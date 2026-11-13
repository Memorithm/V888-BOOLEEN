//! Supervised, opt-in plasticity of *hidden* five-circle neurons.
//!
//! Teacher-side intermediate activations are available only in this
//! instrumented SYNTHETIC experiment. They are privileged local targets,
//! not a plausible biological teaching signal or end-to-end backpropagation.
//! Each fitted hidden neuron preserves its five Walsh mask identities.
use std::collections::BTreeSet;

use crate::fourier5::{Circle, FiveCircle, CIRCLES};
use crate::fourier5_hierarchy::{FiveCircleHierarchy, HierarchyError};
use crate::fourier5_joint_search::{
    fit_output, JointError, MAX_READOUT_EPOCHS, MAX_READOUT_WEIGHT,
};
use crate::fourier5_routing_search::{RoutingSample, MAX_ROUTING_TRAIN};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HiddenError {
    InvalidTraining,
    InvalidTeacher,
    InvalidEpochs,
    DuplicateInput,
    Hierarchy(HierarchyError),
    Joint(JointError),
}

impl From<HierarchyError> for HiddenError {
    fn from(value: HierarchyError) -> Self { Self::Hierarchy(value) }
}
impl From<JointError> for HiddenError {
    fn from(value: JointError) -> Self { Self::Joint(value) }
}

#[derive(Clone, Debug)]
pub struct HiddenFit {
    pub model: FiveCircleHierarchy,
    pub hidden_neurons: usize,
    pub changed_neurons: usize,
    pub hidden_updates: usize,
    pub hidden_example_visits: usize,
    /// Local teacher-label hits on TRAIN after each layer was fitted.
    /// This is NOT end-to-end accuracy or unseen-test performance.
    pub local_correct: usize,
    pub local_examples: usize,
    pub output_updates: usize,
    pub output_epochs: usize,
    pub global_train_correct: usize,
}

fn sign(input: u8, mask: u8) -> i32 {
    if (input & mask).count_ones() % 2 == 0 { 1 } else { -1 }
}

/// Exact integer local perceptron with bounded signed Walsh weights.
/// All mask identities and deterministic sample order stay unchanged.
fn fit_node(
    starting: FiveCircle,
    examples: &[(u8, bool)],
    epochs: usize,
) -> (FiveCircle, usize, usize, usize) {
    let masks: [u8; CIRCLES] = std::array::from_fn(|i| starting.circles()[i].mask);
    let mut weights: [i32; CIRCLES] = std::array::from_fn(|i| {
        let c = starting.circles()[i];
        let value = c.radius.min(MAX_READOUT_WEIGHT as u32) as i32;
        if c.phase_pi { -value } else { value }
    });
    let mut bias = starting.bias().clamp(-MAX_READOUT_WEIGHT, MAX_READOUT_WEIGHT);
    let mut visits = 0;
    let mut updates = 0;
    for _ in 0..epochs {
        let mut mistakes = 0;
        for &(input, expected) in examples {
            visits += 1;
            let score: i32 = bias + weights.iter().zip(masks)
                .map(|(&w, mask)| w * sign(input, mask)).sum::<i32>();
            if (score < 0) != expected {
                let direction = if expected { -1 } else { 1 };
                bias = (bias + direction).clamp(-MAX_READOUT_WEIGHT, MAX_READOUT_WEIGHT);
                for (weight, mask) in weights.iter_mut().zip(masks) {
                    *weight = (*weight + direction * sign(input, mask))
                        .clamp(-MAX_READOUT_WEIGHT, MAX_READOUT_WEIGHT);
                }
                updates += 1;
                mistakes += 1;
            }
        }
        if mistakes == 0 { break; }
    }
    let circles = std::array::from_fn(|i| Circle {
        mask: masks[i],
        radius: weights[i].unsigned_abs(),
        phase_pi: weights[i] < 0,
    });
    let next = FiveCircle::new(circles, bias).expect("existing masks are valid");
    let correct = examples.iter().filter(|&&(input, target)| {
        next.predict(input) == Ok(target)
    }).count();
    (next, updates, visits, correct)
}

/// Stagewise local plasticity, from leaf-facing hidden layer to the readout.
/// All training examples must have unique inputs and match the teacher's
/// final target. Teacher intermediate labels are read only from TRAIN.
/// After fitting all hidden neurons, the final readout is fitted using
/// existing TRAIN output labels via the independent previous mechanism.
pub fn fit_hidden_with_teacher_taps(
    student: &FiveCircleHierarchy,
    teacher: &FiveCircleHierarchy,
    train: &[RoutingSample],
    epochs: usize,
) -> Result<HiddenFit, HiddenError> {
    if student.depth() < 2
        || student.depth() != teacher.depth()
        || student.input_bits() != teacher.input_bits()
    {
        return Err(HiddenError::InvalidTeacher);
    }
    if epochs == 0 || epochs > MAX_READOUT_EPOCHS {
        return Err(HiddenError::InvalidEpochs);
    }
    if train.is_empty() || train.len() > MAX_ROUTING_TRAIN {
        return Err(HiddenError::InvalidTraining);
    }
    let mut seen = BTreeSet::new();
    let mut teacher_traces = Vec::with_capacity(train.len());
    for row in train {
        if row.state.len() != student.input_bits() {
            return Err(HiddenError::InvalidTraining);
        }
        if !seen.insert(row.state.words().to_vec()) {
            return Err(HiddenError::DuplicateInput);
        }
        let trace = teacher.layer_trace(&row.state)?;
        if trace.last().expect("nonempty").1[0] != row.label {
            return Err(HiddenError::InvalidTeacher);
        }
        teacher_traces.push(trace);
    }

    let mut model = student.clone();
    let mut hidden_neurons = 0;
    let mut changed_neurons = 0;
    let mut hidden_updates = 0;
    let mut hidden_example_visits = 0;
    let mut local_correct = 0;
    let mut local_examples = 0;

    for layer in 0..model.depth() - 1 {
        // All neurons in the same layer receive features computed from the
        // prior stage only; no within-layer ordering or label contamination.
        let actual: Vec<_> = train.iter()
            .map(|row| model.layer_trace(&row.state))
            .collect::<Result<Vec<_>, _>>()?;
        let count = actual[0][layer].0.len();
        for neuron in 0..count {
            let local: Vec<(u8,bool)> = actual.iter().zip(&teacher_traces)
                .map(|(observed, goal)| (observed[layer].0[neuron],goal[layer].1[neuron]))
                .collect();
            let previous = model.layer_model(layer, neuron)?;
            let (learned, updates, visits, hits) = fit_node(previous, &local, epochs);
            changed_neurons += usize::from(learned != previous);
            hidden_neurons += 1;
            hidden_updates += updates;
            hidden_example_visits += visits;
            local_correct += hits;
            local_examples += train.len();
            model = model.with_neuron_parameters(layer, neuron, learned)?;
        }
    }
    let output = fit_output(&model, train, epochs)?;
    Ok(HiddenFit {
        model: output.model,
        hidden_neurons,
        changed_neurons,
        hidden_updates,
        hidden_example_visits,
        local_correct,
        local_examples,
        output_updates: output.updates,
        output_epochs: output.epochs_used,
        global_train_correct: output.train_correct,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5_hierarchy::{GateKind, LeafRouting};
    use crate::BooleanState;

    fn samples(teacher: &FiveCircleHierarchy) -> Vec<RoutingSample> {
        let mut rows = Vec::new();
        let mut seed = 0x65ca_4177_019f_0101_u64;
        let mut seen = BTreeSet::new();
        while rows.len() < 128 {
            let mut state = BooleanState::new(teacher.input_bits());
            for i in 0..state.len() {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                if seed & 1024 != 0 { state.set(i, true).unwrap(); }
            }
            if !seen.insert(state.words().to_vec()) { continue; }
            let label = teacher.infer_scalar(&state).unwrap();
            rows.push(RoutingSample { state, label });
        }
        rows
    }

    #[test]
    fn hidden_parameters_change_without_source_or_walsh_mask_change() {
        let teacher = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Majority],
            LeafRouting::Contiguous,
        ).unwrap();
        let base = FiveCircleHierarchy::from_modes(
            &[GateKind::And, GateKind::And],
            LeafRouting::Contiguous,
        ).unwrap();
        let train = samples(&teacher);
        let fit = fit_hidden_with_teacher_taps(&base, &teacher, &train, 16).unwrap();
        assert_eq!(fit.hidden_neurons, 5);
        assert!(fit.changed_neurons > 0);
        assert!(fit.hidden_updates > 0);
        assert_eq!(fit.model.routing(), base.routing());
        assert_eq!(fit.model.directed_contacts(), base.directed_contacts());
        for layer in 0..2 {
            let nodes = if layer == 0 { 5 } else { 1 };
            for index in 0..nodes {
                let prev = base.layer_model(layer, index).unwrap();
                let next = fit.model.layer_model(layer, index).unwrap();
                for slot in 0..CIRCLES {
                    assert_eq!(prev.circles()[slot].mask, next.circles()[slot].mask);
                }
            }
        }
        assert_eq!(fit.global_train_correct, train.len());
    }

    #[test]
    fn rejects_mismatched_teacher_and_unbounded_epoch_budget() {
        let base = FiveCircleHierarchy::from_modes(
            &[GateKind::And, GateKind::And], LeafRouting::Contiguous,
        ).unwrap();
        let wrong_depth = FiveCircleHierarchy::from_modes(
            &[GateKind::And], LeafRouting::Contiguous,
        ).unwrap();
        let train = samples(&base);
        assert!(matches!(
            fit_hidden_with_teacher_taps(&base, &wrong_depth, &train, 8),
            Err(HiddenError::InvalidTeacher)
        ));
        assert!(matches!(
            fit_hidden_with_teacher_taps(&base, &base, &train, 0),
            Err(HiddenError::InvalidEpochs)
        ));
        let mut corrupt = train.clone();
        corrupt[1].label = !corrupt[1].label;
        assert!(matches!(
            fit_hidden_with_teacher_taps(&base, &base, &corrupt, 8),
            Err(HiddenError::InvalidTeacher)
        ));
        let mut duplicate = train.clone();
        duplicate[1] = duplicate[0].clone();
        assert!(matches!(
            fit_hidden_with_teacher_taps(&base, &base, &duplicate, 8),
            Err(HiddenError::DuplicateInput)
        ));
    }

    #[test]
    fn identity_teacher_does_not_mutate_any_hidden_parameters() {
        let base = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Majority],
            LeafRouting::Contiguous,
        ).unwrap();
        let train = samples(&base);
        let fit = fit_hidden_with_teacher_taps(&base, &base, &train, 8).unwrap();
        assert_eq!(fit.changed_neurons, 0);
        assert_eq!(fit.hidden_updates, 0);
        assert_eq!(fit.model.routing(), base.routing());
    }
}

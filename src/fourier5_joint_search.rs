//! Bounded joint search of multiple source swaps and readout parameter plasticity.
//!
//! Existing five-circle layers keep their fixed source fan-in, number of
//! neurons, Walsh masks, and intermediate radii. A deterministic proposal
//! bank supplies multi-swap routings; the FINAL neuron's five signed radii,
//! phases and integer bias may be trained by a bounded supervised perceptron.
//! The two axes are kept separate for matched-capacity comparisons.
use std::collections::BTreeSet;

use crate::fourier5::{Circle, FiveCircle, CIRCLES};
use crate::fourier5_hierarchy::{FiveCircleHierarchy, HierarchyError, BITSLICE_LANES};
use crate::fourier5_routing_search::{
    candidate_swaps, RoutingSample, RoutingSelectionError, MAX_ROUTING_TRAIN,
};

pub const MAX_COMBINATION_BANK: usize = 10;
pub const MAX_COMBINATION_DEPTH: usize = 3;
pub const MAX_READOUT_EPOCHS: usize = 16;
pub const MAX_READOUT_WEIGHT: i32 = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JointError {
    InvalidBudget,
    InvalidTraining,
    DuplicateTrainingInput,
    Hierarchy(HierarchyError),
    Routing(RoutingSelectionError),
}

impl From<HierarchyError> for JointError {
    fn from(value: HierarchyError) -> Self {
        Self::Hierarchy(value)
    }
}
impl From<RoutingSelectionError> for JointError {
    fn from(value: RoutingSelectionError) -> Self {
        Self::Routing(value)
    }
}

#[derive(Clone, Debug)]
pub struct ReadoutFit {
    pub model: FiveCircleHierarchy,
    pub epochs_used: usize,
    pub updates: usize,
    pub train_correct: usize,
}

#[derive(Clone, Debug)]
pub struct JointReport {
    /// Frozen connections, selected only on TRAIN.
    pub route_only: FiveCircleHierarchy,
    pub route_only_correct: usize,
    pub route_only_swaps: Vec<usize>,
    /// Output model trained with ORIGINAL connections.
    pub plastic_only: FiveCircleHierarchy,
    pub plastic_only_correct: usize,
    pub plastic_only_updates: usize,
    /// Output model fitted at EACH multi-swap candidate on the same TRAIN.
    pub joint: FiveCircleHierarchy,
    pub joint_correct: usize,
    pub joint_swaps: Vec<usize>,
    pub joint_updates: usize,
    pub candidates_evaluated: usize,
    pub output_fits: usize,
    pub total_updates: usize,
    pub train_examples: usize,
    /// Number of full model TRAIN classifications for frozen+fitted candidates.
    /// Excludes readout feature extraction and individual perceptron epochs,
    /// which are reported separately as fitting_example_visits.
    pub classification_inferences: usize,
    pub fitting_example_visits: usize,
}

#[derive(Clone, Debug)]
struct PackedTrain {
    bitplanes: Vec<u64>,
    labels: u64,
}

fn pack(model: &FiveCircleHierarchy, train: &[RoutingSample]) -> Result<Vec<PackedTrain>, JointError> {
    if train.len() < BITSLICE_LANES
        || train.len() > MAX_ROUTING_TRAIN
        || train.len() % BITSLICE_LANES != 0
    {
        return Err(JointError::InvalidTraining);
    }
    let mut seen = BTreeSet::new();
    for example in train {
        if example.state.len() != model.input_bits() {
            return Err(JointError::InvalidTraining);
        }
        if !seen.insert(example.state.words().to_vec()) {
            return Err(JointError::DuplicateTrainingInput);
        }
    }
    train
        .chunks_exact(BITSLICE_LANES)
        .map(|chunk| {
            let states: Vec<_> = chunk.iter().map(|row| row.state.clone()).collect();
            let bitplanes = FiveCircleHierarchy::transpose64(&states)?;
            let mut labels = 0_u64;
            for (lane, row) in chunk.iter().enumerate() {
                labels |= u64::from(row.label) << lane;
            }
            Ok(PackedTrain { bitplanes, labels })
        })
        .collect()
}

fn matches(model: &FiveCircleHierarchy, groups: &[PackedTrain]) -> Result<usize, JointError> {
    let mut correct = 0;
    for group in groups {
        let prediction = model.infer_bitplanes(&group.bitplanes)?;
        correct += (64 - (prediction ^ group.labels).count_ones()) as usize;
    }
    Ok(correct)
}

fn sign(input: u8, mask: u8) -> i32 {
    if (input & mask).count_ones() % 2 == 0 {
        1
    } else {
        -1
    }
}

/// Supervised plasticity of the FINAL neuron only. Mask identities and all
/// other neurons are frozen; signed gains, phases and bias adapt online over
/// TRAIN features with saturating integer updates. No held-out labels enter.
/// Does not implement biological spike-timing-dependent plasticity.
pub fn fit_output(
    base: &FiveCircleHierarchy,
    train: &[RoutingSample],
    epochs: usize,
) -> Result<ReadoutFit, JointError> {
    if epochs == 0 || epochs > MAX_READOUT_EPOCHS
        || train.is_empty() || train.len() > MAX_ROUTING_TRAIN
    {
        return Err(JointError::InvalidBudget);
    }
    let current = base.output_model();
    let masks = std::array::from_fn(|i| current.circles()[i].mask);
    let mut weights: [i32; CIRCLES] = std::array::from_fn(|i| {
        let circle = current.circles()[i];
        let radius = (circle.radius.min(MAX_READOUT_WEIGHT as u32)) as i32;
        if circle.phase_pi { -radius } else { radius }
    });
    let mut bias = current.bias().clamp(-MAX_READOUT_WEIGHT, MAX_READOUT_WEIGHT);
    // Cache presynaptic bits before learning: this graph's hidden gates
    // and routing remain immutable throughout the local fit.
    let mut features = Vec::with_capacity(train.len());
    for row in train {
        features.push((base.output_input(&row.state)?, row.label));
    }
    let mut used = 0;
    let mut updates = 0;
    for epoch in 0..epochs {
        let mut mistakes = 0;
        for &(input, label) in &features {
            let score: i32 = bias
                + weights.iter().zip(masks).map(|(&w, mask)| w * sign(input, mask)).sum::<i32>();
            let predicted = score < 0;
            if predicted != label {
                let target = if label { -1 } else { 1 };
                bias = (bias + target).clamp(-MAX_READOUT_WEIGHT, MAX_READOUT_WEIGHT);
                for (weight, mask) in weights.iter_mut().zip(masks) {
                    *weight = (*weight + target * sign(input, mask))
                        .clamp(-MAX_READOUT_WEIGHT, MAX_READOUT_WEIGHT);
                }
                mistakes += 1;
                updates += 1;
            }
        }
        used = epoch + 1;
        if mistakes == 0 {
            break;
        }
    }
    let circles = std::array::from_fn(|i| Circle {
        mask: masks[i],
        radius: weights[i].unsigned_abs(),
        phase_pi: weights[i] < 0,
    });
    let output = FiveCircle::new(circles, bias).expect("five existing valid Walsh masks");
    let model = base.with_output_model(output);
    let correct = features.iter().filter(|&&(input, label)| {
        output.predict(input) == Ok(label)
    }).count();
    Ok(ReadoutFit {
        model,
        epochs_used: used,
        updates,
        train_correct: correct,
    })
}

/// Deterministic, no-label list of 0..=max_swaps distinct bank indices.
/// Lexicographic order permits reproducible selection with explicit lower
/// structural-change cost on equal accuracy.
fn combinations(bank_size: usize, max_swaps: usize) -> Vec<Vec<usize>> {
    let mut selected = vec![Vec::new()];
    for count in 1..=max_swaps {
        fn visit(
            output: &mut Vec<Vec<usize>>,
            path: &mut Vec<usize>,
            next: usize,
            bank_size: usize,
            remaining: usize,
        ) {
            if remaining == 0 {
                output.push(path.clone());
                return;
            }
            for index in next..bank_size {
                if bank_size - index < remaining {
                    break;
                }
                path.push(index);
                visit(output, path, index + 1, bank_size, remaining - 1);
                path.pop();
            }
        }
        visit(&mut selected, &mut Vec::new(), 0, bank_size, count);
    }
    selected
}

/// Apply a *sequence* of distinct proposed swaps. Two proposals may share
/// a leaf; order is deterministic (bank indices ascending). Still a complete
/// permutation, with exactly the original contact budget.
pub fn multiswap_model(
    base: &FiveCircleHierarchy,
    swaps: &[[usize; 2]],
    indices: &[usize],
) -> Result<FiveCircleHierarchy, JointError> {
    let mut route = base.routing().to_vec();
    let mut last = None;
    for &index in indices {
        if index >= swaps.len() || last.is_some_and(|v| index <= v) {
            return Err(JointError::InvalidBudget);
        }
        let [a, b] = swaps[index];
        if a >= route.len() || b >= route.len() || a == b {
            return Err(JointError::InvalidBudget);
        }
        route.swap(a, b);
        last = Some(index);
    }
    Ok(base.rerouted(route)?)
}

/// Exhaustively evaluate all 0, 1, 2, or 3-swap subsets from a bounded
/// seed-generated bank. At EVERY candidate, compare frozen output with
/// the same readout-plasticity procedure on identical TRAIN examples.
/// Neither VALID nor TEST is accepted by this function.
pub fn optimize_joint(
    base: &FiveCircleHierarchy,
    train: &[RoutingSample],
    proposal_seed: u64,
    bank_limit: usize,
    max_swaps: usize,
    epochs: usize,
) -> Result<JointReport, JointError> {
    if bank_limit == 0 || bank_limit > MAX_COMBINATION_BANK
        || !(1..=MAX_COMBINATION_DEPTH).contains(&max_swaps)
        || max_swaps > bank_limit
        || epochs == 0 || epochs > MAX_READOUT_EPOCHS
    {
        return Err(JointError::InvalidBudget);
    }
    let groups = pack(base, train)?;
    let swaps = candidate_swaps(base.input_bits(), proposal_seed, bank_limit)?;
    let bank = combinations(bank_limit, max_swaps);
    let mut unique = BTreeSet::new();

    let mut route_only = base.clone();
    let mut route_only_correct = 0;
    let mut route_only_swaps = Vec::new();
    let mut joint = base.clone();
    let mut joint_correct = 0;
    let mut joint_swaps = Vec::new();
    let mut joint_updates = 0;
    let mut plastic_only = base.clone();
    let mut plastic_only_correct = 0;
    let mut plastic_only_updates = 0;
    let mut candidates_evaluated = 0;
    let mut total_updates = 0;
    let mut fitting_example_visits = 0;

    for indices in &bank {
        let candidate = multiswap_model(base, &swaps, indices)?;
        if !unique.insert(candidate.routing().to_vec()) {
            continue;
        }
        candidates_evaluated += 1;
        let frozen_hits = matches(&candidate, &groups)?;
        let fit = fit_output(&candidate, train, epochs)?;
        fitting_example_visits += fit.epochs_used * train.len();
        total_updates += fit.updates;
        let plastic_hits = matches(&fit.model, &groups)?;
        // Full-candidate scan but no tuning to heldout labels.
        if candidates_evaluated == 1 || frozen_hits > route_only_correct {
            route_only = candidate;
            route_only_correct = frozen_hits;
            route_only_swaps = indices.clone();
        }
        if candidates_evaluated == 1 {
            plastic_only = fit.model.clone();
            plastic_only_correct = plastic_hits;
            plastic_only_updates = fit.updates;
        }
        if candidates_evaluated == 1 || plastic_hits > joint_correct {
            joint = fit.model;
            joint_correct = plastic_hits;
            joint_swaps = indices.clone();
            joint_updates = fit.updates;
        }
    }

    Ok(JointReport {
        route_only,
        route_only_correct,
        route_only_swaps,
        plastic_only,
        plastic_only_correct,
        plastic_only_updates,
        joint,
        joint_correct,
        joint_swaps,
        joint_updates,
        candidates_evaluated,
        output_fits: candidates_evaluated,
        total_updates,
        train_examples: train.len(),
        classification_inferences: candidates_evaluated * train.len() * 2,
        fitting_example_visits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5_hierarchy::{GateKind, LeafRouting};
    use crate::BooleanState;

    fn examples(width: usize, count: usize, model: &FiveCircleHierarchy) -> Vec<RoutingSample> {
        // Constant-time reproducible pseudo-random words; no label reuse
        // across different source records.
        let mut rng = 0x8695_3118_3217_7311_u64;
        let mut seen = BTreeSet::new();
        let mut rows = Vec::new();
        while rows.len() < count {
            let mut state = BooleanState::new(width);
            for i in 0..width {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                if rng & 1 != 0 {
                    state.set(i, true).unwrap();
                }
            }
            if !seen.insert(state.words().to_vec()) {
                continue;
            }
            let label = model.infer_scalar(&state).unwrap();
            rows.push(RoutingSample { state, label });
        }
        rows
    }

    #[test]
    fn exhaustive_combination_count_and_order() {
        assert_eq!(combinations(5, 1).len(), 6);
        assert_eq!(combinations(5, 2).len(), 16);
        assert_eq!(combinations(5, 3).len(), 26);
        let all = combinations(5, 3);
        assert_eq!(all.first(), Some(&Vec::new()));
        assert_eq!(all.last(), Some(&vec![2, 3, 4]));
        assert_eq!(all.iter().collect::<BTreeSet<_>>().len(), all.len());
    }

    #[test]
    fn two_swap_model_differs_from_each_individual_swap() {
        let base = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::And],
            LeafRouting::Contiguous,
        ).unwrap();
        let swaps = candidate_swaps(25, 17, 8).unwrap();
        let two = multiswap_model(&base, &swaps, &[0, 2]).unwrap();
        assert_ne!(two.routing(), base.routing());
        assert_ne!(two.routing(), multiswap_model(&base, &swaps, &[0]).unwrap().routing());
        assert_ne!(two.routing(), multiswap_model(&base, &swaps, &[2]).unwrap().routing());
        assert_eq!(two.neuron_count(), base.neuron_count());
        assert_eq!(two.directed_contacts(), base.directed_contacts());
        assert!(matches!(multiswap_model(&base, &swaps, &[1, 1]), Err(JointError::InvalidBudget)));
        assert!(matches!(multiswap_model(&base, &swaps, &[4, 1]), Err(JointError::InvalidBudget)));
    }

    #[test]
    fn top_readout_plasticity_changes_weights_not_masks_or_connections() {
        let base = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::And],
            LeafRouting::Contiguous,
        ).unwrap();
        let oracle = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Majority],
            LeafRouting::Contiguous,
        ).unwrap();
        let train = examples(25, 256, &oracle);
        let fit = fit_output(&base, &train, 16).unwrap();
        assert_eq!(fit.train_correct, train.len());
        assert!(fit.updates > 0);
        assert_eq!(fit.model.routing(), base.routing());
        assert_eq!(fit.model.neuron_count(), base.neuron_count());
        for (old, new) in base.output_model().circles().iter().zip(fit.model.output_model().circles()) {
            assert_eq!(old.mask, new.mask);
        }
        assert_ne!(fit.model.output_model(), base.output_model());
    }

    #[test]
    fn joint_search_is_at_least_as_good_as_each_train_only_arm() {
        let base = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::And],
            LeafRouting::Contiguous,
        ).unwrap();
        let teacher_base = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Majority],
            LeafRouting::Contiguous,
        ).unwrap();
        let proposals = candidate_swaps(25, 55, 6).unwrap();
        let teacher = multiswap_model(&teacher_base, &proposals, &[0, 2]).unwrap();
        let train = examples(25, 256, &teacher);
        let fit = optimize_joint(&base, &train, 55, 6, 2, 12).unwrap();
        assert!(fit.joint_correct >= fit.plastic_only_correct);
        assert!(fit.joint_correct >= fit.route_only_correct);
        assert_eq!(fit.train_examples, 256);
        assert_eq!(fit.output_fits, fit.candidates_evaluated);
        assert_eq!(fit.classification_inferences, fit.candidates_evaluated * 256 * 2);
        assert_eq!(fit.joint.neuron_count(), base.neuron_count());
        assert_eq!(fit.joint.directed_contacts(), base.directed_contacts());
        assert!(fit.joint_swaps.len() <= 2);
    }

    #[test]
    fn invalid_budgets_training_and_duplicates_rejected() {
        let base = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::And],
            LeafRouting::Contiguous,
        ).unwrap();
        let train = examples(25, 64, &base);
        assert!(matches!(optimize_joint(&base, &train, 1, 0, 2, 4), Err(JointError::InvalidBudget)));
        assert!(matches!(optimize_joint(&base, &train, 1, 6, 4, 4), Err(JointError::InvalidBudget)));
        assert!(matches!(optimize_joint(&base, &train[..63], 1, 6, 2, 4), Err(JointError::InvalidTraining)));
        let mut repeated = train.clone();
        repeated[3] = train[0].clone();
        assert!(matches!(optimize_joint(&base, &repeated, 1, 6, 2, 4), Err(JointError::DuplicateTrainingInput)));
        assert!(matches!(fit_output(&base, &train, 0), Err(JointError::InvalidBudget)));
        assert!(matches!(fit_output(&base, &train, MAX_READOUT_EPOCHS + 1), Err(JointError::InvalidBudget)));
    }
}

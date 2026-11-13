//! Transient radius modulation for the five-circle Boolean Walsh neuron.
//!
//! A gate reads only an explicit five-bit context. An active conjunction adds
//! a nonnegative radius to a fixed Fourier-Walsh mode for this evaluation.
//! Anatomical edges, original synapses and persistent learned weights are NOT
//! changed. This is an experimental computational rule, not a biological model.
use crate::fourier5::{Circle, FiveCircle, FourierError, CIRCLES, INPUT_BITS, MAX_EPOCHS};
use crate::BooleanState;

pub const MAX_EXAMPLES: usize = 4_096;
pub const MAX_NEURONS: usize = 1_000_000;
pub const MAX_BOOST: u32 = 65_535;
const FIVE_BITS: u8 = (1 << INPUT_BITS) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModulationError {
    InvalidInput,
    InvalidGateMask,
    InvalidBoost,
    InvalidWiring,
    MismatchedState,
    TooManyNeurons,
    InvalidTrainingBudget,
    InvalidTrainingExamples,
}

impl From<FourierError> for ModulationError {
    fn from(_: FourierError) -> Self {
        Self::InvalidInput
    }
}

/// A nonzero mask is a conjunction: all selected context neurons must fire.
/// Mask zero means the gate is disabled. A gate never creates new connections.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RadiusGate {
    pub context_mask: u8,
    pub boost: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatedFiveCircle {
    base: FiveCircle,
    gates: [RadiusGate; CIRCLES],
}

fn signed_character(circle: Circle, input: u8) -> i64 {
    let odd = ((input & circle.mask).count_ones() & 1) != 0;
    if odd ^ circle.phase_pi {
        -1
    } else {
        1
    }
}

fn valid_bits(bits: u8) -> bool {
    bits & !FIVE_BITS == 0
}

impl GatedFiveCircle {
    pub fn new(base: FiveCircle, gates: [RadiusGate; CIRCLES]) -> Result<Self, ModulationError> {
        if gates.iter().any(|gate| !valid_bits(gate.context_mask)) {
            return Err(ModulationError::InvalidGateMask);
        }
        if gates.iter().any(|gate| gate.boost > MAX_BOOST) {
            return Err(ModulationError::InvalidBoost);
        }
        Ok(Self { base, gates })
    }

    #[must_use]
    pub fn base(&self) -> &FiveCircle {
        &self.base
    }

    #[must_use]
    pub fn gates(&self) -> &[RadiusGate; CIRCLES] {
        &self.gates
    }

    /// The transient, nonnegative radius; the persistent radius is untouched.
    pub fn effective_radius(&self, circle: usize, context: u8) -> Result<u64, ModulationError> {
        if !valid_bits(context) {
            return Err(ModulationError::InvalidInput);
        }
        let gate = self
            .gates
            .get(circle)
            .ok_or(ModulationError::InvalidWiring)?;
        let base = self
            .base
            .circles()
            .get(circle)
            .ok_or(ModulationError::InvalidWiring)?;
        let enabled = gate.context_mask != 0 && context & gate.context_mask == gate.context_mask;
        Ok(u64::from(base.radius) + if enabled { u64::from(gate.boost) } else { 0 })
    }

    /// Exact i64 Boolean inference; no allocations, floating point or sin/cos.
    pub fn score(&self, input: u8, context: u8) -> Result<i64, ModulationError> {
        if !valid_bits(input) || !valid_bits(context) {
            return Err(ModulationError::InvalidInput);
        }
        let mut response = self.base.score(input)?;
        for (circle, gate) in self.base.circles().iter().zip(self.gates.iter()) {
            if gate.context_mask != 0 && context & gate.context_mask == gate.context_mask {
                response += signed_character(*circle, input) * i64::from(gate.boost);
            }
        }
        Ok(response)
    }

    pub fn predict(&self, input: u8, context: u8) -> Result<bool, ModulationError> {
        Ok(self.score(input, context)? < 0)
    }

    pub fn correct_count(&self, examples: &[(u8, u8, bool)]) -> Result<usize, ModulationError> {
        let mut correct = 0;
        for &(input, context, expected) in examples {
            if self.predict(input, context)? == expected {
                correct += 1;
            }
        }
        Ok(correct)
    }
}

/// A neuron consumes five activity bits and five explicit context bits.
/// Indices are local indices of an existing BooleanState, not biological IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WiredNeuron {
    pub inputs: [usize; CIRCLES],
    pub context: [usize; CIRCLES],
    pub model: GatedFiveCircle,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiveCircleLayer {
    input_width: usize,
    neurons: Vec<WiredNeuron>,
}

fn gather_bits(state: &BooleanState, indices: &[usize; CIRCLES]) -> Result<u8, ModulationError> {
    let mut bits = 0_u8;
    for (bit, &index) in indices.iter().enumerate() {
        if state
            .get(index)
            .map_err(|_| ModulationError::InvalidWiring)?
        {
            bits |= 1_u8 << bit;
        }
    }
    Ok(bits)
}

impl FiveCircleLayer {
    pub fn new(input_width: usize, neurons: Vec<WiredNeuron>) -> Result<Self, ModulationError> {
        if input_width > MAX_NEURONS || neurons.len() > MAX_NEURONS {
            return Err(ModulationError::TooManyNeurons);
        }
        if neurons.iter().any(|neuron| {
            neuron
                .inputs
                .iter()
                .chain(neuron.context.iter())
                .any(|&index| index >= input_width)
        }) {
            return Err(ModulationError::InvalidWiring);
        }
        Ok(Self {
            input_width,
            neurons,
        })
    }

    #[must_use]
    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    /// Install one independently learned model per existing neuron.
    /// The source indices and topology remain immutable. A wrong-length
    /// collection is rejected before any model can be changed.
    pub fn install_models(
        &mut self,
        models: &[GatedFiveCircle],
    ) -> Result<(), ModulationError> {
        if models.len() != self.neurons.len() {
            return Err(ModulationError::MismatchedState);
        }
        for (neuron, &model) in self.neurons.iter_mut().zip(models.iter()) {
            neuron.model = model;
        }
        Ok(())
    }

    #[must_use]
    pub fn model_at(&self, index: usize) -> Option<GatedFiveCircle> {
        self.neurons.get(index).map(|neuron| neuron.model)
    }

    /// Synchronous evaluation from one immutable snapshot, producing bit-packed
    /// outputs in neuron order. No implicit recurrent edges are introduced.
    pub fn step(&self, state: &BooleanState) -> Result<BooleanState, ModulationError> {
        if state.len() != self.input_width {
            return Err(ModulationError::MismatchedState);
        }
        let mut next = BooleanState::new(self.neurons.len());
        for (index, neuron) in self.neurons.iter().enumerate() {
            let input = gather_bits(state, &neuron.inputs)?;
            let context = gather_bits(state, &neuron.context)?;
            if neuron.model.predict(input, context)? {
                next.set(index, true)
                    .map_err(|_| ModulationError::InvalidWiring)?;
            }
        }
        Ok(next)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateTrainingReport {
    pub model: GatedFiveCircle,
    pub converged: bool,
    pub epochs_used: usize,
    pub updates: usize,
    pub correct: usize,
}

/// Supervised bounded updates to transient boosts only. Fixed masks, base
/// radii, bias and phase are never changed. Nonnegative bounded boosts may
/// make some training tasks unrealizable; convergence is not guaranteed.
pub fn train_context_gates(
    mut model: GatedFiveCircle,
    examples: &[(u8, u8, bool)],
    epochs: usize,
) -> Result<GateTrainingReport, ModulationError> {
    if epochs > MAX_EPOCHS {
        return Err(ModulationError::InvalidTrainingBudget);
    }
    if examples.is_empty() || examples.len() > MAX_EXAMPLES {
        return Err(ModulationError::InvalidTrainingExamples);
    }
    if examples
        .iter()
        .any(|&(input, context, _)| !valid_bits(input) || !valid_bits(context))
    {
        return Err(ModulationError::InvalidInput);
    }
    let mut updates = 0;
    let mut epochs_used = 0;
    let mut converged = false;
    for epoch in 0..epochs {
        let mut mistakes = 0;
        let mut changed_any = false;
        for &(input, context, expected) in examples {
            if model.predict(input, context)? == expected {
                continue;
            }
            mistakes += 1;
            // Desired score movement is negative for true, positive for false.
            let desired_sign: i64 = if expected { -1 } else { 1 };
            let mut changed = false;
            for index in 0..CIRCLES {
                let gate = &mut model.gates[index];
                if gate.context_mask == 0 || context & gate.context_mask != gate.context_mask {
                    continue;
                }
                let contribution_sign = signed_character(model.base.circles()[index], input);
                let original = gate.boost;
                if desired_sign == contribution_sign {
                    gate.boost = gate.boost.saturating_add(1).min(MAX_BOOST);
                } else {
                    gate.boost = gate.boost.saturating_sub(1);
                }
                changed |= gate.boost != original;
            }
            if changed {
                updates += 1;
                changed_any = true;
            }
        }
        epochs_used = epoch + 1;
        if mistakes == 0 {
            converged = true;
            break;
        }
        if !changed_any {
            break;
        }
    }
    let correct = model.correct_count(examples)?;
    // A final updated model can be perfect even if the last epoch included
    // mistakes. Report the actually scored result, not an earlier snapshot.
    converged |= correct == examples.len();
    Ok(GateTrainingReport {
        model,
        converged,
        epochs_used,
        updates,
        correct,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> FiveCircle {
        let mut circles = [Circle::default(); CIRCLES];
        circles[0] = Circle {
            mask: 1,
            radius: 1,
            phase_pi: false,
        };
        FiveCircle::new(circles, 2).unwrap()
    }

    fn gated(boost: u32) -> GatedFiveCircle {
        let mut gates = [RadiusGate::default(); CIRCLES];
        gates[0] = RadiusGate {
            context_mask: 1,
            boost,
        };
        GatedFiveCircle::new(base(), gates).unwrap()
    }

    #[test]
    fn context_modulates_radius_without_modifying_persistent_weight() {
        let model = gated(2);
        assert_eq!(model.effective_radius(0, 0), Ok(1));
        assert_eq!(model.effective_radius(0, 1), Ok(3));
        assert_eq!(model.predict(1, 0), Ok(false));
        assert_eq!(model.predict(1, 1), Ok(true));
        assert_eq!(model.base().circles()[0].radius, 1);
        assert_eq!(model.predict(0, 1), Ok(false));
    }

    #[test]
    fn bounded_supervised_updates_learn_context_specific_firing() {
        let examples = [(1, 0, false), (1, 1, true)];
        let start = gated(0);
        assert_eq!(start.correct_count(&examples), Ok(1));
        let trained = train_context_gates(start, &examples, 32).unwrap();
        assert!(trained.converged);
        assert_eq!(trained.correct, 2);
        assert_eq!(trained.model.gates()[0].boost, 2);
        assert_eq!(trained.model.base(), &base());
    }

    #[test]
    fn fixed_neuron_without_context_cannot_fit_opposite_labels() {
        let examples = [(1, 0, false), (1, 1, true)];
        let no_gate = GatedFiveCircle::new(base(), [RadiusGate::default(); CIRCLES]).unwrap();
        let report = train_context_gates(no_gate, &examples, 32).unwrap();
        assert!(!report.converged);
        assert_eq!(report.correct, 1);
        assert_eq!(report.updates, 0);
    }

    #[test]
    fn synchronous_layer_reads_packed_snapshots() {
        let node = WiredNeuron {
            inputs: [0, 1, 2, 3, 4],
            context: [5, 6, 7, 8, 9],
            model: gated(2),
        };
        let layer = FiveCircleLayer::new(10, vec![node, node]).unwrap();
        let mut activity = BooleanState::new(10);
        activity.set(0, true).unwrap();
        assert_eq!(layer.step(&activity).unwrap().active_count(), 0);
        activity.set(5, true).unwrap();
        let snapshot = activity.clone();
        let result = layer.step(&activity).unwrap();
        assert_eq!(result.active_count(), 2);
        assert_eq!(result.len(), 2);
        assert_eq!(activity, snapshot);
        assert_eq!(layer.neuron_count(), 2);
    }

    #[test]
    fn validation_and_caps_are_enforced() {
        let mut gates = [RadiusGate::default(); CIRCLES];
        gates[0].context_mask = 32;
        assert_eq!(
            GatedFiveCircle::new(base(), gates),
            Err(ModulationError::InvalidGateMask)
        );
        gates[0] = RadiusGate {
            context_mask: 1,
            boost: MAX_BOOST + 1,
        };
        assert_eq!(
            GatedFiveCircle::new(base(), gates),
            Err(ModulationError::InvalidBoost)
        );
        let model = gated(2);
        assert_eq!(model.predict(32, 1), Err(ModulationError::InvalidInput));
        assert_eq!(model.predict(0, 32), Err(ModulationError::InvalidInput));
        assert_eq!(
            model.effective_radius(5, 0),
            Err(ModulationError::InvalidWiring)
        );
        let node = WiredNeuron {
            inputs: [0, 1, 2, 3, 10],
            context: [5, 6, 7, 8, 9],
            model,
        };
        assert_eq!(
            FiveCircleLayer::new(10, vec![node]),
            Err(ModulationError::InvalidWiring)
        );
        assert_eq!(
            train_context_gates(model, &[(1, 0, true)], MAX_EPOCHS + 1),
            Err(ModulationError::InvalidTrainingBudget)
        );
        assert_eq!(
            train_context_gates(model, &[], 1),
            Err(ModulationError::InvalidTrainingExamples)
        );
    }

    #[test]
    fn zero_context_mask_never_activates_a_boost() {
        let mut gates = [RadiusGate::default(); CIRCLES];
        gates[0].boost = 3;
        let model = GatedFiveCircle::new(base(), gates).unwrap();
        assert_eq!(model.effective_radius(0, 31), Ok(1));
        assert_eq!(model.score(1, 31), Ok(1));
    }

    #[test]
    fn all_five_context_gates_can_operate_independently() {
        let circles = std::array::from_fn(|i| Circle {
            mask: 1_u8 << i,
            radius: 1,
            phase_pi: false,
        });
        let base = FiveCircle::new(circles, 0).unwrap();
        let gates = std::array::from_fn(|i| RadiusGate {
            context_mask: 1_u8 << i,
            boost: 2,
        });
        let model = GatedFiveCircle::new(base, gates).unwrap();
        for i in 0..CIRCLES {
            assert_eq!(model.effective_radius(i, 1 << i), Ok(3));
            assert_eq!(model.effective_radius(i, 0), Ok(1));
        }
        assert_eq!(model.score(0, 31), Ok(15));
        assert_eq!(model.score(31, 31), Ok(-15));
    }
}

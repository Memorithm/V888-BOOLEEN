//! Persistent exact bit-sliced cache for accepted sequential interventions.
//! Synthetic five-circle circuits only; no BANC anatomical claims.
use std::collections::BTreeSet;
use crate::fourier5::CIRCLES;
use crate::fourier5_hierarchy::{FiveCircleHierarchy, HierarchyError, shannon_64};
use crate::fourier5_incremental::{FiveCircleIncrementalCache, PackedIntervention};

pub struct PersistentSession {
    model: FiveCircleHierarchy,
    planes: Vec<u64>,
    stages: Vec<Vec<u64>>,
    pub rebuilt_gates: usize,
    pub updated_gates: usize,
}

impl PersistentSession {
    pub fn new(model: FiveCircleHierarchy, planes: Vec<u64>) -> Result<Self, HierarchyError> {
        if planes.len() != model.input_bits() {
            return Err(HierarchyError::InvalidInput);
        }
        let mut stages = Vec::with_capacity(model.depth() + 1);
        stages.push(model.routing().iter().map(|&src| planes[usize::from(src)]).collect());
        for layer in 0..model.depth() {
            let mut next = Vec::with_capacity(stages[layer].len() / CIRCLES);
            for neuron in 0..stages[layer].len() / CIRCLES {
                let input = std::array::from_fn(|bit| stages[layer][neuron * CIRCLES + bit]);
                next.push(shannon_64(model.gate_truth(layer, neuron)?, input));
            }
            stages.push(next);
        }
        let rebuilt_gates = model.neuron_count();
        Ok(Self { model, planes, stages, rebuilt_gates, updated_gates: 0 })
    }

    pub fn model(&self) -> &FiveCircleHierarchy { &self.model }
    pub fn output(&self) -> u64 { self.stages.last().expect("nonempty hierarchy")[0] }

    /// Proposals are evaluated against the current accepted model, without
    /// mutating the session. The independent full inference is the oracle.
    pub fn propose(&self, proposal: PackedIntervention) -> Result<u64, HierarchyError> {
        // Reuse the existing exact counterfactual oracle; accepted updates
        // below update only the affected cone, not the whole cache.
        let cache = FiveCircleIncrementalCache::new(&self.model, &self.planes)?;
        Ok(cache.evaluate(proposal)?.output)
    }

    /// Atomic acceptance: validate and construct the candidate first, then
    /// refresh only changed leaves/ancestors. Failed changes leave state intact.
    pub fn accept(&mut self, proposal: PackedIntervention) -> Result<usize, HierarchyError> {
        let (candidate, start_layer, mut affected) = match proposal {
            PackedIntervention::LeafSwap { first, second } => {
                if first >= self.model.input_bits() || second >= self.model.input_bits() || first == second {
                    return Err(HierarchyError::InvalidRouting);
                }
                let mut routing = self.model.routing().to_vec();
                routing.swap(first, second);
                let candidate = self.model.rerouted(routing)?;
                (candidate, 0, BTreeSet::from([first / CIRCLES, second / CIRCLES]))
            }
            PackedIntervention::Parameter { layer, neuron, model } => {
                let old = self.model.layer_model(layer, neuron)?;
                if old.circles().iter().zip(model.circles()).any(|(a,b)| a.mask != b.mask) {
                    return Err(HierarchyError::InvalidLayerShape);
                }
                let candidate = self.model.with_neuron_parameters(layer, neuron, model)?;
                (candidate, layer, BTreeSet::from([neuron]))
            }
        };
        if let PackedIntervention::LeafSwap { first, second } = proposal {
            self.stages[0].swap(first, second);
        }
        let mut evaluations = 0;
        for layer in start_layer..candidate.depth() {
            for &neuron in &affected {
                let input = std::array::from_fn(|bit| self.stages[layer][neuron * CIRCLES + bit]);
                self.stages[layer + 1][neuron] = shannon_64(candidate.gate_truth(layer, neuron)?, input);
                evaluations += 1;
            }
            affected = affected.into_iter().map(|index| index / CIRCLES).collect();
        }
        self.model = candidate;
        self.updated_gates += evaluations;
        Ok(evaluations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5_hierarchy::{GateKind, LeafRouting};
    use crate::fourier5::FiveCircle;

    #[test]
    fn accepted_and_rejected_sequences_match_full_oracle() {
        for depth in 2..=4 {
            let base = FiveCircleHierarchy::from_modes(
                &vec![GateKind::Majority; depth], LeafRouting::SeededShuffle(71)
            ).unwrap();
            let planes: Vec<u64> = (0..base.input_bits())
                .map(|i| (i as u64 + 19).wrapping_mul(0x9e3779b97f4a7c15)).collect();
            let mut session = PersistentSession::new(base.clone(), planes.clone()).unwrap();
            let mut reference = base;
            for step in 0..24 {
                let proposal = if step % 3 == 0 {
                    let old = reference.layer_model(0, step % (reference.input_bits() / CIRCLES)).unwrap();
                    let mut circles = *old.circles();
                    circles[0].radius += 1;
                    PackedIntervention::Parameter {
                        layer: 0, neuron: step % (reference.input_bits() / CIRCLES),
                        model: FiveCircle::new(circles, old.bias()).unwrap(),
                    }
                } else {
                    PackedIntervention::LeafSwap { first: step, second: step + 1 }
                };
                let expected = match proposal {
                    PackedIntervention::LeafSwap { first, second } => {
                        let mut routing = reference.routing().to_vec();
                        routing.swap(first, second);
                        reference.rerouted(routing).unwrap()
                    }
                    PackedIntervention::Parameter { layer, neuron, model } =>
                        reference.with_neuron_parameters(layer, neuron, model).unwrap(),
                };
                assert_eq!(session.propose(proposal).unwrap(), expected.infer_bitplanes(&planes).unwrap());
                if step % 4 != 1 {
                    let gates = session.accept(proposal).unwrap();
                    assert!(gates <= 2 * depth);
                    reference = expected;
                }
                assert_eq!(session.output(), reference.infer_bitplanes(&planes).unwrap());
            }
            assert_eq!(session.rebuilt_gates, reference.neuron_count());
        }
    }
}

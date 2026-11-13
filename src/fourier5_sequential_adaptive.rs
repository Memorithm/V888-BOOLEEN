//! Exact sequential proposals with conservative measured-cost engine choice.
//! Synthetic Boolean trees only; no biological connectivity is inferred.
use crate::fourier5_hierarchy::{FiveCircleHierarchy, HierarchyError};
use crate::fourier5_incremental::{FiveCircleIncrementalCache, PackedIntervention};

#[derive(Clone, Copy, Debug)]
pub struct EngineCalibration {
    pub full_ns_per_gate: u64,
    pub incremental_ns_per_gate: u64,
    pub cache_build_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    Full,
    Incremental,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProposalResult {
    pub output: u64,
    pub engine: Engine,
    pub full_gates: usize,
    pub cone_gates: usize,
    pub accepted: bool,
}

/// A cache is rebuilt from the current model for each proposal, including
/// after accepted interventions. Its build cost is included in engine choice.
/// Unknown calibration (zero rates) always selects the full oracle.
pub fn evaluate_and_apply(
    current: &mut FiveCircleHierarchy,
    planes: &[u64],
    proposal: PackedIntervention,
    accept: bool,
    calibration: EngineCalibration,
) -> Result<ProposalResult, HierarchyError> {
    if planes.len() != current.input_bits() {
        return Err(HierarchyError::InvalidInput);
    }
    let candidate = match proposal {
        PackedIntervention::LeafSwap { first, second } => {
            if first >= current.input_bits() || second >= current.input_bits() || first == second {
                return Err(HierarchyError::InvalidRouting);
            }
            let mut routing = current.routing().to_vec();
            routing.swap(first, second);
            current.rerouted(routing)?
        }
        PackedIntervention::Parameter { layer, neuron, model } => {
            current.with_neuron_parameters(layer, neuron, model)?
        }
    };
    let cache = FiveCircleIncrementalCache::new(current, planes)?;
    let delta = cache.evaluate(proposal)?;
    let full_gates = current.neuron_count();
    let incremental_cost = (cache.build_gate_evaluations() as u128)
        .saturating_mul(calibration.full_ns_per_gate as u128)
        .saturating_add(calibration.cache_build_ns as u128)
        .saturating_add((delta.gate_evaluations as u128)
            .saturating_mul(calibration.incremental_ns_per_gate as u128));
    let full_cost = (full_gates as u128)
        .saturating_mul(calibration.full_ns_per_gate as u128);
    let engine = if calibration.full_ns_per_gate != 0
        && calibration.incremental_ns_per_gate != 0
        && incremental_cost < full_cost {
        Engine::Incremental
    } else {
        Engine::Full
    };
    let output = match engine {
        Engine::Full => candidate.infer_bitplanes(planes)?,
        Engine::Incremental => delta.output,
    };
    if accept {
        *current = candidate;
    }
    Ok(ProposalResult {
        output,
        engine,
        full_gates,
        cone_gates: delta.gate_evaluations,
        accepted: accept,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5::FiveCircle;
    use crate::fourier5_hierarchy::{GateKind, LeafRouting};

    fn planes(count: usize) -> Vec<u64> {
        (0..count).map(|i| (i as u64 + 17).wrapping_mul(0x9e3779b97f4a7c15)).collect()
    }

    #[test]
    fn sequential_swaps_match_independent_full_oracle() {
        for depth in 2..=4 {
            let mut model = FiveCircleHierarchy::from_modes(
                &vec![GateKind::Majority; depth], LeafRouting::SeededShuffle(77)
            ).unwrap();
            let inputs = planes(model.input_bits());
            for step in 0..16 {
                let first = step;
                let second = step + 1;
                let mut routing = model.routing().to_vec();
                routing.swap(first, second);
                let expected = model.rerouted(routing).unwrap();
                let before = model.infer_bitplanes(&inputs).unwrap();
                let accept = step % 2 == 0;
                let result = evaluate_and_apply(
                    &mut model, &inputs,
                    PackedIntervention::LeafSwap { first, second },
                    accept,
                    EngineCalibration {
                        full_ns_per_gate: 10,
                        incremental_ns_per_gate: 1,
                        cache_build_ns: 0,
                    },
                ).unwrap();
                assert_eq!(result.output, expected.infer_bitplanes(&inputs).unwrap());
                assert_eq!(model.infer_bitplanes(&inputs).unwrap(),
                    if accept { result.output } else { before });
            }
        }
    }

    #[test]
    fn hidden_parameter_proposals_match_full_oracle() {
        let mut model = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity, GateKind::Or],
            LeafRouting::Contiguous,
        ).unwrap();
        let inputs = planes(model.input_bits());
        for neuron in [0, 3, 12] {
            let old = model.layer_model(0, neuron).unwrap();
            let mut circles = *old.circles();
            circles[0].radius += 1;
            let next = FiveCircle::new(circles, old.bias() + 1).unwrap();
            let expected = model.with_neuron_parameters(0, neuron, next).unwrap();
            let result = evaluate_and_apply(
                &mut model, &inputs,
                PackedIntervention::Parameter { layer: 0, neuron, model: next },
                true,
                EngineCalibration {
                    full_ns_per_gate: 0,
                    incremental_ns_per_gate: 0,
                    cache_build_ns: 0,
                },
            ).unwrap();
            assert_eq!(result.engine, Engine::Full);
            assert_eq!(result.output, expected.infer_bitplanes(&inputs).unwrap());
        }
    }
}

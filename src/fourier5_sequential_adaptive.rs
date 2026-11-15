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

/// A cache is constructed only when its conservative cost lower bound can beat
/// full inference. Rebuild after acceptance; never reuse stale state.
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
        PackedIntervention::Parameter {
            layer,
            neuron,
            model,
        } => current.with_neuron_parameters(layer, neuron, model)?,
    };
    let full_gates = current.neuron_count();
    // A fresh cache evaluates every baseline gate before any counterfactual
    // work. With nonzero per-gate cost it cannot beat a single full pass.
    // Skip that allocation and traversal entirely when the lower bound
    // already exceeds the full oracle cost.
    let full_cost = (full_gates as u128).saturating_mul(calibration.full_ns_per_gate as u128);
    let cache_lower_bound = (full_gates as u128)
        .saturating_mul(calibration.full_ns_per_gate as u128)
        .saturating_add(calibration.cache_build_ns as u128);
    let can_try_incremental = calibration.full_ns_per_gate != 0
        && calibration.incremental_ns_per_gate != 0
        && cache_lower_bound < full_cost;
    let (output, engine, cone_gates) = if can_try_incremental {
        let cache = FiveCircleIncrementalCache::new(current, planes)?;
        let delta = cache.evaluate(proposal)?;
        let incremental_cost = cache_lower_bound.saturating_add(
            (delta.gate_evaluations as u128)
                .saturating_mul(calibration.incremental_ns_per_gate as u128),
        );
        if incremental_cost < full_cost {
            (delta.output, Engine::Incremental, delta.gate_evaluations)
        } else {
            (
                candidate.infer_bitplanes(planes)?,
                Engine::Full,
                delta.gate_evaluations,
            )
        }
    } else {
        (candidate.infer_bitplanes(planes)?, Engine::Full, 0)
    };
    if accept {
        *current = candidate;
    }
    Ok(ProposalResult {
        output,
        engine,
        full_gates,
        cone_gates,
        accepted: accept,
    })
}


/// A conservative break-even gate budget for a *frozen* batch, evaluated
/// before allocating a cache. Returns None when calibration is missing or
/// the cache cannot amortize. All arithmetic is checked in u128.
///
/// Unlike a gate-count-only heuristic this includes cold-cache construction
/// and the requested minimum saving in parts per million (0..=1_000_000).
/// The returned limit is the largest total cone size whose estimated cost
/// still satisfies the saving. It is an estimate, not a performance claim.
pub fn frozen_batch_cone_budget(
    baseline_gates: usize,
    proposal_count: usize,
    calibration: EngineCalibration,
    minimum_saving_ppm: u32,
) -> Option<u128> {
    if baseline_gates == 0
        || proposal_count == 0
        || minimum_saving_ppm > 1_000_000
        || calibration.full_ns_per_gate == 0
        || calibration.incremental_ns_per_gate == 0
    {
        return None;
    }
    let full = (baseline_gates as u128)
        .checked_mul(proposal_count as u128)?
        .checked_mul(u128::from(calibration.full_ns_per_gate))?;
    let build = (baseline_gates as u128)
        .checked_mul(u128::from(calibration.full_ns_per_gate))?
        .checked_add(u128::from(calibration.cache_build_ns))?;
    // Strict saving: incremental * 1_000_000 <
    // full * (1_000_000 - minimum_saving_ppm).
    // Use quotient/remainder to avoid overflow on the full product.
    let allowance = full
        .checked_mul(u128::from(1_000_000 - minimum_saving_ppm))?
        / 1_000_000;
    if build >= allowance {
        return None;
    }
    // Require strict improvement even with integer truncation.
    Some((allowance - build - 1) / u128::from(calibration.incremental_ns_per_gate))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchEvaluation {
    pub outputs: Vec<u64>,
    pub engine: Engine,
    pub full_gates: usize,
    pub incremental_gates: usize,
    pub cache_build_gates: usize,
}

/// Evaluate independent counterfactual proposals against the SAME frozen
/// baseline. A single cache is amortized across the batch; proposals are
/// never applied here. Use evaluate_and_apply for sequential acceptance.
/// Calibration is caller-measured, not inferred from synthetic gate counts.
pub fn evaluate_frozen_batch(
    baseline: &FiveCircleHierarchy,
    planes: &[u64],
    proposals: &[PackedIntervention],
    calibration: EngineCalibration,
) -> Result<BatchEvaluation, HierarchyError> {
    if planes.len() != baseline.input_bits() {
        return Err(HierarchyError::InvalidInput);
    }
    if proposals.is_empty() {
        return Ok(BatchEvaluation {
            outputs: Vec::new(),
            engine: Engine::Full,
            full_gates: 0,
            incremental_gates: 0,
            cache_build_gates: 0,
        });
    }
    // Validate every proposal with the production hierarchy API, including
    // masks, before choosing an engine. No partial result is returned.
    let mut candidates = Vec::with_capacity(proposals.len());
    for &proposal in proposals {
        let candidate = match proposal {
            PackedIntervention::LeafSwap { first, second } => {
                if first >= baseline.input_bits()
                    || second >= baseline.input_bits()
                    || first == second
                {
                    return Err(HierarchyError::InvalidRouting);
                }
                let mut routing = baseline.routing().to_vec();
                routing.swap(first, second);
                baseline.rerouted(routing)?
            }
            PackedIntervention::Parameter {
                layer,
                neuron,
                model,
            } => baseline.with_neuron_parameters(layer, neuron, model)?,
        };
        candidates.push(candidate);
    }
    let full_gates = baseline.neuron_count().saturating_mul(proposals.len());
    if calibration.full_ns_per_gate == 0 || calibration.incremental_ns_per_gate == 0 {
        return Ok(BatchEvaluation {
            outputs: candidates
                .iter()
                .map(|c| c.infer_bitplanes(planes))
                .collect::<Result<Vec<_>, _>>()?,
            engine: Engine::Full,
            full_gates,
            incremental_gates: 0,
            cache_build_gates: 0,
        });
    }
    // The cold cache must evaluate every baseline gate. If even this
    // lower bound cannot beat the complete batch, avoid allocating the
    // cache and evaluating all counterfactual cones.
    let full_cost_bound = (full_gates as u128)
        .saturating_mul(u128::from(calibration.full_ns_per_gate));
    let cache_build_bound = (baseline.neuron_count() as u128)
        .saturating_mul(u128::from(calibration.full_ns_per_gate))
        .saturating_add(u128::from(calibration.cache_build_ns));
    if cache_build_bound >= full_cost_bound {
        return Ok(BatchEvaluation {
            outputs: candidates
                .iter()
                .map(|candidate| candidate.infer_bitplanes(planes))
                .collect::<Result<Vec<_>, _>>()?,
            engine: Engine::Full,
            full_gates,
            incremental_gates: 0,
            cache_build_gates: 0,
        });
    }
    let cache = FiveCircleIncrementalCache::new(baseline, planes)?;
    let deltas = proposals
        .iter()
        .copied()
        .map(|p| cache.evaluate(p))
        .collect::<Result<Vec<_>, _>>()?;
    let cone_gates: usize = deltas.iter().map(|d| d.gate_evaluations).sum();
    let full_cost = (full_gates as u128).saturating_mul(calibration.full_ns_per_gate as u128);
    let incremental_cost = (cache.build_gate_evaluations() as u128)
        .saturating_mul(calibration.full_ns_per_gate as u128)
        .saturating_add(calibration.cache_build_ns as u128)
        .saturating_add(
            (cone_gates as u128).saturating_mul(calibration.incremental_ns_per_gate as u128),
        );
    let use_incremental = incremental_cost < full_cost;
    let outputs = if use_incremental {
        deltas.iter().map(|d| d.output).collect()
    } else {
        candidates
            .iter()
            .map(|c| c.infer_bitplanes(planes))
            .collect::<Result<Vec<_>, _>>()?
    };
    Ok(BatchEvaluation {
        outputs,
        engine: if use_incremental {
            Engine::Incremental
        } else {
            Engine::Full
        },
        full_gates,
        incremental_gates: cone_gates,
        cache_build_gates: cache.build_gate_evaluations(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5::FiveCircle;
    use crate::fourier5_hierarchy::{GateKind, LeafRouting};

    fn planes(count: usize) -> Vec<u64> {
        (0..count)
            .map(|i| (i as u64 + 17).wrapping_mul(0x9e3779b97f4a7c15))
            .collect()
    }


    #[test]
    fn cone_budget_accounts_for_cold_build_margin_and_invalid_calibration() {
        let calibration = EngineCalibration {
            full_ns_per_gate: 10,
            incremental_ns_per_gate: 2,
            cache_build_ns: 5,
        };
        // Four full passes cost 400; cold build costs 105.
        assert_eq!(frozen_batch_cone_budget(10, 4, calibration, 0), Some(147));
        // 25% margin reduces allowance to 300.
        assert_eq!(frozen_batch_cone_budget(10, 4, calibration, 250_000), Some(97));
        assert_eq!(frozen_batch_cone_budget(10, 1, calibration, 0), None);
        assert_eq!(frozen_batch_cone_budget(10, 4, calibration, 1_000_001), None);
        assert_eq!(frozen_batch_cone_budget(0, 4, calibration, 0), None);
        assert_eq!(frozen_batch_cone_budget(10, 0, calibration, 0), None);
        assert_eq!(frozen_batch_cone_budget(usize::MAX, usize::MAX,
            EngineCalibration { full_ns_per_gate: u64::MAX, ..calibration }, 0), None);
    }

    #[test]
    fn frozen_batch_amortizes_cache_and_matches_full_oracle() {
        for depth in 2..=4 {
            let model = FiveCircleHierarchy::from_modes(
                &vec![GateKind::Majority; depth],
                LeafRouting::SeededShuffle(13),
            )
            .unwrap();
            let input = planes(model.input_bits());
            let proposals: Vec<_> = (0..12)
                .map(|i| PackedIntervention::LeafSwap {
                    first: i,
                    second: i + 1,
                })
                .collect();
            let expected: Vec<u64> = proposals
                .iter()
                .map(|p| {
                    let PackedIntervention::LeafSwap { first, second } = *p else {
                        unreachable!()
                    };
                    let mut routing = model.routing().to_vec();
                    routing.swap(first, second);
                    model
                        .rerouted(routing)
                        .unwrap()
                        .infer_bitplanes(&input)
                        .unwrap()
                })
                .collect();
            let fast = evaluate_frozen_batch(
                &model,
                &input,
                &proposals,
                EngineCalibration {
                    full_ns_per_gate: 100,
                    incremental_ns_per_gate: 1,
                    cache_build_ns: 0,
                },
            )
            .unwrap();
            assert_eq!(fast.outputs, expected);
            assert_eq!(fast.engine, Engine::Incremental);
            assert_eq!(fast.cache_build_gates, model.neuron_count());
            let slow = evaluate_frozen_batch(
                &model,
                &input,
                &proposals,
                EngineCalibration {
                    full_ns_per_gate: 1,
                    incremental_ns_per_gate: 1000,
                    cache_build_ns: 0,
                },
            )
            .unwrap();
            assert_eq!(slow.outputs, expected);
            assert_eq!(slow.engine, Engine::Full);
            assert_eq!(
                model.infer_bitplanes(&input).unwrap(),
                model.infer_bitplanes(&input).unwrap()
            );
        }
    }

    #[test]
    fn cold_cache_preflight_skips_non_amortizable_batch_and_matches_oracle() {
        let model = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity, GateKind::Or],
            LeafRouting::SeededShuffle(19),
        )
        .unwrap();
        let input = planes(model.input_bits());
        let proposal = PackedIntervention::LeafSwap {
            first: 0,
            second: 1,
        };
        let mut routing = model.routing().to_vec();
        routing.swap(0, 1);
        let oracle = model.rerouted(routing).unwrap().infer_bitplanes(&input).unwrap();
        for overhead in [0, 1, 100_000] {
            let outcome = evaluate_frozen_batch(
                &model,
                &input,
                &[proposal],
                EngineCalibration {
                    full_ns_per_gate: 10,
                    incremental_ns_per_gate: 1,
                    cache_build_ns: overhead,
                },
            )
            .unwrap();
            assert_eq!(outcome.outputs, vec![oracle]);
            assert_eq!(outcome.engine, Engine::Full);
            assert_eq!(outcome.cache_build_gates, 0);
            assert_eq!(outcome.incremental_gates, 0);
        }
    }

    #[test]
    fn frozen_batch_rejects_invalid_proposals() {
        let model = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity],
            LeafRouting::Contiguous,
        )
        .unwrap();
        let input = planes(model.input_bits());
        let invalid = [PackedIntervention::LeafSwap {
            first: 0,
            second: 25,
        }];
        assert!(matches!(
            evaluate_frozen_batch(
                &model,
                &input,
                &invalid,
                EngineCalibration {
                    full_ns_per_gate: 1,
                    incremental_ns_per_gate: 1,
                    cache_build_ns: 0
                }
            ),
            Err(HierarchyError::InvalidRouting)
        ));
    }

    #[test]
    fn sequential_swaps_match_independent_full_oracle() {
        for depth in 2..=4 {
            let mut model = FiveCircleHierarchy::from_modes(
                &vec![GateKind::Majority; depth],
                LeafRouting::SeededShuffle(77),
            )
            .unwrap();
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
                    &mut model,
                    &inputs,
                    PackedIntervention::LeafSwap { first, second },
                    accept,
                    EngineCalibration {
                        full_ns_per_gate: 10,
                        incremental_ns_per_gate: 1,
                        cache_build_ns: 0,
                    },
                )
                .unwrap();
                assert_eq!(result.engine, Engine::Full);
                assert_eq!(result.cone_gates, 0);
                assert_eq!(result.output, expected.infer_bitplanes(&inputs).unwrap());
                assert_eq!(
                    model.infer_bitplanes(&inputs).unwrap(),
                    if accept { result.output } else { before }
                );
            }
        }
    }

    #[test]
    fn hidden_parameter_proposals_match_full_oracle() {
        let mut model = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity, GateKind::Or],
            LeafRouting::Contiguous,
        )
        .unwrap();
        let inputs = planes(model.input_bits());
        for neuron in [0, 3, 12] {
            let old = model.layer_model(0, neuron).unwrap();
            let mut circles = *old.circles();
            circles[0].radius += 1;
            let next = FiveCircle::new(circles, old.bias() + 1).unwrap();
            let expected = model.with_neuron_parameters(0, neuron, next).unwrap();
            let result = evaluate_and_apply(
                &mut model,
                &inputs,
                PackedIntervention::Parameter {
                    layer: 0,
                    neuron,
                    model: next,
                },
                true,
                EngineCalibration {
                    full_ns_per_gate: 0,
                    incremental_ns_per_gate: 0,
                    cache_build_ns: 0,
                },
            )
            .unwrap();
            assert_eq!(result.engine, Engine::Full);
            assert_eq!(result.output, expected.infer_bitplanes(&inputs).unwrap());
        }
    }
}

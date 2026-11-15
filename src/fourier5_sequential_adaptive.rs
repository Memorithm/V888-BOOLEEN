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

/// Estimate the minimum independent batch size that amortizes a cold cache.
/// Costs must be measured externally; this does not claim a speedup.
pub fn cold_cache_break_even_batch(
    baseline_gates: usize,
    cone_gates_per_proposal: usize,
    calibration: EngineCalibration,
    minimum_saving_ppm: u32,
) -> Option<u64> {
    if calibration.full_ns_per_gate == 0 || calibration.incremental_ns_per_gate == 0 {
        return None;
    }
    let full = (baseline_gates as u128).checked_mul(u128::from(calibration.full_ns_per_gate))?;
    let incremental = (cone_gates_per_proposal as u128)
        .checked_mul(u128::from(calibration.incremental_ns_per_gate))?;
    let retained = 1_000_000_u128.checked_sub(u128::from(minimum_saving_ppm.min(1_000_000)))?;
    let allowed_full = full.checked_mul(retained)?;
    let incremental_scaled = incremental.checked_mul(1_000_000)?;
    if allowed_full <= incremental_scaled {
        return None;
    }
    let build = full
        .checked_add(u128::from(calibration.cache_build_ns))?
        .checked_mul(1_000_000)?;
    let saving = allowed_full.checked_sub(incremental_scaled)?;
    // Strict saving criterion: n * saving > build.
    u64::try_from(build / saving + 1).ok()
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
    evaluate_frozen_batch_with_margin(baseline, planes, proposals, calibration, 0)
}

/// Require a minimum predicted saving before selecting Incremental.
/// `minimum_saving_ppm` is clamped to [0, 1_000_000]. A stricter
/// margin can only select Full more often; outputs must remain identical.
pub fn evaluate_frozen_batch_with_margin(
    baseline: &FiveCircleHierarchy,
    planes: &[u64],
    proposals: &[PackedIntervention],
    calibration: EngineCalibration,
    minimum_saving_ppm: u32,
) -> Result<BatchEvaluation, HierarchyError> {
    let retained_ppm = 1_000_000_u128.saturating_sub(u128::from(minimum_saving_ppm.min(1_000_000)));
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
    // A cold cache must evaluate every baseline gate before any proposal.
    // If even that lower bound cannot beat Full, avoid allocating the cache.
    // This is a conservative rejection only: it cannot change predictions.
    let full_cost = (full_gates as u128).saturating_mul(calibration.full_ns_per_gate as u128);
    let cold_lower_bound = (baseline.neuron_count() as u128)
        .saturating_mul(calibration.full_ns_per_gate as u128)
        .saturating_add(calibration.cache_build_ns as u128);
    if cold_lower_bound.saturating_mul(1_000_000) >= full_cost.saturating_mul(retained_ppm) {
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
    let cache = FiveCircleIncrementalCache::new(baseline, planes)?;
    let build_cost = (cache.build_gate_evaluations() as u128)
        .saturating_mul(calibration.full_ns_per_gate as u128)
        .saturating_add(calibration.cache_build_ns as u128);
    // Stop computing counterfactual cones as soon as the measured-cost
    // budget is exhausted. Later cones cannot make Incremental profitable.
    // This counts only cones actually evaluated, not estimated future cones.
    let allowed_cost = full_cost.saturating_mul(retained_ppm) / 1_000_000;
    let mut cone_gates = 0_usize;
    let mut outputs_incremental = Vec::with_capacity(proposals.len());
    let mut exhausted = build_cost >= allowed_cost;
    if !exhausted {
        for &proposal in proposals {
            let delta = cache.evaluate(proposal)?;
            cone_gates = cone_gates.saturating_add(delta.gate_evaluations);
            let cost = build_cost.saturating_add(
                (cone_gates as u128).saturating_mul(calibration.incremental_ns_per_gate as u128),
            );
            if cost >= allowed_cost {
                exhausted = true;
                break;
            }
            outputs_incremental.push(delta.output);
        }
    }
    let use_incremental = !exhausted;
    let outputs = if use_incremental {
        outputs_incremental
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
    fn cold_cache_break_even_is_strict_and_conservative() {
        let c = EngineCalibration {
            full_ns_per_gate: 10,
            incremental_ns_per_gate: 1,
            cache_build_ns: 0,
        };
        assert_eq!(cold_cache_break_even_batch(10, 10, c, 0), Some(2));
        assert_eq!(cold_cache_break_even_batch(10, 10, c, 500_000), Some(3));
        assert_eq!(cold_cache_break_even_batch(10, 10, c, 1_000_000), None);
        assert_eq!(cold_cache_break_even_batch(10, 100, c, 0), None);
        assert_eq!(
            cold_cache_break_even_batch(
                10,
                1,
                EngineCalibration {
                    full_ns_per_gate: 0,
                    ..c
                },
                0
            ),
            None
        );
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
    fn cold_cache_lower_bound_skips_unprofitable_build_without_changing_oracle() {
        let model = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity, GateKind::Or],
            LeafRouting::SeededShuffle(31),
        )
        .unwrap();
        let input = planes(model.input_bits());
        for count in [1, 2, 4] {
            let proposals: Vec<_> = (0..count)
                .map(|i| PackedIntervention::LeafSwap {
                    first: i,
                    second: i + 1,
                })
                .collect();
            let expected: Vec<_> = proposals
                .iter()
                .map(|proposal| {
                    let PackedIntervention::LeafSwap { first, second } = *proposal else {
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
            let result = evaluate_frozen_batch(
                &model,
                &input,
                &proposals,
                EngineCalibration {
                    full_ns_per_gate: 10,
                    incremental_ns_per_gate: 1,
                    cache_build_ns: 1_000_000,
                },
            )
            .unwrap();
            assert_eq!(result.outputs, expected);
            assert_eq!(result.engine, Engine::Full);
            assert_eq!(result.cache_build_gates, 0);
            assert_eq!(result.incremental_gates, 0);
        }
    }

    #[test]
    fn safety_margin_changes_engine_not_oracle_predictions() {
        let model = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity, GateKind::Or],
            LeafRouting::SeededShuffle(43),
        )
        .unwrap();
        let input = planes(model.input_bits());
        let proposals: Vec<_> = (0..12)
            .map(|i| PackedIntervention::LeafSwap {
                first: i,
                second: i + 1,
            })
            .collect();
        let calibration = EngineCalibration {
            full_ns_per_gate: 100,
            incremental_ns_per_gate: 1,
            cache_build_ns: 0,
        };
        let normal = evaluate_frozen_batch(&model, &input, &proposals, calibration).unwrap();
        let conservative =
            evaluate_frozen_batch_with_margin(&model, &input, &proposals, calibration, 1_000_000)
                .unwrap();
        assert_eq!(normal.engine, Engine::Incremental);
        assert_eq!(conservative.engine, Engine::Full);
        assert_eq!(normal.outputs, conservative.outputs);
        assert_eq!(conservative.cache_build_gates, 0);
    }

    #[test]
    fn heterogeneous_cones_stop_early_and_keep_full_oracle_outputs() {
        let model = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity, GateKind::Or],
            LeafRouting::SeededShuffle(53),
        )
        .unwrap();
        let input = planes(model.input_bits());
        let proposals: Vec<_> = (0..24)
            .map(|i| PackedIntervention::LeafSwap {
                first: i,
                second: i + 1,
            })
            .collect();
        let expected: Vec<_> = proposals
            .iter()
            .map(|proposal| {
                let PackedIntervention::LeafSwap { first, second } = *proposal else {
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
        let result = evaluate_frozen_batch(
            &model,
            &input,
            &proposals,
            EngineCalibration {
                full_ns_per_gate: 10,
                incremental_ns_per_gate: 1_000_000,
                cache_build_ns: 0,
            },
        )
        .unwrap();
        assert_eq!(result.engine, Engine::Full);
        assert_eq!(result.outputs, expected);
        assert_eq!(result.cache_build_gates, model.neuron_count());
        // The first cone exhausts the entire budget; the rest are skipped.
        assert!(result.incremental_gates > 0);
        assert!(result.incremental_gates < model.neuron_count() * proposals.len());
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

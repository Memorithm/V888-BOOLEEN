//! Persistent exact bit-sliced cache for accepted sequential interventions.
//! Synthetic five-circle circuits only; no BANC anatomical claims.
use crate::fourier5::CIRCLES;
use crate::fourier5_hierarchy::compiled_truth_table;
use crate::fourier5_hierarchy::{shannon_64, FiveCircleHierarchy, HierarchyError};
use crate::fourier5_incremental::PackedIntervention;
use std::collections::{BTreeMap, BTreeSet};

pub struct PersistentSession {
    model: FiveCircleHierarchy,
    stages: Vec<Vec<u64>>,
    pub rebuilt_gates: usize,
    pub updated_gates: usize,
}

impl PersistentSession {
    pub fn new(model: FiveCircleHierarchy, planes: Vec<u64>) -> Result<Self, HierarchyError> {
        if planes.len() != model.input_bits() {
            return Err(HierarchyError::InvalidInput);
        }
        let mut stages: Vec<Vec<u64>> = Vec::with_capacity(model.depth() + 1);
        stages.push(
            model
                .routing()
                .iter()
                .map(|&src| planes[usize::from(src)])
                .collect(),
        );
        for layer in 0..model.depth() {
            let mut next = Vec::with_capacity(stages[layer].len() / CIRCLES);
            for neuron in 0..stages[layer].len() / CIRCLES {
                let input = std::array::from_fn(|bit| stages[layer][neuron * CIRCLES + bit]);
                next.push(shannon_64(model.gate_truth(layer, neuron)?, input));
            }
            stages.push(next);
        }
        let rebuilt_gates = model.neuron_count();
        Ok(Self {
            model,
            stages,
            rebuilt_gates,
            updated_gates: 0,
        })
    }

    pub fn model(&self) -> &FiveCircleHierarchy {
        &self.model
    }
    pub fn output(&self) -> u64 {
        self.stages.last().expect("nonempty hierarchy")[0]
    }

    /// Proposals are evaluated against the current accepted model, without
    /// mutating the session. The independent full inference is the oracle.
    pub fn propose(&self, proposal: PackedIntervention) -> Result<u64, HierarchyError> {
        let mut changes = BTreeMap::new();
        let start_layer = match proposal {
            PackedIntervention::LeafSwap { first, second } => {
                if first >= self.model.input_bits()
                    || second >= self.model.input_bits()
                    || first == second
                {
                    return Err(HierarchyError::InvalidRouting);
                }
                for neuron in BTreeSet::from([first / CIRCLES, second / CIRCLES]) {
                    let input = std::array::from_fn(|bit| {
                        let position = neuron * CIRCLES + bit;
                        let source = if position == first {
                            second
                        } else if position == second {
                            first
                        } else {
                            position
                        };
                        self.stages[0][source]
                    });
                    changes.insert(neuron, shannon_64(self.model.gate_truth(0, neuron)?, input));
                }
                0
            }
            PackedIntervention::Parameter {
                layer,
                neuron,
                model,
            } => {
                let old = self.model.layer_model(layer, neuron)?;
                if old
                    .circles()
                    .iter()
                    .zip(model.circles())
                    .any(|(a, b)| a.mask != b.mask)
                {
                    return Err(HierarchyError::InvalidLayerShape);
                }
                let input = std::array::from_fn(|bit| self.stages[layer][neuron * CIRCLES + bit]);
                changes.insert(neuron, shannon_64(compiled_truth_table(model), input));
                layer
            }
        };
        for layer in start_layer + 1..self.model.depth() {
            let parents: BTreeSet<usize> = changes.keys().map(|&index| index / CIRCLES).collect();
            let mut next = BTreeMap::new();
            for neuron in parents {
                let input = std::array::from_fn(|bit| {
                    let child = neuron * CIRCLES + bit;
                    changes
                        .get(&child)
                        .copied()
                        .unwrap_or(self.stages[layer][child])
                });
                next.insert(
                    neuron,
                    shannon_64(self.model.gate_truth(layer, neuron)?, input),
                );
            }
            changes = next;
        }
        Ok(changes[&0])
    }

    /// Exact per-lane Boolean influence of a proposed intervention on the
    /// current output. A set bit means that lane's final prediction flips.
    /// The session remains unchanged, including on invalid proposals.
    /// This is an output sensitivity, not a hidden teacher activation.
    pub fn output_flip_mask(&self, proposal: PackedIntervention) -> Result<u64, HierarchyError> {
        Ok(self.output() ^ self.propose(proposal)?)
    }

    /// Evaluate a bounded sequence of independent counterfactuals against
    /// the same accepted baseline. No proposal is committed by this method.
    /// The caller supplies the bound to avoid unbounded batch allocation.
    pub fn output_flip_masks(
        &self,
        proposals: &[PackedIntervention],
        max_proposals: usize,
    ) -> Result<Vec<u64>, HierarchyError> {
        if proposals.len() > max_proposals {
            return Err(HierarchyError::InvalidInput);
        }
        proposals
            .iter()
            .copied()
            .map(|p| self.output_flip_mask(p))
            .collect()
    }

    /// Rank independent interventions by output sensitivity per affected gate.
    /// This is an unsupervised diagnostic, not a learning objective or teacher tap.
    /// Ties prefer fewer gates, then the earliest proposal. No state is committed.
    pub fn most_efficient_output_flip(
        &self,
        proposals: &[PackedIntervention],
        max_proposals: usize,
    ) -> Result<Option<(usize, u32, usize)>, HierarchyError> {
        if proposals.len() > max_proposals {
            return Err(HierarchyError::InvalidInput);
        }
        let mut best: Option<(usize, u32, usize)> = None;
        for (index, &proposal) in proposals.iter().enumerate() {
            let mut affected = BTreeSet::new();
            let start_layer = match proposal {
                PackedIntervention::LeafSwap { first, second } => {
                    if first >= self.model.input_bits()
                        || second >= self.model.input_bits()
                        || first == second
                    {
                        return Err(HierarchyError::InvalidRouting);
                    }
                    affected.insert(first / CIRCLES);
                    affected.insert(second / CIRCLES);
                    0
                }
                PackedIntervention::Parameter { layer, neuron, model } => {
                    let old = self.model.layer_model(layer, neuron)?;
                    if old.circles().iter().zip(model.circles()).any(|(a, b)| a.mask != b.mask) {
                        return Err(HierarchyError::InvalidLayerShape);
                    }
                    affected.insert(neuron);
                    layer
                }
            };
            let mut gates = 0usize;
            for _ in start_layer..self.model.depth() {
                gates = gates.checked_add(affected.len()).ok_or(HierarchyError::InvalidInput)?;
                affected = affected.into_iter().map(|node| node / CIRCLES).collect();
            }
            let flips = self.output_flip_mask(proposal)?.count_ones();
            let better = match best {
                None => true,
                Some((_, best_flips, best_gates)) => {
                    (flips as u128) * (best_gates as u128)
                        > (best_flips as u128) * (gates as u128)
                        || ((flips as u128) * (best_gates as u128)
                            == (best_flips as u128) * (gates as u128)
                            && (flips > best_flips
                                || (flips == best_flips && gates < best_gates)))
                }
            };
            if better {
                best = Some((index, flips, gates));
            }
        }
        Ok(best)
    }

    /// Select a counterfactual engine using externally measured nanoseconds per
    /// gate. Cache construction is already paid for by this persistent session.
    /// A zero calibration falls back to the independent full inference.
    /// The returned gate budget is a conservative upper bound on affected gates.
    pub fn propose_adaptive(
        &self,
        proposal: PackedIntervention,
        calibration: crate::fourier5_sequential_adaptive::EngineCalibration,
    ) -> Result<crate::fourier5_sequential_adaptive::ProposalResult, HierarchyError> {
        use crate::fourier5_sequential_adaptive::{Engine, ProposalResult};
        let candidate = match proposal {
            PackedIntervention::LeafSwap { first, second } => {
                if first >= self.model.input_bits()
                    || second >= self.model.input_bits()
                    || first == second
                {
                    return Err(HierarchyError::InvalidRouting);
                }
                let mut routing = self.model.routing().to_vec();
                routing.swap(first, second);
                self.model.rerouted(routing)?
            }
            PackedIntervention::Parameter {
                layer,
                neuron,
                model,
            } => {
                let old = self.model.layer_model(layer, neuron)?;
                if old
                    .circles()
                    .iter()
                    .zip(model.circles())
                    .any(|(a, b)| a.mask != b.mask)
                {
                    return Err(HierarchyError::InvalidLayerShape);
                }
                self.model.with_neuron_parameters(layer, neuron, model)?
            }
        };
        let mut affected = BTreeSet::new();
        let start_layer = match proposal {
            PackedIntervention::LeafSwap { first, second } => {
                affected.insert(first / CIRCLES);
                affected.insert(second / CIRCLES);
                0
            }
            PackedIntervention::Parameter { layer, neuron, .. } => {
                affected.insert(neuron);
                layer
            }
        };
        let mut cone_gates = 0usize;
        for _ in start_layer..self.model.depth() {
            cone_gates = cone_gates.saturating_add(affected.len());
            affected = affected.into_iter().map(|index| index / CIRCLES).collect();
        }
        let full_gates = self.model.neuron_count();
        let full_cost = (full_gates as u128).saturating_mul(calibration.full_ns_per_gate as u128);
        let incremental_cost =
            (cone_gates as u128).saturating_mul(calibration.incremental_ns_per_gate as u128);
        let use_incremental = calibration.full_ns_per_gate > 0
            && calibration.incremental_ns_per_gate > 0
            && incremental_cost < full_cost;
        let output = if use_incremental {
            self.propose(proposal)?
        } else {
            // The full path is independent of the persistent cache.
            // Original routed input planes are reconstructed from the current
            // routed stage, using the inverse permutation.
            let mut planes = vec![0u64; self.model.input_bits()];
            for (position, &source) in self.model.routing().iter().enumerate() {
                planes[usize::from(source)] = self.stages[0][position];
            }
            candidate.infer_bitplanes(&planes)?
        };
        Ok(ProposalResult {
            output,
            engine: if use_incremental {
                Engine::Incremental
            } else {
                Engine::Full
            },
            full_gates,
            cone_gates,
            accepted: false,
        })
    }

    /// Measure both exact engines on this frozen session and choose the faster
    /// observed path. Timing includes candidate construction and input
    /// reconstruction for Full; neither path commits an intervention.
    ///
    /// This is an on-demand diagnostic selector, not a claim that timing every
    /// proposal is faster than a pre-calibrated policy. Measurements are local
    /// to the caller's CPU and must not be reused across machines.
    pub fn propose_measured(
        &self,
        proposal: PackedIntervention,
        repetitions: usize,
    ) -> Result<crate::fourier5_sequential_adaptive::ProposalResult, HierarchyError> {
        use crate::fourier5_sequential_adaptive::{Engine, ProposalResult};
        use std::time::Instant;

        if repetitions == 0 || repetitions > 64 {
            return Err(HierarchyError::InvalidInput);
        }
        // Validate the intervention and construct the independent full oracle.
        let candidate = match proposal {
            PackedIntervention::LeafSwap { first, second } => {
                if first >= self.model.input_bits()
                    || second >= self.model.input_bits()
                    || first == second
                {
                    return Err(HierarchyError::InvalidRouting);
                }
                let mut routing = self.model.routing().to_vec();
                routing.swap(first, second);
                self.model.rerouted(routing)?
            }
            PackedIntervention::Parameter {
                layer,
                neuron,
                model,
            } => {
                let old = self.model.layer_model(layer, neuron)?;
                if old
                    .circles()
                    .iter()
                    .zip(model.circles())
                    .any(|(a, b)| a.mask != b.mask)
                {
                    return Err(HierarchyError::InvalidLayerShape);
                }
                self.model.with_neuron_parameters(layer, neuron, model)?
            }
        };
        let mut planes = vec![0u64; self.model.input_bits()];
        for (position, &source) in self.model.routing().iter().enumerate() {
            planes[usize::from(source)] = self.stages[0][position];
        }
        let mut full_ns = 0u128;
        let mut incremental_ns = 0u128;
        let mut full_output = 0;
        let mut incremental_output;
        // Alternate measurement order to reduce systematic warm-cache bias.
        for round in 0..repetitions {
            if round % 2 == 0 {
                let start = Instant::now();
                full_output =
                    std::hint::black_box(candidate.infer_bitplanes(std::hint::black_box(&planes))?);
                full_ns = full_ns.saturating_add(start.elapsed().as_nanos());
                let start = Instant::now();
                incremental_output =
                    std::hint::black_box(self.propose(std::hint::black_box(proposal))?);
                incremental_ns = incremental_ns.saturating_add(start.elapsed().as_nanos());
            } else {
                let start = Instant::now();
                incremental_output =
                    std::hint::black_box(self.propose(std::hint::black_box(proposal))?);
                incremental_ns = incremental_ns.saturating_add(start.elapsed().as_nanos());
                let start = Instant::now();
                full_output =
                    std::hint::black_box(candidate.infer_bitplanes(std::hint::black_box(&planes))?);
                full_ns = full_ns.saturating_add(start.elapsed().as_nanos());
            }
            if full_output != incremental_output {
                return Err(HierarchyError::InvalidInput);
            }
        }
        let use_incremental = incremental_ns < full_ns;
        Ok(ProposalResult {
            output: full_output,
            engine: if use_incremental {
                Engine::Incremental
            } else {
                Engine::Full
            },
            full_gates: self.model.neuron_count(),
            // This diagnostic does not claim a gate-count estimate.
            cone_gates: 0,
            accepted: false,
        })
    }

    /// Independently verify an adaptive decision against a complete inference.
    /// Diagnostic safety mode: doubles inference work and makes no speed claim.
    /// A mismatch fails closed without mutating the accepted model or cache.
    pub fn propose_verified_adaptive(
        &self,
        proposal: PackedIntervention,
        calibration: crate::fourier5_sequential_adaptive::EngineCalibration,
    ) -> Result<crate::fourier5_sequential_adaptive::ProposalResult, HierarchyError> {
        let selected = self.propose_adaptive(proposal, calibration)?;
        let mut planes = vec![0u64; self.model.input_bits()];
        for (position, &source) in self.model.routing().iter().enumerate() {
            planes[usize::from(source)] = self.stages[0][position];
        }
        let candidate = match proposal {
            PackedIntervention::LeafSwap { first, second } => {
                let mut routing = self.model.routing().to_vec();
                routing.swap(first, second);
                self.model.rerouted(routing)?
            }
            PackedIntervention::Parameter {
                layer,
                neuron,
                model,
            } => self.model.with_neuron_parameters(layer, neuron, model)?,
        };
        if selected.output != candidate.infer_bitplanes(&planes)? {
            return Err(HierarchyError::InvalidInput);
        }
        Ok(selected)
    }

    /// Atomic acceptance: validate and construct the candidate first, then
    /// refresh only changed leaves/ancestors. Failed changes leave state intact.
    pub fn accept(&mut self, proposal: PackedIntervention) -> Result<usize, HierarchyError> {
        let (candidate, start_layer, mut affected) = match proposal {
            PackedIntervention::LeafSwap { first, second } => {
                if first >= self.model.input_bits()
                    || second >= self.model.input_bits()
                    || first == second
                {
                    return Err(HierarchyError::InvalidRouting);
                }
                let mut routing = self.model.routing().to_vec();
                routing.swap(first, second);
                let candidate = self.model.rerouted(routing)?;
                (
                    candidate,
                    0,
                    BTreeSet::from([first / CIRCLES, second / CIRCLES]),
                )
            }
            PackedIntervention::Parameter {
                layer,
                neuron,
                model,
            } => {
                let old = self.model.layer_model(layer, neuron)?;
                if old
                    .circles()
                    .iter()
                    .zip(model.circles())
                    .any(|(a, b)| a.mask != b.mask)
                {
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
                self.stages[layer + 1][neuron] =
                    shannon_64(candidate.gate_truth(layer, neuron)?, input);
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
    use crate::fourier5::FiveCircle;
    use crate::fourier5_hierarchy::{GateKind, LeafRouting};

    #[test]
    fn flip_per_cone_ranking_matches_full_oracle_and_preserves_cache() {
        let model = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity, GateKind::Or],
            LeafRouting::SeededShuffle(91),
        )
        .unwrap();
        let planes: Vec<u64> = (0..model.input_bits())
            .map(|i| (i as u64 + 31).wrapping_mul(0x9e3779b97f4a7c15))
            .collect();
        let session = PersistentSession::new(model.clone(), planes.clone()).unwrap();
        let proposals = [
            PackedIntervention::LeafSwap { first: 0, second: 1 },
            PackedIntervention::LeafSwap { first: 0, second: 17 },
            PackedIntervention::LeafSwap { first: 12, second: 79 },
        ];
        let baseline = model.infer_bitplanes(&planes).unwrap();
        let mut reference: Option<(usize, u32, usize)> = None;
        for (index, proposal) in proposals.iter().enumerate() {
            let PackedIntervention::LeafSwap { first, second } = *proposal else {
                unreachable!()
            };
            let mut routing = model.routing().to_vec();
            routing.swap(first, second);
            let full = model.rerouted(routing).unwrap().infer_bitplanes(&planes).unwrap();
            let flips = (baseline ^ full).count_ones();
            let mut nodes = BTreeSet::from([first / CIRCLES, second / CIRCLES]);
            let mut gates = 0usize;
            for _ in 0..model.depth() {
                gates += nodes.len();
                nodes = nodes.into_iter().map(|n| n / CIRCLES).collect();
            }
            let better = match reference {
                None => true,
                Some((_, old_flips, old_gates)) => {
                    (flips as u128) * (old_gates as u128)
                        > (old_flips as u128) * (gates as u128)
                        || ((flips as u128) * (old_gates as u128)
                            == (old_flips as u128) * (gates as u128)
                            && (flips > old_flips
                                || (flips == old_flips && gates < old_gates)))
                }
            };
            if better {
                reference = Some((index, flips, gates));
            }
        }
        assert_eq!(
            session.most_efficient_output_flip(&proposals, 3).unwrap(),
            reference
        );
        assert_eq!(session.most_efficient_output_flip(&[], 0).unwrap(), None);
        assert!(matches!(
            session.most_efficient_output_flip(&proposals, 2),
            Err(HierarchyError::InvalidInput)
        ));
        assert!(matches!(
            session.most_efficient_output_flip(
                &[PackedIntervention::LeafSwap { first: 0, second: model.input_bits() }],
                1
            ),
            Err(HierarchyError::InvalidRouting)
        ));
        assert_eq!(session.output(), baseline);
        assert_eq!(session.updated_gates, 0);
    }

    #[test]
    fn verified_adaptive_matches_full_after_sequential_acceptances() {
        use crate::fourier5_sequential_adaptive::EngineCalibration;
        let base = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity, GateKind::Or],
            LeafRouting::SeededShuffle(47),
        )
        .unwrap();
        let planes: Vec<u64> = (0..base.input_bits())
            .map(|i| (i as u64 + 5).wrapping_mul(0x9e3779b97f4a7c15))
            .collect();
        let mut session = PersistentSession::new(base, planes.clone()).unwrap();
        for step in 0..16 {
            let proposal = PackedIntervention::LeafSwap {
                first: step,
                second: step + 13,
            };
            for (full, incremental) in [(1, 10000), (10000, 1), (0, 0)] {
                let result = session
                    .propose_verified_adaptive(
                        proposal,
                        EngineCalibration {
                            full_ns_per_gate: full,
                            incremental_ns_per_gate: incremental,
                            cache_build_ns: 0,
                        },
                    )
                    .unwrap();
                let mut routing = session.model().routing().to_vec();
                routing.swap(step, step + 13);
                let oracle = session
                    .model()
                    .rerouted(routing)
                    .unwrap()
                    .infer_bitplanes(&planes)
                    .unwrap();
                assert_eq!(result.output, oracle);
                assert!(!result.accepted);
            }
            if step % 2 == 0 {
                session.accept(proposal).unwrap();
                assert_eq!(
                    session.output(),
                    session.model().infer_bitplanes(&planes).unwrap()
                );
            }
        }
    }

    #[test]
    fn measured_selector_preserves_exact_output_and_rejects_invalid_budget() {
        let model = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity],
            LeafRouting::SeededShuffle(19),
        )
        .unwrap();
        let planes: Vec<u64> = (0..model.input_bits())
            .map(|i| (i as u64 + 3).wrapping_mul(0x9e3779b97f4a7c15))
            .collect();
        let session = PersistentSession::new(model.clone(), planes.clone()).unwrap();
        let proposal = PackedIntervention::LeafSwap {
            first: 0,
            second: 7,
        };
        let mut routing = model.routing().to_vec();
        routing.swap(0, 7);
        let expected = model
            .rerouted(routing)
            .unwrap()
            .infer_bitplanes(&planes)
            .unwrap();
        let result = session.propose_measured(proposal, 3).unwrap();
        assert_eq!(result.output, expected);
        assert!(!result.accepted);
        assert_eq!(session.output(), model.infer_bitplanes(&planes).unwrap());
        assert!(matches!(
            session.propose_measured(proposal, 0),
            Err(HierarchyError::InvalidInput)
        ));
        assert!(matches!(
            session.propose_measured(proposal, 65),
            Err(HierarchyError::InvalidInput)
        ));
    }

    #[test]
    fn adaptive_persistent_engine_matches_full_after_sequential_acceptance() {
        use crate::fourier5_sequential_adaptive::{Engine, EngineCalibration};
        let model = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity, GateKind::Or],
            LeafRouting::SeededShuffle(37),
        )
        .unwrap();
        let planes: Vec<u64> = (0..model.input_bits())
            .map(|i| (i as u64 + 11).wrapping_mul(0x9e3779b97f4a7c15))
            .collect();
        let mut session = PersistentSession::new(model, planes.clone()).unwrap();
        for step in 0..12 {
            let proposal = PackedIntervention::LeafSwap {
                first: step,
                second: step + 7,
            };
            for (calibration, expected_engine) in [
                (
                    EngineCalibration {
                        full_ns_per_gate: 100,
                        incremental_ns_per_gate: 1,
                        cache_build_ns: 0,
                    },
                    Engine::Incremental,
                ),
                (
                    EngineCalibration {
                        full_ns_per_gate: 1,
                        incremental_ns_per_gate: 10000,
                        cache_build_ns: 0,
                    },
                    Engine::Full,
                ),
                (
                    EngineCalibration {
                        full_ns_per_gate: 0,
                        incremental_ns_per_gate: 0,
                        cache_build_ns: 0,
                    },
                    Engine::Full,
                ),
            ] {
                let result = session.propose_adaptive(proposal, calibration).unwrap();
                let mut routing = session.model().routing().to_vec();
                routing.swap(step, step + 7);
                let oracle = session
                    .model()
                    .rerouted(routing)
                    .unwrap()
                    .infer_bitplanes(&planes)
                    .unwrap();
                assert_eq!(result.output, oracle);
                assert_eq!(result.engine, expected_engine);
                assert!(!result.accepted);
            }
            if step % 2 == 0 {
                session.accept(proposal).unwrap();
                assert_eq!(
                    session.output(),
                    session.model().infer_bitplanes(&planes).unwrap()
                );
            }
        }
    }

    #[test]
    fn output_sensitivity_matches_independent_full_inference_and_preserves_state() {
        let base = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Majority, GateKind::Majority],
            LeafRouting::SeededShuffle(23),
        )
        .unwrap();
        let planes: Vec<u64> = (0..base.input_bits())
            .map(|i| (i as u64 + 7).wrapping_mul(0xd6e8feb86659fd93))
            .collect();
        let session = PersistentSession::new(base.clone(), planes.clone()).unwrap();
        let baseline = base.infer_bitplanes(&planes).unwrap();
        let proposals = [
            PackedIntervention::LeafSwap {
                first: 0,
                second: 1,
            },
            PackedIntervention::LeafSwap {
                first: 3,
                second: 42,
            },
        ];
        let actual = session.output_flip_masks(&proposals, 2).unwrap();
        for (proposal, &mask) in proposals.iter().zip(&actual) {
            let PackedIntervention::LeafSwap { first, second } = *proposal else {
                unreachable!()
            };
            let mut routing = base.routing().to_vec();
            routing.swap(first, second);
            let expected = base
                .rerouted(routing)
                .unwrap()
                .infer_bitplanes(&planes)
                .unwrap();
            assert_eq!(mask, baseline ^ expected);
        }
        assert_eq!(session.output(), baseline);
        assert_eq!(session.updated_gates, 0);
        assert!(matches!(
            session.output_flip_masks(&proposals, 1),
            Err(HierarchyError::InvalidInput)
        ));
        assert!(matches!(
            session.output_flip_mask(PackedIntervention::LeafSwap {
                first: 0,
                second: base.input_bits()
            }),
            Err(HierarchyError::InvalidRouting)
        ));
        assert_eq!(session.output(), baseline);
    }

    #[test]
    fn accepted_and_rejected_sequences_match_full_oracle() {
        for depth in 2..=4 {
            let base = FiveCircleHierarchy::from_modes(
                &vec![GateKind::Majority; depth],
                LeafRouting::SeededShuffle(71),
            )
            .unwrap();
            let planes: Vec<u64> = (0..base.input_bits())
                .map(|i| (i as u64 + 19).wrapping_mul(0x9e3779b97f4a7c15))
                .collect();
            let mut session = PersistentSession::new(base.clone(), planes.clone()).unwrap();
            let mut reference = base;
            for step in 0..24 {
                let proposal = if step % 3 == 0 {
                    let old = reference
                        .layer_model(0, step % (reference.input_bits() / CIRCLES))
                        .unwrap();
                    let mut circles = *old.circles();
                    circles[0].radius += 1;
                    PackedIntervention::Parameter {
                        layer: 0,
                        neuron: step % (reference.input_bits() / CIRCLES),
                        model: FiveCircle::new(circles, old.bias()).unwrap(),
                    }
                } else {
                    PackedIntervention::LeafSwap {
                        first: step,
                        second: step + 1,
                    }
                };
                let expected = match proposal {
                    PackedIntervention::LeafSwap { first, second } => {
                        let mut routing = reference.routing().to_vec();
                        routing.swap(first, second);
                        reference.rerouted(routing).unwrap()
                    }
                    PackedIntervention::Parameter {
                        layer,
                        neuron,
                        model,
                    } => reference
                        .with_neuron_parameters(layer, neuron, model)
                        .unwrap(),
                };
                assert_eq!(
                    session.propose(proposal).unwrap(),
                    expected.infer_bitplanes(&planes).unwrap()
                );
                if step % 4 != 1 {
                    let gates = session.accept(proposal).unwrap();
                    assert!(gates <= 2 * depth);
                    reference = expected;
                }
                assert_eq!(
                    session.output(),
                    reference.infer_bitplanes(&planes).unwrap()
                );
            }
            assert_eq!(session.rebuilt_gates, reference.neuron_count());
        }
    }
}

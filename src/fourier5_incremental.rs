//! Exact incremental bit-sliced evaluation of a five-circle Boolean tree.
//!
//! Baseline gates are evaluated once per immutable 64-pattern batch. A
//! counterfactual parameter change recomputes one gate plus its ancestors;
//! a leaf-source exchange recomputes at most two affected paths. Unchanged
//! siblings are read from the baseline cache without reevaluating them.
//! This is an exact Boolean circuit optimization, NOT approximate learning.
use std::collections::{BTreeMap, BTreeSet};

use crate::fourier5::{FiveCircle, CIRCLES};
use crate::fourier5_hierarchy::{
    compiled_truth_table, shannon_64, FiveCircleHierarchy, HierarchyError,
};

#[derive(Clone, Copy, Debug)]
pub enum PackedIntervention {
    Parameter {
        layer: usize,
        neuron: usize,
        model: FiveCircle,
    },
    LeafSwap {
        first: usize,
        second: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IncrementalEvaluation {
    pub output: u64,
    /// Exactly how many of the 32-entry five-circle gates were evaluated
    /// for this proposed intervention, across all 64 parallel lanes.
    pub gate_evaluations: usize,
}

/// Borrowed baseline model and source planes; only the cached gates/outputs
/// are owned. Rebuild this cache after accepting an intervention, NEVER
/// reuse stale layers across successive accepted network states.
pub struct FiveCircleIncrementalCache<'a> {
    base: &'a FiveCircleHierarchy,
    stages: Vec<Vec<u64>>,
    build_gate_evaluations: usize,
}

impl<'a> FiveCircleIncrementalCache<'a> {
    pub fn new(
        base: &'a FiveCircleHierarchy,
        planes: &[u64],
    ) -> Result<Self, HierarchyError> {
        if planes.len() != base.input_bits() {
            return Err(HierarchyError::InvalidInput);
        }
        let mut stages: Vec<Vec<u64>> = Vec::with_capacity(base.depth() + 1);
        stages.push(base.routing().iter().map(|&src| planes[usize::from(src)]).collect());
        let mut evaluations = 0;
        for layer in 0..base.depth() {
            let width = stages[layer].len() / CIRCLES;
            let mut next = Vec::with_capacity(width);
            for neuron in 0..width {
                let input = std::array::from_fn(|bit| stages[layer][neuron * CIRCLES + bit]);
                next.push(shannon_64(base.gate_truth(layer, neuron)?, input));
                evaluations += 1;
            }
            stages.push(next);
        }
        Ok(Self {
            base,
            stages,
            build_gate_evaluations: evaluations,
        })
    }

    #[must_use]
    pub fn baseline_output(&self) -> u64 {
        self.stages.last().expect("hierarchy has at least one layer")[0]
    }

    #[must_use]
    pub fn build_gate_evaluations(&self) -> usize {
        self.build_gate_evaluations
    }

    /// Evaluate only the affected Boolean dependency cone. A swap exchanges
    /// two *positions in the routed input vector*; its two bottom gates
    /// propagate independently until they meet at a common ancestor.
    pub fn evaluate(
        &self,
        change: PackedIntervention,
    ) -> Result<IncrementalEvaluation, HierarchyError> {
        let mut changes: BTreeMap<usize, u64> = BTreeMap::new();
        let start_layer;
        let mut evaluated = 0;

        match change {
            PackedIntervention::Parameter {
                layer,
                neuron,
                model,
            } => {
                let original = self.base.layer_model(layer, neuron)?;
                if original
                    .circles()
                    .iter()
                    .zip(model.circles())
                    .any(|(old, new)| old.mask != new.mask)
                {
                    return Err(HierarchyError::InvalidLayerShape);
                }
                let input = std::array::from_fn(|bit| self.stages[layer][neuron * CIRCLES + bit]);
                changes.insert(neuron, shannon_64(compiled_truth_table(model), input));
                evaluated += 1;
                start_layer = layer;
            }
            PackedIntervention::LeafSwap { first, second } => {
                if first >= self.base.input_bits()
                    || second >= self.base.input_bits()
                    || first == second
                {
                    return Err(HierarchyError::InvalidRouting);
                }
                let affected: BTreeSet<usize> =
                    [first / CIRCLES, second / CIRCLES].into_iter().collect();
                for neuron in affected {
                    let input = std::array::from_fn(|bit| {
                        let position = neuron * CIRCLES + bit;
                        let source_position = if position == first {
                            second
                        } else if position == second {
                            first
                        } else {
                            position
                        };
                        self.stages[0][source_position]
                    });
                    changes.insert(neuron, shannon_64(self.base.gate_truth(0, neuron)?, input));
                    evaluated += 1;
                }
                start_layer = 0;
            }
        }

        for layer in start_layer + 1..self.base.depth() {
            let parents: BTreeSet<usize> = changes.keys().map(|&index| index / CIRCLES).collect();
            let mut updated = BTreeMap::new();
            for neuron in parents {
                let input = std::array::from_fn(|bit| {
                    let child = neuron * CIRCLES + bit;
                    changes
                        .get(&child)
                        .copied()
                        .unwrap_or(self.stages[layer][child])
                });
                updated.insert(neuron, shannon_64(self.base.gate_truth(layer, neuron)?, input));
                evaluated += 1;
            }
            changes = updated;
        }

        Ok(IncrementalEvaluation {
            output: *changes.get(&0).expect("each path ends at the root"),
            gate_evaluations: evaluated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5_hierarchy::{GateKind, LeafRouting};

    fn bitplanes(count: usize, seed: u64) -> Vec<u64> {
        let mut rng = if seed == 0 { 1 } else { seed };
        (0..count)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng
            })
            .collect()
    }

    #[test]
    fn neuron_mutation_is_exact_for_all_layers_and_three_network_widths() {
        for modes in [
            &[GateKind::Majority, GateKind::Parity][..],
            &[GateKind::Parity, GateKind::Majority, GateKind::Or][..],
            &[GateKind::And, GateKind::Parity, GateKind::Majority, GateKind::Or][..],
        ] {
            let base = FiveCircleHierarchy::from_modes(modes, LeafRouting::SeededShuffle(43)).unwrap();
            for seed in [1, 202, 909] {
                let planes = bitplanes(base.input_bits(), seed);
                let cache = FiveCircleIncrementalCache::new(&base, &planes).unwrap();
                assert_eq!(cache.baseline_output(), base.infer_bitplanes(&planes).unwrap());
                assert_eq!(cache.build_gate_evaluations(), base.neuron_count());
                for layer in 0..base.depth() {
                    let nodes = base.input_bits() / CIRCLES.pow((layer + 1) as u32);
                    for index in [0, nodes / 2, nodes - 1] {
                        let old = base.layer_model(layer, index).unwrap();
                        let mut circles = *old.circles();
                        circles[0].radius += 3;
                        circles[0].phase_pi = !circles[0].phase_pi;
                        let next = FiveCircle::new(circles, old.bias() - 2).unwrap();
                        let candidate = base.with_neuron_parameters(layer, index, next).unwrap();
                        let result = cache.evaluate(PackedIntervention::Parameter {
                            layer, neuron: index, model: next
                        }).unwrap();
                        assert_eq!(result.output, candidate.infer_bitplanes(&planes).unwrap(),
                            "width={} seed={} layer={layer} index={index}", base.input_bits(), seed);
                        assert_eq!(result.gate_evaluations, base.depth() - layer);
                        assert!(result.gate_evaluations <= base.neuron_count());
                    }
                }
            }
        }
    }

    #[test]
    fn leaf_exchanges_are_exact_even_when_paths_merge_or_values_match() {
        for modes in [
            &[GateKind::Majority, GateKind::Parity][..],
            &[GateKind::Parity, GateKind::And, GateKind::Majority][..],
            &[GateKind::Or, GateKind::Parity, GateKind::Majority, GateKind::Parity][..],
        ] {
            let base = FiveCircleHierarchy::from_modes(modes, LeafRouting::SeededShuffle(88)).unwrap();
            for seed in [2, 194, 809] {
                let planes = bitplanes(base.input_bits(), seed);
                let cache = FiveCircleIncrementalCache::new(&base, &planes).unwrap();
                for (first, second) in [
                    (0, 1),
                    (0, 5),
                    (0, base.input_bits() / CIRCLES),
                    (base.input_bits() - 1, base.input_bits() / 2),
                ] {
                    let mut route = base.routing().to_vec();
                    route.swap(first, second);
                    let changed = base.rerouted(route).unwrap();
                    let actual = cache.evaluate(PackedIntervention::LeafSwap { first, second }).unwrap();
                    assert_eq!(actual.output, changed.infer_bitplanes(&planes).unwrap(),
                        "width={} seed={seed} swapped={first},{second}", base.input_bits());
                    assert!(actual.gate_evaluations <= 2 * base.depth());
                    assert!(actual.gate_evaluations < base.neuron_count());
                }
            }
        }
    }

    #[test]
    fn rejects_out_of_range_and_changed_mask_without_cache_corruption() {
        let model = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity],
            LeafRouting::Contiguous,
        ).unwrap();
        let planes = bitplanes(25, 42);
        let cache = FiveCircleIncrementalCache::new(&model, &planes).unwrap();
        assert!(matches!(
            cache.evaluate(PackedIntervention::LeafSwap {first: 0, second: 25}),
            Err(HierarchyError::InvalidRouting)
        ));
        assert!(matches!(
            FiveCircleIncrementalCache::new(&model, &planes[..24]),
            Err(HierarchyError::InvalidInput)
        ));
        let wrong = GateKind::Parity.five_circle();
        assert!(matches!(
            cache.evaluate(PackedIntervention::Parameter {layer: 0, neuron: 0, model: wrong}),
            Err(HierarchyError::InvalidLayerShape)
        ));
        assert_eq!(cache.baseline_output(), model.infer_bitplanes(&planes).unwrap());
    }
}

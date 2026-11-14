//! Scalable, exactly equivalent old-vs-cached 64-lane five-circle inference.
//!
//! The original frozen hierarchy intentionally stops at 625 inputs. This
//! bounded benchmark extension uses its exact Shannon/Walsh 5-bit primitive
//! on a *new* ragged five-ary tree, not the original 625-bit public API.
//! Missing children in the last partial group are explicitly FALSE, never
//! fabricated inputs or BANC synapses. Every source bit is used once.
//! All gates at level 0 are Majority; all higher levels are Parity.
//! Model parameters and topology are frozen except one opt-in intervention.
use crate::fourier5::{FiveCircle, CIRCLES};
use crate::fourier5_hierarchy::{compiled_truth_table, shannon_64, GateKind};

pub const SIX_MILLION_INPUTS: usize = 6_250_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WideError {
    InvalidWidth,
    InvalidInput,
    InvalidIntervention,
    IncompatibleWalshMasks,
}

#[derive(Clone, Copy, Debug)]
pub enum WideEdit {
    /// Change the local Fourier gains/phases/threshold, preserving all
    /// original five Walsh masks at that node.
    Parameter {
        layer: usize,
        neuron: usize,
        model: FiveCircle,
    },
    /// Swap the two leaf *positions*, keeping a complete source bijection.
    SourceSwap { first: usize, second: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WideEvaluation {
    pub output: u64,
    /// Number of exact 5-input Fourier gate invocations, each on 64 lanes.
    pub gate_evaluations: usize,
}

pub struct WideFiveTree {
    input_bits: usize,
    level_widths: Vec<usize>,
    base_models: Vec<FiveCircle>,
    base_tables: Vec<u32>,
    logical_neurons: usize,
    padding_slots: usize,
}

fn group_input(source: &[u64], index: usize) -> [u64; CIRCLES] {
    std::array::from_fn(|bit| source.get(index * CIRCLES + bit).copied().unwrap_or(0))
}

fn edited_source(planes: &[u64], position: usize, swap: Option<(usize, usize)>) -> u64 {
    if position >= planes.len() {
        return 0;
    }
    let remapped = match swap {
        Some((left, right)) if position == left => right,
        Some((left, right)) if position == right => left,
        _ => position,
    };
    planes[remapped]
}

impl WideFiveTree {
    /// Exactly input_bits REAL Boolean input positions. Padding is FALSE
    /// only at ragged upper levels, with the number of padded slots tracked.
    pub fn new(input_bits: usize) -> Result<Self, WideError> {
        if !(5..=SIX_MILLION_INPUTS).contains(&input_bits) {
            return Err(WideError::InvalidWidth);
        }
        let mut level_widths = Vec::new();
        let mut nodes = 0;
        let mut padding = 0;
        let mut previous = input_bits;
        while previous > 1 {
            let width = previous.div_ceil(CIRCLES);
            padding += width * CIRCLES - previous;
            nodes += width;
            level_widths.push(width);
            previous = width;
        }
        let base_models: Vec<_> = (0..level_widths.len())
            .map(|level| {
                if level == 0 {
                    GateKind::Majority.five_circle()
                } else {
                    GateKind::Parity.five_circle()
                }
            })
            .collect();
        let base_tables = base_models
            .iter()
            .copied()
            .map(compiled_truth_table)
            .collect();
        Ok(Self {
            input_bits,
            level_widths,
            base_models,
            base_tables,
            logical_neurons: nodes,
            padding_slots: padding,
        })
    }

    #[must_use]
    pub fn input_bits(&self) -> usize {
        self.input_bits
    }

    #[must_use]
    pub fn depth(&self) -> usize {
        self.level_widths.len()
    }

    #[must_use]
    pub fn logical_neurons(&self) -> usize {
        self.logical_neurons
    }

    #[must_use]
    pub fn real_contacts(&self) -> usize {
        self.logical_neurons * CIRCLES - self.padding_slots
    }

    #[must_use]
    pub fn padding_slots(&self) -> usize {
        self.padding_slots
    }

    #[must_use]
    pub fn level_widths(&self) -> &[usize] {
        &self.level_widths
    }

    /// Gate replacement is constrained to the exact same five Walsh masks.
    /// This is the same contract enforced by with_neuron_parameters in the
    /// 625-input public model. No new source or mask is invented.
    fn table_for_edit(&self, edit: WideEdit) -> Result<u32, WideError> {
        match edit {
            WideEdit::Parameter {
                layer,
                neuron,
                model,
            } => {
                if self
                    .level_widths
                    .get(layer)
                    .is_none_or(|&count| neuron >= count)
                {
                    return Err(WideError::InvalidIntervention);
                }
                let original = self.base_models[layer];
                if original
                    .circles()
                    .iter()
                    .zip(model.circles())
                    .any(|(left, right)| left.mask != right.mask)
                {
                    return Err(WideError::IncompatibleWalshMasks);
                }
                Ok(compiled_truth_table(model))
            }
            WideEdit::SourceSwap { first, second } => {
                if first >= self.input_bits || second >= self.input_bits || first == second {
                    return Err(WideError::InvalidIntervention);
                }
                Ok(0)
            }
        }
    }

    /// Generalized ORIGINAL full-recomputation algorithm. Every Fourier
    /// neuron is evaluated even for a one-leaf intervention, just as in
    /// FiveCircleHierarchy::infer_bitplanes, without the old 625-bit cap.
    pub fn evaluate_full(
        &self,
        planes: &[u64],
        edit: Option<WideEdit>,
    ) -> Result<WideEvaluation, WideError> {
        if planes.len() != self.input_bits {
            return Err(WideError::InvalidInput);
        }
        let parameter_table = edit.map(|value| self.table_for_edit(value)).transpose()?;
        let swap = match edit {
            Some(WideEdit::SourceSwap { first, second }) => Some((first, second)),
            _ => None,
        };
        let mut previous = Vec::new();
        for (level, &count) in self.level_widths.iter().enumerate() {
            let mut next = Vec::with_capacity(count);
            for neuron in 0..count {
                let input = if level == 0 {
                    std::array::from_fn(|bit| edited_source(planes, neuron * CIRCLES + bit, swap))
                } else {
                    group_input(&previous, neuron)
                };
                let truth = match edit {
                    Some(WideEdit::Parameter {
                        layer,
                        neuron: target,
                        ..
                    }) if level == layer && neuron == target => {
                        parameter_table.expect("validated parameter truth")
                    }
                    _ => self.base_tables[level],
                };
                next.push(shannon_64(truth, input));
            }
            previous = next;
        }
        Ok(WideEvaluation {
            output: previous[0],
            gate_evaluations: self.logical_neurons,
        })
    }

    /// An independent reference exploiting the fact that the upper layers
    /// are Parity: XOR of every bottom-level MAJORITY of five sources.
    /// It never calls the compiled Walsh LUT or Shannon evaluator.
    pub fn independent_reference(&self, planes: &[u64]) -> Result<u64, WideError> {
        if planes.len() != self.input_bits {
            return Err(WideError::InvalidInput);
        }
        let mut result = 0_u64;
        for index in 0..self.level_widths[0] {
            let five = group_input(planes, index);
            let mut majority = 0_u64;
            for a in 0..CIRCLES {
                for b in a + 1..CIRCLES {
                    for c in b + 1..CIRCLES {
                        majority |= five[a] & five[b] & five[c];
                    }
                }
            }
            result ^= majority;
        }
        Ok(result)
    }

    /// Reference five-circle gates per-level, not source bitplanes. The
    /// baseline representation occupies 8 * logical_neurons bytes at
    /// 64 parallel lanes, plus the unmodified 8 * input_bits source bytes.
    #[must_use]
    pub fn cache_word_bytes(&self) -> usize {
        self.logical_neurons * std::mem::size_of::<u64>()
    }
}

pub struct WideFiveCache<'a> {
    tree: &'a WideFiveTree,
    planes: &'a [u64],
    baseline: Vec<Vec<u64>>,
}

impl<'a> WideFiveCache<'a> {
    /// Exactly one entire hierarchy evaluation, retaining ALL intermediary
    /// outputs to avoid recalculating unchanged sibling subtrees.
    pub fn new(tree: &'a WideFiveTree, planes: &'a [u64]) -> Result<Self, WideError> {
        if planes.len() != tree.input_bits {
            return Err(WideError::InvalidInput);
        }
        let mut baseline: Vec<Vec<u64>> = Vec::with_capacity(tree.depth());
        for (level, &count) in tree.level_widths.iter().enumerate() {
            let mut next = Vec::with_capacity(count);
            for neuron in 0..count {
                let input = if level == 0 {
                    group_input(planes, neuron)
                } else {
                    group_input(&baseline[level - 1], neuron)
                };
                next.push(shannon_64(tree.base_tables[level], input));
            }
            baseline.push(next);
        }
        Ok(Self {
            tree,
            planes,
            baseline,
        })
    }

    #[must_use]
    pub fn output(&self) -> u64 {
        self.baseline.last().expect("nonempty tree")[0]
    }

    /// O(depth), or O(2 * depth) for one source swap, instead of
    /// O(input_bits). Boolean sensitivity pruning stops propagation when
    /// the changed node's *64-lane output* equals the cached output.
    pub fn evaluate_delta(&self, edit: WideEdit) -> Result<WideEvaluation, WideError> {
        let parameter_table = self.tree.table_for_edit(edit)?;
        let mut evaluated = 0;
        let (start_layer, mut changed): (usize, Vec<(usize, u64)>) = match edit {
            WideEdit::Parameter { layer, neuron, .. } => {
                let inputs = if layer == 0 {
                    group_input(self.planes, neuron)
                } else {
                    group_input(&self.baseline[layer - 1], neuron)
                };
                let updated = shannon_64(parameter_table, inputs);
                evaluated += 1;
                if updated == self.baseline[layer][neuron] {
                    (layer, Vec::new())
                } else {
                    (layer, vec![(neuron, updated)])
                }
            }
            WideEdit::SourceSwap { first, second } => {
                let mut affected = vec![first / CIRCLES, second / CIRCLES];
                affected.sort_unstable();
                affected.dedup();
                let mut changes = Vec::with_capacity(affected.len());
                for neuron in affected {
                    let input = std::array::from_fn(|bit| {
                        edited_source(self.planes, neuron * CIRCLES + bit, Some((first, second)))
                    });
                    let updated = shannon_64(self.tree.base_tables[0], input);
                    evaluated += 1;
                    if updated != self.baseline[0][neuron] {
                        changes.push((neuron, updated));
                    }
                }
                (0, changes)
            }
        };

        for level in start_layer + 1..self.tree.depth() {
            if changed.is_empty() {
                break;
            }
            let mut parents: Vec<usize> = changed.iter().map(|&(node, _)| node / CIRCLES).collect();
            parents.sort_unstable();
            parents.dedup();
            let mut updated = Vec::with_capacity(parents.len());
            for neuron in parents {
                let inputs = std::array::from_fn(|bit| {
                    let child = neuron * CIRCLES + bit;
                    changed
                        .iter()
                        .find(|&&(index, _)| index == child)
                        .map_or_else(
                            || self.baseline[level - 1].get(child).copied().unwrap_or(0),
                            |&(_, value)| value,
                        )
                });
                let value = shannon_64(self.tree.base_tables[level], inputs);
                evaluated += 1;
                if value != self.baseline[level][neuron] {
                    updated.push((neuron, value));
                }
            }
            changed = updated;
        }

        Ok(WideEvaluation {
            output: changed
                .first()
                .map_or_else(|| self.output(), |&(_, value)| value),
            gate_evaluations: evaluated,
        })
    }

    #[must_use]
    pub fn baseline_gate_evaluations(&self) -> usize {
        self.tree.logical_neurons
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planes(width: usize, mut state: u64) -> Vec<u64> {
        (0..width)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            })
            .collect()
    }

    fn parameter(tree: &WideFiveTree, layer: usize, neuron: usize) -> WideEdit {
        let base = tree.base_models[layer];
        let mut circles = *base.circles();
        circles[0].phase_pi = !circles[0].phase_pi;
        WideEdit::Parameter {
            layer,
            neuron,
            model: FiveCircle::new(circles, base.bias()).unwrap(),
        }
    }

    #[test]
    fn exact_even_for_non_power_of_five_sizes_and_ragged_upper_levels() {
        for size in [5, 25, 125, 625, 1000, 1250, 6250, 62500] {
            let tree = WideFiveTree::new(size).unwrap();
            assert_eq!(tree.real_contacts(), size + tree.logical_neurons() - 1);
            assert_eq!(tree.level_widths.last(), Some(&1));
            for seed in [99, 1001, 77129] {
                let input = planes(size, seed);
                let full = tree.evaluate_full(&input, None).unwrap();
                let cache = WideFiveCache::new(&tree, &input).unwrap();
                assert_eq!(full.output, cache.output(), "size={size}");
                assert_eq!(
                    full.output,
                    tree.independent_reference(&input).unwrap(),
                    "size={size}"
                );
                let edits = [
                    parameter(&tree, 0, 0),
                    parameter(
                        &tree,
                        tree.depth() / 2,
                        tree.level_widths[tree.depth() / 2] / 2,
                    ),
                    parameter(&tree, tree.depth() - 1, 0),
                    WideEdit::SourceSwap {
                        first: 0,
                        second: 1,
                    },
                    WideEdit::SourceSwap {
                        first: 0,
                        second: size / 2,
                    },
                    WideEdit::SourceSwap {
                        first: size / 3,
                        second: size - 1,
                    },
                ];
                for edit in edits {
                    let full = tree.evaluate_full(&input, Some(edit)).unwrap();
                    let cached = cache.evaluate_delta(edit).unwrap();
                    assert_eq!(
                        full.output, cached.output,
                        "size={size} seed={seed} edit={edit:?}"
                    );
                    assert!(cached.gate_evaluations <= 2 * tree.depth());
                }
            }
        }
    }

    #[test]
    fn six_million_250k_geometry_and_memory_are_exact() {
        let tree = WideFiveTree::new(SIX_MILLION_INPUTS).unwrap();
        assert_eq!(tree.depth(), 10);
        assert_eq!(
            tree.level_widths(),
            &[1_250_000, 250_000, 50_000, 10_000, 2_000, 400, 80, 16, 4, 1,]
        );
        assert_eq!(tree.logical_neurons(), 1_562_501);
        assert_eq!(tree.padding_slots(), 5);
        assert_eq!(tree.real_contacts(), 7_812_500);
        assert_eq!(tree.cache_word_bytes(), 12_500_008);
        assert_eq!(SIX_MILLION_INPUTS * 8, 50_000_000);
    }

    #[test]
    fn reject_missing_inputs_invalid_nodes_masks_and_source_aliases() {
        let tree = WideFiveTree::new(25).unwrap();
        let data = planes(25, 21);
        assert_eq!(WideFiveTree::new(4).err(), Some(WideError::InvalidWidth));
        assert_eq!(
            WideFiveTree::new(SIX_MILLION_INPUTS + 1).err(),
            Some(WideError::InvalidWidth)
        );
        assert_eq!(
            tree.evaluate_full(&data[..24], None).err(),
            Some(WideError::InvalidInput)
        );
        assert!(matches!(
            WideFiveCache::new(&tree, &data[..24]),
            Err(WideError::InvalidInput)
        ));
        let cache = WideFiveCache::new(&tree, &data).unwrap();
        for edit in [
            WideEdit::SourceSwap {
                first: 0,
                second: 0,
            },
            WideEdit::SourceSwap {
                first: 25,
                second: 1,
            },
            WideEdit::Parameter {
                layer: tree.depth(),
                neuron: 0,
                model: GateKind::Parity.five_circle(),
            },
            WideEdit::Parameter {
                layer: 0,
                neuron: 0,
                model: GateKind::Parity.five_circle(),
            },
        ] {
            assert!(tree.evaluate_full(&data, Some(edit)).is_err());
            assert!(cache.evaluate_delta(edit).is_err());
        }
        assert_eq!(
            tree.evaluate_full(&data, None).unwrap().output,
            cache.output()
        );
    }
}

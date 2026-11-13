//! Independently parameterized, persistently mutable five-input Fourier neurons.
//!
//! Unlike a single counterfactual evaluated against a frozen baseline, a
//! committed proposal updates the model and every affected cached descendant
//! along the unique path to the root. No dense routing, BANC synapses or
//! biological learning rule is claimed here.
use crate::fourier5::{FiveCircle, CIRCLES};
use crate::fourier5_hierarchy::{compiled_truth_table, shannon_64, GateKind};
use crate::fourier5_wide::{WideError, WideFiveTree};

/// An edit prepared against one exact model generation. It is not committed
/// unless the caller explicitly accepts it.
#[derive(Clone, Debug)]
pub struct WideProposal {
    generation: u64,
    layer: usize,
    neuron: usize,
    model: FiveCircle,
    truth: u32,
    changes: Vec<(usize, usize, u64)>,
    pub output: u64,
    pub gate_evaluations: usize,
    /// Logical u64 input/cached-child loads, NOT DRAM transactions.
    pub child_word_reads: usize,
    /// Actual changed stage words in the application-level cache.
    pub changed_cache_words: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WideLocalFit {
    /// Number of individually inspected and fitted artificial neurons.
    pub neurons_fitted: usize,
    /// Number whose stored Fourier parameters changed relative to the start.
    pub models_changed: usize,
    /// Number of 64-way node outputs agreeing with the TRAIN-only teacher.
    pub train_correct_lanes: u64,
    pub train_total_lanes: u64,
    pub gate_evaluations: usize,
}

pub struct WidePlasticCircuit<'a> {
    tree: &'a WideFiveTree,
    sources: &'a [u64],
    models: Vec<Vec<FiveCircle>>,
    truths: Vec<Vec<u32>>,
    stages: Vec<Vec<u64>>,
    generation: u64,
    commits: usize,
}

fn children(source: &[u64], neuron: usize) -> [u64; CIRCLES] {
    std::array::from_fn(|bit| source.get(neuron * CIRCLES + bit).copied().unwrap_or(0))
}

impl<'a> WidePlasticCircuit<'a> {
    /// One independent Fourier model and compiled table per physical node.
    pub fn new(tree: &'a WideFiveTree, sources: &'a [u64]) -> Result<Self, WideError> {
        if sources.len() != tree.input_bits() {
            return Err(WideError::InvalidInput);
        }
        let mut models = Vec::with_capacity(tree.depth());
        let mut truths = Vec::with_capacity(tree.depth());
        let mut stages: Vec<Vec<u64>> = Vec::with_capacity(tree.depth());
        for (layer, &width) in tree.level_widths().iter().enumerate() {
            let model = if layer == 0 {
                GateKind::Majority.five_circle()
            } else {
                GateKind::Parity.five_circle()
            };
            let truth = compiled_truth_table(model);
            let previous = if layer == 0 {
                sources
            } else {
                &stages[layer - 1]
            };
            stages.push(
                (0..width)
                    .map(|neuron| shannon_64(truth, children(previous, neuron)))
                    .collect::<Vec<_>>(),
            );
            models.push(vec![model; width]);
            truths.push(vec![truth; width]);
        }
        Ok(Self {
            tree,
            sources,
            models,
            truths,
            stages,
            generation: 0,
            commits: 0,
        })
    }

    #[must_use]
    pub fn output(&self) -> u64 {
        self.stages
            .last()
            .expect("nonempty")
            .first()
            .copied()
            .expect("root")
    }

    #[must_use]
    pub fn commits(&self) -> usize {
        self.commits
    }

    #[must_use]
    pub fn model_bytes(&self) -> usize {
        self.tree.logical_neurons() * std::mem::size_of::<FiveCircle>()
    }

    #[must_use]
    pub fn truth_bytes(&self) -> usize {
        self.tree.logical_neurons() * std::mem::size_of::<u32>()
    }

    #[must_use]
    pub fn cache_bytes(&self) -> usize {
        self.tree.cache_word_bytes()
    }

    /// Move the persistent model to a new 64-lane training minibatch.
    /// All 1,562,501 Fourier models remain unchanged. This is a FULL
    /// cache rebuild (not an incremental source update); its cost must
    /// be measured separately. Outstanding proposals become stale.
    pub fn rebind_sources(&mut self, sources: &'a [u64]) -> Result<(), WideError> {
        if sources.len() != self.tree.input_bits() {
            return Err(WideError::InvalidInput);
        }
        self.sources = sources;
        self.rebuild_cache();
        self.generation += 1;
        Ok(())
    }

    pub fn model(&self, layer: usize, neuron: usize) -> Result<FiveCircle, WideError> {
        self.models
            .get(layer)
            .and_then(|items| items.get(neuron))
            .copied()
            .ok_or(WideError::InvalidIntervention)
    }

    /// An explicit, independent model for EVERY neuron; installation is
    /// distinct from learning. Rebuilds cache only once after installation.
    pub fn install_models(&mut self, mut choice: impl FnMut(usize, usize) -> FiveCircle) {
        for (layer, models) in self.models.iter_mut().enumerate() {
            for (neuron, current) in models.iter_mut().enumerate() {
                let updated = choice(layer, neuron);
                *current = updated;
                self.truths[layer][neuron] = compiled_truth_table(updated);
            }
        }
        self.rebuild_cache();
        self.generation += 1;
    }

    fn rebuild_cache(&mut self) {
        for layer in 0..self.tree.depth() {
            for neuron in 0..self.stages[layer].len() {
                let previous = if layer == 0 {
                    self.sources
                } else {
                    &self.stages[layer - 1]
                };
                self.stages[layer][neuron] =
                    shannon_64(self.truths[layer][neuron], children(previous, neuron));
            }
        }
    }

    /// Recompute with arbitrary held-out inputs and return hidden-layer
    /// outputs, without modifying the persistent training cache.
    pub fn evaluate_trace(&self, sources: &[u64]) -> Result<Vec<Vec<u64>>, WideError> {
        if sources.len() != self.tree.input_bits() {
            return Err(WideError::InvalidInput);
        }
        let mut trace: Vec<Vec<u64>> = Vec::with_capacity(self.tree.depth());
        for (layer, truths) in self.truths.iter().enumerate() {
            let previous = if layer == 0 {
                sources
            } else {
                &trace[layer - 1]
            };
            trace.push(
                truths
                    .iter()
                    .enumerate()
                    .map(|(neuron, &truth)| shannon_64(truth, children(previous, neuron)))
                    .collect::<Vec<_>>(),
            );
        }
        Ok(trace)
    }

    /// Exact full-recomputation comparator for a single candidate edit;
    /// no cache or model state changes. Counts logical gate invocations.
    pub fn evaluate_full_edit(
        &self,
        layer: usize,
        neuron: usize,
        model: FiveCircle,
    ) -> Result<u64, WideError> {
        self.model(layer, neuron)?;
        let replacement = compiled_truth_table(model);
        let mut previous = Vec::<u64>::new();
        for (at_layer, row) in self.truths.iter().enumerate() {
            let input = if at_layer == 0 {
                self.sources
            } else {
                &previous
            };
            let next = row
                .iter()
                .enumerate()
                .map(|(index, &truth)| {
                    let truth = if at_layer == layer && index == neuron {
                        replacement
                    } else {
                        truth
                    };
                    shannon_64(truth, children(input, index))
                })
                .collect::<Vec<_>>();
            previous = next;
        }
        Ok(previous[0])
    }

    /// Stage the exact dependency cone, without changing any parameter or
    /// cached output. The proposal may be rejected for free.
    pub fn propose(
        &self,
        layer: usize,
        neuron: usize,
        model: FiveCircle,
    ) -> Result<WideProposal, WideError> {
        self.model(layer, neuron)?;
        let truth = compiled_truth_table(model);
        let mut changes = Vec::with_capacity(self.tree.depth() - layer);
        let mut reads = 0_usize;
        let mut evaluations = 0_usize;
        let mut current_node = neuron;
        let mut previous_update: Option<(usize, u64)> = None;
        for at_layer in layer..self.tree.depth() {
            let previous = if at_layer == 0 {
                self.sources
            } else {
                &self.stages[at_layer - 1]
            };
            let input: [u64; CIRCLES] = std::array::from_fn(|bit| {
                let child_index = current_node * CIRCLES + bit;
                let original = previous.get(child_index).copied();
                if original.is_some() {
                    reads += 1;
                }
                match previous_update {
                    Some((index, value)) if index == child_index => value,
                    _ => original.unwrap_or(0),
                }
            });
            let table = if at_layer == layer {
                truth
            } else {
                self.truths[at_layer][current_node]
            };
            let updated = shannon_64(table, input);
            evaluations += 1;
            if updated == self.stages[at_layer][current_node] {
                break;
            }
            changes.push((at_layer, current_node, updated));
            previous_update = Some((current_node, updated));
            current_node /= CIRCLES;
        }
        let output = changes
            .last()
            .filter(|(at_layer, _, _)| *at_layer == self.tree.depth() - 1)
            .map_or_else(|| self.output(), |(_, _, value)| *value);
        Ok(WideProposal {
            generation: self.generation,
            layer,
            neuron,
            model,
            truth,
            changed_cache_words: changes.len(),
            changes,
            output,
            gate_evaluations: evaluations,
            child_word_reads: reads,
        })
    }

    /// Commit both parameters and propagated outputs, not merely a
    /// counterfactual. A proposal based on a stale generation is rejected.
    pub fn accept(&mut self, proposal: WideProposal) -> Result<(), WideError> {
        if proposal.generation != self.generation {
            return Err(WideError::InvalidIntervention);
        }
        self.models[proposal.layer][proposal.neuron] = proposal.model;
        self.truths[proposal.layer][proposal.neuron] = proposal.truth;
        for (layer, neuron, output) in proposal.changes {
            self.stages[layer][neuron] = output;
        }
        self.generation += 1;
        self.commits += 1;
        Ok(())
    }

    /// Fit a fully observed artificial teacher circuit on TRAIN activations.
    /// These are PRIVILEGED HIDDEN LABELS, not output-only learning.
    /// Every neuron chooses its own Fourier parameter set. The caller is
    /// responsible for a held-out evaluation after fitting.
    pub fn fit_local_teacher(
        &mut self,
        targets: &[Vec<u64>],
        candidates: &[FiveCircle],
    ) -> Result<WideLocalFit, WideError> {
        if candidates.is_empty()
            || targets.len() != self.tree.depth()
            || targets
                .iter()
                .zip(self.tree.level_widths())
                .any(|(row, &n)| row.len() != n)
        {
            return Err(WideError::InvalidInput);
        }
        let options: Vec<(FiveCircle, u32)> = candidates
            .iter()
            .copied()
            .map(|model| (model, compiled_truth_table(model)))
            .collect();
        let mut report = WideLocalFit::default();
        for (layer, target_row) in targets.iter().enumerate() {
            for (neuron, &label) in target_row.iter().enumerate() {
                let previous = if layer == 0 {
                    self.sources
                } else {
                    &self.stages[layer - 1]
                };
                let input = children(previous, neuron);
                let mut best = (u32::MAX, self.models[layer][neuron], 0_u32, 0_u64);
                for &(model, truth) in &options {
                    let prediction = shannon_64(truth, input);
                    report.gate_evaluations += 1;
                    let errors = (prediction ^ label).count_ones();
                    if errors < best.0 {
                        best = (errors, model, truth, prediction);
                    }
                }
                if self.models[layer][neuron] != best.1 {
                    report.models_changed += 1;
                }
                self.models[layer][neuron] = best.1;
                self.truths[layer][neuron] = best.2;
                self.stages[layer][neuron] = best.3;
                report.train_correct_lanes += u64::from(64 - best.0);
                report.train_total_lanes += 64;
                report.neurons_fitted += 1;
            }
        }
        self.generation += 1;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planes(width: usize, mut seed: u64) -> Vec<u64> {
        (0..width)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            })
            .collect()
    }

    #[test]
    fn rebinding_preserves_learned_parameters_and_invalidates_stale_proposals() {
        let tree = WideFiveTree::new(125).unwrap();
        let first = planes(125, 123);
        let second = planes(125, 456);
        let mut circuit = WidePlasticCircuit::new(&tree, &first).unwrap();
        let chosen = GateKind::And.five_circle();
        circuit
            .accept(circuit.propose(0, 0, chosen).unwrap())
            .unwrap();
        let stale = circuit.propose(0, 1, GateKind::Or.five_circle()).unwrap();
        let previous_commits = circuit.commits();
        circuit.rebind_sources(&second).unwrap();
        assert_eq!(circuit.commits(), previous_commits);
        assert_eq!(circuit.model(0, 0), Ok(chosen));
        assert!(circuit.accept(stale).is_err());
        assert_eq!(
            circuit.output(),
            circuit.evaluate_trace(&second).unwrap().last().unwrap()[0]
        );
        assert_eq!(
            circuit.rebind_sources(&first[..124]),
            Err(WideError::InvalidInput)
        );
        assert_eq!(
            circuit.output(),
            circuit.evaluate_trace(&second).unwrap().last().unwrap()[0]
        );
        circuit.rebind_sources(&first).unwrap();
        assert_eq!(
            circuit.output(),
            circuit.evaluate_trace(&first).unwrap().last().unwrap()[0]
        );
    }

    #[test]
    fn persistent_independent_neuron_edits_are_exact_across_generations() {
        let tree = WideFiveTree::new(125).unwrap();
        let data = planes(125, 0x9876);
        let mut learner = WidePlasticCircuit::new(&tree, &data).unwrap();
        for (layer, neuron, mode) in [
            (0, 0, GateKind::And),
            (1, 3, GateKind::Or),
            (0, 12, GateKind::Parity),
            (2, 0, GateKind::Majority),
        ] {
            let candidate = mode.five_circle();
            let proposal = learner.propose(layer, neuron, candidate).unwrap();
            assert_eq!(
                proposal.output,
                learner
                    .evaluate_full_edit(layer, neuron, candidate)
                    .unwrap()
            );
            assert!(proposal.gate_evaluations <= tree.depth() - layer);
            learner.accept(proposal).unwrap();
            assert_eq!(
                learner.output(),
                learner.evaluate_trace(&data).unwrap().last().unwrap()[0]
            );
            assert_eq!(learner.model(layer, neuron), Ok(candidate));
        }
        assert_eq!(learner.commits(), 4);
        assert_eq!(
            learner.model_bytes(),
            tree.logical_neurons() * std::mem::size_of::<FiveCircle>()
        );
    }

    #[test]
    fn rejection_stale_proposal_and_teacher_fitting_are_separate() {
        let tree = WideFiveTree::new(625).unwrap();
        let data = planes(625, 456);
        let other = planes(625, 789);
        let mut teacher = WidePlasticCircuit::new(&tree, &data).unwrap();
        teacher.install_models(|layer, neuron| {
            match (layer * 3 + neuron) % 4 {
                0 => GateKind::Majority,
                1 => GateKind::Parity,
                2 => GateKind::And,
                _ => GateKind::Or,
            }
            .five_circle()
        });
        let train = teacher.evaluate_trace(&data).unwrap();
        let valid = teacher.evaluate_trace(&other).unwrap();
        let mut learner = WidePlasticCircuit::new(&tree, &data).unwrap();
        let unaccepted = learner.propose(0, 0, GateKind::Or.five_circle()).unwrap();
        assert_eq!(learner.commits(), 0);
        let options = [
            GateKind::Majority.five_circle(),
            GateKind::Parity.five_circle(),
            GateKind::And.five_circle(),
            GateKind::Or.five_circle(),
        ];
        let report = learner.fit_local_teacher(&train, &options).unwrap();
        assert_eq!(report.neurons_fitted, tree.logical_neurons());
        assert!(report.train_correct_lanes <= report.train_total_lanes);
        assert!(learner.accept(unaccepted).is_err());
        let learned_valid = learner.evaluate_trace(&other).unwrap();
        assert_eq!(learned_valid.len(), valid.len());
        assert_eq!(
            learner.output(),
            learner.evaluate_trace(&data).unwrap().last().unwrap()[0]
        );
        assert_eq!(learner.commits(), 0);
        assert!(learner.fit_local_teacher(&[], &options).is_err());
    }
}

//! Per-neuron supervised plasticity for the existing five-circle network.
//!
//! Every neuron owns its own five context gains; the ten incoming indices,
//! Fourier phases, masks and persistent radii remain fixed. The supervision
//! is synthetic, not a measured BANC signal or evidence of neural memory.
use crate::fourier5::{FiveCircle, CIRCLES};
use crate::fourier5_dynamic::{
    train_context_gates, GatedFiveCircle, ModulationError, RadiusGate,
};
use crate::fourier5_recurrent::{MAX_NETWORK_NODES, MIN_NETWORK_NODES};

pub const LOCAL_EXAMPLES: usize = 64;
pub const MAX_LOCAL_EPOCHS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalTask {
    A,
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalLearningError {
    InvalidPopulation,
    InvalidEpochBudget,
    Model(ModulationError),
}

impl From<ModulationError> for LocalLearningError {
    fn from(value: ModulationError) -> Self {
        Self::Model(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalTrainReport {
    pub neurons: usize,
    pub epochs_used_total: usize,
    pub updates_total: usize,
    pub train_correct: usize,
    pub train_examples: usize,
    pub perfect_neurons: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalEvaluation {
    pub neurons: usize,
    pub correct: usize,
    pub examples: usize,
    pub perfect_neurons: usize,
}

#[derive(Clone, Debug)]
pub struct LocalPopulation {
    seed: u64,
    models: Vec<GatedFiveCircle>,
}

fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn task_bit(seed: u64, neuron: usize, task: LocalTask) -> usize {
    let first = (mix64(seed ^ (neuron as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)) % 5) as usize;
    match task {
        LocalTask::A => first,
        LocalTask::B => (first + 1) % CIRCLES,
    }
}

fn target(input: u8, context: u8, bit: usize) -> bool {
    input.count_ones() % 2 != 0 && context & (1 << bit) == 0
}

/// Each context appears with one odd and one even parity word. For TEST,
/// two input bits are flipped, preserving parity but changing each pair.
/// The generated pairs are disjoint from that neuron's TRAIN pairs.
fn examples(seed: u64, neuron: usize, task: LocalTask, test: bool) -> [(u8, u8, bool); LOCAL_EXAMPLES] {
    let bit = task_bit(seed, neuron, task);
    std::array::from_fn(|index| {
        let context = (index / 2) as u8;
        let h = mix64(
            seed ^ (neuron as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)
                ^ u64::from(context).wrapping_mul(0xa076_1d64_78bd_642f),
        );
        let high = ((h as u8) & 15) << 1;
        let odd = high | u8::from(high.count_ones() % 2 == 0);
        let input = odd ^ u8::from(index % 2 != 0) ^ if test { 0b0_0110 } else { 0 };
        (input, context, target(input, context, bit))
    })
}

impl LocalPopulation {
    pub fn new(nodes: usize, seed: u64, base: FiveCircle) -> Result<Self, LocalLearningError> {
        if !(MIN_NETWORK_NODES..=MAX_NETWORK_NODES).contains(&nodes) {
            return Err(LocalLearningError::InvalidPopulation);
        }
        let gates = std::array::from_fn(|i| RadiusGate {
            context_mask: 1_u8 << i,
            boost: 0,
        });
        let model = GatedFiveCircle::new(base, gates)?;
        Ok(Self {
            seed,
            models: vec![model; nodes],
        })
    }

    #[must_use]
    pub fn models(&self) -> &[GatedFiveCircle] {
        &self.models
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.models.len()
    }

    /// FNV-style deterministic **noncryptographic** fingerprint of all five
    /// learned boosts, useful for reproducibility and task interference.
    #[must_use]
    pub fn gain_fingerprint(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for model in &self.models {
            for gate in model.gates() {
                hash = (hash ^ u64::from(gate.boost))
                    .wrapping_mul(0x0000_0100_0000_01b3);
                hash = (hash ^ u64::from(gate.context_mask))
                    .wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    /// Train EVERY neuron on its own 64 supervised examples. Models start
    /// from their current gains, so training A then B is an interference arm.
    /// No sharing of learned weights across different neuron identities.
    pub fn train(
        &mut self,
        task: LocalTask,
        epochs: usize,
    ) -> Result<LocalTrainReport, LocalLearningError> {
        if epochs == 0 || epochs > MAX_LOCAL_EPOCHS {
            return Err(LocalLearningError::InvalidEpochBudget);
        }
        let mut correct = 0;
        let mut perfect = 0;
        let mut updates = 0;
        let mut epochs_used_total = 0;
        for (i, model) in self.models.iter_mut().enumerate() {
            let train = examples(self.seed, i, task, false);
            let fit = train_context_gates(*model, &train, epochs)?;
            *model = fit.model;
            correct += fit.correct;
            perfect += usize::from(fit.correct == LOCAL_EXAMPLES);
            updates += fit.updates;
            epochs_used_total += fit.epochs_used;
        }
        Ok(LocalTrainReport {
            neurons: self.node_count(),
            epochs_used_total,
            updates_total: updates,
            train_correct: correct,
            train_examples: self.node_count() * LOCAL_EXAMPLES,
            perfect_neurons: perfect,
        })
    }

    /// TEST only. Does not change gain values, training order or topology.
    pub fn evaluate(&self, task: LocalTask) -> Result<LocalEvaluation, LocalLearningError> {
        let mut correct = 0;
        let mut perfect = 0;
        for (i, model) in self.models.iter().enumerate() {
            let mut hits = 0;
            for (input, context, expected) in examples(self.seed, i, task, true) {
                if model.predict(input, context)? == expected {
                    hits += 1;
                }
            }
            correct += hits;
            perfect += usize::from(hits == LOCAL_EXAMPLES);
        }
        Ok(LocalEvaluation {
            neurons: self.node_count(),
            correct,
            examples: self.node_count() * LOCAL_EXAMPLES,
            perfect_neurons: perfect,
        })
    }

    /// Exact task-label distribution, computed from explicit deterministic
    /// synthetic target selection. Useful for checking neuron diversity.
    #[must_use]
    pub fn teacher_bit_histogram(&self, task: LocalTask) -> [usize; CIRCLES] {
        let mut bins = [0; CIRCLES];
        for i in 0..self.node_count() {
            bins[task_bit(self.seed, i, task)] += 1;
        }
        bins
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5::Circle;

    fn parity_base() -> FiveCircle {
        let mut circles = [Circle::default(); CIRCLES];
        circles[0] = Circle {
            mask: 31,
            radius: 2,
            phase_pi: false,
        };
        FiveCircle::new(circles, -1).unwrap()
    }

    #[test]
    fn train_test_pairs_disjoint_with_parity_preserved() {
        for i in 0..24 {
            let a = examples(27, i, LocalTask::A, false);
            let t = examples(27, i, LocalTask::A, true);
            for k in 0..LOCAL_EXAMPLES {
                assert_eq!(a[k].1, t[k].1);
                assert_eq!(a[k].0.count_ones() % 2, t[k].0.count_ones() % 2);
                assert_ne!(a[k].0, t[k].0);
                assert_eq!(target(a[k].0, a[k].1, task_bit(27, i, LocalTask::A)), a[k].2);
                assert_eq!(target(t[k].0, t[k].1, task_bit(27, i, LocalTask::A)), t[k].2);
            }
        }
    }

    #[test]
    fn individual_models_learn_different_context_cues() {
        let mut population = LocalPopulation::new(128, 123, parity_base()).unwrap();
        let before = population.gain_fingerprint();
        let evaluation_before = population.evaluate(LocalTask::A).unwrap();
        assert!(evaluation_before.correct < evaluation_before.examples);
        let report = population.train(LocalTask::A, 8).unwrap();
        let after = population.evaluate(LocalTask::A).unwrap();
        assert_eq!(report.neurons, 128);
        assert_eq!(report.train_correct, report.train_examples);
        assert_eq!(after.correct, after.examples);
        assert_eq!(after.perfect_neurons, 128);
        assert_ne!(before, population.gain_fingerprint());
        assert_eq!(population.teacher_bit_histogram(LocalTask::A).iter().sum::<usize>(), 128);
        assert!(population.models().windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn sequential_interference_is_measured_not_hidden() {
        let mut population = LocalPopulation::new(128, 991, parity_base()).unwrap();
        population.train(LocalTask::A, 8).unwrap();
        let bank_a = population.clone();
        let snapshot_a = population.gain_fingerprint();
        assert_eq!(bank_a.evaluate(LocalTask::A).unwrap().correct, 128 * 64);
        let before_b = population.evaluate(LocalTask::B).unwrap();
        assert!(before_b.correct < before_b.examples);
        population.train(LocalTask::B, 8).unwrap();
        let after_b = population.evaluate(LocalTask::B).unwrap();
        assert_eq!(after_b.correct, after_b.examples);
        let retained_a = population.evaluate(LocalTask::A).unwrap();
        assert!(retained_a.correct < retained_a.examples);
        assert_ne!(snapshot_a, population.gain_fingerprint());
        assert_eq!(bank_a.evaluate(LocalTask::A).unwrap().correct, 128 * 64);
        let mut bank_b = LocalPopulation::new(128, 991, parity_base()).unwrap();
        bank_b.train(LocalTask::B, 8).unwrap();
        assert_eq!(bank_b.evaluate(LocalTask::B).unwrap().correct, 128 * 64);
    }

    #[test]
    fn identical_seeds_are_bit_exact_and_different_seeds_change_fingerprint() {
        let mut one = LocalPopulation::new(64, 17, parity_base()).unwrap();
        let mut two = LocalPopulation::new(64, 17, parity_base()).unwrap();
        let mut three = LocalPopulation::new(64, 18, parity_base()).unwrap();
        assert_eq!(one.train(LocalTask::A, 8).unwrap(), two.train(LocalTask::A, 8).unwrap());
        three.train(LocalTask::A, 8).unwrap();
        assert_eq!(one.gain_fingerprint(), two.gain_fingerprint());
        assert_ne!(one.gain_fingerprint(), three.gain_fingerprint());
    }

    #[test]
    fn invalid_populations_and_training_budgets_rejected() {
        assert!(matches!(
            LocalPopulation::new(63, 1, parity_base()),
            Err(LocalLearningError::InvalidPopulation)
        ));
        assert!(matches!(
            LocalPopulation::new(MAX_NETWORK_NODES + 1, 1, parity_base()),
            Err(LocalLearningError::InvalidPopulation)
        ));
        let mut population = LocalPopulation::new(64, 1, parity_base()).unwrap();
        let before = population.gain_fingerprint();
        assert_eq!(
            population.train(LocalTask::A, 0),
            Err(LocalLearningError::InvalidEpochBudget)
        );
        assert_eq!(
            population.train(LocalTask::B, MAX_LOCAL_EPOCHS + 1),
            Err(LocalLearningError::InvalidEpochBudget)
        );
        assert_eq!(before, population.gain_fingerprint());
    }
}

//! Multi-task continual learning of task-addressed Fourier-Walsh readout neurons.
//! Eight (or more) separately learned heads share five frozen features from
//! an existing Boolean circuit. TRAIN output labels only, not hidden teacher
//! activations; no claim that the 1.56M backbone neurons learn here.
//! Each head selects an unknown Walsh mask and sign among 62 candidates.
//! Explicit task IDs prevent overwriting prior heads (capacity grows by task).
use crate::fourier5::{Circle, FiveCircle, CIRCLES};
use crate::fourier5_hierarchy::{compiled_truth_table, shannon_64};

const MAGIC: &[u8; 8] = b"V888TH01";
const HEADER: usize = 8 + 4 + 4 + 4 + 5 * 4;
const SLOT: usize = 2 + 8;
const MAX_TASKS: usize = 1_024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadError {
    InvalidShape,
    InvalidTask,
    NotLearned,
    CorruptCheckpoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskBatch {
    /// Exactly five 64-lane input planes from the SAME frozen feature binding.
    pub features: [u64; CIRCLES],
    /// Only the final observable task label. No target hidden activations.
    pub labels: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadFit {
    pub task: usize,
    pub train_correct: u64,
    pub train_total: u64,
    pub candidates_tested: usize,
    pub changed: bool,
    pub revision: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Head {
    mask: u8,
    phase: bool,
    revision: u64,
}

/// Explicit feature binding prevents accidentally restoring task memories
/// against a different circuit layer or different five source neurons.
/// All five features may be reused across multiple tasks.
#[derive(Clone, Debug)]
pub struct TaskHeadBank {
    feature_layer: u32,
    source_neurons: [u32; CIRCLES],
    heads: Vec<Option<Head>>,
}

fn candidate(mask: u8, phase: bool) -> FiveCircle {
    let mut circles = [Circle::default(); CIRCLES];
    circles[0] = Circle {
        mask,
        radius: 1,
        phase_pi: phase,
    };
    FiveCircle::new(circles, 0).expect("mask between 1 and 31")
}

fn matches(features: [u64; CIRCLES], labels: u64, mask: u8, phase: bool) -> u64 {
    let truth = compiled_truth_table(candidate(mask, phase));
    u64::from(64 - (shannon_64(truth, features) ^ labels).count_ones())
}

fn fingerprint(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for &value in bytes {
        h = (h ^ u64::from(value)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl TaskHeadBank {
    pub fn new(
        tasks: usize,
        feature_layer: u32,
        source_neurons: [u32; CIRCLES],
    ) -> Result<Self, HeadError> {
        if tasks == 0 || tasks > MAX_TASKS {
            return Err(HeadError::InvalidShape);
        }
        if source_neurons
            .iter()
            .enumerate()
            .any(|(i, &x)| source_neurons[..i].contains(&x))
        {
            return Err(HeadError::InvalidShape);
        }
        Ok(Self {
            feature_layer,
            source_neurons,
            heads: vec![None; tasks],
        })
    }

    #[must_use]
    pub fn task_count(&self) -> usize {
        self.heads.len()
    }

    #[must_use]
    pub fn learned_count(&self) -> usize {
        self.heads.iter().filter(|h| h.is_some()).count()
    }

    pub fn model(&self, task: usize) -> Result<Option<FiveCircle>, HeadError> {
        let h = self.heads.get(task).ok_or(HeadError::InvalidTask)?;
        Ok(h.map(|x| candidate(x.mask, x.phase)))
    }

    pub fn revision(&self, task: usize) -> Result<u64, HeadError> {
        Ok(self.heads.get(task).ok_or(HeadError::InvalidTask)?.map_or(0, |h| h.revision))
    }

    pub fn predict(&self, task: usize, features: [u64; CIRCLES]) -> Result<u64, HeadError> {
        let head = self
            .heads
            .get(task)
            .ok_or(HeadError::InvalidTask)?
            .ok_or(HeadError::NotLearned)?;
        Ok(shannon_64(
            compiled_truth_table(candidate(head.mask, head.phase)),
            features,
        ))
    }

    /// Train on complete 64-lane batches. VALID/TEST/OOD labels are never read.
    /// A tie retains the existing head when possible and otherwise chooses
    /// the first deterministic candidate. Each task gets one independent
    /// neuron, so learning task B cannot mutate task A's learned parameters.
    pub fn fit(&mut self, task: usize, train: &[TaskBatch]) -> Result<HeadFit, HeadError> {
        if self.heads.get(task).is_none() {
            return Err(HeadError::InvalidTask);
        }
        if train.len() < 2 || train.len() > 1_024 {
            return Err(HeadError::InvalidShape);
        }
        let old = self.heads[task];
        let mut best: Option<(u64, u8, bool)> = None;
        let mut tested = 0;
        for mask in 1..=31_u8 {
            for phase in [false, true] {
                let correct: u64 = train
                    .iter()
                    .map(|b| matches(b.features, b.labels, mask, phase))
                    .sum();
                tested += 1;
                if best.is_none_or(|(score, _, _)| correct > score)
                    || best.is_some_and(|(score, _, _)| {
                        correct == score
                            && old.is_some_and(|h| h.mask == mask && h.phase == phase)
                    })
                {
                    best = Some((correct, mask, phase));
                }
            }
        }
        let (correct, mask, phase) = best.expect("62 candidates");
        let changed = old.is_none_or(|h| h.mask != mask || h.phase != phase);
        let revision = old.map_or(1, |h| h.revision.saturating_add(u64::from(changed)));
        self.heads[task] = Some(Head {
            mask,
            phase,
            revision,
        });
        Ok(HeadFit {
            task,
            train_correct: correct,
            train_total: train.len() as u64 * 64,
            candidates_tested: tested,
            changed,
            revision,
        })
    }

    /// Stable, self-checking data-only checkpoint: never executes code.
    /// Frozen backbone/feature binding must be validated by caller.
    #[must_use]
    pub fn checkpoint(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(HEADER + self.heads.len() * SLOT + 8);
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&1_u32.to_le_bytes());
        data.extend_from_slice(&(self.heads.len() as u32).to_le_bytes());
        data.extend_from_slice(&self.feature_layer.to_le_bytes());
        for &node in &self.source_neurons {
            data.extend_from_slice(&node.to_le_bytes());
        }
        for head in &self.heads {
            if let Some(h) = head {
                data.extend_from_slice(&[h.mask, u8::from(h.phase)]);
                data.extend_from_slice(&h.revision.to_le_bytes());
            } else {
                data.extend_from_slice(&[0, 0]);
                data.extend_from_slice(&0_u64.to_le_bytes());
            }
        }
        let hash = fingerprint(&data);
        data.extend_from_slice(&hash.to_le_bytes());
        data
    }

    pub fn restore(
        bytes: &[u8],
        tasks: usize,
        feature_layer: u32,
        source_neurons: [u32; CIRCLES],
    ) -> Result<Self, HeadError> {
        let mut bank = Self::new(tasks, feature_layer, source_neurons)?;
        let expected_len = HEADER + tasks * SLOT + 8;
        if bytes.len() != expected_len || &bytes[..8] != MAGIC {
            return Err(HeadError::CorruptCheckpoint);
        }
        let u32_at = |p: usize| u32::from_le_bytes(bytes[p..p + 4].try_into().expect("checked size"));
        if u32_at(8) != 1 || u32_at(12) != tasks as u32 || u32_at(16) != feature_layer {
            return Err(HeadError::CorruptCheckpoint);
        }
        for (i, &neuron) in source_neurons.iter().enumerate() {
            if u32_at(20 + i * 4) != neuron {
                return Err(HeadError::CorruptCheckpoint);
            }
        }
        let checksum = u64::from_le_bytes(bytes[expected_len - 8..].try_into().expect("size"));
        if checksum != fingerprint(&bytes[..expected_len - 8]) {
            return Err(HeadError::CorruptCheckpoint);
        }
        for i in 0..tasks {
            let off = HEADER + i * SLOT;
            let mask = bytes[off];
            let phase = bytes[off + 1];
            let revision = u64::from_le_bytes(bytes[off + 2..off + 10].try_into().expect("size"));
            bank.heads[i] = match (mask, phase, revision) {
                (0, 0, 0) => None,
                (1..=31, 0..=1, 1..) => Some(Head {
                    mask,
                    phase: phase == 1,
                    revision,
                }),
                _ => return Err(HeadError::CorruptCheckpoint),
            };
        }
        Ok(bank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(mask: u8, phase: bool, seed: u64) -> TaskBatch {
        let mut x = seed;
        let features = std::array::from_fn(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        });
        let labels = shannon_64(compiled_truth_table(candidate(mask, phase)), features);
        TaskBatch { features, labels }
    }

    #[test]
    fn two_tasks_learn_new_samples_and_survive_checkpoint() {
        let mut bank = TaskHeadBank::new(2, 7, [0, 1, 2, 3, 4]).unwrap();
        assert_eq!(bank.predict(0, [0; 5]), Err(HeadError::NotLearned));
        let a = [batch(0b10111, false, 42), batch(0b10111, false, 43)];
        let report_a = bank.fit(0, &a).unwrap();
        assert_eq!(report_a.train_correct, 128);
        assert_eq!(report_a.candidates_tested, 62);
        let before = bank.model(0).unwrap();
        let b = [batch(0b01101, true, 82), batch(0b01101, true, 83)];
        assert_eq!(bank.fit(1, &b).unwrap().train_correct, 128);
        assert_eq!(bank.model(0).unwrap(), before);
        assert_eq!(bank.learned_count(), 2);
        for (task, mask, phase, seed) in [(0, 0b10111, false, 2026), (1, 0b01101, true, 2027)] {
            let item = batch(mask, phase, seed);
            assert_eq!(bank.predict(task, item.features), Ok(item.labels));
        }
        let bytes = bank.checkpoint();
        let restored = TaskHeadBank::restore(&bytes, 2, 7, [0, 1, 2, 3, 4]).unwrap();
        assert_eq!(restored.checkpoint(), bytes);
        assert!(TaskHeadBank::restore(&bytes, 2, 6, [0, 1, 2, 3, 4]).is_err());
        let mut damaged = bytes;
        damaged[42] ^= 0x40;
        assert_eq!(
            TaskHeadBank::restore(&damaged, 2, 7, [0, 1, 2, 3, 4]).err(),
            Some(HeadError::CorruptCheckpoint)
        );
    }

    #[test]
    fn reject_bad_inputs_and_keep_unrelated_weights_frozen() {
        assert!(TaskHeadBank::new(0, 0, [0, 1, 2, 3, 4]).is_err());
        assert!(TaskHeadBank::new(1, 0, [1, 1, 2, 3, 4]).is_err());
        let mut bank = TaskHeadBank::new(2, 0, [0, 1, 2, 3, 4]).unwrap();
        assert_eq!(bank.fit(0, &[]), Err(HeadError::InvalidShape));
        assert_eq!(bank.fit(2, &[batch(3, false, 1), batch(3, false, 2)]), Err(HeadError::InvalidTask));
        assert!(bank.model(1).unwrap().is_none());
        assert_eq!(bank.revision(1), Ok(0));
    }
}

//! B888-4 shared-capacity contextual credit assignment.
//!
//! A SINGLE five-input Fourier-Walsh neuron (three frozen circuit features,
//! two observed context bits) is trained on sequential task-output errors.
//! No task-indexed output head is allocated. Functional Walsh coupling masks
//! and phase are mutable; the 1.56M-node feature extractor stays frozen.
//! Exact finite candidate search with *TRAIN-only* past-task constraints.
//! This is not structural BANC synaptic growth or general backpropagation.
use crate::fourier5::{Circle, FiveCircle, CIRCLES};
use crate::fourier5_task_memory::TaskBatch;

const MAGIC: &[u8; 8] = b"V888SC01";
const CHECKPOINT_LEN: usize = 46;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharedError {
    InvalidFeatures,
    InvalidTraining,
    CorruptCheckpoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedUpdate {
    pub accepted: bool,
    pub candidates_tested: usize,
    pub current_before: u64,
    pub current_after: u64,
    pub current_total: u64,
    pub previous_before: u64,
    pub previous_after: u64,
    pub changed_coupling_bits: u32,
    pub commits: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedContextNeuron {
    feature_layer: u32,
    source_neurons: [u32; 3],
    mask: u8,
    phase: bool,
    commits: u64,
}

fn checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// The one active Walsh character; the other four circles remain zero.
/// The five input slots represent three backbone features and two context bits.
#[must_use]
pub fn neuron(mask: u8, phase: bool) -> Option<FiveCircle> {
    if mask == 0 || mask > 31 {
        return None;
    }
    let mut terms = [Circle::default(); CIRCLES];
    terms[0] = Circle {
        mask,
        radius: 1,
        phase_pi: phase,
    };
    FiveCircle::new(terms, 0).ok()
}

#[must_use]
pub fn predict_bits(mask: u8, phase: bool, features: [u64; CIRCLES]) -> u64 {
    let mut out = if phase { u64::MAX } else { 0 };
    for (bit, &plane) in features.iter().enumerate() {
        if mask & (1 << bit) != 0 {
            out ^= plane;
        }
    }
    out
}

fn count_right(mask: u8, phase: bool, sample: &TaskBatch) -> u64 {
    let bad = (predict_bits(mask, phase, sample.features) ^ sample.labels) & sample.observed;
    u64::from(sample.observed.count_ones() - bad.count_ones())
}

fn score_set(mask: u8, phase: bool, batches: &[TaskBatch]) -> u64 {
    batches.iter().map(|b| count_right(mask, phase, b)).sum()
}

impl SharedContextNeuron {
    pub fn new(feature_layer: u32, source_neurons: [u32; 3]) -> Result<Self, SharedError> {
        for (i, &source) in source_neurons.iter().enumerate() {
            if source_neurons[..i].contains(&source) {
                return Err(SharedError::InvalidFeatures);
            }
        }
        Ok(Self {
            feature_layer,
            source_neurons,
            mask: 1,
            phase: false,
            commits: 0,
        })
    }

    #[must_use]
    pub fn model(&self) -> FiveCircle {
        neuron(self.mask, self.phase).expect("initialized valid mask")
    }

    #[must_use]
    pub fn coupling_mask(&self) -> u8 {
        self.mask
    }

    #[must_use]
    pub fn phase(&self) -> bool {
        self.phase
    }

    #[must_use]
    pub fn commits(&self) -> u64 {
        self.commits
    }

    #[must_use]
    pub fn predict(&self, features: [u64; CIRCLES]) -> u64 {
        predict_bits(self.mask, self.phase, features)
    }

    /// Counterfactual scalar reward: change in observed current-task
    /// prediction hits. If replay sets are supplied, NEVER accept any
    /// reduction of prior TRAIN hits. Independent VALID/TEST are not read.
    ///
    /// No mutation is performed for a tie. One neuron/connection-mask
    /// candidate is accepted at most once per task phase.
    pub fn fit(
        &mut self,
        current: &[TaskBatch],
        replay: &[&[TaskBatch]],
    ) -> Result<SharedUpdate, SharedError> {
        if current.is_empty()
            || current.len() > 1_024
            || current.iter().all(|b| b.observed == 0)
            || replay.len() > 128
            || replay
                .iter()
                .any(|r| r.is_empty() || r.len() > 1_024)
        {
            return Err(SharedError::InvalidTraining);
        }
        let old_cur = score_set(self.mask, self.phase, current);
        let old_prior: Vec<u64> = replay
            .iter()
            .map(|items| score_set(self.mask, self.phase, items))
            .collect();
        let previous_before: u64 = old_prior.iter().sum();
        let mut selected = (self.mask, self.phase, old_cur, previous_before);
        let mut tested = 0;
        for mask in 1..=31_u8 {
            for phase in [false, true] {
                tested += 1;
                let mut retained = 0;
                let mut allowed = true;
                for (group, &floor) in replay.iter().zip(&old_prior) {
                    let seen = score_set(mask, phase, group);
                    if seen < floor {
                        allowed = false;
                        break;
                    }
                    retained += seen;
                }
                if !allowed {
                    continue;
                }
                let improvement = score_set(mask, phase, current);
                if improvement > selected.2
                    || (improvement == selected.2 && retained > selected.3)
                {
                    selected = (mask, phase, improvement, retained);
                }
            }
        }
        let accepted = (selected.0, selected.1) != (self.mask, self.phase);
        let changed_coupling_bits = if accepted {
            (selected.0 ^ self.mask).count_ones()
        } else {
            0
        };
        if accepted {
            self.mask = selected.0;
            self.phase = selected.1;
            self.commits += 1;
        }
        Ok(SharedUpdate {
            accepted,
            candidates_tested: tested,
            current_before: old_cur,
            current_after: selected.2,
            current_total: current.iter().map(|x| u64::from(x.observed.count_ones())).sum(),
            previous_before,
            previous_after: selected.3,
            changed_coupling_bits,
            commits: self.commits,
        })
    }

    /// Data-only persistence with binding checks. The hash is for accidental
    /// corruption detection, not authentication or a backbone identity check.
    #[must_use]
    pub fn checkpoint(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(CHECKPOINT_LEN);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&1_u32.to_le_bytes());
        out.extend_from_slice(&self.feature_layer.to_le_bytes());
        for &source in &self.source_neurons {
            out.extend_from_slice(&source.to_le_bytes());
        }
        out.push(self.mask);
        out.push(u8::from(self.phase));
        out.extend_from_slice(&self.commits.to_le_bytes());
        out.extend_from_slice(&checksum(&out).to_le_bytes());
        out
    }

    pub fn restore(
        data: &[u8],
        feature_layer: u32,
        source_neurons: [u32; 3],
    ) -> Result<Self, SharedError> {
        let mut result = Self::new(feature_layer, source_neurons)?;
        if data.len() != CHECKPOINT_LEN || &data[..8] != MAGIC {
            return Err(SharedError::CorruptCheckpoint);
        }
        let u32_at = |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().expect("length"));
        if u32_at(8) != 1 || u32_at(12) != feature_layer {
            return Err(SharedError::CorruptCheckpoint);
        }
        for (i, &source) in source_neurons.iter().enumerate() {
            if u32_at(16 + i * 4) != source {
                return Err(SharedError::CorruptCheckpoint);
            }
        }
        let found = u64::from_le_bytes(data[CHECKPOINT_LEN - 8..].try_into().expect("length"));
        if found != checksum(&data[..CHECKPOINT_LEN - 8])
            || !(1..=31).contains(&data[28])
            || data[29] > 1
        {
            return Err(SharedError::CorruptCheckpoint);
        }
        result.mask = data[28];
        result.phase = data[29] != 0;
        result.commits = u64::from_le_bytes(data[30..38].try_into().expect("length"));
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset(mask: u8, phase: bool, context: u8) -> Vec<TaskBatch> {
        (0..4)
            .map(|i| {
                let mut state = 0x1122_3344_u64 ^ (i as u64 * 0x9919);
                let mut next = || {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state
                };
                let feature = [
                    next(),
                    next(),
                    next(),
                    if context & 1 != 0 { u64::MAX } else { 0 },
                    if context & 2 != 0 { u64::MAX } else { 0 },
                ];
                TaskBatch {
                    labels: predict_bits(mask, phase, feature),
                    features: feature,
                    observed: u64::MAX,
                }
            })
            .collect()
    }

    #[test]
    fn learns_multiple_contexts_without_allocating_task_heads() {
        let mut one = SharedContextNeuron::new(7, [0, 1, 2]).unwrap();
        let target = 0b1_1011;
        let mut memory = Vec::new();
        for context in [0, 1, 2, 3] {
            let task = dataset(target, false, context);
            let replay: Vec<&[TaskBatch]> = memory.iter().map(Vec::as_slice).collect();
            let report = one.fit(&task, &replay).unwrap();
            assert_eq!(report.current_after, report.current_total);
            assert_eq!(report.candidates_tested, 62);
            for prior in &memory {
                assert_eq!(score_set(one.mask, one.phase, prior), 256);
            }
            memory.push(task);
        }
        assert_eq!(one.coupling_mask(), target);
        assert!(one.commits() >= 2);
        let bytes = one.checkpoint();
        assert_eq!(bytes.len(), CHECKPOINT_LEN);
        assert_eq!(
            SharedContextNeuron::restore(&bytes, 7, [0, 1, 2]).unwrap(),
            one
        );
        assert!(SharedContextNeuron::restore(&bytes, 6, [0, 1, 2]).is_err());
        let mut damaged = bytes;
        damaged[28] ^= 0x04;
        assert_eq!(
            SharedContextNeuron::restore(&damaged, 7, [0, 1, 2]),
            Err(SharedError::CorruptCheckpoint)
        );
    }

    #[test]
    fn Fourier_coupling_matches_64_lane_prediction_for_all_masks() {
        use crate::fourier5_hierarchy::{compiled_truth_table, shannon_64};
        let lanes = [
            0xaaaa_aaaa_aaaa_aaaa,
            0xcccc_cccc_cccc_cccc,
            0xf0f0_f0f0_f0f0_f0f0,
            0xff00_ff00_ff00_ff00,
            0xffff_0000_ffff_0000,
        ];
        for mask in 1..=31 {
            for phase in [false, true] {
                assert_eq!(
                    predict_bits(mask, phase, lanes),
                    shannon_64(compiled_truth_table(neuron(mask, phase).unwrap()), lanes)
                );
            }
        }
    }

    #[test]
    fn malformed_training_is_refused_without_state_change() {
        let mut model = SharedContextNeuron::new(7, [0, 1, 2]).unwrap();
        let before = model.clone();
        assert_eq!(model.fit(&[], &[]), Err(SharedError::InvalidTraining));
        assert_eq!(before, model);
        assert_eq!(SharedContextNeuron::new(7, [0, 0, 2]), Err(SharedError::InvalidFeatures));
    }
}

//! Exact output-reward credit across a fixed, shared multi-neuron Boolean circuit.
//!
//! Three mutable hidden Fourier-Walsh neurons each receive five *frozen*
//! backbone features. One mutable Fourier root combines their three outputs
//! and two observable context bits. All FOUR neurons are shared by all tasks:
//! no task-indexed heads and no change to the 1.56M backbone neurons.
//! Counterfactual output reward supports individual and joint hidden edits.
//! This is a bounded realizable synthetic family, not generic BANC learning.
use crate::fourier5::{Circle, FiveCircle, CIRCLES};

const MAGIC: &[u8; 8] = b"V888HC01";
const MAX_GROUPS: usize = 128;
const MAX_EXAMPLES: usize = 1_024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HiddenError {
    InvalidBinding,
    InvalidTraining,
    CorruptCheckpoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WalshGate {
    pub mask: u8,
    pub phase: bool,
}

impl WalshGate {
    #[must_use]
    pub fn predict(self, features: [u64; CIRCLES]) -> u64 {
        let mut output = if self.phase { u64::MAX } else { 0 };
        for (index, plane) in features.iter().enumerate() {
            if self.mask & (1_u8 << index) != 0 {
                output ^= *plane;
            }
        }
        output
    }

    #[must_use]
    pub fn as_five_circle(self) -> FiveCircle {
        let mut circles = [Circle::default(); CIRCLES];
        circles[0] = Circle {
            mask: self.mask,
            radius: 1,
            phase_pi: self.phase,
        };
        FiveCircle::new(circles, 0).expect("valid five-bit Walsh character")
    }
}

/// The 15 feature words are grouped into three disjoint blocks of five.
/// Context is an explicit observable 2-bit input, NOT a task-specific head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HiddenExample {
    pub blocks: [[u64; CIRCLES]; 3],
    pub context: u8,
    pub label: u64,
    pub observed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HiddenCredit {
    pub accepted: bool,
    pub proposals: usize,
    pub hidden_changed: usize,
    pub root_changed: bool,
    pub reward_before: u64,
    pub reward_after: u64,
    pub observed_total: u64,
    pub replay_before: u64,
    pub replay_after: u64,
    pub commits: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiddenCircuit {
    feature_layer: u32,
    source_neurons: [u32; 15],
    hidden: [WalshGate; 3],
    root: WalshGate,
    commits: u64,
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut state = 0xcbf2_9ce4_8422_2325_u64;
    for &b in bytes {
        state = (state ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    state
}

fn valid_gate(gate: WalshGate) -> bool {
    (1..=31).contains(&gate.mask)
}

impl HiddenCircuit {
    pub fn new(layer: u32, source_neurons: [u32; 15]) -> Result<Self, HiddenError> {
        for (index, &node) in source_neurons.iter().enumerate() {
            if source_neurons[..index].contains(&node) {
                return Err(HiddenError::InvalidBinding);
            }
        }
        Ok(Self {
            feature_layer: layer,
            source_neurons,
            hidden: [
                WalshGate { mask: 1, phase: false },
                WalshGate { mask: 1, phase: false },
                WalshGate { mask: 1, phase: false },
            ],
            root: WalshGate { mask: 0b00111, phase: false },
            commits: 0,
        })
    }

    #[must_use]
    pub fn hidden(&self) -> [WalshGate; 3] {
        self.hidden
    }

    #[must_use]
    pub fn root(&self) -> WalshGate {
        self.root
    }

    #[must_use]
    pub fn commits(&self) -> u64 {
        self.commits
    }

    #[must_use]
    pub fn predict(&self, item: &HiddenExample) -> u64 {
        let inputs = [
            self.hidden[0].predict(item.blocks[0]),
            self.hidden[1].predict(item.blocks[1]),
            self.hidden[2].predict(item.blocks[2]),
            if item.context & 1 != 0 { u64::MAX } else { 0 },
            if item.context & 2 != 0 { u64::MAX } else { 0 },
        ];
        self.root.predict(inputs)
    }

    #[must_use]
    pub fn score(&self, data: &[HiddenExample]) -> (u64, u64) {
        data.iter().fold((0, 0), |(correct, total), example| {
            let error = (self.predict(example) ^ example.label) & example.observed;
            let observed = u64::from(example.observed.count_ones());
            (correct + observed - u64::from(error.count_ones()), total + observed)
        })
    }

    fn valid_groups(data: &[HiddenExample], replay: &[&[HiddenExample]]) -> bool {
        !data.is_empty()
            && data.len() <= MAX_EXAMPLES
            && data.iter().any(|e| e.observed != 0)
            && replay.len() <= MAX_GROUPS
            && replay.iter().all(|group| !group.is_empty() && group.len() <= MAX_EXAMPLES)
            && data.iter().chain(replay.iter().flat_map(|group| group.iter()))
                .all(|e| e.context <= 3)
    }

    fn replay_scores(&self, replay: &[&[HiddenExample]]) -> Vec<u64> {
        replay.iter().map(|group| self.score(group).0).collect()
    }

    fn meets_replay(&self, replay: &[&[HiddenExample]], floor: &[u64]) -> Option<u64> {
        let mut score = 0;
        for (data, &minimum) in replay.iter().zip(floor) {
            let correct = self.score(data).0;
            if correct < minimum {
                return None;
            }
            score += correct;
        }
        Some(score)
    }

    fn summarize(
        &mut self,
        proposed: Self,
        best: (u64, u64),
        initial: (u64, u64),
        total: u64,
        proposals: usize,
    ) -> HiddenCredit {
        let hidden_changed = self.hidden.iter().zip(proposed.hidden)
            .filter(|(a, b)| **a != *b).count();
        let root_changed = self.root != proposed.root;
        let accepted = (hidden_changed != 0 || root_changed) && best.0 > initial.0;
        if accepted {
            self.hidden = proposed.hidden;
            self.root = proposed.root;
            self.commits += 1;
        }
        HiddenCredit {
            accepted,
            proposals,
            hidden_changed: if accepted { hidden_changed } else { 0 },
            root_changed: accepted && root_changed,
            reward_before: initial.0,
            reward_after: if accepted { best.0 } else { initial.0 },
            observed_total: total,
            replay_before: initial.1,
            replay_after: if accepted { best.1 } else { initial.1 },
            commits: self.commits,
        }
    }

    /// Exhaustive *joint* counterfactual credit over three hidden coupling
    /// masks and one output phase. Search only TRAIN. An individual hidden
    /// mutation often has NO useful signal with two incorrect parity blocks.
    /// Candidate space: 31^3 * 2 = 59,582 exactly; 0 extra neurons.
    /// Root mask is fixed during this cooperative first-stage search.
    pub fn fit_joint(
        &mut self,
        current: &[HiddenExample],
        replay: &[&[HiddenExample]],
    ) -> Result<HiddenCredit, HiddenError> {
        if !Self::valid_groups(current, replay) {
            return Err(HiddenError::InvalidTraining);
        }
        let initial_score = self.score(current);
        let floors = self.replay_scores(replay);
        let initial_replay: u64 = floors.iter().sum();
        let initial = (initial_score.0, initial_replay);
        let mut proposed = self.clone();
        let mut best = initial;
        let mut candidate = self.clone();
        let mut proposals = 0;
        for a in 1..=31 {
            for b in 1..=31 {
                for c in 1..=31 {
                    candidate.hidden[0] = WalshGate { mask: a, phase: false };
                    candidate.hidden[1] = WalshGate { mask: b, phase: false };
                    candidate.hidden[2] = WalshGate { mask: c, phase: false };
                    for phase in [false, true] {
                        proposals += 1;
                        candidate.root.phase = phase;
                        let reward = candidate.score(current).0;
                        if reward <= best.0 {
                            continue;
                        }
                        if let Some(previous) = candidate.meets_replay(replay, &floors) {
                            proposed.hidden = candidate.hidden;
                            proposed.root = candidate.root;
                            best = (reward, previous);
                        }
                    }
                }
            }
        }
        Ok(self.summarize(proposed, best, initial, initial_score.1, proposals))
    }

    /// Exact gate-wise output credit for all four shared neurons, including
    /// changes to hidden couplings, phase and root context input couplings.
    /// Every accepted proposal improves current observable TRAIN reward and
    /// preserves each previous TRAIN task relative to its current accuracy.
    pub fn fit_coordinate(
        &mut self,
        current: &[HiddenExample],
        replay: &[&[HiddenExample]],
    ) -> Result<HiddenCredit, HiddenError> {
        if !Self::valid_groups(current, replay) {
            return Err(HiddenError::InvalidTraining);
        }
        let before = self.score(current);
        let floors = self.replay_scores(replay);
        let initial_replay: u64 = floors.iter().sum();
        let initial = (before.0, initial_replay);
        let mut chosen = self.clone();
        let mut best = initial;
        let mut proposals = 0;
        // Route output first, then consider each hidden block: shared
        // weights, all gates eligible, no task index in parameter storage.
        for index in 0..4 {
            for mask in 1..=31 {
                for phase in [false, true] {
                    proposals += 1;
                    let mut candidate = self.clone();
                    let gate = WalshGate { mask, phase };
                    if index == 3 {
                        candidate.root = gate;
                    } else {
                        candidate.hidden[index] = gate;
                    }
                    let correct = candidate.score(current).0;
                    if correct <= best.0 {
                        continue;
                    }
                    if let Some(retained) = candidate.meets_replay(replay, &floors) {
                        chosen = candidate;
                        best = (correct, retained);
                    }
                }
            }
        }
        Ok(self.summarize(chosen, best, initial, before.1, proposals))
    }

    /// A fixed-size data-only snapshot with source binding, edit count and
    /// accidental corruption checksum. Not cryptographic authentication.
    #[must_use]
    pub fn checkpoint(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(100);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&1_u32.to_le_bytes());
        out.extend_from_slice(&self.feature_layer.to_le_bytes());
        for &source in &self.source_neurons {
            out.extend_from_slice(&source.to_le_bytes());
        }
        for gate in self.hidden.iter().chain(std::iter::once(&self.root)) {
            out.push(gate.mask);
            out.push(u8::from(gate.phase));
        }
        out.extend_from_slice(&self.commits.to_le_bytes());
        out.extend_from_slice(&fnv(&out).to_le_bytes());
        out
    }

    pub fn restore(
        bytes: &[u8],
        layer: u32,
        source_neurons: [u32; 15],
    ) -> Result<Self, HiddenError> {
        let mut circuit = Self::new(layer, source_neurons)?;
        if bytes.len() != 100 || &bytes[..8] != MAGIC {
            return Err(HiddenError::CorruptCheckpoint);
        }
        let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().expect("length"));
        if u32_at(8) != 1 || u32_at(12) != layer {
            return Err(HiddenError::CorruptCheckpoint);
        }
        for (i, &source) in source_neurons.iter().enumerate() {
            if u32_at(16 + i * 4) != source {
                return Err(HiddenError::CorruptCheckpoint);
            }
        }
        let sum = u64::from_le_bytes(bytes[92..100].try_into().expect("length"));
        if fnv(&bytes[..92]) != sum {
            return Err(HiddenError::CorruptCheckpoint);
        }
        for (index, gate) in circuit.hidden.iter_mut()
            .chain(std::iter::once(&mut circuit.root)).enumerate()
        {
            let mask = bytes[76 + 2 * index];
            let phase = bytes[77 + 2 * index];
            if !(1..=31).contains(&mask) || phase > 1 {
                return Err(HiddenError::CorruptCheckpoint);
            }
            *gate = WalshGate { mask, phase: phase != 0 };
        }
        circuit.commits = u64::from_le_bytes(bytes[84..92].try_into().expect("length"));
        Ok(circuit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(context: u8, target: &HiddenCircuit, seed: u64) -> HiddenExample {
        let mut state = seed;
        let blocks = std::array::from_fn(|_| std::array::from_fn(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        }));
        let mut result = HiddenExample {
            blocks, context, label: 0, observed: u64::MAX,
        };
        result.label = target.predict(&result);
        result
    }

    #[test]
    fn joint_hidden_credit_and_context_replay() {
        let ids = std::array::from_fn(|i| i as u32);
        let mut teacher = HiddenCircuit::new(7, ids).unwrap();
        for (gate, mask) in teacher.hidden.iter_mut().zip([0b01101, 0b10110, 0b11011]) {
            gate.mask = mask;
        }
        teacher.root.mask = 0b11111;
        let mut model = HiddenCircuit::new(7, ids).unwrap();
        let tasks: Vec<Vec<HiddenExample>> = (0..4)
            .map(|context| (0..4).map(|i| {
                sample(context, &teacher, 123 + context as u64 * 11 + i)
            }).collect())
            .collect();
        let first = model.fit_joint(&tasks[0], &[]).unwrap();
        assert!(first.accepted);
        assert!(first.hidden_changed >= 2);
        assert_eq!(model.score(&tasks[0]), (256, 256));
        for phase in 1..4 {
            let replay: Vec<_> = tasks[..phase].iter().map(Vec::as_slice).collect();
            let report = model.fit_coordinate(&tasks[phase], &replay).unwrap();
            assert!(report.reward_after >= report.reward_before);
            for old in &tasks[..=phase] {
                assert_eq!(model.score(old), (256, 256));
            }
        }
        let state = model.checkpoint();
        assert_eq!(state.len(), 100);
        assert_eq!(HiddenCircuit::restore(&state, 7, ids).unwrap(), model);
        let mut corrupt = state;
        corrupt[76] ^= 0x40;
        assert_eq!(
            HiddenCircuit::restore(&corrupt, 7, ids),
            Err(HiddenError::CorruptCheckpoint)
        );
    }

    #[test]
    fn five_circle_reference_matches_compiled_parity_gates() {
        use crate::fourier5_hierarchy::{compiled_truth_table, shannon_64};
        let features = [
            0xaaaa_aaaa_aaaa_aaaa, 0xcccc_cccc_cccc_cccc,
            0xf0f0_f0f0_f0f0_f0f0, 0xff00_ff00_ff00_ff00,
            0xffff_0000_ffff_0000,
        ];
        for mask in 1..=31 {
            for phase in [false, true] {
                let gate = WalshGate { mask, phase };
                assert_eq!(
                    gate.predict(features),
                    shannon_64(compiled_truth_table(gate.as_five_circle()), features)
                );
            }
        }
    }
}

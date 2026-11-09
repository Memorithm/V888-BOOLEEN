//! Experimental binary Hebbian-like, TEACHER-SUPERVISED edge plasticity.
//! Synapses follow only the original directed graph; no anatomy is invented.
//! Four-bit weight counters (0..15) are updated by XOR/AND carry gates.
//! Not calibrated to biological learning or insect memory.
use crate::connectome::Connectome;
use crate::gates;
use crate::BooleanState;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"V888PLS1";
const VERSION: u64 = 1;
const MAX_WEIGHT: u8 = 15;

fn bad(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Saturating four-bit increment, implemented as a Boolean ripple carry.
#[must_use]
pub fn increment_four(value: u8) -> u8 {
    let value = value & MAX_WEIGHT;
    if value == MAX_WEIGHT {
        return value;
    }
    let mut carry = 1_u8;
    let mut out = 0_u8;
    for i in 0..4 {
        let bit = (value >> i) & 1;
        out |= (bit ^ carry) << i;
        carry &= bit;
    }
    out
}

/// Saturating four-bit decrement, implemented as a Boolean ripple borrow.
#[must_use]
pub fn decrement_four(value: u8) -> u8 {
    let value = value & MAX_WEIGHT;
    if value == 0 {
        return 0;
    }
    let mut borrow = 1_u8;
    let mut out = 0_u8;
    for i in 0..4 {
        let bit = (value >> i) & 1;
        out |= (bit ^ borrow) << i;
        borrow &= !bit & 1;
    }
    out
}

/// Edge list fingerprint, frozen in the weight file to reject incompatible
/// graph snapshots; not a cryptographic hash or source attribution.
fn graph_fingerprint(graph: &Connectome) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for value in (graph.ids.len() as u64).to_le_bytes() {
        h = (h ^ u64::from(value)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    for &id in &graph.ids {
        for byte in id.to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    for &offset in &graph.offsets {
        for byte in offset.to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    for (&target, &count) in graph.targets.iter().zip(&graph.counts) {
        for byte in target.to_le_bytes().into_iter().chain(count.to_le_bytes()) {
            h = (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// No tensor/FPU, no new graph edges, one four-bit plastic coefficient per
/// EXISTING directed contact aggregate. Initial weights can be frozen at 0
/// (untrained) or 1 (non-plastic topology-only comparator).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlasticCircuit {
    weights: Vec<u8>,
    topology: u64,
    neuron_count: usize,
}

impl PlasticCircuit {
    pub fn new(graph: &Connectome, initial_weight: u8) -> io::Result<Self> {
        graph.validate()?;
        if initial_weight > MAX_WEIGHT {
            return Err(bad("initial weight not four-bit"));
        }
        Ok(Self {
            weights: vec![initial_weight; graph.edges()],
            topology: graph_fingerprint(graph),
            neuron_count: graph.ids.len(),
        })
    }

    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        self.weights.len().div_ceil(2)
    }

    fn shape(&self, graph: &Connectome, input: &BooleanState) -> io::Result<()> {
        if self.weights.len() != graph.edges()
            || self.neuron_count != graph.ids.len()
            || input.len() != graph.ids.len()
        {
            return Err(bad("plastic circuit shape mismatch"));
        }
        Ok(())
    }

    /// Teacher explicitly specifies which postsynaptic neurons should spike.
    /// Only existing edges from active sources are reinforced/suppressed.
    /// This is SUPERVISED associative learning; teacher is not a fly recording.
    pub fn train(
        &mut self,
        graph: &Connectome,
        input: &BooleanState,
        teacher: &BooleanState,
    ) -> io::Result<u64> {
        self.shape(graph, input)?;
        if teacher.len() != self.neuron_count {
            return Err(bad("teacher length mismatch"));
        }
        let mut updated = 0_u64;
        for source in input.active_indices() {
            for edge in graph.offsets[source]..graph.offsets[source + 1] {
                let edge = edge as usize;
                let target = graph.targets[edge] as usize;
                self.weights[edge] = if teacher
                    .get(target)
                    .map_err(|_| bad("invalid target index"))?
                {
                    increment_four(self.weights[edge])
                } else {
                    decrement_four(self.weights[edge])
                };
                updated = gates::add(u128::from(updated), 1)?
                    .try_into()
                    .map_err(|_| bad("update count overflow"))?;
            }
        }
        Ok(updated)
    }

    /// Weighted-synapse Boolean recurrence; summation and threshold use
    /// XOR/AND ripple carries and bitwise comparators only.
    pub fn predict(
        &self,
        graph: &Connectome,
        input: &BooleanState,
        threshold: u128,
    ) -> io::Result<BooleanState> {
        self.shape(graph, input)?;
        if threshold == 0 {
            return Err(bad("threshold must be positive"));
        }
        let mut inputs = vec![0_u128; self.neuron_count];
        for source in input.active_indices() {
            for edge in graph.offsets[source]..graph.offsets[source + 1] {
                let edge = edge as usize;
                let weight = self.weights[edge];
                if weight == 0 {
                    continue;
                }
                let contacts = u128::from(graph.counts[edge]);
                let mut contribution = 0_u128;
                for shift in 0..4 {
                    if (weight >> shift) & 1 != 0 {
                        contribution = gates::add(contribution, contacts << shift)?;
                    }
                }
                let target = graph.targets[edge] as usize;
                inputs[target] = gates::add(inputs[target], contribution)?;
            }
        }
        let mut result = BooleanState::new(self.neuron_count);
        for (i, &value) in inputs.iter().enumerate() {
            if gates::ge(value, threshold) {
                result
                    .set(i, true)
                    .map_err(|_| bad("invalid output index"))?;
            }
        }
        Ok(result)
    }

    /// Store weights in packed four-bit form with graph topology fingerprint.
    /// Does not replace existing checkpoints.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut out = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
        out.write_all(MAGIC)?;
        out.write_all(&VERSION.to_le_bytes())?;
        out.write_all(&(self.neuron_count as u64).to_le_bytes())?;
        out.write_all(&(self.weights.len() as u64).to_le_bytes())?;
        out.write_all(&self.topology.to_le_bytes())?;
        for group in self.weights.chunks(2) {
            let low = group[0];
            let high = group.get(1).copied().unwrap_or(0);
            out.write_all(&[low | (high << 4)])?;
        }
        out.flush()
    }

    pub fn load(path: &Path, graph: &Connectome) -> io::Result<Self> {
        graph.validate()?;
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        let mut input = BufReader::new(file);
        let mut magic = [0_u8; 8];
        input.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(bad("invalid plastic checkpoint format"));
        }
        let mut read_number = || -> io::Result<u64> {
            let mut raw = [0_u8; 8];
            input.read_exact(&mut raw)?;
            Ok(u64::from_le_bytes(raw))
        };
        let version = read_number()?;
        let n = read_number()?;
        let e = read_number()?;
        let topology = read_number()?;
        if version != VERSION
            || n != graph.ids.len() as u64
            || e != graph.edges() as u64
            || topology != graph_fingerprint(graph)
            || size != 40 + e.div_ceil(2)
        {
            return Err(bad("incompatible graph/plasticity checkpoint"));
        }
        let mut weights = Vec::with_capacity(e as usize);
        for _ in 0..e.div_ceil(2) {
            let mut pair = [0_u8; 1];
            input.read_exact(&mut pair)?;
            weights.push(pair[0] & 15);
            if weights.len() < e as usize {
                weights.push(pair[0] >> 4);
            } else if pair[0] >> 4 != 0 {
                return Err(bad("invalid last padding nibble"));
            }
        }
        Ok(Self {
            weights,
            topology,
            neuron_count: n as usize,
        })
    }

    #[must_use]
    pub fn weights(&self) -> &[u8] {
        &self.weights
    }

    #[must_use]
    pub fn topology_fingerprint(&self) -> u64 {
        self.topology
    }

    /// Saturating four-bit step on one existing edge. Returns whether the
    /// stored coefficient actually changed (false at the 0/15 plateaus).
    pub fn apply_edge_step(&mut self, edge: usize, reinforce: bool) -> io::Result<bool> {
        let current = self
            .weights
            .get(edge)
            .copied()
            .ok_or_else(|| bad("edge index out of range"))?;
        let next = if reinforce {
            increment_four(current)
        } else {
            decrement_four(current)
        };
        self.weights[edge] = next;
        Ok(next != current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture() -> Connectome {
        Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 1, 2, 2],
            targets: vec![2, 2],
            counts: vec![1, 1],
        }
    }
    fn stimulus(id: usize) -> BooleanState {
        let mut state = BooleanState::new(3);
        state.set(id, true).unwrap();
        state
    }

    #[test]
    fn four_bit_gates_saturate_and_match_integer_oracle() {
        for n in 0..=15 {
            assert_eq!(increment_four(n), (n + 1).min(15));
            assert_eq!(decrement_four(n), n.saturating_sub(1));
        }
    }

    #[test]
    fn supervised_recall_beats_identically_sized_frozen_baselines() {
        let graph = fixture();
        let mut learner = PlasticCircuit::new(&graph, 0).unwrap();
        let frozen_zero = learner.clone();
        let frozen_topology = PlasticCircuit::new(&graph, 1).unwrap();
        let positive = stimulus(0);
        let negative = stimulus(1);
        let mut target = BooleanState::new(3);
        target.set(2, true).unwrap();
        for _ in 0..8 {
            assert_eq!(learner.train(&graph, &positive, &target).unwrap(), 1);
            assert_eq!(
                learner
                    .train(&graph, &negative, &BooleanState::new(3))
                    .unwrap(),
                1
            );
        }
        let cases = [(positive, true), (negative, false)];
        let score = |model: &PlasticCircuit| -> usize {
            cases
                .iter()
                .filter(|(stimulus, expected)| {
                    model.predict(&graph, stimulus, 1).unwrap().get(2).unwrap() == *expected
                })
                .count()
        };
        assert_eq!(score(&frozen_zero), 1);
        assert_eq!(score(&frozen_topology), 1);
        assert_eq!(score(&learner), 2);
    }

    #[test]
    fn graph_pinned_checkpoint_roundtrip_and_wrong_graph_rejection() -> io::Result<()> {
        let graph = fixture();
        let mut circuit = PlasticCircuit::new(&graph, 0)?;
        let target = stimulus(2);
        circuit.train(&graph, &stimulus(0), &target)?;
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let file =
            std::env::temp_dir().join(format!("v888-plastic-{}-{unique}.bin", std::process::id()));
        circuit.save(&file)?;
        let after = PlasticCircuit::load(&file, &graph)?;
        assert_eq!(after, circuit);
        let mut bad_graph = graph.clone();
        bad_graph.ids[2] = 31;
        assert!(PlasticCircuit::load(&file, &bad_graph).is_err());
        assert!(circuit.save(&file).is_err());
        std::fs::remove_file(file)?;
        Ok(())
    }

    #[test]
    fn sparse_multi_source_updates_equal_dense_reference() -> io::Result<()> {
        let count = 130;
        let graph = {
            let mut offsets = vec![0_u64];
            let mut targets = Vec::new();
            let mut counts = Vec::new();
            for source in 0..count {
                if matches!(source, 0 | 63 | 64 | 129) {
                    targets.extend([3, 5, 124]);
                    counts.extend([1, 2, 3]);
                }
                offsets.push(targets.len() as u64);
            }
            Connectome {
                ids: (0..count as u64).collect(),
                offsets,
                targets,
                counts,
            }
        };
        graph.validate()?;
        let mut input = BooleanState::new(count);
        for source in [0, 63, 64, 129] {
            input.set(source, true).unwrap();
        }
        let mut teacher = BooleanState::new(count);
        for target in [3, 124] {
            teacher.set(target, true).unwrap();
        }
        let mut fast = PlasticCircuit::new(&graph, 0)?;
        let mut dense_reference = fast.clone();
        for _ in 0..3 {
            fast.train(&graph, &input, &teacher)?;
            for (source, window) in graph.offsets.windows(2).enumerate() {
                if !input.get(source).unwrap() {
                    continue;
                }
                for edge in window[0]..window[1] {
                    let edge = edge as usize;
                    let target = graph.targets[edge] as usize;
                    dense_reference.weights[edge] = if teacher.get(target).unwrap() {
                        increment_four(dense_reference.weights[edge])
                    } else {
                        decrement_four(dense_reference.weights[edge])
                    };
                }
            }
            assert_eq!(fast.weights, dense_reference.weights);
        }
        for threshold in [1_u128, 2, 6, 10, 200] {
            let fast_output = fast.predict(&graph, &input, threshold)?;
            let mut totals = vec![0_u128; count];
            for (source, window) in graph.offsets.windows(2).enumerate() {
                if !input.get(source).unwrap() {
                    continue;
                }
                for edge in window[0]..window[1] {
                    let edge = edge as usize;
                    totals[graph.targets[edge] as usize] +=
                        u128::from(graph.counts[edge]) * u128::from(fast.weights[edge]);
                }
            }
            let expected: Vec<usize> = totals
                .iter()
                .enumerate()
                .filter_map(|(index, &sum)| (sum >= threshold).then_some(index))
                .collect();
            assert_eq!(fast_output.active_indices().collect::<Vec<_>>(), expected);
        }
        Ok(())
    }

    #[test]
    fn shape_and_teacher_validation_fail_closed() -> io::Result<()> {
        let graph = fixture();
        let mut circuit = PlasticCircuit::new(&graph, 0)?;
        let wrong_state = BooleanState::new(2);
        assert!(circuit.predict(&graph, &wrong_state, 1).is_err());
        assert!(circuit.train(&graph, &stimulus(0), &wrong_state).is_err());
        assert!(circuit.predict(&graph, &stimulus(0), 0).is_err());
        assert!(PlasticCircuit::new(&graph, 16).is_err());
        Ok(())
    }
}

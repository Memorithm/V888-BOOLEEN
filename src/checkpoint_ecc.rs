//! Optional SECDED protection for frozen V888 Boolean-state snapshots.
//!
//! The live engine remains one bit per neuron. This codec is intentionally
//! outside propagation, learning and selection. Each 128-bit snapshot block
//! has a separately stored 9-bit shortened extended-Hamming tag (8 syndrome
//! bits and one overall parity bit). One flipped bit per codeword is corrected;
//! every pair of flipped bits in the same codeword is detected and rejected.
//! Three or more flips are outside the guarantee.
//!
//! The tags are currently held as `u16` values: this costs 16/128 = 12.5%
//! metadata for full blocks, rather than the 7.03125% theoretical bit-packed
//! minimum. The shape/length header is NOT protected here. For durable file
//! storage, protect length and framing separately and commit atomically.
//! SECDED is not an authentication or adversarial tamper-detection mechanism.

use crate::BooleanState;

/// The protected block and metadata layout are kept separate from the engine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtectedBooleanSnapshot {
    len: usize,
    blocks: Vec<[u64; 2]>,
    tags: Vec<u16>,
}

/// The location of a corrected bit in a Hamming codeword.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitRepair {
    /// Zero-based 128-bit block index within the snapshot.
    pub block: usize,
    /// One-based Hamming position 1..=136; 137 means overall parity.
    pub codeword_position: u16,
}

/// A new Boolean state and a record of the repairs required to reconstruct it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveredState {
    pub state: BooleanState,
    pub repairs: Vec<BitRepair>,
}

/// Validation and decoding failures. No partially recovered state is returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotError {
    /// Number of blocks or metadata tags disagrees with the declared width.
    InvalidShape,
    /// The 16-bit storage includes values outside the defined 9-bit tag.
    InvalidTag { block: usize },
    /// Two-bit corruption was detected in a protected codeword.
    Uncorrectable { block: usize },
    /// Recovered bits remain set outside the declared Boolean-state length.
    InvalidPadding { block: usize },
    /// The declared neuron count exceeds the caller's recovery allocation budget.
    NeuronBudgetExceeded { len: usize, max_neurons: usize },
}

fn block_count(len: usize) -> usize {
    len / 128 + usize::from(len % 128 != 0)
}

/// Compute the eight parity-check bits from the 128 payload bits.
///
/// Hamming positions 1,2,4,...,128 are reserved for the separate tag.
fn payload_syndrome(data: u128) -> u8 {
    let mut parity = 0u8;
    let mut bit = 0u32;
    for position in 1u16..=136 {
        if !position.is_power_of_two() {
            if data & (1u128 << bit) != 0 {
                parity ^= position as u8;
            }
            bit += 1;
        }
    }
    debug_assert_eq!(bit, 128);
    parity
}

fn make_tag(data: u128) -> u16 {
    let check = payload_syndrome(data);
    let overall = (data.count_ones() ^ check.count_ones()) & 1;
    u16::from(check) | (overall as u16) << 8
}

/// Return the corrected codeword position, or None when already clean.
fn repair(data: &mut u128, tag: &mut u16) -> Result<Option<u16>, ()> {
    if *tag > 0x01ff {
        return Err(());
    }
    let syndrome = payload_syndrome(*data) ^ (*tag as u8);
    let odd = (data.count_ones() ^ tag.count_ones()) & 1 != 0;
    match (syndrome, odd) {
        (0, false) => Ok(None),
        (0, true) => {
            *tag ^= 1u16 << 8;
            Ok(Some(137))
        }
        (_, false) => Err(()),
        (position, true) if position > 136 => Err(()),
        (position, true) => {
            let position = u16::from(position);
            if position.is_power_of_two() {
                *tag ^= 1u16 << position.trailing_zeros();
            } else {
                let mut payload_index = 0u32;
                for p in 1..position {
                    if !p.is_power_of_two() {
                        payload_index += 1;
                    }
                }
                *data ^= 1u128 << payload_index;
            }
            Ok(Some(position))
        }
    }
}

impl ProtectedBooleanSnapshot {
    /// Capture the unchanged, one-bit-per-position Boolean state.
    #[must_use]
    pub fn capture(state: &BooleanState) -> Self {
        let len = state.len();
        let mut blocks = Vec::with_capacity(block_count(len));
        let mut tags = Vec::with_capacity(block_count(len));
        let words = state.words();
        for index in 0..block_count(len) {
            let low = words.get(2 * index).copied().unwrap_or(0);
            let high = words.get(2 * index + 1).copied().unwrap_or(0);
            let data = u128::from(low) | (u128::from(high) << 64);
            blocks.push([low, high]);
            tags.push(make_tag(data));
        }
        Self { len, blocks, tags }
    }

    /// Adopt previously encoded metadata with structural validation.
    ///
    /// The checksum is checked at `recover`, not when constructing the object.
    /// An external durable format must protect the `len` header separately.
    pub fn from_parts(
        len: usize,
        blocks: Vec<[u64; 2]>,
        tags: Vec<u16>,
    ) -> Result<Self, SnapshotError> {
        if blocks.len() != block_count(len) || tags.len() != blocks.len() {
            return Err(SnapshotError::InvalidShape);
        }
        Ok(Self { len, blocks, tags })
    }

    /// Return the framing width, blocks and 9-bit metadata for serialization.
    ///
    /// The caller must protect and version the framing independently.
    #[must_use]
    pub fn into_parts(self) -> (usize, Vec<[u64; 2]>, Vec<u16>) {
        (self.len, self.blocks, self.tags)
    }

    /// Verify, repair single-bit faults and reconstruct a fresh Boolean state.
    ///
    /// No changes are made to the snapshot or a live engine state. If one
    /// block contains two flipped bits, no partial state escapes.
    pub fn recover(&self) -> Result<RecoveredState, SnapshotError> {
        self.recover_bounded(usize::MAX)
    }

    /// Recover only when the declared state size fits an explicit neuron budget.
    ///
    /// The bound is enforced before allocating the destination BooleanState.
    /// The protected snapshot has already been loaded into memory; callers
    /// must separately limit encoded input size and authenticate its framing.
    pub fn recover_bounded(
        &self,
        max_neurons: usize,
    ) -> Result<RecoveredState, SnapshotError> {
        if self.blocks.len() != block_count(self.len) || self.tags.len() != self.blocks.len() {
            return Err(SnapshotError::InvalidShape);
        }
        if self.len > max_neurons {
            return Err(SnapshotError::NeuronBudgetExceeded {
                len: self.len,
                max_neurons,
            });
        }
        let mut state = BooleanState::new(self.len);
        let mut repairs = Vec::new();
        for (block, (&words, &stored_tag)) in self.blocks.iter().zip(&self.tags).enumerate() {
            let mut data = u128::from(words[0]) | (u128::from(words[1]) << 64);
            let mut tag = stored_tag;
            if tag > 0x01ff {
                return Err(SnapshotError::InvalidTag { block });
            }
            match repair(&mut data, &mut tag) {
                Ok(Some(codeword_position)) => repairs.push(BitRepair {
                    block,
                    codeword_position,
                }),
                Ok(None) => {}
                Err(()) => return Err(SnapshotError::Uncorrectable { block }),
            }
            let valid_bits = self.len.saturating_sub(block * 128).min(128);
            if valid_bits < 128 && (data >> valid_bits) != 0 {
                return Err(SnapshotError::InvalidPadding { block });
            }
            while data != 0 {
                let offset = data.trailing_zeros() as usize;
                state
                    .set(block * 128 + offset, true)
                    .expect("corrected and padding-validated bit is in bounds");
                data &= data - 1;
            }
        }
        Ok(RecoveredState { state, repairs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent positioning oracle used to construct fault injections.
    fn flip(codeword: &mut u128, tag: &mut u16, position: u16) {
        if position == 137 {
            *tag ^= 1u16 << 8;
        } else if position.is_power_of_two() {
            *tag ^= 1u16 << position.trailing_zeros();
        } else {
            let index = (1..position).filter(|p| !p.is_power_of_two()).count();
            *codeword ^= 1u128 << index;
        }
    }

    #[test]
    fn fixed_codeword_vectors_match_independent_hamming_oracle() {
        // Independent position-enumeration reference vectors. These are
        // compatible with SciRust's 128-bit SECDED codeword convention:
        // low u64 payload first, then high u64; 9-bit tag separately.
        assert_eq!(make_tag(0), 0x000);
        assert_eq!(make_tag(u128::MAX), 0x077);
        assert_eq!(make_tag(1), 0x103);
        assert_eq!(make_tag(1u128 << 64), 0x148);
        assert_eq!(make_tag(0x1357_9bdf_2468_ace0_fedc_ba98_7654_3210), 0x090);
    }

    #[test]
    fn roundtrip_all_word_edges_and_full_v888_width() {
        for len in [0, 1, 63, 64, 65, 127, 128, 129, 130, 257, 188_508] {
            let mut state = BooleanState::new(len);
            for index in 0..len {
                if index % 17 == 3 || index % 113 == 0 {
                    state.set(index, true).unwrap();
                }
            }
            let snapshot = ProtectedBooleanSnapshot::capture(&state);
            assert_eq!(snapshot.blocks.len(), block_count(len));
            let restored = snapshot.recover().unwrap();
            assert_eq!(restored.state, state);
            assert!(restored.repairs.is_empty());
            let (length, blocks, tags) = snapshot.into_parts();
            let rebuilt = ProtectedBooleanSnapshot::from_parts(length, blocks, tags).unwrap();
            assert_eq!(rebuilt.recover().unwrap().state, state);
        }
    }

    #[test]
    fn exhaustively_corrects_every_single_and_rejects_every_double_fault() {
        for clean in [0u128, u128::MAX, 0x1357_9bdf_2468_ace0_fedc_ba98_7654_3210] {
            let clean_tag = make_tag(clean);
            for first in 1..=137u16 {
                let mut data = clean;
                let mut tag = clean_tag;
                flip(&mut data, &mut tag, first);
                assert_eq!(repair(&mut data, &mut tag), Ok(Some(first)));
                assert_eq!((data, tag), (clean, clean_tag));
                for second in (first + 1)..=137u16 {
                    let mut corrupt_data = clean;
                    let mut corrupt_tag = clean_tag;
                    flip(&mut corrupt_data, &mut corrupt_tag, first);
                    flip(&mut corrupt_data, &mut corrupt_tag, second);
                    let before = (corrupt_data, corrupt_tag);
                    assert_eq!(repair(&mut corrupt_data, &mut corrupt_tag), Err(()));
                    assert_eq!((corrupt_data, corrupt_tag), before);
                }
            }
        }
    }

    #[test]
    fn repair_is_reported_without_mutating_captured_data() {
        let mut original = BooleanState::new(257);
        for bit in [0, 4, 64, 127, 128, 196, 256] {
            original.set(bit, true).unwrap();
        }
        let snapshot = ProtectedBooleanSnapshot::capture(&original);
        let (len, mut blocks, mut tags) = snapshot.clone().into_parts();
        let mut data = u128::from(blocks[1][0]) | (u128::from(blocks[1][1]) << 64);
        flip(&mut data, &mut tags[1], 137);
        blocks[1] = [data as u64, (data >> 64) as u64];
        let damaged = ProtectedBooleanSnapshot::from_parts(len, blocks, tags).unwrap();
        let recovered = damaged.recover().unwrap();
        assert_eq!(recovered.state, original);
        assert_eq!(
            recovered.repairs,
            vec![BitRepair {
                block: 1,
                codeword_position: 137
            }]
        );
        assert_eq!(snapshot.recover().unwrap().state, original);
    }

    #[test]
    fn rejects_uncorrectable_and_malformed_snapshots() {
        let mut original = BooleanState::new(129);
        original.set(0, true).unwrap();
        let snapshot = ProtectedBooleanSnapshot::capture(&original);
        let (len, mut blocks, mut tags) = snapshot.into_parts();
        let mut data = u128::from(blocks[0][0]) | (u128::from(blocks[0][1]) << 64);
        flip(&mut data, &mut tags[0], 3);
        flip(&mut data, &mut tags[0], 5);
        blocks[0] = [data as u64, (data >> 64) as u64];
        let damaged =
            ProtectedBooleanSnapshot::from_parts(len, blocks.clone(), tags.clone()).unwrap();
        assert_eq!(
            damaged.recover(),
            Err(SnapshotError::Uncorrectable { block: 0 })
        );

        tags[0] = 0x200;
        assert_eq!(
            ProtectedBooleanSnapshot::from_parts(len, blocks.clone(), tags)
                .unwrap()
                .recover(),
            Err(SnapshotError::InvalidTag { block: 0 })
        );
        assert_eq!(
            ProtectedBooleanSnapshot::from_parts(len, vec![blocks[0]], vec![0]),
            Err(SnapshotError::InvalidShape)
        );
    }

    #[test]
    fn rejects_validly_tagged_but_nonzero_padding() {
        let original = BooleanState::new(65);
        let snapshot = ProtectedBooleanSnapshot::capture(&original);
        let (len, mut blocks, mut tags) = snapshot.into_parts();
        blocks[0][1] |= 1u64 << 6;
        let data = u128::from(blocks[0][0]) | (u128::from(blocks[0][1]) << 64);
        tags[0] = make_tag(data);
        let rebuilt = ProtectedBooleanSnapshot::from_parts(len, blocks, tags).unwrap();
        assert_eq!(
            rebuilt.recover(),
            Err(SnapshotError::InvalidPadding { block: 0 })
        );
    }

    #[test]
    fn bounded_recovery_rejects_oversize_before_allocating_new_state() {
        let mut original = BooleanState::new(129);
        original.set(0, true).unwrap();
        original.set(128, true).unwrap();
        let encoded = ProtectedBooleanSnapshot::capture(&original);
        assert_eq!(
            encoded.recover_bounded(128),
            Err(SnapshotError::NeuronBudgetExceeded {
                len: 129,
                max_neurons: 128,
            })
        );
        assert_eq!(encoded.recover_bounded(129).unwrap().state, original);
        assert_eq!(encoded.recover().unwrap().state, original);
    }

    #[test]
    fn bounded_recovery_accepts_zero_width_and_rejects_one_bit_over_budget() {
        let empty = ProtectedBooleanSnapshot::capture(&BooleanState::new(0));
        assert_eq!(empty.recover_bounded(0).unwrap().state.len(), 0);
        let single = ProtectedBooleanSnapshot::capture(&BooleanState::new(1));
        assert_eq!(
            single.recover_bounded(0),
            Err(SnapshotError::NeuronBudgetExceeded {
                len: 1,
                max_neurons: 0,
            })
        );
        assert_eq!(single.recover_bounded(1).unwrap().state.len(), 1);
    }

    #[test]
    fn multiple_blocks_can_each_repair_one_error() {
        let mut state = BooleanState::new(256);
        for bit in [0, 1, 63, 64, 127, 128, 191, 255] {
            state.set(bit, true).unwrap();
        }
        let (len, mut blocks, mut tags) = ProtectedBooleanSnapshot::capture(&state).into_parts();
        for block in 0..2 {
            let mut data = u128::from(blocks[block][0]) | (u128::from(blocks[block][1]) << 64);
            flip(&mut data, &mut tags[block], 42);
            blocks[block] = [data as u64, (data >> 64) as u64];
        }
        let snapshot = ProtectedBooleanSnapshot::from_parts(len, blocks, tags).unwrap();
        let restored = snapshot.recover().unwrap();
        assert_eq!(restored.state, state);
        assert_eq!(restored.repairs.len(), 2);
        assert_eq!(restored.repairs[0].block, 0);
        assert_eq!(restored.repairs[1].block, 1);
    }
}

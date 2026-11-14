//! Exact comparison of composed five-circle neurons and equivalent Boolean LUTs.
//!
//! No learning or plasticity. The three arms have identical model topology:
//! scalar Fourier-Walsh, scalar compiled truth tables, and 64-lane bit-sliced
//! truth tables. A separate Boolean oracle checks every output on the same
//! generated sample. Timings are descriptive and include transposition cost.
use std::time::Instant;
use v888_booleen::fourier5_hierarchy::{
    FiveCircleHierarchy, GateKind, LeafRouting, BITSLICE_LANES,
};
use v888_booleen::BooleanState;

#[derive(Clone, Copy)]
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        let mut word = self.0;
        word ^= word << 13;
        word ^= word >> 7;
        word ^= word << 17;
        self.0 = word;
        word
    }
}

fn make_states(random: &mut Random, width: usize) -> Vec<BooleanState> {
    let mut states = Vec::with_capacity(BITSLICE_LANES);
    for _ in 0..BITSLICE_LANES {
        let mut state = BooleanState::new(width);
        let mut word = 0_u64;
        for bit in 0..width {
            if bit % 64 == 0 {
                word = random.next();
            }
            if word & (1_u64 << (bit % 64)) != 0 {
                state.set(bit, true).expect("known in-range index");
            }
        }
        states.push(state);
    }
    states
}

/// Boolean reference without Fourier inference or compiled Walsh truth
/// tables; one independent decision per five incoming Boolean values.
fn oracle(state: &BooleanState, routing: &[u16], kinds: &[GateKind]) -> bool {
    let mut current: Vec<bool> = routing
        .iter()
        .map(|&index| state.get(usize::from(index)).expect("validated routing"))
        .collect();
    for &kind in kinds {
        current = current
            .chunks_exact(5)
            .map(|group| {
                let mut input = 0_u8;
                for (bit, &active) in group.iter().enumerate() {
                    input |= u8::from(active) << bit;
                }
                kind.oracle(input)
            })
            .collect();
    }
    current[0]
}

fn profile(kinds: &[GateKind], cases: usize, seed: u64) -> Result<(), String> {
    if cases == 0 || cases % BITSLICE_LANES != 0 {
        return Err("case count must be a nonzero multiple of 64".into());
    }
    let routing = LeafRouting::SeededShuffle(seed ^ 0x5a43_0f1e);
    let tree =
        FiveCircleHierarchy::from_modes(kinds, routing).map_err(|error| format!("{error:?}"))?;
    let contiguous = FiveCircleHierarchy::from_modes(kinds, LeafRouting::Contiguous)
        .map_err(|error| format!("{error:?}"))?;
    let mut random = Random(seed);
    let mut scalar_ns = 0;
    let mut lut_ns = 0;
    let mut transpose_ns = 0;
    let mut bitplane_ns = 0;
    let mut correct = 0_usize;
    let mut active = 0_usize;
    let mut routing_disagreements = 0_usize;

    for _ in 0..cases / BITSLICE_LANES {
        let states = make_states(&mut random, tree.input_bits());
        let start = Instant::now();
        let scalar: Vec<bool> = states
            .iter()
            .map(|state| tree.infer_scalar(state).expect("valid width"))
            .collect();
        scalar_ns += start.elapsed().as_nanos();

        let start = Instant::now();
        let lut: Vec<bool> = states
            .iter()
            .map(|state| tree.infer_lut_scalar(state).expect("valid width"))
            .collect();
        lut_ns += start.elapsed().as_nanos();

        let start = Instant::now();
        let planes =
            FiveCircleHierarchy::transpose64(&states).map_err(|error| format!("{error:?}"))?;
        transpose_ns += start.elapsed().as_nanos();

        let start = Instant::now();
        let packed = tree
            .infer_bitplanes(&planes)
            .map_err(|error| format!("{error:?}"))?;
        bitplane_ns += start.elapsed().as_nanos();

        for (lane, state) in states.iter().enumerate() {
            let expected = oracle(state, tree.routing(), kinds);
            let bitsliced = packed & (1_u64 << lane) != 0;
            if scalar[lane] != expected || lut[lane] != expected || bitsliced != expected {
                return Err(format!(
                    "incorrect representation at depth={} seed={} lane={lane}",
                    kinds.len(),
                    seed
                ));
            }
            correct += 1;
            active += usize::from(bitsliced);
            routing_disagreements +=
                usize::from(expected != oracle(state, contiguous.routing(), kinds));
        }
    }
    if correct != cases || tree.neuron_count() * 5 != tree.directed_contacts() {
        return Err("incorrect hierarchy size or benchmark accounting".into());
    }
    println!(
        "hierarchy depth={} input_bits={} neurons={} directed_contacts={} cases={} seed={} correct={} active={} permuted_vs_contiguous_disagreements={} scalar_us={} lut_us={} transpose_us={} bitslice_us={} bitslice_including_transpose_us={} logical_model_route_bytes={} compiled_truth_bytes={}",
        tree.depth(),
        tree.input_bits(),
        tree.neuron_count(),
        tree.directed_contacts(),
        cases,
        seed,
        correct,
        active,
        routing_disagreements,
        scalar_ns / 1_000,
        lut_ns / 1_000,
        transpose_ns / 1_000,
        bitplane_ns / 1_000,
        (transpose_ns + bitplane_ns) / 1_000,
        tree.logical_wiring_and_model_bytes(),
        tree.compiled_truth_bytes()
    );
    Ok(())
}

fn main() -> Result<(), String> {
    for seed in [0xa832_0101_u64, 0xa832_0202, 0xa832_0303] {
        // 25-bit full parity: routing permutation must not change its value.
        profile(&[GateKind::Parity, GateKind::Parity], 65_536, seed)?;
        // 125-bit hierarchical composition of disjunction, majority and XOR.
        profile(
            &[GateKind::Parity, GateKind::Majority, GateKind::Parity],
            16_384,
            seed,
        )?;
        // 625-bit heterogeneous hierarchy, 156 five-circle neurons.
        profile(
            &[
                GateKind::Majority,
                GateKind::Parity,
                GateKind::And,
                GateKind::Or,
            ],
            4_096,
            seed,
        )?;
    }
    println!("FOURIER5_HIERARCHICAL_625_BIT_SYNTHETIC_PASS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_oracle_matches_all_64_bit_lanes() {
        profile(
            &[GateKind::Majority, GateKind::Parity, GateKind::And],
            64,
            53,
        )
        .unwrap();
    }

    #[test]
    fn parity_is_invariant_under_exact_leaf_permutations() {
        let kinds = &[GateKind::Parity, GateKind::Parity];
        let contiguous = FiveCircleHierarchy::from_modes(kinds, LeafRouting::Contiguous).unwrap();
        let shuffled =
            FiveCircleHierarchy::from_modes(kinds, LeafRouting::SeededShuffle(19)).unwrap();
        let mut random = Random(173);
        for state in make_states(&mut random, 25) {
            assert_eq!(
                contiguous.infer_scalar(&state),
                shuffled.infer_scalar(&state)
            );
            assert_eq!(
                contiguous.infer_scalar(&state),
                Ok(state.active_count() % 2 != 0)
            );
        }
    }
}

//! Train-only frozen Fourier routing selection on 25, 125 and 625 inputs.
//!
//! Fixed Fourier masks, radii and neuron count. All candidate permutations
//! are proposed *before* any label is generated and are scored on TRAIN only.
//! VALID, TEST and shifted-density TEST never influence routing selection.
//! The planted synthetic teacher is deliberately in the proposal bank:
//! this measures search identifiability, not open-ended topology discovery.
use std::collections::BTreeSet;
use std::time::Instant;

use v888_booleen::fourier5_hierarchy::{FiveCircleHierarchy, GateKind, LeafRouting};
use v888_booleen::fourier5_routing_search::{
    candidate_model, candidate_swaps, select_route, RoutingSample, MAX_SWAP_CANDIDATES,
};
use v888_booleen::BooleanState;

#[derive(Clone, Copy)]
struct Random(u64);

impl Random {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Independent evaluator using only primitive Boolean truth rules,
/// NEVER FiveCircle.predict, compiled LUT or train labels.
fn oracle(state: &BooleanState, routing: &[u16], modes: &[GateKind]) -> bool {
    let mut current: Vec<bool> = routing
        .iter()
        .map(|&source| state.get(usize::from(source)).expect("valid routing"))
        .collect();
    for &mode in modes {
        current = current
            .chunks_exact(5)
            .map(|five| {
                let word = five
                    .iter()
                    .enumerate()
                    .fold(0_u8, |bits, (i, &active)| bits | (u8::from(active) << i));
                mode.oracle(word)
            })
            .collect();
    }
    current[0]
}

/// Populate four globally disjoint input splits. TRAIN / VALID / TEST
/// have Bernoulli(p=1/2) bits; SHIFT has Bernoulli(p=1/4) bits.
/// Any exact duplicate input across splits is resampled (not relabeled).
fn make_samples(
    random: &mut Random,
    seen: &mut BTreeSet<Vec<u64>>,
    count: usize,
    width: usize,
    density_per_256: u64,
    teacher_route: &[u16],
    modes: &[GateKind],
) -> Result<Vec<RoutingSample>, String> {
    let mut rows = Vec::with_capacity(count);
    let mut attempts = 0_usize;
    while rows.len() != count {
        attempts += 1;
        if attempts > count * 8 {
            return Err("could not generate distinct input patterns in budget".into());
        }
        let mut state = BooleanState::new(width);
        for input in 0..width {
            let present = ((random.next() >> 32) & 255) < density_per_256;
            if present {
                state.set(input, true).expect("valid index");
            }
        }
        if !seen.insert(state.words().to_vec()) {
            continue;
        }
        let label = oracle(&state, teacher_route, modes);
        rows.push(RoutingSample { state, label });
    }
    Ok(rows)
}

fn score(model: &FiveCircleHierarchy, rows: &[RoutingSample]) -> Result<usize, String> {
    let mut hits = 0;
    for row in rows {
        hits += usize::from(
            model
                .infer_scalar(&row.state)
                .map_err(|err| format!("{err:?}"))?
                == row.label,
        );
    }
    Ok(hits)
}

fn profile(
    modes: &[GateKind],
    training: usize,
    seed: u64,
    planted_index: usize,
) -> Result<(), String> {
    let base = FiveCircleHierarchy::from_modes(modes, LeafRouting::Contiguous)
        .map_err(|err| format!("{err:?}"))?;
    let width = base.input_bits();
    let proposal_seed = seed ^ 0x9a16_70de;
    // The independent candidate bank uses NO input data and NO labels.
    let pairs = candidate_swaps(width, proposal_seed, MAX_SWAP_CANDIDATES)
        .map_err(|err| format!("{err:?}"))?;
    let teacher =
        candidate_model(&base, &pairs, planted_index).map_err(|err| format!("{err:?}"))?;
    let random_index = 1 + (seed as usize % MAX_SWAP_CANDIDATES);
    let random_control =
        candidate_model(&base, &pairs, random_index).map_err(|err| format!("{err:?}"))?;
    let mut rng = Random::new(seed ^ 0x63f0_11a3);
    let mut seen = BTreeSet::new();
    let train = make_samples(
        &mut rng,
        &mut seen,
        training,
        width,
        128,
        teacher.routing(),
        modes,
    )?;
    let validation = make_samples(
        &mut rng,
        &mut seen,
        512,
        width,
        128,
        teacher.routing(),
        modes,
    )?;
    let test = make_samples(
        &mut rng,
        &mut seen,
        1_024,
        width,
        128,
        teacher.routing(),
        modes,
    )?;
    let shifted = make_samples(
        &mut rng,
        &mut seen,
        512,
        width,
        64,
        teacher.routing(),
        modes,
    )?;
    assert_eq!(
        seen.len(),
        train.len() + validation.len() + test.len() + shifted.len()
    );

    let began = Instant::now();
    // No references to validation, test, shifted or teacher are passed.
    let selected = select_route(&base, &train, proposal_seed, MAX_SWAP_CANDIDATES)
        .map_err(|err| format!("{err:?}"))?;
    let search_ms = began.elapsed().as_millis();

    let begin = Instant::now();
    let train_hits = score(&selected.model, &train)?;
    let baseline_train = score(&base, &train)?;
    let valid_hits = score(&selected.model, &validation)?;
    let test_hits = score(&selected.model, &test)?;
    let shifted_hits = score(&selected.model, &shifted)?;
    let baseline_valid = score(&base, &validation)?;
    let baseline_test = score(&base, &test)?;
    let baseline_shifted = score(&base, &shifted)?;
    let random_valid = score(&random_control, &validation)?;
    let random_test = score(&random_control, &test)?;
    let random_shifted = score(&random_control, &shifted)?;
    let teacher_valid = score(&teacher, &validation)?;
    let teacher_test = score(&teacher, &test)?;
    let teacher_shifted = score(&teacher, &shifted)?;
    let evaluate_ms = begin.elapsed().as_millis();

    println!(
        "routing depth={} input_bits={width} neurons={} contacts={} seed={seed} proposal_seed={proposal_seed} teacher_index={planted_index} random_index={random_index} winner_index={} winner_pair={:?} candidates={} train={train_hits}/{} baseline_train={baseline_train}/{} valid={valid_hits}/{} baseline_valid={baseline_valid}/{} random_valid={random_valid}/{} teacher_valid={teacher_valid}/{} test={test_hits}/{} baseline_test={baseline_test}/{} random_test={random_test}/{} teacher_test={teacher_test}/{} shifted={shifted_hits}/{} baseline_shifted={baseline_shifted}/{} random_shifted={random_shifted}/{} teacher_shifted={teacher_shifted}/{} search_inferences={} search_ms={search_ms} evaluate_ms={evaluate_ms} logical_model_routing_bytes={}",
        modes.len(),
        base.neuron_count(),
        base.directed_contacts(),
        selected.selected_index,
        selected.selected_pair,
        selected.candidates_evaluated,
        train.len(),
        train.len(),
        validation.len(),
        validation.len(),
        validation.len(),
        validation.len(),
        test.len(),
        test.len(),
        test.len(),
        test.len(),
        shifted.len(),
        shifted.len(),
        shifted.len(),
        shifted.len(),
        selected.training_inferences,
        base.logical_wiring_and_model_bytes()
    );
    // Full-budget planted teacher is a candidate, so TRAIN must be exact;
    // test and shifted results must be reported, not forced to be perfect.
    if selected.train_correct != train.len()
        || train_hits != train.len()
        || baseline_train > train_hits
        || teacher_valid != validation.len()
        || teacher_test != test.len()
        || teacher_shifted != shifted.len()
        || selected.candidates_evaluated != MAX_SWAP_CANDIDATES + 1
        || selected.training_inferences != train.len() * (MAX_SWAP_CANDIDATES + 1)
        || selected.model.neuron_count() != base.neuron_count()
        || selected.model.directed_contacts() != base.directed_contacts()
    {
        return Err("inconsistent train-only routing search or oracle contract".into());
    }
    Ok(())
}

fn main() -> Result<(), String> {
    for (seed, planted_index) in [
        (0xb888_0011_u64, 7_usize),
        (0xb888_0022, 31),
        (0xb888_0033, 53),
    ] {
        profile(
            &[GateKind::Majority, GateKind::Parity],
            512,
            seed,
            planted_index,
        )?;
        profile(
            &[GateKind::Majority, GateKind::Parity, GateKind::Parity],
            512,
            seed,
            planted_index,
        )?;
        profile(
            &[
                GateKind::Majority,
                GateKind::Parity,
                GateKind::Majority,
                GateKind::Parity,
            ],
            1_024,
            seed,
            planted_index,
        )?;
    }
    println!("FOURIER5_TRAIN_ONLY_ROUTE_SELECTION_SYNTHETIC_PASS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_validation_test_shifted_are_exactly_disjoint() {
        let modes = [GateKind::Majority, GateKind::Parity];
        let base = FiveCircleHierarchy::from_modes(&modes, LeafRouting::Contiguous).unwrap();
        let mut rng = Random::new(97);
        let mut seen = BTreeSet::new();
        let train = make_samples(&mut rng, &mut seen, 64, 25, 128, base.routing(), &modes).unwrap();
        let validation =
            make_samples(&mut rng, &mut seen, 64, 25, 128, base.routing(), &modes).unwrap();
        let test = make_samples(&mut rng, &mut seen, 64, 25, 128, base.routing(), &modes).unwrap();
        let shifted =
            make_samples(&mut rng, &mut seen, 64, 25, 64, base.routing(), &modes).unwrap();
        assert_eq!(seen.len(), 256);
        assert_eq!(
            train.len() + validation.len() + test.len() + shifted.len(),
            256
        );
        for row in train.iter().chain(&validation).chain(&test).chain(&shifted) {
            assert_eq!(row.label, oracle(&row.state, base.routing(), &modes));
        }
    }

    #[test]
    fn heldout_labels_are_never_available_to_router() {
        let modes = [GateKind::Majority, GateKind::Parity];
        let base = FiveCircleHierarchy::from_modes(&modes, LeafRouting::Contiguous).unwrap();
        let pairs = candidate_swaps(25, 19, 16).unwrap();
        let teacher = candidate_model(&base, &pairs, 7).unwrap();
        let mut rng = Random::new(20);
        let mut seen = BTreeSet::new();
        let train =
            make_samples(&mut rng, &mut seen, 128, 25, 128, teacher.routing(), &modes).unwrap();
        let heldout =
            make_samples(&mut rng, &mut seen, 64, 25, 128, teacher.routing(), &modes).unwrap();
        let first = select_route(&base, &train, 19, 16).unwrap();
        let inverted_heldout: Vec<_> = heldout
            .into_iter()
            .map(|mut row| {
                row.label = !row.label;
                row
            })
            .collect();
        assert_eq!(inverted_heldout.len(), 64);
        let second = select_route(&base, &train, 19, 16).unwrap();
        assert_eq!(first.selected_index, second.selected_index);
        assert_eq!(first.model.routing(), second.model.routing());
        assert_eq!(first.train_correct, second.train_correct);
    }
}

//! Exact paired evaluation of 2-/3-swap Boolean connectivity AND local plasticity.
//!
//! Four arms: original frozen, multi-swap frozen, frozen routing + learned
//! output radii/bias, and jointly selected routing + learned output.
//! The teacher is synthetic and IN the 3-swap bank; the top-gate teacher
//! is majority, while the initial/frozen top gate is AND. Report all
//! errors and positive/negative class counts on genuinely disjoint samples.
use std::collections::BTreeSet;
use std::time::Instant;

use v888_booleen::fourier5_hierarchy::{FiveCircleHierarchy, GateKind, LeafRouting};
use v888_booleen::fourier5_joint_search::{multiswap_model, optimize_joint};
use v888_booleen::fourier5_routing_search::{candidate_swaps, select_route, RoutingSample};
use v888_booleen::BooleanState;

const BANK_SIZE: usize = 8;
const MAX_DEPTH: usize = 3;
const EPOCHS: usize = 12;

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

/// Oracle independent of FiveCircle.predict and all compiled truth tables.
fn oracle(state: &BooleanState, route: &[u16], modes: &[GateKind]) -> bool {
    let mut current: Vec<bool> = route
        .iter()
        .map(|&index| state.get(usize::from(index)).expect("validated source"))
        .collect();
    for &mode in modes {
        current = current
            .chunks_exact(5)
            .map(|group| {
                let mut bits = 0_u8;
                for (bit, &value) in group.iter().enumerate() {
                    bits |= u8::from(value) << bit;
                }
                mode.oracle(bits)
            })
            .collect();
    }
    current[0]
}

fn generate(
    rng: &mut Random,
    seen: &mut BTreeSet<Vec<u64>>,
    count: usize,
    width: usize,
    p_256: u64,
    route: &[u16],
    modes: &[GateKind],
) -> Result<Vec<RoutingSample>, String> {
    let mut rows = Vec::with_capacity(count);
    for _ in 0..count * 16 {
        if rows.len() == count {
            break;
        }
        let mut state = BooleanState::new(width);
        for bit in 0..width {
            if ((rng.next() >> 32) & 255) < p_256 {
                state.set(bit, true).expect("bounded index");
            }
        }
        if !seen.insert(state.words().to_vec()) {
            continue;
        }
        let label = oracle(&state, route, modes);
        rows.push(RoutingSample { state, label });
    }
    if rows.len() != count {
        return Err("unique synthetic sampling budget exceeded".into());
    }
    Ok(rows)
}

#[derive(Clone, Copy, Debug)]
struct Hits {
    correct: usize,
    positives: usize,
    true_positives: usize,
    negatives: usize,
    true_negatives: usize,
}

impl Hits {
    fn total(self) -> usize {
        self.positives + self.negatives
    }
}

fn score(model: &FiveCircleHierarchy, rows: &[RoutingSample]) -> Result<Hits, String> {
    let mut hits = Hits {
        correct: 0,
        positives: 0,
        true_positives: 0,
        negatives: 0,
        true_negatives: 0,
    };
    for row in rows {
        let predicted = model
            .infer_scalar(&row.state)
            .map_err(|e| format!("{e:?}"))?;
        if row.label {
            hits.positives += 1;
            hits.true_positives += usize::from(predicted);
        } else {
            hits.negatives += 1;
            hits.true_negatives += usize::from(!predicted);
        }
        hits.correct += usize::from(row.label == predicted);
    }
    if hits.total() != rows.len() {
        return Err("confusion-matrix accounting broken".into());
    }
    Ok(hits)
}

fn profile(true_modes: &[GateKind], seed: u64, hidden_swaps: &[usize]) -> Result<(), String> {
    let mut initial_modes = true_modes.to_vec();
    *initial_modes.last_mut().ok_or("no output gate")? = GateKind::And;
    let base = FiveCircleHierarchy::from_modes(&initial_modes, LeafRouting::Contiguous)
        .map_err(|e| format!("{e:?}"))?;
    let teacher_base = FiveCircleHierarchy::from_modes(true_modes, LeafRouting::Contiguous)
        .map_err(|e| format!("{e:?}"))?;
    let proposal_seed = seed ^ 0x55aa_1122;
    let bank = candidate_swaps(base.input_bits(), proposal_seed, BANK_SIZE)
        .map_err(|e| format!("{e:?}"))?;
    let teacher =
        multiswap_model(&teacher_base, &bank, hidden_swaps).map_err(|e| format!("{e:?}"))?;
    let teacher_route = teacher.routing();

    // Strict multi-connection challenge: teacher must be different from
    // the identity and every single-swap member of the proposal bank.
    if teacher_route == base.routing()
        || (0..BANK_SIZE).any(|index| {
            multiswap_model(&base, &bank, &[index]).is_ok_and(|m| m.routing() == teacher_route)
        })
    {
        return Err("the target topology is not a genuine multi-swap challenge".into());
    }
    let width = base.input_bits();
    let mut rng = Random::new(seed ^ 0x1a5e_c403);
    let mut seen = BTreeSet::new();
    let train_count = 512;
    let train = generate(
        &mut rng,
        &mut seen,
        train_count,
        width,
        128,
        teacher_route,
        true_modes,
    )?;
    let validation = generate(
        &mut rng,
        &mut seen,
        256,
        width,
        128,
        teacher_route,
        true_modes,
    )?;
    let test = generate(
        &mut rng,
        &mut seen,
        512,
        width,
        128,
        teacher_route,
        true_modes,
    )?;
    let shifted = generate(
        &mut rng,
        &mut seen,
        256,
        width,
        64,
        teacher_route,
        true_modes,
    )?;
    if seen.len() != train.len() + validation.len() + test.len() + shifted.len() {
        return Err("input leak between splits".into());
    }
    let begun = Instant::now();
    let single =
        select_route(&base, &train, proposal_seed, BANK_SIZE).map_err(|e| format!("{e:?}"))?;
    let single_ms = begun.elapsed().as_millis();
    let begun = Instant::now();
    let fit = optimize_joint(&base, &train, proposal_seed, BANK_SIZE, MAX_DEPTH, EPOCHS)
        .map_err(|e| format!("{e:?}"))?;
    let joint_ms = begun.elapsed().as_millis();

    let controls = [
        ("frozen", &base),
        ("single_swap_frozen", &single.model),
        ("multi_swap_frozen", &fit.route_only),
        ("plastic_only", &fit.plastic_only),
        ("joint", &fit.joint),
        ("synthetic_teacher", &teacher),
    ];
    for &(arm, model) in &controls {
        let valid_hits = score(model, &validation)?;
        let test_hits = score(model, &test)?;
        let shifted_hits = score(model, &shifted)?;
        let train_hits = score(model, &train)?;
        println!(
            "joint_arm width={width} depth={} seed={seed} hidden_swaps={:?} arm={arm} train={}/{} valid={}/{} valid_pos={}/{} valid_neg={}/{} test={}/{} test_pos={}/{} test_neg={}/{} shifted={}/{} shifted_pos={}/{} shifted_neg={}/{}",
            true_modes.len(), hidden_swaps,
            train_hits.correct, train_hits.total(),
            valid_hits.correct, valid_hits.total(),
            valid_hits.true_positives, valid_hits.positives,
            valid_hits.true_negatives, valid_hits.negatives,
            test_hits.correct, test_hits.total(),
            test_hits.true_positives, test_hits.positives,
            test_hits.true_negatives, test_hits.negatives,
            shifted_hits.correct, shifted_hits.total(),
            shifted_hits.true_positives, shifted_hits.positives,
            shifted_hits.true_negatives, shifted_hits.negatives
        );
        if arm == "synthetic_teacher"
            && (train_hits.correct != train.len()
                || valid_hits.correct != validation.len()
                || test_hits.correct != test.len()
                || shifted_hits.correct != shifted.len())
        {
            return Err("independent oracle mismatch".into());
        }
    }

    println!(
        "joint_cost width={width} seed={seed} bank_size={BANK_SIZE} max_swaps={MAX_DEPTH} candidates={} single_candidates={} classification_inferences={} fitting_example_visits={} output_fits={} total_updates={} joint_updates={} plastic_only_updates={} route_only_swaps={:?} joint_swaps={:?} route_only_train={}/{} plastic_only_train={}/{} joint_train={}/{} single_ms={single_ms} joint_ms={joint_ms} neurons={} contacts={} model_bytes={}",
        fit.candidates_evaluated, single.candidates_evaluated,
        fit.classification_inferences, fit.fitting_example_visits,
        fit.output_fits, fit.total_updates, fit.joint_updates,
        fit.plastic_only_updates, fit.route_only_swaps, fit.joint_swaps,
        fit.route_only_correct, train.len(), fit.plastic_only_correct, train.len(),
        fit.joint_correct, train.len(), base.neuron_count(), base.directed_contacts(),
        base.logical_wiring_and_model_bytes()
    );
    if fit.candidates_evaluated < 1 + BANK_SIZE
        || fit.joint_correct < fit.route_only_correct
        || fit.joint_correct < fit.plastic_only_correct
        || fit.joint.neuron_count() != base.neuron_count()
        || fit.joint.directed_contacts() != base.directed_contacts()
        || fit.output_fits != fit.candidates_evaluated
        || fit.classification_inferences != 2 * fit.candidates_evaluated * train.len()
    {
        return Err("joint optimization budget/capacity invariant violated".into());
    }
    Ok(())
}

fn main() -> Result<(), String> {
    for (seed, hidden_swaps) in [
        (0x5888_0101_u64, &[0_usize, 2][..]),
        (0x5888_0202_u64, &[0_usize, 2, 5][..]),
        (0x5888_0303_u64, &[1_usize, 4, 7][..]),
    ] {
        profile(
            &[GateKind::Majority, GateKind::Majority],
            seed,
            hidden_swaps,
        )?;
        profile(
            &[GateKind::Parity, GateKind::Majority, GateKind::Majority],
            seed,
            hidden_swaps,
        )?;
        profile(
            &[
                GateKind::Parity,
                GateKind::Majority,
                GateKind::Parity,
                GateKind::Majority,
            ],
            seed,
            hidden_swaps,
        )?;
    }
    println!("FOURIER5_MULTISWAP_AND_READOUT_PLASTICITY_SYNTHETIC_PASS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_evaluation_splits_are_disjoint_and_oracle_labeled() {
        let modes = [GateKind::Majority, GateKind::Majority];
        let base = FiveCircleHierarchy::from_modes(&modes, LeafRouting::Contiguous).unwrap();
        let mut seen = BTreeSet::new();
        let mut rng = Random::new(18);
        let train = generate(&mut rng, &mut seen, 64, 25, 128, base.routing(), &modes).unwrap();
        let valid = generate(&mut rng, &mut seen, 64, 25, 128, base.routing(), &modes).unwrap();
        let test = generate(&mut rng, &mut seen, 64, 25, 128, base.routing(), &modes).unwrap();
        let shift = generate(&mut rng, &mut seen, 64, 25, 64, base.routing(), &modes).unwrap();
        assert_eq!(seen.len(), 256);
        for row in train.iter().chain(&valid).chain(&test).chain(&shift) {
            assert_eq!(row.label, oracle(&row.state, base.routing(), &modes));
        }
    }

    #[test]
    fn two_swaps_and_readout_train_with_no_test_leak() {
        profile(
            &[GateKind::Majority, GateKind::Majority],
            0x5888_0101,
            &[0, 2],
        )
        .unwrap();
    }
}

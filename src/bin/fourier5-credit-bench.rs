//! No-hidden-teacher-tap credit assignment across 25/125/625 Boolean inputs.
//!
//! Labels are constructed by an independent Boolean oracle; the learner
//! receives ONLY TRAIN inputs and final output labels. Teacher intermediate
//! activations are never supplied. A teacher route uses unrelated seeded
//! swaps; it is not planted in the learner's random intervention stream.
//! All methods evaluate an identical bounded number of full-model proposals.
use std::collections::BTreeSet;
use std::time::Instant;

use v888_booleen::fourier5_counterfactual_credit::{
    fit_final_loss, CreditConfig, CreditPolicy, CreditScope,
};
use v888_booleen::fourier5_hierarchy::{FiveCircleHierarchy, GateKind, LeafRouting};
use v888_booleen::fourier5_joint_search::multiswap_model;
use v888_booleen::fourier5_routing_search::{candidate_swaps, RoutingSample};
use v888_booleen::BooleanState;

const TRAIN: usize = 512;
const VALID: usize = 256;
const TEST: usize = 512;
const SHIFT: usize = 256;
const ROUNDS: usize = 8;
const PROPOSALS: usize = 64;

struct Random(u64);
impl Random {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed })
    }
    fn next(&mut self) -> u64 {
        let mut v = self.0;
        v ^= v << 13;
        v ^= v >> 7;
        v ^= v << 17;
        self.0 = v;
        v
    }
}

fn independent_oracle(
    input: &BooleanState,
    routing: &[u16],
    modes: &[GateKind],
) -> bool {
    let mut bits: Vec<bool> = routing
        .iter()
        .map(|&source| input.get(usize::from(source)).expect("validated source"))
        .collect();
    for &mode in modes {
        bits = bits
            .chunks_exact(5)
            .map(|five| {
                let word = five.iter().enumerate().fold(0_u8, |acc, (bit, value)| {
                    acc | (u8::from(*value) << bit)
                });
                mode.oracle(word)
            })
            .collect();
    }
    bits[0]
}

fn generate(
    random: &mut Random,
    unique: &mut BTreeSet<Vec<u64>>,
    count: usize,
    width: usize,
    density: u64,
    routing: &[u16],
    modes: &[GateKind],
) -> Result<Vec<RoutingSample>, String> {
    let mut rows = Vec::with_capacity(count);
    for _ in 0..count * 32 {
        if rows.len() == count { break; }
        let mut state = BooleanState::new(width);
        for bit in 0..width {
            if ((random.next() >> 32) & 255) < density {
                state.set(bit, true).expect("bounded bit");
            }
        }
        if !unique.insert(state.words().to_vec()) { continue; }
        let label = independent_oracle(&state, routing, modes);
        rows.push(RoutingSample { state, label });
    }
    if rows.len() != count { return Err("cannot generate disjoint inputs".into()); }
    Ok(rows)
}

#[derive(Clone, Copy)]
struct Hits {
    correct: usize,
    positives: usize,
    true_positive: usize,
    negatives: usize,
    true_negative: usize,
}

fn score(
    model: &FiveCircleHierarchy,
    rows: &[RoutingSample],
) -> Result<Hits, String> {
    let mut hits = Hits {
        correct: 0, positives: 0, true_positive: 0, negatives: 0, true_negative: 0,
    };
    for row in rows {
        let predicted = model.infer_scalar(&row.state).map_err(|e| format!("{e:?}"))?;
        hits.correct += usize::from(predicted == row.label);
        if row.label {
            hits.positives += 1;
            hits.true_positive += usize::from(predicted);
        } else {
            hits.negatives += 1;
            hits.true_negative += usize::from(!predicted);
        }
    }
    if hits.positives + hits.negatives != rows.len() {
        return Err("broken score accounting".into());
    }
    Ok(hits)
}

fn profile(
    seed: u64,
    modes: &[GateKind],
) -> Result<(), String> {
    let teacher_base = FiveCircleHierarchy::from_modes(
        modes, LeafRouting::Contiguous
    ).map_err(|e| format!("{e:?}"))?;
    let width = teacher_base.input_bits();
    // Independent teacher source changes: NO coordinates or random proposal
    // streams from the learning algorithm are used here.
    let teacher_bank = candidate_swaps(width, seed ^ 0x341a_8b2f, 8)
        .map_err(|e| format!("{e:?}"))?;
    let teacher = multiswap_model(&teacher_base, &teacher_bank, &[0, 3, 6])
        .map_err(|e| format!("{e:?}"))?;
    if teacher.routing() == teacher_base.routing() {
        return Err("unexpected unmodified teacher routing".into());
    }

    // The student starts with the exact same five masks on EVERY neuron,
    // but every fifth leaf-facing neuron starts as AND rather than Majority.
    // Higher layers are parity and are frozen unless a proposal explicitly
    // targets a hidden neuron. Their output labels are NOT given to the fit.
    let mut base = teacher_base.clone();
    let bottom = width / 5;
    let mut perturbed = 0;
    for index in (0..bottom).step_by(5) {
        base = base.with_neuron_parameters(0, index, GateKind::And.five_circle())
            .map_err(|e| format!("{e:?}"))?;
        perturbed += 1;
    }
    if base.routing() == teacher.routing() {
        return Err("teacher routing unexpectedly equals student".into());
    }

    let mut rng = Random::new(seed ^ 0x45ce_991a);
    let mut unique = BTreeSet::new();
    let train = generate(&mut rng, &mut unique, TRAIN, width, 128, teacher.routing(), modes)?;
    let valid = generate(&mut rng, &mut unique, VALID, width, 128, teacher.routing(), modes)?;
    let test = generate(&mut rng, &mut unique, TEST, width, 128, teacher.routing(), modes)?;
    let shifted = generate(&mut rng, &mut unique, SHIFT, width, 64, teacher.routing(), modes)?;
    if unique.len() != TRAIN + VALID + TEST + SHIFT {
        return Err("cross-split duplicate Boolean input".into());
    }
    for row in train.iter().chain(&valid).chain(&test).chain(&shifted) {
        if teacher.infer_scalar(&row.state) != Ok(row.label) {
            return Err("teacher inference and independent oracle mismatch".into());
        }
    }

    let training_seed = seed ^ 0x71d0_cafe;
    let setups = [
        ("readout_only", CreditScope::OutputParameters, CreditPolicy::Greedy),
        ("hidden_only", CreditScope::HiddenParameters, CreditPolicy::Greedy),
        ("connections_only", CreditScope::LeafRouting, CreditPolicy::Greedy),
        ("joint", CreditScope::JointHiddenAndRouting, CreditPolicy::Greedy),
        ("blind_joint", CreditScope::JointHiddenAndRouting, CreditPolicy::Blind),
    ];
    let reference = score(&teacher, &test)?;
    if reference.correct != TEST {
        return Err("oracle not exact on test".into());
    }
    let initial_test = score(&base, &test)?;
    let initial_shifted = score(&base, &shifted)?;
    println!(
        "credit_arm width={width} seed={seed} arm=initial test={}/{} test_tp={}/{} test_tn={}/{} shifted={}/{} shifted_tp={}/{} shifted_tn={}/{}",
        initial_test.correct, TEST, initial_test.true_positive, initial_test.positives,
        initial_test.true_negative, initial_test.negatives,
        initial_shifted.correct, SHIFT, initial_shifted.true_positive,
        initial_shifted.positives, initial_shifted.true_negative,
        initial_shifted.negatives,
    );
    for (name, scope, policy) in setups {
        let started = Instant::now();
        let result = fit_final_loss(
            &base, &train,
            CreditConfig {
                scope, policy, rounds: ROUNDS, proposals_per_round: PROPOSALS,
                seed: training_seed,
            },
        ).map_err(|e| format!("{e:?}"))?;
        let train_ms = started.elapsed().as_millis();
        if result.candidate_evaluations != ROUNDS * PROPOSALS
            || result.full_model_train_inferences != (ROUNDS * PROPOSALS + 1) * TRAIN
            || result.model.neuron_count() != base.neuron_count()
            || result.model.directed_contacts() != base.directed_contacts()
            || (policy == CreditPolicy::Greedy
                && result.final_train_correct < result.initial_train_correct)
        {
            return Err(format!("credit assignment contract violated in arm {name}"));
        }
        if scope == CreditScope::HiddenParameters && result.model.routing() != base.routing() {
            return Err("hidden-only learner changed wiring".into());
        }
        let v = score(&result.model, &valid)?;
        let t = score(&result.model, &test)?;
        let shift = score(&result.model, &shifted)?;
        println!(
            "credit_arm width={width} seed={seed} arm={name} train={}/{} valid={}/{} valid_tp={}/{} valid_tn={}/{} test={}/{} test_tp={}/{} test_tn={}/{} shifted={}/{} shifted_tp={}/{} shifted_tn={}/{} param_moves={} route_moves={} no_gain_rounds={} full_model_inferences={} train_ms={train_ms}",
            result.final_train_correct, TRAIN,
            v.correct, VALID, v.true_positive, v.positives, v.true_negative, v.negatives,
            t.correct, TEST, t.true_positive, t.positives, t.true_negative, t.negatives,
            shift.correct, SHIFT, shift.true_positive, shift.positives,
            shift.true_negative, shift.negatives,
            result.accepted_parameter_moves, result.accepted_routing_moves,
            result.rounds_without_improvement, result.full_model_train_inferences,
        );
    }
    println!(
        "credit_profile width={width} seed={seed} nodes={} contacts={} hidden_parameter_faults={} teacher_route_unplanted=true rounds={ROUNDS} proposals_per_round={PROPOSALS} train={TRAIN} valid={VALID} test={TEST} shifted={SHIFT}",
        base.neuron_count(), base.directed_contacts(), perturbed
    );
    Ok(())
}

fn main() -> Result<(), String> {
    for seed in [0xc088_0101_u64, 0xc088_0202, 0xc088_0303] {
        profile(seed, &[GateKind::Majority, GateKind::Parity])?;
        profile(seed, &[GateKind::Majority, GateKind::Parity, GateKind::Parity])?;
        profile(seed, &[GateKind::Majority, GateKind::Parity, GateKind::Parity, GateKind::Parity])?;
    }
    println!("FOURIER5_FINAL_LABEL_COUNTERFACTUAL_CREDIT_SYNTHETIC_PASS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_oracle_agrees_with_existing_fourier_neurons() {
        let modes = [GateKind::Majority, GateKind::Parity];
        let tree = FiveCircleHierarchy::from_modes(
            &modes, LeafRouting::SeededShuffle(71),
        ).unwrap();
        let mut rng = Random::new(17);
        let mut seen = BTreeSet::new();
        let rows = generate(&mut rng, &mut seen, 64, 25, 128, tree.routing(), &modes).unwrap();
        for row in rows {
            assert_eq!(tree.infer_scalar(&row.state), Ok(row.label));
        }
    }

    #[test]
    fn test_sets_are_input_disjoint_and_shift_has_independent_density() {
        let modes = [GateKind::Majority, GateKind::Parity];
        let tree = FiveCircleHierarchy::from_modes(
            &modes, LeafRouting::Contiguous
        ).unwrap();
        let mut rng = Random::new(23);
        let mut seen = BTreeSet::new();
        let train = generate(&mut rng, &mut seen, 64, 25, 128, tree.routing(), &modes).unwrap();
        let valid = generate(&mut rng, &mut seen, 64, 25, 128, tree.routing(), &modes).unwrap();
        let test = generate(&mut rng, &mut seen, 64, 25, 128, tree.routing(), &modes).unwrap();
        let shifted = generate(&mut rng, &mut seen, 64, 25, 64, tree.routing(), &modes).unwrap();
        assert_eq!(seen.len(), 256);
        assert_eq!(train.len() + valid.len() + test.len() + shifted.len(), 256);
    }
}

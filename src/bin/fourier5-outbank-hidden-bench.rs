//! Out-of-bank connectivity and hidden-layer Fourier plasticity experiment.
//!
//! The teacher is synthesized from a DIFFERENT proposal bank. Its routing
//! is explicitly checked absent from every 0–3-swap combination in the
//! learner's candidate bank. There is no guaranteed perfect solution.
//!
//! TRAIN provides synthetic hidden teacher activation labels exclusively
//! for the two hidden-tap plasticity arms. No hidden signals, model
//! selection or sample inspection from VALID/TEST/SHIFT is permitted.
use std::collections::BTreeSet;
use std::time::Instant;

use v888_booleen::fourier5_hidden_teacher::fit_hidden_with_teacher_taps;
use v888_booleen::fourier5_hierarchy::{FiveCircleHierarchy, GateKind, LeafRouting};
use v888_booleen::fourier5_joint_search::{multiswap_model, optimize_joint};
use v888_booleen::fourier5_routing_search::{candidate_swaps, RoutingSample};
use v888_booleen::BooleanState;

const BANK: usize = 8;
const EPOCHS: usize = 8;
const TRAIN: usize = 512;
const VALID: usize = 256;
const TEST: usize = 512;
const SHIFT: usize = 256;

#[derive(Clone, Copy)]
struct Xorshift(u64);
impl Xorshift {
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

fn oracle(state: &BooleanState, route: &[u16], modes: &[GateKind]) -> bool {
    let mut layer: Vec<bool> = route
        .iter()
        .map(|&idx| state.get(usize::from(idx)).expect("valid route"))
        .collect();
    for &mode in modes {
        layer = layer
            .chunks_exact(5)
            .map(|five| {
                let mut input = 0_u8;
                for (bit, &active) in five.iter().enumerate() {
                    input |= u8::from(active) << bit;
                }
                mode.oracle(input)
            })
            .collect();
    }
    layer[0]
}

fn generate(
    rng: &mut Xorshift,
    unique: &mut BTreeSet<Vec<u64>>,
    count: usize,
    width: usize,
    density: u64,
    truth_route: &[u16],
    truth_modes: &[GateKind],
) -> Result<Vec<RoutingSample>, String> {
    let mut output = Vec::with_capacity(count);
    for _ in 0..count * 32 {
        if output.len() == count {
            break;
        }
        let mut state = BooleanState::new(width);
        for bit in 0..width {
            if ((rng.next() >> 32) & 255) < density {
                state.set(bit, true).expect("valid bit");
            }
        }
        if !unique.insert(state.words().to_vec()) {
            continue;
        }
        let label = oracle(&state, truth_route, truth_modes);
        output.push(RoutingSample { state, label });
    }
    if output.len() != count {
        return Err("unique input generation budget exhausted".into());
    }
    Ok(output)
}

fn in_candidate_bank(
    base: &FiveCircleHierarchy,
    swaps: &[[usize; 2]],
    teacher: &FiveCircleHierarchy,
) -> Result<bool, String> {
    if base.routing() == teacher.routing() {
        return Ok(true);
    }
    for i in 0..swaps.len() {
        for j in i..swaps.len() {
            for k in j..swaps.len() {
                let positions: Vec<usize> = [i, j, k]
                    .into_iter()
                    .filter(|n| *n != swaps.len())
                    .collect();
                if i == j && j == k {
                    if multiswap_model(base, swaps, &[i])
                        .map_err(|e| format!("{e:?}"))?
                        .routing()
                        == teacher.routing()
                    {
                        return Ok(true);
                    }
                } else if i == j && j != k {
                    if multiswap_model(base, swaps, &[i, k])
                        .map_err(|e| format!("{e:?}"))?
                        .routing()
                        == teacher.routing()
                    {
                        return Ok(true);
                    }
                } else if i < j && j == k {
                    if multiswap_model(base, swaps, &[i, j])
                        .map_err(|e| format!("{e:?}"))?
                        .routing()
                        == teacher.routing()
                    {
                        return Ok(true);
                    }
                } else if i < j && j < k {
                    if multiswap_model(base, swaps, &positions)
                        .map_err(|e| format!("{e:?}"))?
                        .routing()
                        == teacher.routing()
                    {
                        return Ok(true);
                    }
                }
            }
        }
    }
    Ok(false)
}

#[derive(Clone, Copy)]
struct Score {
    correct: usize,
    positives: usize,
    true_positives: usize,
    negatives: usize,
    true_negatives: usize,
}
fn score(model: &FiveCircleHierarchy, rows: &[RoutingSample]) -> Result<Score, String> {
    let mut hits = Score {
        correct: 0,
        positives: 0,
        true_positives: 0,
        negatives: 0,
        true_negatives: 0,
    };
    for row in rows {
        let response = model
            .infer_scalar(&row.state)
            .map_err(|e| format!("{e:?}"))?;
        hits.correct += usize::from(response == row.label);
        if row.label {
            hits.positives += 1;
            hits.true_positives += usize::from(response);
        } else {
            hits.negatives += 1;
            hits.true_negatives += usize::from(!response);
        }
    }
    Ok(hits)
}

fn profile(seed: u64, teacher_modes: &[GateKind]) -> Result<(), String> {
    let width = 5_usize.pow(teacher_modes.len() as u32);
    let mut student_modes = teacher_modes.to_vec();
    // All hidden layer Walsh masks remain the same. We start the
    // Majority layers as AND, which is learnable with the same 5 masks.
    for kind in &mut student_modes {
        if *kind == GateKind::Majority {
            *kind = GateKind::And;
        }
    }
    let base = FiveCircleHierarchy::from_modes(&student_modes, LeafRouting::Contiguous)
        .map_err(|e| format!("{e:?}"))?;
    let teacher_base = FiveCircleHierarchy::from_modes(teacher_modes, LeafRouting::Contiguous)
        .map_err(|e| format!("{e:?}"))?;

    let proposal_seed = seed ^ 0x5229_1133;
    let proposal = candidate_swaps(width, proposal_seed, BANK).map_err(|e| format!("{e:?}"))?;
    let teacher_swaps =
        candidate_swaps(width, seed ^ 0x0dd0_7788, BANK).map_err(|e| format!("{e:?}"))?;
    let teacher =
        multiswap_model(&teacher_base, &teacher_swaps, &[0, 3, 6]).map_err(|e| format!("{e:?}"))?;
    if in_candidate_bank(&base, &proposal, &teacher)? {
        return Err("teacher routing unexpectedly belongs to learner bank".into());
    }

    let mut random = Xorshift::new(seed ^ 0x39e1_861d);
    let mut unique = BTreeSet::new();
    let train = generate(
        &mut random,
        &mut unique,
        TRAIN,
        width,
        128,
        teacher.routing(),
        teacher_modes,
    )?;
    let valid = generate(
        &mut random,
        &mut unique,
        VALID,
        width,
        128,
        teacher.routing(),
        teacher_modes,
    )?;
    let test = generate(
        &mut random,
        &mut unique,
        TEST,
        width,
        128,
        teacher.routing(),
        teacher_modes,
    )?;
    let shift = generate(
        &mut random,
        &mut unique,
        SHIFT,
        width,
        64,
        teacher.routing(),
        teacher_modes,
    )?;
    if unique.len() != TRAIN + VALID + TEST + SHIFT {
        return Err("cross-split duplicate input".into());
    }
    // The teacher's labels were produced from a separate Boolean oracle.
    for row in train.iter().chain(&valid).chain(&test).chain(&shift) {
        if teacher.infer_scalar(&row.state) != Ok(row.label) {
            return Err("teacher and Boolean oracle disagree".into());
        }
    }

    let start = Instant::now();
    let joint = optimize_joint(&base, &train, proposal_seed, BANK, 3, EPOCHS)
        .map_err(|e| format!("{e:?}"))?;
    let search_ms = start.elapsed().as_millis();
    let start = Instant::now();
    let hidden_base = fit_hidden_with_teacher_taps(&base, &teacher, &train, EPOCHS)
        .map_err(|e| format!("{e:?}"))?;
    let base_hidden_ms = start.elapsed().as_millis();
    let start = Instant::now();
    let hidden_joint = fit_hidden_with_teacher_taps(&joint.joint, &teacher, &train, EPOCHS)
        .map_err(|e| format!("{e:?}"))?;
    let joint_hidden_ms = start.elapsed().as_millis();

    let arms = [
        ("base_frozen", &base),
        ("route_only", &joint.route_only),
        ("plastic_readout_only", &joint.plastic_only),
        ("outbank_joint_readout", &joint.joint),
        ("hidden_teacher_base", &hidden_base.model),
        ("hidden_teacher_joint", &hidden_joint.model),
        ("oracle", &teacher),
    ];
    for (name, model) in arms {
        let train_hits = score(model, &train)?;
        let valid_hits = score(model, &valid)?;
        let test_hits = score(model, &test)?;
        let shift_hits = score(model, &shift)?;
        println!(
            "outbank width={width} seed={seed} arm={name} train={}/{} valid={}/{} valid_pos={}/{} valid_neg={}/{} test={}/{} test_pos={}/{} test_neg={}/{} shift={}/{} shift_pos={}/{} shift_neg={}/{}",
            train_hits.correct, TRAIN, valid_hits.correct, VALID,
            valid_hits.true_positives, valid_hits.positives,
            valid_hits.true_negatives, valid_hits.negatives,
            test_hits.correct, TEST, test_hits.true_positives, test_hits.positives,
            test_hits.true_negatives, test_hits.negatives,
            shift_hits.correct, SHIFT, shift_hits.true_positives, shift_hits.positives,
            shift_hits.true_negatives, shift_hits.negatives
        );
        if name == "oracle"
            && (train_hits.correct != TRAIN
                || valid_hits.correct != VALID
                || test_hits.correct != TEST
                || shift_hits.correct != SHIFT)
        {
            return Err("reference accuracy mismatch".into());
        }
        if model.neuron_count() != base.neuron_count()
            || model.directed_contacts() != base.directed_contacts()
        {
            return Err("capacity/contact mismatch".into());
        }
    }
    println!(
        "outbank_budget width={width} seed={seed} teacher_in_bank=false neurons={} contacts={} candidates={} classification_inferences={} search_updates={} search_fit_visits={} search_ms={search_ms} base_hidden_ms={base_hidden_ms} joint_hidden_ms={joint_hidden_ms} base_hidden_neurons={} base_hidden_changed={} base_hidden_updates={} base_hidden_visits={} base_hidden_local={}/{} base_hidden_global_train={}/{} joint_hidden_neurons={} joint_hidden_changed={} joint_hidden_updates={} joint_hidden_visits={} joint_hidden_local={}/{} joint_hidden_global_train={}/{}",
        base.neuron_count(),base.directed_contacts(),
        joint.candidates_evaluated,joint.classification_inferences,joint.total_updates,
        joint.fitting_example_visits,
        hidden_base.hidden_neurons,hidden_base.changed_neurons,
        hidden_base.hidden_updates,hidden_base.hidden_example_visits,
        hidden_base.local_correct,hidden_base.local_examples,
        hidden_base.global_train_correct,TRAIN,
        hidden_joint.hidden_neurons,hidden_joint.changed_neurons,
        hidden_joint.hidden_updates,hidden_joint.hidden_example_visits,
        hidden_joint.local_correct,hidden_joint.local_examples,
        hidden_joint.global_train_correct,TRAIN
    );
    Ok(())
}

fn main() -> Result<(), String> {
    for seed in [0x5890_0101_u64, 0x5890_0202, 0x5890_0303] {
        profile(seed, &[GateKind::Majority, GateKind::Majority])?;
        profile(
            seed,
            &[GateKind::Majority, GateKind::Parity, GateKind::Majority],
        )?;
        profile(
            seed,
            &[
                GateKind::Majority,
                GateKind::Parity,
                GateKind::Majority,
                GateKind::Majority,
            ],
        )?;
    }
    println!("FOURIER5_OUT_OF_BANK_HIDDEN_PLASTICITY_SYNTHETIC_PASS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_bank_teacher_is_not_candidate_at_25_bits() {
        let base = FiveCircleHierarchy::from_modes(
            &[GateKind::And, GateKind::And],
            LeafRouting::Contiguous,
        )
        .unwrap();
        let pool = candidate_swaps(25, 11, BANK).unwrap();
        let alt = candidate_swaps(25, 99, BANK).unwrap();
        let teacher = multiswap_model(&base, &alt, &[0, 3, 6]).unwrap();
        assert!(!in_candidate_bank(&base, &pool, &teacher).unwrap());
    }

    #[test]
    fn oracle_agrees_on_25_bit_models() {
        let modes = [GateKind::Majority, GateKind::Majority];
        let tree = FiveCircleHierarchy::from_modes(&modes, LeafRouting::SeededShuffle(17)).unwrap();
        let mut rng = Xorshift::new(29);
        let mut seen = BTreeSet::new();
        let samples = generate(&mut rng, &mut seen, 64, 25, 128, tree.routing(), &modes).unwrap();
        for row in samples {
            assert_eq!(tree.infer_scalar(&row.state), Ok(row.label));
        }
    }
}

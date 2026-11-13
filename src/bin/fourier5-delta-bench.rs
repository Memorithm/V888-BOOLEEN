//! Physical parity-and-cost benchmark: exact cached Boolean dependency
//! cones versus FULL bit-sliced recomputation for identical intervention
//! proposals, policies, train labels and final learned model.
//!
//! This is an inference optimization; it does NOT change candidate choice,
//! learning objective, hidden teacher supervision or model capacity.
use std::collections::BTreeSet;
use std::time::Instant;

use v888_booleen::fourier5_counterfactual_credit::{
    fit_final_loss_with_evaluator, CreditConfig, CreditEvaluationMode, CreditPolicy, CreditScope,
};
use v888_booleen::fourier5_hierarchy::{FiveCircleHierarchy, GateKind, LeafRouting};
use v888_booleen::fourier5_routing_search::RoutingSample;
use v888_booleen::BooleanState;

const TRAIN: usize = 256;
const TEST: usize = 512;
const SHIFT: usize = 256;

struct Random(u64);
impl Random {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed })
    }

    fn next(&mut self) -> u64 {
        let mut n = self.0;
        n ^= n << 13;
        n ^= n >> 7;
        n ^= n << 17;
        self.0 = n;
        n
    }
}

/// Independent Boolean oracle without Fourier or compiled LUT models.
fn oracle(state: &BooleanState, routing: &[u16], modes: &[GateKind]) -> bool {
    let mut current: Vec<bool> = routing
        .iter()
        .map(|&source| state.get(usize::from(source)).expect("valid source"))
        .collect();
    for &mode in modes {
        current = current
            .chunks_exact(5)
            .map(|inputs| {
                let value = inputs.iter().enumerate().fold(0_u8, |acc, (bit, &is_on)| {
                    acc | (u8::from(is_on) << bit)
                });
                mode.oracle(value)
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
    routing: &[u16],
    modes: &[GateKind],
) -> Result<Vec<RoutingSample>, String> {
    let mut samples = Vec::with_capacity(count);
    for _ in 0..count * 32 {
        if samples.len() == count {
            break;
        }
        let mut state = BooleanState::new(width);
        for bit in 0..width {
            if ((rng.next() >> 32) & 255) < p_256 {
                state.set(bit, true).expect("bounded source");
            }
        }
        if !seen.insert(state.words().to_vec()) {
            continue;
        }
        samples.push(RoutingSample {
            label: oracle(&state, routing, modes),
            state,
        });
    }
    if samples.len() != count {
        return Err("distinct sample generation budget exceeded".into());
    }
    Ok(samples)
}

#[derive(Clone, Copy)]
struct Confusion {
    correct: usize,
    tp: usize,
    positives: usize,
    tn: usize,
    negatives: usize,
}

fn confusion(
    model: &FiveCircleHierarchy,
    rows: &[RoutingSample],
) -> Result<Confusion, String> {
    let mut answer = Confusion {
        correct: 0, tp: 0, positives: 0, tn: 0, negatives: 0,
    };
    for row in rows {
        let actual = model.infer_scalar(&row.state).map_err(|err| format!("{err:?}"))?;
        answer.correct += usize::from(actual == row.label);
        if row.label {
            answer.positives += 1;
            answer.tp += usize::from(actual);
        } else {
            answer.negatives += 1;
            answer.tn += usize::from(!actual);
        }
    }
    if answer.positives + answer.negatives != rows.len() {
        return Err("confusion matrix count mismatch".into());
    }
    Ok(answer)
}

fn profile(seed: u64, modes: &[GateKind], scope: CreditScope) -> Result<(), String> {
    let teacher = FiveCircleHierarchy::from_modes(
        modes, LeafRouting::SeededShuffle(seed ^ 0x5eed_08ee)
    ).map_err(|err| format!("{err:?}"))?;
    let width = teacher.input_bits();
    let mut student = FiveCircleHierarchy::from_modes(
        modes, LeafRouting::Contiguous
    ).map_err(|err| format!("{err:?}"))?;
    // Model mismatch is transparent: some frozen first-layer Majorities
    // are replaced by AND gates, which retain the same Walsh singleton masks.
    for neuron in (0..width / 5).step_by(5) {
        student = student
            .with_neuron_parameters(0, neuron, GateKind::And.five_circle())
            .map_err(|err| format!("{err:?}"))?;
    }

    let mut rng = Random::new(seed ^ 0x614b_297a);
    let mut seen = BTreeSet::new();
    let train = generate(&mut rng, &mut seen, TRAIN, width, 128, teacher.routing(), modes)?;
    let test = generate(&mut rng, &mut seen, TEST, width, 128, teacher.routing(), modes)?;
    let shift = generate(&mut rng, &mut seen, SHIFT, width, 64, teacher.routing(), modes)?;
    if seen.len() != TRAIN + TEST + SHIFT {
        return Err("disjoint split guarantee failed".into());
    }
    for sample in train.iter().chain(&test).chain(&shift) {
        if teacher.infer_scalar(&sample.state) != Ok(sample.label) {
            return Err("independent Boolean oracle mismatch".into());
        }
    }

    let settings = CreditConfig {
        scope,
        policy: CreditPolicy::Greedy,
        rounds: 4,
        proposals_per_round: 64,
        seed: seed ^ 0x308b_1007,
    };
    // Explicit paired order: full reference FIRST, incremental SECOND.
    let now = Instant::now();
    let full = fit_final_loss_with_evaluator(
        &student, &train, settings, CreditEvaluationMode::Full
    ).map_err(|err| format!("{err:?}"))?;
    let full_ns = now.elapsed().as_nanos();
    let now = Instant::now();
    let cached = fit_final_loss_with_evaluator(
        &student, &train, settings, CreditEvaluationMode::Incremental
    ).map_err(|err| format!("{err:?}"))?;
    let cached_ns = now.elapsed().as_nanos();

    if full.final_train_correct != cached.final_train_correct
        || full.initial_train_correct != cached.initial_train_correct
        || full.accepted_parameter_moves != cached.accepted_parameter_moves
        || full.accepted_routing_moves != cached.accepted_routing_moves
        || full.model.routing() != cached.model.routing()
        || full.full_model_train_inferences != cached.full_model_train_inferences
        || cached.full_recompute_gate_evaluations != full.actual_gate_evaluations
        || cached.actual_gate_evaluations >= full.actual_gate_evaluations
        || cached.candidate_evaluations != 256
    {
        return Err("full and cached intervention optimization diverged".into());
    }
    for layer in 0..student.depth() {
        let nodes = width / 5_usize.pow((layer + 1) as u32);
        for neuron in 0..nodes {
            if cached.model.layer_model(layer, neuron)
                != full.model.layer_model(layer, neuron)
            {
                return Err("changed Fourier parameters differ after cache optimization".into());
            }
        }
    }
    let test_cached = confusion(&cached.model, &test)?;
    let test_full = confusion(&full.model, &test)?;
    let shifted = confusion(&cached.model, &shift)?;
    if test_cached.correct != test_full.correct || test_cached.tp != test_full.tp {
        return Err("same learnt models disagree on disjoint TEST".into());
    }
    let gate_saving_percent = 100.0
        * (1.0 - cached.actual_gate_evaluations as f64 / full.actual_gate_evaluations as f64);
    println!(
        "delta_cache width={width} depth={} scope={scope:?} seed={seed} train={}/{} test={}/{} test_tp={}/{} test_tn={}/{} shifted={}/{} shift_tp={}/{} shift_tn={}/{} models={} contacts={} proposals={} output_decisions={} baseline_full_gates={} cache_build_gates={} cached_cone_gates={} actual_cached_gates={} exact_gate_saving_percent={gate_saving_percent:.3} full_us={} cache_us={} train_param_moves={} train_route_moves={}",
        modes.len(),
        cached.final_train_correct, TRAIN,
        test_cached.correct, TEST, test_cached.tp, test_cached.positives,
        test_cached.tn, test_cached.negatives,
        shifted.correct, SHIFT, shifted.tp, shifted.positives,
        shifted.tn, shifted.negatives,
        student.neuron_count(), student.directed_contacts(),
        cached.candidate_evaluations, cached.full_model_train_inferences,
        full.actual_gate_evaluations, cached.cache_build_gate_evaluations,
        cached.delta_gate_evaluations, cached.actual_gate_evaluations,
        full_ns / 1_000, cached_ns / 1_000,
        cached.accepted_parameter_moves, cached.accepted_routing_moves,
    );
    Ok(())
}

fn main() -> Result<(), String> {
    for seed in [0xcac8_0101_u64, 0xcac8_0202, 0xcac8_0303] {
        profile(seed, &[GateKind::Majority, GateKind::Parity],
            CreditScope::HiddenParameters)?;
        profile(seed, &[GateKind::Majority, GateKind::Parity],
            CreditScope::JointHiddenAndRouting)?;
        profile(seed, &[GateKind::Majority, GateKind::Parity, GateKind::Parity],
            CreditScope::HiddenParameters)?;
        profile(seed, &[GateKind::Majority, GateKind::Parity, GateKind::Parity],
            CreditScope::JointHiddenAndRouting)?;
        profile(seed, &[GateKind::Majority, GateKind::Parity, GateKind::Parity, GateKind::Parity],
            CreditScope::HiddenParameters)?;
        profile(seed, &[GateKind::Majority, GateKind::Parity, GateKind::Parity, GateKind::Parity],
            CreditScope::JointHiddenAndRouting)?;
    }
    println!("FOURIER5_EXACT_INCREMENTAL_CREDIT_CACHE_PASS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_same_model_and_heldout_scores_from_full_and_cached_training() {
        profile(0xcac8_1212, &[GateKind::Majority, GateKind::Parity],
            CreditScope::JointHiddenAndRouting).unwrap();
    }
}

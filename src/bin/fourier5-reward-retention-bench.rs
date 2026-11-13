//! Output-reward-only continual learning on the 6,250,000-input Fourier tree.
//! Three independent synthetic teachers, disjoint seeded TRAIN-A/TRAIN-B/
//! VALID/TEST input streams, three sequential passes per training batch.
//! Never consult hidden teacher activations for fitting or validation labels
//! for proposal selection. VALID/TEST are inspected *after* each pass.
//! This is a controlled capacity experiment, not a BANC biological model.
use std::time::Instant;

use v888_booleen::fourier5::FiveCircle;
use v888_booleen::fourier5_hierarchy::GateKind;
use v888_booleen::fourier5_wide::{WideFiveTree, SIX_MILLION_INPUTS};
use v888_booleen::fourier5_wide_learning::WidePlasticCircuit;

const TRIALS_PER_PASS: usize = 4_096;
const PASSES_PER_BATCH: usize = 3;
const SEEDS: [u64; 3] = [0x9a77_0001, 0x9a77_0002, 0x9a77_0003];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn words(count: usize, seed: u64) -> Vec<u64> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        })
        .collect()
}

fn hash(layer: usize, neuron: usize, seed: u64) -> u64 {
    let mut x = (neuron as u64) ^ ((layer as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)) ^ seed;
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn complement_parity() -> FiveCircle {
    let original = GateKind::Parity.five_circle();
    let mut circles = *original.circles();
    for circle in &mut circles {
        circle.phase_pi = !circle.phase_pi;
    }
    FiveCircle::new(circles, -original.bias()).expect("existing Walsh masks")
}

fn candidate_models() -> [FiveCircle; 5] {
    [
        GateKind::Majority.five_circle(),
        GateKind::Parity.five_circle(),
        GateKind::And.five_circle(),
        GateKind::Or.five_circle(),
        complement_parity(),
    ]
}

fn correct(predicted: u64, label: u64) -> u32 {
    64 - (predicted ^ label).count_ones()
}

fn score(circuit: &WidePlasticCircuit<'_>, sample: &[u64], label: u64) -> Result<u32, String> {
    let layers = circuit
        .evaluate_trace(sample)
        .map_err(|e| format!("{e:?}"))?;
    Ok(correct(layers.last().expect("one output layer")[0], label))
}

struct PassStats {
    net_accepts: usize,
    replay_rollbacks: usize,
    evaluations: usize,
    child_reads: usize,
    elapsed_us: u128,
}

/// Reward is the observed improvement of 64 final-output labels on TRAIN.
/// The output impact of each proposed neuron change is its exact
/// counterfactual credit. The optional replay constraint uses TRAIN-A only.
/// Structural source-connection plasticity is NOT implemented here.
fn train_pass(
    circuit: &mut WidePlasticCircuit<'_>,
    widths: &[usize],
    target: u64,
    seed: u64,
    options: &[FiveCircle; 5],
    replay: Option<(&[u64], u64, u32)>,
    trials: usize,
) -> Result<PassStats, String> {
    let start = Instant::now();
    let mut rng = Rng(seed);
    let mut best = correct(circuit.output(), target);
    let mut net_accepts = 0;
    let mut replay_rollbacks = 0;
    let mut evaluations = 0;
    let mut child_reads = 0;
    for _ in 0..trials {
        let layer = rng.next() as usize % widths.len();
        let neuron = rng.next() as usize % widths[layer];
        let model = options[rng.next() as usize % options.len()];
        let before_model = circuit.model(layer, neuron).map_err(|e| format!("{e:?}"))?;
        if model == before_model {
            continue;
        }
        let proposal = circuit
            .propose(layer, neuron, model)
            .map_err(|e| format!("{e:?}"))?;
        evaluations += proposal.gate_evaluations;
        child_reads += proposal.child_word_reads;
        let candidate = correct(proposal.output, target);
        if candidate <= best {
            continue;
        }
        circuit.accept(proposal).map_err(|e| format!("{e:?}"))?;
        let preserved = if let Some((old_sources, old_target, old_floor)) = replay {
            score(circuit, old_sources, old_target)? >= old_floor
        } else {
            true
        };
        if preserved {
            best = candidate;
            net_accepts += 1;
        } else {
            let undo = circuit
                .propose(layer, neuron, before_model)
                .map_err(|e| format!("{e:?}"))?;
            circuit.accept(undo).map_err(|e| format!("{e:?}"))?;
            replay_rollbacks += 1;
        }
    }
    if correct(circuit.output(), target) != best {
        return Err("TRAIN reward drift after persistent commits".into());
    }
    if let Some((old_sources, old_target, old_floor)) = replay {
        if score(circuit, old_sources, old_target)? < old_floor {
            return Err("TRAIN-A retention invariant violated".into());
        }
    }
    Ok(PassStats {
        net_accepts,
        replay_rollbacks,
        evaluations,
        child_reads,
        elapsed_us: start.elapsed().as_micros(),
    })
}

struct Assessment<'a> {
    seed: u64,
    a: &'a [u64],
    label_a: u64,
    b: &'a [u64],
    label_b: u64,
    valid: &'a [u64],
    label_valid: u64,
    test: &'a [u64],
    label_test: u64,
}

fn report(
    set: &Assessment<'_>,
    phase: &str,
    pass: usize,
    arm: &str,
    model: &WidePlasticCircuit<'_>,
    stats: &PassStats,
) -> Result<(), String> {
    // No VALID or TEST labels are used by train_pass.
    let seed = set.seed;
    let score_a = score(model, set.a, set.label_a)?;
    let score_b = score(model, set.b, set.label_b)?;
    let score_valid = score(model, set.valid, set.label_valid)?;
    let score_test = score(model, set.test, set.label_test)?;
    println!(
        "REWARD_PASS seed={seed} phase={phase} pass={pass} arm={arm} train_a={score_a}/64 train_b={score_b}/64 valid={score_valid}/64 test={score_test}/64 accepted={} replay_rollbacks={} cached_gate_evals={} cached_child_reads={} elapsed_us={} persistent_commits={}",
        stats.net_accepts,
        stats.replay_rollbacks,
        stats.evaluations,
        stats.child_reads,
        stats.elapsed_us,
        model.commits()
    );
    Ok(())
}

fn experiment(width: usize, seed: u64, trials: usize, passes: usize) -> Result<(), String> {
    let total_start = Instant::now();
    let tree = WideFiveTree::new(width).map_err(|e| format!("{e:?}"))?;
    let a = words(width, seed ^ 0x1111_2222_3333_4444);
    let b = words(width, seed ^ 0x5555_6666_7777_8888);
    let valid = words(width, seed ^ 0x1357_9bdf_2468_ace0);
    let test = words(width, seed ^ 0x0f0f_f0f0_5a5a_a5a5);
    let options = candidate_models();
    let mut teacher = WidePlasticCircuit::new(&tree, &a).map_err(|e| format!("{e:?}"))?;
    teacher.install_models(|layer, neuron| {
        let bucket = hash(layer, neuron, seed) % 10;
        if layer == 0 {
            options[match bucket {
                0 => 2,
                1 => 3,
                2..=5 => 0,
                _ => 1,
            }]
        } else if bucket % 2 == 0 {
            options[1]
        } else {
            options[4]
        }
    });
    // Only the teacher's FINAL output is returned as an observable label.
    let label_a = teacher.output();
    let label_b = teacher
        .evaluate_trace(&b)
        .map_err(|e| format!("{e:?}"))?
        .last()
        .unwrap()[0];
    let label_valid = teacher
        .evaluate_trace(&valid)
        .map_err(|e| format!("{e:?}"))?
        .last()
        .unwrap()[0];
    let label_test = teacher
        .evaluate_trace(&test)
        .map_err(|e| format!("{e:?}"))?
        .last()
        .unwrap()[0];
    drop(teacher);
    println!(
        "REWARD_SPLITS seed={seed} width={width} lanes=64 train_a_positive={}          train_b_positive={} valid_positive={} test_positive={} streams_seeded_separately=true",
        label_a.count_ones(), label_b.count_ones(),
        label_valid.count_ones(), label_test.count_ones()
    );
    if [label_a, label_b, label_valid, label_test]
        .iter()
        .any(|x| !(16..=48).contains(&x.count_ones()))
    {
        return Err("unbalanced final labels; refuse the experiment".into());
    }
    let set = Assessment {
        seed,
        a: &a,
        label_a,
        b: &b,
        label_b,
        valid: &valid,
        label_valid,
        test: &test,
        label_test,
    };
    let mut guarded = WidePlasticCircuit::new(&tree, &a).map_err(|e| format!("{e:?}"))?;
    let empty = PassStats {
        net_accepts: 0,
        replay_rollbacks: 0,
        evaluations: 0,
        child_reads: 0,
        elapsed_us: 0,
    };
    report(&set, "A", 0, "replay_guard", &guarded, &empty)?;
    for pass in 1..=passes {
        let stats = train_pass(
            &mut guarded,
            tree.level_widths(),
            label_a,
            seed ^ (pass as u64).wrapping_mul(0xfeed_f00d),
            &options,
            None,
            trials,
        )?;
        report(&set, "A", pass, "replay_guard", &guarded, &stats)?;
    }
    let retained_floor = correct(guarded.output(), label_a);
    // Construct matched comparator from EXACT A-trained independent models.
    let mut unguarded = WidePlasticCircuit::new(&tree, &a).map_err(|e| format!("{e:?}"))?;
    unguarded.install_models(|layer, neuron| guarded.model(layer, neuron).unwrap());
    if unguarded.output() != guarded.output() {
        return Err("control does not match the A-trained source model".into());
    }
    let rebuild_start = Instant::now();
    guarded.rebind_sources(&b).map_err(|e| format!("{e:?}"))?;
    let guarded_rebuild_us = rebuild_start.elapsed().as_micros();
    let rebuild_start = Instant::now();
    unguarded.rebind_sources(&b).map_err(|e| format!("{e:?}"))?;
    let unguarded_rebuild_us = rebuild_start.elapsed().as_micros();
    println!(
        "REWARD_REBIND seed={seed} width={width} guarded_rebuild_us={guarded_rebuild_us}          unguarded_rebuild_us={unguarded_rebuild_us} retained_a_floor={retained_floor}/64"
    );
    for pass in 1..=passes {
        let proposal_seed = seed ^ (pass as u64).wrapping_mul(0xace5_d00d);
        let guarded_stats = train_pass(
            &mut guarded,
            tree.level_widths(),
            label_b,
            proposal_seed,
            &options,
            Some((&a, label_a, retained_floor)),
            trials,
        )?;
        let unguarded_stats = train_pass(
            &mut unguarded,
            tree.level_widths(),
            label_b,
            proposal_seed,
            &options,
            None,
            trials,
        )?;
        report(&set, "B", pass, "replay_guard", &guarded, &guarded_stats)?;
        report(&set, "B", pass, "no_replay", &unguarded, &unguarded_stats)?;
    }
    if score(&guarded, &a, label_a)? < retained_floor {
        return Err("loss of retained TRAIN-A output accuracy".into());
    }
    println!(
        "REWARD_SUMMARY seed={seed} width={width} trials_per_pass={trials}          passes_per_batch={passes} total_elapsed_ms={} root_only_labels=true          replay_guard_train_only=true structural_plasticity=false",
        total_start.elapsed().as_millis()
    );
    Ok(())
}

fn main() -> Result<(), String> {
    for seed in SEEDS {
        experiment(SIX_MILLION_INPUTS, seed, TRIALS_PER_PASS, PASSES_PER_BATCH)?;
    }
    println!("REWARD_MULTISEED_CONTINUAL_LEARNING_COMPLETE");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_is_defined_only_by_final_labels_and_retention_is_checked() {
        experiment(125, SEEDS[0], 128, 2).unwrap();
    }
}

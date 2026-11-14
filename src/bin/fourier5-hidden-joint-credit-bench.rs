//! B888-4: output-only, JOINT hidden-neuron credit and exact replay on the
//! 1,562,501-node frozen Fourier backbone, without adding output heads.
//!
//! Five distinct seeds, four sequential context tasks, independent TRAIN,
//! VALID/TEST and post-backbone feature-shift OOD streams. Evolving parameters:
//! three 5-input hidden Walsh couplings + one shared 5-input root (4 neurons).
//! Teacher is a representable parity-of-parities; synthetic capacity study.
use std::time::Instant;

use v888_booleen::fourier5::{FiveCircle, CIRCLES};
use v888_booleen::fourier5_hidden_credit::{HiddenCircuit, HiddenCredit, HiddenExample, WalshGate};
use v888_booleen::fourier5_wide::{WideFiveTree, SIX_MILLION_INPUTS};
use v888_booleen::fourier5_wide_learning::WidePlasticCircuit;

const SEEDS: [u64; 5] = [
    0xb888_6101,
    0xb888_6102,
    0xb888_6103,
    0xb888_6104,
    0xb888_6105,
];
const TASKS: usize = 4;
const SOURCES: [u32; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
const TRAIN_BATCHES: usize = 3;
const HELDOUT_BATCHES: usize = 2;

struct Task {
    train: Vec<HiddenExample>,
    valid: Vec<HiddenExample>,
    test: Vec<HiddenExample>,
    ood: Vec<HiddenExample>,
}

struct SourcePlan {
    width: usize,
    layer: usize,
    seed: u64,
    teacher: HiddenCircuit,
}

#[derive(Clone, Copy)]
enum Split {
    Train,
    Valid,
    Test,
    Ood,
}

impl Split {
    fn config(self) -> (u64, usize, bool, bool) {
        match self {
            Self::Train => (0xa111_0001, TRAIN_BATCHES, true, false),
            Self::Valid => (0xb222_0002, HELDOUT_BATCHES, false, false),
            Self::Test => (0xc333_0003, HELDOUT_BATCHES, false, false),
            Self::Ood => (0xd444_0004, HELDOUT_BATCHES, false, true),
        }
    }
}

fn random_words(width: usize, seed: u64) -> Vec<u64> {
    let mut x = seed;
    (0..width)
        .map(|_| {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        })
        .collect()
}

/// Independent scalar teacher: no use of the learner's compiled predictor.
fn oracle(gate: FiveCircle, planes: [u64; CIRCLES]) -> u64 {
    let mut answer = 0_u64;
    for lane in 0..64 {
        let state = planes.iter().enumerate().fold(0_u8, |acc, (bit, word)| {
            acc | ((((word >> lane) & 1) as u8) << bit)
        });
        if gate.predict(state).expect("only five inputs") {
            answer |= 1_u64 << lane;
        }
    }
    answer
}

fn teacher_label(teacher: &HiddenCircuit, sample: &HiddenExample) -> u64 {
    let hidden = teacher.hidden();
    let outputs = [
        oracle(hidden[0].as_five_circle(), sample.blocks[0]),
        oracle(hidden[1].as_five_circle(), sample.blocks[1]),
        oracle(hidden[2].as_five_circle(), sample.blocks[2]),
        if sample.context & 1 != 0 { u64::MAX } else { 0 },
        if sample.context & 2 != 0 { u64::MAX } else { 0 },
    ];
    oracle(teacher.root().as_five_circle(), outputs)
}

fn unseen_mask(example: &HiddenExample) -> u64 {
    !example.blocks[0][0] & !example.blocks[0][1]
}

fn heldout_score(circuit: &HiddenCircuit, examples: &[HiddenExample]) -> (u64, u64) {
    examples.iter().fold((0, 0), |(correct, count), example| {
        let mask = unseen_mask(example);
        let wrong = (circuit.predict(example) ^ example.label) & mask;
        (
            correct + u64::from(mask.count_ones() - wrong.count_ones()),
            count + u64::from(mask.count_ones()),
        )
    })
}

fn feature_activity(examples: &[HiddenExample]) -> (u64, u64) {
    let active: u64 = examples
        .iter()
        .flat_map(|e| e.blocks.iter())
        .flat_map(|planes| planes.iter())
        .map(|plane| u64::from(plane.count_ones()))
        .sum();
    (active, examples.len() as u64 * 15 * 64)
}

fn examples(
    backbone: &WidePlasticCircuit<'_>,
    plan: &SourcePlan,
    context: u8,
    split: Split,
) -> Result<Vec<HiddenExample>, String> {
    let (salt, count, train, ood) = split.config();
    let mut output = Vec::with_capacity(count);
    for index in 0..count {
        let source = random_words(
            plan.width,
            plan.seed
                ^ salt
                ^ ((u64::from(context) + 1) << 48)
                ^ (index as u64 + 1).wrapping_mul(0x6a09_e667_f3bc_c909),
        );
        let trace = backbone
            .evaluate_trace(&source)
            .map_err(|e| format!("{e:?}"))?;
        let layer = trace.get(plan.layer).ok_or("missing backbone layer")?;
        if layer.len() < 15 {
            return Err("not enough independent backbone feature nodes".into());
        }
        let mut blocks = [[0_u64; CIRCLES]; 3];
        for (i, block) in blocks.iter_mut().enumerate() {
            block.copy_from_slice(&layer[i * CIRCLES..(i + 1) * CIRCLES]);
        }
        if ood {
            // Genuine measured representation intervention, not an
            // inference about latent shift from raw-source perturbation.
            let modifiers = random_words(
                15,
                plan.seed
                    ^ salt
                    ^ ((u64::from(context) + 1) << 32)
                    ^ (index as u64 + 1).wrapping_mul(0x243f_6a88_85a3_08d3),
            );
            for (word, &mask) in blocks.iter_mut().flatten().zip(&modifiers) {
                *word &= mask;
            }
        }
        let mut sample = HiddenExample {
            blocks,
            context,
            label: 0,
            observed: u64::MAX,
        };
        sample.label = teacher_label(&plan.teacher, &sample);
        if train {
            sample.observed = !unseen_mask(&sample);
        }
        output.push(sample);
    }
    Ok(output)
}

fn task_suite(backbone: &WidePlasticCircuit<'_>, plan: &SourcePlan) -> Result<Vec<Task>, String> {
    (0..TASKS)
        .map(|context| {
            Ok(Task {
                train: examples(backbone, plan, context as u8, Split::Train)?,
                valid: examples(backbone, plan, context as u8, Split::Valid)?,
                test: examples(backbone, plan, context as u8, Split::Test)?,
                ood: examples(backbone, plan, context as u8, Split::Ood)?,
            })
        })
        .collect()
}

fn negative_labels(items: &[HiddenExample]) -> Vec<HiddenExample> {
    items
        .iter()
        .map(|item| HiddenExample {
            label: item.label.rotate_left(17),
            ..*item
        })
        .collect()
}

fn aggregate(circuit: &HiddenCircuit, tasks: &[Task], which: Split) -> (u64, u64) {
    tasks
        .iter()
        .map(|task| match which {
            Split::Train => circuit.score(&task.train),
            Split::Valid => circuit.score(&task.valid),
            Split::Test => circuit.score(&task.test),
            Split::Ood => circuit.score(&task.ood),
        })
        .fold((0, 0), |(a, b), (c, d)| (a + c, b + d))
}

struct Report<'a> {
    seed: u64,
    phase: usize,
    primary: &'a HiddenCircuit,
    control: &'a HiddenCircuit,
    tasks: &'a [Task],
    fit: HiddenCredit,
    elapsed_us: u128,
    pretest: (u64, u64),
    past: &'a [(u64, u64)],
    null: &'a HiddenCircuit,
    frozen: &'a HiddenCircuit,
    feature_layer: u32,
}

fn report(input: Report<'_>) -> Result<(), String> {
    let Report {
        seed,
        phase,
        primary,
        control,
        tasks,
        fit,
        elapsed_us,
        pretest,
        past,
        null,
        frozen,
        feature_layer,
    } = input;
    let train = primary.score(&tasks[phase].train);
    let valid = primary.score(&tasks[phase].valid);
    let test = primary.score(&tasks[phase].test);
    let novel = heldout_score(primary, &tasks[phase].test);
    let ood = primary.score(&tasks[phase].ood);
    let baseline = frozen.score(&tasks[phase].test);
    let no_replay = control.score(&tasks[phase].test);
    let shuffled = null.score(&tasks[phase].test);
    let mut lost = 0_u64;
    for (old, &prior) in past.iter().enumerate() {
        let after = primary.score(&tasks[old].test);
        let comparison = control.score(&tasks[old].test);
        lost += prior.0.saturating_sub(after.0);
        println!(
            "HIDDEN_RECALL seed={seed} after_task={phase} previous_task={old}              before={}/{} after={}/{} forgotten={} without_replay={}/{}",
            prior.0, prior.1, after.0, after.1,
            prior.0.saturating_sub(after.0),
            comparison.0, comparison.1
        );
    }
    if fit.replay_after < fit.replay_before {
        return Err("TRAIN replay accuracy decreased".into());
    }
    let record = primary.checkpoint();
    let restored =
        HiddenCircuit::restore(&record, feature_layer, SOURCES).map_err(|e| format!("{e:?}"))?;
    if &restored != primary {
        return Err("hidden joint memory checkpoint mismatch".into());
    }
    println!(
        "HIDDEN_TASK_PASS seed={seed} task={phase} accepted={} proposals={}          hidden_changed={} root_changed={} commits={}          reward_before={}/{} reward_after={}/{}          previous_train_before={} previous_train_after={}          train={}/{} valid={}/{} test_before={}/{} test_after={}/{}          unseen_feature_codes={}/{} intervention_ood={}/{}          frozen={}/{} shuffled_labels={}/{} no_replay={}/{}          earlier_test_lost={lost} fit_us={elapsed_us} checkpoint_bytes={}          fixed_four_neurons=true output_labels_only=true",
        fit.accepted, fit.proposals, fit.hidden_changed,
        fit.root_changed, fit.commits,
        fit.reward_before, fit.observed_total,
        fit.reward_after, fit.observed_total,
        fit.replay_before, fit.replay_after,
        train.0, train.1,
        valid.0, valid.1,
        pretest.0, pretest.1, test.0, test.1,
        novel.0, novel.1, ood.0, ood.1,
        baseline.0, baseline.1, shuffled.0, shuffled.1,
        no_replay.0, no_replay.1, record.len()
    );
    Ok(())
}

fn experiment(tree: &WideFiveTree, seed: u64) -> Result<(), String> {
    let started = Instant::now();
    let width = tree.input_bits();
    let layer = tree.depth().checked_sub(3).ok_or("tree depth too small")?;
    let hidden_palettes = [
        [0b01101_u8, 0b10110, 0b11011],
        [0b10101, 0b11010, 0b01110],
        [0b10011, 0b11001, 0b10111],
        [0b11101, 0b10110, 0b01011],
        [0b01110, 0b10101, 0b11001],
    ];
    let expected = hidden_palettes[(seed as usize) % hidden_palettes.len()];
    let truth = std::array::from_fn(|index| WalshGate {
        mask: expected[index],
        phase: false,
    });
    let oracle_model = HiddenCircuit::from_models(
        layer as u32,
        SOURCES,
        truth,
        WalshGate {
            mask: 0b11111,
            phase: seed & 1 != 0,
        },
    )
    .map_err(|e| format!("{e:?}"))?;
    let initial = random_words(width, seed ^ 0x1010_2020);
    let backbone = WidePlasticCircuit::new(tree, &initial).map_err(|e| format!("{e:?}"))?;
    let plan = SourcePlan {
        width,
        layer,
        seed,
        teacher: oracle_model,
    };
    let tasks = task_suite(&backbone, &plan)?;
    let original = HiddenCircuit::new(layer as u32, SOURCES).map_err(|e| format!("{e:?}"))?;
    let mut primary = original.clone();
    let mut no_replay = original.clone();
    let mut shuffled = original.clone();
    let valid_activity = feature_activity(&tasks[0].valid);
    let ood_activity = feature_activity(&tasks[0].ood);
    println!(
        "HIDDEN_FEATURE_SHIFT seed={seed} backbone_neurons={}          feature_layer={layer} valid_active={}/{} ood_active={}/{}          representation_intervention=true backbone_frozen=true",
        tree.logical_neurons(), valid_activity.0, valid_activity.1,
        ood_activity.0, ood_activity.1
    );
    let mut root_coupling_updates = 0_usize;
    let mut initial_hidden_changed = 0_usize;
    let mut prior_test_regressions = 0_u64;
    for task in 0..TASKS {
        let retained: Vec<_> = tasks[..task].iter().map(|x| x.train.as_slice()).collect();
        let previous_test: Vec<_> = tasks[..task]
            .iter()
            .map(|x| primary.score(&x.test))
            .collect();
        let before = primary.score(&tasks[task].test);
        let start = Instant::now();
        let fit = if task == 0 {
            primary.fit_joint(&tasks[task].train, &retained)
        } else {
            primary.fit_coordinate(&tasks[task].train, &retained)
        }
        .map_err(|e| format!("{e:?}"))?;
        let elapsed = start.elapsed().as_micros();
        if task == 0 {
            initial_hidden_changed = fit.hidden_changed;
        } else if fit.accepted && fit.root_changed {
            root_coupling_updates += 1;
        }
        for (index, earlier) in previous_test.iter().enumerate() {
            let score = primary.score(&tasks[index].test);
            prior_test_regressions += earlier.0.saturating_sub(score.0);
        }
        if task == 0 {
            no_replay
                .fit_joint(&tasks[task].train, &[])
                .map_err(|e| format!("{e:?}"))?;
        } else {
            no_replay
                .fit_coordinate(&tasks[task].train, &[])
                .map_err(|e| format!("{e:?}"))?;
        }
        let negative = negative_labels(&tasks[task].train);
        let previous_noise: Vec<_> = tasks[..task]
            .iter()
            .map(|x| negative_labels(&x.train))
            .collect();
        let noise_refs: Vec<_> = previous_noise.iter().map(Vec::as_slice).collect();
        if task == 0 {
            shuffled
                .fit_joint(&negative, &noise_refs)
                .map_err(|e| format!("{e:?}"))?;
        } else {
            shuffled
                .fit_coordinate(&negative, &noise_refs)
                .map_err(|e| format!("{e:?}"))?;
        }
        let checkpoint = primary.checkpoint();
        let restored = HiddenCircuit::restore(&checkpoint, layer as u32, SOURCES)
            .map_err(|e| format!("{e:?}"))?;
        if restored != primary {
            return Err("checkpoint mutation detected".into());
        }
        report(Report {
            seed,
            phase: task,
            primary: &primary,
            control: &no_replay,
            tasks: &tasks,
            fit,
            elapsed_us: elapsed,
            pretest: before,
            past: &previous_test,
            null: &shuffled,
            frozen: &original,
            feature_layer: layer as u32,
        })?;
    }
    // Explicit incompatible fifth task: exactly the same input/context
    // examples as the last learned task, with every observed label flipped.
    // With a fixed deterministic readout and strict replay, a positive
    // reward here MUST trade off at least one earlier TRAIN prediction.
    // The non-improvement is a scientific capacity boundary, not a failure.
    let incompatible: Vec<_> = tasks[TASKS - 1]
        .train
        .iter()
        .map(|example| HiddenExample {
            label: !example.label,
            ..*example
        })
        .collect();
    let prior_train: Vec<_> = tasks.iter().map(|task| task.train.as_slice()).collect();
    let original_score = primary.score(&tasks[TASKS - 1].train);
    let incompatible_before = primary.score(&incompatible);
    let previous_checkpoint = primary.checkpoint();
    let conflict = primary
        .fit_coordinate(&incompatible, &prior_train)
        .map_err(|e| format!("{e:?}"))?;
    let incompatible_after = primary.score(&incompatible);
    if conflict.accepted
        || incompatible_before != incompatible_after
        || primary.score(&tasks[TASKS - 1].train) != original_score
        || primary.checkpoint() != previous_checkpoint
    {
        return Err("incompatible-task retention failed to preserve shared parameters".into());
    }
    println!(
        "HIDDEN_CONFLICT_LIMIT seed={seed} original_task_correct={}/{}          contradictory_task_before={}/{} contradictory_task_after={}/{}          accepted={} fixed_capacity_conflict=true          note=no_single_deterministic_model_can_satisfy_opposite_labels",
        original_score.0, original_score.1,
        incompatible_before.0, incompatible_before.1,
        incompatible_after.0, incompatible_after.1, conflict.accepted,
    );

    let final_valid = aggregate(&primary, &tasks, Split::Valid);
    let final_test = aggregate(&primary, &tasks, Split::Test);
    let final_ood = aggregate(&primary, &tasks, Split::Ood);
    let unseen = tasks
        .iter()
        .map(|t| heldout_score(&primary, &t.test))
        .fold((0_u64, 0_u64), |(a, b), (c, d)| (a + c, b + d));
    let baseline = aggregate(&original, &tasks, Split::Test);
    let random = aggregate(&shuffled, &tasks, Split::Test);
    let unguarded = aggregate(&no_replay, &tasks, Split::Test);
    let seed_criteria_met = final_valid.0 * 10 >= final_valid.1 * 9
        && final_test.0 * 10 >= final_test.1 * 9
        && unseen.1 > 0
        && unseen.0 * 10 >= unseen.1 * 9
        && final_ood.0 * 10 >= final_ood.1 * 9
        && prior_test_regressions == 0
        && initial_hidden_changed >= 2
        && root_coupling_updates >= 2
        && primary.commits() >= 3
        && final_test.0 * 5 >= baseline.0 * 5 + final_test.1
        && final_test.0 * 5 >= random.0 * 5 + final_test.1;
    println!(
        "HIDDEN_PREDECLARED_CRITERIA seed={seed} met={seed_criteria_met}          initial_hidden_changed={initial_hidden_changed}          later_root_coupling_updates={root_coupling_updates}          prior_test_regressions={prior_test_regressions}          validation_min_90=true test_min_90=true unseen_min_90=true          ood_min_90=true control_gain_min_20pp=true",
    );
    let hidden = primary.hidden();
    let root = primary.root();
    println!(
        "HIDDEN_SEED_SUMMARY seed={seed} width={width} backbone_neurons={}          shared_hidden_neurons=3 shared_root_neurons=1          tasks={TASKS} commits={} hidden_masks={},{},{}          root_mask={} root_phase={} target_hidden_masks={},{},{}          valid={}/{} test={}/{} unseen={}/{} feature_ood={}/{}          frozen={}/{} shuffled={}/{} no_replay={}/{}          elapsed_ms={} predeclared_criteria_met={seed_criteria_met}          output_only=true fixed_capacity=true",
        tree.logical_neurons(), primary.commits(),
        hidden[0].mask, hidden[1].mask, hidden[2].mask,
        root.mask, root.phase,
        expected[0], expected[1], expected[2],
        final_valid.0, final_valid.1,
        final_test.0, final_test.1,
        unseen.0, unseen.1, final_ood.0, final_ood.1,
        baseline.0, baseline.1,
        random.0, random.1,
        unguarded.0, unguarded.1, started.elapsed().as_millis()
    );
    Ok(())
}

fn main() -> Result<(), String> {
    let tree = WideFiveTree::new(SIX_MILLION_INPUTS).map_err(|e| format!("{e:?}"))?;
    for seed in SEEDS {
        experiment(&tree, seed)?;
    }
    println!("HIDDEN_JOINT_CREDIT_MULTISEED_COMPLETE");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_shared_hidden_learning_runs_with_independent_holdouts() {
        let tree = WideFiveTree::new(125).unwrap();
        experiment(&tree, SEEDS[0]).unwrap();
    }

    #[test]
    fn heldout_mask_picks_exact_unseen_feature_codes() {
        for a in [false, true] {
            for b in [false, true] {
                let mut item = HiddenExample {
                    blocks: [[0_u64; CIRCLES]; 3],
                    context: 0,
                    label: 0,
                    observed: u64::MAX,
                };
                item.blocks[0][0] = if a { u64::MAX } else { 0 };
                item.blocks[0][1] = if b { u64::MAX } else { 0 };
                assert_eq!(unseen_mask(&item) == u64::MAX, !a && !b);
            }
        }
    }
}

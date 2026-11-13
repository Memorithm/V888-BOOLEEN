//! Equal-capacity shared contextual continual learning.
//! One trainable Fourier neuron receives three frozen wide-tree features and
//! two context bits. Five independent seeds, four sequential tasks, output
//! labels only. No independent task heads, no 1.56M-backbone plasticity.
use std::time::Instant;

use v888_booleen::fourier5::{FiveCircle, CIRCLES};
use v888_booleen::fourier5_shared_context::{neuron, SharedContextNeuron};
use v888_booleen::fourier5_task_memory::TaskBatch;
use v888_booleen::fourier5_wide::{WideFiveTree, SIX_MILLION_INPUTS};
use v888_booleen::fourier5_wide_learning::WidePlasticCircuit;

const SEEDS: [u64; 5] = [0xb888_5101, 0xb888_5102, 0xb888_5103, 0xb888_5104, 0xb888_5105];
const TASKS: usize = 4;

fn data(size: usize, seed: u64) -> Vec<u64> {
    let mut x = seed;
    (0..size)
        .map(|_| {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        })
        .collect()
}

fn oracle(model: FiveCircle, planes: [u64; CIRCLES]) -> u64 {
    let mut result = 0_u64;
    for lane in 0..64 {
        let code = planes.iter().enumerate().fold(0_u8, |bits, (idx, value)| {
            bits | ((((value >> lane) & 1) as u8) << idx)
        });
        if model.predict(code).expect("five input bits") {
            result |= 1_u64 << lane;
        }
    }
    result
}

fn withheld(planes: [u64; CIRCLES]) -> u64 {
    !planes[0] & !planes[1]
}

fn scores(model: &SharedContextNeuron, items: &[TaskBatch], novel: bool) -> (u64, u64) {
    items.iter().fold((0, 0), |(right, count), item| {
        let admit = if novel { withheld(item.features) } else { item.observed };
        let wrong = (model.predict(item.features) ^ item.labels) & admit;
        (
            right + u64::from(admit.count_ones() - wrong.count_ones()),
            count + u64::from(admit.count_ones()),
        )
    })
}

fn total_score(model: &SharedContextNeuron, tasks: &[Task], split: usize) -> (u64, u64) {
    tasks.iter().map(|t| match split {
        0 => scores(model, &t.valid, false),
        1 => scores(model, &t.test, false),
        2 => scores(model, &t.ood, false),
        _ => scores(model, &t.test, true),
    })
    .fold((0, 0), |(a, b), (x, y)| (a + x, b + y))
}

struct Task {
    train: Vec<TaskBatch>,
    valid: Vec<TaskBatch>,
    test: Vec<TaskBatch>,
    ood: Vec<TaskBatch>,
}

fn split(
    circuit: &WidePlasticCircuit<'_>,
    size: usize,
    layer: usize,
    seed: u64,
    task: usize,
    salt: u64,
    batches: usize,
    training: bool,
    ood: bool,
    target: FiveCircle,
) -> Result<Vec<TaskBatch>, String> {
    let mut result = Vec::new();
    for i in 0..batches {
        let source = data(
            size,
            seed ^ salt ^ ((task as u64 + 1) << 48)
                ^ (i as u64 + 1).wrapping_mul(0x6a09_e667_f3bc_c909),
        );
        let trace = circuit.evaluate_trace(&source).map_err(|e| format!("{e:?}"))?;
        let row = trace.get(layer).ok_or("missing feature layer")?;
        if row.len() < 3 {
            return Err("need three independent backbone nodes".into());
        }
        let mut features = [0_u64; CIRCLES];
        features[..3].copy_from_slice(&row[..3]);
        if ood {
            let modifier = data(3, seed ^ salt ^ ((task as u64 + 1) << 32) ^ i as u64);
            for (plane, m) in features[..3].iter_mut().zip(modifier) {
                *plane &= m;
            }
        }
        features[3] = if task & 1 != 0 { u64::MAX } else { 0 };
        features[4] = if task & 2 != 0 { u64::MAX } else { 0 };
        result.push(TaskBatch {
            features,
            labels: oracle(target, features),
            observed: if training { !withheld(features) } else { u64::MAX },
        });
    }
    Ok(result)
}

fn suite(
    circuit: &WidePlasticCircuit<'_>,
    size: usize,
    layer: usize,
    seed: u64,
    target: FiveCircle,
) -> Result<Vec<Task>, String> {
    (0..TASKS)
        .map(|t| {
            Ok(Task {
                train: split(circuit, size, layer, seed, t, 0x1111_0001, 3, true, false, target)?,
                valid: split(circuit, size, layer, seed, t, 0x2222_0002, 2, false, false, target)?,
                test: split(circuit, size, layer, seed, t, 0x3333_0003, 2, false, false, target)?,
                ood: split(circuit, size, layer, seed, t, 0x4444_0004, 2, false, true, target)?,
            })
        })
        .collect()
}

fn feature_activity(items: &[TaskBatch]) -> (u64, u64) {
    let active = items.iter().flat_map(|item| item.features[..3].iter())
        .map(|w| u64::from(w.count_ones())).sum();
    (active, items.len() as u64 * 3 * 64)
}

fn experiment(tree: &WideFiveTree, seed: u64) -> Result<(), String> {
    let started = Instant::now();
    let size = tree.input_bits();
    let layer = tree.depth().checked_sub(3).ok_or("backbone too shallow")?;
    let data_masks = [3_u8, 5, 6, 7, 3];
    let target_mask = 0b1_1000 | data_masks[(seed as usize) % data_masks.len()];
    let target_phase = seed & 1 != 0;
    let target = neuron(target_mask, target_phase).ok_or("target invalid")?;
    let init = data(size, seed ^ 0x3c3c_a5a5);
    let circuit = WidePlasticCircuit::new(tree, &init).map_err(|e| format!("{e:?}"))?;
    let tasks = suite(&circuit, size, layer, seed, target)?;
    let base = SharedContextNeuron::new(layer as u32, [0, 1, 2]).map_err(|e| format!("{e:?}"))?;
    let mut model = base.clone();
    let mut no_replay = base.clone();
    let mut shuffled = base.clone();
    let ordinary = feature_activity(&tasks[0].valid);
    let shifted = feature_activity(&tasks[0].ood);
    println!(
        "SHARED_FEATURE_SHIFT seed={seed} backbone_neurons={}          valid_active={}/{} shifted_active={}/{}          OOD_intervention_after_backbone=true",
        tree.logical_neurons(), ordinary.0, ordinary.1, shifted.0, shifted.1
    );
    let mut initial_test = Vec::new();
    for task in 0..TASKS {
        let previous: Vec<&[TaskBatch]> = tasks[..task].iter().map(|t| t.train.as_slice()).collect();
        let past_test: Vec<_> = tasks[..task].iter()
            .map(|t| scores(&model, &t.test, false))
            .collect();
        let pre = scores(&model, &tasks[task].test, false);
        let now = Instant::now();
        let fit = model.fit(&tasks[task].train, &previous).map_err(|e| format!("{e:?}"))?;
        let fit_us = now.elapsed().as_micros();
        no_replay.fit(&tasks[task].train, &[]).map_err(|e| format!("{e:?}"))?;
        let corrupted: Vec<TaskBatch> = tasks[task].train.iter().map(|b| TaskBatch {
            features: b.features,
            labels: b.labels.rotate_left(17),
            observed: b.observed,
        }).collect();
        let prior_corrupted: Vec<Vec<TaskBatch>> = tasks[..task].iter()
            .map(|t| t.train.iter().map(|b| TaskBatch {
                features: b.features,
                labels: b.labels.rotate_left(17),
                observed: b.observed,
            }).collect())
            .collect();
        let prior_refs: Vec<&[TaskBatch]> = prior_corrupted.iter().map(Vec::as_slice).collect();
        shuffled.fit(&corrupted, &prior_refs).map_err(|e| format!("{e:?}"))?;
        let checkpoint = model.checkpoint();
        let restored = SharedContextNeuron::restore(&checkpoint, layer as u32, [0, 1, 2])
            .map_err(|e| format!("{e:?}"))?;
        if model != restored || fit.previous_after < fit.previous_before {
            return Err("replay retention or persistence invariant failed".into());
        }
        let mut regressions = 0_u64;
        let mut old_correct = 0_u64;
        let mut old_control = 0_u64;
        for (past, &baseline) in past_test.iter().enumerate() {
            let retained = scores(&model, &tasks[past].test, false);
            let control = scores(&no_replay, &tasks[past].test, false);
            regressions += baseline.0.saturating_sub(retained.0);
            old_correct += retained.0;
            old_control += control.0;
            println!(
                "SHARED_RECALL seed={seed} phase={task} past_task={past}                  before={}/{} retained={}/{} lost={} no_replay_correct={}/{}",
                baseline.0, baseline.1, retained.0, retained.1,
                baseline.0.saturating_sub(retained.0), control.0, control.1
            );
        }
        let train = scores(&model, &tasks[task].train, false);
        let valid = scores(&model, &tasks[task].valid, false);
        let test = scores(&model, &tasks[task].test, false);
        let novel = scores(&model, &tasks[task].test, true);
        let ood = scores(&model, &tasks[task].ood, false);
        let frozen = scores(&base, &tasks[task].test, false);
        let negative = scores(&shuffled, &tasks[task].test, false);
        let without = scores(&no_replay, &tasks[task].test, false);
        initial_test.push(test);
        println!(
            "SHARED_TASK_PASS seed={seed} task={task}              mask={} phase_pi={} changed_coupling_bits={} accepted={} commits={}              candidates={} reward_before={}/{} reward_after={}/{}              previous_train_before={} previous_train_after={}              train={}/{} valid={}/{} test_before={}/{} test_after={}/{}              unseen_combinations={}/{} shifted_feature_ood={}/{}              frozen={}/{} shuffled_labels={}/{} no_replay={}/{}              earlier_test_correct={old_correct} earlier_no_replay_correct={old_control}              earlier_test_regressions={regressions} fit_us={fit_us}              checkpoint_bytes={} persistent_exact=true",
            model.coupling_mask(), model.phase(), fit.changed_coupling_bits,
            fit.accepted, fit.commits, fit.candidates_tested,
            fit.current_before, fit.current_total, fit.current_after, fit.current_total,
            fit.previous_before, fit.previous_after,
            train.0, train.1, valid.0, valid.1,
            pre.0, pre.1, test.0, test.1,
            novel.0, novel.1, ood.0, ood.1,
            frozen.0, frozen.1, negative.0, negative.1,
            without.0, without.1, checkpoint.len()
        );
    }
    let valid = total_score(&model, &tasks, 0);
    let test = total_score(&model, &tasks, 1);
    let shifted = total_score(&model, &tasks, 2);
    let unseen = total_score(&model, &tasks, 3);
    let frozen = total_score(&base, &tasks, 1);
    let corrupted = total_score(&shuffled, &tasks, 1);
    let no_replay_score = total_score(&no_replay, &tasks, 1);
    println!(
        "SHARED_SEED_SUMMARY seed={seed} width={size} backbone_neurons={}          shared_trainable_neurons=1 tasks={TASKS} commits={}          learned_mask={} target_mask={target_mask}          valid={}/{} test={}/{} unseen={}/{} feature_ood={}/{}          frozen={}/{} shuffled={}/{} no_replay={}/{}          elapsed_ms={} no_task_heads=true",
        tree.logical_neurons(), model.commits(), model.coupling_mask(),
        valid.0, valid.1, test.0, test.1, unseen.0, unseen.1,
        shifted.0, shifted.1, frozen.0, frozen.1,
        corrupted.0, corrupted.1, no_replay_score.0, no_replay_score.1,
        started.elapsed().as_millis()
    );
    Ok(())
}

fn main() -> Result<(), String> {
    let tree = WideFiveTree::new(SIX_MILLION_INPUTS).map_err(|e| format!("{e:?}"))?;
    for seed in SEEDS {
        experiment(&tree, seed)?;
    }
    println!("SHARED_CONTEXT_MULTISEED_COMPLETE");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_shared_context_reproduction() {
        experiment(&WideFiveTree::new(125).unwrap(), SEEDS[0]).unwrap();
    }

    #[test]
    fn unseen_combinations_are_excluded_from_supervision() {
        for a in [false, true] {
            for b in [false, true] {
                let planes = [
                    if a { u64::MAX } else { 0 },
                    if b { u64::MAX } else { 0 },
                    0, 0, 0,
                ];
                assert_eq!(withheld(planes) == u64::MAX, !a && !b);
            }
        }
    }
}

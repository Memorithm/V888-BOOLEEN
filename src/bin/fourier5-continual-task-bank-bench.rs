//! B888-4: sequential continual learning of multiple observable task outputs
//! from a frozen, physically evaluated wide Fourier backbone.
//!
//! Three source seeds × eight nonidentical task functions; per task 192 TRAIN
//! observations split over 3 fresh 64-lane batches (minus withheld codes),
//! independent 128-lane VALID, 128-lane TEST, 128-lane raw-source OOD.
//! Task-local Fourier heads learn from observable final task labels ONLY.
//! Prior heads are immutable; compare frozen, single-overwritten and
//! shuffled-label controls. No claim of backbone synapse learning or BANC data.
use std::time::Instant;

use v888_booleen::fourier5::{Circle, FiveCircle, CIRCLES};
use v888_booleen::fourier5_task_memory::{TaskBatch, TaskHeadBank};
use v888_booleen::fourier5_wide::{WideFiveTree, SIX_MILLION_INPUTS};
use v888_booleen::fourier5_wide_learning::WidePlasticCircuit;

const SEEDS: [u64; 3] = [0xb888_4001, 0xb888_4002, 0xb888_4003];
const TASKS: usize = 8;
const TRAIN_BATCHES: usize = 3;
const HELDOUT_BATCHES: usize = 2;
const FEATURE_NODES: [u32; CIRCLES] = [0, 1, 2, 3, 4];

fn words(size: usize, seed: u64, biased: bool) -> Vec<u64> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    (0..size)
        .map(|_| {
            let first = next();
            if biased {
                first & next()
            } else {
                first
            }
        })
        .collect()
}

/// Independent scalar oracle. The learner uses a compiled 64-lane
/// predictor; this oracle evaluates one feature code at a time.
fn scalar(model: FiveCircle, planes: [u64; CIRCLES]) -> u64 {
    let mut outputs = 0_u64;
    for bit in 0..64 {
        let word = planes
            .iter()
            .enumerate()
            .fold(0_u8, |v, (i, plane)| v | (((plane >> bit) as u8 & 1) << i));
        if model.predict(word).expect("validated input code") {
            outputs |= 1_u64 << bit;
        }
    }
    outputs
}

fn teacher(mask: u8, phase: bool) -> FiveCircle {
    let mut terms = [Circle::default(); CIRCLES];
    terms[0] = Circle {
        mask,
        radius: 1,
        phase_pi: phase,
    };
    FiveCircle::new(terms, 0).expect("valid Walsh mask")
}

fn heldout_mask(planes: [u64; CIRCLES]) -> u64 {
    // All 8 states with 2 least significant feature bits 00
    // are excluded from ALL TRAIN supervision (no label used on these lanes).
    !planes[0] & !planes[1]
}

fn feature_codes_seen(items: &[TaskBatch]) -> u32 {
    let mut codes = 0_u32;
    for item in items {
        for lane in 0..64 {
            if item.observed & (1_u64 << lane) == 0 {
                continue;
            }
            let input = item.features.iter().enumerate().fold(0_u8, |acc, (i, plane)| {
                acc | ((((plane >> lane) & 1) as u8) << i)
            });
            codes |= 1_u32 << input;
        }
    }
    codes
}

fn feature_plane_set(
    circuit: &WidePlasticCircuit<'_>,
    source: &[u64],
    feature_layer: usize,
) -> Result<[u64; CIRCLES], String> {
    let trace = circuit.evaluate_trace(source).map_err(|e| format!("{e:?}"))?;
    let layer = trace.get(feature_layer).ok_or("missing feature layer")?;
    if layer.len() < CIRCLES {
        return Err("not enough distinct feature neurons".into());
    }
    Ok(std::array::from_fn(|index| layer[index]))
}

fn more_features(
    circuit: &WidePlasticCircuit<'_>,
    width: usize,
    feature_layer: usize,
    seed: u64,
    salt: u64,
    count: usize,
    skewed: bool,
) -> Result<Vec<[u64; CIRCLES]>, String> {
    (0..count)
        .map(|index| {
            let sources = words(
                width,
                seed ^ salt ^ (index as u64 + 1).wrapping_mul(0x6a09_e667_f3bc_c909),
                skewed,
            );
            feature_plane_set(circuit, &sources, feature_layer)
        })
        .collect()
}

fn labeled(features: &[[u64; CIRCLES]], model: FiveCircle, train: bool) -> Vec<TaskBatch> {
    features
        .iter()
        .copied()
        .map(|planes| TaskBatch {
            features: planes,
            labels: scalar(model, planes),
            observed: if train { !heldout_mask(planes) } else { u64::MAX },
        })
        .collect()
}

fn score_model(
    model: FiveCircle,
    samples: &[TaskBatch],
    novel_only: bool,
) -> (u64, u64) {
    samples.iter().fold((0, 0), |(right, total), batch| {
        let mask = if novel_only { heldout_mask(batch.features) } else { u64::MAX };
        let difference = (scalar(model, batch.features) ^ batch.labels) & mask;
        (
            right + u64::from(mask.count_ones() - difference.count_ones()),
            total + u64::from(mask.count_ones()),
        )
    })
}

fn score_bank(bank: &TaskHeadBank, task: usize, samples: &[TaskBatch]) -> Result<(u64, u64), String> {
    let model = bank.model(task).map_err(|e| format!("{e:?}"))?
        .ok_or("missing trained head")?;
    Ok(score_model(model, samples, false))
}

fn untrained_baseline() -> FiveCircle {
    teacher(31, false)
}

struct TaskRecord {
    valid: Vec<TaskBatch>,
    test: Vec<TaskBatch>,
    ood: Vec<TaskBatch>,
    previous_test: (u64, u64),
}

fn experiment(tree: &WideFiveTree, seed: u64, count: usize) -> Result<(), String> {
    let start = Instant::now();
    let width = tree.input_bits();
    let feature_layer = tree.depth().checked_sub(3).ok_or("hierarchy depth under 3")?;
    let first_input = words(width, seed ^ 0x9111_0111, false);
    let circuit = WidePlasticCircuit::new(tree, &first_input).map_err(|e| format!("{e:?}"))?;
    let mut train_features = vec![feature_plane_set(&circuit, &first_input, feature_layer)?];
    train_features.extend(more_features(
        &circuit, width, feature_layer, seed, 0xa111_0001, TRAIN_BATCHES - 1, false,
    )?);
    let valid_features = more_features(
        &circuit, width, feature_layer, seed, 0xb222_0002, HELDOUT_BATCHES, false,
    )?;
    let test_features = more_features(
        &circuit, width, feature_layer, seed, 0xc333_0003, HELDOUT_BATCHES, false,
    )?;
    let ood_features = more_features(
        &circuit, width, feature_layer, seed, 0xd444_0004, HELDOUT_BATCHES, true,
    )?;
    let possible_masks: Vec<u8> = (1..=31_u8)
        .filter(|mask| mask.count_ones() >= 2 && *mask != 31)
        .collect();
    let mut bank = TaskHeadBank::new(count, feature_layer as u32, FEATURE_NODES)
        .map_err(|e| format!("{e:?}"))?;
    let mut shuffled = TaskHeadBank::new(count, feature_layer as u32, FEATURE_NODES)
        .map_err(|e| format!("{e:?}"))?;
    let mut records: Vec<TaskRecord> = Vec::new();
    let feature_valid_ones: u64 = valid_features
        .iter().flat_map(|item| item.iter()).map(|x| u64::from(x.count_ones())).sum();
    let feature_ood_ones: u64 = ood_features
        .iter().flat_map(|item| item.iter()).map(|x| u64::from(x.count_ones())).sum();
    println!(
        "TASK_FEATURE_DISTRIBUTION seed={seed} width={width} backbone_nodes={}          feature_layer={feature_layer} valid_ones={feature_valid_ones}/640          ood_ones={feature_ood_ones}/640 ood_sources_biased_p025=true          task_heads_new=true backbone_frozen=true",
        tree.logical_neurons()
    );

    for task in 0..count {
        let mask = possible_masks[(seed as usize + task * 7) % possible_masks.len()];
        let phase = (seed as usize + task) % 2 != 0;
        let task_teacher = teacher(mask, phase);
        let train = labeled(&train_features, task_teacher, true);
        let valid = labeled(&valid_features, task_teacher, false);
        let test = labeled(&test_features, task_teacher, false);
        let ood = labeled(&ood_features, task_teacher, false);
        let observed_codes = feature_codes_seen(&train);
        if observed_codes & 0x1111_1111 != 0 {
            return Err("combinatorially held-out feature codes leaked into TRAIN".into());
        }
        let snapshot = bank.checkpoint();
        let old_models: Vec<_> = (0..task)
            .map(|old| bank.model(old).expect("previous valid index"))
            .collect();
        let learning_start = Instant::now();
        let fit = bank.fit(task, &train).map_err(|e| format!("{e:?}"))?;
        let fit_us = learning_start.elapsed().as_micros();
        if fit.candidates_tested != 62 {
            return Err("missing Fourier candidate evaluations".into());
        }
        let saved = bank.checkpoint();
        let restored = TaskHeadBank::restore(
            &saved, count, feature_layer as u32, FEATURE_NODES,
        ).map_err(|e| format!("{e:?}"))?;
        if saved != restored.checkpoint() || bank.learned_count() != task + 1 {
            return Err("persistent checkpoint or task count changed".into());
        }
        for (old_task, expected) in old_models.into_iter().enumerate() {
            if bank.model(old_task).map_err(|e| format!("{e:?}"))? != expected {
                return Err("catastrophic parameter overwrite in prior task".into());
            }
        }
        // Negative control: training labels are cyclically assigned to
        // the WRONG sample lanes. Same model class and TRAIN admission mask.
        let shuffled_train: Vec<_> = train
            .iter()
            .map(|b| TaskBatch {
                features: b.features,
                labels: b.labels.rotate_left(17),
                observed: b.observed,
            })
            .collect();
        shuffled.fit(task, &shuffled_train).map_err(|e| format!("{e:?}"))?;
        let head = bank.model(task).map_err(|e| format!("{e:?}"))?
            .ok_or("task head not learned")?;
        let shuffled_head = shuffled.model(task).map_err(|e| format!("{e:?}"))?
            .ok_or("negative control not learned")?;
        let train_score = score_model(head, &train, false);
        let valid_score = score_model(head, &valid, false);
        let test_score = score_model(head, &test, false);
        let novel_score = score_model(head, &test, true);
        let ood_score = score_model(head, &ood, false);
        let baseline = score_model(untrained_baseline(), &test, false);
        let null_score = score_model(shuffled_head, &test, false);
        if novel_score.1 == 0 || fit.train_total < 32 {
            return Err("too few supervised or combinatorially unseen examples".into());
        }
        let record = TaskRecord {
            valid, test, ood, previous_test: test_score,
        };
        records.push(record);
        // Replay every earlier task's independent TEST after EVERY new task.
        let mut past_correct = 0_u64;
        let mut past_total = 0_u64;
        let mut overwrite_correct = 0_u64;
        let mut maximal_forgetting = 0_u64;
        for previous in 0..=task {
            let score = score_bank(&bank, previous, &records[previous].test)?;
            let shared_last = score_model(head, &records[previous].test, false);
            let initial = records[previous].previous_test;
            maximal_forgetting = maximal_forgetting.max(initial.0.saturating_sub(score.0));
            past_correct += score.0;
            past_total += score.1;
            overwrite_correct += shared_last.0;
            println!(
                "TASK_RECALL seed={seed} after_task={task} prior_task={previous}                  test_correct={}/{} initial_correct={}/{} forgotten={}                  single_overwrite_correct={}/{}",
                score.0, score.1, initial.0, initial.1,
                initial.0.saturating_sub(score.0), shared_last.0, shared_last.1
            );
        }
        println!(
            "TASK_LEARNING_PASS seed={seed} task={task} width={width}              train_seen={}/{} seen_codes={} valid={}/{} test={}/{}              novel_combinations={}/{} raw_source_ood={}/{}              frozen_baseline={}/{} shuffled_label_control={}/{}              accepted_head_revision={} fit_us={fit_us}              cumulative_test={past_correct}/{past_total}              single_overwrite_test={overwrite_correct}/{past_total}              max_test_forgetting={maximal_forgetting}              checkpoint_bytes={} checkpoint_exact=true no_hidden_teacher_labels=true",
            fit.train_correct, fit.train_total, observed_codes.count_ones(),
            valid_score.0, valid_score.1,
            test_score.0, test_score.1,
            novel_score.0, novel_score.1,
            ood_score.0, ood_score.1,
            baseline.0, baseline.1,
            null_score.0, null_score.1,
            fit.revision, saved.len()
        );
        if maximal_forgetting != 0 {
            return Err("previous task TEST performance deteriorated".into());
        }
    }
    // Independent holdouts were not consulted during model selection.
    // The frozen backbone never gets parameter updates in this experiment.
    let all_test_correct: u64 = (0..count)
        .map(|task| score_bank(&bank, task, &records[task].test).unwrap().0)
        .sum();
    let all_valid_correct: u64 = (0..count)
        .map(|task| score_bank(&bank, task, &records[task].valid).unwrap().0)
        .sum();
    let all_ood_correct: u64 = (0..count)
        .map(|task| score_bank(&bank, task, &records[task].ood).unwrap().0)
        .sum();
    let total = count as u64 * HELDOUT_BATCHES as u64 * 64;
    println!(
        "TASK_MEMORY_SUMMARY seed={seed} width={width} tasks={count}          valid={all_valid_correct}/{total} test={all_test_correct}/{total}          raw_source_ood={all_ood_correct}/{total} forgotten_test_lanes=0          elapsed_ms={} frozen_backbone=true trained_new_heads={count}          never_before_seen_feature_codes_tested=true",
        start.elapsed().as_millis()
    );
    Ok(())
}

fn main() -> Result<(), String> {
    let tree = WideFiveTree::new(SIX_MILLION_INPUTS).map_err(|e| format!("{e:?}"))?;
    for seed in SEEDS {
        experiment(&tree, seed, TASKS)?;
    }
    println!("CONTINUAL_WALSH_TASK_BANK_MULTISEED_COMPLETE");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_reproducible_task_memory_benchmark() {
        let tree = WideFiveTree::new(125).unwrap();
        experiment(&tree, SEEDS[0], 4).unwrap();
    }

    #[test]
    fn heldout_masks_exclude_exactly_8_of_32_feature_codes() {
        let mut heldout = 0;
        for code in 0..32 {
            let features = std::array::from_fn(|bit| if code & (1 << bit) != 0 { u64::MAX } else { 0 });
            let mask = heldout_mask(features);
            heldout += usize::from(mask == u64::MAX);
        }
        assert_eq!(heldout, 8);
    }
}

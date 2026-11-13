//! Physical learning qualification, not a biological or real-data result.
//! Hidden-teacher labels are privileged; global-only labels are tested
//! separately. Energy is reported ONLY when readable Intel RAPL counters
//! are available; logical word accesses are never called DRAM traffic.
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use v888_booleen::fourier5_hierarchy::GateKind;
use v888_booleen::fourier5_wide::{WideFiveTree, SIX_MILLION_INPUTS};
use v888_booleen::fourier5_wide_learning::WidePlasticCircuit;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut s = self.0;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.0 = s;
        s
    }
}

fn words(count: usize, seed: u64) -> Vec<u64> {
    // SplitMix64 avoids GF(2)-linear inter-position correlations from
    // the xorshift stream reserved for proposal scheduling.
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

fn hash(layer: usize, neuron: usize) -> u64 {
    let mut x = (neuron as u64) ^ ((layer as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn complemented_parity() -> v888_booleen::fourier5::FiveCircle {
    let original = GateKind::Parity.five_circle();
    let mut circles = *original.circles();
    for circle in &mut circles {
        circle.phase_pi = !circle.phase_pi;
    }
    v888_booleen::fourier5::FiveCircle::new(circles, -original.bias())
        .expect("valid complemented Walsh parity")
}

fn matches(predicted: u64, expected: u64) -> u32 {
    64 - (predicted ^ expected).count_ones()
}

fn trace_agreement(left: &[Vec<u64>], right: &[Vec<u64>]) -> (u64, u64) {
    let mut correct = 0_u64;
    let mut total = 0_u64;
    for (a, b) in left.iter().zip(right) {
        for (&x, &y) in a.iter().zip(b) {
            correct += u64::from(matches(x, y));
            total += 64;
        }
    }
    (correct, total)
}

fn peak_rss_kib() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find(|line| line.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

/// Read only top-level package zones to avoid double counting subdomains.
/// sysfs availability/permissions are environment-dependent.
fn rapl() -> Option<Vec<(String, u64, u64)>> {
    let mut zones = Vec::new();
    for item in fs::read_dir("/sys/class/powercap").ok()?.flatten() {
        let name = item.file_name().to_string_lossy().into_owned();
        if !name.starts_with("intel-rapl:") || name.matches(':').count() != 1 {
            continue;
        }
        let path = item.path();
        let value: u64 = fs::read_to_string(path.join("energy_uj"))
            .ok()?
            .trim()
            .parse()
            .ok()?;
        let range: u64 = fs::read_to_string(path.join("max_energy_range_uj"))
            .ok()?
            .trim()
            .parse()
            .ok()?;
        zones.push((name, value, range));
    }
    zones.sort_by(|a, b| a.0.cmp(&b.0));
    if zones.is_empty() {
        None
    } else {
        Some(zones)
    }
}

fn energy_delta(
    first: &Option<Vec<(String, u64, u64)>>,
    last: &Option<Vec<(String, u64, u64)>>,
) -> Option<u64> {
    let (first, last) = (first.as_ref()?, last.as_ref()?);
    if first.len() != last.len() {
        return None;
    }
    let mut total = 0_u64;
    for ((name, start, range), (other, end, other_range)) in first.iter().zip(last) {
        if name != other || range != other_range || *range == 0 {
            return None;
        }
        total += if end >= start {
            end - start
        } else {
            range - start + end
        };
    }
    Some(total)
}

fn main() -> Result<(), String> {
    let width = SIX_MILLION_INPUTS;
    let start = Instant::now();
    let tree = WideFiveTree::new(width).map_err(|e| format!("{e:?}"))?;
    let train = words(width, 0x1234_6250_0001);
    let valid = words(width, 0x9876_6250_0002);
    let options = [
        GateKind::Majority.five_circle(),
        GateKind::Parity.five_circle(),
        GateKind::And.five_circle(),
        GateKind::Or.five_circle(),
        complemented_parity(),
    ];

    // Synthetic teacher has INDEPENDENT per-neuron Fourier functions.
    // Labels of EVERY hidden node are intentionally supplied for the
    // privileged teacher-tapped demonstration; this is not end-to-end.
    let mut teacher =
        WidePlasticCircuit::new(&tree, &train).map_err(|e| format!("teacher: {e:?}"))?;
    teacher.install_models(|layer, neuron| {
        let bucket = hash(layer, neuron) % 10;
        // Use independent parity/complement-parity at every higher
        // node. Repeated majority can amplify bias until the single
        // target output is almost constant (a trivial learning task).
        let choice = if layer == 0 {
            match bucket {
                0 => 2,
                1 => 3,
                2..=5 => 0,
                _ => 1,
            }
        } else if bucket % 2 == 0 {
            1
        } else {
            4
        };
        options[choice]
    });
    let targets = teacher
        .evaluate_trace(&train)
        .map_err(|e| format!("{e:?}"))?;
    let valid_targets = teacher
        .evaluate_trace(&valid)
        .map_err(|e| format!("{e:?}"))?;
    drop(teacher);

    let mut learner =
        WidePlasticCircuit::new(&tree, &train).map_err(|e| format!("learner: {e:?}"))?;
    let prior = learner
        .evaluate_trace(&valid)
        .map_err(|e| format!("{e:?}"))?;
    let prior_agreement = trace_agreement(&prior, &valid_targets);
    let local_start = Instant::now();
    let local = learner
        .fit_local_teacher(&targets, &options)
        .map_err(|e| format!("fit: {e:?}"))?;
    let fit_ms = local_start.elapsed().as_millis();
    let trained = learner
        .evaluate_trace(&train)
        .map_err(|e| format!("{e:?}"))?;
    let heldout = learner
        .evaluate_trace(&valid)
        .map_err(|e| format!("{e:?}"))?;
    let heldout_agreement = trace_agreement(&heldout, &valid_targets);
    let train_agreement = trace_agreement(&trained, &targets);
    if local.neurons_fitted != tree.logical_neurons()
        || train_agreement.0 != local.train_correct_lanes
        || train_agreement.1 != local.train_total_lanes
    {
        return Err("trainer trace/counter mismatch".into());
    }
    println!(
        "LARGE_LOCAL_FIT input_bits={width} nodes={} models_changed={} fit_gate_evals={} fit_ms={fit_ms} train_correct={}/{} valid_before={}/{} valid_after={}/{} train_root_correct={} valid_root_correct={} model_bytes={} truth_bytes={} cache_bytes={} peak_rss_kib={}",
        local.neurons_fitted, local.models_changed, local.gate_evaluations,
        train_agreement.0, train_agreement.1,
        prior_agreement.0, prior_agreement.1,
        heldout_agreement.0, heldout_agreement.1,
        matches(trained.last().unwrap()[0], targets.last().unwrap()[0]),
        matches(heldout.last().unwrap()[0], valid_targets.last().unwrap()[0]),
        learner.model_bytes(), learner.truth_bytes(), learner.cache_bytes(),
        peak_rss_kib().map_or_else(|| "unavailable".to_owned(), |x| x.to_string())
    );

    // A second, strictly output-only learner must never read hidden target
    // labels for training. VALID is inspected only after the trial.
    let mut global =
        WidePlasticCircuit::new(&tree, &train).map_err(|e| format!("global: {e:?}"))?;
    let train_label = targets.last().unwrap()[0];
    let valid_label = valid_targets.last().unwrap()[0];
    let train_positive = train_label.count_ones();
    let valid_positive = valid_label.count_ones();
    println!(
        "LARGE_LABEL_BALANCE train_positive={train_positive}/64 valid_positive={valid_positive}/64"
    );
    if !(16..=48).contains(&train_positive) || !(16..=48).contains(&valid_positive) {
        return Err("degenerate synthetic label balance; no generalization claim".into());
    }
    let global_valid_before = global
        .evaluate_trace(&valid)
        .map_err(|e| format!("{e:?}"))?
        .last()
        .unwrap()[0];
    let initial = matches(global.output(), train_label);
    let mut best = initial;
    let mut rng = Rng(0x2580_fa11_abba_2211);
    let global_start = Instant::now();
    let trials = 4096;
    for _ in 0..trials {
        let layer = (rng.next() as usize) % tree.depth();
        let neuron = (rng.next() as usize) % tree.level_widths()[layer];
        let model = options[(rng.next() as usize) % options.len()];
        let proposal = global
            .propose(layer, neuron, model)
            .map_err(|e| format!("propose: {e:?}"))?;
        let score = matches(proposal.output, train_label);
        if score > best {
            global
                .accept(proposal)
                .map_err(|e| format!("commit: {e:?}"))?;
            best = score;
        }
    }
    let global_ms = global_start.elapsed().as_millis();
    let global_valid_after = global
        .evaluate_trace(&valid)
        .map_err(|e| format!("{e:?}"))?
        .last()
        .unwrap()[0];
    if best != matches(global.output(), train_label) {
        return Err("persistent online objective mismatch".into());
    }
    println!(
        "LARGE_GLOBAL_ONLY trials={trials} commits={} train_before={initial}/64 train_after={best}/64 valid_before={}/64 valid_after={}/64 elapsed_ms={global_ms}",
        global.commits(),
        matches(global_valid_before, valid_label),
        matches(global_valid_after, valid_label)
    );
    drop(global);

    // Paired, exact full-versus-incremental re-evaluation AFTER training.
    // Cached proposals are rejected: both arms see the same frozen model.
    // The cache's initial construction and teacher training are EXCLUDED
    // from these hot-state timings and energy samples.
    let edits: Vec<_> = (0..64)
        .map(|i| {
            let neuron = (i * 79_919 + 11) % tree.level_widths()[0];
            let model = options[(i + 1) % options.len()];
            (0, neuron, model)
        })
        .collect();
    let energy_full_start = rapl();
    let full_start = Instant::now();
    let mut full_outputs = Vec::with_capacity(edits.len());
    for &(layer, neuron, model) in &edits {
        let output = learner
            .evaluate_full_edit(layer, neuron, model)
            .map_err(|e| format!("{e:?}"))?;
        full_outputs.push(black_box(output));
    }
    let full_us = full_start.elapsed().as_micros();
    let energy_full_end = rapl();
    let energy_cached_start = rapl();
    let cached_start = Instant::now();
    let mut logical_reads = 0_usize;
    let mut changed_words = 0_usize;
    let mut gate_evals = 0_usize;
    let mut cached_outputs = Vec::with_capacity(edits.len());
    for &(layer, neuron, model) in &edits {
        let proposal = learner
            .propose(layer, neuron, model)
            .map_err(|e| format!("{e:?}"))?;
        logical_reads += proposal.child_word_reads;
        changed_words += proposal.changed_cache_words;
        gate_evals += proposal.gate_evaluations;
        cached_outputs.push(black_box(proposal.output));
    }
    let cached_us = cached_start.elapsed().as_micros();
    let energy_cached_end = rapl();
    if full_outputs != cached_outputs {
        return Err("large trained circuit: full/delta mismatch".into());
    }
    println!(
        "LARGE_MEMORY_ACCESS_ESTIMATE trials={} full_logical_child_words={} cached_logical_child_words={} cached_changed_words={} full_gate_evals={} cached_gate_evals={} full_us={full_us} cached_us={cached_us} peak_rss_kib={}",
        edits.len(), tree.real_contacts() * edits.len(), logical_reads,
        changed_words, tree.logical_neurons() * edits.len(), gate_evals,
        peak_rss_kib().map_or_else(|| "unavailable".to_owned(), |x| x.to_string())
    );
    match (
        energy_delta(&energy_full_start, &energy_full_end),
        energy_delta(&energy_cached_start, &energy_cached_end),
    ) {
        (Some(full), Some(cached)) if full > 0 && cached > 0 => println!(
            "LARGE_RAPL_PACKAGE_ENERGY status=measured_uncontrolled full_uj={full} cached_uj={cached} note=package_shared_other_processes_and_measurement_granularity"
        ),
        _ => println!("LARGE_RAPL_PACKAGE_ENERGY status=unavailable reason=counter_inaccessible_or_resolution"),
    }
    println!(
        "LARGE_LEARNING_6250000_DONE elapsed_ms={} note=teacher_tapped_and_output_only_are_distinct",
        start.elapsed().as_millis()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_inverted_parity_is_an_exact_complement() {
        let normal = GateKind::Parity.five_circle();
        let inverse = complemented_parity();
        for word in 0..32 {
            assert_eq!(
                inverse.predict(word).unwrap(),
                !normal.predict(word).unwrap()
            );
        }
    }
}

//! Bounded, deterministic sequential cache benchmark. Synthetic DAG only.
//! Usage: cargo run --release --bin sequential-cache-bench -- 10000 1000
//! Run exclusively through the bounded RemoteOps qualification workflow.
use std::time::Instant;
use v888_booleen::fourier5_exact_sequential_cache::{ExactSequentialCache, Gate};

fn proc_status_kib(field: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix(field)
            .and_then(|value| value.split_whitespace().next())
            .and_then(|value| value.parse().ok())
    })
}

fn throughput(steps: usize, nanoseconds: u128) -> f64 {
    if nanoseconds == 0 { return 0.0; }
    (steps as f64) * 1_000_000_000.0 / (nanoseconds as f64)
}

// Independent per-workload timing counters; zero-duration samples are retained.
#[derive(Clone, Default)]
struct WorkloadTiming {
    samples: usize,
    full_ns: u128,
    incremental_ns: u128,
    evaluated: usize,
    // Exact paired samples: preserve tails rather than inferring from aggregate means.
    pairs_ns: Vec<(u128, u128)>,
}
impl WorkloadTiming {
    fn record(&mut self, full_ns: u128, incremental_ns: u128, evaluated: usize) {
        self.samples += 1;
        self.full_ns += full_ns;
        self.incremental_ns += incremental_ns;
        self.evaluated += evaluated;
        self.pairs_ns.push((full_ns, incremental_ns));
    }
    // Nearest-rank p95: no interpolation, saturating arithmetic, conservative tie.
    fn tail_preferred_engine(&self) -> &'static str {
        if self.pairs_ns.len() < 20 { return "Full"; }
        let mut full = self.pairs_ns.iter().map(|p| p.0).collect::<Vec<_>>();
        let mut inc = self.pairs_ns.iter().map(|p| p.1).collect::<Vec<_>>();
        full.sort_unstable();
        inc.sort_unstable();
        let rank = (full.len().saturating_mul(95).saturating_add(99) / 100).saturating_sub(1);
        if inc[rank] < full[rank] { "Incremental" } else { "Full" }
    }
    // TRAIN-only conservative calibration: require both lower total cost and
    // lower nearest-rank p95 on paired samples. A tie or insufficient evidence
    // always selects Full. Never fit this rule on VALID/TEST/OOD.
    fn calibrated_engine(&self) -> &'static str {
        if self.pairs_ns.len() < 20 || self.incremental_ns >= self.full_ns {
            return "Full";
        }
        self.tail_preferred_engine()
    }
    fn preferred_engine(&self) -> &'static str {
        if self.samples == 0 || self.incremental_ns >= self.full_ns {
            "Full"
        } else {
            "Incremental"
        }
    }
}

// Replay the frozen TRAIN choice against paired held-out measurements.
// This is counterfactual accounting, not an execution-time speedup claim.
fn frozen_policy_cost(policy: &[&str; 3], buckets: &[WorkloadTiming; 3]) -> (u128, u128, u128) {
    let mut chosen = 0u128;
    let mut full = 0u128;
    let mut oracle = 0u128;
    for (choice, bucket) in policy.iter().zip(buckets.iter()) {
        chosen += if *choice == "Incremental" { bucket.incremental_ns } else { bucket.full_ns };
        full += bucket.full_ns;
        oracle += bucket.full_ns.min(bucket.incremental_ns);
    }
    (chosen, full, oracle)
}

// Last-write-wins semantics without allocating/copying the entire input vector.
fn batch_is_noop(inputs: &[bool], changes: &[(usize, bool)]) -> bool {
    changes.iter().enumerate().all(|(position, &(index, _))| {
        let final_value = changes[position + 1..].iter()
            .rev().find(|(later, _)| *later == index)
            .map(|(_, value)| *value).unwrap_or(changes[position].1);
        inputs[index] == final_value
    })
}

fn circuit(width: usize) -> Vec<Gate> {
    let mut gates = (0..width).map(Gate::Input).collect::<Vec<_>>();
    for i in 0..width {
        let left = i;
        let right = if i == 0 { width - 1 } else { i - 1 };
        gates.push(Gate::Xor(left, right));
    }
    for i in 0..width {
        gates.push(Gate::And(width + i, (i + 1) % width));
    }
    gates
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let width = args.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(1024);
    let steps = args.get(2).and_then(|s| s.parse::<usize>().ok()).unwrap_or(256);
    if !(2..=100_000).contains(&width) || !(1..=10_000).contains(&steps) {
        eprintln!("width must be 2..=100000; steps 1..=10000");
        std::process::exit(2);
    }
    let gates = circuit(width);
    let mut full_ns = 0u128;
    let mut incremental_ns = 0u128;
    let mut total_evaluated = 0usize;
    let mut total_noop_batches = 0usize;
    // Four disjoint seed domains; freeze per-class TRAIN choices for evaluation.
    let mut frozen_train_policy = ["Full"; 3];
    for (split, seed) in [("TRAIN", 11u64), ("VALID", 23), ("TEST", 37), ("OOD", 53)] {
        // Each split starts from the same baseline; no state crosses split boundaries.
        let mut full = ExactSequentialCache::new(gates.clone(), vec![false; width]).unwrap();
        let mut incremental = full.clone();
        let mut rng = seed;
        let mut split_full = 0u128;
        let mut split_inc = 0u128;
        let mut split_noop_batches = 0usize;
        let mut workload = [WorkloadTiming::default(); 3];
        let rss_start_kib = proc_status_kib("VmRSS:");
        let hwm_start_kib = proc_status_kib("VmHWM:");
        for step in 0..steps {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let index = (rng as usize) % width;
            // Nonhomogeneous bursts, including repeated writes and no-ops.
            let workload_class = if step % 17 == 0 { 2 } else if step % 7 == 0 { 1 } else { 0 };
            let changes = if step % 17 == 0 {
                vec![(index, true), (index, false), ((index + 1) % width, true)]
            } else if step % 7 == 0 {
                vec![(index, false)]
            } else {
                vec![(index, (rng & 1) != 0)]
            };
            let unchanged = batch_is_noop(full.inputs(), &changes);
            if unchanged { split_noop_batches += 1; }
            // Alternate execution order to avoid a systematic first-run advantage.
            let (elapsed_full, elapsed_inc, evaluated) = if step % 2 == 0 {
                let start = Instant::now();
                let mut next_inputs = full.inputs().to_vec();
                for &(i, value) in &changes { next_inputs[i] = value; }
                full = ExactSequentialCache::new(gates.clone(), next_inputs).unwrap();
                let full_ns = start.elapsed().as_nanos();
                let start = Instant::now();
                let evaluated = incremental.update_inputs_atomic(&changes).unwrap();
                (full_ns, start.elapsed().as_nanos(), evaluated)
            } else {
                let start = Instant::now();
                let evaluated = incremental.update_inputs_atomic(&changes).unwrap();
                let inc_ns = start.elapsed().as_nanos();
                let start = Instant::now();
                let mut next_inputs = full.inputs().to_vec();
                for &(i, value) in &changes { next_inputs[i] = value; }
                full = ExactSequentialCache::new(gates.clone(), next_inputs).unwrap();
                (start.elapsed().as_nanos(), inc_ns, evaluated)
            };
            // Full oracle uses the final input state and a complete rebuild.
            assert_eq!(full.inputs(), incremental.inputs(), "split={split} step={step}");
            assert_eq!(full.values(), incremental.values(), "split={split} step={step}");
            split_full += elapsed_full;
            split_inc += elapsed_inc;
            total_evaluated += evaluated;
            workload[workload_class].record(elapsed_full, elapsed_inc, evaluated);
        }
        if split == "TRAIN" {
            for (index, timing) in workload.iter().enumerate() {
                frozen_train_policy[index] = timing.calibrated_engine();
            }
        }
        for (class_index, (class, timing)) in ["single", "forced_false", "burst"].iter().zip(workload.iter()).enumerate() {
            println!(
                "split={split} workload={class} samples={} full_ns={} incremental_ns={} evaluated={} preferred={} p95_preferred={} train_frozen={} full_updates_per_sec={:.2} incremental_updates_per_sec={:.2}",
                timing.samples, timing.full_ns, timing.incremental_ns, timing.evaluated,
                timing.preferred_engine(), timing.tail_preferred_engine(), frozen_train_policy[class_index],
                throughput(timing.samples, timing.full_ns),
                throughput(timing.samples, timing.incremental_ns)
            );
        }
        let (chosen_ns, baseline_full_ns, hindsight_oracle_ns) = frozen_policy_cost(&frozen_train_policy, &workload);
        println!("split={split} frozen_policy_ns={chosen_ns} full_baseline_ns={baseline_full_ns} hindsight_oracle_ns={hindsight_oracle_ns} regret_vs_full_ns={} regret_vs_oracle_ns={}",
            chosen_ns as i128 - baseline_full_ns as i128,
            chosen_ns.saturating_sub(hindsight_oracle_ns));
        total_noop_batches += split_noop_batches;
        full_ns += split_full;
        incremental_ns += split_inc;
        println!(
            "split={split} steps={steps} noop_batches={split_noop_batches} full_ns={split_full} incremental_ns={split_inc} full_updates_per_sec={:.2} incremental_updates_per_sec={:.2} rss_start_kib={rss_start_kib:?} rss_end_kib={:?} vmhwm_start_kib={hwm_start_kib:?} vmhwm_end_kib={:?}",
            throughput(steps, split_full), throughput(steps, split_inc),
            proc_status_kib("VmRSS:"),
            proc_status_kib("VmHWM:")
        );
    }
    println!(
        "width={width} gates={} interventions={} noop_batches={total_noop_batches} full_ns={full_ns} incremental_ns={incremental_ns} evaluated_gates={total_evaluated} process_vmhwm_kib={:?}",
        gates.len(),
        steps * 4,
        proc_status_kib("VmHWM:")
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn workload_policy_is_conservative_and_never_uses_empty_samples() {
        let mut timing = WorkloadTiming::default();
        assert_eq!(timing.preferred_engine(), "Full");
        timing.record(100, 120, 3);
        assert_eq!(timing.preferred_engine(), "Full");
        timing.record(100, 20, 2);
        assert_eq!(timing.samples, 2);
        assert_eq!(timing.evaluated, 5);
        assert_eq!(timing.preferred_engine(), "Incremental");
    }
    #[test]
    fn workload_timing_partition_accounting_is_exact() {
        let mut buckets = [WorkloadTiming::default(); 3];
        for (index, full, incremental) in [(0, 10, 5), (2, 20, 25), (1, 30, 10), (2, 40, 35)] {
            buckets[index].record(full, incremental, 1);
        }
        assert_eq!(buckets.iter().map(|b| b.samples).sum::<usize>(), 4);
        assert_eq!(buckets.iter().map(|b| b.full_ns).sum::<u128>(), 100);
        assert_eq!(buckets.iter().map(|b| b.incremental_ns).sum::<u128>(), 75);
    }
    #[test]
    fn p95_policy_rejects_small_samples_and_tail_regression() {
        let mut t = WorkloadTiming::default();
        for _ in 0..19 { t.record(100, 1, 1); }
        assert_eq!(t.tail_preferred_engine(), "Full");
        t.record(100, 1000, 1);
        // At n=20, nearest-rank p95 is index 18; one outlier is not p95.
        assert_eq!(t.tail_preferred_engine(), "Incremental");
        t.record(100, 1000, 1);
        assert_eq!(t.tail_preferred_engine(), "Full");
    }
    #[test]
    fn p95_policy_handles_ties_and_zero_cost() {
        let mut t = WorkloadTiming::default();
        for _ in 0..20 { t.record(0, 0, 0); }
        assert_eq!(t.tail_preferred_engine(), "Full");
    }
    #[test]
    fn calibrated_policy_requires_mean_and_tail_improvement() {
        let mut t = WorkloadTiming::default();
        for _ in 0..19 { t.record(100, 1, 0); }
        assert_eq!(t.calibrated_engine(), "Full");
        t.record(100, 1, 0);
        assert_eq!(t.calibrated_engine(), "Incremental");
        // Total incremental cost remains lower, but the p95 now regresses.
        t.record(100, 1_000, 0);
        t.record(100, 1_000, 0);
        assert_eq!(t.calibrated_engine(), "Full");
        let mut tied = WorkloadTiming::default();
        for _ in 0..20 { tied.record(100, 100, 0); }
        assert_eq!(tied.calibrated_engine(), "Full");
    }
    #[test]
    fn frozen_policy_reports_held_out_regressions_without_refitting() {
        let mut buckets = std::array::from_fn(|_| WorkloadTiming::default());
        buckets[0].record(100, 200, 1);
        buckets[1].record(100, 20, 1);
        buckets[2].record(100, 80, 1);
        let frozen = ["Incremental", "Full", "Incremental"];
        assert_eq!(frozen_policy_cost(&frozen, &buckets), (380, 300, 200));
        assert_eq!(frozen, ["Incremental", "Full", "Incremental"]);
    }
    #[test]
    fn last_write_wins_noop_matches_independent_state_replay() {
        let state = [false, true, false];
        for changes in [vec![(0, true), (0, false)], vec![(1, false), (1, true)],
                        vec![(0, true), (2, true)], vec![(1, false), (1, false)]] {
            let mut reference = state;
            for &(i, value) in &changes { reference[i] = value; }
            assert_eq!(batch_is_noop(&state, &changes), reference == state);
        }
        assert!(batch_is_noop(&state, &[]));
    }
    #[test]
    fn throughput_is_measured_in_updates_per_second() {
        assert_eq!(throughput(10, 2_000_000_000), 5.0);
        assert_eq!(throughput(10, 0), 0.0);
    }
    #[test]
    fn proc_status_missing_field_is_none() {
        assert_eq!(proc_status_kib("NonexistentV888Field:"), None);
    }
    #[test]
    fn synthetic_dag_rebuild_matches_atomic_updates() {
        let gates = circuit(17);
        let mut inc = ExactSequentialCache::new(gates.clone(), vec![false; 17]).unwrap();
        let mut state = vec![false; 17];
        for step in 0..70 {
            let edits = [(step % 17, true), (step % 17, step % 3 == 0), ((step + 5) % 17, step % 2 == 0)];
            for &(i, value) in &edits { state[i] = value; }
            inc.update_inputs_atomic(&edits).unwrap();
            let full = ExactSequentialCache::new(gates.clone(), state.clone()).unwrap();
            assert_eq!(inc.inputs(), full.inputs());
            assert_eq!(inc.values(), full.values());
        }
    }
}

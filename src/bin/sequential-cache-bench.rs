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
#[derive(Clone, Copy, Default)]
struct WorkloadTiming {
    samples: usize,
    full_ns: u128,
    incremental_ns: u128,
    evaluated: usize,
}
impl WorkloadTiming {
    fn record(&mut self, full_ns: u128, incremental_ns: u128, evaluated: usize) {
        self.samples += 1;
        self.full_ns += full_ns;
        self.incremental_ns += incremental_ns;
        self.evaluated += evaluated;
    }
    fn preferred_engine(&self) -> &'static str {
        if self.samples == 0 || self.incremental_ns >= self.full_ns {
            "Full"
        } else {
            "Incremental"
        }
    }
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
    // Four disjoint seed domains; no tuning or adaptation on VALID/TEST/OOD.
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
            let unchanged = changes.iter().fold(full.inputs().to_vec(), |mut state, &(i, value)| { state[i] = value; state }) == full.inputs();
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
        for (class, timing) in ["single", "forced_false", "burst"].iter().zip(workload.iter()) {
            println!(
                "split={split} workload={class} samples={} full_ns={} incremental_ns={} evaluated={} preferred={} full_updates_per_sec={:.2} incremental_updates_per_sec={:.2}",
                timing.samples, timing.full_ns, timing.incremental_ns, timing.evaluated,
                timing.preferred_engine(),
                throughput(timing.samples, timing.full_ns),
                throughput(timing.samples, timing.incremental_ns)
            );
        }
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

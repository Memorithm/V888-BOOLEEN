//! Reproducible, process-isolated synthetic throughput probe for three V888
//! event schedulers. Does NOT measure joules or claim biological validity.
//!
//! Example:
//! cargo run --release --bin v888-event-bench -- --mode=serial --shape=disjoint --sources=256 --repetitions=8 --workers=4
//! For physical counters, use reviewed Memorithm/RemoteOps host-side RAPL.

use std::env;
use std::fs;
use std::hint::black_box;
use std::time::{Duration, Instant};

use v888_booleen::connectome::Connectome;
use v888_booleen::event_runtime::{
    ContactCacheReport, DisjointAdmissionProfile, EventRuntime, EventWork,
};
use v888_booleen::BooleanState;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Serial,
    Frontier,
    Disjoint,
    Adaptive,
}

impl Mode {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "serial" => Some(Self::Serial),
            "frontier" => Some(Self::Frontier),
            "disjoint" => Some(Self::Disjoint),
            "adaptive" => Some(Self::Adaptive),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::Frontier => "frontier",
            Self::Disjoint => "disjoint",
            Self::Adaptive => "adaptive",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Disjoint,
    Convergent,
    WideDisjoint,
    HeavyDisjoint,
}

impl Shape {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "disjoint" => Some(Self::Disjoint),
            "convergent" => Some(Self::Convergent),
            "wide-disjoint" => Some(Self::WideDisjoint),
            "heavy-disjoint" => Some(Self::HeavyDisjoint),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Disjoint => "disjoint",
            Self::Convergent => "convergent",
            Self::WideDisjoint => "wide-disjoint",
            Self::HeavyDisjoint => "heavy-disjoint",
        }
    }

    fn threshold(self) -> u128 {
        match self {
            Self::Disjoint => 1,
            Self::Convergent => 2,
            Self::WideDisjoint | Self::HeavyDisjoint => 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Config {
    mode: Mode,
    shape: Shape,
    sources: usize,
    repetitions: usize,
    workers: usize,
    cache: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mode: Mode::Serial,
            shape: Shape::Disjoint,
            sources: 64,
            repetitions: 3,
            workers: 4,
            cache: false,
        }
    }
}

impl Config {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut config = Self::default();
        for argument in arguments {
            if argument == "--help" {
                return Err(String::from(
                    "usage: --mode=serial|frontier|disjoint|adaptive --shape=disjoint|convergent|wide-disjoint|heavy-disjoint \
                     --sources=8..4096 --repetitions=1..64 --workers=1..4 --cache=on|off",
                ));
            }
            let (name, value) = argument
                .strip_prefix("--")
                .and_then(|text| text.split_once('='))
                .ok_or_else(|| format!("expected --name=value, got {argument}"))?;
            match name {
                "mode" => {
                    config.mode =
                        Mode::parse(value).ok_or_else(|| format!("unsupported mode {value}"))?;
                }
                "shape" => {
                    config.shape =
                        Shape::parse(value).ok_or_else(|| format!("unsupported shape {value}"))?;
                }
                "sources" => {
                    config.sources = value
                        .parse()
                        .map_err(|_| format!("invalid source count {value}"))?;
                }
                "repetitions" => {
                    config.repetitions = value
                        .parse()
                        .map_err(|_| format!("invalid repetitions {value}"))?;
                }
                "workers" => {
                    config.workers = value
                        .parse()
                        .map_err(|_| format!("invalid workers {value}"))?;
                }
                "cache" => {
                    config.cache = match value {
                        "on" => true,
                        "off" => false,
                        _ => return Err(format!("invalid cache mode {value}")),
                    };
                }
                _ => return Err(format!("unsupported option {name}")),
            }
        }
        if !(8..=4096).contains(&config.sources) {
            return Err(String::from("sources must be in 8..=4096"));
        }
        if !(1..=64).contains(&config.repetitions) {
            return Err(String::from("repetitions must be in 1..=64"));
        }
        if !(1..=4).contains(&config.workers) {
            return Err(String::from("workers must be in 1..=4"));
        }
        Ok(config)
    }
}

fn graph(sources: usize, shape: Shape) -> Connectome {
    let count = sources * 2;
    // wide-disjoint retains one unique target per source, but models
    // 64 physical contact rows with identical presynaptic/postsynaptic IDs.
    // It measures CSR traversal work, NOT 64 independent output neurons.
    let contacts_per_source = match shape {
        Shape::WideDisjoint => 64,
        Shape::HeavyDisjoint => 512,
        _ => 1,
    };
    let total_contacts = sources * contacts_per_source;
    let targets: Vec<u32> = (0..sources)
        .flat_map(|source| {
            let local = match shape {
                Shape::Disjoint | Shape::WideDisjoint | Shape::HeavyDisjoint => source,
                Shape::Convergent => source % 4,
            };
            std::iter::repeat_n((sources + local) as u32, contacts_per_source)
        })
        .collect();
    let graph = Connectome {
        ids: (1..=count as u64).collect(),
        offsets: (0..=sources)
            .map(|index| (index * contacts_per_source) as u64)
            .chain(std::iter::repeat_n(total_contacts as u64, sources))
            .collect(),
        targets,
        counts: vec![1; total_contacts],
    };
    graph.validate().expect("generated graph must validate");
    graph
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Outcome {
    work: EventWork,
    state: BooleanState,
    incoming: Vec<u128>,
    pending: usize,
    last_time: Option<u64>,
    complete_audit: bool,
}

#[derive(Clone, Debug)]
struct Sample {
    elapsed: Duration,
    outcome: Outcome,
    /// Number of actual writer threads used, for the disjoint mode.
    writers: usize,
    cache_report: ContactCacheReport,
    cache_build_ns: u128,
}

fn execute(graph: &Connectome, shape: Shape, mode: Mode, workers: usize, cache: bool) -> Sample {
    execute_with_profile(
        graph,
        shape,
        mode,
        workers,
        cache,
        &mut DisjointAdmissionProfile::default(),
    )
}

fn execute_with_profile(
    graph: &Connectome,
    shape: Shape,
    mode: Mode,
    workers: usize,
    cache: bool,
    profile: &mut DisjointAdmissionProfile,
) -> Sample {
    let sources = graph.ids.len() / 2;
    let mut runtime =
        EventRuntime::new(graph, shape.threshold(), 4 * sources + 16).expect("valid bounded graph");
    // Cache creation is outside the timed event loop. Report its cost
    // separately so amortization is not hidden in any throughput claim.
    let cache_started = Instant::now();
    let cache_report = if cache {
        runtime
            .prepare_contact_cache(sources)
            .expect("bounded synthetic cache creation")
    } else {
        ContactCacheReport::default()
    };
    let cache_build_ns = cache_started.elapsed().as_nanos();
    for &id in &graph.ids[..sources] {
        runtime.schedule_drive(0, id, true).expect("first drive");
        runtime.schedule_drive(3, id, false).expect("second drive");
    }

    // Time only the event run, never construction, drive injection, audit or
    // structural parity verification. The outer host RAPL window can include
    // other process work and is documented separately.
    let started = Instant::now();
    let (work, writers) = match mode {
        Mode::Serial => (runtime.run(usize::MAX).expect("serial run"), 0),
        Mode::Frontier => (
            runtime
                .run_parallel_frontiers(usize::MAX, workers)
                .expect("frontier run")
                .work,
            0,
        ),
        Mode::Disjoint => {
            let report = runtime
                .run_disjoint_parallel_commits(usize::MAX, workers)
                .expect("disjoint run");
            (report.work, report.target_writer_threads)
        }
        Mode::Adaptive => {
            let report = runtime
                .run_disjoint_parallel_commits_with_profile(usize::MAX, workers, profile)
                .expect("adaptive run");
            (report.work, report.target_writer_threads)
        }
    };
    let elapsed = started.elapsed();
    let outcome = Outcome {
        work,
        state: runtime.state().clone(),
        incoming: graph
            .ids
            .iter()
            .map(|&id| runtime.incoming_for(id).expect("valid neuron ID"))
            .collect(),
        pending: runtime.pending_count(),
        last_time: runtime.last_processed_time(),
        complete_audit: runtime.audit_incoming().expect("no counter overflow"),
    };
    black_box(&outcome);
    Sample {
        elapsed,
        outcome,
        writers,
        cache_report,
        cache_build_ns,
    }
}

fn proc_status_kib(field: &str) -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with(field))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn proc_cpu_ticks() -> Option<u64> {
    // Linux /proc/self/stat fields 14 and 15 are aggregate user/system
    // CPU ticks. Process names may contain spaces or parentheses.
    let status = fs::read_to_string("/proc/self/stat").ok()?;
    let suffix = status.rsplit_once(") ")?.1;
    let parts: Vec<&str> = suffix.split_whitespace().collect();
    let user: u64 = parts.get(11)?.parse().ok()?;
    let system: u64 = parts.get(12)?.parse().ok()?;
    user.checked_add(system)
}

fn report(config: Config) {
    let graph = graph(config.sources, config.shape);
    let reference = execute(&graph, config.shape, Mode::Serial, 1, false).outcome;
    assert!(reference.complete_audit);
    assert_eq!(reference.pending, 0);
    // Warm up exactly the chosen execution mode; exclude warmup from timing.
    let mut profile = DisjointAdmissionProfile::default();
    let warmup = execute_with_profile(
        &graph,
        config.shape,
        config.mode,
        config.workers,
        config.cache,
        &mut profile,
    );
    assert_eq!(warmup.outcome, reference);
    let ticks_before = proc_cpu_ticks();
    let mut durations = Vec::new();
    let mut writers = 0_usize;
    let mut cache_build_total_ns = 0_u128;
    let mut retained_pairs = 0_usize;
    let mut covered_rows = 0_usize;
    for _ in 0..config.repetitions {
        let sample = execute_with_profile(
            &graph,
            config.shape,
            config.mode,
            config.workers,
            config.cache,
            &mut profile,
        );
        assert_eq!(sample.outcome, reference, "scheduler parity failure");
        durations.push(sample.elapsed.as_nanos());
        writers = writers.saturating_add(sample.writers);
        cache_build_total_ns += sample.cache_build_ns;
        retained_pairs = sample.cache_report.cached_pairs;
        covered_rows = sample.cache_report.covered_contact_rows;
    }
    let ticks_after = proc_cpu_ticks();
    let (serial_samples, parallel_samples) = profile.observation_counts();
    durations.sort_unstable();
    let total_ns: u128 = durations.iter().sum();
    let peak_rss = proc_status_kib("VmHWM:")
        .map_or_else(|| "unavailable".to_string(), |value| value.to_string());
    let current_rss = proc_status_kib("VmRSS:")
        .map_or_else(|| "unavailable".to_string(), |value| value.to_string());
    let ticks = ticks_before
        .zip(ticks_after)
        .and_then(|(before, after)| after.checked_sub(before))
        .map_or_else(|| "unavailable".to_string(), |value| value.to_string());
    println!(
        "V888_EVENT_BENCH mode={} shape={} sources={} repetitions={} workers={} cache={} \
         scope=event_run_only min_ns={} median_ns={} max_ns={} total_ns={} \
         events_per_run={} transitions_per_run={} contacts_per_run={} \
         writer_threads_total={} calibration_serial_groups={} calibration_parallel_groups={} cache_pairs={} cache_covered_rows={} cache_build_total_ns={} peak_rss_kib={} current_rss_kib={} \
         cpu_ticks_process={} exact_parity=true host_energy_attribution=false",
        config.mode.label(),
        config.shape.label(),
        config.sources,
        config.repetitions,
        config.workers,
        if config.cache { "on" } else { "off" },
        durations[0],
        durations[(durations.len() - 1) / 2],
        durations[durations.len() - 1],
        total_ns,
        reference.work.events_consumed,
        reference.work.state_transitions,
        reference.work.contacts_examined,
        writers,
        serial_samples,
        parallel_samples,
        retained_pairs,
        covered_rows,
        cache_build_total_ns,
        peak_rss,
        current_rss,
        ticks,
    );
}

fn main() {
    match Config::parse(env::args().skip(1)) {
        Ok(config) => report(config),
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_shapes_have_exact_parity_for_all_execution_modes() {
        for shape in [
            Shape::Disjoint,
            Shape::Convergent,
            Shape::WideDisjoint,
            Shape::HeavyDisjoint,
        ] {
            let graph = graph(16, shape);
            let baseline = execute(&graph, shape, Mode::Serial, 1, false).outcome;
            for mode in [
                Mode::Serial,
                Mode::Frontier,
                Mode::Disjoint,
                Mode::Adaptive,
            ] {
                for cache in [false, true] {
                    let sample = execute(&graph, shape, mode, 4, cache);
                    assert_eq!(sample.outcome, baseline);
                    assert!(sample.outcome.complete_audit);
                    assert_eq!(sample.outcome.pending, 0);
                    assert_eq!(sample.outcome.state.active_count(), 0);
                    if cache && matches!(shape, Shape::WideDisjoint | Shape::HeavyDisjoint) {
                        assert_eq!(sample.cache_report.cached_sources, 16);
                        assert_eq!(sample.cache_report.cached_pairs, 16);
                    } else {
                        assert_eq!(sample.cache_report.cached_pairs, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn disjoint_and_convergent_graphs_have_distinct_fanout_patterns() {
        let a = graph(32, Shape::Disjoint);
        let b = graph(32, Shape::Convergent);
        let c = graph(32, Shape::WideDisjoint);
        let d = graph(32, Shape::HeavyDisjoint);
        assert_eq!(a.ids, b.ids);
        assert_eq!(a.targets.len(), 32);
        assert_eq!(c.targets.len(), 32 * 64);
        assert_eq!(c.offsets[1], 64);
        assert_eq!(d.offsets[1], 512);
        assert_eq!(d.targets.len(), 32 * 512);
        assert_eq!(
            a.targets
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            32
        );
        assert_eq!(
            b.targets
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            4
        );
    }

    #[test]
    fn cli_refuses_unbounded_or_invalid_measurements() {
        assert!(Config::parse(["--sources=4097".to_string()]).is_err());
        assert!(Config::parse(["--workers=0".to_string()]).is_err());
        assert!(Config::parse(["--repetitions=65".to_string()]).is_err());
        assert!(Config::parse(["--mode=unknown".to_string()]).is_err());
        assert!(Config::parse(["--typo=12".to_string()]).is_err());
        assert!(Config::parse(["--cache=invalid".to_string()]).is_err());
        let config = Config::parse([
            "--mode=disjoint".to_string(),
            "--shape=convergent".to_string(),
        ])
        .expect("valid selected modes");
        assert_eq!(config.mode, Mode::Disjoint);
        assert_eq!(config.shape, Shape::Convergent);
        let cached = Config::parse(["--cache=on".to_string()]).unwrap();
        assert!(cached.cache);
        let uncached = Config::parse(["--cache=off".to_string()]).unwrap();
        assert!(!uncached.cache);
    }

    #[test]
    fn proc_counters_are_optional_without_affecting_parity() {
        let _ = proc_cpu_ticks();
        let _ = proc_status_kib("VmRSS:");
        assert_eq!(Shape::Disjoint.threshold(), 1);
        assert_eq!(Shape::Convergent.threshold(), 2);
        assert_eq!(Shape::WideDisjoint.threshold(), 1);
        assert_eq!(Shape::HeavyDisjoint.threshold(), 1);
    }
}

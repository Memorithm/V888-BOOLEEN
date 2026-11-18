//! Bounded paired synthetic benchmark for externally calibrated disjoint admission.
//! All cost coefficients are caller-supplied. No estimation of joules,
//! biological fidelity, or deployable speedup is made by this tool.
use std::env;
use std::hint::black_box;
use std::time::Instant;

use v888_booleen::connectome::Connectome;
use v888_booleen::event_runtime::{
    DisjointCommitWork, EventRuntime, EventWork, MeasuredDisjointCosts,
};
use v888_booleen::BooleanState;

#[derive(Clone, Copy, Debug)]
struct Config {
    sources: usize,
    contacts_per_source: usize,
    targets_per_source: usize,
    repetitions: usize,
    workers: usize,
    cache: bool,
    costs: MeasuredDisjointCosts,
}

impl Config {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut sources = 64;
        let mut contacts_per_source = 64;
        let mut targets_per_source = 4;
        let mut repetitions = 5;
        let mut workers = 4;
        let mut cache = false;
        let mut costs = None;

        for arg in arguments {
            let (key, value) = arg
                .strip_prefix("--")
                .and_then(|s| s.split_once('='))
                .ok_or_else(|| format!("invalid argument: {arg}"))?;
            match key {
                "sources" => sources = value.parse().map_err(|_| "invalid sources")?,
                "contacts" => {
                    contacts_per_source = value.parse().map_err(|_| "invalid contacts")?;
                }
                "targets" => {
                    targets_per_source = value.parse().map_err(|_| "invalid targets")?;
                }
                "repetitions" => {
                    repetitions = value.parse().map_err(|_| "invalid repetitions")?;
                }
                "workers" => workers = value.parse().map_err(|_| "invalid workers")?,
                "cache" => {
                    cache = match value {
                        "on" => true,
                        "off" => false,
                        _ => return Err(String::from("cache must be on or off")),
                    }
                }
                "costs" => costs = Some(parse_costs(value)?),
                _ => return Err(format!("unsupported option: {key}")),
            }
        }
        if !(2..=1024).contains(&sources)
            || !(1..=512).contains(&contacts_per_source)
            || !(1..=16).contains(&targets_per_source)
            || targets_per_source > contacts_per_source
            || !(1..=32).contains(&repetitions)
            || !(2..=4).contains(&workers)
        {
            return Err(String::from(
                "synthetic topology or repetition limit exceeded",
            ));
        }
        let costs = costs.ok_or("missing explicit --costs profile")?;
        if costs.workers != workers {
            return Err(String::from("profile worker count must equal --workers"));
        }
        Ok(Self {
            sources,
            contacts_per_source,
            targets_per_source,
            repetitions,
            workers,
            cache,
            costs,
        })
    }
}

fn parse_costs(value: &str) -> Result<MeasuredDisjointCosts, String> {
    let parts: Vec<_> = value.split(',').collect();
    if parts.len() != 12 {
        return Err(String::from("expected 12 comma-separated cost fields"));
    }
    let usize_at = |n: usize| {
        parts[n]
            .parse::<usize>()
            .map_err(|_| format!("invalid integer cost field {n}"))
    };
    let u64_at = |n: usize| {
        parts[n]
            .parse::<u64>()
            .map_err(|_| format!("invalid nanosecond cost field {n}"))
    };
    let c = MeasuredDisjointCosts {
        workers: usize_at(0)?,
        min_producers: usize_at(1)?,
        max_producers: usize_at(2)?,
        min_effective_contacts: usize_at(3)?,
        max_effective_contacts: usize_at(4)?,
        min_distinct_targets: usize_at(5)?,
        max_distinct_targets: usize_at(6)?,
        two_pass_overhead_ns: u64_at(7)?,
        serial_contact_ns: u64_at(8)?,
        serial_target_ns: u64_at(9)?,
        parallel_contact_ns: u64_at(10)?,
        parallel_target_ns: u64_at(11)?,
    };
    // A public CLI cannot invoke the private validity predicate, so repeat
    // its fail-closed constraints before launching a benchmark process.
    if !(2..=32).contains(&c.workers)
        || c.min_producers < 2
        || c.max_producers < c.min_producers
        || c.min_effective_contacts == 0
        || c.max_effective_contacts < c.min_effective_contacts
        || c.min_distinct_targets < 2
        || c.max_distinct_targets < c.min_distinct_targets
        || [
            c.two_pass_overhead_ns,
            c.serial_contact_ns,
            c.serial_target_ns,
            c.parallel_contact_ns,
            c.parallel_target_ns,
        ]
        .contains(&0)
    {
        return Err(String::from(
            "invalid profile geometry or cost coefficients",
        ));
    }
    Ok(c)
}

fn graph(config: Config) -> Connectome {
    let sources = config.sources;
    let targets_per_source = config.targets_per_source;
    let count = sources * (targets_per_source + 1);
    let rows = sources * config.contacts_per_source;
    let graph = Connectome {
        ids: (1..=count as u64).collect(),
        offsets: (0..=sources)
            .map(|n| (n * config.contacts_per_source) as u64)
            .chain(std::iter::repeat_n(rows as u64, count - sources))
            .collect(),
        targets: (0..sources)
            .flat_map(|source| {
                (0..config.contacts_per_source).map(move |row| {
                    (sources + source * targets_per_source + row % targets_per_source) as u32
                })
            })
            .collect(),
        counts: vec![1; rows],
    };
    graph.validate().expect("synthetic CSR graph must validate");
    graph
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Outcome {
    work: EventWork,
    state: BooleanState,
    incoming: Vec<u128>,
    last_time: Option<u64>,
    pending: usize,
    audit: bool,
}

struct Measurement {
    elapsed_ns: u128,
    outcome: Outcome,
    writers: usize,
    accepted: usize,
    rejected: usize,
}

fn execute(config: Config, graph: &Connectome, external: bool) -> Measurement {
    let mut engine = EventRuntime::new(graph, 1, 4 * graph.ids.len() + 16).unwrap();
    if config.cache {
        engine
            .prepare_contact_cache(config.sources * config.targets_per_source)
            .expect("bounded cache build");
    }
    for &id in &graph.ids[..config.sources] {
        engine.schedule_drive(0, id, true).unwrap();
        engine.schedule_drive(3, id, false).unwrap();
    }
    let started = Instant::now();
    let (work, writers, accepted, rejected) = if external {
        let report: DisjointCommitWork = engine
            .run_disjoint_parallel_commits_with_costs(usize::MAX, config.workers, config.costs)
            .expect("validated external-cost scheduler");
        (
            report.work,
            report.target_writer_threads,
            report.cost_profile_accepted,
            report.cost_profile_rejected,
        )
    } else {
        (engine.run(usize::MAX).unwrap(), 0, 0, 0)
    };
    let elapsed_ns = started.elapsed().as_nanos();
    let outcome = Outcome {
        work,
        state: engine.state().clone(),
        incoming: graph
            .ids
            .iter()
            .map(|&id| engine.incoming_for(id).unwrap())
            .collect(),
        last_time: engine.last_processed_time(),
        pending: engine.pending_count(),
        audit: engine.audit_incoming().unwrap(),
    };
    black_box(&outcome);
    Measurement {
        elapsed_ns,
        outcome,
        writers,
        accepted,
        rejected,
    }
}

fn report(config: Config) {
    let graph = graph(config);
    let reference = execute(config, &graph, false).outcome;
    assert!(reference.audit && reference.pending == 0);
    // Warm up both paths, excluding both warmups from medians.
    assert_eq!(execute(config, &graph, true).outcome, reference);
    let mut serial_times = Vec::new();
    let mut external_times = Vec::new();
    let mut writers = 0;
    let mut accepted = 0;
    let mut rejected = 0;
    for sample in 0..config.repetitions {
        // Alternate execution order to reduce fixed first-mode bias.
        let first_is_external = sample % 2 == 1;
        let first = execute(config, &graph, first_is_external);
        let second = execute(config, &graph, !first_is_external);
        assert_eq!(first.outcome, reference, "first run parity");
        assert_eq!(second.outcome, reference, "second run parity");
        for (external, current) in [(first_is_external, first), (!first_is_external, second)] {
            if external {
                external_times.push(current.elapsed_ns);
                writers += current.writers;
                accepted += current.accepted;
                rejected += current.rejected;
            } else {
                serial_times.push(current.elapsed_ns);
            }
        }
    }
    serial_times.sort_unstable();
    external_times.sort_unstable();
    let median_index = (config.repetitions - 1) / 2;
    println!(
        "V888_COST_BENCH sources={} contacts_per_source={} targets_per_source={} \
         repetitions={} workers={} cache={} serial_median_ns={} external_median_ns={} \
         writer_threads_total={} admission_accepted={} admission_rejected={} \
         events={} contacts={} exact_parity=true externally_supplied_costs=true \
         physical_calibration_unverified=true energy_measured=false",
        config.sources,
        config.contacts_per_source,
        config.targets_per_source,
        config.repetitions,
        config.workers,
        if config.cache { "on" } else { "off" },
        serial_times[median_index],
        external_times[median_index],
        writers,
        accepted,
        rejected,
        reference.work.events_consumed,
        reference.work.contacts_examined,
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

    const COSTS: &str = "4,2,1024,1,524288,2,16384,50,15,10,2,3";

    fn config(args: &[&str]) -> Result<Config, String> {
        Config::parse(args.iter().map(|arg| arg.to_string()))
    }

    #[test]
    fn cost_profile_and_budgets_are_mandatory() {
        assert!(config(&["--sources=8"]).is_err());
        assert!(config(&["--costs=1,2,3"]).is_err());
        assert!(config(&["--workers=0", &format!("--costs={COSTS}")]).is_err());
        assert!(config(&["--cache=invalid", &format!("--costs={COSTS}")]).is_err());
        assert!(config(&["--targets=17", &format!("--costs={COSTS}")]).is_err());
        assert!(config(&["--sources=1025", &format!("--costs={COSTS}")]).is_err());
        assert!(config(&["--repetitions=33", &format!("--costs={COSTS}")]).is_err());
        assert!(config(&["--costs=4,2,1024,1,524288,2,16384,0,15,10,2,3"]).is_err());
        assert!(config(&[&format!("--costs={COSTS}")]).is_ok());
    }

    #[test]
    fn paired_cost_runtime_matches_reference_with_and_without_cache() {
        for cache in [false, true] {
            let opts = if cache { "on" } else { "off" };
            let args = vec![
                "--sources=8".to_string(),
                "--contacts=64".to_string(),
                "--targets=4".to_string(),
                format!("--cache={opts}"),
                format!("--costs={COSTS}"),
            ];
            let config = Config::parse(args).unwrap();
            let graph = graph(config);
            graph.validate().unwrap();
            let serial = execute(config, &graph, false);
            let external = execute(config, &graph, true);
            assert_eq!(serial.outcome, external.outcome);
            assert!(external.outcome.audit);
            assert_eq!(external.outcome.pending, 0);
        }
    }
}

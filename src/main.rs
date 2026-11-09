use std::error::Error;
use std::fs::File;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader};
use std::path::Path;
use v888_booleen::circuit_explorer::{ActivityReport, CircuitExplorer, RegionLabels, Routing};
use v888_booleen::connectome::Connectome;
use v888_booleen::consolidation::sweep as consolidation_sweep;
use v888_booleen::engine::BooleanEngine;
use v888_booleen::full_synapse::{pack_records, FullSynapseReader};
use v888_booleen::learning_bench::benchmark;
use v888_booleen::learning_strategies::compare_strategies;
use v888_booleen::plasticity::PlasticCircuit;
use v888_booleen::source_holdout::benchmark_disjoint_sources;
use v888_booleen::synapse_pairs::{pack, propagate_one_step, SynapsePairsReader};
fn require<'a>(args: &'a [String], name: &str) -> Result<&'a str, Box<dyn Error>> {
    let ix = args
        .iter()
        .position(|v| v == name)
        .ok_or(format!("missing {name}"))?;
    Ok(args
        .get(ix + 1)
        .ok_or(format!("missing value for {name}"))?)
}
fn optional<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|v| v == name)
        .and_then(|ix| args.get(ix + 1).map(String::as_str))
}

type Episode = (usize, usize, bool);

fn read_training_episodes(graph: &Connectome, path: &Path) -> Result<Vec<Episode>, Box<dyn Error>> {
    const MAX_EXAMPLES: usize = 65_536;
    let mut records = Vec::new();
    for (i, text) in BufReader::new(File::open(path)?).lines().enumerate() {
        let row = text?;
        if i == 0 {
            if row.trim_end_matches('\r') != "source_id,target_id,fire" {
                return Err("episode CSV requires source_id,target_id,fire header".into());
            }
            continue;
        }
        if row.len() > 100 || records.len() >= MAX_EXAMPLES {
            return Err("episode row size or count exceeded".into());
        }
        let fields: Vec<&str> = row.trim_end_matches('\r').split(',').collect();
        if fields.len() != 3 {
            return Err("invalid labeled episode".into());
        }
        let source_id = fields[0].parse::<u64>()?;
        let target_id = fields[1].parse::<u64>()?;
        let source = graph.index(source_id).ok_or("unknown source ID")?;
        let target = graph.index(target_id).ok_or("unknown target ID")?;
        let fire = match fields[2] {
            "0" => false,
            "1" => true,
            _ => return Err("fire must be 0 or 1".into()),
        };
        records.push((source, target, fire));
    }
    if records.is_empty() {
        return Err("empty learning episodes".into());
    }
    Ok(records)
}

fn run_one_episode(
    model: &PlasticCircuit,
    graph: &Connectome,
    record: Episode,
    threshold: u128,
) -> Result<bool, Box<dyn Error>> {
    let mut stimulus = v888_booleen::BooleanState::new(graph.ids.len());
    stimulus
        .set(record.0, true)
        .map_err(|_| "invalid source index")?;
    let output = model.predict(graph, &stimulus, threshold)?;
    Ok(output.get(record.1).map_err(|_| "invalid target index")?)
}

fn print_circuit_report(step: u32, report: &ActivityReport) {
    println!(
        "circuit_step={step} active={} newly_active={} silenced={} active_sources={} routed_edges={} contact_mass={} regions={:?} sample_ids={:?}",
        report.active_neurons,
        report.newly_active,
        report.silenced,
        report.active_sources_visited,
        report.routed_edges,
        report.contact_mass,
        report.active_by_region,
        report.sampled_neuron_ids
    );
}

fn help() {
    println!("V888 BOOLEEN experimental Boolean connectome (no biological data bundled)");
    println!("  build --edges edges.csv [--neurons ids.txt] --output brain.v888b");
    println!("  info --graph brain.v888b");
    println!("  export --graph brain.v888b --output rows.csv");
    println!("  run --graph brain.v888b --seed ID[,ID] [--threshold 1] [--steps 3]");
    println!("  demo");
    println!("  recall-circuits --graph brain.v888b --cues ID[,ID] [--policy directed|strongest-k|symmetric-hypothesis] [--hops 2] [--min-cues 1] [--top 20] [--budget 100000] [--regions labels.csv]");
    println!("  explore-circuits --graph brain.v888b --seed ID[,ID] [--policy directed|strongest-k|symmetric-hypothesis] [--fanout 8] [--regions labels.csv] [--threshold 1] [--steps 3]");
    println!("  recall-circuits --graph brain.v888b --cues ID[,ID] [--policy directed|strongest-k|symmetric-hypothesis] [--fanout 8] [--hops 2] [--min-cues 2] [--top-k 10] [--edge-budget 100000] [--regions labels.csv]");
    println!("  pack-synapses --expected-rows N --output synapses.v888syn < records.bin");
    println!("  synapse-info --pairs synapses.v888syn");
    println!("  pack-all --expected-rows N --output all.v888all < all30-fields.bin");
    println!(
        "  pack-all --expected-rows N --output - < all30-fields.bin | gzip -1 > all.v888all.gz"
    );
    println!("  all-info --file all.v888all");
    println!("  validate-all --file all.v888all");
    println!("  gzip -dc all.v888all.gz | v888-booleen validate-all --file -");
    println!("  run-synapses --graph brain.v888b --pairs synapses.v888syn --seed ID[,ID] [--threshold 3] [--steps 2]");
    println!("  bench-learning --graph brain.v888b  (supervised retention only)");
    println!("  bench-holdout --graph brain.v888b (unseen-source generalization control)");
    println!("  bench-strategies [--graph brain.v888b] [--seed 1] [--rounds 8]  (equal-budget strategy comparison)");
    println!("  bench-consolidation [--graph brain.v888b] [--seed 1] [--rounds-a 8] [--rounds-b 8] [--periods 2,4,8,1000]  (learn A then B, retention of A)");
    println!("  train-plasticity --graph brain.v888b --episodes labeled.csv --output weights.v888pls [--epochs 1]");
    println!("  score-plasticity --graph brain.v888b --weights weights.v888pls --episodes labeled.csv [--threshold 1]");
    println!("  run-plasticity --graph brain.v888b --weights weights.v888pls --seed ID[,ID] [--threshold 1]");
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("build") => {
            let graph = Connectome::import(
                Path::new(require(&args, "--edges")?),
                optional(&args, "--neurons").map(Path::new),
            )?;
            graph.save(Path::new(require(&args, "--output")?))?;
            println!("saved neurons={} edges={}", graph.ids.len(), graph.edges());
        }
        Some("info") if optional(&args, "--graph").is_some() => {
            let graph = Connectome::load(Path::new(require(&args, "--graph")?))?;
            println!(
                "neurons={} edges={} format=v888b-v1",
                graph.ids.len(),
                graph.edges()
            );
        }
        Some("info") => {
            println!("V888 BOOLEEN experimental engine - no source dataset bundled");
            println!("Use info --graph <file> after importing real BANC v888 data");
        }
        Some("export") => {
            Connectome::load(Path::new(require(&args, "--graph")?))?
                .export_csv(Path::new(require(&args, "--output")?))?;
        }
        Some("recall-circuits") => {
            // Explicitly reject unknown flags; they must never silently
            // increase a recall budget or alter the requested hit count.
            let allowed = [
                "--graph",
                "--cues",
                "--policy",
                "--fanout",
                "--hops",
                "--min-cues",
                "--top",
                "--top-k",
                "--budget",
                "--edge-budget",
                "--regions",
            ];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(format!("unknown recall-circuits option: {flag}").into());
                }
            }
            if (optional(&args, "--top").is_some() && optional(&args, "--top-k").is_some())
                || (optional(&args, "--budget").is_some()
                    && optional(&args, "--edge-budget").is_some())
            {
                return Err("ambiguous recall aliases provided together".into());
            }
            let graph = Connectome::load(Path::new(require(&args, "--graph")?))?;
            let fanout = optional(&args, "--fanout")
                .unwrap_or("8")
                .parse::<usize>()?;
            let policy = Routing::parse(optional(&args, "--policy").unwrap_or("directed"), fanout)?;
            let hops = optional(&args, "--hops").unwrap_or("2").parse::<u8>()?;
            let min_cues = optional(&args, "--min-cues")
                .unwrap_or("2")
                .parse::<usize>()?;
            let top_k = optional(&args, "--top-k")
                .or_else(|| optional(&args, "--top"))
                .unwrap_or("10")
                .parse::<usize>()?;
            let edge_budget = optional(&args, "--edge-budget")
                .or_else(|| optional(&args, "--budget"))
                .unwrap_or("100000")
                .parse::<u64>()?;
            let cues: Vec<u64> = require(&args, "--cues")?
                .split(',')
                .map(str::parse::<u64>)
                .collect::<Result<_, _>>()?;
            let regions = match optional(&args, "--regions") {
                Some(path) => Some(RegionLabels::from_reader(
                    &graph,
                    BufReader::new(File::open(path)?),
                )?),
                None => None,
            };
            let explorer = CircuitExplorer::new(&graph, policy, 1)?;
            let report =
                explorer.recall(&cues, hops, min_cues, top_k, edge_budget, regions.as_ref())?;
            println!(
                "recall_scope=structural_not_learned routing={policy:?} cues={} hops={} edges_examined={} source_expansions={} hits={}",
                cues.len(),
                hops,
                report.edges_examined,
                report.source_expansions,
                report.hits.len(),
            );
            for (rank, hit) in report.hits.iter().enumerate() {
                println!(
                    "recall_rank={} neuron_id={} region={} distinct_cues={} score={} nearest_hops={} evidence={:?}",
                    rank + 1,
                    hit.neuron_id,
                    hit.region,
                    hit.distinct_cues,
                    hit.score,
                    hit.nearest_hops,
                    hit.evidence,
                );
            }
        }
        Some("explore-circuits") => {
            let graph = Connectome::load(Path::new(require(&args, "--graph")?))?;
            let fanout = optional(&args, "--fanout")
                .unwrap_or("8")
                .parse::<usize>()?;
            let policy = Routing::parse(optional(&args, "--policy").unwrap_or("directed"), fanout)?;
            let threshold = optional(&args, "--threshold")
                .unwrap_or("1")
                .parse::<u128>()?;
            let steps = optional(&args, "--steps").unwrap_or("3").parse::<u32>()?;
            if steps > 100 {
                return Err("exploration limited to 100 synchronous steps".into());
            }
            let regions = match optional(&args, "--regions") {
                Some(path) => Some(RegionLabels::from_reader(
                    &graph,
                    BufReader::new(File::open(path)?),
                )?),
                None => None,
            };
            let mut explorer = CircuitExplorer::new(&graph, policy, threshold)?;
            for id in require(&args, "--seed")?.split(',') {
                explorer.stimulate(id.parse::<u64>()?)?;
            }
            println!(
                "routing={policy:?} experimental_dynamics=true region_labels={} source_topology_unchanged=true",
                if regions.is_some() { "provided" } else { "UNMAPPED" }
            );
            print_circuit_report(0, &explorer.snapshot(regions.as_ref())?);
            for step in 1..=steps {
                print_circuit_report(step, &explorer.step(regions.as_ref())?);
            }
        }
        Some("run") => {
            let graph = Connectome::load(Path::new(require(&args, "--graph")?))?;
            let threshold = optional(&args, "--threshold")
                .unwrap_or("1")
                .parse::<u64>()?;
            let steps = optional(&args, "--steps").unwrap_or("3").parse::<u32>()?;
            if steps > 1000 {
                return Err("too many steps (>1000)".into());
            }
            let mut brain = BooleanEngine::new(&graph, threshold)?;
            for seed in require(&args, "--seed")?.split(',') {
                brain.stimulate(&graph, seed.parse::<u64>()?)?;
            }
            println!("step=0 active={}", brain.active.active_count());
            for step in 1..=steps {
                println!("step={step} active={}", brain.step(&graph)?);
            }
        }
        Some("pack-synapses") => {
            let expected = require(&args, "--expected-rows")?.parse::<u64>()?;
            let path = Path::new(require(&args, "--output")?);
            let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
            let result = pack(&mut std::io::stdin().lock(), &mut output, expected);
            if result.is_err() {
                drop(output);
                fs::remove_file(path)?;
            }
            println!("synapses_packed={}", result?);
        }
        Some("pack-all") => {
            let expected = require(&args, "--expected-rows")?.parse::<u64>()?;
            let destination = require(&args, "--output")?;
            if destination == "-" {
                let mut stream = std::io::stdout().lock();
                let rows = pack_records(&mut std::io::stdin().lock(), &mut stream, expected)?;
                eprintln!("full_synapse_rows_packed={rows}");
            } else {
                let path = Path::new(destination);
                let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
                let result = pack_records(&mut std::io::stdin().lock(), &mut output, expected);
                if result.is_err() {
                    drop(output);
                    fs::remove_file(path)?;
                }
                println!("full_synapse_rows_packed={}", result?);
            }
        }
        Some("all-info") => {
            let rows = FullSynapseReader::open(Path::new(require(&args, "--file")?))?;
            println!("full_synapse_rows={} fields=30 format=V888C30A", rows.len());
        }
        Some("validate-all") => {
            let input = require(&args, "--file")?;
            let counted = if input == "-" {
                v888_booleen::full_synapse::validate_stream(&mut std::io::stdin().lock())?
            } else {
                let mut reader = FullSynapseReader::open(Path::new(input))?;
                let mut counted = 0_u64;
                reader.visit(|_row| {
                    counted += 1;
                    Ok(())
                })?;
                counted
            };
            println!("full_synapse_verified_rows={counted} format=V888C30A");
        }
        Some("synapse-info") => {
            let source = SynapsePairsReader::open(Path::new(require(&args, "--pairs")?))?;
            println!("individual_synapse_pairs={} format=V888SYB1", source.len());
        }
        Some("run-synapses") => {
            let graph = Connectome::load(Path::new(require(&args, "--graph")?))?;
            let pair_file = Path::new(require(&args, "--pairs")?);
            let threshold = optional(&args, "--threshold")
                .unwrap_or("3")
                .parse::<u64>()?;
            let steps = optional(&args, "--steps").unwrap_or("2").parse::<u32>()?;
            if steps > 10 {
                return Err("synapse stream limited to 10 steps per invocation".into());
            }
            let mut brain = BooleanEngine::new(&graph, threshold)?;
            for seed in require(&args, "--seed")?.split(',') {
                brain.stimulate(&graph, seed.parse::<u64>()?)?;
            }
            println!("synapse_step=0 active={}", brain.active.active_count());
            for step in 1..=steps {
                let mut reader = SynapsePairsReader::open(pair_file)?;
                let (next, counted) =
                    propagate_one_step(&graph, &brain.active, &mut reader, threshold)?;
                brain.active = next;
                println!(
                    "synapse_step={step} active={} counted_contacts={counted}",
                    brain.active.active_count()
                );
            }
        }
        Some("train-plasticity") => {
            let graph = Connectome::load(Path::new(require(&args, "--graph")?))?;
            let records = read_training_episodes(&graph, Path::new(require(&args, "--episodes")?))?;
            let epochs = optional(&args, "--epochs").unwrap_or("1").parse::<u32>()?;
            if epochs == 0 || epochs > 64 || records.len() as u64 * u64::from(epochs) > 1_000_000 {
                return Err("training episode budget exceeded".into());
            }
            let mut model = PlasticCircuit::new(&graph, 0)?;
            let mut stimulus = v888_booleen::BooleanState::new(graph.ids.len());
            let mut teacher = v888_booleen::BooleanState::new(graph.ids.len());
            let mut updates = 0_u64;
            for _ in 0..epochs {
                for &(source, target, fire) in &records {
                    stimulus.set(source, true).map_err(|_| "invalid source")?;
                    if fire {
                        teacher.set(target, true).map_err(|_| "invalid teacher")?;
                    }
                    updates = updates
                        .checked_add(model.train(&graph, &stimulus, &teacher)?)
                        .ok_or("training update overflow")?;
                    stimulus.set(source, false).map_err(|_| "invalid source")?;
                    if fire {
                        teacher.set(target, false).map_err(|_| "invalid teacher")?;
                    }
                }
            }
            model.save(Path::new(require(&args, "--output")?))?;
            println!(
                "plasticity_training_episodes={} edge_updates={} checkpoint_weight_bytes={}",
                records.len() as u64 * u64::from(epochs),
                updates,
                model.weight_bytes()
            );
            println!("plasticity_supervision=artificial_labels_not_insect_behavior");
        }
        Some("score-plasticity") => {
            let graph = Connectome::load(Path::new(require(&args, "--graph")?))?;
            let model = PlasticCircuit::load(Path::new(require(&args, "--weights")?), &graph)?;
            let records = read_training_episodes(&graph, Path::new(require(&args, "--episodes")?))?;
            let threshold = optional(&args, "--threshold")
                .unwrap_or("1")
                .parse::<u128>()?;
            let mut correct = 0_u64;
            for &record in &records {
                if run_one_episode(&model, &graph, record, threshold)? == record.2 {
                    correct += 1;
                }
            }
            println!(
                "plasticity_correct={} plasticity_examples={}",
                correct,
                records.len()
            );
            println!("plasticity_scope=explicit_supervised_recall");
        }
        Some("run-plasticity") => {
            let graph = Connectome::load(Path::new(require(&args, "--graph")?))?;
            let model = PlasticCircuit::load(Path::new(require(&args, "--weights")?), &graph)?;
            let threshold = optional(&args, "--threshold")
                .unwrap_or("1")
                .parse::<u128>()?;
            let mut input = v888_booleen::BooleanState::new(graph.ids.len());
            for seed in require(&args, "--seed")?.split(',') {
                let id = seed.parse::<u64>()?;
                let index = graph.index(id).ok_or("unknown seed ID")?;
                input.set(index, true).map_err(|_| "invalid seed index")?;
            }
            let output = model.predict(&graph, &input, threshold)?;
            println!(
                "plasticity_active={} plasticity_threshold={threshold}",
                output.active_count()
            );
            println!("plasticity_scope=one_synchronous_experimental_step");
        }
        Some("bench-holdout") => {
            let graph = Connectome::load(Path::new(require(&args, "--graph")?))?;
            let r = benchmark_disjoint_sources(&graph)?;
            println!("v888_holdout_schema=disjoint-source-v1");
            println!("dataset_neurons={} dataset_edges={}", r.neurons, r.edges);
            println!("train_positive_source={} train_negative_source={} test_positive_source={} test_negative_source={} target_id={}",
                r.training_source_ids[0], r.training_source_ids[1], r.test_source_ids[0], r.test_source_ids[1], r.target_id);
            println!(
                "train_episodes={} seen_test_episodes={} unseen_test_episodes={}",
                r.train_episodes, r.seen_test_episodes, r.unseen_test_episodes
            );
            println!("frozen_zero_seen_correct={} frozen_zero_unseen_correct={} frozen_topology_seen_correct={} frozen_topology_unseen_correct={}",
                r.zero_seen_correct, r.zero_unseen_correct, r.topology_seen_correct, r.topology_unseen_correct);
            println!(
                "learned_seen_correct={} learned_unseen_correct={} plastic_weight_bytes={}",
                r.learned_seen_correct, r.learned_unseen_correct, r.weight_bytes
            );
            println!(
                "train_elapsed_us={} eval_elapsed_us={}",
                r.training_elapsed.as_micros(),
                r.evaluation_elapsed.as_micros()
            );
            println!("evaluation_scope=artificial_labels_disjoint_sources_not_biological_learning");
        }
        Some("bench-learning") => {
            let graph = Connectome::load(Path::new(require(&args, "--graph")?))?;
            let r = benchmark(&graph)?;
            println!("v888_learning_report_schema=retention-v1");
            println!("dataset_neurons={} dataset_edges={}", r.neurons, r.edges);
            println!(
                "supervised_positive_source={} supervised_negative_source={} target_neuron={}",
                r.positive_source_id, r.negative_source_id, r.target_id
            );
            println!(
                "training_episodes={} test_episodes={} positive_test_examples={}",
                r.training_episodes, r.tested_episodes, r.positive_examples
            );
            println!(
                "frozen_zero_correct={} frozen_topology_correct={} learned_correct={}",
                r.frozen_zero_correct, r.frozen_topology_correct, r.learned_correct
            );
            println!("plastic_weight_bytes={}", r.learned_weight_bytes);
            println!("train_elapsed_us={} eval_frozen_zero_us={} eval_frozen_topology_us={} eval_learned_us={}", r.train_elapsed.as_micros(), r.frozen_zero_eval_elapsed.as_micros(), r.frozen_topology_eval_elapsed.as_micros(), r.learned_eval_elapsed.as_micros());
            println!("evaluation_scope=seen_pattern_retention_not_generalization_or_biology");
        }
        Some("bench-consolidation") => {
            let seed = optional(&args, "--seed").unwrap_or("1").parse::<u64>()?;
            let rounds_a = optional(&args, "--rounds-a")
                .unwrap_or("8")
                .parse::<u32>()?;
            let rounds_b = optional(&args, "--rounds-b")
                .unwrap_or("8")
                .parse::<u32>()?;
            let periods = optional(&args, "--periods")
                .unwrap_or("2,4,8,1000")
                .split(',')
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>()?;
            let owned = match optional(&args, "--graph") {
                Some(path) => Some(Connectome::load(Path::new(path))?),
                None => None,
            };
            let outcomes = consolidation_sweep(owned.as_ref(), seed, rounds_a, rounds_b, &periods)?;
            println!("v888_consolidation_report_schema=b888-4-consolidation-v1");
            println!(
                "graph={} seed={seed} rounds_a={rounds_a} rounds_b={rounds_b}",
                if owned.is_some() {
                    "provided"
                } else {
                    "synthetic-comparison"
                }
            );
            for r in &outcomes {
                println!(
                    "protect_period={} a_after_a={}/{} a_after_b={}/{} b_after_b={}/{} protected_edges={}",
                    r.config.protect_period, r.a_correct_after_a, r.presentations,
                    r.a_correct_after_b, r.presentations, r.b_correct_after_b, r.presentations,
                    r.protected_edges
                );
                println!(
                    "protect_period={} a_visited={} a_changed={} b_visited={} b_changed={} b_blocked={} b_effective_rate={} observed_activations={}",
                    r.config.protect_period, r.phase_a.edges_visited, r.phase_a.weights_changed,
                    r.phase_b.edges_visited, r.phase_b.weights_changed, r.phase_b.blocked,
                    r.recovery_rate_b().map_or("n/a".to_string(), |v| format!("{v:.4}")),
                    r.observed_activations
                );
                println!(
                    "protect_period={} weight_bytes={} overhead_bytes={} elapsed_us={} topology_unchanged={}",
                    r.config.protect_period, r.weight_bytes, r.overhead_bytes,
                    r.elapsed.as_micros(), r.topology_unchanged
                );
            }
            println!(
                "evaluation_scope=artificial_labels_overlapping_sources_not_biological_memory"
            );
        }
        Some("bench-strategies") => {
            let seed = optional(&args, "--seed").unwrap_or("1").parse::<u64>()?;
            let rounds = optional(&args, "--rounds").unwrap_or("8").parse::<u32>()?;
            let graph_path = optional(&args, "--graph");
            let owned = match graph_path {
                Some(path) => Some(Connectome::load(Path::new(path))?),
                None => None,
            };
            let outcomes = compare_strategies(owned.as_ref(), seed, rounds)?;
            println!("v888_strategy_report_schema=b888-4-v1");
            println!(
                "graph={} seed={} rounds={} strategies={}",
                if graph_path.is_some() {
                    "provided"
                } else {
                    "synthetic-comparison"
                },
                seed,
                rounds,
                outcomes.len()
            );
            for r in &outcomes {
                println!(
                    "strategy={} train_pos={} train_neg={} holdout_pos={} holdout_neg={} target={}",
                    r.strategy.as_label(),
                    r.train_pos_id,
                    r.train_neg_id,
                    r.holdout_pos_id,
                    r.holdout_neg_id,
                    r.target_id
                );
                println!(
                    "strategy={} seen_before={}/{} seen_after={}/{} unseen_after={}/{} seen_after_reversal={}/{} reversal_task={}/{}",
                    r.strategy.as_label(),
                    r.seen_correct_before,
                    r.seen_total,
                    r.seen_correct_after,
                    r.seen_total,
                    r.unseen_correct_after,
                    r.unseen_total,
                    r.seen_correct_after_reversal,
                    r.seen_total,
                    r.reversal_task_correct,
                    r.seen_total
                );
                println!(
                    "strategy={} train_episodes={} reversal_episodes={} edges_visited={} weights_changed={} weight_bytes={} unpacked_weight_bytes={} train_us={} eval_us={} topology_unchanged={}",
                    r.strategy.as_label(),
                    r.train_episodes,
                    r.reversal_episodes,
                    r.edges_visited,
                    r.weights_changed,
                    r.weight_bytes,
                    r.unpacked_weight_bytes,
                    r.train_elapsed.as_micros(),
                    r.eval_elapsed.as_micros(),
                    r.topology_unchanged
                );
            }
            println!("evaluation_scope=artificial_labels_equal_budget_retention_reversal_and_holdout_not_biology");
        }
        Some("demo") => {
            let graph = Connectome {
                ids: vec![10, 20, 30],
                offsets: vec![0, 1, 2, 2],
                targets: vec![1, 2],
                counts: vec![2, 3],
            };
            let mut brain = BooleanEngine::new(&graph, 2)?;
            brain.stimulate(&graph, 10)?;
            for step in 1..=3 {
                println!("demo step={step} active={}", brain.step(&graph)?);
            }
            println!("SYNTHETIC DEMO ONLY; no V888 anatomical data loaded");
        }
        Some("--help") | Some("-h") | None => help(),
        _ => {
            help();
            return Err("unknown command".into());
        }
    }
    Ok(())
}

use std::error::Error;
use std::fs::File;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader};
use std::path::Path;
use v888_booleen::circuit_explorer::{ActivityReport, CircuitExplorer, RegionLabels, Routing};
use v888_booleen::connectome::Connectome;
use v888_booleen::consolidation::{
    matrix as consolidation_matrix, sweep as consolidation_sweep, to_csv as consolidation_csv,
    Scenario, ALL_STRATEGIES,
};
use v888_booleen::consolidation_scale as scale;
use v888_booleen::eligibility_trace as et;
use v888_booleen::engine::BooleanEngine;
use v888_booleen::full_synapse::{pack_records, FullSynapseReader};
use v888_booleen::generalization_bench as genbench;
use v888_booleen::graded_reward as gr;
use v888_booleen::learning_bench::benchmark;
use v888_booleen::learning_strategies::compare_strategies;
use v888_booleen::plasticity::PlasticCircuit;
use v888_booleen::reward_limits as rl;
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
    println!("  bench-consolidation-matrix [--graph brain.v888b] [--seeds 1,2,3] [--rounds-a 8] [--rounds-b 8] [--variants 2,4,1000:1] [--csv out.csv]  (3 strategies x conflict/disjoint x protection, CSV)");
    println!("  bench-consolidation-scale [--seeds 1,2,..] [--sources 64] [--targets 24] [--fan-in 6] [--conflict-pcts 0,50,100] [--rounds-a 8] [--rounds-b 8] [--variants 2,8,1000,1000:1,1000:2] [--csv out.csv]  (GENERATED synthetic graph, many overlapping edges)");
    println!("  bench-generalization [--graph brain.v888b] [--seeds 1,..,10] [--inits 0,1] [--sources 64] [--targets 24] [--fan-in 6] [--active 4] [--train-patterns 32] [--test-patterns 64] [--rounds-a 8] [--rounds-b 8] [--positive-pct 50] [--flip-pct 50] [--csv out.csv]  (held-out generalisation + A-then-B forgetting vs supervised control; GENERATED synthetic graph by default)");
    println!("  bench-reward-limits [--graph brain.v888b] [--seeds 1,..,10] [--init 1] [--csv out.csv]  (reward-modulated under reward noise / sparse reward / credit assignment vs matched supervised control; same task and defaults as bench-generalization; GENERATED synthetic graph by default)");
    println!("  bench-eligibility-traces [--graph brain.v888b] [--seeds 1,..,10] [--init 1] [--csv out.csv]  (eligibility trace + reward-prediction error remedy for scalar / delayed credit vs #34 reward arm and supervised control; GENERATED synthetic graph by default)");
    println!("  bench-graded-reward [--graph brain.v888b] [--seeds 1,..,10] [--init 1] [--csv out.csv]  (graded reward ladder bit/fraction/group-k/per-target through the #36 trace + episode-stamped tags for delayed scalars; GENERATED synthetic graph by default)");
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
        Some("bench-consolidation-matrix") => {
            let seeds = optional(&args, "--seeds")
                .unwrap_or("1,2,3")
                .split(',')
                .map(str::parse::<u64>)
                .collect::<Result<Vec<_>, _>>()?;
            let rounds_a = optional(&args, "--rounds-a")
                .unwrap_or("8")
                .parse::<u32>()?;
            let rounds_b = optional(&args, "--rounds-b")
                .unwrap_or("8")
                .parse::<u32>()?;
            // variants: period[:min_weight], comma-separated
            let variants = optional(&args, "--variants")
                .unwrap_or("2,4,8,1000,1000:1,1000:15")
                .split(',')
                .map(|v| {
                    let (p, w) = v.split_once(':').unwrap_or((v, "0"));
                    Ok::<_, Box<dyn std::error::Error>>((p.parse::<u32>()?, w.parse::<u8>()?))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let owned = match optional(&args, "--graph") {
                Some(path) => Some(Connectome::load(Path::new(path))?),
                None => None,
            };
            let rows = consolidation_matrix(
                owned.as_ref(),
                &seeds,
                rounds_a,
                rounds_b,
                &variants,
                &ALL_STRATEGIES,
                &[Scenario::Conflict, Scenario::Disjoint],
            )?;
            let csv = consolidation_csv(&rows);
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(csv.as_bytes())?;
            }
            println!("v888_consolidation_matrix_schema=b888-4-consolidation-matrix-v1");
            print!("{csv}");
            println!(
                "evaluation_scope=artificial_labels_overlapping_sources_not_biological_memory"
            );
        }
        Some("bench-consolidation-scale") => {
            let list = |name: &str, default: &str| -> Result<Vec<u64>, Box<dyn Error>> {
                Ok(optional(&args, name)
                    .unwrap_or(default)
                    .split(',')
                    .map(str::parse::<u64>)
                    .collect::<Result<Vec<_>, _>>()?)
            };
            let num = |name: &str, default: &str| -> Result<usize, Box<dyn Error>> {
                Ok(optional(&args, name).unwrap_or(default).parse::<usize>()?)
            };
            let seeds = list("--seeds", "1,2,3,4,5,6,7,8,9,10")?;
            let pcts: Vec<u32> = list("--conflict-pcts", "0,50,100")?
                .into_iter()
                .map(|v| v as u32)
                .collect();
            let base = scale::ScaleSpec {
                sources: num("--sources", "64")?,
                targets: num("--targets", "24")?,
                fan_in: num("--fan-in", "6")?,
                conflict_pct: 0,
                rounds_a: num("--rounds-a", "8")? as u32,
                rounds_b: num("--rounds-b", "8")? as u32,
            };
            let variants = optional(&args, "--variants")
                .unwrap_or("2,8,1000,1000:1,1000:2")
                .split(',')
                .map(|v| {
                    let (p, w) = v.split_once(':').unwrap_or((v, "0"));
                    Ok::<_, Box<dyn std::error::Error>>((p.parse::<u32>()?, w.parse::<u8>()?))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let rows = scale::run_grid(&base, &pcts, &seeds, &ALL_STRATEGIES, &variants)?;
            let csv = scale::to_csv(&rows);
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(csv.as_bytes())?;
            }
            println!("v888_consolidation_scale_schema=b888-4-consolidation-scale-v1");
            println!(
                "graph=GENERATED_SYNTHETIC sources={} targets={} fan_in={} seeds={}",
                base.sources,
                base.targets,
                base.fan_in,
                seeds.len()
            );
            println!("conflict_pct,strategy,period,min_weight,seeds,protected_edges_mean,a_after_a,a_after_b,b_after_b,d_a_after_b,d_b_after_b,useful");
            for s in scale::summarize(&rows) {
                println!(
                    "{},{},{},{},{},{:.1},{:.3},{:.3},{:.3},{:+.3},{:+.3},{}",
                    s.conflict_pct,
                    s.strategy.as_label(),
                    s.period,
                    s.min_weight,
                    s.seeds,
                    s.protected_edges,
                    s.a_after_a,
                    s.a_after_b,
                    s.b_after_b,
                    s.d_a_after_b,
                    s.d_b_after_b,
                    scale::is_useful(&s)
                );
            }
            println!("evaluation_scope=generated_synthetic_graph_artificial_labels_not_BANC_not_biological");
        }
        Some("bench-generalization") => {
            let allowed = [
                "--graph",
                "--seeds",
                "--inits",
                "--sources",
                "--targets",
                "--fan-in",
                "--active",
                "--train-patterns",
                "--test-patterns",
                "--rounds-a",
                "--rounds-b",
                "--positive-pct",
                "--flip-pct",
                "--csv",
            ];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(format!("unknown bench-generalization option: {flag}").into());
                }
            }
            let list = |name: &str, default: &str| -> Result<Vec<u64>, Box<dyn Error>> {
                Ok(optional(&args, name)
                    .unwrap_or(default)
                    .split(',')
                    .map(str::parse::<u64>)
                    .collect::<Result<Vec<_>, _>>()?)
            };
            let num = |name: &str, default: &str| -> Result<usize, Box<dyn Error>> {
                Ok(optional(&args, name).unwrap_or(default).parse::<usize>()?)
            };
            let seeds = list("--seeds", "1,2,3,4,5,6,7,8,9,10")?;
            let inits = list("--inits", "0,1")?
                .into_iter()
                .map(u8::try_from)
                .collect::<Result<Vec<_>, _>>()?;
            let shape = genbench::GraphShape {
                sources: num("--sources", "64")?,
                targets: num("--targets", "24")?,
                fan_in: num("--fan-in", "6")?,
            };
            let spec = genbench::GenSpec {
                active: num("--active", "4")?,
                train_patterns: num("--train-patterns", "32")?,
                test_patterns: num("--test-patterns", "64")?,
                rounds_a: u32::try_from(num("--rounds-a", "8")?)?,
                rounds_b: u32::try_from(num("--rounds-b", "8")?)?,
                positive_pct: u32::try_from(num("--positive-pct", "50")?)?,
                flip_pct: u32::try_from(num("--flip-pct", "50")?)?,
                init_weight: 0,
            };
            let owned = match optional(&args, "--graph") {
                Some(path) => Some(Connectome::load(Path::new(path))?),
                None => None,
            };
            let graph_label = if owned.is_some() {
                "PROVIDED_GRAPH"
            } else {
                "GENERATED_SYNTHETIC"
            };
            let rows = genbench::run_grid(owned.as_ref(), &shape, &spec, &seeds, &inits)?;
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(genbench::to_csv(&rows, graph_label).as_bytes())?;
            }
            println!("v888_generalization_schema=b888-4-generalization-v1");
            if owned.is_some() {
                println!("graph=PROVIDED_GRAPH (task labels artificial)");
            } else {
                println!(
                    "graph=GENERATED_SYNTHETIC sources={} targets={} fan_in={} (not BANC; real BANC V888 graph DOI 10.7910/DVN/7WTH1N not available here)",
                    shape.sources, shape.targets, shape.fan_in
                );
            }
            println!(
                "seeds={} inits={:?} active={} train_patterns={} test_patterns={} rounds_a={} rounds_b={} positive_pct={} flip_pct={} threshold=1 metric=balanced_accuracy",
                seeds.len(), inits, spec.active, spec.train_patterns, spec.test_patterns,
                spec.rounds_a, spec.rounds_b, spec.positive_pct, spec.flip_pct
            );
            println!("init,arm,seeds,a_train_after_a,a_test_after_a(gen),a_test_after_b,b_test_after_b,forgetting,d_gen_vs_ctl,gen_better/tie/worse,d_forget_vs_ctl(+=less),forget_better/tie/worse,d_b_vs_ctl,gen_verdict,forget_verdict");
            let fmt = |s: genbench::Spread| {
                format!("{:.3}±{:.3}[{:.3};{:.3}]", s.mean, s.sd, s.min, s.max)
            };
            let mut unseen = 0.0;
            let mut agree = 0.0;
            for s in genbench::summarize(&rows) {
                unseen = s.test_edges_unseen_in_a;
                agree = s.ab_label_agreement;
                println!(
                    "{},{},{},{},{},{},{},{},{:+.3}±{:.3},{}/{}/{},{:+.3}±{:.3},{}/{}/{},{:+.3}±{:.3},{},{}",
                    s.init_weight,
                    s.arm.as_label(),
                    s.seeds,
                    fmt(s.a_train_after_a),
                    fmt(s.a_test_after_a),
                    fmt(s.a_test_after_b),
                    fmt(s.b_test_after_b),
                    fmt(s.forgetting),
                    s.generalisation_vs_control.delta.mean,
                    s.generalisation_vs_control.delta.sd,
                    s.generalisation_vs_control.better,
                    s.generalisation_vs_control.ties,
                    s.generalisation_vs_control.worse,
                    s.forgetting_vs_control.delta.mean,
                    s.forgetting_vs_control.delta.sd,
                    s.forgetting_vs_control.better,
                    s.forgetting_vs_control.ties,
                    s.forgetting_vs_control.worse,
                    s.b_vs_control.mean,
                    s.b_vs_control.sd,
                    s.generalisation_verdict.as_label(),
                    s.forgetting_verdict.as_label()
                );
            }
            println!("test_edges_unseen_in_a_mean={unseen:.3} ab_label_agreement_mean={agree:.3} topology_unchanged={}", rows.iter().all(|r| r.topology_unchanged));
            println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
        }
        Some("bench-eligibility-traces") => {
            let allowed = ["--graph", "--seeds", "--init", "--csv"];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(format!("unknown bench-eligibility-traces option: {flag}").into());
                }
            }
            let seeds = optional(&args, "--seeds")
                .unwrap_or("1,2,3,4,5,6,7,8,9,10")
                .split(',')
                .map(str::parse::<u64>)
                .collect::<Result<Vec<_>, _>>()?;
            let init = optional(&args, "--init").unwrap_or("1").parse::<u8>()?;
            let shape = genbench::GraphShape {
                sources: 64,
                targets: 24,
                fan_in: 6,
            };
            let spec = genbench::GenSpec {
                active: 4,
                train_patterns: 32,
                test_patterns: 64,
                rounds_a: 8,
                rounds_b: 8,
                positive_pct: 50,
                flip_pct: 50,
                init_weight: init,
            };
            let owned = match optional(&args, "--graph") {
                Some(path) => Some(Connectome::load(Path::new(path))?),
                None => None,
            };
            let graph_label = if owned.is_some() {
                "PROVIDED_GRAPH"
            } else {
                "GENERATED_SYNTHETIC"
            };
            let delays = et::DEFAULT_DELAYS;
            let rows = et::run_traces(owned.as_ref(), &shape, &spec, &seeds, &delays)?;
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(et::to_csv(&rows, graph_label).as_bytes())?;
            }
            println!("v888_eligibility_traces_schema=b888-4-eligibility-traces-v1");
            if owned.is_some() {
                println!("graph=PROVIDED_GRAPH (task labels artificial)");
            } else {
                println!("graph=GENERATED_SYNTHETIC sources=64 targets=24 fan_in=6 (not BANC; real BANC V888 graph DOI 10.7910/DVN/7WTH1N not available here)");
            }
            println!(
                "seeds={} init={init} active=4 train_patterns=32 test_patterns=64 rounds_a=8 rounds_b=8 flip_pct=50 threshold=1 metric=balanced_accuracy tag={} baseline_init={} baseline_rate=1/{} step_threshold={} clamp={} reactivate_after={}",
                seeds.len(),
                et::TAG,
                et::BASELINE_INIT,
                et::BASELINE_RATE,
                et::STEP_THRESHOLD,
                et::CREDIT_CLAMP,
                et::REACTIVATE_AFTER
            );
            let fmt = |s: genbench::Spread| format!("{:.3}±{:.3}", s.mean, s.sd);
            let mean = |arm: et::TraceArm, d: u32, f: &dyn Fn(&et::TraceRow) -> f64| {
                let v: Vec<f64> = rows
                    .iter()
                    .filter(|r| {
                        r.arm == arm && (arm == et::TraceArm::RewardCleanReference || r.delay == d)
                    })
                    .map(f)
                    .collect();
                genbench::spread(&v)
            };
            println!("arm_summary: condition,arm,a_train,a_test,forgetting,b_test,updates(A+B),weights_changed(A+B),reactivations(A+B)");
            let mut groups = vec![(
                "per-target-clean".to_string(),
                0,
                et::TraceArm::RewardCleanReference,
            )];
            for &d in &delays {
                for arm in [
                    et::TraceArm::Control,
                    et::TraceArm::Reward,
                    et::TraceArm::Trace,
                    et::TraceArm::TraceReact,
                ] {
                    groups.push((et::delay_label(d), d, arm));
                }
            }
            for (label, d, arm) in groups {
                println!(
                    "{label},{},{},{},{},{},{:.1},{:.1},{:.1}",
                    arm.as_label(),
                    fmt(mean(arm, d, &|r| r.a_train_after_a.balanced())),
                    fmt(mean(arm, d, &|r| r.a_test_after_a.balanced())),
                    fmt(mean(arm, d, &|r| r.forgetting())),
                    fmt(mean(arm, d, &|r| r.b_test_after_b.balanced())),
                    mean(arm, d, &|r| (r.cost_a.update_calls + r.cost_b.update_calls)
                        as f64)
                    .mean,
                    mean(
                        arm,
                        d,
                        &|r| (r.cost_a.weights_changed + r.cost_b.weights_changed) as f64
                    )
                    .mean,
                    mean(
                        arm,
                        d,
                        &|r| (r.cost_a.reactivations + r.cost_b.reactivations) as f64
                    )
                    .mean,
                );
            }
            println!("comparison: condition,x,y,n,x_a_test,y_a_test,d_gen,gen_b/t/w,gen_verdict,d_forget(+=x_less),forget_b/t/w,d_b,forget_verdict,x_reaches_control");
            for c in et::preregistered_comparisons(&rows, &delays) {
                println!(
                    "{},{},{},{},{},{},{:+.3}±{:.3},{}/{}/{},{},{:+.3}±{:.3},{}/{}/{},{:+.3}±{:.3},{},{}",
                    et::delay_label(c.delay),
                    c.x.as_label(),
                    c.y.as_label(),
                    c.n,
                    fmt(c.x_a_test),
                    fmt(c.y_a_test),
                    c.gen.delta.mean,
                    c.gen.delta.sd,
                    c.gen.better,
                    c.gen.ties,
                    c.gen.worse,
                    et::verdict_label(c.gen_verdict),
                    c.forget.delta.mean,
                    c.forget.delta.sd,
                    c.forget.better,
                    c.forget.ties,
                    c.forget.worse,
                    c.b.mean,
                    c.b.sd,
                    et::verdict_label(c.forget_verdict),
                    if c.y == et::TraceArm::Control { et::reaches_control(&c).to_string() } else { "n/a".to_string() },
                );
            }
            println!(
                "topology_unchanged={}",
                rows.iter().all(|r| r.topology_unchanged)
            );
            println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
        }
        Some("bench-graded-reward") => {
            let allowed = ["--graph", "--seeds", "--init", "--csv"];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(format!("unknown bench-graded-reward option: {flag}").into());
                }
            }
            let seeds = optional(&args, "--seeds")
                .unwrap_or("1,2,3,4,5,6,7,8,9,10")
                .split(',')
                .map(str::parse::<u64>)
                .collect::<Result<Vec<_>, _>>()?;
            let init = optional(&args, "--init").unwrap_or("1").parse::<u8>()?;
            let shape = genbench::GraphShape {
                sources: 64,
                targets: 24,
                fan_in: 6,
            };
            let spec = genbench::GenSpec {
                active: 4,
                train_patterns: 32,
                test_patterns: 64,
                rounds_a: 8,
                rounds_b: 8,
                positive_pct: 50,
                flip_pct: 50,
                init_weight: init,
            };
            let owned = match optional(&args, "--graph") {
                Some(path) => Some(Connectome::load(Path::new(path))?),
                None => None,
            };
            let graph_label = if owned.is_some() {
                "PROVIDED_GRAPH"
            } else {
                "GENERATED_SYNTHETIC"
            };
            let rows = gr::run_graded(owned.as_ref(), &shape, &spec, &seeds)?;
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(gr::to_csv(&rows, graph_label).as_bytes())?;
            }
            println!("v888_graded_reward_schema=b888-4-graded-reward-v1");
            if owned.is_some() {
                println!("graph=PROVIDED_GRAPH (task labels artificial)");
            } else {
                println!("graph=GENERATED_SYNTHETIC sources=64 targets=24 fan_in=6 (not BANC; real BANC V888 graph DOI 10.7910/DVN/7WTH1N not available here)");
            }
            println!(
                "seeds={} init={init} active=4 train_patterns=32 test_patterns=64 rounds_a=8 rounds_b=8 flip_pct=50 threshold=1 metric=balanced_accuracy tag={} baseline_init={} baseline_rate=1/{} step_threshold={} clamp={}",
                seeds.len(),
                et::TAG,
                et::BASELINE_INIT,
                et::BASELINE_RATE,
                et::STEP_THRESHOLD,
                et::CREDIT_CLAMP,
            );
            let fmt = |s: genbench::Spread| format!("{:.3}±{:.3}", s.mean, s.sd);
            let stat = |arm: gr::GradedArm, d: u32, f: &dyn Fn(&gr::GradedRow) -> f64| {
                let v: Vec<f64> = rows
                    .iter()
                    .filter(|r| r.arm == arm && r.delay == d)
                    .map(f)
                    .collect();
                genbench::spread(&v)
            };
            println!("arm_summary: arm,delay,a_train,a_test,forgetting,b_test,signals_per_episode,bits_per_episode,episodes_with_signal(A+B),weights_changed(A+B),steps_up(A+B),steps_down(A+B)");
            for (arm, d) in gr::preregistered_arms() {
                println!(
                    "{},{d},{},{},{},{},{:.3},{:.3},{:.1},{:.1},{:.1},{:.1}",
                    arm.label(),
                    fmt(stat(arm, d, &|r| r.a_train_after_a.balanced())),
                    fmt(stat(arm, d, &|r| r.a_test_after_a.balanced())),
                    fmt(stat(arm, d, &|r| r.forgetting())),
                    fmt(stat(arm, d, &|r| r.b_test_after_b.balanced())),
                    stat(arm, d, &gr::GradedRow::signals_per_episode).mean,
                    stat(arm, d, &gr::GradedRow::bits_per_episode).mean,
                    stat(arm, d, &|r| (r.cost_a.episodes_with_signal
                        + r.cost_b.episodes_with_signal)
                        as f64)
                    .mean,
                    stat(arm, d, &|r| (r.cost_a.trace.weights_changed
                        + r.cost_b.trace.weights_changed)
                        as f64)
                    .mean,
                    stat(
                        arm,
                        d,
                        &|r| (r.cost_a.trace.steps_up + r.cost_b.trace.steps_up) as f64
                    )
                    .mean,
                    stat(
                        arm,
                        d,
                        &|r| (r.cost_a.trace.steps_down + r.cost_b.trace.steps_down) as f64
                    )
                    .mean,
                );
            }
            println!("comparison: x,dx,y,dy,n,x_a_test,y_a_test,d_gen,gen_b/t/w,gen_verdict,d_forget(+=x_less),forget_b/t/w,d_b,forget_verdict,x_reaches_control");
            for c in gr::preregistered_comparisons(&rows) {
                println!(
                    "{},{},{},{},{},{},{},{:+.3}±{:.3},{}/{}/{},{},{:+.3}±{:.3},{}/{}/{},{:+.3}±{:.3},{},{}",
                    c.x.0.label(),
                    c.x.1,
                    c.y.0.label(),
                    c.y.1,
                    c.n,
                    fmt(c.x_a_test),
                    fmt(c.y_a_test),
                    c.gen.delta.mean,
                    c.gen.delta.sd,
                    c.gen.better,
                    c.gen.ties,
                    c.gen.worse,
                    et::verdict_label(c.gen_verdict),
                    c.forget.delta.mean,
                    c.forget.delta.sd,
                    c.forget.better,
                    c.forget.ties,
                    c.forget.worse,
                    c.b.mean,
                    c.b.sd,
                    et::verdict_label(c.forget_verdict),
                    if c.y == (gr::GradedArm::Reference(et::TraceArm::Control), 0) { gr::reaches_control(&c).to_string() } else { "n/a".to_string() },
                );
            }
            let chain: Vec<String> = gr::CHAIN
                .iter()
                .map(|&g| {
                    format!(
                        "{}={:.3}",
                        g.label(),
                        gr::mean_a_test(&rows, gr::GradedArm::Ladder(g), 0)
                    )
                })
                .collect();
            println!("h1_chain_a_test: {}", chain.join(" "));
            println!("h1_reading={:?}", gr::read_h1(&rows));
            println!(
                "topology_unchanged={}",
                rows.iter().all(|r| r.topology_unchanged)
            );
            println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
        }
        Some("bench-reward-limits") => {
            let allowed = ["--graph", "--seeds", "--init", "--csv"];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(format!("unknown bench-reward-limits option: {flag}").into());
                }
            }
            let seeds = optional(&args, "--seeds")
                .unwrap_or("1,2,3,4,5,6,7,8,9,10")
                .split(',')
                .map(str::parse::<u64>)
                .collect::<Result<Vec<_>, _>>()?;
            let init = optional(&args, "--init").unwrap_or("1").parse::<u8>()?;
            let shape = genbench::GraphShape {
                sources: 64,
                targets: 24,
                fan_in: 6,
            };
            let spec = genbench::GenSpec {
                active: 4,
                train_patterns: 32,
                test_patterns: 64,
                rounds_a: 8,
                rounds_b: 8,
                positive_pct: 50,
                flip_pct: 50,
                init_weight: init,
            };
            let owned = match optional(&args, "--graph") {
                Some(path) => Some(Connectome::load(Path::new(path))?),
                None => None,
            };
            let graph_label = if owned.is_some() {
                "PROVIDED_GRAPH"
            } else {
                "GENERATED_SYNTHETIC"
            };
            let conditions = rl::default_conditions();
            let rows = rl::run_limits(owned.as_ref(), &shape, &spec, &seeds, &conditions)?;
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(rl::to_csv(&rows, graph_label).as_bytes())?;
            }
            println!("v888_reward_limits_schema=b888-4-reward-limits-v1");
            if owned.is_some() {
                println!("graph=PROVIDED_GRAPH (task labels artificial)");
            } else {
                println!("graph=GENERATED_SYNTHETIC sources=64 targets=24 fan_in=6 (not BANC; real BANC V888 graph DOI 10.7910/DVN/7WTH1N not available here)");
            }
            println!(
                "seeds={} init={init} active=4 train_patterns=32 test_patterns=64 rounds_a=8 rounds_b=8 flip_pct=50 threshold=1 metric=balanced_accuracy",
                seeds.len()
            );
            println!("condition,seeds,reward_a_test,control_a_test,reward_vs_clean_reward,hurts_reward,d_gen_vs_ctl,gen_b/t/w,reward_forgetting,control_forgetting,d_forget_vs_ctl(+=less),forget_b/t/w,reward_b_test,control_b_test,d_b_vs_ctl,gen_verdict,forget_verdict,reward_updates,control_updates");
            let fmt = |s: genbench::Spread| format!("{:.3}±{:.3}", s.mean, s.sd);
            for s in rl::summarize(&rows) {
                println!(
                    "{},{},{},{},{:+.3}±{:.3},{},{:+.3}±{:.3},{}/{}/{},{},{},{:+.3}±{:.3},{}/{}/{},{},{},{:+.3}±{:.3},{},{},{:.1},{:.1}",
                    s.condition.label(),
                    s.seeds,
                    fmt(s.reward_a_test),
                    fmt(s.control_a_test),
                    s.reward_vs_clean.mean,
                    s.reward_vs_clean.sd,
                    s.hurts_reward,
                    s.gen_vs_control.delta.mean,
                    s.gen_vs_control.delta.sd,
                    s.gen_vs_control.better,
                    s.gen_vs_control.ties,
                    s.gen_vs_control.worse,
                    fmt(s.reward_forgetting),
                    fmt(s.control_forgetting),
                    s.forget_vs_control.delta.mean,
                    s.forget_vs_control.delta.sd,
                    s.forget_vs_control.better,
                    s.forget_vs_control.ties,
                    s.forget_vs_control.worse,
                    fmt(s.reward_b_test),
                    fmt(s.control_b_test),
                    s.b_vs_control.mean,
                    s.b_vs_control.sd,
                    s.gen_verdict.as_label(),
                    s.forget_verdict.as_label(),
                    s.reward_update_calls,
                    s.control_update_calls,
                );
            }
            println!(
                "topology_unchanged={}",
                rows.iter().all(|r| r.topology_unchanged)
            );
            println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
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

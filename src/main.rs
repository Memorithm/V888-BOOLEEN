use std::error::Error;
use std::fs::File;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader};
use std::path::Path;
use v888_booleen::circuit_explorer::{ActivityReport, CircuitExplorer, RegionLabels, Routing};
use v888_booleen::connectome::Connectome;
use v888_booleen::consolidated_graded as cg;
use v888_booleen::consolidation::{
    matrix as consolidation_matrix, sweep as consolidation_sweep, to_csv as consolidation_csv,
    Scenario, ALL_STRATEGIES,
};
use v888_booleen::consolidation_scale as scale;
use v888_booleen::context_cue as cc;
use v888_booleen::cue_displacement as cdp;
use v888_booleen::cue_gating as cgate;
use v888_booleen::cue_init as ci;
use v888_booleen::cue_reach as cr;
use v888_booleen::eligibility_trace as et;
use v888_booleen::engine::BooleanEngine;
use v888_booleen::error_gated as eg;
use v888_booleen::full_synapse::{pack_records, FullSynapseReader};
use v888_booleen::generalization_bench as genbench;
use v888_booleen::graded_reward as gr;
use v888_booleen::importance_protection as ip;
use v888_booleen::learning_bench::benchmark;
use v888_booleen::learning_strategies::compare_strategies;
use v888_booleen::novel_split as ns;
use v888_booleen::plasticity::PlasticCircuit;
use v888_booleen::reward_limits as rl;
use v888_booleen::seq_generalisation as sgen;
use v888_booleen::source_holdout::benchmark_disjoint_sources;
use v888_booleen::synapse_pairs::{pack, propagate_one_step, SynapsePairsReader};
use v888_booleen::task_sequence as ts;
use v888_booleen::write_allocation as wa;
use v888_booleen::write_budget as wb;
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
    println!("  bench-consolidated-graded [--graph brain.v888b] [--seeds 1,..,10] [--init 1] [--csv out.csv]  (consolidation 1-of-4 + bounded 1-in-8 A rehearsal on supervised control, trace-group-8, trace-per-target; equal budget; GENERATED synthetic graph by default)");
    println!("  bench-importance-protection [--graph brain.v888b] [--seeds 1,..,10] [--init 1] [--csv out.csv]  (importance-weighted freeze of top-credit A edges 25%/50% + task-gated plasticity vs #39 consolidate; equal budget; GENERATED synthetic graph by default)");
    println!("  bench-error-gated [--graph brain.v888b] [--seeds 1,..,20] [--init 1] [--csv out.csv]  (error-gated / margin-gated-4 supervised updates vs matched supervised control; agree/conflict forgetting split; bootstrap CIs; equal budget; GENERATED synthetic graph by default)");
    println!("  bench-context-cue [--graph brain.v888b] [--seeds 1,..,20] [--flips 25,50,100] [--init 1] [--csv out.csv]  (task-context cue on reserved existing input lines vs no cue, supervised and error-gated; conflict-pair floor; structural ceiling; bootstrap CIs; equal budget; GENERATED synthetic graph by default)");
    println!("  bench-task-sequence [--graph brain.v888b] [--seeds 41,..,80] [--flips 25,50,100] [--init 1] [--csv out.csv]  (A->B->C->D task sequences with one context cue per task; #121 gates and a margin-gate sweep; novel-configuration TEST slice; equal budget; bootstrap CIs; GENERATED synthetic graph by default)");
    println!("  bench-novel-split [--graph brain.v888b] [--seeds 81,..,120] [--flips 25,50,100] [--census] [--csv out.csv]  (B888-4 cycle 12: wider pattern pool, cue cap 4, TEST pairs split into seen / compositional-novel / unseen-edge per task; design-validity gate on novel-pair count; E/E-cue vs error-gated vs supervised at equal budget; GENERATED synthetic graph by default)");
    println!("  bench-seq-generalisation [--graph brain.v888b] [--seeds 321,..,360] [--flips 25,50,100] [--k 8] [--heldout-orders 2] [--threads N] [--csv out.csv] [--census]  (B888-4 cycle 18: GENERALISATION TO LONGER SEQUENCES AND HELD-OUT TASK ORDERS — supervised/reward/error-gated/E/E-cue0 at paced100 and error-gated@Z-pattern on k tasks (64 reserved cue lines split over the k cues), identity order + seeded held-out orders; --census = identity order, writes/refusals and pair counts only, no accuracy; GENERATED synthetic graph by default)");
    println!("  bench-cue-displacement [--graph brain.v888b] [--seeds 281,..,320] [--flips 25,50,100] [--csv out.csv] [--census]  (B888-4 cycle 17: CUE WRITE DISPLACEMENT — E/E-cue0 at paced100 vs the same arm with cue writes not charged (E/E-cue0-exempt) and vs error-gated capped at the pattern-write curve E/E-cue0 left (error-gated@Z-pattern); supervised/reward/error-gated references; CN generalisation and forgetting; --census = writes/refusals only, no accuracy; GENERATED synthetic graph by default)");
    println!("  bench-cue-init [--graph brain.v888b] [--seeds 241,..,280] [--flips 25,50,100] [--budgets free,paced100] [--csv out.csv] [--census]  (B888-4 cycle 16: CUE EDGES START SILENT — E/E-cue@16 with cue-line edges at initial weight 0 (source-major, + pattern-first secondary) vs #127 E/E-cue (init 1), error-gated, supervised and per-target reward under an equal weight-write budget paced to uncapped error-gated; CN generalisation and forgetting; --census = writes/cue writes/refusals only, no accuracy; GENERATED synthetic graph by default)");
    println!("  bench-write-allocation [--graph brain.v888b] [--seeds 201,..,240] [--flips 25,50,100] [--budgets free,paced100] [--csv out.csv] [--census]  (B888-4 cycle 15: WRITE ALLOCATION of E/E-cue@16 under an equal weight-write budget paced to uncapped error-gated — source-major (#126) / pattern-first / cue-first funding of the offered steps, vs error-gated / supervised / per-target reward, plus secondary E-tag (task-local negative tags); CN generalisation and forgetting; --census = writes/cue writes/refusals only, no accuracy; GENERATED synthetic graph by default)");
    println!("  bench-write-budget [--graph brain.v888b] [--seeds 161,..,200] [--flips 25,50,75,100] [--budgets free,paced100,total100,paced50] [--csv out.csv]  (B888-4 cycle 14: EQUAL WEIGHT-WRITE budget per task, capped at a fraction of the uncapped error-gated writes; supervised / error-gated / E/E-cue@16 / per-target reward on the #124 fixed-pool novel-split design over a flip sweep; CN generalisation and forgetting; bootstrap CIs; GENERATED synthetic graph by default)");
    println!("  bench-cue-reach [--graph brain.v888b] [--seeds 121,..,160] [--flips 25,50,100] [--levels 2,4,8,16] [--census] [--no-coupled] [--csv out.csv]  (B888-4 cycle 13: cue size / target reach varied at a FIXED pattern pool on the #123 novel-split design; E/E-cue minus error-gated gain on all / novel pairs and forgetting per level; coupled #123-style pool-vs-reach trade-off; equal budget; bootstrap CIs; GENERATED synthetic graph by default)");
    println!("  bench-cue-gating [--graph brain.v888b] [--seeds 1,..,40] [--flips 25,50,100] [--init 1] [--csv out.csv]  (separate pattern-edge / cue-edge gates and cue-line count under the task-context cue; S0 structural check; equal budget; bootstrap CIs; GENERATED synthetic graph by default)");
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
        Some("bench-importance-protection") => {
            let allowed = ["--graph", "--seeds", "--init", "--csv"];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(
                        format!("unknown bench-importance-protection option: {flag}").into(),
                    );
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
            let rows = ip::run_grid(owned.as_ref(), &shape, &spec, &seeds)?;
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(ip::to_csv(&rows, graph_label).as_bytes())?;
            }
            println!("v888_importance_protection_schema=b888-4-importance-protection-v1");
            if owned.is_some() {
                println!("graph=PROVIDED_GRAPH (task labels artificial)");
            } else {
                println!("graph=GENERATED_SYNTHETIC sources=64 targets=24 fan_in=6 (not BANC; real BANC V888 graph DOI 10.7910/DVN/7WTH1N not available here)");
            }
            println!(
                "seeds={} init={init} active=4 train_patterns=32 test_patterns=64 rounds_a=8 rounds_b=8 flip_pct=50 threshold=1 metric=balanced_accuracy consolidate_period={} gate_salt={:#x}",
                seeds.len(),
                ip::CONSOLIDATE_PERIOD,
                ip::GATE_SALT,
            );
            let fmt = |s: genbench::Spread| format!("{:.3}±{:.3}", s.mean, s.sd);
            let stat = |l: cg::Learner, m: ip::Mechanism, f: &dyn Fn(&ip::Row) -> f64| {
                let v: Vec<f64> = rows
                    .iter()
                    .filter(|r| r.learner == l && r.mechanism == m)
                    .map(f)
                    .collect();
                genbench::spread(&v)
            };
            println!("arm_summary: learner,mechanism,a_train,a_test_after_a,a_test_after_b,forgetting,b_test,protected_edges,weights_changed(A+B),a_blocked,b_protected_attempts,b_blocked");
            for l in cg::LEARNERS {
                for m in ip::MECHANISMS {
                    println!(
                        "{},{},{},{},{},{},{},{:.1},{:.1},{:.1},{:.1},{:.1}",
                        l.label(),
                        m.label(),
                        fmt(stat(l, m, &|r| r.a_train_after_a.balanced())),
                        fmt(stat(l, m, &|r| r.a_test_after_a.balanced())),
                        fmt(stat(l, m, &|r| r.a_test_after_b.balanced())),
                        fmt(stat(l, m, &ip::Row::forgetting)),
                        fmt(stat(l, m, &|r| r.b_test_after_b.balanced())),
                        stat(l, m, &|r| r.protected_edges as f64).mean,
                        stat(
                            l,
                            m,
                            &|r| (r.cost_a.weights_changed + r.cost_b.weights_changed) as f64
                        )
                        .mean,
                        stat(l, m, &|r| r.a_blocked as f64).mean,
                        stat(l, m, &|r| r.cost_b.protected_attempts as f64).mean,
                        stat(l, m, &|r| r.cost_b.blocked as f64).mean,
                    );
                }
            }
            println!("comparison: x_learner,x_mech,y_learner,y_mech,n,d_a_test,d_forget(+=x_less),forget_b/t/w,d_retained_a,d_b,trade_sum,trade_pos/t/neg,passes_rule,better_exchange_h3,tradeoff_vs_control_h4");
            for c in ip::preregistered_comparisons(&rows) {
                let own = c.y.0 == c.x.0;
                let vs_ctl = c.y == (cg::Learner::Control, ip::Mechanism::None)
                    && c.x.0 != cg::Learner::Control;
                println!(
                    "{},{},{},{},{},{:+.3}±{:.3},{:+.3}±{:.3},{}/{}/{},{:+.3}±{:.3},{:+.3}±{:.3},{:+.3}±{:.3},{}/{}/{},{},{},{}",
                    c.x.0.label(),
                    c.x.1.label(),
                    c.y.0.label(),
                    c.y.1.label(),
                    c.n,
                    c.a_test.mean,
                    c.a_test.sd,
                    c.forget.delta.mean,
                    c.forget.delta.sd,
                    c.forget.better,
                    c.forget.ties,
                    c.forget.worse,
                    c.retained.mean,
                    c.retained.sd,
                    c.b.mean,
                    c.b.sd,
                    c.trade.delta.mean,
                    c.trade.delta.sd,
                    c.trade.better,
                    c.trade.ties,
                    c.trade.worse,
                    if own { c.passes().to_string() } else { "n/a".to_string() },
                    if own { c.better_exchange().to_string() } else { "n/a".to_string() },
                    if vs_ctl { c.beats_control_tradeoff().to_string() } else { "n/a".to_string() },
                );
            }
            let (h1, h2, h3, h4) = ip::readings(&rows);
            println!("H1_importance25_passes_both_graded={h1}");
            println!("H2_task_gated_passes_both_graded={h2}");
            println!("H3_better_exchange_than_1to1={h3}");
            println!("H4_tradeoff_vs_control={h4}");
            println!(
                "topology_unchanged={}",
                rows.iter().all(|r| r.topology_unchanged)
            );
            println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
        }
        Some("bench-error-gated") => {
            let allowed = ["--graph", "--seeds", "--init", "--csv"];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(format!("unknown bench-error-gated option: {flag}").into());
                }
            }
            let seeds = optional(&args, "--seeds")
                .unwrap_or("1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
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
            let rows = eg::run_grid(owned.as_ref(), &shape, &spec, &seeds)?;
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(eg::to_csv(&rows, graph_label).as_bytes())?;
            }
            println!("v888_error_gated_schema=b888-4-error-gated-v1");
            if owned.is_some() {
                println!("graph=PROVIDED_GRAPH (task labels artificial)");
            } else {
                println!("graph=GENERATED_SYNTHETIC sources=64 targets=24 fan_in=6 (not BANC; real BANC V888 graph DOI 10.7910/DVN/7WTH1N not available here)");
            }
            println!(
                "seeds={} init={init} active=4 train_patterns=32 test_patterns=64 rounds_a=8 rounds_b=8 flip_pct=50 threshold={} metric=balanced_accuracy margin={} boot_resamples={} boot_seed={:#x}",
                seeds.len(),
                eg::THRESHOLD,
                eg::MARGIN,
                eg::BOOT_RESAMPLES,
                eg::BOOT_SEED,
            );
            let fmt = |s: genbench::Spread| format!("{:.3}±{:.3}", s.mean, s.sd);
            let stat = |g: eg::Gate, f: &dyn Fn(&eg::Row) -> f64| {
                let v: Vec<f64> = rows.iter().filter(|r| r.gate == g).map(f).collect();
                genbench::spread(&v)
            };
            println!("arm_summary: arm,a_train,a_test_after_a,a_test_after_b,forgetting,b_test,agree_forgetting,conflict_forgetting,agree_b_test,conflict_b_test,targets_updated(A+B),attempts(A+B),weights_changed(A+B),saturated_after_a,saturated_after_b");
            for g in eg::ARMS {
                println!(
                    "{},{},{},{},{},{},{},{},{},{},{:.1},{:.1},{:.1},{:.1},{:.1}",
                    g.label(),
                    fmt(stat(g, &|r| r.a_train_after_a.balanced())),
                    fmt(stat(g, &|r| r.a_test_after_a.all.balanced())),
                    fmt(stat(g, &|r| r.a_test_after_b.all.balanced())),
                    fmt(stat(g, &eg::Row::forgetting)),
                    fmt(stat(g, &|r| r.b_test_after_b.all.balanced())),
                    fmt(stat(g, &eg::Row::forgetting_agree)),
                    fmt(stat(g, &eg::Row::forgetting_conflict)),
                    fmt(stat(g, &|r| r.b_test_after_b.agree.balanced())),
                    fmt(stat(g, &|r| r.b_test_after_b.conflict.balanced())),
                    stat(
                        g,
                        &|r| (r.cost_a.targets_updated + r.cost_b.targets_updated) as f64
                    )
                    .mean,
                    stat(g, &|r| (r.cost_a.attempts + r.cost_b.attempts) as f64).mean,
                    stat(
                        g,
                        &|r| (r.cost_a.weights_changed + r.cost_b.weights_changed) as f64
                    )
                    .mean,
                    stat(g, &|r| r.saturated_after_a as f64).mean,
                    stat(g, &|r| r.saturated_after_b as f64).mean,
                );
            }
            let p = |d: &eg::Paired| {
                format!(
                    "{:+.3}±{:.3}[{:+.3},{:+.3}]{}/{}/{}",
                    d.mean, d.sd, d.ci_lo, d.ci_hi, d.better, d.ties, d.worse
                )
            };
            println!("comparison_vs_supervised: x,n,d_forget(+=x_less),d_forget_agree,d_forget_conflict,d_a_test,d_retained_a,d_b_test,trade_sum,passes_rule,generalises  (each: mean±sd[boot95_lo,hi]better/tie/worse)");
            for c in eg::preregistered_comparisons(&rows) {
                println!(
                    "{},{},{},{},{},{},{},{},{},{},{}",
                    c.x.label(),
                    c.forget.n,
                    p(&c.forget),
                    p(&c.forget_agree),
                    p(&c.forget_conflict),
                    p(&c.a_test),
                    p(&c.retained),
                    p(&c.b),
                    p(&c.trade),
                    c.passes(),
                    c.generalises(),
                );
            }
            println!(
                "control_agree_forgetting={}",
                p(&eg::control_agree_forgetting(&rows))
            );
            let (h1, h2, h3, h4) = eg::readings(&rows);
            println!("H1_error_gated_passes={h1}");
            println!("H2_margin_gated_4_passes={h2}");
            println!("H3_generalisation_gain={h3}");
            println!("H4_control_agree_forgetting_avoidable={h4}");
            println!(
                "topology_unchanged={}",
                rows.iter().all(|r| r.topology_unchanged)
            );
            println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
        }
        Some("bench-context-cue") => {
            let allowed = ["--graph", "--seeds", "--flips", "--init", "--csv"];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(format!("unknown bench-context-cue option: {flag}").into());
                }
            }
            let seeds = optional(&args, "--seeds")
                .unwrap_or("1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
                .split(',')
                .map(str::parse::<u64>)
                .collect::<Result<Vec<_>, _>>()?;
            let flips = optional(&args, "--flips")
                .unwrap_or("25,50,100")
                .split(',')
                .map(str::parse::<u32>)
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
                flip_pct: cc::PRIMARY_FLIP,
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
            let rows = cc::run_grid(owned.as_ref(), &shape, &spec, &flips, &seeds)?;
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(cc::to_csv(&rows, graph_label).as_bytes())?;
            }
            println!("v888_context_cue_schema=b888-4-context-cue-v1");
            if owned.is_some() {
                println!("graph=PROVIDED_GRAPH (task labels artificial)");
            } else {
                println!("graph=GENERATED_SYNTHETIC sources=64 targets=24 fan_in=6 (not BANC; real BANC V888 graph DOI 10.7910/DVN/7WTH1N not available here)");
            }
            println!(
                "seeds={} flips={flips:?} init={init} active=4 train_patterns=32 test_patterns=64 rounds_a=8 rounds_b=8 threshold={} cue_cap={} metric=balanced_accuracy boot_resamples={} boot_seed={:#x}",
                seeds.len(),
                cc::THRESHOLD,
                cc::CUE_CAP,
                eg::BOOT_RESAMPLES,
                eg::BOOT_SEED,
            );
            let fmt = |s: genbench::Spread| format!("{:.3}±{:.3}", s.mean, s.sd);
            let stat = |flip: u32, a: cc::Arm, f: &dyn Fn(&cc::Row) -> f64| {
                let v: Vec<f64> = rows
                    .iter()
                    .filter(|r| r.flip == flip && r.arm == a)
                    .map(f)
                    .collect();
                genbench::spread(&v)
            };
            println!("arm_summary: flip,arm,cue_lines_a/b,cue_reach_a/b,a_train,a_test_after_a,a_test_after_b,forgetting,b_test,agree_forgetting,conflict_forgetting,conflict_b_test,conflict_both_correct,ceiling,weights_changed(A+B),cue_weight_mean");
            for &flip in &flips {
                for a in cc::ARMS {
                    println!(
                        "{flip},{},{:.1}/{:.1},{:.1}/{:.1},{},{},{},{},{},{},{},{},{},{},{:.1},{:.2}",
                        a.label(),
                        stat(flip, a, &|r| r.cue_lines_a as f64).mean,
                        stat(flip, a, &|r| r.cue_lines_b as f64).mean,
                        stat(flip, a, &|r| r.cue_reach_a as f64).mean,
                        stat(flip, a, &|r| r.cue_reach_b as f64).mean,
                        fmt(stat(flip, a, &|r| r.a_train_after_a.balanced())),
                        fmt(stat(flip, a, &|r| r.a_test_after_a.all.balanced())),
                        fmt(stat(flip, a, &|r| r.a_test_after_b.all.balanced())),
                        fmt(stat(flip, a, &cc::Row::forgetting)),
                        fmt(stat(flip, a, &|r| r.b_test_after_b.all.balanced())),
                        fmt(stat(flip, a, &cc::Row::forgetting_agree)),
                        fmt(stat(flip, a, &cc::Row::forgetting_conflict)),
                        fmt(stat(flip, a, &|r| r.b_test_after_b.conflict.balanced())),
                        fmt(stat(flip, a, &cc::Row::both_frac)),
                        fmt(stat(flip, a, &cc::Row::ceiling_frac)),
                        stat(
                            flip,
                            a,
                            &|r| (r.cost_a.weights_changed + r.cost_b.weights_changed) as f64
                        )
                        .mean,
                        stat(flip, a, &|r| r.cue_weight_mean).mean,
                    );
                }
            }
            let p = |d: &eg::Paired| {
                format!(
                    "{:+.3}±{:.3}[{:+.3},{:+.3}]{}/{}/{}",
                    d.mean, d.sd, d.ci_lo, d.ci_hi, d.better, d.ties, d.worse
                )
            };
            println!("comparison: flip,x,base,n,d_forget(+=x_less),d_forget_agree,d_forget_conflict,d_a_test,d_retained_a,d_b_test,d_b_conflict,d_both_correct,passes_rule,breaks_floor  (each: mean±sd[boot95_lo,hi]better/tie/worse)");
            for &flip in &flips {
                let mut cmps = cc::cue_comparisons(&rows, flip);
                cmps.push(cc::compare(&rows, flip, cc::ERROR_GATED, cc::SUPERVISED));
                cmps.push(cc::compare(
                    &rows,
                    flip,
                    cc::ERROR_GATED_CUE,
                    cc::SUPERVISED,
                ));
                for c in cmps {
                    println!(
                        "{flip},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                        c.x.label(),
                        c.base.label(),
                        c.forget.n,
                        p(&c.forget),
                        p(&c.forget_agree),
                        p(&c.forget_conflict),
                        p(&c.a_test),
                        p(&c.retained),
                        p(&c.b),
                        p(&c.b_conflict),
                        p(&c.both),
                        c.passes(),
                        c.breaks_floor(),
                    );
                }
            }
            let (h1, h2, h3, h4) = cc::readings(&rows);
            println!("H1_supervised_cue_passes_flip50={h1}");
            println!("H2_error_gated_cue_passes_flip50={h2}");
            println!("H3_conflict_floor_broken_flip50={h3}");
            println!("H4_cue_passes_at_flip25_or_flip100={h4}");
            println!(
                "topology_unchanged={}",
                rows.iter().all(|r| r.topology_unchanged)
            );
            println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
        }
        Some("bench-cue-gating") => {
            let allowed = ["--graph", "--seeds", "--flips", "--init", "--csv"];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(format!("unknown bench-cue-gating option: {flag}").into());
                }
            }
            let seeds: Vec<u64> = match optional(&args, "--seeds") {
                Some(s) => s
                    .split(',')
                    .map(str::parse::<u64>)
                    .collect::<Result<Vec<_>, _>>()?,
                None => cgate::SEEDS.collect(),
            };
            let flips = optional(&args, "--flips")
                .unwrap_or("25,50,100")
                .split(',')
                .map(str::parse::<u32>)
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
                flip_pct: cgate::PRIMARY_FLIP,
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
            let rows =
                cgate::run_grid(owned.as_ref(), &shape, &spec, &flips, &seeds, &cgate::ARMS)?;
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(cgate::to_csv(&rows, graph_label).as_bytes())?;
            }
            println!("v888_cue_gating_schema=b888-4-cue-gating-v1");
            if owned.is_some() {
                println!("graph=PROVIDED_GRAPH (task labels artificial)");
            } else {
                println!("graph=GENERATED_SYNTHETIC sources=64 targets=24 fan_in=6 (not BANC; real BANC V888 graph DOI 10.7910/DVN/7WTH1N not available here)");
            }
            println!(
                "seeds={} flips={flips:?} init={init} active=4 train_patterns=32 test_patterns=64 rounds_a=8 rounds_b=8 threshold={} cue_cap={} metric=balanced_accuracy boot_resamples={} boot_seed={:#x}",
                seeds.len(),
                cgate::THRESHOLD,
                cc::CUE_CAP,
                eg::BOOT_RESAMPLES,
                eg::BOOT_SEED,
            );
            let fmt = |s: genbench::Spread| format!("{:.3}±{:.3}", s.mean, s.sd);
            let stat = |flip: u32, a: cgate::Arm, f: &dyn Fn(&cgate::Row) -> f64| {
                let v: Vec<f64> = rows
                    .iter()
                    .filter(|r| r.flip == flip && r.arm == a)
                    .map(f)
                    .collect();
                genbench::spread(&v)
            };
            println!("arm_summary: flip,arm,cue_lines_a/b,cue_reach_a/b,a_train,a_test_after_a,a_tpr_after_a,a_tnr_after_a,a_test_after_b,a_tpr_after_b,forgetting,b_test,conflict_forgetting,conflict_both_correct,ceiling,weights_changed(A+B),cue_a_weight_after_a,cue_b_weight_after_b,pattern_equals_control_rows");
            for &flip in &flips {
                for a in cgate::ARMS {
                    println!(
                        "{flip},{},{:.1}/{:.1},{:.1}/{:.1},{},{},{},{},{},{},{},{},{},{},{},{:.1},{:.2},{:.2},{}",
                        a.label(),
                        stat(flip, a, &|r| r.cue_lines_a as f64).mean,
                        stat(flip, a, &|r| r.cue_lines_b as f64).mean,
                        stat(flip, a, &|r| r.cue_reach_a as f64).mean,
                        stat(flip, a, &|r| r.cue_reach_b as f64).mean,
                        fmt(stat(flip, a, &|r| r.a_train_after_a.balanced())),
                        fmt(stat(flip, a, &|r| r.a_test_after_a.all.balanced())),
                        fmt(stat(flip, a, &|r| cgate::tpr(&r.a_test_after_a.all))),
                        fmt(stat(flip, a, &|r| cgate::tnr(&r.a_test_after_a.all))),
                        fmt(stat(flip, a, &|r| r.a_test_after_b.all.balanced())),
                        fmt(stat(flip, a, &|r| cgate::tpr(&r.a_test_after_b.all))),
                        fmt(stat(flip, a, &cgate::Row::forgetting)),
                        fmt(stat(flip, a, &|r| r.b_test_after_b.all.balanced())),
                        fmt(stat(flip, a, &cgate::Row::forgetting_conflict)),
                        fmt(stat(flip, a, &cgate::Row::both_frac)),
                        fmt(stat(flip, a, &cgate::Row::ceiling_frac)),
                        stat(
                            flip,
                            a,
                            &|r| (r.cost_a.weights_changed + r.cost_b.weights_changed) as f64
                        )
                        .mean,
                        stat(flip, a, &|r| r.cue_a_after_a).mean,
                        stat(flip, a, &|r| r.cue_b_after_b).mean,
                        rows.iter()
                            .filter(|r| r.flip == flip && r.arm == a && r.pattern_equals_control)
                            .count(),
                    );
                }
            }
            let p = |d: &eg::Paired| {
                format!(
                    "{:+.3}±{:.3}[{:+.3},{:+.3}]{}/{}/{}",
                    d.mean, d.sd, d.ci_lo, d.ci_hi, d.better, d.ties, d.worse
                )
            };
            let low: Vec<u64> = seeds.iter().copied().filter(|&s| s <= 20).collect();
            let high: Vec<u64> = seeds.iter().copied().filter(|&s| s > 20).collect();
            println!("comparison: flip,seeds,x,base,n,d_forget(+=x_less),d_forget_conflict,d_a_test,d_a_tpr,d_a_tnr,d_retained_a,d_b_test,d_both_correct,passes_rule  (each: mean±sd[boot95_lo,hi]better/tie/worse)");
            for &flip in &flips {
                let mut cmps = vec![
                    (cgate::SS, cgate::SUPERVISED),
                    (cgate::SE, cgate::SUPERVISED),
                    (cgate::SS_K1, cgate::SUPERVISED),
                    (cgate::SS_K2, cgate::SUPERVISED),
                    (cgate::SS_K4, cgate::SUPERVISED),
                    (cgate::ES, cgate::ERROR_GATED),
                    (cgate::EE, cgate::ERROR_GATED),
                    (cgate::ES, cgate::SUPERVISED),
                    (cgate::EE, cgate::SUPERVISED),
                    (cgate::ERROR_GATED, cgate::SUPERVISED),
                ];
                cmps.push((cgate::ES, cgate::EE));
                for (label, subset) in [
                    ("all", None),
                    ("1-20", Some(low.as_slice())),
                    ("21-40", Some(high.as_slice())),
                ] {
                    if subset.is_some_and(<[u64]>::is_empty) {
                        continue;
                    }
                    for &(x, base) in &cmps {
                        let c = cgate::compare(&rows, flip, x, base, subset);
                        println!(
                            "{flip},{label},{},{},{},{},{},{},{},{},{},{},{},{}",
                            x.label(),
                            base.label(),
                            c.forget.n,
                            p(&c.forget),
                            p(&c.forget_conflict),
                            p(&c.a_test),
                            p(&c.a_tpr),
                            p(&c.a_tnr),
                            p(&c.retained),
                            p(&c.b),
                            p(&c.both),
                            c.passes(),
                        );
                    }
                    println!(
                        "{flip},{label},H4_attribution,{}",
                        p(&cgate::h4_attribution(&rows, flip, subset))
                    );
                }
            }
            let (s0, h1, h2, h3, h4) = cgate::readings(&rows);
            println!("S0_structural_supervised_pattern_gate_holds={s0}");
            println!("H1_S/E-cue_passes_vs_supervised_flip50={h1}");
            println!("H2_some_S/S-cue-k_passes_vs_supervised_flip50={h2}");
            println!("H3_E/S-cue_passes_vs_error-gated_flip50={h3}");
            println!("H4_pattern_gate_carries_cue_benefit_flip50={h4}");
            println!(
                "topology_unchanged={}",
                rows.iter().all(|r| r.topology_unchanged)
            );
            println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
        }
        Some("bench-novel-split") => {
            let allowed = ["--graph", "--seeds", "--flips", "--census", "--csv"];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(format!("unknown bench-novel-split option: {flag}").into());
                }
            }
            let seeds: Vec<u64> = match optional(&args, "--seeds") {
                Some(s) => s
                    .split(',')
                    .map(str::parse::<u64>)
                    .collect::<Result<Vec<_>, _>>()?,
                None => ns::SEEDS.collect(),
            };
            let flips = optional(&args, "--flips")
                .unwrap_or("25,50,100")
                .split(',')
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>()?;
            let owned = match optional(&args, "--graph") {
                Some(path) => Some(Connectome::load(Path::new(path))?),
                None => None,
            };
            let graph_label = if owned.is_some() {
                "PROVIDED_GRAPH"
            } else {
                "GENERATED_SYNTHETIC"
            };
            let d = ns::DESIGN;
            println!("v888_novel_split_schema=b888-4-novel-split-v1");
            if owned.is_some() {
                println!("graph=PROVIDED_GRAPH (task labels artificial)");
            } else {
                println!(
                    "graph=GENERATED_SYNTHETIC sources={} targets={} fan_in={} (not BANC; real BANC V888 graph DOI 10.7910/DVN/7WTH1N not available here)",
                    d.shape.sources, d.shape.targets, d.shape.fan_in
                );
            }
            println!(
                "seeds={} flips={flips:?} tasks={} active={} train_patterns_per_task={} test_patterns={} rounds_per_task={} init={} threshold={} cue_cap={} min_novel_pairs={} metric=balanced_accuracy boot_resamples={} boot_seed={:#x}",
                seeds.len(),
                ts::K_MAX,
                d.spec.active,
                d.spec.train_patterns,
                d.spec.test_patterns,
                d.spec.rounds_a,
                d.spec.init_weight,
                eg::THRESHOLD,
                d.cue_cap,
                ns::MIN_NOVEL_PAIRS,
                eg::BOOT_RESAMPLES,
                eg::BOOT_SEED,
            );
            if args.iter().any(|a| a == "--census") {
                println!("census: flip,seed,task,pool,cue_reach_min,test_pairs,seen,cn,icn,ue");
                for &flip in &flips {
                    for &seed in &seeds {
                        for (j, c) in ns::census(owned.as_ref(), &d, flip, seed)?
                            .iter()
                            .enumerate()
                        {
                            println!(
                                "{flip},{seed},{},{},{},{},{},{},{},{}",
                                j + 1,
                                c.pool,
                                c.cue_reach_min,
                                c.all,
                                c.seen,
                                c.cn,
                                c.icn,
                                c.ue
                            );
                        }
                    }
                }
                println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
                return Ok(());
            }
            let rows = ns::run_grid(owned.as_ref(), &d, &flips, &seeds, &ns::ARMS)?;
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(ns::to_csv(&rows, graph_label).as_bytes())?;
            }
            print!("{}", ns::report(&rows, &flips));
            println!(
                "topology_unchanged_all={}",
                rows.iter().all(|r| r.topology_unchanged)
            );
            println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
        }
        Some("bench-cue-reach") => {
            let allowed = [
                "--graph",
                "--seeds",
                "--flips",
                "--levels",
                "--census",
                "--no-coupled",
                "--csv",
            ];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(format!("unknown bench-cue-reach option: {flag}").into());
                }
            }
            let seeds: Vec<u64> = match optional(&args, "--seeds") {
                Some(s) => s
                    .split(',')
                    .map(str::parse::<u64>)
                    .collect::<Result<Vec<_>, _>>()?,
                None => cr::SEEDS.collect(),
            };
            let flips = optional(&args, "--flips")
                .unwrap_or("25,50,100")
                .split(',')
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>()?;
            let levels: Vec<usize> = match optional(&args, "--levels") {
                Some(s) => s
                    .split(',')
                    .map(str::parse::<usize>)
                    .collect::<Result<Vec<_>, _>>()?,
                None => cr::LEVELS.to_vec(),
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
            let d = cr::DESIGN;
            println!("v888_cue_reach_schema=b888-4-cue-reach-v1");
            if owned.is_some() {
                println!("graph=PROVIDED_GRAPH (task labels artificial)");
            } else {
                println!(
                    "graph=GENERATED_SYNTHETIC sources={} targets={} fan_in={} (not BANC; real BANC V888 graph DOI 10.7910/DVN/7WTH1N not available here)",
                    d.shape.sources, d.shape.targets, d.shape.fan_in
                );
            }
            println!(
                "seeds={} flips={flips:?} levels={levels:?} reserved_lines_per_cue={} tasks={} active={} train_patterns_per_task={} test_patterns={} rounds_per_task={} init={} threshold={} min_novel_pairs={} mono_tol={} metric=balanced_accuracy boot_resamples={} boot_seed={:#x}",
                seeds.len(),
                cr::FULL,
                ts::K_MAX,
                d.spec.active,
                d.spec.train_patterns,
                d.spec.test_patterns,
                d.spec.rounds_a,
                d.spec.init_weight,
                eg::THRESHOLD,
                ns::MIN_NOVEL_PAIRS,
                cr::MONO_TOL,
                eg::BOOT_RESAMPLES,
                eg::BOOT_SEED,
            );
            if args.iter().any(|a| a == "--census") {
                println!("census_fixed: flip,seed,task,pool,reachable,lines@levels,reach@levels,test_pairs,seen,cn,icn,ue");
                for &flip in &flips {
                    for &seed in &seeds {
                        let fx = cr::fixed_fixture(owned.as_ref(), flip, seed)?;
                        let c = cr::census_fixture(&fx, &levels);
                        for (j, p) in c.pairs.iter().enumerate() {
                            let lines: Vec<String> =
                                c.lines.iter().map(|l| l[j].to_string()).collect();
                            let reach: Vec<String> =
                                c.reach.iter().map(|r| r[j].to_string()).collect();
                            println!(
                                "{flip},{seed},{},{},{},{},{},{},{},{},{},{}",
                                j + 1,
                                c.pool,
                                c.reachable,
                                lines.join("/"),
                                reach.join("/"),
                                p.all,
                                p.seen,
                                p.cn,
                                p.icn,
                                p.ue
                            );
                        }
                    }
                }
                println!(
                    "census_coupled(#123 construction): flip,seed,level,pool,reach@tasks,cn@tasks"
                );
                for &flip in &flips {
                    for &seed in &seeds {
                        for &l in &levels {
                            let fx = cr::coupled_fixture(owned.as_ref(), flip, seed, l)?;
                            let c = cr::census_fixture(&fx, &[l]);
                            let reach: Vec<String> =
                                c.reach[0].iter().map(ToString::to_string).collect();
                            let cn: Vec<String> =
                                c.pairs.iter().map(|p| p.cn.to_string()).collect();
                            println!(
                                "{flip},{seed},{l},{},{},{}",
                                c.pool,
                                reach.join("/"),
                                cn.join("/")
                            );
                        }
                    }
                }
                println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
                return Ok(());
            }
            let mut rows = cr::run_fixed(owned.as_ref(), &flips, &seeds, &levels)?;
            if !args.iter().any(|a| a == "--no-coupled") && flips.contains(&cr::PRIMARY_FLIP) {
                rows.extend(cr::run_coupled(
                    owned.as_ref(),
                    &ns::DESIGN,
                    cr::PRIMARY_FLIP,
                    &seeds,
                    &levels,
                )?);
            }
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(cr::to_csv(&rows, graph_label).as_bytes())?;
            }
            print!("{}", cr::report(&rows, &flips, &levels));
            println!(
                "topology_unchanged_all={}",
                rows.iter().all(|r| r.topology_unchanged)
            );
            println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
        }
        Some("bench-seq-generalisation") => {
            let allowed = [
                "--graph",
                "--seeds",
                "--flips",
                "--k",
                "--heldout-orders",
                "--threads",
                "--csv",
                "--census",
            ];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(format!("unknown bench-seq-generalisation option: {flag}").into());
                }
            }
            let census = args.iter().any(|a| a == "--census");
            let seeds: Vec<u64> = match optional(&args, "--seeds") {
                Some(s) => s
                    .split(',')
                    .map(str::parse::<u64>)
                    .collect::<Result<Vec<_>, _>>()?,
                None if census => sgen::PILOT_SEEDS.collect(),
                None => sgen::SEEDS.collect(),
            };
            let flips = optional(&args, "--flips")
                .unwrap_or(if census { "50" } else { "25,50,100" })
                .split(',')
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>()?;
            let k: usize = optional(&args, "--k").map_or(Ok(sgen::K_RUN), str::parse)?;
            let heldout: usize =
                optional(&args, "--heldout-orders").map_or(Ok(sgen::HELDOUT_ORDERS), str::parse)?;
            let threads: usize =
                optional(&args, "--threads").map_or(Ok(sgen::default_threads()), str::parse)?;
            let owned = match optional(&args, "--graph") {
                Some(path) => Some(Connectome::load(Path::new(path))?),
                None => None,
            };
            let graph_label = if owned.is_some() {
                "PROVIDED_GRAPH"
            } else {
                "GENERATED_SYNTHETIC"
            };
            let d = sgen::DESIGN;
            println!("v888_seq_generalisation_schema=b888-4-seq-generalisation-v1");
            if owned.is_some() {
                println!("graph=PROVIDED_GRAPH (task labels artificial)");
            } else {
                println!(
                    "graph=GENERATED_SYNTHETIC sources={} targets={} fan_in={} (not BANC; real BANC V888 graph DOI 10.7910/DVN/7WTH1N not available here)",
                    d.shape.sources, d.shape.targets, d.shape.fan_in
                );
            }
            if census {
                println!("census(identity order only; writes, refusals and pair counts; no accuracy computed)");
                println!(
                    "fixture: flip,seed,pool,cue_lines_per_cue,cue_reach_per_cue,cn_pairs_per_task"
                );
                let mut lines = Vec::new();
                for &flip in &flips {
                    let (facts, l) = sgen::census(owned.as_ref(), &d, k, flip, &seeds)?;
                    for (seed, pool, lines_, reach, cn) in facts {
                        println!("{flip},{seed},{pool},{lines_:?},{reach:?},{cn:?}");
                    }
                    lines.push((flip, l));
                }
                println!(
                    "writes: flip,seed,setting,pattern_writes,cue_writes,total_writes,refused"
                );
                for (flip, l) in lines {
                    for (seed, s, pw, cw, tw, r) in l {
                        println!("{flip},{seed},{},{pw},{cw},{tw},{r}", s.name());
                    }
                }
                return Ok(());
            }
            println!(
                "seeds={} flips={flips:?} k={k} k_ref={} heldout_orders={heldout} order_salt={:#x} budget=paced100 (per-episode cumulative curve of uncapped error-gated in the same order; Z-pattern = Z@paced100's own pattern-write curve) cue_lines_per_cue={} tasks={k} active={} train_patterns_per_task={} test_patterns={} rounds_per_task={} init={} cue_init(E/E-cue0*)=0 threshold={} min_novel_pairs={} metric=balanced_accuracy boot_resamples={} boot_seed={:#x}",
                seeds.len(),
                sgen::K_REF,
                sgen::ORDER_SALT,
                sgen::cue_cap(&d, k)?,
                d.spec.active,
                d.spec.train_patterns,
                d.spec.test_patterns,
                d.spec.rounds_a,
                d.spec.init_weight,
                eg::THRESHOLD,
                ns::MIN_NOVEL_PAIRS,
                eg::BOOT_RESAMPLES,
                eg::BOOT_SEED,
            );
            let rows =
                sgen::run_grid_threads(owned.as_ref(), &d, k, &flips, &seeds, heldout, threads)?;
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(sgen::to_csv(&rows, graph_label).as_bytes())?;
            }
            print!("{}", sgen::report(&rows, &flips));
            println!(
                "topology_unchanged_all={}",
                rows.iter().all(|r| r.topology_unchanged)
            );
            println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
        }
        Some("bench-cue-displacement") => {
            let allowed = ["--graph", "--seeds", "--flips", "--csv", "--census"];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(format!("unknown bench-cue-displacement option: {flag}").into());
                }
            }
            let census = args.iter().any(|a| a == "--census");
            let seeds: Vec<u64> = match optional(&args, "--seeds") {
                Some(s) => s
                    .split(',')
                    .map(str::parse::<u64>)
                    .collect::<Result<Vec<_>, _>>()?,
                None if census => cdp::PILOT_SEEDS.collect(),
                None => cdp::SEEDS.collect(),
            };
            let flips = optional(&args, "--flips")
                .unwrap_or(if census { "50" } else { "25,50,100" })
                .split(',')
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>()?;
            let owned = match optional(&args, "--graph") {
                Some(path) => Some(Connectome::load(Path::new(path))?),
                None => None,
            };
            let graph_label = if owned.is_some() {
                "PROVIDED_GRAPH"
            } else {
                "GENERATED_SYNTHETIC"
            };
            let d = cdp::DESIGN;
            println!("v888_cue_displacement_schema=b888-4-cue-displacement-v1");
            if owned.is_some() {
                println!("graph=PROVIDED_GRAPH (task labels artificial)");
            } else {
                println!(
                    "graph=GENERATED_SYNTHETIC sources={} targets={} fan_in={} (not BANC; real BANC V888 graph DOI 10.7910/DVN/7WTH1N not available here)",
                    d.shape.sources, d.shape.targets, d.shape.fan_in
                );
            }
            if census {
                println!("census(writes and refusals only; no accuracy computed): flip,seed,setting,pattern_writes,cue_writes,total_writes,refused");
                for &flip in &flips {
                    for (seed, s, pw, cw, tw, r) in cdp::census(owned.as_ref(), flip, &seeds)? {
                        println!("{flip},{seed},{},{pw},{cw},{tw},{r}", s.name());
                    }
                }
                return Ok(());
            }
            println!(
                "seeds={} flips={flips:?} budget=paced100 (per-episode cumulative curve of uncapped error-gated; exempt = cue writes not charged; Z-pattern = Z@paced100's own pattern-write curve) cue_level={} tasks={} active={} train_patterns_per_task={} test_patterns={} rounds_per_task={} init={} cue_init(E/E-cue0*)=0 threshold={} min_novel_pairs={} metric=balanced_accuracy boot_resamples={} boot_seed={:#x}",
                seeds.len(),
                ci::CUE_LEVEL,
                ts::K_MAX,
                d.spec.active,
                d.spec.train_patterns,
                d.spec.test_patterns,
                d.spec.rounds_a,
                d.spec.init_weight,
                eg::THRESHOLD,
                ns::MIN_NOVEL_PAIRS,
                eg::BOOT_RESAMPLES,
                eg::BOOT_SEED,
            );
            let rows = cdp::run_grid(owned.as_ref(), &flips, &seeds)?;
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(cdp::to_csv(&rows, graph_label).as_bytes())?;
            }
            print!("{}", cdp::report(&rows, &flips));
            println!(
                "topology_unchanged_all={}",
                rows.iter().all(|r| r.inner.topology_unchanged)
            );
            println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
        }
        Some("bench-cue-init") => {
            let allowed = [
                "--graph",
                "--seeds",
                "--flips",
                "--budgets",
                "--csv",
                "--census",
            ];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(format!("unknown bench-cue-init option: {flag}").into());
                }
            }
            let census = args.iter().any(|a| a == "--census");
            let seeds: Vec<u64> = match optional(&args, "--seeds") {
                Some(s) => s
                    .split(',')
                    .map(str::parse::<u64>)
                    .collect::<Result<Vec<_>, _>>()?,
                None if census => ci::PILOT_SEEDS.collect(),
                None => ci::SEEDS.collect(),
            };
            let flips = optional(&args, "--flips")
                .unwrap_or(if census { "50" } else { "25,50,100" })
                .split(',')
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>()?;
            let budgets: Vec<wb::Budget> = match optional(&args, "--budgets") {
                Some(s) => s
                    .split(',')
                    .map(wb::Budget::parse)
                    .collect::<Result<Vec<_>, _>>()?,
                None => ci::BUDGETS.to_vec(),
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
            let d = ci::DESIGN;
            println!("v888_cue_init_schema=b888-4-cue-zero-init-v1");
            if owned.is_some() {
                println!("graph=PROVIDED_GRAPH (task labels artificial)");
            } else {
                println!(
                    "graph=GENERATED_SYNTHETIC sources={} targets={} fan_in={} (not BANC; real BANC V888 graph DOI 10.7910/DVN/7WTH1N not available here)",
                    d.shape.sources, d.shape.targets, d.shape.fan_in
                );
            }
            if census {
                println!("census(writes, cue writes, refusals and cue state only; no accuracy computed): flip,seed,arm,budget,writes_total,cue_writes,refused,cue_refused,cue_nonzero_end");
                for &flip in &flips {
                    for (seed, a, b, w, cw, r, cr_, nz) in
                        ci::census(owned.as_ref(), flip, &seeds, &budgets)?
                    {
                        println!(
                            "{flip},{seed},{},{},{w},{cw},{r},{cr_},{nz}",
                            a.name(),
                            b.name()
                        );
                    }
                }
                return Ok(());
            }
            let names: Vec<String> = budgets.iter().map(|b| b.name()).collect();
            println!(
                "seeds={} flips={flips:?} budgets={names:?} (percent of uncapped error-gated effective writes per task; paced = per-episode cumulative curve) cue_level={} tasks={} active={} train_patterns_per_task={} test_patterns={} rounds_per_task={} init={} cue_init(E/E-cue0*)=0 threshold={} min_novel_pairs={} metric=balanced_accuracy boot_resamples={} boot_seed={:#x}",
                seeds.len(),
                ci::CUE_LEVEL,
                ts::K_MAX,
                d.spec.active,
                d.spec.train_patterns,
                d.spec.test_patterns,
                d.spec.rounds_a,
                d.spec.init_weight,
                eg::THRESHOLD,
                ns::MIN_NOVEL_PAIRS,
                eg::BOOT_RESAMPLES,
                eg::BOOT_SEED,
            );
            let rows = ci::run_grid(owned.as_ref(), &flips, &seeds, &budgets)?;
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(ci::to_csv(&rows, graph_label).as_bytes())?;
            }
            print!("{}", ci::report(&rows, &flips, &budgets));
            println!(
                "topology_unchanged_all={}",
                rows.iter().all(|r| r.inner.topology_unchanged)
            );
            println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
        }
        Some("bench-write-allocation") => {
            let allowed = [
                "--graph",
                "--seeds",
                "--flips",
                "--budgets",
                "--csv",
                "--census",
            ];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(format!("unknown bench-write-allocation option: {flag}").into());
                }
            }
            let census = args.iter().any(|a| a == "--census");
            let seeds: Vec<u64> = match optional(&args, "--seeds") {
                Some(s) => s
                    .split(',')
                    .map(str::parse::<u64>)
                    .collect::<Result<Vec<_>, _>>()?,
                None if census => (1..=5).collect(),
                None => wa::SEEDS.collect(),
            };
            let flips = optional(&args, "--flips")
                .unwrap_or(if census { "50" } else { "25,50,100" })
                .split(',')
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>()?;
            let budgets: Vec<wb::Budget> = match optional(&args, "--budgets") {
                Some(s) => s
                    .split(',')
                    .map(wb::Budget::parse)
                    .collect::<Result<Vec<_>, _>>()?,
                None => wa::BUDGETS.to_vec(),
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
            let d = wa::DESIGN;
            println!("v888_write_allocation_schema=b888-4-write-allocation-v1");
            if owned.is_some() {
                println!("graph=PROVIDED_GRAPH (task labels artificial)");
            } else {
                println!(
                    "graph=GENERATED_SYNTHETIC sources={} targets={} fan_in={} (not BANC; real BANC V888 graph DOI 10.7910/DVN/7WTH1N not available here)",
                    d.shape.sources, d.shape.targets, d.shape.fan_in
                );
            }
            if census {
                println!("census(writes, cue writes, refusals and blocked raises only; no accuracy computed): flip,seed,strategy,budget,writes_total,cue_writes,refused,cue_refused,blocked");
                for &flip in &flips {
                    for (seed, s, b, w, cw, r, cr_, bl) in
                        wa::census(owned.as_ref(), flip, &seeds, &budgets)?
                    {
                        println!(
                            "{flip},{seed},{},{},{w},{cw},{r},{cr_},{bl}",
                            s.name(),
                            b.name()
                        );
                    }
                }
                return Ok(());
            }
            let names: Vec<String> = budgets.iter().map(|b| b.name()).collect();
            println!(
                "seeds={} flips={flips:?} budgets={names:?} (percent of uncapped error-gated effective writes per task; paced = per-episode cumulative curve) cue_level={} tasks={} active={} train_patterns_per_task={} test_patterns={} rounds_per_task={} init={} threshold={} min_novel_pairs={} metric=balanced_accuracy boot_resamples={} boot_seed={:#x}",
                seeds.len(),
                wa::CUE_LEVEL,
                ts::K_MAX,
                d.spec.active,
                d.spec.train_patterns,
                d.spec.test_patterns,
                d.spec.rounds_a,
                d.spec.init_weight,
                eg::THRESHOLD,
                ns::MIN_NOVEL_PAIRS,
                eg::BOOT_RESAMPLES,
                eg::BOOT_SEED,
            );
            let rows = wa::run_grid(owned.as_ref(), &flips, &seeds, &budgets)?;
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(wa::to_csv(&rows, graph_label).as_bytes())?;
            }
            print!("{}", wa::report(&rows, &flips, &budgets));
            println!(
                "topology_unchanged_all={}",
                rows.iter().all(|r| r.inner.topology_unchanged)
            );
            println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
        }
        Some("bench-write-budget") => {
            let allowed = ["--graph", "--seeds", "--flips", "--budgets", "--csv"];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(format!("unknown bench-write-budget option: {flag}").into());
                }
            }
            let seeds: Vec<u64> = match optional(&args, "--seeds") {
                Some(s) => s
                    .split(',')
                    .map(str::parse::<u64>)
                    .collect::<Result<Vec<_>, _>>()?,
                None => wb::SEEDS.collect(),
            };
            let flips = optional(&args, "--flips")
                .unwrap_or("25,50,75,100")
                .split(',')
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>()?;
            let budgets: Vec<wb::Budget> = match optional(&args, "--budgets") {
                Some(s) => s
                    .split(',')
                    .map(wb::Budget::parse)
                    .collect::<Result<Vec<_>, _>>()?,
                None => wb::BUDGETS.to_vec(),
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
            let d = wb::DESIGN;
            println!("v888_write_budget_schema=b888-4-write-budget-v1");
            if owned.is_some() {
                println!("graph=PROVIDED_GRAPH (task labels artificial)");
            } else {
                println!(
                    "graph=GENERATED_SYNTHETIC sources={} targets={} fan_in={} (not BANC; real BANC V888 graph DOI 10.7910/DVN/7WTH1N not available here)",
                    d.shape.sources, d.shape.targets, d.shape.fan_in
                );
            }
            let names: Vec<String> = budgets.iter().map(|b| b.name()).collect();
            println!(
                "seeds={} flips={flips:?} budgets={names:?} (percent of uncapped error-gated effective writes per task; paced = per-episode cumulative curve, total = task total) cue_level={} tasks={} active={} train_patterns_per_task={} test_patterns={} rounds_per_task={} init={} threshold={} min_novel_pairs={} mono_tol={} metric=balanced_accuracy boot_resamples={} boot_seed={:#x}",
                seeds.len(),
                wb::CUE_LEVEL,
                ts::K_MAX,
                d.spec.active,
                d.spec.train_patterns,
                d.spec.test_patterns,
                d.spec.rounds_a,
                d.spec.init_weight,
                eg::THRESHOLD,
                ns::MIN_NOVEL_PAIRS,
                wb::MONO_TOL,
                eg::BOOT_RESAMPLES,
                eg::BOOT_SEED,
            );
            let rows = wb::run_grid(owned.as_ref(), &flips, &seeds, &budgets)?;
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(wb::to_csv(&rows, graph_label).as_bytes())?;
            }
            print!("{}", wb::report(&rows, &flips, &budgets));
            println!(
                "topology_unchanged_all={}",
                rows.iter().all(|r| r.inner.topology_unchanged)
            );
            println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
        }
        Some("bench-task-sequence") => {
            let allowed = ["--graph", "--seeds", "--flips", "--init", "--csv"];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(format!("unknown bench-task-sequence option: {flag}").into());
                }
            }
            let seeds: Vec<u64> = match optional(&args, "--seeds") {
                Some(s) => s
                    .split(',')
                    .map(str::parse::<u64>)
                    .collect::<Result<Vec<_>, _>>()?,
                None => ts::SEEDS.collect(),
            };
            let flips = optional(&args, "--flips")
                .unwrap_or("25,50,100")
                .split(',')
                .map(str::parse::<u32>)
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
                flip_pct: ts::PRIMARY_FLIP,
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
            let rows = ts::run_grid(owned.as_ref(), &shape, &spec, &flips, &seeds, &ts::ARMS)?;
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(ts::to_csv(&rows, graph_label).as_bytes())?;
            }
            println!("v888_task_sequence_schema=b888-4-task-sequence-v1");
            if owned.is_some() {
                println!("graph=PROVIDED_GRAPH (task labels artificial)");
            } else {
                println!("graph=GENERATED_SYNTHETIC sources=64 targets=24 fan_in=6 (not BANC; real BANC V888 graph DOI 10.7910/DVN/7WTH1N not available here)");
            }
            println!(
                "seeds={} flips={flips:?} init={init} tasks={} active=4 train_patterns_per_task=32 test_patterns=64 rounds_per_task=8 threshold={} cue_cap={} metric=balanced_accuracy boot_resamples={} boot_seed={:#x}",
                seeds.len(),
                ts::K_MAX,
                eg::THRESHOLD,
                cc::CUE_CAP,
                eg::BOOT_RESAMPLES,
                eg::BOOT_SEED,
            );
            let fmt = |s: genbench::Spread| format!("{:.3}±{:.3}", s.mean, s.sd);
            let stat = |flip: u32, a: ts::Arm, f: &dyn Fn(&ts::Row) -> f64| {
                let v: Vec<f64> = rows
                    .iter()
                    .filter(|r| r.flip == flip && r.arm == a)
                    .map(f)
                    .collect();
                genbench::spread(&v)
            };
            for &flip in &flips {
                println!(
                    "flip={flip} median_novel_pairs_per_task={} test_pairs_mean={:.1} cue_lines_mean={:?} cue_reach_mean={:?}",
                    ts::median_novel_pairs(&rows, flip),
                    stat(flip, ts::SUPERVISED, &|r| r.all[0][0].total() as f64).mean,
                    (0..ts::K_MAX)
                        .map(|j| format!("{:.1}", stat(flip, ts::EE, &|r| r.cue_lines[j] as f64).mean))
                        .collect::<Vec<_>>(),
                    (0..ts::K_MAX)
                        .map(|j| format!("{:.1}", stat(flip, ts::EE, &|r| r.cue_reach[j] as f64).mean))
                        .collect::<Vec<_>>(),
                );
            }
            println!("arm_summary: flip,k,arm,acc,acc_novel,learn,forgetting,last,first_retained,weights_changed_total,saturated_end");
            for &flip in &flips {
                for k in 2..=ts::K_MAX {
                    for a in ts::ARMS {
                        println!(
                            "{flip},{k},{},{},{},{},{},{},{},{:.1},{:.1}",
                            a.label(),
                            fmt(stat(flip, a, &|r| r.acc(k))),
                            fmt(stat(flip, a, &|r| r.acc_novel(k))),
                            fmt(stat(flip, a, &|r| r.learn(k))),
                            fmt(stat(flip, a, &|r| r.forgetting(k))),
                            fmt(stat(flip, a, &|r| r.last(k))),
                            fmt(stat(flip, a, &|r| r.r(k - 1, 0))),
                            stat(flip, a, &|r| r.cost[..k]
                                .iter()
                                .map(|c| c.weights_changed as f64)
                                .sum())
                            .mean,
                            stat(flip, a, &|r| r.saturated_end as f64).mean,
                        );
                    }
                }
            }
            println!(
                "matrix: flip,arm,i,R[i][1..4] (task j on TEST after task i, mean over seeds)"
            );
            for &flip in &flips {
                for a in ts::ARMS {
                    for i in 0..ts::K_MAX {
                        let cells: Vec<String> = (0..ts::K_MAX)
                            .map(|j| format!("{:.3}", stat(flip, a, &|r| r.r(i, j)).mean))
                            .collect();
                        println!("{flip},{},{},{}", a.label(), i + 1, cells.join(","));
                    }
                }
            }
            let p = |d: &eg::Paired| {
                format!(
                    "{:+.3}±{:.3}[{:+.3},{:+.3}]{}/{}/{}",
                    d.mean, d.sd, d.ci_lo, d.ci_hi, d.better, d.ties, d.worse
                )
            };
            let low: Vec<u64> = seeds.iter().copied().filter(|&s| s <= 60).collect();
            let high: Vec<u64> = seeds.iter().copied().filter(|&s| s > 60).collect();
            let cmps = [
                (ts::EE, ts::ERROR_GATED),
                (ts::EE, ts::SUPERVISED),
                (ts::ERROR_GATED, ts::SUPERVISED),
                (ts::SE, ts::SUPERVISED),
                (ts::ES, ts::ERROR_GATED),
                (ts::ES, ts::EE),
                (ts::MARGIN4, ts::ERROR_GATED),
                (ts::MM2, ts::EE),
                (ts::MM4, ts::EE),
                (ts::MM8, ts::EE),
                (ts::MM2, ts::SUPERVISED),
                (ts::MM4, ts::SUPERVISED),
                (ts::MM8, ts::SUPERVISED),
            ];
            println!("comparison: flip,seeds,k,x,base,n,d_acc,d_acc_novel,d_learn,d_last,d_forget(+=x_less),d_first_retained,d_first_learn,passes_seq_rule,passes_pair_rule  (each: mean±sd[boot95_lo,hi]better/tie/worse)");
            for &flip in &flips {
                for (label, subset) in [
                    ("all", None),
                    ("41-60", Some(low.as_slice())),
                    ("61-80", Some(high.as_slice())),
                ] {
                    if subset.is_some_and(<[u64]>::is_empty) {
                        continue;
                    }
                    for k in 2..=ts::K_MAX {
                        for &(x, base) in &cmps {
                            let c = ts::compare(&rows, flip, k, x, base, subset);
                            println!(
                                "{flip},{label},{k},{},{},{},{},{},{},{},{},{},{},{},{}",
                                x.label(),
                                base.label(),
                                c.acc.n,
                                p(&c.acc),
                                p(&c.acc_novel),
                                p(&c.learn),
                                p(&c.last),
                                p(&c.forget),
                                p(&c.first_retained),
                                p(&c.first_learn),
                                c.passes(),
                                c.passes_pair_rule(),
                            );
                        }
                    }
                }
            }
            let rd = ts::readings(&rows);
            println!(
                "H1_E/E-cue_vs_error-gated_two_task_rule_heldout_seeds_flip50={}",
                rd.h1
            );
            println!(
                "H2_E/E-cue_vs_error-gated_seq_rule_k3_and_k4_flip50={}",
                rd.h2
            );
            println!(
                "H3_E/E-cue_vs_supervised_seq_rule_k3_and_k4_flip50={}",
                rd.h3
            );
            println!(
                "H4_E/E-cue_vs_error-gated_novel_config_acc_k3_and_k4_flip50={}",
                rd.h4.map_or("not_evaluable".to_string(), |b| b.to_string())
            );
            println!(
                "H5_some_margin_arm_beats_E/E-cue_seq_rule_k3_flip50={}",
                rd.h5
            );
            println!(
                "topology_unchanged={}",
                rows.iter().all(|r| r.topology_unchanged)
            );
            println!("evaluation_scope=synthetic_or_unverified_graph_artificial_labels_not_BANC_not_biological");
        }
        Some("bench-consolidated-graded") => {
            let allowed = ["--graph", "--seeds", "--init", "--csv"];
            for flag in args.iter().skip(1).filter(|arg| arg.starts_with("--")) {
                if !allowed.contains(&flag.as_str()) {
                    return Err(format!("unknown bench-consolidated-graded option: {flag}").into());
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
            let rows = cg::run_grid(owned.as_ref(), &shape, &spec, &seeds)?;
            if let Some(path) = optional(&args, "--csv") {
                use std::io::Write;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?
                    .write_all(cg::to_csv(&rows, graph_label).as_bytes())?;
            }
            println!("v888_consolidated_graded_schema=b888-4-consolidated-graded-v1");
            if owned.is_some() {
                println!("graph=PROVIDED_GRAPH (task labels artificial)");
            } else {
                println!("graph=GENERATED_SYNTHETIC sources=64 targets=24 fan_in=6 (not BANC; real BANC V888 graph DOI 10.7910/DVN/7WTH1N not available here)");
            }
            println!(
                "seeds={} init={init} active=4 train_patterns=32 test_patterns=64 rounds_a=8 rounds_b=8 flip_pct=50 threshold=1 metric=balanced_accuracy protect_period={} rehearsal_period={}",
                seeds.len(),
                cg::PROTECT_PERIOD,
                cg::REHEARSAL_PERIOD,
            );
            let fmt = |s: genbench::Spread| format!("{:.3}±{:.3}", s.mean, s.sd);
            let stat = |l: cg::Learner, m: cg::Mechanism, f: &dyn Fn(&cg::Row) -> f64| {
                let v: Vec<f64> = rows
                    .iter()
                    .filter(|r| r.learner == l && r.mechanism == m)
                    .map(f)
                    .collect();
                genbench::spread(&v)
            };
            println!("arm_summary: learner,mechanism,a_train,a_test_after_a,a_test_after_b,forgetting,b_test,protected_edges,b_rehearsal_episodes,weights_changed(A+B),b_protected_attempts,b_blocked");
            for l in cg::LEARNERS {
                for m in cg::MECHANISMS {
                    println!(
                        "{},{},{},{},{},{},{},{:.1},{:.1},{:.1},{:.1},{:.1}",
                        l.label(),
                        m.label(),
                        fmt(stat(l, m, &|r| r.a_train_after_a.balanced())),
                        fmt(stat(l, m, &|r| r.a_test_after_a.balanced())),
                        fmt(stat(l, m, &|r| r.a_test_after_b.balanced())),
                        fmt(stat(l, m, &cg::Row::forgetting)),
                        fmt(stat(l, m, &|r| r.b_test_after_b.balanced())),
                        stat(l, m, &|r| r.protected_edges as f64).mean,
                        stat(l, m, &|r| r.cost_b.rehearsal_episodes as f64).mean,
                        stat(
                            l,
                            m,
                            &|r| (r.cost_a.weights_changed + r.cost_b.weights_changed) as f64
                        )
                        .mean,
                        stat(l, m, &|r| r.cost_b.protected_attempts as f64).mean,
                        stat(l, m, &|r| r.cost_b.blocked as f64).mean,
                    );
                }
            }
            println!("comparison: x_learner,x_mech,y_learner,y_mech,n,d_a_test,d_forget(+=x_less),forget_b/t/w,d_retained_a,d_b,passes_rule,tradeoff_vs_control");
            for c in cg::preregistered_comparisons(&rows) {
                println!(
                    "{},{},{},{},{},{:+.3}±{:.3},{:+.3}±{:.3},{}/{}/{},{:+.3}±{:.3},{:+.3}±{:.3},{},{}",
                    c.x.0.label(),
                    c.x.1.label(),
                    c.y.0.label(),
                    c.y.1.label(),
                    c.n,
                    c.a_test.mean,
                    c.a_test.sd,
                    c.forget.delta.mean,
                    c.forget.delta.sd,
                    c.forget.better,
                    c.forget.ties,
                    c.forget.worse,
                    c.retained.mean,
                    c.retained.sd,
                    c.b.mean,
                    c.b.sd,
                    if c.y.0 == c.x.0 { c.passes().to_string() } else { "n/a".to_string() },
                    if c.y == (cg::Learner::Control, cg::Mechanism::None) && c.x.0 != cg::Learner::Control { cg::beats_control_tradeoff(&c).to_string() } else { "n/a".to_string() },
                );
            }
            let (h1, h2, h3) = cg::readings(&rows);
            println!("H1_consolidation_passes_both_graded={h1}");
            println!("H2_rehearsal_passes_both_graded={h2}");
            println!("H3_tradeoff_broken_vs_control={h3}");
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

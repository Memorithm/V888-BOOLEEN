//! Convert observed V888 activity into a bounded, derived recurrent circuit.
//! Source anatomical datasets are never overwritten by default.
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use v888_booleen::activity_atlas::ActivityAtlas;
use v888_booleen::activity_guided::{derive_circuit, SelectionConfig};
use v888_booleen::connectome::Connectome;
use v888_booleen::engine::BooleanEngine;

fn required<'a>(args: &'a [String], key: &str) -> Result<&'a str, Box<dyn Error>> {
    let at = args.iter().position(|arg| arg == key).ok_or(format!("missing {key}"))?;
    Ok(args.get(at + 1).ok_or(format!("missing value for {key}"))?)
}
fn optional<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter().position(|arg| arg == key)
        .and_then(|at| args.get(at + 1).map(String::as_str))
}
fn number<T: std::str::FromStr>(
    args: &[String],
    key: &str,
    default: T,
) -> Result<T, Box<dyn Error>>
where
    <T as std::str::FromStr>::Err: Error + 'static,
{
    match optional(args, key) {
        Some(value) => Ok(value.parse::<T>()?),
        None => Ok(default),
    }
}

fn probe(
    original: &Connectome,
    reduced: &Connectome,
    seeds: &str,
    threshold: u64,
    steps: u32,
) -> Result<(), Box<dyn Error>> {
    if steps > 10 {
        return Err("comparison step limit exceeded (10)".into());
    }
    let mut reference = BooleanEngine::new(original, threshold)?;
    let mut selected = BooleanEngine::new(reduced, threshold)?;
    for text in seeds.split(',') {
        let id = text.parse::<u64>()?;
        reference.stimulate(original, id)?;
        selected.stimulate(reduced, id).map_err(|_| "probe seed not present in selected circuit")?;
    }

    for tick in 0..=steps {
        let mut reference_inside = 0_u64;
        let mut overlap = 0_u64;
        for index in reference.active.active_indices() {
            if let Some(mapped) = reduced.index(original.ids[index]) {
                reference_inside += 1;
                if selected.active.get(mapped).map_err(|_| "invalid circuit index")? {
                    overlap += 1;
                }
            }
        }
        println!(
            "probe_tick={tick} full_graph_active={} full_graph_active_inside_circuit={} reduced_active={} matched_active={overlap}",
            reference.active.active_count(),
            reference_inside,
            selected.active.active_count()
        );
        if tick != steps {
            reference.step(original)?;
            selected.step(reduced)?;
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.iter().any(|arg| arg == "--graph")
        || !args.iter().any(|arg| arg == "--atlas")
        || !args.iter().any(|arg| arg == "--output")
    {
        eprintln!("Usage: v888-select --graph brain.v888b --atlas activity.v888atl --output selected.v888b [--anchors 128] [--max-neurons 2048] [--min-ticks 1] [--hops 1] [--fanout 8] [--max-scanned-edges 20000000] [--max-result-edges 1000000] [--probe ROOT_ID[,ROOT_ID]] [--threshold 1] [--steps 3]");
        return Err("missing required argument".into());
    }
    let graph_path = Path::new(required(&args, "--graph")?);
    let atlas_path = Path::new(required(&args, "--atlas")?);
    let output_path = Path::new(required(&args, "--output")?);
    if output_path == graph_path || output_path == atlas_path || output_path.exists() {
        return Err("refusing to overwrite an existing graph, atlas or output".into());
    }
    let graph = Connectome::load(graph_path)?;
    let atlas = ActivityAtlas::read(&graph, BufReader::new(File::open(atlas_path)?))?;
    let defaults = SelectionConfig::default();
    let config = SelectionConfig {
        anchors: number(&args, "--anchors", defaults.anchors)?,
        max_neurons: number(&args, "--max-neurons", defaults.max_neurons)?,
        min_active_ticks: number(&args, "--min-ticks", defaults.min_active_ticks)?,
        expansion_hops: number(&args, "--hops", defaults.expansion_hops)?,
        fanout: number(&args, "--fanout", defaults.fanout)?,
        max_scanned_edges: number(&args, "--max-scanned-edges", defaults.max_scanned_edges)?,
        max_result_edges: number(&args, "--max-result-edges", defaults.max_result_edges)?,
    };
    let (selected, stats) = derive_circuit(&graph, &atlas, config)?;
    println!(
        "recorded_ticks={} source_neurons={} selected_neurons={} source_edges={} selected_edges={} anchors={} expanded_neurons={} inspected_edges={} selection=activity_guided_experimental",
        stats.recorded_ticks,
        stats.graph_neurons,
        stats.selected_neurons,
        stats.graph_edges,
        stats.selected_edges,
        stats.anchors,
        stats.expanded_neurons,
        stats.inspected_edges
    );
    if let Some(seeds) = optional(&args, "--probe") {
        let threshold = number(&args, "--threshold", 1_u64)?;
        let steps = number(&args, "--steps", 3_u32)?;
        probe(&graph, &selected, seeds, threshold, steps)?;
    }
    selected.save(output_path)?;
    println!("derived_circuit_saved={}", output_path.display());
    println!("source_connectome_unchanged=true derived_projection_is_lossy=true");
    Ok(())
}

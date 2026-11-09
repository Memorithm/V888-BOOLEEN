//! Record and inspect reproducible boolean activation trajectories.
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use v888_booleen::activity_atlas::ActivityAtlas;
use v888_booleen::connectome::Connectome;
use v888_booleen::engine::BooleanEngine;

fn arg<'a>(args: &'a [String], flag: &str) -> Result<&'a str, Box<dyn Error>> {
    let index = args.iter().position(|v| v == flag).ok_or(format!("missing {flag}"))?;
    Ok(args.get(index + 1).ok_or(format!("missing value for {flag}"))?)
}
fn optional<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter().position(|v| v == flag).and_then(|i| args.get(i + 1).map(String::as_str))
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("record") => {
            let graph = Connectome::load(Path::new(arg(&args, "--graph")?))?;
            let threshold = optional(&args, "--threshold").unwrap_or("1").parse::<u64>()?;
            let steps = optional(&args, "--steps").unwrap_or("3").parse::<u32>()?;
            if steps > 1000 {
                return Err("recording limited to 1000 steps".into());
            }
            let mut engine = BooleanEngine::new(&graph, threshold)?;
            for seed in arg(&args, "--seed")?.split(',') {
                engine.stimulate(&graph, seed.parse::<u64>()?)?;
            }
            let mut atlas = ActivityAtlas::new(&graph)?;
            for tick in 0..=steps {
                let event = atlas.observe(&graph, &engine.active)?;
                println!(
                    "tick={tick} active={} new={} silent={}",
                    event.active, event.newly_active, event.silenced
                );
                if tick < steps {
                    engine.step(&graph)?;
                }
            }
            let output_path = arg(&args, "--output")?;
            let output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output_path)?;
            let mut writer = BufWriter::new(output);
            atlas.write(&graph, &mut writer)?;
            writer.flush()?;
            println!("atlas_saved={output_path} ticks={}", atlas.ticks());
        }
        Some("inspect") => {
            let graph = Connectome::load(Path::new(arg(&args, "--graph")?))?;
            let file = BufReader::new(File::open(arg(&args, "--atlas")?)?);
            let atlas = ActivityAtlas::read(&graph, file)?;
            let limit = optional(&args, "--top").unwrap_or("20").parse::<usize>()?;
            if limit > 10_000 {
                return Err("top limit 10000".into());
            }
            println!("atlas_ticks={} graph_neurons={}", atlas.ticks(), graph.ids.len());
            for event in atlas.timeline() {
                println!(
                    "tick={} active={} new={} silent={}",
                    event.tick, event.active, event.newly_active, event.silenced
                );
            }
            for activity in atlas.top_neurons(&graph, limit)? {
                println!(
                    "neuron_id={} active_ticks={} rising={} falling={} longest_streak={} first={:?} last={:?}",
                    activity.neuron_id, activity.active_ticks, activity.rising_edges,
                    activity.falling_edges, activity.longest_active_streak,
                    activity.first_active_tick, activity.last_active_tick
                );
            }
            if let Some(raw_id) = optional(&args, "--neuron") {
                let id = raw_id.parse::<u64>()?;
                let activity = atlas.neuron(&graph, id)?.ok_or("unknown neuron ID")?;
                println!("queried_neuron={activity:?}");
            }
        }
        _ => {
            eprintln!("Usage: v888-activity record --graph brain.v888b --seed ID[,ID] --output history.v888atl [--threshold 1] [--steps 3]");
            eprintln!("       v888-activity inspect --graph brain.v888b --atlas history.v888atl [--top 20] [--neuron ID]");
            return Err("unknown subcommand".into());
        }
    }
    Ok(())
}

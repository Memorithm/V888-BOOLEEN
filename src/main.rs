use std::error::Error;
use std::path::Path;
use v888_booleen::connectome::Connectome;
use v888_booleen::engine::BooleanEngine;
fn require<'a>(args:&'a [String],name:&str)->Result<&'a str,Box<dyn Error>>{
    let ix=args.iter().position(|v|v==name).ok_or(format!("missing {name}"))?;
    Ok(args.get(ix+1).ok_or(format!("missing value for {name}"))?)
}
fn optional<'a>(args:&'a [String],name:&str)->Option<&'a str>{
    args.iter().position(|v|v==name).and_then(|ix|args.get(ix+1).map(String::as_str))
}
fn help(){
    println!("V888 BOOLEEN experimental Boolean connectome (no biological data bundled)");
    println!("  build --edges edges.csv [--neurons ids.txt] --output brain.v888b");
    println!("  info --graph brain.v888b");
    println!("  export --graph brain.v888b --output rows.csv");
    println!("  run --graph brain.v888b --seed ID[,ID] [--threshold 1] [--steps 3]");
    println!("  demo");
}
fn main()->Result<(),Box<dyn Error>>{
    let args:Vec<String>=std::env::args().skip(1).collect();
    match args.first().map(String::as_str){
        Some("build")=>{
            let graph=Connectome::import(Path::new(require(&args,"--edges")?),
                optional(&args,"--neurons").map(Path::new))?;
            graph.save(Path::new(require(&args,"--output")?))?;
            println!("saved neurons={} edges={}",graph.ids.len(),graph.edges());
        }
        Some("info") if optional(&args,"--graph").is_some()=>{
            let graph=Connectome::load(Path::new(require(&args,"--graph")?))?;
            println!("neurons={} edges={} format=v888b-v1",graph.ids.len(),graph.edges());
        }
        Some("export")=>{
            Connectome::load(Path::new(require(&args,"--graph")?))?
                .export_csv(Path::new(require(&args,"--output")?))?;
        }
        Some("run")=>{
            let graph=Connectome::load(Path::new(require(&args,"--graph")?))?;
            let threshold=optional(&args,"--threshold").unwrap_or("1").parse::<u64>()?;
            let steps=optional(&args,"--steps").unwrap_or("3").parse::<u32>()?;
            if steps>1000{return Err("too many steps (>1000)".into());}
            let mut brain=BooleanEngine::new(&graph,threshold)?;
            for seed in require(&args,"--seed")?.split(','){
                brain.stimulate(&graph,seed.parse::<u64>()?)?;
            }
            println!("step=0 active={}",brain.active.active_count());
            for step in 1..=steps{println!("step={step} active={}",brain.step(&graph)?);}
        }
        Some("demo")=>{
            let graph=Connectome{
                ids:vec![10,20,30],offsets:vec![0,1,2,2],
                targets:vec![1,2],counts:vec![2,3]
            };
            let mut brain=BooleanEngine::new(&graph,2)?;
            brain.stimulate(&graph,10)?;
            for step in 1..=3{println!("demo step={step} active={}",brain.step(&graph)?);}
            println!("SYNTHETIC DEMO ONLY; no V888 anatomical data loaded");
        }
        Some("--help")|Some("-h")|None=>help(),
        _=>{help();return Err("unknown command".into());}
    }
    Ok(())
}

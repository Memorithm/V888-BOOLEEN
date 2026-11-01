//! Full recurrent CSR export including self loops; separate from SciRust V8CSR001.
use std::path::Path;
use v888_booleen::connectome::Connectome;
use v888_booleen::recurrent_csr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: cargo run --example export_recurrent_csr -- INPUT.v888b OUTPUT.csr".into());
    }
    let graph = Connectome::load(Path::new(&args[0]))?;
    let path = Path::new(&args[1]);
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    let result = recurrent_csr::write(&graph, &mut output).and_then(|contacts| {
        output.sync_all()?;
        Ok(contacts)
    });
    drop(output);
    match result {
        Ok(contacts) => {
            println!(
                "format=V8CSR002 nodes={} directed_pairs={} contacts={contacts}",
                graph.ids.len(),
                graph.edges()
            );
            println!("source_scope=gcs-v888-v2 identity_equivalence_to_scirust_bool01=NOT_CLAIMED");
            Ok(())
        }
        Err(error) => {
            std::fs::remove_file(path)?;
            Err(error.into())
        }
    }
}


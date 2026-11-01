//! Format adapter to SciRust's V8CSR001 contract. This does not qualify source identity.
use std::io::{self, Write};
use std::path::Path;
use v888_booleen::connectome::Connectome;

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Validate the stricter consumer contract before writing any output bytes.
/// Reject incompatible rows instead of silently filtering or merging them.
fn export(graph: &Connectome, output: &mut impl Write) -> io::Result<u64> {
    graph.validate()?;
    if graph.ids.is_empty()
        || graph.ids.contains(&0)
        || graph.edges() > 20_000_000
    {
        return Err(invalid("consumer node/edge limits or zero root ID"));
    }
    let mut contacts = 0_u64;
    for source in 0..graph.ids.len() {
        let start = graph.offsets[source] as usize;
        let end = graph.offsets[source + 1] as usize;
        let row = &graph.targets[start..end];
        if row.windows(2).any(|pair| pair[0] >= pair[1])
            || row.contains(&(source as u32))
        {
            return Err(invalid("consumer rejects duplicate/unsorted targets or self loops"));
        }
        for &count in &graph.counts[start..end] {
            if count == 0 {
                return Err(invalid("consumer rejects zero contact multiplicity"));
            }
            contacts = contacts
                .checked_add(count)
                .ok_or_else(|| invalid("consumer contact sum overflow"))?;
        }
    }
    output.write_all(b"V8CSR001")?;
    output.write_all(&(graph.ids.len() as u64).to_le_bytes())?;
    output.write_all(&(graph.edges() as u64).to_le_bytes())?;
    for &id in &graph.ids {
        output.write_all(&id.to_le_bytes())?;
    }
    for &offset in &graph.offsets {
        output.write_all(&offset.to_le_bytes())?;
    }
    for &target in &graph.targets {
        output.write_all(&target.to_le_bytes())?;
    }
    for &count in &graph.counts {
        output.write_all(&count.to_le_bytes())?;
    }
    Ok(contacts)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: cargo run --example export_v8csr -- INPUT.v888b OUTPUT.csr".into());
    }
    let graph = Connectome::load(Path::new(&args[0]))?;
    let path = Path::new(&args[1]);
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    let result = export(&graph, &mut output).and_then(|contacts| {
        output.sync_all()?;
        Ok(contacts)
    });
    drop(output);
    match result {
        Ok(contacts) => {
            println!(
                "format=V8CSR001 nodes={} directed_pairs={} contacts={contacts}",
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Connectome {
        Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 2, 2, 3],
            targets: vec![1, 2, 0],
            counts: vec![3, 5, 7],
        }
    }

    #[test]
    fn exact_consumer_layout_keeps_targets_separate_from_counts() {
        let mut bytes = Vec::new();
        assert_eq!(export(&fixture(), &mut bytes).unwrap(), 15);
        let mut expected = b"V8CSR001".to_vec();
        for value in [3_u64, 3, 10, 20, 30, 0, 2, 2, 3] {
            expected.extend_from_slice(&value.to_le_bytes());
        }
        for target in [1_u32, 2, 0] {
            expected.extend_from_slice(&target.to_le_bytes());
        }
        for count in [3_u64, 5, 7] {
            expected.extend_from_slice(&count.to_le_bytes());
        }
        assert_eq!(bytes, expected);
        assert_eq!(bytes.len(), 32 + 16 * 3 + 12 * 3);
    }

    #[test]
    fn incompatible_rows_fail_before_output() {
        let original = fixture();
        let mut cases = Vec::new();
        let mut graph = original.clone();
        graph.ids[0] = 0;
        cases.push(graph);
        let mut graph = original.clone();
        graph.targets[0] = 0;
        cases.push(graph);
        let mut graph = original.clone();
        graph.targets[1] = 1;
        cases.push(graph);
        let mut graph = original.clone();
        graph.targets.swap(0, 1);
        cases.push(graph);
        let mut graph = original.clone();
        graph.counts[0] = 0;
        cases.push(graph);
        let mut graph = original;
        graph.counts[0] = u64::MAX;
        cases.push(graph);
        for graph in cases {
            let mut bytes = Vec::new();
            assert!(export(&graph, &mut bytes).is_err());
            assert!(bytes.is_empty());
        }
    }
}

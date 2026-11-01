//! Recurrent CSR exchange for SML research, preserving self loops.
//!
//! V8CSR002 is a distinct contract from SciRust V8CSR001. Its payload has
//! identical arrays, but permits source == target. No physiological semantics
//! or source equivalence is implied by this format.
use crate::connectome::Connectome;
use std::io::{self, Write};

pub const MAGIC: &[u8; 8] = b"V8CSR002";

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn validate(graph: &Connectome) -> io::Result<u64> {
    graph.validate()?;
    if graph.ids.is_empty() || graph.ids.contains(&0) || graph.edges() > 20_000_000 {
        return Err(invalid("recurrent CSR dimensions or root ID"));
    }
    let mut contacts = 0_u64;
    for source in 0..graph.ids.len() {
        let start = graph.offsets[source] as usize;
        let end = graph.offsets[source + 1] as usize;
        if graph.targets[start..end]
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid("recurrent CSR targets must be strictly increasing"));
        }
    }
    for &count in &graph.counts {
        if count == 0 {
            return Err(invalid("zero contact multiplicity"));
        }
        contacts = contacts
            .checked_add(count)
            .ok_or_else(|| invalid("contact sum overflow"))?;
    }
    Ok(contacts)
}

/// Write the full graph, including self loops, after validating all invariants.
pub fn write(graph: &Connectome, output: &mut impl Write) -> io::Result<u64> {
    let contacts = validate(graph)?;
    output.write_all(MAGIC)?;
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

/// Decode a complete byte slice with strict length and resource ceilings.
/// Validation precedes any use by a recurrent consumer.
pub fn read(bytes: &[u8]) -> io::Result<Connectome> {
    fn take<const N: usize>(bytes: &[u8], cursor: &mut usize) -> io::Result<[u8; N]> {
        let end = cursor
            .checked_add(N)
            .ok_or_else(|| invalid("cursor overflow"))?;
        let value = bytes
            .get(*cursor..end)
            .ok_or_else(|| invalid("truncated recurrent CSR"))?;
        *cursor = end;
        value.try_into().map_err(|_| invalid("invalid field"))
    }
    let mut cursor = 0;
    if &take::<8>(bytes, &mut cursor)? != MAGIC {
        return Err(invalid("wrong recurrent CSR magic"));
    }
    let n = u64::from_le_bytes(take(bytes, &mut cursor)?);
    let e = u64::from_le_bytes(take(bytes, &mut cursor)?);
    if n == 0 || n > 1_000_000 || e > 20_000_000 || bytes.len() as u64 != 32 + 16 * n + 12 * e {
        return Err(invalid("recurrent CSR length or dimensions"));
    }
    let mut ids = Vec::with_capacity(n as usize);
    let mut offsets = Vec::with_capacity(n as usize + 1);
    let mut targets = Vec::with_capacity(e as usize);
    let mut counts = Vec::with_capacity(e as usize);
    for _ in 0..n {
        ids.push(u64::from_le_bytes(take(bytes, &mut cursor)?));
    }
    for _ in 0..=n {
        offsets.push(u64::from_le_bytes(take(bytes, &mut cursor)?));
    }
    for _ in 0..e {
        targets.push(u32::from_le_bytes(take(bytes, &mut cursor)?));
    }
    for _ in 0..e {
        counts.push(u64::from_le_bytes(take(bytes, &mut cursor)?));
    }
    let graph = Connectome {
        ids,
        offsets,
        targets,
        counts,
    };
    validate(&graph)?;
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Connectome {
        Connectome {
            ids: vec![10, 20],
            offsets: vec![0, 2, 3],
            targets: vec![0, 1, 1],
            counts: vec![3, 5, 7],
        }
    }

    #[test]
    fn exact_layout_preserves_both_self_loops() {
        let mut bytes = Vec::new();
        assert_eq!(write(&fixture(), &mut bytes).unwrap(), 15);
        let mut expected = b"V8CSR002".to_vec();
        for value in [2_u64, 3, 10, 20, 0, 2, 3] {
            expected.extend_from_slice(&value.to_le_bytes());
        }
        for target in [0_u32, 1, 1] {
            expected.extend_from_slice(&target.to_le_bytes());
        }
        for count in [3_u64, 5, 7] {
            expected.extend_from_slice(&count.to_le_bytes());
        }
        assert_eq!(bytes, expected);
        assert_eq!(read(&expected).unwrap(), fixture());
    }

    #[test]
    fn malformed_or_wrong_version_bytes_are_rejected() {
        let mut original = Vec::new();
        write(&fixture(), &mut original).unwrap();
        for len in 0..original.len() {
            assert!(read(&original[..len]).is_err());
        }
        let mut bytes = original.clone();
        bytes.push(0);
        assert!(read(&bytes).is_err());
        for (offset, value) in [(7, b'1'), (24, 0), (48, 4), (68, 2), (72, 2), (76, 0)] {
            let mut bytes = original.clone();
            bytes[offset] = value;
            assert!(read(&bytes).is_err(), "accepted corruption at {offset}");
        }
    }

    #[test]
    fn invalid_graphs_write_nothing() {
        let mut graph = fixture();
        graph.counts[0] = u64::MAX;
        let mut bytes = Vec::new();
        assert!(write(&graph, &mut bytes).is_err());
        assert!(bytes.is_empty());
        graph = fixture();
        graph.targets[1] = 0;
        assert!(write(&graph, &mut bytes).is_err());
        assert!(bytes.is_empty());
    }
}

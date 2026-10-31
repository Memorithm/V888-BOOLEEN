//! Lossless and streaming Boolean/bit-level projection of individual synapse
//! pre/post root IDs. All OTHER columns remain in the original SHA-pinned Parquet.
//! Do not confuse this projection with complete reconstruction of the source.
use crate::connectome::Connectome;
use crate::gates;
use crate::BooleanState;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"V888SYB1";
const VERSION: u64 = 1;
const HEADER_SIZE: u64 = 24;
const RECORD_SIZE: u64 = 17;
const MAX_ROWS: u64 = 300_000_000;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// A decoded, individual synapse. Null and a valid zero root ID differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynapsePair {
    pub pre: Option<u64>,
    pub post: Option<u64>,
}

impl SynapsePair {
    #[must_use]
    pub fn encode(self) -> [u8; 17] {
        let mut record = [0_u8; 17];
        if let Some(pre) = self.pre {
            record[0] |= 1;
            record[1..9].copy_from_slice(&pre.to_le_bytes());
        }
        if let Some(post) = self.post {
            record[0] |= 2;
            record[9..17].copy_from_slice(&post.to_le_bytes());
        }
        record
    }

    pub fn decode(record: [u8; 17]) -> io::Result<Self> {
        if record[0] & !3 != 0 {
            return Err(invalid("unknown validity flags"));
        }
        let mut pre_bytes = [0_u8; 8];
        let mut post_bytes = [0_u8; 8];
        pre_bytes.copy_from_slice(&record[1..9]);
        post_bytes.copy_from_slice(&record[9..17]);
        let pre = u64::from_le_bytes(pre_bytes);
        let post = u64::from_le_bytes(post_bytes);
        if record[0] & 1 == 0 && pre != 0 {
            return Err(invalid("nonzero null presynaptic identifier"));
        }
        if record[0] & 2 == 0 && post != 0 {
            return Err(invalid("nonzero null postsynaptic identifier"));
        }
        Ok(Self {
            pre: (record[0] & 1 != 0).then_some(pre),
            post: (record[0] & 2 != 0).then_some(post),
        })
    }
}

/// Encode raw 17-byte records from an exact-ID adapter, with an explicit
/// row count and fixed binary header. Never materializes the input in RAM.
/// Each source record is validated, including the two independent null bits.
pub fn pack<R: Read, W: Write + Seek>(
    source: &mut R,
    output: &mut W,
    expected_rows: u64,
) -> io::Result<u64> {
    if expected_rows == 0 || expected_rows > MAX_ROWS {
        return Err(invalid("invalid expected row count"));
    }
    output.write_all(MAGIC)?;
    output.write_all(&VERSION.to_le_bytes())?;
    output.write_all(&0_u64.to_le_bytes())?;
    let mut rows = 0_u64;
    let mut flag = [0_u8; 1];
    loop {
        match source.read(&mut flag)? {
            0 => break,
            1 => {}
            _ => unreachable!("single-byte buffer"),
        }
        let mut record = [0_u8; 17];
        record[0] = flag[0];
        source.read_exact(&mut record[1..])?;
        let pair = SynapsePair::decode(record)?;
        if rows >= expected_rows {
            return Err(invalid("source exceeds expected synapse row count"));
        }
        output.write_all(&pair.encode())?;
        rows += 1;
    }
    if rows != expected_rows {
        return Err(invalid("synapse row count mismatch"));
    }
    output.seek(SeekFrom::Start(16))?;
    output.write_all(&rows.to_le_bytes())?;
    output.flush()?;
    Ok(rows)
}

/// Verified, bounded random-free streaming reader for the pair projection.
pub struct SynapsePairsReader {
    input: BufReader<File>,
    count: u64,
}

impl SynapsePairsReader {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        let mut input = BufReader::new(file);
        let mut header = [0_u8; 24];
        input.read_exact(&mut header)?;
        if &header[0..8] != MAGIC {
            return Err(invalid("invalid synapse format magic"));
        }
        if u64::from_le_bytes(header[8..16].try_into().map_err(|_| invalid("version"))?) != VERSION {
            return Err(invalid("unsupported synapse format version"));
        }
        let count = u64::from_le_bytes(header[16..24].try_into().map_err(|_| invalid("count"))?);
        if count == 0 || count > MAX_ROWS {
            return Err(invalid("invalid synapse count"));
        }
        let needed = HEADER_SIZE + RECORD_SIZE * count;
        if size != needed {
            return Err(invalid("synapse file length mismatch"));
        }
        Ok(Self { input, count })
    }

    #[must_use]
    pub fn len(&self) -> u64 {
        self.count
    }

    pub fn visit<F>(&mut self, mut visitor: F) -> io::Result<()>
    where
        F: FnMut(SynapsePair) -> io::Result<()>,
    {
        for _ in 0..self.count {
            let mut row = [0_u8; 17];
            self.input.read_exact(&mut row)?;
            visitor(SynapsePair::decode(row)?)?;
        }
        Ok(())
    }
}

/// One SYNCHRONOUS experimental recurrence from synapse-level contacts.
/// Unknown/unproofread neuron IDs are preserved in the source binary, but
/// cannot activate a neuron absent from the explicit graph universe.
pub fn propagate_one_step(
    graph: &Connectome,
    active: &BooleanState,
    pairs: &mut SynapsePairsReader,
    threshold: u64,
) -> io::Result<(BooleanState, u64)> {
    graph.validate()?;
    if threshold == 0 || active.len() != graph.ids.len() {
        return Err(invalid("invalid recurrence arguments"));
    }
    let mut active_pre = HashSet::new();
    for (index, &id) in graph.ids.iter().enumerate() {
        if active.get(index).map_err(|_| invalid("invalid active index"))? {
            active_pre.insert(id);
        }
    }
    let mut sums = vec![0_u128; graph.ids.len()];
    let mut matched_contacts = 0_u64;
    pairs.visit(|pair| {
        if let (Some(pre), Some(post)) = (pair.pre, pair.post) {
            if active_pre.contains(&pre) {
                if let Ok(target) = graph.ids.binary_search(&post) {
                    sums[target] = gates::add(sums[target], 1)?;
                    matched_contacts += 1;
                }
            }
        }
        Ok(())
    })?;
    let mut next = BooleanState::new(graph.ids.len());
    for (i, &sum) in sums.iter().enumerate() {
        if gates::ge(sum, threshold as u128) {
            next.set(i, true).map_err(|_| invalid("invalid target index"))?;
        }
    }
    Ok((next, matched_contacts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn exact_pair_encoding_with_nulls_and_zero() {
        for pair in [
            SynapsePair { pre: None, post: None },
            SynapsePair { pre: Some(0), post: None },
            SynapsePair { pre: None, post: Some(u64::MAX) },
            SynapsePair { pre: Some(u64::MAX), post: Some(0) },
        ] {
            assert_eq!(SynapsePair::decode(pair.encode()).unwrap(), pair);
        }
        let mut invalid = SynapsePair { pre: None, post: None }.encode();
        invalid[1] = 1;
        assert!(SynapsePair::decode(invalid).is_err());
        invalid[1] = 0;
        invalid[0] = 4;
        assert!(SynapsePair::decode(invalid).is_err());
    }

    #[test]
    fn exact_binary_stream_roundtrip() {
        let original = [
            SynapsePair { pre: Some(10), post: Some(20) },
            SynapsePair { pre: Some(10), post: Some(20) },
            SynapsePair { pre: None, post: Some(20) },
            SynapsePair { pre: Some(30), post: None },
        ];
        let raw: Vec<u8> = original.iter().flat_map(|pair| pair.encode()).collect();
        let mut target = Cursor::new(Vec::new());
        assert_eq!(pack(&mut Cursor::new(raw.clone()), &mut target, 4).unwrap(), 4);
        assert_eq!(target.get_ref().len(), 24 + 4 * 17);
        let mut invalid = Cursor::new(Vec::new());
        assert!(pack(&mut Cursor::new(raw), &mut invalid, 5).is_err());
    }

    #[test]
    fn truncated_raw_record_fails_closed() {
        let mut input = Cursor::new(vec![3_u8, 1, 2, 3]);
        let mut target = Cursor::new(Vec::new());
        assert!(pack(&mut input, &mut target, 1).is_err());
    }

    #[test]
    fn invalid_row_count_rejected() {
        let mut empty = Cursor::new(Vec::<u8>::new());
        let mut target = Cursor::new(Vec::new());
        assert!(pack(&mut empty, &mut target, 0).is_err());
        assert!(pack(&mut empty, &mut target, 300_000_001).is_err());
    }
}

//! Compact lossless (for pre/post/count columns) directed BANC graph.
//! Original Feather source, synapse rows, and annotations are separate.
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"V888BOL1";
fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, s.into())
}
fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Connectome {
    pub ids: Vec<u64>,
    pub offsets: Vec<u64>,
    pub targets: Vec<u32>,
    pub counts: Vec<u64>,
}
impl Connectome {
    pub fn validate(&self) -> io::Result<()> {
        if self.ids.len() > 1_000_000 || self.targets.len() > 300_000_000 {
            return Err(invalid("graph too large"));
        }
        if self.ids.windows(2).any(|w| w[0] >= w[1]) {
            return Err(invalid("nonunique/unsorted IDs"));
        }
        if self.offsets.len() != self.ids.len() + 1
            || self.offsets.first() != Some(&0)
            || self.offsets.last() != Some(&(self.targets.len() as u64))
            || self.offsets.windows(2).any(|w| w[0] > w[1])
            || self.targets.len() != self.counts.len()
            || self.targets.iter().any(|&v| v as usize >= self.ids.len())
        {
            return Err(invalid("invalid CSR arrays"));
        }
        Ok(())
    }
    pub fn index(&self, id: u64) -> Option<usize> {
        self.ids.binary_search(&id).ok()
    }
    pub fn edges(&self) -> usize {
        self.targets.len()
    }
    /// Numeric CSV: pre,post,count in first three fields, no quoted fields.
    /// Optional IDs list preserves neurons with no retained edges.
    pub fn import(edges: &Path, neurons: Option<&Path>) -> io::Result<Self> {
        let mut ids = Vec::<u64>::new();
        if let Some(path) = neurons {
            for (i, row) in BufReader::new(File::open(path)?).lines().enumerate() {
                let row = row?;
                if i == 0 && row == "id" {
                    continue;
                }
                ids.push(
                    row.parse()
                        .map_err(|_| invalid(format!("invalid ID line {}", i + 1)))?,
                );
                if ids.len() > 1_000_000 {
                    return Err(invalid("node limit exceeded"));
                }
            }
        }
        let mut raw = Vec::<(u64, u64, u64)>::new();
        for (i, row) in BufReader::new(File::open(edges)?).lines().enumerate() {
            let row = row?;
            let fields: Vec<&str> = row.trim_end_matches('\r').split(',').collect();
            if i == 0 {
                if fields.len() < 3
                    || fields[0].trim_start_matches('\u{feff}') != "pre"
                    || fields[1] != "post"
                    || fields[2] != "count"
                {
                    return Err(invalid("expected pre,post,count CSV header"));
                }
                continue;
            }
            if fields.len() < 3 {
                return Err(invalid(format!("invalid line {}", i + 1)));
            }
            let parse = |j: usize| {
                fields[j]
                    .parse::<u64>()
                    .map_err(|_| invalid(format!("invalid integer at line {}", i + 1)))
            };
            let pre = parse(0)?;
            let post = parse(1)?;
            let count = parse(2)?;
            raw.push((pre, post, count));
            ids.extend([pre, post]);
            if raw.len() > 300_000_000 {
                return Err(invalid("edge limit exceeded"));
            }
        }
        ids.sort_unstable();
        ids.dedup();
        if ids.len() > 1_000_000 || ids.len() > u32::MAX as usize {
            return Err(invalid("node limit exceeded"));
        }
        raw.sort_unstable(); // deterministic output, NO duplicate collapsing.
        let mut offsets = vec![0; ids.len() + 1];
        let mut targets = Vec::with_capacity(raw.len());
        let mut counts = Vec::with_capacity(raw.len());
        let mut cursor = 0;
        for (pre, post, count) in raw {
            let source = ids
                .binary_search(&pre)
                .map_err(|_| invalid("source absent"))?;
            let target = ids
                .binary_search(&post)
                .map_err(|_| invalid("target absent"))?;
            while cursor < source {
                offsets[cursor + 1] = targets.len() as u64;
                cursor += 1;
            }
            targets.push(u32::try_from(target).map_err(|_| invalid("node overflow"))?);
            counts.push(count);
        }
        while cursor < ids.len() {
            offsets[cursor + 1] = targets.len() as u64;
            cursor += 1;
        }
        let result = Self {
            ids,
            offsets,
            targets,
            counts,
        };
        result.validate()?;
        Ok(result)
    }
    /// Write a lossless CSR serialization. The graph may be a derived
    /// projection, but every stored edge row remains exact.
    fn write_to<W: Write>(&self, mut out: W) -> io::Result<()> {
        self.validate()?;
        out.write_all(MAGIC)?;
        out.write_all(&1_u64.to_le_bytes())?;
        out.write_all(&(self.ids.len() as u64).to_le_bytes())?;
        out.write_all(&(self.targets.len() as u64).to_le_bytes())?;
        for id in &self.ids {
            out.write_all(&id.to_le_bytes())?;
        }
        for offset in &self.offsets {
            out.write_all(&offset.to_le_bytes())?;
        }
        for (target, count) in self.targets.iter().zip(&self.counts) {
            out.write_all(&target.to_le_bytes())?;
            out.write_all(&count.to_le_bytes())?;
        }
        out.flush()
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        self.write_to(BufWriter::new(File::create(path)?))
    }

    /// Create a new graph file, refusing to overwrite existing files,
    /// including symlinks to original biological source material.
    pub fn save_new(&self, path: &Path) -> io::Result<()> {
        use std::fs::OpenOptions;
        self.validate()?;
        self.write_to(BufWriter::new(
            OpenOptions::new().write(true).create_new(true).open(path)?,
        ))
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        let mut input = BufReader::new(file);
        let mut magic = [0; 8];
        input.read_exact(&mut magic)?;
        if &magic != MAGIC || read_u64(&mut input)? != 1 {
            return Err(invalid("unknown graph format"));
        }
        let n = read_u64(&mut input)?;
        let e = read_u64(&mut input)?;
        if n > 1_000_000 || n > u32::MAX as u64 || e > 300_000_000 {
            return Err(invalid("size limit"));
        }
        let expected = 32 + 16 * n + 8 + 12 * e;
        if size != expected {
            return Err(invalid("truncated/trailing graph bytes"));
        }
        let mut ids = Vec::with_capacity(n as usize);
        let mut offsets = Vec::with_capacity(n as usize + 1);
        let mut targets = Vec::with_capacity(e as usize);
        let mut counts = Vec::with_capacity(e as usize);
        for _ in 0..n {
            ids.push(read_u64(&mut input)?);
        }
        for _ in 0..=n {
            offsets.push(read_u64(&mut input)?);
        }
        for _ in 0..e {
            targets.push(read_u32(&mut input)?);
            counts.push(read_u64(&mut input)?);
        }
        let graph = Self {
            ids,
            offsets,
            targets,
            counts,
        };
        graph.validate()?;
        Ok(graph)
    }
    /// Exact row multiset after roundtrip. Original source row order is not retained.
    pub fn export_csv(&self, path: &Path) -> io::Result<()> {
        self.validate()?;
        let mut w = BufWriter::new(File::create(path)?);
        writeln!(w, "pre,post,count")?;
        for (source, &id) in self.ids.iter().enumerate() {
            for edge in self.offsets[source]..self.offsets[source + 1] {
                let e = edge as usize;
                writeln!(
                    w,
                    "{},{},{}",
                    id, self.ids[self.targets[e] as usize], self.counts[e]
                )?;
            }
        }
        w.flush()
    }
}

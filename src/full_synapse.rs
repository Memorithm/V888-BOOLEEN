//! Exact, bounded, row-oriented codec for ALL 30 scalar fields in the
//! acquired BANC V888-v2 enriched synapse Parquet schema.
//!
//! `f64` values are treated as raw IEEE-754 u64 words, never quantised or
//! interpreted, including NaN payload and negative zero. String bytes are
//! checked as UTF-8, but never normalised. NULL is independent of the value.
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"V888C30A";
const VERSION: u64 = 1;
const MAX_ROWS: u64 = 300_000_000;
const MAX_TEXT_BYTES: usize = 1_048_576;
pub const FIELD_COUNT: usize = 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnType {
    Text,
    FloatBits,
    Int32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Column {
    pub name: &'static str,
    pub kind: ColumnType,
}

const fn column(name: &'static str, kind: ColumnType) -> Column {
    Column { name, kind }
}

pub const COLUMNS: [Column; FIELD_COUNT] = [
    column("id", ColumnType::Text),
    column("size", ColumnType::FloatBits),
    column("pre_root_id", ColumnType::Text),
    column("post_root_id", ColumnType::Text),
    column("X", ColumnType::FloatBits),
    column("Y", ColumnType::FloatBits),
    column("Z", ColumnType::FloatBits),
    column("pre_x", ColumnType::FloatBits),
    column("pre_y", ColumnType::FloatBits),
    column("pre_z", ColumnType::FloatBits),
    column("post_x", ColumnType::FloatBits),
    column("post_y", ColumnType::FloatBits),
    column("post_z", ColumnType::FloatBits),
    column("pre_supervoxel_id", ColumnType::Text),
    column("post_supervoxel_id", ColumnType::Text),
    column("neuropil", ColumnType::Text),
    column("region", ColumnType::Text),
    column("side", ColumnType::Text),
    column("neuropil_detailed", ColumnType::Text),
    column("acetylcholine", ColumnType::FloatBits),
    column("dopamine", ColumnType::FloatBits),
    column("gaba", ColumnType::FloatBits),
    column("glutamate", ColumnType::FloatBits),
    column("histamine", ColumnType::FloatBits),
    column("octopamine", ColumnType::FloatBits),
    column("serotonin", ColumnType::FloatBits),
    column("tyramine", ColumnType::FloatBits),
    column("syn_top_nt", ColumnType::Text),
    column("syn_top_p", ColumnType::FloatBits),
    column("label", ColumnType::Int32),
];

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cell {
    Null,
    Text(Vec<u8>),
    FloatBits(u64),
    Int32(i32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SynapseRow {
    pub values: [Cell; FIELD_COUNT],
}

impl Default for SynapseRow {
    fn default() -> Self {
        Self {
            values: std::array::from_fn(|_| Cell::Null),
        }
    }
}

fn read_u32<R: Read>(input: &mut R) -> io::Result<u32> {
    let mut raw = [0_u8; 4];
    input.read_exact(&mut raw)?;
    Ok(u32::from_le_bytes(raw))
}

fn read_u64<R: Read>(input: &mut R) -> io::Result<u64> {
    let mut raw = [0_u8; 8];
    input.read_exact(&mut raw)?;
    Ok(u64::from_le_bytes(raw))
}

impl SynapseRow {
    pub fn write_to<W: Write>(&self, output: &mut W) -> io::Result<()> {
        let mut presence = 0_u32;
        for (index, value) in self.values.iter().enumerate() {
            match (COLUMNS[index].kind, value) {
                (_, Cell::Null) => {}
                (ColumnType::Text, Cell::Text(bytes)) => {
                    if bytes.len() > MAX_TEXT_BYTES || std::str::from_utf8(bytes).is_err() {
                        return Err(invalid("invalid UTF-8 or oversized text"));
                    }
                    presence |= 1 << index;
                }
                (ColumnType::FloatBits, Cell::FloatBits(_))
                | (ColumnType::Int32, Cell::Int32(_)) => {
                    presence |= 1 << index;
                }
                _ => return Err(invalid("field type does not match V888 30-column schema")),
            }
        }
        output.write_all(&presence.to_le_bytes())?;
        for value in &self.values {
            match value {
                Cell::Null => {}
                Cell::Text(bytes) => {
                    let len =
                        u32::try_from(bytes.len()).map_err(|_| invalid("text size overflow"))?;
                    output.write_all(&len.to_le_bytes())?;
                    output.write_all(bytes)?;
                }
                Cell::FloatBits(word) => output.write_all(&word.to_le_bytes())?,
                Cell::Int32(value) => output.write_all(&value.to_le_bytes())?,
            }
        }
        Ok(())
    }

    pub fn read_from<R: Read>(input: &mut R) -> io::Result<Self> {
        let presence = read_u32(input)?;
        if presence >> FIELD_COUNT != 0 {
            return Err(invalid("unknown field presence bits"));
        }
        let mut result = Self::default();
        for (index, column) in COLUMNS.iter().enumerate() {
            if presence & (1 << index) == 0 {
                continue;
            }
            result.values[index] = match column.kind {
                ColumnType::Text => {
                    let len = read_u32(input)? as usize;
                    if len > MAX_TEXT_BYTES {
                        return Err(invalid("oversized text field"));
                    }
                    let mut bytes = vec![0_u8; len];
                    input.read_exact(&mut bytes)?;
                    if std::str::from_utf8(&bytes).is_err() {
                        return Err(invalid("invalid UTF-8 text"));
                    }
                    Cell::Text(bytes)
                }
                ColumnType::FloatBits => Cell::FloatBits(read_u64(input)?),
                ColumnType::Int32 => Cell::Int32(read_u32(input)? as i32),
            };
        }
        Ok(result)
    }
}

/// Input is a binary stream of individual 30-column records in schema order:
/// u32 presence bitmap then typed fields. The header is created only after
/// the declared row count passes validation; every row is decoded+re-encoded.
pub fn pack_records<R: Read, W: Write>(
    input: &mut R,
    output: &mut W,
    expected_rows: u64,
) -> io::Result<u64> {
    if expected_rows == 0 || expected_rows > MAX_ROWS {
        return Err(invalid("invalid row count"));
    }
    output.write_all(MAGIC)?;
    output.write_all(&VERSION.to_le_bytes())?;
    output.write_all(&expected_rows.to_le_bytes())?;
    for _ in 0..expected_rows {
        let row = SynapseRow::read_from(input)?;
        row.write_to(output)?;
    }
    let mut extra = [0_u8; 1];
    if input.read(&mut extra)? != 0 {
        return Err(invalid("trailing synapse records"));
    }
    output.flush()?;
    Ok(expected_rows)
}

pub struct FullSynapseReader {
    input: BufReader<File>,
    rows: u64,
}

impl FullSynapseReader {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        if file.metadata()?.len() < 24 {
            return Err(invalid("file shorter than header"));
        }
        let mut input = BufReader::new(file);
        let mut magic = [0_u8; 8];
        input.read_exact(&mut magic)?;
        if &magic != MAGIC || read_u64(&mut input)? != VERSION {
            return Err(invalid("invalid full synapse binary version"));
        }
        let rows = read_u64(&mut input)?;
        if rows == 0 || rows > MAX_ROWS {
            return Err(invalid("invalid row count in file"));
        }
        if input.get_ref().metadata()?.len() < 24 + 4 * rows {
            return Err(invalid("binary shorter than required null bitmaps"));
        }
        Ok(Self { input, rows })
    }

    #[must_use]
    pub fn len(&self) -> u64 {
        self.rows
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    pub fn visit<F>(&mut self, mut visitor: F) -> io::Result<()>
    where
        F: FnMut(SynapseRow) -> io::Result<()>,
    {
        for _ in 0..self.rows {
            visitor(SynapseRow::read_from(&mut self.input)?)?;
        }
        let mut extra = [0_u8; 1];
        if self.input.read(&mut extra)? != 0 {
            return Err(invalid("trailing full synapse data"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn schema_stays_exactly_thirty_columns() {
        assert_eq!(COLUMNS.len(), 30);
        assert_eq!(COLUMNS[0].name, "id");
        assert_eq!(COLUMNS[2].name, "pre_root_id");
        assert_eq!(COLUMNS[19].name, "acetylcholine");
        assert_eq!(COLUMNS[28].name, "syn_top_p");
        assert_eq!(COLUMNS[29].name, "label");
        assert_eq!(
            COLUMNS
                .iter()
                .filter(|c| c.kind == ColumnType::Text)
                .count(),
            10
        );
        assert_eq!(
            COLUMNS
                .iter()
                .filter(|c| c.kind == ColumnType::FloatBits)
                .count(),
            19
        );
        assert_eq!(
            COLUMNS
                .iter()
                .filter(|c| c.kind == ColumnType::Int32)
                .count(),
            1
        );
    }

    #[test]
    fn all_columns_roundtrip_with_nan_payload_and_signed_zero() -> io::Result<()> {
        let mut original = SynapseRow::default();
        for (index, column) in COLUMNS.iter().enumerate() {
            original.values[index] = match column.kind {
                ColumnType::Text => Cell::Text(format!("drosophila-{index}").into_bytes()),
                ColumnType::FloatBits => Cell::FloatBits(0x7ff8_0000_0000_0123),
                ColumnType::Int32 => Cell::Int32(-123),
            };
        }
        original.values[1] = Cell::FloatBits(1_u64 << 63); // -0.0
        original.values[4] = Cell::FloatBits(0); // +0.0
        original.values[15] = Cell::Text(Vec::new()); // empty != NULL
        original.values[17] = Cell::Null;
        let mut raw = Vec::new();
        original.write_to(&mut raw)?;
        let decoded = SynapseRow::read_from(&mut Cursor::new(&raw))?;
        assert_eq!(decoded, original);
        let mut packed = Vec::new();
        assert_eq!(pack_records(&mut Cursor::new(raw), &mut packed, 1)?, 1);
        assert_eq!(&packed[..8], MAGIC);
        assert_eq!(u64::from_le_bytes(packed[16..24].try_into().unwrap()), 1);
        Ok(())
    }

    #[test]
    fn null_only_row_and_repetition_are_preserved() -> io::Result<()> {
        let row = SynapseRow::default();
        let mut raw = Vec::new();
        row.write_to(&mut raw)?;
        row.write_to(&mut raw)?;
        let mut packed = Vec::new();
        assert_eq!(pack_records(&mut Cursor::new(&raw), &mut packed, 2)?, 2);
        assert_eq!(packed.len(), 24 + 2 * 4);
        assert_eq!(&packed[24..], &[0_u8; 8]);
        Ok(())
    }

    #[test]
    fn invalid_mask_wrong_type_and_truncation_fail_closed() {
        let mut raw = Cursor::new(u32::MAX.to_le_bytes());
        assert!(SynapseRow::read_from(&mut raw).is_err());
        let mut row = SynapseRow::default();
        row.values[0] = Cell::FloatBits(0);
        assert!(row.write_to(&mut Vec::new()).is_err());
        let mut row = SynapseRow::default();
        row.values[0] = Cell::Text(vec![0xff]);
        assert!(row.write_to(&mut Vec::new()).is_err());
        let mut row = SynapseRow::default();
        row.values[0] = Cell::Text(vec![b'a'; MAX_TEXT_BYTES + 1]);
        assert!(row.write_to(&mut Vec::new()).is_err());
        let mut out = Vec::new();
        assert!(pack_records(&mut Cursor::new([0_u8; 3]), &mut out, 1).is_err());
        let mut out = Vec::new();
        assert!(pack_records(&mut Cursor::new([0_u8; 4]), &mut out, 0).is_err());
    }

    #[test]
    fn reject_trailing_records_after_expected_count() -> io::Result<()> {
        let mut raw = vec![0_u8; 8]; // two null-only records
        let mut out = Vec::new();
        assert!(pack_records(&mut Cursor::new(&mut raw), &mut out, 1).is_err());
        Ok(())
    }
}

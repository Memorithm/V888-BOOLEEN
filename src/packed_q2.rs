//! Experimental, lossless two-bit packing and bit-sliced arithmetic.
//!
//! Every 64-bit word in each plane represents 64 four-level values.
//! Numeric Q2 dot products are NOT GF(4) products.
//! This module does not change the active neuron/event runtime.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q2Error {
    OutOfBounds,
    InvalidLevel,
    LengthMismatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedQ2 {
    len: usize,
    low: Vec<u64>,
    high: Vec<u64>,
}

impl PackedQ2 {
    #[must_use]
    pub fn new(len: usize) -> Self {
        let count = len / 64 + usize::from(len % 64 != 0);
        Self {
            len,
            low: vec![0; count],
            high: vec![0; count],
        }
    }

    pub fn from_levels(values: &[u8]) -> Result<Self, Q2Error> {
        let mut packed = Self::new(values.len());
        for (index, &value) in values.iter().enumerate() {
            packed.set(index, value)?;
        }
        Ok(packed)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Data payload only: excludes capacity rounding, allocator and Vec metadata.
    #[must_use]
    pub fn allocated_payload_bytes(&self) -> usize {
        (self.low.len() + self.high.len()) * std::mem::size_of::<u64>()
    }

    pub fn set(&mut self, index: usize, value: u8) -> Result<(), Q2Error> {
        if index >= self.len {
            return Err(Q2Error::OutOfBounds);
        }
        if value > 3 {
            return Err(Q2Error::InvalidLevel);
        }
        let word = index / 64;
        let mask = 1_u64 << (index % 64);
        self.low[word] = (self.low[word] & !mask) | (u64::from(value & 1) * mask);
        self.high[word] = (self.high[word] & !mask) | (u64::from(value >> 1) * mask);
        Ok(())
    }

    pub fn get(&self, index: usize) -> Result<u8, Q2Error> {
        if index >= self.len {
            return Err(Q2Error::OutOfBounds);
        }
        let word = index / 64;
        let bit = index % 64;
        Ok(((self.low[word] >> bit) & 1) as u8
            | ((((self.high[word] >> bit) & 1) as u8) << 1))
    }

    fn same_length(&self, other: &Self) -> Result<(), Q2Error> {
        if self.len == other.len {
            Ok(())
        } else {
            Err(Q2Error::LengthMismatch)
        }
    }

    /// Exact dot product with unsigned numeric codes 0,1,2,3.
    /// Four AND+POPCOUNT terms compute up to 64 products per word pair.
    pub fn dot_unsigned(&self, other: &Self) -> Result<u128, Q2Error> {
        self.same_length(other)?;
        let mut sum = 0_u128;
        for i in 0..self.low.len() {
            let n00 = u128::from((self.low[i] & other.low[i]).count_ones());
            let n01 = u128::from((self.low[i] & other.high[i]).count_ones());
            let n10 = u128::from((self.high[i] & other.low[i]).count_ones());
            let n11 = u128::from((self.high[i] & other.high[i]).count_ones());
            sum += n00 + 2 * (n01 + n10) + 4 * n11;
        }
        Ok(sum)
    }

    /// Exact dot with 2-bit two's-complement codes: 0,1,-2,-1.
    pub fn dot_signed(&self, other: &Self) -> Result<i128, Q2Error> {
        self.same_length(other)?;
        let mut sum = 0_i128;
        for i in 0..self.low.len() {
            let n00 = i128::from((self.low[i] & other.low[i]).count_ones());
            let n01 = i128::from((self.low[i] & other.high[i]).count_ones());
            let n10 = i128::from((self.high[i] & other.low[i]).count_ones());
            let n11 = i128::from((self.high[i] & other.high[i]).count_ones());
            sum += n00 - 2 * (n01 + n10) + 4 * n11;
        }
        Ok(sum)
    }

    /// Exact GF(4) inner product: XOR-reduction of per-lane field products.
    ///
    /// Both components are accumulated as POPCOUNT parity over 64 lanes.
    /// The return code represents 0, 1, x, or 1+x in GF(2)[x]/(x²+x+1),
    /// never an integer dot product or multiplication modulo four.
    pub fn gf4_dot(&self, other: &Self) -> Result<u8, Q2Error> {
        self.same_length(other)?;
        let mut sum = 0_u8;
        for i in 0..self.low.len() {
            let p00 = self.low[i] & other.low[i];
            let p01 = self.low[i] & other.high[i];
            let p10 = self.high[i] & other.low[i];
            let p11 = self.high[i] & other.high[i];
            let low_parity = ((p00 ^ p11).count_ones() & 1) as u8;
            let high_parity = ((p01 ^ p10 ^ p11).count_ones() & 1) as u8;
            sum ^= low_parity | (high_parity << 1);
        }
        Ok(sum)
    }

    /// GF(4) addition in polynomial basis: componentwise XOR.
    pub fn gf4_add(&self, other: &Self) -> Result<Self, Q2Error> {
        self.same_length(other)?;
        Ok(Self {
            len: self.len,
            low: self.low.iter().zip(&other.low).map(|(a, b)| a ^ b).collect(),
            high: self.high.iter().zip(&other.high).map(|(a, b)| a ^ b).collect(),
        })
    }

    /// GF(4) multiplication modulo x²+x+1, NOT multiplication mod 4.
    /// The codes 0,1,2,3 represent 0,1,x,1+x.
    pub fn gf4_mul(&self, other: &Self) -> Result<Self, Q2Error> {
        self.same_length(other)?;
        let mut result = Self::new(self.len);
        for i in 0..self.low.len() {
            let p00 = self.low[i] & other.low[i];
            let p01 = self.low[i] & other.high[i];
            let p10 = self.high[i] & other.low[i];
            let p11 = self.high[i] & other.high[i];
            result.low[i] = p00 ^ p11;
            result.high[i] = p01 ^ p10 ^ p11;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::{PackedQ2, Q2Error};

    fn signed(v: u8) -> i128 {
        i128::from(v & 1) - 2 * i128::from(v >> 1)
    }

    fn gf4_ref(a: u8, b: u8) -> u8 {
        let (a0, a1, b0, b1) = (a & 1, a >> 1, b & 1, b >> 1);
        ((a0 & b0) ^ (a1 & b1))
            | (((a0 & b1) ^ (a1 & b0) ^ (a1 & b1)) << 1)
    }

    #[test]
    fn errors_and_word_boundaries() {
        let mut q = PackedQ2::new(65);
        assert_eq!(q.allocated_payload_bytes(), 32);
        assert_eq!(q.set(65, 0), Err(Q2Error::OutOfBounds));
        assert_eq!(q.set(0, 4), Err(Q2Error::InvalidLevel));
        assert_eq!(q.get(65), Err(Q2Error::OutOfBounds));
        assert_eq!(PackedQ2::from_levels(&[0, 4]), Err(Q2Error::InvalidLevel));
        for (i, level) in [(0, 1), (63, 2), (64, 3)] {
            q.set(i, level).unwrap();
            assert_eq!(q.get(i), Ok(level));
        }
        assert_eq!(q.get(1), Ok(0));
        let short = PackedQ2::new(64);
        assert_eq!(q.dot_unsigned(&short), Err(Q2Error::LengthMismatch));
        assert_eq!(q.dot_signed(&short), Err(Q2Error::LengthMismatch));
        assert_eq!(q.gf4_add(&short), Err(Q2Error::LengthMismatch));
        assert_eq!(q.gf4_mul(&short), Err(Q2Error::LengthMismatch));
        assert_eq!(q.gf4_dot(&short), Err(Q2Error::LengthMismatch));
    }

    #[test]
    fn scalar_oracles_exhaustive_algebra_and_many_lengths() {
        for len in [0, 1, 2, 31, 63, 64, 65, 127, 128, 129, 188_508] {
            for seed in 0..4_usize {
                let a: Vec<u8> = (0..len).map(|i| ((i * 17 + i / 5 + seed) % 4) as u8).collect();
                let b: Vec<u8> = (0..len).map(|i| ((i * 13 + i / 7 + seed * 3) % 4) as u8).collect();
                let x = PackedQ2::from_levels(&a).unwrap();
                let y = PackedQ2::from_levels(&b).unwrap();
                assert_eq!(x.is_empty(), len == 0);
                assert_eq!(x.len(), len);
                let expected_unsigned: u128 = a.iter().zip(&b)
                    .map(|(&p, &q)| u128::from(p) * u128::from(q)).sum();
                let expected_signed: i128 = a.iter().zip(&b)
                    .map(|(&p, &q)| signed(p) * signed(q)).sum();
                assert_eq!(x.dot_unsigned(&y), Ok(expected_unsigned));
                assert_eq!(x.dot_signed(&y), Ok(expected_signed));
                let expected_gf4_dot = a
                    .iter()
                    .zip(&b)
                    .fold(0_u8, |acc, (&p, &q)| acc ^ gf4_ref(p, q));
                assert_eq!(x.gf4_dot(&y), Ok(expected_gf4_dot));
                let add = x.gf4_add(&y).unwrap();
                let mul = x.gf4_mul(&y).unwrap();
                for i in 0..len {
                    assert_eq!(x.get(i), Ok(a[i]));
                    assert_eq!(add.get(i), Ok(a[i] ^ b[i]));
                    assert_eq!(mul.get(i), Ok(gf4_ref(a[i], b[i])));
                }
            }
        }
        for a in 0..4 {
            for b in 0..4 {
                let x = PackedQ2::from_levels(&[a]).unwrap();
                let y = PackedQ2::from_levels(&[b]).unwrap();
                assert_eq!(x.gf4_mul(&y).unwrap().get(0), Ok(gf4_ref(a, b)));
                assert_eq!(x.gf4_dot(&y), Ok(gf4_ref(a, b)));
            }
        }
        assert_eq!(gf4_ref(2, 2), 3);
        assert_eq!(gf4_ref(2, 3), 1);
        assert_eq!(gf4_ref(3, 3), 2);
    }
}

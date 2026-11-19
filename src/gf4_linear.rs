//! Classical GF(4) linear maps built from two-bit packed coefficient rows.
//!
//! Matrix entries and activations are field elements, not Boolean contact
//! counts or biological synaptic weights. The event runtime is unchanged.

use crate::packed_q2::{PackedQ2, Q2Error};

/// A reusable, row-packed GF(4) matrix of dimensions output_len × input_len.
///
/// Row ownership avoids repacking weights on each application. Every row has
/// the same exact logical width; the final partial u64 is zero-padded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedGf4Linear {
    input_len: usize,
    rows: Vec<PackedQ2>,
}

impl PackedGf4Linear {
    /// Validate all row widths before accepting a packed coefficient matrix.
    pub fn from_rows(input_len: usize, rows: Vec<PackedQ2>) -> Result<Self, Q2Error> {
        if rows.iter().any(|row| row.len() != input_len) {
            return Err(Q2Error::LengthMismatch);
        }
        Ok(Self { input_len, rows })
    }

    #[must_use]
    pub fn input_len(&self) -> usize {
        self.input_len
    }

    #[must_use]
    pub fn output_len(&self) -> usize {
        self.rows.len()
    }

    /// Packed coefficient payload only, excluding Vec metadata/capacity.
    #[must_use]
    pub fn coefficient_payload_bytes(&self) -> usize {
        self.rows
            .iter()
            .map(PackedQ2::allocated_payload_bytes)
            .sum()
    }

    /// Compose this map with an earlier map: self(previous(input)).
    ///
    /// If self is M×K and previous is K×N, the result is an M×N GF(4)
    /// matrix. A single packed K-lane column is reused across N columns,
    /// avoiding a full transposed coefficient matrix.
    pub fn compose(&self, previous: &Self) -> Result<Self, Q2Error> {
        if self.input_len != previous.output_len() {
            return Err(Q2Error::LengthMismatch);
        }

        let mut rows = vec![PackedQ2::new(previous.input_len); self.output_len()];
        let mut column = PackedQ2::new(self.input_len);
        for col in 0..previous.input_len {
            for (row, coefficients) in previous.rows.iter().enumerate() {
                column.set(row, coefficients.get(col)?)?;
            }
            for (index, weights) in self.rows.iter().enumerate() {
                rows[index].set(col, weights.gf4_dot(&column)?)?;
            }
        }
        Ok(Self {
            input_len: previous.input_len,
            rows,
        })
    }

    /// y = W*x over GF(4). Each output lane is the exact field inner product
    /// of one packed coefficient row and the packed input vector.
    pub fn apply(&self, input: &PackedQ2) -> Result<PackedQ2, Q2Error> {
        let mut result = PackedQ2::new(self.rows.len());
        self.apply_xor(input, &mut result)?;
        Ok(result)
    }

    /// output += W*x over GF(4), with field addition implemented by XOR.
    ///
    /// Both shapes are rejected before writing the first destination lane.
    /// After construction every row has an invariant width, so all field
    /// products and writes are valid throughout this loop.
    pub fn apply_xor(
        &self,
        input: &PackedQ2,
        output: &mut PackedQ2,
    ) -> Result<(), Q2Error> {
        if input.len() != self.input_len || output.len() != self.rows.len() {
            return Err(Q2Error::LengthMismatch);
        }
        for (index, row) in self.rows.iter().enumerate() {
            let updated = output.get(index)? ^ row.gf4_dot(input)?;
            output.set(index, updated)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{PackedGf4Linear, PackedQ2, Q2Error};

    fn field_product(a: u8, b: u8) -> u8 {
        let a0 = a & 1;
        let a1 = a >> 1;
        let b0 = b & 1;
        let b1 = b >> 1;
        ((a0 & b0) ^ (a1 & b1))
            | (((a0 & b1) ^ (a1 & b0) ^ (a1 & b1)) << 1)
    }

    fn scalar_dot(row: &[u8], input: &[u8]) -> u8 {
        row.iter()
            .zip(input)
            .fold(0_u8, |sum, (&a, &b)| sum ^ field_product(a, b))
    }

    #[test]
    fn multiple_outputs_match_independent_scalar_oracle() {
        for width in [0, 1, 63, 64, 65, 129, 188_508] {
            let input_values: Vec<u8> = (0..width)
                .map(|i| ((i * 17 + i / 5 + 2) % 4) as u8)
                .collect();
            let input = PackedQ2::from_levels(&input_values).unwrap();
            let outputs = if width == 188_508 { 2 } else { 65 };
            let scalar_rows: Vec<Vec<u8>> = (0..outputs)
                .map(|r| {
                    (0..width)
                        .map(|i| ((i * 13 + i / 7 + r * 3) % 4) as u8)
                        .collect()
                })
                .collect();
            let packed_rows: Vec<PackedQ2> = scalar_rows
                .iter()
                .map(|row| PackedQ2::from_levels(row).unwrap())
                .collect();
            let matrix = PackedGf4Linear::from_rows(width, packed_rows).unwrap();
            assert_eq!(matrix.input_len(), width);
            assert_eq!(matrix.output_len(), outputs);
            let result = matrix.apply(&input).unwrap();
            for (index, row) in scalar_rows.iter().enumerate() {
                assert_eq!(result.get(index), Ok(scalar_dot(row, &input_values)));
            }
        }
    }

    #[test]
    fn identity_matrix_spans_multiple_output_words() {
        let width = 129;
        let input_values: Vec<u8> = (0..width).map(|i| (i % 4) as u8).collect();
        let rows: Vec<PackedQ2> = (0..width)
            .map(|i| {
                let mut row = PackedQ2::new(width);
                row.set(i, 1).unwrap();
                row
            })
            .collect();
        let matrix = PackedGf4Linear::from_rows(width, rows).unwrap();
        let input = PackedQ2::from_levels(&input_values).unwrap();
        assert_eq!(matrix.apply(&input), Ok(input));
    }

    #[test]
    fn field_linearity_and_repeated_accumulation_cancel() {
        let a = PackedQ2::from_levels(&[0, 1, 2, 3, 2, 0, 3, 1, 2]).unwrap();
        let b = PackedQ2::from_levels(&[3, 2, 1, 0, 3, 1, 2, 0, 3]).unwrap();
        let rows = vec![
            PackedQ2::from_levels(&[1, 2, 3, 0, 1, 2, 3, 0, 1]).unwrap(),
            PackedQ2::from_levels(&[3, 0, 1, 2, 3, 0, 1, 2, 3]).unwrap(),
            PackedQ2::from_levels(&[2, 2, 2, 2, 2, 2, 2, 2, 2]).unwrap(),
        ];
        let matrix = PackedGf4Linear::from_rows(9, rows).unwrap();
        let lhs = matrix.apply(&a.gf4_add(&b).unwrap()).unwrap();
        let rhs = matrix
            .apply(&a)
            .unwrap()
            .gf4_add(&matrix.apply(&b).unwrap())
            .unwrap();
        assert_eq!(lhs, rhs);

        let mut out = PackedQ2::from_levels(&[3, 1, 2]).unwrap();
        let original = out.clone();
        matrix.apply_xor(&a, &mut out).unwrap();
        matrix.apply_xor(&a, &mut out).unwrap();
        assert_eq!(out, original);
    }

    #[test]
    fn empty_dimensions_are_well_defined() {
        let empty_rows = PackedGf4Linear::from_rows(65, Vec::new()).unwrap();
        assert!(empty_rows.apply(&PackedQ2::new(65)).unwrap().is_empty());

        let zero_input_rows = PackedGf4Linear::from_rows(
            0,
            vec![PackedQ2::new(0), PackedQ2::new(0)],
        )
        .unwrap();
        assert_eq!(zero_input_rows.apply(&PackedQ2::new(0)),
            Ok(PackedQ2::from_levels(&[0, 0]).unwrap()));
    }

    #[test]
    fn composition_matches_two_passes_and_scalar_oracle() {
        for input_width in [0, 1, 63, 64, 65, 129] {
            for hidden_width in [0, 1, 2, 65] {
                let output_width = 3;
                let input_values: Vec<u8> = (0..input_width)
                    .map(|i| ((i * 17 + i / 5 + 1) % 4) as u8)
                    .collect();
                let prev_rows: Vec<Vec<u8>> = (0..hidden_width)
                    .map(|r| {
                        (0..input_width)
                            .map(|i| ((i * 13 + r * 7 + i / 3) % 4) as u8)
                            .collect()
                    })
                    .collect();
                let last_rows: Vec<Vec<u8>> = (0..output_width)
                    .map(|r| {
                        (0..hidden_width)
                            .map(|i| ((i * 19 + r * 5 + i / 7) % 4) as u8)
                            .collect()
                    })
                    .collect();
                let previous = PackedGf4Linear::from_rows(
                    input_width,
                    prev_rows
                        .iter()
                        .map(|row| PackedQ2::from_levels(row).unwrap())
                        .collect(),
                )
                .unwrap();
                let last = PackedGf4Linear::from_rows(
                    hidden_width,
                    last_rows
                        .iter()
                        .map(|row| PackedQ2::from_levels(row).unwrap())
                        .collect(),
                )
                .unwrap();
                let composed = last.compose(&previous).unwrap();
                let input = PackedQ2::from_levels(&input_values).unwrap();
                assert_eq!(composed.input_len(), input_width);
                assert_eq!(composed.output_len(), output_width);
                assert_eq!(
                    composed.apply(&input),
                    last.apply(&previous.apply(&input).unwrap())
                );
                for (row_index, row) in last_rows.iter().enumerate() {
                    let scalar_hidden: Vec<u8> = prev_rows
                        .iter()
                        .map(|prev| scalar_dot(prev, &input_values))
                        .collect();
                    assert_eq!(
                        composed.apply(&input).unwrap().get(row_index),
                        Ok(scalar_dot(row, &scalar_hidden))
                    );
                }
            }
        }
    }

    #[test]
    fn composition_checks_dimensions_and_does_not_mutate_operands() {
        let earlier = PackedGf4Linear::from_rows(
            65,
            vec![PackedQ2::new(65); 3],
        )
        .unwrap();
        let later = PackedGf4Linear::from_rows(
            2,
            vec![PackedQ2::new(2); 4],
        )
        .unwrap();
        let before_earlier = earlier.clone();
        let before_later = later.clone();
        assert_eq!(later.compose(&earlier), Err(Q2Error::LengthMismatch));
        assert_eq!(earlier, before_earlier);
        assert_eq!(later, before_later);

        let identity = PackedGf4Linear::from_rows(
            3,
            (0..3)
                .map(|i| {
                    let mut row = PackedQ2::new(3);
                    row.set(i, 1).unwrap();
                    row
                })
                .collect(),
        )
        .unwrap();
        let composed = identity.compose(&earlier).unwrap();
        let input = PackedQ2::from_levels(&[3; 65]).unwrap();
        assert_eq!(composed.apply(&input), earlier.apply(&input));
    }

    #[test]
    fn invalid_dimensions_are_rejected_without_mutation() {
        let rows = vec![PackedQ2::new(64), PackedQ2::new(65)];
        assert!(matches!(
            PackedGf4Linear::from_rows(64, rows),
            Err(Q2Error::LengthMismatch)
        ));
        let matrix = PackedGf4Linear::from_rows(
            65,
            vec![PackedQ2::new(65), PackedQ2::new(65)],
        )
        .unwrap();
        let mut output = PackedQ2::from_levels(&[1, 3]).unwrap();
        let before = output.clone();
        assert_eq!(
            matrix.apply_xor(&PackedQ2::new(64), &mut output),
            Err(Q2Error::LengthMismatch)
        );
        assert_eq!(output, before);
        assert_eq!(
            matrix.apply_xor(&PackedQ2::new(65), &mut PackedQ2::new(1)),
            Err(Q2Error::LengthMismatch)
        );
        assert_eq!(matrix.apply(&PackedQ2::new(64)), Err(Q2Error::LengthMismatch));
    }
}

//! Boolean-only comparison of IEEE-754 probability *bit patterns*.
//!
//! No floating point operations appear in this module. Original bit payloads
//! and NULLs must be preserved upstream; this provides a research gate,
//! not a physiological neurotransmitter model.
use crate::gates;
use std::cmp::Ordering;

const NEGATIVE_SIGN: u64 = 1_u64 << 63;
const PLUS_ONE_BITS: u64 = 0x3ff0_0000_0000_0000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbabilityError {
    OutsideUnitInterval,
}

/// Check that an IEEE-754 bit pattern represents a probability in [0, 1].
/// Positive finite values and signed zeros are accepted. NaN, infinities,
/// negatives other than -0, and values greater than one are rejected.
pub fn validate_probability_bits(bits: u64) -> Result<(), ProbabilityError> {
    let magnitude = bits & !NEGATIVE_SIGN;
    if bits & NEGATIVE_SIGN != 0 && magnitude != 0 {
        return Err(ProbabilityError::OutsideUnitInterval);
    }
    if magnitude > PLUS_ONE_BITS {
        return Err(ProbabilityError::OutsideUnitInterval);
    }
    Ok(())
}

fn sortable_bits(bits: u64) -> u64 {
    if bits & !NEGATIVE_SIGN == 0 {
        0
    } else {
        bits
    }
}

/// On finite [0,1], the positive IEEE-754 encoding is ordinal and can be
/// compared solely with the Boolean AND/XOR/shift comparator.
pub fn compare_bits(left: u64, right: u64) -> Result<Ordering, ProbabilityError> {
    validate_probability_bits(left)?;
    validate_probability_bits(right)?;
    let left = sortable_bits(left);
    let right = sortable_bits(right);
    if left == right {
        Ok(Ordering::Equal)
    } else if gates::ge(left as u128, right as u128) {
        Ok(Ordering::Greater)
    } else {
        Ok(Ordering::Less)
    }
}

/// None represents an unknown measurement, not a zero probability.
pub fn at_least(
    measurement: Option<u64>,
    threshold_bits: u64,
) -> Result<Option<bool>, ProbabilityError> {
    validate_probability_bits(threshold_bits)?;
    match measurement {
        None => Ok(None),
        Some(bits) => Ok(Some(compare_bits(bits, threshold_bits)? != Ordering::Less)),
    }
}

/// Largest of the eight classifier probabilities; ties go to the
/// lowest index, NULLs are not invented. The result is an index, never a
/// physiological excitatory/inhibitory effect.
pub fn argmax_eight(values: [Option<u64>; 8]) -> Result<Option<usize>, ProbabilityError> {
    let mut choice: Option<(usize, u64)> = None;
    for (index, raw) in values.into_iter().enumerate() {
        if let Some(bits) = raw {
            validate_probability_bits(bits)?;
            match choice {
                None => choice = Some((index, bits)),
                Some((_, best)) => {
                    if compare_bits(bits, best)? == Ordering::Greater {
                        choice = Some((index, bits));
                    }
                }
            }
        }
    }
    Ok(choice.map(|(index, _)| index))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_bit_gate_agrees_with_float_oracle_on_valid_values() {
        for i in 0..=100_u64 {
            for j in 0..=100_u64 {
                let a = (i as f64 / 100.0).to_bits();
                let b = (j as f64 / 100.0).to_bits();
                assert_eq!(compare_bits(a, b).unwrap(), i.cmp(&j));
            }
        }
    }

    #[test]
    fn signed_zero_and_null_distinct() {
        let plus_zero = 0_u64;
        let minus_zero = NEGATIVE_SIGN;
        assert_eq!(compare_bits(plus_zero, minus_zero), Ok(Ordering::Equal));
        assert_eq!(at_least(None, plus_zero), Ok(None));
        assert_eq!(at_least(Some(minus_zero), plus_zero), Ok(Some(true)));
        assert_eq!(at_least(Some(PLUS_ONE_BITS), PLUS_ONE_BITS), Ok(Some(true)));
    }

    #[test]
    fn reject_invalid_probability_bit_patterns() {
        for bits in [
            0x7ff0_0000_0000_0000_u64, // +infinity
            0xfff0_0000_0000_0000_u64, // -infinity
            0x7ff8_0000_0000_0001_u64, // NaN with nonzero payload
            0xbff0_0000_0000_0000_u64, // -1
            0x4000_0000_0000_0000_u64, // 2
        ] {
            assert_eq!(
                validate_probability_bits(bits),
                Err(ProbabilityError::OutsideUnitInterval)
            );
        }
    }

    #[test]
    fn argmax_is_stable_on_ties_and_nulls() {
        let half = 0.5_f64.to_bits();
        let one = PLUS_ONE_BITS;
        assert_eq!(argmax_eight([None; 8]), Ok(None));
        assert_eq!(
            argmax_eight([None, Some(half), None, Some(one), Some(one), None, None, None]),
            Ok(Some(3))
        );
        assert_eq!(
            argmax_eight([None, Some(half), None, Some(half), None, None, None, None]),
            Ok(Some(1))
        );
    }
}

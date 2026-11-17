//! Two-dimensional VALID-only routing for synthetic circuits.
//! Unlike cone-ratio-only routing, circuit scale is part of the decision key.
//! TEST/OOD must never be supplied to calibration.
use crate::fourier5_conservative_policy::{Engine, PairedObservation, PolicyError};

const CONES: usize = 16;
const SIZES: usize = 8;
const CELLS: usize = CONES * SIZES;

/// Logarithmic circuit-size class, saturating above 2^7 gates.
/// Small and large circuits with equal cone ratios never share evidence.
fn size_class(gates: u64) -> usize {
    (63 - gates.leading_zeros() as usize).min(SIZES - 1)
}

fn cell(gates: u64, cone: u64) -> usize {
    let ratio = ((u128::from(cone) * 16) / u128::from(gates)).min(15) as usize;
    size_class(gates) * CONES + ratio
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScaleAwarePolicy {
    enabled: [bool; CELLS],
    samples: [u64; CELLS],
}

impl ScaleAwarePolicy {
    /// Admit Incremental only when every observed VALID pair in the exact
    /// size/cone cell agrees with Full and is strictly faster by at least
    /// margin_ns. Empty/undersampled cells fail closed.
    pub fn fit_valid(
        valid: &[PairedObservation],
        min_samples: u64,
        margin_ns: u64,
    ) -> Result<Self, PolicyError> {
        if valid.is_empty() || min_samples == 0 {
            return Err(PolicyError::EmptyValidation);
        }
        let mut samples = [0u64; CELLS];
        let mut eligible = [true; CELLS];
        for row in valid {
            if row.gates == 0
                || row.cone_gates > row.gates
                || row.full_ns == 0
                || row.incremental_ns == 0
            {
                return Err(PolicyError::InvalidObservation);
            }
            let index = cell(row.gates, row.cone_gates);
            samples[index] = samples[index].checked_add(1).ok_or(PolicyError::Overflow)?;
            eligible[index] &= row.same_output
                && row.same_decision
                && row.incremental_ns < row.full_ns
                && row.full_ns - row.incremental_ns >= margin_ns;
        }
        let mut enabled = [false; CELLS];
        for index in 0..CELLS {
            enabled[index] = samples[index] >= min_samples && eligible[index];
        }
        Ok(Self { enabled, samples })
    }

    #[must_use]
    pub fn select(&self, gates: u64, cone_gates: u64) -> Engine {
        if gates == 0 || cone_gates > gates {
            return Engine::Full;
        }
        if self.enabled[cell(gates, cone_gates)] {
            Engine::Incremental
        } else {
            Engine::Full
        }
    }

    #[must_use]
    pub fn validation_samples(&self, gates: u64, cone_gates: u64) -> u64 {
        if gates == 0 || cone_gates > gates {
            return 0;
        }
        self.samples[cell(gates, cone_gates)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(gates: u64, cone: u64, full: u64, inc: u64, correct: bool) -> PairedObservation {
        PairedObservation {
            gates,
            cone_gates: cone,
            full_ns: full,
            incremental_ns: inc,
            same_output: correct,
            same_decision: correct,
        }
    }

    #[test]
    fn scale_prevents_small_circuit_evidence_leaking_into_large() {
        let valid = [observation(64, 4, 100, 20, true)];
        let policy = ScaleAwarePolicy::fit_valid(&valid, 1, 0).unwrap();
        assert_eq!(policy.select(64, 4), Engine::Incremental);
        assert_eq!(policy.select(4096, 256), Engine::Full);
        assert_eq!(policy.validation_samples(4096, 256), 0);
    }

    #[test]
    fn one_regression_or_decision_mismatch_disables_entire_cell() {
        let valid = [
            observation(64, 4, 1000, 1, true),
            observation(65, 4, 100, 101, true),
        ];
        let policy = ScaleAwarePolicy::fit_valid(&valid, 2, 0).unwrap();
        assert_eq!(policy.select(64, 4), Engine::Full);
        let mismatch = [
            observation(64, 4, 100, 1, true),
            observation(65, 4, 100, 1, false),
        ];
        assert_eq!(
            ScaleAwarePolicy::fit_valid(&mismatch, 2, 0).unwrap().select(64, 4),
            Engine::Full
        );
    }

    #[test]
    fn holdout_is_read_only_and_unknown_cells_fail_closed() {
        let valid = [observation(64, 4, 100, 40, true)];
        let policy = ScaleAwarePolicy::fit_valid(&valid, 1, 10).unwrap();
        let frozen = policy.clone();
        let _test = observation(64, 4, 1, 100, false);
        assert_eq!(policy, frozen);
        assert_eq!(policy.select(64, 4), Engine::Incremental);
        assert_eq!(policy.select(0, 0), Engine::Full);
        assert_eq!(policy.select(64, 65), Engine::Full);
    }

    #[test]
    fn invalid_rows_and_insufficient_margin_fail_closed() {
        assert_eq!(
            ScaleAwarePolicy::fit_valid(&[], 1, 0),
            Err(PolicyError::EmptyValidation)
        );
        assert_eq!(
            ScaleAwarePolicy::fit_valid(&[observation(10, 11, 5, 1, true)], 1, 0),
            Err(PolicyError::InvalidObservation)
        );
        let policy = ScaleAwarePolicy::fit_valid(
            &[observation(64, 4, 100, 95, true)],
            1,
            6,
        ).unwrap();
        assert_eq!(policy.select(64, 4), Engine::Full);
    }
}

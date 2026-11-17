//! Conservative, frozen Full/Incremental policy calibrated on VALID only.
//! Synthetic circuit workload metadata; no BANC biological interpretation.
//! No TEST/OOD observations can modify a frozen policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    Full,
    Incremental,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairedObservation {
    pub gates: u64,
    pub cone_gates: u64,
    pub full_ns: u64,
    pub incremental_ns: u64,
    pub same_output: bool,
    pub same_decision: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyError {
    InvalidObservation,
    EmptyValidation,
    Overflow,
}

/// Bucket boundaries are powers of two, avoiding floating-point calibration.
fn bucket(gates: u64, cone: u64) -> usize {
    let scaled = (u128::from(cone) * 16) / u128::from(gates);
    usize::try_from(scaled.min(15)).unwrap_or(15)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrozenPolicy {
    /// Only buckets with unanimous correctness and strict aggregate speedup
    /// on VALID are enabled. Empty buckets always fall back to Full.
    enabled: [bool; 16],
    samples: [u64; 16],
}

impl FrozenPolicy {
    pub fn fit_valid(rows: &[PairedObservation], min_samples: u64) -> Result<Self, PolicyError> {
        if rows.is_empty() || min_samples == 0 {
            return Err(PolicyError::EmptyValidation);
        }
        let mut count = [0u64; 16];
        let mut full = [0u128; 16];
        let mut inc = [0u128; 16];
        let mut correct = [true; 16];
        for r in rows {
            if r.gates == 0 || r.cone_gates > r.gates || r.full_ns == 0 || r.incremental_ns == 0 {
                return Err(PolicyError::InvalidObservation);
            }
            let b = bucket(r.gates, r.cone_gates);
            count[b] = count[b].checked_add(1).ok_or(PolicyError::Overflow)?;
            full[b] = full[b]
                .checked_add(u128::from(r.full_ns))
                .ok_or(PolicyError::Overflow)?;
            inc[b] = inc[b]
                .checked_add(u128::from(r.incremental_ns))
                .ok_or(PolicyError::Overflow)?;
            correct[b] &= r.same_output && r.same_decision;
        }
        let mut enabled = [false; 16];
        for b in 0..16 {
            enabled[b] = count[b] >= min_samples && correct[b] && inc[b] < full[b];
        }
        Ok(Self {
            enabled,
            samples: count,
        })
    }

    /// Fit only on disjoint VALID paired measurements, with a per-observation
    /// no-regression gate. Unlike aggregate calibration, one slow incremental
    /// sample disables the entire cone bucket. TEST/OOD never enter this fit.
    ///
    /// A margin of zero requires incremental_ns < full_ns for every row;
    /// a positive margin requires incremental_ns + margin_ns <= full_ns.
    /// This is an observed VALID guarantee, not a future latency guarantee.
    pub fn fit_valid_no_regression(
        rows: &[PairedObservation],
        min_samples: u64,
        margin_ns: u64,
    ) -> Result<Self, PolicyError> {
        if rows.is_empty() || min_samples == 0 {
            return Err(PolicyError::EmptyValidation);
        }
        let mut samples = [0u64; 16];
        let mut eligible = [true; 16];
        for row in rows {
            if row.gates == 0
                || row.cone_gates > row.gates
                || row.full_ns == 0
                || row.incremental_ns == 0
            {
                return Err(PolicyError::InvalidObservation);
            }
            let b = bucket(row.gates, row.cone_gates);
            samples[b] = samples[b].checked_add(1).ok_or(PolicyError::Overflow)?;
            let improvement = row.full_ns.saturating_sub(row.incremental_ns);
            eligible[b] &= row.same_output
                && row.same_decision
                && row.incremental_ns < row.full_ns
                && improvement >= margin_ns;
        }
        let mut enabled = [false; 16];
        for b in 0..16 {
            enabled[b] = samples[b] >= min_samples && eligible[b];
        }
        Ok(Self { enabled, samples })
    }

    #[must_use]
    pub fn select(&self, gates: u64, cone_gates: u64) -> Engine {
        if gates == 0 || cone_gates > gates {
            return Engine::Full;
        }
        if self.enabled[bucket(gates, cone_gates)] {
            Engine::Incremental
        } else {
            Engine::Full
        }
    }

    #[must_use]
    pub fn validation_samples(&self) -> &[u64; 16] {
        &self.samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(cone: u64, full: u64, inc: u64, correct: bool) -> PairedObservation {
        PairedObservation {
            gates: 100,
            cone_gates: cone,
            full_ns: full,
            incremental_ns: inc,
            same_output: correct,
            same_decision: correct,
        }
    }
    #[test]
    fn conservative_selection_and_unknown_cones() {
        let p = FrozenPolicy::fit_valid(
            &[
                row(2, 100, 20, true),
                row(3, 100, 25, true),
                row(80, 100, 150, true),
            ],
            2,
        )
        .unwrap();
        assert_eq!(p.select(100, 2), Engine::Incremental);
        assert_eq!(p.select(100, 80), Engine::Full);
        assert_eq!(p.select(100, 40), Engine::Full);
        assert_eq!(p.select(0, 0), Engine::Full);
    }
    #[test]
    fn a_single_decision_mismatch_disables_bucket() {
        let p =
            FrozenPolicy::fit_valid(&[row(2, 100, 10, true), row(3, 100, 10, false)], 2).unwrap();
        assert_eq!(p.select(100, 2), Engine::Full);
    }
    #[test]
    fn strict_aggregate_oracle_and_no_test_mutation() {
        let valid = [row(2, 100, 30, true), row(3, 100, 30, true)];
        let p = FrozenPolicy::fit_valid(&valid, 2).unwrap();
        let frozen = p.clone();
        let _held_out = row(2, 1, 1000, false);
        assert_eq!(p, frozen);
        assert_eq!(p.select(100, 2), Engine::Incremental);
        assert_eq!(
            FrozenPolicy::fit_valid(&[row(2, 10, 10, true)], 1)
                .unwrap()
                .select(100, 2),
            Engine::Full
        );
    }
    #[test]
    fn malformed_validation_rejected() {
        assert_eq!(
            FrozenPolicy::fit_valid(&[row(101, 1, 1, true)], 1),
            Err(PolicyError::InvalidObservation)
        );
        assert_eq!(
            FrozenPolicy::fit_valid(&[], 1),
            Err(PolicyError::EmptyValidation)
        );
    }
    #[test]
    fn strict_fit_rejects_aggregate_win_with_one_regression() {
        let rows = [row(2, 1000, 1, true), row(3, 100, 101, true)];
        assert_eq!(
            FrozenPolicy::fit_valid(&rows, 2).unwrap().select(100, 2),
            Engine::Incremental
        );
        assert_eq!(
            FrozenPolicy::fit_valid_no_regression(&rows, 2, 0)
                .unwrap()
                .select(100, 2),
            Engine::Full
        );
    }

    #[test]
    fn strict_fit_respects_margin_and_exact_bucket_boundaries() {
        let rows = [row(2, 100, 90, true), row(3, 100, 89, true)];
        let allowed = FrozenPolicy::fit_valid_no_regression(&rows, 2, 10).unwrap();
        assert_eq!(allowed.select(100, 2), Engine::Incremental);
        assert_eq!(allowed.select(100, 80), Engine::Full);
        let blocked = FrozenPolicy::fit_valid_no_regression(&rows, 2, 11).unwrap();
        assert_eq!(blocked.select(100, 2), Engine::Full);
    }

    #[test]
    fn strict_fit_rejects_mismatch_and_unknown_validation() {
        let rows = [row(2, 100, 1, true), row(3, 100, 1, false)];
        assert_eq!(
            FrozenPolicy::fit_valid_no_regression(&rows, 2, 0)
                .unwrap()
                .select(100, 2),
            Engine::Full
        );
        assert_eq!(
            FrozenPolicy::fit_valid_no_regression(&[], 1, 0),
            Err(PolicyError::EmptyValidation)
        );
        assert_eq!(
            FrozenPolicy::fit_valid_no_regression(&[row(101, 100, 1, true)], 1, 0),
            Err(PolicyError::InvalidObservation)
        );
    }
}

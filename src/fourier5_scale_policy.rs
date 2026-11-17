//! Two-dimensional VALID-only routing for synthetic circuits.
//! Unlike cone-ratio-only routing, circuit scale is part of the decision key.
//! TEST/OOD must never be supplied to calibration.
use crate::fourier5_conservative_policy::{Engine, PairedObservation, PolicyError};

const CONES: usize = 16;
const SIZES: usize = u64::BITS as usize;
const CELLS: usize = CONES * SIZES;

/// Exact base-2 size bucket over the full nonzero u64 range.
/// A large circuit must not inherit calibration from a much smaller circuit.
fn size_class(gates: u64) -> usize {
    (u64::BITS - 1 - gates.leading_zeros()) as usize
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

    /// Fail closed when an incremental cache does not describe the exact
    /// current circuit generation. A stale cache must never be dispatched.
    /// Generations are caller-maintained monotonic revision identifiers;
    /// callers must increment the revision after every accepted mutation.
    #[must_use]
    pub fn select_for_generation(
        &self,
        gates: u64,
        cone_gates: u64,
        circuit_generation: u64,
        cache_generation: Option<u64>,
    ) -> Engine {
        if cache_generation != Some(circuit_generation) {
            return Engine::Full;
        }
        self.select(gates, cone_gates)
    }

    /// Route a sequential intervention only if its cache was rebuilt for the
    /// exact post-mutation revision. Overflow is a hard error: a wrapped
    /// revision could otherwise make an obsolete cache appear current.
    /// The caller must update its circuit revision only after an accepted
    /// mutation, and assign the returned revision to a rebuilt cache.
    pub fn next_generation(current: u64) -> Result<u64, PolicyError> {
        current.checked_add(1).ok_or(PolicyError::Overflow)
    }

    /// Explicitly reject a cache revision from the future as well as stale
    /// revisions; callers may use this after a successful full-oracle check.
    #[must_use]
    pub fn cache_is_current(circuit_generation: u64, cache_generation: Option<u64>) -> bool {
        cache_generation == Some(circuit_generation)
    }

    #[must_use]
    pub fn validation_samples(&self, gates: u64, cone_gates: u64) -> u64 {
        if gates == 0 || cone_gates > gates {
            return 0;
        }
        self.samples[cell(gates, cone_gates)]
    }
}


/// Immutable evaluation of a frozen routing policy on disjoint TEST/OOD pairs.
/// This is an audit, never an input to calibration or automatic policy changes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HoldoutAudit {
    pub total: u64,
    pub routed_incremental: u64,
    pub output_mismatches: u64,
    pub decision_mismatches: u64,
    pub latency_regressions: u64,
    pub full_ns: u128,
    pub selected_ns: u128,
}

impl ScaleAwarePolicy {
    /// Preserve held-out failures per scale/cone cell instead of allowing
    /// a fast category to conceal a slow or incorrect category in totals.
    /// Cell indices use the same frozen VALID routing keys; this never fits
    /// or modifies the policy. Empty cells remain zeroed.
    pub fn audit_holdout_cells(
        &self,
        rows: &[PairedObservation],
    ) -> Result<Vec<HoldoutAudit>, PolicyError> {
        let mut cells = vec![HoldoutAudit::default(); CELLS];
        for row in rows {
            // Reuse the exact single-observation oracle/validation accounting.
            // Do not silently discard invalid rows or counter overflow.
            let single = self.audit_holdout(std::slice::from_ref(row))?;
            let bucket = &mut cells[cell(row.gates, row.cone_gates)];
            bucket.total = bucket.total.checked_add(single.total).ok_or(PolicyError::Overflow)?;
            bucket.routed_incremental = bucket.routed_incremental.checked_add(single.routed_incremental).ok_or(PolicyError::Overflow)?;
            bucket.output_mismatches = bucket.output_mismatches.checked_add(single.output_mismatches).ok_or(PolicyError::Overflow)?;
            bucket.decision_mismatches = bucket.decision_mismatches.checked_add(single.decision_mismatches).ok_or(PolicyError::Overflow)?;
            bucket.latency_regressions = bucket.latency_regressions.checked_add(single.latency_regressions).ok_or(PolicyError::Overflow)?;
            bucket.full_ns = bucket.full_ns.checked_add(single.full_ns).ok_or(PolicyError::Overflow)?;
            bucket.selected_ns = bucket.selected_ns.checked_add(single.selected_ns).ok_or(PolicyError::Overflow)?;
        }
        Ok(cells)
    }

    /// Evaluate selected-engine outcomes against paired Full observations.
    /// Fail on malformed rows or arithmetic overflow rather than dropping data.
    pub fn audit_holdout(&self, rows: &[PairedObservation]) -> Result<HoldoutAudit, PolicyError> {
        let mut audit = HoldoutAudit::default();
        for row in rows {
            if row.gates == 0
                || row.cone_gates > row.gates
                || row.full_ns == 0
                || row.incremental_ns == 0
            {
                return Err(PolicyError::InvalidObservation);
            }
            audit.total = audit.total.checked_add(1).ok_or(PolicyError::Overflow)?;
            audit.full_ns = audit.full_ns.checked_add(u128::from(row.full_ns)).ok_or(PolicyError::Overflow)?;
            let selected = if self.select(row.gates, row.cone_gates) == Engine::Incremental {
                audit.routed_incremental = audit.routed_incremental.checked_add(1).ok_or(PolicyError::Overflow)?;
                if !row.same_output {
                    audit.output_mismatches = audit.output_mismatches.checked_add(1).ok_or(PolicyError::Overflow)?;
                }
                if !row.same_decision {
                    audit.decision_mismatches = audit.decision_mismatches.checked_add(1).ok_or(PolicyError::Overflow)?;
                }
                if row.incremental_ns >= row.full_ns {
                    audit.latency_regressions = audit.latency_regressions.checked_add(1).ok_or(PolicyError::Overflow)?;
                }
                row.incremental_ns
            } else {
                row.full_ns
            };
            audit.selected_ns = audit.selected_ns.checked_add(u128::from(selected)).ok_or(PolicyError::Overflow)?;
        }
        Ok(audit)
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
    fn large_scale_buckets_do_not_inherit_tiny_valid_evidence() {
        // Previously every size >=128 shared the saturated final bucket.
        let valid = [observation(128, 0, 100, 10, true)];
        let policy = ScaleAwarePolicy::fit_valid(&valid, 1, 0).unwrap();
        assert_eq!(policy.select(128, 0), Engine::Incremental);
        assert_eq!(policy.select(1_562_501, 0), Engine::Full);
        assert_eq!(policy.validation_samples(1_562_501, 0), 0);
        assert_eq!(policy.select(u64::MAX, 0), Engine::Full);
    }

    #[test]
    fn large_scale_own_valid_evidence_can_enable_incremental() {
        let valid = [observation(1_562_501, 10, 100, 50, true)];
        let policy = ScaleAwarePolicy::fit_valid(&valid, 1, 1).unwrap();
        assert_eq!(policy.select(1_562_501, 10), Engine::Incremental);
        assert_eq!(policy.select(128, 0), Engine::Full);
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
    fn holdout_audit_exposes_negative_results_without_retraining() {
        let valid = [observation(1_562_501, 10, 100, 40, true)];
        let policy = ScaleAwarePolicy::fit_valid(&valid, 1, 1).unwrap();
        let frozen = policy.clone();
        let mut bad = observation(1_562_501, 10, 100, 130, false);
        bad.same_decision = true;
        let holdout = [bad, observation(128, 0, 50, 1, false)];
        let report = policy.audit_holdout(&holdout).unwrap();
        assert_eq!(report.total, 2);
        assert_eq!(report.routed_incremental, 1);
        assert_eq!(report.output_mismatches, 1);
        assert_eq!(report.decision_mismatches, 0);
        assert_eq!(report.latency_regressions, 1);
        assert_eq!(report.full_ns, 150);
        assert_eq!(report.selected_ns, 180);
        assert_eq!(policy, frozen);
    }

    #[test]
    fn heldout_cell_audit_does_not_mask_negative_large_scale_results() {
        let valid = [
            observation(128, 0, 100, 10, true),
            observation(1_562_501, 10, 100, 10, true),
        ];
        let policy = ScaleAwarePolicy::fit_valid(&valid, 1, 0).unwrap();
        let frozen = policy.clone();
        let rows = [
            observation(128, 0, 1000, 1, true),
            observation(1_562_501, 10, 100, 150, false),
        ];
        let cells = policy.audit_holdout_cells(&rows).unwrap();
        let small = &cells[cell(128, 0)];
        let large = &cells[cell(1_562_501, 10)];
        assert_eq!(small.latency_regressions, 0);
        assert_eq!(large.latency_regressions, 1);
        assert_eq!(large.output_mismatches, 1);
        assert_eq!(large.selected_ns, 150);
        assert_eq!(policy, frozen);
        assert_eq!(cells.iter().map(|a| a.total).sum::<u64>(), 2);
        assert_eq!(
            policy.audit_holdout_cells(&[observation(0, 0, 1, 1, true)]),
            Err(PolicyError::InvalidObservation)
        );
    }

    #[test]
    fn holdout_audit_rejects_invalid_pairs() {
        let policy = ScaleAwarePolicy::fit_valid(&[observation(64, 1, 100, 10, true)], 1, 0).unwrap();
        assert_eq!(policy.audit_holdout(&[]).unwrap(), HoldoutAudit::default());
        assert_eq!(policy.audit_holdout(&[observation(0, 0, 1, 1, true)]), Err(PolicyError::InvalidObservation));
    }

    #[test]
    fn stale_or_missing_cache_generation_forces_full() {
        let valid = [observation(1_562_501, 10, 100, 40, true)];
        let policy = ScaleAwarePolicy::fit_valid(&valid, 1, 1).unwrap();
        assert_eq!(policy.select(1_562_501, 10), Engine::Incremental);
        assert_eq!(policy.select_for_generation(1_562_501, 10, 7, Some(7)), Engine::Incremental);
        assert_eq!(policy.select_for_generation(1_562_501, 10, 8, Some(7)), Engine::Full);
        assert_eq!(policy.select_for_generation(1_562_501, 10, 7, None), Engine::Full);
        assert_eq!(policy.select_for_generation(1_562_501, 10, 7, Some(8)), Engine::Full);
    }

    #[test]
    fn matching_generation_does_not_override_unknown_cells() {
        let valid = [observation(128, 0, 100, 20, true)];
        let policy = ScaleAwarePolicy::fit_valid(&valid, 1, 0).unwrap();
        assert_eq!(policy.select_for_generation(1_562_501, 0, 1, Some(1)), Engine::Full);
        assert_eq!(policy.select_for_generation(0, 0, 1, Some(1)), Engine::Full);
    }

    #[test]
    fn sequential_revisions_advance_without_wraparound() {
        assert_eq!(ScaleAwarePolicy::next_generation(0), Ok(1));
        assert_eq!(ScaleAwarePolicy::next_generation(u64::MAX - 1), Ok(u64::MAX));
        assert_eq!(ScaleAwarePolicy::next_generation(u64::MAX), Err(PolicyError::Overflow));
    }

    #[test]
    fn cache_revision_never_matches_future_or_missing_cache() {
        assert!(ScaleAwarePolicy::cache_is_current(8, Some(8)));
        assert!(!ScaleAwarePolicy::cache_is_current(8, Some(7)));
        assert!(!ScaleAwarePolicy::cache_is_current(8, Some(9)));
        assert!(!ScaleAwarePolicy::cache_is_current(8, None));
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

//! Held-out sequential cost audit for synthetic heterogeneous cone workloads.
//! This is accounting, not a benchmark: callers supply independently measured
//! production-path durations (including cache refresh) and VmHWM observations.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Split {
    Train,
    Valid,
    Test,
    Ood,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub cohort: u64,
    pub split: Split,
    pub cone_bucket: u32,
    pub full_ns: u64,
    pub incremental_refresh_ns: u64,
    pub peak_kib: u64,
    pub state_equal: bool,
    pub cache_equal: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    pub count: u64,
    pub full_ns: u128,
    pub incremental_ns: u128,
    pub peak_kib: u64,
    pub regressions: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditError {
    InvalidMeasurement,
    SplitLeak,
    Overflow,
}

/// Rejects cohort reuse across all splits, not just adjacent observations.
/// Rejects missing physical observations and oracle mismatches. Accumulates
/// using checked u128 arithmetic to avoid false speedup due to overflow.
/// No split, including VALID, is used here to train a policy.
pub fn audit(steps: &[Step]) -> Result<BTreeMap<(Split, u32), Totals>, AuditError> {
    let mut seen = BTreeSet::new();
    let mut result: BTreeMap<(Split, u32), Totals> = BTreeMap::new();
    for step in steps {
        if !seen.insert(step.cohort) {
            return Err(AuditError::SplitLeak);
        }
        if step.full_ns == 0
            || step.incremental_refresh_ns == 0
            || step.peak_kib == 0
            || !step.state_equal
            || !step.cache_equal
        {
            return Err(AuditError::InvalidMeasurement);
        }
        let totals = result.entry((step.split, step.cone_bucket)).or_default();
        totals.count = totals.count.checked_add(1).ok_or(AuditError::Overflow)?;
        totals.full_ns = totals.full_ns.checked_add(u128::from(step.full_ns)).ok_or(AuditError::Overflow)?;
        totals.incremental_ns = totals.incremental_ns.checked_add(u128::from(step.incremental_refresh_ns)).ok_or(AuditError::Overflow)?;
        totals.peak_kib = totals.peak_kib.max(step.peak_kib);
        if step.incremental_refresh_ns >= step.full_ns {
            totals.regressions = totals.regressions.checked_add(1).ok_or(AuditError::Overflow)?;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(cohort: u64, split: Split, bucket: u32, full: u64, incremental: u64) -> Step {
        Step { cohort, split, cone_bucket: bucket, full_ns: full,
            incremental_refresh_ns: incremental, peak_kib: 1024,
            state_equal: true, cache_equal: true }
    }

    #[test]
    fn heterogeneous_sequential_oracle_is_independent_of_order() {
        let input = vec![
            step(1, Split::Train, 0, 100, 40),
            step(2, Split::Valid, 1, 20, 30),
            step(3, Split::Test, 0, 90, 80),
            step(4, Split::Ood, 1, 10, 15),
            step(5, Split::Valid, 1, 50, 20),
        ];
        let expected = audit(&input).unwrap();
        assert_eq!(expected[&(Split::Valid, 1)].count, 2);
        assert_eq!(expected[&(Split::Valid, 1)].full_ns, 70);
        assert_eq!(expected[&(Split::Valid, 1)].incremental_ns, 50);
        assert_eq!(expected[&(Split::Valid, 1)].regressions, 1);
        let mut reversed = input;
        reversed.reverse();
        assert_eq!(audit(&reversed), Ok(expected));
    }

    #[test]
    fn rejects_leakage_and_missing_oracle_evidence() {
        let a = step(9, Split::Valid, 0, 100, 50);
        assert_eq!(audit(&[a, step(9, Split::Test, 1, 100, 50)]), Err(AuditError::SplitLeak));
        assert_eq!(audit(&[Step { cache_equal: false, ..a }]), Err(AuditError::InvalidMeasurement));
        assert_eq!(audit(&[Step { peak_kib: 0, ..a }]), Err(AuditError::InvalidMeasurement));
    }

    #[test]
    fn large_integer_costs_do_not_wrap() {
        let rows = [step(1, Split::Ood, 7, u64::MAX, u64::MAX - 1),
                    step(2, Split::Ood, 7, u64::MAX, u64::MAX)];
        let totals = audit(&rows).unwrap()[&(Split::Ood, 7)];
        assert_eq!(totals.full_ns, 2 * u128::from(u64::MAX));
        assert_eq!(totals.regressions, 1);
    }
}


/// A negative result is retained as evidence instead of being discarded by
/// the strict admission audit. A failed oracle can never authorize Incremental.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NegativeEvidence {
    pub observations: u64,
    pub state_mismatches: u64,
    pub cache_mismatches: u64,
    pub cost_regressions: u64,
    pub maximum_peak_kib: u64,
}

/// Record negative evidence on disjoint cohorts, including mismatched oracles.
/// This is reporting only: it does not train or authorize an admission policy.
/// Missing durations or VmHWM remain errors rather than fabricated zeros.
pub fn audit_negative_evidence(
    steps: &[Step],
) -> Result<BTreeMap<(Split, u32), NegativeEvidence>, AuditError> {
    let mut seen = BTreeSet::new();
    let mut output: BTreeMap<(Split, u32), NegativeEvidence> = BTreeMap::new();
    for step in steps {
        if !seen.insert(step.cohort) {
            return Err(AuditError::SplitLeak);
        }
        if step.full_ns == 0 || step.incremental_refresh_ns == 0 || step.peak_kib == 0 {
            return Err(AuditError::InvalidMeasurement);
        }
        let evidence = output.entry((step.split, step.cone_bucket)).or_default();
        evidence.observations = evidence.observations.checked_add(1).ok_or(AuditError::Overflow)?;
        evidence.state_mismatches = evidence
            .state_mismatches
            .checked_add(u64::from(!step.state_equal))
            .ok_or(AuditError::Overflow)?;
        evidence.cache_mismatches = evidence
            .cache_mismatches
            .checked_add(u64::from(!step.cache_equal))
            .ok_or(AuditError::Overflow)?;
        evidence.cost_regressions = evidence
            .cost_regressions
            .checked_add(u64::from(step.incremental_refresh_ns >= step.full_ns))
            .ok_or(AuditError::Overflow)?;
        evidence.maximum_peak_kib = evidence.maximum_peak_kib.max(step.peak_kib);
    }
    Ok(output)
}

#[cfg(test)]
mod negative_evidence_tests {
    use super::*;

    #[test]
    fn independent_oracle_preserves_failures_in_each_held_out_split() {
        let rows = [
            Step { cohort: 10, split: Split::Valid, cone_bucket: 1,
                full_ns: 100, incremental_refresh_ns: 110, peak_kib: 42,
                state_equal: false, cache_equal: true },
            Step { cohort: 11, split: Split::Test, cone_bucket: 1,
                full_ns: 100, incremental_refresh_ns: 50, peak_kib: 55,
                state_equal: true, cache_equal: false },
            Step { cohort: 12, split: Split::Ood, cone_bucket: 2,
                full_ns: 100, incremental_refresh_ns: 100, peak_kib: 60,
                state_equal: false, cache_equal: false },
        ];
        assert_eq!(audit(&rows), Err(AuditError::InvalidMeasurement));
        let report = audit_negative_evidence(&rows).expect("valid physical observations");
        assert_eq!(report[&(Split::Valid, 1)].state_mismatches, 1);
        assert_eq!(report[&(Split::Valid, 1)].cost_regressions, 1);
        assert_eq!(report[&(Split::Test, 1)].cache_mismatches, 1);
        assert_eq!(report[&(Split::Test, 1)].cost_regressions, 0);
        assert_eq!(report[&(Split::Ood, 2)].state_mismatches, 1);
        assert_eq!(report[&(Split::Ood, 2)].cache_mismatches, 1);
        assert_eq!(report[&(Split::Ood, 2)].maximum_peak_kib, 60);
    }

    #[test]
    fn disjointness_and_missing_measurements_fail_closed() {
        let row = Step { cohort: 1, split: Split::Train, cone_bucket: 0,
            full_ns: 5, incremental_refresh_ns: 6, peak_kib: 1,
            state_equal: false, cache_equal: false };
        assert_eq!(audit_negative_evidence(&[row, Step { split: Split::Test, ..row }]),
            Err(AuditError::SplitLeak));
        assert_eq!(audit_negative_evidence(&[Step { peak_kib: 0, ..row }]),
            Err(AuditError::InvalidMeasurement));
    }
}

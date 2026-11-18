//! Explicit disjointness guard for synthetic sequential calibration and holdout.
//! Sample identifiers are caller-supplied opaque IDs, not biological neuron IDs.
//! A repeated identifier across splits is rejected before any policy is fitted.
use crate::fourier5_sequential_cost_policy::{
    SequentialCost, SequentialCostPolicy, SequentialHoldoutAudit, SequentialPolicyError,
};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Train,
    Valid,
    Test,
    Ood,
}

#[derive(Clone, Copy, Debug)]
pub struct TaggedCost {
    pub sample_id: u128,
    pub split: Split,
    pub cost: SequentialCost,
    pub cache_current: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitError {
    DuplicateSample,
    EmptyValid,
    EmptyHoldout,
    UnexpectedSplit,
    Policy(SequentialPolicyError),
}

impl From<SequentialPolicyError> for SplitError {
    fn from(error: SequentialPolicyError) -> Self {
        Self::Policy(error)
    }
}

/// Fail closed on repeated IDs even within one split: no reused observation
/// may inflate VALID support or contaminate a holdout. Fit only VALID rows.
pub fn fit_disjoint_valid(
    rows: &[TaggedCost],
    min_samples: u64,
    margin_ns: u64,
) -> Result<SequentialCostPolicy, SplitError> {
    let mut seen = BTreeSet::new();
    let mut valid = Vec::new();
    for row in rows {
        if !seen.insert(row.sample_id) {
            return Err(SplitError::DuplicateSample);
        }
        if row.split == Split::Valid {
            valid.push(row.cost);
        }
    }
    if valid.is_empty() {
        return Err(SplitError::EmptyValid);
    }
    Ok(SequentialCostPolicy::fit_valid(
        &valid,
        min_samples,
        margin_ns,
    )?)
}

/// Audit one held-out split without updating the VALID-frozen policy.
/// Each observation has its own cache-freshness bit.
pub fn audit_disjoint_holdout(
    policy: &SequentialCostPolicy,
    rows: &[TaggedCost],
    split: Split,
) -> Result<Vec<SequentialHoldoutAudit>, SplitError> {
    if !matches!(split, Split::Test | Split::Ood) {
        return Err(SplitError::EmptyValid);
    }
    let mut seen = BTreeSet::new();
    let mut audits = Vec::new();
    for row in rows {
        if !seen.insert(row.sample_id) {
            return Err(SplitError::DuplicateSample);
        }
        if row.split == split {
            audits.push(policy.audit_holdout(&[row.cost], row.cache_current)?);
        }
    }
    Ok(audits)
}

/// Audit a held-out partition against the *same* calibration manifest.
/// The original audit function only checks duplicates within its input slice;
/// passing TEST alone cannot reveal an ID already used by VALID. This entry
/// point checks the union before any audit is produced, then refuses an empty
/// requested holdout. The frozen policy is never modified.
pub fn audit_holdout_with_calibration_manifest(
    policy: &SequentialCostPolicy,
    calibration: &[TaggedCost],
    holdout: &[TaggedCost],
    split: Split,
) -> Result<Vec<SequentialHoldoutAudit>, SplitError> {
    if !matches!(split, Split::Test | Split::Ood) {
        return Err(SplitError::EmptyValid);
    }
    let mut seen = BTreeSet::new();
    let mut has_valid = false;
    for row in calibration {
        if !matches!(row.split, Split::Train | Split::Valid) {
            return Err(SplitError::UnexpectedSplit);
        }
        has_valid |= row.split == Split::Valid;
        if !seen.insert(row.sample_id) {
            return Err(SplitError::DuplicateSample);
        }
    }
    if !has_valid {
        return Err(SplitError::EmptyValid);
    }
    // Validate the complete holdout before running the first policy audit.
    // A late duplicate or mislabeled row must never produce partial audits.
    if holdout.is_empty() {
        return Err(SplitError::EmptyHoldout);
    }
    for row in holdout {
        if row.split != split {
            return Err(SplitError::UnexpectedSplit);
        }
        if !seen.insert(row.sample_id) {
            return Err(SplitError::DuplicateSample);
        }
    }
    let mut audits = Vec::with_capacity(holdout.len());
    for row in holdout {
        audits.push(policy.audit_holdout(&[row.cost], row.cache_current)?);
    }
    Ok(audits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5_conservative_policy::Engine;
    fn row(id: u128, split: Split, full: u64, inc: u64) -> TaggedCost {
        TaggedCost {
            sample_id: id,
            split,
            cache_current: true,
            cost: SequentialCost {
                gates: 128,
                cone_gates: 4,
                full_ns: full,
                incremental_ns: inc,
                cache_update_ns: 5,
                same_output: true,
                same_decision: true,
            },
        }
    }
    #[test]
    fn duplicate_id_across_splits_is_rejected() {
        let rows = [row(1, Split::Valid, 100, 10), row(1, Split::Test, 1, 100)];
        assert!(matches!(
            fit_disjoint_valid(&rows, 1, 0),
            Err(SplitError::DuplicateSample)
        ));
    }
    #[test]
    fn test_regression_cannot_change_valid_selection() {
        let rows = [row(1, Split::Valid, 100, 10), row(2, Split::Test, 1, 100)];
        let policy = fit_disjoint_valid(&rows, 1, 0).unwrap();
        assert_eq!(policy.select(128, 4, true), Engine::Incremental);
        let audits = audit_disjoint_holdout(&policy, &rows, Split::Test).unwrap();
        assert_eq!(audits.len(), 1);
        assert_eq!(
            audits[0]
                .cells
                .iter()
                .map(|c| c.latency_regressions)
                .sum::<u64>(),
            1
        );
        assert_eq!(policy.select(128, 4, true), Engine::Incremental);
    }
    #[test]
    fn train_and_ood_never_fit_valid() {
        let rows = [row(1, Split::Train, 100, 10), row(2, Split::Ood, 100, 10)];
        assert!(matches!(
            fit_disjoint_valid(&rows, 1, 0),
            Err(SplitError::EmptyValid)
        ));
    }
    #[test]
    fn calibration_overlap_is_rejected_even_when_holdout_is_passed_separately() {
        let calibration = [
            row(11, Split::Train, 100, 10),
            row(12, Split::Valid, 100, 10),
        ];
        let policy = fit_disjoint_valid(&calibration, 1, 0).unwrap();
        let overlap = [row(12, Split::Test, 100, 10)];
        assert!(matches!(
            audit_holdout_with_calibration_manifest(&policy, &calibration, &overlap, Split::Test),
            Err(SplitError::DuplicateSample)
        ));
        let clean = [row(13, Split::Test, 100, 10)];
        assert_eq!(
            audit_holdout_with_calibration_manifest(&policy, &calibration, &clean, Split::Test)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(policy.select(128, 4, true), Engine::Incremental);
    }

    #[test]
    fn manifest_audit_rejects_mislabelled_and_empty_holdout() {
        let calibration = [row(1, Split::Valid, 100, 10)];
        let policy = fit_disjoint_valid(&calibration, 1, 0).unwrap();
        assert!(matches!(
            audit_holdout_with_calibration_manifest(&policy, &calibration, &[], Split::Ood),
            Err(SplitError::EmptyHoldout)
        ));
        assert!(matches!(
            audit_holdout_with_calibration_manifest(
                &policy,
                &calibration,
                &[row(2, Split::Valid, 100, 10)],
                Split::Test
            ),
            Err(SplitError::UnexpectedSplit)
        ));
    }

    #[test]
    fn late_overlap_and_wrong_split_are_rejected_before_any_audit() {
        let calibration = [row(1, Split::Train, 100, 10), row(2, Split::Valid, 100, 10)];
        let policy = fit_disjoint_valid(&calibration, 1, 0).unwrap();
        // A later duplicate must be rejected even when earlier samples are valid.
        let duplicate = [row(3, Split::Test, 100, 10), row(2, Split::Test, 100, 10)];
        assert!(matches!(
            audit_holdout_with_calibration_manifest(&policy, &calibration, &duplicate, Split::Test),
            Err(SplitError::DuplicateSample)
        ));
        let mislabeled = [row(3, Split::Test, 100, 10), row(4, Split::Ood, 100, 10)];
        assert!(matches!(
            audit_holdout_with_calibration_manifest(
                &policy,
                &calibration,
                &mislabeled,
                Split::Test
            ),
            Err(SplitError::UnexpectedSplit)
        ));
        let clean = [row(3, Split::Test, 100, 10), row(4, Split::Test, 100, 10)];
        assert_eq!(
            audit_holdout_with_calibration_manifest(&policy, &calibration, &clean, Split::Test)
                .unwrap()
                .len(),
            2
        );
    }
}

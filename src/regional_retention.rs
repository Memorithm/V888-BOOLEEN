//! Experimental regional retention across baseline, disruption and recovery phases.
//! Labels are caller-provided annotations, not inferred biological anatomy.
use crate::BooleanState;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetentionError {
    InvalidDimensions,
    InvalidBudget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionalRetention {
    pub region: u32,
    pub selected_members: usize,
    /// Was recall present on the final baseline observation?
    pub baseline_recalled: bool,
    /// Was recall absent on at least one disruption observation?
    pub disrupted: bool,
    /// First 1-based recovery tick completing a fresh durable streak.
    /// None means right-censored at the supplied recovery horizon.
    pub recovery_confirmation: Option<usize>,
    /// Selected annotated active-member observations in the recovery phase.
    pub recovery_active_member_ticks: u128,
}

/// Measure region-specific loss and durable recovery using three explicit phases.
/// The caller supplies the states; this function never mutates a graph or state.
/// A region can be classified as recovered only if it was recalled at baseline,
/// subsequently fell below threshold, and reached a new consecutive streak.
/// Phase lengths are bounded to prevent unbounded experiment allocation.
pub fn evaluate_regional_retention(
    selected: &[usize],
    annotations: &[Option<u32>],
    baseline: &[BooleanState],
    disruption: &[BooleanState],
    recovery: &[BooleanState],
    minimum_active: usize,
    durable_ticks: usize,
) -> Result<Vec<RegionalRetention>, RetentionError> {
    if selected.is_empty()
        || baseline.is_empty()
        || disruption.is_empty()
        || recovery.is_empty()
        || baseline.len() > 1000
        || disruption.len() > 1000
        || recovery.len() > 1000
        || minimum_active == 0
        || durable_ticks == 0
        || durable_ticks > recovery.len()
    {
        return Err(RetentionError::InvalidBudget);
    }
    let universe = annotations.len();
    if universe == 0
        || baseline
            .iter()
            .chain(disruption)
            .chain(recovery)
            .any(|state| state.len() != universe)
    {
        return Err(RetentionError::InvalidDimensions);
    }
    let mut unique = BTreeSet::new();
    let mut regions = BTreeMap::<u32, Vec<usize>>::new();
    for &member in selected {
        if member >= universe || !unique.insert(member) {
            return Err(RetentionError::InvalidDimensions);
        }
        if let Some(region) = annotations[member] {
            regions.entry(region).or_default().push(member);
        }
    }
    if regions.is_empty() {
        return Err(RetentionError::InvalidDimensions);
    }

    let mut output = Vec::with_capacity(regions.len());
    for (region, members) in regions {
        let count = |state: &BooleanState| -> usize {
            members
                .iter()
                .filter(|&&member| {
                    // All indices and state dimensions were validated above.
                    state.get(member) == Ok(true)
                })
                .count()
        };
        let baseline_recalled = count(&baseline[baseline.len() - 1]) >= minimum_active;
        let disrupted = disruption.iter().any(|state| count(state) < minimum_active);
        let mut streak = 0usize;
        let mut confirmation = None;
        let mut active_ticks = 0u128;
        for (index, state) in recovery.iter().enumerate() {
            let active = count(state);
            active_ticks += active as u128;
            if active >= minimum_active {
                streak += 1;
                if baseline_recalled
                    && disrupted
                    && streak == durable_ticks
                    && confirmation.is_none()
                {
                    confirmation = Some(index + 1);
                }
            } else {
                streak = 0;
            }
        }
        output.push(RegionalRetention {
            region,
            selected_members: members.len(),
            baseline_recalled,
            disrupted,
            recovery_confirmation: confirmation,
            recovery_active_member_ticks: active_ticks,
        });
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(bits: &[usize]) -> BooleanState {
        let mut result = BooleanState::new(5);
        for &bit in bits {
            result.set(bit, true).unwrap();
        }
        result
    }

    #[test]
    fn distinguishes_recovery_censoring_and_unannotated_members() {
        let baseline = [state(&[0, 1, 2, 3])];
        let disruption = [state(&[2, 3]), state(&[])];
        let recovery = [state(&[0, 2]), state(&[0, 1, 2]), state(&[0, 1])];
        let original = recovery.clone();
        let results = evaluate_regional_retention(
            &[0, 1, 2, 3, 4],
            &[Some(7), Some(7), Some(9), Some(9), None],
            &baseline,
            &disruption,
            &recovery,
            2,
            2,
        )
        .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].region, 7);
        assert!(results[0].baseline_recalled);
        assert!(results[0].disrupted);
        assert_eq!(results[0].recovery_confirmation, Some(3));
        assert_eq!(results[0].recovery_active_member_ticks, 5);
        assert_eq!(results[1].region, 9);
        assert_eq!(results[1].recovery_confirmation, None);
        assert_eq!(results[1].recovery_active_member_ticks, 2);
        assert_eq!(recovery, original);
    }

    #[test]
    fn refuses_invalid_dimensions_and_budgets() {
        let b = [state(&[0])];
        let a = [Some(1), None, None, None, None];
        assert_eq!(
            evaluate_regional_retention(&[0, 0], &a, &b, &b, &b, 1, 1),
            Err(RetentionError::InvalidDimensions)
        );
        assert_eq!(
            evaluate_regional_retention(&[0], &a, &b, &b, &b, 1, 2),
            Err(RetentionError::InvalidBudget)
        );
        assert_eq!(
            evaluate_regional_retention(&[0], &a[..4], &b, &b, &b, 1, 1),
            Err(RetentionError::InvalidDimensions)
        );
    }

    #[test]
    fn baseline_without_recall_cannot_claim_recovery() {
        let empty = [state(&[])];
        let active = [state(&[0]), state(&[0])];
        let result = evaluate_regional_retention(
            &[0],
            &[Some(1), None, None, None, None],
            &empty,
            &empty,
            &active,
            1,
            2,
        )
        .unwrap();
        assert!(!result[0].baseline_recalled);
        assert_eq!(result[0].recovery_confirmation, None);
    }
}

//! Observed transitions between explicitly annotated regions in Boolean snapshots.
//! These are temporal co-activations, never anatomical or synaptic edges.
use crate::BooleanState;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransitionError {
    EmptyHistory,
    LengthMismatch,
    ZeroBudget,
    TooManyRegions,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionTransition {
    pub from_region: u32,
    pub to_region: u32,
    pub observed_steps: u64,
}

/// Count consecutive-snapshot region transitions without constructing graph edges.
/// A region is active if at least `minimum_active` annotated members are active.
/// Includes self-transitions; skips unannotated neurons. Every adjacent pair
/// contributes at most once per ordered region pair. Input snapshots are chronological.
/// The region budget caps quadratic work and memory, including dormant annotations.
pub fn observe_region_transitions(
    snapshots: &[BooleanState],
    annotations: &[Option<u32>],
    minimum_active: usize,
    maximum_regions: usize,
) -> Result<Vec<RegionTransition>, TransitionError> {
    if snapshots.is_empty() {
        return Err(TransitionError::EmptyHistory);
    }
    if minimum_active == 0 || maximum_regions == 0 {
        return Err(TransitionError::ZeroBudget);
    }
    let len = snapshots[0].len();
    if annotations.len() != len || snapshots.iter().any(|s| s.len() != len) {
        return Err(TransitionError::LengthMismatch);
    }
    let regions: BTreeSet<u32> = annotations.iter().flatten().copied().collect();
    if regions.len() > maximum_regions {
        return Err(TransitionError::TooManyRegions);
    }
    let active_regions = |state: &BooleanState| {
        let mut counts = BTreeMap::<u32, usize>::new();
        for index in state.active_indices() {
            if let Some(region) = annotations[index] {
                *counts.entry(region).or_default() += 1;
            }
        }
        counts
            .into_iter()
            .filter_map(|(region, count)| (count >= minimum_active).then_some(region))
            .collect::<Vec<_>>()
    };
    let mut previous = active_regions(&snapshots[0]);
    let mut transitions = BTreeMap::<(u32, u32), u64>::new();
    for snapshot in &snapshots[1..] {
        let current = active_regions(snapshot);
        for &from in &previous {
            for &to in &current {
                let count = transitions.entry((from, to)).or_default();
                *count = count.saturating_add(1);
            }
        }
        previous = current;
    }
    Ok(transitions
        .into_iter()
        .map(
            |((from_region, to_region), observed_steps)| RegionTransition {
                from_region,
                to_region,
                observed_steps,
            },
        )
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_temporal_pairs_not_anatomical_edges() {
        let mut first = BooleanState::new(4);
        first.set(0, true).unwrap();
        first.set(1, true).unwrap();
        let mut second = BooleanState::new(4);
        second.set(2, true).unwrap();
        second.set(3, true).unwrap();
        let annotations = [Some(7), Some(7), Some(9), None];
        assert_eq!(
            observe_region_transitions(&[first, second], &annotations, 1, 2).unwrap(),
            vec![RegionTransition {
                from_region: 7,
                to_region: 9,
                observed_steps: 1
            }]
        );
    }

    #[test]
    fn threshold_and_budget_are_enforced() {
        let state = BooleanState::new(2);
        let annotations = [Some(1), Some(2)];
        assert!(
            observe_region_transitions(&[state.clone(), state.clone()], &annotations, 2, 2)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            observe_region_transitions(std::slice::from_ref(&state), &annotations, 1, 1),
            Err(TransitionError::TooManyRegions)
        );
        assert_eq!(
            observe_region_transitions(std::slice::from_ref(&state), &annotations, 0, 2),
            Err(TransitionError::ZeroBudget)
        );
        assert_eq!(
            observe_region_transitions(&[state], &[Some(1)], 1, 2),
            Err(TransitionError::LengthMismatch)
        );
    }

    #[test]
    fn one_snapshot_has_no_transitions() {
        assert!(
            observe_region_transitions(&[BooleanState::new(0)], &[], 1, 1)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            observe_region_transitions(&[], &[], 1, 1),
            Err(TransitionError::EmptyHistory)
        );
    }
}

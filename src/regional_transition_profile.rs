//! Bounded, read-only regional transition profiling of simulated Boolean states.
//! Region IDs are explicit caller annotations, never inferred anatomy.
use crate::BooleanState;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionProfileError {
    EmptyHistory,
    InvalidBudget,
    LengthMismatch,
    CountOverflow,
}

/// Per-region observed activity and state changes across consecutive snapshots.
/// The first snapshot contributes occupancy but no transition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionTransitionProfile {
    pub region_id: u32,
    pub members: u64,
    pub active_member_ticks: u128,
    pub rising_edges: u128,
    pub falling_edges: u128,
}

/// Profile up to 1,000 snapshots using only caller-provided region annotations.
/// No mutation, training, topology inference or biological firing claim.
/// Counts are checked; unannotated positions do not contribute.
pub fn profile_region_transitions(
    snapshots: &[BooleanState],
    annotations: &[Option<u32>],
) -> Result<Vec<RegionTransitionProfile>, TransitionProfileError> {
    if snapshots.is_empty() {
        return Err(TransitionProfileError::EmptyHistory);
    }
    if snapshots.len() > 1_000 {
        return Err(TransitionProfileError::InvalidBudget);
    }
    let len = snapshots[0].len();
    if annotations.len() != len || snapshots.iter().any(|s| s.len() != len) {
        return Err(TransitionProfileError::LengthMismatch);
    }
    let mut regions = BTreeMap::<u32, RegionTransitionProfile>::new();
    for (index, &region) in annotations.iter().enumerate() {
        let Some(region_id) = region else { continue };
        let entry = regions.entry(region_id).or_insert(RegionTransitionProfile {
            region_id,
            members: 0,
            active_member_ticks: 0,
            rising_edges: 0,
            falling_edges: 0,
        });
        entry.members = entry
            .members
            .checked_add(1)
            .ok_or(TransitionProfileError::CountOverflow)?;
        let mut previous = false;
        for (tick, snapshot) in snapshots.iter().enumerate() {
            let active = snapshot.get(index).expect("validated dimensions");
            if active {
                entry.active_member_ticks = entry
                    .active_member_ticks
                    .checked_add(1)
                    .ok_or(TransitionProfileError::CountOverflow)?;
            }
            if tick != 0 {
                if !previous && active {
                    entry.rising_edges = entry
                        .rising_edges
                        .checked_add(1)
                        .ok_or(TransitionProfileError::CountOverflow)?;
                } else if previous && !active {
                    entry.falling_edges = entry
                        .falling_edges
                        .checked_add(1)
                        .ok_or(TransitionProfileError::CountOverflow)?;
                }
            }
            previous = active;
        }
    }
    Ok(regions.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_rising_falling_and_initial_occupancy() {
        let mut a = BooleanState::new(5);
        let mut b = BooleanState::new(5);
        let mut c = BooleanState::new(5);
        a.set(0, true).unwrap();
        a.set(4, true).unwrap();
        b.set(1, true).unwrap();
        c.set(0, true).unwrap();
        c.set(1, true).unwrap();
        let history = [a, b, c];
        let original = history.clone();
        let regions =
            profile_region_transitions(&history, &[Some(8), Some(8), None, Some(3), Some(8)])
                .unwrap();
        assert_eq!(
            regions,
            vec![
                RegionTransitionProfile {
                    region_id: 3,
                    members: 1,
                    active_member_ticks: 0,
                    rising_edges: 0,
                    falling_edges: 0
                },
                RegionTransitionProfile {
                    region_id: 8,
                    members: 3,
                    active_member_ticks: 5,
                    rising_edges: 2,
                    falling_edges: 2
                },
            ]
        );
        assert_eq!(history, original);
    }

    #[test]
    fn rejects_invalid_input_before_any_result() {
        let s = BooleanState::new(2);
        assert_eq!(
            profile_region_transitions(&[], &[None, None]),
            Err(TransitionProfileError::EmptyHistory)
        );
        assert_eq!(
            profile_region_transitions(&[s.clone()], &[None]),
            Err(TransitionProfileError::LengthMismatch)
        );
        assert_eq!(
            profile_region_transitions(&[s.clone(), BooleanState::new(3)], &[None, None]),
            Err(TransitionProfileError::LengthMismatch)
        );
        assert_eq!(
            profile_region_transitions(&vec![s; 1_001], &[None, None]),
            Err(TransitionProfileError::InvalidBudget)
        );
    }

    #[test]
    fn unannotated_neurons_are_not_inferred() {
        let mut s = BooleanState::new(2);
        s.set(1, true).unwrap();
        assert!(profile_region_transitions(&[s], &[None, None])
            .unwrap()
            .is_empty());
    }
}

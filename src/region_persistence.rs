//! Experimental, non-anatomical persistence-gated recall over explicitly annotated regions.
//! A region is recalled only after consecutive ticks with at least `threshold` active members.
use crate::BooleanState;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersistenceError {
    EmptyRegion,
    DuplicateMember,
    OutOfBounds,
    InvalidThreshold,
    InvalidDuration,
    TooManyRegions,
    TooManyTicks,
    InconsistentStateLength,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionPersistence {
    pub longest_run: usize,
    pub first_sustained_tick: Option<usize>,
    pub sustained_ticks: usize,
    pub active_member_ticks: u64,
}

/// Measures consecutive activation of annotated regions without modifying states.
/// `first_sustained_tick` is the zero-based tick completing the first qualifying run.
/// Limits bound work: at most 64 regions and 1,000 ticks.
pub fn observe_region_persistence(
    states: &[BooleanState],
    regions: &[Vec<usize>],
    threshold: usize,
    duration: usize,
) -> Result<Vec<RegionPersistence>, PersistenceError> {
    if regions.len() > 64 {
        return Err(PersistenceError::TooManyRegions);
    }
    if states.len() > 1_000 {
        return Err(PersistenceError::TooManyTicks);
    }
    if duration == 0 {
        return Err(PersistenceError::InvalidDuration);
    }
    let width = states.first().map_or(0, BooleanState::len);
    if states.iter().any(|s| s.len() != width) {
        return Err(PersistenceError::InconsistentStateLength);
    }
    for region in regions {
        if region.is_empty() {
            return Err(PersistenceError::EmptyRegion);
        }
        if threshold == 0 || threshold > region.len() {
            return Err(PersistenceError::InvalidThreshold);
        }
        let mut seen = HashSet::with_capacity(region.len());
        for &member in region {
            if member >= width {
                return Err(PersistenceError::OutOfBounds);
            }
            if !seen.insert(member) {
                return Err(PersistenceError::DuplicateMember);
            }
        }
    }
    let mut result = Vec::with_capacity(regions.len());
    for region in regions {
        let mut run = 0usize;
        let mut longest_run = 0usize;
        let mut first_sustained_tick = None;
        let mut sustained_ticks = 0usize;
        let mut active_member_ticks = 0u64;
        for (tick, state) in states.iter().enumerate() {
            let active = region
                .iter()
                .filter(|&&member| state.get(member) == Ok(true))
                .count();
            active_member_ticks += active as u64;
            if active >= threshold {
                run += 1;
                longest_run = longest_run.max(run);
                if run >= duration {
                    first_sustained_tick.get_or_insert(tick);
                    sustained_ticks += 1;
                }
            } else {
                run = 0;
            }
        }
        result.push(RegionPersistence {
            longest_run,
            first_sustained_tick,
            sustained_ticks,
            active_member_ticks,
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(bits: &[usize]) -> BooleanState {
        let mut state = BooleanState::new(4);
        for &bit in bits {
            state.set(bit, true).unwrap();
        }
        state
    }

    #[test]
    fn persistence_resets_after_gap_and_counts_completion_ticks() {
        let states = vec![
            state(&[0, 1]), state(&[0]), state(&[0, 1]),
            state(&[0, 1]), state(&[]), state(&[0, 1]),
        ];
        let result = observe_region_persistence(&states, &[vec![0, 1]], 2, 2).unwrap();
        assert_eq!(result[0].first_sustained_tick, Some(3));
        assert_eq!(result[0].longest_run, 2);
        assert_eq!(result[0].sustained_ticks, 1);
        assert_eq!(result[0].active_member_ticks, 9);
    }

    #[test]
    fn distinct_regions_and_no_sustained_recall() {
        let states = vec![state(&[0, 2]), state(&[0, 2]), state(&[2])];
        let result = observe_region_persistence(
            &states, &[vec![0, 1], vec![2, 3]], 1, 3,
        ).unwrap();
        assert_eq!(result[0].first_sustained_tick, None);
        assert_eq!(result[1].first_sustained_tick, Some(2));
        assert_eq!(result[1].sustained_ticks, 1);
    }

    #[test]
    fn rejects_ambiguous_annotations_and_invalid_inputs() {
        let states = vec![state(&[0])];
        assert_eq!(
            observe_region_persistence(&states, &[vec![0, 0]], 1, 1),
            Err(PersistenceError::DuplicateMember)
        );
        assert_eq!(
            observe_region_persistence(&states, &[vec![4]], 1, 1),
            Err(PersistenceError::OutOfBounds)
        );
        assert_eq!(
            observe_region_persistence(&states, &[vec![0]], 0, 1),
            Err(PersistenceError::InvalidThreshold)
        );
        assert_eq!(
            observe_region_persistence(&states, &[vec![0]], 1, 0),
            Err(PersistenceError::InvalidDuration)
        );
    }
}

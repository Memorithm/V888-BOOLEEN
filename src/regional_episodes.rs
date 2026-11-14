//! Bounded recall episodes for explicitly annotated experimental regions.
//! Region labels are supplied by the caller; no anatomical connections are inferred.
use crate::BooleanState;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpisodeError {
    InvalidBudget,
    LengthMismatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionEpisode {
    pub region_id: u32,
    pub members: usize,
    pub recalled_ticks: u64,
    pub episodes: u64,
    pub longest_episode: u64,
    pub active_member_ticks: u64,
    pub first_recall_tick: Option<usize>,
}

/// Measures independent regional recall episodes over a bounded sequence.
/// A tick recalls a region when at least `minimum_active` of its explicitly
/// annotated members are active. Episodes are maximal consecutive recalled ticks.
/// All regions, including never-recalled regions, are returned in ID order.
/// This is an experimental state statistic, not biological activity evidence.
pub fn measure_regional_episodes(
    states: &[BooleanState],
    annotations: &[Option<u32>],
    minimum_active: usize,
) -> Result<Vec<RegionEpisode>, EpisodeError> {
    if states.is_empty() || states.len() > 1000 || minimum_active == 0 {
        return Err(EpisodeError::InvalidBudget);
    }
    if states.iter().any(|state| state.len() != annotations.len()) {
        return Err(EpisodeError::LengthMismatch);
    }
    let mut groups = BTreeMap::<u32, Vec<usize>>::new();
    for (index, region) in annotations.iter().enumerate() {
        if let Some(id) = region {
            groups.entry(*id).or_default().push(index);
        }
    }
    let mut output = Vec::with_capacity(groups.len());
    for (region_id, members) in groups {
        let mut result = RegionEpisode {
            region_id,
            members: members.len(),
            recalled_ticks: 0,
            episodes: 0,
            longest_episode: 0,
            active_member_ticks: 0,
            first_recall_tick: None,
        };
        let mut streak = 0_u64;
        for (tick, state) in states.iter().enumerate() {
            let active = members
                .iter()
                .filter(|&&member| state.get(member) == Ok(true))
                .count();
            result.active_member_ticks += active as u64;
            if active >= minimum_active {
                result.recalled_ticks += 1;
                if streak == 0 {
                    result.episodes += 1;
                    if result.first_recall_tick.is_none() {
                        result.first_recall_tick = Some(tick + 1);
                    }
                }
                streak += 1;
                result.longest_episode = result.longest_episode.max(streak);
            } else {
                streak = 0;
            }
        }
        output.push(result);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(bits: &[usize]) -> BooleanState {
        let mut s = BooleanState::new(4);
        for &bit in bits {
            s.set(bit, true).unwrap();
        }
        s
    }

    #[test]
    fn counts_separate_episodes_and_unrecalled_regions() {
        let states = [state(&[0, 1]), state(&[0]), state(&[0, 1]), state(&[0, 1])];
        let regions = [Some(9), Some(9), Some(2), None];
        let result = measure_regional_episodes(&states, &regions, 2).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].region_id, 2);
        assert_eq!(result[0].episodes, 0);
        assert_eq!(result[0].active_member_ticks, 0);
        assert_eq!(result[1].region_id, 9);
        assert_eq!(result[1].recalled_ticks, 3);
        assert_eq!(result[1].episodes, 2);
        assert_eq!(result[1].longest_episode, 2);
        assert_eq!(result[1].first_recall_tick, Some(1));
        assert_eq!(result[1].active_member_ticks, 7);
    }

    #[test]
    fn rejects_invalid_budget_and_dimension() {
        let regions = [Some(1), None, None, None];
        assert_eq!(
            measure_regional_episodes(&[], &regions, 1),
            Err(EpisodeError::InvalidBudget)
        );
        assert_eq!(
            measure_regional_episodes(&[state(&[])], &regions, 0),
            Err(EpisodeError::InvalidBudget)
        );
        assert_eq!(
            measure_regional_episodes(&[state(&[])], &regions[..3], 1),
            Err(EpisodeError::LengthMismatch)
        );
        assert_eq!(
            measure_regional_episodes(&vec![state(&[]); 1001], &regions, 1),
            Err(EpisodeError::InvalidBudget)
        );
    }
}

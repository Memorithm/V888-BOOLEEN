//! B888-3: bounded, matched forgetting and recovery across independent episodes.
//! Experimental recall only; no anatomical or biological inference.
use crate::BooleanState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodeError {
    EmptyEpisodes,
    TooManyEpisodes,
    InvalidMembers,
    DuplicateMember,
    InvalidThreshold,
    InvalidBoundary,
    LengthMismatch,
    TooManyTicks,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpisodeRecall {
    pub lost: u32,
    pub retained: u32,
    pub gained: u32,
    pub first_recovered_tick: Option<usize>,
    pub sustained_recovery_tick: Option<usize>,
    pub baseline_activation: u64,
    pub perturbed_activation: u64,
}

/// One matched episode. The boundary is the first evaluated post-intervention tick.
pub struct RecallEpisode<'a> {
    pub baseline: &'a [BooleanState],
    pub perturbed: &'a [BooleanState],
    pub boundary: usize,
}

/// Evaluate frozen, explicitly chosen members across independently matched episodes.
/// All episodes must have the same post-boundary budget and state dimension.
/// Sustained recovery means every remaining tick recalls in both trajectories.
pub fn evaluate_episode_recall(
    members: &[usize],
    threshold: usize,
    episodes: &[RecallEpisode<'_>],
) -> Result<Vec<EpisodeRecall>, EpisodeError> {
    if episodes.is_empty() {
        return Err(EpisodeError::EmptyEpisodes);
    }
    if episodes.len() > 64 {
        return Err(EpisodeError::TooManyEpisodes);
    }
    if members.is_empty() || members.len() > 4096 {
        return Err(EpisodeError::InvalidMembers);
    }
    if threshold == 0 || threshold > members.len() {
        return Err(EpisodeError::InvalidThreshold);
    }
    let mut indices = members.to_vec();
    indices.sort_unstable();
    if indices.windows(2).any(|w| w[0] == w[1]) {
        return Err(EpisodeError::DuplicateMember);
    }
    let first = &episodes[0];
    if first.baseline.is_empty() || first.boundary >= first.baseline.len() {
        return Err(EpisodeError::InvalidBoundary);
    }
    let dimension = first.baseline[0].len();
    if indices.last().is_some_and(|&i| i >= dimension) {
        return Err(EpisodeError::InvalidMembers);
    }
    let budget = first.baseline.len() - first.boundary;
    let mut output = Vec::with_capacity(episodes.len());
    for episode in episodes {
        if episode.baseline.is_empty() || episode.boundary >= episode.baseline.len() {
            return Err(EpisodeError::InvalidBoundary);
        }
        if episode.baseline.len() > 1000 {
            return Err(EpisodeError::TooManyTicks);
        }
        if episode.baseline.len() != episode.perturbed.len()
            || episode.baseline.len() - episode.boundary != budget
            || episode.baseline.iter().chain(episode.perturbed.iter()).any(|s| s.len() != dimension)
        {
            return Err(EpisodeError::LengthMismatch);
        }
        let mut profile = EpisodeRecall {
            lost: 0,
            retained: 0,
            gained: 0,
            first_recovered_tick: None,
            sustained_recovery_tick: None,
            baseline_activation: 0,
            perturbed_activation: 0,
        };
        let mut jointly_recalled = Vec::with_capacity(budget);
        for tick in episode.boundary..episode.baseline.len() {
            let active = |s: &BooleanState| indices.iter().filter(|&&i| s.get(i) == Ok(true)).count();
            let b = active(&episode.baseline[tick]);
            let p = active(&episode.perturbed[tick]);
            profile.baseline_activation += b as u64;
            profile.perturbed_activation += p as u64;
            let br = b >= threshold;
            let pr = p >= threshold;
            profile.lost += u32::from(br && !pr);
            profile.retained += u32::from(br && pr);
            profile.gained += u32::from(!br && pr);
            if br && pr && profile.first_recovered_tick.is_none() {
                profile.first_recovered_tick = Some(tick - episode.boundary);
            }
            jointly_recalled.push(br && pr);
        }
        let mut suffix_all = true;
        for offset in (0..budget).rev() {
            suffix_all &= jointly_recalled[offset];
            if suffix_all {
                profile.sustained_recovery_tick = Some(offset);
            }
        }
        output.push(profile);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(bits: &[usize]) -> BooleanState {
        let mut s = BooleanState::new(4);
        for &i in bits { s.set(i, true).unwrap(); }
        s
    }

    #[test]
    fn measures_independent_equal_budget_episodes() {
        let b = vec![state(&[0,1]), state(&[0,1]), state(&[0,1])];
        let p = vec![state(&[0]), state(&[0,1]), state(&[0,1])];
        let episodes = [
            RecallEpisode { baseline: &b, perturbed: &p, boundary: 0 },
            RecallEpisode { baseline: &b, perturbed: &b, boundary: 0 },
        ];
        let r = evaluate_episode_recall(&[0,1], 2, &episodes).unwrap();
        assert_eq!((r[0].lost, r[0].retained, r[0].first_recovered_tick, r[0].sustained_recovery_tick), (1,2,Some(1),Some(1)));
        assert_eq!((r[1].lost, r[1].sustained_recovery_tick), (0,Some(0)));
        assert_eq!((r[0].baseline_activation, r[0].perturbed_activation), (6,5));
    }

    #[test]
    fn rejects_unequal_budgets_and_duplicate_members() {
        let a = vec![state(&[0]), state(&[0])];
        let b = vec![state(&[0])];
        let episodes = [
            RecallEpisode { baseline: &a, perturbed: &a, boundary: 0 },
            RecallEpisode { baseline: &b, perturbed: &b, boundary: 0 },
        ];
        assert_eq!(evaluate_episode_recall(&[0], 1, &episodes).err(), Some(EpisodeError::LengthMismatch));
        assert_eq!(evaluate_episode_recall(&[0,0], 1, &episodes).err(), Some(EpisodeError::DuplicateMember));
    }

    #[test]
    fn intermittent_recall_is_not_sustained() {
        let b = vec![state(&[0]), state(&[0]), state(&[0])];
        let p = vec![state(&[0]), state(&[]), state(&[])];
        let e = [RecallEpisode { baseline: &b, perturbed: &p, boundary: 0 }];
        let r = evaluate_episode_recall(&[0], 1, &e).unwrap();
        assert_eq!(r[0].first_recovered_tick, Some(0));
        assert_eq!(r[0].sustained_recovery_tick, None);
    }
}

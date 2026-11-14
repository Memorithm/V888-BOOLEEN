//! Experimental, deterministic selection of frozen circuits across held-out episodes.
//! Membership is supplied explicitly; no biological connectivity is inferred.
use crate::BooleanState;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EpisodeSelectionError {
    EmptyCandidates,
    EmptyEpisodes,
    EmptyEpisode,
    EmptyCircuit,
    InvalidThreshold,
    OutOfBounds,
    UnequalBudgets,
    TooLarge,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpisodeCircuitScore {
    pub candidate_index: usize,
    /// Minimum number of successful recall ticks across held-out episodes.
    pub worst_episode_recalls: usize,
    pub total_recalls: usize,
    /// Number of active member bits examined across all episodes.
    pub activation_cost: u64,
}

/// Rank frozen, explicitly supplied circuits by worst-episode recall, then
/// aggregate recall, then lower activation cost, then original candidate order.
///
/// Every episode must have identical tick count and state width. Candidate
/// selection must be performed upstream using TRAIN only; this function does
/// not train, modify, or derive circuit membership from held-out observations.
pub fn rank_episode_robust_circuits(
    candidates: &[Vec<usize>],
    episodes: &[Vec<BooleanState>],
    threshold: usize,
) -> Result<Vec<EpisodeCircuitScore>, EpisodeSelectionError> {
    use EpisodeSelectionError as E;
    if candidates.is_empty() { return Err(E::EmptyCandidates); }
    if episodes.is_empty() { return Err(E::EmptyEpisodes); }
    if candidates.len() > 256 || episodes.len() > 64 { return Err(E::TooLarge); }
    let ticks = episodes[0].len();
    if ticks == 0 { return Err(E::EmptyEpisode); }
    if ticks > 1000 { return Err(E::TooLarge); }
    let width = episodes[0][0].len();
    for episode in episodes {
        if episode.is_empty() { return Err(E::EmptyEpisode); }
        if episode.len() != ticks || episode.iter().any(|s| s.len() != width) {
            return Err(E::UnequalBudgets);
        }
    }
    let mut scores = Vec::with_capacity(candidates.len());
    for (candidate_index, members) in candidates.iter().enumerate() {
        if members.is_empty() { return Err(E::EmptyCircuit); }
        if members.len() > 4096 { return Err(E::TooLarge); }
        if threshold == 0 || threshold > members.len() { return Err(E::InvalidThreshold); }
        let mut sorted = members.clone();
        sorted.sort_unstable();
        if sorted.iter().any(|&i| i >= width) { return Err(E::OutOfBounds); }
        if sorted.windows(2).any(|w| w[0] == w[1]) { return Err(E::InvalidThreshold); }
        let mut worst = usize::MAX;
        let mut total = 0usize;
        let mut cost = 0u64;
        for episode in episodes {
            let mut recalls = 0usize;
            for state in episode {
                let active = sorted.iter().filter(|&&i| state.get(i) == Ok(true)).count();
                cost = cost.checked_add(active as u64).ok_or(E::TooLarge)?;
                if active >= threshold { recalls += 1; }
            }
            worst = worst.min(recalls);
            total = total.checked_add(recalls).ok_or(E::TooLarge)?;
        }
        scores.push(EpisodeCircuitScore {
            candidate_index, worst_episode_recalls: worst,
            total_recalls: total, activation_cost: cost,
        });
    }
    scores.sort_by(|a,b| b.worst_episode_recalls.cmp(&a.worst_episode_recalls)
        .then_with(|| b.total_recalls.cmp(&a.total_recalls))
        .then_with(|| a.activation_cost.cmp(&b.activation_cost))
        .then_with(|| a.candidate_index.cmp(&b.candidate_index)));
    Ok(scores)
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
    fn worst_episode_beats_aggregate_recall() {
        let episodes = vec![
            vec![state(&[0,1,2]), state(&[0,1,2])],
            vec![state(&[2]), state(&[2])],
        ];
        let ranked = rank_episode_robust_circuits(&[vec![0,1], vec![2]], &episodes, 1).unwrap();
        assert_eq!(ranked[0].candidate_index, 1);
        assert_eq!(ranked[0].worst_episode_recalls, 2);
        assert_eq!(ranked[1].worst_episode_recalls, 0);
    }
    #[test]
    fn rejects_unequal_episodes_and_duplicate_members() {
        let bad = vec![vec![state(&[0])], vec![state(&[0]), state(&[0])]];
        assert_eq!(rank_episode_robust_circuits(&[vec![0]], &bad, 1), Err(EpisodeSelectionError::UnequalBudgets));
        assert_eq!(rank_episode_robust_circuits(&[vec![0,0]], &[vec![state(&[0])]], 1), Err(EpisodeSelectionError::InvalidThreshold));
    }
    #[test]
    fn selection_does_not_mutate_inputs() {
        let candidates = vec![vec![1],vec![0]];
        let episodes = vec![vec![state(&[0,1])]];
        let original = episodes.clone();
        let result = rank_episode_robust_circuits(&candidates, &episodes, 1).unwrap();
        assert_eq!(result[0].candidate_index, 0);
        assert_eq!(episodes, original);
    }
}

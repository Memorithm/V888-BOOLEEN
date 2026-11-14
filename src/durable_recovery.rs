//! B888-3: frozen, explicitly selected circuits evaluated for durable recovery.
//! All trajectories are experimental observations, not biological anatomy.
use crate::BooleanState;
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DurableRecovery {
    pub candidate_index: usize,
    pub recovered_episodes: usize,
    pub worst_recovery_delay: Option<usize>,
    pub total_recalled_ticks: usize,
    pub active_member_ticks: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DurableRecoveryError {
    EmptyInput,
    BudgetExceeded,
    UnequalEpisodeLength,
    InvalidCandidate,
    DuplicateMember,
    InvalidThreshold,
    DimensionMismatch,
}

/// Rank frozen candidates by number of episodes with a sustained recall run,
/// then by worst recovery latency (smaller is better), recalled ticks and cost.
/// Every episode is held out and evaluated with the same threshold and horizon.
/// A missing durable recovery is represented by None, never by a fabricated delay.
pub fn rank_durable_recovery(
    candidates: &[Vec<usize>],
    episodes: &[Vec<BooleanState>],
    threshold: usize,
    sustained_ticks: usize,
) -> Result<Vec<DurableRecovery>, DurableRecoveryError> {
    use DurableRecoveryError as E;
    if candidates.is_empty() || episodes.is_empty() { return Err(E::EmptyInput); }
    if candidates.len() > 256 || episodes.len() > 64 { return Err(E::BudgetExceeded); }
    let horizon = episodes[0].len();
    if horizon == 0 || horizon > 1000 || sustained_ticks == 0 || sustained_ticks > horizon {
        return Err(E::BudgetExceeded);
    }
    let width = episodes[0][0].len();
    if episodes.iter().any(|e| e.len() != horizon) { return Err(E::UnequalEpisodeLength); }
    if episodes.iter().flat_map(|e| e.iter()).any(|s| s.len() != width) {
        return Err(E::DimensionMismatch);
    }
    if threshold == 0 { return Err(E::InvalidThreshold); }
    for members in candidates {
        if members.is_empty() || members.len() > 4096 { return Err(E::InvalidCandidate); }
        if threshold > members.len() || members.iter().any(|&m| m >= width) {
            return Err(E::InvalidThreshold);
        }
        let unique: HashSet<_> = members.iter().collect();
        if unique.len() != members.len() { return Err(E::DuplicateMember); }
    }
    let mut ranked = Vec::with_capacity(candidates.len());
    for (candidate_index, members) in candidates.iter().enumerate() {
        let mut recovered_episodes = 0usize;
        let mut worst_recovery_delay = None::<usize>;
        let mut total_recalled_ticks = 0usize;
        let mut active_member_ticks = 0usize;
        for episode in episodes {
            let mut streak = 0usize;
            let mut first_sustained = None;
            for (tick, state) in episode.iter().enumerate() {
                let active = members.iter().filter(|&&m| state.get(m) == Ok(true)).count();
                active_member_ticks += active;
                if active >= threshold {
                    total_recalled_ticks += 1;
                    streak += 1;
                    if streak == sustained_ticks && first_sustained.is_none() {
                        first_sustained = Some(tick + 1 - sustained_ticks);
                    }
                } else {
                    streak = 0;
                }
            }
            if let Some(delay) = first_sustained {
                recovered_episodes += 1;
                worst_recovery_delay = Some(worst_recovery_delay.map_or(delay, |prev| prev.max(delay)));
            }
        }
        ranked.push(DurableRecovery {
            candidate_index, recovered_episodes, worst_recovery_delay,
            total_recalled_ticks, active_member_ticks,
        });
    }
    ranked.sort_by(|a, b| {
        b.recovered_episodes.cmp(&a.recovered_episodes)
            .then_with(|| a.worst_recovery_delay.unwrap_or(usize::MAX)
                .cmp(&b.worst_recovery_delay.unwrap_or(usize::MAX)))
            .then_with(|| b.total_recalled_ticks.cmp(&a.total_recalled_ticks))
            .then_with(|| a.active_member_ticks.cmp(&b.active_member_ticks))
            .then_with(|| a.candidate_index.cmp(&b.candidate_index))
    });
    Ok(ranked)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state(bits: &[usize]) -> BooleanState {
        let mut s = BooleanState::new(4);
        for &bit in bits { s.set(bit, true).unwrap(); }
        s
    }
    #[test]
    fn ranks_sustained_recovery_across_equal_episodes() {
        let episodes = vec![
            vec![state(&[]), state(&[0]), state(&[0]), state(&[1])],
            vec![state(&[]), state(&[0]), state(&[0]), state(&[])],
        ];
        let result = rank_durable_recovery(&[vec![1], vec![0]], &episodes, 1, 2).unwrap();
        assert_eq!(result[0].candidate_index, 1);
        assert_eq!(result[0].recovered_episodes, 2);
        assert_eq!(result[0].worst_recovery_delay, Some(1));
        assert_eq!(result[1].worst_recovery_delay, None);
    }
    #[test]
    fn rejects_leakage_and_invalid_dimensions() {
        assert_eq!(rank_durable_recovery(&[vec![0]], &[vec![state(&[])], vec![]], 1, 1),
            Err(DurableRecoveryError::UnequalEpisodeLength));
        assert_eq!(rank_durable_recovery(&[vec![0, 0]], &[vec![state(&[])]], 1, 1),
            Err(DurableRecoveryError::DuplicateMember));
    }
    #[test]
    fn sustained_run_must_be_consecutive() {
        let episodes = vec![vec![state(&[0]), state(&[]), state(&[0]), state(&[])]];
        let result = rank_durable_recovery(&[vec![0]], &episodes, 1, 2).unwrap();
        assert_eq!(result[0].recovered_episodes, 0);
        assert_eq!(result[0].total_recalled_ticks, 2);
    }
}

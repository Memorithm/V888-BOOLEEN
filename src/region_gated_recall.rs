//! Select frozen experimental subcircuits gated by explicitly annotated regional activity.
//! A gate is a temporal observation, never an inferred anatomical connection.
use crate::{
    subcircuit_recall::{RecallError, Subcircuit},
    BooleanState,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GatedRecallError {
    InvalidBudget,
    EmptyHistory,
    LengthMismatch,
    MissingRegion,
    Recall(RecallError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatedRecallScore {
    pub candidate: usize,
    pub gated_ticks: usize,
    pub recalled_ticks: usize,
    pub longest_recall_streak: usize,
    pub activation_cost: u64,
}

/// Rank frozen candidates by sustained recall *during* activity of an explicitly
/// annotated region. All candidates share identical held-out snapshots and budgets.
/// Ties: recall streak, recall ticks, lower activation cost, original index.
/// A non-active gate breaks the streak; it is never treated as a successful recall.
pub fn rank_region_gated_recall(
    candidates: &[Subcircuit],
    snapshots: &[BooleanState],
    annotations: &[Option<u32>],
    region: u32,
    minimum_region_active: usize,
    minimum_circuit_active: usize,
) -> Result<Vec<GatedRecallScore>, GatedRecallError> {
    if candidates.is_empty()
        || candidates.len() > 256
        || minimum_region_active == 0
        || minimum_circuit_active == 0
        || snapshots.len() > 1000
    {
        return Err(GatedRecallError::InvalidBudget);
    }
    let first = snapshots.first().ok_or(GatedRecallError::EmptyHistory)?;
    let len = first.len();
    if annotations.len() != len
        || snapshots.iter().any(|s| s.len() != len)
        || candidates.iter().any(|c| c.universe_len() != len)
    {
        return Err(GatedRecallError::LengthMismatch);
    }
    let members: Vec<usize> = annotations
        .iter()
        .enumerate()
        .filter_map(|(i, label)| (*label == Some(region)).then_some(i))
        .collect();
    if members.is_empty() {
        return Err(GatedRecallError::MissingRegion);
    }
    let mut scores = Vec::with_capacity(candidates.len());
    for (candidate, circuit) in candidates.iter().enumerate() {
        let mut score = GatedRecallScore {
            candidate,
            gated_ticks: 0,
            recalled_ticks: 0,
            longest_recall_streak: 0,
            activation_cost: 0,
        };
        let mut streak = 0usize;
        for state in snapshots {
            let gate_active = members
                .iter()
                .filter(|&&i| state.get(i) == Ok(true))
                .count()
                >= minimum_region_active;
            if gate_active {
                score.gated_ticks += 1;
                let active = circuit
                    .members()
                    .iter()
                    .filter(|&&i| state.get(i) == Ok(true))
                    .count();
                score.activation_cost += active as u64;
                if circuit
                    .recall(state, minimum_circuit_active)
                    .map_err(GatedRecallError::Recall)?
                {
                    score.recalled_ticks += 1;
                    streak += 1;
                    score.longest_recall_streak = score.longest_recall_streak.max(streak);
                } else {
                    streak = 0;
                }
            } else {
                streak = 0;
            }
        }
        scores.push(score);
    }
    scores.sort_by(|a, b| {
        b.longest_recall_streak
            .cmp(&a.longest_recall_streak)
            .then_with(|| b.recalled_ticks.cmp(&a.recalled_ticks))
            .then_with(|| a.activation_cost.cmp(&b.activation_cost))
            .then_with(|| a.candidate.cmp(&b.candidate))
    });
    Ok(scores)
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
    fn region_gate_breaks_streak_and_ranks_frozen_candidates() {
        let candidates = [
            Subcircuit::new(4, vec![1]).unwrap(),
            Subcircuit::new(4, vec![2]).unwrap(),
        ];
        let history = [
            state(&[0, 1]),
            state(&[0, 1, 2]),
            state(&[1, 2]),
            state(&[0, 1]),
        ];
        let result =
            rank_region_gated_recall(&candidates, &history, &[Some(7), None, None, None], 7, 1, 1)
                .unwrap();
        assert_eq!(result[0].candidate, 0);
        assert_eq!(result[0].gated_ticks, 3);
        assert_eq!(result[0].recalled_ticks, 3);
        assert_eq!(result[0].longest_recall_streak, 2);
        assert_eq!(result[1].recalled_ticks, 1);
    }

    #[test]
    fn rejects_missing_region_and_mismatched_dimensions() {
        let circuit = Subcircuit::new(4, vec![1]).unwrap();
        assert_eq!(
            rank_region_gated_recall(
                std::slice::from_ref(&circuit),
                &[state(&[1])],
                &[None; 4],
                3,
                1,
                1
            ),
            Err(GatedRecallError::MissingRegion)
        );
        assert_eq!(
            rank_region_gated_recall(&[circuit], &[state(&[1])], &[None; 3], 3, 1, 1),
            Err(GatedRecallError::LengthMismatch)
        );
    }

    #[test]
    fn rejects_zero_budgets_and_empty_history() {
        let circuit = Subcircuit::new(4, vec![1]).unwrap();
        assert_eq!(
            rank_region_gated_recall(
                std::slice::from_ref(&circuit),
                &[state(&[1])],
                &[Some(1); 4],
                1,
                0,
                1
            ),
            Err(GatedRecallError::InvalidBudget)
        );
        assert_eq!(
            rank_region_gated_recall(&[circuit], &[], &[Some(1); 4], 1, 1, 1),
            Err(GatedRecallError::EmptyHistory)
        );
    }
}

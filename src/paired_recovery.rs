//! Paired, bounded comparison of post-perturbation recall on frozen circuits.
//! These are experimental activity traces, not anatomical or causal evidence.
use crate::{
    subcircuit_recall::{RecallError, Subcircuit},
    BooleanState,
};

const MAX_STEPS: usize = 1_000;
const MAX_CANDIDATES: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecoveryComparisonError {
    EmptyCandidates,
    InvalidWindow,
    TooManyCandidates,
    Recall(RecallError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryCandidate {
    /// Caller-supplied stable candidate identifier.
    pub id: u32,
    pub circuit: Subcircuit,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryScore {
    pub id: u32,
    /// Number of ticks after suppression with a recalled circuit.
    pub recalled_ticks: usize,
    /// First post-suppression tick (zero-based), or None if never recovered.
    pub first_recall_tick: Option<usize>,
    /// First post-suppression tick starting a sustained run.
    pub sustained_recovery_tick: Option<usize>,
    /// Total active member-ticks in the post-suppression window.
    pub activation_cost: usize,
}

/// Compare fixed candidates on exactly the same held-out activity trace.
/// All candidates use the same threshold, suppression boundary and required
/// consecutive recall duration. Sort: sustained recovery first, earlier
/// recovery, then more recalled ticks, lower cost, lower identifier.
/// This ranking is descriptive, not evidence that a perturbation caused recovery.
pub fn rank_paired_recovery(
    candidates: &[RecoveryCandidate],
    heldout: &[BooleanState],
    suppression_end: usize,
    minimum_active: usize,
    sustained_steps: usize,
) -> Result<Vec<RecoveryScore>, RecoveryComparisonError> {
    if candidates.is_empty() {
        return Err(RecoveryComparisonError::EmptyCandidates);
    }
    if candidates.len() > MAX_CANDIDATES {
        return Err(RecoveryComparisonError::TooManyCandidates);
    }
    if heldout.is_empty()
        || heldout.len() > MAX_STEPS
        || suppression_end >= heldout.len()
        || minimum_active == 0
        || sustained_steps == 0
    {
        return Err(RecoveryComparisonError::InvalidWindow);
    }
    let post = &heldout[suppression_end..];
    if sustained_steps > post.len() {
        return Err(RecoveryComparisonError::InvalidWindow);
    }
    let mut scores = Vec::with_capacity(candidates.len());
    let mut ids = std::collections::BTreeSet::new();
    for candidate in candidates {
        if !ids.insert(candidate.id) {
            return Err(RecoveryComparisonError::InvalidWindow);
        }
        if heldout
            .iter()
            .any(|s| s.len() != candidate.circuit.universe_len())
        {
            return Err(RecoveryComparisonError::Recall(RecallError::LengthMismatch));
        }
        let mut recalled_ticks = 0;
        let mut first_recall_tick = None;
        let mut sustained_recovery_tick = None;
        let mut streak = 0;
        let mut activation_cost = 0usize;
        for (tick, state) in post.iter().enumerate() {
            for &member in candidate.circuit.members() {
                if state
                    .get(member)
                    .map_err(|_| RecoveryComparisonError::Recall(RecallError::OutOfBounds))?
                {
                    activation_cost += 1;
                }
            }
            if candidate
                .circuit
                .recall(state, minimum_active)
                .map_err(RecoveryComparisonError::Recall)?
            {
                recalled_ticks += 1;
                if first_recall_tick.is_none() {
                    first_recall_tick = Some(tick);
                }
                streak += 1;
                if streak == sustained_steps && sustained_recovery_tick.is_none() {
                    sustained_recovery_tick = Some(tick + 1 - sustained_steps);
                }
            } else {
                streak = 0;
            }
        }
        scores.push(RecoveryScore {
            id: candidate.id,
            recalled_ticks,
            first_recall_tick,
            sustained_recovery_tick,
            activation_cost,
        });
    }
    scores.sort_by(|a, b| {
        a.sustained_recovery_tick
            .is_none()
            .cmp(&b.sustained_recovery_tick.is_none())
            .then_with(|| a.sustained_recovery_tick.cmp(&b.sustained_recovery_tick))
            .then_with(|| b.recalled_ticks.cmp(&a.recalled_ticks))
            .then_with(|| a.activation_cost.cmp(&b.activation_cost))
            .then_with(|| a.id.cmp(&b.id))
    });
    Ok(scores)
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
    fn sustained_recovery_beats_intermitent_recall_on_same_trace() {
        let trace = vec![
            state(&[]),
            state(&[0, 2]),
            state(&[0, 1]),
            state(&[0, 1, 2]),
            state(&[0, 1]),
        ];
        let candidates = vec![
            RecoveryCandidate {
                id: 9,
                circuit: Subcircuit::new(4, vec![0, 2]).unwrap(),
            },
            RecoveryCandidate {
                id: 2,
                circuit: Subcircuit::new(4, vec![0, 1]).unwrap(),
            },
        ];
        let ranked = rank_paired_recovery(&candidates, &trace, 1, 2, 2).unwrap();
        assert_eq!(ranked[0].id, 2);
        assert_eq!(ranked[0].sustained_recovery_tick, Some(1));
        assert_eq!(ranked[1].sustained_recovery_tick, None);
        assert_eq!(ranked[0].recalled_ticks, 3);
    }

    #[test]
    fn no_recovery_is_right_censored_and_inputs_unchanged() {
        let trace = vec![state(&[]), state(&[0]), state(&[1])];
        let original = trace.clone();
        let candidates = [RecoveryCandidate {
            id: 1,
            circuit: Subcircuit::new(4, vec![0, 1]).unwrap(),
        }];
        let result = rank_paired_recovery(&candidates, &trace, 1, 2, 2).unwrap();
        assert_eq!(result[0].first_recall_tick, None);
        assert_eq!(result[0].sustained_recovery_tick, None);
        assert_eq!(trace, original);
    }

    #[test]
    fn rejects_invalid_budgets_and_mismatched_dimensions() {
        let candidates = [RecoveryCandidate {
            id: 1,
            circuit: Subcircuit::new(3, vec![0]).unwrap(),
        }];
        let trace = [state(&[0])];
        assert_eq!(
            rank_paired_recovery(&candidates, &trace, 1, 1, 1),
            Err(RecoveryComparisonError::InvalidWindow)
        );
        assert_eq!(
            rank_paired_recovery(&candidates, &trace, 0, 1, 1),
            Err(RecoveryComparisonError::Recall(RecallError::LengthMismatch))
        );
        assert_eq!(
            rank_paired_recovery(&[], &trace, 0, 1, 1),
            Err(RecoveryComparisonError::EmptyCandidates)
        );
    }
}

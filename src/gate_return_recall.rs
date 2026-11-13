//! Observe recall after an explicitly annotated region reactivates.
//! Temporal co-occurrence is not anatomical connectivity or causality.
use crate::{subcircuit_recall::{RecallError, Subcircuit}, BooleanState};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GateReturnError {
    InvalidBudget,
    EmptyHistory,
    LengthMismatch,
    MissingRegion,
    Recall(RecallError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateReturnScore {
    pub candidate: usize,
    pub return_events: usize,
    pub recovered_events: usize,
    pub first_recovery_delay: Option<usize>,
    pub recall_ticks: usize,
    pub activation_cost: u64,
}

/// Count recovery within a fixed horizon after an annotated region rises from
/// inactive to active. A return is counted once, even if the gate stays active.
/// Overlapping horizons are allowed, and are applied identically to all candidates.
/// A candidate recovers if recall succeeds on any tick within its return horizon.
pub fn rank_gate_return_recall(
    candidates: &[Subcircuit],
    snapshots: &[BooleanState],
    annotations: &[Option<u32>],
    region: u32,
    minimum_region_active: usize,
    minimum_circuit_active: usize,
    horizon: usize,
) -> Result<Vec<GateReturnScore>, GateReturnError> {
    if candidates.is_empty()
        || candidates.len() > 256
        || snapshots.len() > 1000
        || minimum_region_active == 0
        || minimum_circuit_active == 0
        || horizon == 0
        || horizon > 1000
    {
        return Err(GateReturnError::InvalidBudget);
    }
    let first = snapshots.first().ok_or(GateReturnError::EmptyHistory)?;
    let len = first.len();
    if annotations.len() != len
        || snapshots.iter().any(|state| state.len() != len)
        || candidates.iter().any(|candidate| candidate.universe_len() != len)
    {
        return Err(GateReturnError::LengthMismatch);
    }
    let members: Vec<usize> = annotations
        .iter()
        .enumerate()
        .filter_map(|(index, label)| (*label == Some(region)).then_some(index))
        .collect();
    if members.is_empty() {
        return Err(GateReturnError::MissingRegion);
    }
    let mut returns = Vec::new();
    let mut previously_active = false;
    for (tick, state) in snapshots.iter().enumerate() {
        let active = members
            .iter()
            .filter(|&&index| state.get(index) == Ok(true))
            .count()
            >= minimum_region_active;
        if active && !previously_active {
            returns.push(tick);
        }
        previously_active = active;
    }

    let mut scores = Vec::with_capacity(candidates.len());
    for (candidate, circuit) in candidates.iter().enumerate() {
        let mut score = GateReturnScore {
            candidate,
            return_events: returns.len(),
            recovered_events: 0,
            first_recovery_delay: None,
            recall_ticks: 0,
            activation_cost: 0,
        };
        for &start in &returns {
            let end = start.saturating_add(horizon).min(snapshots.len());
            let mut recovered = false;
            for (delay, state) in snapshots[start..end].iter().enumerate() {
                let active = circuit
                    .members()
                    .iter()
                    .filter(|&&index| state.get(index) == Ok(true))
                    .count();
                score.activation_cost += active as u64;
                if circuit
                    .recall(state, minimum_circuit_active)
                    .map_err(GateReturnError::Recall)?
                {
                    score.recall_ticks += 1;
                    if !recovered {
                        score.recovered_events += 1;
                        score.first_recovery_delay = Some(
                            score.first_recovery_delay.map_or(delay, |old| old.min(delay)),
                        );
                        recovered = true;
                    }
                }
            }
        }
        scores.push(score);
    }
    scores.sort_by(|a, b| {
        b.recovered_events
            .cmp(&a.recovered_events)
            .then_with(|| b.recall_ticks.cmp(&a.recall_ticks))
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
    fn return_events_rank_recovery_without_mutating_states() {
        let candidates = [
            Subcircuit::new(4, vec![1]).unwrap(),
            Subcircuit::new(4, vec![2]).unwrap(),
        ];
        let history = [
            state(&[]),
            state(&[0]),
            state(&[0, 1]),
            state(&[]),
            state(&[0, 1]),
            state(&[0, 2]),
        ];
        let before = history.clone();
        let scores = rank_gate_return_recall(
            &candidates,
            &history,
            &[Some(7), None, None, None],
            7,
            1,
            1,
            2,
        )
        .unwrap();
        assert_eq!(scores[0].candidate, 0);
        assert_eq!(scores[0].return_events, 2);
        assert_eq!(scores[0].recovered_events, 2);
        assert_eq!(scores[0].first_recovery_delay, Some(0));
        assert_eq!(scores[1].recovered_events, 1);
        assert_eq!(history, before);
    }

    #[test]
    fn no_return_has_zero_recovery() {
        let circuit = Subcircuit::new(4, vec![1]).unwrap();
        let scores = rank_gate_return_recall(
            &[circuit],
            &[state(&[]), state(&[1])],
            &[Some(7), None, None, None],
            7,
            1,
            1,
            2,
        )
        .unwrap();
        assert_eq!(scores[0].return_events, 0);
        assert_eq!(scores[0].first_recovery_delay, None);
    }

    #[test]
    fn rejects_invalid_dimensions_and_horizon() {
        let circuit = Subcircuit::new(4, vec![1]).unwrap();
        assert_eq!(
            rank_gate_return_recall(
                &[circuit.clone()],
                &[state(&[0])],
                &[Some(7); 3],
                7,
                1,
                1,
                2,
            ),
            Err(GateReturnError::LengthMismatch)
        );
        assert_eq!(
            rank_gate_return_recall(
                &[circuit],
                &[state(&[0])],
                &[Some(7); 4],
                7,
                1,
                1,
                0,
            ),
            Err(GateReturnError::InvalidBudget)
        );
    }
}

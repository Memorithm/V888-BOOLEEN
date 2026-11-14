//! Hypothetical ablation-robust circuit choice from frozen candidate memberships.
//! This is an experimental recall rule, not evidence of biological connectivity.
use crate::{subcircuit_recall::{RecallError, Subcircuit}, BooleanState};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RobustCircuit {
    pub candidate_index: usize,
    pub worst_case_recalled: usize,
    pub baseline_recalled: usize,
    pub member_count: usize,
}

/// Choose the circuit retaining the most recall ticks after its worst single-member
/// silencing. All candidates see identical snapshots; no training or mutation occurs.
/// Ties prefer baseline recall, fewer members, then earliest candidate.
pub fn select_ablation_robust(
    candidates: &[Subcircuit],
    snapshots: &[BooleanState],
    minimum_active: usize,
) -> Result<RobustCircuit, RecallError> {
    if candidates.is_empty() { return Err(RecallError::EmptyCircuit); }
    if snapshots.is_empty() { return Err(RecallError::EmptyHistory); }
    if minimum_active == 0 { return Err(RecallError::ZeroBudget); }
    if candidates.len() > 256 || snapshots.len() > 1_000 {
        return Err(RecallError::ZeroBudget);
    }
    let universe = snapshots[0].len();
    if snapshots.iter().any(|s| s.len() != universe)
        || candidates.iter().any(|c| c.universe_len() != universe) {
        return Err(RecallError::LengthMismatch);
    }
    let mut best: Option<RobustCircuit> = None;
    for (candidate_index, circuit) in candidates.iter().enumerate() {
        if circuit.members().len() > 4_096 { return Err(RecallError::ZeroBudget); }
        let mut baseline_recalled = 0usize;
        let mut losses = vec![0usize; circuit.members().len()];
        for state in snapshots {
            let mut active_count = 0usize;
            let mut active_members = Vec::new();
            for (position, &member) in circuit.members().iter().enumerate() {
                if state.get(member).expect("validated membership") {
                    active_count += 1;
                    active_members.push(position);
                }
            }
            if active_count >= minimum_active {
                baseline_recalled += 1;
                if active_count == minimum_active {
                    for position in active_members { losses[position] += 1; }
                }
            }
        }
        let worst_loss = losses.into_iter().max().unwrap_or(0);
        let result = RobustCircuit {
            candidate_index,
            worst_case_recalled: baseline_recalled - worst_loss,
            baseline_recalled,
            member_count: circuit.members().len(),
        };
        let better = best.as_ref().is_none_or(|old| {
            (result.worst_case_recalled, result.baseline_recalled,
             std::cmp::Reverse(result.member_count))
                > (old.worst_case_recalled, old.baseline_recalled,
                   std::cmp::Reverse(old.member_count))
        });
        if better { best = Some(result); }
    }
    Ok(best.expect("nonempty candidates"))
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
    fn prefers_robust_recall_over_fragile_baseline() {
        let fragile = Subcircuit::new(4, vec![0, 1]).unwrap();
        let robust = Subcircuit::new(4, vec![0, 1, 2]).unwrap();
        let history = [state(&[0, 1, 2]), state(&[0, 1, 2]), state(&[0, 1])];
        let selected = select_ablation_robust(&[fragile, robust], &history, 2).unwrap();
        assert_eq!(selected.candidate_index, 1);
        assert_eq!(selected.baseline_recalled, 3);
        assert_eq!(selected.worst_case_recalled, 2);
    }
    #[test]
    fn ties_are_deterministic_and_inputs_unchanged() {
        let a = Subcircuit::new(4, vec![0, 1]).unwrap();
        let b = a.clone();
        let history = [state(&[0, 1])];
        let copy = history.clone();
        assert_eq!(select_ablation_robust(&[a, b], &history, 1).unwrap().candidate_index, 0);
        assert_eq!(history, copy);
    }
    #[test]
    fn rejects_incompatible_dimensions_and_zero_threshold() {
        let c = Subcircuit::new(4, vec![0]).unwrap();
        assert_eq!(select_ablation_robust(&[c.clone()], &[BooleanState::new(3)], 1), Err(RecallError::LengthMismatch));
        assert_eq!(select_ablation_robust(&[c], &[BooleanState::new(4)], 0), Err(RecallError::ZeroBudget));
    }
}

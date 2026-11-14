//! Paired held-out recall robustness of frozen experimental subcircuits.
//! Neither selection nor adaptation occurs on held-out observations.
use crate::{
    subcircuit_recall::{RecallError, Subcircuit},
    BooleanState,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeldoutAblation {
    pub baseline_matches: usize,
    pub worst_case_matches: usize,
    pub worst_member: usize,
    pub total_member_activations: usize,
}

/// Evaluate the worst single-member silencing on identical held-out snapshots.
/// All candidates must have equal membership size and use the same recall threshold.
/// Returns the most robust candidate (highest worst-case matches), then lower
/// activation cost, then lower input index. No biological lesion is implied.
pub fn select_heldout_ablation(
    candidates: &[Subcircuit],
    heldout: &[BooleanState],
    minimum_active: usize,
) -> Result<(usize, HeldoutAblation), RecallError> {
    if candidates.is_empty() {
        return Err(RecallError::EmptyCircuit);
    }
    if heldout.is_empty() {
        return Err(RecallError::EmptyHistory);
    }
    if minimum_active == 0 || candidates.len() > 256 || heldout.len() > 1000 {
        return Err(RecallError::ZeroBudget);
    }
    let universe = candidates[0].universe_len();
    let width = candidates[0].members().len();
    if candidates
        .iter()
        .any(|c| c.universe_len() != universe || c.members().len() != width)
        || heldout.iter().any(|s| s.len() != universe)
    {
        return Err(RecallError::LengthMismatch);
    }
    if width == 0 || width > 4096 {
        return Err(RecallError::ZeroBudget);
    }
    let mut best: Option<(usize, HeldoutAblation)> = None;
    for (candidate_index, circuit) in candidates.iter().enumerate() {
        let mut baseline = 0;
        let mut losses = vec![0usize; width];
        let mut cost = 0usize;
        for state in heldout {
            let mut active = 0usize;
            for &member in circuit.members() {
                if state.get(member).expect("validated index") {
                    active += 1;
                }
            }
            cost += active;
            if active >= minimum_active {
                baseline += 1;
                if active == minimum_active {
                    for (position, &member) in circuit.members().iter().enumerate() {
                        if state.get(member).expect("validated index") {
                            losses[position] += 1;
                        }
                    }
                }
            }
        }
        let (worst_position, &max_loss) = losses
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.cmp(b).then_with(|| ib.cmp(ia)))
            .expect("nonempty circuit");
        let report = HeldoutAblation {
            baseline_matches: baseline,
            worst_case_matches: baseline - max_loss,
            worst_member: circuit.members()[worst_position],
            total_member_activations: cost,
        };
        let replace = best.as_ref().is_none_or( |(_, current)| {
            report.worst_case_matches > current.worst_case_matches
                || (report.worst_case_matches == current.worst_case_matches
                    && report.total_member_activations < current.total_member_activations)
        });
        if replace {
            best = Some((candidate_index, report));
        }
    }
    Ok(best.expect("nonempty candidates"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(indices: &[usize]) -> BooleanState {
        let mut result = BooleanState::new(4);
        for &index in indices {
            result.set(index, true).unwrap();
        }
        result
    }

    #[test]
    fn paired_worst_case_prefers_redundancy() {
        let fragile = Subcircuit::new(4, vec![0, 1, 2]).unwrap();
        let robust = Subcircuit::new(4, vec![1, 2, 3]).unwrap();
        let heldout = [state(&[0, 1, 2, 3]), state(&[1, 2, 3]), state(&[1, 2, 3])];
        let (index, result) = select_heldout_ablation(&[fragile, robust], &heldout, 2).unwrap();
        assert_eq!(index, 1);
        assert_eq!(result.baseline_matches, 3);
        assert_eq!(result.worst_case_matches, 3);
    }

    #[test]
    fn deterministic_tie_and_cost() {
        let a = Subcircuit::new(4, vec![0, 1]).unwrap();
        let b = Subcircuit::new(4, vec![2, 3]).unwrap();
        let heldout = [state(&[0, 1, 2, 3])];
        let (index, report) = select_heldout_ablation(&[a, b], &heldout, 1).unwrap();
        assert_eq!(index, 0);
        assert_eq!(report.worst_member, 0);
        assert_eq!(report.worst_case_matches, 1);
    }

    #[test]
    fn refuses_unpaired_dimensions() {
        let a = Subcircuit::new(4, vec![0, 1]).unwrap();
        let b = Subcircuit::new(4, vec![0, 1, 2]).unwrap();
        assert_eq!(
            select_heldout_ablation(&[a, b], &[state(&[0])], 1),
            Err(RecallError::LengthMismatch)
        );
    }
}

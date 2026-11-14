//! Train-only regional subcircuit selection with independent held-out recall.
//! Region IDs are explicit annotations, not inferred biological anatomy.
use crate::subcircuit_recall::{RecallError, Subcircuit};
use crate::BooleanState;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeldoutRegionalRecall {
    pub region_id: u32,
    pub members: Vec<usize>,
    pub training_matches: usize,
    pub heldout_matches: usize,
    pub heldout_steps: usize,
}

/// Select frequent members independently inside each annotated region using TRAIN
/// only. Evaluate each frozen subcircuit on HELDOUT without fitting or changing it.
/// An empty region or one without enough training observations is not selected.
/// Both phases are bounded; results are sorted by explicit region ID.
pub fn evaluate_heldout_regional_recall(
    training: &[BooleanState],
    heldout: &[BooleanState],
    annotations: &[Option<u32>],
    minimum_training_observations: usize,
    maximum_members_per_region: usize,
    minimum_recall_members: usize,
) -> Result<Vec<HeldoutRegionalRecall>, RecallError> {
    if training.is_empty() || heldout.is_empty() {
        return Err(RecallError::EmptyHistory);
    }
    if training.len() > 1000
        || heldout.len() > 1000
        || minimum_training_observations == 0
        || maximum_members_per_region == 0
        || minimum_recall_members == 0
    {
        return Err(RecallError::ZeroBudget);
    }
    let universe = annotations.len();
    if training
        .iter()
        .chain(heldout)
        .any(|state| state.len() != universe)
    {
        return Err(RecallError::LengthMismatch);
    }
    let mut regions = BTreeMap::<u32, Vec<(usize, usize)>>::new();
    for (index, &region) in annotations.iter().enumerate() {
        if let Some(id) = region {
            regions.entry(id).or_default().push((index, 0));
        }
    }
    // Count only training activity, never held-out activity.
    let mut counts = vec![0usize; universe];
    for state in training {
        for index in state.active_indices() {
            if annotations[index].is_some() {
                counts[index] += 1;
            }
        }
    }
    let mut result = Vec::new();
    for (region_id, mut candidates) in regions {
        candidates.retain(|&(index, _)| counts[index] >= minimum_training_observations);
        candidates
            .sort_unstable_by(|a, b| counts[b.0].cmp(&counts[a.0]).then_with(|| a.0.cmp(&b.0)));
        candidates.truncate(maximum_members_per_region);
        if candidates.is_empty() {
            continue;
        }
        let circuit = Subcircuit::new(
            universe,
            candidates.into_iter().map(|(index, _)| index).collect(),
        )?;
        let training_matches = training
            .iter()
            .filter(|state| circuit.recall(state, minimum_recall_members) == Ok(true))
            .count();
        let heldout_matches = heldout
            .iter()
            .filter(|state| circuit.recall(state, minimum_recall_members) == Ok(true))
            .count();
        result.push(HeldoutRegionalRecall {
            region_id,
            members: circuit.members().to_vec(),
            training_matches,
            heldout_matches,
            heldout_steps: heldout.len(),
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(len: usize, active: &[usize]) -> BooleanState {
        let mut state = BooleanState::new(len);
        for &index in active {
            state.set(index, true).unwrap();
        }
        state
    }

    #[test]
    fn frozen_training_selection_does_not_leak_heldout_activity() {
        let labels = [Some(2), Some(2), Some(1), None];
        let train = [state(4, &[0, 2]), state(4, &[0, 2])];
        let heldout = [state(4, &[1, 2, 3]), state(4, &[1, 3])];
        let outcome = evaluate_heldout_regional_recall(&train, &heldout, &labels, 2, 1, 1).unwrap();
        assert_eq!(
            outcome,
            vec![
                HeldoutRegionalRecall {
                    region_id: 1,
                    members: vec![2],
                    training_matches: 2,
                    heldout_matches: 1,
                    heldout_steps: 2
                },
                HeldoutRegionalRecall {
                    region_id: 2,
                    members: vec![0],
                    training_matches: 2,
                    heldout_matches: 0,
                    heldout_steps: 2
                },
            ]
        );
        assert_eq!(train[0].active_count(), 2);
        assert_eq!(heldout[0].active_count(), 3);
    }

    #[test]
    fn ties_are_stable_and_missing_regions_are_not_inferred() {
        let labels = [Some(7), Some(7), None];
        let train = [state(3, &[0, 1, 2])];
        let heldout = [state(3, &[1])];
        let result = evaluate_heldout_regional_recall(&train, &heldout, &labels, 1, 1, 1).unwrap();
        assert_eq!(result[0].members, vec![0]);
        assert_eq!(result[0].heldout_matches, 0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn validates_both_partitions_and_budgets() {
        let labels = [Some(1)];
        let valid = [state(1, &[0])];
        assert_eq!(
            evaluate_heldout_regional_recall(&[], &valid, &labels, 1, 1, 1),
            Err(RecallError::EmptyHistory)
        );
        assert_eq!(
            evaluate_heldout_regional_recall(&valid, &[], &labels, 1, 1, 1),
            Err(RecallError::EmptyHistory)
        );
        assert_eq!(
            evaluate_heldout_regional_recall(&valid, &valid, &labels, 0, 1, 1),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(
            evaluate_heldout_regional_recall(&valid, &valid, &labels, 1, 0, 1),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(
            evaluate_heldout_regional_recall(&valid, &valid, &labels, 1, 1, 0),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(
            evaluate_heldout_regional_recall(&valid, &[state(2, &[0])], &labels, 1, 1, 1),
            Err(RecallError::LengthMismatch)
        );
    }
}

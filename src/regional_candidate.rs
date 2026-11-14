//! Experimental lag-one regional candidate selection, never anatomical connectivity.
use crate::BooleanState;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateError {
    InvalidBudget,
    LengthMismatch,
    TooManyRegions,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LagCandidate {
    pub source_region: u32,
    pub target_region: u32,
    pub source_exposures: u64,
    pub target_exposures: u64,
    pub joint_exposures: u64,
    pub transitions: u64,
}

/// Rank directed *hypotheses* by lag-one conditional activity, not causality.
///
/// A region is active if any explicitly annotated member is active. Candidates
/// must have at least `minimum_joint` positive pairs and strictly positive
/// lift against the target's unconditional next-tick activity. Equal precision
/// is ordered by region identifiers. No graph or state is mutated.
pub fn select_lag_candidates(
    states: &[BooleanState],
    annotations: &[Option<u32>],
    minimum_joint: u64,
    maximum_candidates: usize,
) -> Result<Vec<LagCandidate>, CandidateError> {
    if states.len() < 2 || states.len() > 1000 || annotations.is_empty()
        || minimum_joint == 0 || maximum_candidates == 0 || maximum_candidates > 4096
    {
        return Err(CandidateError::InvalidBudget);
    }
    if states.iter().any(|state| state.len() != annotations.len()) {
        return Err(CandidateError::LengthMismatch);
    }
    let mut members = BTreeMap::<u32, Vec<usize>>::new();
    for (index, &region) in annotations.iter().enumerate() {
        if let Some(id) = region {
            members.entry(id).or_default().push(index);
        }
    }
    if members.len() > 64 {
        return Err(CandidateError::TooManyRegions);
    }
    let regions: Vec<_> = members.into_iter().collect();
    let mut activity = Vec::with_capacity(states.len());
    for state in states {
        let mut row = Vec::with_capacity(regions.len());
        for (_, indices) in &regions {
            row.push(indices.iter().any(|&i| state.get(i) == Ok(true)));
        }
        activity.push(row);
    }
    let transitions = (states.len() - 1) as u64;
    let mut candidates = Vec::new();
    for (source_index, (source_id, _)) in regions.iter().enumerate() {
        for (target_index, (target_id, _)) in regions.iter().enumerate() {
            if source_id == target_id {
                continue;
            }
            let mut source = 0_u64;
            let mut target = 0_u64;
            let mut joint = 0_u64;
            for pair in activity.windows(2) {
                let a = pair[0][source_index];
                let b = pair[1][target_index];
                source += u64::from(a);
                target += u64::from(b);
                joint += u64::from(a && b);
            }
            if joint >= minimum_joint && source > 0
                && u128::from(joint) * u128::from(transitions)
                    > u128::from(source) * u128::from(target)
            {
                candidates.push(LagCandidate {
                    source_region: *source_id,
                    target_region: *target_id,
                    source_exposures: source,
                    target_exposures: target,
                    joint_exposures: joint,
                    transitions,
                });
            }
        }
    }
    candidates.sort_unstable_by(|a, b| {
        (u128::from(b.joint_exposures) * u128::from(a.source_exposures))
            .cmp(&(u128::from(a.joint_exposures) * u128::from(b.source_exposures)))
            .then_with(|| b.joint_exposures.cmp(&a.joint_exposures))
            .then_with(|| a.source_region.cmp(&b.source_region))
            .then_with(|| a.target_region.cmp(&b.target_region))
    });
    candidates.truncate(maximum_candidates);
    Ok(candidates)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(a: bool, b: bool) -> BooleanState {
        let mut s = BooleanState::new(3);
        s.set(0, a).unwrap();
        s.set(1, b).unwrap();
        s.set(2, true).unwrap(); // unannotated: never contributes
        s
    }

    #[test]
    fn selects_predictive_direction_without_inventing_edges() {
        let states = [
            state(true, false),
            state(false, true),
            state(true, false),
            state(false, true),
            state(false, false),
        ];
        let result = select_lag_candidates(&states, &[Some(10), Some(20), None], 2, 4).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!((result[0].source_region, result[0].target_region), (10, 20));
        assert_eq!((result[0].joint_exposures, result[0].source_exposures), (2, 2));
        assert_eq!(result[0].transitions, 4);
    }

    #[test]
    fn rejects_invalid_budgets_and_dimensions() {
        let states = [state(false, false), state(true, false)];
        assert_eq!(select_lag_candidates(&states, &[None], 1, 1), Err(CandidateError::LengthMismatch));
        assert_eq!(select_lag_candidates(&states, &[Some(1), None, None], 0, 1), Err(CandidateError::InvalidBudget));
        assert_eq!(select_lag_candidates(&states[..1], &[Some(1), None, None], 1, 1), Err(CandidateError::InvalidBudget));
    }

    #[test]
    fn excludes_independent_or_unobserved_regions() {
        let states = [state(false, false), state(false, false), state(false, false)];
        assert!(select_lag_candidates(&states, &[Some(1), Some(2), None], 1, 4).unwrap().is_empty());
    }
}

//! Lagged activation statistics over explicitly annotated, disjoint regions.
//! These temporal associations are not anatomical edges or causal estimates.
use crate::BooleanState;
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnnotatedRegion {
    pub label: String,
    pub members: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaggedPair {
    pub from: usize,
    pub to: usize,
    pub coactive_ticks: u64,
    pub source_ticks: u64,
    pub target_ticks: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LagError {
    EmptyTrace,
    InconsistentStateSize,
    InvalidRegion,
    DuplicateMember,
    InvalidLag,
    BudgetExceeded,
}

/// Count source activity at t and target activity at t + lag.
/// Every pair has exactly trace.len() - lag opportunities.
/// Regions and states are immutable; no graph connectivity is inferred.
pub fn observe_lagged_regions(
    trace: &[BooleanState],
    regions: &[AnnotatedRegion],
    lag: usize,
    min_active: usize,
) -> Result<Vec<LaggedPair>, LagError> {
    if trace.is_empty() {
        return Err(LagError::EmptyTrace);
    }
    if trace.len() > 1000 || regions.len() > 64 {
        return Err(LagError::BudgetExceeded);
    }
    if lag == 0 || lag >= trace.len() {
        return Err(LagError::InvalidLag);
    }
    let neurons = trace[0].len();
    if trace.iter().any(|state| state.len() != neurons) {
        return Err(LagError::InconsistentStateSize);
    }
    let mut seen = HashSet::new();
    for region in regions {
        if region.label.is_empty() || region.members.is_empty() || min_active == 0
            || min_active > region.members.len()
            || region.members.iter().any(|&member| member >= neurons)
        {
            return Err(LagError::InvalidRegion);
        }
        for &member in &region.members {
            if !seen.insert(member) {
                return Err(LagError::DuplicateMember);
            }
        }
    }
    let mut active = vec![vec![false; regions.len()]; trace.len()];
    for (tick, state) in trace.iter().enumerate() {
        for (region_index, region) in regions.iter().enumerate() {
            let count = region.members.iter().filter(|&&member| {
                // Bounds validated above.
                state.get(member) == Ok(true)
            }).count();
            active[tick][region_index] = count >= min_active;
        }
    }
    let mut result = Vec::with_capacity(regions.len() * regions.len());
    for from in 0..regions.len() {
        for to in 0..regions.len() {
            let mut pair = LaggedPair {
                from, to, coactive_ticks: 0, source_ticks: 0, target_ticks: 0,
            };
            for tick in 0..trace.len() - lag {
                let source = active[tick][from];
                let target = active[tick + lag][to];
                pair.source_ticks += u64::from(source);
                pair.target_ticks += u64::from(target);
                pair.coactive_ticks += u64::from(source && target);
            }
            result.push(pair);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state(a: bool, b: bool) -> BooleanState {
        let mut s = BooleanState::new(2);
        s.set(0, a).unwrap();
        s.set(1, b).unwrap();
        s
    }
    fn regions() -> Vec<AnnotatedRegion> {
        vec![
            AnnotatedRegion { label: "a".into(), members: vec![0] },
            AnnotatedRegion { label: "b".into(), members: vec![1] },
        ]
    }
    #[test]
    fn lag_direction_and_marginals() {
        let trace = [state(true, false), state(false, true), state(false, false)];
        let pairs = observe_lagged_regions(&trace, &regions(), 1, 1).unwrap();
        assert_eq!(pairs[1], LaggedPair {
            from: 0, to: 1, coactive_ticks: 1, source_ticks: 1, target_ticks: 1
        });
        assert_eq!(pairs[2].coactive_ticks, 0);
    }
    #[test]
    fn rejects_overlap_and_invalid_lag() {
        let trace = [state(true, false), state(false, true)];
        assert_eq!(observe_lagged_regions(&trace, &regions(), 0, 1), Err(LagError::InvalidLag));
        let mut overlapping = regions();
        overlapping[1].members = vec![0];
        assert_eq!(observe_lagged_regions(&trace, &overlapping, 1, 1), Err(LagError::DuplicateMember));
    }
    #[test]
    fn no_mutation_and_empty_region_set() {
        let trace = [state(true, false), state(false, true)];
        let original = trace.clone();
        assert!(observe_lagged_regions(&trace, &[], 1, 1).unwrap().is_empty());
        assert_eq!(trace, original);
    }
}

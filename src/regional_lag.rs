//! Lag-one regional co-activation observations for experimental directed hypotheses.
//! A temporal association is not an anatomical edge or a causal connection.
use crate::BooleanState;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LagError {
    InvalidBudget,
    LengthMismatch,
    TooManyRegions,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionalLag {
    pub source_region: u32,
    pub target_region: u32,
    /// Number of adjacent snapshot pairs with source active at t.
    pub source_exposures: u64,
    /// Number of pairs with target active at t+1.
    pub target_next_active: u64,
    /// Number of pairs satisfying both observations.
    pub joint: u64,
    /// Number of pairs with source inactive and target active at t+1.
    pub target_without_source: u64,
}

/// Measure every ordered region pair on consecutive snapshots. Regions are
/// explicitly annotated by the caller; no edges or physiology are inferred.
/// An annotated region is active when any of its members is active.
/// Results are ordered by (source_region, target_region), including self pairs.
/// Requires 2..=1000 snapshots and at most 64 distinct annotated regions.
pub fn regional_lag_one(
    states: &[BooleanState],
    annotations: &[Option<u32>],
) -> Result<Vec<RegionalLag>, LagError> {
    if !(2..=1000).contains(&states.len()) || annotations.is_empty() {
        return Err(LagError::InvalidBudget);
    }
    if states.iter().any(|s| s.len() != annotations.len()) {
        return Err(LagError::LengthMismatch);
    }
    let mut regions = BTreeMap::<u32, Vec<usize>>::new();
    for (index, region) in annotations.iter().enumerate() {
        if let Some(id) = region {
            regions.entry(*id).or_default().push(index);
        }
    }
    if regions.len() > 64 {
        return Err(LagError::TooManyRegions);
    }
    let ids: Vec<u32> = regions.keys().copied().collect();
    let members: Vec<&Vec<usize>> = regions.values().collect();
    let mut previous = vec![false; ids.len()];
    let mut output = Vec::with_capacity(ids.len() * ids.len());
    for &source in &ids {
        for &target in &ids {
            output.push(RegionalLag {
                source_region: source,
                target_region: target,
                source_exposures: 0,
                target_next_active: 0,
                joint: 0,
                target_without_source: 0,
            });
        }
    }
    for (tick, state) in states.iter().enumerate() {
        let active: Vec<bool> = members
            .iter()
            .map(|group| group.iter().any(|&index| state.get(index) == Ok(true)))
            .collect();
        if tick != 0 {
            for (source, &was_active) in previous.iter().enumerate() {
                for (target, &is_active) in active.iter().enumerate() {
                    let row = &mut output[source * ids.len() + target];
                    row.source_exposures += u64::from(was_active);
                    row.target_next_active += u64::from(is_active);
                    row.joint += u64::from(was_active && is_active);
                    row.target_without_source += u64::from(!was_active && is_active);
                }
            }
        }
        previous = active;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordered_lag_counts_are_observational_only() {
        let mut a = BooleanState::new(3);
        a.set(0, true).unwrap();
        let mut b = BooleanState::new(3);
        b.set(1, true).unwrap();
        let c = BooleanState::new(3);
        let rows = regional_lag_one(&[a, b, c], &[Some(7), Some(9), None]).unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!((rows[1].source_region, rows[1].target_region), (7, 9));
        assert_eq!(
            (
                rows[1].source_exposures,
                rows[1].target_next_active,
                rows[1].joint,
                rows[1].target_without_source
            ),
            (1, 1, 1, 0)
        );
        assert_eq!(rows[3].joint, 0);
    }
    #[test]
    fn rejects_unmatched_lengths_and_unbounded_histories() {
        let a = BooleanState::new(2);
        assert_eq!(
            regional_lag_one(std::slice::from_ref(&a), &[Some(1), None]),
            Err(LagError::InvalidBudget)
        );
        assert_eq!(
            regional_lag_one(&[a.clone(), a.clone()], &[Some(1)]),
            Err(LagError::LengthMismatch)
        );
        assert_eq!(
            regional_lag_one(&vec![a; 1001], &[Some(1), None]),
            Err(LagError::InvalidBudget)
        );
    }
    #[test]
    fn preserves_explicit_region_ids_and_excludes_unannotated_members() {
        let mut a = BooleanState::new(3);
        a.set(2, true).unwrap();
        let rows = regional_lag_one(&[a.clone(), a], &[Some(42), None, None]).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].joint, 0);
        assert_eq!(rows[0].source_region, 42);
    }
}

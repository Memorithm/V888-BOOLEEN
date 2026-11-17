//! VALID-frozen selection for sequential synthetic interventions.
//! Costs include exact cache rebuild/refresh, not only evaluation.
//! TEST/OOD are audit-only and must never be used for fitting.
use crate::fourier5_conservative_policy::Engine;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequentialCost {
    pub gates: u64,
    pub cone_gates: u64,
    pub full_ns: u64,
    pub incremental_ns: u64,
    pub cache_update_ns: u64,
    pub same_output: bool,
    pub same_decision: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequentialPolicyError {
    EmptyValidation,
    InvalidObservation,
    Overflow,
}

const CONES: usize = 16;
const CELLS: usize = 64 * CONES;

fn bucket(gates: u64, cone: u64) -> Option<usize> {
    if gates == 0 || cone > gates {
        return None;
    }
    let size = (u64::BITS - 1 - gates.leading_zeros()) as usize;
    let ratio = ((u128::from(cone) * 16) / u128::from(gates)).min(15) as usize;
    Some(size * CONES + ratio)
}

#[derive(Clone, Debug)]
pub struct SequentialCostPolicy {
    enabled: [bool; CELLS],
}

impl SequentialCostPolicy {
    /// Every VALID observation in a bucket must beat Full after cache update.
    /// No aggregate mean may hide an individual regression.
    pub fn fit_valid(rows: &[SequentialCost], min_samples: u64, margin_ns: u64)
        -> Result<Self, SequentialPolicyError>
    {
        if rows.is_empty() || min_samples == 0 {
            return Err(SequentialPolicyError::EmptyValidation);
        }
        let mut count = [0u64; CELLS];
        let mut eligible = [true; CELLS];
        for row in rows {
            let index = bucket(row.gates, row.cone_gates)
                .ok_or(SequentialPolicyError::InvalidObservation)?;
            if row.full_ns == 0 || row.incremental_ns == 0 {
                return Err(SequentialPolicyError::InvalidObservation);
            }
            count[index] = count[index].checked_add(1)
                .ok_or(SequentialPolicyError::Overflow)?;
            let total = row.incremental_ns.checked_add(row.cache_update_ns)
                .ok_or(SequentialPolicyError::Overflow)?;
            eligible[index] &= row.same_output && row.same_decision
                && total < row.full_ns
                && row.full_ns - total >= margin_ns;
        }
        let mut enabled = [false; CELLS];
        for i in 0..CELLS {
            enabled[i] = count[i] >= min_samples && eligible[i];
        }
        Ok(Self { enabled })
    }

    #[must_use]
    pub fn select(&self, gates: u64, cone_gates: u64, cache_current: bool) -> Engine {
        match bucket(gates, cone_gates) {
            Some(index) if cache_current && self.enabled[index] => Engine::Incremental,
            _ => Engine::Full,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(gates: u64, full_ns: u64, incremental_ns: u64, cache_update_ns: u64)
        -> SequentialCost
    {
        SequentialCost {
            gates, cone_gates: 4, full_ns, incremental_ns, cache_update_ns,
            same_output: true, same_decision: true,
        }
    }

    #[test]
    fn rebuild_cost_can_reverse_apparent_speedup() {
        let p = SequentialCostPolicy::fit_valid(
            &[sample(128, 100, 10, 95)], 1, 0).unwrap();
        assert_eq!(p.select(128, 4, true), Engine::Full);
    }

    #[test]
    fn separate_large_scale_and_stale_cache() {
        let p = SequentialCostPolicy::fit_valid(
            &[sample(128, 100, 10, 5)], 1, 0).unwrap();
        assert_eq!(p.select(128, 4, true), Engine::Incremental);
        assert_eq!(p.select(128, 4, false), Engine::Full);
        assert_eq!(p.select(1_562_501, 4, true), Engine::Full);
    }

    #[test]
    fn single_regression_or_decision_mismatch_disables_bucket() {
        let mut wrong = sample(128, 100, 10, 5);
        wrong.same_decision = false;
        let p = SequentialCostPolicy::fit_valid(
            &[sample(128, 100, 10, 5), wrong], 2, 0).unwrap();
        assert_eq!(p.select(128, 4, true), Engine::Full);
    }

    #[test]
    fn invalid_and_overflow_fail_closed() {
        assert!(matches!(SequentialCostPolicy::fit_valid(
            &[sample(0, 100, 10, 5)], 1, 0),
            Err(SequentialPolicyError::InvalidObservation)));
        assert!(matches!(SequentialCostPolicy::fit_valid(
            &[sample(128, 100, u64::MAX, 1)], 1, 0),
            Err(SequentialPolicyError::Overflow)));
    }
}

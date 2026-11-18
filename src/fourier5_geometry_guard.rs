//! VALID-only geometry envelope for a frozen sequential Full/Incremental policy.
//! This guard prevents a coarse logarithmic bucket from silently authorizing
//! unseen cone sizes or graph scales. TEST/OOD must never update the envelope.
use crate::fourier5_conservative_policy::Engine;
use crate::fourier5_sequential_cost_policy::{SequentialCost, SequentialCostPolicy, SequentialPolicyError};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Envelope {
    min_gates: u64,
    max_gates: u64,
    min_cone: u64,
    max_cone: u64,
    count: u64,
}

/// Conservative VALID-frozen geometry guard. The underlying measured policy
/// remains responsible for speed and oracle-decision eligibility.
#[derive(Clone, Debug)]
pub struct GeometryGuard {
    policy: SequentialCostPolicy,
    envelopes: BTreeMap<(u32, u32), Envelope>,
    min_samples: u64,
}

fn key(gates: u64, cone: u64) -> Option<(u32, u32)> {
    if gates == 0 || cone > gates {
        return None;
    }
    // Integer-only log2 bins; zero cone is a distinct regime.
    let scale = 63 - gates.leading_zeros();
    let cone_bin = if cone == 0 { 0 } else { 1 + 63 - cone.leading_zeros() };
    Some((scale, cone_bin))
}

impl GeometryGuard {
    /// Fit only from VALID rows; reject malformed timing, geometry and overflow.
    pub fn fit_valid(
        rows: &[SequentialCost],
        min_samples: u64,
        margin_ns: u64,
    ) -> Result<Self, SequentialPolicyError> {
        let policy = SequentialCostPolicy::fit_valid(rows, min_samples, margin_ns)?;
        let mut envelopes: BTreeMap<(u32, u32), Envelope> = BTreeMap::new();
        for row in rows {
            let bin = key(row.gates, row.cone_gates)
                .ok_or(SequentialPolicyError::InvalidObservation)?;
            let entry = envelopes.entry(bin).or_insert(Envelope {
                min_gates: row.gates,
                max_gates: row.gates,
                min_cone: row.cone_gates,
                max_cone: row.cone_gates,
                count: 0,
            });
            entry.count = entry.count.checked_add(1).ok_or(SequentialPolicyError::Overflow)?;
            entry.min_gates = entry.min_gates.min(row.gates);
            entry.max_gates = entry.max_gates.max(row.gates);
            entry.min_cone = entry.min_cone.min(row.cone_gates);
            entry.max_cone = entry.max_cone.max(row.cone_gates);
        }
        Ok(Self { policy, envelopes, min_samples })
    }

    /// Fail closed for unknown scales, cones, stale caches or sparse VALID bins.
    /// This is not an online learning update; TEST/OOD never changes the guard.
    #[must_use]
    pub fn select(&self, gates: u64, cone: u64, cache_current: bool) -> Engine {
        let Some(bin) = key(gates, cone) else { return Engine::Full };
        let Some(env) = self.envelopes.get(&bin) else { return Engine::Full };
        if env.count < self.min_samples
            || gates < env.min_gates || gates > env.max_gates
            || cone < env.min_cone || cone > env.max_cone
        {
            return Engine::Full;
        }
        self.policy.select(gates, cone, cache_current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(gates: u64, cone: u64) -> SequentialCost {
        SequentialCost {
            gates, cone_gates: cone, full_ns: 100,
            incremental_ns: 20, cache_update_ns: 10,
            same_output: true, same_decision: true,
        }
    }

    #[test]
    fn unseen_geometry_in_same_coarse_bucket_falls_back() {
        let guard = GeometryGuard::fit_valid(&[row(128, 4), row(160, 4)], 2, 1).unwrap();
        assert_eq!(guard.select(128, 4, true), Engine::Incremental);
        assert_eq!(guard.select(144, 4, true), Engine::Incremental);
        assert_eq!(guard.select(200, 4, true), Engine::Full);
        assert_eq!(guard.select(144, 5, true), Engine::Full);
        assert_eq!(guard.select(144, 4, false), Engine::Full);
    }

    #[test]
    fn oracle_mismatch_disables_underlying_bucket() {
        let mut wrong = row(160, 4);
        wrong.same_decision = false;
        let guard = GeometryGuard::fit_valid(&[row(128, 4), wrong], 2, 0).unwrap();
        assert_eq!(guard.select(128, 4, true), Engine::Full);
    }

    #[test]
    fn empty_invalid_and_sparse_validation_fail_closed() {
        assert!(matches!(GeometryGuard::fit_valid(&[], 1, 0), Err(SequentialPolicyError::EmptyValidation)));
        assert!(matches!(GeometryGuard::fit_valid(&[row(0, 0)], 1, 0), Err(SequentialPolicyError::InvalidObservation)));
        let guard = GeometryGuard::fit_valid(&[row(128, 0), row(160, 0)], 2, 0).unwrap();
        assert_eq!(guard.select(144, 0, true), Engine::Incremental);
        assert_eq!(guard.select(144, 1, true), Engine::Full);
    }
}

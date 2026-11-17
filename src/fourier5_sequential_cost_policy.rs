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


#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SequentialHoldoutCell {
    pub observations: u64,
    pub incremental_selected: u64,
    pub output_mismatches: u64,
    pub decision_mismatches: u64,
    pub latency_regressions: u64,
    pub total_full_ns: u128,
    pub total_selected_ns: u128,
}

/// Audit-only result. TEST/OOD must not update the frozen VALID policy.
#[derive(Clone, Debug)]
pub struct SequentialHoldoutAudit {
    pub cells: [SequentialHoldoutCell; CELLS],
}

impl SequentialCostPolicy {
    /// Evaluate a frozen predecision on disjoint TEST/OOD paired observations.
    /// Selected cost includes exact cache refresh. Full is the independent oracle.
    /// Never re-fit or enable a bucket from holdout outcomes.
    pub fn audit_holdout(
        &self,
        rows: &[SequentialCost],
        cache_current: bool,
    ) -> Result<SequentialHoldoutAudit, SequentialPolicyError> {
        let mut cells = [SequentialHoldoutCell::default(); CELLS];
        for row in rows {
            let index = bucket(row.gates, row.cone_gates)
                .ok_or(SequentialPolicyError::InvalidObservation)?;
            if row.full_ns == 0 || row.incremental_ns == 0 {
                return Err(SequentialPolicyError::InvalidObservation);
            }
            let incremental_total = row.incremental_ns.checked_add(row.cache_update_ns)
                .ok_or(SequentialPolicyError::Overflow)?;
            let selected_incremental =
                self.select(row.gates, row.cone_gates, cache_current) == Engine::Incremental;
            let cell = &mut cells[index];
            cell.observations = cell.observations.checked_add(1)
                .ok_or(SequentialPolicyError::Overflow)?;
            cell.total_full_ns = cell.total_full_ns.checked_add(u128::from(row.full_ns))
                .ok_or(SequentialPolicyError::Overflow)?;
            let selected_ns = if selected_incremental {
                cell.incremental_selected = cell.incremental_selected.checked_add(1)
                    .ok_or(SequentialPolicyError::Overflow)?;
                if !row.same_output {
                    cell.output_mismatches = cell.output_mismatches.checked_add(1)
                        .ok_or(SequentialPolicyError::Overflow)?;
                }
                if !row.same_decision {
                    cell.decision_mismatches = cell.decision_mismatches.checked_add(1)
                        .ok_or(SequentialPolicyError::Overflow)?;
                }
                if incremental_total >= row.full_ns {
                    cell.latency_regressions = cell.latency_regressions.checked_add(1)
                        .ok_or(SequentialPolicyError::Overflow)?;
                }
                incremental_total
            } else {
                row.full_ns
            };
            cell.total_selected_ns = cell.total_selected_ns.checked_add(u128::from(selected_ns))
                .ok_or(SequentialPolicyError::Overflow)?;
        }
        Ok(SequentialHoldoutAudit { cells })
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
    fn holdout_preserves_large_scale_negative_results_without_refitting() {
        let valid = [sample(128, 100, 10, 5), sample(1_562_501, 100, 10, 5)];
        let policy = SequentialCostPolicy::fit_valid(&valid, 1, 0).unwrap();
        let mut regression = sample(1_562_501, 100, 120, 10);
        regression.same_decision = false;
        let audit = policy.audit_holdout(&[sample(128, 100, 10, 5), regression], true).unwrap();
        let small = bucket(128, 4).unwrap();
        let large = bucket(1_562_501, 4).unwrap();
        assert_eq!(audit.cells[small].latency_regressions, 0);
        assert_eq!(audit.cells[large].latency_regressions, 1);
        assert_eq!(audit.cells[large].decision_mismatches, 1);
        assert_eq!(audit.cells[large].total_selected_ns, 130);
        assert_eq!(policy.select(1_562_501, 4, true), Engine::Incremental);
        let fallback = policy.audit_holdout(&[regression], false).unwrap();
        assert_eq!(fallback.cells[large].incremental_selected, 0);
        assert_eq!(fallback.cells[large].total_selected_ns, 100);
    }

    #[test]
    fn holdout_rejects_overflow_and_invalid_observations() {
        let policy = SequentialCostPolicy::fit_valid(&[sample(128, 100, 10, 5)], 1, 0).unwrap();
        assert!(matches!(policy.audit_holdout(&[sample(0, 100, 10, 5)], true),
            Err(SequentialPolicyError::InvalidObservation)));
        assert!(matches!(policy.audit_holdout(&[sample(128, 100, u64::MAX, 1)], true),
            Err(SequentialPolicyError::Overflow)));
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


impl SequentialCostPolicy {
    /// Audit a heterogeneous ordered intervention stream. Cache freshness is
    /// sampled for EACH intervention, never inferred from the first or last.
    /// The frozen VALID policy is not mutated by TEST/OOD observations.
    pub fn audit_holdout_sequence(
        &self,
        steps: &[(SequentialCost, bool)],
    ) -> Result<SequentialHoldoutAudit, SequentialPolicyError> {
        let mut cells = [SequentialHoldoutCell::default(); CELLS];
        for &(row, cache_current) in steps {
            // Stream each step into its bucket directly. In particular, do not
            // construct a full 1024-cell audit for every intervention.
            let index = bucket(row.gates, row.cone_gates)
                .ok_or(SequentialPolicyError::InvalidObservation)?;
            if row.full_ns == 0 || row.incremental_ns == 0 {
                return Err(SequentialPolicyError::InvalidObservation);
            }
            let incremental_total = row.incremental_ns.checked_add(row.cache_update_ns)
                .ok_or(SequentialPolicyError::Overflow)?;
            let selected = self.select(row.gates, row.cone_gates, cache_current)
                == Engine::Incremental;
            let dst = &mut cells[index];
            dst.observations = dst.observations.checked_add(1)
                .ok_or(SequentialPolicyError::Overflow)?;
            dst.total_full_ns = dst.total_full_ns.checked_add(u128::from(row.full_ns))
                .ok_or(SequentialPolicyError::Overflow)?;
            let selected_ns = if selected {
                dst.incremental_selected = dst.incremental_selected.checked_add(1)
                    .ok_or(SequentialPolicyError::Overflow)?;
                if !row.same_output {
                    dst.output_mismatches = dst.output_mismatches.checked_add(1)
                        .ok_or(SequentialPolicyError::Overflow)?;
                }
                if !row.same_decision {
                    dst.decision_mismatches = dst.decision_mismatches.checked_add(1)
                        .ok_or(SequentialPolicyError::Overflow)?;
                }
                if incremental_total >= row.full_ns {
                    dst.latency_regressions = dst.latency_regressions.checked_add(1)
                        .ok_or(SequentialPolicyError::Overflow)?;
                }
                incremental_total
            } else {
                row.full_ns
            };
            dst.total_selected_ns = dst.total_selected_ns.checked_add(u128::from(selected_ns))
                .ok_or(SequentialPolicyError::Overflow)?;
        }
        Ok(SequentialHoldoutAudit { cells })
    }
}

#[cfg(test)]
mod heterogeneous_sequence_tests {
    use super::*;

    #[test]
    fn per_intervention_cache_freshness_prevents_false_incremental_selection() {
        let valid = [SequentialCost {
            gates: 128, cone_gates: 4, full_ns: 100,
            incremental_ns: 10, cache_update_ns: 5,
            same_output: true, same_decision: true,
        }];
        let policy = SequentialCostPolicy::fit_valid(&valid, 1, 0).unwrap();
        let mut divergent = valid[0];
        divergent.same_output = false;
        divergent.same_decision = false;
        divergent.incremental_ns = 120;
        let steps = [(valid[0], true), (divergent, false), (divergent, true)];
        let audit = policy.audit_holdout_sequence(&steps).unwrap();
        let cell = audit.cells[bucket(128, 4).unwrap()];
        assert_eq!(cell.observations, 3);
        assert_eq!(cell.incremental_selected, 2);
        assert_eq!(cell.output_mismatches, 1);
        assert_eq!(cell.decision_mismatches, 1);
        assert_eq!(cell.latency_regressions, 1);
        assert_eq!(cell.total_full_ns, 300);
        assert_eq!(cell.total_selected_ns, 240);
        assert_eq!(policy.select(128, 4, true), Engine::Incremental);
    }

    #[test]
    fn sequence_rejects_invalid_step_without_modifying_policy() {
        let row = SequentialCost {
            gates: 128, cone_gates: 4, full_ns: 100,
            incremental_ns: 10, cache_update_ns: 5,
            same_output: true, same_decision: true,
        };
        let policy = SequentialCostPolicy::fit_valid(&[row], 1, 0).unwrap();
        let mut invalid = row;
        invalid.gates = 0;
        assert_eq!(
            policy.audit_holdout_sequence(&[(row, true), (invalid, false)]).unwrap_err(),
            SequentialPolicyError::InvalidObservation
        );
        assert_eq!(policy.select(128, 4, true), Engine::Incremental);
    }
}

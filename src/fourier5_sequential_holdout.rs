//! Held-out, non-calibrating sequential audit for a frozen synthetic dispatch policy.
//! TEST/OOD are never used to update the policy. Durations must include cache refresh.
use crate::fourier5_conservative_policy::{Engine, FrozenPolicy};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequentialStep {
    pub gates: u64,
    pub cone_gates: u64,
    pub accepted: bool,
    pub full_ns: u64,
    pub incremental_ns: u64,
    pub same_output: bool,
    pub same_decision: bool,
    /// Process-wide peak resident memory, not per-step allocation.
    pub vmhwm_kib: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditError {
    Empty,
    InvalidStep,
    Overflow,
    OracleMismatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequentialAudit {
    pub steps: u64,
    pub accepted: u64,
    pub incremental_selected: u64,
    pub full_total_ns: u128,
    pub selected_total_ns: u128,
    pub incremental_regressions: u64,
    pub worst_regression_ns: u64,
    pub max_vmhwm_kib: u64,
}

/// Audit one disjoint held-out sequence, preserving order and counting each
/// accepted cache refresh in the caller-supplied timings. A mismatched oracle
/// fails the whole trace rather than silently reporting a speedup.
pub fn audit_sequence(
    policy: &FrozenPolicy,
    steps: &[SequentialStep],
) -> Result<SequentialAudit, AuditError> {
    if steps.is_empty() {
        return Err(AuditError::Empty);
    }
    let mut result = SequentialAudit {
        steps: 0,
        accepted: 0,
        incremental_selected: 0,
        full_total_ns: 0,
        selected_total_ns: 0,
        incremental_regressions: 0,
        worst_regression_ns: 0,
        max_vmhwm_kib: 0,
    };
    for step in steps {
        if step.gates == 0 || step.cone_gates > step.gates
            || step.full_ns == 0 || step.incremental_ns == 0
        {
            return Err(AuditError::InvalidStep);
        }
        if !step.same_output || !step.same_decision {
            return Err(AuditError::OracleMismatch);
        }
        result.steps = result.steps.checked_add(1).ok_or(AuditError::Overflow)?;
        if step.accepted {
            result.accepted = result.accepted.checked_add(1).ok_or(AuditError::Overflow)?;
        }
        result.full_total_ns = result.full_total_ns
            .checked_add(u128::from(step.full_ns)).ok_or(AuditError::Overflow)?;
        let selected = if policy.select(step.gates, step.cone_gates) == Engine::Incremental {
            result.incremental_selected = result.incremental_selected
                .checked_add(1).ok_or(AuditError::Overflow)?;
            if step.incremental_ns > step.full_ns {
                result.incremental_regressions = result.incremental_regressions
                    .checked_add(1).ok_or(AuditError::Overflow)?;
                result.worst_regression_ns = result.worst_regression_ns
                    .max(step.incremental_ns - step.full_ns);
            }
            step.incremental_ns
        } else {
            step.full_ns
        };
        result.selected_total_ns = result.selected_total_ns
            .checked_add(u128::from(selected)).ok_or(AuditError::Overflow)?;
        result.max_vmhwm_kib = result.max_vmhwm_kib.max(step.vmhwm_kib);
    }
    Ok(result)
}

impl SequentialAudit {
    /// Deployment gate: no decision mismatch, no per-step regression, and
    /// strictly lower cumulative wall time on the held-out trace.
    #[must_use]
    pub fn passes_strict_gate(&self) -> bool {
        self.incremental_selected > 0
            && self.incremental_regressions == 0
            && self.selected_total_ns < self.full_total_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5_conservative_policy::PairedObservation;

    fn policy() -> FrozenPolicy {
        FrozenPolicy::fit_valid(&[PairedObservation {
            gates: 100, cone_gates: 2, full_ns: 100,
            incremental_ns: 20, same_output: true, same_decision: true,
        }], 1).unwrap()
    }
    fn step(cone: u64, full: u64, inc: u64) -> SequentialStep {
        SequentialStep {
            gates: 100, cone_gates: cone, accepted: true,
            full_ns: full, incremental_ns: inc,
            same_output: true, same_decision: true, vmhwm_kib: 2048,
        }
    }
    #[test]
    fn heterogeneous_sequence_matches_independent_scalar_oracle() {
        let steps = [step(2, 100, 20), step(80, 50, 90), step(3, 70, 30)];
        let p = policy();
        let result = audit_sequence(&p, &steps).unwrap();
        let oracle: u128 = steps.iter().map(|s| {
            u128::from(if s.cone_gates < 7 { s.incremental_ns } else { s.full_ns })
        }).sum();
        assert_eq!(result.full_total_ns, 220);
        assert_eq!(result.selected_total_ns, oracle);
        assert_eq!(result.incremental_selected, 2);
        assert!(result.passes_strict_gate());
    }
    #[test]
    fn slower_incremental_step_fails_even_when_aggregate_wins() {
        let result = audit_sequence(&policy(), &[step(2, 1000, 1), step(2, 100, 101)]).unwrap();
        assert_eq!(result.incremental_regressions, 1);
        assert_eq!(result.worst_regression_ns, 1);
        assert!(!result.passes_strict_gate());
    }
    #[test]
    fn oracle_mismatch_and_invalid_timing_fail_closed() {
        let mut s = step(2, 100, 20);
        s.same_decision = false;
        assert_eq!(audit_sequence(&policy(), &[s]), Err(AuditError::OracleMismatch));
        s.same_decision = true;
        s.incremental_ns = 0;
        assert_eq!(audit_sequence(&policy(), &[s]), Err(AuditError::InvalidStep));
        assert_eq!(audit_sequence(&policy(), &[]), Err(AuditError::Empty));
    }
    #[test]
    fn rejected_steps_are_counted_without_cache_revision_assumption() {
        let mut s = step(80, 100, 20);
        s.accepted = false;
        s.vmhwm_kib = 8192;
        let result = audit_sequence(&policy(), &[s]).unwrap();
        assert_eq!(result.accepted, 0);
        assert_eq!(result.max_vmhwm_kib, 8192);
        assert_eq!(result.selected_total_ns, 100);
        assert!(!result.passes_strict_gate());
    }
}

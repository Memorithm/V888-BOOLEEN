//! Conservative runtime admission for measured heterogeneous sequential probes.
//! A frozen VALID policy is an input, never trained from TEST/OOD observations.
//! These checks do not infer biological connectivity or learning performance.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequentialAdmission {
    pub full_ns: u64,
    pub incremental_ns: u64,
    pub cache_refresh_ns: u64,
    pub vm_hwm_kib: Option<u64>,
    pub memory_ceiling_kib: u64,
    pub same_output: bool,
    pub same_decision: bool,
    pub cache_epoch_matches: bool,
    pub valid_frozen_admission: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionDecision {
    FullPolicy,
    FullOracleMismatch,
    FullStaleCache,
    FullMemoryUnknown,
    FullMemoryExceeded,
    FullCostRegression,
    Incremental,
}

/// Fail closed on missing measurements, stale epochs, or any measured regression.
/// A tie is Full; costs include the exact cache refresh measured for this step.
#[must_use]
pub fn admit_sequential_step(step: SequentialAdmission) -> AdmissionDecision {
    if !step.valid_frozen_admission {
        return AdmissionDecision::FullPolicy;
    }
    if !step.same_output || !step.same_decision {
        return AdmissionDecision::FullOracleMismatch;
    }
    if !step.cache_epoch_matches {
        return AdmissionDecision::FullStaleCache;
    }
    let Some(memory) = step.vm_hwm_kib else {
        return AdmissionDecision::FullMemoryUnknown;
    };
    if memory > step.memory_ceiling_kib {
        return AdmissionDecision::FullMemoryExceeded;
    }
    if u128::from(step.incremental_ns) + u128::from(step.cache_refresh_ns)
        >= u128::from(step.full_ns)
    {
        return AdmissionDecision::FullCostRegression;
    }
    AdmissionDecision::Incremental
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdmissionTotals {
    pub full: u64,
    pub incremental: u64,
    pub selected_cost_ns: u128,
    pub full_reference_ns: u128,
    pub peak_vm_hwm_kib: Option<u64>,
}

/// Offline accounting, not measured production throughput. Every step is
/// individually checked; no aggregate gain can hide a local regression.
#[must_use]
pub fn audit_sequential_steps(steps: &[SequentialAdmission]) -> Option<AdmissionTotals> {
    let mut totals = AdmissionTotals::default();
    for &step in steps {
        totals.full_reference_ns = totals
            .full_reference_ns
            .checked_add(u128::from(step.full_ns))?;
        totals.peak_vm_hwm_kib = match (totals.peak_vm_hwm_kib, step.vm_hwm_kib) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (None, value) | (value, None) => value,
        };
        if admit_sequential_step(step) == AdmissionDecision::Incremental {
            totals.incremental = totals.incremental.checked_add(1)?;
            totals.selected_cost_ns = totals.selected_cost_ns.checked_add(
                u128::from(step.incremental_ns) + u128::from(step.cache_refresh_ns),
            )?;
        } else {
            totals.full = totals.full.checked_add(1)?;
            totals.selected_cost_ns = totals
                .selected_cost_ns
                .checked_add(u128::from(step.full_ns))?;
        }
    }
    Some(totals)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline() -> SequentialAdmission {
        SequentialAdmission {
            full_ns: 100,
            incremental_ns: 40,
            cache_refresh_ns: 20,
            vm_hwm_kib: Some(200),
            memory_ceiling_kib: 300,
            same_output: true,
            same_decision: true,
            cache_epoch_matches: true,
            valid_frozen_admission: true,
        }
    }

    #[test]
    fn exhaustive_cost_oracle_with_cache_refresh() {
        for full in 0..32 {
            for incremental in 0..32 {
                for cache in 0..32 {
                    let step = SequentialAdmission {
                        full_ns: full,
                        incremental_ns: incremental,
                        cache_refresh_ns: cache,
                        ..baseline()
                    };
                    let expected = incremental + cache < full;
                    assert_eq!(
                        admit_sequential_step(step) == AdmissionDecision::Incremental,
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn fail_closed_on_every_guard_and_overflow_boundary() {
        let mut step = baseline();
        assert_eq!(admit_sequential_step(step), AdmissionDecision::Incremental);
        step.same_decision = false;
        assert_eq!(admit_sequential_step(step), AdmissionDecision::FullOracleMismatch);
        step = baseline();
        step.cache_epoch_matches = false;
        assert_eq!(admit_sequential_step(step), AdmissionDecision::FullStaleCache);
        step = baseline();
        step.vm_hwm_kib = None;
        assert_eq!(admit_sequential_step(step), AdmissionDecision::FullMemoryUnknown);
        step = baseline();
        step.vm_hwm_kib = Some(301);
        assert_eq!(admit_sequential_step(step), AdmissionDecision::FullMemoryExceeded);
        step = baseline();
        step.valid_frozen_admission = false;
        assert_eq!(admit_sequential_step(step), AdmissionDecision::FullPolicy);
        step = baseline();
        step.full_ns = u64::MAX;
        step.incremental_ns = u64::MAX;
        step.cache_refresh_ns = u64::MAX;
        assert_eq!(admit_sequential_step(step), AdmissionDecision::FullCostRegression);
    }

    #[test]
    fn heterogeneous_audit_never_credits_a_regression() {
        let good = baseline();
        let slow = SequentialAdmission {
            incremental_ns: 90,
            ..good
        };
        let bad = SequentialAdmission {
            same_output: false,
            ..good
        };
        let result = audit_sequential_steps(&[good, slow, bad]).unwrap();
        assert_eq!(result.incremental, 1);
        assert_eq!(result.full, 2);
        assert_eq!(result.full_reference_ns, 300);
        assert_eq!(result.selected_cost_ns, 260);
        assert_eq!(result.peak_vm_hwm_kib, Some(200));
    }
}

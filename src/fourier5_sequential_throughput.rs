//! Fail-closed per-class sequential throughput admission for synthetic workloads.
//! Only predeclared VALID class assignments may calibrate this policy.
//! A measured speedup is not evidence of biological fidelity or learning.
use crate::fourier5_sequential_workload_audit::{StepMeasurement, StepVerdict};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThroughputChoice {
    Full,
    Incremental,
}

/// Each independent VALID workload group supplies one complete paired sequence.
/// Durations must include cache maintenance and attempted (including rejected)
/// interventions. VmHWM is a process lifetime peak, not per-step allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataSplit {
    Train,
    Valid,
    Test,
    Ood,
}

pub struct ValidSequence<'a> {
    /// Explicit provenance: only VALID may influence dispatch calibration.
    pub split: DataSplit,
    pub group_id: u64,
    pub steps: &'a [StepMeasurement],
}

/// Require at least two independent groups, no failed oracle comparisons,
/// positive Full timings, strict per-step speedup and a strict aggregate
/// saving margin in *each* group. This cannot mask a slow minority class.
/// The caller is responsible for TRAIN/VALID/TEST/OOD provenance.
#[must_use]
pub fn choose_sequential_throughput(
    groups: &[ValidSequence<'_>],
    minimum_groups: usize,
    minimum_steps_per_group: usize,
    minimum_saving_percent: u8,
    max_vmhwm_kib: u64,
) -> ThroughputChoice {
    if minimum_groups < 2
        || groups.len() < minimum_groups
        || minimum_steps_per_group == 0
        || minimum_saving_percent > 100
        || max_vmhwm_kib == 0
    {
        return ThroughputChoice::Full;
    }
    let mut seen = std::collections::HashSet::new();
    for group in groups {
        if group.split != DataSplit::Valid
            || !seen.insert(group.group_id)
            || group.steps.len() < minimum_steps_per_group
        {
            return ThroughputChoice::Full;
        }
        let mut full = 0_u128;
        let mut incremental = 0_u128;
        for step in group.steps {
            if step.verdict != StepVerdict::Accepted
                || step
                    .peak_rss_kib
                    .is_none_or(|peak| peak == 0 || peak > max_vmhwm_kib)
            {
                return ThroughputChoice::Full;
            }
            let f = step.full.as_nanos();
            let i = step.incremental_with_refresh.as_nanos();
            // A fast majority must not conceal a marginal or regressing step.
            // Apply the predeclared saving threshold to every intervention,
            // including its exact cache-refresh cost.
            let Some(scaled_incremental) = i.checked_mul(100) else {
                return ThroughputChoice::Full;
            };
            let Some(scaled_full) = f.checked_mul(u128::from(100 - minimum_saving_percent)) else {
                return ThroughputChoice::Full;
            };
            // A zero-duration sample is below timer resolution, not proof of speedup.
            // Reject it rather than calibrating dispatch on an unmeasured cost.
            if f == 0 || i == 0 || scaled_incremental >= scaled_full {
                return ThroughputChoice::Full;
            }
            let Some(next_full) = full.checked_add(f) else {
                return ThroughputChoice::Full;
            };
            let Some(next_incremental) = incremental.checked_add(i) else {
                return ThroughputChoice::Full;
            };
            full = next_full;
            incremental = next_incremental;
        }
        let Some(lhs) = incremental.checked_mul(100) else {
            return ThroughputChoice::Full;
        };
        let Some(rhs) = full.checked_mul(u128::from(100 - minimum_saving_percent)) else {
            return ThroughputChoice::Full;
        };
        if lhs >= rhs {
            return ThroughputChoice::Full;
        }
    }
    ThroughputChoice::Incremental
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn row(
        full: u64,
        incremental: u64,
        verdict: StepVerdict,
        peak: Option<u64>,
    ) -> StepMeasurement {
        StepMeasurement {
            verdict,
            full: Duration::from_nanos(full),
            incremental_with_refresh: Duration::from_nanos(incremental),
            peak_rss_kib: peak,
        }
    }

    #[test]
    fn independent_groups_with_exact_states_are_admitted() {
        let a = [row(100, 20, StepVerdict::Accepted, Some(1024))];
        let b = [row(200, 40, StepVerdict::Accepted, Some(2048))];
        let groups = [
            ValidSequence {
                split: DataSplit::Valid,
                group_id: 1,
                steps: &a,
            },
            ValidSequence {
                split: DataSplit::Valid,
                group_id: 2,
                steps: &b,
            },
        ];
        assert_eq!(
            choose_sequential_throughput(&groups, 2, 1, 20, 4096),
            ThroughputChoice::Incremental
        );
    }

    #[test]
    fn independent_scalar_oracle_rejects_slow_minority_group() {
        let a = [row(1000, 1, StepVerdict::Accepted, Some(1024))];
        let b = [row(100, 90, StepVerdict::Accepted, Some(1024))];
        // Even though pooled savings exceed 20%, group b saves only 10%.
        assert!(90_u128 * 100 >= 100_u128 * 80);
        let groups = [
            ValidSequence {
                split: DataSplit::Valid,
                group_id: 1,
                steps: &a,
            },
            ValidSequence {
                split: DataSplit::Valid,
                group_id: 2,
                steps: &b,
            },
        ];
        assert_eq!(
            choose_sequential_throughput(&groups, 2, 1, 20, 4096),
            ThroughputChoice::Full
        );
    }

    #[test]
    fn duplicate_groups_and_missing_peak_fail_closed() {
        let a = [row(100, 1, StepVerdict::Accepted, Some(1024))];
        let b = [row(100, 1, StepVerdict::Accepted, None)];
        let duplicate = [
            ValidSequence {
                split: DataSplit::Valid,
                group_id: 7,
                steps: &a,
            },
            ValidSequence {
                split: DataSplit::Valid,
                group_id: 7,
                steps: &a,
            },
        ];
        assert_eq!(
            choose_sequential_throughput(&duplicate, 2, 1, 20, 4096),
            ThroughputChoice::Full
        );
        let missing = [
            ValidSequence {
                split: DataSplit::Valid,
                group_id: 7,
                steps: &a,
            },
            ValidSequence {
                split: DataSplit::Valid,
                group_id: 8,
                steps: &b,
            },
        ];
        assert_eq!(
            choose_sequential_throughput(&missing, 2, 1, 20, 4096),
            ThroughputChoice::Full
        );
    }

    #[test]
    fn oracle_mismatch_or_memory_ceiling_rejects() {
        let a = [row(100, 1, StepVerdict::Mismatch, Some(1024))];
        let b = [row(100, 1, StepVerdict::Accepted, Some(4097))];
        let groups = [
            ValidSequence {
                split: DataSplit::Valid,
                group_id: 1,
                steps: &a,
            },
            ValidSequence {
                split: DataSplit::Valid,
                group_id: 2,
                steps: &b,
            },
        ];
        assert_eq!(
            choose_sequential_throughput(&groups, 2, 1, 20, 4096),
            ThroughputChoice::Full
        );
    }
    #[test]
    fn independent_step_oracle_rejects_masked_slow_intervention() {
        // Each group saves >20% overall, but its second step saves only 1%.
        // A scalar oracle rejects 99 * 100 >= 100 * (100 - 20).
        assert!(99_u128 * 100 >= 100_u128 * 80);
        let a = [
            row(1000, 1, StepVerdict::Accepted, Some(1024)),
            row(100, 99, StepVerdict::Accepted, Some(1024)),
        ];
        let b = [
            row(1000, 1, StepVerdict::Accepted, Some(1024)),
            row(100, 99, StepVerdict::Accepted, Some(1024)),
        ];
        let groups = [
            ValidSequence {
                split: DataSplit::Valid,
                group_id: 1,
                steps: &a,
            },
            ValidSequence {
                split: DataSplit::Valid,
                group_id: 2,
                steps: &b,
            },
        ];
        assert_eq!(
            choose_sequential_throughput(&groups, 2, 2, 20, 4096),
            ThroughputChoice::Full
        );
    }

    #[test]
    fn independent_step_oracle_accepts_strict_margin_per_intervention() {
        let a = [
            row(100, 50, StepVerdict::Accepted, Some(1024)),
            row(100, 60, StepVerdict::Accepted, Some(1024)),
        ];
        let b = [
            row(200, 100, StepVerdict::Accepted, Some(2048)),
            row(200, 120, StepVerdict::Accepted, Some(2048)),
        ];
        let groups = [
            ValidSequence {
                split: DataSplit::Valid,
                group_id: 1,
                steps: &a,
            },
            ValidSequence {
                split: DataSplit::Valid,
                group_id: 2,
                steps: &b,
            },
        ];
        assert_eq!(
            choose_sequential_throughput(&groups, 2, 2, 20, 4096),
            ThroughputChoice::Incremental
        );
    }

    #[test]
    fn zero_duration_incremental_sample_is_not_a_measured_speedup() {
        // Independent scalar oracle: no positive elapsed time was measured.
        let a = [row(100, 0, StepVerdict::Accepted, Some(1024))];
        let b = [row(100, 50, StepVerdict::Accepted, Some(1024))];
        assert_eq!(a[0].incremental_with_refresh.as_nanos(), 0);
        let groups = [
            ValidSequence {
                split: DataSplit::Valid,
                group_id: 1,
                steps: &a,
            },
            ValidSequence {
                split: DataSplit::Valid,
                group_id: 2,
                steps: &b,
            },
        ];
        assert_eq!(
            choose_sequential_throughput(&groups, 2, 1, 20, 4096),
            ThroughputChoice::Full
        );
    }

    #[test]
    fn disjoint_test_and_ood_groups_never_calibrate_dispatch() {
        let a = [row(100, 1, StepVerdict::Accepted, Some(1024))];
        let b = [row(100, 1, StepVerdict::Accepted, Some(1024))];
        for forbidden in [DataSplit::Train, DataSplit::Test, DataSplit::Ood] {
            let groups = [
                ValidSequence {
                    split: DataSplit::Valid,
                    group_id: 1,
                    steps: &a,
                },
                ValidSequence {
                    split: forbidden,
                    group_id: 2,
                    steps: &b,
                },
            ];
            assert_eq!(
                choose_sequential_throughput(&groups, 2, 1, 20, 4096),
                ThroughputChoice::Full
            );
        }
    }
}

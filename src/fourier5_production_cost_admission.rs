//! Explicitly separated production-cost calibration for sequential synthetic workloads.
//! Verified transaction timings include Full and must NEVER enter this selector.
use std::collections::HashSet;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Train,
    Valid,
    Test,
    Ood,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    Full,
    Incremental,
}

/// Only timings from paired candidate execution (including cache refresh), not
/// Full+candidate+oracle verification, are accepted by this type.
#[derive(Clone, Copy, Debug)]
pub struct PairedProductionCost {
    pub full: Duration,
    pub incremental_with_cache_refresh: Duration,
    pub exact_state_and_cache: bool,
    pub vmhwm_kib: Option<u64>,
}

pub struct ValidationGroup<'a> {
    pub split: Split,
    pub id: u64,
    pub samples: &'a [PairedProductionCost],
}

/// Conservative VALID-only policy. Every step of every independent group must
/// beat Full by the configured strict margin. No pooled minority regressions.
/// This policy does not claim that caller-supplied timing provenance is genuine.
#[must_use]
pub fn select(
    groups: &[ValidationGroup<'_>],
    min_groups: usize,
    min_steps: usize,
    saving_percent: u8,
    peak_limit_kib: u64,
) -> Decision {
    if min_groups < 2
        || groups.len() < min_groups
        || min_steps == 0
        || saving_percent >= 100
        || peak_limit_kib == 0
    {
        return Decision::Full;
    }
    let mut ids = HashSet::new();
    for group in groups {
        if group.split != Split::Valid || !ids.insert(group.id) || group.samples.len() < min_steps {
            return Decision::Full;
        }
        for sample in group.samples {
            let full = sample.full.as_nanos();
            let incremental = sample.incremental_with_cache_refresh.as_nanos();
            let Some(left) = incremental.checked_mul(100) else {
                return Decision::Full;
            };
            let Some(right) = full.checked_mul(u128::from(100 - saving_percent)) else {
                return Decision::Full;
            };
            if !sample.exact_state_and_cache
                || full == 0
                || incremental == 0
                || left >= right
                || !matches!(sample.vmhwm_kib, Some(peak) if peak > 0 && peak <= peak_limit_kib)
            {
                return Decision::Full;
            }
        }
    }
    Decision::Incremental
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cost(full: u64, incremental: u64) -> PairedProductionCost {
        PairedProductionCost {
            full: Duration::from_nanos(full),
            incremental_with_cache_refresh: Duration::from_nanos(incremental),
            exact_state_and_cache: true,
            vmhwm_kib: Some(1024),
        }
    }
    #[test]
    fn paired_valid_groups_select_incremental() {
        let a = [cost(100, 50)];
        let b = [cost(200, 80)];
        let groups = [
            ValidationGroup {
                split: Split::Valid,
                id: 1,
                samples: &a,
            },
            ValidationGroup {
                split: Split::Valid,
                id: 2,
                samples: &b,
            },
        ];
        assert_eq!(select(&groups, 2, 1, 20, 4096), Decision::Incremental);
    }
    #[test]
    fn slow_minority_is_not_hidden_by_fast_group() {
        let a = [cost(10000, 1)];
        let b = [cost(100, 90)];
        let groups = [
            ValidationGroup {
                split: Split::Valid,
                id: 1,
                samples: &a,
            },
            ValidationGroup {
                split: Split::Valid,
                id: 2,
                samples: &b,
            },
        ];
        assert_eq!(select(&groups, 2, 1, 20, 4096), Decision::Full);
    }
    #[test]
    fn provenance_mismatch_and_missing_memory_fail_closed() {
        let a = [cost(100, 10)];
        let b = [cost(100, 10)];
        let mut groups = [
            ValidationGroup {
                split: Split::Valid,
                id: 1,
                samples: &a,
            },
            ValidationGroup {
                split: Split::Test,
                id: 2,
                samples: &b,
            },
        ];
        assert_eq!(select(&groups, 2, 1, 20, 4096), Decision::Full);
        groups[1].split = Split::Valid;
        let missing = [PairedProductionCost {
            vmhwm_kib: None,
            ..cost(100, 10)
        }];
        let groups_missing = [
            ValidationGroup {
                split: Split::Valid,
                id: 1,
                samples: &a,
            },
            ValidationGroup {
                split: Split::Valid,
                id: 2,
                samples: &missing,
            },
        ];
        assert_eq!(select(&groups_missing, 2, 1, 20, 4096), Decision::Full);
    }
}

/// A measured workload stratum. Buckets are declared before looking at VALID
/// timings; an absent bucket is never silently represented by a faster one.
pub struct WorkloadStratum<'a> {
    pub cone_bucket: usize,
    pub groups: &'a [ValidationGroup<'a>],
}

/// Admit Incremental only when every predeclared cone-size bucket has enough
/// disjoint VALID groups and each observed intervention meets the same strict
/// production-cost and exactness requirements as `select`.
///
/// `required_buckets` is a caller-frozen workload contract, not inferred from
/// the validation results. Empty, duplicate, missing or out-of-range buckets
/// fail closed. TEST and OOD must never be passed for calibration.
#[must_use]
pub fn select_stratified(
    strata: &[WorkloadStratum<'_>],
    required_buckets: usize,
    min_groups_per_bucket: usize,
    min_steps_per_group: usize,
    saving_percent: u8,
    peak_limit_kib: u64,
) -> Decision {
    if required_buckets == 0 || strata.len() != required_buckets {
        return Decision::Full;
    }
    let mut covered = vec![false; required_buckets];
    let mut all_group_ids = HashSet::new();
    for stratum in strata {
        let Some(seen) = covered.get_mut(stratum.cone_bucket) else {
            return Decision::Full;
        };
        if *seen {
            return Decision::Full;
        }
        *seen = true;
        if select(
            stratum.groups,
            min_groups_per_bucket,
            min_steps_per_group,
            saving_percent,
            peak_limit_kib,
        ) != Decision::Incremental
        {
            return Decision::Full;
        }
        // Disjointness is global, not merely within individual cone buckets.
        for group in stratum.groups {
            if !all_group_ids.insert(group.id) {
                return Decision::Full;
            }
        }
    }
    if covered.iter().all(|seen| *seen) {
        Decision::Incremental
    } else {
        Decision::Full
    }
}

#[cfg(test)]
mod stratified_tests {
    use super::*;

    fn sample(full: u64, incremental: u64) -> PairedProductionCost {
        PairedProductionCost {
            full: Duration::from_nanos(full),
            incremental_with_cache_refresh: Duration::from_nanos(incremental),
            exact_state_and_cache: true,
            vmhwm_kib: Some(512),
        }
    }

    #[test]
    fn heterogeneous_buckets_require_all_cones() {
        let fast = [sample(100, 40)];
        let slow = [sample(100, 95)];
        let a = [
            ValidationGroup {
                split: Split::Valid,
                id: 1,
                samples: &fast,
            },
            ValidationGroup {
                split: Split::Valid,
                id: 2,
                samples: &fast,
            },
        ];
        let b = [
            ValidationGroup {
                split: Split::Valid,
                id: 3,
                samples: &slow,
            },
            ValidationGroup {
                split: Split::Valid,
                id: 4,
                samples: &slow,
            },
        ];
        let strata = [
            WorkloadStratum {
                cone_bucket: 0,
                groups: &a,
            },
            WorkloadStratum {
                cone_bucket: 1,
                groups: &b,
            },
        ];
        assert_eq!(
            select_stratified(&strata, 2, 2, 1, 10, 1024),
            Decision::Full
        );
        assert_eq!(
            select_stratified(&strata[..1], 2, 2, 1, 10, 1024),
            Decision::Full
        );
    }

    #[test]
    fn disjoint_fast_buckets_admit_but_duplicate_ids_do_not() {
        let fast = [sample(100, 40)];
        let a = [
            ValidationGroup {
                split: Split::Valid,
                id: 1,
                samples: &fast,
            },
            ValidationGroup {
                split: Split::Valid,
                id: 2,
                samples: &fast,
            },
        ];
        let b = [
            ValidationGroup {
                split: Split::Valid,
                id: 3,
                samples: &fast,
            },
            ValidationGroup {
                split: Split::Valid,
                id: 4,
                samples: &fast,
            },
        ];
        let strata = [
            WorkloadStratum {
                cone_bucket: 0,
                groups: &a,
            },
            WorkloadStratum {
                cone_bucket: 1,
                groups: &b,
            },
        ];
        assert_eq!(
            select_stratified(&strata, 2, 2, 1, 10, 1024),
            Decision::Incremental
        );
        let duplicated = [
            ValidationGroup {
                split: Split::Valid,
                id: 1,
                samples: &fast,
            },
            ValidationGroup {
                split: Split::Valid,
                id: 4,
                samples: &fast,
            },
        ];
        let invalid = [
            WorkloadStratum {
                cone_bucket: 0,
                groups: &a,
            },
            WorkloadStratum {
                cone_bucket: 1,
                groups: &duplicated,
            },
        ];
        assert_eq!(
            select_stratified(&invalid, 2, 2, 1, 10, 1024),
            Decision::Full
        );
    }
}

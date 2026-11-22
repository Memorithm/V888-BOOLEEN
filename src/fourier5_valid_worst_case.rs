//! VALID-frozen, worst-case cost admission for heterogeneous synthetic cones.
//! Training/holdout observations cannot update the policy. Costs include cache refresh.
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partition { Train, Valid, Test, Ood }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairedSample {
    pub partition: Partition,
    pub group: u64,
    pub cone_bucket: u32,
    pub full_ns: u64,
    pub incremental_ns: u64,
    pub refresh_ns: u64,
    pub peak_kib: u64,
    pub oracle_equal: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Admission {
    pub cone_bucket: u32,
    pub groups: usize,
    pub worst_full_ns: u64,
    pub worst_incremental_total_ns: u64,
    pub max_peak_kib: u64,
    pub allowed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyError { DuplicateGroup, Overflow, EmptyValidation }

/// Freeze a conservative policy using only disjoint VALID groups.
/// Every sample must be positive, correct, within memory budget, and faster
/// including refresh; otherwise the bucket is rejected. No mean can hide
/// an individual regression.
pub fn freeze_valid_policy(
    samples: &[PairedSample],
    min_groups: usize,
    memory_limit_kib: u64,
) -> Result<Vec<Admission>, PolicyError> {
    let mut groups = BTreeSet::new();
    let mut buckets: BTreeMap<u32, (usize, u64, u64, u64, bool)> = BTreeMap::new();
    for s in samples {
        if s.partition != Partition::Valid { continue; }
        if !groups.insert(s.group) { return Err(PolicyError::DuplicateGroup); }
        let total = s.incremental_ns.checked_add(s.refresh_ns)
            .ok_or(PolicyError::Overflow)?;
        let entry = buckets.entry(s.cone_bucket).or_insert((0, 0, 0, 0, true));
        entry.0 = entry.0.checked_add(1).ok_or(PolicyError::Overflow)?;
        entry.1 = entry.1.max(s.full_ns);
        entry.2 = entry.2.max(total);
        entry.3 = entry.3.max(s.peak_kib);
        entry.4 &= s.oracle_equal && s.full_ns > 0 && s.incremental_ns > 0
            && s.peak_kib > 0 && s.peak_kib <= memory_limit_kib
            && total < s.full_ns;
    }
    if buckets.is_empty() { return Err(PolicyError::EmptyValidation); }
    Ok(buckets.into_iter().map(|(cone_bucket, (groups, worst_full_ns,
        worst_incremental_total_ns, max_peak_kib, valid))| Admission {
        cone_bucket, groups, worst_full_ns, worst_incremental_total_ns,
        max_peak_kib, allowed: valid && groups >= min_groups && min_groups > 0,
    }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample(group: u64, bucket: u32, partition: Partition, full: u64, inc: u64) -> PairedSample {
        PairedSample { partition, group, cone_bucket: bucket, full_ns: full,
            incremental_ns: inc, refresh_ns: 10, peak_kib: 128, oracle_equal: true }
    }
    #[test]
    fn independent_worst_case_oracle() {
        for full in 1..25 {
            for inc in 1..25 {
                let a = freeze_valid_policy(&[sample(1, 3, Partition::Valid, full, inc)], 1, 256).unwrap();
                assert_eq!(a[0].allowed, inc + 10 < full);
            }
        }
    }
    #[test]
    fn holdout_cannot_change_valid_decision() {
        let base = sample(1, 2, Partition::Valid, 100, 20);
        let mut data = vec![base];
        let original = freeze_valid_policy(&data, 1, 256).unwrap();
        data.push(sample(2, 2, Partition::Test, 1, 500));
        data.push(sample(3, 2, Partition::Ood, 1, 500));
        assert_eq!(freeze_valid_policy(&data, 1, 256).unwrap(), original);
    }
    #[test]
    fn non_valid_groups_do_not_poison_valid_policy() {
        let valid = sample(7, 2, Partition::Valid, 100, 20);
        let baseline = freeze_valid_policy(&[valid], 1, 256).unwrap();
        let mixed = [valid, sample(7, 2, Partition::Train, 1, 999),
            sample(7, 2, Partition::Test, 1, 999),
            sample(7, 2, Partition::Ood, 1, 999)];
        assert_eq!(freeze_valid_policy(&mixed, 1, 256).unwrap(), baseline);
        assert_eq!(freeze_valid_policy(&[valid, valid], 1, 256),
            Err(PolicyError::DuplicateGroup));
    }
    #[test]
    fn rejects_regression_missing_groups_and_duplicates() {
        let data = [sample(1, 0, Partition::Valid, 100, 20),
            sample(2, 0, Partition::Valid, 30, 40)];
        assert!(!freeze_valid_policy(&data, 2, 256).unwrap()[0].allowed);
        assert!(!freeze_valid_policy(&data[..1], 2, 256).unwrap()[0].allowed);
        assert_eq!(freeze_valid_policy(&[data[0], data[0]], 1, 256),
            Err(PolicyError::DuplicateGroup));
    }
}

/// Connect a frozen VALID policy to the real epoch-aware dispatcher.
/// No TEST/OOD observations are consulted here. Unknown or duplicated buckets,
/// stale cache epochs, absent VmHWM samples and zero gate budgets fail to Full.
#[must_use]
pub fn choose_frozen_valid(
    intervention: crate::fourier5_cone_epoch_dispatch::Intervention,
    frozen: &[Admission],
    committed_revision: u64,
    peak_limit_kib: u64,
    max_changed_gates: usize,
) -> crate::fourier5_cone_epoch_dispatch::EngineChoice {
    use crate::fourier5_cone_epoch_dispatch::{choose, ConeAdmission, EngineChoice};
    if max_changed_gates == 0 {
        return EngineChoice::Full;
    }
    let Ok(bucket) = u32::try_from(intervention.cone_bucket) else {
        return EngineChoice::Full;
    };
    let mut matches = frozen.iter().filter(|a| a.cone_bucket == bucket);
    let Some(admission) = matches.next() else {
        return EngineChoice::Full;
    };
    if matches.next().is_some() || !admission.allowed || admission.max_peak_kib == 0
        || admission.max_peak_kib > peak_limit_kib
    {
        return EngineChoice::Full;
    }
    let table = [ConeAdmission {
        valid_admitted: true,
        cone_bucket: intervention.cone_bucket,
        max_changed_gates,
    }];
    choose(intervention, &table, committed_revision, peak_limit_kib)
}

#[cfg(test)]
mod dispatch_integration_tests {
    use super::*;
    use crate::fourier5_cone_epoch_dispatch::{CacheEpoch, EngineChoice, Intervention};

    #[test]
    fn frozen_valid_dispatch_matches_independent_truth_table() {
        let valid = [PairedSample {
            partition: Partition::Valid, group: 17, cone_bucket: 2,
            full_ns: 100, incremental_ns: 20, refresh_ns: 10,
            peak_kib: 128, oracle_equal: true,
        }];
        let frozen = freeze_valid_policy(&valid, 1, 256).unwrap();
        for state in 0..3 {
            for cache in 0..3 {
                for cone in 0..4 {
                    for changed in 0..5 {
                        let intervention = Intervention {
                            cone_bucket: cone, changed_gates: changed,
                            cache_epoch: CacheEpoch { state_revision: state, cache_revision: cache },
                            observed_peak_kib: Some(128),
                        };
                        let expected = state == 1 && cache == 1 && cone == 2 && changed <= 2;
                        assert_eq!(
                            choose_frozen_valid(intervention, &frozen, 1, 256, 2)
                                == EngineChoice::Incremental,
                            expected,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn corrupted_policy_and_memory_fail_closed() {
        let sample = PairedSample {
            partition: Partition::Valid, group: 1, cone_bucket: 2,
            full_ns: 100, incremental_ns: 20, refresh_ns: 10,
            peak_kib: 128, oracle_equal: true,
        };
        let policy = freeze_valid_policy(&[sample], 1, 256).unwrap();
        let intervention = Intervention {
            cone_bucket: 2, changed_gates: 1,
            cache_epoch: CacheEpoch { state_revision: 4, cache_revision: 4 },
            observed_peak_kib: Some(128),
        };
        assert_eq!(choose_frozen_valid(intervention, &[policy[0], policy[0]], 4, 256, 2), EngineChoice::Full);
        assert_eq!(choose_frozen_valid(intervention, &policy, 4, 100, 2), EngineChoice::Full);
        assert_eq!(choose_frozen_valid(intervention, &policy, 4, 256, 0), EngineChoice::Full);
        assert_eq!(choose_frozen_valid(Intervention { observed_peak_kib: None, ..intervention }, &policy, 4, 256, 2), EngineChoice::Full);
        let rejected = freeze_valid_policy(&[PairedSample { oracle_equal: false, ..sample }], 1, 256).unwrap();
        assert_eq!(choose_frozen_valid(intervention, &rejected, 4, 256, 2), EngineChoice::Full);
    }
}

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
        if !groups.insert(s.group) { return Err(PolicyError::DuplicateGroup); }
        if s.partition != Partition::Valid { continue; }
        let total = s.incremental_ns.checked_add(s.refresh_ns)
            .ok_or(PolicyError::Overflow)?;
        let entry = buckets.entry(s.cone_bucket).or_insert((0, 0, 0, 0, true));
        entry.0 += 1;
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
    fn rejects_regression_missing_groups_and_duplicates() {
        let data = [sample(1, 0, Partition::Valid, 100, 20),
            sample(2, 0, Partition::Valid, 30, 40)];
        assert!(!freeze_valid_policy(&data, 2, 256).unwrap()[0].allowed);
        assert!(!freeze_valid_policy(&data[..1], 2, 256).unwrap()[0].allowed);
        assert_eq!(freeze_valid_policy(&[data[0], data[0]], 1, 256),
            Err(PolicyError::DuplicateGroup));
    }
}

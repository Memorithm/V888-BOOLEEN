//! Explicitly separated production-cost calibration for sequential synthetic workloads.
//! Verified transaction timings include Full and must NEVER enter this selector.
use std::collections::HashSet;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split { Train, Valid, Test, Ood }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision { Full, Incremental }

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
pub fn select(groups: &[ValidationGroup<'_>], min_groups: usize, min_steps: usize,
              saving_percent: u8, peak_limit_kib: u64) -> Decision {
    if min_groups < 2 || groups.len() < min_groups || min_steps == 0
        || saving_percent >= 100 || peak_limit_kib == 0 {
        return Decision::Full;
    }
    let mut ids = HashSet::new();
    for group in groups {
        if group.split != Split::Valid || !ids.insert(group.id)
            || group.samples.len() < min_steps { return Decision::Full; }
        for sample in group.samples {
            let full = sample.full.as_nanos();
            let incremental = sample.incremental_with_cache_refresh.as_nanos();
            let Some(left) = incremental.checked_mul(100) else { return Decision::Full };
            let Some(right) = full.checked_mul(u128::from(100 - saving_percent))
                else { return Decision::Full };
            if !sample.exact_state_and_cache || full == 0 || incremental == 0
                || left >= right
                || !matches!(sample.vmhwm_kib, Some(peak) if peak > 0 && peak <= peak_limit_kib)
            { return Decision::Full; }
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
            exact_state_and_cache: true, vmhwm_kib: Some(1024),
        }
    }
    #[test]
    fn paired_valid_groups_select_incremental() {
        let a = [cost(100, 50)];
        let b = [cost(200, 80)];
        let groups = [
            ValidationGroup { split: Split::Valid, id: 1, samples: &a },
            ValidationGroup { split: Split::Valid, id: 2, samples: &b },
        ];
        assert_eq!(select(&groups, 2, 1, 20, 4096), Decision::Incremental);
    }
    #[test]
    fn slow_minority_is_not_hidden_by_fast_group() {
        let a = [cost(10000, 1)];
        let b = [cost(100, 90)];
        let groups = [
            ValidationGroup { split: Split::Valid, id: 1, samples: &a },
            ValidationGroup { split: Split::Valid, id: 2, samples: &b },
        ];
        assert_eq!(select(&groups, 2, 1, 20, 4096), Decision::Full);
    }
    #[test]
    fn provenance_mismatch_and_missing_memory_fail_closed() {
        let a = [cost(100, 10)];
        let b = [cost(100, 10)];
        let mut groups = [
            ValidationGroup { split: Split::Valid, id: 1, samples: &a },
            ValidationGroup { split: Split::Test, id: 2, samples: &b },
        ];
        assert_eq!(select(&groups, 2, 1, 20, 4096), Decision::Full);
        groups[1].split = Split::Valid;
        let missing = [PairedProductionCost { vmhwm_kib: None, ..cost(100, 10) }];
        let groups_missing = [
            ValidationGroup { split: Split::Valid, id: 1, samples: &a },
            ValidationGroup { split: Split::Valid, id: 2, samples: &missing },
        ];
        assert_eq!(select(&groups_missing, 2, 1, 20, 4096), Decision::Full);
    }
}

//! Strict provenance and revision gate for sequential synthetic experiments.
//! This is metadata validation, not a physical benchmark or biological claim.

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Split {
    Train,
    Valid,
    Test,
    Ood,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraceRecord {
    pub split: Split,
    pub dataset_id: u64,
    pub seed: u64,
    pub revision_before: u64,
    pub revision_after: u64,
    pub accepted: bool,
    pub cache_matches_full: bool,
    pub decision_matches_full: bool,
    pub vmhwm_kib: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceError {
    Empty,
    SplitLeak,
    BrokenRevision,
    InvalidTransition,
    OracleMismatch,
    MemoryExceeded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraceSummary {
    pub records: usize,
    pub accepted: usize,
    pub peak_vmhwm_kib: u64,
}

/// Validate complete ordered traces, with distinct (dataset, seed) provenance
/// across TRAIN/VALID/TEST/OOD. Each contiguous sequence must have an exact
/// revision transition; accepted changes advance by one, rejected by zero.
/// Peak RSS is process-wide and cannot be attributed to individual records.
pub fn validate_traces(
    records: &[TraceRecord],
    memory_limit_kib: u64,
) -> Result<TraceSummary, TraceError> {
    if records.is_empty() {
        return Err(TraceError::Empty);
    }
    let mut ownership = std::collections::BTreeMap::new();
    let mut previous = std::collections::BTreeMap::new();
    let mut accepted = 0usize;
    let mut peak = 0u64;
    for r in records {
        let key = (r.dataset_id, r.seed);
        if let Some(split) = ownership.insert(key, r.split) {
            if split != r.split {
                return Err(TraceError::SplitLeak);
            }
        }
        if let Some(last) = previous.get(&(r.split, key)) {
            if *last != r.revision_before {
                return Err(TraceError::BrokenRevision);
            }
        }
        let expected = r
            .revision_before
            .checked_add(u64::from(r.accepted))
            .ok_or(TraceError::InvalidTransition)?;
        if expected != r.revision_after {
            return Err(TraceError::InvalidTransition);
        }
        if !r.cache_matches_full || !r.decision_matches_full {
            return Err(TraceError::OracleMismatch);
        }
        if r.vmhwm_kib > memory_limit_kib {
            return Err(TraceError::MemoryExceeded);
        }
        peak = peak.max(r.vmhwm_kib);
        accepted += usize::from(r.accepted);
        previous.insert((r.split, key), r.revision_after);
    }
    Ok(TraceSummary {
        records: records.len(),
        accepted,
        peak_vmhwm_kib: peak,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn record(split: Split, before: u64, accepted: bool) -> TraceRecord {
        TraceRecord {
            split,
            dataset_id: 3,
            seed: 7,
            revision_before: before,
            revision_after: before + u64::from(accepted),
            accepted,
            cache_matches_full: true,
            decision_matches_full: true,
            vmhwm_kib: 2048,
        }
    }
    #[test]
    fn accepted_and_rejected_revisions_match_scalar_oracle() {
        let records = [
            record(Split::Valid, 0, true),
            record(Split::Valid, 1, false),
            record(Split::Valid, 1, true),
        ];
        assert_eq!(
            validate_traces(&records, 4096),
            Ok(TraceSummary {
                records: 3,
                accepted: 2,
                peak_vmhwm_kib: 2048
            })
        );
    }
    #[test]
    fn disjoint_split_enforced() {
        let records = [record(Split::Train, 0, true), record(Split::Test, 0, true)];
        assert_eq!(validate_traces(&records, 4096), Err(TraceError::SplitLeak));
    }
    #[test]
    fn revision_continuity_and_rejection_enforced() {
        let records = [record(Split::Valid, 0, true), record(Split::Valid, 0, true)];
        assert_eq!(
            validate_traces(&records, 4096),
            Err(TraceError::BrokenRevision)
        );
        let mut invalid = record(Split::Valid, 0, false);
        invalid.revision_after = 1;
        assert_eq!(
            validate_traces(&[invalid], 4096),
            Err(TraceError::InvalidTransition)
        );
    }
    #[test]
    fn oracle_and_memory_fail_closed() {
        let mut r = record(Split::Ood, 0, true);
        r.cache_matches_full = false;
        assert_eq!(validate_traces(&[r], 4096), Err(TraceError::OracleMismatch));
        r.cache_matches_full = true;
        assert_eq!(validate_traces(&[r], 1024), Err(TraceError::MemoryExceeded));
    }
}

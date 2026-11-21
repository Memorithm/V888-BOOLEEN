//! Conservative runtime dispatch for synthetic sequential caches.
//! VALID admission is supplied externally; TEST/OOD never calibrate here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineChoice { Full, Incremental }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheEpoch { pub state_revision: u64, pub cache_revision: u64 }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConeAdmission {
    pub valid_admitted: bool,
    pub cone_bucket: usize,
    pub max_changed_gates: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Intervention {
    pub cone_bucket: usize,
    pub changed_gates: usize,
    pub cache_epoch: CacheEpoch,
    pub observed_peak_kib: Option<u64>,
}

/// Only admit a prevalidated cone with a canonical, committed cache epoch.
/// A missing or duplicate bucket, stale cache or absent memory sample falls
/// back to Full. The caller must use the separate measured VALID selector.
#[must_use]
pub fn choose(
    input: Intervention,
    admissions: &[ConeAdmission],
    committed_revision: u64,
    peak_limit_kib: u64,
) -> EngineChoice {
    if peak_limit_kib == 0
        || input.cache_epoch.state_revision != committed_revision
        || input.cache_epoch.cache_revision != committed_revision
        || !matches!(input.observed_peak_kib, Some(peak) if peak > 0 && peak <= peak_limit_kib)
    {
        return EngineChoice::Full;
    }
    // Validate the entire frozen admission table: duplicate unrelated buckets
    // must not make a partially corrupted policy appear admissible.
    let mut seen = std::collections::HashSet::with_capacity(admissions.len());
    let mut admitted = false;
    let mut found = false;
    for bucket in admissions {
        if !seen.insert(bucket.cone_bucket) {
            return EngineChoice::Full;
        }
        if bucket.cone_bucket == input.cone_bucket {
            found = true;
            admitted = bucket.valid_admitted && input.changed_gates <= bucket.max_changed_gates;
        }
    }
    if found && admitted {
        EngineChoice::Incremental
    } else {
        EngineChoice::Full
    }
}

/// Advance both state and cache revisions after a verified commit.
/// Reject overflow; never wrap a revision to zero.
#[must_use]
pub fn next_committed_epoch(current: u64) -> Option<CacheEpoch> {
    let revision = current.checked_add(1)?;
    Some(CacheEpoch { state_revision: revision, cache_revision: revision })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample(state: u64, cache: u64, bucket: usize, gates: usize) -> Intervention {
        Intervention {
            cone_bucket: bucket,
            changed_gates: gates,
            cache_epoch: CacheEpoch { state_revision: state, cache_revision: cache },
            observed_peak_kib: Some(2048),
        }
    }
    #[test]
    fn independent_exhaustive_small_oracle() {
        let valid = [
            ConeAdmission { valid_admitted: true, cone_bucket: 0, max_changed_gates: 2 },
            ConeAdmission { valid_admitted: false, cone_bucket: 1, max_changed_gates: 4 },
        ];
        for state in 0..3 {
            for cache in 0..3 {
                for bucket in 0..3 {
                    for gates in 0..6 {
                        let expected = state == 1 && cache == 1 && bucket == 0 && gates <= 2;
                        assert_eq!(
                            choose(sample(state, cache, bucket, gates), &valid, 1, 4096)
                                == EngineChoice::Incremental,
                            expected
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn duplicate_buckets_missing_memory_and_stale_cache_reject() {
        let entry = ConeAdmission { valid_admitted: true, cone_bucket: 0, max_changed_gates: 5 };
        let input = sample(1, 1, 0, 1);
        assert_eq!(choose(input, &[entry, entry], 1, 4096), EngineChoice::Full);
        assert_eq!(choose(input, &[], 1, 4096), EngineChoice::Full);
        assert_eq!(choose(sample(1, 0, 0, 1), &[entry], 1, 4096), EngineChoice::Full);
        assert_eq!(choose(Intervention { observed_peak_kib: None, ..input }, &[entry], 1, 4096), EngineChoice::Full);
        assert_eq!(choose(input, &[entry], 1, 1024), EngineChoice::Full);
        assert_eq!(choose(input, &[entry], 1, 0), EngineChoice::Full);
    }
    #[test]
    fn independent_oracle_rejects_unrelated_duplicate_admissions() {
        let selected = ConeAdmission {
            valid_admitted: true,
            cone_bucket: 0,
            max_changed_gates: 2,
        };
        let other = ConeAdmission {
            valid_admitted: true,
            cone_bucket: 1,
            max_changed_gates: 2,
        };
        let input = sample(1, 1, 0, 1);
        // The oracle counts unique bucket identifiers across the *whole* table.
        for table in [
            vec![selected, other],
            vec![other, selected],
            vec![selected, other, other],
            vec![other, other, selected],
        ] {
            let unique = table.iter().map(|a| a.cone_bucket)
                .collect::<std::collections::HashSet<_>>().len() == table.len();
            let expected = if unique {
                EngineChoice::Incremental
            } else {
                EngineChoice::Full
            };
            assert_eq!(choose(input, &table, 1, 4096), expected);
        }
    }

    #[test]
    fn epoch_commit_is_atomic_and_overflow_checked() {
        assert_eq!(next_committed_epoch(9), Some(CacheEpoch { state_revision: 10, cache_revision: 10 }));
        assert_eq!(next_committed_epoch(u64::MAX), None);
    }
}

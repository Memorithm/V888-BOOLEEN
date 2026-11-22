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


/// Evidence produced by the caller after an accepted sequential intervention.
/// The canonical Full oracle must compare the *entire* state and cache, not
/// merely the requested output bit. This gate does not run the oracle itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitEvidence {
    pub previous_epoch: CacheEpoch,
    pub full_state_equal: bool,
    pub canonical_cache_equal: bool,
    pub cache_refresh_complete: bool,
}

/// Fail closed: never publish a new cache epoch after an unverified mutation.
/// The caller must keep the previous committed snapshot on `None`.
#[must_use]
pub fn commit_verified_epoch(
    committed_revision: u64,
    evidence: CommitEvidence,
) -> Option<CacheEpoch> {
    if evidence.previous_epoch.state_revision != committed_revision
        || evidence.previous_epoch.cache_revision != committed_revision
        || !evidence.full_state_equal
        || !evidence.canonical_cache_equal
        || !evidence.cache_refresh_complete
    {
        return None;
    }
    next_committed_epoch(committed_revision)
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
    fn independent_truth_table_for_verified_sequential_commit() {
        for state_ok in [false, true] {
            for cache_ok in [false, true] {
                for refresh_ok in [false, true] {
                    for state_revision in [6, 7, 8] {
                        for cache_revision in [6, 7, 8] {
                            let evidence = CommitEvidence {
                                previous_epoch: CacheEpoch { state_revision, cache_revision },
                                full_state_equal: state_ok,
                                canonical_cache_equal: cache_ok,
                                cache_refresh_complete: refresh_ok,
                            };
                            let expected = state_ok && cache_ok && refresh_ok
                                && state_revision == 7 && cache_revision == 7;
                            assert_eq!(
                                commit_verified_epoch(7, evidence),
                                expected.then_some(CacheEpoch {
                                    state_revision: 8,
                                    cache_revision: 8,
                                }),
                            );
                        }
                    }
                }
            }
        }
        let evidence = CommitEvidence {
            previous_epoch: CacheEpoch {
                state_revision: u64::MAX,
                cache_revision: u64::MAX,
            },
            full_state_equal: true,
            canonical_cache_equal: true,
            cache_refresh_complete: true,
        };
        assert_eq!(commit_verified_epoch(u64::MAX, evidence), None);
    }

    #[test]
    fn epoch_commit_is_atomic_and_overflow_checked() {
        assert_eq!(next_committed_epoch(9), Some(CacheEpoch { state_revision: 10, cache_revision: 10 }));
        assert_eq!(next_committed_epoch(u64::MAX), None);
    }
}


/// A verified sequential transition owns its snapshots: no speculative mutation
/// escapes when the incremental candidate differs from the independent Full
/// oracle. The returned cache is canonical even on a fallback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalSnapshot<S, C> {
    pub state: S,
    pub cache: C,
    pub epoch: CacheEpoch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifiedPath {
    Incremental,
    FullFallback,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedTransition<S, C> {
    pub snapshot: CanonicalSnapshot<S, C>,
    pub path: VerifiedPath,
}

/// Execute both paths on separate clones, compare the *entire* resulting
/// state and cache, then publish exactly one canonical epoch. Full is the
/// independent reference; this verification function is NOT a production
/// latency benchmark and must not train the VALID admission policy.
#[must_use]
pub fn execute_verified_transition<S, C, F, I>(
    committed: &CanonicalSnapshot<S, C>,
    full: F,
    incremental: I,
) -> Option<VerifiedTransition<S, C>>
where
    S: Clone + Eq,
    C: Clone + Eq,
    F: FnOnce(&S, &C) -> (S, C),
    I: FnOnce(&S, &C) -> (S, C),
{
    if committed.epoch.state_revision != committed.epoch.cache_revision {
        return None;
    }
    let next = next_committed_epoch(committed.epoch.state_revision)?;
    let (full_state, full_cache) = full(&committed.state, &committed.cache);
    let (candidate_state, candidate_cache) =
        incremental(&committed.state, &committed.cache);
    let equal = candidate_state == full_state && candidate_cache == full_cache;
    let (state, cache, path) = if equal {
        (candidate_state, candidate_cache, VerifiedPath::Incremental)
    } else {
        (full_state, full_cache, VerifiedPath::FullFallback)
    };
    Some(VerifiedTransition {
        snapshot: CanonicalSnapshot { state, cache, epoch: next },
        path,
    })
}

#[cfg(test)]
mod canonical_transaction_tests {
    use super::*;

    #[test]
    fn independent_reference_checks_entire_cache_and_rollback() {
        let start = CanonicalSnapshot {
            state: vec![false, true],
            cache: vec![0_u8, 1],
            epoch: CacheEpoch { state_revision: 4, cache_revision: 4 },
        };
        for corrupt_state in [false, true] {
            for corrupt_cache in [false, true] {
                let outcome = execute_verified_transition(
                    &start,
                    |state, cache| {
                        let mut s = state.to_vec();
                        let mut c = cache.to_vec();
                        s[0] = true;
                        c[0] = 7;
                        (s, c)
                    },
                    |state, cache| {
                        let mut s = state.to_vec();
                        let mut c = cache.to_vec();
                        s[0] = !corrupt_state;
                        c[0] = if corrupt_cache { 8 } else { 7 };
                        (s, c)
                    },
                )
                .expect("valid epoch");
                assert_eq!(outcome.snapshot.state, vec![true, true]);
                assert_eq!(outcome.snapshot.cache, vec![7, 1]);
                assert_eq!(outcome.snapshot.epoch.state_revision, 5);
                assert_eq!(outcome.snapshot.epoch.cache_revision, 5);
                assert_eq!(
                    outcome.path,
                    if corrupt_state || corrupt_cache {
                        VerifiedPath::FullFallback
                    } else {
                        VerifiedPath::Incremental
                    }
                );
            }
        }
        assert_eq!(start.state, vec![false, true]);
        assert_eq!(start.cache, vec![0, 1]);
    }

    #[test]
    fn stale_or_overflowed_epoch_never_invokes_either_engine() {
        use std::cell::Cell;
        for epoch in [
            CacheEpoch { state_revision: 3, cache_revision: 2 },
            CacheEpoch { state_revision: u64::MAX, cache_revision: u64::MAX },
        ] {
            let called = Cell::new(false);
            let start = CanonicalSnapshot { state: 0_u8, cache: 0_u8, epoch };
            let result = execute_verified_transition(
                &start,
                |_, _| { called.set(true); (1, 1) },
                |_, _| { called.set(true); (1, 1) },
            );
            assert!(result.is_none());
            assert!(!called.get());
        }
    }

    #[test]
    fn sequential_commits_rebuild_from_last_canonical_snapshot() {
        let mut snapshot = CanonicalSnapshot {
            state: 0_u64,
            cache: 0_u64,
            epoch: CacheEpoch { state_revision: 0, cache_revision: 0 },
        };
        for step in 1..=8_u64 {
            let result = execute_verified_transition(
                &snapshot,
                |state, _| (*state + 1, (*state + 1) * 3),
                |state, cache| {
                    let next = *state + 1;
                    (next, if step == 3 { *cache } else { next * 3 })
                },
            )
            .expect("epoch");
            assert_eq!(result.snapshot.state, step);
            assert_eq!(result.snapshot.cache, step * 3);
            assert_eq!(result.snapshot.epoch.state_revision, step);
            assert_eq!(
                result.path,
                if step == 3 { VerifiedPath::FullFallback } else { VerifiedPath::Incremental }
            );
            snapshot = result.snapshot;
        }
    }
}


/// Production dispatch: Full is invoked only when the frozen VALID policy
/// rejects Incremental. Both engines return a *complete canonical* state/cache
/// snapshot. Unlike the verified oracle path, this does not prove equality at
/// runtime; correctness requires independently qualified engine equivalence.
/// The old snapshot is never modified and epoch overflow rejects before work.
#[must_use]
pub fn execute_dispatched_transition<S, C, F, I>(
    committed: &CanonicalSnapshot<S, C>,
    intervention: Intervention,
    admissions: &[ConeAdmission],
    peak_limit_kib: u64,
    full: F,
    incremental: I,
) -> Option<VerifiedTransition<S, C>>
where
    F: FnOnce(&S, &C) -> (S, C),
    I: FnOnce(&S, &C) -> (S, C),
{
    let revision = committed.epoch.state_revision;
    if committed.epoch.cache_revision != revision {
        return None;
    }
    let next = next_committed_epoch(revision)?;
    let path = choose(intervention, admissions, revision, peak_limit_kib);
    let (state, cache) = match path {
        EngineChoice::Full => full(&committed.state, &committed.cache),
        EngineChoice::Incremental => incremental(&committed.state, &committed.cache),
    };
    Some(VerifiedTransition {
        snapshot: CanonicalSnapshot { state, cache, epoch: next },
        path: match path {
            EngineChoice::Full => VerifiedPath::FullFallback,
            EngineChoice::Incremental => VerifiedPath::Incremental,
        },
    })
}

#[cfg(test)]
mod dispatched_transition_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn independent_oracle_for_sequential_mixed_cones() {
        let admissions = [
            ConeAdmission { valid_admitted: true, cone_bucket: 0, max_changed_gates: 2 },
            ConeAdmission { valid_admitted: false, cone_bucket: 1, max_changed_gates: 9 },
        ];
        let mut snapshot = CanonicalSnapshot {
            state: 0_u64,
            cache: 0_u64,
            epoch: CacheEpoch { state_revision: 0, cache_revision: 0 },
        };
        for step in 0..16_u64 {
            let full_called = Cell::new(false);
            let incremental_called = Cell::new(false);
            let bucket = (step % 3) as usize;
            let gates = (step % 4) as usize;
            let admissible = bucket == 0 && gates <= 2;
            let intervention = Intervention {
                cone_bucket: bucket,
                changed_gates: gates,
                cache_epoch: snapshot.epoch,
                observed_peak_kib: Some(1024),
            };
            let result = execute_dispatched_transition(
                &snapshot, intervention, &admissions, 2048,
                |state, _| {
                    full_called.set(true);
                    (*state + 1, (*state + 1) * 7)
                },
                |state, _| {
                    incremental_called.set(true);
                    (*state + 1, (*state + 1) * 7)
                },
            ).expect("valid sequential epoch");
            assert_eq!(incremental_called.get(), admissible);
            assert_eq!(full_called.get(), !admissible);
            assert_eq!(result.snapshot.state, step + 1);
            assert_eq!(result.snapshot.cache, (step + 1) * 7);
            assert_eq!(result.snapshot.epoch.state_revision, step + 1);
            assert_eq!(result.snapshot.epoch.cache_revision, step + 1);
            snapshot = result.snapshot;
        }
    }

    #[test]
    fn invalid_epochs_never_execute_and_missing_memory_falls_back() {
        for epoch in [
            CacheEpoch { state_revision: 5, cache_revision: 4 },
            CacheEpoch { state_revision: u64::MAX, cache_revision: u64::MAX },
        ] {
            let called = Cell::new(false);
            let snapshot = CanonicalSnapshot { state: 1_u8, cache: 1_u8, epoch };
            let intervention = Intervention {
                cone_bucket: 0, changed_gates: 0,
                cache_epoch: epoch, observed_peak_kib: Some(100),
            };
            let outcome = execute_dispatched_transition(
                &snapshot, intervention, &[], 200,
                |_, _| { called.set(true); (2, 2) },
                |_, _| { called.set(true); (2, 2) },
            );
            assert!(outcome.is_none());
            assert!(!called.get());
        }
        let snapshot = CanonicalSnapshot {
            state: 1_u8, cache: 1_u8,
            epoch: CacheEpoch { state_revision: 2, cache_revision: 2 },
        };
        let intervention = Intervention {
            cone_bucket: 0, changed_gates: 0,
            cache_epoch: snapshot.epoch, observed_peak_kib: None,
        };
        let result = execute_dispatched_transition(
            &snapshot, intervention,
            &[ConeAdmission { valid_admitted: true, cone_bucket: 0, max_changed_gates: 2 }],
            200,
            |_, _| (3, 9),
            |_, _| panic!("incremental must not execute without memory evidence"),
        ).expect("Full fallback");
        assert_eq!(result.path, VerifiedPath::FullFallback);
        assert_eq!((result.snapshot.state, result.snapshot.cache), (3, 9));
        assert_eq!((snapshot.state, snapshot.cache), (1, 1));
    }
}


/// Periodic independent Full checkpoints for production-like sequential runs.
/// An admitted Incremental step is verified at every `audit_interval`-th
/// committed revision. A zero interval disables Incremental entirely.
/// Checkpoint results are never used to recalibrate frozen VALID admissions.
#[must_use]
pub fn execute_checkpointed_transition<S, C, F, I>(
    committed: &CanonicalSnapshot<S, C>,
    intervention: Intervention,
    admissions: &[ConeAdmission],
    peak_limit_kib: u64,
    audit_interval: u64,
    full: F,
    incremental: I,
) -> Option<VerifiedTransition<S, C>>
where
    S: Clone + Eq,
    C: Clone + Eq,
    F: FnOnce(&S, &C) -> (S, C),
    I: FnOnce(&S, &C) -> (S, C),
{
    let revision = committed.epoch.state_revision;
    if committed.epoch.cache_revision != revision {
        return None;
    }
    let next = next_committed_epoch(revision)?;
    let choice = if audit_interval == 0 {
        EngineChoice::Full
    } else {
        choose(intervention, admissions, revision, peak_limit_kib)
    };
    if choice == EngineChoice::Full {
        let (state, cache) = full(&committed.state, &committed.cache);
        return Some(VerifiedTransition {
            snapshot: CanonicalSnapshot {
                state,
                cache,
                epoch: next,
            },
            path: VerifiedPath::FullFallback,
        });
    }
    if next.state_revision % audit_interval == 0 {
        return execute_verified_transition(committed, full, incremental);
    }
    let (state, cache) = incremental(&committed.state, &committed.cache);
    Some(VerifiedTransition {
        snapshot: CanonicalSnapshot {
            state,
            cache,
            epoch: next,
        },
        path: VerifiedPath::Incremental,
    })
}

#[cfg(test)]
mod checkpointed_transition_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn independent_sequential_checkpoint_oracle_with_corrupted_cache() {
        let admissions = [ConeAdmission {
            valid_admitted: true,
            cone_bucket: 0,
            max_changed_gates: 2,
        }];
        let mut snapshot = CanonicalSnapshot {
            state: 0_u64,
            cache: 0_u64,
            epoch: CacheEpoch {
                state_revision: 0,
                cache_revision: 0,
            },
        };
        for step in 1..=12_u64 {
            let full_calls = Cell::new(0);
            let incremental_calls = Cell::new(0);
            let intervention = Intervention {
                cone_bucket: 0,
                changed_gates: 1,
                cache_epoch: snapshot.epoch,
                observed_peak_kib: Some(1024),
            };
            let result = execute_checkpointed_transition(
                &snapshot,
                intervention,
                &admissions,
                2048,
                3,
                |state, _| {
                    full_calls.set(full_calls.get() + 1);
                    (*state + 1, (*state + 1) * 5)
                },
                |state, _| {
                    incremental_calls.set(incremental_calls.get() + 1);
                    (*state + 1, if step == 6 { 0 } else { (*state + 1) * 5 })
                },
            )
            .expect("valid epoch");
            assert_eq!(full_calls.get(), u32::from(step % 3 == 0));
            assert_eq!(incremental_calls.get(), 1);
            assert_eq!(result.snapshot.state, step);
            assert_eq!(result.snapshot.cache, step * 5);
            assert_eq!(
                result.path,
                if step == 6 {
                    VerifiedPath::FullFallback
                } else {
                    VerifiedPath::Incremental
                }
            );
            snapshot = result.snapshot;
        }
    }

    #[test]
    fn disabled_audit_and_invalid_epoch_never_run_incremental() {
        let admission = [ConeAdmission {
            valid_admitted: true,
            cone_bucket: 0,
            max_changed_gates: 1,
        }];
        let snapshot = CanonicalSnapshot {
            state: 2_u8,
            cache: 4_u8,
            epoch: CacheEpoch {
                state_revision: 2,
                cache_revision: 2,
            },
        };
        let intervention = Intervention {
            cone_bucket: 0,
            changed_gates: 1,
            cache_epoch: snapshot.epoch,
            observed_peak_kib: Some(100),
        };
        let result = execute_checkpointed_transition(
            &snapshot,
            intervention,
            &admission,
            200,
            0,
            |_, _| (3, 6),
            |_, _| panic!("incremental must not execute"),
        )
        .expect("Full only");
        assert_eq!(result.path, VerifiedPath::FullFallback);
        assert_eq!((result.snapshot.state, result.snapshot.cache), (3, 6));
        let stale = CanonicalSnapshot {
            epoch: CacheEpoch {
                state_revision: 2,
                cache_revision: 1,
            },
            ..snapshot
        };
        assert!(execute_checkpointed_transition(
            &stale,
            intervention,
            &admission,
            200,
            3,
            |_, _| panic!("Full must not execute"),
            |_, _| panic!("Incremental must not execute"),
        )
        .is_none());
    }
}

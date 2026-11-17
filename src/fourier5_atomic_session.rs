//! Atomic synthetic sequential intervention: a failed cache rebuild never publishes
//! a partially changed circuit. This is an opt-in clone-based reference path,
//! not a claim about the performance of the wide circuit engine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtomicSession<C, K> {
    pub circuit: C,
    pub cache: K,
    pub revision: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomicError<E> {
    Mutation(E),
    Rebuild(E),
    RevisionOverflow,
}

/// A rejected intervention leaves both objects unchanged. For an accepted
/// intervention, mutation and cache rebuild run on private copies. The pair
/// is published only after both callbacks succeed. This prevents a caller's
/// partially-mutating callback from corrupting the committed session.
///
/// The callbacks must not mutate external state: only C and K are transactional.
pub fn apply_atomic<C: Clone, K: Clone, E>(
    session: &mut AtomicSession<C, K>,
    mutate: impl FnOnce(&mut C) -> Result<bool, E>,
    rebuild: impl FnOnce(&C, &mut K) -> Result<(), E>,
) -> Result<bool, AtomicError<E>> {
    // Do not run an unrollbackable callback when the revision cannot advance.
    let next = session
        .revision
        .checked_add(1)
        .ok_or(AtomicError::RevisionOverflow)?;
    let mut candidate = session.circuit.clone();
    let accepted = mutate(&mut candidate).map_err(AtomicError::Mutation)?;
    if !accepted {
        return Ok(false);
    }
    let mut next_cache = session.cache.clone();
    rebuild(&candidate, &mut next_cache).map_err(AtomicError::Rebuild)?;
    session.circuit = candidate;
    session.cache = next_cache;
    session.revision = next;
    Ok(true)
}

#[cfg(test)]
type TestOperation = Box<dyn FnMut(&mut Vec<bool>) -> Result<bool, &'static str>>;

#[cfg(test)]
mod tests {
    use super::*;

    // Independent full-recomputation oracle: cache equals the sum of all bits.
    fn full(c: &[bool]) -> usize {
        c.iter().filter(|&&bit| bit).count()
    }

    #[test]
    fn sequential_commits_match_full_oracle_after_every_acceptance() {
        let mut s = AtomicSession {
            circuit: vec![false; 19],
            cache: 0usize,
            revision: 0,
        };
        for index in [3, 7, 3, 18, 0, 7] {
            let accepted = apply_atomic(
                &mut s,
                |c| {
                    c[index] = !c[index];
                    Ok::<_, ()>(true)
                },
                |c, k| {
                    *k = full(c);
                    Ok::<_, ()>(())
                },
            )
            .unwrap();
            assert!(accepted);
            assert_eq!(s.cache, full(&s.circuit));
        }
        assert_eq!(s.revision, 6);
    }

    #[test]
    fn failed_partial_mutation_and_failed_rebuild_are_atomic() {
        let original = AtomicSession {
            circuit: vec![false, true],
            cache: 1usize,
            revision: 9,
        };
        let mut s = original.clone();
        assert_eq!(
            apply_atomic(
                &mut s,
                |c| {
                    c[0] = true;
                    Err::<bool, _>("mutation")
                },
                |_, _| Ok(())
            ),
            Err(AtomicError::Mutation("mutation"))
        );
        assert_eq!(s, original);
        assert_eq!(
            apply_atomic(
                &mut s,
                |c| {
                    c[0] = true;
                    Ok::<_, &str>(true)
                },
                |_, k| {
                    *k = 99;
                    Err("rebuild")
                }
            ),
            Err(AtomicError::Rebuild("rebuild"))
        );
        assert_eq!(s, original);
        assert!(!apply_atomic(
            &mut s,
            |c| {
                c[0] = true;
                Ok::<_, &str>(false)
            },
            |_, _| panic!("rejected interventions must not rebuild")
        )
        .unwrap());
        assert_eq!(s, original);
    }

    #[test]
    fn overflow_fails_before_mutation() {
        let mut s = AtomicSession {
            circuit: vec![false],
            cache: 0usize,
            revision: u64::MAX,
        };
        let before = s.clone();
        assert_eq!(
            apply_atomic(
                &mut s,
                |_| panic!("must not mutate on overflow"),
                |_, _| Ok::<(), ()>(())
            ),
            Err(AtomicError::RevisionOverflow)
        );
        assert_eq!(s, before);
    }
}

/// Lower-memory atomic commit path: construct a new cache from the accepted
/// candidate instead of cloning the previous cache. The cache factory is
/// invoked only after acceptance; failures leave the committed pair intact.
/// This is useful when the cached cone state is large, but still clones C.
/// No speed or memory benefit is claimed without physical measurement.
pub fn apply_atomic_fresh_cache<C: Clone, K, E>(
    session: &mut AtomicSession<C, K>,
    mutate: impl FnOnce(&mut C) -> Result<bool, E>,
    build_cache: impl FnOnce(&C) -> Result<K, E>,
) -> Result<bool, AtomicError<E>> {
    // Fail before invoking a potentially side-effecting callback when no
    // accepted revision can be represented. This matches apply_atomic.
    let next_revision = session
        .revision
        .checked_add(1)
        .ok_or(AtomicError::RevisionOverflow)?;
    let mut candidate = session.circuit.clone();
    let accepted = mutate(&mut candidate).map_err(AtomicError::Mutation)?;
    if !accepted {
        return Ok(false);
    }
    let cache = build_cache(&candidate).map_err(AtomicError::Rebuild)?;
    session.circuit = candidate;
    session.cache = cache;
    session.revision = next_revision;
    Ok(true)
}

#[cfg(test)]
mod fresh_cache_tests {
    use super::*;

    struct NonCloneCache(usize);

    #[test]
    fn sequential_nonclone_cache_matches_independent_full_oracle() {
        let mut s = AtomicSession {
            circuit: vec![false; 23],
            cache: NonCloneCache(0),
            revision: 0,
        };
        for index in [0, 22, 11, 0, 5, 22] {
            let accepted = apply_atomic_fresh_cache(
                &mut s,
                |bits| {
                    bits[index] = !bits[index];
                    Ok::<_, ()>(true)
                },
                |bits| Ok::<_, ()>(NonCloneCache(bits.iter().filter(|&&v| v).count())),
            )
            .unwrap();
            assert!(accepted);
            assert_eq!(s.cache.0, s.circuit.iter().filter(|&&v| v).count());
        }
        assert_eq!(s.revision, 6);
    }

    #[test]
    fn reject_and_failed_build_preserve_committed_nonclone_cache() {
        let mut s = AtomicSession {
            circuit: vec![true, false],
            cache: NonCloneCache(1),
            revision: 7,
        };
        assert!(!apply_atomic_fresh_cache(
            &mut s,
            |bits| {
                bits[1] = true;
                Ok::<_, &str>(false)
            },
            |_| panic!("rejected candidate must not build cache"),
        )
        .unwrap());
        assert_eq!(s.circuit, [true, false]);
        assert_eq!(s.cache.0, 1);
        assert_eq!(s.revision, 7);
        assert_eq!(
            apply_atomic_fresh_cache(
                &mut s,
                |bits| {
                    bits[1] = true;
                    Ok::<_, &str>(true)
                },
                |_| Err::<NonCloneCache, _>("failed"),
            ),
            Err(AtomicError::Rebuild("failed"))
        );
        assert_eq!(s.circuit, [true, false]);
        assert_eq!(s.cache.0, 1);
        assert_eq!(s.revision, 7);
    }

    #[test]
    fn overflow_preserves_state_and_does_not_build_cache() {
        let mut s = AtomicSession {
            circuit: vec![false],
            cache: NonCloneCache(0),
            revision: u64::MAX,
        };
        assert_eq!(
            apply_atomic_fresh_cache(
                &mut s,
                |bits| {
                    bits[0] = true;
                    Ok::<_, ()>(true)
                },
                |_| panic!("overflow must prevent cache build"),
            ),
            Err(AtomicError::RevisionOverflow)
        );
        assert_eq!(s.circuit, [false]);
        assert_eq!(s.cache.0, 0);
        assert_eq!(s.revision, u64::MAX);
    }
}

/// Apply a sequence of interventions on a private circuit candidate, then
/// construct exactly one cache for the final accepted state. This is an atomic
/// batch, not a replacement for per-intervention sequential latency measurement:
/// intermediate accepted states are deliberately not published or observable.
/// A rejected intervention may modify the private candidate; those changes are
/// discarded by restoring its pre-intervention snapshot.
pub fn apply_atomic_batch<C: Clone, K, E>(
    session: &mut AtomicSession<C, K>,
    interventions: &mut [impl FnMut(&mut C) -> Result<bool, E>],
    build_cache: impl FnOnce(&C) -> Result<K, E>,
) -> Result<usize, AtomicError<E>> {
    let mut candidate = session.circuit.clone();
    let mut accepted = 0usize;
    for intervention in interventions {
        let mut proposed = candidate.clone();
        if intervention(&mut proposed).map_err(AtomicError::Mutation)? {
            // Check revision before accepting the candidate.
            let next_count = accepted
                .checked_add(1)
                .ok_or(AtomicError::RevisionOverflow)?;
            let increment = u64::try_from(next_count).map_err(|_| AtomicError::RevisionOverflow)?;
            session
                .revision
                .checked_add(increment)
                .ok_or(AtomicError::RevisionOverflow)?;
            candidate = proposed;
            accepted = next_count;
        }
    }
    if accepted == 0 {
        return Ok(0);
    }
    let cache = build_cache(&candidate).map_err(AtomicError::Rebuild)?;
    let increment = u64::try_from(accepted).map_err(|_| AtomicError::RevisionOverflow)?;
    session.circuit = candidate;
    session.cache = cache;
    session.revision += increment;
    Ok(accepted)
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    fn oracle(bits: &[bool]) -> usize {
        bits.iter().filter(|&&value| value).count()
    }

    #[test]
    fn batch_rebuilds_once_and_matches_independent_full_oracle() {
        let mut session = AtomicSession {
            circuit: vec![false; 8],
            cache: 0usize,
            revision: 4,
        };
        let mut operations: Vec<TestOperation> = vec![
            Box::new(|bits| {
                bits[1] = true;
                Ok(true)
            }),
            Box::new(|bits| {
                bits[2] = true;
                Ok(false)
            }),
            Box::new(|bits| {
                bits[7] = true;
                Ok(true)
            }),
        ];
        let mut builds = 0;
        let count = apply_atomic_batch(&mut session, &mut operations, |bits| {
            builds += 1;
            Ok::<_, &'static str>(oracle(bits))
        })
        .unwrap();
        assert_eq!(count, 2);
        assert_eq!(builds, 1);
        assert_eq!(session.revision, 6);
        assert!(!session.circuit[2]);
        assert_eq!(session.cache, oracle(&session.circuit));
    }

    #[test]
    fn batch_failure_rolls_back_every_prior_acceptance() {
        let mut session = AtomicSession {
            circuit: vec![false; 4],
            cache: 0usize,
            revision: 2,
        };
        let original = session.clone();
        let mut operations: Vec<TestOperation> = vec![
            Box::new(|bits| {
                bits[0] = true;
                Ok(true)
            }),
            Box::new(|bits| {
                bits[1] = true;
                Err("mutation failed")
            }),
        ];
        assert_eq!(
            apply_atomic_batch(&mut session, &mut operations, |_| Ok(0)),
            Err(AtomicError::Mutation("mutation failed"))
        );
        assert_eq!(session, original);
        operations.pop();
        assert_eq!(
            apply_atomic_batch(&mut session, &mut operations, |_| Err::<usize, _>(
                "cache failed"
            )),
            Err(AtomicError::Rebuild("cache failed"))
        );
        assert_eq!(session, original);
    }

    #[test]
    fn batch_rejects_revision_overflow_and_skips_empty_rebuild() {
        let mut session = AtomicSession {
            circuit: vec![false],
            cache: 0usize,
            revision: u64::MAX,
        };
        let mut rejected = [|bits: &mut Vec<bool>| {
            bits[0] = true;
            Ok::<_, ()>(false)
        }];
        assert_eq!(
            apply_atomic_batch(&mut session, &mut rejected, |_| panic!(
                "no accepted changes"
            )),
            Ok(0)
        );
        assert_eq!(session.circuit, [false]);
        let mut accepted = [|bits: &mut Vec<bool>| {
            bits[0] = true;
            Ok::<_, ()>(true)
        }];
        assert_eq!(
            apply_atomic_batch(&mut session, &mut accepted, |_| panic!(
                "overflow before rebuild"
            )),
            Err(AtomicError::RevisionOverflow)
        );
        assert_eq!(session.revision, u64::MAX);
        assert_eq!(session.circuit, [false]);
    }
}

/// Execute a batch with an exact cache rebuild after EVERY accepted change,
/// while publishing the entire batch atomically. Unlike apply_atomic_batch,
/// intermediate accepted states are available to the observer and the cache
/// is never stale at those observation points. The observer must not have
/// external side effects: callbacks are outside the rollback boundary.
/// Rejected proposals are discarded even if their callback mutated them.
pub fn apply_atomic_observed_batch<C: Clone, K, E>(
    session: &mut AtomicSession<C, K>,
    interventions: &mut [impl FnMut(&mut C) -> Result<bool, E>],
    mut build_cache: impl FnMut(&C) -> Result<K, E>,
    mut observe: impl FnMut(&C, &K, u64) -> Result<(), E>,
) -> Result<usize, AtomicError<E>> {
    let mut candidate = session.circuit.clone();
    let mut candidate_cache: Option<K> = None;
    let mut accepted = 0usize;
    for intervention in interventions {
        let mut proposed = candidate.clone();
        if !intervention(&mut proposed).map_err(AtomicError::Mutation)? {
            continue;
        }
        let next_count = accepted
            .checked_add(1)
            .ok_or(AtomicError::RevisionOverflow)?;
        let increment = u64::try_from(next_count).map_err(|_| AtomicError::RevisionOverflow)?;
        let revision = session
            .revision
            .checked_add(increment)
            .ok_or(AtomicError::RevisionOverflow)?;
        let cache = build_cache(&proposed).map_err(AtomicError::Rebuild)?;
        observe(&proposed, &cache, revision).map_err(AtomicError::Rebuild)?;
        candidate = proposed;
        candidate_cache = Some(cache);
        accepted = next_count;
    }
    if let Some(cache) = candidate_cache {
        let increment = u64::try_from(accepted).map_err(|_| AtomicError::RevisionOverflow)?;
        session.circuit = candidate;
        session.cache = cache;
        session.revision += increment;
    }
    Ok(accepted)
}

#[cfg(test)]
mod observed_batch_tests {
    use super::*;

    fn full(bits: &[bool]) -> usize {
        bits.iter().filter(|&&value| value).count()
    }

    #[test]
    fn every_intermediate_cache_matches_independent_full_oracle() {
        let mut session = AtomicSession {
            circuit: vec![false; 8],
            cache: 0usize,
            revision: 10,
        };
        let mut operations: Vec<TestOperation> = vec![
            Box::new(|bits| {
                bits[0] = true;
                Ok(true)
            }),
            Box::new(|bits| {
                bits[1] = true;
                Ok(false)
            }),
            Box::new(|bits| {
                bits[7] = true;
                Ok(true)
            }),
        ];
        let mut observed = Vec::new();
        let count = apply_atomic_observed_batch(
            &mut session,
            &mut operations,
            |bits| Ok::<_, &'static str>(full(bits)),
            |bits, cache, revision| {
                assert_eq!(*cache, full(bits));
                observed.push((revision, *cache));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(count, 2);
        assert_eq!(observed, [(11, 1), (12, 2)]);
        assert_eq!(session.revision, 12);
        assert_eq!(session.cache, full(&session.circuit));
        assert!(!session.circuit[1]);
    }

    #[test]
    fn observer_failure_rolls_back_entire_batch() {
        let mut session = AtomicSession {
            circuit: vec![false; 3],
            cache: 0usize,
            revision: 4,
        };
        let original = session.clone();
        let mut operations: Vec<TestOperation> = vec![
            Box::new(|bits| {
                bits[0] = true;
                Ok(true)
            }),
            Box::new(|bits| {
                bits[1] = true;
                Ok(true)
            }),
        ];
        let result = apply_atomic_observed_batch(
            &mut session,
            &mut operations,
            |bits| Ok::<_, &'static str>(full(bits)),
            |_, _, revision| {
                if revision == 6 {
                    Err("observer")
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(result, Err(AtomicError::Rebuild("observer")));
        assert_eq!(session, original);
    }

    #[test]
    fn rejected_only_and_overflow_never_rebuild() {
        let mut session = AtomicSession {
            circuit: vec![false],
            cache: 0usize,
            revision: u64::MAX,
        };
        let mut rejected = [|bits: &mut Vec<bool>| {
            bits[0] = true;
            Ok::<_, ()>(false)
        }];
        assert_eq!(
            apply_atomic_observed_batch(
                &mut session,
                &mut rejected,
                |_| panic!("no rebuild"),
                |_, _, _| panic!("no observation"),
            ),
            Ok(0)
        );
        assert_eq!(session.circuit, [false]);
        let mut accepted = [|bits: &mut Vec<bool>| {
            bits[0] = true;
            Ok::<_, ()>(true)
        }];
        assert_eq!(
            apply_atomic_observed_batch(
                &mut session,
                &mut accepted,
                |_| panic!("overflow before rebuild"),
                |_, _, _| panic!("overflow before observation"),
            ),
            Err(AtomicError::RevisionOverflow)
        );
        assert_eq!(session.revision, u64::MAX);
        assert_eq!(session.circuit, [false]);
    }
}

/// Optimistic-concurrency guard for an exact sequential intervention.
/// A stale expected revision is rejected before cloning or calling user code.
/// On a match, the candidate is rebuilt and committed atomically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuardedAtomicError<E> {
    StaleRevision { expected: u64, actual: u64 },
    Atomic(AtomicError<E>),
}

pub fn apply_atomic_at_revision<C: Clone, K, E>(
    session: &mut AtomicSession<C, K>,
    expected_revision: u64,
    mutate: impl FnOnce(&mut C) -> Result<bool, E>,
    build_cache: impl FnOnce(&C) -> Result<K, E>,
) -> Result<bool, GuardedAtomicError<E>> {
    if session.revision != expected_revision {
        return Err(GuardedAtomicError::StaleRevision {
            expected: expected_revision,
            actual: session.revision,
        });
    }
    apply_atomic_fresh_cache(session, mutate, build_cache).map_err(GuardedAtomicError::Atomic)
}

#[cfg(test)]
mod revision_guard_tests {
    use super::*;

    fn full_oracle(bits: &[bool]) -> usize {
        bits.iter().filter(|&&bit| bit).count()
    }

    #[test]
    fn stale_revision_never_executes_mutation_or_rebuild() {
        let mut session = AtomicSession {
            circuit: vec![false, true],
            cache: 1usize,
            revision: 7,
        };
        let result = apply_atomic_at_revision(
            &mut session,
            6,
            |_| -> Result<bool, ()> { panic!("stale mutation must not execute") },
            |_| -> Result<usize, ()> { panic!("stale rebuild must not execute") },
        );
        assert_eq!(
            result,
            Err(GuardedAtomicError::StaleRevision {
                expected: 6,
                actual: 7
            })
        );
        assert_eq!(session.revision, 7);
        assert_eq!(session.cache, 1);
        assert_eq!(session.circuit, [false, true]);
    }

    #[test]
    fn guarded_sequential_updates_match_full_oracle_and_reject_replay() {
        let mut session = AtomicSession {
            circuit: vec![false; 12],
            cache: 0usize,
            revision: 0,
        };
        for index in [1, 4, 1, 11, 0] {
            let expected = session.revision;
            assert!(apply_atomic_at_revision(
                &mut session,
                expected,
                |bits| {
                    bits[index] = !bits[index];
                    Ok::<_, ()>(true)
                },
                |bits| Ok::<_, ()>(full_oracle(bits)),
            )
            .unwrap());
            assert_eq!(session.cache, full_oracle(&session.circuit));
            assert_eq!(session.revision, expected + 1);
            assert_eq!(
                apply_atomic_at_revision(
                    &mut session,
                    expected,
                    |_| -> Result<bool, ()> { panic!("replayed mutation") },
                    |_| -> Result<usize, ()> { panic!("replayed rebuild") },
                ),
                Err(GuardedAtomicError::StaleRevision {
                    expected,
                    actual: expected + 1
                })
            );
        }
    }

    #[test]
    fn overflow_prevents_mutation_even_with_matching_revision() {
        let mut session = AtomicSession {
            circuit: vec![false],
            cache: 0usize,
            revision: u64::MAX,
        };
        assert_eq!(
            apply_atomic_at_revision(
                &mut session,
                u64::MAX,
                |_| -> Result<bool, ()> { panic!("overflow mutation") },
                |_| -> Result<usize, ()> { panic!("overflow rebuild") },
            ),
            Err(GuardedAtomicError::Atomic(AtomicError::RevisionOverflow))
        );
        assert_eq!(session.circuit, [false]);
    }
}

/// An independent full-recomputation oracle must agree with the proposed
/// incremental cache before either object is committed. A mismatch is a
/// fail-closed error, not an accepted (and possibly incorrect) intervention.
/// The oracle and equality predicate must be side-effect-free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifiedAtomicError<E> {
    Mutation(E),
    Incremental(E),
    Oracle(E),
    RevisionOverflow,
    OracleMismatch,
}

pub fn apply_atomic_verified<C: Clone, K, O, E>(
    session: &mut AtomicSession<C, K>,
    mutate: impl FnOnce(&mut C) -> Result<bool, E>,
    incremental: impl FnOnce(&C, &K) -> Result<K, E>,
    full_oracle: impl FnOnce(&C) -> Result<O, E>,
    agrees: impl FnOnce(&K, &O) -> bool,
) -> Result<bool, VerifiedAtomicError<E>> {
    let next = session
        .revision
        .checked_add(1)
        .ok_or(VerifiedAtomicError::RevisionOverflow)?;
    let mut candidate = session.circuit.clone();
    if !mutate(&mut candidate).map_err(VerifiedAtomicError::Mutation)? {
        return Ok(false);
    }
    let candidate_cache =
        incremental(&candidate, &session.cache).map_err(VerifiedAtomicError::Incremental)?;
    let expected = full_oracle(&candidate).map_err(VerifiedAtomicError::Oracle)?;
    if !agrees(&candidate_cache, &expected) {
        return Err(VerifiedAtomicError::OracleMismatch);
    }
    session.circuit = candidate;
    session.cache = candidate_cache;
    session.revision = next;
    Ok(true)
}

#[cfg(test)]
mod verified_atomic_tests {
    use super::*;

    fn full(bits: &[bool]) -> usize {
        bits.iter().filter(|&&bit| bit).count()
    }

    #[test]
    fn verified_sequential_updates_match_independent_full_recompute() {
        let mut session = AtomicSession {
            circuit: vec![false; 16],
            cache: 0usize,
            revision: 0,
        };
        for (step, index) in [1, 8, 1, 15, 0].into_iter().enumerate() {
            assert!(apply_atomic_verified(
                &mut session,
                |bits| {
                    bits[index] = !bits[index];
                    Ok::<_, ()>(true)
                },
                |bits, _old| Ok::<_, ()>(full(bits)),
                |bits| Ok::<_, ()>(bits.iter().map(|v| usize::from(*v)).sum::<usize>()),
                |incremental, oracle| incremental == oracle,
            )
            .unwrap());
            assert_eq!(session.cache, full(&session.circuit));
            assert_eq!(session.revision as usize, step + 1);
        }
    }

    #[test]
    fn oracle_mismatch_and_oracle_error_leave_state_unchanged() {
        let original = AtomicSession {
            circuit: vec![true, false],
            cache: 1usize,
            revision: 4,
        };
        let mut session = original.clone();
        assert_eq!(
            apply_atomic_verified(
                &mut session,
                |bits| {
                    bits[1] = true;
                    Ok::<_, &str>(true)
                },
                |_, _| Ok(0usize),
                |bits| Ok(full(bits)),
                |a, b| a == b,
            ),
            Err(VerifiedAtomicError::OracleMismatch)
        );
        assert_eq!(session, original);
        assert_eq!(
            apply_atomic_verified(
                &mut session,
                |bits| {
                    bits[1] = true;
                    Ok::<_, &str>(true)
                },
                |_, _| Ok(2usize),
                |_| Err::<usize, _>("oracle unavailable"),
                |a, b| a == b,
            ),
            Err(VerifiedAtomicError::Oracle("oracle unavailable"))
        );
        assert_eq!(session, original);
    }

    #[test]
    fn rejected_and_overflow_interventions_do_not_run_oracle() {
        let mut session = AtomicSession {
            circuit: vec![false],
            cache: 0usize,
            revision: 1,
        };
        assert!(!apply_atomic_verified(
            &mut session,
            |bits| {
                bits[0] = true;
                Ok::<_, ()>(false)
            },
            |_, _| -> Result<usize, ()> { panic!("rejected incremental") },
            |_| -> Result<usize, ()> { panic!("rejected oracle") },
            |_, _| panic!("rejected comparison"),
        )
        .unwrap());
        assert_eq!(session.circuit, [false]);
        session.revision = u64::MAX;
        assert_eq!(
            apply_atomic_verified(
                &mut session,
                |_| -> Result<bool, ()> { panic!("overflow mutation") },
                |_, _| -> Result<usize, ()> { panic!("overflow incremental") },
                |_| -> Result<usize, ()> { panic!("overflow oracle") },
                |_, _| panic!("overflow comparison"),
            ),
            Err(VerifiedAtomicError::RevisionOverflow)
        );
    }
}

//! Fail-closed epoch tracking for sequential synthetic cache interventions.
//! This is a reusable admission guard, not a cache implementation or a BANC model.
use crate::fourier5_conservative_policy::Engine;
use crate::fourier5_sequential_cost_policy::SequentialCostPolicy;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpochError {
    StaleProposal,
    Overflow,
    OracleMismatch,
    InvalidGeneration,
}

/// An intervention is evaluated against a snapshot of the current cache epoch.
/// The caller must rebuild/refresh the actual cache before publishing the new epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheEpochGuard {
    generation: u64,
    cache_generation: Option<u64>,
}

impl CacheEpochGuard {
    #[must_use]
    pub fn new() -> Self {
        Self {
            generation: 0,
            cache_generation: None,
        }
    }

    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    #[must_use]
    pub fn cache_current(&self) -> bool {
        self.cache_generation == Some(self.generation)
    }

    /// Explicitly certify an externally rebuilt cache for the current state.
    /// Never accepts a future or old generation.
    pub fn certify_rebuild(&mut self, generation: u64) -> Result<(), EpochError> {
        if generation != self.generation {
            return Err(EpochError::InvalidGeneration);
        }
        self.cache_generation = Some(generation);
        Ok(())
    }


    /// Atomically apply a sequential intervention and publish its refreshed cache.
    /// Both Boolean output and downstream decision must agree with an independent
    /// Full oracle. A stale proposal is rejected before any candidate work.
    /// Rebuild and oracle closures operate only on private candidate copies.
    pub fn verified_refresh<S, C, I, B, F, D>(
        &mut self,
        expected_generation: u64,
        state: &mut S,
        cache: &mut C,
        intervention: I,
        rebuild: B,
        full: F,
        incremental: D,
    ) -> Result<Option<u64>, EpochError>
    where
        S: Clone,
        I: FnOnce(&mut S) -> bool,
        B: FnOnce(&S) -> C,
        F: FnOnce(&S) -> (bool, bool),
        D: FnOnce(&S, &C) -> (bool, bool),
    {
        if expected_generation != self.generation {
            return Err(EpochError::StaleProposal);
        }
        // Reject overflow before performing potentially expensive work.
        let next = self.generation.checked_add(1).ok_or(EpochError::Overflow)?;
        let mut candidate = state.clone();
        if !intervention(&mut candidate) {
            return Ok(None);
        }
        let candidate_cache = rebuild(&candidate);
        let expected = full(&candidate);
        let observed = incremental(&candidate, &candidate_cache);
        if expected != observed {
            return Err(EpochError::OracleMismatch);
        }
        // All fallible work is complete: publish state, cache and epoch together.
        *state = candidate;
        *cache = candidate_cache;
        self.generation = next;
        self.cache_generation = Some(next);
        Ok(Some(next))
    }

    /// Choose the engine before the intervention, without consulting its oracle outcome.
    #[must_use]
    pub fn select(&self, policy: &SequentialCostPolicy, gates: u64, cone: u64) -> Engine {
        policy.select(gates, cone, self.cache_current())
    }

    /// Accept a verified intervention atomically. Even if a cache refresh is
    /// planned, the old cache becomes invalid until explicitly certified.
    /// Rejected or stale interventions leave the state and cache unchanged.
    pub fn accept_verified(
        &mut self,
        expected_generation: u64,
        same_output: bool,
        same_decision: bool,
    ) -> Result<u64, EpochError> {
        if expected_generation != self.generation {
            return Err(EpochError::StaleProposal);
        }
        if !same_output || !same_decision {
            return Err(EpochError::OracleMismatch);
        }
        let next = self.generation.checked_add(1).ok_or(EpochError::Overflow)?;
        self.generation = next;
        self.cache_generation = None;
        Ok(next)
    }
}

impl Default for CacheEpochGuard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5_sequential_cost_policy::SequentialCost;

    fn policy() -> SequentialCostPolicy {
        SequentialCostPolicy::fit_valid(
            &[SequentialCost {
                gates: 128,
                cone_gates: 4,
                full_ns: 100,
                incremental_ns: 10,
                cache_update_ns: 5,
                same_output: true,
                same_decision: true,
            }],
            1,
            0,
        )
        .unwrap()
    }

    #[test]
    fn sequential_commit_requires_rebuild_before_incremental_reuse() {
        let p = policy();
        let mut guard = CacheEpochGuard::new();
        assert_eq!(guard.select(&p, 128, 4), Engine::Full);
        guard.certify_rebuild(0).unwrap();
        assert_eq!(guard.select(&p, 128, 4), Engine::Incremental);
        assert_eq!(guard.accept_verified(0, true, true), Ok(1));
        assert_eq!(guard.select(&p, 128, 4), Engine::Full);
        assert_eq!(guard.certify_rebuild(0), Err(EpochError::InvalidGeneration));
        guard.certify_rebuild(1).unwrap();
        assert_eq!(guard.select(&p, 128, 4), Engine::Incremental);
        assert_eq!(guard.accept_verified(1, true, true), Ok(2));
        assert_eq!(guard.select(&p, 128, 4), Engine::Full);
    }

    #[test]
    fn rejected_and_stale_proposals_are_transactional() {
        let mut guard = CacheEpochGuard::new();
        guard.certify_rebuild(0).unwrap();
        assert_eq!(guard.accept_verified(0, false, true), Err(EpochError::OracleMismatch));
        assert_eq!(guard.accept_verified(0, true, false), Err(EpochError::OracleMismatch));
        assert_eq!(guard.accept_verified(1, true, true), Err(EpochError::StaleProposal));
        assert_eq!(guard.generation(), 0);
        assert!(guard.cache_current());
        guard.accept_verified(0, true, true).unwrap();
        assert_eq!(guard.accept_verified(0, true, true), Err(EpochError::StaleProposal));
        assert_eq!(guard.generation(), 1);
        assert!(!guard.cache_current());
    }


    #[test]
    fn verified_refresh_commits_cache_and_epoch_as_one_transaction() {
        let mut guard = CacheEpochGuard::new();
        let mut state = vec![false, false];
        let mut cache = 0_usize;
        for index in [0, 1] {
            let epoch = guard.generation();
            let result = guard.verified_refresh(
                epoch, &mut state, &mut cache,
                |s| { s[index] = true; true },
                |s| s.iter().filter(|&&bit| bit).count(),
                |s| (s.iter().any(|&bit| bit), s.iter().all(|&bit| bit)),
                |s, c| (*c > 0, *c == s.len()),
            );
            assert_eq!(result, Ok(Some(epoch + 1)));
            assert!(guard.cache_current());
        }
        assert_eq!(state, vec![true, true]);
        assert_eq!(cache, 2);
        assert_eq!(guard.generation(), 2);
    }

    #[test]
    fn verified_refresh_rejects_decision_mismatch_without_partial_commit() {
        let mut guard = CacheEpochGuard::new();
        guard.certify_rebuild(0).unwrap();
        let mut state = vec![false];
        let mut cache = 0_usize;
        let result = guard.verified_refresh(
            0, &mut state, &mut cache,
            |s| { s[0] = true; true },
            |s| usize::from(s[0]),
            |_| (true, true),
            |_, _| (true, false),
        );
        assert_eq!(result, Err(EpochError::OracleMismatch));
        assert_eq!(state, vec![false]);
        assert_eq!(cache, 0);
        assert_eq!(guard.generation(), 0);
        assert!(guard.cache_current());
    }

    #[test]
    fn stale_and_declined_proposals_do_not_rebuild_or_mutate() {
        let mut guard = CacheEpochGuard::new();
        let mut state = 0_u8;
        let mut cache = 0_u8;
        assert_eq!(guard.verified_refresh(
            1, &mut state, &mut cache,
            |_| panic!("stale intervention ran"),
            |_| panic!("stale rebuild ran"),
            |_| (false, false),
            |_, _| (false, false),
        ), Err(EpochError::StaleProposal));
        assert_eq!(guard.verified_refresh(
            0, &mut state, &mut cache,
            |_| false,
            |_| panic!("declined rebuild ran"),
            |_| (false, false),
            |_, _| (false, false),
        ), Ok(None));
        assert_eq!((state, cache, guard.generation()), (0, 0, 0));
    }

    #[test]
    fn transactional_refresh_supports_non_clone_cache_without_unneeded_copy() {
        struct NonCloneCache(u8);
        let mut guard = CacheEpochGuard::new();
        let mut state = 1_u8;
        let mut cache = NonCloneCache(1);
        let result = guard.verified_refresh(
            0,
            &mut state,
            &mut cache,
            |candidate| { *candidate = 2; true },
            |candidate| NonCloneCache(*candidate),
            |candidate| (*candidate > 0, *candidate == 2),
            |candidate, candidate_cache| (
                candidate_cache.0 > 0,
                candidate_cache.0 == *candidate && *candidate == 2,
            ),
        );
        assert_eq!(result, Ok(Some(1)));
        assert_eq!(state, 2);
        assert_eq!(cache.0, 2);
        assert_eq!(guard.generation(), 1);
        assert!(guard.cache_current());
    }

    #[test]
    fn exhausted_epoch_fails_before_intervention_and_keeps_cache_certified() {
        let mut guard = CacheEpochGuard {
            generation: u64::MAX,
            cache_generation: Some(u64::MAX),
        };
        let mut state = 1_u8;
        let mut cache = 1_u8;
        let result = guard.verified_refresh(
            u64::MAX,
            &mut state,
            &mut cache,
            |_| panic!("overflow ran intervention"),
            |_| panic!("overflow ran rebuild"),
            |_| panic!("overflow ran Full oracle"),
            |_, _| panic!("overflow ran Incremental oracle"),
        );
        assert_eq!(result, Err(EpochError::Overflow));
        assert_eq!((state, cache), (1, 1));
        assert_eq!(guard.generation(), u64::MAX);
        assert!(guard.cache_current());
    }

    #[test]
    fn geometry_and_uncertified_cache_fail_closed() {
        let p = policy();
        let mut guard = CacheEpochGuard::new();
        guard.certify_rebuild(0).unwrap();
        assert_eq!(guard.select(&p, 0, 0), Engine::Full);
        assert_eq!(guard.select(&p, 128, 129), Engine::Full);
        assert_eq!(guard.select(&p, 1_562_501, 4), Engine::Full);
    }
}

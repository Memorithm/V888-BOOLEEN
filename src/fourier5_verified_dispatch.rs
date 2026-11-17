//! Sequential fail-closed dispatch: never publish an unverified incremental result.
//! Synthetic circuit API; no anatomical or learning claims.
use crate::fourier5_conservative_policy::Engine;
use crate::fourier5_scale_policy::ScaleAwarePolicy;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheStamp {
    pub circuit_id: u128,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchError {
    FullOracleFailed,
    GenerationOverflow,
}

/// The full evaluator must be independent of the incremental implementation.
/// A successful intervention increments the revision, invalidating the old cache.
/// The caller must rebuild the cache for the returned revision before reuse.
pub fn dispatch_verified<T: Eq>(
    policy: &ScaleAwarePolicy,
    gates: u64,
    cone_gates: u64,
    circuit: CacheStamp,
    cache: Option<CacheStamp>,
    mut full: impl FnMut() -> Result<T, DispatchError>,
    mut incremental: impl FnMut() -> Result<T, DispatchError>,
) -> Result<(T, Engine), DispatchError> {
    let route = policy.select_for_cache_identity(
        gates,
        cone_gates,
        circuit.circuit_id,
        circuit.generation,
        cache.map(|stamp| stamp.circuit_id),
        cache.map(|stamp| stamp.generation),
    );
    if route == Engine::Full {
        return full().map(|result| (result, Engine::Full));
    }
    // Incremental errors or disagreements fall back to the independent oracle.
    // Never return an incremental value unless the oracle agrees.
    let candidate = incremental();
    let reference = full()?;
    if candidate.as_ref().is_ok_and(|value| value == &reference) {
        Ok((reference, Engine::Incremental))
    } else {
        Ok((reference, Engine::Full))
    }
}

pub fn after_accepted_intervention(stamp: CacheStamp) -> Result<CacheStamp, DispatchError> {
    let generation = ScaleAwarePolicy::next_generation(stamp.generation)
        .map_err(|_| DispatchError::GenerationOverflow)?;
    Ok(CacheStamp { generation, ..stamp })
}


/// Owns the cache revision for sequential synthetic interventions. A cache is
/// published only after its rebuild succeeds for the next circuit generation.
/// The caller must apply the corresponding circuit mutation transactionally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequentialCacheState {
    circuit: CacheStamp,
    cache: Option<CacheStamp>,
}

impl SequentialCacheState {
    pub fn new(circuit: CacheStamp, cache: Option<CacheStamp>) -> Self {
        Self {
            circuit,
            cache: cache.filter(|stamp| *stamp == circuit),
        }
    }

    pub fn circuit(&self) -> CacheStamp {
        self.circuit
    }

    pub fn cache(&self) -> Option<CacheStamp> {
        self.cache
    }

    /// Commit the revision only after a successful cache rebuild.
    /// On error the previous state is unchanged.
    pub fn accept_with_rebuild(
        &mut self,
        rebuild: impl FnOnce(CacheStamp) -> Result<(), DispatchError>,
    ) -> Result<CacheStamp, DispatchError> {
        let next = after_accepted_intervention(self.circuit)?;
        rebuild(next)?;
        self.circuit = next;
        self.cache = Some(next);
        Ok(next)
    }

    /// Stage a circuit mutation and cache rebuild on private clones. Publish both
    /// only after an independent Full oracle validates the staged result.
    /// This is an actual transaction for Clone state, unlike metadata-only
    /// accept_with_rebuild. No state is published on any failure.
    pub fn transact_verified<C: Clone, K: Clone, T: Eq>(
        &mut self,
        circuit: &mut C,
        cache: &mut K,
        mutate: impl FnOnce(&mut C) -> Result<(), DispatchError>,
        rebuild: impl FnOnce(&C, &mut K) -> Result<(), DispatchError>,
        full_oracle: impl FnOnce(&C) -> Result<T, DispatchError>,
        cache_oracle: impl FnOnce(&C, &K) -> Result<T, DispatchError>,
    ) -> Result<CacheStamp, DispatchError> {
        let next = after_accepted_intervention(self.circuit)?;
        let mut staged_circuit = circuit.clone();
        let mut staged_cache = cache.clone();
        mutate(&mut staged_circuit)?;
        rebuild(&staged_circuit, &mut staged_cache)?;
        let reference = full_oracle(&staged_circuit)?;
        if cache_oracle(&staged_circuit, &staged_cache)? != reference {
            return Err(DispatchError::FullOracleFailed);
        }
        *circuit = staged_circuit;
        *cache = staged_cache;
        self.circuit = next;
        self.cache = Some(next);
        Ok(next)
    }

    /// Build a fresh cache instead of cloning the old one. The staged circuit and
    /// newly constructed cache remain private until independent Full verification.
    /// This avoids cloning a potentially large stale cache on every intervention.
    /// A failed build or oracle check leaves all published state unchanged.
    pub fn transact_with_fresh_cache<C: Clone, K, T: Eq>(
        &mut self,
        circuit: &mut C,
        cache: &mut K,
        mutate: impl FnOnce(&mut C) -> Result<(), DispatchError>,
        build_cache: impl FnOnce(&C) -> Result<K, DispatchError>,
        full_oracle: impl FnOnce(&C) -> Result<T, DispatchError>,
        cache_oracle: impl FnOnce(&C, &K) -> Result<T, DispatchError>,
    ) -> Result<CacheStamp, DispatchError> {
        let next = after_accepted_intervention(self.circuit)?;
        let mut staged = circuit.clone();
        mutate(&mut staged)?;
        let fresh = build_cache(&staged)?;
        let reference = full_oracle(&staged)?;
        if cache_oracle(&staged, &fresh)? != reference {
            return Err(DispatchError::FullOracleFailed);
        }
        *circuit = staged;
        *cache = fresh;
        self.circuit = next;
        self.cache = Some(next);
        Ok(next)
    }

    /// Apply a sequence of interventions atomically. Each staged step receives
    /// a newly constructed cache and an independent Full comparison. A failure
    /// at any step leaves the published circuit, cache and revision unchanged.
    /// Only the final verified cache is published; K need not implement Clone.
    pub fn transact_sequential_fresh<C: Clone, K, T: Eq, I>(
        &mut self,
        circuit: &mut C,
        cache: &mut K,
        interventions: I,
        mut mutate: impl FnMut(&mut C, I::Item) -> Result<(), DispatchError>,
        mut build_cache: impl FnMut(&C) -> Result<K, DispatchError>,
        mut full_oracle: impl FnMut(&C) -> Result<T, DispatchError>,
        mut cache_oracle: impl FnMut(&C, &K) -> Result<T, DispatchError>,
    ) -> Result<CacheStamp, DispatchError>
    where
        I: IntoIterator,
    {
        let mut staged = circuit.clone();
        let mut next = self.circuit;
        let mut last_cache = None;
        for intervention in interventions {
            next = after_accepted_intervention(next)?;
            mutate(&mut staged, intervention)?;
            let fresh = build_cache(&staged)?;
            let reference = full_oracle(&staged)?;
            if cache_oracle(&staged, &fresh)? != reference {
                return Err(DispatchError::FullOracleFailed);
            }
            last_cache = Some(fresh);
        }
        if let Some(fresh) = last_cache {
            *circuit = staged;
            *cache = fresh;
            self.circuit = next;
            self.cache = Some(next);
        }
        Ok(self.circuit)
    }

    /// Commit an accepted intervention without a rebuilt cache: Full is required.
    pub fn accept_without_cache(&mut self) -> Result<CacheStamp, DispatchError> {
        let next = after_accepted_intervention(self.circuit)?;
        self.circuit = next;
        self.cache = None;
        Ok(next)
    }

    /// Reject a rebuild from a different circuit or generation.
    pub fn publish_rebuilt_cache(&mut self, stamp: CacheStamp) -> bool {
        if stamp != self.circuit {
            return false;
        }
        self.cache = Some(stamp);
        true
    }

    /// Dispatch using the state-owned circuit/cache stamps, preventing callers
    /// from accidentally pairing a new circuit revision with an old cache.
    /// Full remains an independent oracle for every incremental result.
    pub fn dispatch<T: Eq>(
        &self,
        policy: &ScaleAwarePolicy,
        gates: u64,
        cone_gates: u64,
        full: impl FnMut() -> Result<T, DispatchError>,
        incremental: impl FnMut() -> Result<T, DispatchError>,
    ) -> Result<(T, Engine), DispatchError> {
        dispatch_verified(
            policy,
            gates,
            cone_gates,
            self.circuit,
            self.cache,
            full,
            incremental,
        )
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5_conservative_policy::PairedObservation;
    use std::cell::Cell;

    fn policy() -> ScaleAwarePolicy {
        ScaleAwarePolicy::fit_valid(&[PairedObservation {
            gates: 128, cone_gates: 4, full_ns: 100, incremental_ns: 10,
            same_output: true, same_decision: true,
        }], 1, 0).unwrap()
    }

    #[test]
    fn oracle_rejects_wrong_incremental_output() {
        let stamp = CacheStamp { circuit_id: 7, generation: 2 };
        let result = dispatch_verified(&policy(), 128, 4, stamp, Some(stamp),
            || Ok(42), || Ok(41));
        assert_eq!(result, Ok((42, Engine::Full)));
    }

    #[test]
    fn stale_and_cross_circuit_cache_never_execute_incremental() {
        let stamp = CacheStamp { circuit_id: 7, generation: 2 };
        for other in [
            CacheStamp { circuit_id: 8, generation: 2 },
            CacheStamp { circuit_id: 7, generation: 1 },
        ] {
            let calls = Cell::new(0);
            let result = dispatch_verified(&policy(), 128, 4, stamp, Some(other),
                || Ok(42), || { calls.set(calls.get() + 1); Ok(42) });
            assert_eq!(result, Ok((42, Engine::Full)));
            assert_eq!(calls.get(), 0);
        }
    }

    #[test]
    fn accepted_mutation_invalidates_old_cache_until_rebuild() {
        let before = CacheStamp { circuit_id: 7, generation: 2 };
        let after = after_accepted_intervention(before).unwrap();
        assert_eq!(after.generation, 3);
        assert_eq!(dispatch_verified(&policy(), 128, 4, after, Some(before),
            || Ok(1), || Err(DispatchError::FullOracleFailed)), Ok((1, Engine::Full)));
        assert_eq!(dispatch_verified(&policy(), 128, 4, after, Some(after),
            || Ok(1), || Ok(1)), Ok((1, Engine::Incremental)));
        assert_eq!(after_accepted_intervention(CacheStamp { generation: u64::MAX, ..before }),
            Err(DispatchError::GenerationOverflow));
    }

    #[test]
    fn full_oracle_failure_is_not_hidden() {
        let stamp = CacheStamp { circuit_id: 7, generation: 2 };
        assert_eq!(dispatch_verified::<u8>(&policy(), 128, 4, stamp, Some(stamp),
            || Err(DispatchError::FullOracleFailed), || Ok(1)),
            Err(DispatchError::FullOracleFailed));
    }

    #[test]
    fn sequential_rebuild_commits_only_on_success() {
        let initial = CacheStamp { circuit_id: 17, generation: 5 };
        let mut state = SequentialCacheState::new(initial, Some(initial));
        assert_eq!(
            state.accept_with_rebuild(|_| Err(DispatchError::FullOracleFailed)),
            Err(DispatchError::FullOracleFailed)
        );
        assert_eq!(state.circuit(), initial);
        assert_eq!(state.cache(), Some(initial));
        let next = state.accept_with_rebuild(|stamp| {
            assert_eq!(stamp.generation, 6);
            Ok(())
        }).unwrap();
        assert_eq!(state.circuit(), next);
        assert_eq!(state.cache(), Some(next));
        assert!(!state.publish_rebuilt_cache(initial));
        assert_eq!(state.cache(), Some(next));
    }

    #[test]
    fn sequential_cache_invalidation_and_overflow_are_fail_closed() {
        let initial = CacheStamp { circuit_id: 19, generation: 9 };
        let mut state = SequentialCacheState::new(initial, Some(initial));
        let next = state.accept_without_cache().unwrap();
        assert_eq!(state.cache(), None);
        assert!(!state.publish_rebuilt_cache(initial));
        assert!(state.publish_rebuilt_cache(next));
        assert_eq!(state.cache(), Some(next));
        let terminal = CacheStamp { generation: u64::MAX, ..initial };
        let mut saturated = SequentialCacheState::new(terminal, Some(terminal));
        assert_eq!(saturated.accept_without_cache(), Err(DispatchError::GenerationOverflow));
        assert_eq!(saturated.cache(), Some(terminal));
        assert_eq!(saturated.circuit(), terminal);
    }


    #[test]
    fn transactional_mutation_rolls_back_on_oracle_disagreement() {
        let initial = CacheStamp { circuit_id: 29, generation: 4 };
        let mut state = SequentialCacheState::new(initial, Some(initial));
        let mut circuit = vec![false, true];
        let mut cache = vec![false, true];
        let result = state.transact_verified(
            &mut circuit, &mut cache,
            |c| { c[0] = true; Ok(()) },
            |c, k| { *k = c.clone(); k[1] = false; Ok(()) },
            |c| Ok(c.clone()),
            |_, k| Ok(k.clone()),
        );
        assert_eq!(result, Err(DispatchError::FullOracleFailed));
        assert_eq!(circuit, vec![false, true]);
        assert_eq!(cache, vec![false, true]);
        assert_eq!(state.circuit(), initial);
        assert_eq!(state.cache(), Some(initial));
    }

    #[test]
    fn transactional_mutation_publishes_only_verified_generation() {
        let initial = CacheStamp { circuit_id: 31, generation: 0 };
        let mut state = SequentialCacheState::new(initial, Some(initial));
        let mut circuit = vec![false, false];
        let mut cache = circuit.clone();
        let next = state.transact_verified(
            &mut circuit, &mut cache,
            |c| { c[1] = true; Ok(()) },
            |c, k| { *k = c.clone(); Ok(()) },
            |c| Ok(c.clone()),
            |_, k| Ok(k.clone()),
        ).unwrap();
        assert_eq!(next.generation, 1);
        assert_eq!(circuit, vec![false, true]);
        assert_eq!(cache, circuit);
        assert_eq!(state.cache(), Some(next));
        assert!(!state.publish_rebuilt_cache(initial));
    }

    #[test]
    fn fresh_cache_transaction_supports_non_clone_cache_and_rollback() {
        struct NonClone(Vec<bool>);
        let initial = CacheStamp { circuit_id: 37, generation: 3 };
        let mut state = SequentialCacheState::new(initial, Some(initial));
        let mut circuit = vec![false, true];
        let mut cache = NonClone(circuit.clone());
        let failed = state.transact_with_fresh_cache(
            &mut circuit, &mut cache,
            |c| { c[0] = true; Ok(()) },
            |c| Ok(NonClone(vec![false; c.len()])),
            |c| Ok(c.clone()),
            |_, k| Ok(k.0.clone()),
        );
        assert_eq!(failed, Err(DispatchError::FullOracleFailed));
        assert_eq!(circuit, vec![false, true]);
        assert_eq!(cache.0, circuit);
        assert_eq!(state.circuit(), initial);
        let next = state.transact_with_fresh_cache(
            &mut circuit, &mut cache,
            |c| { c[0] = true; Ok(()) },
            |c| Ok(NonClone(c.clone())),
            |c| Ok(c.clone()),
            |_, k| Ok(k.0.clone()),
        ).unwrap();
        assert_eq!(next.generation, 4);
        assert_eq!(circuit, vec![true, true]);
        assert_eq!(cache.0, circuit);
        assert_eq!(state.cache(), Some(next));
    }

    #[test]
    fn fresh_cache_builder_failure_does_not_publish_revision() {
        struct NonClone;
        let initial = CacheStamp { circuit_id: 41, generation: 0 };
        let mut state = SequentialCacheState::new(initial, Some(initial));
        let mut circuit = 1_u8;
        let mut cache = NonClone;
        let result = state.transact_with_fresh_cache(
            &mut circuit, &mut cache,
            |c| { *c = 2; Ok(()) },
            |_| Err(DispatchError::FullOracleFailed),
            |c| Ok(*c),
            |_, _| Ok(2_u8),
        );
        assert_eq!(result, Err(DispatchError::FullOracleFailed));
        assert_eq!(circuit, 1);
        assert_eq!(state.circuit(), initial);
        assert_eq!(state.cache(), Some(initial));
    }

    #[test]
    fn sequential_fresh_batch_verifies_every_step_and_rolls_back() {
        struct NonClone(Vec<bool>);
        let initial = CacheStamp { circuit_id: 51, generation: 3 };
        let mut state = SequentialCacheState::new(initial, Some(initial));
        let mut circuit = vec![false, false, false];
        let mut cache = NonClone(circuit.clone());
        let mut checked = 0;
        let rejected = state.transact_sequential_fresh(
            &mut circuit, &mut cache, [0_usize, 1],
            |c, i| { c[i] = true; Ok(()) },
            |c| Ok(NonClone(c.clone())),
            |c| { checked += 1; Ok(c.clone()) },
            |_, k| {
                let mut value = k.0.clone();
                if value[1] { value[1] = false; }
                Ok(value)
            },
        );
        assert_eq!(rejected, Err(DispatchError::FullOracleFailed));
        assert_eq!(checked, 2);
        assert_eq!(circuit, vec![false; 3]);
        assert_eq!(cache.0, circuit);
        assert_eq!(state.circuit(), initial);
        let accepted = state.transact_sequential_fresh(
            &mut circuit, &mut cache, [0_usize, 2],
            |c, i| { c[i] = true; Ok(()) },
            |c| Ok(NonClone(c.clone())),
            |c| Ok(c.clone()),
            |_, k| Ok(k.0.clone()),
        ).unwrap();
        assert_eq!(accepted.generation, 5);
        assert_eq!(circuit, vec![true, false, true]);
        assert_eq!(cache.0, circuit);
        assert_eq!(state.cache(), Some(accepted));
    }

    #[test]
    fn sequential_fresh_empty_and_overflow_are_non_mutating() {
        struct NonClone;
        let initial = CacheStamp { circuit_id: 52, generation: u64::MAX };
        let mut state = SequentialCacheState::new(initial, Some(initial));
        let mut circuit = 0_u8;
        let mut cache = NonClone;
        let empty = state.transact_sequential_fresh(
            &mut circuit, &mut cache, std::iter::empty::<u8>(),
            |c, _| { *c = 1; Ok(()) },
            |_| Ok(NonClone), |c| Ok(*c), |_, _| Ok(0_u8),
        );
        assert_eq!(empty, Ok(initial));
        let overflow = state.transact_sequential_fresh(
            &mut circuit, &mut cache, [1_u8],
            |c, value| { *c = value; Ok(()) },
            |_| Ok(NonClone), |c| Ok(*c), |_, _| Ok(1_u8),
        );
        assert_eq!(overflow, Err(DispatchError::GenerationOverflow));
        assert_eq!(circuit, 0);
        assert_eq!(state.circuit(), initial);
        assert_eq!(state.cache(), Some(initial));
    }

    #[test]
    fn state_owned_dispatch_tracks_sequential_cache_rebuilds() {
        let initial = CacheStamp { circuit_id: 23, generation: 0 };
        let mut state = SequentialCacheState::new(initial, Some(initial));
        let calls = Cell::new(0);
        assert_eq!(state.dispatch(&policy(), 128, 4,
            || Ok(9), || { calls.set(calls.get() + 1); Ok(9) }),
            Ok((9, Engine::Incremental)));
        assert_eq!(calls.get(), 1);

        let next = state.accept_without_cache().unwrap();
        assert_eq!(state.dispatch(&policy(), 128, 4,
            || Ok(10), || { calls.set(calls.get() + 1); Ok(10) }),
            Ok((10, Engine::Full)));
        assert_eq!(calls.get(), 1);
        assert!(!state.publish_rebuilt_cache(initial));
        assert!(state.publish_rebuilt_cache(next));
        assert_eq!(state.dispatch(&policy(), 128, 4,
            || Ok(10), || { calls.set(calls.get() + 1); Ok(11) }),
            Ok((10, Engine::Full)));
        assert_eq!(calls.get(), 2);
        assert_eq!(state.dispatch(&policy(), 128, 4,
            || Ok(10), || { calls.set(calls.get() + 1); Ok(10) }),
            Ok((10, Engine::Incremental)));
        assert_eq!(calls.get(), 3);
    }

}

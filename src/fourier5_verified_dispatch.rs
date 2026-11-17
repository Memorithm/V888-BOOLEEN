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

}

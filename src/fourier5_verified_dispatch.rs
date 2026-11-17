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
}

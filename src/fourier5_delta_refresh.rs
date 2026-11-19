//! Atomic delta-refresh for sequential synthetic Boolean cache interventions.
//! Unlike full cache rebuild, refresh mutates a cloned cache candidate. Neither
//! live state nor live cache changes unless the independent Full oracle agrees.
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaOutcome { Declined, OracleMismatch, Committed }

#[derive(Clone, Debug)]
pub struct DeltaMeasure {
    pub outcome: DeltaOutcome,
    pub elapsed: Duration,
    pub refresh: Duration,
    pub full: Duration,
    pub incremental: Duration,
}

/// Transactionally apply a state edit and an exact incremental cache update.
/// Refresh must handle all dependencies of the intervention. Full is independent
/// of the cache, and its result must include any decision bits being protected.
pub fn verified_delta<S, C, O, I, R, F, D>(
    state: &mut S, cache: &mut C, intervene: I, refresh: R,
    full: F, incremental: D,
) -> DeltaMeasure
where
    S: Clone, C: Clone, O: Eq,
    I: FnOnce(&mut S) -> bool,
    R: FnOnce(&S, &mut C),
    F: FnOnce(&S) -> O,
    D: FnOnce(&S, &C) -> O,
{
    let started = Instant::now();
    let mut candidate = state.clone();
    if !intervene(&mut candidate) {
        return DeltaMeasure {
            outcome: DeltaOutcome::Declined, elapsed: started.elapsed(),
            refresh: Duration::ZERO, full: Duration::ZERO,
            incremental: Duration::ZERO,
        };
    }
    let mut candidate_cache = cache.clone();
    let phase = Instant::now();
    refresh(&candidate, &mut candidate_cache);
    let refresh_time = phase.elapsed();
    let phase = Instant::now();
    let expected = full(&candidate);
    let full_time = phase.elapsed();
    let phase = Instant::now();
    let observed = incremental(&candidate, &candidate_cache);
    let incremental_time = phase.elapsed();
    let outcome = if expected == observed {
        *state = candidate;
        *cache = candidate_cache;
        DeltaOutcome::Committed
    } else {
        DeltaOutcome::OracleMismatch
    };
    DeltaMeasure {
        outcome, elapsed: started.elapsed(), refresh: refresh_time,
        full: full_time, incremental: incremental_time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent scalar oracle: recompute from state, not the cached count.
    fn oracle(bits: &[bool]) -> (usize, bool) {
        let count = bits.iter().filter(|&&bit| bit).count();
        (count, count >= 3)
    }

    #[test]
    fn heterogeneous_sequential_updates_match_independent_oracle() {
        let mut bits = vec![false; 7];
        let mut cache = 0_usize;
        for index in [0, 6, 2, 2, 4, 0, 5] {
            let before = bits[index];
            let result = verified_delta(
                &mut bits, &mut cache,
                |candidate| { candidate[index] = !candidate[index]; true },
                |candidate, count| {
                    if before { *count -= 1; } else { *count += 1; }
                    assert_eq!(*count, candidate.iter().filter(|&&b| b).count());
                },
                |candidate| oracle(candidate),
                |_, count| (*count, *count >= 3),
            );
            assert_eq!(result.outcome, DeltaOutcome::Committed);
            assert_eq!(cache, oracle(&bits).0);
            assert!(result.elapsed >= result.refresh + result.full + result.incremental);
        }
    }

    #[test]
    fn wrong_delta_and_wrong_decision_roll_back_both_objects() {
        for corrupt_decision in [false, true] {
            let mut bits = vec![false, false, false];
            let mut cache = 0_usize;
            let result = verified_delta(
                &mut bits, &mut cache,
                |candidate| { candidate[0] = true; true },
                |_, count| *count = if corrupt_decision { 1 } else { 9 },
                |candidate| oracle(candidate),
                |_, count| (*count, corrupt_decision),
            );
            assert_eq!(result.outcome, DeltaOutcome::OracleMismatch);
            assert_eq!(bits, vec![false; 3]);
            assert_eq!(cache, 0);
        }
    }

    #[test]
    fn decline_skips_cache_refresh_and_oracles() {
        let mut state = 1_u8;
        let mut cache = 1_u8;
        let result = verified_delta::<_, _, u8, _, _, _, _>(
            &mut state, &mut cache, |_| false,
            |_, _| panic!("refresh called"), |_| panic!("full called"),
            |_, _| panic!("incremental called"),
        );
        assert_eq!(result.outcome, DeltaOutcome::Declined);
        assert_eq!((state, cache), (1, 1));
    }
}

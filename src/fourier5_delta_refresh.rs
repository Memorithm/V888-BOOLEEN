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

 
/// Fallible transactional refresh. An error from the intervention or cache
/// updater aborts before the live state/cache are published.
pub fn try_verified_delta<S, C, O, E, I, R, F, D>(
    state: &mut S,
    cache: &mut C,
    intervene: I,
    refresh: R,
    full: F,
    incremental: D,
) -> Result<DeltaMeasure, E>
where
    S: Clone,
    C: Clone,
    O: Eq,
    I: FnOnce(&mut S) -> Result<bool, E>,
    R: FnOnce(&S, &mut C) -> Result<(), E>,
    F: FnOnce(&S) -> Result<O, E>,
    D: FnOnce(&S, &C) -> Result<O, E>,
{
    let started = Instant::now();
    let mut candidate = state.clone();
    if !intervene(&mut candidate)? {
        return Ok(DeltaMeasure {
            outcome: DeltaOutcome::Declined,
            elapsed: started.elapsed(),
            refresh: Duration::ZERO,
            full: Duration::ZERO,
            incremental: Duration::ZERO,
        });
    }
    let mut candidate_cache = cache.clone();
    let phase = Instant::now();
    refresh(&candidate, &mut candidate_cache)?;
    let refresh_time = phase.elapsed();
    let phase = Instant::now();
    let expected = full(&candidate)?;
    let full_time = phase.elapsed();
    let phase = Instant::now();
    let observed = incremental(&candidate, &candidate_cache)?;
    let incremental_time = phase.elapsed();
    let outcome = if expected == observed {
        *state = candidate;
        *cache = candidate_cache;
        DeltaOutcome::Committed
    } else {
        DeltaOutcome::OracleMismatch
    };
    Ok(DeltaMeasure {
        outcome,
        elapsed: started.elapsed(),
        refresh: refresh_time,
        full: full_time,
        incremental: incremental_time,
    })
}

#[cfg(test)]
mod fallible_tests {
    use super::*;

    #[test]
    fn errors_at_each_stage_leave_both_live_objects_unchanged() {
        for failing_stage in 0..4 {
            let mut state = 1_u8;
            let mut cache = 1_u8;
            let result = try_verified_delta(
                &mut state,
                &mut cache,
                |candidate| {
                    *candidate = 2;
                    if failing_stage == 0 { Err("intervention") } else { Ok(true) }
                },
                |_, candidate_cache| {
                    *candidate_cache = 2;
                    if failing_stage == 1 { Err("refresh") } else { Ok(()) }
                },
                |_| {
                    if failing_stage == 2 { Err("full") } else { Ok(2_u8) }
                },
                |_, _| {
                    if failing_stage == 3 { Err("incremental") } else { Ok(2_u8) }
                },
            );
            assert!(result.is_err());
            assert_eq!((state, cache), (1, 1));
        }
    }

    #[test]
    fn success_commits_and_mismatch_rolls_back() {
        for observed in [2_u8, 3_u8] {
            let mut state = 1_u8;
            let mut cache = 1_u8;
            let result = try_verified_delta(
                &mut state,
                &mut cache,
                |candidate| { *candidate = 2; Ok::<_, ()>(true) },
                |_, candidate_cache| { *candidate_cache = 2; Ok::<_, ()>(()) },
                |candidate| Ok::<_, ()>(*candidate),
                |_, _| Ok::<_, ()>(observed),
            ).unwrap();
            assert_eq!(result.outcome, if observed == 2 {
                DeltaOutcome::Committed
            } else {
                DeltaOutcome::OracleMismatch
            });
            assert_eq!((state, cache), if observed == 2 { (2, 2) } else { (1, 1) });
        }
    }
}


/// A sequence is published atomically: even a late mismatch or declined edit
/// discards every earlier candidate commit. Each accepted step is independently
/// checked against Full before the next edit observes the updated candidate cache.
#[derive(Clone, Debug)]
pub struct DeltaBatchMeasure {
    pub outcome: DeltaOutcome,
    pub steps: Vec<DeltaMeasure>,
}

/// Apply an ordered heterogeneous edit batch without exposing intermediate states.
/// An error from any stage leaves both live objects untouched. Timings include
/// oracle verification, not merely the cheaper incremental path.
pub fn try_verified_delta_batch<S, C, O, E, T, I, R, F, D>(
    state: &mut S,
    cache: &mut C,
    edits: impl IntoIterator<Item = T>,
    mut intervene: I,
    mut refresh: R,
    mut full: F,
    mut incremental: D,
) -> Result<DeltaBatchMeasure, E>
where
    S: Clone,
    C: Clone,
    O: Eq,
    I: FnMut(&mut S, T) -> Result<bool, E>,
    R: FnMut(&S, &mut C) -> Result<(), E>,
    F: FnMut(&S) -> Result<O, E>,
    D: FnMut(&S, &C) -> Result<O, E>,
{
    let mut candidate_state = state.clone();
    let mut candidate_cache = cache.clone();
    let mut steps = Vec::new();
    for edit in edits {
        let measure = try_verified_delta(
            &mut candidate_state,
            &mut candidate_cache,
            |s| intervene(s, edit),
            |s, c| refresh(s, c),
            |s| full(s),
            |s, c| incremental(s, c),
        )?;
        let outcome = measure.outcome;
        steps.push(measure);
        if outcome != DeltaOutcome::Committed {
            return Ok(DeltaBatchMeasure { outcome, steps });
        }
    }
    if steps.is_empty() {
        return Ok(DeltaBatchMeasure {
            outcome: DeltaOutcome::Declined,
            steps,
        });
    }
    *state = candidate_state;
    *cache = candidate_cache;
    Ok(DeltaBatchMeasure {
        outcome: DeltaOutcome::Committed,
        steps,
    })
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    fn run(edits: &[usize], corrupt_at: Option<usize>) -> (DeltaBatchMeasure, Vec<bool>, usize) {
        let mut state = vec![false; 5];
        let mut cache = 0_usize;
        let mut step = 0_usize;
        let result = try_verified_delta_batch(
            &mut state,
            &mut cache,
            edits.iter().copied(),
            |bits, index| {
                bits[index] = !bits[index];
                Ok::<_, ()>(true)
            },
            |bits, count| {
                *count = bits.iter().filter(|&&bit| bit).count();
                step += 1;
                if corrupt_at == Some(step) {
                    *count += 1;
                }
                Ok::<_, ()>(())
            },
            |bits| Ok::<_, ()>(bits.iter().filter(|&&bit| bit).count()),
            |_, count| Ok::<_, ()>(*count),
        ).unwrap();
        (result, state, cache)
    }

    #[test]
    fn sequential_batch_commits_exact_cache_and_state() {
        let (result, state, cache) = run(&[0, 2, 0, 4], None);
        assert_eq!(result.outcome, DeltaOutcome::Committed);
        assert_eq!(result.steps.len(), 4);
        assert_eq!(state, vec![false, false, true, false, true]);
        assert_eq!(cache, 2);
    }

    #[test]
    fn late_mismatch_rolls_back_every_earlier_step() {
        let (result, state, cache) = run(&[0, 2, 4], Some(3));
        assert_eq!(result.outcome, DeltaOutcome::OracleMismatch);
        assert_eq!(result.steps.len(), 3);
        assert_eq!(state, vec![false; 5]);
        assert_eq!(cache, 0);
    }

    #[test]
    fn empty_batch_does_not_publish() {
        let (result, state, cache) = run(&[], None);
        assert_eq!(result.outcome, DeltaOutcome::Declined);
        assert!(result.steps.is_empty());
        assert_eq!(state, vec![false; 5]);
        assert_eq!(cache, 0);
    }
}

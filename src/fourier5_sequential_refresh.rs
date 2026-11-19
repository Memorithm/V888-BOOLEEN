//! Exact sequential cache refresh harness for synthetic Boolean workloads.
//! Measures the entire accepted intervention path, not CSV ingestion.
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepMeasure {
    pub accepted: bool,
    pub oracle_equal: bool,
    pub elapsed: Duration,
    pub refresh: Duration,
    /// Time spent computing the independent Full oracle.
    pub full: Duration,
    /// Time spent evaluating the refreshed Incremental cache.
    pub incremental: Duration,
}

/// Apply an intervention to a *candidate* state. Commit only after an independently
/// computed Full oracle agrees with the incrementally evaluated refreshed cache.
/// A rejected candidate cannot mutate either the live state or its cache.
pub fn verified_step<S, C, O, I, B, F, D>(
    state: &mut S,
    cache: &mut C,
    intervention: I,
    rebuild: B,
    full: F,
    incremental: D,
) -> StepMeasure
where
    S: Clone,
    O: Eq,
    I: FnOnce(&mut S) -> bool,
    B: FnOnce(&S) -> C,
    F: FnOnce(&S) -> O,
    D: FnOnce(&S, &C) -> O,
{
    let started = Instant::now();
    let mut candidate = state.clone();
    if !intervention(&mut candidate) {
        return StepMeasure {
            accepted: false,
            oracle_equal: true,
            elapsed: started.elapsed(),
            refresh: Duration::ZERO,
            full: Duration::ZERO,
            incremental: Duration::ZERO,
        };
    }
    let refresh_started = Instant::now();
    let candidate_cache = rebuild(&candidate);
    let refresh = refresh_started.elapsed();
    let full_started = Instant::now();
    let expected = full(&candidate);
    let full_elapsed = full_started.elapsed();
    let incremental_started = Instant::now();
    let actual = incremental(&candidate, &candidate_cache);
    let incremental_elapsed = incremental_started.elapsed();
    let oracle_equal = expected == actual;
    if oracle_equal {
        *state = candidate;
        *cache = candidate_cache;
    }
    StepMeasure {
        accepted: oracle_equal,
        oracle_equal,
        elapsed: started.elapsed(),
        refresh,
        full: full_elapsed,
        incremental: incremental_elapsed,
    }
}

/// Execute a bounded sequence of interventions, rebuilding and independently
/// verifying the cache after every accepted step. Stop at the first rejection;
/// later interventions are never evaluated. The live state remains at the last
/// verified generation, rather than rolling back earlier successful steps.
pub fn verified_sequence<S, C, O, T, I, B, F, D>(
    state: &mut S,
    cache: &mut C,
    interventions: &[T],
    mut intervene: I,
    mut rebuild: B,
    mut full: F,
    mut incremental: D,
) -> SequenceMeasure
where
    S: Clone,
    O: Eq,
    I: FnMut(&mut S, &T) -> bool,
    B: FnMut(&S) -> C,
    F: FnMut(&S) -> O,
    D: FnMut(&S, &C) -> O,
{
    let started = Instant::now();
    let mut result = SequenceMeasure::default();
    for item in interventions {
        let step = verified_step(
            state,
            cache,
            |candidate| intervene(candidate, item),
            |candidate| rebuild(candidate),
            |candidate| full(candidate),
            |candidate, candidate_cache| incremental(candidate, candidate_cache),
        );
        result.attempted += 1;
        result.refresh += step.refresh;
        result.full += step.full;
        result.incremental += step.incremental;
        if !step.accepted {
            result.stopped_on_rejection = true;
            break;
        }
        result.committed += 1;
    }
    result.elapsed = started.elapsed();
    result
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SequenceMeasure {
    pub attempted: usize,
    pub committed: usize,
    pub stopped_on_rejection: bool,
    pub elapsed: Duration,
    pub refresh: Duration,
    pub full: Duration,
    pub incremental: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_rebuilds_each_committed_generation_and_stops_on_mismatch() {
        let mut state = vec![false; 4];
        let mut cache = 0_usize;
        let mut visited = Vec::new();
        let result = verified_sequence(
            &mut state,
            &mut cache,
            &[0_usize, 1, 2, 3],
            |candidate, index| {
                visited.push(*index);
                candidate[*index] = true;
                true
            },
            |candidate| candidate.iter().filter(|&&v| v).count(),
            |candidate| candidate.iter().filter(|&&v| v).count(),
            |_, refreshed| if *refreshed == 3 { 99 } else { *refreshed },
        );
        assert_eq!(visited, [0, 1, 2]);
        assert_eq!((result.attempted, result.committed), (3, 2));
        assert!(result.stopped_on_rejection);
        assert_eq!(state, [true, true, false, false]);
        assert_eq!(cache, 2);
        assert!(result.elapsed >= result.refresh + result.full + result.incremental);
    }

    #[test]
    fn empty_sequence_does_not_touch_cache_or_oracles() {
        let mut state = 7_u8;
        let mut cache = 7_u8;
        let result = verified_sequence::<_, _, u8, u8, _, _, _, _>(
            &mut state,
            &mut cache,
            &[],
            |_, _| panic!("unexpected intervention"),
            |_| panic!("unexpected rebuild"),
            |_| panic!("unexpected full"),
            |_, _| panic!("unexpected incremental"),
        );
        assert_eq!((result.attempted, result.committed), (0, 0));
        assert!(!result.stopped_on_rejection);
        assert_eq!((state, cache), (7, 7));
    }

    #[test]
    fn sequential_refresh_commits_each_generation() {
        let mut state = vec![false, false, false];
        let mut cache = 0_usize;
        for index in [0, 2, 1] {
            let result = verified_step(
                &mut state,
                &mut cache,
                |s| {
                    s[index] = true;
                    true
                },
                |s| s.iter().filter(|&&b| b).count(),
                |s| s.iter().filter(|&&b| b).count(),
                |_, c| *c,
            );
            assert!(result.accepted);
            assert!(result.elapsed >= result.refresh + result.full + result.incremental);
        }
        assert_eq!(cache, 3);
        assert_eq!(state, vec![true; 3]);
    }

    #[test]
    fn mismatch_rolls_back_state_and_cache() {
        let mut state = vec![false];
        let mut cache = 0_usize;
        let result = verified_step(
            &mut state,
            &mut cache,
            |s| {
                s[0] = true;
                true
            },
            |_| 0,
            |s| usize::from(s[0]),
            |_, c| *c,
        );
        assert!(!result.accepted);
        assert!(!result.oracle_equal);
        assert_eq!(state, vec![false]);
        assert_eq!(cache, 0);
    }

    #[test]
    fn declined_intervention_never_rebuilds() {
        let mut state = 0_u8;
        let mut cache = 0_u8;
        let result = verified_step(
            &mut state,
            &mut cache,
            |_| false,
            |_| panic!("rebuild on declined intervention"),
            |_| 0_u8,
            |_, _| 0_u8,
        );
        assert!(!result.accepted);
        assert!(result.oracle_equal);
        assert_eq!(result.full, Duration::ZERO);
        assert_eq!(result.incremental, Duration::ZERO);
    }

    #[test]
    fn oracle_and_incremental_execute_once_in_order() {
        use std::cell::RefCell;
        let phases = RefCell::new(Vec::new());
        let mut state = 1_u8;
        let mut cache = 1_u8;
        let measure = verified_step(
            &mut state,
            &mut cache,
            |candidate| {
                *candidate = 2;
                true
            },
            |candidate| {
                phases.borrow_mut().push("rebuild");
                *candidate
            },
            |candidate| {
                phases.borrow_mut().push("full");
                *candidate
            },
            |_, refreshed| {
                phases.borrow_mut().push("incremental");
                *refreshed
            },
        );
        assert_eq!(*phases.borrow(), ["rebuild", "full", "incremental"]);
        assert!(measure.accepted);
        assert!(measure.elapsed >= measure.refresh + measure.full + measure.incremental);
        assert_eq!((state, cache), (2, 2));
    }
    #[test]
    fn distinct_output_and_decision_mismatches_fail_closed() {
        for observed in [(false, true), (true, false), (false, false)] {
            let mut state = false;
            let mut cache = false;
            let result = verified_step(
                &mut state,
                &mut cache,
                |candidate| {
                    *candidate = true;
                    true
                },
                |candidate| *candidate,
                |candidate| (*candidate, *candidate),
                |_, _| observed,
            );
            assert!(!result.accepted);
            assert!(!result.oracle_equal);
            assert!(!state);
            assert!(!cache);
        }
    }

    #[test]
    fn non_clone_cache_commits_only_after_oracle_agreement() {
        struct NonCloneCache(u8);
        let mut state = 0_u8;
        let mut cache = NonCloneCache(0);
        let first = verified_step(
            &mut state,
            &mut cache,
            |candidate| {
                *candidate = 1;
                true
            },
            |candidate| NonCloneCache(*candidate),
            |candidate| *candidate,
            |_, candidate_cache| candidate_cache.0,
        );
        assert!(first.accepted);
        assert_eq!((state, cache.0), (1, 1));
        let rejected = verified_step(
            &mut state,
            &mut cache,
            |candidate| {
                *candidate = 2;
                true
            },
            |_| NonCloneCache(99),
            |candidate| *candidate,
            |_, candidate_cache| candidate_cache.0,
        );
        assert!(!rejected.accepted);
        assert_eq!((state, cache.0), (1, 1));
    }
}

/// Resource observations taken in the same process as the sequential workload.
/// VmHWM is a Linux process-wide high-water mark, not per-step allocated bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceResourceMeasure {
    pub sequence: SequenceMeasure,
    pub vm_hwm_kib: Option<u64>,
    /// Process high-water mark before executing the sequence (not per-step RSS).
    pub vm_hwm_before_kib: Option<u64>,
    /// Nonnegative increase in process high-water mark during this sequence.
    /// Zero does not imply no allocation: previously reached peaks are invisible.
    pub vm_hwm_growth_kib: Option<u64>,
    pub committed_per_second: Option<u64>,
}

/// Execute and instrument the actual sequence, rather than a CSV audit.
/// The throughput denominator includes candidate cloning, cache refresh,
/// Full-oracle verification and rejected interventions.
pub fn verified_sequence_resources<S, C, O, T, I, B, F, D>(
    state: &mut S,
    cache: &mut C,
    interventions: &[T],
    intervene: I,
    rebuild: B,
    full: F,
    incremental: D,
) -> SequenceResourceMeasure
where
    S: Clone,
    O: Eq,
    I: FnMut(&mut S, &T) -> bool,
    B: FnMut(&S) -> C,
    F: FnMut(&S) -> O,
    D: FnMut(&S, &C) -> O,
{
    let vm_hwm_before_kib = read_process_vm_hwm_kib();
    let sequence = verified_sequence(
        state,
        cache,
        interventions,
        intervene,
        rebuild,
        full,
        incremental,
    );
    let nanos = sequence.elapsed.as_nanos();
    let committed_per_second = if nanos == 0 {
        None
    } else {
        u128::try_from(sequence.committed)
            .ok()
            .and_then(|count| count.checked_mul(1_000_000_000))
            .map(|rate| rate / nanos)
            .and_then(|rate| u64::try_from(rate).ok())
    };
    let vm_hwm_kib = read_process_vm_hwm_kib();
    let vm_hwm_growth_kib = match (vm_hwm_before_kib, vm_hwm_kib) {
        (Some(before), Some(after)) => after.checked_sub(before),
        _ => None,
    };
    SequenceResourceMeasure {
        sequence,
        vm_hwm_kib,
        vm_hwm_before_kib,
        vm_hwm_growth_kib,
        committed_per_second,
    }
}

fn parse_vm_hwm_kib(status: &str) -> Option<u64> {
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let mut fields = line["VmHWM:".len()..].split_whitespace();
    let value = fields.next()?.parse::<u64>().ok()?;
    if fields.next()? != "kB" || fields.next().is_some() {
        return None;
    }
    Some(value)
}

#[cfg(target_os = "linux")]
fn read_process_vm_hwm_kib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| parse_vm_hwm_kib(&status))
}

#[cfg(not(target_os = "linux"))]
fn read_process_vm_hwm_kib() -> Option<u64> {
    None
}

#[cfg(test)]
mod resource_tests {
    use super::*;

    #[test]
    fn vm_hwm_parser_rejects_missing_or_wrong_units() {
        assert_eq!(
            parse_vm_hwm_kib("Name:\ttest\nVmHWM:\t2048 kB\n"),
            Some(2048)
        );
        assert_eq!(parse_vm_hwm_kib("VmHWM: 2048 MB"), None);
        assert_eq!(parse_vm_hwm_kib("VmHWM: invalid kB"), None);
        assert_eq!(parse_vm_hwm_kib("VmRSS: 2048 kB"), None);
    }

    #[test]
    fn resource_sequence_preserves_oracle_rollback() {
        let mut state = 0_u8;
        let mut cache = 0_u8;
        let result = verified_sequence_resources(
            &mut state,
            &mut cache,
            &[1_u8, 2_u8],
            |candidate, value| {
                *candidate = *value;
                true
            },
            |candidate| *candidate,
            |candidate| *candidate,
            |_, candidate_cache| {
                if *candidate_cache == 2 {
                    99
                } else {
                    *candidate_cache
                }
            },
        );
        assert_eq!(
            (result.sequence.attempted, result.sequence.committed),
            (2, 1)
        );
        assert!(result.sequence.stopped_on_rejection);
        assert_eq!((state, cache), (1, 1));
        assert!(result.committed_per_second.is_some());
        if let (Some(before), Some(after), Some(growth)) = (
            result.vm_hwm_before_kib,
            result.vm_hwm_kib,
            result.vm_hwm_growth_kib,
        ) {
            assert_eq!(after.checked_sub(before), Some(growth));
        }
    }
}

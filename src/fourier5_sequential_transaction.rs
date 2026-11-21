//! Transactional sequential execution with an independently rebuilt cache oracle.
//! Synthetic states only. No biological interpretation or speedup claim.
use crate::fourier5_sequential_workload_audit::{process_vm_hwm_kib, StepMeasurement, StepVerdict};
use std::time::Instant;

/// A failed authoritative evaluation or cache reconstruction stops the sequence.
/// No partially verified state/cache pair is returned as successful.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransactionError {
    Full,
    Rebuild,
    /// A configured intervention count or observed peak RSS bound was exceeded.
    BudgetExceeded,
}

/// The incremental callback receives an immutable snapshot of the last verified
/// state and cache, and must return both updated objects by value. It cannot
/// mutate the committed cache through this interface. The Full oracle computes
/// the next state independently; rebuild computes the canonical cache from it.
/// On mismatch or incremental error, Full wins and its cache is rebuilt.
/// Timings are paired and include the incremental callback's cache maintenance.
/// Both the authoritative Full evaluation and canonical cache rebuild are
/// charged on every step: this reports the complete cost of *verified execution*,
/// not unverified production latency.
/// Full timing remains the standalone reference; verified timings necessarily
/// include Full and must not be used to calibrate unverified production dispatch.
/// Backwards-compatible verified execution; use the profiled variant to obtain
/// the initial cache reconstruction duration (excluded from per-step rows).
pub fn execute_verified_sequence<S, C, I, E, F, D, R>(
    initial_state: S,
    initial_cache: C,
    inputs: &[I],
    full: F,
    incremental: D,
    rebuild: R,
) -> Result<(S, C, Vec<StepMeasurement>), TransactionError>
where
    S: Clone + Eq,
    C: Clone + Eq,
    F: FnMut(&S, &I) -> Result<S, E>,
    D: FnMut(&S, &C, &I) -> Result<(S, C), E>,
    R: FnMut(&S) -> Result<C, E>,
{
    execute_verified_sequence_profiled(
        initial_state,
        initial_cache,
        inputs,
        full,
        incremental,
        rebuild,
    )
    .map(|(state, cache, rows, _initial_rebuild)| (state, cache, rows))
}

/// Same transactional semantics as `execute_verified_sequence`, but returns
/// the measured initial canonical cache reconstruction cost separately.
/// This cost must be included in end-to-end cold-start benchmarks.
pub fn execute_verified_sequence_profiled<S, C, I, E, F, D, R>(
    initial_state: S,
    initial_cache: C,
    inputs: &[I],
    full: F,
    incremental: D,
    rebuild: R,
) -> Result<(S, C, Vec<StepMeasurement>, std::time::Duration), TransactionError>
where
    S: Clone + Eq,
    C: Clone + Eq,
    F: FnMut(&S, &I) -> Result<S, E>,
    D: FnMut(&S, &C, &I) -> Result<(S, C), E>,
    R: FnMut(&S) -> Result<C, E>,
{
    execute_verified_sequence_limited(
        initial_state,
        initial_cache,
        inputs,
        full,
        incremental,
        rebuild,
        usize::MAX,
        None,
    )
}

/// A requested memory ceiling must have a real, nonzero observation.
fn within_observed_peak_budget(limit: Option<u64>, observed: Option<u64>) -> bool {
    match limit {
        None => true,
        Some(maximum) => matches!(observed, Some(peak) if peak > 0 && peak <= maximum),
    }
}

/// Bounded transactional execution. Rejects an oversized sequence before invoking
/// callbacks or allocating rows. An optional VmHWM ceiling is checked at entry
/// and after each verified step; VmHWM is process-wide and is NOT an allocator
/// limit or a substitute for RemoteOps cgroup memory enforcement.
#[allow(clippy::too_many_arguments)]
pub fn execute_verified_sequence_limited<S, C, I, E, F, D, R>(
    initial_state: S,
    initial_cache: C,
    inputs: &[I],
    mut full: F,
    mut incremental: D,
    mut rebuild: R,
    max_steps: usize,
    max_observed_vm_hwm_kib: Option<u64>,
) -> Result<(S, C, Vec<StepMeasurement>, std::time::Duration), TransactionError>
where
    S: Clone + Eq,
    C: Clone + Eq,
    F: FnMut(&S, &I) -> Result<S, E>,
    D: FnMut(&S, &C, &I) -> Result<(S, C), E>,
    R: FnMut(&S) -> Result<C, E>,
{
    if inputs.len() > max_steps {
        return Err(TransactionError::BudgetExceeded);
    }
    if !within_observed_peak_budget(max_observed_vm_hwm_kib, process_vm_hwm_kib()) {
        return Err(TransactionError::BudgetExceeded);
    }
    // Verify the supplied initial cache before allowing any incremental call.
    let initial_rebuild_start = Instant::now();
    let initial_canonical = rebuild(&initial_state).map_err(|_| TransactionError::Rebuild)?;
    let initial_rebuild_time = initial_rebuild_start.elapsed();
    if initial_canonical != initial_cache {
        return Err(TransactionError::Rebuild);
    }
    // Cache construction can itself cross the process peak budget. Reject
    // before either engine runs, rather than waiting until the first step.
    if !within_observed_peak_budget(max_observed_vm_hwm_kib, process_vm_hwm_kib()) {
        return Err(TransactionError::BudgetExceeded);
    }
    let mut state = initial_state;
    let mut cache = initial_cache;
    let mut rows = Vec::with_capacity(inputs.len());
    for input in inputs {
        let start = Instant::now();
        let expected = full(&state, input).map_err(|_| TransactionError::Full)?;
        let full_time = start.elapsed();
        let start = Instant::now();
        let candidate = incremental(&state, &cache, input);
        let incremental_time = start.elapsed();
        // Independent cache reconstruction is outside the timed fast path.
        // It is a correctness oracle, not a claimed incremental cost.
        let rebuild_start = Instant::now();
        let canonical = rebuild(&expected).map_err(|_| TransactionError::Rebuild)?;
        let rebuild_time = rebuild_start.elapsed();
        let verdict = match candidate {
            Ok((next_state, next_cache)) if next_state == expected && next_cache == canonical => {
                state = next_state;
                cache = next_cache;
                StepVerdict::Accepted
            }
            Ok(_) => {
                state = expected;
                cache = canonical;
                StepVerdict::Mismatch
            }
            Err(_) => {
                state = expected;
                cache = canonical;
                StepVerdict::IncrementalError
            }
        };
        // Verified execution runs all three operations, even when accepted.
        // Charge Full oracle evaluation, candidate execution, and canonical
        // rebuild. Reuse the canonical cache on recovery (no second rebuild).
        let verified_time = full_time
            .saturating_add(incremental_time)
            .saturating_add(rebuild_time);
        // Use the same kernel observation for enforcement and reporting.
        // Two reads could otherwise report a peak larger than the admitted one.
        let observed_peak = process_vm_hwm_kib();
        if !within_observed_peak_budget(max_observed_vm_hwm_kib, observed_peak) {
            return Err(TransactionError::BudgetExceeded);
        }
        rows.push(StepMeasurement {
            verdict,
            full: full_time,
            incremental_with_refresh: verified_time,
            peak_rss_kib: observed_peak,
        });
    }
    Ok((state, cache, rows, initial_rebuild_time))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_budget_fails_closed_without_observation() {
        assert!(!within_observed_peak_budget(Some(6144), None));
        assert!(!within_observed_peak_budget(Some(6144), Some(0)));
        assert!(!within_observed_peak_budget(Some(6144), Some(6145)));
        assert!(within_observed_peak_budget(Some(6144), Some(6144)));
        assert!(within_observed_peak_budget(None, None));
    }

    #[test]
    fn bounded_sequence_rejects_excess_before_callbacks() {
        use std::cell::Cell;
        let calls = Cell::new(0_u32);
        let result = execute_verified_sequence_limited(
            0_u8,
            0_u8,
            &[1_u8, 2],
            |s, d| {
                calls.set(calls.get() + 1);
                Ok::<_, ()>(s + d)
            },
            |s, _, d| Ok::<_, ()>((s + d, s + d)),
            |s| Ok::<_, ()>(*s),
            1,
            None,
        );
        assert!(matches!(result, Err(TransactionError::BudgetExceeded)));
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn bounded_sequence_accepts_exact_step_limit() {
        let (state, cache, rows, _) = execute_verified_sequence_limited(
            0_u8,
            0_u8,
            &[1_u8, 2],
            |s, d| Ok::<_, ()>(s + d),
            |s, _, d| Ok::<_, ()>((s + d, s + d)),
            |s| Ok::<_, ()>(*s),
            2,
            None,
        )
        .unwrap();
        assert_eq!((state, cache), (3, 3));
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn stale_cache_with_correct_output_is_recovered_before_next_step() {
        let (state, cache, rows) = execute_verified_sequence(
            0_u32,
            0_u32,
            &[1_u32, 2, 3],
            |s, d| Ok::<_, ()>(s + d),
            |s, c, d| Ok::<_, ()>((s + d, c + d + u32::from(*d == 2))),
            |s| Ok::<_, ()>(*s),
        )
        .unwrap();
        assert_eq!((state, cache), (6, 6));
        assert_eq!(
            rows.iter().map(|r| r.verdict).collect::<Vec<_>>(),
            vec![
                StepVerdict::Accepted,
                StepVerdict::Mismatch,
                StepVerdict::Accepted
            ]
        );
    }

    #[test]
    fn incremental_error_recovers_full_and_keeps_sequence_live() {
        let (state, cache, rows) = execute_verified_sequence(
            0_i32,
            0_i32,
            &[1_i32, 2, 3],
            |s, d| Ok::<_, &'static str>(s + d),
            |s, _, d| {
                if *d == 2 {
                    Err("candidate")
                } else {
                    Ok((s + d, s + d))
                }
            },
            |s| Ok::<_, &'static str>(*s),
        )
        .unwrap();
        assert_eq!((state, cache), (6, 6));
        assert_eq!(rows[1].verdict, StepVerdict::IncrementalError);
        assert_eq!(rows[2].verdict, StepVerdict::Accepted);
    }

    #[test]
    fn invalid_initial_cache_fails_before_any_mutation() {
        let result = execute_verified_sequence(
            0_u8,
            1_u8,
            &[1_u8],
            |s, d| Ok::<_, ()>(s + d),
            |s, _, d| Ok::<_, ()>((s + d, s + d)),
            |s| Ok::<_, ()>(*s),
        );
        assert!(matches!(result, Err(TransactionError::Rebuild)));
    }

    #[test]
    fn mismatch_rebuilds_once_per_step_and_recovers_exactly() {
        use std::cell::Cell;
        let rebuild_calls = Cell::new(0_u32);
        let (state, cache, rows) = execute_verified_sequence(
            0_u32,
            0_u32,
            &[1_u32, 2],
            |s, d| Ok::<_, ()>(s + d),
            |s, _, d| Ok::<_, ()>((s + d, s + d + 1)),
            |s| {
                rebuild_calls.set(rebuild_calls.get() + 1);
                Ok::<_, ()>(*s)
            },
        )
        .unwrap();
        // One initial validation plus one canonical reconstruction per step.
        assert_eq!(rebuild_calls.get(), 3);
        assert_eq!((state, cache), (3, 3));
        assert!(rows.iter().all(|r| r.verdict == StepVerdict::Mismatch));
    }

    #[test]
    fn accepted_step_accounts_for_canonical_verification_cost() {
        use std::time::Duration;
        let (_, _, rows) = execute_verified_sequence(
            0_u8,
            0_u8,
            &[1_u8],
            |s, d| Ok::<_, ()>(s + d),
            |s, _, d| Ok::<_, ()>((s + d, s + d)),
            |s| {
                if *s == 1 {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Ok::<_, ()>(*s)
            },
        )
        .unwrap();
        assert_eq!(rows[0].verdict, StepVerdict::Accepted);
        assert!(rows[0].incremental_with_refresh >= Duration::from_millis(5));
    }

    #[test]
    fn verified_step_charges_authoritative_full_oracle_too() {
        use std::time::Duration;
        let (_, _, rows) = execute_verified_sequence(
            0_u8,
            0_u8,
            &[1_u8],
            |s, d| {
                std::thread::sleep(Duration::from_millis(4));
                Ok::<_, ()>(s + d)
            },
            |s, _, d| Ok::<_, ()>((s + d, s + d)),
            |s| Ok::<_, ()>(*s),
        )
        .unwrap();
        assert_eq!(rows[0].verdict, StepVerdict::Accepted);
        // Independent arithmetic oracle: verified >= Full + candidate + rebuild.
        assert!(rows[0].incremental_with_refresh >= rows[0].full);
        assert!(rows[0].full >= Duration::from_millis(4));
    }

    #[test]
    fn profiled_cold_start_reports_initial_rebuild_separately() {
        use std::time::Duration;
        let (state, cache, rows, initial) = execute_verified_sequence_profiled(
            0_u8,
            0_u8,
            &[1_u8],
            |s, d| Ok::<_, ()>(s + d),
            |s, _, d| Ok::<_, ()>((s + d, s + d)),
            |s| {
                if *s == 0 {
                    std::thread::sleep(Duration::from_millis(3));
                }
                Ok::<_, ()>(*s)
            },
        )
        .unwrap();
        assert_eq!((state, cache), (1, 1));
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].verdict, StepVerdict::Accepted);
        assert!(initial >= Duration::from_millis(3));
    }

    #[test]
    fn initial_rebuild_failure_does_not_call_either_engine() {
        use std::cell::Cell;
        let full_calls = Cell::new(0_u32);
        let incremental_calls = Cell::new(0_u32);
        let result = execute_verified_sequence(
            0_u8,
            0_u8,
            &[1_u8],
            |s, d| {
                full_calls.set(full_calls.get() + 1);
                Ok::<_, ()>(s + d)
            },
            |s, _, d| {
                incremental_calls.set(incremental_calls.get() + 1);
                Ok::<_, ()>((s + d, s + d))
            },
            |_| Err::<u8, ()>(()),
        );
        assert!(matches!(result, Err(TransactionError::Rebuild)));
        assert_eq!(full_calls.get(), 0);
        assert_eq!(incremental_calls.get(), 0);
    }

    #[test]
    fn authoritative_failure_never_returns_unverified_commit() {
        let result = execute_verified_sequence(
            0_i32,
            0_i32,
            &[1_i32, 2],
            |s, d| if *d == 2 { Err("full") } else { Ok(s + d) },
            |s, _, d| Ok::<_, &'static str>((s + d, s + d)),
            |s| Ok::<_, &'static str>(*s),
        );
        assert!(matches!(result, Err(TransactionError::Full)));
    }
}

/// Exact accounting for a *verified* heterogeneous sequential run.
/// This is not a production Incremental benchmark: every step also runs Full
/// and rebuilds a canonical cache. None means a missing observation or
/// arithmetic overflow; no partial throughput is published.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedSequenceProfile {
    pub steps: usize,
    pub accepted: usize,
    pub recovered: usize,
    /// Sum of verified costs for accepted steps only.
    pub accepted_verified_nanos: u128,
    /// Sum of verified costs for recovered steps only.
    pub recovered_verified_nanos: u128,
    /// Maximum observed process VmHWM among recovered steps.
    pub recovered_peak_rss_kib: Option<u64>,
    pub cold_start_nanos: u128,
    pub full_nanos: u128,
    pub verified_step_nanos: u128,
    pub total_verified_nanos: u128,
    /// Steps per second, rounded down. None for zero elapsed time.
    pub throughput_steps_per_second: Option<u128>,
    /// Process lifetime high-water mark, not a workload allocation.
    pub process_peak_rss_kib: u64,
}

/// Fail closed if any row has no valid memory observation, or if a timing
/// aggregate overflows. Includes rejected/recovered steps and cold start.
#[must_use]
pub fn profile_verified_sequence(
    rows: &[StepMeasurement],
    initial_rebuild: std::time::Duration,
) -> Option<VerifiedSequenceProfile> {
    let mut accepted = 0_usize;
    let mut recovered = 0_usize;
    let mut full_nanos = 0_u128;
    let mut accepted_verified_nanos = 0_u128;
    let mut recovered_verified_nanos = 0_u128;
    let mut recovered_peak_rss_kib: Option<u64> = None;
    let mut verified_step_nanos = 0_u128;
    let mut peak = 0_u64;
    for row in rows {
        match row.verdict {
            StepVerdict::Accepted => accepted = accepted.checked_add(1)?,
            StepVerdict::Mismatch | StepVerdict::IncrementalError => {
                recovered = recovered.checked_add(1)?
            }
            StepVerdict::FullError => return None,
        }
        let observed = row.peak_rss_kib.filter(|&value| value > 0)?;
        peak = peak.max(observed);
        if row.incremental_with_refresh < row.full {
            return None;
        }
        match row.verdict {
            StepVerdict::Accepted => {
                accepted_verified_nanos =
                    accepted_verified_nanos.checked_add(row.incremental_with_refresh.as_nanos())?;
            }
            StepVerdict::Mismatch | StepVerdict::IncrementalError => {
                recovered_verified_nanos = recovered_verified_nanos
                    .checked_add(row.incremental_with_refresh.as_nanos())?;
                recovered_peak_rss_kib = Some(recovered_peak_rss_kib.unwrap_or(0).max(observed));
            }
            StepVerdict::FullError => return None,
        }
        full_nanos = full_nanos.checked_add(row.full.as_nanos())?;
        verified_step_nanos =
            verified_step_nanos.checked_add(row.incremental_with_refresh.as_nanos())?;
    }
    let cold_start_nanos = initial_rebuild.as_nanos();
    let total_verified_nanos = cold_start_nanos.checked_add(verified_step_nanos)?;
    let throughput_steps_per_second = if total_verified_nanos == 0 {
        None
    } else {
        Some((rows.len() as u128).checked_mul(1_000_000_000)? / total_verified_nanos)
    };
    Some(VerifiedSequenceProfile {
        steps: rows.len(),
        accepted,
        recovered,
        accepted_verified_nanos,
        recovered_verified_nanos,
        recovered_peak_rss_kib,
        cold_start_nanos,
        full_nanos,
        verified_step_nanos,
        total_verified_nanos,
        throughput_steps_per_second,
        process_peak_rss_kib: peak,
    })
}

#[cfg(test)]
mod profile_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn independent_arithmetic_oracle_includes_recovery_and_cold_start() {
        let rows = [
            StepMeasurement {
                verdict: StepVerdict::Accepted,
                full: Duration::from_nanos(11),
                incremental_with_refresh: Duration::from_nanos(31),
                peak_rss_kib: Some(2048),
            },
            StepMeasurement {
                verdict: StepVerdict::Mismatch,
                full: Duration::from_nanos(17),
                incremental_with_refresh: Duration::from_nanos(43),
                peak_rss_kib: Some(4096),
            },
        ];
        let profile = profile_verified_sequence(&rows, Duration::from_nanos(26)).unwrap();
        assert_eq!(
            (profile.steps, profile.accepted, profile.recovered),
            (2, 1, 1)
        );
        assert_eq!((profile.full_nanos, profile.verified_step_nanos), (28, 74));
        assert_eq!(profile.total_verified_nanos, 100);
        assert_eq!(profile.accepted_verified_nanos, 31);
        assert_eq!(profile.recovered_verified_nanos, 43);
        assert_eq!(profile.recovered_peak_rss_kib, Some(4096));
        assert_eq!(profile.throughput_steps_per_second, Some(20_000_000));
        assert_eq!(profile.process_peak_rss_kib, 4096);
    }

    #[test]
    fn rejects_verified_cost_below_full_oracle() {
        let row = StepMeasurement {
            verdict: StepVerdict::Mismatch,
            full: Duration::from_nanos(20),
            incremental_with_refresh: Duration::from_nanos(19),
            peak_rss_kib: Some(512),
        };
        assert!(profile_verified_sequence(&[row], Duration::ZERO).is_none());
    }

    #[test]
    fn missing_peak_and_zero_duration_fail_closed() {
        let mut row = StepMeasurement {
            verdict: StepVerdict::Accepted,
            full: Duration::ZERO,
            incremental_with_refresh: Duration::ZERO,
            peak_rss_kib: None,
        };
        assert!(profile_verified_sequence(&[row.clone()], Duration::ZERO).is_none());
        row.peak_rss_kib = Some(1024);
        let profile = profile_verified_sequence(&[row], Duration::ZERO).unwrap();
        assert_eq!(profile.throughput_steps_per_second, None);
    }
}

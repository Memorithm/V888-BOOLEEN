//! Transactional sequential execution with an independently rebuilt cache oracle.
//! Synthetic states only. No biological interpretation or speedup claim.
use std::time::Instant;
use crate::fourier5_sequential_workload_audit::{process_vm_hwm_kib, StepMeasurement, StepVerdict};

/// A failed authoritative evaluation or cache reconstruction stops the sequence.
/// No partially verified state/cache pair is returned as successful.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransactionError {
    Full,
    Rebuild,
}

/// The incremental callback receives an immutable snapshot of the last verified
/// state and cache, and must return both updated objects by value. It cannot
/// mutate the committed cache through this interface. The Full oracle computes
/// the next state independently; rebuild computes the canonical cache from it.
/// On mismatch or incremental error, Full wins and its cache is rebuilt.
/// Timings are paired and include the incremental callback's cache maintenance.
/// The canonical oracle rebuild is charged on every step: this reports the
/// conservative cost of *verified execution*, not unverified production latency.
/// Full timing remains the standalone reference, so these paired figures are
/// unsuitable for claims of production speedup without a separate benchmark.
pub fn execute_verified_sequence<S, C, I, E, F, D, R>(
    initial_state: S,
    initial_cache: C,
    inputs: &[I],
    mut full: F,
    mut incremental: D,
    mut rebuild: R,
) -> Result<(S, C, Vec<StepMeasurement>), TransactionError>
where
    S: Clone + Eq,
    C: Clone + Eq,
    F: FnMut(&S, &I) -> Result<S, E>,
    D: FnMut(&S, &C, &I) -> Result<(S, C), E>,
    R: FnMut(&S) -> Result<C, E>,
{
    // Verify the supplied initial cache before allowing any incremental call.
    if rebuild(&initial_state).map_err(|_| TransactionError::Rebuild)? != initial_cache {
        return Err(TransactionError::Rebuild);
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
            Ok((next_state, next_cache))
                if next_state == expected && next_cache == canonical =>
            {
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
        // Charge verification on accepted paths too: a verified execution
        // cannot omit the oracle's reconstruction from its measured cost.
        // The same canonical cache is reused on recovery (no second rebuild).
        let verified_time = incremental_time.saturating_add(rebuild_time);
        rows.push(StepMeasurement {
            verdict,
            full: full_time,
            incremental_with_refresh: verified_time,
            peak_rss_kib: process_vm_hwm_kib(),
        });
    }
    Ok((state, cache, rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_cache_with_correct_output_is_recovered_before_next_step() {
        let (state, cache, rows) = execute_verified_sequence(
            0_u32, 0_u32, &[1_u32, 2, 3],
            |s, d| Ok::<_, ()>(s + d),
            |s, c, d| Ok::<_, ()>((s + d, c + d + u32::from(*d == 2))),
            |s| Ok::<_, ()>(*s),
        ).unwrap();
        assert_eq!((state, cache), (6, 6));
        assert_eq!(rows.iter().map(|r| r.verdict).collect::<Vec<_>>(),
            vec![StepVerdict::Accepted, StepVerdict::Mismatch, StepVerdict::Accepted]);
    }

    #[test]
    fn incremental_error_recovers_full_and_keeps_sequence_live() {
        let (state, cache, rows) = execute_verified_sequence(
            0_i32, 0_i32, &[1_i32, 2, 3],
            |s, d| Ok::<_, &'static str>(s + d),
            |s, _, d| if *d == 2 { Err("candidate") } else { Ok((s + d, s + d)) },
            |s| Ok::<_, &'static str>(*s),
        ).unwrap();
        assert_eq!((state, cache), (6, 6));
        assert_eq!(rows[1].verdict, StepVerdict::IncrementalError);
        assert_eq!(rows[2].verdict, StepVerdict::Accepted);
    }

    #[test]
    fn invalid_initial_cache_fails_before_any_mutation() {
        let result = execute_verified_sequence(
            0_u8, 1_u8, &[1_u8],
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
            0_u32, 0_u32, &[1_u32, 2],
            |s, d| Ok::<_, ()>(s + d),
            |s, _, d| Ok::<_, ()>((s + d, s + d + 1)),
            |s| {
                rebuild_calls.set(rebuild_calls.get() + 1);
                Ok::<_, ()>(*s)
            },
        ).unwrap();
        // One initial validation plus one canonical reconstruction per step.
        assert_eq!(rebuild_calls.get(), 3);
        assert_eq!((state, cache), (3, 3));
        assert!(rows.iter().all(|r| r.verdict == StepVerdict::Mismatch));
    }

    #[test]
    fn accepted_step_accounts_for_canonical_verification_cost() {
        use std::time::Duration;
        let (_, _, rows) = execute_verified_sequence(
            0_u8, 0_u8, &[1_u8],
            |s, d| Ok::<_, ()>(s + d),
            |s, _, d| Ok::<_, ()>((s + d, s + d)),
            |s| {
                if *s == 1 {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Ok::<_, ()>(*s)
            },
        ).unwrap();
        assert_eq!(rows[0].verdict, StepVerdict::Accepted);
        assert!(rows[0].incremental_with_refresh >= Duration::from_millis(5));
    }

    #[test]
    fn authoritative_failure_never_returns_unverified_commit() {
        let result = execute_verified_sequence(
            0_i32, 0_i32, &[1_i32, 2],
            |s, d| if *d == 2 { Err("full") } else { Ok(s + d) },
            |s, _, d| Ok::<_, &'static str>((s + d, s + d)),
            |s| Ok::<_, &'static str>(*s),
        );
        assert!(matches!(result, Err(TransactionError::Full)));
    }
}

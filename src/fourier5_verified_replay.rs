//! Execute a frozen dispatch decision against a full scalar oracle.
//! Synthetic circuits only. The caller supplies independent engines; no BANC topology.
//! This is a correctness-first execution path, not a latency benchmark.
use crate::fourier5_measured_dispatch::{select_before_decision, BucketPolicy, EngineChoice};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayError<E> {
    Full(E),
    Incremental(E),
    DecisionMismatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayOutcome<T> {
    pub output: T,
    pub choice: EngineChoice,
    /// True if a selected incremental result disagreed with the full oracle.
    pub fallback: bool,
}

/// Run a real full oracle for every intervention. The incremental engine is
/// invoked only if the frozen TRAIN/VALID table allows it *before* acceptance
/// is known. No TEST/OOD observation changes the policy.
///
/// Both callbacks must be observational (no external mutation). The full
/// callback is authoritative, including acceptance and output. A mismatch
/// falls back to Full and is reported, never silently promoted.
/// Incremental errors also fall back to Full; full errors fail closed.
pub fn execute_verified<T: Eq, E>(
    gates: usize,
    cone_gates: usize,
    frozen: &[BucketPolicy],
    full: impl FnOnce() -> Result<(T, bool), E>,
    incremental: impl FnOnce() -> Result<(T, bool), E>,
) -> Result<ReplayOutcome<(T, bool)>, ReplayError<E>> {
    let choice = select_before_decision(gates, cone_gates, frozen);
    let reference = full().map_err(ReplayError::Full)?;
    if choice == EngineChoice::Full {
        return Ok(ReplayOutcome {
            output: reference,
            choice,
            fallback: false,
        });
    }
    match incremental() {
        Ok(candidate) if candidate == reference => Ok(ReplayOutcome {
            output: candidate,
            choice,
            fallback: false,
        }),
        _ => Ok(ReplayOutcome {
            output: reference,
            choice: EngineChoice::Full,
            fallback: true,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5_measured_dispatch::ConeBucket;
    use std::cell::Cell;

    fn policies() -> [BucketPolicy; 2] {
        [true, false].map(|accepted| BucketPolicy {
            bucket: ConeBucket {
                gates: 100,
                cone_gates: 3,
                accepted,
            },
            choice: EngineChoice::Incremental,
            pairs: 3,
            full_total_ns: 300,
            incremental_total_ns: 90,
        })
    }

    #[test]
    fn independent_scalar_oracle_agrees_on_heterogeneous_sequences() {
        let policy = policies();
        for (input, accepted) in [(0b10101u8, true), (0b00111, false), (0b11111, true)] {
            let oracle = || {
                Ok::<_, ()>((
                    (0..5).filter(|i| input & (1 << i) != 0).count() % 2 == 1,
                    accepted,
                ))
            };
            let incremental = || Ok::<_, ()>((input.count_ones() % 2 == 1, accepted));
            let result = execute_verified(100, 3, &policy, oracle, incremental).unwrap();
            assert_eq!(result.choice, EngineChoice::Incremental);
            assert!(!result.fallback);
        }
    }

    #[test]
    fn mismatch_and_incremental_error_return_authoritative_full() {
        let policy = policies();
        let wrong = execute_verified(
            100,
            3,
            &policy,
            || Ok::<_, ()>((true, true)),
            || Ok::<_, ()>((false, true)),
        )
        .unwrap();
        assert_eq!(wrong.output, (true, true));
        assert_eq!(wrong.choice, EngineChoice::Full);
        assert!(wrong.fallback);
        let failed = execute_verified(
            100,
            3,
            &policy,
            || Ok::<_, &'static str>((true, false)),
            || Err::<(bool, bool), _>("incremental failure"),
        )
        .unwrap();
        assert_eq!(failed.output, (true, false));
        assert!(failed.fallback);
    }

    #[test]
    fn unknown_or_unvalidated_bucket_never_runs_incremental() {
        let called = Cell::new(false);
        let result = execute_verified(
            100,
            4,
            &policies(),
            || Ok::<_, ()>((7u8, false)),
            || {
                called.set(true);
                Ok::<_, ()>((9u8, true))
            },
        )
        .unwrap();
        assert_eq!(result.output, (7, false));
        assert_eq!(result.choice, EngineChoice::Full);
        assert!(!called.get());
    }

    #[test]
    fn full_error_is_not_masked_by_incremental() {
        let result = execute_verified(
            100,
            3,
            &policies(),
            || Err::<(bool, bool), _>("full failed"),
            || Ok::<_, &'static str>((true, true)),
        );
        assert_eq!(result, Err(ReplayError::Full("full failed")));
    }
}

//! Sequential Full-authoritative recovery for synthetic cache states.
//! Unlike strict audit (which rejects mismatches), this mode commits the
//! independently reconstructed Full state on an Incremental mismatch/error.
//! Full failure never advances state; no holdout observations train a policy.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryVerdict {
    IncrementalVerified,
    FullAfterMismatch,
    FullAfterIncrementalError,
    FullUnavailable,
}

/// Each step starts from the last *committed* state, not a rejected candidate.
/// The state must include the complete cache, not just observable outputs.
/// The caller must provide an independent Full implementation.
pub fn recover_sequence<S, I, E, F, D>(
    initial: S,
    inputs: &[I],
    mut full: F,
    mut incremental: D,
) -> (S, Vec<RecoveryVerdict>)
where
    S: Clone + Eq,
    F: FnMut(&S, &I) -> Result<S, E>,
    D: FnMut(&S, &I) -> Result<S, E>,
{
    let mut committed = initial;
    let mut verdicts = Vec::with_capacity(inputs.len());
    for input in inputs {
        // Both engines see the same immutable committed snapshot.
        let candidate = incremental(&committed, input);
        match full(&committed, input) {
            Err(_) => verdicts.push(RecoveryVerdict::FullUnavailable),
            Ok(reference) => {
                let verdict = match candidate {
                    Ok(actual) if actual == reference => RecoveryVerdict::IncrementalVerified,
                    Ok(_) => RecoveryVerdict::FullAfterMismatch,
                    Err(_) => RecoveryVerdict::FullAfterIncrementalError,
                };
                // Full is authoritative even when the candidate is wrong.
                committed = reference;
                verdicts.push(verdict);
            }
        }
    }
    (committed, verdicts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_recovery_advances_from_full_state_after_stale_cache() {
        // (visible output, internal cache epoch); output-only equality is unsafe.
        let (state, verdicts) = recover_sequence(
            (0_u32, 0_u32),
            &[1_u32, 2, 3],
            |&(out, epoch), &delta| Ok::<_, ()>((out + delta, epoch + 1)),
            |&(out, epoch), &delta| Ok::<_, ()>((out + delta, epoch + u32::from(delta != 2))),
        );
        assert_eq!(state, (6, 3));
        assert_eq!(
            verdicts,
            vec![
                RecoveryVerdict::IncrementalVerified,
                RecoveryVerdict::FullAfterMismatch,
                RecoveryVerdict::IncrementalVerified,
            ]
        );
    }

    #[test]
    fn independent_full_error_preserves_previous_committed_state() {
        let (state, verdicts) = recover_sequence(
            0_i32,
            &[1, 2, 3, 4],
            |&state, &delta| {
                if delta == 2 {
                    Err(())
                } else {
                    Ok(state + delta)
                }
            },
            |&state, &delta| {
                if delta == 3 {
                    Err(())
                } else {
                    Ok(state + delta)
                }
            },
        );
        assert_eq!(state, 8);
        assert_eq!(
            verdicts,
            vec![
                RecoveryVerdict::IncrementalVerified,
                RecoveryVerdict::FullUnavailable,
                RecoveryVerdict::FullAfterIncrementalError,
                RecoveryVerdict::IncrementalVerified,
            ]
        );
    }

    #[test]
    fn oracle_reference_matches_direct_sequential_full_with_rejections() {
        let inputs = [3_i64, -2, 7, 0, 4];
        let (state, verdicts) = recover_sequence(
            0_i64,
            &inputs,
            |&state, &delta| {
                if delta == 0 {
                    Err(())
                } else {
                    Ok(state + delta)
                }
            },
            |&state, &delta| Ok::<_, ()>(state + delta + i64::from(delta == 7)),
        );
        let expected = inputs.iter().filter(|&&x| x != 0).sum::<i64>();
        assert_eq!(state, expected);
        assert_eq!(verdicts[2], RecoveryVerdict::FullAfterMismatch);
        assert_eq!(verdicts[3], RecoveryVerdict::FullUnavailable);
    }
}

//! Atomic multi-intervention commit for synthetic Boolean circuits.
//! Each accepted proposal refreshes the candidate cache and is checked against
//! an independent Full oracle. Any failure rolls back the *entire* batch.
//! Callbacks must be deterministic and free of external side effects.
use crate::fourier5_transactional_oracle::{apply_verified, StepOutcome};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchReport<E> {
    pub committed: usize,
    pub rejected: usize,
    pub failed_at: Option<usize>,
    pub failure: Option<StepOutcome<E>>,
}

/// Apply proposals in order to private candidate state/cache. Publish them
/// together only if every accepted intervention passes its Full oracle.
/// A rejected proposal is a no-op and does not invalidate the batch.
pub fn apply_atomic_batch<S, C, P, E>(
    state: &mut S,
    cache: &mut C,
    proposals: impl IntoIterator<Item = P>,
    mut propose: impl FnMut(&mut S, P) -> Result<bool, E>,
    mut refresh: impl FnMut(&S, &mut C) -> Result<(), E>,
    mut full: impl FnMut(&S) -> Result<C, E>,
) -> BatchReport<E>
where
    S: Clone,
    C: Clone + PartialEq + std::fmt::Debug,
{
    let mut next_state = state.clone();
    let mut next_cache = cache.clone();
    let mut committed = 0usize;
    let mut rejected = 0usize;
    for (index, proposal) in proposals.into_iter().enumerate() {
        let outcome = apply_verified(
            &mut next_state,
            &mut next_cache,
            proposal,
            &mut propose,
            &mut refresh,
            &mut full,
        );
        match outcome {
            StepOutcome::Committed => committed += 1,
            StepOutcome::Rejected => rejected += 1,
            failure => {
                return BatchReport {
                    committed: 0,
                    rejected,
                    failed_at: Some(index),
                    failure: Some(failure),
                };
            }
        }
    }
    *state = next_state;
    *cache = next_cache;
    BatchReport { committed, rejected, failed_at: None, failure: None }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent Full oracle counts all set bits; incremental refresh toggles
    // parity using the change between previous and candidate state.
    fn full(bits: &[bool]) -> Result<bool, &'static str> {
        Ok(bits.iter().filter(|&&b| b).count() % 2 != 0)
    }

    #[test]
    fn accepted_sequential_changes_are_committed_together() {
        let mut state = vec![false; 4];
        let mut cache = false;
        let report = apply_atomic_batch(
            &mut state, &mut cache, [0usize, 1, 0],
            |bits, i| { bits[i] = !bits[i]; Ok::<_, &str>(true) },
            |bits, cached| { *cached = bits.iter().fold(false, |a, b| a ^ b); Ok::<_, &str>(()) },
            full,
        );
        assert_eq!(report, BatchReport { committed: 3, rejected: 0, failed_at: None, failure: None });
        assert_eq!(state, vec![false, true, false, false]);
        assert!(cache);
    }

    #[test]
    fn late_oracle_mismatch_rolls_back_earlier_commits() {
        let mut state = vec![false; 3];
        let mut cache = false;
        let report = apply_atomic_batch(
            &mut state, &mut cache, [0usize, 1],
            |bits, i| { bits[i] = true; Ok::<_, &str>(true) },
            |bits, cached| { *cached = bits[0]; Ok::<_, &str>(()) },
            full,
        );
        assert_eq!(report.failed_at, Some(1));
        assert_eq!(report.failure, Some(StepOutcome::OracleMismatch));
        assert_eq!(report.committed, 0);
        assert_eq!(state, vec![false; 3]);
        assert!(!cache);
    }

    #[test]
    fn rejected_proposal_is_noop_and_does_not_block_later_commit() {
        let mut state = vec![false; 2];
        let mut cache = false;
        let report = apply_atomic_batch(
            &mut state, &mut cache, [0usize, 1],
            |bits, i| { bits[i] = true; Ok::<_, &str>(i == 1) },
            |bits, cached| { *cached = bits.iter().filter(|&&b| b).count() % 2 == 1; Ok::<_, &str>(()) },
            full,
        );
        assert_eq!(report.committed, 1);
        assert_eq!(report.rejected, 1);
        assert_eq!(state, vec![false, true]);
        assert!(cache);
    }

    #[test]
    fn empty_batch_does_not_change_state() {
        let mut state = vec![true];
        let mut cache = true;
        let report = apply_atomic_batch(
            &mut state, &mut cache, std::iter::empty::<usize>(),
            |_, _| Ok::<_, &str>(true),
            |_, _| Ok::<_, &str>(()),
            full,
        );
        assert_eq!(report.committed, 0);
        assert_eq!(report.failed_at, None);
        assert_eq!(state, vec![true]);
        assert!(cache);
    }
}

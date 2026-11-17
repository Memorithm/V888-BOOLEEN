//! Transactional sequential cache validation for synthetic Boolean circuits.
//! An independent Full oracle gates every accepted incremental mutation.
use std::fmt::Debug;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepOutcome<E> {
    Rejected,
    Committed,
    Aborted { error: E },
    OracleMismatch,
}

/// Keep committed state unchanged until the incremental cache and independent
/// Full recomputation agree. A rejected proposal never changes the session.
/// Callbacks must be deterministic and have no external side effects.
pub fn apply_verified<S, C, P, E>(
    state: &mut S,
    cache: &mut C,
    proposal: P,
    mut propose: impl FnMut(&mut S, P) -> Result<bool, E>,
    mut refresh: impl FnMut(&S, &mut C) -> Result<(), E>,
    mut full: impl FnMut(&S) -> Result<C, E>,
) -> StepOutcome<E>
where
    S: Clone,
    C: Clone + PartialEq + Debug,
{
    let mut candidate_state = state.clone();
    match propose(&mut candidate_state, proposal) {
        Err(error) => return StepOutcome::Aborted { error },
        Ok(false) => return StepOutcome::Rejected,
        Ok(true) => {}
    }
    let mut candidate_cache = cache.clone();
    if let Err(error) = refresh(&candidate_state, &mut candidate_cache) {
        return StepOutcome::Aborted { error };
    }
    let expected = match full(&candidate_state) {
        Ok(value) => value,
        Err(error) => return StepOutcome::Aborted { error },
    };
    if candidate_cache != expected {
        return StepOutcome::OracleMismatch;
    }
    *state = candidate_state;
    *cache = candidate_cache;
    StepOutcome::Committed
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent reference: recompute parity by scanning all bits, not
    // by applying the incremental XOR update.
    fn parity(bits: &[bool]) -> bool {
        bits.iter().filter(|&&bit| bit).count() % 2 == 1
    }

    #[test]
    fn sequential_accepted_updates_preserve_exact_cache() {
        let mut bits = vec![false; 17];
        let mut cached = false;
        for index in [0, 8, 16, 8, 2, 0] {
            let result = apply_verified(
                &mut bits, &mut cached, index,
                |candidate, i| { candidate[i] = !candidate[i]; Ok::<_, ()>(true) },
                |candidate, value| { *value = candidate.iter().fold(false, |a, b| a ^ b); Ok::<_, ()>(()) },
                |candidate| Ok::<_, ()>(parity(candidate)),
            );
            assert_eq!(result, StepOutcome::Committed);
            assert_eq!(cached, parity(&bits));
        }
    }

    #[test]
    fn rejected_proposal_does_not_commit() {
        let mut bits = vec![false; 2];
        let mut cached = false;
        let result = apply_verified(
            &mut bits, &mut cached, 1,
            |candidate, i| { candidate[i] = true; Ok::<_, ()>(false) },
            |_, _| Ok::<_, ()>(()),
            |_| Ok::<_, ()>(true),
        );
        assert_eq!(result, StepOutcome::Rejected);
        assert_eq!(bits, vec![false; 2]);
        assert!(!cached);
    }

    #[test]
    fn incorrect_incremental_cache_rolls_back() {
        let mut bits = vec![false; 2];
        let mut cached = false;
        let result = apply_verified(
            &mut bits, &mut cached, 0,
            |candidate, i| { candidate[i] = true; Ok::<_, ()>(true) },
            |_, _| Ok::<_, ()>(()),
            |candidate| Ok::<_, ()>(parity(candidate)),
        );
        assert_eq!(result, StepOutcome::OracleMismatch);
        assert_eq!(bits, vec![false; 2]);
        assert!(!cached);
    }

    #[test]
    fn full_oracle_failure_rolls_back() {
        let mut bits = vec![false];
        let mut cached = false;
        let result = apply_verified(
            &mut bits, &mut cached, 0,
            |candidate, i| { candidate[i] = true; Ok::<_, &str>(true) },
            |_, value| { *value = true; Ok::<_, &str>(()) },
            |_| Err::<bool, _>("oracle failed"),
        );
        assert_eq!(result, StepOutcome::Aborted { error: "oracle failed" });
        assert_eq!(bits, vec![false]);
        assert!(!cached);
    }
}

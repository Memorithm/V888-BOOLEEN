//! Atomic synthetic sequential intervention: a failed cache rebuild never publishes
//! a partially changed circuit. This is an opt-in clone-based reference path,
//! not a claim about the performance of the wide circuit engine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtomicSession<C, K> {
    pub circuit: C,
    pub cache: K,
    pub revision: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomicError<E> {
    Mutation(E),
    Rebuild(E),
    RevisionOverflow,
}

/// A rejected intervention leaves both objects unchanged. For an accepted
/// intervention, mutation and cache rebuild run on private copies. The pair
/// is published only after both callbacks succeed. This prevents a caller's
/// partially-mutating callback from corrupting the committed session.
///
/// The callbacks must not mutate external state: only C and K are transactional.
pub fn apply_atomic<C: Clone, K: Clone, E>(
    session: &mut AtomicSession<C, K>,
    mutate: impl FnOnce(&mut C) -> Result<bool, E>,
    rebuild: impl FnOnce(&C, &mut K) -> Result<(), E>,
) -> Result<bool, AtomicError<E>> {
    // Do not run an unrollbackable callback when the revision cannot advance.
    let next = session.revision.checked_add(1).ok_or(AtomicError::RevisionOverflow)?;
    let mut candidate = session.circuit.clone();
    let accepted = mutate(&mut candidate).map_err(AtomicError::Mutation)?;
    if !accepted {
        return Ok(false);
    }
    let mut next_cache = session.cache.clone();
    rebuild(&candidate, &mut next_cache).map_err(AtomicError::Rebuild)?;
    session.circuit = candidate;
    session.cache = next_cache;
    session.revision = next;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent full-recomputation oracle: cache equals the sum of all bits.
    fn full(c: &[bool]) -> usize { c.iter().filter(|&&bit| bit).count() }

    #[test]
    fn sequential_commits_match_full_oracle_after_every_acceptance() {
        let mut s = AtomicSession { circuit: vec![false; 19], cache: 0usize, revision: 0 };
        for index in [3, 7, 3, 18, 0, 7] {
            let accepted = apply_atomic(&mut s,
                |c| { c[index] = !c[index]; Ok::<_, ()>(true) },
                |c, k| { *k = full(c); Ok::<_, ()>(()) }).unwrap();
            assert!(accepted);
            assert_eq!(s.cache, full(&s.circuit));
        }
        assert_eq!(s.revision, 6);
    }

    #[test]
    fn failed_partial_mutation_and_failed_rebuild_are_atomic() {
        let original = AtomicSession { circuit: vec![false, true], cache: 1usize, revision: 9 };
        let mut s = original.clone();
        assert_eq!(apply_atomic(&mut s,
            |c| { c[0] = true; Err::<bool, _>("mutation") },
            |_, _| Ok(())), Err(AtomicError::Mutation("mutation")));
        assert_eq!(s, original);
        assert_eq!(apply_atomic(&mut s,
            |c| { c[0] = true; Ok::<_, &str>(true) },
            |_, k| { *k = 99; Err("rebuild") }), Err(AtomicError::Rebuild("rebuild")));
        assert_eq!(s, original);
        assert!(!apply_atomic(&mut s,
            |c| { c[0] = true; Ok::<_, &str>(false) },
            |_, _| panic!("rejected interventions must not rebuild")).unwrap());
        assert_eq!(s, original);
    }

    #[test]
    fn overflow_fails_before_mutation() {
        let mut s = AtomicSession { circuit: vec![false], cache: 0usize, revision: u64::MAX };
        let before = s.clone();
        assert_eq!(apply_atomic(&mut s,
            |_| panic!("must not mutate on overflow"),
            |_, _| Ok(())), Err(AtomicError::RevisionOverflow));
        assert_eq!(s, before);
    }
}

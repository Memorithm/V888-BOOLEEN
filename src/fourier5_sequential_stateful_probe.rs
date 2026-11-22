//! Stateful heterogeneous sequential probe. Every accepted intervention advances
//! the canonical Full state; the incremental path is observational only.
//! Synthetic research harness, not an inference speedup claim.
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceError {
    InvalidCone,
    FullFailed,
    CacheFailed,
    IncrementalFailed,
    RevisionOverflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepEvidence {
    pub revision: u64,
    pub gates: u64,
    pub cone_gates: u64,
    pub full_ns: u64,
    pub cache_refresh_ns: u64,
    pub incremental_ns: u64,
    pub output_equal: bool,
    pub decision_equal: bool,
    pub incremental_first: bool,
    /// Linux process-wide high-water mark, not per-step allocation.
    pub vm_hwm_kib: Option<u64>,
}

impl StepEvidence {
    #[must_use]
    pub fn incremental_total_ns(self) -> u128 {
        u128::from(self.cache_refresh_ns) + u128::from(self.incremental_ns)
    }

    #[must_use]
    pub fn admissible(self, memory_cap_kib: u64) -> bool {
        self.output_equal
            && self.decision_equal
            && self.vm_hwm_kib.is_some_and(|v| v <= memory_cap_kib)
            && self.incremental_total_ns() < u128::from(self.full_ns)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequentialProbe<S> {
    pub canonical_state: S,
    pub revision: u64,
    pub evidence: Vec<StepEvidence>,
}

fn elapsed(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX).max(1)
}

/// Executes each mutation against the *previous committed Full state*.
/// Cache refresh and both evaluations are measured separately, alternating
/// evaluation order. A failed step returns an error without publishing a
/// partially updated sequence. All output comparisons use an independent Full
/// reference; no measured timings are fabricated.
pub fn run_sequential_probe<S, M, F, R, I, D>(
    initial: S,
    interventions: &[(u64, u64)],
    mut mutate: M,
    mut full: F,
    mut refresh: R,
    mut incremental: I,
    mut same_decision: D,
) -> Result<SequentialProbe<S>, SequenceError>
where
    S: Clone + Eq,
    M: FnMut(&mut S, usize),
    F: FnMut(&S) -> Result<S, SequenceError>,
    R: FnMut(&S) -> Result<(), SequenceError>,
    I: FnMut(&S) -> Result<S, SequenceError>,
    D: FnMut(&S, &S) -> bool,
{
    let mut canonical = initial;
    let mut revision = 0_u64;
    let mut evidence = Vec::with_capacity(interventions.len());
    for (index, &(gates, cone_gates)) in interventions.iter().enumerate() {
        if gates == 0 || cone_gates > gates {
            return Err(SequenceError::InvalidCone);
        }
        let next_revision = revision.checked_add(1).ok_or(SequenceError::RevisionOverflow)?;
        let mut input = canonical.clone();
        mutate(&mut input, index);
        let start = Instant::now();
        refresh(&input).map_err(|_| SequenceError::CacheFailed)?;
        let cache_refresh_ns = elapsed(start);
        let incremental_first = index % 2 == 1;
        let (reference, full_ns, candidate, incremental_ns) = if incremental_first {
            let start = Instant::now();
            let candidate = incremental(&input).map_err(|_| SequenceError::IncrementalFailed)?;
            let incremental_ns = elapsed(start);
            let start = Instant::now();
            let reference = full(&input).map_err(|_| SequenceError::FullFailed)?;
            (reference, elapsed(start), candidate, incremental_ns)
        } else {
            let start = Instant::now();
            let reference = full(&input).map_err(|_| SequenceError::FullFailed)?;
            let full_ns = elapsed(start);
            let start = Instant::now();
            let candidate = incremental(&input).map_err(|_| SequenceError::IncrementalFailed)?;
            (reference, full_ns, candidate, elapsed(start))
        };
        let output_equal = reference == candidate;
        let decision_equal = same_decision(&reference, &candidate);
        evidence.push(StepEvidence {
            revision: next_revision,
            gates,
            cone_gates,
            full_ns,
            cache_refresh_ns,
            incremental_ns,
            output_equal,
            decision_equal,
            incremental_first,
            vm_hwm_kib: crate::fourier5_sequential_physical_step::linux_vm_hwm_kib(),
        });
        canonical = reference;
        revision = next_revision;
    }
    Ok(SequentialProbe { canonical_state: canonical, revision, evidence })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heterogeneous_sequence_matches_independent_scalar_oracle() {
        let cones = [(100, 1), (100, 70), (100, 4), (100, 99), (100, 0)];
        let result = run_sequential_probe(
            0_i64,
            &cones,
            |s, index| *s += (index + 1) as i64,
            |s| Ok(*s * 2),
            |_| Ok(()),
            |s| Ok(*s * 2),
            |a, b| a == b,
        ).unwrap();
        let mut oracle = 0_i64;
        for index in 0..cones.len() {
            oracle = (oracle + (index + 1) as i64) * 2;
        }
        assert_eq!(result.canonical_state, oracle);
        assert_eq!(result.revision, 5);
        assert!(result.evidence.iter().enumerate().all(|(i, e)| e.incremental_first == (i % 2 == 1) && e.output_equal && e.decision_equal && e.revision == (i + 1) as u64));
    }

    #[test]
    fn corrupt_incremental_never_commits() {
        let result = run_sequential_probe(
            1_u64, &[(20, 1), (20, 15), (20, 2)],
            |s, _| *s += 1,
            |s| Ok(*s * 3),
            |_| Ok(()),
            |s| Ok(*s * 3 + u64::from(*s == 8)),
            |a, b| a == b,
        ).unwrap();
        assert_eq!(result.canonical_state, 81);
        assert!(!result.evidence[1].output_equal);
        assert!(result.evidence[0].output_equal);
    }

    #[test]
    fn invalid_cone_and_refresh_failure_are_fail_closed() {
        let invalid = run_sequential_probe(0_u8, &[(2, 3)], |_, _| {}, Ok, |_| Ok(()), Ok, |a, b| a == b);
        assert_eq!(invalid, Err(SequenceError::InvalidCone));
        let failure = run_sequential_probe(0_u8, &[(2, 1)], |_, _| {}, Ok, |_| Err(SequenceError::CacheFailed), Ok, |a, b| a == b);
        assert_eq!(failure, Err(SequenceError::CacheFailed));
    }

    #[test]
    fn admission_requires_memory_and_total_cost() {
        let mut e = StepEvidence { revision: 1, gates: 10, cone_gates: 2, full_ns: 100, cache_refresh_ns: 40, incremental_ns: 59, output_equal: true, decision_equal: true, incremental_first: false, vm_hwm_kib: Some(100) };
        assert!(e.admissible(100));
        e.incremental_ns = 60;
        assert!(!e.admissible(100));
        e.incremental_ns = 1;
        e.vm_hwm_kib = None;
        assert!(!e.admissible(100));
    }
}

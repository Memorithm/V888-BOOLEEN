//! Exact sequential paired-workload accounting for synthetic circuits.
//! Never use held-out observations to promote a dispatch policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceError { Empty, ZeroDuration, Overflow, OracleMismatch, DecisionMismatch, RevisionMismatch }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairedIntervention {
    pub accepted: bool,
    pub full_ns: u128,
    pub incremental_ns: u128,
    pub full_output: u64,
    pub incremental_output: u64,
    pub full_decision: bool,
    pub incremental_decision: bool,
    pub revision_before: u64,
    pub revision_after: u64,
    /// Process-wide peak RSS high-water mark; NOT attributable to one step.
    pub vm_hwm_kib: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SequenceTotals {
    pub steps: u64,
    pub accepted: u64,
    pub rejected: u64,
    pub full_ns: u128,
    pub incremental_ns: u128,
    pub full_wins: u64,
    pub incremental_wins: u64,
    pub ties: u64,
    pub max_vm_hwm_kib: Option<u64>,
}

impl SequenceTotals {
    /// Reject the entire trace on the first discrepancy; no partial result.
    pub fn verify(trace: &[PairedIntervention]) -> Result<Self, SequenceError> {
        if trace.is_empty() { return Err(SequenceError::Empty); }
        let mut result = Self::default();
        let mut previous_revision: Option<u64> = None;
        for item in trace {
            if previous_revision.is_some_and(|revision| revision != item.revision_before) {
                return Err(SequenceError::RevisionMismatch);
            }
            if item.full_ns == 0 || item.incremental_ns == 0 {
                return Err(SequenceError::ZeroDuration);
            }
            if item.full_output != item.incremental_output {
                return Err(SequenceError::OracleMismatch);
            }
            if item.full_decision != item.incremental_decision
                || item.full_decision != item.accepted {
                return Err(SequenceError::DecisionMismatch);
            }
            let expected = item.revision_before.checked_add(u64::from(item.accepted))
                .ok_or(SequenceError::Overflow)?;
            if item.revision_after != expected {
                return Err(SequenceError::RevisionMismatch);
            }
            previous_revision = Some(item.revision_after);
            result.steps = result.steps.checked_add(1).ok_or(SequenceError::Overflow)?;
            if item.accepted {
                result.accepted = result.accepted.checked_add(1).ok_or(SequenceError::Overflow)?;
            } else {
                result.rejected = result.rejected.checked_add(1).ok_or(SequenceError::Overflow)?;
            }
            result.full_ns = result.full_ns.checked_add(item.full_ns)
                .ok_or(SequenceError::Overflow)?;
            result.incremental_ns = result.incremental_ns.checked_add(item.incremental_ns)
                .ok_or(SequenceError::Overflow)?;
            if item.full_ns < item.incremental_ns {
                result.full_wins = result.full_wins.checked_add(1).ok_or(SequenceError::Overflow)?;
            } else if item.full_ns > item.incremental_ns {
                result.incremental_wins = result.incremental_wins.checked_add(1)
                    .ok_or(SequenceError::Overflow)?;
            } else {
                result.ties = result.ties.checked_add(1).ok_or(SequenceError::Overflow)?;
            }
            if let Some(kib) = item.vm_hwm_kib {
                result.max_vm_hwm_kib = Some(result.max_vm_hwm_kib.map_or(kib, |v| v.max(kib)));
            }
        }
        Ok(result)
    }

    /// Compare exact integer total durations; no floating point rounding.
    #[must_use]
    pub fn incremental_faster(&self) -> bool { self.incremental_ns < self.full_ns }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn record(accepted: bool, revision: u64, full: u128, inc: u128) -> PairedIntervention {
        PairedIntervention {
            accepted, full_ns: full, incremental_ns: inc,
            full_output: 17, incremental_output: 17,
            full_decision: accepted, incremental_decision: accepted,
            revision_before: revision,
            revision_after: revision + u64::from(accepted),
            vm_hwm_kib: Some(4096),
        }
    }

    #[test]
    fn mixed_sequential_trace_exact_totals_and_rejections() {
        let trace = [record(true, 0, 100, 10), record(false, 1, 20, 40),
                     record(true, 1, 30, 30)];
        let got = SequenceTotals::verify(&trace).unwrap();
        assert_eq!((got.steps, got.accepted, got.rejected), (3, 2, 1));
        assert_eq!((got.full_ns, got.incremental_ns), (150, 80));
        assert_eq!((got.full_wins, got.incremental_wins, got.ties), (1, 1, 1));
        assert_eq!(got.max_vm_hwm_kib, Some(4096));
        assert!(got.incremental_faster());
    }

    #[test]
    fn independent_oracle_rejects_corrupt_outputs_decisions_and_revisions() {
        let base = record(true, 5, 20, 10);
        let mut bad = base;
        bad.incremental_output ^= 1;
        assert_eq!(SequenceTotals::verify(&[base, bad]), Err(SequenceError::OracleMismatch));
        bad = base;
        bad.incremental_decision = false;
        assert_eq!(SequenceTotals::verify(&[bad]), Err(SequenceError::DecisionMismatch));
        bad = base;
        bad.revision_after = 5;
        assert_eq!(SequenceTotals::verify(&[bad]), Err(SequenceError::RevisionMismatch));
    }

    #[test]
    fn trace_revision_discontinuity_is_rejected() {
        let first = record(true, 0, 10, 5);
        let skipped = record(false, 9, 10, 5);
        assert_eq!(SequenceTotals::verify(&[first, skipped]), Err(SequenceError::RevisionMismatch));
    }

    #[test]
    fn empty_zero_and_overflow_fail_closed() {
        assert_eq!(SequenceTotals::verify(&[]), Err(SequenceError::Empty));
        let mut bad = record(false, 0, 1, 1);
        bad.incremental_ns = 0;
        assert_eq!(SequenceTotals::verify(&[bad]), Err(SequenceError::ZeroDuration));
        bad = record(true, 0, 1, 1);
        bad.revision_before = u64::MAX;
        assert_eq!(SequenceTotals::verify(&[bad]), Err(SequenceError::Overflow));
        let mut huge = record(false, 0, u128::MAX, 1);
        huge.vm_hwm_kib = None;
        assert_eq!(SequenceTotals::verify(&[huge, huge]), Err(SequenceError::Overflow));
    }
}

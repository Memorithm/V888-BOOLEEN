//! Streaming, split-isolated accounting for real sequential delta refreshes.
//! Only actual measured durations are accepted; this module invents no timings.
use crate::fourier5_delta_refresh::{DeltaMeasure, DeltaOutcome};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Split { Train, Valid, Test, Ood }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditError { ZeroFullTime, Overflow, InvalidGeometry, OutOfOrder }

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub attempted: u64,
    pub committed: u64,
    pub declined: u64,
    pub mismatched: u64,
    pub full_ns: u128,
    pub incremental_ns: u128,
    pub refresh_ns: u128,
    pub elapsed_ns: u128,
    pub peak_vm_hwm_kib: u64,
}

impl Counters {
    pub fn verified_cost_ns(&self) -> Option<u128> {
        self.full_ns.checked_add(self.incremental_ns)?.checked_add(self.refresh_ns)
    }
    pub fn committed_per_second(&self) -> Option<f64> {
        if self.elapsed_ns == 0 { return None; }
        Some(self.committed as f64 * 1_000_000_000.0 / self.elapsed_ns as f64)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalSample {
    pub split: Split,
    pub sequence: u64,
    pub gates: u64,
    pub cone: u64,
    pub vm_hwm_kib: u64,
    pub measure: DeltaMeasure,
}

/// Four isolated accumulators: holdout results never influence VALID dispatch.
#[derive(Default, Debug)]
pub struct SequentialAudit {
    totals: [Counters; 4],
    last_sequence: [Option<u64>; 4],
}

impl SequentialAudit {
    pub fn totals(&self, split: Split) -> Counters { self.totals[index(split)] }

    /// Validate and atomically incorporate a physically observed sample.
    /// Rejected samples cannot alter counters or ordering.
    pub fn record(&mut self, sample: PhysicalSample) -> Result<(), AuditError> {
        let i = index(sample.split);
        if sample.gates == 0 || sample.cone > sample.gates {
            return Err(AuditError::InvalidGeometry);
        }
        if self.last_sequence[i].is_some_and(|last| sample.sequence <= last) {
            return Err(AuditError::OutOfOrder);
        }
        if sample.measure.outcome != DeltaOutcome::Declined
            && sample.measure.full == Duration::ZERO {
            return Err(AuditError::ZeroFullTime);
        }
        let mut next = self.totals[i];
        next.attempted = next.attempted.checked_add(1).ok_or(AuditError::Overflow)?;
        match sample.measure.outcome {
            DeltaOutcome::Committed => next.committed = next.committed.checked_add(1).ok_or(AuditError::Overflow)?,
            DeltaOutcome::Declined => next.declined = next.declined.checked_add(1).ok_or(AuditError::Overflow)?,
            DeltaOutcome::OracleMismatch => next.mismatched = next.mismatched.checked_add(1).ok_or(AuditError::Overflow)?,
        }
        next.full_ns = next.full_ns.checked_add(sample.measure.full.as_nanos()).ok_or(AuditError::Overflow)?;
        next.incremental_ns = next.incremental_ns.checked_add(sample.measure.incremental.as_nanos()).ok_or(AuditError::Overflow)?;
        next.refresh_ns = next.refresh_ns.checked_add(sample.measure.refresh.as_nanos()).ok_or(AuditError::Overflow)?;
        next.elapsed_ns = next.elapsed_ns.checked_add(sample.measure.elapsed.as_nanos()).ok_or(AuditError::Overflow)?;
        next.peak_vm_hwm_kib = next.peak_vm_hwm_kib.max(sample.vm_hwm_kib);
        self.totals[i] = next;
        self.last_sequence[i] = Some(sample.sequence);
        Ok(())
    }
}

const fn index(split: Split) -> usize {
    match split { Split::Train => 0, Split::Valid => 1, Split::Test => 2, Split::Ood => 3 }
}

/// Linux process high-water RSS, in KiB; unavailable elsewhere.
/// VmHWM is a process-wide high-water mark, not per-operation allocation.
pub fn linux_vm_hwm_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix("VmHWM:")?;
        let mut fields = value.split_whitespace();
        let number = fields.next()?.parse::<u64>().ok()?;
        if fields.next()? == "kB" { Some(number) } else { None }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(split: Split, sequence: u64, outcome: DeltaOutcome) -> PhysicalSample {
        PhysicalSample {
            split, sequence, gates: 100, cone: 7, vm_hwm_kib: 1234,
            measure: DeltaMeasure {
                outcome, elapsed: Duration::from_nanos(20),
                refresh: Duration::from_nanos(3),
                full: Duration::from_nanos(10),
                incremental: Duration::from_nanos(4),
            },
        }
    }

    #[test]
    fn holdouts_do_not_mutate_valid_accounting() {
        let mut audit = SequentialAudit::default();
        audit.record(sample(Split::Valid, 0, DeltaOutcome::Committed)).unwrap();
        let frozen = audit.totals(Split::Valid);
        for split in [Split::Train, Split::Test, Split::Ood] {
            audit.record(sample(split, 0, DeltaOutcome::OracleMismatch)).unwrap();
        }
        assert_eq!(audit.totals(Split::Valid), frozen);
        assert_eq!(frozen.verified_cost_ns(), Some(17));
        assert_eq!(frozen.committed_per_second(), Some(50_000_000.0));
    }

    #[test]
    fn invalid_and_reordered_samples_are_atomic() {
        let mut audit = SequentialAudit::default();
        audit.record(sample(Split::Valid, 4, DeltaOutcome::Committed)).unwrap();
        let before = audit.totals(Split::Valid);
        assert_eq!(audit.record(sample(Split::Valid, 4, DeltaOutcome::Committed)), Err(AuditError::OutOfOrder));
        let mut invalid = sample(Split::Valid, 5, DeltaOutcome::Committed);
        invalid.cone = 101;
        assert_eq!(audit.record(invalid), Err(AuditError::InvalidGeometry));
        assert_eq!(audit.totals(Split::Valid), before);
        audit.record(sample(Split::Valid, 5, DeltaOutcome::Committed)).unwrap();
        assert_eq!(audit.totals(Split::Valid).attempted, 2);
    }

    #[test]
    fn full_oracle_time_is_never_omitted() {
        let mut audit = SequentialAudit::default();
        let mut invalid = sample(Split::Test, 0, DeltaOutcome::Committed);
        invalid.measure.full = Duration::ZERO;
        assert_eq!(audit.record(invalid), Err(AuditError::ZeroFullTime));
        audit.record(sample(Split::Test, 0, DeltaOutcome::Committed)).unwrap();
        assert_eq!(audit.totals(Split::Test).full_ns, 10);
    }
}

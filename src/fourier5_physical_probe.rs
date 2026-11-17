//! Linux process-wide high-water RSS sampling for bounded sequential experiments.
//! This is a measurement utility, not a biological or performance claim.
use std::io;
use std::time::{Duration, Instant};

/// A sampled process-wide high-water resident set size in KiB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeakRss {
    pub kib: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseError {
    Missing,
    Duplicate,
    Malformed,
    Overflow,
}

/// Strictly parse the VmHWM field from Linux /proc/self/status.
/// Reject ambiguous units, duplicates and truncated values.
pub fn parse_vmhwm(status: &str) -> Result<PeakRss, ParseError> {
    let mut result = None;
    for line in status.lines() {
        let Some(rest) = line.strip_prefix("VmHWM:") else {
            continue;
        };
        if result.is_some() {
            return Err(ParseError::Duplicate);
        }
        let mut fields = rest.split_whitespace();
        let number = fields.next().ok_or(ParseError::Malformed)?;
        let unit = fields.next().ok_or(ParseError::Malformed)?;
        if unit != "kB" || fields.next().is_some() || !number.bytes().all(|b| b.is_ascii_digit()) {
            return Err(ParseError::Malformed);
        }
        let kib = number.parse::<u64>().map_err(|_| ParseError::Overflow)?;
        result = Some(PeakRss { kib });
    }
    result.ok_or(ParseError::Missing)
}

/// Read Linux's process-wide high-water RSS. This is NOT an instantaneous or
/// per-intervention allocation measurement, and can only increase per process.
#[cfg(target_os = "linux")]
pub fn process_peak_rss() -> io::Result<PeakRss> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    parse_vmhwm(&status).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{e:?}")))
}

#[cfg(not(target_os = "linux"))]
pub fn process_peak_rss() -> io::Result<PeakRss> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "VmHWM requires Linux",
    ))
}

/// Time one intervention including the caller's complete cache refresh.
/// The callback must perform the full operation; this utility does not infer
/// whether a cache was rebuilt or whether an intervention was accepted.
pub fn time_intervention<T, E>(
    operation: impl FnOnce() -> Result<T, E>,
) -> (Duration, Result<T, E>) {
    let start = Instant::now();
    let result = operation();
    (start.elapsed(), result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_linux_status_without_confusing_vmrss() {
        let status = "Name:\tprobe\nVmRSS:\t2048 kB\nVmHWM:\t4096 kB\n";
        assert_eq!(parse_vmhwm(status), Ok(PeakRss { kib: 4096 }));
    }

    #[test]
    fn rejects_duplicates_and_unknown_units() {
        assert_eq!(
            parse_vmhwm("VmHWM: 1 kB\nVmHWM: 2 kB"),
            Err(ParseError::Duplicate)
        );
        assert_eq!(parse_vmhwm("VmHWM: 2 MB"), Err(ParseError::Malformed));
        assert_eq!(parse_vmhwm("VmRSS: 2 kB"), Err(ParseError::Missing));
    }

    #[test]
    fn rejects_overflow_and_trailing_fields() {
        assert_eq!(
            parse_vmhwm("VmHWM: 18446744073709551616 kB"),
            Err(ParseError::Overflow)
        );
        assert_eq!(
            parse_vmhwm("VmHWM: 12 kB extra"),
            Err(ParseError::Malformed)
        );
    }

    #[test]
    fn times_success_and_failure_without_changing_result() {
        let (_, ok) = time_intervention(|| Ok::<_, ()>(17));
        let (_, err) = time_intervention(|| Err::<(), _>(9));
        assert_eq!(ok, Ok(17));
        assert_eq!(err, Err(9));
    }
}

/// One real process-level sample around an entire sequential intervention.
/// The high-water mark is cumulative and cannot attribute memory to this step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterventionSample {
    pub elapsed: Duration,
    pub before: PeakRss,
    pub after: PeakRss,
}

impl InterventionSample {
    /// Conservative non-attributable increase in process-wide peak RSS.
    #[must_use]
    pub fn peak_increase_kib(self) -> u64 {
        self.after.kib.saturating_sub(self.before.kib)
    }
}

/// Measure a complete intervention, including any cache reconstruction inside
/// the callback. Failure to sample memory before the operation prevents mutation;
/// failure after the operation returns an I/O error even if mutation committed.
/// Callers must NOT interpret that error as a rollback; reconcile session state.
pub fn sample_intervention<T, E>(
    operation: impl FnOnce() -> Result<T, E>,
) -> io::Result<(InterventionSample, Result<T, E>)> {
    let before = process_peak_rss()?;
    let (elapsed, outcome) = time_intervention(operation);
    let after = process_peak_rss()?;
    Ok((
        InterventionSample {
            elapsed,
            before,
            after,
        },
        outcome,
    ))
}

#[cfg(test)]
mod sequential_sample_tests {
    use super::*;

    #[test]
    fn peak_delta_is_saturating_not_an_allocation_claim() {
        let sample = InterventionSample {
            elapsed: Duration::from_nanos(17),
            before: PeakRss { kib: 900 },
            after: PeakRss { kib: 950 },
        };
        assert_eq!(sample.peak_increase_kib(), 50);
        assert_eq!(
            InterventionSample {
                after: PeakRss { kib: 800 },
                ..sample
            }
            .peak_increase_kib(),
            0
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn physical_sample_preserves_success_and_error() {
        let (ok_sample, ok) = sample_intervention(|| Ok::<_, u8>(42)).unwrap();
        assert_eq!(ok, Ok(42));
        assert!(ok_sample.after.kib >= ok_sample.before.kib);
        let (_, error) = sample_intervention(|| Err::<u8, _>(7)).unwrap();
        assert_eq!(error, Err(7));
    }
}

/// A completed operation with its outcome preserved even when post-operation
/// process-memory sampling fails. A post-sample error is NOT a rollback.
#[derive(Debug)]
pub struct PreservedIntervention<T, E> {
    pub elapsed: Duration,
    pub before: PeakRss,
    pub after: io::Result<PeakRss>,
    pub outcome: Result<T, E>,
}

/// Run one operation exactly once. A failed pre-sample prevents execution;
/// a failed post-sample never discards the operation's actual outcome.
pub fn sample_intervention_preserving_outcome<T, E>(
    operation: impl FnOnce() -> Result<T, E>,
) -> io::Result<PreservedIntervention<T, E>> {
    sample_intervention_with(process_peak_rss, operation)
}

fn sample_intervention_with<T, E>(
    mut sample: impl FnMut() -> io::Result<PeakRss>,
    operation: impl FnOnce() -> Result<T, E>,
) -> io::Result<PreservedIntervention<T, E>> {
    let before = sample()?;
    let (elapsed, outcome) = time_intervention(operation);
    let after = sample();
    Ok(PreservedIntervention {
        elapsed,
        before,
        after,
        outcome,
    })
}

#[cfg(test)]
mod preserving_outcome_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn pre_sample_failure_does_not_run_mutation() {
        let calls = Cell::new(0);
        let result = sample_intervention_with(
            || Err(io::Error::other("before")),
            || {
                calls.set(calls.get() + 1);
                Ok::<_, ()>(())
            },
        );
        assert!(result.is_err());
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn post_sample_failure_preserves_committed_result() {
        let reads = Cell::new(0);
        let mutations = Cell::new(0);
        let result = sample_intervention_with(
            || {
                reads.set(reads.get() + 1);
                if reads.get() == 1 {
                    Ok(PeakRss { kib: 12 })
                } else {
                    Err(io::Error::other("after"))
                }
            },
            || {
                mutations.set(mutations.get() + 1);
                Ok::<_, u8>(17)
            },
        )
        .unwrap();
        assert_eq!(mutations.get(), 1);
        assert_eq!(result.outcome, Ok(17));
        assert_eq!(result.before.kib, 12);
        assert!(result.after.is_err());
    }

    #[test]
    fn post_sample_success_preserves_rejected_intervention() {
        let result =
            sample_intervention_with(|| Ok(PeakRss { kib: 20 }), || Err::<(), _>("rejected"))
                .unwrap();
        assert_eq!(result.outcome, Err("rejected"));
        assert_eq!(result.after.unwrap().kib, 20);
    }
}

/// Two process-wide Linux memory indicators sampled from the same status read.
/// VmRSS is instantaneous; VmHWM is the historical maximum. Neither isolates
/// allocations by an individual intervention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemorySnapshot {
    pub current_rss: PeakRss,
    pub peak_rss: PeakRss,
}

/// Parse both RSS fields atomically from one status document. A missing,
/// duplicate, malformed or overflowing field fails the entire snapshot.
pub fn parse_memory_snapshot(status: &str) -> Result<MemorySnapshot, ParseError> {
    let peak_rss = parse_vmhwm(status)?;
    let mut current = None;
    for line in status.lines() {
        let Some(rest) = line.strip_prefix("VmRSS:") else {
            continue;
        };
        if current.is_some() {
            return Err(ParseError::Duplicate);
        }
        let mut fields = rest.split_whitespace();
        let number = fields.next().ok_or(ParseError::Malformed)?;
        let unit = fields.next().ok_or(ParseError::Malformed)?;
        if unit != "kB" || fields.next().is_some() || !number.bytes().all(|b| b.is_ascii_digit()) {
            return Err(ParseError::Malformed);
        }
        current = Some(PeakRss {
            kib: number.parse::<u64>().map_err(|_| ParseError::Overflow)?,
        });
    }
    let current_rss = current.ok_or(ParseError::Missing)?;
    if current_rss.kib > peak_rss.kib {
        return Err(ParseError::Malformed);
    }
    Ok(MemorySnapshot {
        current_rss,
        peak_rss,
    })
}

/// Read current and peak RSS from a single /proc/self/status snapshot.
#[cfg(target_os = "linux")]
pub fn process_memory_snapshot() -> io::Result<MemorySnapshot> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    parse_memory_snapshot(&status)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{e:?}")))
}

#[cfg(not(target_os = "linux"))]
pub fn process_memory_snapshot() -> io::Result<MemorySnapshot> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "Linux /proc required",
    ))
}

#[cfg(test)]
mod memory_snapshot_tests {
    use super::*;

    #[test]
    fn distinguishes_current_and_historical_memory() {
        let snapshot = parse_memory_snapshot("VmRSS: 120 kB\nVmHWM: 900 kB\n").unwrap();
        assert_eq!(snapshot.current_rss.kib, 120);
        assert_eq!(snapshot.peak_rss.kib, 900);
    }

    #[test]
    fn rejects_missing_duplicate_and_impossible_current_rss() {
        assert_eq!(
            parse_memory_snapshot("VmHWM: 5 kB\n"),
            Err(ParseError::Missing)
        );
        assert_eq!(
            parse_memory_snapshot("VmRSS: 2 kB\nVmRSS: 3 kB\nVmHWM: 5 kB\n"),
            Err(ParseError::Duplicate)
        );
        assert_eq!(
            parse_memory_snapshot("VmRSS: 6 kB\nVmHWM: 5 kB\n"),
            Err(ParseError::Malformed)
        );
    }

    #[test]
    fn rejects_overflow_and_ambiguous_units() {
        assert_eq!(
            parse_memory_snapshot("VmRSS: 18446744073709551616 kB\nVmHWM: 5 kB\n"),
            Err(ParseError::Overflow)
        );
        assert_eq!(
            parse_memory_snapshot("VmRSS: 4 MB\nVmHWM: 5 kB\n"),
            Err(ParseError::Malformed)
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_snapshot_respects_peak_invariant() {
        let sample = process_memory_snapshot().unwrap();
        assert!(sample.current_rss.kib <= sample.peak_rss.kib);
    }
}

/// One sequential step with process-wide memory sampled around the complete
/// operation, including any cache refresh performed by the callback.
/// A failed post-sample is retained alongside the committed operation outcome.
#[derive(Debug)]
pub struct SequentialStep<T, E> {
    pub index: usize,
    pub elapsed: Duration,
    pub before: MemorySnapshot,
    pub after: io::Result<MemorySnapshot>,
    pub outcome: Result<T, E>,
}

/// Execute operations in order, exactly once each, without losing outcomes on
/// post-sampling failures. A pre-sample failure stops before the next mutation.
/// No rollback, oracle equivalence, or cache correctness is implied.
pub fn sample_sequential<T, E, I, F>(
    operations: I,
    mut operation: F,
) -> io::Result<Vec<SequentialStep<T, E>>>
where
    I: IntoIterator,
    F: FnMut(I::Item) -> Result<T, E>,
{
    sample_sequential_with(operations, process_memory_snapshot, &mut operation)
}

fn sample_sequential_with<T, E, I, F, S>(
    operations: I,
    mut sample: S,
    operation: &mut F,
) -> io::Result<Vec<SequentialStep<T, E>>>
where
    I: IntoIterator,
    F: FnMut(I::Item) -> Result<T, E>,
    S: FnMut() -> io::Result<MemorySnapshot>,
{
    let mut steps = Vec::new();
    for (index, input) in operations.into_iter().enumerate() {
        let before = sample()?;
        let (elapsed, outcome) = time_intervention(|| operation(input));
        let after = sample();
        steps.push(SequentialStep {
            index,
            elapsed,
            before,
            after,
            outcome,
        });
    }
    Ok(steps)
}

#[cfg(test)]
mod sequential_execution_tests {
    use super::*;
    use std::cell::Cell;

    fn snapshot(current: u64, peak: u64) -> MemorySnapshot {
        MemorySnapshot {
            current_rss: PeakRss { kib: current },
            peak_rss: PeakRss { kib: peak },
        }
    }

    #[test]
    fn heterogeneous_steps_execute_once_and_in_order() {
        let mut seen = Vec::new();
        let steps = sample_sequential_with([3_i32, -2, 7], || Ok(snapshot(12, 20)), &mut |input| {
            seen.push(input);
            if input < 0 {
                Err(input)
            } else {
                Ok(input * 2)
            }
        })
        .unwrap();
        assert_eq!(seen, vec![3, -2, 7]);
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].outcome, Ok(6));
        assert_eq!(steps[1].outcome, Err(-2));
        assert_eq!(steps[2].outcome, Ok(14));
        assert_eq!(
            steps.iter().map(|s| s.index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn failed_post_sample_preserves_outcome_and_continues() {
        let reads = Cell::new(0);
        let steps = sample_sequential_with(
            [1_u8, 2],
            || {
                reads.set(reads.get() + 1);
                if reads.get() == 2 {
                    Err(io::Error::other("post"))
                } else {
                    Ok(snapshot(8, 16))
                }
            },
            &mut |n| Ok::<_, ()>(n),
        )
        .unwrap();
        assert!(steps[0].after.is_err());
        assert_eq!(steps[0].outcome, Ok(1));
        assert_eq!(steps[1].outcome, Ok(2));
        assert!(steps[1].after.is_ok());
    }

    #[test]
    fn failed_pre_sample_prevents_next_mutation() {
        let reads = Cell::new(0);
        let mutations = Cell::new(0);
        let result = sample_sequential_with(
            [1, 2],
            || {
                reads.set(reads.get() + 1);
                if reads.get() == 3 {
                    Err(io::Error::other("pre"))
                } else {
                    Ok(snapshot(8, 16))
                }
            },
            &mut |_| {
                mutations.set(mutations.get() + 1);
                Ok::<_, ()>(())
            },
        );
        assert!(result.is_err());
        assert_eq!(mutations.get(), 1);
    }
}

/// Preserves every completed step if sampling fails before a later operation.
/// Unlike `sample_sequential`, the partial trace is returned on pre-sample
/// failure, so a caller can reconcile already committed state without guessing.
#[derive(Debug)]
pub struct SequentialRun<T, E> {
    pub completed: Vec<SequentialStep<T, E>>,
    pub stopped_before: Option<usize>,
    pub sampling_error: Option<io::Error>,
}

impl<T, E> SequentialRun<T, E> {
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.sampling_error.is_none()
    }
}

/// Execute sequential operations while retaining completed results on an
/// unsuccessful pre-sample. Post-sample failures remain attached to the step.
/// This does not validate cache coherence; callers must use a Full oracle.
pub fn sample_sequential_recoverable<T, E, I, F>(
    operations: I,
    mut operation: F,
) -> SequentialRun<T, E>
where
    I: IntoIterator,
    F: FnMut(I::Item) -> Result<T, E>,
{
    sample_sequential_recoverable_with(operations, process_memory_snapshot, &mut operation)
}

fn sample_sequential_recoverable_with<T, E, I, F, S>(
    operations: I,
    mut sample: S,
    operation: &mut F,
) -> SequentialRun<T, E>
where
    I: IntoIterator,
    F: FnMut(I::Item) -> Result<T, E>,
    S: FnMut() -> io::Result<MemorySnapshot>,
{
    let mut completed = Vec::new();
    for (index, input) in operations.into_iter().enumerate() {
        let before = match sample() {
            Ok(value) => value,
            Err(error) => {
                return SequentialRun {
                    completed,
                    stopped_before: Some(index),
                    sampling_error: Some(error),
                };
            }
        };
        let (elapsed, outcome) = time_intervention(|| operation(input));
        let after = sample();
        completed.push(SequentialStep {
            index,
            elapsed,
            before,
            after,
            outcome,
        });
    }
    SequentialRun {
        completed,
        stopped_before: None,
        sampling_error: None,
    }
}

#[cfg(test)]
mod recoverable_sequence_tests {
    use super::*;
    use std::cell::Cell;

    fn snap() -> MemorySnapshot {
        MemorySnapshot {
            current_rss: PeakRss { kib: 8 },
            peak_rss: PeakRss { kib: 16 },
        }
    }

    #[test]
    fn preserves_completed_mutations_when_next_pre_sample_fails() {
        let reads = Cell::new(0);
        let calls = Cell::new(0);
        let run = sample_sequential_recoverable_with(
            [10, 20, 30],
            || {
                reads.set(reads.get() + 1);
                if reads.get() == 5 {
                    Err(io::Error::other("pre"))
                } else {
                    Ok(snap())
                }
            },
            &mut |n| {
                calls.set(calls.get() + 1);
                Ok::<_, ()>(n)
            },
        );
        assert!(!run.is_complete());
        assert_eq!(run.stopped_before, Some(2));
        assert_eq!(calls.get(), 2);
        assert_eq!(run.completed.len(), 2);
        assert_eq!(run.completed[0].outcome, Ok(10));
        assert_eq!(run.completed[1].outcome, Ok(20));
    }

    #[test]
    fn post_sample_failure_is_not_a_rollback_or_stop() {
        let reads = Cell::new(0);
        let run = sample_sequential_recoverable_with(
            [1, 2],
            || {
                reads.set(reads.get() + 1);
                if reads.get() == 2 {
                    Err(io::Error::other("post"))
                } else {
                    Ok(snap())
                }
            },
            &mut |n| Ok::<_, ()>(n),
        );
        assert!(run.is_complete());
        assert_eq!(run.completed.len(), 2);
        assert!(run.completed[0].after.is_err());
        assert_eq!(run.completed[1].outcome, Ok(2));
    }

    #[test]
    fn first_pre_sample_failure_executes_nothing() {
        let calls = Cell::new(0);
        let run =
            sample_sequential_recoverable_with([1], || Err(io::Error::other("pre")), &mut |_| {
                calls.set(calls.get() + 1);
                Ok::<_, ()>(())
            });
        assert_eq!(calls.get(), 0);
        assert_eq!(run.stopped_before, Some(0));
        assert!(run.completed.is_empty());
    }
}

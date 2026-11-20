//! Bounded heterogeneous sequential workload audit for synthetic state machines.
//! Full is an independent oracle; a rejected step never changes accepted state.
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepVerdict {
    Accepted,
    Mismatch,
    FullError,
    IncrementalError,
}

#[derive(Clone, Debug)]
pub struct StepMeasurement {
    pub verdict: StepVerdict,
    pub full: Duration,
    /// Includes incremental cache refresh/rebuild performed by the closure.
    pub incremental_with_refresh: Duration,
    pub peak_rss_kib: Option<u64>,
}

/// Linux VmHWM is a process lifetime high-water mark, not per-step memory.
/// None on unsupported systems or when the kernel does not expose it.
#[must_use]
pub fn process_vm_hwm_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let value = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))?;
    let mut parts = value.split_whitespace();
    let amount = parts.next()?.parse::<u64>().ok()?;
    if parts.next()? != "kB" || parts.next().is_some() {
        return None;
    }
    Some(amount)
}

/// Runs both paths from the same last accepted state, commits only on exact
/// equality. The caller supplies independent implementations; the module
/// cannot establish independence from closure types alone.
/// Errors are classified, not swallowed into a success measurement.
pub fn audit_sequence<S, I, E, F, D>(
    initial: S,
    inputs: &[I],
    mut full: F,
    mut incremental: D,
) -> (S, Vec<StepMeasurement>)
where
    S: Clone + Eq,
    F: FnMut(&S, &I) -> Result<S, E>,
    D: FnMut(&S, &I) -> Result<S, E>,
{
    let mut accepted = initial;
    let mut report = Vec::with_capacity(inputs.len());
    for input in inputs {
        let start = Instant::now();
        let reference = full(&accepted, input);
        let full_time = start.elapsed();
        // Always execute both paths for a paired observation, even on Full
        // error; no unverified state is committed.
        let start = Instant::now();
        let candidate = incremental(&accepted, input);
        let incremental_time = start.elapsed();
        let verdict = match (reference, candidate) {
            (Err(_), _) => StepVerdict::FullError,
            (Ok(_), Err(_)) => StepVerdict::IncrementalError,
            (Ok(expected), Ok(actual)) if expected == actual => {
                accepted = actual;
                StepVerdict::Accepted
            }
            (Ok(_), Ok(_)) => StepVerdict::Mismatch,
        };
        report.push(StepMeasurement {
            verdict,
            full: full_time,
            incremental_with_refresh: incremental_time,
            peak_rss_kib: process_vm_hwm_kib(),
        });
    }
    (accepted, report)
}


#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorkloadSummary {
    pub accepted: usize,
    pub mismatches: usize,
    pub full_errors: usize,
    pub incremental_errors: usize,
    pub full_nanos: u128,
    pub incremental_nanos: u128,
    /// Process-wide high-water mark; not the allocation of this workload.
    pub process_peak_rss_kib: Option<u64>,
}

/// Aggregate a heterogeneous sequence without hiding rejected interventions.
/// Returns None on arithmetic overflow rather than publishing a partial total.
/// Durations include all attempted steps, not only successful interventions.
#[must_use]
pub fn summarize_workload(rows: &[StepMeasurement]) -> Option<WorkloadSummary> {
    let mut summary = WorkloadSummary::default();
    for row in rows {
        let count = match row.verdict {
            StepVerdict::Accepted => &mut summary.accepted,
            StepVerdict::Mismatch => &mut summary.mismatches,
            StepVerdict::FullError => &mut summary.full_errors,
            StepVerdict::IncrementalError => &mut summary.incremental_errors,
        };
        *count = count.checked_add(1)?;
        summary.full_nanos = summary.full_nanos.checked_add(row.full.as_nanos())?;
        summary.incremental_nanos = summary
            .incremental_nanos
            .checked_add(row.incremental_with_refresh.as_nanos())?;
        summary.process_peak_rss_kib = match (summary.process_peak_rss_kib, row.peak_rss_kib) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, value) => value,
        };
    }
    Some(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_includes_rejections_in_paired_cost_and_peak() {
        let rows = [
            StepMeasurement {
                verdict: StepVerdict::Accepted,
                full: Duration::from_nanos(10),
                incremental_with_refresh: Duration::from_nanos(3),
                peak_rss_kib: Some(1024),
            },
            StepMeasurement {
                verdict: StepVerdict::Mismatch,
                full: Duration::from_nanos(20),
                incremental_with_refresh: Duration::from_nanos(8),
                peak_rss_kib: Some(2048),
            },
            StepMeasurement {
                verdict: StepVerdict::FullError,
                full: Duration::from_nanos(5),
                incremental_with_refresh: Duration::from_nanos(7),
                peak_rss_kib: None,
            },
            StepMeasurement {
                verdict: StepVerdict::IncrementalError,
                full: Duration::from_nanos(9),
                incremental_with_refresh: Duration::from_nanos(4),
                peak_rss_kib: Some(512),
            },
        ];
        let summary = summarize_workload(&rows).unwrap();
        assert_eq!((summary.accepted, summary.mismatches, summary.full_errors, summary.incremental_errors), (1, 1, 1, 1));
        assert_eq!((summary.full_nanos, summary.incremental_nanos), (44, 22));
        assert_eq!(summary.process_peak_rss_kib, Some(2048));
    }

    #[test]
    fn empty_summary_has_no_invented_memory_measurement() {
        assert_eq!(summarize_workload(&[]), Some(WorkloadSummary::default()));
    }


    #[test]
    fn independent_oracle_rejects_stale_internal_cache_even_if_output_matches() {
        // (visible output, internal cache epoch)
        let inputs = [1_u8, 2, 3, 4];
        let (state, rows) = audit_sequence(
            (0_u8, 0_u8),
            &inputs,
            |&(output, epoch), &delta| Ok::<_, ()>((output + delta, epoch + 1)),
            |&(output, epoch), &delta| Ok::<_, ()>((output + delta, epoch + u8::from(delta != 2))),
        );
        assert_eq!(state, (8, 3)); // 1 + rejected 2 + 3 + 4
        assert_eq!(
            rows.iter().map(|r| r.verdict).collect::<Vec<_>>(),
            vec![
                StepVerdict::Accepted,
                StepVerdict::Mismatch,
                StepVerdict::Accepted,
                StepVerdict::Accepted,
            ]
        );
    }

    #[test]
    fn errors_do_not_commit_and_later_steps_use_last_accepted_state() {
        let (state, rows) = audit_sequence(
            0_i32,
            &[1_i32, 2, 3, 4],
            |&state, &delta| {
                if delta == 2 {
                    Err("oracle")
                } else {
                    Ok(state + delta)
                }
            },
            |&state, &delta| {
                if delta == 3 {
                    Err("incremental")
                } else {
                    Ok(state + delta)
                }
            },
        );
        assert_eq!(state, 5);
        assert_eq!(
            rows.iter().map(|r| r.verdict).collect::<Vec<_>>(),
            vec![
                StepVerdict::Accepted,
                StepVerdict::FullError,
                StepVerdict::IncrementalError,
                StepVerdict::Accepted,
            ]
        );
    }

    #[test]
    fn empty_workload_is_identity() {
        let (state, rows) = audit_sequence(
            vec![true, false],
            &[] as &[u8],
            |state, _| Ok::<_, ()>(state.clone()),
            |state, _| Ok::<_, ()>(state.clone()),
        );
        assert_eq!(state, vec![true, false]);
        assert!(rows.is_empty());
    }
}

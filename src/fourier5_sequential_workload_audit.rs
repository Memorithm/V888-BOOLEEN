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

#[cfg(test)]
mod tests {
    use super::*;

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

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


/// Counterbalanced paired audit: odd steps time Incremental before Full,
/// even steps Full before Incremental. This reduces fixed execution-order
/// bias in heterogeneous synthetic workloads, without claiming to remove
/// cache, thermal or process-level interference. Both paths always start
/// from the same accepted state; only exact equality commits a candidate.
/// Full errors take precedence, regardless of evaluation order.
pub fn audit_sequence_counterbalanced<S, I, E, F, D>(
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
    for (index, input) in inputs.iter().enumerate() {
        let (reference, full_time, candidate, incremental_time) = if index % 2 == 0 {
            let start = Instant::now();
            let reference = full(&accepted, input);
            let full_time = start.elapsed();
            let start = Instant::now();
            let candidate = incremental(&accepted, input);
            (reference, full_time, candidate, start.elapsed())
        } else {
            let start = Instant::now();
            let candidate = incremental(&accepted, input);
            let incremental_time = start.elapsed();
            let start = Instant::now();
            let reference = full(&accepted, input);
            (reference, start.elapsed(), candidate, incremental_time)
        };
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
mod counterbalanced_tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn order_alternates_and_rejected_steps_do_not_change_state() {
        let calls = RefCell::new(Vec::new());
        let (state, rows) = audit_sequence_counterbalanced(
            (0_i32, 0_u32),
            &[1_i32, 2, 3, 4],
            |&(value, epoch), &delta| {
                calls.borrow_mut().push(('F', delta));
                Ok::<_, ()>((value + delta, epoch + 1))
            },
            |&(value, epoch), &delta| {
                calls.borrow_mut().push(('I', delta));
                Ok::<_, ()>((value + delta, epoch + u32::from(delta != 2)))
            },
        );
        assert_eq!(calls.into_inner(), vec![
            ('F', 1), ('I', 1), ('I', 2), ('F', 2),
            ('F', 3), ('I', 3), ('I', 4), ('F', 4),
        ]);
        assert_eq!(state, (8, 3));
        assert_eq!(
            rows.iter().map(|row| row.verdict).collect::<Vec<_>>(),
            vec![StepVerdict::Accepted, StepVerdict::Mismatch,
                 StepVerdict::Accepted, StepVerdict::Accepted]
        );
    }

    #[test]
    fn full_error_precedes_incremental_error_in_both_orders() {
        let (state, rows) = audit_sequence_counterbalanced(
            0_i32,
            &[1_i32, 2],
            |_, _| Err::<i32, ()>(()),
            |_, _| Err::<i32, ()>(()),
        );
        assert_eq!(state, 0);
        assert!(rows.iter().all(|row| row.verdict == StepVerdict::FullError));
    }

    #[test]
    fn oracle_matches_independent_scalar_replay_on_heterogeneous_steps() {
        let deltas = [3_i64, -2, 7, 0, -9, 11, 1];
        let (state, rows) = audit_sequence_counterbalanced(
            0_i64,
            &deltas,
            |&value, &delta| value.checked_add(delta).ok_or(()),
            |&value, &delta| value.checked_add(delta).ok_or(()),
        );
        assert_eq!(state, deltas.iter().copied().sum::<i64>());
        assert!(rows.iter().all(|row| row.verdict == StepVerdict::Accepted));
    }
}

/// Paired audit with execution order balanced *within each workload class*.
/// Classes must be assigned independently of measured timings and outcomes.
/// This prevents an alternating global order from always placing the same
/// engine first for a class appearing only at even (or odd) positions.
/// The order counter advances on rejected steps as well.
pub fn audit_sequence_stratified<S, I, K, E, F, D, C>(
    initial: S,
    inputs: &[I],
    mut class_of: C,
    mut full: F,
    mut incremental: D,
) -> (S, Vec<StepMeasurement>)
where
    S: Clone + Eq,
    K: Eq,
    C: FnMut(&I) -> K,
    F: FnMut(&S, &I) -> Result<S, E>,
    D: FnMut(&S, &I) -> Result<S, E>,
{
    let mut accepted = initial;
    let mut classes: Vec<(K, usize)> = Vec::new();
    let mut report = Vec::with_capacity(inputs.len());
    for input in inputs {
        let class = class_of(input);
        let first_incremental = if let Some((_, count)) =
            classes.iter_mut().find(|(key, _)| *key == class)
        {
            let odd = *count % 2 == 1;
            *count += 1;
            odd
        } else {
            classes.push((class, 1));
            false
        };
        let (reference, full_time, candidate, incremental_time) = if first_incremental {
            let start = Instant::now();
            let candidate = incremental(&accepted, input);
            let incremental_time = start.elapsed();
            let start = Instant::now();
            let reference = full(&accepted, input);
            (reference, start.elapsed(), candidate, incremental_time)
        } else {
            let start = Instant::now();
            let reference = full(&accepted, input);
            let full_time = start.elapsed();
            let start = Instant::now();
            let candidate = incremental(&accepted, input);
            (reference, full_time, candidate, start.elapsed())
        };
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
mod stratified_tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn each_class_balances_even_when_interleaved() {
        let calls = RefCell::new(Vec::new());
        let inputs = [(0_u8, 1_i32), (1, 2), (0, 3), (1, 4), (0, 5)];
        let (state, rows) = audit_sequence_stratified(
            0_i32,
            &inputs,
            |&(class, _)| class,
            |&value, &(class, delta)| {
                calls.borrow_mut().push(('F', class));
                Ok::<_, ()>(value + delta)
            },
            |&value, &(class, delta)| {
                calls.borrow_mut().push(('I', class));
                Ok::<_, ()>(value + delta)
            },
        );
        assert_eq!(state, 15);
        assert!(rows.iter().all(|row| row.verdict == StepVerdict::Accepted));
        assert_eq!(
            calls.into_inner(),
            vec![
                ('F', 0), ('I', 0),
                ('F', 1), ('I', 1),
                ('I', 0), ('F', 0),
                ('I', 1), ('F', 1),
                ('F', 0), ('I', 0),
            ]
        );
    }

    #[test]
    fn rejected_step_still_advances_class_order_and_rolls_back() {
        let calls = RefCell::new(Vec::new());
        let (state, rows) = audit_sequence_stratified(
            (0_i32, 0_u32),
            &[1_i32, 2, 3],
            |_| 0_u8,
            |&(value, epoch), &delta| {
                calls.borrow_mut().push('F');
                Ok::<_, ()>((value + delta, epoch + 1))
            },
            |&(value, epoch), &delta| {
                calls.borrow_mut().push('I');
                Ok::<_, ()>((value + delta, epoch + u32::from(delta != 2)))
            },
        );
        assert_eq!(state, (4, 2));
        assert_eq!(
            rows.iter().map(|row| row.verdict).collect::<Vec<_>>(),
            vec![StepVerdict::Accepted, StepVerdict::Mismatch, StepVerdict::Accepted]
        );
        assert_eq!(calls.into_inner(), vec!['F', 'I', 'I', 'F', 'F', 'I']);
    }
}


/// Per-class paired totals for heterogeneous sequential workloads.
/// Rejected steps contribute to cost but never to accepted throughput.
/// VmHWM is a process-lifetime peak and cannot be attributed to a class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassCost<K> {
    pub class: K,
    pub attempted: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub full_ns: u128,
    pub incremental_with_refresh_ns: u128,
}

/// Aggregate paired observations without mixing classes or hiding failures.
/// Returns None for mismatched input lengths or arithmetic overflow.
pub fn summarize_stratified_costs<K: Eq + Clone>(
    classes: &[K],
    rows: &[StepMeasurement],
) -> Option<Vec<ClassCost<K>>> {
    if classes.len() != rows.len() {
        return None;
    }
    let mut totals: Vec<ClassCost<K>> = Vec::new();
    for (class, row) in classes.iter().zip(rows) {
        let position = totals.iter().position(|entry| entry.class == *class);
        let index = if let Some(index) = position {
            index
        } else {
            totals.push(ClassCost {
                class: class.clone(),
                attempted: 0,
                accepted: 0,
                rejected: 0,
                full_ns: 0,
                incremental_with_refresh_ns: 0,
            });
            totals.len() - 1
        };
        let entry = &mut totals[index];
        entry.attempted = entry.attempted.checked_add(1)?;
        if row.verdict == StepVerdict::Accepted {
            entry.accepted = entry.accepted.checked_add(1)?;
        } else {
            entry.rejected = entry.rejected.checked_add(1)?;
        }
        entry.full_ns = entry.full_ns.checked_add(row.full.as_nanos())?;
        entry.incremental_with_refresh_ns = entry
            .incremental_with_refresh_ns
            .checked_add(row.incremental_with_refresh.as_nanos())?;
    }
    Some(totals)
}

#[cfg(test)]
mod stratified_cost_tests {
    use super::*;

    #[test]
    fn rejected_steps_are_charged_to_their_class() {
        let classes = [0_u8, 1, 0];
        let rows = [
            StepMeasurement {
                verdict: StepVerdict::Accepted,
                full: Duration::from_nanos(10),
                incremental_with_refresh: Duration::from_nanos(3),
                peak_rss_kib: None,
            },
            StepMeasurement {
                verdict: StepVerdict::Mismatch,
                full: Duration::from_nanos(20),
                incremental_with_refresh: Duration::from_nanos(40),
                peak_rss_kib: None,
            },
            StepMeasurement {
                verdict: StepVerdict::FullError,
                full: Duration::from_nanos(30),
                incremental_with_refresh: Duration::from_nanos(7),
                peak_rss_kib: None,
            },
        ];
        let costs = summarize_stratified_costs(&classes, &rows).unwrap();
        assert_eq!(costs.len(), 2);
        assert_eq!((costs[0].attempted, costs[0].accepted, costs[0].rejected), (2, 1, 1));
        assert_eq!((costs[0].full_ns, costs[0].incremental_with_refresh_ns), (40, 10));
        assert_eq!((costs[1].attempted, costs[1].accepted, costs[1].rejected), (1, 0, 1));
        assert_eq!((costs[1].full_ns, costs[1].incremental_with_refresh_ns), (20, 40));
    }

    #[test]
    fn malformed_pairing_is_rejected() {
        let classes = [1_u8];
        assert!(summarize_stratified_costs(&classes, &[]).is_none());
        assert_eq!(summarize_stratified_costs::<u8>(&[], &[]), Some(vec![]));
    }
}

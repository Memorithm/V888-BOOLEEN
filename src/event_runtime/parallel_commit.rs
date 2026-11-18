//! Disjoint-target parallel commits for bounded event frontiers.
//!
//! Two-phase protocol:
//! (1) stage and validate every event from an immutable snapshot;
//! (2) update *nonoverlapping* incoming-counter slices in scoped threads.
//! Neuron bits, event queue and bookkeeping remain owner-serialized. This
//! does parallelize real state writes but not the entire commit transaction.
//!
//! Only contiguous DRIVE prefixes are admitted. RECHECKs retain the exact
//! reference sequence, whose same-time read-after-write effects are causal.
//! When any preflight fails, replay one reference event at a time to keep
//! the original first-error and partial-commit semantics.

use std::collections::BTreeSet;
use std::thread;

use super::parallel::{prepare_one, PreparedEvent};
use super::{EventError, EventKey, EventRuntime, EventWork, DRIVE, RECHECK};

const MAX_COMMIT_THREADS: usize = 32;
const EVENTS_PER_THREAD: usize = 32;
/// Preliminary admission threshold: measured runtime cost, not joule calibration.
const MIN_PARALLEL_CONTACTS: usize = 65_536;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DisjointCommitWork {
    /// Exactly the same logical work counts as serial EventRuntime::run.
    pub work: EventWork,
    /// Successfully committed groups with at least two drive events.
    pub disjoint_groups: usize,
    /// Number of drive events covered by successful disjoint groups.
    pub grouped_events: usize,
    /// Number of distinct weighted target counters updated by worker threads.
    pub target_counters_written: usize,
    /// Total spawned writers, not hardware concurrency or measured speedup.
    pub target_writer_threads: usize,
    /// Serial fallback events, including RECHECK and overlapping contacts.
    pub serial_events: usize,
    /// Maximum number of events in one disjoint group.
    pub max_group_width: usize,
}

struct StagedCommit {
    event: PreparedEvent,
    updates: Vec<(usize, u128)>,
    next_time: Option<u64>,
    inserted: usize,
    coalesced: usize,
}

impl<'g> EventRuntime<'g> {
    /// Commit disjoint incoming-counter updates in actual scoped Rust threads.
    ///
    /// Exact sequential semantics, including queue-capacity failures, are
    /// retained. The opt-in implementation groups only consecutive, same-tick
    /// drives whose nonzero outgoing target sets are pairwise disjoint.
    /// Overlapping targets and all RECHECK events fall back to serial work.
    /// Mutations of neuron bits and BTreeMap event keys are deliberately
    /// owner-serialized; only disjoint weighted incoming writes run in threads.
    ///
    /// Maximum 32 threads and at most 32 drives per thread per group.
    /// A group can use fewer writers when it touches few distinct targets.
    /// The method does not combine the separate contact-budget admission.
    pub fn run_disjoint_parallel_commits(
        &mut self,
        max_events: usize,
        requested_threads: usize,
    ) -> Result<DisjointCommitWork, EventError> {
        if requested_threads == 0 {
            return Err(EventError::InvalidWorkerCount);
        }
        if requested_threads == 1 {
            return Ok(DisjointCommitWork {
                work: self.run(max_events)?,
                ..DisjointCommitWork::default()
            });
        }
        // Work this small cannot amortize the two scoped thread passes.
        // Direct delegation avoids repeated graph scans and temporary buffers.
        if self.graph.targets.len() < MIN_PARALLEL_CONTACTS {
            let work = self.run(max_events)?;
            return Ok(DisjointCommitWork {
                serial_events: work.events_consumed,
                work,
                ..DisjointCommitWork::default()
            });
        }
        let threads = requested_threads.min(MAX_COMMIT_THREADS);
        let mut result = DisjointCommitWork::default();
        while result.work.events_consumed < max_events && !self.pending.is_empty() {
            let allowance = (max_events - result.work.events_consumed)
                .min(threads.saturating_mul(EVENTS_PER_THREAD));
            let group = self.disjoint_drive_prefix(allowance);
            // RECHECK does not enter a DRIVE prefix. Always consume the
            // earliest event so that a nonempty queue makes progress.
            if group.is_empty() {
                self.consume_one(&mut result.work)?;
                result.serial_events += 1;
                continue;
            }
            let mut contacts = 0usize;
            let mut changing_sources = 0usize;
            for &((_, _, source), requested) in &group {
                if self.state.get(source).expect("validated neuron index") != requested {
                    let fanout =
                        (self.graph.offsets[source + 1] - self.graph.offsets[source]) as usize;
                    if fanout > 0 {
                        changing_sources += 1;
                        contacts = contacts.saturating_add(fanout);
                    }
                }
            }
            // Threaded writes require at least two independent producers,
            // not merely two queued drives (other drives can be no-ops).
            if changing_sources < 2 || contacts < MIN_PARALLEL_CONTACTS {
                // A rejected prefix can be consumed in canonical order.
                // Do not replan the same tiny group after each event.
                for _ in 0..group.len() {
                    self.consume_one(&mut result.work)?;
                    result.serial_events += 1;
                }
                continue;
            }

            // Parallel preflight has no side effects. Any failed preparation
            // must be reported at its original sequential event boundary.
            let worker_count = threads.min(group.len());
            let chunk_size = group.len().div_ceil(worker_count);
            let graph = self.graph;
            let state = &self.state;
            let incoming = &self.incoming;
            let threshold = self.threshold;
            let prepared = thread::scope(|scope| {
                let mut handles = Vec::new();
                for chunk in group.chunks(chunk_size) {
                    handles.push(scope.spawn(move || {
                        chunk
                            .iter()
                            .map(|&(key, requested)| {
                                prepare_one(graph, state, incoming, threshold, key, requested)
                            })
                            .collect::<Vec<_>>()
                    }));
                }
                let mut ordered = Vec::with_capacity(group.len());
                for handle in handles {
                    ordered.extend(handle.join().expect("disjoint preflight worker panicked"));
                }
                ordered
            });
            let mut events = Vec::with_capacity(group.len());
            let mut preparation_failed = false;
            for stage in prepared {
                match stage {
                    Ok(event) => events.push(event),
                    Err(_) => {
                        preparation_failed = true;
                        break;
                    }
                }
            }
            if preparation_failed {
                self.consume_one(&mut result.work)?;
                result.serial_events += 1;
                continue;
            }

            // On a prospective error, use reference single-event commits.
            // It is not permissible to reject a whole group if its earlier
            // events would have been committed by the baseline scheduler.
            let Some(stages) = self.preflight_disjoint_commits(events) else {
                self.consume_one(&mut result.work)?;
                result.serial_events += 1;
                continue;
            };
            let targets_written: usize = stages.iter().map(|stage| stage.updates.len()).sum();
            let writer_threads = self.apply_disjoint_counter_writes(&stages, threads);

            // All staging errors have already been checked; these mutations
            // cannot introduce an arithmetic or queue-capacity error.
            for stage in stages {
                let key = stage.event.key;
                let removed = self.pending.remove(&key);
                debug_assert_eq!(removed, Some(stage.event.requested));
                let source = key.2;
                let changed =
                    self.state.get(source).expect("validated source") != stage.event.desired;
                if changed {
                    self.state
                        .set(source, stage.event.desired)
                        .expect("validated source");
                    if let Some(next_time) = stage.next_time {
                        for (target, _) in stage.updates {
                            let _ = self.pending.insert((next_time, RECHECK, target), false);
                        }
                    }
                    result.work.state_transitions += 1;
                    if stage.event.desired {
                        result.work.rising_transitions += 1;
                    } else {
                        result.work.falling_transitions += 1;
                    }
                    result.work.contacts_examined += stage.event.contacts_examined;
                    result.work.wakeups_inserted += stage.inserted;
                    result.work.wakeups_coalesced += stage.coalesced;
                }
                self.last_time = Some(key.0);
                result.work.events_consumed += 1;
            }
            result.disjoint_groups += 1;
            result.grouped_events += group.len();
            result.max_group_width = result.max_group_width.max(group.len());
            result.target_counters_written += targets_written;
            result.target_writer_threads += writer_threads;
        }
        Ok(result)
    }

    /// Select only a true causal prefix: never reorder a conflicting event.
    /// A source that stays unchanged does not write its outgoing targets.
    fn disjoint_drive_prefix(&self, limit: usize) -> Vec<(EventKey, bool)> {
        let Some((&(time, phase, _), _)) = self.pending.first_key_value() else {
            return Vec::new();
        };
        if phase != DRIVE {
            return Vec::new();
        }
        let mut touched = BTreeSet::new();
        let mut selected = Vec::new();
        for (&key, &requested) in self.pending.iter().take(limit) {
            if key.0 != time || key.1 != DRIVE {
                break;
            }
            let source = key.2;
            if self.state.get(source).expect("validated source") != requested {
                let first = self.graph.offsets[source] as usize;
                let last = self.graph.offsets[source + 1] as usize;
                if (first..last).any(|edge| {
                    self.graph.counts[edge] != 0
                        && touched.contains(&(self.graph.targets[edge] as usize))
                }) {
                    break;
                }
                for edge in first..last {
                    if self.graph.counts[edge] != 0 {
                        touched.insert(self.graph.targets[edge] as usize);
                    }
                }
            }
            selected.push((key, requested));
        }
        selected
    }

    /// Returns None when exact sequential preflight would encounter an error.
    /// Disjoint target ownership ensures snapshots stay valid across events.
    fn preflight_disjoint_commits(&self, events: Vec<PreparedEvent>) -> Option<Vec<StagedCommit>> {
        let mut projected_len = self.pending.len();
        let mut stages = Vec::with_capacity(events.len());
        for event in events {
            let future = if event.deltas.is_empty() {
                None
            } else {
                Some(event.key.0.checked_add(1)?)
            };
            let mut updates = Vec::with_capacity(event.deltas.len());
            let mut inserted = 0usize;
            let mut coalesced = 0usize;
            if let Some(next) = future {
                for &(target, delta) in &event.deltas {
                    let after = if event.desired {
                        self.incoming[target].checked_add(delta)?
                    } else {
                        self.incoming[target].checked_sub(delta)?
                    };
                    if self.pending.contains_key(&(next, RECHECK, target)) {
                        coalesced += 1;
                    } else {
                        inserted += 1;
                    }
                    updates.push((target, after));
                }
            }
            projected_len = projected_len.checked_sub(1)?.checked_add(inserted)?;
            if projected_len > self.max_pending {
                return None;
            }
            stages.push(StagedCommit {
                event,
                updates,
                next_time: future,
                inserted,
                coalesced,
            });
        }
        Some(stages)
    }

    /// Assign prechecked, disjoint target counters into disjoint mutable
    /// slices. Cut positions follow sorted *active target indices* for
    /// balanced writes, rather than splitting a mostly idle dense graph.
    fn apply_disjoint_counter_writes(
        &mut self,
        stages: &[StagedCommit],
        requested_threads: usize,
    ) -> usize {
        let mut writes = Vec::new();
        for event in stages {
            writes.extend_from_slice(&event.updates);
        }
        if writes.is_empty() {
            return 0;
        }
        writes.sort_unstable_by_key(|(target, _)| *target);
        debug_assert!(writes.windows(2).all(|pair| pair[0].0 < pair[1].0));
        let threads = requested_threads.min(writes.len());
        let total = writes.len();
        let num_neurons = self.incoming.len();
        thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            let mut remaining = self.incoming.as_mut_slice();
            let mut base = 0usize;
            for thread_index in 0..threads {
                let first = thread_index * total / threads;
                let last = (thread_index + 1) * total / threads;
                let end = if thread_index + 1 == threads {
                    num_neurons
                } else {
                    writes[last].0
                };
                let (segment, rest) = remaining.split_at_mut(end - base);
                remaining = rest;
                let batch = &writes[first..last];
                handles.push(scope.spawn(move || {
                    for &(target, after) in batch {
                        segment[target - base] = after;
                    }
                }));
                base = end;
            }
            for handle in handles {
                handle.join().expect("disjoint counter writer panicked");
            }
        });
        threads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome::Connectome;

    fn compare(
        graph: &Connectome,
        threshold: u128,
        drives: &[(u64, u64, bool)],
        max_events: usize,
        max_pending: usize,
        workers: usize,
    ) -> DisjointCommitWork {
        let mut reference = EventRuntime::new(graph, threshold, max_pending).unwrap();
        let mut tested = EventRuntime::new(graph, threshold, max_pending).unwrap();
        for &(time, source, value) in drives {
            reference.schedule_drive(time, source, value).unwrap();
            tested.schedule_drive(time, source, value).unwrap();
        }
        let expected = reference.run(max_events).unwrap();
        let actual = tested
            .run_disjoint_parallel_commits(max_events, workers)
            .unwrap();
        assert_eq!(actual.work, expected);
        assert_eq!(tested.state(), reference.state());
        assert_eq!(tested.incoming, reference.incoming);
        assert_eq!(tested.pending, reference.pending);
        assert_eq!(
            tested.last_processed_time(),
            reference.last_processed_time()
        );
        assert_eq!(tested.audit_incoming(), Ok(true));
        actual
    }

    fn eight_disjoint_sources() -> Connectome {
        // 8192 contact rows for each of eight distinct sources.
        // The 65536 contacts exercise the admitted parallel path.
        Connectome {
            ids: (10..26).collect(),
            offsets: (0..=8)
                .map(|i| i * 8192)
                .chain(std::iter::repeat_n(65_536, 8))
                .collect(),
            targets: (0..8)
                .flat_map(|i| std::iter::repeat_n(8 + i, 8192))
                .collect(),
            counts: vec![1; 65_536],
        }
    }

    #[test]
    fn distinct_targets_are_written_by_multiple_threads() {
        let graph = eight_disjoint_sources();
        let drives = (10..18).map(|id| (0, id, true)).collect::<Vec<_>>();
        let result = compare(&graph, 1, &drives, 20, 64, 4);
        assert_eq!(result.disjoint_groups, 1);
        assert_eq!(result.grouped_events, 8);
        assert_eq!(result.target_counters_written, 8);
        assert_eq!(result.target_writer_threads, 4);
        assert_eq!(result.max_group_width, 8);
        assert_eq!(result.work.state_transitions, 16);
    }

    #[test]
    fn a_conflicting_target_splits_prefix_without_reordering() {
        let graph = Connectome {
            ids: vec![10, 20, 30, 40, 50],
            offsets: vec![0, 1, 2, 3, 3, 3],
            targets: vec![3, 3, 4],
            counts: vec![1; 3],
        };
        let report = compare(
            &graph,
            2,
            &[(0, 10, true), (0, 20, true), (0, 30, true)],
            12,
            12,
            3,
        );
        assert_eq!(report.work.events_consumed, 5);
        assert_eq!(report.work.wakeups_inserted, 2);
        assert_eq!(report.work.wakeups_coalesced, 1);
        // Three-contact graphs now bypass thread admission entirely.
        assert_eq!(report.disjoint_groups, 0);
        assert_eq!(report.grouped_events, 0);
        assert_eq!(report.serial_events, report.work.events_consumed);
    }

    #[test]
    fn independent_negative_fronts_are_written_in_parallel() {
        let graph = eight_disjoint_sources();
        let mut drives = (10..18).map(|id| (0, id, true)).collect::<Vec<_>>();
        drives.extend((10..18).map(|id| (3, id, false)));
        let stats = compare(&graph, 1, &drives, 100, 64, 4);
        assert_eq!(stats.disjoint_groups, 2);
        assert_eq!(stats.work.rising_transitions, 16);
        assert_eq!(stats.work.falling_transitions, 16);
        assert_eq!(stats.target_counters_written, 16);
    }

    #[test]
    fn isolated_rechecks_always_advance_after_large_graph_serial_fallback() {
        let graph = eight_disjoint_sources();
        let mut runtime = EventRuntime::new(&graph, 1, 32).unwrap();
        runtime.schedule_drive(0, 10, true).unwrap();
        let first = runtime.run_disjoint_parallel_commits(2, 4).unwrap();
        assert_eq!(first.work.events_consumed, 2);
        assert_eq!(first.work.state_transitions, 2);
        assert_eq!(first.serial_events, 2);
        assert_eq!(runtime.pending_count(), 0);
        assert_eq!(runtime.audit_incoming(), Ok(true));
    }

    #[test]
    fn idle_drive_does_not_count_as_parallel_writer() {
        // One large source changes, but the other DRIVE is a no-op.
        // Group length alone must never allocate two writer threads.
        let graph = Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 65_536, 65_536, 65_536],
            targets: vec![2; 65_536],
            counts: vec![1; 65_536],
        };
        let result = compare(
            &graph,
            1,
            &[(0, 10, true), (0, 20, false)],
            8,
            8,
            4,
        );
        assert_eq!(result.disjoint_groups, 0);
        assert_eq!(result.target_writer_threads, 0);
        assert_eq!(result.serial_events, result.work.events_consumed);
        assert_eq!(result.work.state_transitions, 2);
    }

    #[test]
    fn worker_count_one_delegates_to_exact_serial_engine() {
        let graph = eight_disjoint_sources();
        let drives = [(0, 10, true), (0, 11, true)];
        let stats = compare(&graph, 1, &drives, 12, 16, 1);
        assert_eq!(stats.target_writer_threads, 0);
        assert_eq!(stats.disjoint_groups, 0);
    }

    #[test]
    fn zero_event_budget_and_invalid_workers_preserve_state() {
        let graph = eight_disjoint_sources();
        let mut runtime = EventRuntime::new(&graph, 1, 16).unwrap();
        runtime.schedule_drive(0, 10, true).unwrap();
        assert!(matches!(
            runtime.run_disjoint_parallel_commits(10, 0),
            Err(EventError::InvalidWorkerCount)
        ));
        assert_eq!(
            runtime.run_disjoint_parallel_commits(0, 4).unwrap().work,
            EventWork::default()
        );
        assert_eq!(runtime.pending_count(), 1);
        assert_eq!(runtime.state().active_count(), 0);
    }

    #[test]
    fn capacity_error_preserves_partial_commits_and_all_counters() {
        let graph = Connectome {
            ids: vec![10, 20, 30, 40, 50, 60],
            offsets: vec![0, 1, 4, 4, 4, 4, 4],
            targets: vec![2, 3, 4, 5],
            counts: vec![1; 4],
        };
        let mut reference = EventRuntime::new(&graph, 1, 3).unwrap();
        let mut tested = EventRuntime::new(&graph, 1, 3).unwrap();
        for runtime in [&mut reference, &mut tested] {
            runtime.schedule_drive(0, 10, true).unwrap();
            runtime.schedule_drive(0, 20, true).unwrap();
        }
        assert_eq!(reference.run(5), Err(EventError::PendingBudgetExceeded));
        assert_eq!(
            tested.run_disjoint_parallel_commits(5, 4),
            Err(EventError::PendingBudgetExceeded)
        );
        assert_eq!(tested.state(), reference.state());
        assert_eq!(tested.incoming, reference.incoming);
        assert_eq!(tested.pending, reference.pending);
        assert_eq!(
            tested.last_processed_time(),
            reference.last_processed_time()
        );
        assert_eq!(tested.state().active_count(), 1);
    }

    #[test]
    fn logical_time_overflow_matches_reference_without_mutation() {
        let graph = eight_disjoint_sources();
        let mut reference = EventRuntime::new(&graph, 1, 16).unwrap();
        let mut tested = EventRuntime::new(&graph, 1, 16).unwrap();
        for runtime in [&mut reference, &mut tested] {
            runtime.schedule_drive(u64::MAX, 10, true).unwrap();
            runtime.schedule_drive(u64::MAX, 11, true).unwrap();
        }
        assert_eq!(reference.run(5), Err(EventError::LogicalTimeOverflow));
        assert_eq!(
            tested.run_disjoint_parallel_commits(5, 4),
            Err(EventError::LogicalTimeOverflow)
        );
        assert_eq!(tested.state(), reference.state());
        assert_eq!(tested.incoming, reference.incoming);
        assert_eq!(tested.pending, reference.pending);
    }

    #[test]
    fn partial_event_slices_are_equivalent_for_every_prefix() {
        let graph = eight_disjoint_sources();
        let drives = (10..18).map(|id| (0, id, true)).collect::<Vec<_>>();
        for limit in 0..=24 {
            let _ = compare(&graph, 1, &drives, limit, 64, 8);
        }
    }

    #[test]
    fn overlapping_feedback_rechecks_fall_back_to_serial() {
        let graph = Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 1, 2, 3],
            targets: vec![1, 2, 0],
            counts: vec![1; 3],
        };
        let report = compare(&graph, 1, &[(0, 10, true), (0, 20, true)], 16, 16, 4);
        // Two drives, two t+1 rechecks, then a t+2 no-op recheck.
        assert_eq!(report.work.events_consumed, 5);
        assert_eq!(report.work.state_transitions, 3);
    }
}

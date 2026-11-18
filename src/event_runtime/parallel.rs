//! Opt-in bounded multi-threaded preflight for causal event frontiers.
//!
//! Preparation reads only immutable graph/state snapshots. Commit remains
//! serial and strictly ordered to preserve the event-runtime reference,
//! including wakeup deduplication, per-event failure atomicity and counters.
//! This is NOT a claim of parallel commits or physical energy savings.

use std::collections::BTreeSet;

use super::contacts::{effective_outgoing_work, outgoing_with_cache, ContactCache};
use std::thread;

use super::{EventError, EventKey, EventRuntime, EventWork, DRIVE, RECHECK};
use crate::connectome::Connectome;
use crate::BooleanState;

const MAX_WORKER_THREADS: usize = 32;
const EVENTS_PER_WORKER: usize = 32;
/// Conservative research admission threshold; benchmark, not energy calibrated.
const MIN_PARALLEL_CONTACTS: usize = 65_536;
const MIN_PARALLEL_EVENTS: usize = 4;

/// Logical work is comparable with `EventRuntime::run`. Additional fields
/// count preflight staging, not measured parallel speedup or energy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParallelFrontierWork {
    pub work: EventWork,
    pub parallel_groups: usize,
    pub prepared_events: usize,
    pub threads_spawned: usize,
    pub max_group_width: usize,
}

pub(super) struct PreparedEvent {
    pub(super) key: EventKey,
    pub(super) requested: bool,
    pub(super) desired: bool,
    pub(super) contacts_examined: usize,
    pub(super) deltas: Vec<(usize, u128)>,
}

pub(super) fn prepare_one(
    graph: &Connectome,
    cache: Option<&ContactCache>,
    state: &BooleanState,
    incoming: &[u128],
    threshold: u128,
    key: EventKey,
    requested: bool,
) -> Result<PreparedEvent, EventError> {
    let (_, phase, source) = key;
    let current = state.get(source).expect("validated source index");
    let desired = if phase == DRIVE {
        requested
    } else {
        incoming[source] >= threshold
    };
    let (contacts_examined, deltas) = if desired == current {
        (0, Vec::new())
    } else {
        let first = graph.offsets[source] as usize;
        let last = graph.offsets[source + 1] as usize;
        (last - first, outgoing_with_cache(graph, cache, source)?)
    };
    Ok(PreparedEvent {
        key,
        requested,
        desired,
        contacts_examined,
        deltas,
    })
}

impl<'g> EventRuntime<'g> {
    /// Run up to `max_events`, preparing a stable causal prefix in worker
    /// threads and committing events in their exact original order.
    ///
    /// A group never crosses a logical time or DRIVE/RECHECK phase boundary.
    /// Consecutive RECHECKs share a group only when no earlier transitioning
    /// source in the group can write to the later source's incoming counter.
    /// Shared *other* targets are safe: they are updated in serial order.
    ///
    /// This conservative dependency guard is essential: the original engine
    /// immediately updates weighted inputs on a state transition, while
    /// scheduling the target's next recheck at t+1. Therefore blindly taking
    /// all t-local RECHECKs from one snapshot can change the reference result.
    ///
    /// Workers are capped at 32; staging at 32 events per worker. The caller
    /// may pass 1 to select the exact sequential execution path. Contact
    /// budgets are handled by the separate sequential `run_with_contact_budget`.
    pub fn run_parallel_frontiers(
        &mut self,
        max_events: usize,
        requested_workers: usize,
    ) -> Result<ParallelFrontierWork, EventError> {
        if requested_workers == 0 {
            return Err(EventError::InvalidWorkerCount);
        }
        if requested_workers == 1 {
            return Ok(ParallelFrontierWork {
                work: self.run(max_events)?,
                ..ParallelFrontierWork::default()
            });
        }
        // A small entire graph cannot amortize scoped-thread creation.
        // Skip dependency scans and allocations on this common path.
        if self.graph.targets.len() < MIN_PARALLEL_CONTACTS {
            return Ok(ParallelFrontierWork {
                work: self.run(max_events)?,
                ..ParallelFrontierWork::default()
            });
        }
        let threads = requested_workers.min(MAX_WORKER_THREADS);
        let mut report = ParallelFrontierWork::default();
        while report.work.events_consumed < max_events && !self.pending.is_empty() {
            let allowance = (max_events - report.work.events_consumed)
                .min(threads.saturating_mul(EVENTS_PER_WORKER));
            let group = self.independent_prefix(allowance);
            // Admit threads only for real transitions with sufficient actual
            // preparation work (cached pairs or uncached CSR rows). A large
            // logical fanout need not justify threads after compression.
            // Consume a rejected causal prefix in order, rather
            // than repeatedly planning it or spawning empty worker groups.
            let contacts: usize = group
                .iter()
                .map(|&((_, phase, source), requested)| {
                    let current = self.state.get(source).expect("validated neuron index");
                    let desired = if phase == DRIVE {
                        requested
                    } else {
                        self.incoming[source] >= self.threshold
                    };
                    if current == desired {
                        0
                    } else {
                        effective_outgoing_work(self.graph, self.contact_cache.as_ref(), source)
                    }
                })
                .sum();
            if group.len() < MIN_PARALLEL_EVENTS || contacts < MIN_PARALLEL_CONTACTS {
                for _ in 0..group.len() {
                    self.consume_one(&mut report.work)?;
                }
                continue;
            }
            let worker_count = threads.min(group.len());
            let chunk_size = group.len().div_ceil(worker_count);
            let graph = self.graph;
            let cache = self.contact_cache.as_ref();
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
                                prepare_one(
                                    graph, cache, state, incoming, threshold, key, requested,
                                )
                            })
                            .collect::<Vec<_>>()
                    }));
                }
                let mut ordered = Vec::with_capacity(group.len());
                for handle in handles {
                    ordered.extend(handle.join().expect("frontier preparation worker panicked"));
                }
                ordered
            });
            report.parallel_groups += 1;
            report.threads_spawned += group.len().div_ceil(chunk_size);
            report.max_group_width = report.max_group_width.max(group.len());
            for staged in prepared {
                self.commit_prepared(staged?, &mut report.work)?;
                report.prepared_events += 1;
            }
        }
        Ok(report)
    }

    /// Select only a prefix. Never reorder an event to obtain a wider group.
    fn independent_prefix(&self, limit: usize) -> Vec<(EventKey, bool)> {
        let Some((&(time, phase, _), _)) = self.pending.first_key_value() else {
            return Vec::new();
        };
        let mut touched_by_earlier_rechecks = BTreeSet::new();
        let mut selected = Vec::new();
        for (&key, &requested) in self.pending.iter().take(limit) {
            let (_, candidate_phase, source) = key;
            if key.0 != time || candidate_phase != phase {
                break;
            }
            if phase == RECHECK && touched_by_earlier_rechecks.contains(&source) {
                break;
            }
            if phase == RECHECK {
                let current = self.state.get(source).expect("validated source index");
                let desired = self.incoming[source] >= self.threshold;
                if desired != current {
                    if let Some(reduced) = self.contact_cache.as_ref().and_then(|c| c.get(&source)) {
                        // Cached pairs represent exactly the nonzero CSR
                        // destinations. Avoid rescanning large repeated runs.
                        for &(target, _) in reduced {
                            touched_by_earlier_rechecks.insert(target);
                        }
                    } else {
                        let first = self.graph.offsets[source] as usize;
                        let last = self.graph.offsets[source + 1] as usize;
                        for edge in first..last {
                            if self.graph.counts[edge] > 0 {
                                touched_by_earlier_rechecks
                                    .insert(self.graph.targets[edge] as usize);
                            }
                        }
                    }
                }
            }
            selected.push((key, requested));
        }
        selected
    }

    /// Replay exact serial effects using precomputed *edge counts*, never
    /// snapshot target sums: earlier commits may modify overlapping targets.
    fn commit_prepared(
        &mut self,
        prepared: PreparedEvent,
        work: &mut EventWork,
    ) -> Result<(), EventError> {
        let (&key, &requested) = self.pending.first_key_value().expect("group is nonempty");
        debug_assert_eq!(key, prepared.key);
        debug_assert_eq!(requested, prepared.requested);
        let (time, phase, source) = key;
        let current = self.state.get(source).expect("validated source index");
        let desired = if phase == DRIVE {
            requested
        } else {
            self.incoming[source] >= self.threshold
        };
        // Conservative fallback if a future policy relaxes group selection.
        if desired != prepared.desired {
            return self.consume_one(work);
        }
        let mut staged = Vec::with_capacity(prepared.deltas.len());
        let mut inserted = 0usize;
        let mut coalesced = 0usize;
        let future = if prepared.deltas.is_empty() {
            None
        } else {
            Some(time.checked_add(1).ok_or(EventError::LogicalTimeOverflow)?)
        };
        if let Some(next) = future {
            for (target, delta) in &prepared.deltas {
                let prior = self.incoming[*target];
                let after = if desired {
                    prior.checked_add(*delta)
                } else {
                    prior.checked_sub(*delta)
                }
                .ok_or(EventError::ArithmeticOverflow)?;
                if self.pending.contains_key(&(next, RECHECK, *target)) {
                    coalesced += 1;
                } else {
                    inserted += 1;
                }
                staged.push((*target, after));
            }
            if self.pending.len() - 1 + inserted > self.max_pending {
                return Err(EventError::PendingBudgetExceeded);
            }
        }
        let _ = self.pending.pop_first();
        if current != desired {
            self.state
                .set(source, desired)
                .expect("validated source index");
            if let Some(next) = future {
                for (target, after) in staged {
                    self.incoming[target] = after;
                    let _ = self.pending.insert((next, RECHECK, target), false);
                }
            }
            work.state_transitions += 1;
            if desired {
                work.rising_transitions += 1;
            } else {
                work.falling_transitions += 1;
            }
            work.contacts_examined += prepared.contacts_examined;
            work.wakeups_inserted += inserted;
            work.wakeups_coalesced += coalesced;
        }
        if phase == RECHECK {
            work.threshold_checks += 1;
        }
        self.last_time = Some(time);
        work.events_consumed += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> Connectome {
        // 0 -> 2,3; 1 -> 2,4; 2 -> 0,3; 3 -> 1. Includes
        // convergence and cyclic feedback among concurrently active neurons.
        Connectome {
            ids: vec![10, 20, 30, 40, 50],
            offsets: vec![0, 2, 4, 6, 7, 7],
            targets: vec![2, 3, 2, 4, 0, 3, 1],
            counts: vec![1; 7],
        }
    }

    fn compare(
        graph: &Connectome,
        threshold: u128,
        drives: &[(u64, u64, bool)],
        event_limit: usize,
        pending_limit: usize,
        worker_count: usize,
    ) -> ParallelFrontierWork {
        let mut sequential = EventRuntime::new(graph, threshold, pending_limit).unwrap();
        let mut concurrent = EventRuntime::new(graph, threshold, pending_limit).unwrap();
        for &(time, id, value) in drives {
            sequential.schedule_drive(time, id, value).unwrap();
            concurrent.schedule_drive(time, id, value).unwrap();
        }
        let expected = sequential.run(event_limit).unwrap();
        let actual = concurrent
            .run_parallel_frontiers(event_limit, worker_count)
            .unwrap();
        assert_eq!(actual.work, expected);
        assert_eq!(concurrent.state(), sequential.state());
        assert_eq!(concurrent.incoming, sequential.incoming);
        assert_eq!(concurrent.pending, sequential.pending);
        assert_eq!(
            concurrent.last_processed_time(),
            sequential.last_processed_time()
        );
        assert_eq!(concurrent.audit_incoming(), Ok(true));
        actual
    }

    #[test]
    fn cached_recheck_dependencies_keep_exact_causal_prefix() {
        // At t+1, neuron 30 will write to the already queued 40.
        // Its cached 65,536-contact fanout must still block speculative
        // staging of 40 from a stale same-timestamp incoming snapshot.
        let graph = Connectome {
            ids: vec![10, 20, 30, 40],
            offsets: vec![0, 1, 2, 65_538, 65_538],
            targets: [vec![2, 3], vec![3; 65_536]].concat(),
            counts: vec![1; 65_538],
        };
        graph.validate().unwrap();
        let mut reference = EventRuntime::new(&graph, 1, 8).unwrap();
        let mut cached = EventRuntime::new(&graph, 1, 8).unwrap();
        assert_eq!(cached.prepare_contact_cache(1).unwrap().cached_sources, 1);
        for engine in [&mut reference, &mut cached] {
            engine.schedule_drive(0, 10, true).unwrap();
            engine.schedule_drive(0, 20, true).unwrap();
            assert_eq!(engine.run(2).unwrap().events_consumed, 2);
        }
        assert_eq!(reference.independent_prefix(8).len(), 1);
        assert_eq!(cached.independent_prefix(8).len(), 1);
        let expected = reference.run(8).unwrap();
        let actual = cached.run_parallel_frontiers(8, 4).unwrap();
        assert_eq!(actual.work, expected);
        assert_eq!(cached.incoming, reference.incoming);
        assert_eq!(cached.pending, reference.pending);
        assert_eq!(cached.state(), reference.state());
        assert_eq!(cached.audit_incoming(), Ok(true));
    }

    #[test]
    fn simultaneous_independent_drives_are_prepared_by_multiple_threads() {
        // Four independent sources, each carrying 16_384 contact rows.
        // The group is large enough to exercise the actual thread path.
        let graph = Connectome {
            ids: (10..18).collect(),
            offsets: (0..=4)
                .map(|i| i * 16_384)
                .chain(std::iter::repeat_n(65_536, 4))
                .collect(),
            targets: (0..4)
                .flat_map(|i| std::iter::repeat_n(4 + i, 16_384))
                .collect(),
            counts: vec![1; 65_536],
        };
        let stats = compare(
            &graph,
            1,
            &[(0, 13, true), (0, 11, true), (0, 10, true), (0, 12, true)],
            16,
            16,
            4,
        );
        assert_eq!(stats.parallel_groups, 1);
        assert_eq!(stats.prepared_events, 4);
        assert_eq!(stats.threads_spawned, 4);
        assert_eq!(stats.max_group_width, 4);
        assert_eq!(stats.work.state_transitions, 8);
    }

    #[test]
    fn convergent_drives_preserve_exact_wakeup_coalescing() {
        let graph = Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 1, 2, 2],
            targets: vec![2, 2],
            counts: vec![1, 1],
        };
        let stats = compare(&graph, 2, &[(0, 20, true), (0, 10, true)], 10, 10, 3);
        assert_eq!(stats.work.wakeups_inserted, 1);
        assert_eq!(stats.work.wakeups_coalesced, 1);
        assert_eq!(stats.parallel_groups, 0);
    }

    #[test]
    fn independent_rechecks_remain_exact_under_small_graph_fallback() {
        let graph = Connectome {
            ids: vec![10, 20, 30, 40],
            offsets: vec![0, 1, 1, 2, 2],
            targets: vec![1, 3],
            counts: vec![1, 1],
        };
        let stats = compare(&graph, 1, &[(0, 10, true), (0, 30, true)], 16, 16, 4);
        assert_eq!(stats.work.state_transitions, 4);
        assert_eq!(stats.parallel_groups, 0);
        assert_eq!(stats.prepared_events, 0);
        assert_eq!(stats.work.threshold_checks, 2);
    }

    #[test]
    fn a_later_failing_event_preserves_earlier_successful_event() {
        let graph = Connectome {
            ids: vec![10, 20, 30, 40, 50],
            offsets: vec![0, 0, 3, 3, 3, 3],
            targets: vec![2, 3, 4],
            counts: vec![1, 1, 1],
        };
        let mut serial = EventRuntime::new(&graph, 1, 2).unwrap();
        let mut parallel = EventRuntime::new(&graph, 1, 2).unwrap();
        for runtime in [&mut serial, &mut parallel] {
            runtime.schedule_drive(0, 10, true).unwrap();
            runtime.schedule_drive(0, 20, true).unwrap();
        }
        assert_eq!(serial.run(2), Err(EventError::PendingBudgetExceeded));
        assert!(matches!(
            parallel.run_parallel_frontiers(2, 2),
            Err(EventError::PendingBudgetExceeded)
        ));
        assert_eq!(serial.state().active_count(), 1);
        assert_eq!(serial.pending, parallel.pending);
        assert_eq!(serial.incoming, parallel.incoming);
        assert_eq!(serial.state(), parallel.state());
        assert_eq!(serial.last_processed_time(), Some(0));
        assert_eq!(parallel.last_processed_time(), Some(0));
    }

    #[test]
    fn rechecks_never_read_stale_same_time_incoming_contributions() {
        // t=1 rechecks for nodes 1 and 2. Processing node 1 updates
        // incoming[2] immediately, so node 2 must observe the update at t=1.
        let graph = Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 2, 3, 3],
            targets: vec![1, 2, 2],
            counts: vec![2, 1, 1],
        };
        let stats = compare(&graph, 2, &[(0, 10, true)], 16, 16, 4);
        assert_eq!(stats.work.state_transitions, 3);
        assert_eq!(stats.work.contacts_examined, 3);
    }

    #[test]
    fn feedback_and_partial_slices_reproduce_every_reference_counter() {
        for limit in 0..=40 {
            let _ = compare(
                &graph(),
                1,
                &[(0, 10, true), (0, 20, true), (6, 10, false), (7, 20, false)],
                limit,
                64,
                4,
            );
        }
    }

    #[test]
    fn singleton_worker_matches_reference_without_spawning_threads() {
        let stats = compare(&graph(), 1, &[(0, 10, true), (0, 20, true)], 20, 32, 1);
        assert_eq!(stats.threads_spawned, 0);
        assert_eq!(stats.parallel_groups, 0);
    }

    #[test]
    fn invalid_worker_count_and_empty_budget_are_noops() {
        let graph = graph();
        let mut runtime = EventRuntime::new(&graph, 1, 8).unwrap();
        runtime.schedule_drive(0, 10, true).unwrap();
        assert!(matches!(
            runtime.run_parallel_frontiers(1, 0),
            Err(EventError::InvalidWorkerCount)
        ));
        let zero = runtime.run_parallel_frontiers(0, 4).unwrap();
        assert_eq!(zero.work, EventWork::default());
        assert_eq!(runtime.pending_count(), 1);
    }

    #[test]
    fn failing_event_does_not_mutate_queue_or_state() {
        let graph = Connectome {
            ids: vec![10, 20, 30, 40],
            offsets: vec![0, 3, 3, 3, 3],
            targets: vec![1, 2, 3],
            counts: vec![1, 1, 1],
        };
        let mut serial = EventRuntime::new(&graph, 1, 2).unwrap();
        let mut parallel = EventRuntime::new(&graph, 1, 2).unwrap();
        for runtime in [&mut serial, &mut parallel] {
            runtime.schedule_drive(0, 10, true).unwrap();
            runtime.schedule_drive(0, 20, true).unwrap();
        }
        assert_eq!(serial.run(2), Err(EventError::PendingBudgetExceeded));
        assert!(matches!(
            parallel.run_parallel_frontiers(2, 4),
            Err(EventError::PendingBudgetExceeded)
        ));
        assert_eq!(serial.pending, parallel.pending);
        assert_eq!(serial.incoming, parallel.incoming);
        assert_eq!(serial.state(), parallel.state());
    }

    #[test]
    fn logical_time_overflow_is_causally_identical() {
        let graph = Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 1, 2, 2],
            targets: vec![2, 2],
            counts: vec![1, 1],
        };
        let mut serial = EventRuntime::new(&graph, 1, 8).unwrap();
        let mut parallel = EventRuntime::new(&graph, 1, 8).unwrap();
        for runtime in [&mut serial, &mut parallel] {
            runtime.schedule_drive(u64::MAX, 10, true).unwrap();
            runtime.schedule_drive(u64::MAX, 20, true).unwrap();
        }
        assert_eq!(serial.run(2), Err(EventError::LogicalTimeOverflow));
        assert!(matches!(
            parallel.run_parallel_frontiers(2, 4),
            Err(EventError::LogicalTimeOverflow)
        ));
        assert_eq!(serial.pending, parallel.pending);
        assert_eq!(serial.state(), parallel.state());
    }
}

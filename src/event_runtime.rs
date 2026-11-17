//! Experimental asynchronous, event-driven Boolean propagation over a fixed CSR graph.
//!
//! Unlike `BooleanEngine::step`, this runtime never scans all neurons or
//! constructs a dense input vector per tick. Only actual state transitions
//! visit outgoing contacts. A changed target is checked at the *next* logical
//! time; duplicate wakeups for the same (time, target) coalesce.
//!
//! This is a deterministic computational hypothesis, NOT a biological model.
//! Logical work counters are not measurements of energy or physical time.

use std::collections::BTreeMap;

use crate::connectome::Connectome;
use crate::BooleanState;

mod parallel;
mod parallel_commit;
pub use parallel::ParallelFrontierWork;
pub use parallel_commit::DisjointCommitWork;

const DRIVE: u8 = 0;
const RECHECK: u8 = 1;
type EventKey = (u64, u8, usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventError {
    InvalidGraph,
    InvalidThreshold,
    InvalidPendingBudget,
    InvalidWorkerCount,
    UnknownNeuron,
    TimeNotFuture,
    ConflictingDrive,
    PendingBudgetExceeded,
    ArithmeticOverflow,
    LogicalTimeOverflow,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EventWork {
    /// Events actually consumed, including no-ops.
    pub events_consumed: usize,
    /// Boolean neurons whose state changed.
    pub state_transitions: usize,
    /// Positive activation fronts: false -> true.
    pub rising_transitions: usize,
    /// Negative activation fronts: true -> false.
    pub falling_transitions: usize,
    /// CSR contacts examined only on a real transition.
    pub contacts_examined: usize,
    /// Threshold comparisons during target rechecks.
    pub threshold_checks: usize,
    /// New coalesced target wakeups inserted into the queue.
    pub wakeups_inserted: usize,
    /// Target wakeups already queued at the same logical time.
    pub wakeups_coalesced: usize,
}

/// Results of one explicitly resource-bounded execution slice.
/// A blocked event is not consumed or mutated; the caller may resume with a
/// larger contact budget. The budget counts examined CSR contact rows, not
/// energy or all CPU/memory operations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BudgetedWork {
    pub work: EventWork,
    pub blocked_on_contact_budget: bool,
}

/// One owner of the mutable state; callers may inject timestamped drives.
/// Queue order is (logical time, drive-before-recheck, neuron index).
/// There is no wall-clock wait, OS thread pool, or network message transport.
pub struct EventRuntime<'g> {
    graph: &'g Connectome,
    state: BooleanState,
    incoming: Vec<u128>,
    threshold: u128,
    pending: BTreeMap<EventKey, bool>,
    max_pending: usize,
    last_time: Option<u64>,
}

impl<'g> EventRuntime<'g> {
    pub fn new(
        graph: &'g Connectome,
        threshold: u128,
        max_pending: usize,
    ) -> Result<Self, EventError> {
        graph.validate().map_err(|_| EventError::InvalidGraph)?;
        if threshold == 0 {
            return Err(EventError::InvalidThreshold);
        }
        if max_pending == 0 {
            return Err(EventError::InvalidPendingBudget);
        }
        Ok(Self {
            graph,
            state: BooleanState::new(graph.ids.len()),
            incoming: vec![0; graph.ids.len()],
            threshold,
            pending: BTreeMap::new(),
            max_pending,
            last_time: None,
        })
    }

    #[must_use]
    pub fn state(&self) -> &BooleanState {
        &self.state
    }

    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    #[must_use]
    pub fn next_event_time(&self) -> Option<u64> {
        self.pending.first_key_value().map(|(key, _)| key.0)
    }

    #[must_use]
    pub fn last_processed_time(&self) -> Option<u64> {
        self.last_time
    }

    /// Weighted sum currently contributed by active presynaptic sources.
    /// This is a logical counter, not a voltage, current or energy estimate.
    pub fn incoming_for(&self, neuron_id: u64) -> Result<u128, EventError> {
        let index = self
            .graph
            .index(neuron_id)
            .ok_or(EventError::UnknownNeuron)?;
        Ok(self.incoming[index])
    }

    /// Explicit full reference audit, never used on the hot event path.
    /// Recomputes weighted incoming sums independently from active source
    /// bits; the dense oracle buffer is deliberately confined to this audit.
    pub fn audit_incoming(&self) -> Result<bool, EventError> {
        let mut expected = vec![0_u128; self.graph.ids.len()];
        for source in self.state.active_indices() {
            let first = self.graph.offsets[source] as usize;
            let last = self.graph.offsets[source + 1] as usize;
            for edge in first..last {
                let target = self.graph.targets[edge] as usize;
                expected[target] = expected[target]
                    .checked_add(u128::from(self.graph.counts[edge]))
                    .ok_or(EventError::ArithmeticOverflow)?;
            }
        }
        Ok(expected == self.incoming)
    }

    /// Schedule a caller-controlled Boolean assignment, identified by the
    /// original graph neuron ID. Repeated identical drives are idempotent;
    /// conflicting drives at one time are rejected without mutation.
    /// A return value of false means that this drive was already pending.
    pub fn schedule_drive(
        &mut self,
        logical_time: u64,
        neuron_id: u64,
        active: bool,
    ) -> Result<bool, EventError> {
        if self.last_time.is_some_and(|last| logical_time <= last) {
            return Err(EventError::TimeNotFuture);
        }
        let index = self
            .graph
            .index(neuron_id)
            .ok_or(EventError::UnknownNeuron)?;
        let key = (logical_time, DRIVE, index);
        if let Some(&old) = self.pending.get(&key) {
            return if old == active {
                Ok(false)
            } else {
                Err(EventError::ConflictingDrive)
            };
        }
        if self.pending.len() >= self.max_pending {
            return Err(EventError::PendingBudgetExceeded);
        }
        let _ = self.pending.insert(key, active);
        Ok(true)
    }

    /// Consume no more than `max_events` in stable causal order. A bounded
    /// call can stop with work pending; the caller decides whether to resume.
    /// On an error, the failing event and its effects remain uncommitted.
    /// Earlier successfully consumed events in the same call stay committed.
    pub fn run(&mut self, max_events: usize) -> Result<EventWork, EventError> {
        Ok(self.run_with_contact_budget(max_events, usize::MAX)?.work)
    }

    /// Opt-in logical work admission. The first event that would exceed
    /// `max_contacts` remains queued and is never partially committed.
    /// Events are not reordered or dropped, preserving causal determinism.
    /// Rechecks with no change may run with zero contact budget.
    ///
    /// A recurring event whose fanout exceeds the supplied per-slice
    /// allowance cannot make progress until the caller increases the budget.
    /// The budget is a count of CSR rows, NOT a calibrated energy constraint.
    pub fn run_with_contact_budget(
        &mut self,
        max_events: usize,
        max_contacts: usize,
    ) -> Result<BudgetedWork, EventError> {
        let mut work = EventWork::default();
        let mut blocked_on_contact_budget = false;
        for _ in 0..max_events {
            let Some((&(_, phase, source), &requested)) = self.pending.first_key_value() else {
                break;
            };
            let current = self.state.get(source).expect("validated neuron index");
            let desired = if phase == DRIVE {
                requested
            } else {
                self.incoming[source] >= self.threshold
            };
            let required_contacts = if current == desired {
                0
            } else {
                (self.graph.offsets[source + 1] - self.graph.offsets[source]) as usize
            };
            if required_contacts > max_contacts.saturating_sub(work.contacts_examined) {
                blocked_on_contact_budget = true;
                break;
            }
            self.consume_one(&mut work)?;
        }
        Ok(BudgetedWork {
            work,
            blocked_on_contact_budget,
        })
    }

    fn consume_one(&mut self, work: &mut EventWork) -> Result<(), EventError> {
        let (&(time, phase, source), &requested) =
            self.pending.first_key_value().expect("queue is nonempty");
        let current = self.state.get(source).expect("validated neuron index");
        let desired = if phase == DRIVE {
            requested
        } else {
            self.incoming[source] >= self.threshold
        };

        // Stage every outgoing counter change and every future wakeup before
        // modifying state or queue. This keeps a rejected event atomic.
        let mut staged = Vec::new();
        let mut contact_count = 0;
        let mut inserted = 0;
        let mut coalesced = 0;
        let mut next_time = None;
        if current != desired {
            let first = self.graph.offsets[source] as usize;
            let last = self.graph.offsets[source + 1] as usize;
            contact_count = last - first;
            let mut deltas = BTreeMap::<usize, u128>::new();
            for edge in first..last {
                let count = u128::from(self.graph.counts[edge]);
                if count == 0 {
                    continue;
                }
                let target = self.graph.targets[edge] as usize;
                let item = deltas.entry(target).or_insert(0);
                *item = item
                    .checked_add(count)
                    .ok_or(EventError::ArithmeticOverflow)?;
            }
            if !deltas.is_empty() {
                let following = time.checked_add(1).ok_or(EventError::LogicalTimeOverflow)?;
                for (target, delta) in deltas {
                    let old = self.incoming[target];
                    let new = if desired {
                        old.checked_add(delta)
                    } else {
                        old.checked_sub(delta)
                    }
                    .ok_or(EventError::ArithmeticOverflow)?;
                    if self.pending.contains_key(&(following, RECHECK, target)) {
                        coalesced += 1;
                    } else {
                        inserted += 1;
                    }
                    staged.push((target, new));
                }
                next_time = Some(following);
            }
            if self.pending.len() - 1 + inserted > self.max_pending {
                return Err(EventError::PendingBudgetExceeded);
            }
        }

        // The event is removed only once all preflight checks have passed.
        let _ = self.pending.pop_first();
        if current != desired {
            self.state
                .set(source, desired)
                .expect("validated neuron index");
            if let Some(following) = next_time {
                for (target, value) in staged {
                    self.incoming[target] = value;
                    let _ = self.pending.insert((following, RECHECK, target), false);
                }
            }
            work.state_transitions += 1;
            if desired {
                work.rising_transitions += 1;
            } else {
                work.falling_transitions += 1;
            }
            work.contacts_examined += contact_count;
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

    fn chain() -> Connectome {
        Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 1, 2, 2],
            targets: vec![1, 2],
            counts: vec![1, 1],
        }
    }

    fn converge() -> Connectome {
        Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 1, 2, 2],
            targets: vec![2, 2],
            counts: vec![1, 1],
        }
    }

    fn fork() -> Connectome {
        Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 2, 2, 2],
            targets: vec![1, 2],
            counts: vec![1, 1],
        }
    }

    #[test]
    fn sparse_chain_runs_without_global_sweeps_and_can_turn_off() {
        let graph = chain();
        let mut runtime = EventRuntime::new(&graph, 1, 16).unwrap();
        assert_eq!(runtime.schedule_drive(0, 10, true), Ok(true));
        let on = runtime.run(10).unwrap();
        assert_eq!(on.events_consumed, 3);
        assert_eq!(on.state_transitions, 3);
        assert_eq!(on.rising_transitions, 3);
        assert_eq!(on.falling_transitions, 0);
        assert_eq!(on.contacts_examined, 2);
        assert_eq!(on.threshold_checks, 2);
        assert_eq!(runtime.state().active_count(), 3);
        assert_eq!(runtime.incoming_for(30), Ok(1));
        assert_eq!(runtime.next_event_time(), None);

        runtime.schedule_drive(10, 10, false).unwrap();
        let off = runtime.run(10).unwrap();
        assert_eq!(off.events_consumed, 3);
        assert_eq!(off.state_transitions, 3);
        assert_eq!(off.rising_transitions, 0);
        assert_eq!(off.falling_transitions, 3);
        assert_eq!(runtime.state().active_count(), 0);
        assert_eq!(runtime.incoming_for(30), Ok(0));
    }

    #[test]
    fn convergent_drives_coalesce_one_recheck_and_respect_threshold() {
        let graph = converge();
        let mut runtime = EventRuntime::new(&graph, 2, 4).unwrap();
        runtime.schedule_drive(0, 20, true).unwrap();
        runtime.schedule_drive(0, 10, true).unwrap();
        let first = runtime.run(2).unwrap();
        assert_eq!(first.events_consumed, 2);
        assert_eq!(first.wakeups_inserted, 1);
        assert_eq!(first.wakeups_coalesced, 1);
        assert_eq!(runtime.next_event_time(), Some(1));
        assert_eq!(runtime.incoming_for(30), Ok(2));
        assert_eq!(runtime.run(1).unwrap().threshold_checks, 1);
        assert_eq!(runtime.state().active_count(), 3);
        runtime.schedule_drive(2, 10, false).unwrap();
        runtime.run(10).unwrap();
        assert_eq!(runtime.state().active_count(), 1);
        assert_eq!(runtime.incoming_for(30), Ok(1));
    }

    #[test]
    fn insertion_order_does_not_affect_result_or_work() {
        let graph = converge();
        let mut a = EventRuntime::new(&graph, 2, 8).unwrap();
        let mut b = EventRuntime::new(&graph, 2, 8).unwrap();
        for id in [10, 20] {
            a.schedule_drive(0, id, true).unwrap();
        }
        for id in [20, 10] {
            b.schedule_drive(0, id, true).unwrap();
        }
        assert_eq!(a.run(20), b.run(20));
        assert_eq!(a.state(), b.state());
    }

    #[test]
    fn queue_exhaustion_rejects_event_without_partial_mutation() {
        let graph = fork();
        let mut runtime = EventRuntime::new(&graph, 1, 1).unwrap();
        runtime.schedule_drive(0, 10, true).unwrap();
        assert_eq!(runtime.run(1), Err(EventError::PendingBudgetExceeded));
        assert_eq!(runtime.state().active_count(), 0);
        assert_eq!(runtime.incoming_for(20), Ok(0));
        assert_eq!(runtime.pending_count(), 1);
        assert_eq!(runtime.next_event_time(), Some(0));
    }

    #[test]
    fn invalid_inputs_duplicates_and_no_op_do_not_propagate() {
        let graph = chain();
        assert!(matches!(
            EventRuntime::new(&graph, 0, 1),
            Err(EventError::InvalidThreshold)
        ));
        assert!(matches!(
            EventRuntime::new(&graph, 1, 0),
            Err(EventError::InvalidPendingBudget)
        ));
        let mut runtime = EventRuntime::new(&graph, 1, 4).unwrap();
        assert_eq!(
            runtime.schedule_drive(0, 999, true),
            Err(EventError::UnknownNeuron)
        );
        assert_eq!(runtime.schedule_drive(0, 10, false), Ok(true));
        assert_eq!(runtime.schedule_drive(0, 10, false), Ok(false));
        assert_eq!(
            runtime.schedule_drive(0, 10, true),
            Err(EventError::ConflictingDrive)
        );
        let no_op = runtime.run(10).unwrap();
        assert_eq!(no_op.events_consumed, 1);
        assert_eq!(no_op.state_transitions, 0);
        assert_eq!(no_op.contacts_examined, 0);
        assert_eq!(
            runtime.schedule_drive(0, 10, true),
            Err(EventError::TimeNotFuture)
        );
    }

    #[test]
    fn independent_oracle_matches_every_partial_cyclic_transition() {
        let graph = Connectome {
            ids: vec![10, 20],
            offsets: vec![0, 1, 2],
            targets: vec![1, 0],
            counts: vec![1, 1],
        };
        let mut runtime = EventRuntime::new(&graph, 1, 4).unwrap();
        runtime.schedule_drive(0, 10, true).unwrap();
        for _ in 0..3 {
            assert_eq!(runtime.run(1).unwrap().events_consumed, 1);
            assert_eq!(runtime.audit_incoming(), Ok(true));
        }
        assert_eq!(runtime.state().active_count(), 2);
        runtime.schedule_drive(3, 10, false).unwrap();
        for _ in 0..3 {
            assert_eq!(runtime.run(1).unwrap().events_consumed, 1);
            assert_eq!(runtime.audit_incoming(), Ok(true));
        }
        assert_eq!(runtime.state().active_count(), 0);
        assert_eq!(runtime.pending_count(), 0);
    }

    #[test]
    fn duplicate_contacts_preserve_full_u128_multiplicity() {
        let graph = Connectome {
            ids: vec![10, 20],
            offsets: vec![0, 2, 2],
            targets: vec![1, 1],
            counts: vec![u64::MAX, u64::MAX],
        };
        let threshold = u128::from(u64::MAX) * 2;
        let mut runtime = EventRuntime::new(&graph, threshold, 4).unwrap();
        runtime.schedule_drive(0, 10, true).unwrap();
        let first = runtime.run(2).unwrap();
        assert_eq!(first.wakeups_inserted, 1);
        assert_eq!(first.contacts_examined, 2);
        assert_eq!(runtime.incoming_for(20), Ok(threshold));
        assert_eq!(runtime.state().active_count(), 2);
        assert_eq!(runtime.audit_incoming(), Ok(true));
        runtime.schedule_drive(2, 10, false).unwrap();
        runtime.run(2).unwrap();
        assert_eq!(runtime.incoming_for(20), Ok(0));
        assert_eq!(runtime.audit_incoming(), Ok(true));
        assert_eq!(runtime.state().active_count(), 0);
    }

    #[test]
    fn timestamp_overflow_preserves_pending_event_and_state() {
        let graph = chain();
        let mut runtime = EventRuntime::new(&graph, 1, 4).unwrap();
        runtime.schedule_drive(u64::MAX, 10, true).unwrap();
        assert_eq!(runtime.run(1), Err(EventError::LogicalTimeOverflow));
        assert_eq!(runtime.pending_count(), 1);
        assert_eq!(runtime.next_event_time(), Some(u64::MAX));
        assert_eq!(runtime.last_processed_time(), None);
        assert_eq!(runtime.state().active_count(), 0);
        assert_eq!(runtime.incoming_for(20), Ok(0));
    }

    #[test]
    fn contact_budget_blocks_atomically_then_allows_resume() {
        let graph = fork();
        let mut runtime = EventRuntime::new(&graph, 1, 4).unwrap();
        runtime.schedule_drive(0, 10, true).unwrap();
        let blocked = runtime.run_with_contact_budget(10, 1).unwrap();
        assert!(blocked.blocked_on_contact_budget);
        assert_eq!(blocked.work.events_consumed, 0);
        assert_eq!(runtime.pending_count(), 1);
        assert_eq!(runtime.state().active_count(), 0);
        assert_eq!(runtime.incoming_for(20), Ok(0));

        let resumed = runtime.run_with_contact_budget(1, 2).unwrap();
        assert!(!resumed.blocked_on_contact_budget);
        assert_eq!(resumed.work.events_consumed, 1);
        assert_eq!(resumed.work.contacts_examined, 2);
        assert_eq!(runtime.pending_count(), 2);
        let zero_contact = runtime.run_with_contact_budget(10, 0).unwrap();
        assert!(!zero_contact.blocked_on_contact_budget);
        assert_eq!(zero_contact.work.events_consumed, 2);
        assert_eq!(zero_contact.work.contacts_examined, 0);
        assert_eq!(runtime.state().active_count(), 3);
        assert_eq!(runtime.audit_incoming(), Ok(true));
    }

    #[test]
    fn bounded_slices_equal_unbounded_chain_and_preserve_order() {
        let graph = chain();
        let mut bounded = EventRuntime::new(&graph, 1, 8).unwrap();
        let mut baseline = EventRuntime::new(&graph, 1, 8).unwrap();
        for runtime in [&mut bounded, &mut baseline] {
            runtime.schedule_drive(0, 10, true).unwrap();
        }
        let expected = baseline.run(10).unwrap();
        assert_eq!(expected.events_consumed, 3);

        let first = bounded.run_with_contact_budget(1, 1).unwrap();
        assert_eq!(first.work.events_consumed, 1);
        assert_eq!(first.work.contacts_examined, 1);
        let blocked = bounded.run_with_contact_budget(10, 0).unwrap();
        assert!(blocked.blocked_on_contact_budget);
        assert_eq!(blocked.work.events_consumed, 0);
        assert_eq!(bounded.next_event_time(), Some(1));
        let second = bounded.run_with_contact_budget(1, 1).unwrap();
        assert_eq!(second.work.events_consumed, 1);
        assert_eq!(second.work.contacts_examined, 1);
        let third = bounded.run_with_contact_budget(10, 0).unwrap();
        assert_eq!(third.work.events_consumed, 1);
        assert!(!third.blocked_on_contact_budget);
        assert_eq!(bounded.state(), baseline.state());
        assert_eq!(bounded.incoming, baseline.incoming);
        assert_eq!(
            bounded.last_processed_time(),
            baseline.last_processed_time()
        );
        assert_eq!(bounded.next_event_time(), None);
    }

    #[test]
    fn zero_event_budget_is_noop_and_zero_contact_noop_drive_proceeds() {
        let graph = chain();
        let mut runtime = EventRuntime::new(&graph, 1, 4).unwrap();
        runtime.schedule_drive(0, 10, false).unwrap();
        let zero_events = runtime.run_with_contact_budget(0, 0).unwrap();
        assert_eq!(zero_events.work.events_consumed, 0);
        assert!(!zero_events.blocked_on_contact_budget);
        assert_eq!(runtime.pending_count(), 1);
        let one = runtime.run_with_contact_budget(1, 0).unwrap();
        assert_eq!(one.work.events_consumed, 1);
        assert_eq!(one.work.contacts_examined, 0);
        assert_eq!(runtime.pending_count(), 0);
    }

    #[test]
    fn zero_contact_weights_do_not_wake_targets() {
        let mut graph = chain();
        graph.counts[0] = 0;
        let mut runtime = EventRuntime::new(&graph, 1, 4).unwrap();
        runtime.schedule_drive(0, 10, true).unwrap();
        let work = runtime.run(10).unwrap();
        assert_eq!(work.events_consumed, 1);
        assert_eq!(work.contacts_examined, 1);
        assert_eq!(work.wakeups_inserted, 0);
        assert_eq!(runtime.state().active_count(), 1);
    }
}

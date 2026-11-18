//! Atomic ingress of a bounded batch of timestamped Boolean drives.
//!
//! A caller may prepare commands independently, then publish them as one
//! transaction. No graph state or event queue changes until all commands,
//! duplicates and queue capacity have been checked.

use std::collections::BTreeMap;

use super::{EventError, EventKey, EventRuntime, DRIVE};

/// Exact queue-ingress accounting; not execution work or a throughput claim.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DriveBatchReport {
    /// Newly scheduled distinct drives.
    pub inserted: usize,
    /// Inputs identical to an already queued drive.
    pub already_pending: usize,
    /// Repetitions of an earlier command within this batch.
    pub repeated_in_batch: usize,
}

impl EventRuntime<'_> {
    /// Publish an entire drive batch atomically, in canonical event-key order.
    ///
    /// All inputs must be for known neuron IDs and strictly later than the
    /// last processed logical time. Identical duplicates are idempotent.
    /// Opposite assignments for the same (time, neuron), either within the
    /// batch or against the existing queue, reject the whole batch.
    ///
    /// The queue-capacity check counts only distinct new commands. On any
    /// error the pending queue, neuron bits, counters and logical clock
    /// remain untouched. This operation does NOT run or parallelize events.
    pub fn schedule_drive_batch(
        &mut self,
        drives: &[(u64, u64, bool)],
    ) -> Result<DriveBatchReport, EventError> {
        let mut staged = BTreeMap::<EventKey, bool>::new();
        let mut report = DriveBatchReport::default();

        for &(logical_time, neuron_id, requested) in drives {
            if self.last_time.is_some_and(|last| logical_time <= last) {
                return Err(EventError::TimeNotFuture);
            }
            let source = self
                .graph
                .index(neuron_id)
                .ok_or(EventError::UnknownNeuron)?;
            let key = (logical_time, DRIVE, source);

            if let Some(&existing) = self.pending.get(&key) {
                if existing != requested {
                    return Err(EventError::ConflictingDrive);
                }
                report.already_pending += 1;
            } else if let Some(&earlier) = staged.get(&key) {
                if earlier != requested {
                    return Err(EventError::ConflictingDrive);
                }
                report.repeated_in_batch += 1;
            } else {
                staged.insert(key, requested);
            }
        }

        if self
            .pending
            .len()
            .checked_add(staged.len())
            .is_none_or(|len| len > self.max_pending)
        {
            return Err(EventError::PendingBudgetExceeded);
        }

        report.inserted = staged.len();
        // All staged keys were proved absent from pending. append preserves
        // BTreeMap ordering without exposing a partially committed batch.
        self.pending.append(&mut staged);
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome::Connectome;

    fn chain() -> Connectome {
        Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 1, 2, 2],
            targets: vec![1, 2],
            counts: vec![1, 1],
        }
    }

    #[test]
    fn batch_preserves_reference_execution_and_causal_order() {
        let graph = chain();
        let drives = [
            (4, 10, true),
            (2, 20, true),
            (4, 10, true),
            (8, 20, false),
            (2, 20, true),
            (8, 10, false),
        ];
        let mut batch = EventRuntime::new(&graph, 1, 32).unwrap();
        let mut reference = EventRuntime::new(&graph, 1, 32).unwrap();
        assert_eq!(
            batch.schedule_drive_batch(&drives),
            Ok(DriveBatchReport {
                inserted: 4,
                already_pending: 0,
                repeated_in_batch: 2,
            })
        );
        for &(time, id, bit) in &drives {
            reference.schedule_drive(time, id, bit).unwrap();
        }
        assert_eq!(batch.run(100).unwrap(), reference.run(100).unwrap());
        assert_eq!(batch.state(), reference.state());
        assert_eq!(batch.pending_count(), reference.pending_count());
        assert_eq!(batch.last_processed_time(), reference.last_processed_time());
        for id in [10, 20, 30] {
            assert_eq!(batch.incoming_for(id), reference.incoming_for(id));
        }
        assert_eq!(batch.audit_incoming(), Ok(true));
    }

    #[test]
    fn conflicting_inputs_and_unknown_ids_roll_back_every_insertion() {
        let graph = chain();
        let mut runtime = EventRuntime::new(&graph, 1, 4).unwrap();
        assert_eq!(
            runtime.schedule_drive_batch(&[(1, 10, true), (2, 20, true), (1, 10, false)]),
            Err(EventError::ConflictingDrive)
        );
        assert_eq!(
            runtime.schedule_drive_batch(&[(1, 10, true), (2, 999, true)]),
            Err(EventError::UnknownNeuron)
        );
        assert_eq!(runtime.pending_count(), 0);
        assert_eq!(runtime.next_event_time(), None);
        assert_eq!(runtime.state().active_count(), 0);
        assert_eq!(runtime.last_processed_time(), None);
    }

    #[test]
    fn existing_conflict_and_capacity_preserve_preexisting_queue() {
        let graph = chain();
        let mut runtime = EventRuntime::new(&graph, 1, 1).unwrap();
        runtime.schedule_drive(3, 10, true).unwrap();
        assert_eq!(
            runtime.schedule_drive_batch(&[(4, 20, true), (3, 10, false)]),
            Err(EventError::ConflictingDrive)
        );
        assert_eq!(
            runtime.schedule_drive_batch(&[(3, 10, true), (4, 20, true)]),
            Err(EventError::PendingBudgetExceeded)
        );
        assert_eq!(
            runtime.schedule_drive_batch(&[(3, 10, true), (3, 10, true)]),
            Ok(DriveBatchReport {
                inserted: 0,
                already_pending: 2,
                repeated_in_batch: 0,
            })
        );
        assert_eq!(runtime.pending_count(), 1);
        assert_eq!(runtime.next_event_time(), Some(3));
        assert_eq!(runtime.state().active_count(), 0);
    }

    #[test]
    fn past_event_and_empty_batch_are_safe_after_partial_progress() {
        let graph = chain();
        let mut runtime = EventRuntime::new(&graph, 1, 8).unwrap();
        assert_eq!(
            runtime.schedule_drive_batch(&[]),
            Ok(DriveBatchReport::default())
        );
        runtime.schedule_drive(2, 10, true).unwrap();
        runtime.run(1).unwrap();
        assert_eq!(runtime.last_processed_time(), Some(2));
        assert_eq!(
            runtime.schedule_drive_batch(&[(3, 30, true), (2, 20, false)]),
            Err(EventError::TimeNotFuture)
        );
        assert_eq!(runtime.pending_count(), 1);
        assert_eq!(runtime.next_event_time(), Some(3));
        assert_eq!(runtime.incoming_for(20), Ok(1));
        assert_eq!(runtime.audit_incoming(), Ok(true));
    }

    #[test]
    fn batch_is_compatible_with_disjoint_commit_scheduler() {
        let graph = chain();
        let drives = [(1, 10, true), (1, 20, true), (4, 10, false)];
        let mut reference = EventRuntime::new(&graph, 1, 16).unwrap();
        let mut parallel = EventRuntime::new(&graph, 1, 16).unwrap();
        reference.schedule_drive_batch(&drives).unwrap();
        parallel.schedule_drive_batch(&drives).unwrap();
        let expected = reference.run(100).unwrap();
        let actual = parallel.run_disjoint_parallel_commits(100, 2).unwrap();
        assert_eq!(actual.work, expected);
        assert_eq!(parallel.state(), reference.state());
        assert_eq!(parallel.audit_incoming(), Ok(true));
    }
}

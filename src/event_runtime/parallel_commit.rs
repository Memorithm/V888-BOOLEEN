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
use std::time::{Duration, Instant};

use super::admission::MeasuredDisjointCosts;
use super::contacts::effective_outgoing_work;
use super::parallel::{prepare_one, PreparedEvent};
use super::{EventError, EventKey, EventRuntime, EventWork, DRIVE, RECHECK};

const MAX_COMMIT_THREADS: usize = 32;
const EVENTS_PER_THREAD: usize = 32;
/// Preliminary admission threshold: measured runtime cost, not joule calibration.
const MIN_PARALLEL_CONTACTS: usize = 65_536;
/// Sparse target writes cannot amortize two sets of scoped threads even when
/// the uncached CSR fanout is large. This is a conservative CPU heuristic.
const MIN_PARALLEL_TARGETS: usize = 1_024;
/// Small groups never create two thread passes, even while probing.
const MIN_PROBE_CONTACTS: usize = 256;
const MIN_PROBE_TARGETS: usize = 32;
const COST_BUCKETS: usize = 12;
const COST_OBSERVATIONS: u8 = 2;
const PROBE_EVERY: u64 = 32;

/// Characteristics of the *current causal prefix*, not global graph averages.
/// A cached row has a smaller effective preparation count than its raw CSR size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrefixTopology {
    changing_sources: usize,
    raw_contacts: usize,
    effective_contacts: usize,
    distinct_targets: usize,
}

impl PrefixTopology {
    fn probe_eligible(self) -> bool {
        self.changing_sources >= 2
            && self.effective_contacts >= MIN_PROBE_CONTACTS
            && self.distinct_targets >= MIN_PROBE_TARGETS
    }

    fn legacy_eligible(self) -> bool {
        self.changing_sources >= 2
            && self.effective_contacts >= MIN_PARALLEL_CONTACTS
            && self.distinct_targets >= MIN_PARALLEL_TARGETS
    }

    /// Three target-cardinality bands, whether the cache reduced traversal,
    /// and whether the anatomical fanout repeats target IDs heavily.
    fn bucket(self) -> usize {
        let target_band = if self.distinct_targets < 128 {
            0
        } else if self.distinct_targets < MIN_PARALLEL_TARGETS {
            1
        } else {
            2
        };
        let cached = usize::from(self.raw_contacts > self.effective_contacts.saturating_mul(4));
        let repeated = usize::from(self.raw_contacts > self.distinct_targets.saturating_mul(16));
        target_band * 4 + cached * 2 + repeated
    }

    fn units(self) -> usize {
        self.effective_contacts
            .saturating_add(self.distinct_targets)
            .saturating_add(self.changing_sources.saturating_mul(8))
            .max(1)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CostSample {
    /// Smoothed nanoseconds per 1024 modeled units; never treated as energy.
    ns_per_1024_units: u128,
    observations: u8,
}

impl CostSample {
    fn record(&mut self, topology: PrefixTopology, elapsed: Duration) {
        let sample = elapsed.as_nanos().max(1).saturating_mul(1_024) / topology.units() as u128;
        if self.observations == 0 {
            self.ns_per_1024_units = sample;
        } else {
            self.ns_per_1024_units = self
                .ns_per_1024_units
                .saturating_mul(3)
                .saturating_add(sample)
                / 4;
        }
        self.observations = self.observations.saturating_add(1);
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CostBucket {
    serial: CostSample,
    parallel: CostSample,
    candidates: u64,
}

/// Optional reusable, locally measured admission profile for disjoint commits.
///
/// No wall-clock observation changes Boolean arithmetic, queue order, or the
/// disjointness proof. Each bucket learns from actual scoped-thread and serial
/// groups of comparable target width, cache reduction and target repetition.
/// Keep one profile across repeated calls/runs to retain the measurements.
/// No process-global state, persisted calibration, joule or speedup claim.
#[derive(Clone, Debug)]
pub struct DisjointAdmissionProfile {
    buckets: [CostBucket; COST_BUCKETS],
}

impl Default for DisjointAdmissionProfile {
    fn default() -> Self {
        Self {
            buckets: [CostBucket::default(); COST_BUCKETS],
        }
    }
}

impl DisjointAdmissionProfile {
    /// Returns the number of recorded serial and parallel group measurements.
    #[must_use]
    pub fn observation_counts(&self) -> (usize, usize) {
        self.buckets
            .iter()
            .fold((0, 0), |(serial, parallel), bucket| {
                (
                    serial + usize::from(bucket.serial.observations),
                    parallel + usize::from(bucket.parallel.observations),
                )
            })
    }

    fn should_parallel(&mut self, topology: PrefixTopology) -> bool {
        if !topology.probe_eligible() {
            return false;
        }
        let bucket = &mut self.buckets[topology.bucket()];
        bucket.candidates = bucket.candidates.saturating_add(1);
        let serial_ready = bucket.serial.observations >= COST_OBSERVATIONS;
        let parallel_ready = bucket.parallel.observations >= COST_OBSERVATIONS;

        if serial_ready && parallel_ready {
            // Require a 15% measured margin to avoid unstable tie decisions.
            // Revisit the disfavored path periodically, in either direction.
            let parallel_faster = bucket.parallel.ns_per_1024_units.saturating_mul(100)
                < bucket.serial.ns_per_1024_units.saturating_mul(85);
            return if bucket.candidates % PROBE_EVERY == 0 {
                !parallel_faster
            } else {
                parallel_faster
            };
        }
        if serial_ready {
            // Controlled experiment after two real serial observations,
            // including on sub-65k topologies when they contain enough writes.
            return true;
        }
        if topology.legacy_eligible() {
            // Retain the historical first-admission behavior, but also acquire
            // serial baselines every third candidate to calibrate this bucket.
            return bucket.parallel.observations == 0 || bucket.candidates % 3 != 0;
        }
        false
    }

    fn observe(&mut self, topology: PrefixTopology, parallel: bool, elapsed: Duration) {
        if !topology.probe_eligible() {
            return;
        }
        let bucket = &mut self.buckets[topology.bucket()];
        if parallel {
            bucket.parallel.record(topology, elapsed);
        } else {
            bucket.serial.record(topology, elapsed);
        }
    }
}

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
    /// Candidate prefixes accepted by the optional external cost model.
    pub cost_profile_accepted: usize,
    /// Candidate prefixes rejected by the optional external cost model.
    pub cost_profile_rejected: usize,
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
        self.run_disjoint_parallel_commits_with_admission(max_events, requested_threads, None, None)
    }

    /// Opt-in locally observed admission. Retain measured topology-class
    /// costs across calls or independent repeated runs. The default entry
    /// point instead uses the conservative, fixed historical thresholds.
    /// Timing affects only scheduling, never Boolean arithmetic.
    pub fn run_disjoint_parallel_commits_with_profile(
        &mut self,
        max_events: usize,
        requested_threads: usize,
        profile: &mut DisjointAdmissionProfile,
    ) -> Result<DisjointCommitWork, EventError> {
        self.run_disjoint_parallel_commits_with_admission(
            max_events,
            requested_threads,
            Some(profile),
            None,
        )
    }

    /// Optional external-cost admission. The default online calibration
    /// remains available through run_disjoint_parallel_commits_with_profile.
    /// This method never trains the online profile or changes neuron rules.
    pub fn run_disjoint_parallel_commits_with_costs(
        &mut self,
        max_events: usize,
        requested_threads: usize,
        costs: MeasuredDisjointCosts,
    ) -> Result<DisjointCommitWork, EventError> {
        if requested_threads == 0 {
            return Err(EventError::InvalidWorkerCount);
        }
        if !costs.valid() || costs.workers != requested_threads.min(MAX_COMMIT_THREADS) {
            return Err(EventError::InvalidCostProfile);
        }
        self.run_disjoint_parallel_commits_with_admission(
            max_events,
            requested_threads,
            None,
            Some(costs),
        )
    }

    fn run_disjoint_parallel_commits_with_admission(
        &mut self,
        max_events: usize,
        requested_threads: usize,
        mut profile: Option<&mut DisjointAdmissionProfile>,
        costs: Option<MeasuredDisjointCosts>,
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
        // Necessary global bounds avoid constructing a conflict set when
        // there cannot be enough contacts or distinct destinations to
        // amortize scoped threads. Keep external calibration independent.
        let (minimum_contacts, minimum_targets) = if profile.is_some() {
            (MIN_PROBE_CONTACTS, MIN_PROBE_TARGETS)
        } else {
            (MIN_PARALLEL_CONTACTS, MIN_PARALLEL_TARGETS)
        };
        if costs.is_none()
            && (self.graph.targets.len() < minimum_contacts
                || self.graph.ids.len() < minimum_targets)
        {
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
            let (group, distinct_targets) = self.disjoint_drive_prefix(allowance);
            // RECHECK does not enter a DRIVE prefix. Always consume the
            // earliest event so that a nonempty queue makes progress.
            if group.is_empty() {
                self.consume_one(&mut result.work)?;
                result.serial_events += 1;
                continue;
            }
            let mut work_to_prepare = 0usize;
            let mut raw_contacts = 0usize;
            let mut changing_sources = 0usize;
            for &((_, _, source), requested) in &group {
                if self.state.get(source).expect("validated neuron index") != requested {
                    let remaining =
                        effective_outgoing_work(self.graph, self.contact_cache.as_ref(), source);
                    if remaining > 0 {
                        changing_sources += 1;
                        work_to_prepare = work_to_prepare.saturating_add(remaining);
                        raw_contacts = raw_contacts.saturating_add(
                            (self.graph.offsets[source + 1] - self.graph.offsets[source]) as usize,
                        );
                    }
                }
            }
            // Threaded writes require at least two independent producers,
            // not merely two queued drives (other drives can be no-ops).
            // Compressed cached rows have already paid their CSR scan cost.
            // Do not launch two thread passes for only a handful of pairs.
            let topology = PrefixTopology {
                changing_sources,
                raw_contacts,
                effective_contacts: work_to_prepare,
                distinct_targets,
            };
            let admit = if let Some(measured) = costs {
                measured.admits(
                    topology.changing_sources,
                    topology.effective_contacts,
                    topology.distinct_targets,
                    threads,
                )
            } else if let Some(adaptive) = profile.as_deref_mut() {
                adaptive.should_parallel(topology)
            } else {
                topology.legacy_eligible()
            };
            if costs.is_some() {
                if admit {
                    result.cost_profile_accepted += 1;
                } else {
                    result.cost_profile_rejected += 1;
                }
            }
            if !admit {
                // A rejected prefix can be consumed in canonical order.
                // Do not replan the same tiny group after each event.
                let serial_started = profile.as_ref().map(|_| Instant::now());
                for _ in 0..group.len() {
                    self.consume_one(&mut result.work)?;
                    result.serial_events += 1;
                }
                if let (Some(adaptive), Some(started)) = (profile.as_deref_mut(), serial_started) {
                    adaptive.observe(topology, false, started.elapsed());
                }
                continue;
            }

            let parallel_started = profile.as_ref().map(|_| Instant::now());
            // Parallel preflight has no side effects. Any failed preparation
            // must be reported at its original sequential event boundary.
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
            if let (Some(adaptive), Some(started)) = (profile.as_deref_mut(), parallel_started) {
                adaptive.observe(topology, true, started.elapsed());
            }
        }
        Ok(result)
    }

    /// Select only a true causal prefix: never reorder a conflicting event.
    /// Return the number of distinct nonzero writable targets for admission.
    /// A source that stays unchanged does not write its outgoing targets.
    fn disjoint_drive_prefix(&self, limit: usize) -> (Vec<(EventKey, bool)>, usize) {
        let Some((&(time, phase, _), _)) = self.pending.first_key_value() else {
            return (Vec::new(), 0);
        };
        if phase != DRIVE {
            return (Vec::new(), 0);
        }
        let mut touched = BTreeSet::new();
        let mut selected = Vec::new();
        for (&key, &requested) in self.pending.iter().take(limit) {
            if key.0 != time || key.1 != DRIVE {
                break;
            }
            let source = key.2;
            if self.state.get(source).expect("validated source") != requested {
                if let Some(reduced) = self.contact_cache.as_ref().and_then(|c| c.get(&source)) {
                    // Exact nonzero target set; no repeated CSR fanout scan.
                    if reduced.iter().any(|(target, _)| touched.contains(target)) {
                        break;
                    }
                    for &(target, _) in reduced {
                        touched.insert(target);
                    }
                } else {
                    let first = self.graph.offsets[source] as usize;
                    let last = self.graph.offsets[source + 1] as usize;
                    // Avoid repeated BTreeSet lookups for adjacent CSR rows
                    // with the same nonzero target. Zeros do not create writes.
                    let mut last_target = None;
                    if (first..last).any(|edge| {
                        if self.graph.counts[edge] == 0 {
                            return false;
                        }
                        let target = self.graph.targets[edge] as usize;
                        if last_target == Some(target) {
                            return false;
                        }
                        last_target = Some(target);
                        touched.contains(&target)
                    }) {
                        break;
                    }
                    let mut last_target = None;
                    for edge in first..last {
                        if self.graph.counts[edge] != 0 {
                            let target = self.graph.targets[edge] as usize;
                            if last_target != Some(target) {
                                touched.insert(target);
                                last_target = Some(target);
                            }
                        }
                    }
                }
            }
            selected.push((key, requested));
        }
        (selected, touched.len())
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
    fn cached_conflict_guard_does_not_admit_shared_target() {
        let graph = Connectome {
            ids: (10..26).collect(),
            offsets: (0..=8)
                .map(|source| source * 8192)
                .chain(std::iter::repeat_n(65_536, 8))
                .collect(),
            // First and second source both write neuron 18 (index 8).
            targets: (0..8)
                .flat_map(|source| {
                    let target = if source < 2 { 8 } else { source + 8 };
                    std::iter::repeat_n(target, 8192)
                })
                .collect(),
            counts: vec![1; 65_536],
        };
        graph.validate().unwrap();
        let mut reference = EventRuntime::new(&graph, 1, 64).unwrap();
        let mut cached = EventRuntime::new(&graph, 1, 64).unwrap();
        assert_eq!(cached.prepare_contact_cache(8).unwrap().cached_pairs, 8);
        for engine in [&mut reference, &mut cached] {
            for id in 10..18 {
                engine.schedule_drive(0, id, true).unwrap();
            }
        }
        assert_eq!(
            reference.disjoint_drive_prefix(8),
            (vec![((0, DRIVE, 0), true)], 1)
        );
        assert_eq!(
            cached.disjoint_drive_prefix(8),
            (vec![((0, DRIVE, 0), true)], 1)
        );
        let expected = reference.run(64).unwrap();
        let actual = cached.run_disjoint_parallel_commits(64, 4).unwrap();
        assert_eq!(actual.work, expected);
        assert_eq!(cached.state(), reference.state());
        assert_eq!(cached.incoming, reference.incoming);
        assert_eq!(cached.pending, reference.pending);
        assert_eq!(cached.audit_incoming(), Ok(true));
    }

    #[test]
    fn few_distinct_targets_skip_threads_despite_large_uncached_fanout() {
        let graph = eight_disjoint_sources();
        let drives = (10..18).map(|id| (0, id, true)).collect::<Vec<_>>();
        let result = compare(&graph, 1, &drives, 20, 64, 4);
        assert_eq!(result.disjoint_groups, 0);
        assert_eq!(result.target_counters_written, 0);
        assert_eq!(result.target_writer_threads, 0);
        assert_eq!(result.serial_events, result.work.events_consumed);
        assert_eq!(result.work.state_transitions, 16);
        assert_eq!(result.work.contacts_examined, 65_536);
    }

    #[test]
    fn many_distinct_targets_still_use_real_parallel_writers() {
        // 65_536 uncached CSR rows contain 1_024 disjoint weighted targets.
        // This is deliberately above both thread-admission thresholds.
        let target_count = MIN_PARALLEL_TARGETS;
        let node_count = target_count + 2;
        let mut offsets = vec![65_536; node_count + 1];
        offsets[0] = 0;
        offsets[1] = 32_768;
        let graph = Connectome {
            ids: (10..10 + node_count as u64).collect(),
            offsets,
            targets: (0..2)
                .flat_map(|source| {
                    (0..target_count / 2).flat_map(move |target| {
                        std::iter::repeat_n((2 + source * (target_count / 2) + target) as u32, 64)
                    })
                })
                .collect(),
            counts: vec![1; 65_536],
        };
        graph.validate().unwrap();
        let result = compare(
            &graph,
            1,
            &[(0, 10, true), (0, 11, true)],
            node_count,
            2_048,
            4,
        );
        assert_eq!(result.disjoint_groups, 1);
        assert_eq!(result.grouped_events, 2);
        assert_eq!(result.target_counters_written, target_count);
        assert_eq!(result.target_writer_threads, 4);
        assert_eq!(result.work.contacts_examined, 65_536);
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
    fn independent_negative_fronts_skip_threads_when_writes_are_sparse() {
        let graph = eight_disjoint_sources();
        let mut drives = (10..18).map(|id| (0, id, true)).collect::<Vec<_>>();
        drives.extend((10..18).map(|id| (3, id, false)));
        let stats = compare(&graph, 1, &drives, 100, 64, 4);
        assert_eq!(stats.disjoint_groups, 0);
        assert_eq!(stats.work.rising_transitions, 16);
        assert_eq!(stats.work.falling_transitions, 16);
        assert_eq!(stats.target_counters_written, 0);
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
        let result = compare(&graph, 1, &[(0, 10, true), (0, 20, false)], 8, 8, 4);
        assert_eq!(result.disjoint_groups, 0);
        assert_eq!(result.target_writer_threads, 0);
        assert_eq!(result.serial_events, result.work.events_consumed);
        assert_eq!(result.work.state_transitions, 2);
    }

    #[test]
    fn measured_costs_reject_slow_parallel_and_admit_faster_parallel() {
        let shape = PrefixTopology {
            changing_sources: 2,
            raw_contacts: 512,
            effective_contacts: 512,
            distinct_targets: 256,
        };
        let mut slow = DisjointAdmissionProfile::default();
        let mut fast = DisjointAdmissionProfile::default();
        assert!(!slow.should_parallel(shape)); // Cold sub-65k path.
        for _ in 0..2 {
            slow.observe(shape, false, Duration::from_millis(2));
            slow.observe(shape, true, Duration::from_millis(8));
            fast.observe(shape, false, Duration::from_millis(8));
            fast.observe(shape, true, Duration::from_millis(2));
        }
        assert!(!slow.should_parallel(shape));
        assert!(fast.should_parallel(shape));
        assert_eq!(slow.observation_counts(), (2, 2));
        assert_eq!(fast.observation_counts(), (2, 2));
        assert!(shape.probe_eligible());
        assert!(!shape.legacy_eligible());
    }

    #[test]
    fn periodic_probes_resample_the_disfavored_path_in_both_directions() {
        let shape = PrefixTopology {
            changing_sources: 2,
            raw_contacts: 512,
            effective_contacts: 512,
            distinct_targets: 256,
        };
        let mut fast = DisjointAdmissionProfile::default();
        let mut slow = DisjointAdmissionProfile::default();
        for _ in 0..2 {
            fast.observe(shape, false, Duration::from_millis(8));
            fast.observe(shape, true, Duration::from_millis(2));
            slow.observe(shape, false, Duration::from_millis(2));
            slow.observe(shape, true, Duration::from_millis(8));
        }
        for _ in 0..PROBE_EVERY - 1 {
            assert!(fast.should_parallel(shape));
            assert!(!slow.should_parallel(shape));
        }
        assert!(!fast.should_parallel(shape)); // Refresh serial baseline.
        assert!(slow.should_parallel(shape)); // Refresh parallel baseline.
    }

    #[test]
    fn topology_buckets_separate_caching_repetition_and_target_width() {
        let medium = PrefixTopology {
            changing_sources: 2,
            raw_contacts: 512,
            effective_contacts: 512,
            distinct_targets: 256,
        };
        let cached = PrefixTopology {
            raw_contacts: 8_192,
            effective_contacts: 512,
            ..medium
        };
        let repeated = PrefixTopology {
            raw_contacts: 8_192,
            effective_contacts: 8_192,
            ..medium
        };
        let narrow = PrefixTopology {
            distinct_targets: 32,
            ..medium
        };
        assert_ne!(medium.bucket(), cached.bucket());
        assert_ne!(medium.bucket(), repeated.bucket());
        assert_ne!(medium.bucket(), narrow.bucket());
        assert!(medium.bucket() < COST_BUCKETS);
        let single_producer = PrefixTopology {
            changing_sources: 1,
            ..medium
        };
        let too_few_targets = PrefixTopology {
            distinct_targets: 8,
            ..medium
        };
        assert!(!single_producer.probe_eligible());
        assert!(!too_few_targets.probe_eligible());
    }

    #[test]
    fn calibrated_subthreshold_parallel_commit_preserves_full_serial_state() {
        let target_count = 256usize;
        let node_count = target_count + 2;
        let mut offsets = vec![target_count as u64; node_count + 1];
        offsets[0] = 0;
        offsets[1] = (target_count / 2) as u64;
        let graph = Connectome {
            ids: (10..10 + node_count as u64).collect(),
            offsets,
            targets: (2..2 + target_count as u32).collect(),
            counts: vec![1; target_count],
        };
        graph.validate().unwrap();

        let mut profile = DisjointAdmissionProfile::default();
        let shape = PrefixTopology {
            changing_sources: 2,
            raw_contacts: target_count,
            effective_contacts: target_count,
            distinct_targets: target_count,
        };
        // Inject measured examples only for this test, never for production.
        for _ in 0..2 {
            profile.observe(shape, false, Duration::from_millis(8));
            profile.observe(shape, true, Duration::from_millis(2));
        }
        let mut reference = EventRuntime::new(&graph, 1, 1024).unwrap();
        let mut adapted = EventRuntime::new(&graph, 1, 1024).unwrap();
        for runtime in [&mut reference, &mut adapted] {
            for (time, value) in [(0, true), (3, false)] {
                runtime.schedule_drive(time, 10, value).unwrap();
                runtime.schedule_drive(time, 11, value).unwrap();
            }
        }
        let expected = reference.run(usize::MAX).unwrap();
        let actual = adapted
            .run_disjoint_parallel_commits_with_profile(usize::MAX, 4, &mut profile)
            .unwrap();
        assert!(actual.disjoint_groups >= 1);
        assert_eq!(actual.work, expected);
        assert_eq!(adapted.state(), reference.state());
        assert_eq!(adapted.incoming, reference.incoming);
        assert_eq!(adapted.pending, reference.pending);
        assert_eq!(adapted.last_time, reference.last_time);
        assert_eq!(adapted.audit_incoming(), Ok(true));
    }

    #[test]
    fn calibrated_preflight_capacity_error_keeps_exact_partial_state() {
        let target_count = 256usize;
        let node_count = target_count + 2;
        let mut offsets = vec![target_count as u64; node_count + 1];
        offsets[0] = 0;
        offsets[1] = (target_count / 2) as u64;
        let graph = Connectome {
            ids: (10..10 + node_count as u64).collect(),
            offsets,
            targets: (2..2 + target_count as u32).collect(),
            counts: vec![1; target_count],
        };
        graph.validate().unwrap();
        let topology = PrefixTopology {
            changing_sources: 2,
            raw_contacts: target_count,
            effective_contacts: target_count,
            distinct_targets: target_count,
        };
        let mut profile = DisjointAdmissionProfile::default();
        for _ in 0..2 {
            profile.observe(topology, false, Duration::from_millis(8));
            profile.observe(topology, true, Duration::from_millis(2));
        }
        let mut serial = EventRuntime::new(&graph, 1, 130).unwrap();
        let mut tuned = EventRuntime::new(&graph, 1, 130).unwrap();
        for engine in [&mut serial, &mut tuned] {
            engine.schedule_drive(0, 10, true).unwrap();
            engine.schedule_drive(0, 11, true).unwrap();
        }
        assert_eq!(
            serial.run(usize::MAX),
            Err(EventError::PendingBudgetExceeded)
        );
        assert_eq!(
            tuned.run_disjoint_parallel_commits_with_profile(usize::MAX, 4, &mut profile),
            Err(EventError::PendingBudgetExceeded)
        );
        assert_eq!(tuned.state(), serial.state());
        assert_eq!(tuned.incoming, serial.incoming);
        assert_eq!(tuned.pending, serial.pending);
        assert_eq!(tuned.last_time, serial.last_time);
        assert_eq!(tuned.audit_incoming(), Ok(true));
    }

    #[test]
    fn wide_legacy_path_uses_threads_before_adaptive_calibration() {
        let large = PrefixTopology {
            changing_sources: 2,
            raw_contacts: 65_536,
            effective_contacts: 65_536,
            distinct_targets: 1_024,
        };
        let mut profile = DisjointAdmissionProfile::default();
        assert!(profile.should_parallel(large));
        for _ in 0..2 {
            profile.observe(large, false, Duration::from_millis(1));
            profile.observe(large, true, Duration::from_millis(10));
        }
        assert!(!profile.should_parallel(large));
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

    fn small_disjoint_graph() -> Connectome {
        Connectome {
            ids: (10..20).collect(),
            offsets: [0, 4, 8]
                .into_iter()
                .chain(std::iter::repeat_n(8, 8))
                .collect(),
            targets: (2..10).collect(),
            counts: vec![1; 8],
        }
    }

    fn small_measured_costs() -> MeasuredDisjointCosts {
        // Synthetic positive coefficients prove selection logic, not speedup.
        MeasuredDisjointCosts {
            workers: 2,
            min_producers: 2,
            max_producers: 2,
            min_effective_contacts: 8,
            max_effective_contacts: 8,
            min_distinct_targets: 8,
            max_distinct_targets: 8,
            two_pass_overhead_ns: 100,
            serial_contact_ns: 15,
            serial_target_ns: 10,
            parallel_contact_ns: 2,
            parallel_target_ns: 3,
        }
    }

    #[test]
    fn favorable_measured_profile_admits_small_topology_without_changing_results() {
        let graph = small_disjoint_graph();
        graph.validate().unwrap();
        let mut baseline = EventRuntime::new(&graph, 1, 32).unwrap();
        let mut measured = EventRuntime::new(&graph, 1, 32).unwrap();
        for runtime in [&mut baseline, &mut measured] {
            runtime.schedule_drive(0, 10, true).unwrap();
            runtime.schedule_drive(0, 11, true).unwrap();
        }
        let expected = baseline.run(32).unwrap();
        let actual = measured
            .run_disjoint_parallel_commits_with_costs(32, 2, small_measured_costs())
            .unwrap();
        assert_eq!(actual.work, expected);
        assert_eq!(actual.disjoint_groups, 1);
        assert_eq!(actual.cost_profile_accepted, 1);
        assert_eq!(actual.cost_profile_rejected, 0);
        assert_eq!(actual.target_writer_threads, 2);
        assert_eq!(actual.target_counters_written, 8);
        assert_eq!(measured.state(), baseline.state());
        assert_eq!(measured.incoming, baseline.incoming);
        assert_eq!(measured.pending, baseline.pending);
        assert_eq!(measured.last_time, baseline.last_time);
        assert_eq!(measured.audit_incoming(), Ok(true));
    }

    #[test]
    fn expensive_or_unmatched_profile_falls_back_without_changing_results() {
        let graph = small_disjoint_graph();
        let mut baseline = EventRuntime::new(&graph, 1, 32).unwrap();
        let mut expensive = EventRuntime::new(&graph, 1, 32).unwrap();
        for runtime in [&mut baseline, &mut expensive] {
            runtime.schedule_drive(0, 10, true).unwrap();
            runtime.schedule_drive(0, 11, true).unwrap();
        }
        let expected = baseline.run(32).unwrap();
        let mut costs = small_measured_costs();
        costs.two_pass_overhead_ns = 1_000_000;
        let report = expensive
            .run_disjoint_parallel_commits_with_costs(32, 2, costs)
            .unwrap();
        assert_eq!(report.work, expected);
        assert_eq!(report.disjoint_groups, 0);
        assert_eq!(report.cost_profile_accepted, 0);
        assert_eq!(report.cost_profile_rejected, 1);
        assert_eq!(report.target_writer_threads, 0);
        assert_eq!(expensive.incoming, baseline.incoming);
        assert_eq!(expensive.pending, baseline.pending);
        assert_eq!(expensive.state(), baseline.state());

        let mut out_of_envelope = EventRuntime::new(&graph, 1, 32).unwrap();
        out_of_envelope.schedule_drive(0, 10, true).unwrap();
        out_of_envelope.schedule_drive(0, 11, true).unwrap();
        let mut costs = small_measured_costs();
        costs.min_effective_contacts = 9;
        costs.max_effective_contacts = 12;
        let report = out_of_envelope
            .run_disjoint_parallel_commits_with_costs(32, 2, costs)
            .unwrap();
        assert_eq!(report.work, expected);
        assert_eq!(report.disjoint_groups, 0);
        assert_eq!(out_of_envelope.incoming, baseline.incoming);
    }

    #[test]
    fn invalid_cost_profile_is_rejected_atomically() {
        let graph = small_disjoint_graph();
        let mut runtime = EventRuntime::new(&graph, 1, 32).unwrap();
        runtime.schedule_drive(0, 10, true).unwrap();
        let mut invalid = small_measured_costs();
        invalid.serial_contact_ns = 0;
        assert_eq!(
            runtime.run_disjoint_parallel_commits_with_costs(32, 2, invalid),
            Err(EventError::InvalidCostProfile)
        );
        assert_eq!(
            runtime.run_disjoint_parallel_commits_with_costs(32, 3, small_measured_costs()),
            Err(EventError::InvalidCostProfile)
        );
        assert_eq!(runtime.state().active_count(), 0);
        assert_eq!(runtime.pending_count(), 1);
        assert_eq!(runtime.next_event_time(), Some(0));
        assert_eq!(runtime.last_time, None);
    }

    #[test]
    fn cached_topology_uses_effective_pairs_not_raw_csr_row_count() {
        // 512 logical CSR rows compress to eight disjoint weighted pairs.
        // A profile calibrated only for eight effective iterations must
        // admit cached execution and reject the same graph without cache.
        let graph = Connectome {
            ids: (10..20).collect(),
            offsets: [0, 256, 512]
                .into_iter()
                .chain(std::iter::repeat_n(512, 8))
                .collect(),
            targets: (2..10)
                .flat_map(|target| std::iter::repeat_n(target, 64))
                .collect(),
            counts: vec![1; 512],
        };
        graph.validate().unwrap();
        let mut reference = EventRuntime::new(&graph, 1, 32).unwrap();
        let mut cached = EventRuntime::new(&graph, 1, 32).unwrap();
        let mut uncached = EventRuntime::new(&graph, 1, 32).unwrap();
        let cache_report = cached.prepare_contact_cache(8).unwrap();
        assert_eq!(cache_report.cached_pairs, 8);
        assert_eq!(cache_report.covered_contact_rows, 512);

        for runtime in [&mut reference, &mut cached, &mut uncached] {
            runtime.schedule_drive(0, 10, true).unwrap();
            runtime.schedule_drive(0, 11, true).unwrap();
        }
        let expected = reference.run(32).unwrap();
        let cached_work = cached
            .run_disjoint_parallel_commits_with_costs(32, 2, small_measured_costs())
            .unwrap();
        let uncached_work = uncached
            .run_disjoint_parallel_commits_with_costs(32, 2, small_measured_costs())
            .unwrap();
        assert_eq!(cached_work.work, expected);
        assert_eq!(uncached_work.work, expected);
        assert_eq!(cached_work.cost_profile_accepted, 1);
        assert_eq!(cached_work.disjoint_groups, 1);
        assert_eq!(uncached_work.cost_profile_accepted, 0);
        assert_eq!(uncached_work.disjoint_groups, 0);
        assert_eq!(cached.state(), reference.state());
        assert_eq!(cached.incoming, reference.incoming);
        assert_eq!(cached.pending, reference.pending);
        assert_eq!(uncached.state(), reference.state());
        assert_eq!(uncached.incoming, reference.incoming);
        assert_eq!(uncached.pending, reference.pending);
        assert_eq!(cached.audit_incoming(), Ok(true));
        assert_eq!(uncached.audit_incoming(), Ok(true));
    }

    #[test]
    fn cost_admitted_prefix_preserves_first_queue_capacity_error_and_partial_commit() {
        let graph = small_disjoint_graph();
        let mut reference = EventRuntime::new(&graph, 1, 6).unwrap();
        let mut modeled = EventRuntime::new(&graph, 1, 6).unwrap();
        for runtime in [&mut reference, &mut modeled] {
            runtime.schedule_drive(0, 10, true).unwrap();
            runtime.schedule_drive(0, 11, true).unwrap();
        }
        assert_eq!(reference.run(32), Err(EventError::PendingBudgetExceeded));
        assert_eq!(
            modeled.run_disjoint_parallel_commits_with_costs(32, 2, small_measured_costs()),
            Err(EventError::PendingBudgetExceeded)
        );
        assert_eq!(modeled.state(), reference.state());
        assert_eq!(modeled.incoming, reference.incoming);
        assert_eq!(modeled.pending, reference.pending);
        assert_eq!(modeled.last_time, reference.last_time);
        assert_eq!(modeled.state().active_count(), 1);
        assert_eq!(modeled.pending_count(), 5);
        assert_eq!(modeled.audit_incoming(), Ok(true));
    }

    #[test]
    fn cost_admitted_prefix_keeps_overflowing_logical_time_uncommitted() {
        let graph = small_disjoint_graph();
        let mut reference = EventRuntime::new(&graph, 1, 32).unwrap();
        let mut modeled = EventRuntime::new(&graph, 1, 32).unwrap();
        for runtime in [&mut reference, &mut modeled] {
            runtime.schedule_drive(u64::MAX, 10, true).unwrap();
            runtime.schedule_drive(u64::MAX, 11, true).unwrap();
        }
        assert_eq!(reference.run(32), Err(EventError::LogicalTimeOverflow));
        assert_eq!(
            modeled.run_disjoint_parallel_commits_with_costs(32, 2, small_measured_costs()),
            Err(EventError::LogicalTimeOverflow)
        );
        assert_eq!(modeled.state(), reference.state());
        assert_eq!(modeled.incoming, reference.incoming);
        assert_eq!(modeled.pending, reference.pending);
        assert_eq!(modeled.last_time, reference.last_time);
        assert_eq!(modeled.pending_count(), 2);
        assert_eq!(modeled.audit_incoming(), Ok(true));
    }
}

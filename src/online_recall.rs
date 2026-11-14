//! Streaming activation statistics for explicitly selected experimental subcircuits.
//! Observations are simulated states, never evidence of biological firing.
use crate::subcircuit_recall::{RecallError, Subcircuit};
use crate::BooleanState;

/// Bounded-memory online recall. No snapshots or anatomical edges are stored.
#[derive(Clone, Debug)]
pub struct OnlineRecall {
    circuit: Subcircuit,
    counts: Vec<u64>,
    ticks: u64,
    current_streak: u64,
    longest_streak: u64,
    quiet_streak: u64,
    last_recovery_delay: Option<u64>,
    recovery_events: u64,
}

impl OnlineRecall {
    pub fn new(circuit: Subcircuit) -> Self {
        let counts = vec![0; circuit.members().len()];
        Self {
            circuit,
            counts,
            ticks: 0,
            current_streak: 0,
            longest_streak: 0,
            quiet_streak: 0,
            last_recovery_delay: None,
            recovery_events: 0,
        }
    }

    /// Update atomically: invalid dimensions or counter overflow leave all counters unchanged.
    /// A matching observation has at least `minimum_active` selected bits set.
    pub fn observe(
        &mut self,
        state: &BooleanState,
        minimum_active: usize,
    ) -> Result<bool, RecallError> {
        if minimum_active == 0 {
            return Err(RecallError::ZeroBudget);
        }
        // Validate before modifying counters.
        let matches = self.circuit.recall(state, minimum_active)?;
        let next_ticks = self.ticks.checked_add(1).ok_or(RecallError::ZeroBudget)?;
        let next_streak = if matches {
            self.current_streak
                .checked_add(1)
                .ok_or(RecallError::ZeroBudget)?
        } else {
            0
        };
        let next_quiet = if matches {
            0
        } else {
            self.quiet_streak
                .checked_add(1)
                .ok_or(RecallError::ZeroBudget)?
        };
        let recovered = matches && self.quiet_streak > 0;
        let next_recoveries = if recovered {
            self.recovery_events.checked_add(1).ok_or(RecallError::ZeroBudget)?
        } else {
            self.recovery_events
        };
        let mut increments = Vec::with_capacity(self.counts.len());
        for (position, &member) in self.circuit.members().iter().enumerate() {
            if state.get(member) == Ok(true) {
                self.counts[position]
                    .checked_add(1)
                    .ok_or(RecallError::ZeroBudget)?;
                increments.push(position);
            }
        }
        self.ticks = next_ticks;
        if recovered {
            self.last_recovery_delay = Some(self.quiet_streak);
        }
        self.quiet_streak = next_quiet;
        self.recovery_events = next_recoveries;
        self.current_streak = next_streak;
        self.longest_streak = self.longest_streak.max(next_streak);
        for position in increments {
            self.counts[position] += 1;
        }
        Ok(matches)
    }

    /// Observe the current state of the actual BooleanEngine, without copying a snapshot.
    /// The caller controls when the engine is stimulated or stepped. This is
    /// experimental activation tracking, not a biological firing measurement.
    pub fn observe_engine_state(
        &mut self,
        engine: &crate::engine::BooleanEngine,
        minimum_active: usize,
    ) -> Result<bool, RecallError> {
        self.observe(&engine.active, minimum_active)
    }

    /// Advance a validated synthetic graph and observe each resulting engine state.
    /// The initial state is not observed. At most 1000 steps are permitted, matching
    /// the CLI engine bound. A step failure preserves previously completed steps;
    /// no rollback of engine dynamics is implied. No anatomical edges are created.
    pub fn observe_engine_steps(
        &mut self,
        engine: &mut crate::engine::BooleanEngine,
        graph: &crate::connectome::Connectome,
        minimum_active: usize,
        steps: usize,
    ) -> std::io::Result<Vec<bool>> {
        use std::io::{Error, ErrorKind};
        if steps == 0 || steps > 1000 || minimum_active == 0 {
            return Err(Error::new(ErrorKind::InvalidInput, "invalid recall budget"));
        }
        graph.validate()?;
        if engine.active.len() != graph.ids.len() || self.circuit.universe_len() != graph.ids.len()
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "graph/state size mismatch",
            ));
        }
        // Validate the recall threshold before any engine mutation.
        self.circuit
            .recall(&engine.active, minimum_active)
            .map_err(|_| Error::new(ErrorKind::InvalidInput, "invalid recall state"))?;
        let mut matches = Vec::with_capacity(steps);
        for _ in 0..steps {
            engine.step(graph)?;
            let recalled = self
                .observe_engine_state(engine, minimum_active)
                .map_err(|_| Error::new(ErrorKind::InvalidData, "recall counter overflow"))?;
            matches.push(recalled);
        }
        Ok(matches)
    }

    /// Aggregate streaming activation counts by caller-provided region labels.
    /// Only selected and explicitly annotated members contribute; no anatomical
    /// classification is inferred. u128 prevents overflow when summing counters.
    /// Returns (region_id, observed_activations, selected_members), sorted by ID.
    pub fn regional_activity_totals(
        &self,
        annotations: &[Option<u32>],
    ) -> Result<Vec<(u32, u128, usize)>, RecallError> {
        if annotations.len() != self.circuit.universe_len() {
            return Err(RecallError::LengthMismatch);
        }
        let mut totals = std::collections::BTreeMap::<u32, (u128, usize)>::new();
        for (&member, &count) in self.circuit.members().iter().zip(&self.counts) {
            if let Some(region) = annotations[member] {
                let entry = totals.entry(region).or_default();
                entry.0 += u128::from(count);
                entry.1 += 1;
            }
        }
        Ok(totals
            .into_iter()
            .map(|(region, (activations, members))| (region, activations, members))
            .collect())
    }

    /// Exact regional activation occupancy: (region, active_member_ticks, possible_member_ticks).
    /// The denominator includes all observed ticks for every explicitly annotated
    /// selected member, including inactive observations. No floating-point rounding
    /// or anatomical inference is performed.
    pub fn regional_occupancy(
        &self,
        annotations: &[Option<u32>],
    ) -> Result<Vec<(u32, u128, u128)>, RecallError> {
        let totals = self.regional_activity_totals(annotations)?;
        Ok(totals
            .into_iter()
            .map(|(region, active, members)| {
                (region, active, (self.ticks as u128) * (members as u128))
            })
            .collect())
    }

    /// Build a recallable circuit from the most active explicitly annotated region.
    /// Candidate members must each have at least `minimum_member_ticks` activations.
    /// Region rank uses summed activity, then ascending region ID; members use
    /// descending activity, then ascending index. No graph edges are created.
    pub fn select_region_subcircuit(
        &self,
        annotations: &[Option<u32>],
        minimum_member_ticks: u64,
        maximum_members: usize,
    ) -> Result<Subcircuit, RecallError> {
        if maximum_members == 0 || minimum_member_ticks == 0 {
            return Err(RecallError::ZeroBudget);
        }
        if annotations.len() != self.circuit.universe_len() {
            return Err(RecallError::LengthMismatch);
        }
        let mut regions = std::collections::BTreeMap::<u32, u128>::new();
        for (&member, &count) in self.circuit.members().iter().zip(&self.counts) {
            if count >= minimum_member_ticks {
                if let Some(region) = annotations[member] {
                    *regions.entry(region).or_default() += u128::from(count);
                }
            }
        }
        let selected_region = regions
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
            .map(|(region, _)| region)
            .ok_or(RecallError::EmptyCircuit)?;
        let mut members: Vec<(usize, u64)> = self
            .circuit
            .members()
            .iter()
            .copied()
            .zip(self.counts.iter().copied())
            .filter(|&(member, count)| {
                count >= minimum_member_ticks && annotations[member] == Some(selected_region)
            })
            .collect();
        members.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        members.truncate(maximum_members);
        Subcircuit::new(
            self.circuit.universe_len(),
            members.into_iter().map(|(member, _)| member).collect(),
        )
    }

    /// Select explicitly annotated regions with sufficient observed activation.
    /// Regions are ordered by descending active-member ticks, then region ID.
    /// No biological classification or anatomical edge is inferred.
    pub fn select_active_regions(
        &self,
        annotations: &[Option<u32>],
        minimum_active_ticks: u128,
        minimum_members: usize,
        maximum_regions: usize,
    ) -> Result<Vec<u32>, RecallError> {
        if minimum_members == 0 || maximum_regions == 0 {
            return Err(RecallError::ZeroBudget);
        }
        let mut candidates = self.regional_activity_totals(annotations)?;
        candidates.retain(|(_, active, members)| {
            *active >= minimum_active_ticks && *members >= minimum_members
        });
        candidates.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        candidates.truncate(maximum_regions);
        Ok(candidates
            .into_iter()
            .map(|(region, _, _)| region)
            .collect())
    }

    /// Rank explicitly annotated regions with low historical occupancy for targeted
    /// experimental reactivation. A region is eligible when its exact activity
    /// fraction is at most max_active_percent/100 (inclusive). Zero observations
    /// are rejected: no silence can be inferred without a sampled tick.
    /// Returns region IDs ordered by lowest occupancy, then ascending region ID.
    pub fn select_quiet_regions(
        &self,
        annotations: &[Option<u32>],
        max_active_percent: u8,
        minimum_members: usize,
        maximum_regions: usize,
    ) -> Result<Vec<u32>, RecallError> {
        if minimum_members == 0 || maximum_regions == 0 || max_active_percent > 100 {
            return Err(RecallError::ZeroBudget);
        }
        if self.ticks == 0 {
            return Err(RecallError::ZeroBudget);
        }
        let mut candidates = self.regional_occupancy(annotations)?;
        candidates.retain(|(_, active, possible)| {
            *possible >= (self.ticks as u128) * (minimum_members as u128)
                && *active * 100 <= *possible * u128::from(max_active_percent)
        });
        // Compare rational occupancies without floating point. Each region's
        // denominator is ticks * member_count, so compare per-member activity.
        candidates.sort_unstable_by(|a, b| {
            (a.1 * b.2).cmp(&(b.1 * a.2)).then_with(|| a.0.cmp(&b.0))
        });
        candidates.truncate(maximum_regions);
        Ok(candidates.into_iter().map(|(region, _, _)| region).collect())
    }

    /// Atomically observe one simulated state and select the most active annotated regions.
    /// Invalid annotations or budgets must not consume an observation tick.
    /// The returned match describes the current circuit recall, while region ranking
    /// uses cumulative observed activity including this tick.
    pub fn observe_and_select_regions(
        &mut self,
        state: &BooleanState,
        minimum_active: usize,
        annotations: &[Option<u32>],
        minimum_active_ticks: u128,
        minimum_members: usize,
        maximum_regions: usize,
    ) -> Result<(bool, Vec<u32>), RecallError> {
        if annotations.len() != self.circuit.universe_len() {
            return Err(RecallError::LengthMismatch);
        }
        if minimum_members == 0 || maximum_regions == 0 {
            return Err(RecallError::ZeroBudget);
        }
        let recalled = self.observe(state, minimum_active)?;
        let regions = self.select_active_regions(
            annotations,
            minimum_active_ticks,
            minimum_members,
            maximum_regions,
        )?;
        Ok((recalled, regions))
    }

    /// Recall the most active explicitly annotated region against the live engine state.
    /// Selection uses previously observed ticks; the current state is not consumed.
    /// Returns the selected member indices and whether their recall threshold is met.
    /// All graph connectivity remains unchanged and no anatomical label is inferred.
    pub fn recall_selected_region(
        &self,
        engine: &crate::engine::BooleanEngine,
        annotations: &[Option<u32>],
        minimum_member_ticks: u64,
        maximum_members: usize,
        minimum_active: usize,
    ) -> Result<(Vec<usize>, bool), RecallError> {
        if minimum_active == 0 {
            return Err(RecallError::ZeroBudget);
        }
        let selected =
            self.select_region_subcircuit(annotations, minimum_member_ticks, maximum_members)?;
        let recalled = selected.recall(&engine.active, minimum_active)?;
        Ok((selected.members().to_vec(), recalled))
    }

    /// Consecutive simulated observations without circuit recall.
    #[must_use]
    pub fn quiet_streak(&self) -> u64 {
        self.quiet_streak
    }

    /// Last completed silence-to-recall interval, measured in observation ticks.
    #[must_use]
    pub fn last_recovery_delay(&self) -> Option<u64> {
        self.last_recovery_delay
    }

    /// Number of transitions from a nonempty silent interval to a matching recall.
    #[must_use]
    pub fn recovery_events(&self) -> u64 {
        self.recovery_events
    }

    /// Consume at most `maximum_steps` simulated states, stopping when recall
    /// remains positive for `required_streak` consecutive observations.
    /// Returns the number of observations consumed, or None if not recovered.
    /// This is an experimental temporal criterion, not biological evidence.
    /// Input validation happens before the first observation.
    pub fn observe_until_durable<'a>(
        &mut self,
        states: impl IntoIterator<Item = &'a BooleanState>,
        minimum_active: usize,
        required_streak: u64,
        maximum_steps: usize,
    ) -> Result<Option<usize>, RecallError> {
        if minimum_active == 0 || required_streak == 0 || maximum_steps == 0 || maximum_steps > 1000 {

            return Err(RecallError::ZeroBudget);
        }
        if usize::try_from(required_streak).ok().is_none_or(|n| n > maximum_steps) {
            return Err(RecallError::ZeroBudget);
        }
        if self.circuit.members().len() < minimum_active {
            return Err(RecallError::ZeroBudget);
        }
        for (index, state) in states.into_iter().take(maximum_steps).enumerate() {
            self.observe(state, minimum_active)?;
            if self.current_streak >= required_streak {
                return Ok(Some(index + 1));
            }
        }
        Ok(None)
    }

    /// Measure the first fresh durable recall run in bounded simulated states.
    /// Returns 1-based (first positive tick, confirmation tick), or None.
    /// Prior streaks cannot satisfy this invocation's recovery criterion.
    pub fn measure_recovery_latency<'a>(
        &mut self,
        states: impl IntoIterator<Item = &'a BooleanState>,
        minimum_active: usize,
        required_streak: usize,
        maximum_steps: usize,
    ) -> Result<Option<(usize, usize)>, RecallError> {
        if minimum_active == 0
            || minimum_active > self.circuit.members().len()
            || required_streak == 0
            || maximum_steps == 0
            || maximum_steps > 1000
            || required_streak > maximum_steps
        {
            return Err(RecallError::ZeroBudget);
        }
        let mut consecutive = 0usize;
        for (index, state) in states.into_iter().take(maximum_steps).enumerate() {
            if self.observe(state, minimum_active)? {
                consecutive += 1;
                if consecutive == required_streak {
                    return Ok(Some((index + 2 - consecutive, index + 1)));
                }
            } else {
                consecutive = 0;
            }
        }
        Ok(None)
    }

    #[must_use]
    pub fn ticks(&self) -> u64 {
        self.ticks
    }

    #[must_use]
    pub fn latest_streak(&self) -> u64 {
        self.current_streak
    }

    #[must_use]
    pub fn longest_streak(&self) -> u64 {
        self.longest_streak
    }

    pub fn member_counts(&self) -> impl Iterator<Item = (usize, u64)> + '_ {
        self.circuit
            .members()
            .iter()
            .copied()
            .zip(self.counts.iter().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recalls_selected_region_against_live_engine_without_consuming_ticks() {
        use crate::connectome::Connectome;
        use crate::engine::BooleanEngine;

        // Synthetic chain; region IDs are explicit experimental labels.
        let graph = Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 1, 2, 2],
            targets: vec![1, 2],
            counts: vec![1, 1],
        };
        let mut engine = BooleanEngine::new(&graph, 1).unwrap();
        let mut tracker = OnlineRecall::new(Subcircuit::new(3, vec![0, 1, 2]).unwrap());
        let regions = [Some(7), Some(7), Some(9)];
        engine.stimulate(&graph, 10).unwrap();
        tracker.observe_engine_state(&engine, 1).unwrap();
        engine.step(&graph).unwrap();
        tracker.observe_engine_state(&engine, 1).unwrap();
        assert_eq!(
            tracker.recall_selected_region(&engine, &regions, 1, 2, 1),
            Ok((vec![0, 1], true))
        );
        engine.step(&graph).unwrap();
        assert_eq!(
            tracker.recall_selected_region(&engine, &regions, 1, 2, 1),
            Ok((vec![0, 1], false))
        );
        assert_eq!(tracker.ticks(), 2);
        assert_eq!(
            tracker.recall_selected_region(&engine, &regions[..2], 1, 2, 1),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(
            tracker.recall_selected_region(&engine, &regions, 1, 2, 0),
            Err(RecallError::ZeroBudget)
        );
    }

    #[test]
    fn recovery_latency_tracks_silence_and_reactivation_without_snapshots() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(2, vec![0]).unwrap());
        let quiet = BooleanState::new(2);
        let mut active = BooleanState::new(2);
        active.set(0, true).unwrap();
        assert_eq!(tracker.last_recovery_delay(), None);
        assert_eq!(tracker.observe(&quiet, 1), Ok(false));
        assert_eq!(tracker.observe(&quiet, 1), Ok(false));
        assert_eq!(tracker.quiet_streak(), 2);
        assert_eq!(tracker.observe(&active, 1), Ok(true));
        assert_eq!(tracker.last_recovery_delay(), Some(2));
        assert_eq!(tracker.recovery_events(), 1);
        assert_eq!(tracker.quiet_streak(), 0);
        assert_eq!(tracker.observe(&active, 1), Ok(true));
        assert_eq!(tracker.recovery_events(), 1);
        assert_eq!(tracker.observe(&quiet, 1), Ok(false));
        assert_eq!(tracker.observe(&active, 1), Ok(true));
        assert_eq!(tracker.last_recovery_delay(), Some(1));
        assert_eq!(tracker.recovery_events(), 2);
        assert_eq!(tracker.observe(&quiet, 0), Err(RecallError::ZeroBudget));
        assert_eq!(tracker.recovery_events(), 2);
    }

    #[test]
    fn recovery_latency_requires_new_durable_run() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(2, vec![0]).unwrap());
        let mut active = BooleanState::new(2);
        active.set(0, true).unwrap();
        let quiet = BooleanState::new(2);
        tracker.observe(&active, 1).unwrap();
        tracker.observe(&active, 1).unwrap();
        assert_eq!(tracker.measure_recovery_latency([&quiet, &active, &quiet, &active, &active], 1, 2, 5), Ok(Some((4, 5))));
        assert_eq!(tracker.ticks(), 7);
        assert_eq!(tracker.measure_recovery_latency([&active], 1, 2, 1), Err(RecallError::ZeroBudget));
        assert_eq!(tracker.ticks(), 7);
        assert_eq!(tracker.measure_recovery_latency([&quiet, &active], 1, 2, 2), Ok(None));
        assert_eq!(tracker.ticks(), 9);
    }

    #[test]
    fn durable_recall_stops_early_and_preserves_bounded_observation() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(2, vec![0, 1]).unwrap());
        let mut active = BooleanState::new(2);
        active.set(0, true).unwrap();
        let quiet = BooleanState::new(2);
        let sequence = [&quiet, &active, &quiet, &active, &active, &active, &quiet];
        assert_eq!(tracker.observe_until_durable(sequence, 1, 3, 7), Ok(Some(6)));
        assert_eq!(tracker.ticks(), 6);
        assert_eq!(tracker.latest_streak(), 3);
        assert_eq!(tracker.member_counts().collect::<Vec<_>>(), vec![(0, 4), (1, 0)]);
        assert_eq!(tracker.observe_until_durable([&quiet], 1, 1, 0), Err(RecallError::ZeroBudget));
        assert_eq!(tracker.ticks(), 6);
        assert_eq!(tracker.observe_until_durable([&quiet, &active], 1, 2, 2), Ok(None));
        assert_eq!(tracker.latest_streak(), 1);
    }

    #[test]
    fn streams_without_retaining_history_and_preserves_member_order() {
        let circuit = Subcircuit::new(5, vec![4, 1, 3]).unwrap();
        let mut tracker = OnlineRecall::new(circuit);
        let mut active = BooleanState::new(5);
        active.set(1, true).unwrap();
        active.set(4, true).unwrap();
        let quiet = BooleanState::new(5);
        assert_eq!(tracker.observe(&active, 2), Ok(true));
        assert_eq!(tracker.observe(&active, 2), Ok(true));
        assert_eq!(tracker.observe(&quiet, 2), Ok(false));
        assert_eq!(tracker.observe(&active, 2), Ok(true));
        assert_eq!(tracker.ticks(), 4);
        assert_eq!(tracker.latest_streak(), 1);
        assert_eq!(tracker.longest_streak(), 2);
        assert_eq!(
            tracker.member_counts().collect::<Vec<_>>(),
            vec![(1, 3), (3, 0), (4, 3)]
        );
    }

    #[test]
    fn region_totals_stream_without_anatomical_inference() {
        let circuit = Subcircuit::new(5, vec![0, 1, 3, 4]).unwrap();
        let mut tracker = OnlineRecall::new(circuit);
        let labels = [Some(9), Some(2), Some(2), Some(9), None];
        assert_eq!(
            tracker.regional_activity_totals(&labels),
            Ok(vec![(2, 0, 1), (9, 0, 2)])
        );
        let mut first = BooleanState::new(5);
        first.set(0, true).unwrap();
        first.set(1, true).unwrap();
        first.set(2, true).unwrap();
        tracker.observe(&first, 1).unwrap();
        let mut second = BooleanState::new(5);
        second.set(3, true).unwrap();
        second.set(4, true).unwrap();
        tracker.observe(&second, 1).unwrap();
        assert_eq!(
            tracker.regional_activity_totals(&labels),
            Ok(vec![(2, 1, 1), (9, 2, 2)])
        );
        assert_eq!(
            tracker.regional_activity_totals(&labels[..4]),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(tracker.ticks(), 2);
    }

    #[test]
    fn observes_actual_engine_steps_without_retaining_snapshots() {
        use crate::connectome::Connectome;
        use crate::engine::BooleanEngine;

        // Synthetic directed chain 10 -> 20 -> 30. No biological claims.
        let graph = Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 1, 2, 2],
            targets: vec![1, 2],
            counts: vec![1, 1],
        };
        let mut engine = BooleanEngine::new(&graph, 1).unwrap();
        let mut tracker = OnlineRecall::new(Subcircuit::new(3, vec![0, 1, 2]).unwrap());
        engine.stimulate(&graph, 10).unwrap();
        assert_eq!(tracker.observe_engine_state(&engine, 1), Ok(true));
        assert_eq!(
            tracker.member_counts().collect::<Vec<_>>(),
            vec![(0, 1), (1, 0), (2, 0)]
        );
        engine.step(&graph).unwrap();
        assert_eq!(tracker.observe_engine_state(&engine, 1), Ok(true));
        engine.step(&graph).unwrap();
        assert_eq!(tracker.observe_engine_state(&engine, 1), Ok(true));
        engine.step(&graph).unwrap();
        assert_eq!(tracker.observe_engine_state(&engine, 1), Ok(false));
        assert_eq!(tracker.ticks(), 4);
        assert_eq!(tracker.latest_streak(), 0);
        assert_eq!(tracker.longest_streak(), 3);
        assert_eq!(
            tracker.member_counts().collect::<Vec<_>>(),
            vec![(0, 1), (1, 1), (2, 1)]
        );
        assert_eq!(
            tracker.regional_activity_totals(&[Some(4), Some(4), Some(8)]),
            Ok(vec![(4, 2, 2), (8, 1, 1)])
        );
    }

    #[test]
    fn regional_occupancy_counts_inactive_ticks_and_unannotated_members() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(4, vec![0, 1, 3]).unwrap());
        let labels = [Some(7), Some(7), Some(9), None];
        assert_eq!(tracker.regional_occupancy(&labels), Ok(vec![(7, 0, 0)]));
        let mut first = BooleanState::new(4);
        first.set(0, true).unwrap();
        first.set(3, true).unwrap();
        tracker.observe(&first, 1).unwrap();
        tracker.observe(&BooleanState::new(4), 1).unwrap();
        assert_eq!(tracker.regional_occupancy(&labels), Ok(vec![(7, 1, 4)]));
        assert_eq!(
            tracker.regional_occupancy(&labels[..3]),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(tracker.ticks(), 2);
    }

    #[test]
    fn selects_recallable_region_from_streamed_activity() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(5, vec![0, 1, 2, 4]).unwrap());
        let labels = [Some(7), Some(7), Some(9), None, Some(9)];
        let mut first = BooleanState::new(5);
        first.set(0, true).unwrap();
        first.set(1, true).unwrap();
        first.set(2, true).unwrap();
        tracker.observe(&first, 1).unwrap();
        let mut second = BooleanState::new(5);
        second.set(0, true).unwrap();
        second.set(4, true).unwrap();
        tracker.observe(&second, 1).unwrap();
        let selected = tracker.select_region_subcircuit(&labels, 1, 2).unwrap();
        assert_eq!(selected.members(), &[0, 1]);
        assert_eq!(selected.recall(&first, 2), Ok(true));
        assert_eq!(selected.recall(&second, 2), Ok(false));
        assert_eq!(
            tracker.select_region_subcircuit(&labels, 3, 2),
            Err(RecallError::EmptyCircuit)
        );
        assert_eq!(
            tracker.select_region_subcircuit(&labels[..4], 1, 2),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(
            tracker.select_region_subcircuit(&labels, 1, 0),
            Err(RecallError::ZeroBudget)
        );
    }

    #[test]
    fn selects_annotated_regions_by_streamed_activity_with_bounded_budget() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(4, vec![0, 1, 2, 3]).unwrap());
        let labels = [Some(9), Some(9), Some(2), None];
        let mut state = BooleanState::new(4);
        state.set(0, true).unwrap();
        state.set(2, true).unwrap();
        state.set(3, true).unwrap();
        tracker.observe(&state, 1).unwrap();
        assert_eq!(
            tracker.select_active_regions(&labels, 1, 1, 2),
            Ok(vec![2, 9])
        );
        assert_eq!(tracker.select_active_regions(&labels, 1, 2, 2), Ok(vec![9]));
        assert_eq!(tracker.select_active_regions(&labels, 2, 1, 2), Ok(vec![]));
        assert_eq!(tracker.select_active_regions(&labels, 0, 1, 1), Ok(vec![2]));
        assert_eq!(
            tracker.select_active_regions(&labels, 0, 0, 1),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(
            tracker.select_active_regions(&labels[..3], 0, 1, 1),
            Err(RecallError::LengthMismatch)
        );
    }

    #[test]
    fn observes_and_selects_regions_atomically() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(4, vec![0, 1, 2]).unwrap());
        let labels = [Some(8), Some(8), Some(3), None];
        let mut state = BooleanState::new(4);
        state.set(0, true).unwrap();
        state.set(2, true).unwrap();
        assert_eq!(
            tracker.observe_and_select_regions(&state, 2, &labels, 1, 1, 2),
            Ok((true, vec![3, 8]))
        );
        assert_eq!(tracker.ticks(), 1);
        assert_eq!(
            tracker.observe_and_select_regions(&state, 2, &labels[..3], 1, 1, 2),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(
            tracker.observe_and_select_regions(&state, 2, &labels, 1, 0, 2),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(
            tracker.observe_and_select_regions(&BooleanState::new(3), 2, &labels, 1, 1, 2),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(tracker.ticks(), 1);
        assert_eq!(
            tracker.member_counts().collect::<Vec<_>>(),
            vec![(0, 1), (1, 0), (2, 1)]
        );
        assert_eq!(
            tracker.observe_and_select_regions(&BooleanState::new(4), 2, &labels, 2, 1, 2),
            Ok((false, vec![]))
        );
        assert_eq!(tracker.ticks(), 2);
    }

    #[test]
    fn bounded_engine_steps_observe_propagation_and_reject_bad_budgets() {
        use crate::connectome::Connectome;
        use crate::engine::BooleanEngine;
        let graph = Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 1, 2, 2],
            targets: vec![1, 2],
            counts: vec![1, 1],
        };
        let mut engine = BooleanEngine::new(&graph, 1).unwrap();
        let mut tracker = OnlineRecall::new(Subcircuit::new(3, vec![1, 2]).unwrap());
        engine.stimulate(&graph, 10).unwrap();
        assert!(tracker
            .observe_engine_steps(&mut engine, &graph, 1, 0)
            .is_err());
        assert!(tracker
            .observe_engine_steps(&mut engine, &graph, 1, 1001)
            .is_err());
        assert!(tracker
            .observe_engine_steps(&mut engine, &graph, 0, 1)
            .is_err());
        assert_eq!(tracker.ticks(), 0);
        assert_eq!(
            tracker
                .observe_engine_steps(&mut engine, &graph, 1, 3)
                .unwrap(),
            vec![true, true, false]
        );
        assert_eq!(tracker.ticks(), 3);
        assert_eq!(tracker.longest_streak(), 2);
        assert_eq!(tracker.latest_streak(), 0);
        assert_eq!(
            tracker.member_counts().collect::<Vec<_>>(),
            vec![(1, 1), (2, 1)]
        );
        assert_eq!(
            tracker.regional_occupancy(&[None, Some(7), Some(7)]),
            Ok(vec![(7, 2, 6)])
        );
    }

    #[test]
    fn quiet_region_selection_is_bounded_and_annotation_only() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(5, vec![0, 1, 2, 3, 4]).unwrap());
        let labels = [Some(8), Some(8), Some(3), Some(3), None];
        assert_eq!(tracker.select_quiet_regions(&labels, 50, 1, 2), Err(RecallError::ZeroBudget));
        let mut state = BooleanState::new(5);
        state.set(0, true).unwrap();
        state.set(1, true).unwrap();
        state.set(2, true).unwrap();
        tracker.observe(&state, 1).unwrap();
        tracker.observe(&BooleanState::new(5), 1).unwrap();
        // Region 3: 1/4; region 8: 2/4; unannotated index 4 excluded.
        assert_eq!(tracker.select_quiet_regions(&labels, 50, 2, 2), Ok(vec![3, 8]));
        assert_eq!(tracker.select_quiet_regions(&labels, 25, 2, 2), Ok(vec![3]));
        assert_eq!(tracker.select_quiet_regions(&labels, 0, 2, 2), Ok(vec![]));
        assert_eq!(tracker.select_quiet_regions(&labels, 100, 2, 1), Ok(vec![3]));
        assert_eq!(tracker.select_quiet_regions(&labels[..4], 50, 2, 2), Err(RecallError::LengthMismatch));
        assert_eq!(tracker.select_quiet_regions(&labels, 101, 2, 2), Err(RecallError::ZeroBudget));
        assert_eq!(tracker.ticks(), 2);
    }

    #[test]
    fn rejects_bad_input_without_mutation() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(3, vec![0]).unwrap());
        assert_eq!(
            tracker.observe(&BooleanState::new(2), 1),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(
            tracker.observe(&BooleanState::new(3), 0),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(tracker.ticks(), 0);
        assert_eq!(tracker.member_counts().collect::<Vec<_>>(), vec![(0, 0)]);
    }
}

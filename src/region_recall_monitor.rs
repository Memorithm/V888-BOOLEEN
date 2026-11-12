//! Region labels are explicit experimental annotations, never inferred anatomy.
use crate::BooleanState;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionRecall {
    pub region: u32,
    pub members: usize,
    pub active_member_ticks: u128,
    pub recalled_ticks: u64,
    pub current_streak: u64,
    pub longest_streak: u64,
    /// Last observed tick with confirmed regional recall; None if never recalled.
    pub last_recalled_tick: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionMonitorError {
    InvalidDimensions,
    InvalidBudget,
    Overflow,
}

/// Streaming regional recall without storing state snapshots or graph edges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionRecallMonitor {
    universe: usize,
    members: Vec<(usize, u32)>,
    regions: BTreeMap<u32, RegionRecall>,
    ticks: u64,
}

impl RegionRecallMonitor {
    pub fn new(
        universe: usize,
        selected: &[usize],
        annotations: &[Option<u32>],
    ) -> Result<Self, RegionMonitorError> {
        if annotations.len() != universe || selected.is_empty() {
            return Err(RegionMonitorError::InvalidDimensions);
        }
        let mut seen = BTreeSet::new();
        let mut members = Vec::new();
        let mut regions = BTreeMap::<u32, RegionRecall>::new();
        for &index in selected {
            if index >= universe || !seen.insert(index) {
                return Err(RegionMonitorError::InvalidDimensions);
            }
            if let Some(region) = annotations[index] {
                members.push((index, region));
                let entry = regions.entry(region).or_insert(RegionRecall {
                    region, members: 0, active_member_ticks: 0,
                    recalled_ticks: 0, current_streak: 0, longest_streak: 0,
                    last_recalled_tick: None,
                });
                entry.members += 1;
            }
        }
        if members.is_empty() { return Err(RegionMonitorError::InvalidBudget); }
        Ok(Self { universe, members, regions, ticks: 0 })
    }

    /// Each region recalls if at least minimum_active annotated members fire.
    /// Counter updates are atomic even if a counter overflows.
    pub fn observe(
        &mut self,
        state: &BooleanState,
        minimum_active: usize,
    ) -> Result<(), RegionMonitorError> {
        if state.len() != self.universe {
            return Err(RegionMonitorError::InvalidDimensions);
        }
        if minimum_active == 0 { return Err(RegionMonitorError::InvalidBudget); }
        let next = self.ticks.checked_add(1).ok_or(RegionMonitorError::Overflow)?;
        let mut active = BTreeMap::<u32, usize>::new();
        for &(index, region) in &self.members {
            if state.get(index) == Ok(true) {
                *active.entry(region).or_default() += 1;
            }
        }
        let mut updated = self.regions.clone();
        for (&region, stats) in &mut updated {
            let count = active.get(&region).copied().unwrap_or(0);
            stats.active_member_ticks = stats.active_member_ticks
                .checked_add(count as u128).ok_or(RegionMonitorError::Overflow)?;
            if count >= minimum_active {
                stats.recalled_ticks = stats.recalled_ticks.checked_add(1)
                    .ok_or(RegionMonitorError::Overflow)?;
                stats.last_recalled_tick = Some(next);
                stats.current_streak = stats.current_streak.checked_add(1)
                    .ok_or(RegionMonitorError::Overflow)?;
            } else {
                stats.current_streak = 0;
            }
            stats.longest_streak = stats.longest_streak.max(stats.current_streak);
        }
        self.regions = updated;
        self.ticks = next;
        Ok(())
    }

    /// Select regions with sustained current recall, using only observed ticks.
    /// Results are ordered by longest current streak, then active occupancy,
    /// then explicit region ID. This never changes the graph or the monitor.
    pub fn sustained_recall_regions(
        &self,
        minimum_streak: u64,
        minimum_members: usize,
        maximum_regions: usize,
    ) -> Result<Vec<u32>, RegionMonitorError> {
        if minimum_streak == 0 || minimum_members == 0 || maximum_regions == 0 {
            return Err(RegionMonitorError::InvalidBudget);
        }
        let mut candidates: Vec<&RegionRecall> = self.regions.values()
            .filter(|stats| stats.members >= minimum_members
                && stats.current_streak >= minimum_streak)
            .collect();
        candidates.sort_unstable_by(|a, b| {
            b.current_streak.cmp(&a.current_streak)
                .then_with(|| b.active_member_ticks.cmp(&a.active_member_ticks))
                .then_with(|| a.region.cmp(&b.region))
        });
        candidates.truncate(maximum_regions);
        Ok(candidates.into_iter().map(|stats| stats.region).collect())
    }

    /// Rank explicitly annotated regions by elapsed ticks since confirmed recall.
    /// Never-recalled regions have age equal to all observed ticks. The query is
    /// read-only and cannot infer biological silence or modify connectivity.
    pub fn recall_gap_regions(
        &self,
        minimum_gap: u64,
        minimum_members: usize,
        maximum_regions: usize,
    ) -> Result<Vec<(u32, u64)>, RegionMonitorError> {
        if minimum_gap == 0 || minimum_members == 0 || maximum_regions == 0 {
            return Err(RegionMonitorError::InvalidBudget);
        }
        let mut candidates: Vec<(u32, u64)> = self.regions.values()
            .filter(|stats| stats.members >= minimum_members)
            .filter_map(|stats| {
                let gap = self.ticks - stats.last_recalled_tick.unwrap_or(0);
                (gap >= minimum_gap).then_some((stats.region, gap))
            })
            .collect();
        candidates.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        candidates.truncate(maximum_regions);
        Ok(candidates)
    }

    /// Return actual selected neuron indices for the most overdue annotated regions.
    /// The candidate set is frozen by caller-supplied annotations and past observations.
    /// No edges are added and no future observations are consulted.
    pub fn recall_gap_subcircuit_members(
        &self,
        minimum_gap: u64,
        minimum_members: usize,
        maximum_regions: usize,
        maximum_members: usize,
    ) -> Result<Vec<usize>, RegionMonitorError> {
        if maximum_members == 0 {
            return Err(RegionMonitorError::InvalidBudget);
        }
        let regions = self.recall_gap_regions(minimum_gap, minimum_members, maximum_regions)?;
        let mut members = Vec::new();
        for (region, _) in regions {
            let mut indices: Vec<usize> = self.members.iter()
                .filter_map(|&(index, label)| (label == region).then_some(index))
                .collect();
            indices.sort_unstable();
            for index in indices {
                if members.len() == maximum_members {
                    return Ok(members);
                }
                members.push(index);
            }
        }
        Ok(members)
    }

    /// Allocate a bounded recovery candidate set fairly across overdue regions.
    /// One annotated member per region is selected per round, preventing a large
    /// region from exhausting the entire budget before smaller regions are seen.
    /// The region priority is fixed by observed recall gaps, never by anatomy.
    pub fn interleaved_recall_gap_members(
        &self,
        minimum_gap: u64,
        minimum_members: usize,
        maximum_regions: usize,
        maximum_members: usize,
    ) -> Result<Vec<usize>, RegionMonitorError> {
        if maximum_members == 0 {
            return Err(RegionMonitorError::InvalidBudget);
        }
        let regions = self.recall_gap_regions(minimum_gap, minimum_members, maximum_regions)?;
        let mut groups: Vec<Vec<usize>> = regions
            .iter()
            .map(|&(region, _)| {
                let mut indices: Vec<usize> = self.members.iter()
                    .filter_map(|&(index, label)| (label == region).then_some(index))
                    .collect();
                indices.sort_unstable();
                indices
            })
            .collect();
        let mut result = Vec::new();
        let mut round = 0;
        loop {
            let mut advanced = false;
            for group in &mut groups {
                if let Some(&index) = group.get(round) {
                    result.push(index);
                    advanced = true;
                    if result.len() == maximum_members {
                        return Ok(result);
                    }
                }
            }
            if !advanced {
                return Ok(result);
            }
            round += 1;
        }
    }

    /// Materialize a bit-packed recovery seed from the interleaved regional policy.
    /// The returned state is an experimental stimulation proposal, not an
    /// anatomical observation; it never mutates the monitor or graph.
    pub fn interleaved_recall_gap_mask(
        &self,
        minimum_gap: u64,
        minimum_members: usize,
        maximum_regions: usize,
        maximum_members: usize,
    ) -> Result<BooleanState, RegionMonitorError> {
        let indices = self.interleaved_recall_gap_members(
            minimum_gap, minimum_members, maximum_regions, maximum_members,
        )?;
        let mut mask = BooleanState::new(self.universe);
        for index in indices {
            mask.set(index, true)
                .map_err(|_| RegionMonitorError::InvalidDimensions)?;
        }
        Ok(mask)
    }

    /// Apply a bounded, annotation-only recovery proposal to an existing state.
    /// Already-active bits are preserved; the monitor and graph are unchanged.
    /// Returns the number of newly activated members, not the number proposed.
    pub fn stimulate_interleaved_recall_gap(
        &self,
        state: &BooleanState,
        minimum_gap: u64,
        minimum_members: usize,
        maximum_regions: usize,
        maximum_members: usize,
    ) -> Result<(BooleanState, usize), RegionMonitorError> {
        if state.len() != self.universe {
            return Err(RegionMonitorError::InvalidDimensions);
        }
        let candidates = self.interleaved_recall_gap_members(
            minimum_gap, minimum_members, maximum_regions, maximum_members,
        )?;
        let mut next = state.clone();
        let mut activated = 0;
        for index in candidates {
            let was_active = next.get(index)
                .map_err(|_| RegionMonitorError::InvalidDimensions)?;
            if !was_active {
                next.set(index, true)
                    .map_err(|_| RegionMonitorError::InvalidDimensions)?;
                activated += 1;
            }
        }
        Ok((next, activated))
    }

    /// Stimulate only inactive members of overdue explicitly annotated regions.
    /// Existing active members never consume the new-activation budget.
    /// This is a hypothetical recovery policy, not inferred anatomy.
    pub fn stimulate_inactive_recall_gap(
        &self,
        state: &BooleanState,
        minimum_gap: u64,
        minimum_members: usize,
        maximum_regions: usize,
        maximum_new_activations: usize,
    ) -> Result<(BooleanState, usize), RegionMonitorError> {
        if state.len() != self.universe {
            return Err(RegionMonitorError::InvalidDimensions);
        }
        if maximum_new_activations == 0 {
            return Err(RegionMonitorError::InvalidBudget);
        }
        let regions = self.recall_gap_regions(minimum_gap, minimum_members, maximum_regions)?;
        let groups: Vec<Vec<usize>> = regions.iter().map(|&(region, _)| {
            let mut indices: Vec<usize> = self.members.iter()
                .filter_map(|&(index, label)| (label == region).then_some(index))
                .collect();
            indices.sort_unstable();
            indices
        }).collect();
        let mut positions = vec![0usize; groups.len()];
        let mut next = state.clone();
        let mut activated = 0;
        loop {
            let mut advanced = false;
            for (group, position) in groups.iter().zip(positions.iter_mut()) {
                while *position < group.len() && next.get(group[*position]) == Ok(true) {
                    *position += 1;
                }
                if let Some(&index) = group.get(*position) {
                    next.set(index, true)
                        .map_err(|_| RegionMonitorError::InvalidDimensions)?;
                    *position += 1;
                    activated += 1;
                    advanced = true;
                    if activated == maximum_new_activations {
                        return Ok((next, activated));
                    }
                }
            }
            if !advanced {
                return Ok((next, activated));
            }
        }
    }

    /// Run bounded hypothetical recovery with caller-provided graph propagation.
    /// The callback owns the transition rule; this method never edits graph edges.
    /// A successful recall requires an observed threshold crossing after a step.
    /// Returns (final state, propagation steps used, confirmed region IDs).
    pub fn recover_with_propagation<F>(
        &self,
        state: &BooleanState,
        minimum_gap: u64,
        minimum_members: usize,
        maximum_regions: usize,
        maximum_new_activations: usize,
        maximum_steps: usize,
        minimum_active_for_recall: usize,
        mut propagate: F,
    ) -> Result<(BooleanState, usize, Vec<u32>), RegionMonitorError>
    where
        F: FnMut(&BooleanState) -> Result<BooleanState, RegionMonitorError>,
    {
        if state.len() != self.universe {
            return Err(RegionMonitorError::InvalidDimensions);
        }
        if maximum_steps == 0 || minimum_active_for_recall == 0 {
            return Err(RegionMonitorError::InvalidBudget);
        }
        let target_regions = self.recall_gap_regions(
            minimum_gap, minimum_members, maximum_regions,
        )?;
        let (mut current, _) = self.stimulate_inactive_recall_gap(
            state, minimum_gap, minimum_members, maximum_regions,
            maximum_new_activations,
        )?;
        let mut observer = self.clone();
        for step in 1..=maximum_steps {
            let next = propagate(&current)?;
            if next.len() != self.universe {
                return Err(RegionMonitorError::InvalidDimensions);
            }
            observer.observe(&next, minimum_active_for_recall)?;
            current = next;
            let confirmed: Vec<u32> = target_regions
                .iter()
                .filter_map(|&(region, _)| {
                    observer.regions.get(&region).and_then(|stats| {
                        (stats.last_recalled_tick == Some(observer.ticks))
                            .then_some(region)
                    })
                })
                .collect();
            if !confirmed.is_empty() {
                return Ok((current, step, confirmed));
            }
        }
        Ok((current, maximum_steps, Vec::new()))
    }

    /// Confirm durable recall only after consecutive post-propagation observations.
    /// A single transient threshold crossing is not counted as recovery.
    /// Stimulation is hypothetical; the callback supplies the transition rule.
    /// Returns the final state, executed steps and durably recovered region IDs.
    pub fn recover_durable_with_propagation<F>(
        &self,
        state: &BooleanState,
        minimum_gap: u64,
        minimum_members: usize,
        maximum_regions: usize,
        maximum_new_activations: usize,
        maximum_steps: usize,
        minimum_active_for_recall: usize,
        required_consecutive_ticks: u64,
        mut propagate: F,
    ) -> Result<(BooleanState, usize, Vec<u32>), RegionMonitorError>
    where
        F: FnMut(&BooleanState) -> Result<BooleanState, RegionMonitorError>,
    {
        if state.len() != self.universe {
            return Err(RegionMonitorError::InvalidDimensions);
        }
        if maximum_steps == 0 || minimum_active_for_recall == 0
            || required_consecutive_ticks == 0
        {
            return Err(RegionMonitorError::InvalidBudget);
        }
        let targets = self.recall_gap_regions(
            minimum_gap, minimum_members, maximum_regions,
        )?;
        let (mut current, _) = self.stimulate_inactive_recall_gap(
            state, minimum_gap, minimum_members, maximum_regions,
            maximum_new_activations,
        )?;
        let mut observer = self.clone();
        for step in 1..=maximum_steps {
            let next = propagate(&current)?;
            if next.len() != self.universe {
                return Err(RegionMonitorError::InvalidDimensions);
            }
            observer.observe(&next, minimum_active_for_recall)?;
            current = next;
            let confirmed: Vec<u32> = targets.iter().filter_map(|&(region, _)| {
                observer.regions.get(&region).and_then(|stats| {
                    (stats.current_streak >= required_consecutive_ticks)
                        .then_some(region)
                })
            }).collect();
            if !confirmed.is_empty() {
                return Ok((current, step, confirmed));
            }
        }
        Ok((current, maximum_steps, Vec::new()))
    }

    /// Evaluate first and durable regional recall on a bounded, held-out state trace.
    /// The trace is observational: no stimulation, graph edits or monitor mutation.
    /// Durability requires consecutive observations, not a single transient spike.
    pub fn recall_trace_latency(
        &self,
        states: &[BooleanState],
        minimum_active: usize,
        durable_ticks: usize,
    ) -> Result<Vec<(u32, Option<usize>, Option<usize>)>, RegionMonitorError> {
        if states.is_empty() || states.len() > 1000
            || minimum_active == 0 || durable_ticks == 0
        {
            return Err(RegionMonitorError::InvalidBudget);
        }
        if states.iter().any(|state| state.len() != self.universe) {
            return Err(RegionMonitorError::InvalidDimensions);
        }
        let mut output = Vec::with_capacity(self.regions.len());
        for &region in self.regions.keys() {
            let mut first = None;
            let mut durable = None;
            let mut streak = 0usize;
            for (tick, state) in states.iter().enumerate() {
                let active = self.members.iter()
                    .filter(|&&(index, label)| label == region && state.get(index) == Ok(true))
                    .count();
                if active >= minimum_active {
                    if first.is_none() { first = Some(tick + 1); }
                    streak += 1;
                    if streak >= durable_ticks && durable.is_none() {
                        durable = Some(tick + 1);
                    }
                } else {
                    streak = 0;
                }
            }
            output.push((region, first, durable));
        }
        Ok(output)
    }

    /// Compare two equal-horizon observational traces on identical explicit regions.
    /// Returns per-region (id, baseline_first, candidate_first, baseline_durable,
    /// candidate_durable), with one-based confirmation ticks. None is censored.
    /// This is descriptive, not evidence of a causal stimulation benefit.
    pub fn compare_recall_traces(
        &self,
        baseline: &[BooleanState],
        candidate: &[BooleanState],
        minimum_active: usize,
        durable_ticks: usize,
    ) -> Result<Vec<(u32, Option<usize>, Option<usize>, Option<usize>, Option<usize>)>, RegionMonitorError> {
        if baseline.len() != candidate.len() {
            return Err(RegionMonitorError::InvalidDimensions);
        }
        let a = self.recall_trace_latency(baseline, minimum_active, durable_ticks)?;
        let b = self.recall_trace_latency(candidate, minimum_active, durable_ticks)?;
        Ok(a.into_iter().zip(b).map(|((region, first_a, durable_a), (other, first_b, durable_b))| {
            debug_assert_eq!(region, other);
            (region, first_a, first_b, durable_a, durable_b)
        }).collect())
    }

    pub fn ticks(&self) -> u64 { self.ticks }

    pub fn regions(&self) -> impl Iterator<Item = &RegionRecall> {
        self.regions.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_regions_track_streaks_and_occupancy() {
        let mut monitor = RegionRecallMonitor::new(
            5, &[0, 1, 2, 3, 4],
            &[Some(7), Some(7), Some(9), Some(9), None],
        ).unwrap();
        let mut state = BooleanState::new(5);
        for i in [0, 1, 2, 4] { state.set(i, true).unwrap(); }
        monitor.observe(&state, 2).unwrap();
        monitor.observe(&state, 2).unwrap();
        state.set(1, false).unwrap();
        state.set(3, true).unwrap();
        monitor.observe(&state, 2).unwrap();
        let stats: Vec<_> = monitor.regions().collect();
        assert_eq!((stats[0].active_member_ticks, stats[0].recalled_ticks,
            stats[0].longest_streak, stats[0].current_streak), (5, 2, 2, 0));
        assert_eq!((stats[1].active_member_ticks, stats[1].recalled_ticks,
            stats[1].longest_streak), (4, 1, 1));
        assert_eq!(monitor.ticks(), 3);
    }

    #[test]
    fn sustained_regions_use_current_streak_and_stable_ties() {
        let mut monitor = RegionRecallMonitor::new(
            4, &[0, 1, 2, 3], &[Some(7), Some(7), Some(9), Some(9)],
        ).unwrap();
        let mut state = BooleanState::new(4);
        for i in 0..4 { state.set(i, true).unwrap(); }
        monitor.observe(&state, 2).unwrap();
        monitor.observe(&state, 2).unwrap();
        assert_eq!(monitor.sustained_recall_regions(2, 2, 2).unwrap(), vec![7, 9]);
        state.set(1, false).unwrap();
        monitor.observe(&state, 2).unwrap();
        assert_eq!(monitor.sustained_recall_regions(2, 2, 2).unwrap(), vec![9]);
        assert_eq!(monitor.sustained_recall_regions(1, 3, 2).unwrap(), vec![]);
        assert_eq!(monitor.sustained_recall_regions(0, 1, 1),
            Err(RegionMonitorError::InvalidBudget));
        assert_eq!(monitor.ticks(), 3);
    }

    #[test]
    fn recall_gap_tracks_last_confirmation_and_never_recalled_regions() {
        let mut monitor = RegionRecallMonitor::new(
            4, &[0, 1, 2, 3], &[Some(7), Some(7), Some(9), Some(9)],
        ).unwrap();
        let mut state = BooleanState::new(4);
        state.set(0, true).unwrap();
        state.set(1, true).unwrap();
        monitor.observe(&state, 2).unwrap();
        assert_eq!(monitor.recall_gap_regions(1, 2, 2).unwrap(), vec![(9, 1)]);
        state.set(0, false).unwrap();
        state.set(1, false).unwrap();
        monitor.observe(&state, 2).unwrap();
        monitor.observe(&state, 2).unwrap();
        assert_eq!(monitor.recall_gap_regions(1, 2, 2).unwrap(), vec![(9, 3), (7, 2)]);
        assert_eq!(monitor.recall_gap_regions(2, 2, 1).unwrap(), vec![(9, 3)]);
        assert_eq!(monitor.recall_gap_regions(0, 2, 2), Err(RegionMonitorError::InvalidBudget));
        assert_eq!(monitor.ticks(), 3);
    }

    #[test]
    fn gap_subcircuit_members_are_bounded_and_annotated() {
        let mut monitor = RegionRecallMonitor::new(
            6, &[5, 3, 1, 4, 0],
            &[Some(7), Some(9), None, Some(7), Some(9), Some(7)],
        ).unwrap();
        let mut state = BooleanState::new(6);
        for index in [0, 3, 5] { state.set(index, true).unwrap(); }
        monitor.observe(&state, 2).unwrap();
        state.set(0, false).unwrap();
        state.set(3, false).unwrap();
        state.set(5, false).unwrap();
        monitor.observe(&state, 2).unwrap();
        assert_eq!(monitor.recall_gap_subcircuit_members(1, 2, 2, 4).unwrap(),
            vec![1, 4, 0, 3]);
        assert_eq!(monitor.recall_gap_subcircuit_members(1, 2, 2, 2).unwrap(),
            vec![1, 4]);
        assert_eq!(monitor.recall_gap_subcircuit_members(3, 2, 2, 4).unwrap(),
            Vec::<usize>::new());
        assert_eq!(monitor.recall_gap_subcircuit_members(1, 2, 2, 0),
            Err(RegionMonitorError::InvalidBudget));
        assert_eq!(monitor.ticks(), 2);
    }

    #[test]
    fn interleaved_recovery_prevents_large_region_starvation() {
        let mut monitor = RegionRecallMonitor::new(
            6, &[5, 4, 3, 2, 1, 0],
            &[Some(7), Some(7), Some(7), Some(7), Some(9), Some(9)],
        ).unwrap();
        let state = BooleanState::new(6);
        monitor.observe(&state, 2).unwrap();
        assert_eq!(monitor.recall_gap_subcircuit_members(1, 2, 2, 3).unwrap(),
            vec![0, 1, 2]);
        assert_eq!(monitor.interleaved_recall_gap_members(1, 2, 2, 3).unwrap(),
            vec![0, 4, 1]);
        assert_eq!(monitor.interleaved_recall_gap_members(1, 2, 2, 6).unwrap(),
            vec![0, 4, 1, 5, 2, 3]);
        assert_eq!(monitor.interleaved_recall_gap_members(2, 2, 2, 3).unwrap(),
            Vec::<usize>::new());
        assert_eq!(monitor.interleaved_recall_gap_members(1, 2, 2, 0),
            Err(RegionMonitorError::InvalidBudget));
    }

    #[test]
    fn interleaved_recovery_mask_is_bit_packed_and_read_only() {
        let mut monitor = RegionRecallMonitor::new(
            6, &[5, 4, 3, 2, 1, 0],
            &[Some(7), Some(7), Some(7), Some(7), Some(9), Some(9)],
        ).unwrap();
        monitor.observe(&BooleanState::new(6), 2).unwrap();
        let before = monitor.clone();
        let mask = monitor.interleaved_recall_gap_mask(1, 2, 2, 3).unwrap();
        assert_eq!(mask.len(), 6);
        for index in 0..6 {
            assert_eq!(mask.get(index), Ok([0, 1, 4].contains(&index)));
        }
        let empty = monitor.interleaved_recall_gap_mask(2, 2, 2, 3).unwrap();
        for index in 0..6 {
            assert_eq!(empty.get(index), Ok(false));
        }
        assert_eq!(
            monitor.interleaved_recall_gap_mask(1, 2, 2, 0),
            Err(RegionMonitorError::InvalidBudget),
        );
        assert_eq!(monitor, before);
    }

    #[test]
    fn stimulation_preserves_existing_bits_and_counts_new_activations() {
        let mut monitor = RegionRecallMonitor::new(
            6, &[0, 1, 2, 3, 4, 5],
            &[Some(7), Some(7), Some(7), Some(7), Some(9), Some(9)],
        ).unwrap();
        monitor.observe(&BooleanState::new(6), 2).unwrap();
        let before = monitor.clone();
        let mut input = BooleanState::new(6);
        input.set(0, true).unwrap();
        input.set(5, true).unwrap();
        let (output, newly_activated) = monitor
            .stimulate_interleaved_recall_gap(&input, 1, 2, 2, 3).unwrap();
        assert_eq!(newly_activated, 2);
        for index in 0..6 {
            assert_eq!(output.get(index), Ok([0, 1, 4, 5].contains(&index)));
        }
        assert_eq!(input.get(1), Ok(false));
        assert_eq!(monitor, before);
        assert_eq!(
            monitor.stimulate_interleaved_recall_gap(&BooleanState::new(5), 1, 2, 2, 3),
            Err(RegionMonitorError::InvalidDimensions),
        );
        assert_eq!(
            monitor.stimulate_interleaved_recall_gap(&input, 1, 2, 2, 0),
            Err(RegionMonitorError::InvalidBudget),
        );
    }

    #[test]
    fn inactive_budget_skips_active_members_and_preserves_history() {
        let mut monitor = RegionRecallMonitor::new(
            6, &[0, 1, 2, 3, 4, 5],
            &[Some(7), Some(7), Some(7), Some(7), Some(9), Some(9)],
        ).unwrap();
        monitor.observe(&BooleanState::new(6), 2).unwrap();
        let before = monitor.clone();
        let mut state = BooleanState::new(6);
        state.set(0, true).unwrap();
        state.set(4, true).unwrap();
        let (old, old_count) = monitor
            .stimulate_interleaved_recall_gap(&state, 1, 2, 2, 2).unwrap();
        assert_eq!(old_count, 0);
        assert_eq!(old, state);
        let (new, count) = monitor
            .stimulate_inactive_recall_gap(&state, 1, 2, 2, 2).unwrap();
        assert_eq!(count, 2);
        for index in 0..6 {
            assert_eq!(new.get(index), Ok([0, 1, 4, 5].contains(&index)));
        }
        assert_eq!(monitor, before);
        assert_eq!(state.get(1), Ok(false));
        assert_eq!(
            monitor.stimulate_inactive_recall_gap(&BooleanState::new(5), 1, 2, 2, 2),
            Err(RegionMonitorError::InvalidDimensions),
        );
        assert_eq!(
            monitor.stimulate_inactive_recall_gap(&state, 1, 2, 2, 0),
            Err(RegionMonitorError::InvalidBudget),
        );
    }

    #[test]
    fn bounded_recovery_confirms_only_post_propagation_recall() {
        let mut monitor = RegionRecallMonitor::new(
            4, &[0, 1, 2, 3],
            &[Some(7), Some(7), Some(9), Some(9)],
        ).unwrap();
        monitor.observe(&BooleanState::new(4), 2).unwrap();
        let before = monitor.clone();
        let state = BooleanState::new(4);
        let (recovered, steps, regions) = monitor.recover_with_propagation(
            &state, 1, 2, 2, 2, 3, 1,
            |input| Ok(input.clone()),
        ).unwrap();
        assert_eq!(steps, 1);
        assert_eq!(regions, vec![7, 9]);
        assert_eq!(recovered.get(0), Ok(true));
        assert_eq!(recovered.get(2), Ok(true));
        assert_eq!(monitor, before);

        let (silent, steps, regions) = monitor.recover_with_propagation(
            &state, 1, 2, 2, 2, 2, 1,
            |_| Ok(BooleanState::new(4)),
        ).unwrap();
        assert_eq!(steps, 2);
        assert!(regions.is_empty());
        assert_eq!(silent, state);
        assert_eq!(monitor, before);
        assert_eq!(
            monitor.recover_with_propagation(
                &state, 1, 2, 2, 2, 0, 1,
                |input| Ok(input.clone()),
            ),
            Err(RegionMonitorError::InvalidBudget),
        );
        assert_eq!(
            monitor.recover_with_propagation(
                &state, 1, 2, 2, 2, 1, 1,
                |_| Ok(BooleanState::new(3)),
            ),
            Err(RegionMonitorError::InvalidDimensions),
        );
    }

    #[test]
    fn durable_recovery_rejects_transient_recall_and_confirms_streak() {
        let mut monitor = RegionRecallMonitor::new(
            2, &[0, 1], &[Some(7), Some(7)],
        ).unwrap();
        monitor.observe(&BooleanState::new(2), 2).unwrap();
        let before = monitor.clone();
        let input = BooleanState::new(2);
        let mut tick = 0;
        let (_, steps, confirmed) = monitor.recover_durable_with_propagation(
            &input, 1, 2, 1, 2, 3, 2, 2,
            |_| {
                tick += 1;
                let mut next = BooleanState::new(2);
                if tick != 2 {
                    next.set(0, true).unwrap();
                    next.set(1, true).unwrap();
                }
                Ok(next)
            },
        ).unwrap();
        assert_eq!(steps, 3);
        assert!(confirmed.is_empty());
        let (_, steps, confirmed) = monitor.recover_durable_with_propagation(
            &input, 1, 2, 1, 2, 3, 2, 2,
            |state| Ok(state.clone()),
        ).unwrap();
        assert_eq!(steps, 2);
        assert_eq!(confirmed, vec![7]);
        assert_eq!(monitor, before);
        assert_eq!(
            monitor.recover_durable_with_propagation(
                &input, 1, 2, 1, 2, 3, 2, 0,
                |state| Ok(state.clone()),
            ),
            Err(RegionMonitorError::InvalidBudget),
        );
    }

    #[test]
    fn trace_latency_separates_transient_and_durable_recall() {
        let monitor = RegionRecallMonitor::new(
            4, &[0, 1, 2, 3], &[Some(7), Some(7), Some(9), Some(9)],
        ).unwrap();
        let before = monitor.clone();
        let mut states = vec![BooleanState::new(4); 4];
        for tick in [0, 2, 3] {
            for index in [0, 1] { states[tick].set(index, true).unwrap(); }
        }
        for tick in [1, 2, 3] {
            for index in [2, 3] { states[tick].set(index, true).unwrap(); }
        }
        assert_eq!(monitor.recall_trace_latency(&states, 2, 2).unwrap(),
            vec![(7, Some(1), Some(4)), (9, Some(2), Some(3))]);
        assert_eq!(monitor.recall_trace_latency(&states[..1], 2, 2).unwrap(),
            vec![(7, Some(1), None), (9, None, None)]);
        assert_eq!(monitor.recall_trace_latency(&[], 2, 2),
            Err(RegionMonitorError::InvalidBudget));
        assert_eq!(monitor.recall_trace_latency(&states, 2, 0),
            Err(RegionMonitorError::InvalidBudget));
        assert_eq!(monitor.recall_trace_latency(&[BooleanState::new(3)], 2, 1),
            Err(RegionMonitorError::InvalidDimensions));
        assert_eq!(monitor, before);
    }

    #[test]
    fn paired_traces_preserve_censoring_and_equal_budgets() {
        let monitor = RegionRecallMonitor::new(
            4, &[0, 1, 2, 3], &[Some(7), Some(7), Some(9), Some(9)],
        ).unwrap();
        let before = monitor.clone();
        let baseline = vec![BooleanState::new(4); 3];
        let mut candidate = baseline.clone();
        for tick in [1, 2] {
            candidate[tick].set(0, true).unwrap();
            candidate[tick].set(1, true).unwrap();
        }
        assert_eq!(
            monitor.compare_recall_traces(&baseline, &candidate, 2, 2).unwrap(),
            vec![(7, None, Some(2), None, Some(3)), (9, None, None, None, None)],
        );
        assert_eq!(
            monitor.compare_recall_traces(&baseline[..2], &candidate, 2, 2),
            Err(RegionMonitorError::InvalidDimensions),
        );
        assert_eq!(
            monitor.compare_recall_traces(&baseline, &candidate, 0, 2),
            Err(RegionMonitorError::InvalidBudget),
        );
        assert_eq!(monitor, before);
    }

    #[test]
    fn invalid_observations_preserve_counters() {
        let mut monitor = RegionRecallMonitor::new(
            2, &[0], &[Some(1), None],
        ).unwrap();
        let before = monitor.clone();
        assert_eq!(monitor.observe(&BooleanState::new(2), 0),
            Err(RegionMonitorError::InvalidBudget));
        assert_eq!(monitor.observe(&BooleanState::new(3), 1),
            Err(RegionMonitorError::InvalidDimensions));
        assert_eq!(monitor, before);
    }
}

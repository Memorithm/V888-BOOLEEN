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

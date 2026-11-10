//! Deterministic, bounded recall of explicitly indexed experimental subcircuits.
//! Membership is supplied by the caller; no anatomical connections are inferred.
use crate::BooleanState;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecallError {
    LengthMismatch,
    OutOfBounds,
    DuplicateMember,
    EmptyCircuit,
    ZeroBudget,
    EmptyHistory,
}

/// An explicitly selected subset of neuron indices, not a graph or anatomical claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Subcircuit {
    members: Vec<usize>,
    universe_len: usize,
}

impl Subcircuit {
    pub fn new(universe_len: usize, mut members: Vec<usize>) -> Result<Self, RecallError> {
        if members.is_empty() {
            return Err(RecallError::EmptyCircuit);
        }
        if members.iter().any(|&index| index >= universe_len) {
            return Err(RecallError::OutOfBounds);
        }
        members.sort_unstable();
        if members.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(RecallError::DuplicateMember);
        }
        Ok(Self {
            members,
            universe_len,
        })
    }

    #[must_use]
    pub fn members(&self) -> &[usize] {
        &self.members
    }

    /// Recall a circuit when at least `minimum_active` of its members are active.
    /// The caller controls the activation threshold; no learning is implied.
    pub fn recall(&self, state: &BooleanState, minimum_active: usize) -> Result<bool, RecallError> {
        if state.len() != self.universe_len {
            return Err(RecallError::LengthMismatch);
        }
        if minimum_active == 0 {
            return Err(RecallError::ZeroBudget);
        }
        if minimum_active > self.members.len() {
            return Ok(false);
        }
        let mut active = 0;
        for &index in &self.members {
            if state.get(index).expect("validated circuit index") {
                active += 1;
                if active >= minimum_active {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Detect repeated circuit activation across caller-supplied simulation snapshots.
    /// Each snapshot is evaluated independently; no anatomical or temporal edges are inferred.
    /// A matching snapshot contains at least `minimum_active` active members.
    /// At least `minimum_matching_steps` snapshots must match.
    pub fn recall_over_time(
        &self,
        snapshots: &[BooleanState],
        minimum_active: usize,
        minimum_matching_steps: usize,
    ) -> Result<bool, RecallError> {
        if snapshots.is_empty() {
            return Err(RecallError::EmptyHistory);
        }
        if minimum_matching_steps == 0 || minimum_active == 0 {
            return Err(RecallError::ZeroBudget);
        }
        if snapshots
            .iter()
            .any(|state| state.len() != self.universe_len)
        {
            return Err(RecallError::LengthMismatch);
        }
        if minimum_matching_steps > snapshots.len() || minimum_active > self.members.len() {
            return Ok(false);
        }
        let mut matches = 0usize;
        for snapshot in snapshots {
            if self.recall(snapshot, minimum_active)? {
                matches += 1;
                if matches >= minimum_matching_steps {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Recall only when the latest `required_consecutive_steps` snapshots all match.
    /// Unlike `recall_over_time`, earlier intermittent matches cannot satisfy this rule.
    /// The caller supplies snapshots in chronological order, oldest first.
    pub fn recall_latest_streak(
        &self,
        snapshots: &[BooleanState],
        minimum_active: usize,
        required_consecutive_steps: usize,
    ) -> Result<bool, RecallError> {
        if snapshots.is_empty() {
            return Err(RecallError::EmptyHistory);
        }
        if minimum_active == 0 || required_consecutive_steps == 0 {
            return Err(RecallError::ZeroBudget);
        }
        if snapshots
            .iter()
            .any(|state| state.len() != self.universe_len)
        {
            return Err(RecallError::LengthMismatch);
        }
        if required_consecutive_steps > snapshots.len() || minimum_active > self.members.len() {
            return Ok(false);
        }
        for snapshot in snapshots.iter().rev().take(required_consecutive_steps) {
            if !self.recall(snapshot, minimum_active)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Select up to `maximum_members` frequently active indices from observed snapshots.
    /// Ties are resolved by ascending index. Only caller-supplied activity is used:
    /// this does not learn or infer anatomical edges.
    pub fn from_frequent_activity(
        snapshots: &[BooleanState],
        minimum_observations: usize,
        maximum_members: usize,
    ) -> Result<Self, RecallError> {
        if snapshots.is_empty() {
            return Err(RecallError::EmptyHistory);
        }
        if minimum_observations == 0 || maximum_members == 0 {
            return Err(RecallError::ZeroBudget);
        }
        let universe_len = snapshots[0].len();
        if snapshots.iter().any(|state| state.len() != universe_len) {
            return Err(RecallError::LengthMismatch);
        }
        let mut counts = vec![0usize; universe_len];
        for state in snapshots {
            for index in state.active_indices() {
                counts[index] += 1;
            }
        }
        let mut candidates: Vec<(usize, usize)> = counts
            .into_iter()
            .enumerate()
            .filter_map(|(index, count)| (count >= minimum_observations).then_some((index, count)))
            .collect();
        candidates.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        candidates.truncate(maximum_members);
        if candidates.is_empty() {
            return Err(RecallError::EmptyCircuit);
        }
        Self::new(
            universe_len,
            candidates.into_iter().map(|(index, _)| index).collect(),
        )
    }

    /// Count observed activations for each explicitly selected member, in sorted index order.
    /// Counts are empirical snapshot frequencies, not learned weights or anatomy.
    /// Memory is bounded by circuit membership, independently of history length.
    pub fn member_activation_counts(
        &self,
        snapshots: &[BooleanState],
    ) -> Result<Vec<(usize, usize)>, RecallError> {
        if snapshots.is_empty() {
            return Err(RecallError::EmptyHistory);
        }
        if snapshots
            .iter()
            .any(|state| state.len() != self.universe_len)
        {
            return Err(RecallError::LengthMismatch);
        }
        let mut counts = Vec::with_capacity(self.members.len());
        for &member in &self.members {
            let count = snapshots
                .iter()
                .filter(|snapshot| snapshot.get(member) == Ok(true))
                .count();
            counts.push((member, count));
        }
        Ok(counts)
    }

    /// Count active circuit members per explicitly annotated region.
    /// Unannotated members are omitted; no anatomical region is inferred.
    /// Returns sorted (region_id, active_members, total_members) records.
    pub fn activation_by_annotated_region(
        &self,
        state: &BooleanState,
        annotations: &[Option<u32>],
    ) -> Result<Vec<(u32, usize, usize)>, RecallError> {
        if state.len() != self.universe_len || annotations.len() != self.universe_len {
            return Err(RecallError::LengthMismatch);
        }
        let mut counts = std::collections::BTreeMap::<u32, (usize, usize)>::new();
        for &member in &self.members {
            if let Some(region) = annotations[member] {
                let entry = counts.entry(region).or_default();
                entry.1 += 1;
                if state.get(member) == Ok(true) {
                    entry.0 += 1;
                }
            }
        }
        Ok(counts
            .into_iter()
            .map(|(region, (active, total))| (region, active, total))
            .collect())
    }

    /// Materialize recalled members as a packed BooleanState, with no invented edges.
    pub fn select(&self, state: &BooleanState) -> Result<BooleanState, RecallError> {
        if state.len() != self.universe_len {
            return Err(RecallError::LengthMismatch);
        }
        let mut selected = BooleanState::new(self.universe_len);
        for &index in &self.members {
            if state.get(index).expect("validated circuit index") {
                selected.set(index, true).expect("validated circuit index");
            }
        }
        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_is_deterministic_and_bounded() {
        let circuit = Subcircuit::new(130, vec![129, 64, 1]).unwrap();
        assert_eq!(circuit.members(), &[1, 64, 129]);
        let mut state = BooleanState::new(130);
        state.set(1, true).unwrap();
        state.set(129, true).unwrap();
        state.set(2, true).unwrap();
        assert_eq!(circuit.recall(&state, 2), Ok(true));
        assert_eq!(circuit.recall(&state, 3), Ok(false));
        assert_eq!(circuit.recall(&state, 4), Ok(false));
        assert_eq!(
            circuit
                .select(&state)
                .unwrap()
                .active_indices()
                .collect::<Vec<_>>(),
            vec![1, 129]
        );
    }

    #[test]
    fn temporal_recall_counts_matching_snapshots_not_individual_spikes() {
        let circuit = Subcircuit::new(4, vec![1, 3]).unwrap();
        let mut first = BooleanState::new(4);
        first.set(1, true).unwrap();
        first.set(3, true).unwrap();
        let mut second = BooleanState::new(4);
        second.set(1, true).unwrap();
        let mut third = BooleanState::new(4);
        third.set(3, true).unwrap();
        let history = [first.clone(), second, first, third];
        assert_eq!(circuit.recall_over_time(&history, 2, 2), Ok(true));
        assert_eq!(circuit.recall_over_time(&history, 2, 3), Ok(false));
        assert_eq!(circuit.recall_over_time(&history, 2, 5), Ok(false));
        assert_eq!(circuit.recall_over_time(&history, 1, 3), Ok(true));
        assert_eq!(
            circuit.recall_over_time(&[], 1, 1),
            Err(RecallError::EmptyHistory)
        );
        assert_eq!(
            circuit.recall_over_time(&history, 0, 1),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(
            circuit.recall_over_time(&history, 1, 0),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(
            circuit.recall_over_time(&[BooleanState::new(5)], 1, 1),
            Err(RecallError::LengthMismatch)
        );
    }

    #[test]
    fn latest_streak_requires_contiguous_recent_activation() {
        let circuit = Subcircuit::new(3, vec![0, 2]).unwrap();
        let mut active = BooleanState::new(3);
        active.set(0, true).unwrap();
        active.set(2, true).unwrap();
        let quiet = BooleanState::new(3);
        let history = [active.clone(), quiet.clone(), active.clone(), active];
        assert_eq!(circuit.recall_over_time(&history, 2, 3), Ok(true));
        assert_eq!(circuit.recall_latest_streak(&history, 2, 3), Ok(false));
        assert_eq!(circuit.recall_latest_streak(&history, 2, 2), Ok(true));
        assert_eq!(circuit.recall_latest_streak(&history, 2, 5), Ok(false));
        assert_eq!(
            circuit.recall_latest_streak(&history, 0, 1),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(
            circuit.recall_latest_streak(&history, 2, 0),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(
            circuit.recall_latest_streak(&[], 2, 1),
            Err(RecallError::EmptyHistory)
        );
        assert_eq!(
            circuit.recall_latest_streak(&[BooleanState::new(2)], 1, 1),
            Err(RecallError::LengthMismatch)
        );
    }

    #[test]
    fn frequent_activity_selects_top_members_with_stable_ties() {
        let mut first = BooleanState::new(5);
        first.set(1, true).unwrap();
        first.set(3, true).unwrap();
        let mut second = BooleanState::new(5);
        second.set(1, true).unwrap();
        second.set(2, true).unwrap();
        let mut third = BooleanState::new(5);
        third.set(2, true).unwrap();
        third.set(3, true).unwrap();
        let history = [first, second, third];
        let selected = Subcircuit::from_frequent_activity(&history, 2, 2).unwrap();
        assert_eq!(selected.members(), &[1, 2]);
        assert_eq!(
            Subcircuit::from_frequent_activity(&history, 3, 2),
            Err(RecallError::EmptyCircuit)
        );
        assert_eq!(
            Subcircuit::from_frequent_activity(&history, 1, 0),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(
            Subcircuit::from_frequent_activity(&[BooleanState::new(2), BooleanState::new(3)], 1, 1),
            Err(RecallError::LengthMismatch)
        );
    }

    #[test]
    fn member_activation_counts_preserve_explicit_membership_and_zeroes() {
        let circuit = Subcircuit::new(5, vec![4, 1, 3]).unwrap();
        let mut first = BooleanState::new(5);
        first.set(1, true).unwrap();
        first.set(2, true).unwrap();
        let mut second = BooleanState::new(5);
        second.set(1, true).unwrap();
        second.set(4, true).unwrap();
        assert_eq!(
            circuit.member_activation_counts(&[first, second]),
            Ok(vec![(1, 2), (3, 0), (4, 1)])
        );
        assert_eq!(
            circuit.member_activation_counts(&[]),
            Err(RecallError::EmptyHistory)
        );
        assert_eq!(
            circuit.member_activation_counts(&[BooleanState::new(4)]),
            Err(RecallError::LengthMismatch)
        );
    }

    #[test]
    fn regional_recall_counts_only_explicitly_annotated_members() {
        let circuit = Subcircuit::new(6, vec![5, 0, 2, 3]).unwrap();
        let mut state = BooleanState::new(6);
        state.set(0, true).unwrap();
        state.set(3, true).unwrap();
        state.set(4, true).unwrap();
        let labels = [Some(8), None, Some(8), Some(2), Some(2), None];
        assert_eq!(
            circuit.activation_by_annotated_region(&state, &labels),
            Ok(vec![(2, 1, 1), (8, 1, 2)])
        );
        assert_eq!(
            circuit.activation_by_annotated_region(&state, &labels[..5]),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(
            circuit.activation_by_annotated_region(&BooleanState::new(5), &labels),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(
            circuit.activation_by_annotated_region(&state, &[None; 6]),
            Ok(vec![])
        );
    }

    #[test]
    fn invalid_membership_and_budget_are_rejected() {
        assert_eq!(Subcircuit::new(2, vec![]), Err(RecallError::EmptyCircuit));
        assert_eq!(Subcircuit::new(2, vec![2]), Err(RecallError::OutOfBounds));
        assert_eq!(
            Subcircuit::new(2, vec![1, 1]),
            Err(RecallError::DuplicateMember)
        );
        let circuit = Subcircuit::new(2, vec![1]).unwrap();
        assert_eq!(
            circuit.recall(&BooleanState::new(1), 1),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(
            circuit.select(&BooleanState::new(1)),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(
            circuit.recall(&BooleanState::new(2), 0),
            Err(RecallError::ZeroBudget)
        );
    }
}

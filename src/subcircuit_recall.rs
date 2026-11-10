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

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
}

impl OnlineRecall {
    pub fn new(circuit: Subcircuit) -> Self {
        let counts = vec![0; circuit.members().len()];
        Self { circuit, counts, ticks: 0, current_streak: 0, longest_streak: 0 }
    }

    /// Update atomically: invalid dimensions or counter overflow leave all counters unchanged.
    /// A matching observation has at least `minimum_active` selected bits set.
    pub fn observe(&mut self, state: &BooleanState, minimum_active: usize) -> Result<bool, RecallError> {
        if minimum_active == 0 {
            return Err(RecallError::ZeroBudget);
        }
        // Validate before modifying counters.
        let matches = self.circuit.recall(state, minimum_active)?;
        let next_ticks = self.ticks.checked_add(1).ok_or(RecallError::ZeroBudget)?;
        let next_streak = if matches {
            self.current_streak.checked_add(1).ok_or(RecallError::ZeroBudget)?
        } else { 0 };
        let mut increments = Vec::with_capacity(self.counts.len());
        for (position, &member) in self.circuit.members().iter().enumerate() {
            if state.get(member) == Ok(true) {
                self.counts[position].checked_add(1).ok_or(RecallError::ZeroBudget)?;
                increments.push(position);
            }
        }
        self.ticks = next_ticks;
        self.current_streak = next_streak;
        self.longest_streak = self.longest_streak.max(next_streak);
        for position in increments {
            self.counts[position] += 1;
        }
        Ok(matches)
    }

    #[must_use]
    pub fn ticks(&self) -> u64 { self.ticks }

    #[must_use]
    pub fn latest_streak(&self) -> u64 { self.current_streak }

    #[must_use]
    pub fn longest_streak(&self) -> u64 { self.longest_streak }

    #[must_use]
    pub fn member_counts(&self) -> impl Iterator<Item = (usize, u64)> + '_ {
        self.circuit.members().iter().copied().zip(self.counts.iter().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(tracker.member_counts().collect::<Vec<_>>(), vec![(1, 3), (3, 0), (4, 3)]);
    }

    #[test]
    fn rejects_bad_input_without_mutation() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(3, vec![0]).unwrap());
        assert_eq!(tracker.observe(&BooleanState::new(2), 1), Err(RecallError::LengthMismatch));
        assert_eq!(tracker.observe(&BooleanState::new(3), 0), Err(RecallError::ZeroBudget));
        assert_eq!(tracker.ticks(), 0);
        assert_eq!(tracker.member_counts().collect::<Vec<_>>(), vec![(0, 0)]);
    }
}

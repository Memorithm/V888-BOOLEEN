//! Streaming, bounded observation of an explicitly selected experimental subcircuit.
//! No anatomical edges or learning weights are inferred from activation history.
use crate::{subcircuit_recall::Subcircuit, BooleanState};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnlineRecallError {
    LengthMismatch,
    CounterOverflow,
    ZeroBudget,
}

/// O(members) memory regardless of the number of observed simulation steps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnlineRecall {
    circuit: Subcircuit,
    counts: Vec<u64>,
    observed_steps: u64,
    latest_streak: u64,
    longest_streak: u64,
}

impl OnlineRecall {
    #[must_use]
    pub fn new(circuit: Subcircuit) -> Self {
        let counts = vec![0; circuit.members().len()];
        Self {
            circuit,
            counts,
            observed_steps: 0,
            latest_streak: 0,
            longest_streak: 0,
        }
    }

    /// Record one chronological snapshot; all counters are checked before mutation.
    /// A matching step has at least `minimum_active` active selected members.
    pub fn observe(
        &mut self,
        snapshot: &BooleanState,
        minimum_active: usize,
    ) -> Result<bool, OnlineRecallError> {
        if self.circuit.select(snapshot).is_err() {
            return Err(OnlineRecallError::LengthMismatch);
        }
        if minimum_active == 0 {
            return Err(OnlineRecallError::ZeroBudget);
        }
        let mut active = 0usize;
        for (index, &member) in self.circuit.members().iter().enumerate() {
            if snapshot.get(member) == Ok(true) {
                active += 1;
                if self.counts[index] == u64::MAX {
                    return Err(OnlineRecallError::CounterOverflow);
                }
            }
        }
        let next_steps = self
            .observed_steps
            .checked_add(1)
            .ok_or(OnlineRecallError::CounterOverflow)?;
        let matched = active >= minimum_active;
        let next_streak = if matched {
            self.latest_streak
                .checked_add(1)
                .ok_or(OnlineRecallError::CounterOverflow)?
        } else {
            0
        };
        for (index, &member) in self.circuit.members().iter().enumerate() {
            if snapshot.get(member) == Ok(true) {
                self.counts[index] += 1;
            }
        }
        self.observed_steps = next_steps;
        self.latest_streak = next_streak;
        self.longest_streak = self.longest_streak.max(next_streak);
        Ok(matched)
    }

    #[must_use]
    pub fn observed_steps(&self) -> u64 {
        self.observed_steps
    }

    #[must_use]
    pub fn latest_streak(&self) -> u64 {
        self.latest_streak
    }

    #[must_use]
    pub fn longest_streak(&self) -> u64 {
        self.longest_streak
    }

    #[must_use]
    pub fn member_counts(&self) -> Vec<(usize, u64)> {
        self.circuit
            .members()
            .iter()
            .copied()
            .zip(self.counts.iter().copied())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn online_counts_and_streaks_are_bounded_and_deterministic() {
        let circuit = Subcircuit::new(5, vec![4, 1]).unwrap();
        let mut tracker = OnlineRecall::new(circuit);
        let mut both = BooleanState::new(5);
        both.set(1, true).unwrap();
        both.set(4, true).unwrap();
        let quiet = BooleanState::new(5);
        assert_eq!(tracker.observe(&both, 2), Ok(true));
        assert_eq!(tracker.observe(&quiet, 2), Ok(false));
        assert_eq!(tracker.observe(&both, 2), Ok(true));
        assert_eq!(tracker.observe(&both, 2), Ok(true));
        assert_eq!(tracker.member_counts(), vec![(1, 3), (4, 3)]);
        assert_eq!(tracker.observed_steps(), 4);
        assert_eq!(tracker.latest_streak(), 2);
        assert_eq!(tracker.longest_streak(), 2);
        let before = tracker.clone();
        assert_eq!(
            tracker.observe(&BooleanState::new(4), 2),
            Err(OnlineRecallError::LengthMismatch)
        );
        assert_eq!(tracker, before);
    }
}

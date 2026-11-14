//! Experimental B888-3 recovery of a frozen, explicitly selected subcircuit.
//! No anatomical edges or biological dynamics are inferred.
use crate::BooleanState;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecoveryError {
    EmptyMembers,
    DuplicateMember,
    MemberOutOfBounds,
    InvalidThreshold,
    UnequalWindows,
    WindowTooLong,
    StateDimensionMismatch,
    InvalidReleaseTick,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryProfile {
    /// Number of ticks meeting the recall threshold in the baseline window.
    pub baseline_recall_ticks: usize,
    /// Number of ticks meeting the threshold after the release tick.
    pub post_release_recall_ticks: usize,
    /// First post-release tick offset with a recalled state, if any.
    pub first_recovery_offset: Option<usize>,
    /// First offset starting a run of `sustain_ticks` consecutive recalls.
    pub sustained_recovery_offset: Option<usize>,
    /// Sum of active selected members after release.
    pub post_release_activation_cost: usize,
}

/// Compare paired synthetic baseline and perturbed traces without learning.
/// `release_tick` is the first tick after temporary suppression ends.
/// A recovery is measured against an absolute, predeclared threshold, not
/// attributed to a causal intervention or to actual biological recovery.
pub fn measure_recovery(
    members: &[usize],
    baseline: &[BooleanState],
    perturbed: &[BooleanState],
    threshold: usize,
    release_tick: usize,
    sustain_ticks: usize,
) -> Result<RecoveryProfile, RecoveryError> {
    if members.is_empty() {
        return Err(RecoveryError::EmptyMembers);
    }
    if threshold == 0 || threshold > members.len() || sustain_ticks == 0 {
        return Err(RecoveryError::InvalidThreshold);
    }
    if baseline.len() != perturbed.len() {
        return Err(RecoveryError::UnequalWindows);
    }
    if baseline.is_empty() || baseline.len() > 1000 {
        return Err(RecoveryError::WindowTooLong);
    }
    if release_tick >= baseline.len() {
        return Err(RecoveryError::InvalidReleaseTick);
    }
    let dimension = baseline[0].len();
    if baseline.iter().chain(perturbed).any(|s| s.len() != dimension) {
        return Err(RecoveryError::StateDimensionMismatch);
    }
    let mut sorted = members.to_vec();
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err(RecoveryError::DuplicateMember);
    }
    if sorted.iter().any(|&member| member >= dimension) {
        return Err(RecoveryError::MemberOutOfBounds);
    }
    let count = |state: &BooleanState| -> usize {
        sorted.iter().filter(|&&member| state.get(member) == Ok(true)).count()
    };
    let baseline_recall_ticks = baseline.iter().filter(|s| count(s) >= threshold).count();
    let mut first_recovery_offset = None;
    let mut sustained_recovery_offset = None;
    let mut post_release_recall_ticks = 0;
    let mut post_release_activation_cost = 0;
    let mut run = 0;
    for (offset, state) in perturbed[release_tick..].iter().enumerate() {
        let active = count(state);
        post_release_activation_cost += active;
        if active >= threshold {
            post_release_recall_ticks += 1;
            if first_recovery_offset.is_none() {
                first_recovery_offset = Some(offset);
            }
            run += 1;
            if run == sustain_ticks && sustained_recovery_offset.is_none() {
                sustained_recovery_offset = Some(offset + 1 - sustain_ticks);
            }
        } else {
            run = 0;
        }
    }
    Ok(RecoveryProfile {
        baseline_recall_ticks,
        post_release_recall_ticks,
        first_recovery_offset,
        sustained_recovery_offset,
        post_release_activation_cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(bits: &[usize]) -> BooleanState {
        let mut s = BooleanState::new(4);
        for &bit in bits {
            s.set(bit, true).unwrap();
        }
        s
    }

    #[test]
    fn delayed_sustained_recovery_is_distinct_from_first_recall() {
        let baseline = vec![state(&[0, 1]); 6];
        let perturbed = vec![
            state(&[]), state(&[]), state(&[0, 1]), state(&[]),
            state(&[0, 1]), state(&[0, 1]),
        ];
        let p = measure_recovery(&[0, 1], &baseline, &perturbed, 2, 2, 2).unwrap();
        assert_eq!(p.baseline_recall_ticks, 6);
        assert_eq!(p.first_recovery_offset, Some(0));
        assert_eq!(p.sustained_recovery_offset, Some(2));
        assert_eq!(p.post_release_recall_ticks, 3);
        assert_eq!(p.post_release_activation_cost, 6);
    }

    #[test]
    fn never_recovers_and_does_not_mutate_traces() {
        let baseline = vec![state(&[0, 1]); 3];
        let perturbed = vec![state(&[]); 3];
        let original = perturbed.clone();
        let p = measure_recovery(&[0, 1], &baseline, &perturbed, 2, 1, 2).unwrap();
        assert_eq!(p.first_recovery_offset, None);
        assert_eq!(p.sustained_recovery_offset, None);
        assert_eq!(perturbed, original);
    }

    #[test]
    fn rejects_ambiguous_or_invalid_experiments() {
        let traces = vec![state(&[0]); 2];
        assert_eq!(measure_recovery(&[0, 0], &traces, &traces, 1, 0, 1), Err(RecoveryError::DuplicateMember));
        assert_eq!(measure_recovery(&[4], &traces, &traces, 1, 0, 1), Err(RecoveryError::MemberOutOfBounds));
        assert_eq!(measure_recovery(&[0], &traces, &traces, 1, 2, 1), Err(RecoveryError::InvalidReleaseTick));
        assert_eq!(measure_recovery(&[0], &traces, &traces[..1], 1, 0, 1), Err(RecoveryError::UnequalWindows));
    }
}

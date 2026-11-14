//! Experimental paired forgetting/recovery statistics on frozen, explicitly chosen members.
//! This is not a biological inference or an anatomical graph mutation.
use crate::BooleanState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForgettingError {
    EmptyTrace,
    LengthMismatch,
    InvalidBoundary,
    InvalidThreshold,
    InvalidMember,
    DuplicateMember,
    TooManyObservations,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForgettingProfile {
    pub baseline_recall: u32,
    pub perturbed_recall: u32,
    pub lost_recall: u32,
    pub gained_recall: u32,
    pub recovered_recall: u32,
    pub first_recovery_tick: Option<usize>,
    pub baseline_activation_cost: u64,
    pub perturbed_activation_cost: u64,
}

/// Compare equal-length baseline and perturbed traces without changing either trace.
/// The boundary marks the first post-perturbation observation. A recovery requires
/// a perturbed recall at a tick where the baseline also recalls.
/// Both activation costs count only the selected members after the boundary.
pub fn measure_paired_forgetting(
    members: &[usize],
    threshold: usize,
    baseline: &[BooleanState],
    perturbed: &[BooleanState],
    boundary: usize,
) -> Result<ForgettingProfile, ForgettingError> {
    if baseline.is_empty() {
        return Err(ForgettingError::EmptyTrace);
    }
    if baseline.len() != perturbed.len() || baseline.iter().any(|s| s.len() != baseline[0].len())
        || perturbed.iter().any(|s| s.len() != baseline[0].len())
    {
        return Err(ForgettingError::LengthMismatch);
    }
    if baseline.len() > 1000 {
        return Err(ForgettingError::TooManyObservations);
    }
    if boundary >= baseline.len() {
        return Err(ForgettingError::InvalidBoundary);
    }
    if members.is_empty() || threshold == 0 || threshold > members.len() {
        return Err(ForgettingError::InvalidThreshold);
    }
    let mut sorted = members.to_vec();
    sorted.sort_unstable();
    if sorted.last().is_some_and(|&m| m >= baseline[0].len()) {
        return Err(ForgettingError::InvalidMember);
    }
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err(ForgettingError::DuplicateMember);
    }

    let mut result = ForgettingProfile {
        baseline_recall: 0,
        perturbed_recall: 0,
        lost_recall: 0,
        gained_recall: 0,
        recovered_recall: 0,
        first_recovery_tick: None,
        baseline_activation_cost: 0,
        perturbed_activation_cost: 0,
    };
    for tick in boundary..baseline.len() {
        let count = |state: &BooleanState| -> usize {
            sorted.iter().filter(|&&m| state.get(m) == Ok(true)).count()
        };
        let b = count(&baseline[tick]);
        let p = count(&perturbed[tick]);
        result.baseline_activation_cost += b as u64;
        result.perturbed_activation_cost += p as u64;
        let br = b >= threshold;
        let pr = p >= threshold;
        result.baseline_recall += u32::from(br);
        result.perturbed_recall += u32::from(pr);
        result.lost_recall += u32::from(br && !pr);
        result.gained_recall += u32::from(!br && pr);
        if br && pr {
            result.recovered_recall += 1;
            if result.first_recovery_tick.is_none() {
                result.first_recovery_tick = Some(tick);
            }
        }
    }
    Ok(result)
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
    fn detects_loss_then_recovery_on_matched_ticks() {
        let base = vec![state(&[0, 1]), state(&[0, 1]), state(&[0, 1])];
        let pert = vec![state(&[0, 1]), state(&[0]), state(&[0, 1])];
        let p = measure_paired_forgetting(&[0, 1], 2, &base, &pert, 1).unwrap();
        assert_eq!((p.lost_recall, p.recovered_recall, p.first_recovery_tick), (1, 1, Some(2)));
        assert_eq!((p.baseline_activation_cost, p.perturbed_activation_cost), (4, 3));
    }

    #[test]
    fn rejects_duplicate_members_and_mismatched_dimensions() {
        let trace = vec![state(&[0])];
        assert_eq!(measure_paired_forgetting(&[0, 0], 1, &trace, &trace, 0), Err(ForgettingError::DuplicateMember));
        let other = vec![BooleanState::new(5)];
        assert_eq!(measure_paired_forgetting(&[0], 1, &trace, &other, 0), Err(ForgettingError::LengthMismatch));
    }

    #[test]
    fn counts_gained_recall_separately_from_recovery() {
        let base = vec![state(&[]), state(&[0])];
        let pert = vec![state(&[0]), state(&[])];
        let p = measure_paired_forgetting(&[0], 1, &base, &pert, 0).unwrap();
        assert_eq!((p.gained_recall, p.lost_recall, p.recovered_recall), (1, 1, 0));
        assert_eq!(p.first_recovery_tick, None);
    }
}

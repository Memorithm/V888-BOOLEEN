//! Frozen, explicitly annotated regional recall across ordered held-out windows.
//! This is a synthetic-state statistic, not evidence of biological memory.
use crate::BooleanState;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowRecallError {
    InvalidBudget,
    LengthMismatch,
    Overflow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionalWindowRecall {
    pub region: u32,
    pub members: usize,
    /// Number of held-out windows in which the frozen region recalled at least once.
    pub recalled_windows: usize,
    /// Number of positive recall ticks per window, in input order.
    pub positive_ticks: Vec<usize>,
    /// Lowest number of positive recall ticks over all held-out windows.
    pub minimum_positive_ticks: usize,
}

/// Evaluate a frozen set of annotated regions over disjoint, ordered windows.
///
/// No membership is learned from held-out states. A recall tick requires at least
/// `minimum_active` active members of the same annotated region. All windows
/// must be nonempty; their total length is bounded to avoid unbounded work.
/// The caller is responsible for ensuring that held-out windows were not used
/// for candidate selection or training.
pub fn evaluate_regional_window_stability(
    annotations: &[Option<u32>],
    frozen_regions: &[u32],
    windows: &[Vec<BooleanState>],
    minimum_active: usize,
) -> Result<Vec<RegionalWindowRecall>, WindowRecallError> {
    if annotations.is_empty()
        || frozen_regions.is_empty()
        || windows.is_empty()
        || windows.len() > 64
        || minimum_active == 0
        || frozen_regions.len() > 64
    {
        return Err(WindowRecallError::InvalidBudget);
    }
    let mut total = 0usize;
    for window in windows {
        if window.is_empty() {
            return Err(WindowRecallError::InvalidBudget);
        }
        total = total.checked_add(window.len()).ok_or(WindowRecallError::Overflow)?;
        if total > 1000 {
            return Err(WindowRecallError::InvalidBudget);
        }
        if window.iter().any(|state| state.len() != annotations.len()) {
            return Err(WindowRecallError::LengthMismatch);
        }
    }
    let mut members = BTreeMap::<u32, usize>::new();
    for &region in frozen_regions {
        if members.insert(region, 0).is_some() {
            return Err(WindowRecallError::InvalidBudget);
        }
    }
    for region in annotations.iter().flatten() {
        if let Some(count) = members.get_mut(region) {
            *count = count.checked_add(1).ok_or(WindowRecallError::Overflow)?;
        }
    }
    if members.values().any(|&count| count < minimum_active) {
        return Err(WindowRecallError::InvalidBudget);
    }
    let mut result = Vec::with_capacity(members.len());
    for (region, count) in members {
        let mut positive_ticks = Vec::with_capacity(windows.len());
        for window in windows {
            let mut positive = 0usize;
            for state in window {
                let active = state.active_indices()
                    .filter(|&index| annotations[index] == Some(region))
                    .count();
                if active >= minimum_active {
                    positive = positive.checked_add(1).ok_or(WindowRecallError::Overflow)?;
                }
            }
            positive_ticks.push(positive);
        }
        let recalled_windows = positive_ticks.iter().filter(|&&n| n > 0).count();
        let minimum_positive_ticks = positive_ticks.iter().copied().min().unwrap_or(0);
        result.push(RegionalWindowRecall {
            region, members: count, recalled_windows, positive_ticks, minimum_positive_ticks,
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state(bits: &[usize]) -> BooleanState {
        let mut s = BooleanState::new(4);
        for &bit in bits { s.set(bit, true).unwrap(); }
        s
    }
    #[test]
    fn frozen_membership_and_temporal_instability() {
        let annotations = [Some(2), Some(2), Some(9), None];
        let windows = vec![vec![state(&[0, 1, 3]), state(&[0, 1])], vec![state(&[2]), state(&[0])]];
        let result = evaluate_regional_window_stability(&annotations, &[9, 2], &windows, 2).unwrap_err();
        assert_eq!(result, WindowRecallError::InvalidBudget);
        let result = evaluate_regional_window_stability(&annotations, &[2], &windows, 2).unwrap();
        assert_eq!(result[0].positive_ticks, vec![2, 0]);
        assert_eq!(result[0].recalled_windows, 1);
        assert_eq!(result[0].minimum_positive_ticks, 0);
        assert_eq!(result[0].members, 2);
    }
    #[test]
    fn dimensions_and_duplicate_regions_are_rejected() {
        let a = [Some(2); 4];
        assert_eq!(evaluate_regional_window_stability(&a, &[2, 2], &[vec![state(&[0])]], 1), Err(WindowRecallError::InvalidBudget));
        assert_eq!(evaluate_regional_window_stability(&a, &[2], &[vec![BooleanState::new(3)]], 1), Err(WindowRecallError::LengthMismatch));
    }
    #[test]
    fn counts_all_windows_without_mutating_states() {
        let a = [Some(1), Some(1), Some(3), None];
        let windows = vec![vec![state(&[0, 2]), state(&[1])], vec![state(&[2]), state(&[])]];
        let before = windows.clone();
        let result = evaluate_regional_window_stability(&a, &[3, 1], &windows, 1).unwrap();
        assert_eq!(windows, before);
        assert_eq!(result[0].region, 1);
        assert_eq!(result[0].positive_ticks, vec![2, 0]);
        assert_eq!(result[1].positive_ticks, vec![1, 1]);
        assert_eq!(result[1].recalled_windows, 2);
    }
}

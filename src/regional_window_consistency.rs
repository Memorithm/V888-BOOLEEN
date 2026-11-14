//! B888-3: frozen, explicitly annotated regional recall across equal-budget windows.
//! Scores are experimental Boolean-state statistics, not biological claims.
use crate::BooleanState;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowError {
    EmptyWindows,
    EmptyRegion,
    DuplicateNeuron,
    OutOfBounds,
    UnequalBudget,
    TooManyWindows,
    Overflow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowConsistency {
    pub window_positive_ticks: Vec<u64>,
    pub positive_windows: usize,
    pub minimum_positive_ticks: u64,
    pub maximum_positive_ticks: u64,
    pub total_positive_ticks: u64,
    pub total_member_activations: u64,
}

/// Evaluate a frozen member set over independent, equally sized windows.
/// A tick is positive if any annotated member is active. No members are learned
/// or changed using evaluation observations. All windows must have the same
/// nonzero number of ticks; each tick must have the same neuron dimension.
pub fn evaluate_window_consistency(
    members: &[usize],
    windows: &[&[BooleanState]],
) -> Result<WindowConsistency, WindowError> {
    if windows.is_empty() {
        return Err(WindowError::EmptyWindows);
    }
    if windows.len() > 1_000 {
        return Err(WindowError::TooManyWindows);
    }
    if members.is_empty() {
        return Err(WindowError::EmptyRegion);
    }
    let budget = windows[0].len();
    if budget == 0 || budget > 1_000 {
        return Err(WindowError::UnequalBudget);
    }
    let dimension = windows[0][0].len();
    let mut sorted = members.to_vec();
    sorted.sort_unstable();
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(WindowError::DuplicateNeuron);
    }
    if sorted.iter().any(|&index| index >= dimension) {
        return Err(WindowError::OutOfBounds);
    }
    let mut result = WindowConsistency {
        window_positive_ticks: Vec::with_capacity(windows.len()),
        positive_windows: 0,
        minimum_positive_ticks: u64::MAX,
        maximum_positive_ticks: 0,
        total_positive_ticks: 0,
        total_member_activations: 0,
    };
    for window in windows {
        if window.len() != budget {
            return Err(WindowError::UnequalBudget);
        }
        let mut positives = 0_u64;
        for state in *window {
            if state.len() != dimension {
                return Err(WindowError::UnequalBudget);
            }
            let mut any = false;
            for &member in &sorted {
                if state.get(member).map_err(|_| WindowError::OutOfBounds)? {
                    result.total_member_activations = result
                        .total_member_activations
                        .checked_add(1)
                        .ok_or(WindowError::Overflow)?;
                    any = true;
                }
            }
            if any {
                positives = positives.checked_add(1).ok_or(WindowError::Overflow)?;
            }
        }
        if positives > 0 {
            result.positive_windows += 1;
        }
        result.minimum_positive_ticks = result.minimum_positive_ticks.min(positives);
        result.maximum_positive_ticks = result.maximum_positive_ticks.max(positives);
        result.total_positive_ticks = result
            .total_positive_ticks
            .checked_add(positives)
            .ok_or(WindowError::Overflow)?;
        result.window_positive_ticks.push(positives);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state(bits: &[usize]) -> BooleanState {
        let mut state = BooleanState::new(4);
        for &bit in bits {
            state.set(bit, true).unwrap();
        }
        state
    }
    #[test]
    fn equal_budget_windows_measure_intermitttent_recall() {
        let a = [state(&[0, 1]), state(&[])];
        let b = [state(&[]), state(&[])];
        let c = [state(&[1]), state(&[1])];
        let before = a.clone();
        let r = evaluate_window_consistency(&[1], &[&a, &b, &c]).unwrap();
        assert_eq!(r.window_positive_ticks, vec![1, 0, 2]);
        assert_eq!(
            (
                r.positive_windows,
                r.minimum_positive_ticks,
                r.maximum_positive_ticks
            ),
            (2, 0, 2)
        );
        assert_eq!((r.total_positive_ticks, r.total_member_activations), (3, 3));
        assert_eq!(a, before);
    }
    #[test]
    fn rejects_leakage_prone_or_invalid_windows() {
        let a = [state(&[1])];
        let b = [state(&[1]), state(&[])];
        assert_eq!(
            evaluate_window_consistency(&[1], &[&a, &b]),
            Err(WindowError::UnequalBudget)
        );
        assert_eq!(
            evaluate_window_consistency(&[1, 1], &[&a]),
            Err(WindowError::DuplicateNeuron)
        );
        assert_eq!(
            evaluate_window_consistency(&[4], &[&a]),
            Err(WindowError::OutOfBounds)
        );
        assert_eq!(
            evaluate_window_consistency(&[], &[&a]),
            Err(WindowError::EmptyRegion)
        );
    }
}

//! Paired experimental recall decisions on explicitly annotated regions.
//! Decisions are not evidence of biological function or hardware efficiency.
use crate::region_recall_monitor::{RegionMonitorError, RegionRecallMonitor};
use crate::BooleanState;

/// A candidate is accepted only when durable recall is strictly earlier (or
/// baseline recall is censored) and neither activity nor switching exceeds the
/// corresponding baseline cost plus the caller's explicit allowance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecallTradeoff {
    pub region: u32,
    pub baseline_confirmation: Option<usize>,
    pub candidate_confirmation: Option<usize>,
    pub baseline_active_ticks: u128,
    pub candidate_active_ticks: u128,
    pub baseline_transitions: u128,
    pub candidate_transitions: u128,
    pub accepted: bool,
}

impl RegionRecallMonitor {
    /// Evaluate an equal-horizon, paired recall experiment without changing
    /// either trajectory or the underlying connectivity. Missing confirmation
    /// means right-censoring, never a zero-latency recovery.
    pub fn select_paired_recall_tradeoffs(
        &self,
        baseline: &[BooleanState],
        candidate: &[BooleanState],
        minimum_active: usize,
        consecutive: usize,
        activity_allowance: u128,
        transition_allowance: u128,
    ) -> Result<Vec<RecallTradeoff>, RegionMonitorError> {
        let durable = self.paired_durable_recall_activity(
            baseline, candidate, minimum_active, consecutive,
        )?;
        let transitions = self.paired_region_transition_cost(baseline, candidate)?;
        if durable.len() != transitions.len() {
            return Err(RegionMonitorError::InvalidDimensions);
        }
        durable.into_iter().zip(transitions).map(
            |((region, before, after, active_before, active_after),
              (transition_region, flips_before, flips_after))| {
                if region != transition_region {
                    return Err(RegionMonitorError::InvalidDimensions);
                }
                let earlier = match (before, after) {
                    (_, None) => false,
                    (None, Some(_)) => true,
                    (Some(b), Some(c)) => c < b,
                };
                // Saturating comparison treats overflowed allowances as an
                // unbounded cap, rather than wrapping to a false rejection.
                let accepted = earlier
                    && active_after <= active_before.saturating_add(activity_allowance)
                    && flips_after <= flips_before.saturating_add(transition_allowance);
                Ok(RecallTradeoff {
                    region,
                    baseline_confirmation: before,
                    candidate_confirmation: after,
                    baseline_active_ticks: active_before,
                    candidate_active_ticks: active_after,
                    baseline_transitions: flips_before,
                    candidate_transitions: flips_after,
                    accepted,
                })
            },
        ).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tradeoff_requires_confirmed_improvement_and_explicit_cost_allowances() {
        let monitor = RegionRecallMonitor::new(
            5, &[0, 1, 2, 3, 4],
            &[Some(7), Some(7), Some(9), Some(9), None],
        ).unwrap();
        let baseline = vec![BooleanState::new(5); 3];
        let mut candidate = baseline.clone();
        for tick in [1, 2] {
            candidate[tick].set(0, true).unwrap();
            candidate[tick].set(1, true).unwrap();
        }
        candidate[0].set(4, true).unwrap(); // Unannotated, excluded.
        let strict = monitor.select_paired_recall_tradeoffs(
            &baseline, &candidate, 2, 2, 0, 0,
        ).unwrap();
        assert_eq!(strict.len(), 2);
        assert_eq!(strict[0].region, 7);
        assert_eq!(strict[0].candidate_confirmation, Some(3));
        assert_eq!(strict[0].baseline_confirmation, None);
        assert_eq!(strict[0].candidate_active_ticks, 4);
        assert_eq!(strict[0].candidate_transitions, 2);
        assert!(!strict[0].accepted);
        assert!(!strict[1].accepted);
        let allowed = monitor.select_paired_recall_tradeoffs(
            &baseline, &candidate, 2, 2, 4, 2,
        ).unwrap();
        assert!(allowed[0].accepted);
        assert!(!allowed[1].accepted);
        assert_eq!(
            monitor.select_paired_recall_tradeoffs(
                &baseline[..2], &candidate, 2, 2, 4, 2,
            ),
            Err(RegionMonitorError::InvalidDimensions),
        );
    }
}

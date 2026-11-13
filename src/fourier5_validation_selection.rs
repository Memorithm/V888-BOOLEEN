//! Validation-only selection of final-label credit models.
//! TEST/OOD never enter this API; all interventions are trained on TRAIN.
use std::collections::BTreeSet;

use crate::fourier5_counterfactual_credit::{
    fit_final_loss, CreditConfig, CreditError, CreditResult,
};
use crate::fourier5_hierarchy::FiveCircleHierarchy;
use crate::fourier5_routing_search::RoutingSample;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Confusion {
    pub tp: usize,
    pub tn: usize,
    pub fp: usize,
    pub fn_: usize,
}

impl Confusion {
    /// Balanced accuracy is compared as exact fractions, without floats.
    /// Requires both classes in VALID.
    fn score(&self) -> (u128, u128) {
        let positive = (self.tp + self.fn_) as u128;
        let negative = (self.tn + self.fp) as u128;
        (
            self.tp as u128 * negative + self.tn as u128 * positive,
            2 * positive * negative,
        )
    }
}

#[derive(Clone, Debug)]
pub struct SelectedCredit {
    pub result: CreditResult,
    pub validation: Confusion,
    pub selected_seed: u64,
    pub fitted_candidates: usize,
    pub candidate_evaluations: usize,
}

fn evaluate(
    model: &FiveCircleHierarchy,
    valid: &[RoutingSample],
) -> Result<Confusion, CreditError> {
    let mut c = Confusion::default();
    for row in valid {
        let predicted = model.infer_scalar(&row.state)?;
        match (predicted, row.label) {
            (true, true) => c.tp += 1,
            (false, false) => c.tn += 1,
            (true, false) => c.fp += 1,
            (false, true) => c.fn_ += 1,
        }
    }
    Ok(c)
}

/// Fit each fixed seed on TRAIN only; select by VALID balanced accuracy.
/// Tie-break: fewer accepted interventions, then earlier seed in the supplied
/// list. No TEST/OOD access and no calibration against held-out test labels.
/// Every seed receives identical rounds and proposal counts.
pub fn select_by_validation(
    base: &FiveCircleHierarchy,
    train: &[RoutingSample],
    validation: &[RoutingSample],
    template: CreditConfig,
    seeds: &[u64],
) -> Result<SelectedCredit, CreditError> {
    if seeds.is_empty() || seeds.len() > 32 || train.is_empty() || validation.is_empty() {
        return Err(CreditError::InvalidBudget);
    }
    let mut seen = BTreeSet::new();
    for row in train.iter().chain(validation) {
        if row.state.len() != base.input_bits() {
            return Err(CreditError::InvalidInput);
        }
        if !seen.insert(row.state.words().to_vec()) {
            return Err(CreditError::DuplicateInput);
        }
    }
    let positives = validation.iter().filter(|r| r.label).count();
    if positives == 0 || positives == validation.len() {
        return Err(CreditError::InvalidInput);
    }
    let mut winner: Option<SelectedCredit> = None;
    let mut total_candidate_evaluations = 0_usize;
    for &seed in seeds {
        let config = CreditConfig { seed, ..template };
        let result = fit_final_loss(base, train, config)?;
        total_candidate_evaluations = total_candidate_evaluations
            .checked_add(result.candidate_evaluations)
            .ok_or(CreditError::InvalidBudget)?;
        let validation_score = evaluate(&result.model, validation)?;
        let replace = match &winner {
            None => true,
            Some(old) => {
                let (a, b) = validation_score.score();
                let (c, d) = old.validation.score();
                let left = a * d;
                let right = c * b;
                left > right
                    || (left == right
                        && result.accepted_parameter_moves + result.accepted_routing_moves
                            < old.result.accepted_parameter_moves
                                + old.result.accepted_routing_moves)
            }
        };
        if replace {
            winner = Some(SelectedCredit {
                result,
                validation: validation_score,
                selected_seed: seed,
                fitted_candidates: seeds.len(),
                candidate_evaluations: 0,
            });
        }
    }
    let mut selected = winner.expect("nonempty seed list");
    selected.candidate_evaluations = total_candidate_evaluations;
    Ok(selected)
}

/// Decision with a frozen, equal-size control eligible for deployment.
/// The control is never fitted or modified, and TEST/OOD are not inspected.
#[derive(Clone, Debug)]
pub enum GuardedSelection {
    Frozen {
        validation: Confusion,
        fitted_candidates: usize,
        candidate_evaluations: usize,
    },
    Trained(SelectedCredit),
}

/// Train the same fixed-budget candidates as `select_by_validation`, but
/// reject the winner when it does not strictly beat the frozen circuit on
/// independent VALID balanced accuracy. Ties favor the zero-intervention
/// control, avoiding needless changes to a circuit with equal VALID score.
pub fn select_or_keep_frozen(
    base: &FiveCircleHierarchy,
    train: &[RoutingSample],
    validation: &[RoutingSample],
    template: CreditConfig,
    seeds: &[u64],
) -> Result<GuardedSelection, CreditError> {
    select_or_keep_frozen_with_class_floor(
        base, train, validation, template, seeds, 0, 1,
    )
}

/// Opt-in deployment guard: require at least `min_recall_num / min_recall_den`
/// recall for EACH label on independent VALID. No TEST/OOD access. This
/// prevents accepting a balanced-score improvement that sacrifices one class
/// below the pre-registered minimum. A failed floor keeps the frozen circuit;
/// it does not claim that the frozen circuit itself meets the floor.
pub fn select_or_keep_frozen_with_class_floor(
    base: &FiveCircleHierarchy,
    train: &[RoutingSample],
    validation: &[RoutingSample],
    template: CreditConfig,
    seeds: &[u64],
    min_recall_num: usize,
    min_recall_den: usize,
) -> Result<GuardedSelection, CreditError> {
    if min_recall_den == 0 || min_recall_num > min_recall_den {
        return Err(CreditError::InvalidBudget);
    }
    // The selector performs split disjointness, class, width and budget checks.
    let selected = select_by_validation(base, train, validation, template, seeds)?;
    let frozen = evaluate(base, validation)?;
    if strictly_better_balanced(&selected.validation, &frozen)
        && meets_class_floor(&selected.validation, min_recall_num, min_recall_den)
    {
        Ok(GuardedSelection::Trained(selected))
    } else {
        Ok(GuardedSelection::Frozen {
            validation: frozen,
            fitted_candidates: selected.fitted_candidates,
            candidate_evaluations: selected.candidate_evaluations,
        })
    }
}

fn strictly_better_balanced(candidate: &Confusion, reference: &Confusion) -> bool {
    let (a, b) = candidate.score();
    let (c, d) = reference.score();
    a * d > c * b
}


/// Exact integer comparison, avoiding floating-point threshold drift.
fn meets_class_floor(c: &Confusion, numerator: usize, denominator: usize) -> bool {
    let pos = c.tp as u128 + c.fn_ as u128;
    let neg = c.tn as u128 + c.fp as u128;
    pos != 0
        && neg != 0
        && (c.tp as u128) * (denominator as u128) >= pos * (numerator as u128)
        && (c.tn as u128) * (denominator as u128) >= neg * (numerator as u128)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5_counterfactual_credit::{CreditPolicy, CreditScope};
    use crate::fourier5_hierarchy::{GateKind, LeafRouting};
    use crate::BooleanState;

    #[test]
    fn class_floor_blocks_single_class_collapse() {
        let collapsed = Confusion { tp: 8, tn: 0, fp: 2, fn_: 0 };
        let balanced = Confusion { tp: 7, tn: 1, fp: 1, fn_: 1 };
        assert!(!meets_class_floor(&collapsed, 1, 2));
        assert!(meets_class_floor(&balanced, 1, 2));
        assert!(meets_class_floor(&collapsed, 0, 1));
    }

    #[test]
    fn class_floor_rejects_invalid_ratio_before_training() {
        let base = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority], LeafRouting::Contiguous,
        ).unwrap();
        let config = CreditConfig {
            scope: CreditScope::OutputParameters,
            policy: CreditPolicy::Greedy,
            rounds: 1,
            proposals_per_round: 2,
            seed: 1,
        };
        assert!(matches!(
            select_or_keep_frozen_with_class_floor(
                &base, &[], &[], config, &[1], 2, 1,
            ),
            Err(CreditError::InvalidBudget)
        ));
    }

    #[test]
    fn frozen_control_wins_ties_and_loses_to_better_balanced_model() {
        let frozen = Confusion { tp: 2, tn: 2, fp: 2, fn_: 2 };
        let equal = Confusion { tp: 2, tn: 2, fp: 2, fn_: 2 };
        let better = Confusion { tp: 3, tn: 3, fp: 1, fn_: 1 };
        assert!(!strictly_better_balanced(&equal, &frozen));
        assert!(strictly_better_balanced(&better, &frozen));
        assert!(!strictly_better_balanced(&frozen, &better));
    }

    #[test]
    fn frozen_guard_preserves_base_when_training_cannot_improve_validation() {
        let base = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity], LeafRouting::Contiguous
        ).unwrap();
        let mut train = Vec::new();
        let mut validation = Vec::new();
        for n in 0..128_usize {
            let mut state = BooleanState::new(25);
            for bit in 0..25 {
                state.set(bit, (n >> bit) & 1 != 0).unwrap();
            }
            let label = base.infer_scalar(&state).unwrap();
            let row = RoutingSample { state, label };
            if n < 64 { train.push(row); } else { validation.push(row); }
        }
        // The oracle labels are the frozen circuit itself: no trained
        // candidate can strictly improve its perfect VALID accuracy.
        let config = CreditConfig {
            scope: CreditScope::OutputParameters,
            policy: CreditPolicy::Greedy,
            rounds: 1,
            proposals_per_round: 2,
            seed: 7,
        };
        // A parity output over binary-count patterns contains both classes.
        let choice = select_or_keep_frozen(&base, &train, &validation, config, &[7]).unwrap();
        assert!(matches!(choice, GuardedSelection::Frozen { .. }));
    }

    #[test]
    fn empty_training_split_is_rejected_before_fitting() {
        let model = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority], LeafRouting::Contiguous
        ).unwrap();
        let mut state = BooleanState::new(5);
        state.set(0, true).unwrap();
        let validation = [
            RoutingSample { state: state.clone(), label: true },
            RoutingSample { state: BooleanState::new(5), label: false },
        ];
        let config = CreditConfig {
            scope: CreditScope::OutputParameters,
            policy: CreditPolicy::Greedy,
            rounds: 1,
            proposals_per_round: 2,
            seed: 1,
        };
        assert!(matches!(
            select_by_validation(&model, &[], &validation, config, &[1]),
            Err(CreditError::InvalidBudget)
        ));
    }

    #[test]
    fn independent_balanced_score_and_disjoint_validation() {
        let confusion = Confusion { tp: 1, tn: 3, fp: 1, fn_: 3 };
        assert_eq!(confusion.score(), (16, 32));
        let model = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity], LeafRouting::Contiguous
        ).unwrap();
        let mut rows = Vec::new();
        for n in 0..128 {
            let mut state = BooleanState::new(25);
            for bit in 0..25 {
                state.set(bit, (n >> (bit % 7)) & 1 != 0).unwrap();
            }
            rows.push(RoutingSample { state, label: n % 2 == 0 });
        }
        let config = CreditConfig {
            scope: CreditScope::OutputParameters,
            policy: CreditPolicy::Greedy,
            rounds: 1,
            proposals_per_round: 2,
            seed: 1,
        };
        assert!(matches!(
            select_by_validation(&model, &rows[..64], &rows[64..], config, &[1]),
            Err(CreditError::DuplicateInput)
        ));
    }
}

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
    if seeds.is_empty() || seeds.len() > 32 || validation.is_empty() {
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
    for &seed in seeds {
        let config = CreditConfig { seed, ..template };
        let result = fit_final_loss(base, train, config)?;
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
    selected.candidate_evaluations =
        selected.result.candidate_evaluations * seeds.len();
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5_counterfactual_credit::{CreditPolicy, CreditScope};
    use crate::fourier5_hierarchy::{GateKind, LeafRouting};
    use crate::BooleanState;

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

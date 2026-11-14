//! Strict, independent scalar evaluation of frozen, routing-only, readout-only
//! and joint models. No held-out labels can update any candidate.
use std::collections::BTreeSet;

use crate::fourier5_hierarchy::{FiveCircleHierarchy, HierarchyError};
use crate::fourier5_joint_search::JointReport;
use crate::fourier5_routing_search::RoutingSample;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuditError {
    EmptySplit,
    InvalidWidth,
    DuplicateOrLeakedInput,
    Inference(HierarchyError),
}

impl From<HierarchyError> for AuditError {
    fn from(error: HierarchyError) -> Self {
        Self::Inference(error)
    }
}

/// Confusion counts for the positive (true) Boolean output class.
/// This preserves class balance information when OOD prevalence shifts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Confusion {
    pub true_positive: usize,
    pub true_negative: usize,
    pub false_positive: usize,
    pub false_negative: usize,
}

impl Confusion {
    fn observe(&mut self, predicted: bool, expected: bool) {
        match (predicted, expected) {
            (true, true) => self.true_positive += 1,
            (false, false) => self.true_negative += 1,
            (true, false) => self.false_positive += 1,
            (false, true) => self.false_negative += 1,
        }
    }

    #[must_use]
    pub fn correct(self) -> usize {
        self.true_positive + self.true_negative
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitAccuracy {
    pub examples: usize,
    pub frozen_correct: usize,
    pub route_correct: usize,
    pub plastic_correct: usize,
    pub joint_correct: usize,
    pub frozen_confusion: Confusion,
    pub route_confusion: Confusion,
    pub plastic_confusion: Confusion,
    pub joint_confusion: Confusion,
}

/// Evaluate disjoint VALID, TEST and OOD partitions after TRAIN-only fitting.
/// Duplicate input states within or across *all* partitions are rejected;
/// this prevents identical Boolean observations from leaking between splits.
/// Uses scalar inference as an independent oracle from the packed TRAIN scorer.
pub fn audit_disjoint_splits(
    frozen: &FiveCircleHierarchy,
    report: &JointReport,
    train: &[RoutingSample],
    validation: &[RoutingSample],
    test: &[RoutingSample],
    ood: &[RoutingSample],
) -> Result<[SplitAccuracy; 3], AuditError> {
    let width = frozen.input_bits();
    if [train, validation, test, ood]
        .iter()
        .any(|split| split.is_empty())
    {
        return Err(AuditError::EmptySplit);
    }
    if [
        report.route_only.input_bits(),
        report.plastic_only.input_bits(),
        report.joint.input_bits(),
    ]
    .iter()
    .any(|&candidate_width| candidate_width != width)
    {
        return Err(AuditError::InvalidWidth);
    }
    // Equal inference capacity: optimization may change routing and output
    // parameters, but must not change neuron or directed-contact counts.
    let neuron_count = frozen.neuron_count();
    let contacts = frozen.directed_contacts();
    for model in [&report.route_only, &report.plastic_only, &report.joint] {
        if model.neuron_count() != neuron_count || model.directed_contacts() != contacts {
            return Err(AuditError::InvalidWidth);
        }
    }
    let mut seen = BTreeSet::new();
    for row in train.iter().chain(validation).chain(test).chain(ood) {
        if row.state.len() != width {
            return Err(AuditError::InvalidWidth);
        }
        if !seen.insert(row.state.words().to_vec()) {
            return Err(AuditError::DuplicateOrLeakedInput);
        }
    }
    let evaluate = |split: &[RoutingSample]| -> Result<SplitAccuracy, AuditError> {
        let mut scores = SplitAccuracy {
            examples: split.len(),
            frozen_correct: 0,
            route_correct: 0,
            plastic_correct: 0,
            joint_correct: 0,
            frozen_confusion: Confusion::default(),
            route_confusion: Confusion::default(),
            plastic_confusion: Confusion::default(),
            joint_confusion: Confusion::default(),
        };
        for row in split {
            scores
                .frozen_confusion
                .observe(frozen.infer_scalar(&row.state)?, row.label);
            scores
                .route_confusion
                .observe(report.route_only.infer_scalar(&row.state)?, row.label);
            scores
                .plastic_confusion
                .observe(report.plastic_only.infer_scalar(&row.state)?, row.label);
            scores
                .joint_confusion
                .observe(report.joint.infer_scalar(&row.state)?, row.label);
        }
        scores.frozen_correct = scores.frozen_confusion.correct();
        scores.route_correct = scores.route_confusion.correct();
        scores.plastic_correct = scores.plastic_confusion.correct();
        scores.joint_correct = scores.joint_confusion.correct();
        Ok(scores)
    };
    Ok([evaluate(validation)?, evaluate(test)?, evaluate(ood)?])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confusion_counts_all_four_cases_without_class_imbalance_masking() {
        let mut counts = Confusion::default();
        for (prediction, label) in [(true, true), (false, false), (true, false), (false, true)] {
            counts.observe(prediction, label);
        }
        assert_eq!(counts.true_positive, 1);
        assert_eq!(counts.true_negative, 1);
        assert_eq!(counts.false_positive, 1);
        assert_eq!(counts.false_negative, 1);
        assert_eq!(counts.correct(), 2);
    }
}

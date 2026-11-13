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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitAccuracy {
    pub examples: usize,
    pub frozen_correct: usize,
    pub route_correct: usize,
    pub plastic_correct: usize,
    pub joint_correct: usize,
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
    if [train, validation, test, ood].iter().any(|split| split.is_empty()) {
        return Err(AuditError::EmptySplit);
    }
    if [report.route_only.input_bits(), report.plastic_only.input_bits(), report.joint.input_bits()]
        .iter()
        .any(|&candidate_width| candidate_width != width)
    {
        return Err(AuditError::InvalidWidth);
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
        };
        for row in split {
            scores.frozen_correct += usize::from(frozen.infer_scalar(&row.state)? == row.label);
            scores.route_correct += usize::from(report.route_only.infer_scalar(&row.state)? == row.label);
            scores.plastic_correct += usize::from(report.plastic_only.infer_scalar(&row.state)? == row.label);
            scores.joint_correct += usize::from(report.joint.infer_scalar(&row.state)? == row.label);
        }
        Ok(scores)
    };
    Ok([evaluate(validation)?, evaluate(test)?, evaluate(ood)?])
}

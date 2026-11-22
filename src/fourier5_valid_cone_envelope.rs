//! Conservative cone-class admission calibrated exclusively on disjoint VALID measurements.
//! This module does not measure hardware or infer biological behaviour.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidObservation {
    pub sample_id: u64,
    pub cone_class: u32,
    pub gates: u64,
    pub cone_gates: u64,
    pub full_ns: u64,
    pub refresh_ns: u64,
    pub incremental_ns: u64,
    pub oracle_equal: bool,
    pub decision_equal: bool,
    pub vm_hwm_kib: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConeEnvelope {
    pub cone_class: u32,
    pub max_cone_gates: u64,
    pub min_full_ns: u64,
    pub max_incremental_total_ns: u128,
    pub samples: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CalibrationError {
    Empty,
    DuplicateSample,
    InvalidObservation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrozenConePolicy {
    envelopes: BTreeMap<u32, ConeEnvelope>,
}

impl FrozenConePolicy {
    /// Fail closed on any invalid VALID observation. A class is admitted only
    /// when every VALID observation agrees with Full and has a positive margin.
    pub fn calibrate(
        valid: &[ValidObservation],
        min_samples_per_class: usize,
        memory_limit_kib: u64,
    ) -> Result<Self, CalibrationError> {
        if valid.is_empty() || min_samples_per_class == 0 {
            return Err(CalibrationError::Empty);
        }
        let mut seen = BTreeSet::new();
        let mut groups: BTreeMap<u32, Vec<ValidObservation>> = BTreeMap::new();
        for &sample in valid {
            if !seen.insert(sample.sample_id) {
                return Err(CalibrationError::DuplicateSample);
            }
            if sample.gates == 0
                || sample.cone_gates == 0
                || sample.cone_gates > sample.gates
                || sample.full_ns == 0
                || sample.vm_hwm_kib.is_none()
            {
                return Err(CalibrationError::InvalidObservation);
            }
            groups.entry(sample.cone_class).or_default().push(sample);
        }
        let mut envelopes = BTreeMap::new();
        for (cone_class, samples) in groups {
            if samples.len() < min_samples_per_class {
                continue;
            }
            let min_full_ns = samples.iter().map(|s| s.full_ns).min().unwrap_or(0);
            let max_incremental_total_ns = samples
                .iter()
                .map(|s| u128::from(s.refresh_ns) + u128::from(s.incremental_ns))
                .max()
                .unwrap_or(u128::MAX);
            let safe = samples.iter().all(|s| {
                s.oracle_equal
                    && s.decision_equal
                    && s.vm_hwm_kib.is_some_and(|m| m <= memory_limit_kib)
            });
            if safe && max_incremental_total_ns < u128::from(min_full_ns) {
                envelopes.insert(
                    cone_class,
                    ConeEnvelope {
                        cone_class,
                        max_cone_gates: samples
                            .iter()
                            .map(|s| s.cone_gates)
                            .max()
                            .unwrap_or(0),
                        min_full_ns,
                        max_incremental_total_ns,
                        samples: samples.len(),
                    },
                );
            }
        }
        Ok(Self { envelopes })
    }

    /// Runtime routing uses only frozen VALID evidence and current geometry.
    /// No TEST/OOD outcomes or measured timings are fed back into calibration.
    pub fn admits(&self, cone_class: u32, gates: u64, cone_gates: u64) -> bool {
        self.envelopes.get(&cone_class).is_some_and(|e| {
            gates > 0 && cone_gates > 0 && cone_gates <= gates && cone_gates <= e.max_cone_gates
        })
    }

    pub fn envelopes(&self) -> &BTreeMap<u32, ConeEnvelope> {
        &self.envelopes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(id: u64, class: u32) -> ValidObservation {
        ValidObservation {
            sample_id: id,
            cone_class: class,
            gates: 100,
            cone_gates: 10,
            full_ns: 100,
            refresh_ns: 20,
            incremental_ns: 30,
            oracle_equal: true,
            decision_equal: true,
            vm_hwm_kib: Some(1024),
        }
    }

    #[test]
    fn independent_worst_case_oracle_rejects_class_with_one_regression() {
        let mut rows = vec![sample(1, 4), sample(2, 4), sample(3, 9), sample(4, 9)];
        rows[1].refresh_ns = 80;
        let policy = FrozenConePolicy::calibrate(&rows, 2, 2048).unwrap();
        assert!(!policy.admits(4, 100, 10));
        assert!(policy.admits(9, 100, 10));
        assert!(!policy.admits(9, 100, 11));
        assert!(!policy.admits(10, 100, 10));
    }

    #[test]
    fn oracle_disallows_duplicate_provenance_and_missing_memory() {
        assert_eq!(
            FrozenConePolicy::calibrate(&[sample(1, 1), sample(1, 2)], 1, 2048),
            Err(CalibrationError::DuplicateSample)
        );
        let mut invalid = sample(1, 1);
        invalid.vm_hwm_kib = None;
        assert_eq!(
            FrozenConePolicy::calibrate(&[invalid], 1, 2048),
            Err(CalibrationError::InvalidObservation)
        );
    }

    #[test]
    fn oracle_requires_all_decisions_and_memory_within_limit() {
        let mut rows = vec![sample(1, 1), sample(2, 1)];
        rows[1].decision_equal = false;
        assert!(FrozenConePolicy::calibrate(&rows, 2, 2048)
            .unwrap()
            .envelopes()
            .is_empty());
        rows[1].decision_equal = true;
        rows[1].vm_hwm_kib = Some(2049);
        assert!(FrozenConePolicy::calibrate(&rows, 2, 2048)
            .unwrap()
            .envelopes()
            .is_empty());
    }
}

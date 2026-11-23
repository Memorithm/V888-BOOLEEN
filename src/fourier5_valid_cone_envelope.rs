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
    /// Only geometries actually sampled sufficiently on VALID may use Incremental.
    admitted_geometries: BTreeSet<(u32, u64, u64)>,
}

/// Cache generation is advanced by the caller after every accepted mutation.
/// An incremental evaluation is never permitted against a stale generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequentialRoute {
    Full,
    Incremental,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequentialCacheState {
    /// Generation of the committed circuit.
    pub circuit_epoch: u64,
    /// Generation for which the cache was constructed and verified.
    pub cache_epoch: Option<u64>,
}

impl SequentialCacheState {
    /// Accepting a mutation invalidates the previous cache, including when
    /// its topology is unchanged. Overflow fails closed rather than wrapping.
    pub fn accept_mutation(&mut self) {
        self.circuit_epoch = self.circuit_epoch.saturating_add(1);
        self.cache_epoch = None;
    }

    /// Call only after rebuilding the cache from the committed Full state
    /// and checking its output against the independent Full oracle.
    pub fn mark_cache_verified(&mut self) {
        self.cache_epoch = Some(self.circuit_epoch);
    }
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
        let mut admitted_geometries = BTreeSet::new();
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
                let mut geometry_counts = BTreeMap::<(u64, u64), usize>::new();
                for sample in &samples {
                    *geometry_counts.entry((sample.gates, sample.cone_gates)).or_default() += 1;
                }
                for ((gates, cone_gates), count) in geometry_counts {
                    if count >= min_samples_per_class {
                        admitted_geometries.insert((cone_class, gates, cone_gates));
                    }
                }
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
        Ok(Self {
            envelopes,
            admitted_geometries,
        })
    }

    /// Calibrate each exact VALID geometry independently, avoiding the class-wide
    /// penalty when another geometry in the same cone class regresses.
    /// TEST/OOD observations must never be supplied to this method.
    pub fn calibrate_stratified(
        valid: &[ValidObservation],
        min_samples_per_geometry: usize,
        memory_limit_kib: u64,
        minimum_saving_ns: u64,
    ) -> Result<Self, CalibrationError> {
        if valid.is_empty() || min_samples_per_geometry == 0 {
            return Err(CalibrationError::Empty);
        }
        let mut seen = BTreeSet::new();
        let mut groups: BTreeMap<(u32, u64, u64), Vec<ValidObservation>> = BTreeMap::new();
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
            groups
                .entry((sample.cone_class, sample.gates, sample.cone_gates))
                .or_default()
                .push(sample);
        }
        let mut envelopes: BTreeMap<u32, ConeEnvelope> = BTreeMap::new();
        let mut admitted_geometries = BTreeSet::new();
        for ((cone_class, gates, cone_gates), samples) in groups {
            if samples.len() < min_samples_per_geometry {
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
            if !safe
                || max_incremental_total_ns + u128::from(minimum_saving_ns)
                    >= u128::from(min_full_ns)
            {
                continue;
            }
            admitted_geometries.insert((cone_class, gates, cone_gates));
            let entry = envelopes.entry(cone_class).or_insert(ConeEnvelope {
                cone_class,
                max_cone_gates: 0,
                min_full_ns: u64::MAX,
                max_incremental_total_ns: 0,
                samples: 0,
            });
            entry.max_cone_gates = entry.max_cone_gates.max(cone_gates);
            entry.min_full_ns = entry.min_full_ns.min(min_full_ns);
            entry.max_incremental_total_ns =
                entry.max_incremental_total_ns.max(max_incremental_total_ns);
            entry.samples += samples.len();
        }
        Ok(Self {
            envelopes,
            admitted_geometries,
        })
    }

    /// Sequential dispatch is safe only for a cache verified for the current
    /// committed circuit generation and a geometry admitted on frozen VALID.
    /// Full is the mandatory fallback; this method does not rebuild caches.
    pub fn route_sequential(
        &self,
        state: SequentialCacheState,
        cone_class: u32,
        gates: u64,
        cone_gates: u64,
    ) -> SequentialRoute {
        if state.cache_epoch == Some(state.circuit_epoch)
            && self.admits(cone_class, gates, cone_gates)
        {
            SequentialRoute::Incremental
        } else {
            SequentialRoute::Full
        }
    }

    /// Runtime routing uses only frozen VALID evidence and current geometry.
    /// No TEST/OOD outcomes or measured timings are fed back into calibration.
    pub fn admits(&self, cone_class: u32, gates: u64, cone_gates: u64) -> bool {
        self.envelopes.contains_key(&cone_class)
            && self
                .admitted_geometries
                .contains(&(cone_class, gates, cone_gates))
    }

    pub fn envelopes(&self) -> &BTreeMap<u32, ConeEnvelope> {
        &self.envelopes
    }
}


/// A measured sequential intervention, including cache rebuild/refresh cost.
/// The caller supplies timings and VmHWM from a bounded RemoteOps run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequentialMeasurement {
    pub sample_id: u64,
    pub cone_class: u32,
    pub gates: u64,
    pub cone_gates: u64,
    pub full_ns: u64,
    pub incremental_ns: u64,
    pub rebuild_ns: u64,
    pub vm_hwm_kib: Option<u64>,
    pub oracle_equal: bool,
    pub decision_equal: bool,
}

/// Aggregated observed costs for a heterogeneous sequential workload.
/// No synthetic timings are generated by this aggregator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequentialSummary {
    pub interventions: usize,
    pub full_total_ns: u128,
    pub incremental_total_ns: u128,
    pub peak_vm_hwm_kib: u64,
    pub oracle_equal: bool,
    pub decision_equal: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequentialMeasureError {
    Empty,
    DuplicateSample,
    InvalidMeasurement,
}

/// Summarize a disjoint measured sequence without using TEST/OOD for calibration.
/// All rebuild costs are charged to Incremental, including after accepted mutations.
pub fn summarize_sequential(
    measurements: &[SequentialMeasurement],
) -> Result<SequentialSummary, SequentialMeasureError> {
    if measurements.is_empty() {
        return Err(SequentialMeasureError::Empty);
    }
    let mut seen = BTreeSet::new();
    let mut result = SequentialSummary {
        interventions: 0,
        full_total_ns: 0,
        incremental_total_ns: 0,
        peak_vm_hwm_kib: 0,
        oracle_equal: true,
        decision_equal: true,
    };
    for m in measurements {
        if !seen.insert(m.sample_id) {
            return Err(SequentialMeasureError::DuplicateSample);
        }
        if m.gates == 0 || m.cone_gates == 0 || m.cone_gates > m.gates
            || m.full_ns == 0 || m.vm_hwm_kib.is_none()
        {
            return Err(SequentialMeasureError::InvalidMeasurement);
        }
        result.interventions += 1;
        result.full_total_ns += u128::from(m.full_ns);
        result.incremental_total_ns += u128::from(m.rebuild_ns) + u128::from(m.incremental_ns);
        result.peak_vm_hwm_kib = result.peak_vm_hwm_kib.max(m.vm_hwm_kib.unwrap_or(0));
        result.oracle_equal &= m.oracle_equal;
        result.decision_equal &= m.decision_equal;
    }
    Ok(result)
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
    fn sequential_heterogeneous_oracle_charges_every_rebuild_and_reports_regression() {
        let rows = [
            SequentialMeasurement { sample_id: 1, cone_class: 1, gates: 100, cone_gates: 5,
                full_ns: 100, incremental_ns: 20, rebuild_ns: 30,
                vm_hwm_kib: Some(1000), oracle_equal: true, decision_equal: true },
            SequentialMeasurement { sample_id: 2, cone_class: 2, gates: 1000, cone_gates: 200,
                full_ns: 200, incremental_ns: 150, rebuild_ns: 100,
                vm_hwm_kib: Some(2000), oracle_equal: true, decision_equal: false },
        ];
        let summary = summarize_sequential(&rows).unwrap();
        assert_eq!(summary.interventions, 2);
        assert_eq!(summary.full_total_ns, 300);
        assert_eq!(summary.incremental_total_ns, 300);
        assert_eq!(summary.peak_vm_hwm_kib, 2000);
        assert!(summary.oracle_equal);
        assert!(!summary.decision_equal);
        assert_eq!(summarize_sequential(&[rows[0], rows[0]]),
            Err(SequentialMeasureError::DuplicateSample));
        let mut bad = rows[0];
        bad.vm_hwm_kib = None;
        assert_eq!(summarize_sequential(&[bad]),
            Err(SequentialMeasureError::InvalidMeasurement));
    }

    #[test]
    fn stratified_oracle_isolates_regressing_geometry_and_enforces_margin() {
        let mut rows = vec![sample(1, 4), sample(2, 4), sample(3, 4), sample(4, 4)];
        rows[2].gates = 200;
        rows[2].cone_gates = 20;
        rows[3].gates = 200;
        rows[3].cone_gates = 20;
        rows[3].refresh_ns = 90;
        let old = FrozenConePolicy::calibrate(&rows, 2, 2048).unwrap();
        assert!(!old.admits(4, 100, 10));
        let policy = FrozenConePolicy::calibrate_stratified(&rows, 2, 2048, 49).unwrap();
        assert!(policy.admits(4, 100, 10));
        assert!(!policy.admits(4, 200, 20));
        assert!(!policy.admits(4, 150, 15));
        let boundary = FrozenConePolicy::calibrate_stratified(&rows, 2, 2048, 50).unwrap();
        assert!(!boundary.admits(4, 100, 10));
        rows[1].decision_equal = false;
        let unsafe_policy = FrozenConePolicy::calibrate_stratified(&rows, 2, 2048, 0).unwrap();
        assert!(!unsafe_policy.admits(4, 100, 10));
    }

    #[test]
    fn sequential_cache_epoch_oracle_never_reuses_stale_cache() {
        let policy = FrozenConePolicy::calibrate(&[sample(1, 3), sample(2, 3)], 2, 2048)
            .unwrap();
        let mut state = SequentialCacheState {
            circuit_epoch: 7,
            cache_epoch: None,
        };
        assert_eq!(policy.route_sequential(state, 3, 100, 10), SequentialRoute::Full);
        state.mark_cache_verified();
        assert_eq!(policy.route_sequential(state, 3, 100, 10), SequentialRoute::Incremental);
        assert_eq!(policy.route_sequential(state, 3, 100, 11), SequentialRoute::Full);
        state.accept_mutation();
        assert_eq!(state.circuit_epoch, 8);
        assert_eq!(policy.route_sequential(state, 3, 100, 10), SequentialRoute::Full);
        state.mark_cache_verified();
        assert_eq!(policy.route_sequential(state, 3, 100, 10), SequentialRoute::Incremental);
    }

    #[test]
    fn sequential_epoch_overflow_fails_closed() {
        let policy = FrozenConePolicy::calibrate(&[sample(1, 3), sample(2, 3)], 2, 2048)
            .unwrap();
        let mut state = SequentialCacheState {
            circuit_epoch: u64::MAX,
            cache_epoch: Some(u64::MAX),
        };
        state.accept_mutation();
        assert_eq!(state.cache_epoch, None);
        assert_eq!(policy.route_sequential(state, 3, 100, 10), SequentialRoute::Full);
    }

    #[test]
    fn oracle_requires_repeated_exact_valid_geometry_not_interpolation() {
        let rows = [
            sample(1, 7),
            sample(2, 7),
            ValidObservation { sample_id: 3, gates: 1000, cone_gates: 20, ..sample(3, 7) },
            ValidObservation { sample_id: 4, gates: 1000, cone_gates: 20, ..sample(4, 7) },
            ValidObservation { sample_id: 5, gates: 500, cone_gates: 15, ..sample(5, 7) },
        ];
        let policy = FrozenConePolicy::calibrate(&rows, 2, 2048).unwrap();
        for gates in [99, 100, 101, 500, 999, 1000, 1001] {
            for cone in [9, 10, 15, 20, 21] {
                let expected = (gates == 100 && cone == 10)
                    || (gates == 1000 && cone == 20);
                assert_eq!(policy.admits(7, gates, cone), expected);
            }
        }
        assert!(!policy.admits(8, 100, 10));
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

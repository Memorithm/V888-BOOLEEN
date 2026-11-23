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
    pub min_gates: u64,
    pub max_gates: u64,
    pub min_full_ns: u64,
    pub max_incremental_total_ns: u128,
    pub samples: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CalibrationError {
    Empty,
    DuplicateSample,
    InvalidObservation,
    InvalidMargin,
    GenerationExhausted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrozenConePolicy {
    envelopes: BTreeMap<u32, ConeEnvelope>,
    observed_geometry: BTreeMap<u32, BTreeSet<(u64, u64)>>,
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
        let mut observed_geometry = BTreeMap::new();
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
                let mut counts = BTreeMap::new();
                for sample in &samples {
                    *counts.entry((sample.gates, sample.cone_gates)).or_insert(0usize) += 1;
                }
                let supported = counts
                    .into_iter()
                    .filter_map(|(geometry, count)| {
                        (count >= min_samples_per_class).then_some(geometry)
                    })
                    .collect();
                observed_geometry.insert(cone_class, supported);
                envelopes.insert(
                    cone_class,
                    ConeEnvelope {
                        cone_class,
                        max_cone_gates: samples
                            .iter()
                            .map(|s| s.cone_gates)
                            .max()
                            .unwrap_or(0),
                        min_gates: samples.iter().map(|s| s.gates).min().unwrap_or(0),
                        max_gates: samples.iter().map(|s| s.gates).max().unwrap_or(0),
                        min_full_ns,
                        max_incremental_total_ns,
                        samples: samples.len(),
                    },
                );
            }
        }
        Ok(Self {
            envelopes,
            observed_geometry,
        })
    }

    /// Calibrate each exact geometry independently. A slow geometry must not
    /// suppress a distinct, repeatedly measured safe geometry in the same class.
    /// All observations remain VALID-only; no TEST/OOD feedback is consumed.
    pub fn calibrate_stratified(
        valid: &[ValidObservation],
        min_samples_per_geometry: usize,
        memory_limit_kib: u64,
    ) -> Result<Self, CalibrationError> {
        Self::calibrate_stratified_with_margin(valid, min_samples_per_geometry, memory_limit_kib, 0)
    }

    /// Freeze a worst-case VALID admission margin per exact geometry.
    /// margin_percent is a minimum fractional saving relative to the fastest
    /// Full observation, including the entire Incremental cache refresh cost.
    /// This policy is frozen before TEST/OOD and never learns from runtime outcomes.
    pub fn calibrate_stratified_with_margin(
        valid: &[ValidObservation],
        min_samples_per_geometry: usize,
        memory_limit_kib: u64,
        margin_percent: u8,
    ) -> Result<Self, CalibrationError> {
        if margin_percent > 99 {
            return Err(CalibrationError::InvalidMargin);
        }
        if valid.is_empty() || min_samples_per_geometry == 0 {
            return Err(CalibrationError::Empty);
        }
        let mut seen = BTreeSet::new();
        let mut strata: BTreeMap<(u32, u64, u64), Vec<ValidObservation>> = BTreeMap::new();
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
            strata
                .entry((sample.cone_class, sample.gates, sample.cone_gates))
                .or_default()
                .push(sample);
        }

        let mut envelopes: BTreeMap<u32, ConeEnvelope> = BTreeMap::new();
        let mut observed_geometry: BTreeMap<u32, BTreeSet<(u64, u64)>> = BTreeMap::new();
        for ((class, gates, cone), rows) in strata {
            if rows.len() < min_samples_per_geometry {
                continue;
            }
            let min_full = rows.iter().map(|r| r.full_ns).min().unwrap_or(0);
            let max_incremental = rows
                .iter()
                .map(|r| u128::from(r.refresh_ns) + u128::from(r.incremental_ns))
                .max()
                .unwrap_or(u128::MAX);
            // Checked arithmetic: overflow must reject, never admit.
            let conservative_cost = max_incremental.checked_mul(100);
            let permitted_cost = u128::from(min_full)
                .checked_mul(u128::from(100 - margin_percent));
            if !matches!((conservative_cost, permitted_cost), (Some(cost), Some(limit)) if cost < limit)
                || rows.iter().any(|r| {
                    !r.oracle_equal
                        || !r.decision_equal
                        || !r.vm_hwm_kib.is_some_and(|m| m <= memory_limit_kib)
                })
            {
                continue;
            }
            observed_geometry.entry(class).or_default().insert((gates, cone));
            let envelope = envelopes.entry(class).or_insert(ConeEnvelope {
                cone_class: class,
                max_cone_gates: cone,
                min_gates: gates,
                max_gates: gates,
                min_full_ns: min_full,
                max_incremental_total_ns: max_incremental,
                samples: 0,
            });
            envelope.max_cone_gates = envelope.max_cone_gates.max(cone);
            envelope.min_gates = envelope.min_gates.min(gates);
            envelope.max_gates = envelope.max_gates.max(gates);
            envelope.min_full_ns = envelope.min_full_ns.min(min_full);
            envelope.max_incremental_total_ns =
                envelope.max_incremental_total_ns.max(max_incremental);
            envelope.samples += rows.len();
        }
        Ok(Self { envelopes, observed_geometry })
    }

    /// Require independent VALID seeds for every admitted geometry. Repeated
    /// measurements of one seed cannot masquerade as independent support.
    /// The ordinary stratified safety and latency oracle is applied first.
    pub fn calibrate_seed_replicated(
        valid: &[(u64, ValidObservation)],
        min_distinct_seeds: usize,
        memory_limit_kib: u64,
        margin_percent: u8,
    ) -> Result<Self, CalibrationError> {
        if min_distinct_seeds == 0 || valid.is_empty() {
            return Err(CalibrationError::Empty);
        }
        let mut seeds: BTreeMap<(u32, u64, u64), BTreeSet<u64>> = BTreeMap::new();
        let mut rows = Vec::with_capacity(valid.len());
        for &(seed, observation) in valid {
            seeds.entry((observation.cone_class, observation.gates, observation.cone_gates))
                .or_default()
                .insert(seed);
            rows.push(observation);
        }
        let mut policy = Self::calibrate_stratified_with_margin(
            &rows, min_distinct_seeds, memory_limit_kib, margin_percent,
        )?;
        for (&class, geometry) in &mut policy.observed_geometry {
            geometry.retain(|&(gates, cone)| {
                seeds.get(&(class, gates, cone))
                    .is_some_and(|unique| unique.len() >= min_distinct_seeds)
            });
        }
        policy.envelopes.retain(|class, _| {
            policy.observed_geometry.get(class).is_some_and(|g| !g.is_empty())
        });
        Ok(policy)
    }

    /// Only exact geometry pairs observed during VALID are admitted; an interval
    /// between measured sizes is not evidence of safe incremental execution.
    /// Runtime routing uses only frozen VALID evidence and current geometry.
    /// No TEST/OOD outcomes or measured timings are fed back into calibration.
    pub fn admits(&self, cone_class: u32, gates: u64, cone_gates: u64) -> bool {
        self.envelopes.get(&cone_class).is_some_and(|e| {
            gates >= e.min_gates
                && gates <= e.max_gates
                && cone_gates > 0
                && cone_gates <= gates
                && cone_gates <= e.max_cone_gates
                && self.observed_geometry.get(&cone_class).is_some_and(|geometry| {
                    geometry.contains(&(gates, cone_gates))
                })
        })
    }

    /// Select a route for a sequential intervention. An accepted mutation makes
    /// every previously computed cache generation stale; only a completed
    /// rebuild for the current circuit generation permits Incremental.
    /// The caller must acknowledge cache rebuild only after independent Full
    /// equality verification. This function never repairs a stale cache.
    pub fn route_sequential(
        &self,
        cone_class: u32,
        gates: u64,
        cone_gates: u64,
        circuit_generation: u64,
        verified_cache_generation: Option<u64>,
    ) -> SequentialRoute {
        if verified_cache_generation == Some(circuit_generation)
            && self.admits(cone_class, gates, cone_gates)
        {
            SequentialRoute::Incremental
        } else {
            SequentialRoute::Full
        }
    }

    pub fn envelopes(&self) -> &BTreeMap<u32, ConeEnvelope> {
        &self.envelopes
    }
}

/// Conservative dispatch decision, independent of the runtime engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequentialRoute {
    Full,
    Incremental,
}

/// Transactional cache validity for successive accepted circuit mutations.
/// This tracks verification; it does not reconstruct the engine cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequentialCacheTracker {
    generation: u64,
    verified_generation: Option<u64>,
    exhausted: bool,
}

impl SequentialCacheTracker {
    pub fn new() -> Self {
        Self { generation: 0, verified_generation: None, exhausted: false }
    }

    /// A mutation is accepted only when the generation can advance.
    /// On overflow, permanently fail closed instead of reusing a cache token.
    pub fn accept_mutation(&mut self) -> Result<u64, CalibrationError> {
        self.verified_generation = None;
        match self.generation.checked_add(1) {
            Some(next) if !self.exhausted => {
                self.generation = next;
                Ok(next)
            }
            _ => {
                self.exhausted = true;
                Err(CalibrationError::GenerationExhausted)
            }
        }
    }

    /// A rebuild is trusted only after an independent Full equality check.
    /// A failed comparison invalidates any previous verified generation.
    pub fn confirm_rebuild(&mut self, oracle_equal: bool) -> bool {
        self.verified_generation = None;
        if oracle_equal && !self.exhausted {
            self.verified_generation = Some(self.generation);
            true
        } else {
            false
        }
    }

    pub fn route(
        &self,
        policy: &FrozenConePolicy,
        cone_class: u32,
        gates: u64,
        cone_gates: u64,
    ) -> SequentialRoute {
        if self.exhausted {
            SequentialRoute::Full
        } else {
            policy.route_sequential(
                cone_class, gates, cone_gates, self.generation, self.verified_generation,
            )
        }
    }
}

impl Default for SequentialCacheTracker {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transactional_cache_tracker_oracle_after_sequential_mutations() {
        let policy = FrozenConePolicy::calibrate_stratified(
            &[sample(1, 61), sample(2, 61)], 2, 2048,
        ).unwrap();
        let mut tracker = SequentialCacheTracker::new();
        assert_eq!(tracker.route(&policy, 61, 100, 10), SequentialRoute::Full);
        assert!(tracker.confirm_rebuild(true));
        assert_eq!(tracker.route(&policy, 61, 100, 10), SequentialRoute::Incremental);
        for generation in 1..=4 {
            assert_eq!(tracker.accept_mutation(), Ok(generation));
            assert_eq!(tracker.route(&policy, 61, 100, 10), SequentialRoute::Full);
            assert!(!tracker.confirm_rebuild(false));
            assert_eq!(tracker.route(&policy, 61, 100, 10), SequentialRoute::Full);
            assert!(tracker.confirm_rebuild(true));
            assert_eq!(tracker.route(&policy, 61, 100, 10), SequentialRoute::Incremental);
            assert_eq!(tracker.route(&policy, 61, 100, 11), SequentialRoute::Full);
        }
    }

    #[test]
    fn transactional_cache_tracker_fails_closed_on_generation_overflow() {
        let policy = FrozenConePolicy::calibrate_stratified(
            &[sample(1, 62), sample(2, 62)], 2, 2048,
        ).unwrap();
        let mut tracker = SequentialCacheTracker {
            generation: u64::MAX, verified_generation: Some(u64::MAX), exhausted: false,
        };
        assert_eq!(tracker.accept_mutation(), Err(CalibrationError::GenerationExhausted));
        assert!(!tracker.confirm_rebuild(true));
        assert_eq!(tracker.route(&policy, 62, 100, 10), SequentialRoute::Full);
        assert_eq!(tracker.accept_mutation(), Err(CalibrationError::GenerationExhausted));
    }


    #[test]
    fn independent_seed_replicate_oracle() {
        let repeated = [(7, sample(1, 52)), (7, sample(2, 52))];
        let policy = FrozenConePolicy::calibrate_seed_replicated(&repeated, 2, 2048, 0).unwrap();
        assert!(!policy.admits(52, 100, 10));
        let distinct = [(7, sample(1, 52)), (8, sample(2, 52))];
        let policy = FrozenConePolicy::calibrate_seed_replicated(&distinct, 2, 2048, 0).unwrap();
        assert!(policy.admits(52, 100, 10));
        let mut divergent = sample(2, 52);
        divergent.decision_equal = false;
        let policy = FrozenConePolicy::calibrate_seed_replicated(
            &[(7, sample(1, 52)), (8, divergent)], 2, 2048, 0,
        ).unwrap();
        assert!(!policy.admits(52, 100, 10));
    }

    #[test]
    fn sequential_route_requires_current_verified_cache_and_valid_geometry() {
        let rows = [sample(1, 44), sample(2, 44)];
        let policy = FrozenConePolicy::calibrate_stratified(&rows, 2, 2048).unwrap();
        // Independent oracle: only exact generation equality AND admitted
        // geometry can route Incremental. Every other case is Full.
        for circuit_generation in [0_u64, 1, 9, u64::MAX] {
            for cache_generation in [None, Some(0), Some(1), Some(9), Some(u64::MAX)] {
                for cone in [0_u64, 10, 11] {
                    let expected = cache_generation == Some(circuit_generation) && cone == 10;
                    assert_eq!(
                        policy.route_sequential(44, 100, cone, circuit_generation, cache_generation),
                        if expected { SequentialRoute::Incremental } else { SequentialRoute::Full },
                    );
                }
            }
        }
        assert_eq!(policy.route_sequential(44, 101, 10, 9, Some(9)), SequentialRoute::Full);
    }

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
    fn frozen_margin_oracle_checks_worst_case_per_geometry() {
        let mut rows = vec![sample(1, 31), sample(2, 31)];
        rows[0].refresh_ns = 10;
        rows[0].incremental_ns = 40;
        rows[1].refresh_ns = 20;
        rows[1].incremental_ns = 40;
        // Independent integer oracle: worst 60ns, fastest Full 100ns.
        for margin in [0u8, 20, 39, 40, 41, 99] {
            let policy = FrozenConePolicy::calibrate_stratified_with_margin(&rows, 2, 2048, margin).unwrap();
            assert_eq!(policy.admits(31, 100, 10), margin < 40, "margin={margin}");
        }
        assert_eq!(
            FrozenConePolicy::calibrate_stratified_with_margin(&rows, 2, 2048, 100),
            Err(CalibrationError::InvalidMargin)
        );
    }

    #[test]
    fn margin_oracle_rejects_full_timing_tail_and_overflow() {
        let mut rows = vec![sample(1, 32), sample(2, 32)];
        rows[1].full_ns = 70;
        rows[1].refresh_ns = 20;
        rows[1].incremental_ns = 40;
        assert!(!FrozenConePolicy::calibrate_stratified_with_margin(&rows, 2, 2048, 20)
            .unwrap().admits(32, 100, 10));
        assert!(FrozenConePolicy::calibrate_stratified_with_margin(&rows, 2, 2048, 10)
            .unwrap().admits(32, 100, 10));
        rows[1].incremental_ns = u64::MAX;
        assert!(!FrozenConePolicy::calibrate_stratified_with_margin(&rows, 2, 2048, 0)
            .unwrap().admits(32, 100, 10));
    }

    #[test]
    fn stratified_oracle_preserves_safe_geometry_despite_slow_neighbor() {
        let mut rows = Vec::new();
        for (id, cone, refresh) in [(1, 4, 10), (2, 4, 20), (3, 8, 90), (4, 8, 90)] {
            let mut r = sample(id, 7);
            r.cone_gates = cone;
            r.refresh_ns = refresh;
            rows.push(r);
        }
        let policy = FrozenConePolicy::calibrate_stratified(&rows, 2, 2048).unwrap();
        assert!(policy.admits(7, 100, 4));
        assert!(!policy.admits(7, 100, 8));
        assert!(!policy.admits(7, 100, 6));
        assert!(!FrozenConePolicy::calibrate(&rows, 2, 2048)
            .unwrap()
            .admits(7, 100, 4));
    }

    #[test]
    fn stratified_oracle_fails_closed_on_bad_provenance_and_decisions() {
        let mut rows = vec![sample(1, 3), sample(2, 3)];
        rows[1].decision_equal = false;
        assert!(!FrozenConePolicy::calibrate_stratified(&rows, 2, 2048)
            .unwrap()
            .admits(3, 100, 10));
        rows[1].decision_equal = true;
        rows[1].vm_hwm_kib = Some(2049);
        assert!(!FrozenConePolicy::calibrate_stratified(&rows, 2, 2048)
            .unwrap()
            .admits(3, 100, 10));
        rows[1].vm_hwm_kib = Some(1024);
        rows[1].sample_id = 1;
        assert_eq!(
            FrozenConePolicy::calibrate_stratified(&rows, 2, 2048),
            Err(CalibrationError::DuplicateSample)
        );
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
    fn oracle_rejects_unseen_circuit_sizes_even_for_known_cone_class() {
        let mut low = sample(1, 7);
        low.gates = 100;
        let mut high = sample(2, 7);
        high.gates = 200;
        let policy = FrozenConePolicy::calibrate(&[low, high], 1, 2048).unwrap();
        // Independent range oracle: only observed circuit-size interval is admissible.
        for gates in [0, 1, 99, 100, 150, 200, 201, 1_000_000] {
            let expected = gates == 100 || gates == 200;
            assert_eq!(policy.admits(7, gates, 10), expected, "gates={gates}");
        }
        assert!(!policy.admits(7, 200, 11));
        assert!(!policy.admits(7, 150, 10));
    }

    #[test]
    fn independent_exact_geometry_oracle_rejects_unmeasured_cones() {
        let mut first = sample(1, 12);
        first.cone_gates = 4;
        let mut second = sample(2, 12);
        second.cone_gates = 8;
        let policy = FrozenConePolicy::calibrate(&[first, second], 1, 2048).unwrap();
        for cone in 0..=12 {
            let expected = cone == 4 || cone == 8;
            assert_eq!(policy.admits(12, 100, cone), expected, "cone={cone}");
        }
        assert!(!policy.admits(12, 101, 4));
    }

    #[test]
    fn independent_geometry_support_oracle() {
        let mut rows = Vec::new();
        for (id, cone) in [(1, 4), (2, 4), (3, 8), (4, 10), (5, 10)] {
            let mut observation = sample(id, 15);
            observation.cone_gates = cone;
            rows.push(observation);
        }
        let policy = FrozenConePolicy::calibrate(&rows, 2, 2048).unwrap();
        for cone in 0..=12 {
            let expected = cone == 4 || cone == 10;
            assert_eq!(policy.admits(15, 100, cone), expected, "cone={cone}");
        }
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

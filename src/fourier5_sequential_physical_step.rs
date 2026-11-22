//! Physical paired timing for sequential synthetic circuit interventions.
//! The caller owns the circuit mutation. A fresh cache must be built before
//! incremental evaluation; the independent Full result is always returned.
//! Measurements are observations, never a substitute for split provenance.
use crate::fourier5_sequential_cost_policy::SequentialCost;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeError {
    CacheBuild,
    FullEvaluation,
    IncrementalEvaluation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairedStep<T> {
    pub verified_output: T,
    pub cost: SequentialCost,
    /// Linux process-wide peak resident memory, in KiB; None elsewhere.
    pub vm_hwm_kib: Option<u64>,
}

fn elapsed_ns(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos())
        .unwrap_or(u64::MAX)
        .max(1)
}

/// Read process-wide VmHWM. This is not per-step memory allocation.
#[must_use]
pub fn linux_vm_hwm_kib() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
        let mut parts = line.split_whitespace();
        if parts.next()? != "VmHWM:" {
            return None;
        }
        let value = parts.next()?.parse::<u64>().ok()?;
        (parts.next()? == "kB").then_some(value)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Run a measured paired step on the *already mutated* circuit.
/// Cache rebuild is measured, then both engines run on the same state.
/// A disagreement never escapes as an accepted incremental output.
/// Timings are order-sensitive: use balanced alternating orders in the bench.
pub fn measure_sequential_step<T: Eq>(
    gates: u64,
    cone_gates: u64,
    rebuild_cache: impl FnOnce() -> Result<(), ProbeError>,
    full: impl FnOnce() -> Result<T, ProbeError>,
    incremental: impl FnOnce() -> Result<T, ProbeError>,
    same_decision: impl FnOnce(&T, &T) -> bool,
) -> Result<PairedStep<T>, ProbeError> {
    if gates == 0 || cone_gates > gates {
        return Err(ProbeError::FullEvaluation);
    }
    let start = Instant::now();
    rebuild_cache().map_err(|_| ProbeError::CacheBuild)?;
    let cache_update_ns = elapsed_ns(start);

    let start = Instant::now();
    let reference = full().map_err(|_| ProbeError::FullEvaluation)?;
    let full_ns = elapsed_ns(start);

    let start = Instant::now();
    let candidate = incremental().map_err(|_| ProbeError::IncrementalEvaluation)?;
    let incremental_ns = elapsed_ns(start);
    let same_output = candidate == reference;
    let decision_equal = same_decision(&reference, &candidate);
    Ok(PairedStep {
        verified_output: reference,
        cost: SequentialCost {
            gates,
            cone_gates,
            full_ns,
            incremental_ns,
            cache_update_ns,
            same_output,
            same_decision: decision_equal,
        },
        vm_hwm_kib: linux_vm_hwm_kib(),
    })
}

/// Balanced paired measurement: alternate which engine runs first to reduce
/// systematic order bias. Cache construction always precedes both evaluations.
/// The verified result is always Full, regardless of evaluation order.
pub fn measure_sequential_step_balanced<T: Eq>(
    gates: u64,
    cone_gates: u64,
    incremental_first: bool,
    rebuild_cache: impl FnOnce() -> Result<(), ProbeError>,
    full: impl FnOnce() -> Result<T, ProbeError>,
    incremental: impl FnOnce() -> Result<T, ProbeError>,
    same_decision: impl FnOnce(&T, &T) -> bool,
) -> Result<PairedStep<T>, ProbeError> {
    if gates == 0 || cone_gates > gates {
        return Err(ProbeError::FullEvaluation);
    }
    let start = Instant::now();
    rebuild_cache().map_err(|_| ProbeError::CacheBuild)?;
    let cache_update_ns = elapsed_ns(start);
    let (reference, full_ns, candidate, incremental_ns) = if incremental_first {
        let start = Instant::now();
        let candidate = incremental().map_err(|_| ProbeError::IncrementalEvaluation)?;
        let incremental_ns = elapsed_ns(start);
        let start = Instant::now();
        let reference = full().map_err(|_| ProbeError::FullEvaluation)?;
        (reference, elapsed_ns(start), candidate, incremental_ns)
    } else {
        let start = Instant::now();
        let reference = full().map_err(|_| ProbeError::FullEvaluation)?;
        let full_ns = elapsed_ns(start);
        let start = Instant::now();
        let candidate = incremental().map_err(|_| ProbeError::IncrementalEvaluation)?;
        (reference, full_ns, candidate, elapsed_ns(start))
    };
    let same_output = reference == candidate;
    let decision_equal = same_decision(&reference, &candidate);
    Ok(PairedStep {
        verified_output: reference,
        cost: SequentialCost {
            gates,
            cone_gates,
            full_ns,
            incremental_ns,
            cache_update_ns,
            same_output,
            same_decision: decision_equal,
        },
        vm_hwm_kib: linux_vm_hwm_kib(),
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequentialBatch<T> {
    pub steps: Vec<PairedStep<T>>,
    /// Total wall-clock duration, including per-step cache rebuilds and probes.
    pub elapsed_ns: u64,
    /// Sum of measured Full timings, checked for overflow.
    pub full_ns: u64,
    /// Sum of measured Incremental timings, checked for overflow.
    pub incremental_ns: u64,
    /// Sum of measured cache rebuild timings, checked for overflow.
    pub cache_update_ns: u64,
    /// Number of steps whose output or decision disagreed with the Full oracle.
    pub disagreements: usize,
    /// Process-wide high-water mark at the end of the batch, not per-step usage.
    pub vm_hwm_kib: Option<u64>,
}

impl<T> SequentialBatch<T> {
    /// Observed throughput in steps per second, including rebuild and both engines.
    /// Returns None for an empty batch or an unrepresentable duration.
    #[must_use]
    pub fn steps_per_second(&self) -> Option<f64> {
        if self.steps.is_empty() || self.elapsed_ns == 0 {
            return None;
        }
        Some(self.steps.len() as f64 * 1_000_000_000.0 / self.elapsed_ns as f64)
    }
}

/// Measure a heterogeneous sequence with alternating engine order. The caller
/// must mutate its circuit to the next accepted state before each callback.
/// Every step rebuilds its cache and checks the independent Full oracle.
/// No policy is trained on these observations; split isolation is external.
/// A failed step returns an error rather than a partial successful batch.
pub fn measure_sequential_batch<T: Eq>(
    geometries: &[(u64, u64)],
    mut rebuild_cache: impl FnMut(usize) -> Result<(), ProbeError>,
    mut full: impl FnMut(usize) -> Result<T, ProbeError>,
    mut incremental: impl FnMut(usize) -> Result<T, ProbeError>,
    mut same_decision: impl FnMut(&T, &T) -> bool,
) -> Result<SequentialBatch<T>, ProbeError> {
    // Validate the entire planned sequence before any callback can mutate state.
    // A late malformed geometry must not leave a partially applied workload.
    if geometries
        .iter()
        .any(|&(gates, cone)| gates == 0 || cone > gates)
    {
        return Err(ProbeError::FullEvaluation);
    }
    let batch_start = Instant::now();
    let mut steps = Vec::with_capacity(geometries.len());
    let (mut full_ns, mut incremental_ns, mut cache_update_ns) = (0_u64, 0_u64, 0_u64);
    let mut disagreements = 0_usize;
    for (index, &(gates, cone_gates)) in geometries.iter().enumerate() {
        let step = measure_sequential_step_balanced(
            gates,
            cone_gates,
            index % 2 == 1,
            || rebuild_cache(index),
            || full(index),
            || incremental(index),
            |a, b| same_decision(a, b),
        )?;
        full_ns = full_ns
            .checked_add(step.cost.full_ns)
            .ok_or(ProbeError::FullEvaluation)?;
        incremental_ns = incremental_ns
            .checked_add(step.cost.incremental_ns)
            .ok_or(ProbeError::IncrementalEvaluation)?;
        cache_update_ns = cache_update_ns
            .checked_add(step.cost.cache_update_ns)
            .ok_or(ProbeError::CacheBuild)?;
        if !step.cost.same_output || !step.cost.same_decision {
            disagreements += 1;
        }
        steps.push(step);
    }
    Ok(SequentialBatch {
        steps,
        elapsed_ns: elapsed_ns(batch_start),
        full_ns,
        incremental_ns,
        cache_update_ns,
        disagreements,
        vm_hwm_kib: linux_vm_hwm_kib(),
    })
}

/// Conservative post-hoc cost accounting for a verified paired sequence.
/// A mismatch cannot be counted as a successful incremental decision.
/// These are component timings, NOT a production speedup: the physical probe
/// runs both engines, and its observed wall time is `elapsed_ns`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VerifiedRoutingCost {
    pub safe_incremental_steps: u64,
    pub fallback_full_steps: u64,
    pub full_reference_ns: u128,
    pub safe_selected_ns: u128,
    pub incremental_cache_ns: u128,
}

impl<T> SequentialBatch<T> {
    /// Replay a VALID-frozen routing decision against measured oracle pairs.
    /// A failed output/decision check falls back to Full for accounting.
    /// Cache refresh is charged only when Incremental was actually selected.
    /// This is an offline upper-bound diagnostic, not a dispatch implementation.
    #[must_use]
    pub fn verified_routing_cost(
        &self,
        select_incremental: impl Fn(u64, u64, usize) -> bool,
    ) -> VerifiedRoutingCost {
        let mut result = VerifiedRoutingCost::default();
        for (index, step) in self.steps.iter().enumerate() {
            let cost = &step.cost;
            result.full_reference_ns += u128::from(cost.full_ns);
            if select_incremental(cost.gates, cost.cone_gates, index)
                && cost.same_output
                && cost.same_decision
            {
                result.safe_incremental_steps += 1;
                let cache_ns = u128::from(cost.cache_update_ns);
                result.incremental_cache_ns += cache_ns;
                result.safe_selected_ns += u128::from(cost.incremental_ns) + cache_ns;
            } else {
                result.fallback_full_steps += 1;
                result.safe_selected_ns += u128::from(cost.full_ns);
            }
        }
        result
    }
}

#[cfg(test)]
mod verified_routing_tests {
    use super::*;

    #[test]
    fn mismatch_is_not_credited_as_incremental_speedup() {
        let batch = measure_sequential_batch(
            &[(128, 4), (256, 8), (512, 16)],
            |_| Ok(()),
            |i| Ok(i as u8),
            |i| Ok((i + usize::from(i == 1)) as u8),
            |a, b| a == b,
        )
        .unwrap();
        let report = batch.verified_routing_cost(|_, _, _| true);
        assert_eq!(report.safe_incremental_steps, 2);
        assert_eq!(report.fallback_full_steps, 1);
        let oracle = batch
            .steps
            .iter()
            .map(|s| u128::from(s.cost.full_ns))
            .sum::<u128>();
        assert_eq!(report.full_reference_ns, oracle);
        let expected = batch
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if i == 1 {
                    u128::from(s.cost.full_ns)
                } else {
                    u128::from(s.cost.incremental_ns) + u128::from(s.cost.cache_update_ns)
                }
            })
            .sum::<u128>();
        assert_eq!(report.safe_selected_ns, expected);
    }

    #[test]
    fn disabled_route_never_charges_cache_refresh() {
        let batch = measure_sequential_batch(
            &[(128, 4)],
            |_| Ok(()),
            |_| Ok(7_u8),
            |_| Ok(7_u8),
            |a, b| a == b,
        )
        .unwrap();
        let report = batch.verified_routing_cost(|_, _, _| false);
        assert_eq!(report.safe_incremental_steps, 0);
        assert_eq!(report.fallback_full_steps, 1);
        assert_eq!(report.incremental_cache_ns, 0);
        assert_eq!(report.safe_selected_ns, report.full_reference_ns);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn heterogeneous_batch_alternates_and_preserves_full_oracle() {
        let order = std::cell::RefCell::new(Vec::new());
        let batch = measure_sequential_batch(
            &[(64, 1), (128, 7), (1_562_501, 35)],
            |i| {
                order.borrow_mut().push((i, "cache"));
                Ok(())
            },
            |i| {
                order.borrow_mut().push((i, "full"));
                Ok(i as u8)
            },
            |i| {
                order.borrow_mut().push((i, "incremental"));
                Ok((i + usize::from(i == 1)) as u8)
            },
            |a, b| a == b,
        )
        .unwrap();
        assert_eq!(
            batch
                .steps
                .iter()
                .map(|s| s.verified_output)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(batch.disagreements, 1);
        assert_eq!(
            *order.borrow(),
            vec![
                (0, "cache"),
                (0, "full"),
                (0, "incremental"),
                (1, "cache"),
                (1, "incremental"),
                (1, "full"),
                (2, "cache"),
                (2, "full"),
                (2, "incremental"),
            ]
        );
        assert!(batch.steps_per_second().unwrap() > 0.0);
        assert!(batch.full_ns > 0 && batch.incremental_ns > 0 && batch.cache_update_ns > 0);
    }

    #[test]
    fn batch_rejects_invalid_geometry_before_callbacks_and_empty_has_no_rate() {
        let calls = Cell::new(0);
        let empty = measure_sequential_batch::<u8>(
            &[],
            |_| {
                calls.set(1);
                Ok(())
            },
            |_| Ok(0),
            |_| Ok(0),
            |a, b| a == b,
        )
        .unwrap();
        assert!(empty.steps_per_second().is_none());
        let error = measure_sequential_batch::<u8>(
            &[(0, 0)],
            |_| {
                calls.set(1);
                Ok(())
            },
            |_| Ok(0),
            |_| Ok(0),
            |a, b| a == b,
        );
        assert_eq!(error, Err(ProbeError::FullEvaluation));
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn late_invalid_geometry_rejects_entire_batch_without_side_effects() {
        let calls = Cell::new(0_u32);
        let result = measure_sequential_batch::<u8>(
            &[(64, 1), (128, 8), (16, 17)],
            |_| {
                calls.set(calls.get() + 1);
                Ok(())
            },
            |_| {
                calls.set(calls.get() + 1);
                Ok(1)
            },
            |_| {
                calls.set(calls.get() + 1);
                Ok(1)
            },
            |a, b| a == b,
        );
        assert_eq!(result, Err(ProbeError::FullEvaluation));
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn balanced_order_is_observable_and_oracle_remains_full() {
        for first in [false, true] {
            let order = std::cell::RefCell::new(Vec::new());
            let step = measure_sequential_step_balanced(
                128,
                4,
                first,
                || {
                    order.borrow_mut().push("cache");
                    Ok(())
                },
                || {
                    order.borrow_mut().push("full");
                    Ok(7_u8)
                },
                || {
                    order.borrow_mut().push("incremental");
                    Ok(8_u8)
                },
                |a, b| a == b,
            )
            .unwrap();
            let expected = if first {
                vec!["cache", "incremental", "full"]
            } else {
                vec!["cache", "full", "incremental"]
            };
            assert_eq!(*order.borrow(), expected);
            assert_eq!(step.verified_output, 7);
            assert!(!step.cost.same_output);
        }
    }

    #[test]
    fn oracle_never_returns_disagreeing_incremental_value() {
        let rebuilt = Cell::new(false);
        let step = measure_sequential_step(
            128,
            4,
            || {
                rebuilt.set(true);
                Ok(())
            },
            || {
                assert!(rebuilt.get());
                Ok(7_u8)
            },
            || {
                assert!(rebuilt.get());
                Ok(8_u8)
            },
            |a, b| a == b,
        )
        .unwrap();
        assert_eq!(step.verified_output, 7);
        assert!(!step.cost.same_output);
        assert!(!step.cost.same_decision);
        assert!(step.cost.full_ns > 0);
        assert!(step.cost.incremental_ns > 0);
        assert!(step.cost.cache_update_ns > 0);
    }

    #[test]
    fn failed_cache_rebuild_prevents_both_engines() {
        let calls = Cell::new(0);
        let result = measure_sequential_step::<u8>(
            64,
            1,
            || Err(ProbeError::CacheBuild),
            || {
                calls.set(calls.get() + 1);
                Ok(1)
            },
            || {
                calls.set(calls.get() + 1);
                Ok(1)
            },
            |a, b| a == b,
        );
        assert_eq!(result, Err(ProbeError::CacheBuild));
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn invalid_geometry_does_not_mutate_cache() {
        let calls = Cell::new(0);
        let result = measure_sequential_step::<u8>(
            4,
            5,
            || {
                calls.set(1);
                Ok(())
            },
            || Ok(0),
            || Ok(0),
            |a, b| a == b,
        );
        assert_eq!(result, Err(ProbeError::FullEvaluation));
        assert_eq!(calls.get(), 0);
    }
}

/// Execute a policy-selected intervention with a mandatory independent Full
/// oracle. This is a correctness gate, not an accelerated production dispatcher:
/// Full is evaluated on every step, including successful Incremental routes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifiedRoute {
    Full,
    IncrementalVerified,
    IncrementalRejected,
    CacheRebuildFailed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedDispatch<T> {
    pub output: T,
    pub route: VerifiedRoute,
    /// End-to-end wall time, including oracle evaluation and any cache refresh.
    pub elapsed_ns: u64,
    pub vm_hwm_kib: Option<u64>,
}

/// Execute Full once per intervention and never expose an unverified candidate.
/// Cache failures fail closed to Full; a Full failure is always an error.
/// The caller must apply the accepted circuit mutation before this function.
pub fn dispatch_sequential_verified<T: Eq>(
    gates: u64,
    cone_gates: u64,
    select_incremental: bool,
    rebuild_cache: impl FnOnce() -> Result<(), ProbeError>,
    full: impl FnOnce() -> Result<T, ProbeError>,
    incremental: impl FnOnce() -> Result<T, ProbeError>,
    same_decision: impl FnOnce(&T, &T) -> bool,
) -> Result<VerifiedDispatch<T>, ProbeError> {
    if gates == 0 || cone_gates > gates {
        return Err(ProbeError::FullEvaluation);
    }
    let started = Instant::now();
    if !select_incremental {
        let output = full().map_err(|_| ProbeError::FullEvaluation)?;
        return Ok(VerifiedDispatch {
            output,
            route: VerifiedRoute::Full,
            elapsed_ns: elapsed_ns(started),
            vm_hwm_kib: linux_vm_hwm_kib(),
        });
    }
    if rebuild_cache().is_err() {
        let output = full().map_err(|_| ProbeError::FullEvaluation)?;
        return Ok(VerifiedDispatch {
            output,
            route: VerifiedRoute::CacheRebuildFailed,
            elapsed_ns: elapsed_ns(started),
            vm_hwm_kib: linux_vm_hwm_kib(),
        });
    }
    // The independent oracle must not depend on the incremental result.
    let reference = full().map_err(|_| ProbeError::FullEvaluation)?;
    let verified = match incremental() {
        Ok(candidate) => candidate == reference && same_decision(&reference, &candidate),
        Err(_) => false,
    };
    Ok(VerifiedDispatch {
        output: reference,
        route: if verified {
            VerifiedRoute::IncrementalVerified
        } else {
            VerifiedRoute::IncrementalRejected
        },
        elapsed_ns: elapsed_ns(started),
        vm_hwm_kib: linux_vm_hwm_kib(),
    })
}

#[cfg(test)]
mod verified_dispatch_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn unselected_route_never_builds_cache_or_runs_incremental() {
        let calls = Cell::new(0);
        let result = dispatch_sequential_verified(
            64,
            2,
            false,
            || {
                calls.set(calls.get() + 1);
                Ok(())
            },
            || Ok(7_u8),
            || {
                calls.set(calls.get() + 1);
                Ok(9_u8)
            },
            |a, b| a == b,
        )
        .unwrap();
        assert_eq!(result.output, 7);
        assert_eq!(result.route, VerifiedRoute::Full);
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn mismatch_and_incremental_error_cannot_change_oracle_decision() {
        for fail in [false, true] {
            let result = dispatch_sequential_verified(
                128,
                8,
                true,
                || Ok(()),
                || Ok(3_u8),
                || {
                    if fail {
                        Err(ProbeError::IncrementalEvaluation)
                    } else {
                        Ok(4_u8)
                    }
                },
                |a, b| a == b,
            )
            .unwrap();
            assert_eq!(result.output, 3);
            assert_eq!(result.route, VerifiedRoute::IncrementalRejected);
        }
    }

    #[test]
    fn cache_failure_falls_back_to_full_without_incremental_call() {
        let called = Cell::new(false);
        let result = dispatch_sequential_verified(
            128,
            8,
            true,
            || Err(ProbeError::CacheBuild),
            || Ok(5_u8),
            || {
                called.set(true);
                Ok(5_u8)
            },
            |a, b| a == b,
        )
        .unwrap();
        assert_eq!(result.route, VerifiedRoute::CacheRebuildFailed);
        assert_eq!(result.output, 5);
        assert!(!called.get());
    }

    #[test]
    fn valid_candidate_still_returns_independent_full_output() {
        let result = dispatch_sequential_verified(
            256,
            4,
            true,
            || Ok(()),
            || Ok(11_u8),
            || Ok(11_u8),
            |a, b| a == b,
        )
        .unwrap();
        assert_eq!(result.route, VerifiedRoute::IncrementalVerified);
        assert_eq!(result.output, 11);
        assert!(result.elapsed_ns > 0);
    }
}

/// Exact cache-generation guard for sequential accepted interventions.
/// Each accepted mutation invalidates the previous cache; only a rebuild
/// explicitly committed for the current generation permits Incremental.
/// The caller must invoke accept_intervention after EVERY accepted mutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequentialCacheEpoch {
    generation: u64,
    cached_generation: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheEpochError {
    GenerationExhausted,
    StaleRebuild,
}

impl Default for SequentialCacheEpoch {
    fn default() -> Self {
        Self::new()
    }
}

impl SequentialCacheEpoch {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            generation: 0,
            cached_generation: None,
        }
    }

    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// A token is captured BEFORE rebuilding the cache. It must be committed
    /// only after a successful rebuild of that exact circuit generation.
    #[must_use]
    pub const fn rebuild_token(&self) -> u64 {
        self.generation
    }

    pub fn accept_intervention(&mut self) -> Result<u64, CacheEpochError> {
        let next = self
            .generation
            .checked_add(1)
            .ok_or(CacheEpochError::GenerationExhausted)?;
        self.generation = next;
        self.cached_generation = None;
        Ok(next)
    }

    pub fn commit_rebuild(&mut self, token: u64) -> Result<(), CacheEpochError> {
        if token != self.generation {
            return Err(CacheEpochError::StaleRebuild);
        }
        self.cached_generation = Some(token);
        Ok(())
    }

    /// Quarantine a cache after an oracle mismatch or incremental failure.
    /// A subsequent selected intervention must rebuild for the same generation.
    pub fn invalidate_cache(&mut self) {
        self.cached_generation = None;
    }

    #[must_use]
    pub const fn incremental_allowed(&self) -> bool {
        matches!(self.cached_generation, Some(cached) if cached == self.generation)
    }

    /// Fail closed: a stale cache always routes to Full, even when the
    /// VALID-frozen selector prefers Incremental.
    #[must_use]
    pub const fn route_incremental(&self, selected: bool) -> bool {
        selected && self.incremental_allowed()
    }
}

#[cfg(test)]
mod sequential_cache_epoch_tests {
    use super::{CacheEpochError, SequentialCacheEpoch};

    #[test]
    fn accepted_mutation_invalidates_cache_until_exact_rebuild() {
        let mut guard = SequentialCacheEpoch::new();
        assert!(!guard.route_incremental(true));
        guard.commit_rebuild(guard.rebuild_token()).unwrap();
        assert!(guard.route_incremental(true));
        let previous = guard.rebuild_token();
        guard.accept_intervention().unwrap();
        assert!(!guard.route_incremental(true));
        assert_eq!(
            guard.commit_rebuild(previous),
            Err(CacheEpochError::StaleRebuild)
        );
        assert!(!guard.incremental_allowed());
        guard.commit_rebuild(guard.rebuild_token()).unwrap();
        assert!(guard.route_incremental(true));
        assert!(!guard.route_incremental(false));
    }

    #[test]
    fn rejected_intervention_preserves_cache_and_epoch() {
        let mut guard = SequentialCacheEpoch::new();
        guard.commit_rebuild(0).unwrap();
        let before = guard;
        // No accept_intervention call for a rejected proposal.
        assert_eq!(guard, before);
        assert!(guard.incremental_allowed());
    }

    #[test]
    fn stale_commit_never_validates_newer_generation() {
        let mut guard = SequentialCacheEpoch::new();
        let old = guard.rebuild_token();
        guard.accept_intervention().unwrap();
        guard.accept_intervention().unwrap();
        assert_eq!(
            guard.commit_rebuild(old),
            Err(CacheEpochError::StaleRebuild)
        );
        assert!(!guard.incremental_allowed());
    }
}

/// Explicit geometry and frozen routing choice for a single measured step.
/// Group these conditions so the fallible callbacks remain separate and the
/// exact epoch-provenance contract is visible at every call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpochDispatchConfig {
    pub gates: u64,
    pub cone_gates: u64,
    pub select_incremental: bool,
}

/// Bind the verified dispatcher to an exact cache generation. A cache rebuild
/// failure routes to Full; no Incremental result escapes without a Full oracle.
/// The caller must mark every accepted mutation via accept_intervention().
pub fn dispatch_sequential_epoch_verified<T: Eq>(
    epoch: &mut SequentialCacheEpoch,
    config: EpochDispatchConfig,
    rebuild_cache: impl FnOnce() -> Result<(), ProbeError>,
    full: impl FnOnce() -> Result<T, ProbeError>,
    incremental: impl FnOnce() -> Result<T, ProbeError>,
    same_decision: impl FnOnce(&T, &T) -> bool,
) -> Result<VerifiedDispatch<T>, ProbeError> {
    if config.gates == 0 || config.cone_gates > config.gates {
        return Err(ProbeError::FullEvaluation);
    }
    let ready = if !config.select_incremental {
        false
    } else if epoch.incremental_allowed() {
        true
    } else {
        let token = epoch.rebuild_token();
        rebuild_cache().is_ok() && epoch.commit_rebuild(token).is_ok()
    };
    let result = dispatch_sequential_verified(
        config.gates,
        config.cone_gates,
        ready,
        || Ok(()),
        full,
        incremental,
        same_decision,
    );
    // A mismatching candidate is evidence that the cache or incremental engine
    // cannot be trusted for this generation. Never reuse it without rebuilding.
    if matches!(
        &result,
        Ok(VerifiedDispatch {
            route: VerifiedRoute::IncrementalRejected,
            ..
        }) | Err(_)
    ) {
        epoch.invalidate_cache();
    }
    result
}

#[cfg(test)]
mod epoch_verified_dispatch_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn mutation_requires_fresh_rebuild_but_reuses_current_cache() {
        let mut epoch = SequentialCacheEpoch::new();
        let builds = Cell::new(0);
        for expected in [1, 1, 2] {
            if expected == 2 {
                epoch.accept_intervention().unwrap();
            }
            let result = dispatch_sequential_epoch_verified(
                &mut epoch,
                EpochDispatchConfig {
                    gates: 128,
                    cone_gates: 4,
                    select_incremental: true,
                },
                || {
                    builds.set(builds.get() + 1);
                    Ok(())
                },
                || Ok(7_u8),
                || Ok(7_u8),
                |a, b| a == b,
            )
            .unwrap();
            assert_eq!(result.route, VerifiedRoute::IncrementalVerified);
            assert_eq!(builds.get(), expected);
        }
    }

    #[test]
    fn failed_rebuild_never_calls_incremental() {
        let mut epoch = SequentialCacheEpoch::new();
        let calls = Cell::new(0);
        let result = dispatch_sequential_epoch_verified(
            &mut epoch,
            EpochDispatchConfig {
                gates: 128,
                cone_gates: 4,
                select_incremental: true,
            },
            || Err(ProbeError::CacheBuild),
            || Ok(7_u8),
            || {
                calls.set(1);
                Ok(8_u8)
            },
            |a, b| a == b,
        )
        .unwrap();
        assert_eq!(result.route, VerifiedRoute::Full);
        assert_eq!(result.output, 7);
        assert_eq!(calls.get(), 0);
        assert!(!epoch.incremental_allowed());
    }

    #[test]
    fn invalid_geometry_does_not_touch_cache_or_engine() {
        let mut epoch = SequentialCacheEpoch::new();
        let calls = Cell::new(0);
        let result = dispatch_sequential_epoch_verified(
            &mut epoch,
            EpochDispatchConfig {
                gates: 4,
                cone_gates: 5,
                select_incremental: true,
            },
            || {
                calls.set(1);
                Ok(())
            },
            || {
                calls.set(1);
                Ok(0_u8)
            },
            || {
                calls.set(1);
                Ok(0_u8)
            },
            |a, b| a == b,
        );
        assert_eq!(result, Err(ProbeError::FullEvaluation));
        assert_eq!(calls.get(), 0);
    }
}

#[cfg(test)]
mod epoch_quarantine_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn disagreement_quarantines_cache_and_forces_next_rebuild() {
        let mut epoch = SequentialCacheEpoch::new();
        let rebuilds = Cell::new(0);
        let first = dispatch_sequential_epoch_verified(
            &mut epoch,
            EpochDispatchConfig {
                gates: 128,
                cone_gates: 4,
                select_incremental: true,
            },
            || {
                rebuilds.set(rebuilds.get() + 1);
                Ok(())
            },
            || Ok(7_u8),
            || Ok(8_u8),
            |a, b| a == b,
        )
        .unwrap();
        assert_eq!(first.output, 7);
        assert_eq!(first.route, VerifiedRoute::IncrementalRejected);
        assert!(!epoch.incremental_allowed());
        let second = dispatch_sequential_epoch_verified(
            &mut epoch,
            EpochDispatchConfig {
                gates: 128,
                cone_gates: 4,
                select_incremental: true,
            },
            || {
                rebuilds.set(rebuilds.get() + 1);
                Ok(())
            },
            || Ok(7_u8),
            || Ok(7_u8),
            |a, b| a == b,
        )
        .unwrap();
        assert_eq!(second.route, VerifiedRoute::IncrementalVerified);
        assert_eq!(rebuilds.get(), 2);
        assert!(epoch.incremental_allowed());
    }

    #[test]
    fn full_oracle_failure_quarantines_previously_current_cache() {
        let mut epoch = SequentialCacheEpoch::new();
        epoch.commit_rebuild(epoch.rebuild_token()).unwrap();
        let result = dispatch_sequential_epoch_verified::<u8>(
            &mut epoch,
            EpochDispatchConfig {
                gates: 128,
                cone_gates: 4,
                select_incremental: true,
            },
            || panic!("current cache must not be rebuilt"),
            || Err(ProbeError::FullEvaluation),
            || Ok(7),
            |a, b| a == b,
        );
        assert_eq!(result, Err(ProbeError::FullEvaluation));
        assert!(!epoch.incremental_allowed());
    }
}


/// Strict measured routing accounting for heterogeneous sequential workloads.
/// A missing process-wide peak, invalid timing, mismatched oracle, or memory
/// excess forces Full. TEST/OOD may be evaluated here but MUST NOT calibrate
/// the frozen selector supplied by the caller.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoundedRoutingCost {
    pub full_steps: u64,
    pub incremental_steps: u64,
    pub reference_ns: u128,
    pub selected_ns: u128,
    pub rejected_for_memory: u64,
    pub rejected_for_oracle: u64,
    pub rejected_for_cost: u64,
}

impl<T> SequentialBatch<T> {
    /// Account for cache refresh and the actual measured Full fallback cost.
    /// This is a post-hoc paired-probe estimate, NOT production wall time.
    #[must_use]
    pub fn bounded_routing_cost(
        &self,
        max_vmhwm_kib: u64,
        frozen_select: impl Fn(u64, u64, usize) -> bool,
    ) -> BoundedRoutingCost {
        let mut report = BoundedRoutingCost::default();
        for (index, step) in self.steps.iter().enumerate() {
            let c = &step.cost;
            report.reference_ns += u128::from(c.full_ns);
            let memory_ok = max_vmhwm_kib != 0
                && step
                    .vm_hwm_kib
                    .is_some_and(|peak| peak > 0 && peak <= max_vmhwm_kib);
            let oracle_ok = c.same_output && c.same_decision;
            let incremental_total = u128::from(c.incremental_ns)
                + u128::from(c.cache_update_ns);
            let cost_ok = c.full_ns > 0
                && c.incremental_ns > 0
                && c.cache_update_ns > 0
                && incremental_total < u128::from(c.full_ns);
            if frozen_select(c.gates, c.cone_gates, index)
                && memory_ok
                && oracle_ok
                && cost_ok
            {
                report.incremental_steps += 1;
                report.selected_ns += incremental_total;
            } else {
                report.full_steps += 1;
                report.selected_ns += u128::from(c.full_ns);
                if !memory_ok {
                    report.rejected_for_memory += 1;
                }
                if !oracle_ok {
                    report.rejected_for_oracle += 1;
                }
                if !cost_ok {
                    report.rejected_for_cost += 1;
                }
            }
        }
        report
    }
}

#[cfg(test)]
mod bounded_routing_tests {
    use super::*;

    fn step(full: u64, incremental: u64, cache: u64, peak: Option<u64>, equal: bool) -> PairedStep<u8> {
        PairedStep {
            verified_output: 1,
            cost: SequentialCost {
                gates: 256,
                cone_gates: 8,
                full_ns: full,
                incremental_ns: incremental,
                cache_update_ns: cache,
                same_output: equal,
                same_decision: equal,
            },
            vm_hwm_kib: peak,
        }
    }

    #[test]
    fn independent_scalar_oracle_rejects_all_unsafe_steps() {
        let rows = vec![
            step(100, 20, 10, Some(1024), true),
            step(100, 90, 20, Some(1024), true),
            step(100, 20, 10, None, true),
            step(100, 20, 10, Some(2048), true),
            step(100, 20, 10, Some(1024), false),
        ];
        let batch = SequentialBatch {
            steps: rows,
            elapsed_ns: 1000,
            full_ns: 500,
            incremental_ns: 170,
            cache_update_ns: 60,
            disagreements: 1,
            vm_hwm_kib: Some(2048),
        };
        let report = batch.bounded_routing_cost(1024, |_, _, _| true);
        assert_eq!(report.incremental_steps, 1);
        assert_eq!(report.full_steps, 4);
        assert_eq!(report.reference_ns, 500);
        assert_eq!(report.selected_ns, 430);
        assert_eq!(report.rejected_for_memory, 2);
        assert_eq!(report.rejected_for_oracle, 1);
        assert_eq!(report.rejected_for_cost, 1);
    }

    #[test]
    fn disabled_selector_never_credits_incremental() {
        let batch = SequentialBatch {
            steps: vec![step(100, 10, 10, Some(512), true)],
            elapsed_ns: 200,
            full_ns: 100,
            incremental_ns: 10,
            cache_update_ns: 10,
            disagreements: 0,
            vm_hwm_kib: Some(512),
        };
        let report = batch.bounded_routing_cost(1024, |_, _, _| false);
        assert_eq!(report.full_steps, 1);
        assert_eq!(report.incremental_steps, 0);
        assert_eq!(report.selected_ns, 100);
    }
}

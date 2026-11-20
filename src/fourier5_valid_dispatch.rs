//! Conservative VALID-only dispatch for synthetic Full/Incremental evaluations.
//! An uncalibrated geometry always selects Full. TEST/OOD never update policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineChoice { Full, Incremental }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyError { InvalidGeometry, ZeroDuration, Overflow }

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Bin {
    samples: u64,
    full_ns: u128,
    incremental_ns: u128,
    mismatches: u64,
    slow_samples: u64,
}

/// Conservative bucket policy; each bucket must have enough VALID samples,
/// no oracle mismatches, and an observed cost margin before dispatch.
#[derive(Clone, Debug)]
pub struct ValidDispatch {
    bins: [Bin; 16],
    min_samples: u64,
    margin_percent: u64,
}

impl ValidDispatch {
    #[must_use]
    pub fn new(min_samples: u64, margin_percent: u64) -> Self {
        Self {
            bins: [Bin::default(); 16],
            min_samples: min_samples.max(1),
            margin_percent: margin_percent.min(99),
        }
    }

    fn bucket(gates: u64, cone: u64) -> Result<usize, PolicyError> {
        if gates == 0 || cone > gates {
            return Err(PolicyError::InvalidGeometry);
        }
        let size = if gates < 1_024 { 0 } else if gates < 65_536 { 1 }
            else if gates < 1_048_576 { 2 } else { 3 };
        // Compare exact rational coverage: integer truncation incorrectly
        // classified e.g. 199/10_000 as <=1%, despite being 1.99%.
        let scaled_cone = u128::from(cone) * 100;
        let total = u128::from(gates);
        let coverage = if scaled_cone <= total { 0 }
            else if scaled_cone <= total * 10 { 1 }
            else if scaled_cone <= total * 50 { 2 } else { 3 };
        Ok(size * 4 + coverage)
    }

    /// Ingest a measured VALID observation, including refresh and cache build.
    /// The equality flag must compare all outputs and protected decisions.
    /// TRAIN, TEST and OOD must never call this method.
    pub fn observe_valid(
        &mut self, gates: u64, cone: u64, full_ns: u64,
        incremental_with_refresh_ns: u64, equal: bool,
    ) -> Result<(), PolicyError> {
        let index = Self::bucket(gates, cone)?;
        if full_ns == 0 || incremental_with_refresh_ns == 0 {
            return Err(PolicyError::ZeroDuration);
        }
        let mut next = self.bins[index];
        next.samples = next.samples.checked_add(1).ok_or(PolicyError::Overflow)?;
        next.full_ns = next.full_ns.checked_add(u128::from(full_ns))
            .ok_or(PolicyError::Overflow)?;
        next.incremental_ns = next.incremental_ns
            .checked_add(u128::from(incremental_with_refresh_ns))
            .ok_or(PolicyError::Overflow)?;
        // Averages alone can hide expensive heterogeneous refreshes. Require
        // every VALID sample to satisfy the margin, not only the pooled sum.
        let fast_enough = u128::from(incremental_with_refresh_ns) * 100
            < u128::from(full_ns) * u128::from(100 - self.margin_percent);
        if !fast_enough {
            next.slow_samples = next.slow_samples.checked_add(1)
                .ok_or(PolicyError::Overflow)?;
        }
        if !equal {
            next.mismatches = next.mismatches.checked_add(1)
                .ok_or(PolicyError::Overflow)?;
        }
        self.bins[index] = next;
        Ok(())
    }

    /// Dispatch from a frozen VALID calibration; no holdout feedback.
    #[must_use]
    pub fn choose(&self, gates: u64, cone: u64) -> EngineChoice {
        let Ok(index) = Self::bucket(gates, cone) else {
            return EngineChoice::Full;
        };
        let bin = self.bins[index];
        if bin.samples < self.min_samples || bin.mismatches != 0 || bin.slow_samples != 0 {
            return EngineChoice::Full;
        }
        // Division-free comparison, with checked arithmetic and fail-closed overflow.
        let Some(lhs) = bin.incremental_ns.checked_mul(100) else {
            return EngineChoice::Full;
        };
        let Some(rhs) = bin.full_ns.checked_mul(u128::from(100 - self.margin_percent)) else {
            return EngineChoice::Full;
        };
        if lhs < rhs { EngineChoice::Incremental } else { EngineChoice::Full }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_valid_evidence_and_sufficient_margin() {
        let mut policy = ValidDispatch::new(2, 10);
        assert_eq!(policy.choose(10_000, 50), EngineChoice::Full);
        policy.observe_valid(10_000, 50, 100, 40, true).unwrap();
        assert_eq!(policy.choose(10_000, 50), EngineChoice::Full);
        policy.observe_valid(10_000, 50, 100, 50, true).unwrap();
        assert_eq!(policy.choose(10_000, 50), EngineChoice::Incremental);
        assert_eq!(policy.choose(10_000, 9_000), EngineChoice::Full);
    }

    #[test]
    fn oracle_mismatch_disables_entire_geometry_bucket() {
        let mut policy = ValidDispatch::new(1, 0);
        policy.observe_valid(10_000, 50, 100, 1, false).unwrap();
        assert_eq!(policy.choose(10_000, 50), EngineChoice::Full);
    }

    #[test]
    fn heterogeneous_slow_tail_disables_average_speedup() {
        let mut policy = ValidDispatch::new(3, 10);
        policy.observe_valid(10_000, 50, 1000, 10, true).unwrap();
        policy.observe_valid(10_000, 50, 1000, 10, true).unwrap();
        policy.observe_valid(10_000, 50, 100, 150, true).unwrap();
        // The pooled incremental average is faster, but one VALID case regresses.
        assert_eq!(policy.choose(10_000, 50), EngineChoice::Full);
    }

    #[test]
    fn all_valid_samples_must_clear_configured_margin() {
        let mut policy = ValidDispatch::new(2, 10);
        policy.observe_valid(10_000, 50, 100, 89, true).unwrap();
        policy.observe_valid(10_000, 50, 100, 89, true).unwrap();
        assert_eq!(policy.choose(10_000, 50), EngineChoice::Incremental);
        policy.observe_valid(10_000, 50, 100, 90, true).unwrap();
        assert_eq!(policy.choose(10_000, 50), EngineChoice::Full);
    }

    #[test]
    fn fractional_cone_boundaries_never_leak_valid_calibration() {
        let mut policy = ValidDispatch::new(1, 0);
        policy.observe_valid(10_000, 100, 100, 10, true).unwrap();
        assert_eq!(policy.choose(10_000, 100), EngineChoice::Incremental);
        // 1.01% and 1.99% are not in the <=1% bucket.
        for cone in [101, 199] {
            assert_eq!(policy.choose(10_000, cone), EngineChoice::Full);
        }
        policy.observe_valid(10_000, 1_000, 100, 10, true).unwrap();
        assert_eq!(policy.choose(10_000, 1_000), EngineChoice::Incremental);
        assert_eq!(policy.choose(10_000, 1_001), EngineChoice::Full);
        policy.observe_valid(10_000, 5_000, 100, 10, true).unwrap();
        assert_eq!(policy.choose(10_000, 5_000), EngineChoice::Incremental);
        assert_eq!(policy.choose(10_000, 5_001), EngineChoice::Full);
    }

    #[test]
    fn invalid_observation_is_atomic_and_fails_closed() {
        let mut policy = ValidDispatch::new(1, 10);
        assert_eq!(policy.observe_valid(0, 0, 100, 1, true),
            Err(PolicyError::InvalidGeometry));
        assert_eq!(policy.observe_valid(10_000, 50, 0, 1, true),
            Err(PolicyError::ZeroDuration));
        assert_eq!(policy.choose(10_000, 50), EngineChoice::Full);
        policy.observe_valid(10_000, 50, 100, 20, true).unwrap();
        assert_eq!(policy.choose(10_000, 50), EngineChoice::Incremental);
        assert_eq!(policy.choose(10_000, 10_001), EngineChoice::Full);
    }
}


/// Immutable calibration artifact. Its API deliberately exposes no VALID ingestion,
/// so holdout evaluation cannot mutate dispatch decisions after calibration.
#[derive(Clone, Debug)]
pub struct FrozenValidDispatch {
    policy: ValidDispatch,
}

impl ValidDispatch {
    /// End calibration before TEST/OOD. Consumes the mutable policy.
    #[must_use]
    pub fn freeze(self) -> FrozenValidDispatch {
        FrozenValidDispatch { policy: self }
    }
}

impl FrozenValidDispatch {
    /// Select a path without permitting holdout feedback into the calibration.
    #[must_use]
    pub fn choose(&self, gates: u64, cone: u64) -> EngineChoice {
        self.policy.choose(gates, cone)
    }
}

#[cfg(test)]
mod frozen_dispatch_tests {
    use super::*;

    #[test]
    fn frozen_decisions_match_calibration_across_geometry() {
        let mut calibration = ValidDispatch::new(2, 10);
        calibration.observe_valid(10_000, 100, 100, 20, true).unwrap();
        calibration.observe_valid(10_000, 100, 100, 30, true).unwrap();
        let geometries = [(10_000, 100), (10_000, 101), (10_000, 5_000),
                          (0, 0), (10_000, 10_001)];
        let expected: Vec<_> = geometries.iter()
            .map(|&(gates, cone)| calibration.choose(gates, cone)).collect();
        let frozen = calibration.freeze();
        let actual: Vec<_> = geometries.iter()
            .map(|&(gates, cone)| frozen.choose(gates, cone)).collect();
        assert_eq!(actual, expected);
        assert_eq!(frozen.choose(10_000, 100), EngineChoice::Incremental);
        assert_eq!(frozen.choose(10_000, 101), EngineChoice::Full);
    }

    #[test]
    fn freeze_keeps_fail_closed_mismatch_decision() {
        let mut calibration = ValidDispatch::new(1, 0);
        calibration.observe_valid(10_000, 100, 100, 1, false).unwrap();
        let frozen = calibration.freeze();
        assert_eq!(frozen.choose(10_000, 100), EngineChoice::Full);
    }
}


/// Exact-output verification mode for frozen VALID dispatch.
/// This is a correctness guard, NOT a speed benchmark: when Incremental is
/// selected it deliberately runs Full as an independent oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifiedPath {
    Full,
    IncrementalVerified,
    FullAfterMismatch,
}

impl FrozenValidDispatch {
    /// Execute the selected engine and return an oracle-verified decision.
    ///
    /// `full` must be an independent implementation, not a wrapper around
    /// `incremental`. Closures are evaluated at most once each. This API
    /// never learns from TEST/OOD outputs and cannot mutate the frozen policy.
    pub fn execute_verified<T, F, I>(
        &self,
        gates: u64,
        cone: u64,
        full: F,
        incremental: I,
    ) -> (T, VerifiedPath)
    where
        T: Eq,
        F: FnOnce() -> T,
        I: FnOnce() -> T,
    {
        if self.choose(gates, cone) == EngineChoice::Full {
            return (full(), VerifiedPath::Full);
        }
        let candidate = incremental();
        let reference = full();
        if candidate == reference {
            (candidate, VerifiedPath::IncrementalVerified)
        } else {
            (reference, VerifiedPath::FullAfterMismatch)
        }
    }
}

#[cfg(test)]
mod verified_dispatch_tests {
    use super::*;
    use std::cell::Cell;

    fn calibrated() -> FrozenValidDispatch {
        let mut policy = ValidDispatch::new(1, 10);
        policy.observe_valid(10_000, 100, 100, 20, true).unwrap();
        policy.freeze()
    }

    #[test]
    fn verified_incremental_preserves_full_oracle_decision() {
        let (result, path) = calibrated().execute_verified(
            10_000, 100, || vec![true, false], || vec![true, false],
        );
        assert_eq!(result, vec![true, false]);
        assert_eq!(path, VerifiedPath::IncrementalVerified);
    }

    #[test]
    fn mismatch_returns_independent_full_output() {
        let (result, path) = calibrated().execute_verified(
            10_000, 100, || vec![false, true], || vec![true, false],
        );
        assert_eq!(result, vec![false, true]);
        assert_eq!(path, VerifiedPath::FullAfterMismatch);
    }

    #[test]
    fn uncalibrated_geometry_never_runs_incremental() {
        let called = Cell::new(false);
        let (result, path) = calibrated().execute_verified(
            10_000, 101, || 7_u64, || { called.set(true); 99_u64 },
        );
        assert_eq!((result, path), (7, VerifiedPath::Full));
        assert!(!called.get());
    }

    #[test]
    fn verification_executes_each_engine_once() {
        let full_calls = Cell::new(0);
        let inc_calls = Cell::new(0);
        let (_, path) = calibrated().execute_verified(
            10_000, 100,
            || { full_calls.set(full_calls.get() + 1); 3 },
            || { inc_calls.set(inc_calls.get() + 1); 4 },
        );
        assert_eq!(path, VerifiedPath::FullAfterMismatch);
        assert_eq!((full_calls.get(), inc_calls.get()), (1, 1));
    }
}


/// Result of a fallible verified dispatch. An Incremental failure is never
/// published: the independent Full oracle becomes the authoritative result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallibleVerifiedPath {
    Full,
    IncrementalVerified,
    FullAfterMismatch,
    FullAfterIncrementalError,
}

impl FrozenValidDispatch {
    /// Fail-closed fallible execution. Full errors propagate; Incremental
    /// errors fall back to Full. Neither error nor mismatch changes calibration.
    ///
    /// This diagnostic path executes Full for every Incremental selection;
    /// it must not be used to claim Incremental-only throughput.
    pub fn try_execute_verified<T, E, F, I>(
        &self,
        gates: u64,
        cone: u64,
        full: F,
        incremental: I,
    ) -> Result<(T, FallibleVerifiedPath), E>
    where
        T: Eq,
        F: FnOnce() -> Result<T, E>,
        I: FnOnce() -> Result<T, E>,
    {
        if self.choose(gates, cone) == EngineChoice::Full {
            return full().map(|value| (value, FallibleVerifiedPath::Full));
        }
        let candidate = incremental();
        let reference = full()?;
        match candidate {
            Ok(value) if value == reference => {
                Ok((value, FallibleVerifiedPath::IncrementalVerified))
            }
            Ok(_) => Ok((reference, FallibleVerifiedPath::FullAfterMismatch)),
            Err(_) => Ok((reference, FallibleVerifiedPath::FullAfterIncrementalError)),
        }
    }
}

#[cfg(test)]
mod fallible_verified_dispatch_tests {
    use super::*;
    use std::cell::Cell;

    fn policy() -> FrozenValidDispatch {
        let mut p = ValidDispatch::new(1, 10);
        p.observe_valid(10_000, 100, 100, 20, true).unwrap();
        p.freeze()
    }

    #[test]
    fn incremental_failure_falls_back_to_independent_full() {
        let (value, path) = policy().try_execute_verified(
            10_000, 100, || Ok::<_, &'static str>(vec![false, true]),
            || Err::<Vec<bool>, _>("incremental failure"),
        ).unwrap();
        assert_eq!(value, vec![false, true]);
        assert_eq!(path, FallibleVerifiedPath::FullAfterIncrementalError);
        assert_eq!(policy().choose(10_000, 100), EngineChoice::Incremental);
    }

    #[test]
    fn full_error_propagates_even_when_incremental_succeeds() {
        let result = policy().try_execute_verified(
            10_000, 100, || Err::<u8, _>("oracle failed"),
            || Ok::<_, &'static str>(7_u8),
        );
        assert_eq!(result, Err("oracle failed"));
    }

    #[test]
    fn uncalibrated_path_skips_incremental_even_if_full_fails() {
        let called = Cell::new(false);
        let result = policy().try_execute_verified(
            10_000, 101, || Err::<u8, _>("full failed"),
            || { called.set(true); Ok::<_, &'static str>(7) },
        );
        assert_eq!(result, Err("full failed"));
        assert!(!called.get());
    }

    #[test]
    fn equality_and_mismatch_return_correct_paths() {
        for (incremental, expected) in [
            (42_u8, FallibleVerifiedPath::IncrementalVerified),
            (43_u8, FallibleVerifiedPath::FullAfterMismatch),
        ] {
            let (value, path) = policy().try_execute_verified(
                10_000, 100, || Ok::<_, ()>(42_u8),
                || Ok::<_, ()>(incremental),
            ).unwrap();
            assert_eq!((value, path), (42, expected));
        }
    }
}


/// Runtime quarantine for a frozen VALID policy. A failed oracle check
/// disables Incremental for the entire geometry bucket for this session.
/// Quarantine is independent of VALID calibration and never trains on holdouts.
#[derive(Clone, Debug)]
pub struct QuarantinedDispatch {
    frozen: FrozenValidDispatch,
    quarantined: [bool; 16],
}

impl FrozenValidDispatch {
    /// Start a new evaluation session with no quarantined buckets.
    #[must_use]
    pub fn with_runtime_quarantine(self) -> QuarantinedDispatch {
        QuarantinedDispatch {
            frozen: self,
            quarantined: [false; 16],
        }
    }
}

impl QuarantinedDispatch {
    /// Select Full for any invalid, uncalibrated or quarantined geometry.
    #[must_use]
    pub fn choose(&self, gates: u64, cone: u64) -> EngineChoice {
        let Ok(bucket) = ValidDispatch::bucket(gates, cone) else {
            return EngineChoice::Full;
        };
        if self.quarantined[bucket] {
            EngineChoice::Full
        } else {
            self.frozen.choose(gates, cone)
        }
    }

    /// Return the exact quarantine bitmap for a reproducible session audit.
    /// Bit i corresponds to the same size/cone bucket as VALID calibration.
    /// This is diagnostic state only and must not modify frozen calibration.
    #[must_use]
    pub fn quarantine_mask(&self) -> u16 {
        self.quarantined.iter().enumerate().fold(0_u16, |mask, (i, &flag)| {
            if flag { mask | (1_u16 << i) } else { mask }
        })
    }

    /// Execute with an independent Full oracle. A mismatch or Incremental
    /// error permanently quarantines the bucket until a new session is made.
    /// Full failures propagate without publishing an unverified result and
    /// quarantine the bucket because the candidate cannot be certified.
    /// This is a safety mechanism, not a throughput optimization.
    pub fn try_execute<T, E, F, I>(
        &mut self,
        gates: u64,
        cone: u64,
        full: F,
        incremental: I,
    ) -> Result<(T, FallibleVerifiedPath), E>
    where
        T: Eq,
        F: FnOnce() -> Result<T, E>,
        I: FnOnce() -> Result<T, E>,
    {
        if self.choose(gates, cone) == EngineChoice::Full {
            return full().map(|value| (value, FallibleVerifiedPath::Full));
        }
        let bucket = ValidDispatch::bucket(gates, cone)
            .expect("incremental selection requires valid geometry");
        let candidate = incremental();
        // An unavailable Full oracle cannot certify the candidate. Quarantine
        // this geometry until a fresh session, even when Incremental succeeded.
        let reference = match full() {
            Ok(value) => value,
            Err(error) => {
                self.quarantined[bucket] = true;
                return Err(error);
            }
        };
        match candidate {
            Ok(value) if value == reference => {
                Ok((value, FallibleVerifiedPath::IncrementalVerified))
            }
            Ok(_) => {
                self.quarantined[bucket] = true;
                Ok((reference, FallibleVerifiedPath::FullAfterMismatch))
            }
            Err(_) => {
                self.quarantined[bucket] = true;
                Ok((reference, FallibleVerifiedPath::FullAfterIncrementalError))
            }
        }
    }
}

#[cfg(test)]
mod quarantine_tests {
    use super::*;
    use std::cell::Cell;

    fn session() -> QuarantinedDispatch {
        let mut valid = ValidDispatch::new(1, 10);
        valid.observe_valid(10_000, 100, 100, 20, true).unwrap();
        valid.observe_valid(10_000, 500, 100, 20, true).unwrap();
        valid.freeze().with_runtime_quarantine()
    }

    #[test]
    fn mismatch_quarantines_bucket_and_preserves_full_result() {
        let mut session = session();
        let (value, path) = session.try_execute(
            10_000, 100, || Ok::<_, ()>(9_u8), || Ok::<_, ()>(8_u8),
        ).unwrap();
        assert_eq!((value, path), (9, FallibleVerifiedPath::FullAfterMismatch));
        assert_eq!(session.choose(10_000, 100), EngineChoice::Full);
        assert_eq!(session.choose(10_000, 99), EngineChoice::Full);
        assert_eq!(session.choose(10_000, 500), EngineChoice::Incremental);
        let called = Cell::new(false);
        let (_, path) = session.try_execute(
            10_000, 100, || Ok::<_, ()>(9_u8),
            || { called.set(true); Ok::<_, ()>(8_u8) },
        ).unwrap();
        assert_eq!(path, FallibleVerifiedPath::Full);
        assert!(!called.get());
    }

    #[test]
    fn incremental_and_full_oracle_errors_quarantine_bucket() {
        let mut session = session();
        let result = session.try_execute(
            10_000, 100, || Err::<u8, _>("full failed"),
            || Ok::<_, &str>(3_u8),
        );
        assert_eq!(result, Err("full failed"));
        assert_eq!(session.choose(10_000, 100), EngineChoice::Full);
        // Another calibrated geometry remains available independently.
        assert_eq!(session.choose(10_000, 500), EngineChoice::Incremental);
        let (value, path) = session.try_execute(
            10_000, 500, || Ok::<_, &str>(7_u8),
            || Err::<u8, _>("incremental failed"),
        ).unwrap();
        assert_eq!((value, path), (7, FallibleVerifiedPath::FullAfterIncrementalError));
        assert_eq!(session.choose(10_000, 500), EngineChoice::Full);
    }

    #[test]
    fn quarantine_mask_tracks_independent_geometry_buckets() {
        let mut session = session();
        assert_eq!(session.quarantine_mask(), 0);
        let _ = session.try_execute(
            10_000, 100, || Ok::<_, ()>(1_u8), || Ok::<_, ()>(2_u8),
        ).unwrap();
        let first = session.quarantine_mask();
        assert_eq!(first.count_ones(), 1);
        assert_eq!(session.choose(10_000, 100), EngineChoice::Full);
        assert_eq!(session.choose(10_000, 500), EngineChoice::Incremental);
        let _ = session.try_execute(
            10_000, 500, || Ok::<_, ()>(1_u8), || Err::<u8, _>(()),
        ).unwrap();
        assert_eq!(session.quarantine_mask().count_ones(), 2);
        assert_eq!(session.quarantine_mask() & first, first);
    }

    #[test]
    fn matching_outputs_leave_bucket_enabled() {
        let mut session = session();
        let (value, path) = session.try_execute(
            10_000, 100, || Ok::<_, ()>(7_u8), || Ok::<_, ()>(7_u8),
        ).unwrap();
        assert_eq!((value, path), (7, FallibleVerifiedPath::IncrementalVerified));
        assert_eq!(session.choose(10_000, 100), EngineChoice::Incremental);
        assert_eq!(session.choose(0, 0), EngineChoice::Full);
    }
}

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
        let ratio = (u128::from(cone) * 100) / u128::from(gates);
        let coverage = if ratio <= 1 { 0 } else if ratio <= 10 { 1 }
            else if ratio <= 50 { 2 } else { 3 };
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
        if bin.samples < self.min_samples || bin.mismatches != 0 {
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

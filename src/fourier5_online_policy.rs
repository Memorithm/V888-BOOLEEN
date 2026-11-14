//! Bounded, fixed-memory online cost selection for synthetic Full/Incremental inference.
//! Feed only paired timing samples from the same accepted state after oracle equivalence.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineChoice {
    Full,
    Incremental,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyError {
    ZeroCost,
    InvalidCone,
    Overflow,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Bucket {
    samples: u32,
    full_ns: u128,
    incremental_ns: u128,
}

/// Four cone-fraction buckets; fixed memory, no allocations per observation.
/// Calibration must exclude TEST/OOD to prevent selection leakage.
#[derive(Clone, Debug, Default)]
pub struct OnlineEnginePolicy {
    buckets: [Bucket; 4],
}

impl OnlineEnginePolicy {
    fn bucket(cone: usize, total: usize) -> Result<usize, PolicyError> {
        if total == 0 || cone > total {
            return Err(PolicyError::InvalidCone);
        }
        let c = cone as u128;
        let t = total as u128;
        Ok(if c * 16 <= t {
            0
        } else if c * 4 <= t {
            1
        } else if c * 2 <= t {
            2
        } else {
            3
        })
    }

    /// The caller must verify exact equality of both outputs before recording
    /// a sample. Zero times are rejected and overflow cannot mutate state.
    pub fn observe(
        &mut self,
        cone: usize,
        total: usize,
        full_ns: u64,
        incremental_ns: u64,
    ) -> Result<(), PolicyError> {
        if full_ns == 0 || incremental_ns == 0 {
            return Err(PolicyError::ZeroCost);
        }
        let i = Self::bucket(cone, total)?;
        let mut b = self.buckets[i];
        b.samples = b.samples.checked_add(1).ok_or(PolicyError::Overflow)?;
        b.full_ns = b
            .full_ns
            .checked_add(u128::from(full_ns))
            .ok_or(PolicyError::Overflow)?;
        b.incremental_ns = b
            .incremental_ns
            .checked_add(u128::from(incremental_ns))
            .ok_or(PolicyError::Overflow)?;
        self.buckets[i] = b;
        Ok(())
    }

    /// Fail closed to Full unless eight paired observations establish a
    /// strict 10% aggregate latency advantage for the cone bucket.
    pub fn choose(&self, cone: usize, total: usize) -> Result<EngineChoice, PolicyError> {
        let b = self.buckets[Self::bucket(cone, total)?];
        if b.samples >= 8
            && b.incremental_ns
                .checked_mul(10)
                .is_some_and(|value| value < b.full_ns.saturating_mul(9))
        {
            Ok(EngineChoice::Incremental)
        } else {
            Ok(EngineChoice::Full)
        }
    }

    pub fn sample_count(&self, cone: usize, total: usize) -> Result<u32, PolicyError> {
        Ok(self.buckets[Self::bucket(cone, total)?].samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fail_closed_until_paired_evidence() {
        let mut policy = OnlineEnginePolicy::default();
        for _ in 0..7 {
            policy.observe(2, 100, 100, 20).unwrap();
        }
        assert_eq!(policy.choose(2, 100), Ok(EngineChoice::Full));
        policy.observe(2, 100, 100, 20).unwrap();
        assert_eq!(policy.choose(2, 100), Ok(EngineChoice::Incremental));
        assert_eq!(policy.choose(40, 100), Ok(EngineChoice::Full));
    }

    #[test]
    fn independent_integer_threshold_oracle() {
        for (full, incremental, expected) in [
            (100, 89, EngineChoice::Incremental),
            (100, 90, EngineChoice::Full),
            (100, 150, EngineChoice::Full),
        ] {
            let mut policy = OnlineEnginePolicy::default();
            for _ in 0..8 {
                policy.observe(40, 100, full, incremental).unwrap();
            }
            assert_eq!(policy.choose(40, 100), Ok(expected));
        }
    }

    #[test]
    fn rejects_invalid_samples_without_mutation() {
        let mut policy = OnlineEnginePolicy::default();
        assert_eq!(
            policy.observe(101, 100, 5, 2),
            Err(PolicyError::InvalidCone)
        );
        assert_eq!(policy.observe(1, 100, 0, 2), Err(PolicyError::ZeroCost));
        assert_eq!(policy.sample_count(1, 100), Ok(0));
        assert_eq!(policy.choose(1, 0), Err(PolicyError::InvalidCone));
    }
}

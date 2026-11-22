//! Runtime circuit breaker for sequential synthetic Full/Incremental execution.
//! VALID admission is frozen elsewhere; TEST/OOD observations never tune it.
//! Physical timings must include cache refresh and be measured by the caller.

/// A frozen, VALID-derived safety margin: Incremental is only faster when
/// `incremental_ns * denominator <= full_ns * numerator`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrozenMargin {
    pub numerator: u64,
    pub denominator: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeasuredStep {
    pub full_reference_ns: u64,
    pub incremental_with_refresh_ns: u64,
    pub observed_peak_kib: u64,
    pub oracle_state_equal: bool,
    pub oracle_cache_equal: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BreakerState {
    Armed,
    FullOnly,
}

/// One-way fail-closed breaker. It cannot be reset by TEST/OOD measurements.
/// A new instance requires a separately frozen VALID admission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequentialBreaker {
    state: BreakerState,
    steps: u64,
    peak_limit_kib: u64,
    margin: FrozenMargin,
}

impl SequentialBreaker {
    /// Invalid margins and zero memory limits start permanently in Full mode.
    #[must_use]
    pub fn new(margin: FrozenMargin, peak_limit_kib: u64) -> Self {
        let valid = margin.denominator > 0
            && margin.numerator > 0
            && margin.numerator <= margin.denominator
            && peak_limit_kib > 0;
        Self {
            state: if valid {
                BreakerState::Armed
            } else {
                BreakerState::FullOnly
            },
            steps: 0,
            peak_limit_kib,
            margin,
        }
    }

    #[must_use]
    pub fn state(&self) -> BreakerState {
        self.state
    }

    #[must_use]
    pub fn observed_steps(&self) -> u64 {
        self.steps
    }

    /// Observe a *paired* measurement on the same pre-intervention snapshot.
    /// Both paths must return complete state and canonical cache equality.
    /// Missing, zero, overflowing or regressing measurements trip the breaker.
    /// This does not itself execute the independent oracle or sample VmHWM.
    pub fn observe(&mut self, step: MeasuredStep) -> BreakerState {
        if self.state == BreakerState::FullOnly {
            return self.state;
        }
        let valid = step.full_reference_ns > 0
            && step.incremental_with_refresh_ns > 0
            && step.observed_peak_kib > 0
            && step.observed_peak_kib <= self.peak_limit_kib
            && step.oracle_state_equal
            && step.oracle_cache_equal
            && step
                .incremental_with_refresh_ns
                .checked_mul(self.margin.denominator)
                .zip(step.full_reference_ns.checked_mul(self.margin.numerator))
                .is_some_and(|(incremental, full)| incremental <= full);
        match self.steps.checked_add(1) {
            Some(next) => self.steps = next,
            None => self.state = BreakerState::FullOnly,
        }
        if !valid {
            self.state = BreakerState::FullOnly;
        }
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(full: u64, incremental: u64) -> MeasuredStep {
        MeasuredStep {
            full_reference_ns: full,
            incremental_with_refresh_ns: incremental,
            observed_peak_kib: 1024,
            oracle_state_equal: true,
            oracle_cache_equal: true,
        }
    }

    #[test]
    fn independent_integer_oracle_for_mixed_sequential_costs() {
        for full in 1..=30_u64 {
            for incremental in 1..=30_u64 {
                let mut breaker = SequentialBreaker::new(
                    FrozenMargin {
                        numerator: 9,
                        denominator: 10,
                    },
                    2048,
                );
                let expected = (incremental as u128) * 10 <= (full as u128) * 9;
                assert_eq!(
                    breaker.observe(step(full, incremental)) == BreakerState::Armed,
                    expected
                );
                assert_eq!(breaker.observed_steps(), 1);
                if !expected {
                    assert_eq!(breaker.observe(step(100, 1)), BreakerState::FullOnly);
                    assert_eq!(breaker.observed_steps(), 1);
                }
            }
        }
    }

    #[test]
    fn fail_closed_on_cache_divergence_memory_and_overflow() {
        let margin = FrozenMargin {
            numerator: 9,
            denominator: 10,
        };
        let mut invalid = [
            MeasuredStep {
                oracle_cache_equal: false,
                ..step(100, 1)
            },
            MeasuredStep {
                oracle_state_equal: false,
                ..step(100, 1)
            },
            MeasuredStep {
                observed_peak_kib: 4096,
                ..step(100, 1)
            },
            step(0, 1),
            step(100, 0),
            step(u64::MAX, u64::MAX),
        ];
        for observation in &mut invalid {
            let mut breaker = SequentialBreaker::new(margin, 2048);
            assert_eq!(breaker.observe(*observation), BreakerState::FullOnly);
            assert_eq!(breaker.observe(step(100, 1)), BreakerState::FullOnly);
        }
        for bad in [
            FrozenMargin {
                numerator: 0,
                denominator: 10,
            },
            FrozenMargin {
                numerator: 11,
                denominator: 10,
            },
            FrozenMargin {
                numerator: 1,
                denominator: 0,
            },
        ] {
            assert_eq!(SequentialBreaker::new(bad, 2048).state(), BreakerState::FullOnly);
        }
        assert_eq!(SequentialBreaker::new(margin, 0).state(), BreakerState::FullOnly);
    }

    #[test]
    fn sequential_heterogeneous_regression_latches_full() {
        let mut breaker = SequentialBreaker::new(
            FrozenMargin {
                numerator: 9,
                denominator: 10,
            },
            2048,
        );
        for (index, (full, incremental)) in [(100, 40), (100, 50), (20, 30), (100, 1)]
            .into_iter()
            .enumerate()
        {
            let state = breaker.observe(step(full, incremental));
            assert_eq!(
                state,
                if index < 2 {
                    BreakerState::Armed
                } else {
                    BreakerState::FullOnly
                }
            );
        }
        assert_eq!(breaker.observed_steps(), 3);
    }
}

//! Runtime safety circuit breaker for a frozen synthetic Full/Incremental policy.
//! This is NOT calibration: TEST/OOD samples cannot enable Incremental.
//! A mismatch or slower verified Incremental step permanently disables it
//! for this session; no hidden adaptation on held-out data.
use crate::fourier5_measured_dispatch::EngineChoice;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BreakerError {
    ZeroDuration,
    CounterOverflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedStep {
    pub full_ns: u128,
    pub incremental_ns: u128,
    pub same_output: bool,
    pub same_decision: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeBreaker {
    frozen: EngineChoice,
    tripped: bool,
    verified_steps: u64,
}

impl RuntimeBreaker {
    #[must_use]
    pub fn new(frozen: EngineChoice) -> Self {
        Self { frozen, tripped: false, verified_steps: 0 }
    }

    /// A tripped breaker never re-enables Incremental. Recalibration requires
    /// a new session and independent TRAIN/VALID evidence.
    #[must_use]
    pub fn choice(&self) -> EngineChoice {
        if self.tripped { EngineChoice::Full } else { self.frozen }
    }

    #[must_use]
    pub fn verified_steps(&self) -> u64 { self.verified_steps }

    /// Observe a paired exact-oracle step. The result may only REMOVE the
    /// Incremental permission; it must never promote Full to Incremental.
    /// A slower individual Incremental step is treated as a regression.
    /// Counters are checked before state mutation.
    pub fn observe(&mut self, step: VerifiedStep) -> Result<EngineChoice, BreakerError> {
        if step.full_ns == 0 || step.incremental_ns == 0 {
            return Err(BreakerError::ZeroDuration);
        }
        let next = self.verified_steps.checked_add(1)
            .ok_or(BreakerError::CounterOverflow)?;
        self.verified_steps = next;
        if !step.same_output || !step.same_decision
            || step.incremental_ns > step.full_ns {
            self.tripped = true;
        }
        Ok(self.choice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(full: u128, inc: u128, correct: bool) -> VerifiedStep {
        VerifiedStep { full_ns: full, incremental_ns: inc,
            same_output: correct, same_decision: correct }
    }

    #[test]
    fn full_is_never_promoted_even_after_fast_verified_steps() {
        let mut breaker = RuntimeBreaker::new(EngineChoice::Full);
        for _ in 0..10 {
            assert_eq!(breaker.observe(step(100, 1, true)),
                Ok(EngineChoice::Full));
        }
        assert_eq!(breaker.verified_steps(), 10);
    }

    #[test]
    fn regression_or_mismatch_latches_full_for_remaining_sequence() {
        for bad in [step(10, 11, true), step(10, 1, false),
            VerifiedStep { same_output: true, same_decision: false,
                ..step(10, 1, true) }] {
            let mut breaker = RuntimeBreaker::new(EngineChoice::Incremental);
            assert_eq!(breaker.observe(step(10, 1, true)),
                Ok(EngineChoice::Incremental));
            assert_eq!(breaker.observe(bad), Ok(EngineChoice::Full));
            assert_eq!(breaker.observe(step(100, 1, true)), Ok(EngineChoice::Full));
        }
    }

    #[test]
    fn malformed_samples_are_rejected_without_mutating_state() {
        let mut breaker = RuntimeBreaker::new(EngineChoice::Incremental);
        let before = breaker;
        assert_eq!(breaker.observe(step(0, 1, true)),
            Err(BreakerError::ZeroDuration));
        assert_eq!(breaker, before);
        assert_eq!(breaker.observe(step(1, 0, true)),
            Err(BreakerError::ZeroDuration));
        assert_eq!(breaker, before);
    }
}

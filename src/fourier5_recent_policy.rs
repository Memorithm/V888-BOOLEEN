//! Recent, paired, exact-output-verified Full/Incremental policy for synthetic circuits.
//! A bounded rolling window adapts when workload locality changes. No TEST/OOD calibration.
use crate::fourier5_hierarchy::HierarchyError;
use crate::fourier5_incremental::PackedIntervention;
use crate::fourier5_persistent_session::PersistentSession;
use crate::fourier5_sequential_adaptive::{Engine, EngineCalibration};
use std::time::Instant;

const WINDOW: usize = 16;
const MIN_PAIRS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecentPolicyError {
    Inference(HierarchyError),
    DivergentOutputs,
    InvalidCone,
    InvalidHorizon,
}

impl From<HierarchyError> for RecentPolicyError {
    fn from(error: HierarchyError) -> Self {
        Self::Inference(error)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Pair {
    full: u64,
    incremental: u64,
}

#[derive(Clone, Copy, Debug)]
struct Window {
    pairs: [Pair; WINDOW],
    len: usize,
    next: usize,
}
impl Default for Window {
    fn default() -> Self {
        Self {
            pairs: [Pair::default(); WINDOW],
            len: 0,
            next: 0,
        }
    }
}
impl Window {
    fn record(&mut self, pair: Pair) {
        self.pairs[self.next] = pair;
        self.next = (self.next + 1) % WINDOW;
        self.len = (self.len + 1).min(WINDOW);
    }
    fn choice(&self) -> Engine {
        if self.len < MIN_PAIRS {
            return Engine::Full;
        }
        let (full, incremental) = self.pairs[..self.len]
            .iter()
            .fold((0_u128, 0_u128), |(f, i), p| {
                (f + u128::from(p.full), i + u128::from(p.incremental))
            });
        if incremental * 10 < full * 9 {
            Engine::Incremental
        } else {
            Engine::Full
        }
    }
}

/// Calibration state is fixed-size and independent of the circuit size.
/// Timing must be collected on TRAIN/VALID only, never TEST/OOD.
#[derive(Clone, Debug, Default)]
pub struct RecentPairedPolicy {
    windows: [Window; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedDecision {
    pub output: u64,
    pub selected: Engine,
    pub cone_gates: usize,
    pub full_gates: usize,
    pub full_ns: u64,
    pub incremental_ns: u64,
}

impl RecentPairedPolicy {
    fn bucket(cone: usize, total: usize) -> Result<usize, RecentPolicyError> {
        if total == 0 || cone > total {
            return Err(RecentPolicyError::InvalidCone);
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

    /// Only equal outputs are admitted. Zero nanosecond readings are rounded to one
    /// to avoid a zero-clock sample causing a spurious infinite speedup.
    pub fn record_verified(
        &mut self,
        cone: usize,
        total: usize,
        full_output: u64,
        incremental_output: u64,
        full_ns: u64,
        incremental_ns: u64,
    ) -> Result<Engine, RecentPolicyError> {
        let bucket = Self::bucket(cone, total)?;
        if full_output != incremental_output {
            return Err(RecentPolicyError::DivergentOutputs);
        }
        self.windows[bucket].record(Pair {
            full: full_ns.max(1),
            incremental: incremental_ns.max(1),
        });
        Ok(self.windows[bucket].choice())
    }

    /// Choose for a *future* sequence of homogeneous-cost proposals, including
    /// the one-off cost of building or refreshing the cache. The caller must
    /// supply a conservative setup measurement and a predeclared horizon.
    /// Observations must originate exclusively from disjoint TRAIN/VALID.
    /// This never calibrates on TEST/OOD and does not mutate policy state.
    ///
    /// If the cache is already valid, setup_ns may be zero. The caller must
    /// account separately for accepted interventions that invalidate it.
    pub fn choose_amortized(
        &self,
        cone: usize,
        total: usize,
        horizon: usize,
        setup_ns: u64,
    ) -> Result<Engine, RecentPolicyError> {
        if horizon == 0 || horizon > 1_000_000 {
            return Err(RecentPolicyError::InvalidHorizon);
        }
        let window = &self.windows[Self::bucket(cone, total)?];
        if window.len < MIN_PAIRS {
            return Ok(Engine::Full);
        }
        let (full, incremental) = window.pairs[..window.len]
            .iter()
            .fold((0_u128, 0_u128), |(f, i), p| {
                (f + u128::from(p.full), i + u128::from(p.incremental))
            });
        // Compare integer cross-products, retaining the strict 10% margin.
        // u64 timings, at most 16 samples and 1e6 proposals cannot overflow u128.
        let h = horizon as u128;
        let n = window.len as u128;
        let incremental_total = incremental * h + u128::from(setup_ns) * n;
        let full_total = full * h;
        Ok(if incremental_total * 10 < full_total * 9 {
            Engine::Incremental
        } else {
            Engine::Full
        })
    }

    pub fn choose(&self, cone: usize, total: usize) -> Result<Engine, RecentPolicyError> {
        Ok(self.windows[Self::bucket(cone, total)?].choice())
    }

    /// Paired single-proposal diagnostic; both engines operate on the same accepted
    /// session. Full reconstructs its inputs independently of the cached stages.
    /// This method measures both engines, so it is not a throughput optimization.
    /// The session is never mutated; accepted interventions remain caller-owned.
    pub fn measure_verified(
        &mut self,
        session: &PersistentSession,
        proposal: PackedIntervention,
    ) -> Result<VerifiedDecision, RecentPolicyError> {
        let start = Instant::now();
        let full = session.propose_adaptive(
            proposal,
            EngineCalibration {
                full_ns_per_gate: 0,
                incremental_ns_per_gate: 0,
                cache_build_ns: 0,
            },
        )?;
        let full_ns = start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        let start = Instant::now();
        let incremental_output = session.propose(proposal)?;
        let incremental_ns = start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        let selected = self.record_verified(
            full.cone_gates,
            full.full_gates,
            full.output,
            incremental_output,
            full_ns,
            incremental_ns,
        )?;
        Ok(VerifiedDecision {
            output: full.output,
            selected,
            cone_gates: full.cone_gates,
            full_gates: full.full_gates,
            full_ns,
            incremental_ns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn window_recovers_after_workload_reversal() {
        let mut p = RecentPairedPolicy::default();
        for _ in 0..16 {
            p.record_verified(2, 100, 7, 7, 100, 10).unwrap();
        }
        assert_eq!(p.choose(2, 100), Ok(Engine::Incremental));
        for _ in 0..16 {
            p.record_verified(2, 100, 7, 7, 10, 100).unwrap();
        }
        assert_eq!(p.choose(2, 100), Ok(Engine::Full));
    }
    #[test]
    fn rejects_mismatch_without_poisoning_window() {
        let mut p = RecentPairedPolicy::default();
        for _ in 0..7 {
            p.record_verified(2, 100, 1, 1, 100, 10).unwrap();
        }
        assert_eq!(
            p.record_verified(2, 100, 1, 0, 100, 10),
            Err(RecentPolicyError::DivergentOutputs)
        );
        assert_eq!(p.choose(2, 100), Ok(Engine::Full));
        p.record_verified(2, 100, 1, 1, 100, 10).unwrap();
        assert_eq!(p.choose(2, 100), Ok(Engine::Incremental));
    }
    #[test]
    fn independent_bucket_and_strict_threshold() {
        let mut p = RecentPairedPolicy::default();
        for _ in 0..8 {
            p.record_verified(50, 100, 0, 0, 100, 90).unwrap();
        }
        assert_eq!(p.choose(50, 100), Ok(Engine::Full));
        assert_eq!(p.choose(2, 100), Ok(Engine::Full));
        assert_eq!(p.choose(101, 100), Err(RecentPolicyError::InvalidCone));
    }

    #[test]
    fn amortized_horizon_accounts_for_cache_setup_and_strict_margin() {
        let mut p = RecentPairedPolicy::default();
        for _ in 0..8 {
            p.record_verified(2, 100, 0x55, 0x55, 100, 20).unwrap();
        }
        // 8 observations: full=800, incremental=160, setup=1000.
        // One proposal is slower after setup; 20 proposals amortize it.
        assert_eq!(p.choose_amortized(2, 100, 1, 1000), Ok(Engine::Full));
        assert_eq!(
            p.choose_amortized(2, 100, 20, 1000),
            Ok(Engine::Incremental)
        );
        assert_eq!(
            p.choose_amortized(2, 100, 0, 0),
            Err(RecentPolicyError::InvalidHorizon)
        );
        assert_eq!(
            p.choose_amortized(2, 100, 1_000_001, 0),
            Err(RecentPolicyError::InvalidHorizon)
        );
        assert_eq!(
            p.choose_amortized(101, 100, 2, 0),
            Err(RecentPolicyError::InvalidCone)
        );
        assert_eq!(p.choose(2, 100), Ok(Engine::Incremental));
    }

    #[test]
    fn amortized_policy_fails_closed_without_sufficient_pairs() {
        let mut p = RecentPairedPolicy::default();
        for _ in 0..7 {
            p.record_verified(2, 100, 1, 1, 100, 1).unwrap();
        }
        assert_eq!(p.choose_amortized(2, 100, 100, 0), Ok(Engine::Full));
    }
}

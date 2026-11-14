//! Bounded heterogeneous sequential intervention profiling for synthetic circuits.
//! Uses the independent Full oracle in execute_verified_sequence for every step.
//! VmHWM is process-wide Linux peak RSS, not per-step allocation or private RSS.
use crate::fourier5_hierarchy::HierarchyError;
use crate::fourier5_incremental::PackedIntervention;
use crate::fourier5_persistent_session::PersistentSession;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequentialSample {
    pub accepted: bool,
    pub output: u64,
    pub updated_gates: usize,
    pub elapsed_ns: u128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequentialProfile {
    pub samples: Vec<SequentialSample>,
    pub total_ns: u128,
    pub accepted_count: usize,
    pub rejected_count: usize,
    pub updated_gates: usize,
    /// Linux process-wide high-water RSS in KiB, when available.
    pub vm_hwm_kib: Option<u64>,
}

/// Deterministic integer-only latency and throughput statistics for a completed trace.
/// Rates use total wall-clock step time (including the independent Full oracle).
/// The caller must report cache construction separately when comparing end-to-end runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequentialStatistics {
    pub steps: usize,
    pub min_ns: u128,
    pub median_ns: u128,
    pub p95_ns: u128,
    pub max_ns: u128,
    /// Completed interventions per second, rounded down; None for empty/zero-duration traces.
    pub steps_per_second: Option<u128>,
}

/// Separate accepted and rejected interventions before aggregating latencies.
/// This avoids hiding expensive cache updates inside a pooled percentile.
/// Both subsets include independent Full-oracle verification time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AcceptanceStatistics {
    pub accepted: Option<SequentialStatistics>,
    pub rejected: Option<SequentialStatistics>,
}

impl SequentialProfile {
    /// Calculate independent latency distributions for accepted and rejected steps.
    /// Each denominator is the sum of its own measured step durations, never
    /// the overall trace duration. Empty subsets have no statistics.
    pub fn statistics_by_acceptance(&self) -> AcceptanceStatistics {
        let subset = |accepted: bool| {
            let samples: Vec<SequentialSample> = self.samples.iter().copied()
                .filter(|sample| sample.accepted == accepted).collect();
            let total_ns = samples.iter().fold(0_u128, |sum, sample| {
                sum.saturating_add(sample.elapsed_ns)
            });
            SequentialProfile {
                samples,
                total_ns,
                accepted_count: 0,
                rejected_count: 0,
                updated_gates: 0,
                vm_hwm_kib: None,
            }.statistics()
        };
        AcceptanceStatistics {
            accepted: subset(true),
            rejected: subset(false),
        }
    }

    /// Nearest-rank p95; median is the lower middle observation for even samples.
    /// All calculations are overflow-safe and avoid floating-point precision loss.
    pub fn statistics(&self) -> Option<SequentialStatistics> {
        if self.samples.is_empty() {
            return None;
        }
        let mut durations: Vec<u128> = self.samples.iter().map(|s| s.elapsed_ns).collect();
        durations.sort_unstable();
        let n = durations.len();
        let p95_index = n.saturating_mul(95).saturating_add(99) / 100;
        let rate = if self.total_ns == 0 {
            None
        } else {
            Some((n as u128).saturating_mul(1_000_000_000) / self.total_ns)
        };
        Some(SequentialStatistics {
            steps: n,
            min_ns: durations[0],
            median_ns: durations[(n - 1) / 2],
            p95_ns: durations[p95_index.saturating_sub(1).min(n - 1)],
            max_ns: durations[n - 1],
            steps_per_second: rate,
        })
    }
}

/// Cold-start trace: measures cache construction separately from verified steps.
/// The reported end-to-end time includes construction and all oracle checks.
/// VmHWM remains process-wide and must not be interpreted as per-run allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColdSequentialProfile {
    pub cache_build_ns: u128,
    pub end_to_end_ns: u128,
    pub trace: SequentialProfile,
}

impl ColdSequentialProfile {
    /// End-to-end completed interventions per second, including cold cache build
    /// and the independent Full oracle. Returns None for empty or zero-time traces.
    /// This is a diagnostic throughput, NOT unverified production inference speed.
    pub fn end_to_end_steps_per_second(&self) -> Option<u128> {
        if self.trace.samples.is_empty() || self.end_to_end_ns == 0 {
            return None;
        }
        Some(
            (self.trace.samples.len() as u128)
                .saturating_mul(1_000_000_000)
                / self.end_to_end_ns,
        )
    }

    /// Fraction of end-to-end time spent constructing the cold cache,
    /// expressed in parts per million, rounded down.
    /// None means a zero-duration or inconsistent trace.
    pub fn cache_build_share_ppm(&self) -> Option<u32> {
        if self.end_to_end_ns == 0 || self.cache_build_ns > self.end_to_end_ns {
            return None;
        }
        let share = self.cache_build_ns
            .saturating_mul(1_000_000)
            / self.end_to_end_ns;
        Some(share.min(1_000_000) as u32)
    }
}

/// Construct a fresh cache, then execute the exact verified sequential trace.
/// This prevents comparing a warm incremental cache with a cold Full baseline.
pub fn profile_cold_verified_sequence(
    model: crate::fourier5_hierarchy::FiveCircleHierarchy,
    planes: Vec<u64>,
    steps: &[(PackedIntervention, bool)],
    max_steps: usize,
) -> Result<ColdSequentialProfile, HierarchyError> {
    // Validate the step budget before allocating or building the cache.
    if steps.len() > max_steps {
        return Err(HierarchyError::InvalidInput);
    }
    let start = Instant::now();
    let mut session = PersistentSession::new(model, planes)?;
    let cache_build_ns = start.elapsed().as_nanos();
    let trace = profile_verified_sequence(&mut session, steps, max_steps)?;
    let end_to_end_ns = cache_build_ns.saturating_add(trace.total_ns);
    Ok(ColdSequentialProfile {
        cache_build_ns,
        end_to_end_ns,
        trace,
    })
}

/// Read Linux process-wide peak RSS. None on other OSes or malformed/unavailable procfs.
/// This counter is cumulative over the process lifetime and cannot isolate this call.
pub fn process_vm_hwm_kib() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        parse_vm_hwm_kib(&status)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn parse_vm_hwm_kib(status: &str) -> Option<u64> {
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let mut fields = line["VmHWM:".len()..].split_whitespace();
    let amount = fields.next()?.parse::<u64>().ok()?;
    if fields.next()? != "kB" || fields.next().is_some() {
        return None;
    }
    Some(amount)
}

/// Profile a sequence of accepted/rejected changes against the evolving exact
/// Full oracle. Each step includes proposal, oracle and cache update timings.
/// The bound is checked before any mutation. Later invalid steps may leave an
/// already accepted prefix applied; callers must not assume transactionality.
pub fn profile_verified_sequence(
    session: &mut PersistentSession,
    steps: &[(PackedIntervention, bool)],
    max_steps: usize,
) -> Result<SequentialProfile, HierarchyError> {
    if steps.len() > max_steps {
        return Err(HierarchyError::InvalidInput);
    }
    let mut samples = Vec::with_capacity(steps.len());
    let mut total_ns = 0_u128;
    let mut accepted_count = 0_usize;
    let mut rejected_count = 0_usize;
    let mut updated_gates = 0_usize;
    for &(proposal, accepted) in steps {
        let start = Instant::now();
        let result = session.execute_verified_sequence(&[(proposal, accepted)], 1)?;
        let elapsed_ns = start.elapsed().as_nanos();
        let (output, gates) = result[0];
        total_ns = total_ns.saturating_add(elapsed_ns);
        updated_gates = updated_gates.saturating_add(gates);
        if accepted {
            accepted_count += 1;
        } else {
            rejected_count += 1;
        }
        samples.push(SequentialSample {
            accepted,
            output,
            updated_gates: gates,
            elapsed_ns,
        });
    }
    Ok(SequentialProfile {
        samples,
        total_ns,
        accepted_count,
        rejected_count,
        updated_gates,
        vm_hwm_kib: process_vm_hwm_kib(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5_hierarchy::{FiveCircleHierarchy, GateKind, LeafRouting};

    #[test]
    fn cold_metrics_include_setup_and_reject_inconsistent_time() {
        let sample = SequentialSample {
            accepted: true, output: 1, updated_gates: 3, elapsed_ns: 200,
        };
        let trace = SequentialProfile {
            samples: vec![sample],
            total_ns: 200,
            accepted_count: 1,
            rejected_count: 0,
            updated_gates: 3,
            vm_hwm_kib: None,
        };
        let cold = ColdSequentialProfile {
            cache_build_ns: 800,
            end_to_end_ns: 1000,
            trace: trace.clone(),
        };
        assert_eq!(cold.end_to_end_steps_per_second(), Some(1_000_000));
        assert_eq!(cold.cache_build_share_ppm(), Some(800_000));
        assert_eq!(cold.trace.statistics().unwrap().steps_per_second, Some(5_000_000));
        let empty = ColdSequentialProfile {
            cache_build_ns: 0,
            end_to_end_ns: 0,
            trace: SequentialProfile { samples: vec![], ..trace.clone() },
        };
        assert_eq!(empty.end_to_end_steps_per_second(), None);
        assert_eq!(empty.cache_build_share_ppm(), None);
        let inconsistent = ColdSequentialProfile {
            cache_build_ns: 2,
            end_to_end_ns: 1,
            trace,
        };
        assert_eq!(inconsistent.cache_build_share_ppm(), None);
    }

    #[test]
    fn acceptance_statistics_keep_heterogeneous_costs_separate() {
        let mut profile = SequentialProfile {
            samples: Vec::new(),
            total_ns: 0,
            accepted_count: 0,
            rejected_count: 0,
            updated_gates: 0,
            vm_hwm_kib: None,
        };
        assert_eq!(profile.statistics_by_acceptance(), AcceptanceStatistics {
            accepted: None, rejected: None,
        });
        for (accepted, elapsed_ns) in [(true, 100_u128), (false, 2), (true, 300), (false, 4)] {
            profile.samples.push(SequentialSample {
                accepted, output: 0, updated_gates: 0, elapsed_ns,
            });
            profile.total_ns += elapsed_ns;
        }
        let stats = profile.statistics_by_acceptance();
        let accepted = stats.accepted.unwrap();
        let rejected = stats.rejected.unwrap();
        assert_eq!((accepted.steps, accepted.min_ns, accepted.max_ns), (2, 100, 300));
        assert_eq!((rejected.steps, rejected.min_ns, rejected.max_ns), (2, 2, 4));
        assert_eq!(accepted.steps_per_second, Some(5_000_000));
        assert_eq!(rejected.steps_per_second, Some(333_333_333));
        assert_ne!(accepted.steps_per_second, profile.statistics().unwrap().steps_per_second);
    }

    #[test]
    fn integer_statistics_cover_nonuniform_latency_and_empty_trace() {
        let mut profile = SequentialProfile {
            samples: Vec::new(),
            total_ns: 0,
            accepted_count: 0,
            rejected_count: 0,
            updated_gates: 0,
            vm_hwm_kib: None,
        };
        assert_eq!(profile.statistics(), None);
        for duration in [100_u128, 1, 50, 10, 20, 5, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17] {
            profile.samples.push(SequentialSample {
                accepted: false,
                output: 0,
                updated_gates: 0,
                elapsed_ns: duration,
            });
            profile.total_ns += duration;
        }
        let stats = profile.statistics().unwrap();
        assert_eq!(stats.steps, 20);
        assert_eq!(stats.min_ns, 1);
        assert_eq!(stats.median_ns, 10);
        assert_eq!(stats.p95_ns, 50);
        assert_eq!(stats.max_ns, 100);
        assert_eq!(stats.steps_per_second, Some(20_000_000_000 / profile.total_ns));
        profile.total_ns = 0;
        assert_eq!(profile.statistics().unwrap().steps_per_second, None);
    }

    #[test]
    fn cold_trace_accounts_for_build_and_rejects_oversized_budget() {
        let model = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity],
            LeafRouting::SeededShuffle(71),
        ).unwrap();
        let planes = vec![0x55aa_55aa_55aa_55aa; model.input_bits()];
        let steps = [(PackedIntervention::LeafSwap { first: 0, second: 1 }, true)];
        assert!(matches!(
            profile_cold_verified_sequence(model.clone(), planes.clone(), &steps, 0),
            Err(HierarchyError::InvalidInput)
        ));
        let cold = profile_cold_verified_sequence(model.clone(), planes.clone(), &steps, 1).unwrap();
        assert_eq!(cold.trace.samples.len(), 1);
        assert_eq!(cold.trace.accepted_count, 1);
        assert_eq!(cold.end_to_end_ns, cold.cache_build_ns + cold.trace.total_ns);
        let mut oracle_routing = model.routing().to_vec();
        oracle_routing.swap(0, 1);
        let oracle = model.rerouted(oracle_routing).unwrap();
        assert_eq!(cold.trace.samples[0].output, oracle.infer_bitplanes(&planes).unwrap());
    }

    #[test]
    fn parses_linux_hwm_strictly() {
        assert_eq!(parse_vm_hwm_kib("Name:\ttest\nVmHWM:\t1234 kB\n"), Some(1234));
        assert_eq!(parse_vm_hwm_kib("VmHWM:\t12 MB\n"), None);
        assert_eq!(parse_vm_hwm_kib("VmHWM:\tbad kB\n"), None);
        assert_eq!(parse_vm_hwm_kib("VmRSS:\t1234 kB\n"), None);
    }

    #[test]
    fn heterogeneous_sequence_matches_independent_full_and_counts() {
        let base = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity, GateKind::Or],
            LeafRouting::SeededShuffle(53),
        ).unwrap();
        let planes: Vec<u64> = (0..base.input_bits())
            .map(|i| (i as u64 + 17).wrapping_mul(0x9e3779b97f4a7c15))
            .collect();
        let mut session = PersistentSession::new(base.clone(), planes.clone()).unwrap();
        let steps = [
            (PackedIntervention::LeafSwap { first: 0, second: 19 }, true),
            (PackedIntervention::LeafSwap { first: 3, second: 27 }, false),
            (PackedIntervention::LeafSwap { first: 7, second: 38 }, true),
        ];
        assert!(matches!(
            profile_verified_sequence(&mut session, &steps, 2),
            Err(HierarchyError::InvalidInput)
        ));
        assert_eq!(session.output(), base.infer_bitplanes(&planes).unwrap());
        let profile = profile_verified_sequence(&mut session, &steps, 3).unwrap();
        assert_eq!(profile.samples.len(), 3);
        assert_eq!((profile.accepted_count, profile.rejected_count), (2, 1));
        assert_eq!(profile.samples[1].updated_gates, 0);
        assert_eq!(
            profile.updated_gates,
            profile.samples.iter().map(|s| s.updated_gates).sum::<usize>()
        );
        let mut oracle = base;
        for ((proposal, accepted), sample) in steps.iter().zip(&profile.samples) {
            let PackedIntervention::LeafSwap { first, second } = *proposal else { unreachable!() };
            let mut routing = oracle.routing().to_vec();
            routing.swap(first, second);
            let candidate = oracle.rerouted(routing).unwrap();
            assert_eq!(sample.output, candidate.infer_bitplanes(&planes).unwrap());
            if *accepted { oracle = candidate; }
        }
        assert_eq!(session.output(), oracle.infer_bitplanes(&planes).unwrap());
    }
}

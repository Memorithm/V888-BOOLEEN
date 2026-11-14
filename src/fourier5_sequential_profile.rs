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

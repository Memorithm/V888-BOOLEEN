//! Phase-local diagnostics for nonstationary, oracle-verified sequential traces.
//! Each phase is contiguous: no mixing of early cold and late warm interventions.
use crate::fourier5_sequential_profile::SequentialSample;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseCost {
    pub start: usize,
    pub end: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub updated_gates: usize,
    pub elapsed_ns: u128,
    /// Diagnostic verified interventions/s; includes Full-oracle time.
    pub verified_steps_per_second: Option<u128>,
}

/// Partition samples into consecutive non-overlapping phases.
/// A zero window is invalid. The final phase may be shorter.
/// The result is descriptive, never a source of training labels.
pub fn phase_costs(samples: &[SequentialSample], window: usize) -> Option<Vec<PhaseCost>> {
    if window == 0 {
        return None;
    }
    let mut phases = Vec::new();
    for (phase, chunk) in samples.chunks(window).enumerate() {
        let start = phase.checked_mul(window)?;
        let end = start.checked_add(chunk.len())?;
        let accepted = chunk.iter().filter(|s| s.accepted).count();
        let updated_gates = chunk.iter().try_fold(0usize, |a, s| a.checked_add(s.updated_gates))?;
        let elapsed_ns = chunk.iter().try_fold(0u128, |a, s| a.checked_add(s.elapsed_ns))?;
        let verified_steps_per_second = if elapsed_ns == 0 {
            None
        } else {
            Some((chunk.len() as u128).checked_mul(1_000_000_000)? / elapsed_ns)
        };
        phases.push(PhaseCost {
            start, end, accepted, rejected: chunk.len() - accepted,
            updated_gates, elapsed_ns, verified_steps_per_second,
        });
    }
    Some(phases)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample(accepted: bool, ns: u128, gates: usize) -> SequentialSample {
        SequentialSample { accepted, output: 0, updated_gates: gates, elapsed_ns: ns }
    }

    #[test]
    fn phases_preserve_order_and_detect_regime_shift() {
        let trace = [
            sample(true, 100, 7), sample(false, 100, 0),
            sample(true, 10, 2), sample(true, 10, 3),
            sample(false, 5, 0),
        ];
        let phases = phase_costs(&trace, 2).unwrap();
        assert_eq!(phases.len(), 3);
        assert_eq!((phases[0].start, phases[0].end, phases[0].accepted, phases[0].updated_gates), (0, 2, 1, 7));
        assert_eq!((phases[1].start, phases[1].end, phases[1].accepted, phases[1].updated_gates), (2, 4, 2, 5));
        assert_eq!((phases[2].start, phases[2].end, phases[2].rejected), (4, 5, 1));
        assert_eq!(phases[0].verified_steps_per_second, Some(10_000_000));
        assert_eq!(phases[1].verified_steps_per_second, Some(100_000_000));
        // Independent scalar oracle for total measured time and gate work.
        assert_eq!(phases.iter().map(|p| p.elapsed_ns).sum::<u128>(), 225);
        assert_eq!(phases.iter().map(|p| p.updated_gates).sum::<usize>(), 12);
    }

    #[test]
    fn empty_zero_duration_and_invalid_window() {
        assert_eq!(phase_costs(&[], 1), Some(vec![]));
        assert_eq!(phase_costs(&[], 0), None);
        let phases = phase_costs(&[sample(false, 0, 0)], 8).unwrap();
        assert_eq!(phases[0].verified_steps_per_second, None);
    }

    #[test]
    fn arithmetic_overflow_is_rejected_not_silently_saturated() {
        assert_eq!(phase_costs(&[sample(true, u128::MAX, 0), sample(true, 1, 0)], 2), None);
        assert_eq!(phase_costs(&[sample(true, 1, usize::MAX), sample(true, 1, 1)], 2), None);
    }
}

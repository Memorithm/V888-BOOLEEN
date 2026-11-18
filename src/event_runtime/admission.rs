//! Conservative opt-in admission from a measured, topology-bounded cost model.
//!
//! The default parallel scheduler keeps its historical fixed thresholds.
//! This model affects only serial-vs-parallel scheduling, never Boolean rules.

/// Physical-time coefficients supplied by an external paired measurement.
///
/// All fields are explicit. These numbers are NOT derived from V888 biology
/// and are NOT validated measurements merely because this type was built.
/// Measurements must include both scoped-thread passes and be acquired under
/// the same hardware affinity, cache configuration and worker count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeasuredDisjointCosts {
    /// Requested worker count under which the coefficients were obtained.
    pub workers: usize,
    /// Measured topology envelope. A group outside any bound runs serially.
    pub min_producers: usize,
    pub max_producers: usize,
    pub min_effective_contacts: usize,
    pub max_effective_contacts: usize,
    pub min_distinct_targets: usize,
    pub max_distinct_targets: usize,
    /// Combined fixed cost of both scoped-thread launch/join passes.
    pub two_pass_overhead_ns: u64,
    /// Positive per-item estimates from independent serial/reference trials.
    pub serial_contact_ns: u64,
    pub serial_target_ns: u64,
    /// Positive per-item estimates from parallel trials.
    pub parallel_contact_ns: u64,
    pub parallel_target_ns: u64,
}

impl MeasuredDisjointCosts {
    pub(super) fn valid(self) -> bool {
        (2..=32).contains(&self.workers)
            && self.min_producers >= 2
            && self.max_producers >= self.min_producers
            && self.min_effective_contacts > 0
            && self.max_effective_contacts >= self.min_effective_contacts
            && self.min_distinct_targets >= 2
            && self.max_distinct_targets >= self.min_distinct_targets
            && self.two_pass_overhead_ns > 0
            && self.serial_contact_ns > 0
            && self.serial_target_ns > 0
            && self.parallel_contact_ns > 0
            && self.parallel_target_ns > 0
    }

    /// Use a positive predicted margin only *inside* the measured envelope.
    /// Each estimate has checked u128 arithmetic; overflow refuses parallel.
    pub(super) fn admits(
        self,
        producers: usize,
        contacts: usize,
        targets: usize,
        workers: usize,
    ) -> bool {
        if !self.valid()
            || self.workers != workers
            || producers < self.min_producers
            || producers > self.max_producers
            || contacts < self.min_effective_contacts
            || contacts > self.max_effective_contacts
            || targets < self.min_distinct_targets
            || targets > self.max_distinct_targets
        {
            return false;
        }
        let prep_threads = workers.min(producers);
        let writer_threads = workers.min(targets);
        let serial = (contacts as u128)
            .checked_mul(u128::from(self.serial_contact_ns))
            .and_then(|ns| {
                (targets as u128)
                    .checked_mul(u128::from(self.serial_target_ns))
                    .and_then(|extra| ns.checked_add(extra))
            });
        let parallel = (contacts.div_ceil(prep_threads) as u128)
            .checked_mul(u128::from(self.parallel_contact_ns))
            .and_then(|ns| {
                (targets.div_ceil(writer_threads) as u128)
                    .checked_mul(u128::from(self.parallel_target_ns))
                    .and_then(|extra| ns.checked_add(extra))
            })
            .and_then(|ns| ns.checked_add(u128::from(self.two_pass_overhead_ns)));
        matches!((serial, parallel), (Some(serial), Some(parallel)) if parallel < serial)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured() -> MeasuredDisjointCosts {
        // Synthetic coefficients test the decision formula, not physical data.
        MeasuredDisjointCosts {
            workers: 4,
            min_producers: 2,
            max_producers: 16,
            min_effective_contacts: 8,
            max_effective_contacts: 100_000,
            min_distinct_targets: 8,
            max_distinct_targets: 10_000,
            two_pass_overhead_ns: 100,
            serial_contact_ns: 15,
            serial_target_ns: 10,
            parallel_contact_ns: 2,
            parallel_target_ns: 3,
        }
    }

    #[test]
    fn topology_and_worker_envelope_gate_parallelism() {
        let costs = measured();
        assert!(costs.admits(2, 8, 8, 4));
        assert!(!costs.admits(1, 8, 8, 4));
        assert!(!costs.admits(2, 7, 8, 4));
        assert!(!costs.admits(2, 8, 7, 4));
        assert!(!costs.admits(2, 8, 8, 3));
        assert!(!costs.admits(17, 8, 8, 4));
        assert!(!costs.admits(2, 100_001, 8, 4));
    }

    #[test]
    fn measured_cost_can_refuse_parallel_even_on_large_groups() {
        let mut costs = measured();
        costs.two_pass_overhead_ns = 1_000_000_000;
        assert!(!costs.admits(8, 65_536, 1_024, 4));
    }

    #[test]
    fn invalid_and_overflowing_profiles_fail_closed() {
        let mut costs = measured();
        costs.parallel_contact_ns = 0;
        assert!(!costs.valid());
        assert!(!costs.admits(2, 8, 8, 4));
        costs = measured();
        costs.min_distinct_targets = 10;
        costs.max_distinct_targets = 8;
        assert!(!costs.valid());
        costs = measured();
        costs.max_effective_contacts = usize::MAX;
        costs.max_distinct_targets = usize::MAX;
        costs.serial_contact_ns = u64::MAX;
        costs.serial_target_ns = u64::MAX;
        costs.parallel_contact_ns = u64::MAX;
        costs.parallel_target_ns = u64::MAX;
        assert!(!costs.admits(8, usize::MAX, usize::MAX, 4));
    }
}

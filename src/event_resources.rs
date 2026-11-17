//! An experimental, additive logical-work algebra for Boolean events.
//!
//! Approach A: account for distinct observable operations after replay.
//! Approach B: allow a caller to apply explicit prospective budgets using
//! `EventRuntime::run_with_contact_budget`. Neither is a physical energy
//! model unless externally calibrated and independently measured.
//!
//! Coefficients are caller-provided dimensionless proxy prices; this module
//! deliberately has no defaults labelled as joules or watts.

use crate::event_runtime::EventWork;

/// Named axes prevent an arbitrary scalar cost from hiding which work occurred.
/// Rise/fall replace the redundant total state-transition axis.
#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceAxis {
    Events,
    RisingFronts,
    FallingFronts,
    ContactsExamined,
    ThresholdChecks,
    WakeupsInserted,
    WakeupsCoalesced,
}

pub const RESOURCE_AXES: usize = 7;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EventResourceVector {
    counters: [u128; RESOURCE_AXES],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EventResourceTariffs {
    /// Exactly one nonnegative synthetic cost per ResourceAxis.
    /// Coefficients must be set by the caller; they are NOT physical units.
    pub coefficients: [u128; RESOURCE_AXES],
}

impl EventResourceVector {
    /// The zero vector represents no observed logical work.
    #[must_use]
    pub fn zero() -> Self {
        Self::default()
    }

    /// Produce an exact vector for a bounded event-execution slice.
    #[must_use]
    pub fn from_work(work: EventWork) -> Self {
        let wide = |n: usize| u128::try_from(n).expect("usize always fits in u128");
        Self {
            counters: [
                wide(work.events_consumed),
                wide(work.rising_transitions),
                wide(work.falling_transitions),
                wide(work.contacts_examined),
                wide(work.threshold_checks),
                wide(work.wakeups_inserted),
                wide(work.wakeups_coalesced),
            ],
        }
    }

    #[must_use]
    pub fn amount(&self, axis: ResourceAxis) -> u128 {
        self.counters[axis as usize]
    }

    /// Add work from disjoint execution slices; no saturating wrap or float.
    #[must_use]
    pub fn checked_add(self, other: Self) -> Option<Self> {
        let mut counters = [0; RESOURCE_AXES];
        for (index, cell) in counters.iter_mut().enumerate() {
            *cell = self.counters[index].checked_add(other.counters[index])?;
        }
        Some(Self { counters })
    }

    /// Dot product in caller-defined arbitrary proxy units.
    /// Returns None if multiplication or summation exceeds u128.
    /// This cannot be interpreted as measured energy without calibration.
    #[must_use]
    pub fn checked_proxy_score(&self, tariffs: EventResourceTariffs) -> Option<u128> {
        self.counters
            .iter()
            .zip(tariffs.coefficients.iter())
            .try_fold(0_u128, |total, (&count, &price)| {
                total.checked_add(count.checked_mul(price)?)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_work() -> EventWork {
        EventWork {
            events_consumed: 3,
            state_transitions: 2,
            rising_transitions: 1,
            falling_transitions: 1,
            contacts_examined: 5,
            threshold_checks: 1,
            wakeups_inserted: 2,
            wakeups_coalesced: 1,
        }
    }

    #[test]
    fn vector_is_exact_and_front_polarity_is_observable() {
        let vector = EventResourceVector::from_work(sample_work());
        assert_eq!(vector.amount(ResourceAxis::Events), 3);
        assert_eq!(vector.amount(ResourceAxis::RisingFronts), 1);
        assert_eq!(vector.amount(ResourceAxis::FallingFronts), 1);
        assert_eq!(vector.amount(ResourceAxis::ContactsExamined), 5);
        assert_eq!(vector.amount(ResourceAxis::ThresholdChecks), 1);
        assert_eq!(vector.amount(ResourceAxis::WakeupsInserted), 2);
        assert_eq!(vector.amount(ResourceAxis::WakeupsCoalesced), 1);
    }

    #[test]
    fn checked_addition_is_associative_with_zero_in_safe_range() {
        let work = EventResourceVector::from_work(sample_work());
        let zero = EventResourceVector::zero();
        assert_eq!(work.checked_add(zero), Some(work));
        assert_eq!(zero.checked_add(work), Some(work));
        let twice = work.checked_add(work).unwrap();
        let thrice_left = twice.checked_add(work);
        let thrice_right = work.checked_add(twice);
        assert_eq!(thrice_left, thrice_right);
        assert_eq!(twice.amount(ResourceAxis::ContactsExamined), 10);
    }

    #[test]
    fn price_is_explicit_linear_proxy_and_catches_overflow() {
        let work = EventResourceVector::from_work(sample_work());
        let tariffs = EventResourceTariffs {
            coefficients: [1, 2, 3, 4, 5, 6, 7],
        };
        assert_eq!(
            work.checked_proxy_score(tariffs),
            Some(3 + 2 + 3 + 20 + 5 + 12 + 7)
        );
        let twice = work.checked_add(work).unwrap();
        assert_eq!(twice.checked_proxy_score(tariffs), Some(104));
        let large = EventResourceTariffs {
            coefficients: [u128::MAX; RESOURCE_AXES],
        };
        assert_eq!(work.checked_proxy_score(large), None);
    }

    #[test]
    fn overflow_in_vector_addition_returns_none_not_wrap() {
        let huge = EventResourceVector {
            counters: [u128::MAX; RESOURCE_AXES],
        };
        let one = EventResourceVector {
            counters: [1; RESOURCE_AXES],
        };
        assert_eq!(huge.checked_add(one), None);
    }
}

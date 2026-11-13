//! Explicit eligibility traces for *delayed* supervised local plasticity.
//!
//! The bank captures ONLY presynaptic five-bit data/context pairs, not
//! rewards or labels. The network then runs synchronous recurrent ticks.
//! A later teacher signal updates per-neuron gains from the stored CUE,
//! not from the potentially unrelated state at reward time.
//! This is a bounded computational hypothesis, NOT a biological STDP rule.
use crate::fourier5_recurrent::{
    RecurrentError, RecurrentFiveCircle, TickStats, MAX_NETWORK_NODES, MIN_NETWORK_NODES,
};

pub const MAX_TRACE_DELAY: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EligibilityError {
    InvalidPopulation,
    InvalidDelay,
    PopulationMismatch,
    OutstandingCue,
    NoCue,
    NoElapsedTick,
    Expired,
    InvalidTargets,
    Recurrent(RecurrentError),
}

impl From<RecurrentError> for EligibilityError {
    fn from(err: RecurrentError) -> Self {
        Self::Recurrent(err)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeedbackReport {
    pub age: usize,
    pub updated_neurons: usize,
    pub supervised_neurons: usize,
}

/// One outstanding cue per neuron, with a shared bounded delay counter.
/// Approximate cue storage: 2 bytes per neuron excluding Vec metadata.
#[derive(Clone, Debug)]
pub struct EligibilityBank {
    nodes: usize,
    max_delay: usize,
    age: usize,
    pending: Option<Vec<(u8, u8)>>,
}

impl EligibilityBank {
    pub fn new(nodes: usize, max_delay: usize) -> Result<Self, EligibilityError> {
        if !(MIN_NETWORK_NODES..=MAX_NETWORK_NODES).contains(&nodes) {
            return Err(EligibilityError::InvalidPopulation);
        }
        if !(1..=MAX_TRACE_DELAY).contains(&max_delay) {
            return Err(EligibilityError::InvalidDelay);
        }
        Ok(Self {
            nodes,
            max_delay,
            age: 0,
            pending: None,
        })
    }

    #[must_use]
    pub fn cue_age(&self) -> Option<usize> {
        self.pending.as_ref().map(|_| self.age)
    }

    #[must_use]
    pub fn stored_cue_bytes(&self) -> usize {
        self.pending
            .as_ref()
            .map_or(0, |p| std::mem::size_of_val(p.as_slice()))
    }

    /// New cue is refused while the previous unconsumed trace exists.
    pub fn capture(&mut self, network: &RecurrentFiveCircle) -> Result<(), EligibilityError> {
        if network.node_count() != self.nodes {
            return Err(EligibilityError::PopulationMismatch);
        }
        if self.pending.is_some() {
            return Err(EligibilityError::OutstandingCue);
        }
        let traces = network.capture_local_signals()?;
        self.pending = Some(traces);
        self.age = 0;
        Ok(())
    }

    /// Recurrent dynamics continue while local presynaptic cues remain
    /// frozen. Enforces max_delay BEFORE doing another expensive tick.
    pub fn tick(&mut self, network: &mut RecurrentFiveCircle) -> Result<TickStats, EligibilityError> {
        if self.pending.is_none() {
            return Err(EligibilityError::NoCue);
        }
        if network.node_count() != self.nodes {
            return Err(EligibilityError::PopulationMismatch);
        }
        if self.age >= self.max_delay {
            return Err(EligibilityError::Expired);
        }
        let stats = network.step()?;
        self.age += 1;
        Ok(stats)
    }

    /// Delayed supervision modifies per-neuron gains only; no state or
    /// contact index is modified. Labels arrive ONLY here, never at cue.
    /// Successful feedback consumes the trace. A failed validation does
    /// not consume it or partially change neuron gains.
    pub fn reinforce(
        &mut self,
        network: &mut RecurrentFiveCircle,
        labels: &[bool],
    ) -> Result<FeedbackReport, EligibilityError> {
        let Some(pending) = self.pending.as_ref() else {
            return Err(EligibilityError::NoCue);
        };
        if self.age == 0 {
            return Err(EligibilityError::NoElapsedTick);
        }
        if self.age > self.max_delay {
            return Err(EligibilityError::Expired);
        }
        if network.node_count() != self.nodes {
            return Err(EligibilityError::PopulationMismatch);
        }
        if labels.len() != self.nodes {
            return Err(EligibilityError::InvalidTargets);
        }
        let updated = network.apply_delayed_feedback(pending, labels)?;
        let report = FeedbackReport {
            age: self.age,
            updated_neurons: updated,
            supervised_neurons: self.nodes,
        };
        self.pending = None;
        self.age = 0;
        Ok(report)
    }

    /// Explicitly discard stale unsupervised cues. No model update.
    pub fn discard(&mut self) {
        self.pending = None;
        self.age = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5::{Circle, FiveCircle, CIRCLES};
    use crate::fourier5_dynamic::{GatedFiveCircle, RadiusGate};
    use crate::fourier5_recurrent::{NetworkConfig, SyntheticTopology};

    fn fixture(nodes: usize) -> RecurrentFiveCircle {
        let mut circles = [Circle::default(); CIRCLES];
        circles[0] = Circle {
            mask: 31,
            radius: 2,
            phase_pi: false,
        };
        let base = FiveCircle::new(circles, -1).unwrap();
        let gates = std::array::from_fn(|i| RadiusGate {
            context_mask: if i == 0 { 0 } else { 1 << (i - 1) },
            boost: 0,
        });
        let model = GatedFiveCircle::new(base, gates).unwrap();
        RecurrentFiveCircle::seeded(
            NetworkConfig {
                nodes,
                topology: SyntheticTopology::LocalRing,
                seed: 77,
                initial_activity_per_256: 128,
            },
            model,
        )
        .unwrap()
    }

    #[test]
    fn captures_exact_original_cue_and_delivers_reward_after_ticks() {
        let mut net = fixture(128);
        let signals = net.capture_local_signals().unwrap();
        let mut bank = EligibilityBank::new(128, 16).unwrap();
        bank.capture(&net).unwrap();
        assert_eq!(bank.stored_cue_bytes(), 256);
        assert_eq!(bank.cue_age(), Some(0));
        assert_eq!(bank.capture(&net), Err(EligibilityError::OutstandingCue));
        let labels: Vec<_> = signals
            .iter()
            .map(|&(input, context)| input.count_ones() % 2 == 1 && context & 1 == 0)
            .collect();
        for _ in 0..8 {
            bank.tick(&mut net).unwrap();
        }
        assert_eq!(bank.cue_age(), Some(8));
        let fingerprint_before = net.wiring_fingerprint();
        let state_before = net.state().clone();
        let report = bank.reinforce(&mut net, &labels).unwrap();
        assert_eq!(report.age, 8);
        assert_eq!(report.supervised_neurons, 128);
        assert!(report.updated_neurons > 0);
        assert_eq!(net.wiring_fingerprint(), fingerprint_before);
        assert_eq!(net.state(), &state_before);
        assert_eq!(bank.cue_age(), None);
        assert_eq!(bank.reinforce(&mut net, &labels), Err(EligibilityError::NoCue));
    }

    #[test]
    fn rejects_expired_trace_without_extra_tick_or_model_updates() {
        let mut net = fixture(64);
        let mut bank = EligibilityBank::new(64, 2).unwrap();
        bank.capture(&net).unwrap();
        bank.tick(&mut net).unwrap();
        bank.tick(&mut net).unwrap();
        let time = net.tick_count();
        let model = net.neuron_model(0);
        assert_eq!(bank.tick(&mut net), Err(EligibilityError::Expired));
        assert_eq!(net.tick_count(), time);
        assert_eq!(net.neuron_model(0), model);
        bank.discard();
        assert_eq!(bank.cue_age(), None);
        bank.capture(&net).unwrap();
        assert_eq!(bank.cue_age(), Some(0));
    }

    #[test]
    fn invalid_target_length_keeps_trace_and_model() {
        let mut net = fixture(64);
        let mut bank = EligibilityBank::new(64, 3).unwrap();
        bank.capture(&net).unwrap();
        assert_eq!(bank.reinforce(&mut net, &[false; 64]), Err(EligibilityError::NoElapsedTick));
        bank.tick(&mut net).unwrap();
        let before = net.neuron_model(0);
        assert_eq!(bank.reinforce(&mut net, &[false; 63]), Err(EligibilityError::InvalidTargets));
        assert_eq!(bank.cue_age(), Some(1));
        assert_eq!(net.neuron_model(0), before);
    }

    #[test]
    fn resource_and_population_guards_are_explicit() {
        assert!(matches!(EligibilityBank::new(63, 1), Err(EligibilityError::InvalidPopulation)));
        assert!(matches!(EligibilityBank::new(250_001, 1), Err(EligibilityError::InvalidPopulation)));
        assert!(matches!(EligibilityBank::new(64, 0), Err(EligibilityError::InvalidDelay)));
        assert!(matches!(
            EligibilityBank::new(64, MAX_TRACE_DELAY + 1),
            Err(EligibilityError::InvalidDelay)
        ));
        let mut net = fixture(64);
        let mut bank = EligibilityBank::new(65, 1).unwrap();
        assert_eq!(bank.capture(&net), Err(EligibilityError::PopulationMismatch));
        assert_eq!(bank.tick(&mut net), Err(EligibilityError::NoCue));
    }

    #[test]
    fn state_reset_preserves_learned_models_and_wiring() {
        let mut net = fixture(64);
        let original = net.state().clone();
        let fingerprint = net.wiring_fingerprint();
        let model = net.neuron_model(3);
        let mut switched = original.clone();
        switched.set(4, !switched.get(4).unwrap()).unwrap();
        assert!(net.replace_state(switched.clone()).is_ok());
        assert_eq!(net.state(), &switched);
        assert_eq!(net.wiring_fingerprint(), fingerprint);
        assert_eq!(net.neuron_model(3), model);
        assert_eq!(net.replace_state(crate::BooleanState::new(65)), Err(RecurrentError::MismatchedNetwork));
        assert_eq!(net.state(), &switched);
    }
}

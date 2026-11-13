//! Opt-in delayed-feedback eligibility for a five-circle recurrent readout.
//!
//! One observed presynaptic cue is retained as a bounded eligibility trace.
//! The teacher arrives ONLY after recurrent ticks; a single local inhibitory
//! gain can then be updated. The four context gate masks and ten anatomical
//! source indices never change. This is a synthetic, supervised mechanism,
//! not a biological synapse model or autonomous long-term memory.
use crate::fourier5::{Circle, FiveCircle, CIRCLES};
use crate::fourier5_dynamic::{GatedFiveCircle, ModulationError, RadiusGate, MAX_BOOST};
use crate::fourier5_recurrent::{RecurrentError, RecurrentFiveCircle};

pub const MAX_ELIGIBILITY_TICKS: usize = 512;
pub const MAX_ELIGIBILITY_EVENTS: usize = 4_096;
pub const CONTEXT_ASSOCIATIONS: usize = CIRCLES - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EligibilityError {
    InvalidBudget,
    InvalidCue,
    InvalidModel,
    PendingCue,
    NoCue,
    TooEarly,
    Expired,
    TopologyChanged,
    EventBudget,
    Recurrent(RecurrentError),
    Model(ModulationError),
}

impl From<RecurrentError> for EligibilityError {
    fn from(value: RecurrentError) -> Self {
        Self::Recurrent(value)
    }
}
impl From<ModulationError> for EligibilityError {
    fn from(value: ModulationError) -> Self {
        Self::Model(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Eligibility {
    neuron: usize,
    input: u8,
    context: u8,
    circle: usize,
    recorded_tick: usize,
    wiring_fingerprint: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Feedback {
    pub prediction_before: bool,
    pub expected: bool,
    pub gain_before: u32,
    pub gain_after: u32,
    pub changed: bool,
    pub delay_ticks: usize,
}

/// One pending cue at a time. A bounded eligibility trace stores no label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelayedEligibility {
    max_delay: usize,
    pending: Option<Eligibility>,
    delivered: usize,
    updates: usize,
}

fn valid_readout(model: &GatedFiveCircle) -> bool {
    let circles = model.base().circles();
    circles[0]
        == (Circle {
            mask: 31,
            radius: 2,
            phase_pi: false,
        })
        && model.base().bias() == -1
        && model.gates()[0].context_mask == 0
        && (1..CIRCLES).all(|i| {
            circles[i] == Circle::default() && model.gates()[i].context_mask == 1_u8 << (i - 1)
        })
}

/// Fixed, explicit five-circle readout used only for synthetic qualification.
/// The first mode computes parity; the four remaining modes inhibit it
/// when their matching contextual input is present.
#[must_use]
pub fn associative_readout() -> GatedFiveCircle {
    let mut circles = [Circle::default(); CIRCLES];
    circles[0] = Circle {
        mask: 31,
        radius: 2,
        phase_pi: false,
    };
    let base = FiveCircle::new(circles, -1).expect("valid fixed Walsh masks");
    let gates = std::array::from_fn(|i| RadiusGate {
        context_mask: if i == 0 { 0 } else { 1_u8 << (i - 1) },
        boost: 0,
    });
    GatedFiveCircle::new(base, gates).expect("valid fixed contextual masks")
}

impl DelayedEligibility {
    pub fn new(max_delay: usize) -> Result<Self, EligibilityError> {
        if !(1..=MAX_ELIGIBILITY_TICKS).contains(&max_delay) {
            return Err(EligibilityError::InvalidBudget);
        }
        Ok(Self {
            max_delay,
            pending: None,
            delivered: 0,
            updates: 0,
        })
    }

    #[must_use]
    pub fn pending(&self) -> bool {
        self.pending.is_some()
    }

    #[must_use]
    pub fn delivered(&self) -> usize {
        self.delivered
    }

    #[must_use]
    pub fn updates(&self) -> usize {
        self.updates
    }

    /// Record a *real sample* of the ten existing presynaptic source bits.
    /// The selected neuron must receive odd parity data and exactly one
    /// of four contextual cue bits. No teacher or target is passed here.
    pub fn record(
        &mut self,
        network: &RecurrentFiveCircle,
        neuron: usize,
    ) -> Result<(), EligibilityError> {
        if self.pending.is_some() {
            return Err(EligibilityError::PendingCue);
        }
        if self.delivered >= MAX_ELIGIBILITY_EVENTS {
            return Err(EligibilityError::EventBudget);
        }
        let (input, context) = network.sample_neuron_sources(neuron)?;
        if input.count_ones() % 2 == 0 || context.count_ones() != 1 || context & 0b1_0000 != 0 {
            return Err(EligibilityError::InvalidCue);
        }
        let model = network
            .neuron_model(neuron)
            .ok_or(EligibilityError::InvalidCue)?;
        if !valid_readout(&model) {
            return Err(EligibilityError::InvalidModel);
        }
        self.pending = Some(Eligibility {
            neuron,
            input,
            context,
            circle: context.trailing_zeros() as usize + 1,
            recorded_tick: network.tick_count(),
            wiring_fingerprint: network.wiring_fingerprint(),
        });
        Ok(())
    }

    /// Delayed teacher outcome. The original cue is *not* replayed or
    /// re-presented at feedback time. Only a prediction error updates one
    /// existing local gate by +/-1, saturating at 0 and MAX_BOOST.
    pub fn feedback(
        &mut self,
        network: &mut RecurrentFiveCircle,
        expected: bool,
    ) -> Result<Feedback, EligibilityError> {
        let trace = self.pending.ok_or(EligibilityError::NoCue)?;
        if network.wiring_fingerprint() != trace.wiring_fingerprint {
            return Err(EligibilityError::TopologyChanged);
        }
        let elapsed = network
            .tick_count()
            .checked_sub(trace.recorded_tick)
            .ok_or(EligibilityError::TopologyChanged)?;
        if elapsed == 0 {
            return Err(EligibilityError::TooEarly);
        }
        if elapsed > self.max_delay {
            self.pending = None;
            return Err(EligibilityError::Expired);
        }
        let current = network
            .neuron_model(trace.neuron)
            .ok_or(EligibilityError::InvalidCue)?;
        if !valid_readout(&current) {
            return Err(EligibilityError::InvalidModel);
        }
        let before = current.predict(trace.input, trace.context)?;
        let mut gates = *current.gates();
        let gain_before = gates[trace.circle].boost;
        if before != expected {
            gates[trace.circle].boost = if expected {
                gain_before.saturating_sub(1)
            } else {
                gain_before.saturating_add(1).min(MAX_BOOST)
            };
        }
        let gain_after = gates[trace.circle].boost;
        let changed = gain_before != gain_after;
        if changed {
            let next = GatedFiveCircle::new(*current.base(), gates)?;
            network.update_neuron_model(trace.neuron, next)?;
            self.updates += 1;
        }
        self.delivered += 1;
        self.pending = None;
        Ok(Feedback {
            prediction_before: before,
            expected,
            gain_before,
            gain_after,
            changed,
            delay_ticks: elapsed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5_recurrent::{NetworkConfig, SyntheticTopology};

    const READOUT: usize = 19;

    fn network(seed: u64) -> RecurrentFiveCircle {
        RecurrentFiveCircle::seeded(
            NetworkConfig {
                nodes: 128,
                seed,
                topology: SyntheticTopology::LocalRing,
                initial_activity_per_256: 128,
            },
            associative_readout(),
        )
        .unwrap()
    }

    fn teach(
        net: &mut RecurrentFiveCircle,
        plastic: &mut DelayedEligibility,
        cue: usize,
        expected: bool,
        delay: usize,
    ) -> Feedback {
        net.drive_neuron_sources(READOUT, 0b0_0001, 1 << cue)
            .unwrap();
        plastic.record(net, READOUT).unwrap();
        for _ in 0..delay {
            net.step().unwrap();
        }
        plastic.feedback(net, expected).unwrap()
    }

    fn query(net: &RecurrentFiveCircle, cue: usize) -> bool {
        let mut probe = net.clone();
        probe
            .drive_neuron_sources(READOUT, 0b0_0111, (1 << cue) | 0b1_0000)
            .unwrap();
        probe.step().unwrap();
        probe.state().get(READOUT).unwrap()
    }

    #[test]
    fn delayed_cue_reward_and_512_tick_retention() {
        let mut net = network(42);
        let mut learner = DelayedEligibility::new(128).unwrap();
        let targets = [false, true, false, true];
        for _ in 0..4 {
            for (cue, &expected) in targets.iter().enumerate() {
                let result = teach(&mut net, &mut learner, cue, expected, 64);
                assert_eq!(result.delay_ticks, 64);
            }
        }
        assert_eq!(learner.delivered(), 16);
        assert_eq!(learner.updates(), 6);
        let fingerprint = net.wiring_fingerprint();
        let model = net.neuron_model(READOUT).unwrap();
        for _ in 0..512 {
            net.step().unwrap();
        }
        assert_eq!(net.neuron_model(READOUT), Some(model));
        assert_eq!(net.wiring_fingerprint(), fingerprint);
        for (cue, &expected) in targets.iter().enumerate() {
            assert_eq!(query(&net, cue), expected);
        }
    }

    #[test]
    fn equal_capacity_frozen_and_wrong_feedback_controls() {
        let mut trained = network(17);
        let frozen = trained.clone();
        let mut shuffled = trained.clone();
        let mut a = DelayedEligibility::new(128).unwrap();
        let mut b = DelayedEligibility::new(128).unwrap();
        let targets = [false, true, false, true];
        for _ in 0..4 {
            for (cue, &expected) in targets.iter().enumerate() {
                teach(&mut trained, &mut a, cue, expected, 8);
                teach(&mut shuffled, &mut b, cue, !expected, 8);
            }
        }
        let score = |net: &RecurrentFiveCircle| {
            targets
                .iter()
                .enumerate()
                .filter(|(cue, expected)| query(net, *cue) == **expected)
                .count()
        };
        assert_eq!(score(&trained), 4);
        assert_eq!(score(&frozen), 2);
        assert_eq!(score(&shuffled), 0);
        assert_eq!(trained.wiring_fingerprint(), frozen.wiring_fingerprint());
        assert_eq!(trained.wiring_fingerprint(), shuffled.wiring_fingerprint());
    }

    #[test]
    fn early_expired_and_missing_feedback_never_modify_weights() {
        let mut net = network(99);
        let mut plastic = DelayedEligibility::new(8).unwrap();
        let before = net.neuron_model(READOUT);
        assert_eq!(
            plastic.feedback(&mut net, false),
            Err(EligibilityError::NoCue)
        );
        net.drive_neuron_sources(READOUT, 1, 1).unwrap();
        plastic.record(&net, READOUT).unwrap();
        assert_eq!(
            plastic.record(&net, READOUT),
            Err(EligibilityError::PendingCue)
        );
        assert_eq!(
            plastic.feedback(&mut net, false),
            Err(EligibilityError::TooEarly)
        );
        for _ in 0..9 {
            net.step().unwrap();
        }
        assert_eq!(
            plastic.feedback(&mut net, false),
            Err(EligibilityError::Expired)
        );
        assert_eq!(net.neuron_model(READOUT), before);
        assert!(!plastic.pending());
        assert_eq!(plastic.delivered(), 0);
        assert_eq!(plastic.updates(), 0);
    }

    #[test]
    fn invalid_cues_and_budgets_fail_closed() {
        assert_eq!(
            DelayedEligibility::new(0),
            Err(EligibilityError::InvalidBudget)
        );
        assert_eq!(
            DelayedEligibility::new(MAX_ELIGIBILITY_TICKS + 1),
            Err(EligibilityError::InvalidBudget)
        );
        let mut net = network(1);
        let mut plastic = DelayedEligibility::new(16).unwrap();
        let model = net.neuron_model(READOUT);
        let state = net.state().clone();
        net.drive_neuron_sources(READOUT, 0, 1).unwrap();
        assert_eq!(
            plastic.record(&net, READOUT),
            Err(EligibilityError::InvalidCue)
        );
        net.drive_neuron_sources(READOUT, 1, 3).unwrap();
        assert_eq!(
            plastic.record(&net, READOUT),
            Err(EligibilityError::InvalidCue)
        );
        net.drive_neuron_sources(READOUT, 1, 16).unwrap();
        assert_eq!(
            plastic.record(&net, READOUT),
            Err(EligibilityError::InvalidCue)
        );
        assert_eq!(net.neuron_model(READOUT), model);
        assert_eq!(plastic.updates(), 0);
        assert_eq!(
            net.drive_neuron_sources(READOUT, 32, 1),
            Err(RecurrentError::InvalidContact)
        );
        assert_eq!(
            net.drive_neuron_sources(128, 1, 1),
            Err(RecurrentError::InvalidContact)
        );
        assert_ne!(net.state(), &state);
    }

    #[test]
    fn exact_feedback_changes_only_targeted_gain() {
        let mut net = network(7);
        let mut learner = DelayedEligibility::new(16).unwrap();
        let baseline = net.neuron_model(READOUT).unwrap();
        let before_other = net.neuron_model(READOUT + 1);
        let feedback = teach(&mut net, &mut learner, 2, false, 1);
        assert!(feedback.changed);
        let model = net.neuron_model(READOUT).unwrap();
        assert_eq!(model.gates()[3].boost, 1);
        for i in [0, 1, 2, 4] {
            assert_eq!(model.gates()[i], baseline.gates()[i]);
        }
        assert_eq!(net.neuron_model(READOUT + 1), before_other);
    }
}

//! Bounded, deterministic learning of the five *context* gate masks.
//!
//! Unlike the independent Fourier-Walsh mode masks in fourier5_mask_learning,
//! these masks decide which five-bit contextual conjunction temporarily
//! increases an existing circle's radius. The base neuron is immutable.
//! Synthetic supervised examples are not anatomical evidence.
use crate::fourier5::{FiveCircle, CIRCLES, MAX_EPOCHS};
use crate::fourier5_dynamic::{
    train_context_gates, GatedFiveCircle, ModulationError, RadiusGate,
};

pub const MAX_CONTEXT_SAMPLES: usize = 512;
pub const CONTEXT_BITS: u8 = 31;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextSearchError {
    EmptyTraining,
    TooManySamples,
    DuplicatePair,
    InvalidBits,
    InvalidEpochBudget,
    Model(ModulationError),
}

impl From<ModulationError> for ContextSearchError {
    fn from(error: ModulationError) -> Self {
        Self::Model(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextSample {
    pub input: u8,
    pub context: u8,
    pub fire: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextMaskReport {
    pub model: GatedFiveCircle,
    pub selected: usize,
    pub train_correct: usize,
    pub train_hinge: i64,
    pub final_updates: usize,
    /// Number of candidate refits, including baseline.
    pub candidates_evaluated: usize,
    /// Total supervised examples actually visited by all candidate refits.
    pub sample_visits: u64,
}

fn validate(samples: &[ContextSample], epochs: usize) -> Result<(), ContextSearchError> {
    if samples.is_empty() {
        return Err(ContextSearchError::EmptyTraining);
    }
    if samples.len() > MAX_CONTEXT_SAMPLES {
        return Err(ContextSearchError::TooManySamples);
    }
    if epochs == 0 || epochs > MAX_EPOCHS {
        return Err(ContextSearchError::InvalidEpochBudget);
    }
    let mut seen = [0_u64; 16]; // 32 data states x 32 context states.
    for sample in samples {
        if sample.input & !CONTEXT_BITS != 0 || sample.context & !CONTEXT_BITS != 0 {
            return Err(ContextSearchError::InvalidBits);
        }
        let slot = usize::from(sample.input) * 32 + usize::from(sample.context);
        let (word, bit) = (slot / 64, slot % 64);
        let flag = 1_u64 << bit;
        if seen[word] & flag != 0 {
            return Err(ContextSearchError::DuplicatePair);
        }
        seen[word] |= flag;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Fit {
    model: GatedFiveCircle,
    mistakes: usize,
    hinge: i64,
    updates: usize,
    visits: u64,
}

fn fit(
    base: FiveCircle,
    masks: [u8; CIRCLES],
    samples: &[(u8, u8, bool)],
    epochs: usize,
) -> Result<Fit, ContextSearchError> {
    let gates = std::array::from_fn(|i| RadiusGate {
        context_mask: masks[i],
        boost: 0,
    });
    let initial = GatedFiveCircle::new(base, gates)?;
    let trained = train_context_gates(initial, samples, epochs)?;
    let mut errors = 0;
    let mut hinge = 0_i64;
    for &(input, context, fire) in samples {
        let score = trained.model.score(input, context)?;
        if (score < 0) != fire {
            errors += 1;
        }
        let label = if fire { -1 } else { 1 };
        hinge += (1_i64 - label * score).max(0);
    }
    Ok(Fit {
        model: trained.model,
        mistakes: errors,
        hinge,
        updates: trained.updates,
        visits: (trained.epochs_used as u64) * (samples.len() as u64),
    })
}

/// Train-only greedy selection of up to five contextual conjunctions.
///
/// A candidate tries one gate mask (1..31) for the current circle and refits
/// *all* selected gains from zero. Improvement is decided by TRAIN errors,
/// then hinge loss. Ties favor the smaller mask. Stops on no improvement.
/// Does not read validation or test examples and does not claim global optimum.
pub fn learn_context_masks(
    base: FiveCircle,
    samples: &[ContextSample],
    epochs: usize,
) -> Result<ContextMaskReport, ContextSearchError> {
    validate(samples, epochs)?;
    let tuples: Vec<_> = samples
        .iter()
        .map(|s| (s.input, s.context, s.fire))
        .collect();
    let mut masks = [0_u8; CIRCLES];
    let mut best = fit(base, masks, &tuples, epochs)?;
    let mut candidates_evaluated = 1;
    let mut sample_visits = best.visits;

    let mut selected = 0;
    for slot in 0..CIRCLES {
        if best.mistakes == 0 {
            break;
        }
        let mut best_trial: Option<(u8, Fit)> = None;
        for mask in 1..=CONTEXT_BITS {
            masks[slot] = mask;
            let trial = fit(base, masks, &tuples, epochs)?;
            candidates_evaluated += 1;
            sample_visits += trial.visits;
            if best_trial.as_ref().is_none_or(|(prior_mask, prior)| {
                (trial.mistakes, trial.hinge, mask) < (prior.mistakes, prior.hinge, *prior_mask)
            }) {
                best_trial = Some((mask, trial));
            }
        }
        masks[slot] = 0;
        let Some((mask, improved)) = best_trial else {
            break;
        };
        if (improved.mistakes, improved.hinge) >= (best.mistakes, best.hinge) {
            break;
        }
        masks[slot] = mask;
        best = improved;
        selected += 1;
    }
    Ok(ContextMaskReport {
        model: best.model,
        selected,
        train_correct: samples.len() - best.mistakes,
        train_hinge: best.hinge,
        final_updates: best.updates,
        candidates_evaluated,
        sample_visits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5::Circle;

    fn base() -> FiveCircle {
        let mut circles = [Circle::default(); CIRCLES];
        circles[0] = Circle {
            mask: 1,
            radius: 1,
            phase_pi: false,
        };
        FiveCircle::new(circles, 2).unwrap()
    }

    fn examples(start: u8, end: u8) -> Vec<ContextSample> {
        (start..end)
            .flat_map(|context| {
                [0_u8, 1_u8].map(move |input| ContextSample {
                    input,
                    context,
                    fire: input == 1 && context & 0b01010 == 0b01010,
                })
            })
            .collect()
    }

    #[test]
    fn learns_a_two_bit_context_conjunction_without_given_mask() {
        let train = examples(0, 16);
        let test = examples(16, 32);
        let report = learn_context_masks(base(), &train, 32).unwrap();
        assert_eq!(report.selected, 1);
        assert_eq!(report.model.gates()[0].context_mask, 0b01010);
        assert_eq!(report.model.gates()[0].boost, 2);
        assert_eq!(report.train_correct, train.len());
        assert_eq!(
            report
                .model
                .correct_count(&test.iter().map(|s| (s.input, s.context, s.fire)).collect::<Vec<_>>()),
            Ok(test.len())
        );
        assert_eq!(report.model.base(), &base());
        assert!(report.sample_visits >= train.len() as u64);
    }

    #[test]
    fn always_negative_target_needs_no_context_gate() {
        let train: Vec<_> = (0..16)
            .map(|context| ContextSample {
                input: 0,
                context,
                fire: false,
            })
            .collect();
        let report = learn_context_masks(base(), &train, 8).unwrap();
        assert_eq!(report.selected, 0);
        assert_eq!(report.candidates_evaluated, 1);
        assert_eq!(report.train_correct, train.len());
    }

    #[test]
    fn train_split_excludes_heldout_context_high_bit() {
        let train = examples(0, 16);
        let test = examples(16, 32);
        assert_eq!(train.len(), 32);
        assert_eq!(test.len(), 32);
        assert!(train.iter().all(|s| s.context & 16 == 0));
        assert!(test.iter().all(|s| s.context & 16 != 0));
        assert_eq!(train.iter().filter(|s| s.fire).count(), 4);
        assert_eq!(test.iter().filter(|s| s.fire).count(), 4);
    }

    #[test]
    fn training_is_reproducible_and_independent_of_test_labels() {
        let samples = examples(0, 16);
        let fit1 = learn_context_masks(base(), &samples, 32).unwrap();
        let fit2 = learn_context_masks(base(), &samples, 32).unwrap();
        assert_eq!(fit1, fit2);
        let flipped_test: Vec<_> = examples(16, 32)
            .into_iter()
            .map(|s| ContextSample { fire: !s.fire, ..s })
            .collect();
        assert_eq!(
            fit1.model.correct_count(
                &flipped_test
                    .iter()
                    .map(|s| (s.input, s.context, s.fire))
                    .collect::<Vec<_>>()
            ),
            Ok(0)
        );
    }

    #[test]
    fn rejects_duplicates_invalid_bits_and_resource_limits() {
        let one = ContextSample {
            input: 1,
            context: 2,
            fire: false,
        };
        assert_eq!(
            learn_context_masks(base(), &[], 1),
            Err(ContextSearchError::EmptyTraining)
        );
        assert_eq!(
            learn_context_masks(base(), &[one, one], 1),
            Err(ContextSearchError::DuplicatePair)
        );
        assert_eq!(
            learn_context_masks(base(), &[ContextSample { context: 32, ..one }], 1),
            Err(ContextSearchError::InvalidBits)
        );
        assert_eq!(
            learn_context_masks(base(), &[ContextSample { input: 32, ..one }], 1),
            Err(ContextSearchError::InvalidBits)
        );
        assert_eq!(
            learn_context_masks(base(), &[one], 0),
            Err(ContextSearchError::InvalidEpochBudget)
        );
        assert_eq!(
            learn_context_masks(base(), &[one], MAX_EPOCHS + 1),
            Err(ContextSearchError::InvalidEpochBudget)
        );
        assert_eq!(
            learn_context_masks(base(), &vec![one; MAX_CONTEXT_SAMPLES + 1], 1),
            Err(ContextSearchError::TooManySamples)
        );
    }
}

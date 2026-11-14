//! Data-driven selection of up to five discrete Boolean interaction masks.
//!
//! Two matched hypothesis families (Walsh characters or Boolean conjunctions)
//! share the same greedy subset-search procedure, data and update budget.
//! This is not an anatomical or physiological model of BANC V888.
use crate::fourier5::{Circle, FiveCircle, CIRCLES, TABLE_SIZE};

pub const MAX_MASK_EPOCHS: usize = 128;
const VALID_BITS: u8 = (TABLE_SIZE - 1) as u8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskSearchError {
    EmptyTraining,
    TooManyExamples,
    DuplicateInput,
    InvalidInput,
    InvalidEpochBudget,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureFamily {
    Walsh,
    Conjunction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskSample {
    pub input: u8,
    pub fire: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseMaskModel {
    pub family: FeatureFamily,
    pub masks: [u8; CIRCLES],
    pub signed_radii: [i32; CIRCLES],
    pub bias: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskSearchReport {
    pub model: SparseMaskModel,
    pub train_correct: usize,
    pub selected: usize,
    pub perceptron_updates: usize,
}

fn check_sample_set(examples: &[MaskSample]) -> Result<(), MaskSearchError> {
    if examples.is_empty() {
        return Err(MaskSearchError::EmptyTraining);
    }
    if examples.len() > TABLE_SIZE {
        return Err(MaskSearchError::TooManyExamples);
    }
    let mut seen = 0_u32;
    for sample in examples {
        if sample.input & !VALID_BITS != 0 {
            return Err(MaskSearchError::InvalidInput);
        }
        let flag = 1_u32 << sample.input;
        if seen & flag != 0 {
            return Err(MaskSearchError::DuplicateInput);
        }
        seen |= flag;
    }
    Ok(())
}

/// Both families use 31 available nonzero masks; input and output are exact.
fn character(input: u8, mask: u8, family: FeatureFamily) -> i32 {
    match family {
        FeatureFamily::Walsh => {
            if (input & mask).count_ones() & 1 == 0 {
                1
            } else {
                -1
            }
        }
        FeatureFamily::Conjunction => {
            if input & mask == mask {
                -1
            } else {
                1
            }
        }
    }
}

impl SparseMaskModel {
    pub fn score(&self, input: u8) -> Result<i64, MaskSearchError> {
        if input & !VALID_BITS != 0 {
            return Err(MaskSearchError::InvalidInput);
        }
        let mut score = i64::from(self.bias);
        for (mask, weight) in self.masks.iter().zip(self.signed_radii) {
            if *mask != 0 {
                score += i64::from(weight) * i64::from(character(input, *mask, self.family));
            }
        }
        Ok(score)
    }

    pub fn predict(&self, input: u8) -> Result<bool, MaskSearchError> {
        Ok(self.score(input)? < 0)
    }

    pub fn count_correct(&self, samples: &[MaskSample]) -> Result<usize, MaskSearchError> {
        check_sample_set(samples)?;
        let mut hits = 0;
        for sample in samples {
            if self.predict(sample.input)? == sample.fire {
                hits += 1;
            }
        }
        Ok(hits)
    }

    /// A learned Walsh model becomes an actual FiveCircle inference neuron.
    /// A conjunction is deliberately not disguised as a Fourier character.
    #[must_use]
    pub fn as_five_circle(&self) -> Option<FiveCircle> {
        if self.family != FeatureFamily::Walsh {
            return None;
        }
        let circles = std::array::from_fn(|index| Circle {
            mask: self.masks[index],
            radius: self.signed_radii[index].unsigned_abs(),
            phase_pi: self.signed_radii[index] < 0,
        });
        FiveCircle::new(circles, self.bias).ok()
    }

    #[must_use]
    pub fn active_modes(&self) -> usize {
        self.masks
            .iter()
            .zip(self.signed_radii)
            .filter(|(mask, weight)| **mask != 0 && *weight != 0)
            .count()
    }
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    model: SparseMaskModel,
    errors: usize,
    hinge: i64,
    updates: usize,
}

/// Refit all selected signed radii and bias from zero on TRAIN only.
fn refit(samples: &[MaskSample], family: FeatureFamily, chosen: &[u8], epochs: usize) -> Candidate {
    let mut model = SparseMaskModel {
        family,
        masks: [0; CIRCLES],
        signed_radii: [0; CIRCLES],
        bias: 0,
    };
    for (slot, &mask) in chosen.iter().enumerate() {
        model.masks[slot] = mask;
    }
    let mut updates = 0;
    for _ in 0..epochs {
        let mut mistakes = 0;
        for &MaskSample { input, fire } in samples {
            let target = if fire { -1_i64 } else { 1_i64 };
            let score = model.score(input).expect("validated TRAIN input");
            if target * score <= 0 {
                model.bias += target as i32;
                for (slot, &mask) in chosen.iter().enumerate() {
                    model.signed_radii[slot] += target as i32 * character(input, mask, family);
                }
                mistakes += 1;
                updates += 1;
            }
        }
        if mistakes == 0 {
            break;
        }
    }
    let mut errors = 0;
    let mut hinge = 0_i64;
    for &MaskSample { input, fire } in samples {
        let response = model.score(input).expect("validated TRAIN input");
        if (response < 0) != fire {
            errors += 1;
        }
        let target = if fire { -1_i64 } else { 1_i64 };
        hinge += (1 - target * response).max(0);
    }
    Candidate {
        model,
        errors,
        hinge,
        updates,
    }
}

/// Learns both masks and radii, with the same search budget for both families.
///
/// Greedy mode selection minimizes (training mistakes, hinge loss, mask id).
/// No held-out labels are read, and no global optimality is promised.
pub fn learn_five_masks(
    samples: &[MaskSample],
    family: FeatureFamily,
    epochs: usize,
) -> Result<MaskSearchReport, MaskSearchError> {
    check_sample_set(samples)?;
    if epochs == 0 || epochs > MAX_MASK_EPOCHS {
        return Err(MaskSearchError::InvalidEpochBudget);
    }
    let mut selected = Vec::with_capacity(CIRCLES);
    let mut fitted = refit(samples, family, &[], epochs);
    for _ in 0..CIRCLES {
        if fitted.errors == 0 {
            break;
        }
        let mut best: Option<(u8, Candidate)> = None;
        for mask in 1..=VALID_BITS {
            if selected.contains(&mask) {
                continue;
            }
            selected.push(mask);
            let trial = refit(samples, family, &selected, epochs);
            selected.pop();
            let replace = best.as_ref().is_none_or(|(best_mask, previous)| {
                (trial.errors, trial.hinge, mask) < (previous.errors, previous.hinge, *best_mask)
            });
            if replace {
                best = Some((mask, trial));
            }
        }
        let Some((mask, trial)) = best else {
            break;
        };
        if (trial.errors, trial.hinge) >= (fitted.errors, fitted.hinge) {
            break;
        }
        selected.push(mask);
        fitted = trial;
    }
    Ok(MaskSearchReport {
        model: fitted.model,
        train_correct: samples.len() - fitted.errors,
        selected: selected.len(),
        perceptron_updates: fitted.updates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn train_on_lower_half(target: impl Fn(u8) -> bool) -> Vec<MaskSample> {
        (0..16)
            .map(|input| MaskSample {
                input,
                fire: target(input),
            })
            .collect()
    }

    #[test]
    fn learns_four_bit_parity_mask_not_provided_by_oracle() {
        let train = train_on_lower_half(|x| (x & 15).count_ones() % 2 != 0);
        let fit = learn_five_masks(&train, FeatureFamily::Walsh, 64).unwrap();
        assert_eq!(fit.train_correct, 16);
        assert_eq!(fit.model.masks[0], 15);
        assert_eq!(fit.selected, 1);
        let neuron = fit.model.as_five_circle().unwrap();
        for x in 0_u8..32 {
            let expected = (x & 15).count_ones() % 2 != 0;
            assert_eq!(fit.model.predict(x), Ok(expected));
            assert_eq!(neuron.predict(x), Ok(expected));
        }
    }

    #[test]
    fn conjunction_baseline_learns_four_way_and() {
        let train = train_on_lower_half(|x| (x & 15) == 15);
        let fit = learn_five_masks(&train, FeatureFamily::Conjunction, 64).unwrap();
        assert_eq!(fit.train_correct, 16);
        assert_eq!(fit.model.masks[0], 15);
        assert_eq!(fit.selected, 1);
        assert!(fit.model.as_five_circle().is_none());
        for x in 0..32 {
            assert_eq!(fit.model.predict(x), Ok((x & 15) == 15));
        }
    }

    #[test]
    fn all_false_labels_require_no_circles() {
        let train = train_on_lower_half(|_| false);
        for family in [FeatureFamily::Walsh, FeatureFamily::Conjunction] {
            let fit = learn_five_masks(&train, family, 64).unwrap();
            assert_eq!(fit.selected, 0);
            assert_eq!(fit.model.active_modes(), 0);
            assert_eq!(fit.train_correct, 16);
            assert_eq!(fit.model.predict(31), Ok(false));
        }
    }

    #[test]
    fn rejects_duplicate_samples_invalid_bits_and_unbounded_epochs() {
        let one = [MaskSample {
            input: 0,
            fire: true,
        }];
        assert_eq!(
            learn_five_masks(&[], FeatureFamily::Walsh, 2),
            Err(MaskSearchError::EmptyTraining)
        );
        assert_eq!(
            learn_five_masks(&[one[0], one[0]], FeatureFamily::Walsh, 2),
            Err(MaskSearchError::DuplicateInput)
        );
        assert_eq!(
            learn_five_masks(
                &[MaskSample {
                    input: 32,
                    fire: true,
                }],
                FeatureFamily::Walsh,
                2,
            ),
            Err(MaskSearchError::InvalidInput)
        );
        assert_eq!(
            learn_five_masks(&one, FeatureFamily::Walsh, 0),
            Err(MaskSearchError::InvalidEpochBudget)
        );
        assert_eq!(
            learn_five_masks(&one, FeatureFamily::Walsh, MAX_MASK_EPOCHS + 1),
            Err(MaskSearchError::InvalidEpochBudget)
        );
        let model = learn_five_masks(&one, FeatureFamily::Walsh, 2)
            .unwrap()
            .model;
        assert_eq!(model.predict(32), Err(MaskSearchError::InvalidInput));
    }

    #[test]
    fn no_test_labels_are_needed_and_model_is_repeatable() {
        let train = train_on_lower_half(|x| (x & 7).count_ones() % 2 != 0);
        let first = learn_five_masks(&train, FeatureFamily::Walsh, 64).unwrap();
        let second = learn_five_masks(&train, FeatureFamily::Walsh, 64).unwrap();
        assert_eq!(first, second);
        let heldout: Vec<_> = (16..32)
            .map(|x| MaskSample {
                input: x,
                fire: (x & 7).count_ones() % 2 != 0,
            })
            .collect();
        assert_eq!(first.model.count_correct(&heldout), Ok(16));
        let changed_labels: Vec<_> = heldout
            .iter()
            .map(|sample| MaskSample {
                input: sample.input,
                fire: !sample.fire,
            })
            .collect();
        assert_eq!(first.model.count_correct(&changed_labels), Ok(0));
    }

    #[test]
    fn family_search_supports_five_modes_and_is_finite() {
        let train = train_on_lower_half(|x| (x & 15).count_ones() >= 3);
        for family in [FeatureFamily::Walsh, FeatureFamily::Conjunction] {
            let fit = learn_five_masks(&train, family, 64).unwrap();
            assert!(fit.selected <= CIRCLES);
            assert!(fit.train_correct <= train.len());
            assert!(fit
                .model
                .masks
                .iter()
                .all(|&mask| usize::from(mask) < TABLE_SIZE));
        }
        assert_eq!(crate::fourier5::INPUT_BITS, 5);
    }
}

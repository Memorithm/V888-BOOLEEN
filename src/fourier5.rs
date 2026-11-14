//! Experimental five-circle Boolean Fourier neuron.
//!
//! A circle is a Walsh character with a nonnegative radius and a binary
//! phase (0 or pi). Inference uses exact integer operations, not trigonometry.
//! This experimental rule makes no claim about the physiology of BANC V888.

pub const INPUT_BITS: usize = 5;
pub const TABLE_SIZE: usize = 1 << INPUT_BITS;
pub const CIRCLES: usize = 5;
pub const MAX_EPOCHS: usize = 1_024;
const INPUT_MASK: u8 = (TABLE_SIZE - 1) as u8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FourierError {
    InvalidInput,
    InvalidMask,
    ExcessiveEpochBudget,
}

/// Fourier-Walsh mode: radius * (-1)^(phase_pi XOR parity(input & mask)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Circle {
    pub mask: u8,
    pub radius: u32,
    pub phase_pi: bool,
}

fn sign(input: u8, mask: u8) -> i64 {
    if (input & mask).count_ones() % 2 == 0 {
        1
    } else {
        -1
    }
}

fn valid_input(input: u8) -> Result<(), FourierError> {
    if input & !INPUT_MASK != 0 {
        Err(FourierError::InvalidInput)
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FiveCircle {
    circles: [Circle; CIRCLES],
    bias: i32,
}

impl FiveCircle {
    pub fn new(circles: [Circle; CIRCLES], bias: i32) -> Result<Self, FourierError> {
        if circles.iter().any(|circle| circle.mask & !INPUT_MASK != 0) {
            return Err(FourierError::InvalidMask);
        }
        Ok(Self { circles, bias })
    }

    #[must_use]
    pub fn circles(&self) -> &[Circle; CIRCLES] {
        &self.circles
    }

    #[must_use]
    pub fn bias(&self) -> i32 {
        self.bias
    }

    pub fn score(&self, input: u8) -> Result<i64, FourierError> {
        valid_input(input)?;
        Ok(i64::from(self.bias)
            + self
                .circles
                .iter()
                .map(|circle| {
                    let contribution = i64::from(circle.radius) * sign(input, circle.mask);
                    if circle.phase_pi {
                        -contribution
                    } else {
                        contribution
                    }
                })
                .sum::<i64>())
    }

    pub fn predict(&self, input: u8) -> Result<bool, FourierError> {
        Ok(self.score(input)? < 0)
    }

    /// Number of correctly classified states out of all 32 possibilities.
    pub fn correct_count(&self, table: &[bool; TABLE_SIZE]) -> usize {
        (0..TABLE_SIZE)
            .filter(|&x| self.predict(x as u8) == Ok(table[x]))
            .count()
    }
}

/// The 32 Walsh coefficients, scaled by 32 to remain exact integers.
#[must_use]
pub fn walsh_transform(table: &[bool; TABLE_SIZE]) -> [i32; TABLE_SIZE] {
    std::array::from_fn(|mask| {
        (0..TABLE_SIZE)
            .map(|x| {
                let label = if table[x] { -1 } else { 1 };
                label * sign(x as u8, mask as u8) as i32
            })
            .sum()
    })
}

/// Reference reconstruction using ALL 32 modes, not a five-circle model.
/// Its score is exactly +32 for false and -32 for true.
pub fn full_walsh_score(coefficients: &[i32; TABLE_SIZE], input: u8) -> Result<i64, FourierError> {
    valid_input(input)?;
    Ok(coefficients
        .iter()
        .enumerate()
        .map(|(mask, weight)| i64::from(*weight) * sign(input, mask as u8))
        .sum())
}

/// Select five strongest nonconstant modes, keeping the DC term as bias.
/// Not an optimal classifier: spectral energy and classification differ.
#[must_use]
pub fn strongest_five(coefficients: &[i32; TABLE_SIZE]) -> FiveCircle {
    let mut order: [u8; TABLE_SIZE - 1] = std::array::from_fn(|i| (i + 1) as u8);
    order.sort_unstable_by(|a, b| {
        coefficients[usize::from(*b)]
            .unsigned_abs()
            .cmp(&coefficients[usize::from(*a)].unsigned_abs())
            .then_with(|| a.cmp(b))
    });
    let mut circles = [Circle::default(); CIRCLES];
    for (circle, mask) in circles.iter_mut().zip(order) {
        let weight = coefficients[usize::from(mask)];
        *circle = Circle {
            mask,
            radius: weight.unsigned_abs(),
            phase_pi: weight < 0,
        };
    }
    FiveCircle {
        circles,
        bias: coefficients[0],
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingReport {
    pub model: FiveCircle,
    pub converged: bool,
    pub epochs_used: usize,
    pub updates: usize,
}

/// Deterministic perceptron updates for fixed coupling masks.
/// No convergence guarantee for nonseparable mask/function pairs.
pub fn train_fixed_masks(
    masks: [u8; CIRCLES],
    table: &[bool; TABLE_SIZE],
    epochs: usize,
) -> Result<TrainingReport, FourierError> {
    if masks.iter().any(|mask| mask & !INPUT_MASK != 0) {
        return Err(FourierError::InvalidMask);
    }
    if epochs > MAX_EPOCHS {
        return Err(FourierError::ExcessiveEpochBudget);
    }
    let mut weights = [0_i32; CIRCLES];
    let mut bias = 0_i32;
    let mut updates = 0;
    let mut epochs_used = 0;
    let mut converged = false;
    for epoch in 0..epochs {
        let mut mistakes = 0;
        for x in 0..TABLE_SIZE {
            let x = x as u8;
            let target = if table[usize::from(x)] { -1 } else { 1 };
            let score = i64::from(bias)
                + weights
                    .iter()
                    .zip(masks)
                    .map(|(&weight, mask)| i64::from(weight) * sign(x, mask))
                    .sum::<i64>();
            if i64::from(target) * score <= 0 {
                bias += target;
                for (weight, mask) in weights.iter_mut().zip(masks) {
                    *weight += target * sign(x, mask) as i32;
                }
                mistakes += 1;
                updates += 1;
            }
        }
        epochs_used = epoch + 1;
        if mistakes == 0 {
            converged = true;
            break;
        }
    }
    let circles = std::array::from_fn(|i| Circle {
        mask: masks[i],
        radius: weights[i].unsigned_abs(),
        phase_pi: weights[i] < 0,
    });
    Ok(TrainingReport {
        model: FiveCircle { circles, bias },
        converged,
        epochs_used,
        updates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn singletons() -> [Circle; CIRCLES] {
        std::array::from_fn(|i| Circle {
            mask: 1 << i,
            radius: 1,
            phase_pi: false,
        })
    }

    #[test]
    fn parity_five_is_exact_with_a_coupled_circle() {
        let mut circles = [Circle::default(); CIRCLES];
        circles[0] = Circle {
            mask: 31,
            radius: 7,
            phase_pi: false,
        };
        let neuron = FiveCircle::new(circles, 0).unwrap();
        for input in 0..TABLE_SIZE {
            assert_eq!(neuron.predict(input as u8), Ok(input.count_ones() % 2 != 0));
        }
    }

    #[test]
    fn five_circles_represent_or_and_majority() {
        let or = FiveCircle::new(singletons(), -4).unwrap();
        let and = FiveCircle::new(singletons(), 4).unwrap();
        let majority = FiveCircle::new(singletons(), 0).unwrap();
        for x in 0..TABLE_SIZE {
            assert_eq!(or.predict(x as u8), Ok(x != 0));
            assert_eq!(and.predict(x as u8), Ok(x == 31));
            assert_eq!(majority.predict(x as u8), Ok(x.count_ones() >= 3));
        }
    }

    #[test]
    fn perceptron_learns_radii_and_phases() {
        let parity = std::array::from_fn(|x: usize| x.count_ones() % 2 != 0);
        let fit = train_fixed_masks([31, 1, 2, 4, 8], &parity, 128).unwrap();
        assert!(fit.converged);
        assert_eq!(fit.model.correct_count(&parity), TABLE_SIZE);
        assert!(fit.model.circles()[0].radius > 0);

        let or = std::array::from_fn(|x: usize| x != 0);
        let fit = train_fixed_masks([1, 2, 4, 8, 16], &or, 128).unwrap();
        assert!(fit.converged);
        assert_eq!(fit.model.correct_count(&or), TABLE_SIZE);
    }

    #[test]
    fn independent_singletons_cannot_learn_parity_five() {
        let parity = std::array::from_fn(|x: usize| x.count_ones() % 2 != 0);
        let fit = train_fixed_masks([1, 2, 4, 8, 16], &parity, 128).unwrap();
        assert!(!fit.converged);
        assert!(fit.model.correct_count(&parity) < TABLE_SIZE);
    }

    #[test]
    fn full_walsh_oracle_reconstructs_every_test_table() {
        for kind in 0..5 {
            let table = std::array::from_fn(|x: usize| match kind {
                0 => x.count_ones() % 2 != 0,
                1 => x != 0,
                2 => x == 31,
                3 => x.count_ones() >= 3,
                _ => (x as u8 & 0b1_0101) == 0b0_0101,
            });
            let coefficients = walsh_transform(&table);
            for (input, &expected) in table.iter().enumerate() {
                assert_eq!(
                    full_walsh_score(&coefficients, input as u8),
                    Ok(if expected { -32 } else { 32 })
                );
            }
        }
    }

    #[test]
    fn naive_top_five_modes_are_not_universal() {
        let table = std::array::from_fn(|x: usize| x != 0);
        let sparse = strongest_five(&walsh_transform(&table));
        assert_eq!(sparse.correct_count(&table), 31);
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let neuron = FiveCircle::new(singletons(), 0).unwrap();
        assert_eq!(neuron.predict(32), Err(FourierError::InvalidInput));
        let mut bad = singletons();
        bad[0].mask = 32;
        assert_eq!(FiveCircle::new(bad, 0), Err(FourierError::InvalidMask));
        assert_eq!(
            train_fixed_masks([1, 2, 4, 8, 32], &[false; 32], 1),
            Err(FourierError::InvalidMask)
        );
        assert_eq!(
            train_fixed_masks([1, 2, 4, 8, 16], &[false; 32], 1_025),
            Err(FourierError::ExcessiveEpochBudget)
        );
    }
}

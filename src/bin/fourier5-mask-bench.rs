//! Synthetic, preregistered comparison of two five-mask Boolean families.
//!
//! Both models learn discrete interaction masks, signed integer radii and bias
//! from the SAME 16 training labels. No test label is supplied to the trainer.
use v888_booleen::fourier5_mask_learning::{learn_five_masks, FeatureFamily, MaskSample};

type Target = fn(u8) -> bool;

fn parity_four(x: u8) -> bool {
    (x & 15).count_ones() % 2 != 0
}
fn and_four(x: u8) -> bool {
    x & 15 == 15
}
fn majority_four(x: u8) -> bool {
    (x & 15).count_ones() >= 3
}

fn split(start: u8, end: u8, target: Target) -> Vec<MaskSample> {
    (start..end)
        .map(|input| MaskSample {
            input,
            fire: target(input),
        })
        .collect()
}

fn run(task: &str, label: Target, family: FeatureFamily) -> Result<(usize, usize), String> {
    let train = split(0, 16, label);
    let test = split(16, 32, label);
    let fit = learn_five_masks(&train, family, 64).map_err(|err| format!("{err:?}"))?;
    let correct_test = fit
        .model
        .count_correct(&test)
        .map_err(|err| format!("{err:?}"))?;
    let name = match family {
        FeatureFamily::Walsh => "walsh",
        FeatureFamily::Conjunction => "conjunction",
    };
    println!(
        "task={task} family={name} train={}/16 test={correct_test}/16 modes={} masks={:?} weights={:?} bias={} updates={}",
        fit.train_correct,
        fit.model.active_modes(),
        fit.model.masks,
        fit.model.signed_radii,
        fit.model.bias,
        fit.perceptron_updates,
    );
    Ok((fit.train_correct, correct_test))
}

fn main() -> Result<(), String> {
    for (name, target) in [
        ("parity4", parity_four as Target),
        ("and4", and_four as Target),
        ("majority4", majority_four as Target),
    ] {
        let walsh = run(name, target, FeatureFamily::Walsh)?;
        let conjunction = run(name, target, FeatureFamily::Conjunction)?;
        if name == "parity4" && walsh != (16, 16) {
            return Err(format!("parity4 Walsh oracle failed: {walsh:?}"));
        }
        if name == "and4" && conjunction != (16, 16) {
            return Err(format!("and4 conjunction oracle failed: {conjunction:?}"));
        }
    }
    println!("FOURIER5_DISCRETE_MASK_SYNTHETIC_BENCH_PASS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tasks_are_independent_of_fifth_heldout_bit() {
        for target in [
            parity_four as Target,
            and_four as Target,
            majority_four as Target,
        ] {
            for input in 0..16 {
                assert_eq!(target(input), target(input + 16));
            }
        }
    }

    #[test]
    fn disjoint_train_test_and_class_counts() {
        for (target, positives) in [
            (parity_four as Target, 8),
            (and_four as Target, 1),
            (majority_four as Target, 5),
        ] {
            let train = split(0, 16, target);
            let heldout = split(16, 32, target);
            assert!(train.iter().all(|s| s.input < 16));
            assert!(heldout.iter().all(|s| s.input >= 16));
            assert_eq!(train.iter().filter(|s| s.fire).count(), positives);
            assert_eq!(heldout.iter().filter(|s| s.fire).count(), positives);
        }
    }
}

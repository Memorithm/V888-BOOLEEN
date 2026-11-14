//! Reproducible synthetic five-circle context-gating comparison.
//!
//! Run: cargo run --offline --bin fourier5-bench
//! This is a deliberately small structural control, not a biological or
//! broad out-of-distribution generalization benchmark.
use v888_booleen::fourier5::{Circle, FiveCircle, CIRCLES};
use v888_booleen::fourier5_dynamic::{train_context_gates, GatedFiveCircle, RadiusGate};

fn examples(start: u8, end: u8) -> Vec<(u8, u8, bool)> {
    let mut result = Vec::new();
    for input in start..end {
        for context in 0_u8..=1 {
            let target = input & 1 == 1 && context == 1;
            result.push((input, context, target));
        }
    }
    result
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut circles = [Circle::default(); CIRCLES];
    circles[0] = Circle {
        mask: 1,
        radius: 1,
        phase_pi: false,
    };
    let base = FiveCircle::new(circles, 2).map_err(|err| format!("{err:?}"))?;
    let mut gates = [RadiusGate::default(); CIRCLES];
    gates[0] = RadiusGate {
        context_mask: 1,
        boost: 0,
    };
    let initial = GatedFiveCircle::new(base, gates).map_err(|err| format!("{err:?}"))?;
    let train = examples(0, 16); // low fifth bit
    let test = examples(16, 32); // held-out high fifth bit
    let report = train_context_gates(initial, &train, 64).map_err(|err| format!("{err:?}"))?;

    let static_correct = test
        .iter()
        .filter(|(x, _, expected)| base.predict(*x) == Ok(*expected))
        .count();
    let gated_correct = report
        .model
        .correct_count(&test)
        .map_err(|err| format!("{err:?}"))?;

    println!(
        "task=synthetic_context_and train={} holdout={} static_correct={} gated_correct={} epochs={} updates={} boost={} converged={}",
        train.len(),
        test.len(),
        static_correct,
        gated_correct,
        report.epochs_used,
        report.updates,
        report.model.gates()[0].boost,
        report.converged
    );

    // Every x with its first bit on has opposite context-dependent labels,
    // hence no context-blind deterministic classifier can exceed 24/32 here.
    assert_eq!(test.len(), 32);
    assert_eq!(static_correct, 24);
    assert_eq!(gated_correct, 32);
    assert!(report.converged);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disjoint_fifth_bit_heldout_is_reproducible() {
        let train = examples(0, 16);
        let test = examples(16, 32);
        assert_eq!(train.len(), 32);
        assert_eq!(test.len(), 32);
        assert!(train.iter().all(|(x, _, _)| x & 16 == 0));
        assert!(test.iter().all(|(x, _, _)| x & 16 != 0));
        assert_eq!(train.iter().filter(|(_, _, label)| *label).count(), 8);
        assert_eq!(test.iter().filter(|(_, _, label)| *label).count(), 8);
    }

    #[test]
    fn contextual_labels_prove_a_context_blind_capacity_limit() {
        let test = examples(16, 32);
        let mut upper_bound = 0;
        for input in 16..32 {
            let labels: Vec<bool> = test
                .iter()
                .filter(|(x, _, _)| *x == input)
                .map(|(_, _, label)| *label)
                .collect();
            let true_count = labels.iter().filter(|&&x| x).count();
            upper_bound += true_count.max(labels.len() - true_count);
        }
        assert_eq!(upper_bound, 24);
    }
}

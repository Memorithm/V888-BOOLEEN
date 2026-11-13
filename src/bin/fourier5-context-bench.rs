//! Bounded synthetic demonstration of *learned* context gate masks.
//! Run inside trusted RemoteOps: cargo run --offline --bin fourier5-context-bench
//! Do not interpret accuracy as biological fidelity or broad OOD transfer.
use v888_booleen::fourier5::{Circle, FiveCircle, CIRCLES};
use v888_booleen::fourier5_context_search::{learn_context_masks, ContextSample};
use v888_booleen::fourier5_dynamic::{train_context_gates, GatedFiveCircle, RadiusGate};

fn samples(first: u8, end: u8) -> Vec<ContextSample> {
    (first..end)
        .flat_map(|context| {
            [0, 1].map(move |input| ContextSample {
                input,
                context,
                fire: input == 1 && (context & 10) == 10,
            })
        })
        .collect()
}

fn base() -> FiveCircle {
    let circles = std::array::from_fn(|slot| Circle {
        mask: 1,
        radius: if slot == 0 { 1 } else { 0 },
        phase_pi: false,
    });
    FiveCircle::new(circles, 2).expect("known valid base")
}

fn correct_static(base: &FiveCircle, cases: &[ContextSample]) -> usize {
    cases
        .iter()
        .filter(|case| base.predict(case.input) == Ok(case.fire))
        .count()
}

fn correct_context(model: &GatedFiveCircle, cases: &[ContextSample]) -> Result<usize, String> {
    model
        .correct_count(
            &cases
                .iter()
                .map(|s| (s.input, s.context, s.fire))
                .collect::<Vec<_>>(),
        )
        .map_err(|err| format!("{err:?}"))
}

fn main() -> Result<(), String> {
    let train = samples(0, 16);
    let test = samples(16, 32);
    let learned = learn_context_masks(base(), &train, 32).map_err(|err| format!("{err:?}"))?;

    // Same five circles / five context sources, but predetermined singleton
    // context masks. Both approaches have access to identical examples.
    let singleton_gates = std::array::from_fn(|i| RadiusGate {
        context_mask: 1 << i,
        boost: 0,
    });
    let singleton =
        GatedFiveCircle::new(base(), singleton_gates).map_err(|err| format!("{err:?}"))?;
    let tuples = train
        .iter()
        .map(|s| (s.input, s.context, s.fire))
        .collect::<Vec<_>>();
    let fixed_fit =
        train_context_gates(singleton, &tuples, 32).map_err(|err| format!("{err:?}"))?;
    let static_hits = correct_static(&base(), &test);
    let singleton_hits = correct_context(&fixed_fit.model, &test)?;
    let learned_hits = correct_context(&learned.model, &test)?;

    println!(
        "task=context_pair_conjunction train={} heldout={} static_hits={} singleton_hits={} learned_hits={} learned_gates={} gate_mask={} gate_boost={} candidates={} sample_visits={} fixed_epochs={} fixed_updates={}",
        train.len(),
        test.len(),
        static_hits,
        singleton_hits,
        learned_hits,
        learned.selected,
        learned.model.gates()[0].context_mask,
        learned.model.gates()[0].boost,
        learned.candidates_evaluated,
        learned.sample_visits,
        fixed_fit.epochs_used,
        fixed_fit.updates
    );
    // The control test is only held out on irrelevant context bit 4.
    if train.len() != 32
        || test.len() != 32
        || static_hits != 28
        || learned_hits != 32
        || learned.train_correct != 32
        || learned.model.gates()[0].context_mask != 10
    {
        return Err("synthetic control failed".into());
    }
    println!("FOURIER5_CONTEXT_MASK_LEARNING_SYNTHETIC_PASS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separation_is_context_only_with_positive_cases_in_both_splits() {
        let train = samples(0, 16);
        let test = samples(16, 32);
        assert_eq!(train.len(), 32);
        assert_eq!(test.len(), 32);
        assert_eq!(train.iter().filter(|s| s.fire).count(), 4);
        assert_eq!(test.iter().filter(|s| s.fire).count(), 4);
        assert!(train.iter().all(|s| s.context & 16 == 0));
        assert!(test.iter().all(|s| s.context & 16 != 0));
    }
}

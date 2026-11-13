//! Reproducible large-network stress test on *synthetic* recurrent wiring.
//!
//! Three scales, up to 188,508 neurons and 1,885,080 directed contacts per
//! replica. Uses a nontrivial parity-5 AND NOT(context-conjunction) target,
//! train/held-out disjoint 10-bit examples, context-aware controls, and a
//! controlled one-bit perturbation of recurrent trajectories.
//!
//! Run ONLY via budgeted offline RemoteOps CI, not on production workloads.
use std::time::Instant;
use v888_booleen::fourier5::{Circle, FiveCircle, CIRCLES};
use v888_booleen::fourier5_context_search::{learn_context_masks, ContextSample};
use v888_booleen::fourier5_dynamic::{train_context_gates, GatedFiveCircle, RadiusGate};
use v888_booleen::fourier5_recurrent::{NetworkConfig, RecurrentFiveCircle, SyntheticTopology};

fn base() -> FiveCircle {
    let mut circles = [Circle::default(); CIRCLES];
    circles[1] = Circle {
        mask: 31,
        radius: 2,
        phase_pi: false,
    };
    FiveCircle::new(circles, -1).expect("base parameters validated by construction")
}

fn teacher(input: u8, context: u8) -> bool {
    (input.count_ones() & 1) == 1 && (context & 5) != 5
}

#[derive(Clone, Copy)]
struct Scrambler(u64);

impl Scrambler {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn split(seed: u64) -> (Vec<ContextSample>, Vec<ContextSample>) {
    // Exactly 1024 unique pairs x in 0..32, c in 0..32; Fisher-Yates
    // seeded permutation, then 512 TRAIN / 512 held-out pairs. No leak.
    let mut ids: Vec<u16> = (0..1024).collect();
    let mut rng = Scrambler(seed);
    for i in (1..ids.len()).rev() {
        let j = (rng.next() % (i + 1) as u64) as usize;
        ids.swap(i, j);
    }
    let convert = |rows: &[u16]| {
        rows.iter()
            .map(|&id| {
                let input = (id / 32) as u8;
                let context = (id % 32) as u8;
                ContextSample {
                    input,
                    context,
                    fire: teacher(input, context),
                }
            })
            .collect::<Vec<_>>()
    };
    (convert(&ids[..512]), convert(&ids[512..]))
}

fn score_base(model: &FiveCircle, examples: &[ContextSample]) -> usize {
    examples
        .iter()
        .filter(|sample| model.predict(sample.input) == Ok(sample.fire))
        .count()
}

fn score_gated(model: &GatedFiveCircle, examples: &[ContextSample]) -> Result<usize, String> {
    let cases: Vec<_> = examples
        .iter()
        .map(|s| (s.input, s.context, s.fire))
        .collect();
    model
        .correct_count(&cases)
        .map_err(|err| format!("{err:?}"))
}

fn train_benchmark(seed: u64) -> Result<GatedFiveCircle, String> {
    let (training, held_out) = split(seed);
    let began = Instant::now();
    let learned = learn_context_masks(base(), &training, 32).map_err(|err| format!("{err:?}"))?;
    let learned_time = began.elapsed();

    let singleton_gates = std::array::from_fn(|i| RadiusGate {
        context_mask: 1_u8 << i,
        boost: 0,
    });
    let fixed_model =
        GatedFiveCircle::new(base(), singleton_gates).map_err(|err| format!("{err:?}"))?;
    let fixed_inputs: Vec<_> = training
        .iter()
        .map(|s| (s.input, s.context, s.fire))
        .collect();
    let began = Instant::now();
    let fixed =
        train_context_gates(fixed_model, &fixed_inputs, 32).map_err(|err| format!("{err:?}"))?;
    let fixed_time = began.elapsed();

    let static_hits = score_base(&base(), &held_out);
    let fixed_hits = score_gated(&fixed.model, &held_out)?;
    let learned_hits = score_gated(&learned.model, &held_out)?;
    println!(
        "train_seed={seed} train={} test={} target_parity5_context2 static_hits={} fixed_hits={} learned_hits={} selected={} mask={} boost={} candidates={} search_visits={} search_ms={} fixed_updates={} fixed_ms={}",
        training.len(),
        held_out.len(),
        static_hits,
        fixed_hits,
        learned_hits,
        learned.selected,
        learned.model.gates()[0].context_mask,
        learned.model.gates()[0].boost,
        learned.candidates_evaluated,
        learned.sample_visits,
        learned_time.as_millis(),
        fixed.updates,
        fixed_time.as_millis()
    );
    if training.len() != 512 || held_out.len() != 512 || learned.model.base() != &base() {
        return Err("broken training protocol".into());
    }
    Ok(learned.model)
}

fn run_profile(
    nodes: usize,
    steps: usize,
    seed: u64,
    topology: SyntheticTopology,
    model: GatedFiveCircle,
) -> Result<(), String> {
    let config = NetworkConfig {
        nodes,
        topology,
        seed,
        initial_activity_per_256: 128,
    };
    let began = Instant::now();
    let mut trained =
        RecurrentFiveCircle::seeded(config, model).map_err(|err| format!("{err:?}"))?;
    let mut perturbed = trained.clone();
    perturbed
        .flip(nodes / 2)
        .map_err(|err| format!("{err:?}"))?;
    let no_context = GatedFiveCircle::new(base(), [RadiusGate::default(); CIRCLES])
        .map_err(|err| format!("{err:?}"))?;
    let mut static_control =
        RecurrentFiveCircle::seeded(config, no_context).map_err(|err| format!("{err:?}"))?;
    let build_ms = began.elapsed().as_millis();
    if trained.wiring_fingerprint() != static_control.wiring_fingerprint()
        || trained.state() != static_control.state()
        || trained
            .hamming_distance(&perturbed)
            .map_err(|err| format!("{err:?}"))?
            != 1
    {
        return Err("invalid paired initial condition".into());
    }

    let mut last_active = 0;
    let mut last_static = 0;
    let mut last_hamming = 1;
    let mut max_hamming = 1;
    let mut rising_total = 0_u64;
    let mut falling_total = 0_u64;
    let began = Instant::now();
    for _ in 0..steps {
        let stats = trained.step().map_err(|err| format!("{err:?}"))?;
        let static_stats = static_control.step().map_err(|err| format!("{err:?}"))?;
        perturbed.step().map_err(|err| format!("{err:?}"))?;
        last_hamming = trained
            .hamming_distance(&perturbed)
            .map_err(|err| format!("{err:?}"))?;
        max_hamming = max_hamming.max(last_hamming);
        last_active = stats.active;
        last_static = static_stats.active;
        rising_total += stats.rising;
        falling_total += stats.falling;
    }
    let elapsed_ms = began.elapsed().as_millis();
    let contacts_per_tick = trained.contact_count();
    let logical_bytes_per_replica = trained.estimated_state_and_wiring_bytes();
    println!(
        "scale_nodes={nodes} topology={topology:?} seed={seed} directed_contacts={} steps={steps} initial_pairs_hamming=1 build_ms={build_ms} simulate_ms={elapsed_ms} logical_bytes_per_replica={logical_bytes_per_replica} learned_active={last_active} static_active={last_static} rising_total={rising_total} falling_total={falling_total} final_perturbation_hamming={last_hamming} max_perturbation_hamming={max_hamming} wiring_fingerprint={:016x}",
        contacts_per_tick,
        trained.wiring_fingerprint()
    );
    if contacts_per_tick != nodes * 10
        || trained.tick_count() != steps
        || last_active > nodes as u64
        || last_hamming > nodes as u64
    {
        return Err("network telemetry invariant violated".into());
    }
    Ok(())
}

fn main() -> Result<(), String> {
    // Two independent random TRAIN/TEST partitions; one learned rule is
    // selected for the main profile, the other is reported separately.
    let shared = train_benchmark(0x57ad_b0a1)?;
    let _second_seed = train_benchmark(0xa08c_11ef)?;

    run_profile(8_192, 32, 0x0801, SyntheticTopology::LocalRing, shared)?;
    run_profile(
        65_536,
        16,
        0x1fff_8081,
        SyntheticTopology::SeededPseudorandom,
        shared,
    )?;
    run_profile(
        188_508,
        8,
        0x88b2_2026,
        SyntheticTopology::SeededPseudorandom,
        shared,
    )?;
    println!("FOURIER5_LARGE_RECURRENT_SYNTHETIC_BENCH_PASS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn randomized_holdout_does_not_duplicate_training_pairs() {
        for seed in [0x57ad_b0a1, 0xa08c_11ef] {
            let (train, test) = split(seed);
            assert_eq!(train.len(), 512);
            assert_eq!(test.len(), 512);
            let mut seen = [false; 1024];
            for item in train.iter().chain(test.iter()) {
                let id = usize::from(item.input) * 32 + usize::from(item.context);
                assert!(!seen[id]);
                seen[id] = true;
                assert_eq!(teacher(item.input, item.context), item.fire);
            }
            assert!(seen.into_iter().all(|x| x));
        }
    }

    #[test]
    fn frozen_base_cannot_model_contextual_suppression() {
        let b = base();
        let (train, test) = split(0x57ad_b0a1);
        assert!(score_base(&b, &train) < train.len());
        assert!(score_base(&b, &test) < test.len());
        assert_eq!(teacher(1, 0), true);
        assert_eq!(teacher(1, 5), false);
        assert_eq!(b.predict(1), Ok(true));
    }
}

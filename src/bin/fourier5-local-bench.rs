//! Large-scale independent five-circle plasticity with controlled A/B interference.
//!
//! CPU-only synthetic experiment. Persistent learned context gains are unique
//! per neuron; wiring and Boolean state storage follow the existing V888
//! recurrent model. No biological memory claim.
use std::time::Instant;
use v888_booleen::fourier5::{Circle, FiveCircle, CIRCLES};
use v888_booleen::fourier5_dynamic::{GatedFiveCircle, RadiusGate};
use v888_booleen::fourier5_local_learning::{LocalPopulation, LocalTask};
use v888_booleen::fourier5_recurrent::{NetworkConfig, RecurrentFiveCircle, SyntheticTopology};

fn parity_base() -> FiveCircle {
    let mut circles = [Circle::default(); CIRCLES];
    circles[0] = Circle {
        mask: 31,
        radius: 2,
        phase_pi: false,
    };
    FiveCircle::new(circles, -1).expect("synthetic base is valid")
}

fn untrained_model() -> GatedFiveCircle {
    let gates = std::array::from_fn(|i| RadiusGate {
        context_mask: 1_u8 << i,
        boost: 0,
    });
    GatedFiveCircle::new(parity_base(), gates).expect("synthetic gates are valid")
}

fn state_hamming_words(start: &[u64], current: &[u64]) -> u64 {
    start
        .iter()
        .zip(current)
        .map(|(&a, &b)| u64::from((a ^ b).count_ones()))
        .sum()
}

fn profile(
    nodes: usize,
    ticks: usize,
    seed: u64,
    topology: SyntheticTopology,
) -> Result<(), String> {
    let initial = Instant::now();
    let mut task_a =
        LocalPopulation::new(nodes, seed, parity_base()).map_err(|error| format!("{error:?}"))?;
    let untrained_a = task_a
        .evaluate(LocalTask::A)
        .map_err(|error| format!("{error:?}"))?;
    let untrained_b = task_a
        .evaluate(LocalTask::B)
        .map_err(|error| format!("{error:?}"))?;
    let train_a = task_a
        .train(LocalTask::A, 8)
        .map_err(|error| format!("{error:?}"))?;
    let trained_a = task_a
        .evaluate(LocalTask::A)
        .map_err(|error| format!("{error:?}"))?;
    let a_fingerprint = task_a.gain_fingerprint();
    let phase_a_ms = initial.elapsed().as_millis();

    // "Shared" is the SAME per-neuron gain vector modified by task B.
    let mut sequential_b = task_a.clone();
    let before_b = sequential_b
        .evaluate(LocalTask::B)
        .map_err(|error| format!("{error:?}"))?;
    let started = Instant::now();
    let train_shared_b = sequential_b
        .train(LocalTask::B, 8)
        .map_err(|error| format!("{error:?}"))?;
    let sequential_a = sequential_b
        .evaluate(LocalTask::A)
        .map_err(|error| format!("{error:?}"))?;
    let sequential_b_test = sequential_b
        .evaluate(LocalTask::B)
        .map_err(|error| format!("{error:?}"))?;
    let sequential_b_ms = started.elapsed().as_millis();

    // A costed task-indexed two-head control: a fresh separately learned B.
    let started = Instant::now();
    let mut task_b =
        LocalPopulation::new(nodes, seed, parity_base()).map_err(|error| format!("{error:?}"))?;
    let train_independent_b = task_b
        .train(LocalTask::B, 8)
        .map_err(|error| format!("{error:?}"))?;
    let bank_b = task_b
        .evaluate(LocalTask::B)
        .map_err(|error| format!("{error:?}"))?;
    let bank_b_ms = started.elapsed().as_millis();

    println!(
        "individual_learning nodes={nodes} seed={seed} train_per_node=64 test_per_node=64 total_train_examples={} untrained_a={}/{} untrained_b={}/{} phase_a={}/{} before_b={}/{} sequential_retained_a={}/{} sequential_b={}/{} taskbank_a={}/{} taskbank_b={}/{} a_perfect_neurons={} shared_b_perfect_neurons={} bank_b_perfect_neurons={} phase_a_updates={} shared_b_updates={} bank_b_updates={} phase_a_ms={} shared_b_ms={} bank_b_ms={} a_fingerprint={:016x} teacher_histogram={:?}",
        train_a.train_examples,
        untrained_a.correct,
        untrained_a.examples,
        untrained_b.correct,
        untrained_b.examples,
        trained_a.correct,
        trained_a.examples,
        before_b.correct,
        before_b.examples,
        sequential_a.correct,
        sequential_a.examples,
        sequential_b_test.correct,
        sequential_b_test.examples,
        trained_a.correct,
        trained_a.examples,
        bank_b.correct,
        bank_b.examples,
        trained_a.perfect_neurons,
        sequential_b_test.perfect_neurons,
        bank_b.perfect_neurons,
        train_a.updates_total,
        train_shared_b.updates_total,
        train_independent_b.updates_total,
        phase_a_ms,
        sequential_b_ms,
        bank_b_ms,
        a_fingerprint,
        task_a.teacher_bit_histogram(LocalTask::A)
    );
    if train_a.train_examples != nodes * 64
        || train_a.train_correct != train_a.train_examples
        || trained_a.correct != trained_a.examples
        || sequential_b_test.correct != sequential_b_test.examples
        || bank_b.correct != bank_b.examples
        || sequential_a.correct >= trained_a.correct
        || before_b.correct >= sequential_b_test.correct
        || task_a.gain_fingerprint() != a_fingerprint
    {
        return Err("per-neuron supervised A/B qualification failed".into());
    }

    let config = NetworkConfig {
        nodes,
        seed: seed ^ 0x33b8_2026,
        topology,
        initial_activity_per_256: 128,
    };
    let started = Instant::now();
    let base = untrained_model();
    let mut recurrent_a =
        RecurrentFiveCircle::seeded(config, base).map_err(|error| format!("{error:?}"))?;
    let mut recurrent_shared_b =
        RecurrentFiveCircle::seeded(config, base).map_err(|error| format!("{error:?}"))?;
    let mut recurrent_untrained =
        RecurrentFiveCircle::seeded(config, base).map_err(|error| format!("{error:?}"))?;
    recurrent_a
        .install_local_models(task_a.models())
        .map_err(|error| format!("{error:?}"))?;
    recurrent_shared_b
        .install_local_models(sequential_b.models())
        .map_err(|error| format!("{error:?}"))?;
    let build_ms = started.elapsed().as_millis();
    let initial_state = recurrent_a.state().words().to_vec();
    if recurrent_a.state() != recurrent_shared_b.state()
        || recurrent_a.state() != recurrent_untrained.state()
        || recurrent_a.wiring_fingerprint() != recurrent_shared_b.wiring_fingerprint()
        || recurrent_a.wiring_fingerprint() != recurrent_untrained.wiring_fingerprint()
        || recurrent_a.neuron_model(0) != Some(task_a.models()[0])
    {
        return Err("paired wiring or independent model install invalid".into());
    }
    let fingerprints_before = (task_a.gain_fingerprint(), sequential_b.gain_fingerprint());
    let started = Instant::now();
    let mut max_distance_a_b = 0;
    let mut last_distance_a_b = 0;
    let mut last_distance_a_untrained = 0;
    let mut a_rising = 0;
    let mut a_falling = 0;
    let mut last_a_active = 0;
    let mut last_b_active = 0;
    let mut last_untrained_active = 0;
    for _ in 0..ticks {
        let a = recurrent_a.step().map_err(|error| format!("{error:?}"))?;
        let b = recurrent_shared_b
            .step()
            .map_err(|error| format!("{error:?}"))?;
        let control = recurrent_untrained
            .step()
            .map_err(|error| format!("{error:?}"))?;
        last_a_active = a.active;
        last_b_active = b.active;
        last_untrained_active = control.active;
        a_rising += a.rising;
        a_falling += a.falling;
        last_distance_a_b = recurrent_a
            .hamming_distance(&recurrent_shared_b)
            .map_err(|error| format!("{error:?}"))?;
        max_distance_a_b = max_distance_a_b.max(last_distance_a_b);
        last_distance_a_untrained = recurrent_a
            .hamming_distance(&recurrent_untrained)
            .map_err(|error| format!("{error:?}"))?;
    }
    let simulation_ms = started.elapsed().as_millis();
    let initial_final_hamming = state_hamming_words(&initial_state, recurrent_a.state().words());
    println!(
        "recurrent_local nodes={nodes} topology={topology:?} contacts={} ticks={ticks} build_ms={build_ms} simulation_ms={simulation_ms} stored_model_bytes={} logical_wiring_and_state_bytes={} a_active={last_a_active} b_active={last_b_active} untrained_active={last_untrained_active} a_rising={a_rising} a_falling={a_falling} a_b_hamming={last_distance_a_b} max_a_b_hamming={max_distance_a_b} a_untrained_hamming={last_distance_a_untrained} initial_final_hamming={initial_final_hamming} wiring_fingerprint={:016x}",
        recurrent_a.contact_count(),
        std::mem::size_of_val(task_a.models()),
        recurrent_a.estimated_state_and_wiring_bytes(),
        recurrent_a.wiring_fingerprint()
    );
    if recurrent_a.tick_count() != ticks
        || recurrent_a.contact_count() != nodes * 10
        || last_distance_a_b > nodes as u64
        || last_distance_a_untrained > nodes as u64
        || task_a.gain_fingerprint() != fingerprints_before.0
        || sequential_b.gain_fingerprint() != fingerprints_before.1
        || task_a
            .evaluate(LocalTask::A)
            .map_err(|e| format!("{e:?}"))?
            != trained_a
    {
        return Err("recurrent long-horizon qualification failed".into());
    }
    Ok(())
}

fn main() -> Result<(), String> {
    profile(8_192, 512, 0xf020_0101, SyntheticTopology::LocalRing)?;
    profile(
        65_536,
        256,
        0xf020_0202,
        SyntheticTopology::SeededPseudorandom,
    )?;
    profile(
        188_508,
        128,
        0xf020_0303,
        SyntheticTopology::SeededPseudorandom,
    )?;
    println!("FOURIER5_PER_NEURON_LONG_HORIZON_SYNTHETIC_PASS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learned_local_gain_population_can_be_installed_on_existing_graph() {
        let mut pop = LocalPopulation::new(128, 1, parity_base()).unwrap();
        pop.train(LocalTask::A, 8).unwrap();
        let conf = NetworkConfig {
            nodes: 128,
            seed: 9,
            topology: SyntheticTopology::LocalRing,
            initial_activity_per_256: 128,
        };
        let mut graph = RecurrentFiveCircle::seeded(conf, untrained_model()).unwrap();
        let wiring = graph.wiring_fingerprint();
        let initial_state = graph.state().clone();
        graph.install_local_models(pop.models()).unwrap();
        assert_eq!(graph.wiring_fingerprint(), wiring);
        assert_eq!(graph.state(), &initial_state);
        for i in 0..128 {
            assert_eq!(graph.neuron_model(i), Some(pop.models()[i]));
        }
        assert!(graph.install_local_models(&pop.models()[..127]).is_err());
        assert_eq!(graph.neuron_model(127), Some(pop.models()[127]));
        for _ in 0..128 {
            graph.step().unwrap();
        }
        assert_eq!(graph.tick_count(), 128);
        assert_eq!(graph.wiring_fingerprint(), wiring);
    }

    #[test]
    fn untrained_model_has_five_explicit_singleton_context_gates() {
        let model = untrained_model();
        for (i, gate) in model.gates().iter().enumerate() {
            assert_eq!(gate.context_mask, 1 << i);
            assert_eq!(gate.boost, 0);
        }
    }
}

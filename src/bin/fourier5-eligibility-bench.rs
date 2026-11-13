//! B888-4: bounded delayed teacher, recurrent ticks and associative recall.
//! The plasticity mechanism is opt-in and runs on synthetic fixed topology.
use std::time::Instant;
use v888_booleen::fourier5_eligibility::{associative_readout, DelayedEligibility};
use v888_booleen::fourier5_recurrent::{
    NetworkConfig, RecurrentFiveCircle, SyntheticTopology,
};

const NODES: usize = 128;
const READOUT: usize = 19;
const RETENTION_TICKS: usize = 512;
const EPOCHS: usize = 4;

fn target(seed: u64, cue: usize) -> bool {
    ((cue + (seed as usize & 1)) & 1) != 0
}

fn query(network: &RecurrentFiveCircle, cue: usize) -> Result<bool, String> {
    let mut probe = network.clone();
    // Test word differs from TRAIN (odd parity preserved). The fifth
    // context bit is an unseen, irrelevant distractor at recall.
    probe
        .drive_neuron_sources(READOUT, 0b0_0111, (1 << cue) | 0b1_0000)
        .map_err(|e| format!("{e:?}"))?;
    probe.step().map_err(|e| format!("{e:?}"))?;
    probe
        .state()
        .get(READOUT)
        .map_err(|e| format!("{e:?}"))
}

fn score(network: &RecurrentFiveCircle, seed: u64) -> Result<usize, String> {
    let mut correct = 0;
    for cue in 0..4 {
        correct += usize::from(query(network, cue)? == target(seed, cue));
    }
    Ok(correct)
}

fn run(seed: u64, delay: usize) -> Result<(), String> {
    let config = NetworkConfig {
        nodes: NODES,
        seed,
        topology: SyntheticTopology::LocalRing,
        initial_activity_per_256: 128,
    };
    let started = Instant::now();
    let mut learned = RecurrentFiveCircle::seeded(config, associative_readout())
        .map_err(|e| format!("{e:?}"))?;
    let frozen = learned.clone();
    let mut wrong_reward = learned.clone();
    let initial_fingerprint = learned.wiring_fingerprint();
    let mut eligible = DelayedEligibility::new(delay).map_err(|e| format!("{e:?}"))?;
    let mut wrong_eligible = DelayedEligibility::new(delay).map_err(|e| format!("{e:?}"))?;
    let mut delivered_after_delay = 0;
    for _ in 0..EPOCHS {
        for cue in 0..4 {
            for (net, plastic, label) in [
                (&mut learned, &mut eligible, target(seed, cue)),
                (&mut wrong_reward, &mut wrong_eligible, !target(seed, cue)),
            ] {
                net.drive_neuron_sources(READOUT, 1, 1 << cue)
                    .map_err(|e| format!("{e:?}"))?;
                plastic.record(net, READOUT).map_err(|e| format!("{e:?}"))?;
                for _ in 0..delay {
                    net.step().map_err(|e| format!("{e:?}"))?;
                }
                let report = plastic.feedback(net, label).map_err(|e| format!("{e:?}"))?;
                if report.delay_ticks != delay {
                    return Err("feedback delivered at incorrect recurrent tick".into());
                }
                delivered_after_delay += 1;
            }
        }
    }
    let train_ms = started.elapsed().as_millis();
    let frozen_before_retention = score(&frozen, seed)?;
    let learned_before_retention = score(&learned, seed)?;
    let wrong_before_retention = score(&wrong_reward, seed)?;
    let learned_model = learned.neuron_model(READOUT);
    let wrong_model = wrong_reward.neuron_model(READOUT);
    let started = Instant::now();
    for _ in 0..RETENTION_TICKS {
        learned.step().map_err(|e| format!("{e:?}"))?;
        wrong_reward.step().map_err(|e| format!("{e:?}"))?;
    }
    let retention_ms = started.elapsed().as_millis();
    let frozen_after_retention = score(&frozen, seed)?;
    let learned_after_retention = score(&learned, seed)?;
    let wrong_after_retention = score(&wrong_reward, seed)?;
    let wiring_ok = learned.wiring_fingerprint() == initial_fingerprint
        && wrong_reward.wiring_fingerprint() == initial_fingerprint
        && frozen.wiring_fingerprint() == initial_fingerprint;
    let weights_stable = learned.neuron_model(READOUT) == learned_model
        && wrong_reward.neuron_model(READOUT) == wrong_model;
    println!(
        "eligibility_recall seed={seed} neurons={NODES} contacts={} feedback_delay={delay} retention_ticks={RETENTION_TICKS} episodes={} delivered={} updates={} wrong_reward_updates={} train_ms={train_ms} retention_ms={retention_ms} learned_before={learned_before_retention}/4 learned_after={learned_after_retention}/4 frozen_before={frozen_before_retention}/4 frozen_after={frozen_after_retention}/4 wrong_before={wrong_before_retention}/4 wrong_after={wrong_after_retention}/4 wiring_ok={wiring_ok} weights_stable={weights_stable}",
        learned.contact_count(),
        EPOCHS * 4,
        delivered_after_delay,
        eligible.updates(),
        wrong_eligible.updates(),
    );
    if !wiring_ok
        || !weights_stable
        || learned_after_retention != 4
        || learned_before_retention != 4
        || frozen_after_retention != 2
        || frozen_before_retention != 2
        || wrong_before_retention != 0
        || wrong_after_retention != 0
        || eligible.delivered() != EPOCHS * 4
        || wrong_eligible.delivered() != EPOCHS * 4
    {
        return Err("bounded delayed-feedback associative recall failed".into());
    }
    Ok(())
}

fn main() -> Result<(), String> {
    for seed in [11_u64, 42, 93] {
        for delay in [8_usize, 64, 128] {
            run(seed, delay)?;
        }
    }
    println!("FOURIER5_DELAYED_ELIGIBILITY_SYNTHETIC_PASS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matched_delay_controls() {
        run(11, 8).unwrap();
    }
}

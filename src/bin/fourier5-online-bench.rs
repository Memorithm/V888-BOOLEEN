//! Controlled cue -> recurrent delay -> delayed feedback -> cued recall.
//!
//! A seeded teacher labels each neuron's original cue AFTER the delay. Three
//! matched arms: eligibility (correct past cue), no-trace reward-time input,
//! frozen parameters. TRAIN/TEST pair collisions are explicitly excluded.
//! This tests local delayed supervised association, NOT autonomous replay of
//! a missing cue or biological timing-dependent plasticity.
use std::time::Instant;
use v888_booleen::fourier5::{Circle, FiveCircle, CIRCLES};
use v888_booleen::fourier5_dynamic::{GatedFiveCircle, RadiusGate};
use v888_booleen::fourier5_population_eligibility::EligibilityBank;
use v888_booleen::fourier5_recurrent::{NetworkConfig, RecurrentFiveCircle, SyntheticTopology};
use v888_booleen::BooleanState;

const EVAL_EPISODES: usize = 24;
const TRAIN_EPISODES: usize = 128;

/// Stable SplitMix64. Synthetic stimuli and teacher IDs do not use RNG
/// facilities, file state or system entropy.
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[derive(Clone, Copy)]
enum Task {
    A,
    B,
}

fn teacher_bit(seed: u64, index: usize, task: Task) -> usize {
    let selected = (mix64(seed ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)) % 4) as usize;
    match task {
        Task::A => selected,
        Task::B => (selected + 1) % 4,
    }
}

fn teacher(seed: u64, index: usize, input: u8, context: u8, task: Task) -> bool {
    input.count_ones() % 2 != 0 && context & (1 << teacher_bit(seed, index, task)) == 0
}

fn stimulus(nodes: usize, seed: u64, episode: u64) -> BooleanState {
    let mut state = BooleanState::new(nodes);
    let mut word = 0_u64;
    let mut cursor = mix64(seed ^ episode.wrapping_mul(0xd6e8_feb8_6659_fd93));
    for i in 0..nodes {
        if i % 64 == 0 {
            cursor = mix64(cursor);
            word = cursor;
        }
        if (word >> (i % 64)) & 1 != 0 {
            state.set(i, true).expect("valid stimulus index");
        }
    }
    state
}

fn base() -> GatedFiveCircle {
    let mut circles = [Circle::default(); CIRCLES];
    circles[0] = Circle {
        mask: 31,
        radius: 2,
        phase_pi: false,
    };
    let original = FiveCircle::new(circles, -1).expect("known valid base");
    let gates = std::array::from_fn(|i| RadiusGate {
        context_mask: if i == 0 { 0 } else { 1 << (i - 1) },
        boost: 0,
    });
    GatedFiveCircle::new(original, gates).expect("known valid gates")
}

fn create(nodes: usize, seed: u64) -> RecurrentFiveCircle {
    RecurrentFiveCircle::seeded(
        NetworkConfig {
            nodes,
            seed,
            topology: SyntheticTopology::SeededPseudorandom,
            initial_activity_per_256: 128,
        },
        base(),
    )
    .expect("valid synthetic configuration")
}

/// 1024 possible (data,context) combinations per neuron. Used ONLY by the
/// benchmark to exclude already-seen TRAIN pairs from all TEST metrics.
#[derive(Clone)]
struct Seen(Vec<[u64; 16]>);

impl Seen {
    fn new(nodes: usize) -> Self {
        Self(vec![[0; 16]; nodes])
    }

    fn index(input: u8, context: u8) -> usize {
        usize::from(input) * 32 + usize::from(context)
    }

    fn mark(&mut self, node: usize, input: u8, context: u8) {
        let id = Self::index(input, context);
        self.0[node][id / 64] |= 1_u64 << (id % 64);
    }

    fn has(&self, node: usize, input: u8, context: u8) -> bool {
        let id = Self::index(input, context);
        self.0[node][id / 64] & (1_u64 << (id % 64)) != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RecallMetrics {
    correct_cued: usize,
    correct_autonomous: usize,
    novel: usize,
    evaluated_episodes: usize,
}

fn evaluate(
    network: &mut RecurrentFiveCircle,
    seen: &Seen,
    seed: u64,
    task: Task,
    delay: usize,
) -> Result<RecallMetrics, String> {
    let nodes = network.node_count();
    let mut result = RecallMetrics {
        correct_cued: 0,
        correct_autonomous: 0,
        novel: 0,
        evaluated_episodes: EVAL_EPISODES,
    };
    for ep in 0..EVAL_EPISODES {
        let cue = stimulus(nodes, seed ^ 0x81ee_4023_fad0_5b18, ep as u64);
        network.replace_state(cue).map_err(|e| format!("{e:?}"))?;
        let pairs = network
            .capture_local_signals()
            .map_err(|e| format!("{e:?}"))?;
        let mut goals = vec![false; nodes];
        let mut novel = vec![false; nodes];
        for (index, &(input, context)) in pairs.iter().enumerate() {
            goals[index] = teacher(seed, index, input, context, task);
            if !seen.has(index, input, context) {
                novel[index] = true;
                result.novel += 1;
                let prediction = network
                    .neuron_model(index)
                    .ok_or("missing local model")?
                    .predict(input, context)
                    .map_err(|e| format!("{e:?}"))?;
                result.correct_cued += usize::from(prediction == goals[index]);
            }
        }
        for _ in 0..delay {
            network.step().map_err(|e| format!("{e:?}"))?;
        }
        for index in 0..nodes {
            if novel[index] {
                let observed = network.state().get(index).map_err(|e| format!("{e:?}"))?;
                result.correct_autonomous += usize::from(observed == goals[index]);
            }
        }
    }
    if result.novel == 0 {
        return Err("empty test holdout after excluding pair collisions".into());
    }
    Ok(result)
}

fn run(nodes: usize, delay: usize, seed: u64) -> Result<(), String> {
    let time = Instant::now();
    let mut trace_network = create(nodes, seed);
    let mut stale_network = create(nodes, seed);
    let mut frozen_network = create(nodes, seed);
    let fingerprint = trace_network.wiring_fingerprint();
    let mut bank = EligibilityBank::new(nodes, delay).map_err(|e| format!("{e:?}"))?;
    let mut seen = Seen::new(nodes);
    let build_ms = time.elapsed().as_millis();
    let mut updated_trace = [0_usize; 2];
    let mut updated_stale = [0_usize; 2];
    let mut saved_a: Option<RecurrentFiveCircle> = None;
    let mut phase_ms = [0_u128; 2];

    for (task_index, task) in [Task::A, Task::B].into_iter().enumerate() {
        let phase_begin = Instant::now();
        for episode in 0..TRAIN_EPISODES {
            let cue = stimulus(
                nodes,
                seed ^ (task_index as u64).wrapping_mul(0x63ac_f129_1fc0_e3d5),
                episode as u64,
            );
            trace_network
                .replace_state(cue.clone())
                .map_err(|e| format!("{e:?}"))?;
            stale_network
                .replace_state(cue)
                .map_err(|e| format!("{e:?}"))?;
            bank.capture(&trace_network).map_err(|e| format!("{e:?}"))?;
            let pairs = trace_network
                .capture_local_signals()
                .map_err(|e| format!("{e:?}"))?;
            if pairs
                != stale_network
                    .capture_local_signals()
                    .map_err(|e| format!("{e:?}"))?
            {
                return Err("different training cue in paired models".into());
            }
            let mut labels = Vec::with_capacity(nodes);
            for (i, &(input, context)) in pairs.iter().enumerate() {
                seen.mark(i, input, context);
                labels.push(teacher(seed, i, input, context, task));
            }
            // Neither model has seen a teacher signal at cue time.
            for _ in 0..delay {
                bank.tick(&mut trace_network)
                    .map_err(|e| format!("{e:?}"))?;
                stale_network.step().map_err(|e| format!("{e:?}"))?;
            }
            let report = bank
                .reinforce(&mut trace_network, &labels)
                .map_err(|e| format!("{e:?}"))?;
            if report.age != delay || report.supervised_neurons != nodes {
                return Err("delayed feedback age or node count mismatch".into());
            }
            updated_trace[task_index] += report.updated_neurons;
            let stale_pairs = stale_network
                .capture_local_signals()
                .map_err(|e| format!("{e:?}"))?;
            updated_stale[task_index] += stale_network
                .apply_delayed_feedback(&stale_pairs, &labels)
                .map_err(|e| format!("{e:?}"))?;
        }
        phase_ms[task_index] = phase_begin.elapsed().as_millis();
        if task_index == 0 {
            // Exact A-head weights are frozen before subsequent B learning.
            saved_a = Some(trace_network.clone());
        }
    }
    // All TEST arms use exactly the same sample states, teacher, previously
    // seen-pair exclusion mask and cue-delay length. Evaluation is frozen.
    let mut frozen_b = create(nodes, seed);
    let mut saved_a = saved_a.ok_or("A model bank not captured")?;
    let testing = Instant::now();
    let bank_a = evaluate(&mut saved_a, &seen, seed, Task::A, delay)?;
    let before_b = evaluate(&mut saved_a, &seen, seed, Task::B, delay)?;
    let trace_a = evaluate(&mut trace_network, &seen, seed, Task::A, delay)?;
    let trace_b = evaluate(&mut trace_network, &seen, seed, Task::B, delay)?;
    let stale_a = evaluate(&mut stale_network, &seen, seed, Task::A, delay)?;
    let stale_b = evaluate(&mut stale_network, &seen, seed, Task::B, delay)?;
    let frozen_a = evaluate(&mut frozen_network, &seen, seed, Task::A, delay)?;
    let frozen_b = evaluate(&mut frozen_b, &seen, seed, Task::B, delay)?;
    let evaluate_ms = testing.elapsed().as_millis();

    if trace_network.wiring_fingerprint() != fingerprint
        || stale_network.wiring_fingerprint() != fingerprint
        || frozen_network.wiring_fingerprint() != fingerprint
        || trace_a.novel != bank_a.novel
        || trace_b.novel != before_b.novel
        || trace_a.novel != stale_a.novel
        || trace_a.novel != frozen_a.novel
        || trace_b.novel != stale_b.novel
        || trace_b.novel != frozen_b.novel
        || bank.cue_age().is_some()
        || updated_trace.iter().sum::<usize>() == 0
    {
        return Err("paired topology / temporal protocol invariant failed".into());
    }
    println!(
        "delayed_online nodes={nodes} delay={delay} episodes_per_task={TRAIN_EPISODES} test_episodes={EVAL_EPISODES} contacts={} cue_bytes={} build_ms={} train_a_ms={} train_b_ms={} evaluate_ms={} trace_updates_a={} trace_updates_b={} stale_updates_a={} stale_updates_b={} A_novel={} A_trace={}/{} A_saved={}/{} A_stale={}/{} A_frozen={}/{} B_before={}/{} B_novel={} B_trace={}/{} B_stale={}/{} B_frozen={}/{} A_autonomous={}/{} B_autonomous={}/{} wiring={:016x}",
        trace_network.contact_count(),
        nodes * 2,
        build_ms,
        phase_ms[0], phase_ms[1], evaluate_ms,
        updated_trace[0], updated_trace[1],
        updated_stale[0], updated_stale[1],
        trace_a.novel,
        trace_a.correct_cued, trace_a.novel,
        bank_a.correct_cued, bank_a.novel,
        stale_a.correct_cued, stale_a.novel,
        frozen_a.correct_cued, frozen_a.novel,
        before_b.correct_cued, before_b.novel,
        trace_b.novel,
        trace_b.correct_cued, trace_b.novel,
        stale_b.correct_cued, stale_b.novel,
        frozen_b.correct_cued, frozen_b.novel,
        trace_a.correct_autonomous, trace_a.novel,
        trace_b.correct_autonomous, trace_b.novel,
        fingerprint
    );
    // Report negative results without forcing "trace > stale" or perfect
    // recall. Success means validity of a real online training experiment.
    Ok(())
}

fn main() -> Result<(), String> {
    run(1_024, 32, 0x4512_0001)?;
    run(8_192, 8, 0x4512_0002)?;
    println!("FOURIER5_DELAYED_ONLINE_PLASTICITY_EXPERIMENT_PASS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_exclusion_is_exact_for_1024_values() {
        let mut seen = Seen::new(64);
        for input in 0..32 {
            for context in 0..32 {
                assert!(!seen.has(0, input, context));
                seen.mark(0, input, context);
                assert!(seen.has(0, input, context));
                assert!(!seen.has(1, input, context));
            }
        }
        for input in 0..32 {
            for context in 0..32 {
                assert!(seen.has(0, input, context));
            }
        }
    }

    #[test]
    fn teacher_tasks_are_distinct_for_context_sensitive_parity() {
        for node in 0..64 {
            let a = teacher_bit(43, node, Task::A);
            let b = teacher_bit(43, node, Task::B);
            assert_ne!(a, b);
            assert!(a < 4 && b < 4);
            assert!(teacher(43, node, 1, 0, Task::A));
            assert!(!teacher(43, node, 0, 0, Task::A));
            assert!(!teacher(43, node, 1, 1 << a, Task::A));
        }
    }

    #[test]
    fn test_stimulus_domain_differs_from_training_seed_domain() {
        let a = stimulus(257, 9, 7);
        let b = stimulus(257, 9, 7);
        let c = stimulus(257, 10, 7);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}

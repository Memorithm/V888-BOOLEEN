//! Reproducible bounded *retention* benchmark on the real graph.
//! Uses anatomy for connectivity, but labels are a user-defined artificial
//! supervised task; not behavior, cognition, or out-of-distribution learning.
use crate::connectome::Connectome;
use crate::plasticity::PlasticCircuit;
use crate::BooleanState;
use std::io::{self, ErrorKind};
use std::time::{Duration, Instant};

fn bad(s: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, s)
}

#[derive(Clone, Debug)]
pub struct RetentionReport {
    pub neurons: usize,
    pub edges: usize,
    pub positive_source_id: u64,
    pub negative_source_id: u64,
    pub target_id: u64,
    pub training_episodes: u64,
    pub tested_episodes: u64,
    pub positive_examples: u64,
    pub frozen_zero_correct: u64,
    pub frozen_topology_correct: u64,
    pub learned_correct: u64,
    pub learned_weight_bytes: usize,
    pub train_elapsed: Duration,
    pub frozen_zero_eval_elapsed: Duration,
    pub frozen_topology_eval_elapsed: Duration,
    pub learned_eval_elapsed: Duration,
}

/// Finds a target with two different presynaptic source neurons in the
/// observed graph, without inventing connections. Ignores zero-count entries.
fn find_pair(graph: &Connectome) -> io::Result<(usize, usize, usize)> {
    let mut first_source = vec![None::<usize>; graph.ids.len()];
    for source in 0..graph.ids.len() {
        for edge in graph.offsets[source]..graph.offsets[source + 1] {
            let edge = edge as usize;
            if graph.counts[edge] == 0 {
                continue;
            }
            let target = graph.targets[edge] as usize;
            if source == target {
                continue;
            }
            match first_source[target] {
                Some(other) if other != source => return Ok((other, source, target)),
                None => first_source[target] = Some(source),
                _ => {}
            }
        }
    }
    Err(bad("no target with two distinct real sources"))
}

pub fn benchmark(graph: &Connectome) -> io::Result<RetentionReport> {
    graph.validate()?;
    let (good, bad_source, target) = find_pair(graph)?;
    let mut learned = PlasticCircuit::new(graph, 0)?;
    let frozen_zero = learned.clone();
    let frozen_topology = PlasticCircuit::new(graph, 1)?;
    let mut good_input = BooleanState::new(graph.ids.len());
    good_input.set(good, true).map_err(|_| bad("good input"))?;
    let mut bad_input = BooleanState::new(graph.ids.len());
    bad_input
        .set(bad_source, true)
        .map_err(|_| bad("bad input"))?;
    let mut desired = BooleanState::new(graph.ids.len());
    desired.set(target, true).map_err(|_| bad("target"))?;
    let no_spike = BooleanState::new(graph.ids.len());

    const ROUNDS: u64 = 8;
    const TEST_EPISODES: u64 = 40;
    let start = Instant::now();
    for _ in 0..ROUNDS {
        learned.train(graph, &good_input, &desired)?;
        learned.train(graph, &bad_input, &no_spike)?;
    }
    let train_elapsed = start.elapsed();

    let score = |model: &PlasticCircuit| -> io::Result<(u64, Duration)> {
        let start = Instant::now();
        let mut correct = 0_u64;
        for i in 0..TEST_EPISODES {
            // Test events are held out in TIME but reuse the same input
            // patterns as training. This tests retention, NOT generalisation.
            let positive = i % 2 == 0;
            let inputs = if positive { &good_input } else { &bad_input };
            let prediction = model.predict(graph, inputs, 1)?;
            if prediction.get(target).map_err(|_| bad("target index"))? == positive {
                correct += 1;
            }
        }
        Ok((correct, start.elapsed()))
    };

    let (frozen_zero_correct, frozen_zero_eval_elapsed) = score(&frozen_zero)?;
    let (frozen_topology_correct, frozen_topology_eval_elapsed) = score(&frozen_topology)?;
    let (learned_correct, learned_eval_elapsed) = score(&learned)?;

    Ok(RetentionReport {
        neurons: graph.ids.len(),
        edges: graph.edges(),
        positive_source_id: graph.ids[good],
        negative_source_id: graph.ids[bad_source],
        target_id: graph.ids[target],
        training_episodes: 2 * ROUNDS,
        tested_episodes: TEST_EPISODES,
        positive_examples: TEST_EPISODES / 2,
        frozen_zero_correct,
        frozen_topology_correct,
        learned_correct,
        learned_weight_bytes: learned.weight_bytes(),
        train_elapsed,
        frozen_zero_eval_elapsed,
        frozen_topology_eval_elapsed,
        learned_eval_elapsed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controlled_retention_beats_two_frozen_equal_topology_controls() {
        let graph = Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 1, 2, 2],
            targets: vec![2, 2],
            counts: vec![1, 1],
        };
        let report = benchmark(&graph).unwrap();
        assert_eq!(report.training_episodes, 16);
        assert_eq!(report.tested_episodes, 40);
        assert_eq!(report.frozen_zero_correct, 20);
        assert_eq!(report.frozen_topology_correct, 20);
        assert_eq!(report.learned_correct, 40);
        assert_eq!(report.learned_weight_bytes, 1);
        assert_eq!(report.positive_source_id, 10);
        assert_eq!(report.target_id, 30);
    }

    #[test]
    fn no_qualifying_pair_is_rejected() {
        let graph = Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 1, 1, 1],
            targets: vec![2],
            counts: vec![1],
        };
        assert!(benchmark(&graph).is_err());
    }
}

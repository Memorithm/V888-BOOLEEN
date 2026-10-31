//! A disjoint-source generalization control on the *existing* V888 graph.
//! Synaptic anatomy is real; supervised binary labels are ARTIFICIAL.
//! This measures limitations of a local edge-weight lookup; never claim
//! learning of insect behaviors from these scores.
use crate::connectome::Connectome;
use crate::plasticity::PlasticCircuit;
use crate::BooleanState;
use std::io::{self, ErrorKind};
use std::time::{Duration, Instant};

fn invalid(s: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, s)
}

#[derive(Debug, Clone)]
pub struct HoldoutReport {
    pub neurons: usize,
    pub edges: usize,
    pub training_source_ids: [u64; 2],
    pub test_source_ids: [u64; 2],
    pub target_id: u64,
    pub train_episodes: u32,
    pub seen_test_episodes: u32,
    pub unseen_test_episodes: u32,
    pub zero_seen_correct: u32,
    pub zero_unseen_correct: u32,
    pub topology_seen_correct: u32,
    pub topology_unseen_correct: u32,
    pub learned_seen_correct: u32,
    pub learned_unseen_correct: u32,
    pub weight_bytes: usize,
    pub training_elapsed: Duration,
    pub evaluation_elapsed: Duration,
}

/// Find four DIFFERENT observed presynaptic sources for one postsynaptic
/// target. We use the first deterministic match in ascending source order.
fn find_four_sources(graph: &Connectome) -> io::Result<([usize; 4], usize)> {
    let mut sources = vec![[usize::MAX; 4]; graph.ids.len()];
    let mut cardinality = vec![0_u8; graph.ids.len()];
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
            let count = cardinality[target] as usize;
            if sources[target][..count].contains(&source) {
                continue;
            }
            if count < 4 {
                sources[target][count] = source;
                cardinality[target] += 1;
                if count == 3 {
                    return Ok((sources[target], target));
                }
            }
        }
    }
    Err(invalid("no target with four distinct observed sources"))
}

fn stimulus(neurons: usize, source: usize) -> io::Result<BooleanState> {
    let mut state = BooleanState::new(neurons);
    state.set(source, true).map_err(|_| invalid("invalid source"))?;
    Ok(state)
}

fn score(
    model: &PlasticCircuit,
    graph: &Connectome,
    source_positive: &BooleanState,
    source_negative: &BooleanState,
    target: usize,
) -> io::Result<u32> {
    const PRESENTATIONS: u32 = 40;
    let mut correct = 0_u32;
    for example in 0..PRESENTATIONS {
        let positive = example & 1 == 0;
        let input = if positive { source_positive } else { source_negative };
        if model
            .predict(graph, input, 1)?
            .get(target)
            .map_err(|_| invalid("invalid target"))?
            == positive
        {
            correct += 1;
        }
    }
    Ok(correct)
}

/// 2 trained source neurons, 2 NEVER-TRAINED source neurons, same target.
/// These held-out sources have existing anatomical edges, but no learned
/// coefficients. Therefore this local plasticity rule is expected to FAIL
/// at transferring the positive association to the held-out source.
/// Labels are balanced to prevent misleading majority-class accuracy.
pub fn benchmark_disjoint_sources(graph: &Connectome) -> io::Result<HoldoutReport> {
    graph.validate()?;
    let (sources, target) = find_four_sources(graph)?;
    if sources.iter().any(|&s| s == usize::MAX) {
        return Err(invalid("source selection incomplete"));
    }
    let input: Vec<BooleanState> = sources
        .iter()
        .map(|&index| stimulus(graph.ids.len(), index))
        .collect::<io::Result<Vec<_>>>()?;
    let mut taught_positive = BooleanState::new(graph.ids.len());
    taught_positive
        .set(target, true)
        .map_err(|_| invalid("teacher index"))?;
    let taught_negative = BooleanState::new(graph.ids.len());
    let mut learned = PlasticCircuit::new(graph, 0)?;
    let frozen_zero = learned.clone();
    let frozen_topology = PlasticCircuit::new(graph, 1)?;

    let start = Instant::now();
    for _ in 0..8 {
        learned.train(graph, &input[0], &taught_positive)?;
        learned.train(graph, &input[1], &taught_negative)?;
    }
    let training_elapsed = start.elapsed();
    let evaluation = Instant::now();
    let zero_seen = score(&frozen_zero, graph, &input[0], &input[1], target)?;
    let zero_unseen = score(&frozen_zero, graph, &input[2], &input[3], target)?;
    let topology_seen = score(&frozen_topology, graph, &input[0], &input[1], target)?;
    let topology_unseen = score(&frozen_topology, graph, &input[2], &input[3], target)?;
    let learned_seen = score(&learned, graph, &input[0], &input[1], target)?;
    let learned_unseen = score(&learned, graph, &input[2], &input[3], target)?;
    let evaluation_elapsed = evaluation.elapsed();
    Ok(HoldoutReport {
        neurons: graph.ids.len(),
        edges: graph.edges(),
        training_source_ids: [graph.ids[sources[0]], graph.ids[sources[1]]],
        test_source_ids: [graph.ids[sources[2]], graph.ids[sources[3]]],
        target_id: graph.ids[target],
        train_episodes: 16,
        seen_test_episodes: 40,
        unseen_test_episodes: 40,
        zero_seen_correct: zero_seen,
        zero_unseen_correct: zero_unseen,
        topology_seen_correct: topology_seen,
        topology_unseen_correct: topology_unseen,
        learned_seen_correct: learned_seen,
        learned_unseen_correct: learned_unseen,
        weight_bytes: learned.weight_bytes(),
        training_elapsed,
        evaluation_elapsed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holdout_reveals_absent_transfer_to_untrained_sources() {
        let graph = Connectome {
            ids: vec![10, 20, 30, 40, 50],
            offsets: vec![0, 1, 2, 3, 4, 4],
            targets: vec![4, 4, 4, 4],
            counts: vec![1, 1, 1, 1],
        };
        let r = benchmark_disjoint_sources(&graph).unwrap();
        assert_eq!(r.training_source_ids, [10, 20]);
        assert_eq!(r.test_source_ids, [30, 40]);
        assert_eq!(r.target_id, 50);
        assert_eq!(r.learned_seen_correct, 40);
        assert_eq!(r.learned_unseen_correct, 20);
        assert_eq!(r.zero_seen_correct, 20);
        assert_eq!(r.zero_unseen_correct, 20);
        assert_eq!(r.topology_seen_correct, 20);
        assert_eq!(r.topology_unseen_correct, 20);
        assert_eq!(r.weight_bytes, 2);
    }

    #[test]
    fn refuse_without_four_distinct_presynaptic_neurons() {
        let graph = Connectome {
            ids: vec![10, 20, 30, 40, 50],
            offsets: vec![0, 1, 2, 3, 3, 3],
            targets: vec![4, 4, 4],
            counts: vec![1, 1, 1],
        };
        assert!(benchmark_disjoint_sources(&graph).is_err());
    }

    #[test]
    fn repeated_contact_from_same_source_not_counted_twice() {
        let graph = Connectome {
            ids: vec![10, 20, 30, 40, 50],
            offsets: vec![0, 2, 3, 4, 4, 4],
            targets: vec![4, 4, 4, 4],
            counts: vec![1, 1, 1, 1],
        };
        assert!(benchmark_disjoint_sources(&graph).is_err());
    }
}

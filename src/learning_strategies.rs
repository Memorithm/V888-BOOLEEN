//! Experimental learning-strategy hypotheses on a frozen directed graph.
//!
//! Strategies update four-bit coefficients on *existing* edges only. They are
//! not models of biological plasticity, insect memory, or BANC physiology.
//! BANC V888 attributions remain those declared in `docs/SOURCES.md`.
use crate::connectome::Connectome;
use crate::plasticity::PlasticCircuit;
use crate::BooleanState;
use std::io::{self, ErrorKind};
use std::time::{Duration, Instant};

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Executable update rules. `Supervised` is the existing teacher-driven
/// control (`PlasticCircuit::train`). The other two are additional hypotheses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LearningStrategy {
    Supervised,
    AssociativeCoactivation,
    RewardModulated,
}

impl LearningStrategy {
    #[must_use]
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Supervised => "supervised",
            Self::AssociativeCoactivation => "associative-coactivation",
            Self::RewardModulated => "reward-modulated",
        }
    }

    pub fn parse(value: &str) -> io::Result<Self> {
        match value {
            "supervised" => Ok(Self::Supervised),
            "associative-coactivation" | "associative" => Ok(Self::AssociativeCoactivation),
            "reward-modulated" | "reward" => Ok(Self::RewardModulated),
            _ => Err(bad("unknown learning strategy")),
        }
    }
}

/// Explicit external reward. `None` leaves every weight unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewardSignal {
    None,
    Positive,
    Negative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StrategyConfig {
    pub strategy: LearningStrategy,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct UpdateReport {
    pub edges_visited: u64,
    pub weights_changed: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrategyOutcome {
    pub strategy: LearningStrategy,
    pub seed: u64,
    pub train_pos_id: u64,
    pub train_neg_id: u64,
    pub holdout_pos_id: u64,
    pub holdout_neg_id: u64,
    pub target_id: u64,
    pub train_episodes: u64,
    pub reversal_episodes: u64,
    pub seen_total: u64,
    pub unseen_total: u64,
    pub seen_correct_before: u64,
    pub seen_correct_after: u64,
    pub unseen_correct_after: u64,
    pub seen_correct_after_reversal: u64,
    pub reversal_task_correct: u64,
    pub edges_visited: u64,
    pub weights_changed: u64,
    pub train_elapsed: Duration,
    pub eval_elapsed: Duration,
    pub weight_bytes: usize,
    pub unpacked_weight_bytes: usize,
    pub topology_unchanged: bool,
}

fn add_count(total: u64, increment: u64) -> io::Result<u64> {
    total
        .checked_add(increment)
        .ok_or_else(|| bad("update counter overflow"))
}

/// Apply one strategy step to existing edges only. `observed_or_teacher` is
/// the teacher bitmap for `Supervised` and the observed activity bitmap for
/// the other two rules. `reward` is ignored except by `RewardModulated`.
pub fn apply_update(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    strategy: LearningStrategy,
    input: &BooleanState,
    observed_or_teacher: &BooleanState,
    reward: RewardSignal,
) -> io::Result<UpdateReport> {
    graph.validate()?;
    if circuit.weights().len() != graph.edges()
        || input.len() != graph.ids.len()
        || observed_or_teacher.len() != graph.ids.len()
    {
        return Err(bad("learning strategy shape mismatch"));
    }
    if strategy == LearningStrategy::RewardModulated && reward == RewardSignal::None {
        return Ok(UpdateReport::default());
    }

    let mut report = UpdateReport::default();
    for source in input.active_indices() {
        if source >= graph.ids.len() {
            return Err(bad("active source out of range"));
        }
        for edge in graph.offsets[source]..graph.offsets[source + 1] {
            let edge = edge as usize;
            let target = graph.targets[edge] as usize;
            let post = observed_or_teacher
                .get(target)
                .map_err(|_| bad("invalid target index"))?;
            let reinforce = match strategy {
                LearningStrategy::Supervised | LearningStrategy::AssociativeCoactivation => post,
                LearningStrategy::RewardModulated => {
                    if !post {
                        report.edges_visited = add_count(report.edges_visited, 1)?;
                        continue;
                    }
                    match reward {
                        RewardSignal::Positive => true,
                        RewardSignal::Negative => false,
                        RewardSignal::None => unreachable!("reward none returns earlier"),
                    }
                }
            };
            report.edges_visited = add_count(report.edges_visited, 1)?;
            if circuit.apply_edge_step(edge, reinforce)? {
                report.weights_changed = add_count(report.weights_changed, 1)?;
            }
        }
    }
    Ok(report)
}

fn stimulus(neurons: usize, source: usize) -> io::Result<BooleanState> {
    let mut state = BooleanState::new(neurons);
    state
        .set(source, true)
        .map_err(|_| bad("invalid source index"))?;
    Ok(state)
}

fn observed_pair(
    neurons: usize,
    source: usize,
    target: usize,
    include_target: bool,
) -> io::Result<BooleanState> {
    let mut state = stimulus(neurons, source)?;
    if include_target {
        state
            .set(target, true)
            .map_err(|_| bad("invalid target index"))?;
    }
    Ok(state)
}

fn teacher_for(neurons: usize, target: usize, fire: bool) -> io::Result<BooleanState> {
    let mut state = BooleanState::new(neurons);
    if fire {
        state
            .set(target, true)
            .map_err(|_| bad("invalid target index"))?;
    }
    Ok(state)
}

/// Deterministic LCG shuffle so a seed reproduces the episode order exactly.
fn shuffle_with_seed<T>(items: &mut [T], seed: u64) {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    for i in (1..items.len()).rev() {
        state = state
            .wrapping_mul(0x5851_F42D_4C95_7F2D)
            .wrapping_add(0x1405_7B7E_F767_814F);
        let j = (state as usize) % (i + 1);
        items.swap(i, j);
    }
}

fn apply_labeled_episode(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    strategy: LearningStrategy,
    source: usize,
    target: usize,
    fire: bool,
) -> io::Result<UpdateReport> {
    let neurons = graph.ids.len();
    match strategy {
        LearningStrategy::Supervised => {
            let input = stimulus(neurons, source)?;
            let teacher = teacher_for(neurons, target, fire)?;
            apply_update(
                circuit,
                graph,
                strategy,
                &input,
                &teacher,
                RewardSignal::None,
            )
        }
        LearningStrategy::AssociativeCoactivation => {
            let observed = observed_pair(neurons, source, target, fire)?;
            apply_update(
                circuit,
                graph,
                strategy,
                &observed,
                &observed,
                RewardSignal::None,
            )
        }
        LearningStrategy::RewardModulated => {
            let observed = observed_pair(neurons, source, target, true)?;
            let reward = if fire {
                RewardSignal::Positive
            } else {
                RewardSignal::Negative
            };
            apply_update(circuit, graph, strategy, &observed, &observed, reward)
        }
    }
}

fn accumulate(total: &mut UpdateReport, step: UpdateReport) -> io::Result<()> {
    total.edges_visited = add_count(total.edges_visited, step.edges_visited)?;
    total.weights_changed = add_count(total.weights_changed, step.weights_changed)?;
    Ok(())
}

fn score_pair(
    model: &PlasticCircuit,
    graph: &Connectome,
    positive: &BooleanState,
    negative: &BooleanState,
    target: usize,
    presentations: u64,
) -> io::Result<u64> {
    let mut correct = 0_u64;
    for example in 0..presentations {
        let fire = example % 2 == 0;
        let input = if fire { positive } else { negative };
        if model
            .predict(graph, input, 1)?
            .get(target)
            .map_err(|_| bad("invalid target index"))?
            == fire
        {
            correct += 1;
        }
    }
    Ok(correct)
}

/// First target that already has four distinct observed presynaptic sources.
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
    Err(bad("no target with four distinct observed sources"))
}

/// Synthetic graph used when no connectome file is supplied: two training
/// sources, two held-out sources, one shared target, plus a distractor edge.
#[must_use]
pub fn comparison_graph() -> Connectome {
    Connectome {
        ids: vec![10, 20, 30, 40, 50, 60],
        offsets: vec![0, 2, 3, 3, 4, 5, 5],
        targets: vec![2, 5, 2, 2, 2],
        counts: vec![1, 1, 1, 1, 1],
    }
}

#[derive(Clone, Copy)]
struct PhaseSpec {
    strategy: LearningStrategy,
    seed: u64,
    rounds: u32,
    pos: usize,
    neg: usize,
    target: usize,
    reversed: bool,
}

fn run_phase(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    spec: PhaseSpec,
) -> io::Result<UpdateReport> {
    let mut episodes = Vec::with_capacity((spec.rounds as usize) * 2);
    for _ in 0..spec.rounds {
        episodes.push((spec.pos, !spec.reversed));
        episodes.push((spec.neg, spec.reversed));
    }
    shuffle_with_seed(&mut episodes, spec.seed);
    let mut total = UpdateReport::default();
    for (source, fire) in episodes {
        accumulate(
            &mut total,
            apply_labeled_episode(circuit, graph, spec.strategy, source, spec.target, fire)?,
        )?;
    }
    Ok(total)
}

/// Equal-budget comparison of the three strategies on one frozen graph.
/// Forgetting is measured by reversing the trained labels; generalization is
/// measured on two never-activated sources that already synapse onto the
/// same target. Labels are artificial.
pub fn compare_strategies(
    graph: Option<&Connectome>,
    seed: u64,
    rounds: u32,
) -> io::Result<Vec<StrategyOutcome>> {
    if rounds == 0 || rounds > 64 {
        return Err(bad("rounds must be in 1..=64"));
    }
    let owned = comparison_graph();
    let graph = match graph {
        Some(graph) => {
            graph.validate()?;
            graph
        }
        None => {
            owned.validate()?;
            &owned
        }
    };
    let original = graph.clone();
    let (sources, target) = find_four_sources(graph)?;
    let [train_pos, train_neg, holdout_pos, holdout_neg] = sources;
    let neurons = graph.ids.len();
    let train_pos_in = stimulus(neurons, train_pos)?;
    let train_neg_in = stimulus(neurons, train_neg)?;
    let holdout_pos_in = stimulus(neurons, holdout_pos)?;
    let holdout_neg_in = stimulus(neurons, holdout_neg)?;
    const PRESENTATIONS: u64 = 40;
    let mut outcomes = Vec::new();
    for strategy in [
        LearningStrategy::Supervised,
        LearningStrategy::AssociativeCoactivation,
        LearningStrategy::RewardModulated,
    ] {
        let mut circuit = PlasticCircuit::new(graph, 0)?;
        let seen_before = score_pair(
            &circuit,
            graph,
            &train_pos_in,
            &train_neg_in,
            target,
            PRESENTATIONS,
        )?;
        let train_clock = Instant::now();
        let mut totals = run_phase(
            &mut circuit,
            graph,
            PhaseSpec {
                strategy,
                seed,
                rounds,
                pos: train_pos,
                neg: train_neg,
                target,
                reversed: false,
            },
        )?;
        let mut train_elapsed = train_clock.elapsed();
        let eval_clock = Instant::now();
        let seen_after = score_pair(
            &circuit,
            graph,
            &train_pos_in,
            &train_neg_in,
            target,
            PRESENTATIONS,
        )?;
        let unseen_after = score_pair(
            &circuit,
            graph,
            &holdout_pos_in,
            &holdout_neg_in,
            target,
            PRESENTATIONS,
        )?;
        let mut eval_elapsed = eval_clock.elapsed();
        let reversal_clock = Instant::now();
        let reversal = run_phase(
            &mut circuit,
            graph,
            PhaseSpec {
                strategy,
                seed: seed ^ 0xA5A5_A5A5_A5A5_A5A5,
                rounds,
                pos: train_pos,
                neg: train_neg,
                target,
                reversed: true,
            },
        )?;
        accumulate(&mut totals, reversal)?;
        train_elapsed += reversal_clock.elapsed();
        let eval_clock = Instant::now();
        let seen_after_reversal = score_pair(
            &circuit,
            graph,
            &train_pos_in,
            &train_neg_in,
            target,
            PRESENTATIONS,
        )?;
        let reversal_task = score_pair(
            &circuit,
            graph,
            &train_neg_in,
            &train_pos_in,
            target,
            PRESENTATIONS,
        )?;
        eval_elapsed += eval_clock.elapsed();
        outcomes.push(StrategyOutcome {
            strategy,
            seed,
            train_pos_id: graph.ids[train_pos],
            train_neg_id: graph.ids[train_neg],
            holdout_pos_id: graph.ids[holdout_pos],
            holdout_neg_id: graph.ids[holdout_neg],
            target_id: graph.ids[target],
            train_episodes: u64::from(rounds) * 2,
            reversal_episodes: u64::from(rounds) * 2,
            seen_total: PRESENTATIONS,
            unseen_total: PRESENTATIONS,
            seen_correct_before: seen_before,
            seen_correct_after: seen_after,
            unseen_correct_after: unseen_after,
            seen_correct_after_reversal: seen_after_reversal,
            reversal_task_correct: reversal_task,
            edges_visited: totals.edges_visited,
            weights_changed: totals.weights_changed,
            train_elapsed,
            eval_elapsed,
            weight_bytes: circuit.weight_bytes(),
            unpacked_weight_bytes: circuit.weights().len(),
            topology_unchanged: *graph == original,
        });
    }
    if graph != &original {
        return Err(bad("connectome anatomy was modified"));
    }
    Ok(outcomes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture() -> Connectome {
        comparison_graph()
    }

    #[test]
    fn supervised_update_matches_existing_train() -> io::Result<()> {
        let graph = fixture();
        let mut via_strategy = PlasticCircuit::new(&graph, 0)?;
        let mut via_train = via_strategy.clone();
        let input = stimulus(graph.ids.len(), 0)?;
        let teacher = teacher_for(graph.ids.len(), 2, true)?;
        let report = apply_update(
            &mut via_strategy,
            &graph,
            LearningStrategy::Supervised,
            &input,
            &teacher,
            RewardSignal::None,
        )?;
        let visited = via_train.train(&graph, &input, &teacher)?;
        assert_eq!(report.edges_visited, visited);
        assert_eq!(via_strategy, via_train);
        assert_eq!(via_strategy.weights()[0], 1);
        assert_eq!(via_strategy.weights()[1], 0);
        Ok(())
    }

    #[test]
    fn each_strategy_changes_weights_as_specified() -> io::Result<()> {
        let graph = fixture();
        let observed = observed_pair(graph.ids.len(), 0, 2, true)?;
        let teacher = teacher_for(graph.ids.len(), 2, true)?;
        let input = stimulus(graph.ids.len(), 0)?;

        let mut supervised = PlasticCircuit::new(&graph, 8)?;
        apply_update(
            &mut supervised,
            &graph,
            LearningStrategy::Supervised,
            &input,
            &teacher,
            RewardSignal::None,
        )?;
        assert_eq!(supervised.weights()[0], 9);
        assert_eq!(supervised.weights()[1], 7);

        let mut associative = PlasticCircuit::new(&graph, 8)?;
        apply_update(
            &mut associative,
            &graph,
            LearningStrategy::AssociativeCoactivation,
            &observed,
            &observed,
            RewardSignal::None,
        )?;
        assert_eq!(associative.weights()[0], 9);
        assert_eq!(associative.weights()[1], 7);

        let mut rewarded = PlasticCircuit::new(&graph, 8)?;
        apply_update(
            &mut rewarded,
            &graph,
            LearningStrategy::RewardModulated,
            &observed,
            &observed,
            RewardSignal::Positive,
        )?;
        assert_eq!(rewarded.weights()[0], 9);
        assert_eq!(rewarded.weights()[1], 8);

        let mut punished = PlasticCircuit::new(&graph, 8)?;
        apply_update(
            &mut punished,
            &graph,
            LearningStrategy::RewardModulated,
            &observed,
            &observed,
            RewardSignal::Negative,
        )?;
        assert_eq!(punished.weights()[0], 7);
        assert_eq!(punished.weights()[1], 8);

        let mut silent = PlasticCircuit::new(&graph, 8)?;
        let before = silent.weights().to_vec();
        let noop = apply_update(
            &mut silent,
            &graph,
            LearningStrategy::RewardModulated,
            &observed,
            &observed,
            RewardSignal::None,
        )?;
        assert_eq!(noop, UpdateReport::default());
        assert_eq!(silent.weights(), before.as_slice());
        Ok(())
    }

    #[test]
    fn identical_seeds_are_bit_exact_and_checkpoints_resume() -> io::Result<()> {
        let graph = fixture();
        let mut first = PlasticCircuit::new(&graph, 0)?;
        let mut second = first.clone();
        for circuit in [&mut first, &mut second] {
            run_phase(
                circuit,
                &graph,
                PhaseSpec {
                    strategy: LearningStrategy::AssociativeCoactivation,
                    seed: 7,
                    rounds: 8,
                    pos: 0,
                    neg: 1,
                    target: 2,
                    reversed: false,
                },
            )?;
        }
        assert_eq!(first, second);

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let file = std::env::temp_dir().join(format!(
            "v888-b8884-{}-{unique}.v888pls",
            std::process::id()
        ));
        let mut continuous = PlasticCircuit::new(&graph, 0)?;
        run_phase(
            &mut continuous,
            &graph,
            PhaseSpec {
                strategy: LearningStrategy::RewardModulated,
                seed: 11,
                rounds: 4,
                pos: 0,
                neg: 1,
                target: 2,
                reversed: false,
            },
        )?;
        continuous.save(&file)?;
        let mut resumed = PlasticCircuit::load(&file, &graph)?;
        assert_eq!(resumed, continuous);
        let resume_spec = PhaseSpec {
            strategy: LearningStrategy::RewardModulated,
            seed: 13,
            rounds: 4,
            pos: 0,
            neg: 1,
            target: 2,
            reversed: false,
        };
        run_phase(&mut continuous, &graph, resume_spec)?;
        run_phase(&mut resumed, &graph, resume_spec)?;
        assert_eq!(resumed, continuous);
        std::fs::remove_file(file)?;
        Ok(())
    }

    #[test]
    fn strategies_never_change_anatomy() -> io::Result<()> {
        let graph = fixture();
        let original = graph.clone();
        let fingerprint = PlasticCircuit::new(&graph, 0)?.topology_fingerprint();
        let _ = compare_strategies(Some(&graph), 1, 8)?;
        assert_eq!(graph, original);
        assert_eq!(
            PlasticCircuit::new(&graph, 0)?.topology_fingerprint(),
            fingerprint
        );
        Ok(())
    }

    #[test]
    fn equal_budget_comparison_shows_retention_forgetting_and_failed_transfer() -> io::Result<()> {
        let outcomes = compare_strategies(None, 1, 8)?;
        assert_eq!(outcomes.len(), 3);
        for outcome in &outcomes {
            assert!(outcome.topology_unchanged);
            assert_eq!(outcome.seen_total, 40);
            assert_eq!(outcome.unseen_total, 40);
            assert_eq!(outcome.train_episodes, 16);
            assert_eq!(outcome.reversal_episodes, 16);
            assert_eq!(outcome.seen_correct_before, 20);
            assert_eq!(outcome.seen_correct_after, 40);
            // Held-out sources share the target anatomically but never receive
            // an update. Transfer is expected to fail (chance = 20/40).
            assert_eq!(outcome.unseen_correct_after, 20);
            assert_eq!(outcome.seen_correct_after_reversal, 0);
            assert_eq!(outcome.reversal_task_correct, 40);
            assert!(outcome.weights_changed > 0);
            assert_eq!(outcome.weight_bytes, 3);
            assert_eq!(outcome.unpacked_weight_bytes, 5);
        }
        Ok(())
    }
}

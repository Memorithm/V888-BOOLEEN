//! B888-4: two forgetting-reduction mechanisms — #25-style **consolidation**
//! (protected edges accept 1 of `PROTECT_PERIOD` update attempts during task
//! B) and **bounded rehearsal** (1 of `REHEARSAL_PERIOD` B episodes replaced
//! by a task-A training episode, equal budget) — applied to the supervised
//! control and to the #38 `group-8` / `per-target` graded-reward learners.
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph; nothing here is
//! a claim about sleep, replay, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/CONSOLIDATED_GRADED.md`
//! (committed before any run). Only weights of existing edges change.
use crate::connectome::Connectome;
use crate::generalization_bench::{
    self as gb, build_task, encode, episode_order, evaluate, Confusion, GenSpec, GraphShape, Spread,
};
use crate::graded_reward::{train_graded_stream, Grain};
use crate::learning_strategies::{apply_update_gated, LearningStrategy, RewardSignal};
use crate::plasticity::PlasticCircuit;
use crate::reward_limits as rl;
use std::io::{self, ErrorKind};

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Protected edges accept one of this many update attempts during B.
pub const PROTECT_PERIOD: u32 = 4;
/// One B episode in this many is replaced by an A rehearsal episode.
pub const REHEARSAL_PERIOD: usize = 8;
/// Salt of the deterministic A-train rehearsal order.
pub const REHEARSAL_SALT: u64 = 0x5EAE_A55E;
const B_SALT: u64 = 0xB0B0_B0B0;

/// Learning rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Learner {
    Control,
    Graded(Grain),
}

impl Learner {
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::Control => "supervised-matched".to_string(),
            Self::Graded(g) => format!("trace-{}", g.label()),
        }
    }
}

pub const LEARNERS: [Learner; 3] = [
    Learner::Control,
    Learner::Graded(Grain::Groups(8)),
    Learner::Graded(Grain::PerTarget),
];

/// Forgetting-reduction mechanism (acts only during task B).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mechanism {
    None,
    Consolidate,
    Rehearse,
    Both,
}

impl Mechanism {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Consolidate => "consolidate",
            Self::Rehearse => "rehearse",
            Self::Both => "both",
        }
    }
    fn consolidates(self) -> bool {
        matches!(self, Self::Consolidate | Self::Both)
    }
    fn rehearses(self) -> bool {
        matches!(self, Self::Rehearse | Self::Both)
    }
}

pub const MECHANISMS: [Mechanism; 4] = [
    Mechanism::None,
    Mechanism::Consolidate,
    Mechanism::Rehearse,
    Mechanism::Both,
];

/// Per-phase instrumentation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseCost {
    pub episodes: u64,
    pub rehearsal_episodes: u64,
    pub attempts: u64,
    pub weights_changed: u64,
    pub protected_attempts: u64,
    pub blocked: u64,
}

/// One (learner, mechanism, seed) measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub learner: Learner,
    pub mechanism: Mechanism,
    pub seed: u64,
    pub a_train_after_a: Confusion,
    pub a_test_after_a: Confusion,
    pub a_test_after_b: Confusion,
    pub b_test_after_b: Confusion,
    pub protected_edges: u64,
    pub cost_a: PhaseCost,
    pub cost_b: PhaseCost,
    pub topology_unchanged: bool,
}

impl Row {
    #[must_use]
    pub fn forgetting(&self) -> f64 {
        self.a_test_after_a.balanced() - self.a_test_after_b.balanced()
    }
}

pub(crate) type Item<'a> = (&'a [usize], &'a [bool]);

/// B-phase stream: the #38 B order with every `REHEARSAL_PERIOD`-th slot
/// optionally replaced by the next A-train episode of a fixed shuffle.
pub(crate) fn b_stream<'a>(
    task: &'a gb::Task,
    spec: &GenSpec,
    seed: u64,
    rehearse: bool,
) -> Vec<Item<'a>> {
    let b = episode_order(task.b_train.len(), spec.rounds_b, seed ^ B_SALT);
    let a = episode_order(task.a_train.len(), spec.rounds_a, seed ^ REHEARSAL_SALT);
    let mut next_a = a.into_iter().cycle();
    b.into_iter()
        .enumerate()
        .map(|(k, idx)| {
            if rehearse && k % REHEARSAL_PERIOD == REHEARSAL_PERIOD - 1 {
                let i = next_a.next().unwrap_or(0);
                (task.a_train[i].as_slice(), task.valence_a.as_slice())
            } else {
                (task.b_train[idx].as_slice(), task.valence_b.as_slice())
            }
        })
        .collect()
}

pub(crate) fn train(
    learner: Learner,
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    items: &[Item<'_>],
    allow: &mut dyn FnMut(usize) -> bool,
) -> io::Result<PhaseCost> {
    let mut cost = PhaseCost {
        episodes: items.len() as u64,
        ..PhaseCost::default()
    };
    match learner {
        Learner::Control => {
            for &(pattern, valence) in items {
                let (input, _, teacher) = encode(graph, pattern, valence)?;
                let r = apply_update_gated(
                    circuit,
                    graph,
                    LearningStrategy::Supervised,
                    &input,
                    &teacher,
                    RewardSignal::None,
                    allow,
                )?;
                cost.attempts += r.edges_visited;
                cost.weights_changed += r.weights_changed;
            }
        }
        Learner::Graded(grain) => {
            let c = train_graded_stream(circuit, graph, items, grain, 0, false, allow)?;
            cost.attempts = c.trace.update_calls;
            cost.weights_changed = c.trace.weights_changed;
        }
    }
    Ok(cost)
}

/// Run A then B for one (learner, mechanism).
pub fn run_arm(
    graph: &Connectome,
    task: &gb::Task,
    spec: &GenSpec,
    learner: Learner,
    mechanism: Mechanism,
    seed: u64,
) -> io::Result<Row> {
    let original = graph.clone();
    let mut circuit = PlasticCircuit::new(graph, spec.init_weight)?;
    let fingerprint = circuit.topology_fingerprint();
    let a_items: Vec<Item<'_>> = episode_order(task.a_train.len(), spec.rounds_a, seed)
        .into_iter()
        .map(|i| (task.a_train[i].as_slice(), task.valence_a.as_slice()))
        .collect();
    let cost_a = train(learner, &mut circuit, graph, &a_items, &mut |_| true)?;
    let a_train_after_a = evaluate(&circuit, graph, &task.a_train, &task.valence_a)?;
    let a_test_after_a = evaluate(&circuit, graph, &task.test, &task.valence_a)?;
    let protected: Vec<bool> = if mechanism.consolidates() {
        circuit
            .weights()
            .iter()
            .map(|&w| w != spec.init_weight)
            .collect()
    } else {
        vec![false; graph.edges()]
    };
    let protected_edges = protected.iter().filter(|&&p| p).count() as u64;
    let b_items = b_stream(task, spec, seed, mechanism.rehearses());
    let rehearsal_episodes = if mechanism.rehearses() {
        (b_items.len() / REHEARSAL_PERIOD) as u64
    } else {
        0
    };
    let mut attempts = vec![0_u32; graph.edges()];
    let (mut protected_attempts, mut blocked) = (0_u64, 0_u64);
    let mut cost_b = train(learner, &mut circuit, graph, &b_items, &mut |edge| {
        if !protected[edge] {
            return true;
        }
        protected_attempts += 1;
        attempts[edge] = attempts[edge].saturating_add(1);
        if attempts[edge] % PROTECT_PERIOD == 0 {
            true
        } else {
            blocked += 1;
            false
        }
    })?;
    cost_b.rehearsal_episodes = rehearsal_episodes;
    cost_b.protected_attempts = protected_attempts;
    cost_b.blocked = blocked;
    let a_test_after_b = evaluate(&circuit, graph, &task.test, &task.valence_a)?;
    let b_test_after_b = evaluate(&circuit, graph, &task.test, &task.valence_b)?;
    Ok(Row {
        learner,
        mechanism,
        seed,
        a_train_after_a,
        a_test_after_a,
        a_test_after_b,
        b_test_after_b,
        protected_edges,
        cost_a,
        cost_b,
        topology_unchanged: *graph == original
            && circuit.topology_fingerprint() == fingerprint
            && circuit.weights().len() == graph.edges(),
    })
}

/// Grid: seed × learner × mechanism (12 rows per seed).
pub fn run_grid(
    graph: Option<&Connectome>,
    shape: &GraphShape,
    spec: &GenSpec,
    seeds: &[u64],
) -> io::Result<Vec<Row>> {
    if seeds.is_empty() {
        return Err(bad("need at least one seed"));
    }
    let mut rows = Vec::new();
    for &seed in seeds {
        let g = rl::graph_for(graph, shape, seed)?;
        let task = build_task(&g, spec, seed)?;
        for learner in LEARNERS {
            for mechanism in MECHANISMS {
                rows.push(run_arm(&g, &task, spec, learner, mechanism, seed)?);
            }
        }
    }
    Ok(rows)
}

pub const CSV_HEADER: &str = "schema,graph,learner,mechanism,seed,a_train_bal_after_a,a_test_bal_after_a,a_test_bal_after_b,b_test_bal_after_b,forgetting,protected_edges,a_episodes,a_attempts,a_weights_changed,b_episodes,b_rehearsal_episodes,b_attempts,b_weights_changed,b_protected_attempts,b_blocked,topology_unchanged";

/// Deterministic CSV (no timing column).
#[must_use]
pub fn to_csv(rows: &[Row], graph_label: &str) -> String {
    let mut out = String::from(CSV_HEADER);
    out.push('\n');
    for r in rows {
        out.push_str(&format!(
            "b888-4-consolidated-graded-v1,{graph_label},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{},{},{},{},{},{}\n",
            r.learner.label(),
            r.mechanism.label(),
            r.seed,
            r.a_train_after_a.balanced(),
            r.a_test_after_a.balanced(),
            r.a_test_after_b.balanced(),
            r.b_test_after_b.balanced(),
            r.forgetting(),
            r.protected_edges,
            r.cost_a.episodes,
            r.cost_a.attempts,
            r.cost_a.weights_changed,
            r.cost_b.episodes,
            r.cost_b.rehearsal_episodes,
            r.cost_b.attempts,
            r.cost_b.weights_changed,
            r.cost_b.protected_attempts,
            r.cost_b.blocked,
            r.topology_unchanged,
        ));
    }
    out
}

/// Paired comparison of x against y (same seeds).
#[derive(Clone, Debug, PartialEq)]
pub struct Comparison {
    pub x: (Learner, Mechanism),
    pub y: (Learner, Mechanism),
    pub n: usize,
    /// x − y on A test after A.
    pub a_test: Spread,
    /// y forgetting − x forgetting (+ = x forgets less).
    pub forget: gb::Paired,
    /// x − y on A test after B (retained A).
    pub retained: Spread,
    /// x − y on B test after B.
    pub b: Spread,
}

impl Comparison {
    /// Pre-registered rule: forgetting reduction ≥ margin in ≥ 8/10 seeds,
    /// A-test and B-test mean drop ≤ margin.
    #[must_use]
    pub fn passes(&self) -> bool {
        self.n > 0
            && self.forget.delta.mean >= gb::MARGIN
            && (self.forget.better as f64) >= 0.8 * self.n as f64
            && self.a_test.mean > -gb::MARGIN
            && self.b.mean > -gb::MARGIN
    }
}

#[must_use]
pub fn compare(rows: &[Row], x: (Learner, Mechanism), y: (Learner, Mechanism)) -> Comparison {
    let (mut da, mut df, mut dr, mut db) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for rx in rows
        .iter()
        .filter(|r| r.learner == x.0 && r.mechanism == x.1)
    {
        if let Some(ry) = rows
            .iter()
            .find(|r| r.learner == y.0 && r.mechanism == y.1 && r.seed == rx.seed)
        {
            da.push(rx.a_test_after_a.balanced() - ry.a_test_after_a.balanced());
            df.push(ry.forgetting() - rx.forgetting());
            dr.push(rx.a_test_after_b.balanced() - ry.a_test_after_b.balanced());
            db.push(rx.b_test_after_b.balanced() - ry.b_test_after_b.balanced());
        }
    }
    Comparison {
        x,
        y,
        n: df.len(),
        a_test: gb::spread(&da),
        forget: gb::paired(&df),
        retained: gb::spread(&dr),
        b: gb::spread(&db),
    }
}

/// Pre-registered comparisons: every (learner, mechanism ≠ none) vs (learner,
/// none), then every graded (learner, mechanism) vs (control, none).
#[must_use]
pub fn preregistered_comparisons(rows: &[Row]) -> Vec<Comparison> {
    let mut out = Vec::new();
    for l in LEARNERS {
        for m in MECHANISMS.iter().skip(1) {
            out.push(compare(rows, (l, *m), (l, Mechanism::None)));
        }
    }
    for l in LEARNERS.iter().skip(1) {
        for m in MECHANISMS {
            out.push(compare(rows, (*l, m), (Learner::Control, Mechanism::None)));
        }
    }
    out
}

/// H3 rule vs the control: A and B deltas > −margin and forgetting lower by
/// ≥ margin in ≥ 8/10 seeds.
#[must_use]
pub fn beats_control_tradeoff(c: &Comparison) -> bool {
    c.n > 0
        && c.a_test.mean > -gb::MARGIN
        && c.b.mean > -gb::MARGIN
        && c.forget.delta.mean >= gb::MARGIN
        && (c.forget.better as f64) >= 0.8 * c.n as f64
}

/// Pre-stated readings: (H1, H2, H3).
#[must_use]
pub fn readings(rows: &[Row]) -> (bool, bool, bool) {
    let graded = &LEARNERS[1..];
    let pass = |l: Learner, m: Mechanism| compare(rows, (l, m), (l, Mechanism::None)).passes();
    let h1 = graded.iter().all(|&l| pass(l, Mechanism::Consolidate));
    let h2 = graded.iter().all(|&l| pass(l, Mechanism::Rehearse));
    let h3 = graded.iter().any(|&l| {
        [Mechanism::Consolidate, Mechanism::Rehearse]
            .iter()
            .any(|&m| {
                pass(l, m)
                    && beats_control_tradeoff(&compare(
                        rows,
                        (l, m),
                        (Learner::Control, Mechanism::None),
                    ))
            })
    });
    (h1, h2, h3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eligibility_trace::{self as et, TraceArm};
    use crate::graded_reward::{self as gr, GradedArm};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn shape() -> GraphShape {
        GraphShape {
            sources: 40,
            targets: 12,
            fan_in: 6,
        }
    }

    fn spec() -> GenSpec {
        GenSpec {
            active: 3,
            train_patterns: 16,
            test_patterns: 16,
            rounds_a: 4,
            rounds_b: 4,
            positive_pct: 50,
            flip_pct: 50,
            init_weight: 1,
        }
    }

    fn setup(seed: u64) -> (Connectome, gb::Task) {
        let g = rl::graph_for(None, &shape(), seed).unwrap();
        let t = build_task(&g, &spec(), seed).unwrap();
        (g, t)
    }

    #[test]
    fn none_arms_are_bit_identical_to_the_38_rows() {
        for seed in [1, 2, 3] {
            let (g, task) = setup(seed);
            let ctl = et::run_trace_arm(&g, &task, &spec(), 0, TraceArm::Control, seed).unwrap();
            let ours =
                run_arm(&g, &task, &spec(), Learner::Control, Mechanism::None, seed).unwrap();
            assert_eq!(ours.a_test_after_a, ctl.a_test_after_a);
            assert_eq!(ours.a_test_after_b, ctl.a_test_after_b);
            assert_eq!(ours.b_test_after_b, ctl.b_test_after_b);
            assert_eq!(
                ours.cost_a.weights_changed + ours.cost_b.weights_changed,
                ctl.cost_a.weights_changed + ctl.cost_b.weights_changed
            );
            for grain in [Grain::Groups(8), Grain::PerTarget] {
                let theirs =
                    gr::run_graded_arm(&g, &task, &spec(), 0, GradedArm::Ladder(grain), seed)
                        .unwrap();
                let ours = run_arm(
                    &g,
                    &task,
                    &spec(),
                    Learner::Graded(grain),
                    Mechanism::None,
                    seed,
                )
                .unwrap();
                assert_eq!(ours.a_test_after_a, theirs.a_test_after_a);
                assert_eq!(ours.a_test_after_b, theirs.a_test_after_b);
                assert_eq!(ours.b_test_after_b, theirs.b_test_after_b);
                assert_eq!(
                    ours.cost_b.weights_changed,
                    theirs.cost_b.trace.weights_changed
                );
                assert_eq!(ours.cost_b.blocked, 0);
            }
        }
    }

    #[test]
    fn rehearsal_stream_keeps_the_budget_and_replaces_one_in_eight() {
        let (_, task) = setup(5);
        let plain = b_stream(&task, &spec(), 5, false);
        let r = b_stream(&task, &spec(), 5, true);
        assert_eq!(plain.len(), r.len());
        assert_eq!(plain.len(), 16 * 4);
        let mut replaced = 0;
        for (k, (p, q)) in plain.iter().zip(&r).enumerate() {
            if k % REHEARSAL_PERIOD == REHEARSAL_PERIOD - 1 {
                replaced += 1;
                assert_eq!(q.1, task.valence_a.as_slice());
                assert!(task.a_train.iter().any(|a| a.as_slice() == q.0));
            } else {
                assert_eq!(p, q);
            }
        }
        assert_eq!(replaced, plain.len() / REHEARSAL_PERIOD);
    }

    #[test]
    fn mechanisms_act_only_in_b_and_only_on_existing_weights() {
        let (g, task) = setup(6);
        for learner in LEARNERS {
            let base = run_arm(&g, &task, &spec(), learner, Mechanism::None, 6).unwrap();
            for m in MECHANISMS {
                let r = run_arm(&g, &task, &spec(), learner, m, 6).unwrap();
                assert!(r.topology_unchanged);
                assert_eq!(r.a_test_after_a, base.a_test_after_a);
                assert_eq!(r.cost_a, base.cost_a);
                if m.consolidates() {
                    assert!(r.protected_edges > 0);
                    assert!(r.cost_b.blocked <= r.cost_b.protected_attempts);
                    // per edge, floor(attempts / PROTECT_PERIOD) pass
                    let allowed = r.cost_b.protected_attempts - r.cost_b.blocked;
                    assert!(allowed * u64::from(PROTECT_PERIOD) <= r.cost_b.protected_attempts);
                } else {
                    assert_eq!((r.protected_edges, r.cost_b.blocked), (0, 0));
                }
                assert_eq!(r.cost_b.rehearsal_episodes > 0, m.rehearses());
            }
        }
    }

    #[test]
    fn grid_is_deterministic_and_csv_complete() {
        let a = run_grid(None, &shape(), &spec(), &[1, 2]).unwrap();
        let b = run_grid(None, &shape(), &spec(), &[1, 2]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 2 * 12);
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        assert!(csv.contains(",trace-per-target,both,2,"));
        let cmp = preregistered_comparisons(&a);
        assert_eq!(cmp.len(), 9 + 8);
        assert!(cmp.iter().all(|c| c.n == 2));
        let self_cmp = compare(
            &a,
            (Learner::Control, Mechanism::None),
            (Learner::Control, Mechanism::None),
        );
        assert!(!self_cmp.passes());
        let _ = readings(&a);
        assert!(run_grid(None, &shape(), &spec(), &[]).is_err());
    }

    #[test]
    fn checkpoint_after_a_resumes_b_bit_exactly() -> io::Result<()> {
        let (g, task) = setup(7);
        let learner = Learner::Graded(Grain::PerTarget);
        let full = run_arm(&g, &task, &spec(), learner, Mechanism::Both, 7)?;
        let mut c = PlasticCircuit::new(&g, 1)?;
        let a_items: Vec<Item<'_>> = episode_order(task.a_train.len(), 4, 7)
            .into_iter()
            .map(|i| (task.a_train[i].as_slice(), task.valence_a.as_slice()))
            .collect();
        train(learner, &mut c, &g, &a_items, &mut |_| true)?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| bad("clock"))?
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("v888-cg-{}-{nonce}.v888pls", std::process::id()));
        c.save(&path)?;
        let mut resumed = PlasticCircuit::load(&path, &g)?;
        std::fs::remove_file(&path)?;
        let protected: Vec<bool> = resumed.weights().iter().map(|&w| w != 1).collect();
        let mut attempts = vec![0_u32; g.edges()];
        let b_items = b_stream(&task, &spec(), 7, true);
        train(learner, &mut resumed, &g, &b_items, &mut |e| {
            if !protected[e] {
                return true;
            }
            attempts[e] += 1;
            attempts[e] % PROTECT_PERIOD == 0
        })?;
        assert_eq!(
            evaluate(&resumed, &g, &task.test, &task.valence_a)?,
            full.a_test_after_b
        );
        assert_eq!(
            evaluate(&resumed, &g, &task.test, &task.valence_b)?,
            full.b_test_after_b
        );
        Ok(())
    }
}

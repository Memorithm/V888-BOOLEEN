//! B888-4: equal-budget **generalisation and forgetting** comparison of the
//! executable learning strategies against the supervised control.
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present in this repository or on the
//! build machine, so the default run uses the deterministic generated graph of
//! [`crate::consolidation_scale::generate_graph`]. Labels are artificial
//! (hidden per-edge valences). Nothing here is a claim about biology, insect
//! memory or BANC physiology. A user-built `.v888b` can be passed to the same
//! code path; no such run has been made.
//!
//! Protocol (fixed in code and in `docs/GENERALIZATION_FORGETTING.md` before
//! any number was looked at):
//! * Patterns are sets of `active` distinct input neurons (neurons with at
//!   least one outgoing non-zero edge). Per seed, three disjoint pattern sets
//!   are drawn: A-train, B-train and TEST. TEST patterns never appear in any
//!   training set, so TEST measures generalisation to new input combinations
//!   (individual edges may or may not have been seen; the fraction of TEST
//!   edges never activated during A training is reported).
//! * Task A: every original edge gets a hidden valence (`positive_pct` %
//!   positive). For a pattern, a target reached by an active edge must fire
//!   iff at least one active edge into it is positive (OR rule). With
//!   threshold 1 this rule is exactly representable by 4-bit weights
//!   (positive edges >= 1, negative edges = 0), so 1.0 is reachable.
//! * Task B: same rule, but `flip_pct` % of the edge valences are flipped
//!   (conflict). A perfect B learner scores the reported A/B agreement on A.
//! * Equal budget: every arm sees the same graph, the same pattern sets, the
//!   same episode order (`rounds` x training patterns, LCG shuffle) and the
//!   same initial weight. Only the update rule differs.
//! * Arms: `frozen` (no update; baseline), `supervised` (control,
//!   `PlasticCircuit::train` semantics), `associative-coactivation` (teacher
//!   clamps the post-synaptic state, no teacher error signal), and
//!   `reward-modulated` (three-factor: eligibility = active pre AND the
//!   circuit's *own* predicted post spike; per-target reward +1 if that spike
//!   was correct, -1 otherwise; silent targets receive no update).
//! * Metric: balanced accuracy (mean of TPR and TNR) over (pattern, reached
//!   target) pairs, so constant predictors score 0.5. Forgetting is
//!   `bal(A test after A) - bal(A test after B)`.
//! * Pre-stated verdict rule vs the same-seed, same-init supervised control:
//!   a strategy "beats" the control on a metric if the mean paired difference
//!   is >= 0.05 in its favour AND it is better in >= 80 % of seeds; it is
//!   "worse" if the mean paired difference is >= 0.05 against it; otherwise
//!   "no clear difference". The forgetting verdict additionally requires that
//!   B test balanced accuracy does not drop by more than 0.05.
use crate::connectome::Connectome;
use crate::consolidation_scale::{generate_graph, ScaleSpec};
use crate::learning_strategies::{apply_update, LearningStrategy, RewardSignal};
use crate::plasticity::PlasticCircuit;
use crate::BooleanState;
use std::io::{self, ErrorKind};
use std::time::{Duration, Instant};

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Generated-graph shape (ignored when a graph is supplied).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphShape {
    pub sources: usize,
    pub targets: usize,
    pub fan_in: usize,
}

/// Task and budget parameters shared by every arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenSpec {
    pub active: usize,
    pub train_patterns: usize,
    pub test_patterns: usize,
    pub rounds_a: u32,
    pub rounds_b: u32,
    pub positive_pct: u32,
    pub flip_pct: u32,
    pub init_weight: u8,
}

impl GenSpec {
    fn validate(&self) -> io::Result<()> {
        if self.active == 0
            || self.train_patterns == 0
            || self.test_patterns == 0
            || self.rounds_a == 0
            || self.rounds_b == 0
            || self.rounds_a > 64
            || self.rounds_b > 64
            || self.positive_pct > 100
            || self.flip_pct > 100
            || self.init_weight > 15
            || self.train_patterns > 4096
            || self.test_patterns > 4096
        {
            return Err(bad("invalid generalisation spec"));
        }
        Ok(())
    }
}

/// One compared arm. `Frozen` never updates (baseline at the initial weight).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    Frozen,
    Learn(LearningStrategy),
}

impl Arm {
    #[must_use]
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Frozen => "frozen",
            Self::Learn(s) => s.as_label(),
        }
    }
}

/// Every arm, control (`supervised`) second, always in this order.
pub const ALL_ARMS: [Arm; 4] = [
    Arm::Frozen,
    Arm::Learn(LearningStrategy::Supervised),
    Arm::Learn(LearningStrategy::AssociativeCoactivation),
    Arm::Learn(LearningStrategy::RewardModulated),
];

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x6E4E_B888_4C0F_FEE5)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(0x5851_F42D_4C95_7F2D)
            .wrapping_add(0x1405_7B7E_F767_814F);
        self.0 >> 17
    }
    fn pct(&mut self, pct: u32) -> bool {
        (self.next() % 100) < u64::from(pct)
    }
}

/// Confusion counts over (pattern, reached target) pairs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Confusion {
    pub tp: u64,
    pub tn: u64,
    pub fp: u64,
    pub fn_: u64,
}

impl Confusion {
    #[must_use]
    pub fn total(&self) -> u64 {
        self.tp + self.tn + self.fp + self.fn_
    }
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        if self.total() == 0 {
            return 0.0;
        }
        (self.tp + self.tn) as f64 / self.total() as f64
    }
    /// Mean of TPR and TNR; a class with no example counts as 0.5.
    #[must_use]
    pub fn balanced(&self) -> f64 {
        let rate = |hit: u64, miss: u64| {
            if hit + miss == 0 {
                0.5
            } else {
                hit as f64 / (hit + miss) as f64
            }
        };
        0.5 * (rate(self.tp, self.fn_) + rate(self.tn, self.fp))
    }
}

/// Deterministic per-seed task: pattern sets and valences for A and B.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Task {
    pub pool: Vec<usize>,
    pub a_train: Vec<Vec<usize>>,
    pub b_train: Vec<Vec<usize>>,
    pub test: Vec<Vec<usize>>,
    pub valence_a: Vec<bool>,
    pub valence_b: Vec<bool>,
}

fn usable(graph: &Connectome, edge: usize) -> bool {
    graph.counts[edge] != 0
}

/// Draw the task for one seed. Fails if the graph cannot supply enough
/// distinct patterns.
pub fn build_task(graph: &Connectome, spec: &GenSpec, seed: u64) -> io::Result<Task> {
    graph.validate()?;
    spec.validate()?;
    let pool: Vec<usize> = (0..graph.ids.len())
        .filter(|&s| (graph.offsets[s]..graph.offsets[s + 1]).any(|e| usable(graph, e as usize)))
        .collect();
    if pool.len() < spec.active {
        return Err(bad("not enough input neurons for the pattern size"));
    }
    let mut rng = Lcg::new(seed);
    let needed = 2 * spec.train_patterns + spec.test_patterns;
    let mut seen: Vec<Vec<usize>> = Vec::with_capacity(needed);
    let mut attempts = 0_usize;
    while seen.len() < needed {
        attempts += 1;
        if attempts > needed * 1000 {
            return Err(bad("could not draw enough distinct patterns"));
        }
        let mut pattern = Vec::with_capacity(spec.active);
        while pattern.len() < spec.active {
            let s = pool[(rng.next() as usize) % pool.len()];
            if !pattern.contains(&s) {
                pattern.push(s);
            }
        }
        pattern.sort_unstable();
        if !seen.contains(&pattern) {
            seen.push(pattern);
        }
    }
    let test = seen.split_off(2 * spec.train_patterns);
    let b_train = seen.split_off(spec.train_patterns);
    let a_train = seen;
    let valence_a: Vec<bool> = (0..graph.edges())
        .map(|_| rng.pct(spec.positive_pct))
        .collect();
    let valence_b: Vec<bool> = valence_a
        .iter()
        .map(|&v| if rng.pct(spec.flip_pct) { !v } else { v })
        .collect();
    Ok(Task {
        pool,
        a_train,
        b_train,
        test,
        valence_a,
        valence_b,
    })
}

/// Input bitmap, the list of reached targets and the teacher bitmap.
pub(crate) fn encode(
    graph: &Connectome,
    pattern: &[usize],
    valence: &[bool],
) -> io::Result<(BooleanState, Vec<usize>, BooleanState)> {
    let n = graph.ids.len();
    let mut input = BooleanState::new(n);
    let mut teacher = BooleanState::new(n);
    let mut reached = Vec::new();
    for &s in pattern {
        input.set(s, true).map_err(|_| bad("bad source"))?;
    }
    for &s in pattern {
        for edge in graph.offsets[s]..graph.offsets[s + 1] {
            let edge = edge as usize;
            if !usable(graph, edge) {
                continue;
            }
            let t = graph.targets[edge] as usize;
            if !reached.contains(&t) {
                reached.push(t);
            }
            if valence[edge] {
                teacher.set(t, true).map_err(|_| bad("bad target"))?;
            }
        }
    }
    Ok((input, reached, teacher))
}

/// Per-phase instrumentation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseCost {
    pub episodes: u64,
    pub update_calls: u64,
    pub weights_changed: u64,
}

pub(crate) fn episode_order(patterns: usize, rounds: u32, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..rounds).flat_map(|_| 0..patterns).collect();
    crate::learning_strategies::shuffle_with_seed(&mut order, seed);
    order
}

/// Train one phase. Public so a phase can be checkpointed between A and B
/// with the existing `V888PLS1` weight file and resumed bit-exactly.
pub fn train_phase(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    arm: Arm,
    patterns: &[Vec<usize>],
    valence: &[bool],
    budget: (u32, u64),
) -> io::Result<PhaseCost> {
    let (rounds, seed) = budget;
    let mut cost = PhaseCost::default();
    for k in episode_order(patterns.len(), rounds, seed) {
        cost.episodes += 1;
        let strategy = match arm {
            Arm::Frozen => continue,
            Arm::Learn(s) => s,
        };
        let (input, reached, teacher) = encode(graph, &patterns[k], valence)?;
        match strategy {
            LearningStrategy::Supervised => {
                let r = apply_update(
                    circuit,
                    graph,
                    strategy,
                    &input,
                    &teacher,
                    RewardSignal::None,
                )?;
                cost.update_calls += 1;
                cost.weights_changed += r.weights_changed;
            }
            LearningStrategy::AssociativeCoactivation => {
                // Teacher-clamped post-synaptic activity; no error signal.
                let mut observed = teacher.clone();
                for s in input.active_indices() {
                    observed.set(s, true).map_err(|_| bad("bad source"))?;
                }
                let r = apply_update(
                    circuit,
                    graph,
                    strategy,
                    &input,
                    &observed,
                    RewardSignal::None,
                )?;
                cost.update_calls += 1;
                cost.weights_changed += r.weights_changed;
            }
            LearningStrategy::RewardModulated => {
                // Three-factor: own prediction gives eligibility, per-target
                // scalar reward says whether that spike was right.
                let prediction = circuit.predict(graph, &input, 1)?;
                for &t in &reached {
                    if !prediction.get(t).map_err(|_| bad("bad target"))? {
                        continue;
                    }
                    let mut post = BooleanState::new(graph.ids.len());
                    post.set(t, true).map_err(|_| bad("bad target"))?;
                    let reward = if teacher.get(t).map_err(|_| bad("bad target"))? {
                        RewardSignal::Positive
                    } else {
                        RewardSignal::Negative
                    };
                    let r = apply_update(circuit, graph, strategy, &input, &post, reward)?;
                    cost.update_calls += 1;
                    cost.weights_changed += r.weights_changed;
                }
            }
        }
    }
    Ok(cost)
}

/// Score a circuit on a pattern set under a valence assignment.
pub fn evaluate(
    circuit: &PlasticCircuit,
    graph: &Connectome,
    patterns: &[Vec<usize>],
    valence: &[bool],
) -> io::Result<Confusion> {
    let mut c = Confusion::default();
    for p in patterns {
        let (input, reached, teacher) = encode(graph, p, valence)?;
        let out = circuit.predict(graph, &input, 1)?;
        for t in reached {
            let want = teacher.get(t).map_err(|_| bad("bad target"))?;
            let got = out.get(t).map_err(|_| bad("bad target"))?;
            match (want, got) {
                (true, true) => c.tp += 1,
                (false, false) => c.tn += 1,
                (false, true) => c.fp += 1,
                (true, false) => c.fn_ += 1,
            }
        }
    }
    Ok(c)
}

/// One (seed, arm) measurement.
#[derive(Clone, Debug)]
pub struct GenRow {
    pub arm: Arm,
    pub seed: u64,
    pub init_weight: u8,
    pub edges: usize,
    pub a_train_after_a: Confusion,
    pub a_test_after_a: Confusion,
    pub a_test_after_b: Confusion,
    pub b_test_after_b: Confusion,
    /// Fraction of TEST (pattern, edge) activations never activated in A-train.
    pub test_edges_unseen_in_a: f64,
    /// Fraction of TEST (pattern, target) pairs whose A and B labels agree.
    pub ab_label_agreement: f64,
    pub cost_a: PhaseCost,
    pub cost_b: PhaseCost,
    pub weight_bytes: usize,
    pub topology_unchanged: bool,
    pub elapsed: Duration,
}

impl GenRow {
    #[must_use]
    pub fn forgetting(&self) -> f64 {
        self.a_test_after_a.balanced() - self.a_test_after_b.balanced()
    }
}

fn task_stats(graph: &Connectome, task: &Task) -> io::Result<(f64, f64)> {
    let mut seen_edge = vec![false; graph.edges()];
    for p in &task.a_train {
        for &s in p {
            for e in graph.offsets[s]..graph.offsets[s + 1] {
                seen_edge[e as usize] = true;
            }
        }
    }
    let (mut unseen, mut total) = (0_u64, 0_u64);
    let (mut agree, mut pairs) = (0_u64, 0_u64);
    for p in &task.test {
        for &s in p {
            for e in graph.offsets[s]..graph.offsets[s + 1] {
                if usable(graph, e as usize) {
                    total += 1;
                    unseen += u64::from(!seen_edge[e as usize]);
                }
            }
        }
        let (_, reached, ta) = encode(graph, p, &task.valence_a)?;
        let (_, _, tb) = encode(graph, p, &task.valence_b)?;
        for t in reached {
            pairs += 1;
            agree += u64::from(
                ta.get(t).map_err(|_| bad("bad target"))?
                    == tb.get(t).map_err(|_| bad("bad target"))?,
            );
        }
    }
    let frac = |a: u64, b: u64| if b == 0 { 0.0 } else { a as f64 / b as f64 };
    Ok((frac(unseen, total), frac(agree, pairs)))
}

/// Run A then B for one arm on one seed's task.
pub fn run_arm(
    graph: &Connectome,
    task: &Task,
    spec: &GenSpec,
    arm: Arm,
    seed: u64,
) -> io::Result<GenRow> {
    spec.validate()?;
    let original = graph.clone();
    let start = Instant::now();
    let mut circuit = PlasticCircuit::new(graph, spec.init_weight)?;
    let fingerprint = circuit.topology_fingerprint();
    let cost_a = train_phase(
        &mut circuit,
        graph,
        arm,
        &task.a_train,
        &task.valence_a,
        (spec.rounds_a, seed),
    )?;
    let a_train_after_a = evaluate(&circuit, graph, &task.a_train, &task.valence_a)?;
    let a_test_after_a = evaluate(&circuit, graph, &task.test, &task.valence_a)?;
    let cost_b = train_phase(
        &mut circuit,
        graph,
        arm,
        &task.b_train,
        &task.valence_b,
        (spec.rounds_b, seed ^ 0xB0B0_B0B0),
    )?;
    let a_test_after_b = evaluate(&circuit, graph, &task.test, &task.valence_a)?;
    let b_test_after_b = evaluate(&circuit, graph, &task.test, &task.valence_b)?;
    let (test_edges_unseen_in_a, ab_label_agreement) = task_stats(graph, task)?;
    Ok(GenRow {
        arm,
        seed,
        init_weight: spec.init_weight,
        edges: graph.edges(),
        a_train_after_a,
        a_test_after_a,
        a_test_after_b,
        b_test_after_b,
        test_edges_unseen_in_a,
        ab_label_agreement,
        cost_a,
        cost_b,
        weight_bytes: circuit.weight_bytes(),
        topology_unchanged: *graph == original
            && circuit.topology_fingerprint() == fingerprint
            && circuit.weights().len() == graph.edges(),
        elapsed: start.elapsed(),
    })
}

/// Full grid: init weight x seed x arm. Without a supplied graph each seed
/// generates its own synthetic graph (seed varies graph, patterns, valences
/// and episode order). With a supplied graph only the task varies.
pub fn run_grid(
    graph: Option<&Connectome>,
    shape: &GraphShape,
    spec: &GenSpec,
    seeds: &[u64],
    inits: &[u8],
) -> io::Result<Vec<GenRow>> {
    if seeds.is_empty() || inits.is_empty() {
        return Err(bad("need at least one seed and one init weight"));
    }
    let mut rows = Vec::new();
    for &init in inits {
        let spec = GenSpec {
            init_weight: init,
            ..*spec
        };
        for &seed in seeds {
            let owned;
            let g = match graph {
                Some(g) => g,
                None => {
                    let scale = ScaleSpec {
                        sources: shape.sources,
                        targets: shape.targets,
                        fan_in: shape.fan_in,
                        conflict_pct: 0,
                        rounds_a: 1,
                        rounds_b: 1,
                    };
                    owned = generate_graph(&scale, seed)?.graph;
                    &owned
                }
            };
            let task = build_task(g, &spec, seed)?;
            for arm in ALL_ARMS {
                rows.push(run_arm(g, &task, &spec, arm, seed)?);
            }
        }
    }
    Ok(rows)
}

pub const CSV_HEADER: &str = "schema,graph,init_weight,arm,seed,edges,a_train_bal_after_a,a_test_bal_after_a,a_test_acc_after_a,a_test_bal_after_b,b_test_bal_after_b,forgetting,test_pairs,test_edges_unseen_in_a,ab_label_agreement,a_episodes,a_update_calls,a_weights_changed,b_episodes,b_update_calls,b_weights_changed,weight_bytes,topology_unchanged,elapsed_us";

/// CSV, one row per (init, seed, arm). `elapsed_us` is the only
/// non-deterministic column.
#[must_use]
pub fn to_csv(rows: &[GenRow], graph_label: &str) -> String {
    let mut out = String::from(CSV_HEADER);
    out.push('\n');
    for r in rows {
        out.push_str(&format!(
            "b888-4-generalization-v1,{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{},{},{},{},{},{},{},{},{}\n",
            graph_label,
            r.init_weight,
            r.arm.as_label(),
            r.seed,
            r.edges,
            r.a_train_after_a.balanced(),
            r.a_test_after_a.balanced(),
            r.a_test_after_a.accuracy(),
            r.a_test_after_b.balanced(),
            r.b_test_after_b.balanced(),
            r.forgetting(),
            r.a_test_after_a.total(),
            r.test_edges_unseen_in_a,
            r.ab_label_agreement,
            r.cost_a.episodes,
            r.cost_a.update_calls,
            r.cost_a.weights_changed,
            r.cost_b.episodes,
            r.cost_b.update_calls,
            r.cost_b.weights_changed,
            r.weight_bytes,
            r.topology_unchanged,
            r.elapsed.as_micros()
        ));
    }
    out
}

/// Mean, sample standard deviation, min and max.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spread {
    pub mean: f64,
    pub sd: f64,
    pub min: f64,
    pub max: f64,
}

#[must_use]
pub fn spread(values: &[f64]) -> Spread {
    if values.is_empty() {
        return Spread {
            mean: 0.0,
            sd: 0.0,
            min: 0.0,
            max: 0.0,
        };
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let sd = if values.len() > 1 {
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    } else {
        0.0
    };
    Spread {
        mean,
        sd,
        min: values.iter().copied().fold(f64::INFINITY, f64::min),
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

/// Paired comparison with the same-seed supervised control. `better` counts
/// seeds where the arm is strictly better (higher generalisation, lower
/// forgetting).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Paired {
    pub delta: Spread,
    pub better: usize,
    pub ties: usize,
    pub worse: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Control,
    Beats,
    Worse,
    NoClearDifference,
}

impl Verdict {
    #[must_use]
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Control => "control",
            Self::Beats => "beats_control",
            Self::Worse => "worse_than_control",
            Self::NoClearDifference => "no_clear_difference",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArmSummary {
    pub init_weight: u8,
    pub arm: Arm,
    pub seeds: usize,
    pub a_train_after_a: Spread,
    pub a_test_after_a: Spread,
    pub a_test_after_b: Spread,
    pub b_test_after_b: Spread,
    pub forgetting: Spread,
    pub test_edges_unseen_in_a: f64,
    pub ab_label_agreement: f64,
    pub generalisation_vs_control: Paired,
    pub forgetting_vs_control: Paired,
    pub b_vs_control: Spread,
    pub generalisation_verdict: Verdict,
    pub forgetting_verdict: Verdict,
}

pub(crate) const MARGIN: f64 = 0.05;
const EPS: f64 = 1e-12;

pub(crate) fn paired(deltas: &[f64]) -> Paired {
    Paired {
        delta: spread(deltas),
        better: deltas.iter().filter(|&&d| d > EPS).count(),
        ties: deltas.iter().filter(|&&d| d.abs() <= EPS).count(),
        worse: deltas.iter().filter(|&&d| d < -EPS).count(),
    }
}

pub(crate) fn verdict(p: &Paired, n: usize, extra_ok: bool) -> Verdict {
    if p.delta.mean >= MARGIN && (p.better as f64) >= 0.8 * n as f64 && extra_ok {
        Verdict::Beats
    } else if p.delta.mean <= -MARGIN {
        Verdict::Worse
    } else {
        Verdict::NoClearDifference
    }
}

/// Per (init weight, arm) summary with paired deltas vs supervised control.
/// Deltas are oriented so that positive = in the arm's favour (higher
/// generalisation; *lower* forgetting).
#[must_use]
pub fn summarize(rows: &[GenRow]) -> Vec<ArmSummary> {
    let mut keys: Vec<(u8, Arm)> = Vec::new();
    for r in rows {
        if !keys.contains(&(r.init_weight, r.arm)) {
            keys.push((r.init_weight, r.arm));
        }
    }
    let control_arm = Arm::Learn(LearningStrategy::Supervised);
    let mut out = Vec::new();
    for (init, arm) in keys {
        let sel: Vec<&GenRow> = rows
            .iter()
            .filter(|r| r.init_weight == init && r.arm == arm)
            .collect();
        let col = |f: &dyn Fn(&GenRow) -> f64| -> Vec<f64> { sel.iter().map(|r| f(r)).collect() };
        let mut d_gen = Vec::new();
        let mut d_forget = Vec::new();
        let mut d_b = Vec::new();
        for r in &sel {
            if let Some(c) = rows
                .iter()
                .find(|c| c.init_weight == init && c.seed == r.seed && c.arm == control_arm)
            {
                d_gen.push(r.a_test_after_a.balanced() - c.a_test_after_a.balanced());
                d_forget.push(c.forgetting() - r.forgetting());
                d_b.push(r.b_test_after_b.balanced() - c.b_test_after_b.balanced());
            }
        }
        let gen_p = paired(&d_gen);
        let forget_p = paired(&d_forget);
        let b_spread = spread(&d_b);
        let n = d_gen.len();
        let (gv, fv) = if arm == control_arm {
            (Verdict::Control, Verdict::Control)
        } else {
            (
                verdict(&gen_p, n, true),
                verdict(&forget_p, n, b_spread.mean > -MARGIN),
            )
        };
        out.push(ArmSummary {
            init_weight: init,
            arm,
            seeds: sel.len(),
            a_train_after_a: spread(&col(&|r| r.a_train_after_a.balanced())),
            a_test_after_a: spread(&col(&|r| r.a_test_after_a.balanced())),
            a_test_after_b: spread(&col(&|r| r.a_test_after_b.balanced())),
            b_test_after_b: spread(&col(&|r| r.b_test_after_b.balanced())),
            forgetting: spread(&col(&|r| r.forgetting())),
            test_edges_unseen_in_a: spread(&col(&|r| r.test_edges_unseen_in_a)).mean,
            ab_label_agreement: spread(&col(&|r| r.ab_label_agreement)).mean,
            generalisation_vs_control: gen_p,
            forgetting_vs_control: forget_p,
            b_vs_control: b_spread,
            generalisation_verdict: gv,
            forgetting_verdict: fv,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn shape() -> GraphShape {
        GraphShape {
            sources: 40,
            targets: 12,
            fan_in: 6,
        }
    }

    fn spec(init: u8) -> GenSpec {
        GenSpec {
            active: 3,
            train_patterns: 16,
            test_patterns: 16,
            rounds_a: 4,
            rounds_b: 4,
            positive_pct: 50,
            flip_pct: 50,
            init_weight: init,
        }
    }

    fn graph(seed: u64) -> Connectome {
        let s = shape();
        generate_graph(
            &ScaleSpec {
                sources: s.sources,
                targets: s.targets,
                fan_in: s.fan_in,
                conflict_pct: 0,
                rounds_a: 1,
                rounds_b: 1,
            },
            seed,
        )
        .unwrap()
        .graph
    }

    #[test]
    fn task_sets_are_disjoint_deterministic_and_valid() {
        let g = graph(3);
        let t1 = build_task(&g, &spec(1), 3).unwrap();
        let t2 = build_task(&g, &spec(1), 3).unwrap();
        assert_eq!(t1, t2);
        assert_ne!(t1, build_task(&g, &spec(1), 4).unwrap());
        for p in &t1.test {
            assert!(!t1.a_train.contains(p) && !t1.b_train.contains(p));
            assert_eq!(p.len(), 3);
        }
        for p in &t1.a_train {
            assert!(!t1.b_train.contains(p));
        }
        assert_eq!(t1.valence_a.len(), g.edges());
        // only original sources are ever activated
        assert!(t1.pool.iter().all(|&s| s < 40));
    }

    #[test]
    fn flip_zero_keeps_b_equal_to_a_and_flip_hundred_inverts() {
        let g = graph(1);
        let mut s = spec(1);
        s.flip_pct = 0;
        let t = build_task(&g, &s, 1).unwrap();
        assert_eq!(t.valence_a, t.valence_b);
        s.flip_pct = 100;
        let t = build_task(&g, &s, 1).unwrap();
        assert!(t.valence_a.iter().zip(&t.valence_b).all(|(a, b)| a != b));
    }

    #[test]
    fn invalid_specs_and_impossible_tasks_rejected() {
        let g = graph(1);
        let mut s = spec(1);
        s.active = 0;
        assert!(build_task(&g, &s, 1).is_err());
        let mut s = spec(16);
        s.init_weight = 16;
        assert!(build_task(&g, &s, 1).is_err());
        let mut s = spec(1);
        s.active = 41;
        assert!(build_task(&g, &s, 1).is_err());
        // a 2-source graph cannot supply 48 distinct 1-patterns
        let tiny = crate::learning_strategies::comparison_graph();
        let mut s = spec(1);
        s.active = 1;
        assert!(build_task(&tiny, &s, 1).is_err());
    }

    #[test]
    fn supervised_learns_seen_patterns_and_frozen_never_changes() {
        let g = graph(2);
        let task = build_task(&g, &spec(0), 2).unwrap();
        let frozen = run_arm(&g, &task, &spec(0), Arm::Frozen, 2).unwrap();
        assert_eq!(frozen.cost_a.weights_changed, 0);
        assert_eq!(frozen.cost_a.update_calls, 0);
        assert!((frozen.a_test_after_a.balanced() - 0.5).abs() < 1e-12);
        let sup = run_arm(
            &g,
            &task,
            &spec(0),
            Arm::Learn(LearningStrategy::Supervised),
            2,
        )
        .unwrap();
        // Not 1.0: a negative edge co-active with a positive one is
        // reinforced by the OR teacher (measured 0.878 on this seed).
        assert!(sup.a_train_after_a.balanced() > 0.8);
        assert!(sup.topology_unchanged);
        assert_eq!(sup.cost_a.episodes, 16 * 4);
        assert_eq!(sup.cost_a.episodes, frozen.cost_a.episodes);
    }

    #[test]
    fn reward_arm_cannot_learn_from_silent_start() {
        // From init weight 0 nothing ever fires, so the three-factor rule has
        // no eligibility and no update happens: an honest structural limit.
        let g = graph(4);
        let task = build_task(&g, &spec(0), 4).unwrap();
        let r = run_arm(
            &g,
            &task,
            &spec(0),
            Arm::Learn(LearningStrategy::RewardModulated),
            4,
        )
        .unwrap();
        assert_eq!(r.cost_a.weights_changed + r.cost_b.weights_changed, 0);
        let r1 = run_arm(
            &g,
            &task,
            &spec(1),
            Arm::Learn(LearningStrategy::RewardModulated),
            4,
        )
        .unwrap();
        assert!(r1.cost_a.weights_changed > 0);
    }

    #[test]
    fn reward_from_init_one_matches_supervised_on_task_a_only() {
        // Under the OR rule a positive edge is never decremented (it makes
        // its target's label 1), so from init 1 every label-1 target fires
        // and the three-factor rule sees the same eligibility as the
        // teacher. Task B differs: a zeroed edge that must now fire alone is
        // silent and can no longer be raised by the reward rule.
        let g = graph(7);
        let s = spec(1);
        let task = build_task(&g, &s, 7).unwrap();
        let mut sup = PlasticCircuit::new(&g, 1).unwrap();
        let mut rew = sup.clone();
        let sup_arm = Arm::Learn(LearningStrategy::Supervised);
        let rew_arm = Arm::Learn(LearningStrategy::RewardModulated);
        train_phase(
            &mut sup,
            &g,
            sup_arm,
            &task.a_train,
            &task.valence_a,
            (4, 7),
        )
        .unwrap();
        train_phase(
            &mut rew,
            &g,
            rew_arm,
            &task.a_train,
            &task.valence_a,
            (4, 7),
        )
        .unwrap();
        assert_eq!(sup.weights(), rew.weights());
        let b = (4, 7 ^ 0xB0B0_B0B0);
        train_phase(&mut sup, &g, sup_arm, &task.b_train, &task.valence_b, b).unwrap();
        train_phase(&mut rew, &g, rew_arm, &task.b_train, &task.valence_b, b).unwrap();
        assert_ne!(sup.weights(), rew.weights());
    }

    #[test]
    fn associative_equals_supervised_on_bipartite_generated_graph() {
        // Generated targets have no outgoing edges, so clamping the teacher
        // into the observed state reproduces the supervised update exactly.
        let g = graph(5);
        let task = build_task(&g, &spec(1), 5).unwrap();
        let s = run_arm(
            &g,
            &task,
            &spec(1),
            Arm::Learn(LearningStrategy::Supervised),
            5,
        )
        .unwrap();
        let a = run_arm(
            &g,
            &task,
            &spec(1),
            Arm::Learn(LearningStrategy::AssociativeCoactivation),
            5,
        )
        .unwrap();
        assert_eq!(s.a_test_after_b, a.a_test_after_b);
        assert_eq!(s.cost_b, a.cost_b);
    }

    #[test]
    fn grid_is_reproducible_and_budgets_equal() {
        let rows1 = run_grid(None, &shape(), &spec(1), &[1, 2], &[0, 1]).unwrap();
        let rows2 = run_grid(None, &shape(), &spec(1), &[1, 2], &[0, 1]).unwrap();
        assert_eq!(rows1.len(), 2 * 2 * 4);
        for (a, b) in rows1.iter().zip(&rows2) {
            assert_eq!(
                (
                    a.a_test_after_a,
                    a.a_test_after_b,
                    a.b_test_after_b,
                    a.cost_a,
                    a.cost_b
                ),
                (
                    b.a_test_after_a,
                    b.a_test_after_b,
                    b.b_test_after_b,
                    b.cost_a,
                    b.cost_b
                )
            );
        }
        for chunk in rows1.chunks(4) {
            assert!(chunk
                .iter()
                .all(|r| r.cost_a.episodes == chunk[0].cost_a.episodes));
            assert!(chunk
                .iter()
                .all(|r| r.cost_b.episodes == chunk[0].cost_b.episodes));
            assert!(chunk.iter().all(|r| r.topology_unchanged));
        }
        let csv = to_csv(&rows1, "GENERATED_SYNTHETIC");
        assert_eq!(csv.lines().count(), 1 + rows1.len());
        assert!(csv.starts_with(CSV_HEADER));
        let summary = summarize(&rows1);
        assert_eq!(summary.len(), 8);
        let control = summary
            .iter()
            .find(|s| s.arm == Arm::Learn(LearningStrategy::Supervised))
            .unwrap();
        assert_eq!(control.generalisation_verdict, Verdict::Control);
        assert!(control.generalisation_vs_control.delta.mean.abs() < 1e-12);
    }

    #[test]
    fn checkpoint_between_a_and_b_resumes_bit_exactly() -> io::Result<()> {
        let g = graph(6);
        let s = spec(1);
        let task = build_task(&g, &s, 6)?;
        for arm in ALL_ARMS {
            let mut continuous = PlasticCircuit::new(&g, 1)?;
            train_phase(
                &mut continuous,
                &g,
                arm,
                &task.a_train,
                &task.valence_a,
                (4, 6),
            )?;
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| bad("clock"))?
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "v888-gen-{}-{nonce}-{}.v888pls",
                std::process::id(),
                arm.as_label()
            ));
            continuous.save(&path)?;
            let mut resumed = PlasticCircuit::load(&path, &g)?;
            std::fs::remove_file(&path)?;
            let b = (4, 6 ^ 0xB0B0_B0B0);
            let c1 = train_phase(&mut continuous, &g, arm, &task.b_train, &task.valence_b, b)?;
            let c2 = train_phase(&mut resumed, &g, arm, &task.b_train, &task.valence_b, b)?;
            assert_eq!(c1, c2);
            assert_eq!(resumed.weights(), continuous.weights());
        }
        Ok(())
    }

    #[test]
    fn spread_and_balanced_accuracy_oracles() {
        let s = spread(&[1.0, 2.0, 3.0]);
        assert!((s.mean - 2.0).abs() < 1e-12 && (s.sd - 1.0).abs() < 1e-12);
        assert_eq!((s.min, s.max), (1.0, 3.0));
        let c = Confusion {
            tp: 3,
            tn: 1,
            fp: 1,
            fn_: 1,
        };
        assert!((c.balanced() - 0.625).abs() < 1e-12);
        assert!((c.accuracy() - 4.0 / 6.0).abs() < 1e-12);
    }
}

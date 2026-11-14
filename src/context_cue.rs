//! B888-4 cycle 9: an explicit **task-context cue** fed through original,
//! existing edges. Two disjoint sets of existing input nodes are reserved as
//! cue lines (cue A, cue B); during task A (training and evaluation) the cue
//! A lines are switched on, during task B the cue B lines. The cue edges are
//! ordinary edges of the graph and are learned like any other edge. No edge
//! is added or removed. Compared at equal budget: the supervised control,
//! #117's error gate, and their cued versions, at three label-flip rates.
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph; nothing here is
//! a claim about synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/CONTEXT_CUE.md`
//! (committed before any run). Only weights of existing edges change.
use crate::connectome::Connectome;
use crate::error_gated::{self as eg, drive, paired, Gate, Paired, PhaseCost, Split, EFFECT};
use crate::gates;
use crate::generalization_bench::{
    self as gb, build_task_excluding, encode, Confusion, GenSpec, GraphShape,
};
use crate::learning_strategies::{apply_update_gated, LearningStrategy, RewardSignal};
use crate::plasticity::PlasticCircuit;
use crate::reward_limits as rl;
use crate::BooleanState;
use std::io::{self, ErrorKind};

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Pre-registered maximum number of lines per cue.
pub const CUE_CAP: usize = 12;
/// Pre-registered label-flip rates (50 is primary).
pub const FLIPS: [u32; 3] = [25, 50, 100];
pub const PRIMARY_FLIP: u32 = 50;
pub use crate::error_gated::THRESHOLD;

/// Reserved cue lines (existing input nodes) and how many targets each
/// cue reaches through its original out-edges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cue {
    pub a: Vec<usize>,
    pub b: Vec<usize>,
    pub reach_a: usize,
    pub reach_b: usize,
}

impl Cue {
    /// Every cue line (A then B), the nodes excluded from the pattern pool.
    #[must_use]
    pub fn lines(&self) -> Vec<usize> {
        self.a.iter().chain(&self.b).copied().collect()
    }
}

fn out_targets(graph: &Connectome, s: usize) -> Vec<usize> {
    let mut t: Vec<usize> = (graph.offsets[s]..graph.offsets[s + 1])
        .map(|e| e as usize)
        .filter(|&e| gb::usable(graph, e))
        .map(|e| graph.targets[e] as usize)
        .collect();
    t.sort_unstable();
    t.dedup();
    t
}

fn greedy_cover(graph: &Connectome, all_targets: &[usize], taken: &[usize]) -> Vec<usize> {
    greedy_cover_capped(graph, all_targets, taken, CUE_CAP)
}

fn greedy_cover_capped(
    graph: &Connectome,
    all_targets: &[usize],
    taken: &[usize],
    cap: usize,
) -> Vec<usize> {
    let mut covered: Vec<usize> = Vec::new();
    let mut chosen: Vec<usize> = Vec::new();
    while chosen.len() < cap && covered.len() < all_targets.len() {
        let mut best: Option<(usize, usize)> = None;
        for s in 0..graph.ids.len() {
            if taken.contains(&s) || chosen.contains(&s) {
                continue;
            }
            let gain = out_targets(graph, s)
                .iter()
                .filter(|t| !covered.contains(t))
                .count();
            if gain > 0 && best.is_none_or(|(_, g)| gain > g) {
                best = Some((s, gain));
            }
        }
        let Some((s, _)) = best else { break };
        chosen.push(s);
        for t in out_targets(graph, s) {
            if !covered.contains(&t) {
                covered.push(t);
            }
        }
    }
    chosen
}

fn reach(graph: &Connectome, lines: &[usize]) -> usize {
    let mut t: Vec<usize> = lines.iter().flat_map(|&s| out_targets(graph, s)).collect();
    t.sort_unstable();
    t.dedup();
    t.len()
}

/// The pre-registered greedy cue selection (deterministic function of the
/// graph): cue A first, then cue B over the remaining inputs.
pub fn choose_cues(graph: &Connectome) -> io::Result<Cue> {
    graph.validate()?;
    let mut all_targets: Vec<usize> = (0..graph.edges())
        .filter(|&e| gb::usable(graph, e))
        .map(|e| graph.targets[e] as usize)
        .collect();
    all_targets.sort_unstable();
    all_targets.dedup();
    let a = greedy_cover(graph, &all_targets, &[]);
    let b = greedy_cover(graph, &all_targets, &a);
    if a.is_empty() || b.is_empty() {
        return Err(bad("graph has no input usable as a cue line"));
    }
    Ok(Cue {
        reach_a: reach(graph, &a),
        reach_b: reach(graph, &b),
        a,
        b,
    })
}

/// `n` greedy-cover cues (cap [`CUE_CAP`] lines each), chosen in order over
/// the inputs not taken by earlier cues (B888-4 cycle 11 task sequences).
/// The first two cues are exactly [`choose_cues`]'s A and B (tested).
pub fn choose_n_cues(graph: &Connectome, n: usize) -> io::Result<Vec<Vec<usize>>> {
    choose_n_cues_capped(graph, n, CUE_CAP)
}

/// [`choose_n_cues`] with a per-cue line cap (B888-4 cycle 12 uses fewer
/// lines per cue to leave a wider pattern pool). With `cap == CUE_CAP` this
/// is exactly [`choose_n_cues`] (tested).
pub fn choose_n_cues_capped(
    graph: &Connectome,
    n: usize,
    cap: usize,
) -> io::Result<Vec<Vec<usize>>> {
    graph.validate()?;
    if cap == 0 {
        return Err(bad("cue cap must be positive"));
    }
    let mut all_targets: Vec<usize> = (0..graph.edges())
        .filter(|&e| gb::usable(graph, e))
        .map(|e| graph.targets[e] as usize)
        .collect();
    all_targets.sort_unstable();
    all_targets.dedup();
    let mut cues: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut taken: Vec<usize> = Vec::new();
    for _ in 0..n {
        let c = greedy_cover_capped(graph, &all_targets, &taken, cap);
        if c.is_empty() {
            return Err(bad("graph has no input usable as a cue line"));
        }
        taken.extend(&c);
        cues.push(c);
    }
    Ok(cues)
}

/// One compared arm: a #117 gate, with or without the context cue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arm {
    pub gate: Gate,
    pub cued: bool,
}

impl Arm {
    #[must_use]
    pub fn label(self) -> String {
        if self.cued {
            format!("{}+cue", self.gate.label())
        } else {
            self.gate.label()
        }
    }
    /// The same gate without the cue.
    #[must_use]
    pub fn base(self) -> Self {
        Self {
            gate: self.gate,
            cued: false,
        }
    }
}

pub const SUPERVISED: Arm = Arm {
    gate: Gate::Supervised,
    cued: false,
};
pub const ERROR_GATED: Arm = Arm {
    gate: Gate::ErrorGated,
    cued: false,
};
pub const SUPERVISED_CUE: Arm = Arm {
    gate: Gate::Supervised,
    cued: true,
};
pub const ERROR_GATED_CUE: Arm = Arm {
    gate: Gate::ErrorGated,
    cued: true,
};
pub const ARMS: [Arm; 4] = [SUPERVISED, ERROR_GATED, SUPERVISED_CUE, ERROR_GATED_CUE];

/// `encode` plus the active cue lines. Teacher and reached targets come from
/// the pattern's edges only.
pub fn encode_ctx(
    graph: &Connectome,
    pattern: &[usize],
    valence: &[bool],
    cue: &[usize],
) -> io::Result<(BooleanState, Vec<usize>, BooleanState)> {
    if cue.iter().any(|c| pattern.contains(c)) {
        return Err(bad("pattern contains a cue line"));
    }
    let (mut input, reached, teacher) = encode(graph, pattern, valence)?;
    for &c in cue {
        input.set(c, true).map_err(|_| bad("bad cue line"))?;
    }
    Ok((input, reached, teacher))
}

/// One gated supervised episode in a context (`cue` may be empty). Same
/// rule as `error_gated::step`; the gate decision uses the drive including
/// the cue, on the weights before the update.
pub fn step(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    gate: Gate,
    pattern: &[usize],
    valence: &[bool],
    cue: &[usize],
    cost: &mut PhaseCost,
) -> io::Result<()> {
    let (input, reached, teacher) = encode_ctx(graph, pattern, valence, cue)?;
    let mut update = vec![gate == Gate::Supervised; graph.ids.len()];
    if gate != Gate::Supervised {
        let d = drive(circuit, graph, &input)?;
        for &t in &reached {
            let want = teacher.get(t).map_err(|_| bad("bad target"))?;
            let fires = gates::ge(d[t], THRESHOLD);
            update[t] = match gate {
                Gate::Supervised => true,
                Gate::ErrorGated => fires != want,
                Gate::MarginGated(m) => {
                    if want {
                        !gates::ge(d[t], m)
                    } else {
                        fires
                    }
                }
            };
        }
    }
    cost.episodes += 1;
    cost.targets_seen += reached.len() as u64;
    cost.targets_updated += reached.iter().filter(|&&t| update[t]).count() as u64;
    let r = apply_update_gated(
        circuit,
        graph,
        LearningStrategy::Supervised,
        &input,
        &teacher,
        RewardSignal::None,
        &mut |edge| update[graph.targets[edge] as usize],
    )?;
    cost.attempts += r.edges_visited;
    cost.weights_changed += r.weights_changed;
    Ok(())
}

/// Train a phase (every item in the same context).
pub fn train(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    gate: Gate,
    items: &[(&[usize], &[bool])],
    cue: &[usize],
    cost: &mut PhaseCost,
) -> io::Result<()> {
    for &(p, v) in items {
        step(circuit, graph, gate, p, v, cue, cost)?;
    }
    Ok(())
}

fn tally(c: &mut Confusion, want: bool, got: bool) {
    match (want, got) {
        (true, true) => c.tp += 1,
        (false, false) => c.tn += 1,
        (false, true) => c.fp += 1,
        (true, false) => c.fn_ += 1,
    }
}

/// `error_gated::evaluate_split` in a context (equal to it when `cue` is
/// empty, tested).
pub fn evaluate_split(
    circuit: &PlasticCircuit,
    graph: &Connectome,
    patterns: &[Vec<usize>],
    valence: &[bool],
    other: &[bool],
    cue: &[usize],
) -> io::Result<Split> {
    let mut s = Split::default();
    for p in patterns {
        let (input, reached, teacher) = encode_ctx(graph, p, valence, cue)?;
        let (_, _, teacher_other) = encode(graph, p, other)?;
        let out = circuit.predict(graph, &input, THRESHOLD)?;
        for t in reached {
            let want = teacher.get(t).map_err(|_| bad("bad target"))?;
            let alt = teacher_other.get(t).map_err(|_| bad("bad target"))?;
            let got = out.get(t).map_err(|_| bad("bad target"))?;
            tally(&mut s.all, want, got);
            let bucket = if want == alt {
                &mut s.agree
            } else {
                &mut s.conflict
            };
            tally(bucket, want, got);
        }
    }
    Ok(s)
}

/// (conflict pairs correct in both contexts, conflict pairs) for the final
/// circuit; A judged with `cue_a` on, B with `cue_b` on.
pub fn conflict_both_correct(
    circuit: &PlasticCircuit,
    graph: &Connectome,
    task: &gb::Task,
    cue_a: &[usize],
    cue_b: &[usize],
) -> io::Result<(u64, u64)> {
    let (mut both, mut total) = (0_u64, 0_u64);
    for p in &task.test {
        let (ia, reached, ta) = encode_ctx(graph, p, &task.valence_a, cue_a)?;
        let (ib, _, tb) = encode_ctx(graph, p, &task.valence_b, cue_b)?;
        let oa = circuit.predict(graph, &ia, THRESHOLD)?;
        let ob = circuit.predict(graph, &ib, THRESHOLD)?;
        for t in reached {
            let (wa, wb) = (
                ta.get(t).map_err(|_| bad("bad target"))?,
                tb.get(t).map_err(|_| bad("bad target"))?,
            );
            if wa == wb {
                continue;
            }
            total += 1;
            if oa.get(t).map_err(|_| bad("bad target"))? == wa
                && ob.get(t).map_err(|_| bad("bad target"))? == wb
            {
                both += 1;
            }
        }
    }
    Ok((both, total))
}

/// Structural ceiling of an additive cue (see the doc): per target, only
/// conflict pairs of one direction can be right in both contexts, and the
/// direction A=1,B=0 needs cue A to reach the target (A=0,B=1 needs cue B).
/// Returns (fixable conflict pairs, conflict pairs) on the TEST set.
pub fn ceiling(graph: &Connectome, task: &gb::Task, cue: &Cue) -> io::Result<(u64, u64)> {
    let n = graph.ids.len();
    let reached_by = |lines: &[usize]| {
        let mut r = vec![false; n];
        for &s in lines {
            for t in out_targets(graph, s) {
                r[t] = true;
            }
        }
        r
    };
    let (ra, rb) = (reached_by(&cue.a), reached_by(&cue.b));
    let mut a_only = vec![0_u64; n];
    let mut b_only = vec![0_u64; n];
    for p in &task.test {
        let (_, reached, ta) = encode(graph, p, &task.valence_a)?;
        let (_, _, tb) = encode(graph, p, &task.valence_b)?;
        for t in reached {
            match (
                ta.get(t).map_err(|_| bad("bad target"))?,
                tb.get(t).map_err(|_| bad("bad target"))?,
            ) {
                (true, false) => a_only[t] += 1,
                (false, true) => b_only[t] += 1,
                _ => {}
            }
        }
    }
    let mut fix = 0;
    for t in 0..n {
        let x = if ra[t] { a_only[t] } else { 0 };
        let y = if rb[t] { b_only[t] } else { 0 };
        fix += x.max(y);
    }
    Ok((fix, a_only.iter().sum::<u64>() + b_only.iter().sum::<u64>()))
}

/// One (flip, arm, seed) measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub arm: Arm,
    pub flip: u32,
    pub seed: u64,
    pub cue_lines_a: usize,
    pub cue_lines_b: usize,
    pub cue_reach_a: usize,
    pub cue_reach_b: usize,
    pub targets: usize,
    pub a_train_after_a: Confusion,
    pub a_test_after_a: Split,
    pub a_test_after_b: Split,
    pub b_test_after_b: Split,
    pub conflict_both: (u64, u64),
    pub ceiling: (u64, u64),
    pub cost_a: PhaseCost,
    pub cost_b: PhaseCost,
    pub saturated_after_a: u64,
    pub saturated_after_b: u64,
    /// Mean final weight over the out-edges of the cue lines.
    pub cue_weight_mean: f64,
    pub topology_unchanged: bool,
}

fn frac(x: (u64, u64)) -> f64 {
    if x.1 == 0 {
        0.0
    } else {
        x.0 as f64 / x.1 as f64
    }
}

impl Row {
    #[must_use]
    pub fn forgetting(&self) -> f64 {
        self.a_test_after_a.all.balanced() - self.a_test_after_b.all.balanced()
    }
    #[must_use]
    pub fn forgetting_agree(&self) -> f64 {
        self.a_test_after_a.agree.balanced() - self.a_test_after_b.agree.balanced()
    }
    #[must_use]
    pub fn forgetting_conflict(&self) -> f64 {
        self.a_test_after_a.conflict.balanced() - self.a_test_after_b.conflict.balanced()
    }
    #[must_use]
    pub fn both_frac(&self) -> f64 {
        frac(self.conflict_both)
    }
    #[must_use]
    pub fn ceiling_frac(&self) -> f64 {
        frac(self.ceiling)
    }
}

/// Run A then B for one arm. Uncued arms ignore `cue` (never activated).
pub fn run_arm(
    graph: &Connectome,
    task: &gb::Task,
    spec: &GenSpec,
    arm: Arm,
    seed: u64,
    cue: &Cue,
) -> io::Result<Row> {
    let original = graph.clone();
    let mut circuit = PlasticCircuit::new(graph, spec.init_weight)?;
    let fingerprint = circuit.topology_fingerprint();
    let (va, vb) = (&task.valence_a, &task.valence_b);
    let (ca, cb): (&[usize], &[usize]) = if arm.cued {
        (&cue.a, &cue.b)
    } else {
        (&[], &[])
    };
    let mut cost_a = PhaseCost::default();
    train(
        &mut circuit,
        graph,
        arm.gate,
        &eg::a_items(task, spec, seed),
        ca,
        &mut cost_a,
    )?;
    let mut a_train_after_a = Confusion::default();
    for p in &task.a_train {
        let (input, reached, teacher) = encode_ctx(graph, p, va, ca)?;
        let out = circuit.predict(graph, &input, THRESHOLD)?;
        for t in reached {
            tally(
                &mut a_train_after_a,
                teacher.get(t).map_err(|_| bad("bad target"))?,
                out.get(t).map_err(|_| bad("bad target"))?,
            );
        }
    }
    let a_test_after_a = evaluate_split(&circuit, graph, &task.test, va, vb, ca)?;
    let saturated_after_a = eg::saturated(circuit.weights());
    let mut cost_b = PhaseCost::default();
    train(
        &mut circuit,
        graph,
        arm.gate,
        &eg::b_items(task, spec, seed),
        cb,
        &mut cost_b,
    )?;
    let a_test_after_b = evaluate_split(&circuit, graph, &task.test, va, vb, ca)?;
    let b_test_after_b = evaluate_split(&circuit, graph, &task.test, vb, va, cb)?;
    let conflict_both = conflict_both_correct(&circuit, graph, task, ca, cb)?;
    let cue_edges: Vec<usize> = cue
        .lines()
        .iter()
        .flat_map(|&s| (graph.offsets[s]..graph.offsets[s + 1]).map(|e| e as usize))
        .collect();
    let cue_weight_mean = if cue_edges.is_empty() {
        0.0
    } else {
        cue_edges
            .iter()
            .map(|&e| f64::from(circuit.weights()[e]))
            .sum::<f64>()
            / cue_edges.len() as f64
    };
    let mut targets: Vec<u32> = graph.targets.clone();
    targets.sort_unstable();
    targets.dedup();
    Ok(Row {
        arm,
        flip: spec.flip_pct,
        seed,
        cue_lines_a: cue.a.len(),
        cue_lines_b: cue.b.len(),
        cue_reach_a: cue.reach_a,
        cue_reach_b: cue.reach_b,
        targets: targets.len(),
        a_train_after_a,
        a_test_after_a,
        a_test_after_b,
        b_test_after_b,
        conflict_both,
        ceiling: ceiling(graph, task, cue)?,
        cost_a,
        cost_b,
        saturated_after_a,
        saturated_after_b: eg::saturated(circuit.weights()),
        cue_weight_mean,
        topology_unchanged: *graph == original
            && circuit.topology_fingerprint() == fingerprint
            && circuit.weights().len() == graph.edges(),
    })
}

/// Grid: flip × seed × arm (4 rows per (flip, seed)).
pub fn run_grid(
    graph: Option<&Connectome>,
    shape: &GraphShape,
    spec: &GenSpec,
    flips: &[u32],
    seeds: &[u64],
) -> io::Result<Vec<Row>> {
    if seeds.is_empty() || flips.is_empty() {
        return Err(bad("need at least one seed and one flip rate"));
    }
    let mut rows = Vec::new();
    for &flip in flips {
        let spec = GenSpec {
            flip_pct: flip,
            ..*spec
        };
        for &seed in seeds {
            let g = rl::graph_for(graph, shape, seed)?;
            let cue = choose_cues(&g)?;
            let task = build_task_excluding(&g, &spec, seed, &cue.lines())?;
            for arm in ARMS {
                rows.push(run_arm(&g, &task, &spec, arm, seed, &cue)?);
            }
        }
    }
    Ok(rows)
}

pub const CSV_HEADER: &str = "schema,graph,flip_pct,arm,seed,cue_lines_a,cue_lines_b,cue_reach_a,cue_reach_b,targets,a_train_bal_after_a,a_test_bal_after_a,a_test_bal_after_b,b_test_bal_after_b,forgetting,agree_a_test_after_a,agree_a_test_after_b,agree_b_test,agree_forgetting,conflict_a_test_after_a,conflict_a_test_after_b,conflict_b_test,conflict_forgetting,test_pairs_agree,test_pairs_conflict,conflict_both_correct,conflict_both_frac,ceiling_fixable,ceiling_frac,a_attempts,a_weights_changed,b_attempts,b_weights_changed,a_targets_updated,b_targets_updated,saturated_after_a,saturated_after_b,cue_weight_mean,topology_unchanged";

/// Deterministic CSV (no timing column).
#[must_use]
pub fn to_csv(rows: &[Row], graph_label: &str) -> String {
    let mut out = String::from(CSV_HEADER);
    out.push('\n');
    for r in rows {
        out.push_str(&format!(
            "b888-4-context-cue-v1,{graph_label},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{:.6},{},{:.6},{},{},{},{},{},{},{},{},{:.6},{}\n",
            r.flip,
            r.arm.label(),
            r.seed,
            r.cue_lines_a,
            r.cue_lines_b,
            r.cue_reach_a,
            r.cue_reach_b,
            r.targets,
            r.a_train_after_a.balanced(),
            r.a_test_after_a.all.balanced(),
            r.a_test_after_b.all.balanced(),
            r.b_test_after_b.all.balanced(),
            r.forgetting(),
            r.a_test_after_a.agree.balanced(),
            r.a_test_after_b.agree.balanced(),
            r.b_test_after_b.agree.balanced(),
            r.forgetting_agree(),
            r.a_test_after_a.conflict.balanced(),
            r.a_test_after_b.conflict.balanced(),
            r.b_test_after_b.conflict.balanced(),
            r.forgetting_conflict(),
            r.a_test_after_a.agree.total(),
            r.a_test_after_a.conflict.total(),
            r.conflict_both.0,
            r.both_frac(),
            r.ceiling.0,
            r.ceiling_frac(),
            r.cost_a.attempts,
            r.cost_a.weights_changed,
            r.cost_b.attempts,
            r.cost_b.weights_changed,
            r.cost_a.targets_updated,
            r.cost_b.targets_updated,
            r.saturated_after_a,
            r.saturated_after_b,
            r.cue_weight_mean,
            r.topology_unchanged,
        ));
    }
    out
}

/// Arm `x` paired by seed against arm `base` at one flip rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Comparison {
    pub x: Arm,
    pub base: Arm,
    pub flip: u32,
    /// base − x (positive = x forgets less).
    pub forget: Paired,
    pub forget_agree: Paired,
    pub forget_conflict: Paired,
    /// x − base.
    pub a_test: Paired,
    pub retained: Paired,
    pub b: Paired,
    pub b_conflict: Paired,
    pub both: Paired,
}

impl Comparison {
    /// Pre-registered pass rule (H1/H2/H4).
    #[must_use]
    pub fn passes(&self) -> bool {
        self.forget.strong() && self.a_test.mean >= -EFFECT && self.b.mean >= -EFFECT
    }
    /// Pre-registered conflict-floor rule (H3).
    #[must_use]
    pub fn breaks_floor(&self) -> bool {
        self.forget_conflict.strong() && self.b_conflict.mean >= -EFFECT
    }
}

#[must_use]
pub fn compare(rows: &[Row], flip: u32, x: Arm, base: Arm) -> Comparison {
    let p: Vec<(&Row, &Row)> = rows
        .iter()
        .filter(|r| r.flip == flip && r.arm == x)
        .filter_map(|r| {
            rows.iter()
                .find(|c| c.flip == flip && c.arm == base && c.seed == r.seed)
                .map(|c| (r, c))
        })
        .collect();
    let d =
        |f: &dyn Fn(&Row, &Row) -> f64| paired(&p.iter().map(|(r, c)| f(r, c)).collect::<Vec<_>>());
    Comparison {
        x,
        base,
        flip,
        forget: d(&|r, c| c.forgetting() - r.forgetting()),
        forget_agree: d(&|r, c| c.forgetting_agree() - r.forgetting_agree()),
        forget_conflict: d(&|r, c| c.forgetting_conflict() - r.forgetting_conflict()),
        a_test: d(&|r, c| r.a_test_after_a.all.balanced() - c.a_test_after_a.all.balanced()),
        retained: d(&|r, c| r.a_test_after_b.all.balanced() - c.a_test_after_b.all.balanced()),
        b: d(&|r, c| r.b_test_after_b.all.balanced() - c.b_test_after_b.all.balanced()),
        b_conflict: d(&|r, c| {
            r.b_test_after_b.conflict.balanced() - c.b_test_after_b.conflict.balanced()
        }),
        both: d(&|r, c| r.both_frac() - c.both_frac()),
    }
}

/// The pre-registered comparisons (cued vs same gate uncued) at one flip.
#[must_use]
pub fn cue_comparisons(rows: &[Row], flip: u32) -> Vec<Comparison> {
    [SUPERVISED_CUE, ERROR_GATED_CUE]
        .iter()
        .map(|&x| compare(rows, flip, x, x.base()))
        .collect()
}

/// (H1, H2, H3, H4) as pre-registered.
#[must_use]
pub fn readings(rows: &[Row]) -> (bool, bool, bool, bool) {
    let p = cue_comparisons(rows, PRIMARY_FLIP);
    let h4 = [25, 100]
        .iter()
        .any(|&f| cue_comparisons(rows, f).iter().any(Comparison::passes));
    (
        p[0].passes(),
        p[1].passes(),
        p.iter().any(Comparison::breaks_floor),
        h4,
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn choose_n_cues_prefix_matches_choose_cues() {
        for seed in 1..=6 {
            let g = crate::reward_limits::graph_for(
                None,
                &super::gb::GraphShape {
                    sources: 64,
                    targets: 24,
                    fan_in: 6,
                },
                seed,
            )
            .unwrap();
            let two = super::choose_cues(&g).unwrap();
            let n = super::choose_n_cues(&g, 4).unwrap();
            assert_eq!(n.len(), 4);
            assert_eq!(n[0], two.a);
            assert_eq!(n[1], two.b);
            let mut all: Vec<usize> = n.concat();
            let len = all.len();
            all.sort_unstable();
            all.dedup();
            assert_eq!(all.len(), len, "cues must be disjoint");
        }
    }

    use super::*;
    use crate::generalization_bench::build_task;
    use std::fs;
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

    fn setup(seed: u64) -> (Connectome, Cue, gb::Task) {
        let g = rl::graph_for(None, &shape(), seed).unwrap();
        let cue = choose_cues(&g).unwrap();
        let t = build_task_excluding(&g, &spec(), seed, &cue.lines()).unwrap();
        (g, cue, t)
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("v888-cc-{tag}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn cues_are_disjoint_existing_inputs_and_deterministic() {
        for seed in 1..=4 {
            let (g, cue, task) = setup(seed);
            assert_eq!(cue, choose_cues(&g).unwrap());
            assert!(cue.a.len() <= CUE_CAP && cue.b.len() <= CUE_CAP);
            assert!(cue.a.iter().all(|s| !cue.b.contains(s)));
            for &s in &cue.lines() {
                assert!(!out_targets(&g, s).is_empty(), "cue line has an edge");
                assert!(!task.pool.contains(&s));
                assert!(task
                    .a_train
                    .iter()
                    .chain(&task.b_train)
                    .chain(&task.test)
                    .all(|p| !p.contains(&s)));
            }
            assert_eq!(cue.reach_a, reach(&g, &cue.a));
            assert!(cue.reach_a > 0 && cue.reach_b > 0);
        }
    }

    #[test]
    fn excluding_nothing_is_build_task() {
        let g = rl::graph_for(None, &shape(), 3).unwrap();
        assert_eq!(
            build_task_excluding(&g, &spec(), 3, &[]).unwrap(),
            build_task(&g, &spec(), 3).unwrap()
        );
    }

    #[test]
    fn uncued_arms_are_bit_identical_to_the_117_arms() {
        for seed in [1, 2] {
            let g = rl::graph_for(None, &shape(), seed).unwrap();
            let cue = choose_cues(&g).unwrap();
            let task = build_task(&g, &spec(), seed).unwrap();
            for (arm, gate) in [
                (SUPERVISED, Gate::Supervised),
                (ERROR_GATED, Gate::ErrorGated),
            ] {
                let ours = run_arm(&g, &task, &spec(), arm, seed, &cue).unwrap();
                let theirs = eg::run_arm(&g, &task, &spec(), gate, seed).unwrap();
                assert_eq!(ours.a_train_after_a, theirs.a_train_after_a);
                assert_eq!(ours.a_test_after_a, theirs.a_test_after_a);
                assert_eq!(ours.a_test_after_b, theirs.a_test_after_b);
                assert_eq!(ours.b_test_after_b, theirs.b_test_after_b);
                assert_eq!(ours.cost_a, theirs.cost_a);
                assert_eq!(ours.cost_b, theirs.cost_b);
                assert_eq!(ours.saturated_after_b, theirs.saturated_after_b);
                assert_eq!(ours.conflict_both.0, 0);
            }
        }
    }

    #[test]
    fn empty_cue_split_matches_117_split_and_cue_changes_input() {
        let (g, cue, task) = setup(5);
        let c = PlasticCircuit::new(&g, 1).unwrap();
        let (va, vb) = (&task.valence_a, &task.valence_b);
        assert_eq!(
            evaluate_split(&c, &g, &task.test, va, vb, &[]).unwrap(),
            eg::evaluate_split(&c, &g, &task.test, va, vb).unwrap()
        );
        let (i0, r0, t0) = encode_ctx(&g, &task.test[0], va, &[]).unwrap();
        let (i1, r1, t1) = encode_ctx(&g, &task.test[0], va, &cue.a).unwrap();
        assert_eq!((r0, t0), (r1, t1));
        assert_eq!(
            i1.active_indices().count(),
            i0.active_indices().count() + cue.a.len()
        );
        assert!(encode_ctx(&g, &[cue.a[0]], va, &cue.a).is_err());
    }

    #[test]
    fn topology_is_unchanged_and_only_active_edges_move() {
        let (g, cue, task) = setup(6);
        for arm in ARMS {
            let r = run_arm(&g, &task, &spec(), arm, 6, &cue).unwrap();
            assert!(r.topology_unchanged, "{}", arm.label());
        }
        let before = g.clone();
        let mut c = PlasticCircuit::new(&g, 1).unwrap();
        let fp = c.topology_fingerprint();
        let w0 = c.weights().to_vec();
        let mut cost = PhaseCost::default();
        step(
            &mut c,
            &g,
            Gate::Supervised,
            &task.a_train[0],
            &task.valence_a,
            &cue.a,
            &mut cost,
        )
        .unwrap();
        assert_eq!(g, before);
        assert_eq!(c.topology_fingerprint(), fp);
        assert_eq!(c.weights().len(), g.edges());
        let mut active = task.a_train[0].clone();
        active.extend(&cue.a);
        for (e, (a, b)) in w0.iter().zip(c.weights()).enumerate() {
            if a != b {
                let src = (0..g.ids.len())
                    .find(|&s| (g.offsets[s] as usize..g.offsets[s + 1] as usize).contains(&e))
                    .unwrap();
                assert!(active.contains(&src));
            }
        }
    }

    #[test]
    fn checkpoint_after_a_and_mid_b_resumes_bit_exactly() -> io::Result<()> {
        let (g, cue, task) = setup(8);
        for arm in ARMS {
            let full = run_arm(&g, &task, &spec(), arm, 8, &cue)?;
            let (ca, cb): (&[usize], &[usize]) = if arm.cued {
                (&cue.a, &cue.b)
            } else {
                (&[], &[])
            };
            let mut c = PlasticCircuit::new(&g, 1)?;
            let mut cost_a = PhaseCost::default();
            train(
                &mut c,
                &g,
                arm.gate,
                &eg::a_items(&task, &spec(), 8),
                ca,
                &mut cost_a,
            )?;
            assert_eq!(cost_a, full.cost_a);
            let p = tmp("a");
            c.save(&p)?;
            let mut c = PlasticCircuit::load(&p, &g)?;
            fs::remove_file(&p)?;
            let b = eg::b_items(&task, &spec(), 8);
            let (first, rest) = b.split_at(b.len() / 2);
            let mut cost_b = PhaseCost::default();
            train(&mut c, &g, arm.gate, first, cb, &mut cost_b)?;
            let p = tmp("b");
            c.save(&p)?;
            let mut c = PlasticCircuit::load(&p, &g)?;
            fs::remove_file(&p)?;
            train(&mut c, &g, arm.gate, rest, cb, &mut cost_b)?;
            assert_eq!(cost_b, full.cost_b);
            let (va, vb) = (&task.valence_a, &task.valence_b);
            assert_eq!(
                evaluate_split(&c, &g, &task.test, va, vb, ca)?,
                full.a_test_after_b
            );
            assert_eq!(
                evaluate_split(&c, &g, &task.test, vb, va, cb)?,
                full.b_test_after_b
            );
            assert_eq!(
                conflict_both_correct(&c, &g, &task, ca, cb)?,
                full.conflict_both
            );
            assert_eq!(eg::saturated(c.weights()), full.saturated_after_b);
        }
        Ok(())
    }

    #[test]
    fn both_correct_never_exceeds_the_structural_ceiling() {
        let rows = run_grid(None, &shape(), &spec(), &[25, 100], &[1, 2, 3]).unwrap();
        for r in &rows {
            assert!(
                r.conflict_both.0 <= r.ceiling.0,
                "{} seed {}",
                r.arm.label(),
                r.seed
            );
            assert_eq!(r.conflict_both.1, r.ceiling.1);
            assert_eq!(r.conflict_both.1, r.a_test_after_a.conflict.total());
            if !r.arm.cued {
                assert_eq!(r.conflict_both.0, 0);
            }
        }
    }

    #[test]
    fn grid_is_deterministic_and_csv_complete() {
        let a = run_grid(None, &shape(), &spec(), &[50], &[1, 2]).unwrap();
        let b = run_grid(None, &shape(), &spec(), &[50], &[1, 2]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 2 * ARMS.len());
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        let cols = CSV_HEADER.split(',').count();
        assert!(csv.lines().all(|l| l.split(',').count() == cols));
        assert!(csv.contains(",error-gated+cue,2,"));
        let cmp = cue_comparisons(&a, 50);
        assert_eq!(cmp.len(), 2);
        assert!(cmp.iter().all(|c| c.forget.n == 2));
        let id = compare(&a, 50, SUPERVISED, SUPERVISED);
        assert_eq!(id.forget.ties, 2);
        assert!(!id.passes() && !id.breaks_floor());
        let _ = readings(&a);
        assert!(a.iter().all(|r| r.topology_unchanged));
        assert!(run_grid(None, &shape(), &spec(), &[50], &[]).is_err());
        assert!(run_grid(None, &shape(), &spec(), &[], &[1]).is_err());
    }
}

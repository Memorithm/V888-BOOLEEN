//! B888-4 cycle 10: separate gates for **pattern edges** and **cue edges**
//! under #120's task-context cue, plus a cue-line-count sweep, to explain
//! and address #120's H1 failure (`supervised+cue` lost A-test accuracy).
//!
//! An arm is (pattern-edge gate, cue-edge gate, active lines per cue). The
//! pattern edges are the out-edges of pattern inputs, the cue edges the
//! out-edges of the active cue lines. With equal gates and all lines the
//! arms are bit-identical to #120 (tested). No edge is added or removed.
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph; nothing here is
//! a claim about synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/CUE_GATING.md`
//! (committed before any run).
use crate::connectome::Connectome;
use crate::context_cue::{self as cc, Cue};
use crate::error_gated::{self as eg, drive, paired, Gate, Paired, PhaseCost, Split, EFFECT};
use crate::gates;
use crate::generalization_bench::{
    self as gb, build_task_excluding, Confusion, GenSpec, GraphShape,
};
use crate::learning_strategies::{apply_update_gated, LearningStrategy, RewardSignal};
use crate::plasticity::PlasticCircuit;
use crate::reward_limits as rl;
use std::io::{self, ErrorKind};

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

pub use crate::error_gated::THRESHOLD;
/// Pre-registered label-flip rates (50 is primary).
pub const FLIPS: [u32; 3] = [25, 50, 100];
pub const PRIMARY_FLIP: u32 = 50;
/// Pre-registered seeds: 1..=40 (1..=20 are #120's seeds).
pub const SEEDS: std::ops::RangeInclusive<u64> = 1..=40;

/// One compared arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arm {
    /// Gate of the pattern edges.
    pub pattern: Gate,
    /// Gate of the cue edges; `None` = no cue.
    pub cue: Option<Gate>,
    /// Active lines per cue (greedy-order prefix); `None` = all lines.
    pub lines: Option<usize>,
}

fn letter(g: Gate) -> &'static str {
    match g {
        Gate::Supervised => "S",
        Gate::ErrorGated => "E",
        Gate::MarginGated(_) => "M",
    }
}

impl Arm {
    #[must_use]
    pub fn label(self) -> String {
        match self.cue {
            None => self.pattern.label(),
            Some(c) => {
                let k = self.lines.map(|k| format!("-k{k}")).unwrap_or_default();
                format!("{}/{}-cue{k}", letter(self.pattern), letter(c))
            }
        }
    }
    /// The uncued arm with the same pattern gate.
    #[must_use]
    pub fn control(self) -> Self {
        Self {
            pattern: self.pattern,
            cue: None,
            lines: None,
        }
    }
    /// Active cue lines (cue A prefix, cue B prefix) for this arm.
    #[must_use]
    pub fn active(self, cue: &Cue) -> (Vec<usize>, Vec<usize>) {
        if self.cue.is_none() {
            return (Vec::new(), Vec::new());
        }
        let k = |v: &[usize]| v[..self.lines.unwrap_or(v.len()).min(v.len())].to_vec();
        (k(&cue.a), k(&cue.b))
    }
}

const fn arm(pattern: Gate, cue: Option<Gate>, lines: Option<usize>) -> Arm {
    Arm {
        pattern,
        cue,
        lines,
    }
}
const S: Gate = Gate::Supervised;
const E: Gate = Gate::ErrorGated;
pub const SUPERVISED: Arm = arm(S, None, None);
pub const ERROR_GATED: Arm = arm(E, None, None);
pub const SS: Arm = arm(S, Some(S), None);
pub const SE: Arm = arm(S, Some(E), None);
pub const ES: Arm = arm(E, Some(S), None);
pub const EE: Arm = arm(E, Some(E), None);
pub const SS_K1: Arm = arm(S, Some(S), Some(1));
pub const SS_K2: Arm = arm(S, Some(S), Some(2));
pub const SS_K4: Arm = arm(S, Some(S), Some(4));
/// The pre-registered arms, in CSV order.
pub const ARMS: [Arm; 9] = [SUPERVISED, ERROR_GATED, SS, SE, ES, EE, SS_K1, SS_K2, SS_K4];

/// Does `gate` allow the edges into target `t`? (#117 semantics: the
/// supervised gate updates every target, the others only reached ones.)
fn allows(gate: Gate, reached: bool, want: bool, d: u128) -> bool {
    match gate {
        Gate::Supervised => true,
        Gate::ErrorGated => reached && gates::ge(d, THRESHOLD) != want,
        Gate::MarginGated(m) => {
            reached
                && if want {
                    !gates::ge(d, m)
                } else {
                    gates::ge(d, THRESHOLD)
                }
        }
    }
}

/// Per-edge flag: does the edge leave one of `lines`?
#[must_use]
pub fn edge_mask(graph: &Connectome, lines: &[usize]) -> Vec<bool> {
    let mut m = vec![false; graph.edges()];
    for &s in lines {
        for e in graph.offsets[s]..graph.offsets[s + 1] {
            m[e as usize] = true;
        }
    }
    m
}

/// One episode in a context with separate pattern / cue gates. The gate
/// decision uses the drive including the cue, on the weights before the
/// update. With `pattern_gate == cue_gate` this is `context_cue::step`.
#[allow(clippy::too_many_arguments)]
pub fn step(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    pattern_gate: Gate,
    cue_gate: Gate,
    pattern: &[usize],
    valence: &[bool],
    cue: &[usize],
    cue_edge: &[bool],
    cost: &mut PhaseCost,
) -> io::Result<()> {
    if cue_edge.len() != graph.edges() {
        return Err(bad("cue edge mask shape mismatch"));
    }
    let (input, reached, teacher) = cc::encode_ctx(graph, pattern, valence, cue)?;
    let n = graph.ids.len();
    let d = if pattern_gate == Gate::Supervised && cue_gate == Gate::Supervised {
        vec![0; n]
    } else {
        drive(circuit, graph, &input)?
    };
    let mut is_reached = vec![false; n];
    for &t in &reached {
        is_reached[t] = true;
    }
    let mut upd_p = vec![false; n];
    let mut upd_c = vec![false; n];
    for t in 0..n {
        let want = teacher.get(t).map_err(|_| bad("bad target"))?;
        upd_p[t] = allows(pattern_gate, is_reached[t], want, d[t]);
        upd_c[t] = allows(cue_gate, is_reached[t], want, d[t]);
    }
    cost.episodes += 1;
    cost.targets_seen += reached.len() as u64;
    cost.targets_updated += reached.iter().filter(|&&t| upd_p[t]).count() as u64;
    let r = apply_update_gated(
        circuit,
        graph,
        LearningStrategy::Supervised,
        &input,
        &teacher,
        RewardSignal::None,
        &mut |edge| {
            let t = graph.targets[edge] as usize;
            if cue_edge[edge] {
                upd_c[t]
            } else {
                upd_p[t]
            }
        },
    )?;
    cost.attempts += r.edges_visited;
    cost.weights_changed += r.weights_changed;
    Ok(())
}

/// Train a phase in one context.
#[allow(clippy::too_many_arguments)]
pub fn train(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    arm: Arm,
    items: &[(&[usize], &[bool])],
    cue: &[usize],
    cue_edge: &[bool],
    cost: &mut PhaseCost,
) -> io::Result<()> {
    let cg = arm.cue.unwrap_or(arm.pattern);
    for &(p, v) in items {
        step(circuit, graph, arm.pattern, cg, p, v, cue, cue_edge, cost)?;
    }
    Ok(())
}

fn rate(hit: u64, miss: u64) -> f64 {
    if hit + miss == 0 {
        0.5
    } else {
        hit as f64 / (hit + miss) as f64
    }
}

/// True-positive rate (0.5 if no positive pair).
#[must_use]
pub fn tpr(c: &Confusion) -> f64 {
    rate(c.tp, c.fn_)
}
/// True-negative rate (0.5 if no negative pair).
#[must_use]
pub fn tnr(c: &Confusion) -> f64 {
    rate(c.tn, c.fp)
}

fn mean_weight(circuit: &PlasticCircuit, graph: &Connectome, lines: &[usize]) -> f64 {
    let w: Vec<f64> = lines
        .iter()
        .flat_map(|&s| graph.offsets[s]..graph.offsets[s + 1])
        .map(|e| f64::from(circuit.weights()[e as usize]))
        .collect();
    if w.is_empty() {
        0.0
    } else {
        w.iter().sum::<f64>() / w.len() as f64
    }
}

/// FNV-1a hash of the weights of every edge not leaving a reserved cue line
/// (the pattern edges).
#[must_use]
pub fn pattern_hash(circuit: &PlasticCircuit, all_cue_edges: &[bool]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (e, &w) in circuit.weights().iter().enumerate() {
        if all_cue_edges[e] {
            continue;
        }
        for b in (e as u64).to_le_bytes().into_iter().chain([w]) {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
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
    /// Mean weight of the active cue A / cue B out-edges after A and after B.
    pub cue_a_after_a: f64,
    pub cue_a_after_b: f64,
    pub cue_b_after_a: f64,
    pub cue_b_after_b: f64,
    pub pattern_hash_after_a: u64,
    pub pattern_hash_after_b: u64,
    /// Filled by `run_grid`: pattern-edge weights equal the uncued control
    /// with the same pattern gate after A and after B.
    pub pattern_equals_control: bool,
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

/// Run A then B for one arm. `cue` holds the reserved #120 lines; the arm
/// activates a prefix of them (none if uncued).
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
    let (ca, cb) = arm.active(cue);
    let cue_edge = edge_mask(graph, &[ca.as_slice(), cb.as_slice()].concat());
    let all_cue = edge_mask(graph, &cue.lines());
    let mut cost_a = PhaseCost::default();
    train(
        &mut circuit,
        graph,
        arm,
        &eg::a_items(task, spec, seed),
        &ca,
        &cue_edge,
        &mut cost_a,
    )?;
    let mut a_train_after_a = Confusion::default();
    for p in &task.a_train {
        let (input, reached, teacher) = cc::encode_ctx(graph, p, va, &ca)?;
        let out = circuit.predict(graph, &input, THRESHOLD)?;
        for t in reached {
            let want = teacher.get(t).map_err(|_| bad("bad target"))?;
            let got = out.get(t).map_err(|_| bad("bad target"))?;
            match (want, got) {
                (true, true) => a_train_after_a.tp += 1,
                (false, false) => a_train_after_a.tn += 1,
                (false, true) => a_train_after_a.fp += 1,
                (true, false) => a_train_after_a.fn_ += 1,
            }
        }
    }
    let a_test_after_a = cc::evaluate_split(&circuit, graph, &task.test, va, vb, &ca)?;
    let saturated_after_a = eg::saturated(circuit.weights());
    let (cue_a_after_a, cue_b_after_a) = (
        mean_weight(&circuit, graph, &ca),
        mean_weight(&circuit, graph, &cb),
    );
    let pattern_hash_after_a = pattern_hash(&circuit, &all_cue);
    let mut cost_b = PhaseCost::default();
    train(
        &mut circuit,
        graph,
        arm,
        &eg::b_items(task, spec, seed),
        &cb,
        &cue_edge,
        &mut cost_b,
    )?;
    let a_test_after_b = cc::evaluate_split(&circuit, graph, &task.test, va, vb, &ca)?;
    let b_test_after_b = cc::evaluate_split(&circuit, graph, &task.test, vb, va, &cb)?;
    let conflict_both = cc::conflict_both_correct(&circuit, graph, task, &ca, &cb)?;
    let active = Cue {
        reach_a: reach(graph, &ca),
        reach_b: reach(graph, &cb),
        a: ca.clone(),
        b: cb.clone(),
    };
    Ok(Row {
        arm,
        flip: spec.flip_pct,
        seed,
        cue_lines_a: ca.len(),
        cue_lines_b: cb.len(),
        cue_reach_a: active.reach_a,
        cue_reach_b: active.reach_b,
        a_train_after_a,
        a_test_after_a,
        a_test_after_b,
        b_test_after_b,
        conflict_both,
        ceiling: cc::ceiling(graph, task, &active)?,
        cost_a,
        cost_b,
        saturated_after_a,
        saturated_after_b: eg::saturated(circuit.weights()),
        cue_a_after_a,
        cue_a_after_b: mean_weight(&circuit, graph, &ca),
        cue_b_after_a,
        cue_b_after_b: mean_weight(&circuit, graph, &cb),
        pattern_hash_after_a,
        pattern_hash_after_b: pattern_hash(&circuit, &all_cue),
        pattern_equals_control: arm.cue.is_none(),
        topology_unchanged: *graph == original
            && circuit.topology_fingerprint() == fingerprint
            && circuit.weights().len() == graph.edges(),
    })
}

fn reach(graph: &Connectome, lines: &[usize]) -> usize {
    let mut t: Vec<u32> = lines
        .iter()
        .flat_map(|&s| graph.offsets[s]..graph.offsets[s + 1])
        .filter(|&e| gb::usable(graph, e as usize))
        .map(|e| graph.targets[e as usize])
        .collect();
    t.sort_unstable();
    t.dedup();
    t.len()
}

/// Grid: flip × seed × arm, same task and reserved cue lines for every arm
/// of a (flip, seed).
pub fn run_grid(
    graph: Option<&Connectome>,
    shape: &GraphShape,
    spec: &GenSpec,
    flips: &[u32],
    seeds: &[u64],
    arms: &[Arm],
) -> io::Result<Vec<Row>> {
    if seeds.is_empty() || flips.is_empty() || arms.is_empty() {
        return Err(bad("need at least one seed, flip rate and arm"));
    }
    let mut rows = Vec::new();
    for &flip in flips {
        let spec = GenSpec {
            flip_pct: flip,
            ..*spec
        };
        for &seed in seeds {
            let g = rl::graph_for(graph, shape, seed)?;
            let cue = cc::choose_cues(&g)?;
            let task = build_task_excluding(&g, &spec, seed, &cue.lines())?;
            let mut block: Vec<Row> = arms
                .iter()
                .map(|&a| run_arm(&g, &task, &spec, a, seed, &cue))
                .collect::<io::Result<_>>()?;
            let controls: Vec<(Gate, u64, u64)> = block
                .iter()
                .filter(|r| r.arm.cue.is_none())
                .map(|r| {
                    (
                        r.arm.pattern,
                        r.pattern_hash_after_a,
                        r.pattern_hash_after_b,
                    )
                })
                .collect();
            for r in &mut block {
                if r.arm.cue.is_some() {
                    r.pattern_equals_control = controls.iter().any(|&(g, a, b)| {
                        g == r.arm.pattern
                            && a == r.pattern_hash_after_a
                            && b == r.pattern_hash_after_b
                    });
                }
            }
            rows.extend(block);
        }
    }
    Ok(rows)
}

pub const CSV_HEADER: &str = "schema,graph,flip_pct,arm,pattern_gate,cue_gate,seed,cue_lines_a,cue_lines_b,cue_reach_a,cue_reach_b,a_train_bal_after_a,a_test_bal_after_a,a_test_tpr_after_a,a_test_tnr_after_a,a_test_fn_after_a,a_test_fp_after_a,a_test_bal_after_b,a_test_tpr_after_b,a_test_tnr_after_b,b_test_bal_after_b,forgetting,agree_forgetting,conflict_forgetting,conflict_b_test,test_pairs_agree,test_pairs_conflict,conflict_both_correct,conflict_both_frac,ceiling_fixable,ceiling_frac,a_weights_changed,b_weights_changed,a_targets_updated,b_targets_updated,saturated_after_a,saturated_after_b,cue_a_weight_after_a,cue_a_weight_after_b,cue_b_weight_after_a,cue_b_weight_after_b,pattern_hash_after_a,pattern_hash_after_b,pattern_equals_control,topology_unchanged";

/// Deterministic CSV (no timing column).
#[must_use]
pub fn to_csv(rows: &[Row], graph_label: &str) -> String {
    let mut out = String::from(CSV_HEADER);
    out.push('\n');
    for r in rows {
        let (aa, ab) = (&r.a_test_after_a.all, &r.a_test_after_b.all);
        out.push_str(&format!(
            "b888-4-cue-gating-v1,{graph_label},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{:.6},{},{:.6},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:016x},{:016x},{},{}\n",
            r.flip,
            r.arm.label(),
            letter(r.arm.pattern),
            r.arm.cue.map_or("-", letter),
            r.seed,
            r.cue_lines_a,
            r.cue_lines_b,
            r.cue_reach_a,
            r.cue_reach_b,
            r.a_train_after_a.balanced(),
            aa.balanced(),
            tpr(aa),
            tnr(aa),
            aa.fn_,
            aa.fp,
            ab.balanced(),
            tpr(ab),
            tnr(ab),
            r.b_test_after_b.all.balanced(),
            r.forgetting(),
            r.forgetting_agree(),
            r.forgetting_conflict(),
            r.b_test_after_b.conflict.balanced(),
            r.a_test_after_a.agree.total(),
            r.a_test_after_a.conflict.total(),
            r.conflict_both.0,
            r.both_frac(),
            r.ceiling.0,
            r.ceiling_frac(),
            r.cost_a.weights_changed,
            r.cost_b.weights_changed,
            r.cost_a.targets_updated,
            r.cost_b.targets_updated,
            r.saturated_after_a,
            r.saturated_after_b,
            r.cue_a_after_a,
            r.cue_a_after_b,
            r.cue_b_after_a,
            r.cue_b_after_b,
            r.pattern_hash_after_a,
            r.pattern_hash_after_b,
            r.pattern_equals_control,
            r.topology_unchanged,
        ));
    }
    out
}

/// Arm `x` paired by seed against `base` at one flip, on `seeds` (all if
/// `None`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Comparison {
    pub x: Arm,
    pub base: Arm,
    pub flip: u32,
    /// base − x (positive = x forgets less).
    pub forget: Paired,
    pub forget_conflict: Paired,
    /// x − base.
    pub a_test: Paired,
    pub a_tpr: Paired,
    pub a_tnr: Paired,
    pub retained: Paired,
    pub b: Paired,
    pub both: Paired,
}

impl Comparison {
    /// #120's pre-registered pass rule (unchanged).
    #[must_use]
    pub fn passes(&self) -> bool {
        self.forget.strong() && self.a_test.mean >= -EFFECT && self.b.mean >= -EFFECT
    }
}

fn pairs<'a>(
    rows: &'a [Row],
    flip: u32,
    x: Arm,
    base: Arm,
    seeds: Option<&[u64]>,
) -> Vec<(&'a Row, &'a Row)> {
    rows.iter()
        .filter(|r| r.flip == flip && r.arm == x && seeds.is_none_or(|s| s.contains(&r.seed)))
        .filter_map(|r| {
            rows.iter()
                .find(|c| c.flip == flip && c.arm == base && c.seed == r.seed)
                .map(|c| (r, c))
        })
        .collect()
}

#[must_use]
pub fn compare(rows: &[Row], flip: u32, x: Arm, base: Arm, seeds: Option<&[u64]>) -> Comparison {
    let p = pairs(rows, flip, x, base, seeds);
    let d =
        |f: &dyn Fn(&Row, &Row) -> f64| paired(&p.iter().map(|(r, c)| f(r, c)).collect::<Vec<_>>());
    Comparison {
        x,
        base,
        flip,
        forget: d(&|r, c| c.forgetting() - r.forgetting()),
        forget_conflict: d(&|r, c| c.forgetting_conflict() - r.forgetting_conflict()),
        a_test: d(&|r, c| r.a_test_after_a.all.balanced() - c.a_test_after_a.all.balanced()),
        a_tpr: d(&|r, c| tpr(&r.a_test_after_a.all) - tpr(&c.a_test_after_a.all)),
        a_tnr: d(&|r, c| tnr(&r.a_test_after_a.all) - tnr(&c.a_test_after_a.all)),
        retained: d(&|r, c| r.a_test_after_b.all.balanced() - c.a_test_after_b.all.balanced()),
        b: d(&|r, c| r.b_test_after_b.all.balanced() - c.b_test_after_b.all.balanced()),
        both: d(&|r, c| r.both_frac() - c.both_frac()),
    }
}

/// H4 statistic: per seed, [F(error-gated) − F(E/E-cue)] −
/// [F(supervised) − F(S/E-cue)].
#[must_use]
pub fn h4_attribution(rows: &[Row], flip: u32, seeds: Option<&[u64]>) -> Paired {
    let f = |a: Arm, s: u64| {
        rows.iter()
            .find(|r| r.flip == flip && r.arm == a && r.seed == s)
            .map(Row::forgetting)
    };
    let v: Vec<f64> = rows
        .iter()
        .filter(|r| {
            r.flip == flip && r.arm == SUPERVISED && seeds.is_none_or(|s| s.contains(&r.seed))
        })
        .filter_map(|r| {
            let s = r.seed;
            Some((f(ERROR_GATED, s)? - f(EE, s)?) - (f(SUPERVISED, s)? - f(SE, s)?))
        })
        .collect();
    paired(&v)
}

/// S0: every S-pattern-gate row has pattern weights equal to the control and
/// FN = 0 on A test after A, and its A test is ≤ the control's.
#[must_use]
pub fn s0_holds(rows: &[Row]) -> bool {
    rows.iter()
        .filter(|r| r.arm.pattern == Gate::Supervised)
        .all(|r| {
            let ctl = rows
                .iter()
                .find(|c| c.flip == r.flip && c.seed == r.seed && c.arm == SUPERVISED);
            r.pattern_equals_control
                && r.a_test_after_a.all.fn_ == 0
                && ctl.is_some_and(|c| {
                    r.a_test_after_a.all.balanced() <= c.a_test_after_a.all.balanced()
                })
        })
}

/// Pre-registered readings (S0, H1, H2, H3, H4) at the primary flip.
#[must_use]
pub fn readings(rows: &[Row]) -> (bool, bool, bool, bool, bool) {
    let f = PRIMARY_FLIP;
    (
        s0_holds(rows),
        compare(rows, f, SE, SUPERVISED, None).passes(),
        [SS_K1, SS_K2, SS_K4]
            .iter()
            .any(|&k| compare(rows, f, k, SUPERVISED, None).passes()),
        compare(rows, f, ES, ERROR_GATED, None).passes(),
        h4_attribution(rows, f, None).strong(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let cue = cc::choose_cues(&g).unwrap();
        let t = build_task_excluding(&g, &spec(), seed, &cue.lines()).unwrap();
        (g, cue, t)
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("v888-cg-{tag}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn labels_and_active_lines() {
        let (_, cue, _) = setup(1);
        assert_eq!(SUPERVISED.label(), "supervised");
        assert_eq!(SE.label(), "S/E-cue");
        assert_eq!(SS_K2.label(), "S/S-cue-k2");
        assert_eq!(SUPERVISED.active(&cue), (vec![], vec![]));
        assert_eq!(SS.active(&cue), (cue.a.clone(), cue.b.clone()));
        let (a, b) = SS_K1.active(&cue);
        assert_eq!((a, b), (vec![cue.a[0]], vec![cue.b[0]]));
        let big = arm(S, Some(S), Some(99));
        assert_eq!(big.active(&cue), (cue.a.clone(), cue.b.clone()));
        assert_eq!(SE.control(), SUPERVISED);
        assert_eq!(ES.control(), ERROR_GATED);
    }

    #[test]
    fn equal_gates_are_bit_identical_to_the_120_arms() {
        for seed in [1, 2, 3] {
            let (g, cue, task) = setup(seed);
            for (ours, theirs) in [
                (SUPERVISED, cc::SUPERVISED),
                (ERROR_GATED, cc::ERROR_GATED),
                (SS, cc::SUPERVISED_CUE),
                (EE, cc::ERROR_GATED_CUE),
            ] {
                let a = run_arm(&g, &task, &spec(), ours, seed, &cue).unwrap();
                let b = cc::run_arm(&g, &task, &spec(), theirs, seed, &cue).unwrap();
                assert_eq!(a.a_train_after_a, b.a_train_after_a);
                assert_eq!(a.a_test_after_a, b.a_test_after_a);
                assert_eq!(a.a_test_after_b, b.a_test_after_b);
                assert_eq!(a.b_test_after_b, b.b_test_after_b);
                assert_eq!(a.conflict_both, b.conflict_both);
                assert_eq!(a.cost_a, b.cost_a);
                assert_eq!(a.cost_b, b.cost_b);
                assert_eq!(a.saturated_after_b, b.saturated_after_b);
                if ours.cue.is_some() {
                    assert_eq!(a.ceiling, b.ceiling);
                }
            }
        }
    }

    #[test]
    fn structural_s0_supervised_pattern_gate() {
        let rows = run_grid(
            None,
            &shape(),
            &spec(),
            &[25, 50, 100],
            &[1, 2, 3, 4],
            &ARMS,
        )
        .unwrap();
        assert!(s0_holds(&rows));
        for r in &rows {
            assert!(r.topology_unchanged);
            if r.arm.pattern == Gate::Supervised {
                assert_eq!(r.a_test_after_a.all.fn_, 0, "{}", r.arm.label());
                assert!(r.pattern_equals_control);
            }
        }
    }

    #[test]
    fn cue_edge_error_gate_only_decrements_in_phase_a() {
        // Under an S pattern gate there is no FN in A, so S/E cue edges can
        // only go down during A (init 1 → mean ≤ 1).
        for seed in 1..=4 {
            let (g, cue, task) = setup(seed);
            let r = run_arm(&g, &task, &spec(), SE, seed, &cue).unwrap();
            assert!(r.cue_a_after_a <= 1.0);
            // cue A is off during B, so its edges are frozen in B.
            assert_eq!(r.cue_a_after_a, r.cue_a_after_b);
            assert_eq!(r.cue_b_after_a, 1.0);
        }
    }

    #[test]
    fn checkpoint_resume_is_bit_identical() -> io::Result<()> {
        let (g, cue, task) = setup(7);
        for a in [SE, ES, SS_K2] {
            let full = run_arm(&g, &task, &spec(), a, 7, &cue)?;
            let (ca, cb) = a.active(&cue);
            let mask = edge_mask(&g, &[ca.as_slice(), cb.as_slice()].concat());
            let mut c = PlasticCircuit::new(&g, 1)?;
            let mut cost_a = PhaseCost::default();
            train(
                &mut c,
                &g,
                a,
                &eg::a_items(&task, &spec(), 7),
                &ca,
                &mask,
                &mut cost_a,
            )?;
            assert_eq!(cost_a, full.cost_a);
            let p = tmp("a");
            c.save(&p)?;
            let mut c = PlasticCircuit::load(&p, &g)?;
            fs::remove_file(&p)?;
            let b = eg::b_items(&task, &spec(), 7);
            let (first, rest) = b.split_at(b.len() / 2);
            let mut cost_b = PhaseCost::default();
            train(&mut c, &g, a, first, &cb, &mask, &mut cost_b)?;
            let p = tmp("b");
            c.save(&p)?;
            let mut c = PlasticCircuit::load(&p, &g)?;
            fs::remove_file(&p)?;
            train(&mut c, &g, a, rest, &cb, &mask, &mut cost_b)?;
            assert_eq!(cost_b, full.cost_b);
            let (va, vb) = (&task.valence_a, &task.valence_b);
            assert_eq!(
                cc::evaluate_split(&c, &g, &task.test, va, vb, &ca)?,
                full.a_test_after_b
            );
            assert_eq!(
                cc::evaluate_split(&c, &g, &task.test, vb, va, &cb)?,
                full.b_test_after_b
            );
            let all = edge_mask(&g, &cue.lines());
            assert_eq!(pattern_hash(&c, &all), full.pattern_hash_after_b);
        }
        Ok(())
    }

    #[test]
    fn grid_is_deterministic_and_csv_complete() {
        let a = run_grid(None, &shape(), &spec(), &[50], &[1, 2], &ARMS).unwrap();
        let b = run_grid(None, &shape(), &spec(), &[50], &[1, 2], &ARMS).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 2 * ARMS.len());
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        let cols = CSV_HEADER.split(',').count();
        assert!(csv.lines().all(|l| l.split(',').count() == cols));
        assert!(csv.contains(",S/E-cue,S,E,2,"));
        let id = compare(&a, 50, SUPERVISED, SUPERVISED, None);
        assert_eq!(id.forget.ties, 2);
        assert!(!id.passes());
        assert_eq!(compare(&a, 50, SE, SUPERVISED, Some(&[1])).forget.n, 1);
        assert_eq!(h4_attribution(&a, 50, None).n, 2);
        let _ = readings(&a);
        assert!(run_grid(None, &shape(), &spec(), &[50], &[], &ARMS).is_err());
        assert!(run_grid(None, &shape(), &spec(), &[50], &[1], &[]).is_err());
    }

    #[test]
    fn step_rejects_bad_mask() {
        let (g, cue, task) = setup(1);
        let mut c = PlasticCircuit::new(&g, 1).unwrap();
        let mut cost = PhaseCost::default();
        assert!(step(
            &mut c,
            &g,
            S,
            E,
            &task.a_train[0],
            &task.valence_a,
            &cue.a,
            &[],
            &mut cost
        )
        .is_err());
    }
}

//! B888-4 cycle 11: do the #121 gates keep their benefit on **longer task
//! sequences** (A→B→C→D, one context cue per task), on **held-out seeds**,
//! and on **novel-configuration** test pairs; and does a positive-margin
//! gate on both edge classes improve on the both-error-gated arm?
//!
//! Every arm acts only on the original edges of the graph (no edge added or
//! removed, checked per row). A run trains the tasks in sequence, each with
//! the same episode budget, and evaluates every task on held-out TEST
//! patterns after every task, giving an accuracy matrix R[i][j] (task j
//! measured after training task i). Sequences of length 2 and 3 are the
//! prefixes of the length-4 run.
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph; nothing here is
//! a claim about synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/TASK_SEQUENCE.md`
//! (committed before any run).
use crate::connectome::Connectome;
use crate::context_cue as cc;
use crate::cue_gating::{edge_mask, step};
use crate::error_gated::{paired, Gate, Paired, PhaseCost, EFFECT, THRESHOLD};
use crate::generalization_bench::{self as gb, Confusion, GenSpec, GraphShape};
use crate::plasticity::PlasticCircuit;
use crate::reward_limits as rl;
use std::collections::HashSet;
use std::io::{self, ErrorKind};
use std::ops::Range;

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Number of tasks in a run (sequences of length 2 and 3 are prefixes).
pub const K_MAX: usize = 4;
/// Pre-registered label-flip rates between consecutive tasks (50 primary).
pub const FLIPS: [u32; 3] = [25, 50, 100];
pub const PRIMARY_FLIP: u32 = 50;
/// Pre-registered seeds: 41..=80, never used by an earlier B888-4 cycle.
pub const SEEDS: std::ops::RangeInclusive<u64> = 41..=80;
/// H4 is evaluable only if the median number of novel-configuration test
/// pairs per (row, task) is at least this.
pub const MIN_NOVEL_PAIRS: u64 = 20;
/// Pre-registered margins of the margin-gate sweep.
pub const MARGINS: [u128; 3] = [2, 4, 8];

/// One compared arm: gate of the pattern edges and of the cue edges
/// (`None` = no cue line is ever active).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arm {
    pub pattern: Gate,
    pub cue: Option<Gate>,
}

fn letter(g: Gate) -> String {
    match g {
        Gate::Supervised => "S".to_string(),
        Gate::ErrorGated => "E".to_string(),
        Gate::MarginGated(m) => format!("M{m}"),
    }
}

impl Arm {
    #[must_use]
    pub fn label(self) -> String {
        match self.cue {
            None => self.pattern.label(),
            Some(c) => format!("{}/{}-cue", letter(self.pattern), letter(c)),
        }
    }
}

const S: Gate = Gate::Supervised;
const E: Gate = Gate::ErrorGated;
const fn arm(pattern: Gate, cue: Option<Gate>) -> Arm {
    Arm { pattern, cue }
}
pub const SUPERVISED: Arm = arm(S, None);
pub const ERROR_GATED: Arm = arm(E, None);
pub const MARGIN4: Arm = arm(Gate::MarginGated(4), None);
pub const SE: Arm = arm(S, Some(E));
pub const ES: Arm = arm(E, Some(S));
pub const EE: Arm = arm(E, Some(E));
pub const MM2: Arm = arm(Gate::MarginGated(2), Some(Gate::MarginGated(2)));
pub const MM4: Arm = arm(Gate::MarginGated(4), Some(Gate::MarginGated(4)));
pub const MM8: Arm = arm(Gate::MarginGated(8), Some(Gate::MarginGated(8)));
/// The pre-registered arms, in CSV order.
pub const ARMS: [Arm; 9] = [SUPERVISED, ERROR_GATED, MARGIN4, SE, ES, EE, MM2, MM4, MM8];
/// The margin-sweep arms (H5), in `MARGINS` order.
pub const MARGIN_ARMS: [Arm; 3] = [MM2, MM4, MM8];

/// Deterministic per-seed task sequence: one training set and one valence
/// per task, one shared held-out TEST set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeqTask {
    pub pool: Vec<usize>,
    pub train: Vec<Vec<Vec<usize>>>,
    pub test: Vec<Vec<usize>>,
    pub valence: Vec<Vec<bool>>,
}

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x5E9_B888_4C11_7A5C)
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

/// Draw `k` training sets of `spec.train_patterns`, one TEST set of
/// `spec.test_patterns` (all patterns distinct, none containing an
/// `exclude` node), and a valence chain: task 1 is positive with
/// `positive_pct`, task i+1 flips each edge label of task i with
/// `flip_pct`.
pub fn build_sequence(
    graph: &Connectome,
    spec: &GenSpec,
    seed: u64,
    exclude: &[usize],
    k: usize,
) -> io::Result<SeqTask> {
    graph.validate()?;
    if k == 0
        || spec.active == 0
        || spec.train_patterns == 0
        || spec.test_patterns == 0
        || spec.train_patterns > 4096
        || spec.test_patterns > 4096
        || spec.positive_pct > 100
        || spec.flip_pct > 100
        || spec.rounds_a == 0
        || spec.rounds_a > 64
        || spec.init_weight > 15
    {
        return Err(bad("invalid task-sequence spec"));
    }
    let pool: Vec<usize> = (0..graph.ids.len())
        .filter(|s| !exclude.contains(s))
        .filter(|&s| {
            (graph.offsets[s]..graph.offsets[s + 1]).any(|e| gb::usable(graph, e as usize))
        })
        .collect();
    if pool.len() < spec.active {
        return Err(bad("not enough input neurons for the pattern size"));
    }
    let mut rng = Lcg::new(seed);
    let needed = k * spec.train_patterns + spec.test_patterns;
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
    let test = seen.split_off(k * spec.train_patterns);
    let train: Vec<Vec<Vec<usize>>> = seen
        .chunks(spec.train_patterns)
        .map(<[Vec<usize>]>::to_vec)
        .collect();
    let mut valence: Vec<Vec<bool>> = Vec::with_capacity(k);
    valence.push(
        (0..graph.edges())
            .map(|_| rng.pct(spec.positive_pct))
            .collect(),
    );
    for i in 1..k {
        let next: Vec<bool> = valence[i - 1]
            .iter()
            .map(|&v| if rng.pct(spec.flip_pct) { !v } else { v })
            .collect();
        valence.push(next);
    }
    Ok(SeqTask {
        pool,
        train,
        test,
        valence,
    })
}

/// Per reached target of `pattern`: (target, sorted usable pattern edges
/// into it) — the input configuration the target sees.
#[must_use]
pub fn configurations(graph: &Connectome, pattern: &[usize]) -> Vec<(usize, Vec<usize>)> {
    let mut out: Vec<(usize, Vec<usize>)> = Vec::new();
    for &s in pattern {
        for e in graph.offsets[s]..graph.offsets[s + 1] {
            let e = e as usize;
            if !gb::usable(graph, e) {
                continue;
            }
            let t = graph.targets[e] as usize;
            match out.iter_mut().find(|(x, _)| *x == t) {
                Some((_, v)) => v.push(e),
                None => out.push((t, vec![e])),
            }
        }
    }
    for (_, v) in &mut out {
        v.sort_unstable();
    }
    out
}

/// Every (target, configuration) seen in a training set.
#[must_use]
pub fn seen_configurations(
    graph: &Connectome,
    train: &[Vec<usize>],
) -> HashSet<(usize, Vec<usize>)> {
    train
        .iter()
        .flat_map(|p| configurations(graph, p))
        .collect()
}

fn tally(c: &mut Confusion, want: bool, got: bool) {
    match (want, got) {
        (true, true) => c.tp += 1,
        (false, false) => c.tn += 1,
        (false, true) => c.fp += 1,
        (true, false) => c.fn_ += 1,
    }
}

/// (all pairs, novel-configuration pairs) confusion of one task on the TEST
/// patterns, with `cue` on. A pair is novel if its (target, configuration)
/// never occurred in that task's training set.
pub fn evaluate(
    circuit: &PlasticCircuit,
    graph: &Connectome,
    patterns: &[Vec<usize>],
    valence: &[bool],
    cue: &[usize],
    seen: &HashSet<(usize, Vec<usize>)>,
) -> io::Result<(Confusion, Confusion)> {
    let mut all = Confusion::default();
    let mut novel = Confusion::default();
    for p in patterns {
        let (input, _, teacher) = cc::encode_ctx(graph, p, valence, cue)?;
        let out = circuit.predict(graph, &input, THRESHOLD)?;
        for (t, conf) in configurations(graph, p) {
            let want = teacher.get(t).map_err(|_| bad("bad target"))?;
            let got = out.get(t).map_err(|_| bad("bad target"))?;
            tally(&mut all, want, got);
            if !seen.contains(&(t, conf)) {
                tally(&mut novel, want, got);
            }
        }
    }
    Ok((all, novel))
}

/// Episode order of task `i` (task 0 uses the seed itself).
#[must_use]
pub fn task_order(spec: &GenSpec, seed: u64, i: usize) -> Vec<usize> {
    let s = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    gb::episode_order(spec.train_patterns, spec.rounds_a, s)
}

/// Active cue lines of task `i` for `arm` (none if uncued).
#[must_use]
pub fn active_cue(arm: Arm, cues: &[Vec<usize>], i: usize) -> Vec<usize> {
    if arm.cue.is_some() {
        cues[i].clone()
    } else {
        Vec::new()
    }
}

/// Train episodes `range` of task `i`'s order (checkpoint/resume unit).
#[allow(clippy::too_many_arguments)]
pub fn train_task(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    arm: Arm,
    task: &SeqTask,
    spec: &GenSpec,
    seed: u64,
    i: usize,
    range: Range<usize>,
    cues: &[Vec<usize>],
    cost: &mut PhaseCost,
) -> io::Result<()> {
    if i >= task.train.len() || i >= cues.len() {
        return Err(bad("task index out of range"));
    }
    let order = task_order(spec, seed, i);
    let items = order.get(range).ok_or_else(|| bad("episode range"))?;
    let cue = active_cue(arm, cues, i);
    let mask = edge_mask(graph, &cues.concat());
    let cg = arm.cue.unwrap_or(arm.pattern);
    for &ix in items {
        step(
            circuit,
            graph,
            arm.pattern,
            cg,
            &task.train[i][ix],
            &task.valence[i],
            &cue,
            &mask,
            cost,
        )?;
    }
    Ok(())
}

/// One (flip, arm, seed) measurement of a K_MAX-task run.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub arm: Arm,
    pub flip: u32,
    pub seed: u64,
    pub cue_lines: [usize; K_MAX],
    pub cue_reach: [usize; K_MAX],
    /// `all[i][j]`: task j on TEST after training task i.
    pub all: [[Confusion; K_MAX]; K_MAX],
    pub novel: [[Confusion; K_MAX]; K_MAX],
    pub cost: [PhaseCost; K_MAX],
    pub saturated_end: u64,
    /// Mean weight of the out-edges of cue j after the last task.
    pub cue_weight_end: [f64; K_MAX],
    pub topology_unchanged: bool,
}

fn mean(v: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = v.collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

impl Row {
    #[must_use]
    pub fn r(&self, i: usize, j: usize) -> f64 {
        self.all[i][j].balanced()
    }
    #[must_use]
    pub fn rn(&self, i: usize, j: usize) -> f64 {
        self.novel[i][j].balanced()
    }
    /// Mean final accuracy over the first `k` tasks after task k.
    #[must_use]
    pub fn acc(&self, k: usize) -> f64 {
        mean((0..k).map(|j| self.r(k - 1, j)))
    }
    /// Same on novel-configuration pairs.
    #[must_use]
    pub fn acc_novel(&self, k: usize) -> f64 {
        mean((0..k).map(|j| self.rn(k - 1, j)))
    }
    /// Mean just-learned accuracy R[j][j] over the first `k` tasks.
    #[must_use]
    pub fn learn(&self, k: usize) -> f64 {
        mean((0..k).map(|j| self.r(j, j)))
    }
    /// Mean forgetting R[j][j] − R[k][j] over the first k−1 tasks.
    #[must_use]
    pub fn forgetting(&self, k: usize) -> f64 {
        mean((0..k.saturating_sub(1)).map(|j| self.r(j, j) - self.r(k - 1, j)))
    }
    /// Accuracy of the last task right after it.
    #[must_use]
    pub fn last(&self, k: usize) -> f64 {
        self.r(k - 1, k - 1)
    }
    /// Novel-configuration pairs of task j on TEST (same after every task).
    #[must_use]
    pub fn novel_pairs(&self, j: usize) -> u64 {
        self.novel[0][j].total()
    }
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

fn mean_weight(circuit: &PlasticCircuit, graph: &Connectome, lines: &[usize]) -> f64 {
    mean(
        lines
            .iter()
            .flat_map(|&s| graph.offsets[s]..graph.offsets[s + 1])
            .map(|e| f64::from(circuit.weights()[e as usize])),
    )
}

/// Run the K_MAX tasks in sequence for one arm.
#[allow(clippy::needless_range_loop)]
pub fn run_arm(
    graph: &Connectome,
    task: &SeqTask,
    spec: &GenSpec,
    arm: Arm,
    seed: u64,
    cues: &[Vec<usize>],
) -> io::Result<Row> {
    if task.train.len() != K_MAX || task.valence.len() != K_MAX || cues.len() != K_MAX {
        return Err(bad("need K_MAX tasks and cues"));
    }
    let original = graph.clone();
    let mut circuit = PlasticCircuit::new(graph, spec.init_weight)?;
    let fingerprint = circuit.topology_fingerprint();
    let seen: Vec<HashSet<(usize, Vec<usize>)>> = task
        .train
        .iter()
        .map(|t| seen_configurations(graph, t))
        .collect();
    let episodes = spec.train_patterns * spec.rounds_a as usize;
    let mut all = [[Confusion::default(); K_MAX]; K_MAX];
    let mut novel = [[Confusion::default(); K_MAX]; K_MAX];
    let mut cost = [PhaseCost::default(); K_MAX];
    for i in 0..K_MAX {
        train_task(
            &mut circuit,
            graph,
            arm,
            task,
            spec,
            seed,
            i,
            0..episodes,
            cues,
            &mut cost[i],
        )?;
        for j in 0..K_MAX {
            let (a, n) = evaluate(
                &circuit,
                graph,
                &task.test,
                &task.valence[j],
                &active_cue(arm, cues, j),
                &seen[j],
            )?;
            all[i][j] = a;
            novel[i][j] = n;
        }
    }
    let mut cue_lines = [0; K_MAX];
    let mut cue_reach = [0; K_MAX];
    let mut cue_weight_end = [0.0; K_MAX];
    for j in 0..K_MAX {
        let c = active_cue(arm, cues, j);
        cue_lines[j] = c.len();
        cue_reach[j] = reach(graph, &c);
        cue_weight_end[j] = mean_weight(&circuit, graph, &c);
    }
    Ok(Row {
        arm,
        flip: spec.flip_pct,
        seed,
        cue_lines,
        cue_reach,
        all,
        novel,
        cost,
        saturated_end: crate::error_gated::saturated(circuit.weights()),
        cue_weight_end,
        topology_unchanged: *graph == original
            && circuit.topology_fingerprint() == fingerprint
            && circuit.weights().len() == graph.edges(),
    })
}

/// Grid: flip × seed × arm; every arm of a (flip, seed) sees the same
/// graph, cues, patterns, valences and episode orders.
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
            let cues = cc::choose_n_cues(&g, K_MAX)?;
            let task = build_sequence(&g, &spec, seed, &cues.concat(), K_MAX)?;
            for &a in arms {
                rows.push(run_arm(&g, &task, &spec, a, seed, &cues)?);
            }
        }
    }
    Ok(rows)
}

/// CSV header (generated; see `to_csv`).
#[must_use]
pub fn csv_header() -> String {
    let mut h: Vec<String> = [
        "schema",
        "graph",
        "flip_pct",
        "arm",
        "pattern_gate",
        "cue_gate",
        "seed",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    for j in 1..=K_MAX {
        h.push(format!("cue_lines_{j}"));
    }
    for j in 1..=K_MAX {
        h.push(format!("cue_reach_{j}"));
    }
    h.push("test_pairs".to_string());
    for j in 1..=K_MAX {
        h.push(format!("novel_pairs_{j}"));
    }
    for i in 1..=K_MAX {
        for j in 1..=K_MAX {
            h.push(format!("r_{i}_{j}"));
        }
    }
    for i in 1..=K_MAX {
        for j in 1..=K_MAX {
            h.push(format!("rn_{i}_{j}"));
        }
    }
    for name in ["acc", "acc_novel", "learn", "forgetting", "last"] {
        for k in 2..=K_MAX {
            h.push(format!("{name}_k{k}"));
        }
    }
    for j in 1..=K_MAX {
        h.push(format!("weights_changed_{j}"));
    }
    for j in 1..=K_MAX {
        h.push(format!("targets_updated_{j}"));
    }
    for j in 1..=K_MAX {
        h.push(format!("cue_weight_end_{j}"));
    }
    h.push("saturated_end".to_string());
    h.push("topology_unchanged".to_string());
    h.join(",")
}

/// Deterministic CSV (no timing column).
#[must_use]
pub fn to_csv(rows: &[Row], graph_label: &str) -> String {
    let mut out = csv_header();
    out.push('\n');
    for r in rows {
        let mut f: Vec<String> = vec![
            "b888-4-task-sequence-v1".to_string(),
            graph_label.to_string(),
            r.flip.to_string(),
            r.arm.label(),
            letter(r.arm.pattern),
            r.arm.cue.map_or("-".to_string(), letter),
            r.seed.to_string(),
        ];
        f.extend(r.cue_lines.iter().map(ToString::to_string));
        f.extend(r.cue_reach.iter().map(ToString::to_string));
        f.push(r.all[0][0].total().to_string());
        f.extend((0..K_MAX).map(|j| r.novel_pairs(j).to_string()));
        for i in 0..K_MAX {
            f.extend((0..K_MAX).map(|j| format!("{:.6}", r.r(i, j))));
        }
        for i in 0..K_MAX {
            f.extend((0..K_MAX).map(|j| format!("{:.6}", r.rn(i, j))));
        }
        for m in [
            Row::acc,
            Row::acc_novel,
            Row::learn,
            Row::forgetting,
            Row::last,
        ] {
            f.extend((2..=K_MAX).map(|k| format!("{:.6}", m(r, k))));
        }
        f.extend(r.cost.iter().map(|c| c.weights_changed.to_string()));
        f.extend(r.cost.iter().map(|c| c.targets_updated.to_string()));
        f.extend(r.cue_weight_end.iter().map(|w| format!("{w:.6}")));
        f.push(r.saturated_end.to_string());
        f.push(r.topology_unchanged.to_string());
        out.push_str(&f.join(","));
        out.push('\n');
    }
    out
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

/// Arm `x` vs `base`, paired by seed, at one flip and sequence length `k`.
/// Every field is oriented so that > 0 favours `x`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Comparison {
    pub x: Arm,
    pub base: Arm,
    pub flip: u32,
    pub k: usize,
    pub acc: Paired,
    pub acc_novel: Paired,
    pub learn: Paired,
    pub last: Paired,
    /// base − x.
    pub forget: Paired,
    /// R[k][1] (first task retained after task k).
    pub first_retained: Paired,
    /// R[1][1] (first task right after it).
    pub first_learn: Paired,
}

impl Comparison {
    /// Pre-registered sequence rule: mean final accuracy over the k tasks
    /// strongly better, and neither the last task nor the mean just-learned
    /// accuracy drops by more than `EFFECT`.
    #[must_use]
    pub fn passes(&self) -> bool {
        self.acc.strong() && self.last.mean >= -EFFECT && self.learn.mean >= -EFFECT
    }
    /// #120/#121 two-task rule (meaningful at k = 2): forgetting reduction
    /// strong, A test after A and B test after B drops ≤ `EFFECT`.
    #[must_use]
    pub fn passes_pair_rule(&self) -> bool {
        self.forget.strong() && self.first_learn.mean >= -EFFECT && self.last.mean >= -EFFECT
    }
}

#[must_use]
pub fn compare(
    rows: &[Row],
    flip: u32,
    k: usize,
    x: Arm,
    base: Arm,
    seeds: Option<&[u64]>,
) -> Comparison {
    let p = pairs(rows, flip, x, base, seeds);
    let d =
        |f: &dyn Fn(&Row, &Row) -> f64| paired(&p.iter().map(|(r, c)| f(r, c)).collect::<Vec<_>>());
    Comparison {
        x,
        base,
        flip,
        k,
        acc: d(&|r, c| r.acc(k) - c.acc(k)),
        acc_novel: d(&|r, c| r.acc_novel(k) - c.acc_novel(k)),
        learn: d(&|r, c| r.learn(k) - c.learn(k)),
        last: d(&|r, c| r.last(k) - c.last(k)),
        forget: d(&|r, c| c.forgetting(k) - r.forgetting(k)),
        first_retained: d(&|r, c| r.r(k - 1, 0) - c.r(k - 1, 0)),
        first_learn: d(&|r, c| r.r(0, 0) - c.r(0, 0)),
    }
}

/// Median number of novel-configuration TEST pairs per (row, task) at a flip.
#[must_use]
pub fn median_novel_pairs(rows: &[Row], flip: u32) -> u64 {
    let mut v: Vec<u64> = rows
        .iter()
        .filter(|r| r.flip == flip)
        .flat_map(|r| (0..K_MAX).map(move |j| r.novel_pairs(j)))
        .collect();
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

/// Pre-registered readings at the primary flip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Readings {
    /// E/E-cue vs error-gated, two-task rule, held-out seeds.
    pub h1: bool,
    /// E/E-cue vs error-gated, sequence rule at k = 3 and k = 4.
    pub h2: bool,
    /// E/E-cue vs supervised, sequence rule at k = 3 and k = 4.
    pub h3: bool,
    /// E/E-cue vs error-gated on novel-configuration pairs (strong ΔACC at
    /// k = 3 and 4); `None` if not evaluable.
    pub h4: Option<bool>,
    /// Some M_m/M_m-cue passes the sequence rule vs E/E-cue at k = 3.
    pub h5: bool,
}

#[must_use]
pub fn readings(rows: &[Row]) -> Readings {
    let f = PRIMARY_FLIP;
    let both = |x: Arm, base: Arm| {
        [3, 4]
            .iter()
            .all(|&k| compare(rows, f, k, x, base, None).passes())
    };
    let h4 = (median_novel_pairs(rows, f) >= MIN_NOVEL_PAIRS).then(|| {
        [3, 4].iter().all(|&k| {
            compare(rows, f, k, EE, ERROR_GATED, None)
                .acc_novel
                .strong()
        })
    });
    Readings {
        h1: compare(rows, f, 2, EE, ERROR_GATED, None).passes_pair_rule(),
        h2: both(EE, ERROR_GATED),
        h3: both(EE, SUPERVISED),
        h4,
        h5: MARGIN_ARMS
            .iter()
            .any(|&m| compare(rows, f, 3, m, EE, None).passes()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn shape() -> GraphShape {
        GraphShape {
            sources: 64,
            targets: 16,
            fan_in: 6,
        }
    }

    fn spec() -> GenSpec {
        GenSpec {
            active: 3,
            train_patterns: 12,
            test_patterns: 16,
            rounds_a: 3,
            rounds_b: 3,
            positive_pct: 50,
            flip_pct: 50,
            init_weight: 1,
        }
    }

    fn setup(seed: u64) -> (Connectome, Vec<Vec<usize>>, SeqTask) {
        let g = rl::graph_for(None, &shape(), seed).unwrap();
        let cues = cc::choose_n_cues(&g, K_MAX).unwrap();
        let t = build_sequence(&g, &spec(), seed, &cues.concat(), K_MAX).unwrap();
        (g, cues, t)
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("v888-ts-{tag}-{}-{nonce}", std::process::id()))
    }

    /// Feasibility of the pre-registered grid (no training): every bench
    /// seed supplies K_MAX disjoint cues and enough distinct patterns.
    #[test]
    fn bench_seeds_are_feasible() {
        let shape = GraphShape {
            sources: 64,
            targets: 24,
            fan_in: 6,
        };
        let spec = GenSpec {
            active: 4,
            train_patterns: 32,
            test_patterns: 64,
            rounds_a: 8,
            rounds_b: 8,
            positive_pct: 50,
            flip_pct: 50,
            init_weight: 1,
        };
        for seed in SEEDS {
            let g = rl::graph_for(None, &shape, seed).unwrap();
            let cues = cc::choose_n_cues(&g, K_MAX).unwrap();
            assert!(cues.iter().all(|c| !c.is_empty() && c.len() <= cc::CUE_CAP));
            build_sequence(&g, &spec, seed, &cues.concat(), K_MAX).unwrap();
        }
    }

    #[test]
    fn labels() {
        assert_eq!(SUPERVISED.label(), "supervised");
        assert_eq!(MARGIN4.label(), "margin-gated-4");
        assert_eq!(EE.label(), "E/E-cue");
        assert_eq!(MM8.label(), "M8/M8-cue");
        let mut l: Vec<String> = ARMS.iter().map(|a| a.label()).collect();
        l.sort();
        l.dedup();
        assert_eq!(l.len(), ARMS.len());
    }

    #[test]
    fn sequence_is_deterministic_disjoint_and_chained() {
        let (g, cues, t) = setup(3);
        let (_, _, t2) = setup(3);
        assert_eq!(t, t2);
        assert_eq!(t.train.len(), K_MAX);
        let mut all: Vec<&Vec<usize>> = t.train.iter().flatten().chain(&t.test).collect();
        let n = all.len();
        assert_eq!(n, K_MAX * 12 + 16);
        all.sort();
        all.dedup();
        assert_eq!(all.len(), n);
        let lines = cues.concat();
        assert!(t
            .train
            .iter()
            .flatten()
            .chain(&t.test)
            .all(|p| p.iter().all(|s| !lines.contains(s))));
        // flip 100 alternates, flip 0 repeats
        let s100 = GenSpec {
            flip_pct: 100,
            ..spec()
        };
        let t100 = build_sequence(&g, &s100, 3, &lines, K_MAX).unwrap();
        assert!(t100.valence[0]
            .iter()
            .zip(&t100.valence[1])
            .all(|(a, b)| a != b));
        assert_eq!(t100.valence[0], t100.valence[2]);
        let s0 = GenSpec {
            flip_pct: 0,
            ..spec()
        };
        let t0 = build_sequence(&g, &s0, 3, &lines, K_MAX).unwrap();
        assert!(t0.valence.iter().all(|v| *v == t0.valence[0]));
        assert!(build_sequence(&g, &spec(), 3, &lines, 0).is_err());
    }

    #[test]
    fn novel_split_is_correct() {
        let (g, _, t) = setup(5);
        // A training pattern's own configurations are never novel.
        let seen = seen_configurations(&g, &t.train[0]);
        let c = PlasticCircuit::new(&g, 1).unwrap();
        let (all, novel) = evaluate(&c, &g, &t.train[0], &t.valence[0], &[], &seen).unwrap();
        assert!(all.total() > 0);
        assert_eq!(novel.total(), 0);
        // With nothing seen, every pair is novel.
        let (all, novel) = evaluate(&c, &g, &t.test, &t.valence[0], &[], &HashSet::new()).unwrap();
        assert_eq!(all, novel);
        // Pair count equals the reached-target count of `encode`.
        let mut n = 0;
        for p in &t.test {
            n += gb::encode(&g, p, &t.valence[0]).unwrap().1.len() as u64;
        }
        assert_eq!(all.total(), n);
    }

    #[test]
    fn margin_one_equals_error_gate_and_uncued_ignores_cues() {
        let (g, cues, t) = setup(2);
        let m1 = arm(Gate::MarginGated(1), Some(Gate::MarginGated(1)));
        let a = run_arm(&g, &t, &spec(), m1, 2, &cues).unwrap();
        let b = run_arm(&g, &t, &spec(), EE, 2, &cues).unwrap();
        assert_eq!(a.all, b.all);
        assert_eq!(a.cost, b.cost);
        let none: Vec<Vec<usize>> = vec![vec![cues[0][0]]; K_MAX];
        let u1 = run_arm(&g, &t, &spec(), SUPERVISED, 2, &cues).unwrap();
        let u2 = run_arm(&g, &t, &spec(), SUPERVISED, 2, &none).unwrap();
        assert_eq!(u1.all, u2.all);
        assert_eq!(u1.cue_lines, [0; K_MAX]);
    }

    #[test]
    fn error_gated_cue_edges_are_frozen_after_their_task() {
        // Cue j is active only during task j (and its evaluations), so its
        // out-edges cannot change after task j.
        let (g, cues, t) = setup(4);
        let episodes = 12 * 3;
        let mut c = PlasticCircuit::new(&g, 1).unwrap();
        let mut snaps: Vec<Vec<u8>> = Vec::new();
        for i in 0..K_MAX {
            let mut cost = PhaseCost::default();
            train_task(
                &mut c,
                &g,
                EE,
                &t,
                &spec(),
                4,
                i,
                0..episodes,
                &cues,
                &mut cost,
            )
            .unwrap();
            snaps.push(c.weights().to_vec());
        }
        for (j, cue) in cues.iter().enumerate() {
            let m = edge_mask(&g, cue);
            for snap in snaps.iter().skip(j + 1) {
                for (e, &is) in m.iter().enumerate() {
                    if is {
                        assert_eq!(snap[e], snaps[j][e]);
                    }
                }
            }
        }
    }

    #[test]
    fn row_metrics_match_definitions() {
        let (g, cues, t) = setup(6);
        let r = run_arm(&g, &t, &spec(), EE, 6, &cues).unwrap();
        assert!(r.topology_unchanged);
        let k = 3;
        let acc = (r.r(2, 0) + r.r(2, 1) + r.r(2, 2)) / 3.0;
        assert!((r.acc(k) - acc).abs() < 1e-12);
        let f = ((r.r(0, 0) - r.r(2, 0)) + (r.r(1, 1) - r.r(2, 1))) / 2.0;
        assert!((r.forgetting(k) - f).abs() < 1e-12);
        assert!((r.forgetting(2) - (r.r(0, 0) - r.r(1, 0))).abs() < 1e-12);
        assert_eq!(r.last(k), r.r(2, 2));
        for j in 0..K_MAX {
            assert_eq!(r.novel_pairs(j), r.novel[3][j].total());
        }
    }

    #[test]
    fn checkpoint_resume_is_bit_identical() -> io::Result<()> {
        let (g, cues, t) = setup(7);
        let episodes = 12 * 3;
        for a in [EE, MM4, SE] {
            let full = run_arm(&g, &t, &spec(), a, 7, &cues)?;
            let mut c = PlasticCircuit::new(&g, 1)?;
            let mut cost = [PhaseCost::default(); K_MAX];
            for (i, ci) in cost.iter_mut().enumerate().take(2) {
                train_task(&mut c, &g, a, &t, &spec(), 7, i, 0..episodes, &cues, ci)?;
            }
            let p = tmp("k2");
            c.save(&p)?;
            let mut c = PlasticCircuit::load(&p, &g)?;
            fs::remove_file(&p)?;
            let half = episodes / 2;
            train_task(
                &mut c,
                &g,
                a,
                &t,
                &spec(),
                7,
                2,
                0..half,
                &cues,
                &mut cost[2],
            )?;
            let p = tmp("k3");
            c.save(&p)?;
            let mut c = PlasticCircuit::load(&p, &g)?;
            fs::remove_file(&p)?;
            train_task(
                &mut c,
                &g,
                a,
                &t,
                &spec(),
                7,
                2,
                half..episodes,
                &cues,
                &mut cost[2],
            )?;
            train_task(
                &mut c,
                &g,
                a,
                &t,
                &spec(),
                7,
                3,
                0..episodes,
                &cues,
                &mut cost[3],
            )?;
            assert_eq!(cost, full.cost);
            for j in 0..K_MAX {
                let seen = seen_configurations(&g, &t.train[j]);
                let (all, novel) = evaluate(
                    &c,
                    &g,
                    &t.test,
                    &t.valence[j],
                    &active_cue(a, &cues, j),
                    &seen,
                )?;
                assert_eq!(all, full.all[3][j]);
                assert_eq!(novel, full.novel[3][j]);
            }
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
        let cols = csv_header().split(',').count();
        assert!(csv.lines().all(|l| l.split(',').count() == cols));
        assert!(csv.contains(",M4/M4-cue,M4,M4,"));
        let id = compare(&a, 50, 3, SUPERVISED, SUPERVISED, None);
        assert_eq!(id.acc.ties, 2);
        assert!(!id.passes());
        assert!(!id.passes_pair_rule());
        assert_eq!(compare(&a, 50, 4, EE, SUPERVISED, Some(&[1])).acc.n, 1);
        let _ = readings(&a);
        assert!(median_novel_pairs(&a, 50) <= a[0].all[0][0].total());
        assert!(run_grid(None, &shape(), &spec(), &[50], &[], &ARMS).is_err());
        assert!(run_grid(None, &shape(), &spec(), &[50], &[1], &[]).is_err());
    }
}

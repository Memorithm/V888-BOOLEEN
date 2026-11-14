//! B888-4 cycle 12: a task-sequence design in which **generalisation to
//! never-seen input combinations can be measured**, and the measurement.
//!
//! Cycle 11 (#122) could not evaluate generalisation: four reserved cues of
//! up to 12 lines left ~25 pattern inputs, so 98 % of TEST pairs repeated a
//! training configuration. This module widens the graph, caps each cue at a
//! few lines, and splits every TEST pair of task j into three classes with
//! respect to task j's training set:
//!
//! * **seen**: the (target, configuration) occurred in training;
//! * **compositional novel (CN)**: the configuration never occurred, but
//!   every edge in it was active in some training configuration of that
//!   target — the label is then fixed by per-edge valences that were trained;
//! * **unseen edge (UE)**: at least one edge was never active in training.
//!
//! The target rule is the OR of per-edge valences
//! ([`crate::generalization_bench`]), so a CN label is determined by trained
//! quantities and generalisation is possible in principle; a pure lookup
//! table of training configurations scores chance (0.5 balanced) on CN.
//! [`TaskSplit::inferable`] marks the CN pairs whose label is logically
//! implied by the training labels under the OR rule (ICN).
//!
//! Every arm acts only on the original edges (checked per row).
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph; nothing here is
//! a claim about synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/NOVEL_SPLIT.md`
//! (committed before any run).
use crate::connectome::Connectome;
use crate::context_cue as cc;
use crate::error_gated::{paired, Gate, Paired, PhaseCost, EFFECT, THRESHOLD};
use crate::generalization_bench::{self as gb, Confusion, GenSpec, GraphShape};
use crate::plasticity::PlasticCircuit;
use crate::reward_limits as rl;
use crate::task_sequence::{self as ts, Arm, SeqTask, K_MAX};
use std::collections::{HashMap, HashSet};
use std::io::{self, ErrorKind};

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Pre-registered label-flip rates between consecutive tasks (50 primary).
pub const FLIPS: [u32; 3] = [25, 50, 100];
pub const PRIMARY_FLIP: u32 = 50;
/// Pre-registered seeds: 81..=120, never used by an earlier B888-4 cycle.
pub const SEEDS: std::ops::RangeInclusive<u64> = 81..=120;
/// Design-validity gate: a hypothesis on a pair class is evaluable only if
/// the median number of TEST pairs of that class per (seed, task) is at
/// least this.
pub const MIN_NOVEL_PAIRS: u64 = 20;
/// H5 bound on the seen − CN accuracy gap of `E/E-cue`.
pub const GAP_BOUND: f64 = 0.10;

/// The pre-registered design (graph shape, task, budget, cue cap).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Design {
    pub shape: GraphShape,
    pub spec: GenSpec,
    pub cue_cap: usize,
}

/// Cycle-12 primary design (see `docs/NOVEL_SPLIT.md`).
pub const DESIGN: Design = Design {
    shape: GraphShape {
        sources: 128,
        targets: 32,
        fan_in: 8,
    },
    spec: GenSpec {
        active: 6,
        train_patterns: 32,
        test_patterns: 128,
        rounds_a: 8,
        rounds_b: 8,
        positive_pct: 50,
        flip_pct: PRIMARY_FLIP,
        init_weight: 1,
    },
    cue_cap: 4,
};

pub const SUPERVISED: Arm = ts::SUPERVISED;
pub const ERROR_GATED: Arm = ts::ERROR_GATED;
pub const SS: Arm = Arm {
    pattern: Gate::Supervised,
    cue: Some(Gate::Supervised),
};
pub const EE: Arm = ts::EE;
/// The pre-registered arms, in CSV order.
pub const ARMS: [Arm; 4] = [SUPERVISED, ERROR_GATED, SS, EE];

/// Class of one (TEST pattern, reached target) pair w.r.t. one task's
/// training set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Seen,
    Compositional,
    UnseenEdge,
}

/// What task j's training set reveals: seen configurations, edges ever
/// active, and edges whose valence the OR-rule labels imply.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TaskSplit {
    pub seen: HashSet<(usize, Vec<usize>)>,
    pub edges: HashSet<usize>,
    pub known_negative: HashSet<usize>,
    pub known_positive: HashSet<usize>,
}

fn label(conf: &[usize], valence: &[bool]) -> bool {
    conf.iter().any(|&e| valence[e])
}

impl TaskSplit {
    /// Build from training patterns and the task's valence.
    #[must_use]
    pub fn new(graph: &Connectome, train: &[Vec<usize>], valence: &[bool]) -> Self {
        let mut s = Self::default();
        let mut positives: Vec<Vec<usize>> = Vec::new();
        for p in train {
            for (t, conf) in ts::configurations(graph, p) {
                s.edges.extend(conf.iter().copied());
                if label(&conf, valence) {
                    positives.push(conf.clone());
                } else {
                    s.known_negative.extend(conf.iter().copied());
                }
                s.seen.insert((t, conf));
            }
        }
        // OR rule: a positive configuration whose other edges are all known
        // negative implies its remaining edge is positive. Known negatives
        // come only from negative labels, so one pass is the fixpoint.
        for conf in positives {
            let open: Vec<usize> = conf
                .iter()
                .copied()
                .filter(|e| !s.known_negative.contains(e))
                .collect();
            if open.len() == 1 {
                s.known_positive.insert(open[0]);
            }
        }
        s
    }

    #[must_use]
    pub fn classify(&self, target: usize, conf: &[usize]) -> Class {
        if self.seen.contains(&(target, conf.to_vec())) {
            Class::Seen
        } else if conf.iter().all(|e| self.edges.contains(e)) {
            Class::Compositional
        } else {
            Class::UnseenEdge
        }
    }

    /// Is the label of `conf` implied by the training labels (OR rule)?
    #[must_use]
    pub fn inferable(&self, conf: &[usize], valence: &[bool]) -> bool {
        if label(conf, valence) {
            conf.iter().any(|e| self.known_positive.contains(e))
        } else {
            conf.iter().all(|e| self.known_negative.contains(e))
        }
    }
}

/// Confusions of one task on TEST, per pair class.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Eval {
    pub all: Confusion,
    pub seen: Confusion,
    pub cn: Confusion,
    /// Inferable subset of `cn`.
    pub icn: Confusion,
    pub ue: Confusion,
}

fn tally(c: &mut Confusion, want: bool, got: bool) {
    match (want, got) {
        (true, true) => c.tp += 1,
        (false, false) => c.tn += 1,
        (false, true) => c.fp += 1,
        (true, false) => c.fn_ += 1,
    }
}

/// Evaluate one task on `patterns` with `cue` on.
pub fn evaluate(
    circuit: &PlasticCircuit,
    graph: &Connectome,
    patterns: &[Vec<usize>],
    valence: &[bool],
    cue: &[usize],
    split: &TaskSplit,
) -> io::Result<Eval> {
    let mut ev = Eval::default();
    for p in patterns {
        let (input, _, teacher) = cc::encode_ctx(graph, p, valence, cue)?;
        let out = circuit.predict(graph, &input, THRESHOLD)?;
        for (t, conf) in ts::configurations(graph, p) {
            let want = teacher.get(t).map_err(|_| bad("bad target"))?;
            let got = out.get(t).map_err(|_| bad("bad target"))?;
            if want != label(&conf, valence) {
                return Err(bad("teacher disagrees with OR rule"));
            }
            tally(&mut ev.all, want, got);
            match split.classify(t, &conf) {
                Class::Seen => tally(&mut ev.seen, want, got),
                Class::Compositional => {
                    tally(&mut ev.cn, want, got);
                    if split.inferable(&conf, valence) {
                        tally(&mut ev.icn, want, got);
                    }
                }
                Class::UnseenEdge => tally(&mut ev.ue, want, got),
            }
        }
    }
    Ok(ev)
}

/// Pair class used by the per-class metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slice {
    All,
    Seen,
    Cn,
    Icn,
    Ue,
}

impl Slice {
    #[must_use]
    pub fn pick(self, e: &Eval) -> Confusion {
        match self {
            Self::All => e.all,
            Self::Seen => e.seen,
            Self::Cn => e.cn,
            Self::Icn => e.icn,
            Self::Ue => e.ue,
        }
    }
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Seen => "seen",
            Self::Cn => "cn",
            Self::Icn => "icn",
            Self::Ue => "ue",
        }
    }
}

pub const SLICES: [Slice; 5] = [Slice::All, Slice::Seen, Slice::Cn, Slice::Icn, Slice::Ue];

/// One (flip, arm, seed) measurement of a K_MAX-task run.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub arm: Arm,
    pub flip: u32,
    pub seed: u64,
    pub pool: usize,
    pub cue_lines: [usize; K_MAX],
    pub cue_reach: [usize; K_MAX],
    /// `ev[i][j]`: task j on TEST after training task i.
    pub ev: [[Eval; K_MAX]; K_MAX],
    pub cost: [PhaseCost; K_MAX],
    pub saturated_end: u64,
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
    pub fn r(&self, s: Slice, i: usize, j: usize) -> f64 {
        s.pick(&self.ev[i][j]).balanced()
    }
    /// Mean final accuracy over the first `k` tasks after task k.
    #[must_use]
    pub fn acc(&self, s: Slice, k: usize) -> f64 {
        mean((0..k).map(|j| self.r(s, k - 1, j)))
    }
    /// Mean just-learned accuracy R[j][j] over the first `k` tasks.
    #[must_use]
    pub fn learn(&self, s: Slice, k: usize) -> f64 {
        mean((0..k).map(|j| self.r(s, j, j)))
    }
    /// Mean forgetting R[j][j] − R[k][j] over the first k−1 tasks.
    #[must_use]
    pub fn forgetting(&self, s: Slice, k: usize) -> f64 {
        mean((0..k.saturating_sub(1)).map(|j| self.r(s, j, j) - self.r(s, k - 1, j)))
    }
    /// Accuracy of the last task right after it.
    #[must_use]
    pub fn last(&self, s: Slice, k: usize) -> f64 {
        self.r(s, k - 1, k - 1)
    }
    /// TEST pairs of slice `s` for task j (same after every task).
    #[must_use]
    pub fn pairs(&self, s: Slice, j: usize) -> u64 {
        s.pick(&self.ev[0][j]).total()
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

/// Per-seed fixture: graph, cues, tasks and per-task splits. Every arm of a
/// (flip, seed) sees exactly this.
#[derive(Clone, Debug)]
pub struct Fixture {
    pub graph: Connectome,
    pub cues: Vec<Vec<usize>>,
    pub task: SeqTask,
    pub splits: Vec<TaskSplit>,
}

pub fn fixture(
    graph: Option<&Connectome>,
    design: &Design,
    flip: u32,
    seed: u64,
) -> io::Result<Fixture> {
    let spec = GenSpec {
        flip_pct: flip,
        ..design.spec
    };
    let g = rl::graph_for(graph, &design.shape, seed)?;
    let cues = cc::choose_n_cues_capped(&g, K_MAX, design.cue_cap)?;
    let task = ts::build_sequence(&g, &spec, seed, &cues.concat(), K_MAX)?;
    let splits = (0..K_MAX)
        .map(|j| TaskSplit::new(&g, &task.train[j], &task.valence[j]))
        .collect();
    Ok(Fixture {
        graph: g,
        cues,
        task,
        splits,
    })
}

/// Run the K_MAX tasks in sequence for one arm on a fixture.
#[allow(clippy::needless_range_loop)]
pub fn run_arm(fx: &Fixture, spec: &GenSpec, arm: Arm, seed: u64) -> io::Result<Row> {
    let graph = &fx.graph;
    if fx.task.train.len() != K_MAX || fx.cues.len() != K_MAX || fx.splits.len() != K_MAX {
        return Err(bad("need K_MAX tasks, cues and splits"));
    }
    let original = graph.clone();
    let mut circuit = PlasticCircuit::new(graph, spec.init_weight)?;
    let fingerprint = circuit.topology_fingerprint();
    let episodes = spec.train_patterns * spec.rounds_a as usize;
    let mut ev = [[Eval::default(); K_MAX]; K_MAX];
    let mut cost = [PhaseCost::default(); K_MAX];
    for i in 0..K_MAX {
        ts::train_task(
            &mut circuit,
            graph,
            arm,
            &fx.task,
            spec,
            seed,
            i,
            0..episodes,
            &fx.cues,
            &mut cost[i],
        )?;
        for j in 0..K_MAX {
            ev[i][j] = evaluate(
                &circuit,
                graph,
                &fx.task.test,
                &fx.task.valence[j],
                &ts::active_cue(arm, &fx.cues, j),
                &fx.splits[j],
            )?;
        }
    }
    let mut cue_lines = [0; K_MAX];
    let mut cue_reach = [0; K_MAX];
    for j in 0..K_MAX {
        let c = ts::active_cue(arm, &fx.cues, j);
        cue_lines[j] = c.len();
        cue_reach[j] = reach(graph, &c);
    }
    Ok(Row {
        arm,
        flip: spec.flip_pct,
        seed,
        pool: fx.task.pool.len(),
        cue_lines,
        cue_reach,
        ev,
        cost,
        saturated_end: crate::error_gated::saturated(circuit.weights()),
        topology_unchanged: *graph == original
            && circuit.topology_fingerprint() == fingerprint
            && circuit.weights().len() == graph.edges(),
    })
}

/// Grid: flip × seed × arm.
pub fn run_grid(
    graph: Option<&Connectome>,
    design: &Design,
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
            ..design.spec
        };
        for &seed in seeds {
            let fx = fixture(graph, design, flip, seed)?;
            for &a in arms {
                rows.push(run_arm(&fx, &spec, a, seed)?);
            }
        }
    }
    Ok(rows)
}

/// Design-only census (no learning): per (seed, task) TEST pair counts per
/// class at one flip. Used by the design-validity gate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    pub pool: usize,
    pub cue_reach_min: usize,
    pub all: u64,
    pub seen: u64,
    pub cn: u64,
    pub icn: u64,
    pub ue: u64,
}

pub fn census(
    graph: Option<&Connectome>,
    design: &Design,
    flip: u32,
    seed: u64,
) -> io::Result<Vec<Census>> {
    let fx = fixture(graph, design, flip, seed)?;
    let reach_min = fx
        .cues
        .iter()
        .map(|c| reach(&fx.graph, c))
        .min()
        .unwrap_or(0);
    let mut out = Vec::with_capacity(K_MAX);
    for j in 0..K_MAX {
        let mut c = Census {
            pool: fx.task.pool.len(),
            cue_reach_min: reach_min,
            ..Census::default()
        };
        for p in &fx.task.test {
            for (t, conf) in ts::configurations(&fx.graph, p) {
                c.all += 1;
                match fx.splits[j].classify(t, &conf) {
                    Class::Seen => c.seen += 1,
                    Class::Compositional => {
                        c.cn += 1;
                        if fx.splits[j].inferable(&conf, &fx.task.valence[j]) {
                            c.icn += 1;
                        }
                    }
                    Class::UnseenEdge => c.ue += 1,
                }
            }
        }
        out.push(c);
    }
    Ok(out)
}

/// Median TEST pairs of slice `s` per (row, task) at a flip, over one arm
/// (counts do not depend on the arm).
#[must_use]
pub fn median_pairs(rows: &[Row], flip: u32, s: Slice) -> u64 {
    let Some(first) = rows.iter().find(|r| r.flip == flip).map(|r| r.arm) else {
        return 0;
    };
    let mut v: Vec<u64> = rows
        .iter()
        .filter(|r| r.flip == flip && r.arm == first)
        .flat_map(|r| (0..K_MAX).map(move |j| r.pairs(s, j)))
        .collect();
    v.sort_unstable();
    v[v.len() / 2]
}

/// Design-validity gate for slice `s` at a flip.
#[must_use]
pub fn evaluable(rows: &[Row], flip: u32, s: Slice) -> bool {
    median_pairs(rows, flip, s) >= MIN_NOVEL_PAIRS
}

fn pairs(rows: &[Row], flip: u32, x: Arm, base: Arm) -> Vec<(&Row, &Row)> {
    rows.iter()
        .filter(|r| r.flip == flip && r.arm == x)
        .filter_map(|r| {
            rows.iter()
                .find(|c| c.flip == flip && c.arm == base && c.seed == r.seed)
                .map(|c| (r, c))
        })
        .collect()
}

/// Arm `x` vs `base`, paired by seed, on slice `s` at sequence length `k`.
/// Every field is oriented so that > 0 favours `x`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Comparison {
    pub x: Arm,
    pub base: Arm,
    pub flip: u32,
    pub k: usize,
    pub slice: Slice,
    pub acc: Paired,
    pub learn: Paired,
    pub last: Paired,
    /// base − x.
    pub forget: Paired,
}

impl Comparison {
    /// Pre-registered rule: Δ ACC strong, Δ LEARN and Δ LAST ≥ −`EFFECT`.
    #[must_use]
    pub fn passes(&self) -> bool {
        self.acc.strong() && self.last.mean >= -EFFECT && self.learn.mean >= -EFFECT
    }
}

#[must_use]
pub fn compare(rows: &[Row], flip: u32, k: usize, s: Slice, x: Arm, base: Arm) -> Comparison {
    let p = pairs(rows, flip, x, base);
    let d =
        |f: &dyn Fn(&Row, &Row) -> f64| paired(&p.iter().map(|(r, c)| f(r, c)).collect::<Vec<_>>());
    Comparison {
        x,
        base,
        flip,
        k,
        slice: s,
        acc: d(&|r, c| r.acc(s, k) - c.acc(s, k)),
        learn: d(&|r, c| r.learn(s, k) - c.learn(s, k)),
        last: d(&|r, c| r.last(s, k) - c.last(s, k)),
        forget: d(&|r, c| c.forgetting(s, k) - r.forgetting(s, k)),
    }
}

/// Per-seed `f(row)` of one arm at a flip, paired statistics.
#[must_use]
pub fn per_arm(rows: &[Row], flip: u32, arm: Arm, f: &dyn Fn(&Row) -> f64) -> Paired {
    paired(
        &rows
            .iter()
            .filter(|r| r.flip == flip && r.arm == arm)
            .map(f)
            .collect::<Vec<_>>(),
    )
}

/// H1 statistic: `LEARN_cn_4 − 0.5` of `E/E-cue` (chance = 0.5 balanced).
#[must_use]
pub fn h1(rows: &[Row], flip: u32) -> Paired {
    per_arm(rows, flip, EE, &|r| r.learn(Slice::Cn, K_MAX) - 0.5)
}

/// H5 statistic: `ACC_seen_4 − ACC_cn_4` of `E/E-cue`.
#[must_use]
pub fn h5(rows: &[Row], flip: u32) -> Paired {
    per_arm(rows, flip, EE, &|r| {
        r.acc(Slice::Seen, K_MAX) - r.acc(Slice::Cn, K_MAX)
    })
}

/// Pre-registered readings at the primary flip:
/// (gate, H1, H2, H3, H4, H5). H1–H5 are `None` (not evaluable) if the CN
/// gate fails.
#[must_use]
pub fn readings(rows: &[Row]) -> (bool, [Option<bool>; 5]) {
    let f = PRIMARY_FLIP;
    let gate = evaluable(rows, f, Slice::Cn);
    if !gate {
        return (false, [None; 5]);
    }
    let h1 = h1(rows, f).strong();
    let h2 = compare(rows, f, K_MAX, Slice::Cn, EE, ERROR_GATED).passes();
    let h3 = compare(rows, f, K_MAX, Slice::Cn, EE, SUPERVISED).passes();
    let h4 = compare(rows, f, K_MAX, Slice::Cn, EE, ERROR_GATED)
        .forget
        .strong();
    let g = h5(rows, f);
    let h5 = g.n > 0 && g.ci_hi <= GAP_BOUND;
    (gate, [Some(h1), Some(h2), Some(h3), Some(h4), Some(h5)])
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
        "pool",
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
    for s in SLICES {
        for j in 1..=K_MAX {
            h.push(format!("pairs_{}_{j}", s.name()));
        }
    }
    for s in SLICES {
        for i in 1..=K_MAX {
            for j in 1..=K_MAX {
                h.push(format!("r_{}_{i}_{j}", s.name()));
            }
        }
    }
    for s in SLICES {
        for name in ["acc", "learn", "forgetting", "last"] {
            for k in 2..=K_MAX {
                h.push(format!("{name}_{}_k{k}", s.name()));
            }
        }
    }
    for j in 1..=K_MAX {
        h.push(format!("weights_changed_{j}"));
    }
    for j in 1..=K_MAX {
        h.push(format!("targets_updated_{j}"));
    }
    h.push("saturated_end".to_string());
    h.push("topology_unchanged".to_string());
    h.join(",")
}

fn letter(g: Gate) -> String {
    match g {
        Gate::Supervised => "S".to_string(),
        Gate::ErrorGated => "E".to_string(),
        Gate::MarginGated(m) => format!("M{m}"),
    }
}

/// Deterministic CSV (no timing column).
#[must_use]
pub fn to_csv(rows: &[Row], graph_label: &str) -> String {
    let mut out = csv_header();
    out.push('\n');
    type Metric = fn(&Row, Slice, usize) -> f64;
    let metrics: [Metric; 4] = [Row::acc, Row::learn, Row::forgetting, Row::last];
    for r in rows {
        let mut f: Vec<String> = vec![
            "b888-4-novel-split-v1".to_string(),
            graph_label.to_string(),
            r.flip.to_string(),
            r.arm.label(),
            letter(r.arm.pattern),
            r.arm.cue.map_or("-".to_string(), letter),
            r.seed.to_string(),
            r.pool.to_string(),
        ];
        f.extend(r.cue_lines.iter().map(ToString::to_string));
        f.extend(r.cue_reach.iter().map(ToString::to_string));
        for s in SLICES {
            f.extend((0..K_MAX).map(|j| r.pairs(s, j).to_string()));
        }
        for s in SLICES {
            for i in 0..K_MAX {
                f.extend((0..K_MAX).map(|j| format!("{:.6}", r.r(s, i, j))));
            }
        }
        for s in SLICES {
            for m in metrics {
                f.extend((2..=K_MAX).map(|k| format!("{:.6}", m(r, s, k))));
            }
        }
        f.extend(r.cost.iter().map(|c| c.weights_changed.to_string()));
        f.extend(r.cost.iter().map(|c| c.targets_updated.to_string()));
        f.push(r.saturated_end.to_string());
        f.push(r.topology_unchanged.to_string());
        out.push_str(&f.join(","));
        out.push('\n');
    }
    out
}

/// Per-arm mean of `f` at a flip (for reports).
#[must_use]
pub fn arm_mean(rows: &[Row], flip: u32, arm: Arm, f: &dyn Fn(&Row) -> f64) -> f64 {
    mean(
        rows.iter()
            .filter(|r| r.flip == flip && r.arm == arm)
            .map(f),
    )
}

/// Mean count over rows of one arm (weights changed summed over tasks).
#[must_use]
pub fn weights_changed(rows: &[Row], flip: u32, arm: Arm) -> f64 {
    arm_mean(rows, flip, arm, &|r| {
        r.cost.iter().map(|c| c.weights_changed as f64).sum()
    })
}

/// Distribution of novel pairs: (min, median, max) per (seed, task).
#[must_use]
pub fn pair_range(rows: &[Row], flip: u32, s: Slice) -> (u64, u64, u64) {
    let Some(first) = rows.iter().find(|r| r.flip == flip).map(|r| r.arm) else {
        return (0, 0, 0);
    };
    let mut v: Vec<u64> = rows
        .iter()
        .filter(|r| r.flip == flip && r.arm == first)
        .flat_map(|r| (0..K_MAX).map(move |j| r.pairs(s, j)))
        .collect();
    v.sort_unstable();
    (v[0], v[v.len() / 2], v[v.len() - 1])
}

/// Map of seed → per-task CN counts (diagnostic).
#[must_use]
pub fn cn_by_seed(rows: &[Row], flip: u32) -> HashMap<u64, [u64; K_MAX]> {
    let mut m = HashMap::new();
    for r in rows.iter().filter(|r| r.flip == flip && r.arm == ARMS[0]) {
        let mut a = [0; K_MAX];
        for (j, x) in a.iter_mut().enumerate() {
            *x = r.pairs(Slice::Cn, j);
        }
        m.insert(r.seed, a);
    }
    m
}

/// Human-readable report of a grid (deterministic; printed by the CLI).
#[must_use]
pub fn report(rows: &[Row], flips: &[u32]) -> String {
    use std::fmt::Write;
    let mut o = String::new();
    let p = |d: &Paired| {
        format!(
            "{:+.3}±{:.3}[{:+.3},{:+.3}]{}/{}/{}",
            d.mean, d.sd, d.ci_lo, d.ci_hi, d.better, d.ties, d.worse
        )
    };
    for &flip in flips {
        let _ = write!(o, "gate: flip={flip}");
        for s in SLICES {
            let (lo, med, hi) = pair_range(rows, flip, s);
            let _ = write!(o, " {}_pairs_min/med/max={lo}/{med}/{hi}", s.name());
        }
        let _ = writeln!(
            o,
            " min_required={MIN_NOVEL_PAIRS} cn_evaluable={} icn_evaluable={}",
            evaluable(rows, flip, Slice::Cn),
            evaluable(rows, flip, Slice::Icn)
        );
    }
    let _ = writeln!(
        o,
        "arm_summary: flip,k,arm,slice,acc,learn,forgetting,last (mean over seeds)"
    );
    for &flip in flips {
        for k in 2..=K_MAX {
            for a in ARMS {
                for s in SLICES {
                    let _ = writeln!(
                        o,
                        "{flip},{k},{},{},{:.3},{:.3},{:.3},{:.3}",
                        a.label(),
                        s.name(),
                        arm_mean(rows, flip, a, &|r| r.acc(s, k)),
                        arm_mean(rows, flip, a, &|r| r.learn(s, k)),
                        arm_mean(rows, flip, a, &|r| r.forgetting(s, k)),
                        arm_mean(rows, flip, a, &|r| r.last(s, k)),
                    );
                }
            }
        }
    }
    let _ = writeln!(o, "cost: flip,arm,weights_changed_total,saturated_end");
    for &flip in flips {
        for a in ARMS {
            let _ = writeln!(
                o,
                "{flip},{},{:.1},{:.1}",
                a.label(),
                weights_changed(rows, flip, a),
                arm_mean(rows, flip, a, &|r| r.saturated_end as f64)
            );
        }
    }
    let _ = writeln!(
        o,
        "learn_minus_chance: flip,arm,slice,LEARN_4-0.5 (mean±sd[boot95]>0/=0/<0)"
    );
    for &flip in flips {
        for a in ARMS {
            for s in [Slice::Seen, Slice::Cn, Slice::Icn, Slice::Ue] {
                let d = per_arm(rows, flip, a, &|r| r.learn(s, K_MAX) - 0.5);
                let _ = writeln!(o, "{flip},{},{},{}", a.label(), s.name(), p(&d));
            }
        }
    }
    let _ = writeln!(o, "gap_seen_minus_cn: flip,arm,ACC_seen_4-ACC_cn_4");
    for &flip in flips {
        for a in ARMS {
            let d = per_arm(rows, flip, a, &|r| {
                r.acc(Slice::Seen, K_MAX) - r.acc(Slice::Cn, K_MAX)
            });
            let _ = writeln!(o, "{flip},{},{}", a.label(), p(&d));
        }
    }
    let cmps = [
        (EE, ERROR_GATED),
        (EE, SUPERVISED),
        (ERROR_GATED, SUPERVISED),
        (SS, SUPERVISED),
        (EE, SS),
    ];
    let _ = writeln!(
        o,
        "comparison: flip,k,slice,x,base,n,d_acc,d_learn,d_last,d_forget(+=x_less),passes_rule"
    );
    for &flip in flips {
        for k in 2..=K_MAX {
            for s in SLICES {
                for &(x, base) in &cmps {
                    let c = compare(rows, flip, k, s, x, base);
                    let _ = writeln!(
                        o,
                        "{flip},{k},{},{},{},{},{},{},{},{},{}",
                        s.name(),
                        x.label(),
                        base.label(),
                        c.acc.n,
                        p(&c.acc),
                        p(&c.learn),
                        p(&c.last),
                        p(&c.forget),
                        c.passes()
                    );
                }
            }
        }
    }
    if flips.contains(&PRIMARY_FLIP) {
        let (gate, h) = readings(rows);
        let show = |x: Option<bool>| match x {
            None => "not_evaluable".to_string(),
            Some(true) => "pass".to_string(),
            Some(false) => "fail".to_string(),
        };
        let _ = writeln!(
            o,
            "readings(flip={PRIMARY_FLIP}): design_gate={} H1={} H2={} H3={} H4={} H5={}",
            if gate { "pass" } else { "fail" },
            show(h[0]),
            show(h[1]),
            show(h[2]),
            show(h[3]),
            show(h[4])
        );
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn small() -> Design {
        Design {
            shape: GraphShape {
                sources: 48,
                targets: 12,
                fan_in: 6,
            },
            spec: GenSpec {
                active: 4,
                train_patterns: 8,
                test_patterns: 24,
                rounds_a: 2,
                rounds_b: 2,
                ..DESIGN.spec
            },
            cue_cap: 3,
        }
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("v888-novel-split-{tag}-{}.pls", std::process::id()))
    }

    #[test]
    fn split_classes_and_or_rule_inference() {
        // Hand-built split: edges 0,1,2 into target 9; edge 3 never seen.
        let valence = vec![false, true, false, true];
        let mut s = TaskSplit::default();
        s.seen.insert((9, vec![0]));
        s.seen.insert((9, vec![0, 1]));
        s.seen.insert((9, vec![2]));
        s.edges.extend([0, 1, 2]);
        s.known_negative.extend([0, 2]);
        s.known_positive.insert(1);
        assert_eq!(s.classify(9, &[0]), Class::Seen);
        assert_eq!(s.classify(9, &[0, 2]), Class::Compositional);
        assert_eq!(s.classify(9, &[1, 2]), Class::Compositional);
        assert_eq!(s.classify(9, &[0, 3]), Class::UnseenEdge);
        assert!(s.inferable(&[0, 2], &valence));
        assert!(s.inferable(&[1, 2], &valence));
        assert!(!s.inferable(&[0, 3], &valence));
    }

    #[test]
    fn task_split_new_derives_known_valences() {
        let d = small();
        let fx = fixture(None, &d, 50, 3).unwrap();
        for j in 0..K_MAX {
            let sp = &fx.splits[j];
            let v = &fx.task.valence[j];
            // Every known edge's valence is the true one.
            assert!(sp.known_negative.iter().all(|&e| !v[e]));
            assert!(sp.known_positive.iter().all(|&e| v[e]));
            // Training configurations are never novel.
            for p in &fx.task.train[j] {
                for (t, c) in ts::configurations(&fx.graph, p) {
                    assert_eq!(sp.classify(t, &c), Class::Seen);
                }
            }
        }
    }

    #[test]
    fn census_counts_add_up_and_cues_respect_cap() {
        let d = small();
        let fx = fixture(None, &d, 50, 5).unwrap();
        assert!(fx
            .cues
            .iter()
            .all(|c| !c.is_empty() && c.len() <= d.cue_cap));
        let all: Vec<usize> = fx.cues.concat();
        for p in fx.task.train.iter().flatten().chain(&fx.task.test) {
            assert!(p.iter().all(|s| !all.contains(s)));
        }
        for c in census(None, &d, 50, 5).unwrap() {
            assert_eq!(c.all, c.seen + c.cn + c.ue);
            assert!(c.icn <= c.cn);
        }
    }

    #[test]
    fn primary_design_passes_gate_and_old_design_does_not() {
        // Design-only census on a pilot seed (no learning involved).
        let cn: Vec<u64> = census(None, &DESIGN, PRIMARY_FLIP, 1)
            .unwrap()
            .iter()
            .map(|c| c.cn)
            .collect();
        assert!(cn.iter().all(|&n| n >= MIN_NOVEL_PAIRS));
        let old = Design {
            shape: GraphShape {
                sources: 64,
                targets: 24,
                fan_in: 6,
            },
            spec: GenSpec {
                active: 4,
                test_patterns: 64,
                ..DESIGN.spec
            },
            cue_cap: cc::CUE_CAP,
        };
        let cn_old: Vec<u64> = census(None, &old, PRIMARY_FLIP, 1)
            .unwrap()
            .iter()
            .map(|c| c.cn)
            .collect();
        assert!(cn_old.iter().all(|&n| n < MIN_NOVEL_PAIRS));
    }

    #[test]
    fn capped_cues_match_uncapped_at_default_cap() {
        let g = rl::graph_for(None, &DESIGN.shape, 7).unwrap();
        assert_eq!(
            cc::choose_n_cues_capped(&g, K_MAX, cc::CUE_CAP).unwrap(),
            cc::choose_n_cues(&g, K_MAX).unwrap()
        );
        assert!(cc::choose_n_cues_capped(&g, K_MAX, 0).is_err());
    }

    #[test]
    fn run_is_topology_preserving_and_resumes_bit_exactly() {
        let d = small();
        let fx = fixture(None, &d, 50, 4).unwrap();
        for a in ARMS {
            let full = run_arm(&fx, &d.spec, a, 4).unwrap();
            assert!(full.topology_unchanged);
            if a.cue.is_none() {
                assert_eq!(full.cue_lines, [0; K_MAX]);
            }
            // Re-run with a checkpoint after task 2 and mid task 3.
            let mut c = PlasticCircuit::new(&fx.graph, d.spec.init_weight).unwrap();
            let episodes = d.spec.train_patterns * d.spec.rounds_a as usize;
            let mut cost = [PhaseCost::default(); K_MAX];
            for (i, ci) in cost.iter_mut().enumerate().take(2) {
                ts::train_task(
                    &mut c,
                    &fx.graph,
                    a,
                    &fx.task,
                    &d.spec,
                    4,
                    i,
                    0..episodes,
                    &fx.cues,
                    ci,
                )
                .unwrap();
            }
            let p = tmp("k2");
            c.save(&p).unwrap();
            let mut c = PlasticCircuit::load(&p, &fx.graph).unwrap();
            fs::remove_file(&p).unwrap();
            let half = episodes / 2;
            for r in [0..half, half..episodes] {
                ts::train_task(
                    &mut c,
                    &fx.graph,
                    a,
                    &fx.task,
                    &d.spec,
                    4,
                    2,
                    r,
                    &fx.cues,
                    &mut cost[2],
                )
                .unwrap();
                let p = tmp("k3");
                c.save(&p).unwrap();
                c = PlasticCircuit::load(&p, &fx.graph).unwrap();
                fs::remove_file(&p).unwrap();
            }
            ts::train_task(
                &mut c,
                &fx.graph,
                a,
                &fx.task,
                &d.spec,
                4,
                3,
                0..episodes,
                &fx.cues,
                &mut cost[3],
            )
            .unwrap();
            assert_eq!(cost, full.cost);
            for j in 0..K_MAX {
                let e = evaluate(
                    &c,
                    &fx.graph,
                    &fx.task.test,
                    &fx.task.valence[j],
                    &ts::active_cue(a, &fx.cues, j),
                    &fx.splits[j],
                )
                .unwrap();
                assert_eq!(e, full.ev[3][j]);
                assert_eq!(e.all.total(), e.seen.total() + e.cn.total() + e.ue.total());
            }
        }
    }

    #[test]
    fn grid_is_deterministic_csv_complete_and_readings_gate() {
        let d = small();
        let a = run_grid(None, &d, &[50], &[1, 2, 3], &ARMS).unwrap();
        let b = run_grid(None, &d, &[50], &[1, 2, 3], &ARMS).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 3 * ARMS.len());
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        let cols = csv_header().split(',').count();
        assert!(csv.lines().all(|l| l.split(',').count() == cols));
        assert!(csv.contains(",S/S-cue,S,S,"));
        let id = compare(&a, 50, K_MAX, Slice::Cn, SUPERVISED, SUPERVISED);
        assert_eq!(id.acc.ties, 3);
        assert!(!id.passes());
        let (gate, h) = readings(&a);
        assert_eq!(gate, evaluable(&a, 50, Slice::Cn));
        if !gate {
            assert_eq!(h, [None; 5]);
        }
        let (lo, med, hi) = pair_range(&a, 50, Slice::Cn);
        assert!(lo <= med && med <= hi);
        assert_eq!(med, median_pairs(&a, 50, Slice::Cn));
        assert_eq!(cn_by_seed(&a, 50).len(), 3);
        let rep = report(&a, &[50]);
        assert!(rep.contains("readings(flip=50): design_gate="));
        assert_eq!(rep, report(&b, &[50]));
        assert!(weights_changed(&a, 50, SUPERVISED) >= weights_changed(&a, 50, ERROR_GATED));
        assert!(run_grid(None, &d, &[50], &[], &ARMS).is_err());
        assert!(run_grid(None, &d, &[50], &[1], &[]).is_err());
    }
}

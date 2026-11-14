//! B888-4 cycle 13: **cue reach at a fixed pattern pool**.
//!
//! Cycle 12 (#123) found that on the novel-split design the `E/E-cue` arm
//! adds almost nothing over `error-gated` (+0.003 on compositional-novel
//! pairs, +0.011 on all pairs), whereas #122 had measured +0.215 on all
//! pairs. In #123 each cue was capped at 4 lines and reached only 14–22 of
//! the 32 targets, while #122's cues reached every target. Cue reach and
//! pattern-pool width changed together, so the two were confounded.
//!
//! This module separates them. Per seed it reserves four disjoint greedy
//! cues of up to [`FULL`] lines (enough to reach every target), removes all
//! reserved lines from the pattern pool **once**, and then runs the cued arms
//! with the first `L` lines of each cue for `L` in [`LEVELS`]. Graph, pool,
//! patterns, valences, pair classes and episode order are identical at every
//! level; only the number of active cue lines (and so the cue reach) changes.
//! Reserved lines that a level does not use are never active.
//!
//! A secondary "coupled" family reproduces the #123 construction (reserve
//! only `L` lines per cue) so the pool-versus-reach trade-off is reported
//! explicitly.
//!
//! Every arm acts only on the original edges (checked per row).
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph; nothing here is
//! a claim about synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/CUE_REACH.md`
//! (committed before any learning run).
use crate::connectome::Connectome;
use crate::context_cue as cc;
use crate::error_gated::{paired, Gate, Paired, PhaseCost, EFFECT, THRESHOLD};
use crate::generalization_bench::{self as gb, Confusion, GenSpec};
use crate::novel_split::{self as ns, Design, Eval, Fixture, Slice, SLICES};
use crate::plasticity::PlasticCircuit;
use crate::task_sequence::{self as ts, Arm, K_MAX};
use std::io::{self, ErrorKind};

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Pre-registered label-flip rates (50 primary).
pub const FLIPS: [u32; 3] = [25, 50, 100];
pub const PRIMARY_FLIP: u32 = 50;
/// Pre-registered seeds: 121..=160, never used by an earlier B888-4 cycle.
pub const SEEDS: std::ops::RangeInclusive<u64> = 121..=160;
/// Cue sizes (lines per cue) compared at a fixed pool.
pub const LEVELS: [usize; 4] = [2, 4, 8, 16];
/// Lines reserved per cue in the fixed-pool family (the largest level).
pub const FULL: usize = 16;
/// Monotonicity tolerance for the dose-response hypotheses (H1, H2).
pub const MONO_TOL: f64 = 0.005;
/// Fixed-pool design: #123 graph and task, cues reserved up to `FULL`.
pub const DESIGN: Design = Design {
    cue_cap: FULL,
    ..ns::DESIGN
};

pub const SUPERVISED: Arm = ns::SUPERVISED;
pub const ERROR_GATED: Arm = ns::ERROR_GATED;
pub const SS: Arm = ns::SS;
pub const EE: Arm = ns::EE;

/// How the pattern pool is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Primary: every level shares one pool (all `FULL`-line cues reserved).
    Fixed,
    /// Secondary: #123 construction, only the level's lines reserved.
    Coupled,
}

impl Family {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Coupled => "coupled",
        }
    }
}

/// The first `lines` lines of every cue (greedy order: largest marginal
/// reach first). Nested by construction.
#[must_use]
pub fn prefix_cues(cues: &[Vec<usize>], lines: usize) -> Vec<Vec<usize>> {
    cues.iter()
        .map(|c| c[..lines.min(c.len())].to_vec())
        .collect()
}

/// Sorted targets reached by usable edges of `lines`.
#[must_use]
pub fn reach_set(graph: &Connectome, lines: &[usize]) -> Vec<usize> {
    let mut t: Vec<usize> = lines
        .iter()
        .flat_map(|&s| graph.offsets[s]..graph.offsets[s + 1])
        .filter(|&e| gb::usable(graph, e as usize))
        .map(|e| graph.targets[e as usize] as usize)
        .collect();
    t.sort_unstable();
    t.dedup();
    t
}

/// Number of targets with at least one usable in-edge.
#[must_use]
pub fn reachable_targets(graph: &Connectome) -> usize {
    let mut t: Vec<u32> = (0..graph.edges())
        .filter(|&e| gb::usable(graph, e))
        .map(|e| graph.targets[e])
        .collect();
    t.sort_unstable();
    t.dedup();
    t.len()
}

/// Fixed-pool fixture of one (flip, seed): cues of up to `FULL` lines.
pub fn fixed_fixture(graph: Option<&Connectome>, flip: u32, seed: u64) -> io::Result<Fixture> {
    ns::fixture(graph, &DESIGN, flip, seed)
}

/// The fixed fixture with each cue cut to its first `lines` lines. Pool,
/// patterns, valences and splits are unchanged.
#[must_use]
pub fn at_level(base: &Fixture, lines: usize) -> Fixture {
    Fixture {
        cues: prefix_cues(&base.cues, lines),
        ..base.clone()
    }
}

/// Coupled (#123-style) fixture: only `lines` lines per cue reserved.
pub fn coupled_fixture(
    graph: Option<&Connectome>,
    flip: u32,
    seed: u64,
    lines: usize,
) -> io::Result<Fixture> {
    ns::fixture(
        graph,
        &Design {
            cue_cap: lines,
            ..ns::DESIGN
        },
        flip,
        seed,
    )
}

/// One measurement: (family, flip, seed, arm, level) on a K_MAX-task run.
/// `level` is the number of cue lines per cue (0 for uncued arms of the
/// fixed family; for the coupled family the level of the fixture).
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub family: Family,
    pub level: usize,
    pub arm: Arm,
    pub flip: u32,
    pub seed: u64,
    pub pool: usize,
    pub reachable: usize,
    pub cue_lines: [usize; K_MAX],
    /// Targets reached by cue j of this row's fixture at this level
    /// (also filled for uncued rows: the cue they would have had).
    pub cue_targets: Vec<Vec<usize>>,
    /// `ev[i][j]`: task j on TEST after training task i, per pair class.
    pub ev: [[Eval; K_MAX]; K_MAX],
    /// Per-target confusions over all TEST pairs: `tgt[(i*K+j)*n + t]`.
    pub tgt: Vec<Confusion>,
    pub nodes: usize,
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

fn add(a: &mut Confusion, b: &Confusion) {
    a.tp += b.tp;
    a.tn += b.tn;
    a.fp += b.fp;
    a.fn_ += b.fn_;
}

/// Pair subset used by the metrics: a pair class, or all pairs whose target
/// is (not) reached by a given per-task target set.
#[derive(Clone, Copy, Debug)]
pub enum Part<'a> {
    Class(Slice),
    Reached(&'a [Vec<usize>]),
    Unreached(&'a [Vec<usize>]),
}

impl Row {
    fn conf(&self, p: Part<'_>, i: usize, j: usize) -> Confusion {
        match p {
            Part::Class(s) => s.pick(&self.ev[i][j]),
            Part::Reached(sets) | Part::Unreached(sets) => {
                let inside = matches!(p, Part::Reached(_));
                let base = (i * K_MAX + j) * self.nodes;
                let mut c = Confusion::default();
                for t in 0..self.nodes {
                    if sets[j].binary_search(&t).is_ok() == inside {
                        add(&mut c, &self.tgt[base + t]);
                    }
                }
                c
            }
        }
    }
    #[must_use]
    pub fn r(&self, p: Part<'_>, i: usize, j: usize) -> f64 {
        self.conf(p, i, j).balanced()
    }
    #[must_use]
    pub fn acc(&self, p: Part<'_>, k: usize) -> f64 {
        mean((0..k).map(|j| self.r(p, k - 1, j)))
    }
    #[must_use]
    pub fn learn(&self, p: Part<'_>, k: usize) -> f64 {
        mean((0..k).map(|j| self.r(p, j, j)))
    }
    #[must_use]
    pub fn forgetting(&self, p: Part<'_>, k: usize) -> f64 {
        mean((0..k.saturating_sub(1)).map(|j| self.r(p, j, j) - self.r(p, k - 1, j)))
    }
    #[must_use]
    pub fn last(&self, p: Part<'_>, k: usize) -> f64 {
        self.r(p, k - 1, k - 1)
    }
    #[must_use]
    pub fn pairs(&self, p: Part<'_>, j: usize) -> u64 {
        self.conf(p, 0, j).total()
    }
    /// Cue reach of task j as a fraction of reachable targets.
    #[must_use]
    pub fn reach_frac(&self, j: usize) -> f64 {
        if self.reachable == 0 {
            0.0
        } else {
            self.cue_targets[j].len() as f64 / self.reachable as f64
        }
    }
}

/// Evaluate one task on `patterns` (with `cue` on): per-class confusions and
/// per-target all-pair confusions (length = node count).
pub fn evaluate(
    circuit: &PlasticCircuit,
    graph: &Connectome,
    patterns: &[Vec<usize>],
    valence: &[bool],
    cue: &[usize],
    split: &ns::TaskSplit,
) -> io::Result<(Eval, Vec<Confusion>)> {
    let ev = ns::evaluate(circuit, graph, patterns, valence, cue, split)?;
    let mut per = vec![Confusion::default(); graph.ids.len()];
    for p in patterns {
        let (input, _, teacher) = cc::encode_ctx(graph, p, valence, cue)?;
        let out = circuit.predict(graph, &input, THRESHOLD)?;
        for (t, _) in ts::configurations(graph, p) {
            let want = teacher.get(t).map_err(|_| bad("bad target"))?;
            let got = out.get(t).map_err(|_| bad("bad target"))?;
            let c = &mut per[t];
            match (want, got) {
                (true, true) => c.tp += 1,
                (false, false) => c.tn += 1,
                (false, true) => c.fp += 1,
                (true, false) => c.fn_ += 1,
            }
        }
    }
    Ok((ev, per))
}

/// Run the K_MAX tasks for one arm on a fixture (the fixture's cues are the
/// level's cues). `level` is recorded as given.
#[allow(clippy::needless_range_loop)]
pub fn run_arm(
    fx: &Fixture,
    spec: &GenSpec,
    family: Family,
    level: usize,
    arm: Arm,
    seed: u64,
) -> io::Result<Row> {
    let graph = &fx.graph;
    if fx.task.train.len() != K_MAX || fx.cues.len() != K_MAX || fx.splits.len() != K_MAX {
        return Err(bad("need K_MAX tasks, cues and splits"));
    }
    let original = graph.clone();
    let n = graph.ids.len();
    let mut circuit = PlasticCircuit::new(graph, spec.init_weight)?;
    let fingerprint = circuit.topology_fingerprint();
    let episodes = spec.train_patterns * spec.rounds_a as usize;
    let mut ev = [[Eval::default(); K_MAX]; K_MAX];
    let mut tgt = vec![Confusion::default(); K_MAX * K_MAX * n];
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
            let (e, per) = evaluate(
                &circuit,
                graph,
                &fx.task.test,
                &fx.task.valence[j],
                &ts::active_cue(arm, &fx.cues, j),
                &fx.splits[j],
            )?;
            ev[i][j] = e;
            tgt[(i * K_MAX + j) * n..(i * K_MAX + j + 1) * n].copy_from_slice(&per);
        }
    }
    let mut cue_lines = [0; K_MAX];
    for j in 0..K_MAX {
        cue_lines[j] = ts::active_cue(arm, &fx.cues, j).len();
    }
    Ok(Row {
        family,
        level,
        arm,
        flip: spec.flip_pct,
        seed,
        pool: fx.task.pool.len(),
        reachable: reachable_targets(graph),
        cue_lines,
        cue_targets: fx.cues.iter().map(|c| reach_set(graph, c)).collect(),
        ev,
        tgt,
        nodes: n,
        cost,
        saturated_end: crate::error_gated::saturated(circuit.weights()),
        topology_unchanged: *graph == original
            && circuit.topology_fingerprint() == fingerprint
            && circuit.weights().len() == graph.edges(),
    })
}

/// Primary grid (fixed pool): per (flip, seed) `supervised` and
/// `error-gated` once, then `E/E-cue` and `S/S-cue` at every level.
pub fn run_fixed(
    graph: Option<&Connectome>,
    flips: &[u32],
    seeds: &[u64],
    levels: &[usize],
) -> io::Result<Vec<Row>> {
    run_fixed_design(graph, &DESIGN, flips, seeds, levels)
}

/// [`run_fixed`] on an arbitrary design (tests use a small one). Levels
/// above `design.cue_cap` are rejected.
pub fn run_fixed_design(
    graph: Option<&Connectome>,
    design: &Design,
    flips: &[u32],
    seeds: &[u64],
    levels: &[usize],
) -> io::Result<Vec<Row>> {
    if seeds.is_empty() || flips.is_empty() || levels.is_empty() {
        return Err(bad("need at least one seed, flip rate and level"));
    }
    if levels.iter().any(|&l| l == 0 || l > design.cue_cap) {
        return Err(bad("cue level must be in 1..=reserved cap"));
    }
    let mut rows = Vec::new();
    for &flip in flips {
        let spec = GenSpec {
            flip_pct: flip,
            ..design.spec
        };
        for &seed in seeds {
            let base = ns::fixture(graph, design, flip, seed)?;
            for a in [SUPERVISED, ERROR_GATED] {
                rows.push(run_arm(&base, &spec, Family::Fixed, 0, a, seed)?);
            }
            for &l in levels {
                let fx = at_level(&base, l);
                for a in [EE, SS] {
                    rows.push(run_arm(&fx, &spec, Family::Fixed, l, a, seed)?);
                }
            }
        }
    }
    Ok(rows)
}

/// Secondary grid (coupled, #123 construction): per (seed, level)
/// `error-gated` and `E/E-cue` on a fixture that reserves only the level's
/// lines.
pub fn run_coupled(
    graph: Option<&Connectome>,
    design: &Design,
    flip: u32,
    seeds: &[u64],
    levels: &[usize],
) -> io::Result<Vec<Row>> {
    if seeds.is_empty() || levels.is_empty() || levels.contains(&0) {
        return Err(bad("need at least one seed and positive levels"));
    }
    let spec = GenSpec {
        flip_pct: flip,
        ..design.spec
    };
    let mut rows = Vec::new();
    for &seed in seeds {
        for &l in levels {
            let fx = ns::fixture(
                graph,
                &Design {
                    cue_cap: l,
                    ..*design
                },
                flip,
                seed,
            )?;
            for a in [ERROR_GATED, EE] {
                rows.push(run_arm(&fx, &spec, Family::Coupled, l, a, seed)?);
            }
        }
    }
    Ok(rows)
}

/// Label-free census of one fixed fixture: pool, reachable targets, cue
/// lines and reach per (level, task), and per-task pair counts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Census {
    pub pool: usize,
    pub reachable: usize,
    /// `[level][task]`.
    pub lines: Vec<[usize; K_MAX]>,
    pub reach: Vec<[usize; K_MAX]>,
    pub pairs: [ns::Census; K_MAX],
}

pub fn census_fixture(fx: &Fixture, levels: &[usize]) -> Census {
    let mut lines = Vec::new();
    let mut reach = Vec::new();
    for &l in levels {
        let c = prefix_cues(&fx.cues, l);
        let mut a = [0; K_MAX];
        let mut b = [0; K_MAX];
        for j in 0..K_MAX {
            a[j] = c[j].len();
            b[j] = reach_set(&fx.graph, &c[j]).len();
        }
        lines.push(a);
        reach.push(b);
    }
    let mut pairs = [ns::Census::default(); K_MAX];
    for (j, c) in pairs.iter_mut().enumerate() {
        c.pool = fx.task.pool.len();
        for p in &fx.task.test {
            for (t, conf) in ts::configurations(&fx.graph, p) {
                c.all += 1;
                match fx.splits[j].classify(t, &conf) {
                    ns::Class::Seen => c.seen += 1,
                    ns::Class::Compositional => {
                        c.cn += 1;
                        if fx.splits[j].inferable(&conf, &fx.task.valence[j]) {
                            c.icn += 1;
                        }
                    }
                    ns::Class::UnseenEdge => c.ue += 1,
                }
            }
        }
    }
    Census {
        pool: fx.task.pool.len(),
        reachable: reachable_targets(&fx.graph),
        lines,
        reach,
        pairs,
    }
}

fn find(
    rows: &[Row],
    family: Family,
    flip: u32,
    seed: u64,
    arm: Arm,
    level: usize,
) -> Option<&Row> {
    rows.iter().find(|r| {
        r.family == family && r.flip == flip && r.seed == seed && r.arm == arm && r.level == level
    })
}

/// A (arm, level) setting inside one family.
pub type Setting = (Arm, usize);

/// x vs base paired by seed. Every field is oriented so that > 0 favours x.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Comparison {
    pub acc: Paired,
    pub learn: Paired,
    pub last: Paired,
    /// base − x.
    pub forget: Paired,
}

impl Comparison {
    /// Pre-registered sequence rule (as #122/#123): Δ ACC strong, Δ LEARN
    /// and Δ LAST ≥ −`EFFECT`.
    #[must_use]
    pub fn passes(&self) -> bool {
        self.acc.strong() && self.last.mean >= -EFFECT && self.learn.mean >= -EFFECT
    }
}

/// Which pair subset a comparison uses. `Reached`/`Unreached` use the cue
/// targets of the x row (so the base is scored on the same pairs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sub {
    Class(Slice),
    Reached,
    Unreached,
}

impl Sub {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Class(s) => s.name(),
            Self::Reached => "cue_reached",
            Self::Unreached => "cue_unreached",
        }
    }
}

fn part<'a>(s: Sub, x: &'a Row) -> Part<'a> {
    match s {
        Sub::Class(c) => Part::Class(c),
        Sub::Reached => Part::Reached(&x.cue_targets),
        Sub::Unreached => Part::Unreached(&x.cue_targets),
    }
}

/// Paired comparison of setting `x` against `base` in one family.
#[must_use]
pub fn compare(
    rows: &[Row],
    family: Family,
    flip: u32,
    k: usize,
    s: Sub,
    x: Setting,
    base: Setting,
) -> Comparison {
    let pairs: Vec<(&Row, &Row)> = rows
        .iter()
        .filter(|r| r.family == family && r.flip == flip && r.arm == x.0 && r.level == x.1)
        .filter_map(|r| find(rows, family, flip, r.seed, base.0, base.1).map(|c| (r, c)))
        .collect();
    let d = |f: &dyn Fn(&Row, &Row, Part<'_>) -> f64| {
        paired(
            &pairs
                .iter()
                .map(|(r, c)| f(r, c, part(s, r)))
                .collect::<Vec<_>>(),
        )
    };
    Comparison {
        acc: d(&|r, c, p| r.acc(p, k) - c.acc(p, k)),
        learn: d(&|r, c, p| r.learn(p, k) - c.learn(p, k)),
        last: d(&|r, c, p| r.last(p, k) - c.last(p, k)),
        forget: d(&|r, c, p| c.forgetting(p, k) - r.forgetting(p, k)),
    }
}

/// The base of the gain in the fixed family: uncued `error-gated`.
pub const EG0: Setting = (ERROR_GATED, 0);

/// Gain of `E/E-cue` at `level` over `error-gated` (fixed family).
#[must_use]
pub fn gain(rows: &[Row], flip: u32, k: usize, s: Sub, level: usize) -> Comparison {
    compare(rows, Family::Fixed, flip, k, s, (EE, level), EG0)
}

/// Median of a per-(seed, task) count over the rows of one setting.
fn median(mut v: Vec<u64>) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

/// Median CN pairs per (seed, task) in the fixed family at a flip.
#[must_use]
pub fn median_cn(rows: &[Row], flip: u32) -> u64 {
    median(
        rows.iter()
            .filter(|r| r.family == Family::Fixed && r.flip == flip && r.arm == ERROR_GATED)
            .flat_map(|r| (0..K_MAX).map(move |j| r.pairs(Part::Class(Slice::Cn), j)))
            .collect(),
    )
}

/// Median cue reach per (seed, task) of `E/E-cue` at a level (fixed family).
#[must_use]
pub fn median_reach(rows: &[Row], flip: u32, level: usize) -> u64 {
    median(
        rows.iter()
            .filter(|r| {
                r.family == Family::Fixed && r.flip == flip && r.arm == EE && r.level == level
            })
            .flat_map(|r| r.cue_targets.iter().map(|t| t.len() as u64))
            .collect(),
    )
}

/// Median reachable targets per seed (fixed family).
#[must_use]
pub fn median_reachable(rows: &[Row], flip: u32) -> u64 {
    median(
        rows.iter()
            .filter(|r| r.family == Family::Fixed && r.flip == flip && r.arm == ERROR_GATED)
            .map(|r| r.reachable as u64)
            .collect(),
    )
}

/// Design gates at the primary flip: (CN gate, reach gate). The reach gate
/// needs the median reach to increase strictly with the level and to equal
/// the reachable targets at the largest level.
#[must_use]
pub fn gates(rows: &[Row], levels: &[usize]) -> (bool, bool) {
    let f = PRIMARY_FLIP;
    let cn = median_cn(rows, f) >= ns::MIN_NOVEL_PAIRS;
    let m: Vec<u64> = levels.iter().map(|&l| median_reach(rows, f, l)).collect();
    let reach = !m.is_empty()
        && m.windows(2).all(|w| w[1] > w[0])
        && m.last() == Some(&median_reachable(rows, f));
    (cn, reach)
}

/// Dose-response rule on subset `s` at k = 4: consecutive level means of the
/// gain never drop by more than `MONO_TOL`, and the paired difference
/// `E/E-cue`@largest − `E/E-cue`@smallest has 95 % CI lower bound > 0.
#[must_use]
pub fn dose_response(rows: &[Row], flip: u32, s: Sub, levels: &[usize]) -> bool {
    if levels.len() < 2 {
        return false;
    }
    let m: Vec<f64> = levels
        .iter()
        .map(|&l| gain(rows, flip, K_MAX, s, l).acc.mean)
        .collect();
    let mono = m.windows(2).all(|w| w[1] >= w[0] - MONO_TOL);
    let hi = levels[levels.len() - 1];
    let lo = levels[0];
    let end = compare(rows, Family::Fixed, flip, K_MAX, s, (EE, hi), (EE, lo));
    mono && end.acc.n > 0 && end.acc.ci_lo > 0.0
}

/// Pre-registered readings at the primary flip: (CN gate, reach gate,
/// [H1..H5]); H1–H5 are `None` if either gate fails.
#[must_use]
pub fn readings(rows: &[Row], levels: &[usize]) -> (bool, bool, [Option<bool>; 5]) {
    let f = PRIMARY_FLIP;
    let (cn, reach) = gates(rows, levels);
    if !(cn && reach) {
        return (cn, reach, [None; 5]);
    }
    let full = levels[levels.len() - 1];
    let all = Sub::Class(Slice::All);
    let novel = Sub::Class(Slice::Cn);
    let h1 = dose_response(rows, f, all, levels);
    let h2 = dose_response(rows, f, novel, levels);
    let h3 = gain(rows, f, K_MAX, all, full).passes();
    let h4 = gain(rows, f, K_MAX, novel, full).passes();
    let h5 = gain(rows, f, K_MAX, novel, full).forget.strong();
    (
        cn,
        reach,
        [Some(h1), Some(h2), Some(h3), Some(h4), Some(h5)],
    )
}

fn letter(g: Gate) -> String {
    match g {
        Gate::Supervised => "S".to_string(),
        Gate::ErrorGated => "E".to_string(),
        Gate::MarginGated(m) => format!("M{m}"),
    }
}

/// CSV header (generated; see `to_csv`).
#[must_use]
pub fn csv_header() -> String {
    let mut h: Vec<String> = [
        "schema",
        "graph",
        "family",
        "flip_pct",
        "arm",
        "pattern_gate",
        "cue_gate",
        "level",
        "seed",
        "pool",
        "reachable_targets",
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
    for name in ["reached", "unreached"] {
        for m in ["acc", "learn", "forgetting"] {
            h.push(format!("{m}_cue_{name}_k{K_MAX}"));
        }
    }
    for j in 1..=K_MAX {
        h.push(format!("weights_changed_{j}"));
    }
    h.push("saturated_end".to_string());
    h.push("topology_unchanged".to_string());
    h.join(",")
}

/// Deterministic CSV (no timing column). The reached / unreached columns
/// use the row's own cue targets (for uncued rows: the cue of its fixture,
/// i.e. the `FULL`-line cue in the fixed family).
#[must_use]
pub fn to_csv(rows: &[Row], graph_label: &str) -> String {
    let mut out = csv_header();
    out.push('\n');
    for r in rows {
        let mut f: Vec<String> = vec![
            "b888-4-cue-reach-v1".to_string(),
            graph_label.to_string(),
            r.family.name().to_string(),
            r.flip.to_string(),
            r.arm.label(),
            letter(r.arm.pattern),
            r.arm.cue.map_or("-".to_string(), letter),
            r.level.to_string(),
            r.seed.to_string(),
            r.pool.to_string(),
            r.reachable.to_string(),
        ];
        f.extend(r.cue_lines.iter().map(ToString::to_string));
        f.extend(r.cue_targets.iter().map(|t| t.len().to_string()));
        for s in SLICES {
            f.extend((0..K_MAX).map(|j| r.pairs(Part::Class(s), j).to_string()));
        }
        for s in SLICES {
            for i in 0..K_MAX {
                f.extend((0..K_MAX).map(|j| format!("{:.6}", r.r(Part::Class(s), i, j))));
            }
        }
        for s in SLICES {
            let p = Part::Class(s);
            for k in 2..=K_MAX {
                f.push(format!("{:.6}", r.acc(p, k)));
            }
            for k in 2..=K_MAX {
                f.push(format!("{:.6}", r.learn(p, k)));
            }
            for k in 2..=K_MAX {
                f.push(format!("{:.6}", r.forgetting(p, k)));
            }
            for k in 2..=K_MAX {
                f.push(format!("{:.6}", r.last(p, k)));
            }
        }
        for p in [
            Part::Reached(&r.cue_targets),
            Part::Unreached(&r.cue_targets),
        ] {
            f.push(format!("{:.6}", r.acc(p, K_MAX)));
            f.push(format!("{:.6}", r.learn(p, K_MAX)));
            f.push(format!("{:.6}", r.forgetting(p, K_MAX)));
        }
        f.extend(r.cost.iter().map(|c| c.weights_changed.to_string()));
        f.push(r.saturated_end.to_string());
        f.push(r.topology_unchanged.to_string());
        out.push_str(&f.join(","));
        out.push('\n');
    }
    out
}

/// Mean of `f` over the rows of one (family, flip, arm, level).
#[must_use]
pub fn setting_mean(
    rows: &[Row],
    family: Family,
    flip: u32,
    x: Setting,
    f: &dyn Fn(&Row) -> f64,
) -> f64 {
    mean(
        rows.iter()
            .filter(|r| r.family == family && r.flip == flip && r.arm == x.0 && r.level == x.1)
            .map(f),
    )
}

/// OLS slope of the per-(seed, level) gain on `s` against mean cue reach
/// fraction (fixed family, k = 4), in accuracy per unit reach fraction.
#[must_use]
pub fn reach_slope(rows: &[Row], flip: u32, s: Sub, levels: &[usize]) -> f64 {
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for &l in levels {
        for r in rows
            .iter()
            .filter(|r| r.family == Family::Fixed && r.flip == flip && r.arm == EE && r.level == l)
        {
            if let Some(b) = find(rows, Family::Fixed, flip, r.seed, ERROR_GATED, 0) {
                let p = part(s, r);
                let x = mean((0..K_MAX).map(|j| r.reach_frac(j)));
                pts.push((x, r.acc(p, K_MAX) - b.acc(p, K_MAX)));
            }
        }
    }
    let n = pts.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let sxx: f64 = pts.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    if sxx == 0.0 {
        0.0
    } else {
        sxy / sxx
    }
}

/// Human-readable report (deterministic; printed by the CLI).
#[must_use]
pub fn report(rows: &[Row], flips: &[u32], levels: &[usize]) -> String {
    use std::fmt::Write;
    let mut o = String::new();
    let p = |d: &Paired| {
        format!(
            "{:+.3}±{:.3}[{:+.3},{:+.3}]{}/{}/{}",
            d.mean, d.sd, d.ci_lo, d.ci_hi, d.better, d.ties, d.worse
        )
    };
    let fixed = Family::Fixed;
    let _ = writeln!(
        o,
        "census(fixed): flip,level,cue_lines_mean,reach_median,reach_min,reach_max,reach_frac_mean,pool_min/max,reachable_median,cn_median"
    );
    for &flip in flips {
        let rs: Vec<&Row> = rows
            .iter()
            .filter(|r| r.family == fixed && r.flip == flip && r.arm == ERROR_GATED)
            .collect();
        let pmin = rs.iter().map(|r| r.pool).min().unwrap_or(0);
        let pmax = rs.iter().map(|r| r.pool).max().unwrap_or(0);
        for &l in levels {
            let ee: Vec<&Row> = rows
                .iter()
                .filter(|r| r.family == fixed && r.flip == flip && r.arm == EE && r.level == l)
                .collect();
            let reach: Vec<usize> = ee
                .iter()
                .flat_map(|r| r.cue_targets.iter().map(Vec::len))
                .collect();
            let _ = writeln!(
                o,
                "{flip},{l},{:.2},{},{},{},{:.3},{pmin}/{pmax},{},{}",
                mean(
                    ee.iter()
                        .flat_map(|r| r.cue_lines.iter().map(|&x| x as f64))
                ),
                median_reach(rows, flip, l),
                reach.iter().min().unwrap_or(&0),
                reach.iter().max().unwrap_or(&0),
                mean(ee.iter().flat_map(|r| (0..K_MAX).map(|j| r.reach_frac(j)))),
                median_reachable(rows, flip),
                median_cn(rows, flip),
            );
        }
    }
    let _ = writeln!(
        o,
        "setting_summary(fixed,k=4): flip,arm,level,acc_all,acc_cn,acc_icn,learn_all,learn_cn,fgt_all,fgt_cn,last_all,weights_changed,saturated_end"
    );
    let mut settings: Vec<Setting> = vec![(SUPERVISED, 0), EG0];
    for &l in levels {
        settings.push((EE, l));
    }
    for &l in levels {
        settings.push((SS, l));
    }
    for &flip in flips {
        for &x in &settings {
            let m = |f: &dyn Fn(&Row) -> f64| setting_mean(rows, fixed, flip, x, f);
            let (a, c, ic) = (
                Part::Class(Slice::All),
                Part::Class(Slice::Cn),
                Part::Class(Slice::Icn),
            );
            let _ = writeln!(
                o,
                "{flip},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{:.1}",
                x.0.label(),
                x.1,
                m(&|r| r.acc(a, K_MAX)),
                m(&|r| r.acc(c, K_MAX)),
                m(&|r| r.acc(ic, K_MAX)),
                m(&|r| r.learn(a, K_MAX)),
                m(&|r| r.learn(c, K_MAX)),
                m(&|r| r.forgetting(a, K_MAX)),
                m(&|r| r.forgetting(c, K_MAX)),
                m(&|r| r.last(a, K_MAX)),
                m(&|r| r.cost.iter().map(|c| c.weights_changed as f64).sum()),
                m(&|r| r.saturated_end as f64),
            );
        }
    }
    let subs = [
        Sub::Class(Slice::All),
        Sub::Class(Slice::Cn),
        Sub::Class(Slice::Icn),
        Sub::Class(Slice::Seen),
        Sub::Class(Slice::Ue),
        Sub::Reached,
        Sub::Unreached,
    ];
    let _ = writeln!(
        o,
        "gain(fixed, E/E-cue@level - error-gated): flip,k,subset,level,d_acc,d_learn,d_last,d_forget(+=cue_less),passes_rule"
    );
    for &flip in flips {
        for k in 2..=K_MAX {
            for s in subs {
                for &l in levels {
                    let c = gain(rows, flip, k, s, l);
                    let _ = writeln!(
                        o,
                        "{flip},{k},{},{l},{},{},{},{},{}",
                        s.name(),
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
    let _ = writeln!(
        o,
        "level_contrast(fixed,k=4): flip,subset,x_level,base_level,d_acc(E/E-cue@x - E/E-cue@base),d_forget"
    );
    for &flip in flips {
        for s in subs {
            for w in levels.windows(2) {
                let c = compare(rows, fixed, flip, K_MAX, s, (EE, w[1]), (EE, w[0]));
                let _ = writeln!(
                    o,
                    "{flip},{},{},{},{},{}",
                    s.name(),
                    w[1],
                    w[0],
                    p(&c.acc),
                    p(&c.forget)
                );
            }
            if levels.len() > 2 {
                let (hi, lo) = (levels[levels.len() - 1], levels[0]);
                let c = compare(rows, fixed, flip, K_MAX, s, (EE, hi), (EE, lo));
                let _ = writeln!(
                    o,
                    "{flip},{},{hi},{lo},{},{}",
                    s.name(),
                    p(&c.acc),
                    p(&c.forget)
                );
            }
        }
    }
    let _ = writeln!(
        o,
        "reach_slope(fixed,k=4): flip,subset,gain_per_unit_reach_fraction"
    );
    for &flip in flips {
        for s in [Sub::Class(Slice::All), Sub::Class(Slice::Cn)] {
            let _ = writeln!(
                o,
                "{flip},{},{:+.4}",
                s.name(),
                reach_slope(rows, flip, s, levels)
            );
        }
    }
    let _ = writeln!(
        o,
        "ss_cost(fixed, S/S-cue@level - supervised): flip,subset,level,d_acc,d_last"
    );
    for &flip in flips {
        for s in [Sub::Class(Slice::All), Sub::Class(Slice::Cn)] {
            for &l in levels {
                let c = compare(rows, fixed, flip, K_MAX, s, (SS, l), (SUPERVISED, 0));
                let _ = writeln!(o, "{flip},{},{l},{},{}", s.name(), p(&c.acc), p(&c.last));
            }
        }
    }
    let coupled: Vec<usize> = {
        let mut v: Vec<usize> = rows
            .iter()
            .filter(|r| r.family == Family::Coupled)
            .map(|r| r.level)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    if !coupled.is_empty() {
        let _ = writeln!(
            o,
            "coupled(#123 construction, k=4): flip,level,pool_mean,reach_mean,cn_median,eg_acc_all,eg_acc_cn,ee_acc_all,ee_acc_cn,gain_all,gain_cn,d_forget_cn"
        );
        let cf: Vec<u32> = {
            let mut v: Vec<u32> = rows
                .iter()
                .filter(|r| r.family == Family::Coupled)
                .map(|r| r.flip)
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        for flip in cf {
            for &l in &coupled {
                let fam = Family::Coupled;
                let eg = (ERROR_GATED, l);
                let ee = (EE, l);
                let a = Part::Class(Slice::All);
                let c = Part::Class(Slice::Cn);
                let cn = median(
                    rows.iter()
                        .filter(|r| {
                            r.family == fam
                                && r.flip == flip
                                && r.arm == ERROR_GATED
                                && r.level == l
                        })
                        .flat_map(|r| (0..K_MAX).map(move |j| r.pairs(c, j)))
                        .collect(),
                );
                let ga = compare(rows, fam, flip, K_MAX, Sub::Class(Slice::All), ee, eg);
                let gc = compare(rows, fam, flip, K_MAX, Sub::Class(Slice::Cn), ee, eg);
                let _ = writeln!(
                    o,
                    "{flip},{l},{:.1},{:.1},{cn},{:.3},{:.3},{:.3},{:.3},{},{},{}",
                    setting_mean(rows, fam, flip, eg, &|r| r.pool as f64),
                    setting_mean(rows, fam, flip, ee, &|r| mean(
                        r.cue_targets.iter().map(|t| t.len() as f64)
                    )),
                    setting_mean(rows, fam, flip, eg, &|r| r.acc(a, K_MAX)),
                    setting_mean(rows, fam, flip, eg, &|r| r.acc(c, K_MAX)),
                    setting_mean(rows, fam, flip, ee, &|r| r.acc(a, K_MAX)),
                    setting_mean(rows, fam, flip, ee, &|r| r.acc(c, K_MAX)),
                    p(&ga.acc),
                    p(&gc.acc),
                    p(&gc.forget),
                );
            }
        }
    }
    if flips.contains(&PRIMARY_FLIP) {
        let (cn, reach, h) = readings(rows, levels);
        let show = |x: Option<bool>| match x {
            None => "not_evaluable".to_string(),
            Some(true) => "pass".to_string(),
            Some(false) => "fail".to_string(),
        };
        let pf = |b: bool| if b { "pass" } else { "fail" };
        let _ = writeln!(
            o,
            "readings(flip={PRIMARY_FLIP}): cn_gate={} reach_gate={} H1={} H2={} H3={} H4={} H5={}",
            pf(cn),
            pf(reach),
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
    use crate::generalization_bench::GraphShape;
    use crate::reward_limits as rl;
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
                ..ns::DESIGN.spec
            },
            cue_cap: 4,
        }
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("v888-cue-reach-{tag}-{}.pls", std::process::id()))
    }

    #[test]
    fn prefixes_nest_and_reach_grows_with_level() {
        let g = rl::graph_for(None, &DESIGN.shape, 3).unwrap();
        let full = cc::choose_n_cues_capped(&g, K_MAX, FULL).unwrap();
        let mut prev: Vec<usize> = vec![0; K_MAX];
        for &l in &LEVELS {
            let c = prefix_cues(&full, l);
            for j in 0..K_MAX {
                assert!(c[j].len() <= l);
                assert_eq!(c[j][..], full[j][..c[j].len()]);
                let r = reach_set(&g, &c[j]).len();
                assert!(r >= prev[j]);
                prev[j] = r;
            }
        }
        // At FULL every cue of the pilot seed reaches every reachable target.
        assert!(prev.iter().all(|&r| r == reachable_targets(&g)));
        assert_eq!(prefix_cues(&full, 0), vec![Vec::<usize>::new(); K_MAX]);
        assert_eq!(prefix_cues(&full, 1000), full);
    }

    #[test]
    fn fixed_family_shares_pool_patterns_and_splits_across_levels() {
        let base = fixed_fixture(None, 50, 2).unwrap();
        let reserved = base.cues.concat();
        for &l in &LEVELS {
            let fx = at_level(&base, l);
            assert_eq!(fx.task, base.task);
            assert_eq!(fx.splits, base.splits);
            assert_eq!(fx.graph, base.graph);
            // Unused reserved lines are never in a pattern.
            for p in fx.task.train.iter().flatten().chain(&fx.task.test) {
                assert!(p.iter().all(|s| !reserved.contains(s)));
            }
        }
        // The coupled family at the largest level is the fixed fixture.
        let c = coupled_fixture(None, 50, 2, FULL).unwrap();
        assert_eq!(c.task, base.task);
        assert_eq!(c.cues, base.cues);
        // Smaller coupled levels widen the pool.
        let c2 = coupled_fixture(None, 50, 2, 2).unwrap();
        assert!(c2.task.pool.len() > base.task.pool.len());
    }

    #[test]
    fn census_counts_add_up() {
        let fx = fixed_fixture(None, 50, 1).unwrap();
        let c = census_fixture(&fx, &LEVELS);
        assert_eq!(c.lines.len(), LEVELS.len());
        assert_eq!(c.pool, fx.task.pool.len());
        for p in &c.pairs {
            assert_eq!(p.all, p.seen + p.cn + p.ue);
            assert!(p.icn <= p.cn);
        }
        for j in 0..K_MAX {
            assert!(c.reach.windows(2).all(|w| w[1][j] >= w[0][j]));
        }
    }

    #[test]
    fn per_target_confusions_match_all_pairs_and_partition() {
        let d = small();
        let base = ns::fixture(None, &d, 50, 4).unwrap();
        let fx = at_level(&base, 2);
        let r = run_arm(&fx, &d.spec, Family::Fixed, 2, EE, 4).unwrap();
        assert!(r.topology_unchanged);
        assert_eq!(r.cue_lines.iter().max(), Some(&2));
        for i in 0..K_MAX {
            for j in 0..K_MAX {
                let a = r.conf(Part::Class(Slice::All), i, j);
                let mut s = r.conf(Part::Reached(&r.cue_targets), i, j);
                add(&mut s, &r.conf(Part::Unreached(&r.cue_targets), i, j));
                assert_eq!(a, s);
            }
        }
        let u = run_arm(&base, &d.spec, Family::Fixed, 0, ERROR_GATED, 4).unwrap();
        assert_eq!(u.cue_lines, [0; K_MAX]);
    }

    #[test]
    fn run_resumes_bit_exactly_from_checkpoints() {
        let d = small();
        let base = ns::fixture(None, &d, 50, 6).unwrap();
        let fx = at_level(&base, 2);
        for a in [SUPERVISED, ERROR_GATED, EE, SS] {
            let full = run_arm(&fx, &d.spec, Family::Fixed, 2, a, 6).unwrap();
            let mut c = PlasticCircuit::new(&fx.graph, d.spec.init_weight).unwrap();
            let episodes = d.spec.train_patterns * d.spec.rounds_a as usize;
            let mut cost = [PhaseCost::default(); K_MAX];
            let half = episodes / 2;
            for (i, ci) in cost.iter_mut().enumerate() {
                for r in [0..half, half..episodes] {
                    ts::train_task(
                        &mut c, &fx.graph, a, &fx.task, &d.spec, 6, i, r, &fx.cues, ci,
                    )
                    .unwrap();
                    let p = tmp("mid");
                    c.save(&p).unwrap();
                    c = PlasticCircuit::load(&p, &fx.graph).unwrap();
                    fs::remove_file(&p).unwrap();
                }
            }
            assert_eq!(cost, full.cost);
            let n = fx.graph.ids.len();
            for j in 0..K_MAX {
                let (e, per) = evaluate(
                    &c,
                    &fx.graph,
                    &fx.task.test,
                    &fx.task.valence[j],
                    &ts::active_cue(a, &fx.cues, j),
                    &fx.splits[j],
                )
                .unwrap();
                assert_eq!(e, full.ev[K_MAX - 1][j]);
                let b = ((K_MAX - 1) * K_MAX + j) * n;
                assert_eq!(per[..], full.tgt[b..b + n]);
            }
        }
    }

    #[test]
    fn grid_is_deterministic_csv_complete_and_readings_gate() {
        let d = small();
        let levels = [1, 2, 4];
        let run = || {
            let mut rows = run_fixed_design(None, &d, &[50], &[1, 2, 3], &levels).unwrap();
            rows.extend(run_coupled(None, &d, 50, &[1, 2, 3], &levels).unwrap());
            rows
        };
        let a = run();
        let b = run();
        assert_eq!(a, b);
        assert_eq!(a.len(), 3 * (2 + 2 * levels.len()) + 3 * 2 * levels.len());
        assert!(a.iter().all(|r| r.topology_unchanged));
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        let cols = csv_header().split(',').count();
        assert!(csv.lines().all(|l| l.split(',').count() == cols));
        // Self comparison is all ties and fails the rule.
        let id = compare(&a, Family::Fixed, 50, K_MAX, Sub::Reached, (EE, 2), (EE, 2));
        assert_eq!(id.acc.ties, 3);
        assert!(!id.passes());
        let (cn, reach, h) = readings(&a, &levels);
        if !(cn && reach) {
            assert_eq!(h, [None; 5]);
        }
        let rep = report(&a, &[50], &levels);
        assert!(rep.contains("readings(flip=50): cn_gate="));
        assert!(rep.contains("coupled(#123 construction"));
        assert_eq!(rep, report(&b, &[50], &levels));
        let s = reach_slope(&a, 50, Sub::Class(Slice::All), &levels);
        assert!(s.is_finite());
        assert!(run_fixed_design(None, &d, &[50], &[], &levels).is_err());
        assert!(run_fixed_design(None, &d, &[50], &[1], &[0]).is_err());
        assert!(run_fixed_design(None, &d, &[50], &[1], &[d.cue_cap + 1]).is_err());
        assert!(run_coupled(None, &d, 50, &[1], &[0]).is_err());
    }

    #[test]
    fn dose_response_and_gates_on_hand_rows() {
        // Gates on an empty grid fail and readings are not evaluable.
        let (cn, reach, h) = readings(&[], &LEVELS);
        assert!(!cn && !reach);
        assert_eq!(h, [None; 5]);
        assert!(!dose_response(&[], 50, Sub::Class(Slice::All), &[2]));
    }

    #[test]
    fn primary_design_passes_label_free_gates_on_pilot_seed() {
        // Census only (no learning): CN pairs and reach ordering, seed 1.
        let fx = fixed_fixture(None, PRIMARY_FLIP, 1).unwrap();
        let c = census_fixture(&fx, &LEVELS);
        assert!(c.pairs.iter().all(|p| p.cn >= ns::MIN_NOVEL_PAIRS));
        for j in 0..K_MAX {
            assert!(c.reach.windows(2).all(|w| w[1][j] > w[0][j]));
            assert_eq!(c.reach[LEVELS.len() - 1][j], c.reachable);
        }
    }
}

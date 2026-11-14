//! B888-4 cycle 14: **equal weight-write budget** across strategies, over a
//! label-flip sweep.
//!
//! Cycle 13 (#124) compared arms at an equal *episode* budget, but the arms
//! did not write the same number of weights: on the fixed-pool design
//! `E/E-cue@16` changed 445 weights over four tasks, `error-gated` 209 and
//! `supervised` 3594. Part of the cue gain could therefore be bought by
//! extra writes. Its flip dependence (CN gain −0.009 / +0.018 / +0.110 at
//! flips 25 / 50 / 100) was also only a secondary reading.
//!
//! This module limits the number of **effective weight writes** (a 4-bit
//! step that actually changes a stored coefficient; steps at the 0/15
//! plateaus are free and never refused) per task, as a fraction of what the
//! uncapped `error-gated` arm writes on the same (flip, seed, task):
//! * [`Budget::Paced`] (primary): after every episode `e` an arm may have
//!   written at most `pct` % of the writes uncapped `error-gated` had made
//!   by episode `e` (its cumulative write curve). Every arm therefore gets
//!   the same write **schedule** as `error-gated`, and `error-gated` itself
//!   is unchanged at 100 % (tested bit-exactly).
//! * [`Budget::Total`] (secondary): one cap on the task total, spent in
//!   episode order (front-loaded: an arm that writes a lot per episode
//!   exhausts it in the first episodes).
//!
//! A step that would change a weight when no allowance is left is refused
//! (counted); the arm keeps seeing episodes. Inside an episode edges are
//! visited source-major (as `learning_strategies::apply_update_gated`).
//!
//! Strategies (all on the #124 fixed-pool fixture, cue level 16 = full
//! reach for the cued arm):
//! * `supervised` (S): teacher step on every edge of every active line;
//! * `error-gated` (E): teacher step only on the edges into a reached target
//!   the circuit gets wrong;
//! * `E/E-cue` (EE): `error-gated` on pattern and cue edges with the task's
//!   full cue on;
//! * `reward` (R): reward-modulated, clean per-target reward (as the
//!   `PerTarget` reference of `reward_limits`): for every reached target
//!   that fires, the edges from the active lines into it are reinforced if
//!   firing was right and weakened otherwise. Uncued.
//!
//! Every arm acts only on the original edges (checked per row).
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph; nothing here is
//! a claim about synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/WRITE_BUDGET.md`
//! (committed before any learning run).
use crate::connectome::Connectome;
use crate::context_cue as cc;
use crate::cue_reach::{self as cr, Family, Part};
use crate::error_gated::{drive, paired, Paired, PhaseCost, EFFECT, THRESHOLD};
use crate::generalization_bench::{Confusion, GenSpec};
use crate::novel_split::{self as ns, Design, Eval, Fixture, Slice, SLICES};
use crate::plasticity::{decrement_four, increment_four, PlasticCircuit};
use crate::task_sequence::{self as ts, Arm, K_MAX};
use std::io::{self, ErrorKind};

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Pre-registered label-flip sweep (50 primary, 100 high-conflict).
pub const FLIPS: [u32; 4] = [25, 50, 75, 100];
pub const PRIMARY_FLIP: u32 = 50;
pub const HIGH_FLIP: u32 = 100;
/// Pre-registered seeds: 161..=200, never used by an earlier B888-4 cycle.
pub const SEEDS: std::ops::RangeInclusive<u64> = 161..=200;
/// Cue level of the cued arm (lines per cue; full reach on the #124 pool).
pub const CUE_LEVEL: usize = cr::FULL;
/// Write budget of a run, in percent of the uncapped `error-gated` writes
/// of the same (flip, seed, task).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Budget {
    /// No limit.
    Free,
    /// Cumulative writes after each episode ≤ pct % of `error-gated`'s.
    Paced(u32),
    /// Task total ≤ pct % of `error-gated`'s, spent in episode order.
    Total(u32),
}

impl Budget {
    #[must_use]
    pub fn name(self) -> String {
        match self {
            Self::Free => "free".to_string(),
            Self::Paced(p) => format!("paced{p}"),
            Self::Total(p) => format!("total{p}"),
        }
    }
    /// Parse `free`, `pacedN` or `totalN`.
    pub fn parse(s: &str) -> Result<Self, std::num::ParseIntError> {
        if s == "free" {
            Ok(Self::Free)
        } else if let Some(p) = s.strip_prefix("paced") {
            p.parse().map(Self::Paced)
        } else if let Some(p) = s.strip_prefix("total") {
            p.parse().map(Self::Total)
        } else {
            s.parse::<u32>().map(Self::Paced)
        }
    }
    fn pct(self) -> Option<u32> {
        match self {
            Self::Free => None,
            Self::Paced(p) | Self::Total(p) => Some(p),
        }
    }
}

/// Pre-registered budgets. Primary: `Paced(100)`.
pub const BUDGETS: [Budget; 4] = [
    Budget::Free,
    Budget::Paced(100),
    Budget::Total(100),
    Budget::Paced(50),
];
pub const PRIMARY_BUDGET: Budget = Budget::Paced(100);
/// Monotonicity tolerance of the flip dose-response (H3).
pub const MONO_TOL: f64 = 0.005;
/// Non-inferiority margin of H7.
pub const NONINF: f64 = EFFECT;
/// Fixed-pool design of #124 (cues reserved up to 16 lines).
pub const DESIGN: Design = cr::DESIGN;

/// Update rule of one arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Strategy {
    Supervised,
    ErrorGated,
    EeCue,
    Reward,
}

/// The pre-registered strategies, in CSV order.
pub const STRATEGIES: [Strategy; 4] = [
    Strategy::Supervised,
    Strategy::ErrorGated,
    Strategy::EeCue,
    Strategy::Reward,
];

impl Strategy {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Supervised => "supervised",
            Self::ErrorGated => "error-gated",
            Self::EeCue => "E/E-cue",
            Self::Reward => "reward",
        }
    }
    /// Is the task cue on for this strategy?
    #[must_use]
    pub fn cued(self) -> bool {
        self == Self::EeCue
    }
    /// Gate arm used for the cue bookkeeping of `cue_reach` rows. The
    /// reward arm is uncued and is recorded with the uncued supervised arm.
    #[must_use]
    pub fn arm(self) -> Arm {
        match self {
            Self::Supervised | Self::Reward => cr::SUPERVISED,
            Self::ErrorGated => cr::ERROR_GATED,
            Self::EeCue => cr::EE,
        }
    }
}

/// Budget label for reports.
#[must_use]
pub fn budget_name(b: Budget) -> String {
    b.name()
}

/// Write limit of one task: `allow[e]` = maximum cumulative writes during
/// episode `e` (0-based); `None` = unlimited.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Limit(pub Option<Vec<u64>>);

impl Limit {
    fn at(&self, episode: usize) -> u64 {
        self.0
            .as_ref()
            .map_or(u64::MAX, |a| a.get(episode).copied().unwrap_or(u64::MAX))
    }
}

/// Per-task write accounting of the budget.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WriteLedger {
    /// Final allowance of this task (`u64::MAX` if unlimited).
    pub cap: u64,
    /// Effective writes made.
    pub writes: u64,
    /// Steps refused that would have changed a weight.
    pub refused: u64,
    /// 1-based episode of the first refusal (0 = never).
    pub first_refusal: u64,
    /// Cumulative writes after each episode.
    pub cum: Vec<u64>,
    /// Allowance of the current episode (set by `train_task`).
    pub allow: u64,
}

impl WriteLedger {
    #[must_use]
    pub fn new(limit: &Limit) -> Self {
        Self {
            cap: limit
                .0
                .as_ref()
                .map_or(u64::MAX, |a| a.last().copied().unwrap_or(0)),
            allow: u64::MAX,
            ..Self::default()
        }
    }
}

/// One episode of `strategy` under a write ledger. Gate decisions use the
/// drive (including the cue) on the weights before the update. With an
/// uncapped ledger S / E / EE are bit-identical to `cue_gating::step` and R
/// to the clean per-target reward step (tested).
#[allow(clippy::too_many_arguments)]
pub fn step(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    strategy: Strategy,
    pattern: &[usize],
    valence: &[bool],
    cue: &[usize],
    ledger: &mut WriteLedger,
    cost: &mut PhaseCost,
) -> io::Result<()> {
    let (input, reached, teacher) = cc::encode_ctx(graph, pattern, valence, cue)?;
    let n = graph.ids.len();
    let mut want = vec![false; n];
    for (t, w) in want.iter_mut().enumerate() {
        *w = teacher.get(t).map_err(|_| bad("bad target"))?;
    }
    // Per-target: is the incoming update offered, and in which direction.
    let mut offer = vec![strategy == Strategy::Supervised; n];
    let mut dir = want.clone();
    if strategy != Strategy::Supervised {
        let d = drive(circuit, graph, &input)?;
        for &t in &reached {
            let fires = crate::gates::ge(d[t], THRESHOLD);
            match strategy {
                Strategy::ErrorGated | Strategy::EeCue => offer[t] = fires != want[t],
                Strategy::Reward => {
                    offer[t] = fires;
                    dir[t] = want[t];
                }
                Strategy::Supervised => {}
            }
        }
    }
    cost.episodes += 1;
    cost.targets_seen += reached.len() as u64;
    cost.targets_updated += reached.iter().filter(|&&t| offer[t]).count() as u64;
    for source in input.active_indices() {
        for edge in graph.offsets[source]..graph.offsets[source + 1] {
            let edge = edge as usize;
            let t = graph.targets[edge] as usize;
            if !offer[t] {
                continue;
            }
            if ledger.writes >= ledger.allow {
                let w = circuit.weights()[edge];
                let next = if dir[t] {
                    increment_four(w)
                } else {
                    decrement_four(w)
                };
                if next != w {
                    ledger.refused += 1;
                    if ledger.first_refusal == 0 {
                        ledger.first_refusal = cost.episodes;
                    }
                    continue;
                }
            }
            cost.attempts += 1;
            if circuit.apply_edge_step(edge, dir[t])? {
                cost.weights_changed += 1;
                ledger.writes += 1;
            }
        }
    }
    ledger.cum.push(ledger.writes);
    Ok(())
}

/// Train all episodes of task `i` under a write limit.
#[allow(clippy::too_many_arguments)]
pub fn train_task(
    circuit: &mut PlasticCircuit,
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    strategy: Strategy,
    i: usize,
    limit: &Limit,
    cost: &mut PhaseCost,
) -> io::Result<WriteLedger> {
    let order = ts::task_order(spec, seed, i);
    let cue = if strategy.cued() {
        fx.cues[i].clone()
    } else {
        Vec::new()
    };
    let mut ledger = WriteLedger::new(limit);
    for (e, &ix) in order.iter().enumerate() {
        ledger.allow = limit.at(e);
        step(
            circuit,
            &fx.graph,
            strategy,
            &fx.task.train[i][ix],
            &fx.task.valence[i],
            &cue,
            &mut ledger,
            cost,
        )?;
    }
    ledger.allow = ledger.cap;
    Ok(ledger)
}

/// One measurement: (flip, seed, strategy, budget) on a K_MAX-task run.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub strategy: Strategy,
    pub budget: Budget,
    /// Metrics and cue bookkeeping (`inner.level` = cue lines of the cued
    /// arm, 0 otherwise).
    pub inner: cr::Row,
    pub ledger: [WriteLedger; K_MAX],
    /// Final weights equal to the uncapped run of the same strategy.
    pub same_as_free: bool,
}

/// Run one strategy with per-task write limits.
#[allow(clippy::needless_range_loop)]
pub fn run_arm(
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    strategy: Strategy,
    limits: &[Limit; K_MAX],
) -> io::Result<(cr::Row, [WriteLedger; K_MAX], Vec<u8>)> {
    let graph = &fx.graph;
    if fx.task.train.len() != K_MAX || fx.cues.len() != K_MAX || fx.splits.len() != K_MAX {
        return Err(bad("need K_MAX tasks, cues and splits"));
    }
    graph.validate()?;
    let original = graph.clone();
    let n = graph.ids.len();
    let mut circuit = PlasticCircuit::new(graph, spec.init_weight)?;
    let fingerprint = circuit.topology_fingerprint();
    let mut ev = [[Eval::default(); K_MAX]; K_MAX];
    let mut tgt = vec![Confusion::default(); K_MAX * K_MAX * n];
    let mut cost = [PhaseCost::default(); K_MAX];
    let mut ledger: [WriteLedger; K_MAX] = Default::default();
    for i in 0..K_MAX {
        ledger[i] = train_task(
            &mut circuit,
            fx,
            spec,
            seed,
            strategy,
            i,
            &limits[i],
            &mut cost[i],
        )?;
        for j in 0..K_MAX {
            let cue = if strategy.cued() {
                fx.cues[j].clone()
            } else {
                Vec::new()
            };
            let (e, per) = cr::evaluate(
                &circuit,
                graph,
                &fx.task.test,
                &fx.task.valence[j],
                &cue,
                &fx.splits[j],
            )?;
            ev[i][j] = e;
            tgt[(i * K_MAX + j) * n..(i * K_MAX + j + 1) * n].copy_from_slice(&per);
        }
    }
    let mut cue_lines = [0; K_MAX];
    if strategy.cued() {
        for j in 0..K_MAX {
            cue_lines[j] = fx.cues[j].len();
        }
    }
    let row = cr::Row {
        family: Family::Fixed,
        level: if strategy.cued() { CUE_LEVEL } else { 0 },
        arm: strategy.arm(),
        flip: spec.flip_pct,
        seed,
        pool: fx.task.pool.len(),
        reachable: cr::reachable_targets(graph),
        cue_lines,
        cue_targets: fx.cues.iter().map(|c| cr::reach_set(graph, c)).collect(),
        ev,
        tgt,
        nodes: n,
        cost,
        saturated_end: crate::error_gated::saturated(circuit.weights()),
        topology_unchanged: *graph == original
            && circuit.topology_fingerprint() == fingerprint
            && circuit.weights().len() == graph.edges(),
    };
    Ok((row, ledger, circuit.weights().to_vec()))
}

/// Per-task limits of `budget` from the reference cumulative write curves
/// `curves[i][e]` (floor of pct %).
#[must_use]
pub fn limits(curves: &[Vec<u64>], budget: Budget) -> [Limit; K_MAX] {
    let mut out: [Limit; K_MAX] = Default::default();
    for (l, c) in out.iter_mut().zip(curves) {
        let total = c.last().copied().unwrap_or(0);
        *l = Limit(match budget {
            Budget::Free => None,
            Budget::Paced(p) => Some(c.iter().map(|&w| w * u64::from(p) / 100).collect()),
            Budget::Total(p) => Some(vec![total * u64::from(p) / 100; c.len()]),
        });
    }
    out
}

/// Free limits (unlimited) for every task.
#[must_use]
pub fn free() -> [Limit; K_MAX] {
    Default::default()
}

/// Full grid on a design: per (flip, seed) uncapped `error-gated` first
/// (its per-task writes set the caps), then every (strategy, budget).
pub fn run_grid_design(
    graph: Option<&Connectome>,
    design: &Design,
    flips: &[u32],
    seeds: &[u64],
    budgets: &[Budget],
) -> io::Result<Vec<Row>> {
    if seeds.is_empty() || flips.is_empty() || budgets.is_empty() {
        return Err(bad("need at least one seed, flip rate and budget"));
    }
    if budgets
        .iter()
        .filter_map(|b| b.pct())
        .any(|p| p == 0 || p > 100)
    {
        return Err(bad("budget must be in 1..=100 % or uncapped"));
    }
    let mut rows = Vec::new();
    for &flip in flips {
        let spec = GenSpec {
            flip_pct: flip,
            ..design.spec
        };
        for &seed in seeds {
            let fx = ns::fixture(graph, design, flip, seed)?;
            let (_, eg, _) = run_arm(&fx, &spec, seed, Strategy::ErrorGated, &free())?;
            let curves: Vec<Vec<u64>> = eg.iter().map(|l| l.cum.clone()).collect();
            for s in STRATEGIES {
                let (_, _, free_w) = run_arm(&fx, &spec, seed, s, &free())?;
                for &b in budgets {
                    let (inner, ledger, w) = run_arm(&fx, &spec, seed, s, &limits(&curves, b))?;
                    rows.push(Row {
                        strategy: s,
                        budget: b,
                        inner,
                        ledger,
                        same_as_free: w == free_w,
                    });
                }
            }
        }
    }
    Ok(rows)
}

/// Pre-registered grid on the #124 fixed-pool design.
pub fn run_grid(
    graph: Option<&Connectome>,
    flips: &[u32],
    seeds: &[u64],
    budgets: &[Budget],
) -> io::Result<Vec<Row>> {
    run_grid_design(graph, &DESIGN, flips, seeds, budgets)
}

/// A (strategy, budget) setting.
pub type Setting = (Strategy, Budget);

/// The base of every gain: uncapped `error-gated` (identical to its runs at
/// 100 % budgets).
pub const EG: Setting = (Strategy::ErrorGated, Budget::Free);

fn find(rows: &[Row], flip: u32, seed: u64, x: Setting) -> Option<&Row> {
    rows.iter().find(|r| {
        r.inner.flip == flip && r.inner.seed == seed && r.strategy == x.0 && r.budget == x.1
    })
}

fn rows_of(rows: &[Row], flip: u32, x: Setting) -> impl Iterator<Item = &Row> {
    rows.iter()
        .filter(move |r| r.inner.flip == flip && r.strategy == x.0 && r.budget == x.1)
}

/// x vs base paired by seed (same oriented fields as `cue_reach`).
#[must_use]
pub fn compare(
    rows: &[Row],
    flip: u32,
    k: usize,
    s: Slice,
    x: Setting,
    base: Setting,
) -> cr::Comparison {
    let p = Part::Class(s);
    let pairs: Vec<(&Row, &Row)> = rows_of(rows, flip, x)
        .filter_map(|r| find(rows, flip, r.inner.seed, base).map(|c| (r, c)))
        .collect();
    let d = |f: &dyn Fn(&cr::Row, &cr::Row) -> f64| {
        paired(
            &pairs
                .iter()
                .map(|(r, c)| f(&r.inner, &c.inner))
                .collect::<Vec<_>>(),
        )
    };
    cr::Comparison {
        acc: d(&|r, c| r.acc(p, k) - c.acc(p, k)),
        learn: d(&|r, c| r.learn(p, k) - c.learn(p, k)),
        last: d(&|r, c| r.last(p, k) - c.last(p, k)),
        forget: d(&|r, c| c.forgetting(p, k) - r.forgetting(p, k)),
    }
}

/// Per-seed ACC gain of x over base at k = 4 on `s`, keyed by seed.
fn seed_gains(rows: &[Row], flip: u32, s: Slice, x: Setting, base: Setting) -> Vec<(u64, f64)> {
    let p = Part::Class(s);
    rows_of(rows, flip, x)
        .filter_map(|r| {
            find(rows, flip, r.inner.seed, base)
                .map(|c| (r.inner.seed, r.inner.acc(p, K_MAX) - c.inner.acc(p, K_MAX)))
        })
        .collect()
}

/// Paired (by seed) difference of the gain of x over base between two
/// flips: gain@hi − gain@lo.
#[must_use]
pub fn flip_contrast(
    rows: &[Row],
    s: Slice,
    x: Setting,
    base: Setting,
    lo: u32,
    hi: u32,
) -> Paired {
    let a = seed_gains(rows, lo, s, x, base);
    let b = seed_gains(rows, hi, s, x, base);
    let d: Vec<f64> = b
        .iter()
        .filter_map(|(seed, gh)| a.iter().find(|(s2, _)| s2 == seed).map(|(_, gl)| gh - gl))
        .collect();
    paired(&d)
}

fn median(mut v: Vec<u64>) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

/// Median CN pairs per (seed, task) at a flip.
#[must_use]
pub fn median_cn(rows: &[Row], flip: u32) -> u64 {
    median(
        rows_of(rows, flip, EG)
            .flat_map(|r| (0..K_MAX).map(move |j| r.inner.pairs(Part::Class(Slice::Cn), j)))
            .collect(),
    )
}

/// Fraction of (seed, task) where the cap refused at least one write.
#[must_use]
pub fn binding_frac(rows: &[Row], flip: u32, x: Setting) -> f64 {
    let v: Vec<bool> = rows_of(rows, flip, x)
        .flat_map(|r| r.ledger.iter().map(|l| l.refused > 0))
        .collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().filter(|&&b| b).count() as f64 / v.len() as f64
    }
}

/// Design gates: (G1 budget integrity, G2 budget binds for EE at the
/// primary flip, G3 CN pairs). G1: every row keeps its topology, every
/// task's writes ≤ its final allowance, and `error-gated` at every 100 %
/// budget is bit-identical to its uncapped run with no refusal.
#[must_use]
pub fn gates(rows: &[Row]) -> (bool, bool, bool) {
    let g1 = !rows.is_empty()
        && rows.iter().all(|r| {
            r.inner.topology_unchanged
                && r.ledger.iter().all(|l| l.writes <= l.cap)
                && (r.strategy != Strategy::ErrorGated
                    || r.budget.pct().is_some_and(|p| p < 100)
                    || (r.same_as_free && r.ledger.iter().all(|l| l.refused == 0)))
        });
    let g2 = binding_frac(rows, PRIMARY_FLIP, (Strategy::EeCue, PRIMARY_BUDGET)) >= 0.5;
    let g3 = median_cn(rows, PRIMARY_FLIP) >= ns::MIN_NOVEL_PAIRS;
    (g1, g2, g3)
}

/// Number of pre-registered hypotheses.
pub const N_HYP: usize = 7;

/// Pre-registered readings: (gates, [H1..H7]); hypotheses are `None` if a
/// gate fails or their flip is missing.
#[must_use]
pub fn readings(rows: &[Row], flips: &[u32]) -> ((bool, bool, bool), [Option<bool>; N_HYP]) {
    let g = gates(rows);
    let mut h = [None; N_HYP];
    if !(g.0 && g.1 && g.2) {
        return (g, h);
    }
    let ee = (Strategy::EeCue, PRIMARY_BUDGET);
    let ee_free = (Strategy::EeCue, Budget::Free);
    let s1 = (Strategy::Supervised, PRIMARY_BUDGET);
    let r1 = (Strategy::Reward, PRIMARY_BUDGET);
    let (all, cn) = (Slice::All, Slice::Cn);
    let f = PRIMARY_FLIP;
    let hf = HIGH_FLIP;
    let has = |x: u32| flips.contains(&x);
    if has(f) {
        let c = compare(rows, f, K_MAX, all, ee, EG);
        h[0] = Some(c.passes());
        let free = compare(rows, f, K_MAX, all, ee_free, EG);
        h[1] = Some(c.acc.n > 0 && c.acc.ci_lo > 0.0 && c.acc.mean >= 0.5 * free.acc.mean);
        h[5] = Some(compare(rows, f, K_MAX, cn, EG, s1).passes());
        let r = compare(rows, f, K_MAX, all, r1, EG);
        h[6] = Some(r.acc.n > 0 && r.acc.ci_lo > -NONINF);
    }
    let sweep: Vec<u32> = FLIPS.iter().copied().filter(|x| has(*x)).collect();
    if sweep.len() == FLIPS.len() {
        let m: Vec<f64> = sweep
            .iter()
            .map(|&x| compare(rows, x, K_MAX, cn, ee, EG).acc.mean)
            .collect();
        let mono = m.windows(2).all(|w| w[1] >= w[0] - MONO_TOL);
        let end = flip_contrast(rows, cn, ee, EG, FLIPS[0], FLIPS[FLIPS.len() - 1]);
        h[2] = Some(mono && end.n > 0 && end.ci_lo > 0.0);
    }
    if has(hf) {
        let c = compare(rows, hf, K_MAX, cn, ee, EG);
        h[3] = Some(c.passes());
        h[4] = Some(c.forget.strong());
    }
    (g, h)
}

/// CSV header (generated; see `to_csv`).
#[must_use]
pub fn csv_header() -> String {
    let mut h: Vec<String> = [
        "schema",
        "graph",
        "flip_pct",
        "strategy",
        "budget",
        "seed",
        "pool",
        "reachable_targets",
        "cue_lines_total",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
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
    for name in ["cap", "writes", "refused", "first_refusal", "attempts"] {
        for j in 1..=K_MAX {
            h.push(format!("{name}_{j}"));
        }
    }
    for name in ["saturated_end", "same_as_free", "topology_unchanged"] {
        h.push(name.to_string());
    }
    h.join(",")
}

/// Deterministic CSV (no timing column).
#[must_use]
pub fn to_csv(rows: &[Row], graph_label: &str) -> String {
    let mut out = csv_header();
    out.push('\n');
    for r in rows {
        let x = &r.inner;
        let mut f: Vec<String> = vec![
            "b888-4-write-budget-v1".to_string(),
            graph_label.to_string(),
            x.flip.to_string(),
            r.strategy.name().to_string(),
            r.budget.name(),
            x.seed.to_string(),
            x.pool.to_string(),
            x.reachable.to_string(),
            x.cue_lines.iter().sum::<usize>().to_string(),
        ];
        for s in SLICES {
            f.extend((0..K_MAX).map(|j| x.pairs(Part::Class(s), j).to_string()));
        }
        for s in SLICES {
            for i in 0..K_MAX {
                f.extend((0..K_MAX).map(|j| format!("{:.6}", x.r(Part::Class(s), i, j))));
            }
        }
        for s in SLICES {
            let p = Part::Class(s);
            for k in 2..=K_MAX {
                f.push(format!("{:.6}", x.acc(p, k)));
            }
            for k in 2..=K_MAX {
                f.push(format!("{:.6}", x.learn(p, k)));
            }
            for k in 2..=K_MAX {
                f.push(format!("{:.6}", x.forgetting(p, k)));
            }
            for k in 2..=K_MAX {
                f.push(format!("{:.6}", x.last(p, k)));
            }
        }
        f.extend(r.ledger.iter().map(|l| {
            if l.cap == u64::MAX {
                "-".to_string()
            } else {
                l.cap.to_string()
            }
        }));
        f.extend(r.ledger.iter().map(|l| l.writes.to_string()));
        f.extend(r.ledger.iter().map(|l| l.refused.to_string()));
        f.extend(r.ledger.iter().map(|l| l.first_refusal.to_string()));
        f.extend(x.cost.iter().map(|c| c.attempts.to_string()));
        f.push(x.saturated_end.to_string());
        f.push(r.same_as_free.to_string());
        f.push(x.topology_unchanged.to_string());
        out.push_str(&f.join(","));
        out.push('\n');
    }
    out
}

fn mean(v: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = v.collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Mean of `f` over the rows of one (flip, setting).
#[must_use]
pub fn setting_mean(rows: &[Row], flip: u32, x: Setting, f: &dyn Fn(&Row) -> f64) -> f64 {
    mean(rows_of(rows, flip, x).map(f))
}

fn yes(b: Option<bool>) -> &'static str {
    match b {
        Some(true) => "pass",
        Some(false) => "fail",
        None => "not_evaluable",
    }
}

/// Human-readable report (deterministic; printed by the CLI).
#[must_use]
pub fn report(rows: &[Row], flips: &[u32], budgets: &[Budget]) -> String {
    use std::fmt::Write;
    let mut o = String::new();
    let p = |d: &Paired| {
        format!(
            "{:+.3}±{:.3}[{:+.3},{:+.3}]{}/{}/{}",
            d.mean, d.sd, d.ci_lo, d.ci_hi, d.better, d.ties, d.worse
        )
    };
    let (a, c, ic) = (
        Part::Class(Slice::All),
        Part::Class(Slice::Cn),
        Part::Class(Slice::Icn),
    );
    let _ = writeln!(
        o,
        "setting_summary(k=4): flip,strategy,budget,acc_all,acc_cn,acc_icn,acc_seen,learn_all,learn_cn,fgt_all,fgt_cn,last_all,writes_total,cap_total,refused_total,binding_frac,first_refusal_episode_mean(bound tasks),saturated_end"
    );
    for &flip in flips {
        for s in STRATEGIES {
            for &b in budgets {
                let x = (s, b);
                let m = |f: &dyn Fn(&Row) -> f64| setting_mean(rows, flip, x, f);
                let ex: Vec<f64> = rows_of(rows, flip, x)
                    .flat_map(|r| r.ledger.iter())
                    .filter(|l| l.first_refusal > 0)
                    .map(|l| l.first_refusal as f64)
                    .collect();
                let _ = writeln!(
                    o,
                    "{flip},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{},{:.1},{:.3},{:.1},{:.1}",
                    s.name(),
                    budget_name(b),
                    m(&|r| r.inner.acc(a, K_MAX)),
                    m(&|r| r.inner.acc(c, K_MAX)),
                    m(&|r| r.inner.acc(ic, K_MAX)),
                    m(&|r| r.inner.acc(Part::Class(Slice::Seen), K_MAX)),
                    m(&|r| r.inner.learn(a, K_MAX)),
                    m(&|r| r.inner.learn(c, K_MAX)),
                    m(&|r| r.inner.forgetting(a, K_MAX)),
                    m(&|r| r.inner.forgetting(c, K_MAX)),
                    m(&|r| r.inner.last(a, K_MAX)),
                    m(&|r| r.ledger.iter().map(|l| l.writes as f64).sum()),
                    if b != Budget::Free {
                        format!(
                            "{:.1}",
                            m(&|r| r.ledger.iter().map(|l| l.cap as f64).sum())
                        )
                    } else {
                        "-".to_string()
                    },
                    m(&|r| r.ledger.iter().map(|l| l.refused as f64).sum()),
                    binding_frac(rows, flip, x),
                    mean(ex.into_iter()),
                    m(&|r| r.inner.saturated_end as f64),
                );
            }
        }
    }
    let _ = writeln!(
        o,
        "compare: flip,k,subset,x,base,d_acc,d_learn,d_last,d_forget(+=x_forgets_less),passes_rule"
    );
    let subs = [Slice::All, Slice::Cn, Slice::Icn, Slice::Seen, Slice::Ue];
    let mut pairs: Vec<(Setting, Setting)> = Vec::new();
    for &b in budgets {
        pairs.push(((Strategy::EeCue, b), EG));
        pairs.push(((Strategy::Supervised, b), EG));
        pairs.push(((Strategy::Reward, b), EG));
        pairs.push((EG, (Strategy::Supervised, b)));
        pairs.push(((Strategy::Reward, b), (Strategy::Supervised, b)));
        if b != Budget::Free {
            pairs.push(((Strategy::ErrorGated, b), EG));
            pairs.push(((Strategy::EeCue, b), (Strategy::EeCue, Budget::Free)));
        }
    }
    for &flip in flips {
        for k in 2..=K_MAX {
            for s in subs {
                for &(x, base) in &pairs {
                    let cmp = compare(rows, flip, k, s, x, base);
                    let _ = writeln!(
                        o,
                        "{flip},{k},{},{}@{},{}@{},{},{},{},{},{}",
                        s.name(),
                        x.0.name(),
                        budget_name(x.1),
                        base.0.name(),
                        budget_name(base.1),
                        p(&cmp.acc),
                        p(&cmp.learn),
                        p(&cmp.last),
                        p(&cmp.forget),
                        cmp.passes()
                    );
                }
            }
        }
    }
    let _ = writeln!(
        o,
        "flip_contrast(k=4, gain of x over error-gated at flip hi minus at flip lo, paired by seed): subset,x,lo,hi,d"
    );
    for s in [Slice::All, Slice::Cn] {
        for &b in budgets {
            let x = (Strategy::EeCue, b);
            for w in flips.windows(2) {
                let d = flip_contrast(rows, s, x, EG, w[0], w[1]);
                let _ = writeln!(
                    o,
                    "{},E/E-cue@{},{},{},{}",
                    s.name(),
                    budget_name(b),
                    w[0],
                    w[1],
                    p(&d)
                );
            }
            if let (Some(&lo), Some(&hi)) = (flips.first(), flips.last()) {
                let d = flip_contrast(rows, s, x, EG, lo, hi);
                let _ = writeln!(
                    o,
                    "{},E/E-cue@{},{lo},{hi},{}",
                    s.name(),
                    budget_name(b),
                    p(&d)
                );
            }
        }
    }
    let (g, h) = readings(rows, flips);
    let _ = writeln!(
        o,
        "readings: G1_cap_integrity={} G2_cap_binds_ee={} (binding_frac={:.3}) G3_cn_pairs={} (median_cn={})",
        g.0,
        g.1,
        binding_frac(rows, PRIMARY_FLIP, (Strategy::EeCue, PRIMARY_BUDGET)),
        g.2,
        median_cn(rows, PRIMARY_FLIP)
    );
    for (i, v) in h.iter().enumerate() {
        let _ = writeln!(o, "H{}={}", i + 1, yes(*v));
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cue_gating::edge_mask;
    use crate::error_gated::Gate;
    use crate::generalization_bench::GraphShape;
    use crate::learning_strategies::{apply_update, LearningStrategy, RewardSignal};
    use crate::BooleanState;

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

    #[test]
    fn uncapped_gate_steps_match_cue_gating_bit_exactly() {
        let d = small();
        let fx = ns::fixture(None, &d, 50, 3).unwrap();
        let g = &fx.graph;
        let mask = edge_mask(g, &fx.cues.concat());
        for (s, pg) in [
            (Strategy::Supervised, Gate::Supervised),
            (Strategy::ErrorGated, Gate::ErrorGated),
            (Strategy::EeCue, Gate::ErrorGated),
        ] {
            let mut a = PlasticCircuit::new(g, 1).unwrap();
            let mut b = a.clone();
            let (mut ca, mut cb) = (PhaseCost::default(), PhaseCost::default());
            let mut l = WriteLedger::new(&Limit::default());
            for i in 0..K_MAX {
                let cue = if s.cued() { fx.cues[i].clone() } else { vec![] };
                for p in &fx.task.train[i] {
                    let v = &fx.task.valence[i];
                    step(&mut a, g, s, p, v, &cue, &mut l, &mut ca).unwrap();
                    crate::cue_gating::step(&mut b, g, pg, pg, p, v, &cue, &mask, &mut cb).unwrap();
                }
            }
            assert_eq!(a.weights(), b.weights(), "{}", s.name());
            assert_eq!(ca, cb, "{}", s.name());
            assert_eq!(l.writes, ca.weights_changed);
            assert_eq!(l.refused, 0);
        }
    }

    #[test]
    fn uncapped_reward_step_matches_per_target_reward_updates() {
        let d = small();
        let fx = ns::fixture(None, &d, 50, 5).unwrap();
        let g = &fx.graph;
        let mut a = PlasticCircuit::new(g, 1).unwrap();
        let mut b = a.clone();
        let mut ca = PhaseCost::default();
        let mut l = WriteLedger::new(&Limit::default());
        let mut changed = 0;
        for i in 0..K_MAX {
            for p in &fx.task.train[i] {
                let v = &fx.task.valence[i];
                step(&mut a, g, Strategy::Reward, p, v, &[], &mut l, &mut ca).unwrap();
                let (input, reached, teacher) = cc::encode_ctx(g, p, v, &[]).unwrap();
                let pred = b.predict(g, &input, THRESHOLD).unwrap();
                for &t in &reached {
                    if !pred.get(t).unwrap() {
                        continue;
                    }
                    let mut post = BooleanState::new(g.ids.len());
                    post.set(t, true).unwrap();
                    let sign = if teacher.get(t).unwrap() {
                        RewardSignal::Positive
                    } else {
                        RewardSignal::Negative
                    };
                    let r = apply_update(
                        &mut b,
                        g,
                        LearningStrategy::RewardModulated,
                        &input,
                        &post,
                        sign,
                    )
                    .unwrap();
                    changed += r.weights_changed;
                }
            }
        }
        assert_eq!(a.weights(), b.weights());
        assert_eq!(ca.weights_changed, changed);
        assert!(changed > 0);
    }

    #[test]
    fn limits_are_enforced_and_matched_error_gated_is_unchanged() {
        let d = small();
        let fx = ns::fixture(None, &d, 50, 2).unwrap();
        let spec = d.spec;
        let (free_row, eg, wf) = run_arm(&fx, &spec, 2, Strategy::ErrorGated, &free()).unwrap();
        let curves: Vec<Vec<u64>> = eg.iter().map(|l| l.cum.clone()).collect();
        assert!(curves.iter().any(|c| c.last() > Some(&0)));
        for b in [Budget::Paced(100), Budget::Total(100)] {
            let (same, l, w) =
                run_arm(&fx, &spec, 2, Strategy::ErrorGated, &limits(&curves, b)).unwrap();
            assert_eq!(w, wf);
            assert_eq!(same.ev, free_row.ev);
            assert!(l.iter().all(|x| x.refused == 0));
        }
        for s in STRATEGIES {
            for b in [Budget::Paced(100), Budget::Total(100), Budget::Paced(50)] {
                let lim = limits(&curves, b);
                let (r, l, _) = run_arm(&fx, &spec, 2, s, &lim).unwrap();
                assert!(r.topology_unchanged);
                for j in 0..K_MAX {
                    assert!(l[j].writes <= l[j].cap);
                    assert_eq!(l[j].writes, r.cost[j].weights_changed);
                    assert_eq!(l[j].cum.len(), curves[j].len());
                    for (e, &c) in l[j].cum.iter().enumerate() {
                        assert!(
                            c <= lim[j].at(e),
                            "{} {} task {j} ep {e}",
                            s.name(),
                            b.name()
                        );
                    }
                    assert_eq!(l[j].refused > 0, l[j].first_refusal > 0);
                }
            }
        }
        // A zero allowance never changes a weight.
        let zero: [Limit; K_MAX] = std::array::from_fn(|_| Limit(Some(vec![0; 16])));
        let (_, l, w) = run_arm(&fx, &spec, 2, Strategy::Supervised, &zero).unwrap();
        assert!(l.iter().all(|x| x.writes == 0 && x.refused > 0));
        assert!(w.iter().all(|&x| x == spec.init_weight));
        let c = vec![vec![1, 3, 3, 10]; K_MAX];
        assert_eq!(limits(&c, Budget::Paced(50))[0].0, Some(vec![0, 1, 1, 5]));
        assert_eq!(limits(&c, Budget::Total(50))[0].0, Some(vec![5; 4]));
        assert_eq!(limits(&c, Budget::Free)[0].0, None);
        assert_eq!(Budget::parse("paced100"), Ok(Budget::Paced(100)));
        assert_eq!(Budget::parse("total50"), Ok(Budget::Total(50)));
        assert_eq!(Budget::parse("free"), Ok(Budget::Free));
        assert!(Budget::parse("x").is_err());
    }

    #[test]
    fn grid_is_deterministic_csv_complete_and_readings_gate() {
        let d = small();
        let flips = [25, 50, 75, 100];
        let run = || run_grid_design(None, &d, &flips, &[1, 2, 3], &BUDGETS).unwrap();
        let a = run();
        let b = run();
        assert_eq!(a, b);
        assert_eq!(a.len(), flips.len() * 3 * STRATEGIES.len() * BUDGETS.len());
        assert!(a.iter().all(|r| r.inner.topology_unchanged));
        assert!(a
            .iter()
            .filter(|r| r.budget == Budget::Free)
            .all(|r| r.same_as_free));
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        let cols = csv_header().split(',').count();
        assert!(csv.lines().all(|l| l.split(',').count() == cols));
        let id = compare(&a, 50, K_MAX, Slice::All, EG, EG);
        assert_eq!(id.acc.ties, 3);
        assert!(!id.passes());
        let (g, h) = readings(&a, &flips);
        assert!(g.0);
        if !(g.0 && g.1 && g.2) {
            assert_eq!(h, [None; N_HYP]);
        } else {
            assert!(h.iter().all(Option::is_some));
        }
        let rep = report(&a, &flips, &BUDGETS);
        assert!(rep.contains("readings: G1_cap_integrity=true"));
        assert_eq!(rep, report(&b, &flips, &BUDGETS));
        assert!(run_grid_design(None, &d, &[50], &[], &BUDGETS).is_err());
        assert!(run_grid_design(None, &d, &[50], &[1], &[Budget::Paced(0)]).is_err());
        assert!(run_grid_design(None, &d, &[50], &[1], &[Budget::Total(101)]).is_err());
    }

    #[test]
    fn readings_on_empty_grid_are_not_evaluable() {
        let (g, h) = readings(&[], &FLIPS);
        assert!(!g.0 && !g.1);
        assert_eq!(h, [None; N_HYP]);
    }
}

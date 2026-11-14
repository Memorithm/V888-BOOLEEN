//! B888-4 cycle 16: **cue edges that start silent** (initial weight 0 on
//! cue-line edges) for the cued error-gated arm under an equal weight-write
//! budget.
//!
//! Cycle 15 (#127) showed that the funding order of the offered steps does
//! not restore the cued arm's learning at `error-gated`'s write schedule.
//! Its post-hoc reading (untested there): with threshold 1 and initial
//! weight 1, an untouched cue edge makes every cued target fire, so the cue
//! carries a fixed write cost (uncapped ≈ 63 writes per task at flip 50)
//! before it stops causing errors, more than `error-gated`'s whole ≈ 51
//! writes per task. This module tests that reading directly: the update
//! rule, the episodes, the budgets and the allocation orders are the
//! unchanged `write_allocation` code paths; only the **initial weight of the
//! cue-line edges** changes ([`Arm::cue_zero`]). Every non-cue edge starts at
//! `spec.init_weight` as before. With a zero cue edge, a cue adds no drive
//! until the error gate raises it on a false negative in its own context.
//!
//! Arms ([`ARMS`]): `supervised`, `error-gated`, `reward` (uncued, the #126 /
//! #127 code paths; cue lines are never active for them, so the cue init is
//! irrelevant and not applied), `E/E-cue` (#127 source-major, init 1 — the
//! reference cued arm, bit-identical to #127, tested), `E/E-cue0`
//! (source-major, cue edges at 0, primary) and `E/E-cue0-pattern-first`
//! (secondary).
//!
//! Engine state for checkpoint/resume = circuit weights
//! (`PlasticCircuit::save`, which holds the zeroed cue edges) +
//! `write_allocation::Checkpoint` (stats, ledger, cost, task, next episode);
//! resuming mid-task is bit-exact for every arm (tested).
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph; "cue" and
//! "silent" name parts of an update rule and of its initial state, not
//! claims about synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/CUE_ZERO_INIT.md`
//! (committed before any learning run on the hypothesis seeds).
use crate::connectome::Connectome;
use crate::cue_reach::{self as cr, Family, Part};
use crate::error_gated::{paired, PhaseCost, EFFECT};
use crate::generalization_bench::GenSpec;
use crate::novel_split::{self as ns, Design, Eval, Fixture, Slice, SLICES};
use crate::plasticity::PlasticCircuit;
use crate::task_sequence::{self as ts, K_MAX};
use crate::write_allocation::{self as wa, Alloc, Checkpoint, Stats, Strategy};
use crate::write_budget::{self as wb, Budget, Limit, WriteLedger};
use std::io::{self, ErrorKind};

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Pre-registered flips (50 primary, 100 high conflict, 25 secondary).
pub const FLIPS: [u32; 3] = [25, 50, 100];
pub const PRIMARY_FLIP: u32 = 50;
pub const HIGH_FLIP: u32 = 100;
/// Pre-registered seeds: 241..=280, never used by an earlier B888-4 cycle.
pub const SEEDS: std::ops::RangeInclusive<u64> = 241..=280;
/// Pre-registered budgets. Primary: `Paced(100)`.
pub const BUDGETS: [Budget; 2] = [Budget::Free, Budget::Paced(100)];
pub const PRIMARY_BUDGET: Budget = Budget::Paced(100);
/// Cue level of the cued arms (full reach on the #124 pool).
pub const CUE_LEVEL: usize = wb::CUE_LEVEL;
/// Fixed-pool design of #124/#126/#127.
pub const DESIGN: Design = wb::DESIGN;
/// Pilot seeds (writes only, no accuracy).
pub const PILOT_SEEDS: std::ops::RangeInclusive<u64> = 1..=5;

/// One arm: a `write_allocation` update rule plus the cue-edge init.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Arm {
    pub rule: Strategy,
    /// Cue-line edges start at weight 0 (only meaningful for cued rules).
    pub cue_zero: bool,
}

const fn arm(rule: Strategy, cue_zero: bool) -> Arm {
    Arm { rule, cue_zero }
}

pub const SUP: Arm = arm(Strategy::Supervised, false);
pub const ERR: Arm = arm(Strategy::ErrorGated, false);
pub const REW: Arm = arm(Strategy::Reward, false);
/// #127 source-major `E/E-cue`, cue edges at the common init (reference).
pub const CUE1: Arm = arm(Strategy::EeCue(Alloc::SourceMajor), false);
/// Source-major `E/E-cue` with cue edges at 0 (primary).
pub const CUE0: Arm = arm(Strategy::EeCue(Alloc::SourceMajor), true);
/// Pattern-first `E/E-cue` with cue edges at 0 (secondary).
pub const CUE0_PF: Arm = arm(Strategy::EeCue(Alloc::PatternFirst), true);

/// The pre-registered arms, in CSV order.
pub const ARMS: [Arm; 6] = [SUP, ERR, REW, CUE1, CUE0, CUE0_PF];

impl Arm {
    #[must_use]
    pub fn name(self) -> &'static str {
        match (self.rule, self.cue_zero) {
            (Strategy::EeCue(Alloc::SourceMajor), true) => "E/E-cue0",
            (Strategy::EeCue(Alloc::PatternFirst), true) => "E/E-cue0-pattern-first",
            (Strategy::EeCue(Alloc::CueFirst), true) => "E/E-cue0-cue-first",
            (r, _) => r.name(),
        }
    }
    /// Parse a pre-registered arm name.
    pub fn parse(s: &str) -> io::Result<Self> {
        ARMS.iter()
            .copied()
            .find(|x| x.name() == s)
            .ok_or_else(|| bad("unknown arm"))
    }
    #[must_use]
    pub fn cued(self) -> bool {
        self.rule.cued()
    }
    /// Is the cue-edge zero init actually applied? (cued rules only.)
    #[must_use]
    pub fn zero_applied(self) -> bool {
        self.cue_zero && self.cued()
    }
    fn row_arm(self) -> ts::Arm {
        match self.rule {
            Strategy::Supervised | Strategy::Reward => cr::SUPERVISED,
            Strategy::ErrorGated | Strategy::Tagged => cr::ERROR_GATED,
            Strategy::EeCue(_) => cr::EE,
        }
    }
}

/// Per edge: is its source a cue line of any task?
pub fn cue_edges(fx: &Fixture) -> io::Result<Vec<bool>> {
    let g = &fx.graph;
    let n = g.ids.len();
    let mut line = vec![false; n];
    for c in &fx.cues {
        for &s in c {
            *line
                .get_mut(s)
                .ok_or_else(|| bad("cue line out of range"))? = true;
        }
    }
    let mut out = vec![false; g.edges()];
    for (s, &on) in line.iter().enumerate() {
        if on {
            for e in g.offsets[s]..g.offsets[s + 1] {
                out[e as usize] = true;
            }
        }
    }
    Ok(out)
}

/// Initial circuit of `arm`: every edge at `spec.init_weight`, cue-line
/// edges at 0 if the zero init applies.
pub fn initial_circuit(fx: &Fixture, spec: &GenSpec, arm: Arm) -> io::Result<PlasticCircuit> {
    let c = PlasticCircuit::new(&fx.graph, spec.init_weight)?;
    if !arm.zero_applied() {
        return Ok(c);
    }
    let mut w = c.weights().to_vec();
    for (x, on) in w.iter_mut().zip(cue_edges(fx)?) {
        if on {
            *x = 0;
        }
    }
    PlasticCircuit::from_weights(&fx.graph, w)
}

/// Does `w` hold the documented initial weights of `arm`?
pub fn init_matches(fx: &Fixture, spec: &GenSpec, arm: Arm, w: &[u8]) -> io::Result<bool> {
    let cue = cue_edges(fx)?;
    Ok(w.len() == cue.len()
        && w.iter().zip(&cue).all(|(&x, &on)| {
            x == if on && arm.zero_applied() {
                0
            } else {
                spec.init_weight
            }
        }))
}

/// Per-task instrumentation beyond `write_allocation::Stats`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CueState {
    /// Cue-line edges with a non-zero weight at the end of the task.
    pub cue_nonzero_end: u64,
    /// Sum of cue-line edge weights at the end of the task.
    pub cue_mass_end: u64,
}

fn cue_state(w: &[u8], cue: &[bool]) -> CueState {
    let mut s = CueState::default();
    for (&x, &on) in w.iter().zip(cue) {
        if on && x > 0 {
            s.cue_nonzero_end += 1;
            s.cue_mass_end += u64::from(x);
        }
    }
    s
}

/// Output of [`run_arm`]: row, per-task ledgers, stats, cue states, final
/// weights.
pub type ArmRun = (
    cr::Row,
    [WriteLedger; K_MAX],
    [Stats; K_MAX],
    [CueState; K_MAX],
    Vec<u8>,
);

/// Train one task fully from a fresh per-task checkpoint.
fn train_task(
    circuit: &mut PlasticCircuit,
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    arm: Arm,
    i: usize,
    limit: &Limit,
) -> io::Result<Checkpoint> {
    let episodes = ts::task_order(spec, seed, i).len();
    let mut cp = Checkpoint::start(i, fx.graph.edges(), limit);
    wa::train_range(circuit, fx, spec, seed, arm.rule, limit, &mut cp, episodes)?;
    Ok(cp)
}

/// Run one arm over the K_MAX tasks with per-task write limits.
pub fn run_arm(
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    arm: Arm,
    limits: &[Limit; K_MAX],
) -> io::Result<ArmRun> {
    let graph = &fx.graph;
    if fx.task.train.len() != K_MAX || fx.cues.len() != K_MAX || fx.splits.len() != K_MAX {
        return Err(bad("need K_MAX tasks, cues and splits"));
    }
    if arm.rule == Strategy::Tagged {
        return Err(bad("E-tag is not an arm of this cycle"));
    }
    graph.validate()?;
    let original = graph.clone();
    let n = graph.ids.len();
    let cue = cue_edges(fx)?;
    let mut circuit = initial_circuit(fx, spec, arm)?;
    let fingerprint = circuit.topology_fingerprint();
    let mut ev = [[Eval::default(); K_MAX]; K_MAX];
    let mut tgt = vec![crate::generalization_bench::Confusion::default(); K_MAX * K_MAX * n];
    let mut cost = [PhaseCost::default(); K_MAX];
    let mut ledger: [WriteLedger; K_MAX] = Default::default();
    let mut stats = [Stats::default(); K_MAX];
    let mut cues = [CueState::default(); K_MAX];
    for i in 0..K_MAX {
        let cp = train_task(&mut circuit, fx, spec, seed, arm, i, &limits[i])?;
        ledger[i] = cp.ledger;
        stats[i] = cp.stats;
        cost[i] = cp.cost;
        cues[i] = cue_state(circuit.weights(), &cue);
        for j in 0..K_MAX {
            let c: &[usize] = if arm.cued() { &fx.cues[j] } else { &[] };
            let (e, per) = cr::evaluate(
                &circuit,
                graph,
                &fx.task.test,
                &fx.task.valence[j],
                c,
                &fx.splits[j],
            )?;
            ev[i][j] = e;
            tgt[(i * K_MAX + j) * n..(i * K_MAX + j + 1) * n].copy_from_slice(&per);
        }
    }
    let mut cue_lines = [0; K_MAX];
    if arm.cued() {
        for (l, c) in cue_lines.iter_mut().zip(&fx.cues) {
            *l = c.len();
        }
    }
    let row = cr::Row {
        family: Family::Fixed,
        level: if arm.cued() { CUE_LEVEL } else { 0 },
        arm: arm.row_arm(),
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
    Ok((row, ledger, stats, cues, circuit.weights().to_vec()))
}

/// One measurement: (flip, seed, arm, budget) on a K_MAX-task run.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub arm: Arm,
    pub budget: Budget,
    pub inner: cr::Row,
    pub ledger: [WriteLedger; K_MAX],
    pub stats: [Stats; K_MAX],
    pub cue: [CueState; K_MAX],
    /// Final weights equal to the uncapped run of the same arm.
    pub same_as_free: bool,
    /// Uncapped final weights equal across allocations with the same init
    /// (`E/E-cue0-pattern-first` vs `E/E-cue0`; true for other arms).
    pub free_alloc_invariant: bool,
    /// Initial weights are the documented init of the arm.
    pub init_ok: bool,
    /// Digest of the final weights.
    pub digest: u64,
}

/// Full grid: per (flip, seed) uncapped `error-gated` first (its write
/// curves set the caps), then every (arm, budget).
pub fn run_grid_design(
    graph: Option<&Connectome>,
    design: &Design,
    flips: &[u32],
    seeds: &[u64],
    budgets: &[Budget],
    arms: &[Arm],
) -> io::Result<Vec<Row>> {
    if seeds.is_empty() || flips.is_empty() || budgets.is_empty() || arms.is_empty() {
        return Err(bad("need at least one seed, flip, budget and arm"));
    }
    if budgets.iter().any(|b| match b {
        Budget::Free => false,
        Budget::Paced(p) | Budget::Total(p) => *p == 0 || *p > 100,
    }) {
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
            let (_, eg, _, _, _) = run_arm(&fx, &spec, seed, ERR, &wb::free())?;
            let curves: Vec<Vec<u64>> = eg.iter().map(|l| l.cum.clone()).collect();
            let mut free_by_init: Vec<(bool, Vec<u8>)> = Vec::new();
            for &a in arms {
                let init = initial_circuit(&fx, &spec, a)?;
                let init_ok = init_matches(&fx, &spec, a, init.weights())?;
                let free = run_arm(&fx, &spec, seed, a, &wb::free())?;
                let invariant = if a.cued() {
                    let src = run_arm(
                        &fx,
                        &spec,
                        seed,
                        arm(Strategy::EeCue(Alloc::SourceMajor), a.cue_zero),
                        &wb::free(),
                    )?;
                    free_by_init.push((a.cue_zero, src.4.clone()));
                    src.4 == free.4
                } else {
                    true
                };
                for &b in budgets {
                    let (inner, ledger, stats, cue, w) = if b == Budget::Free {
                        free.clone()
                    } else {
                        run_arm(&fx, &spec, seed, a, &wb::limits(&curves, b))?
                    };
                    rows.push(Row {
                        arm: a,
                        budget: b,
                        inner,
                        ledger,
                        stats,
                        cue,
                        same_as_free: w == free.4,
                        free_alloc_invariant: invariant,
                        init_ok,
                        digest: wa::digest(&w),
                    });
                }
            }
        }
    }
    Ok(rows)
}

/// Pre-registered grid on the fixed-pool design.
pub fn run_grid(
    graph: Option<&Connectome>,
    flips: &[u32],
    seeds: &[u64],
    budgets: &[Budget],
) -> io::Result<Vec<Row>> {
    run_grid_design(graph, &DESIGN, flips, seeds, budgets, &ARMS)
}

/// An (arm, budget) setting.
pub type Setting = (Arm, Budget);
/// The base of every gain: uncapped `error-gated`.
pub const EG: Setting = (ERR, Budget::Free);
/// Primary arm: source-major `E/E-cue0` at `paced100`.
pub const Z1: Setting = (CUE0, PRIMARY_BUDGET);
/// #127 `E/E-cue` (init 1) at `paced100`.
pub const C1: Setting = (CUE1, PRIMARY_BUDGET);
pub const ZP1: Setting = (CUE0_PF, PRIMARY_BUDGET);
pub const S1: Setting = (SUP, PRIMARY_BUDGET);
pub const R1: Setting = (REW, PRIMARY_BUDGET);
pub const Z_FREE: Setting = (CUE0, Budget::Free);
pub const C_FREE: Setting = (CUE1, Budget::Free);

fn find(rows: &[Row], flip: u32, seed: u64, x: Setting) -> Option<&Row> {
    rows.iter()
        .find(|r| r.inner.flip == flip && r.inner.seed == seed && r.arm == x.0 && r.budget == x.1)
}

fn rows_of(rows: &[Row], flip: u32, x: Setting) -> impl Iterator<Item = &Row> {
    rows.iter()
        .filter(move |r| r.inner.flip == flip && r.arm == x.0 && r.budget == x.1)
}

fn pairs_of(rows: &[Row], flip: u32, x: Setting, base: Setting) -> Vec<(&Row, &Row)> {
    rows_of(rows, flip, x)
        .filter_map(|r| find(rows, flip, r.inner.seed, base).map(|c| (r, c)))
        .collect()
}

/// x vs base paired by seed at k (same oriented fields as `cue_reach`).
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
    let pairs = pairs_of(rows, flip, x, base);
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

/// Effective writes over the K_MAX tasks.
#[must_use]
pub fn total_writes(r: &Row) -> u64 {
    r.ledger.iter().map(|l| l.writes).sum()
}

/// Cue-edge writes over the K_MAX tasks.
#[must_use]
pub fn cue_writes(r: &Row) -> u64 {
    r.stats.iter().map(|s| s.cue_writes).sum()
}

fn cue_share(r: &Row) -> f64 {
    let w = total_writes(r);
    if w == 0 {
        0.0
    } else {
        cue_writes(r) as f64 / w as f64
    }
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

/// Fraction of seeds where x and y end with different weights.
#[must_use]
pub fn differ_frac(rows: &[Row], flip: u32, x: Setting, y: Setting) -> f64 {
    let p = pairs_of(rows, flip, x, y);
    if p.is_empty() {
        0.0
    } else {
        p.iter().filter(|(a, b)| a.digest != b.digest).count() as f64 / p.len() as f64
    }
}

/// H2 statistic: (seeds where uncapped `E/E-cue0` makes fewer cue writes
/// over the four tasks than uncapped `error-gated` makes writes in total,
/// number of paired seeds).
#[must_use]
pub fn cue_cost_below_budget(rows: &[Row], flip: u32) -> (usize, usize) {
    let p = pairs_of(rows, flip, Z_FREE, EG);
    (
        p.iter()
            .filter(|(z, e)| cue_writes(z) < total_writes(e))
            .count(),
        p.len(),
    )
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

/// Design gates: (G1 integrity, G2 zero init changes the capped run, G3 CN).
/// G1: every row keeps its topology and its documented initial weights;
/// writes ≤ allowance; `error-gated` at 100 % budgets is bit-identical to
/// its uncapped run with no refusal; the two zero-init allocations are
/// bit-identical uncapped.
#[must_use]
pub fn gates(rows: &[Row]) -> (bool, bool, bool) {
    let g1 = !rows.is_empty()
        && rows.iter().all(|r| {
            r.inner.topology_unchanged
                && r.init_ok
                && r.free_alloc_invariant
                && r.ledger.iter().all(|l| l.writes <= l.cap)
                && (r.arm != ERR
                    || matches!(r.budget, Budget::Paced(p) | Budget::Total(p) if p < 100)
                    || (r.same_as_free && r.ledger.iter().all(|l| l.refused == 0)))
        });
    let g2 = differ_frac(rows, PRIMARY_FLIP, Z1, C1) >= 0.5;
    let g3 = median_cn(rows, PRIMARY_FLIP) >= ns::MIN_NOVEL_PAIRS;
    (g1, g2, g3)
}

/// Number of pre-registered hypotheses.
pub const N_HYP: usize = 7;

/// Pre-registered readings: (gates, [H1..H7]); `None` if a gate fails or
/// the hypothesis' flip is missing.
#[must_use]
pub fn readings(rows: &[Row], flips: &[u32]) -> ((bool, bool, bool), [Option<bool>; N_HYP]) {
    let g = gates(rows);
    let mut h = [None; N_HYP];
    if !(g.0 && g.1 && g.2) {
        return (g, h);
    }
    let (all, cn) = (Slice::All, Slice::Cn);
    let f = PRIMARY_FLIP;
    if flips.contains(&f) {
        h[0] = Some(compare(rows, f, K_MAX, all, Z1, EG).passes());
        let (below, n) = cue_cost_below_budget(rows, f);
        h[1] = Some(n > 0 && below * 5 >= n * 4);
        h[2] = Some(compare(rows, f, K_MAX, all, Z1, C1).learn.strong());
        h[3] = Some(compare(rows, f, K_MAX, cn, Z1, EG).passes());
        h[4] = Some(compare(rows, f, K_MAX, cn, Z1, S1).passes());
        h[6] = Some(compare(rows, f, K_MAX, all, Z_FREE, EG).passes());
    }
    if flips.contains(&HIGH_FLIP) {
        let c = compare(rows, HIGH_FLIP, K_MAX, cn, Z1, EG);
        h[5] = Some(c.forget.strong() && c.learn.mean >= -EFFECT);
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
        "arm",
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
    for name in [
        "cap",
        "writes",
        "refused",
        "first_refusal",
        "attempts",
        "cue_writes",
        "cue_refused",
        "cue_nonzero_end",
        "cue_mass_end",
    ] {
        for j in 1..=K_MAX {
            h.push(format!("{name}_{j}"));
        }
    }
    for name in [
        "saturated_end",
        "same_as_free",
        "free_alloc_invariant",
        "init_ok",
        "weights_digest",
        "topology_unchanged",
    ] {
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
            "b888-4-cue-zero-init-v1".to_string(),
            graph_label.to_string(),
            x.flip.to_string(),
            r.arm.name().to_string(),
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
        f.extend(r.stats.iter().map(|t| t.cue_writes.to_string()));
        f.extend(r.stats.iter().map(|t| t.cue_refused.to_string()));
        f.extend(r.cue.iter().map(|t| t.cue_nonzero_end.to_string()));
        f.extend(r.cue.iter().map(|t| t.cue_mass_end.to_string()));
        f.push(x.saturated_end.to_string());
        f.push(r.same_as_free.to_string());
        f.push(r.free_alloc_invariant.to_string());
        f.push(r.init_ok.to_string());
        f.push(format!("{:016x}", r.digest));
        f.push(x.topology_unchanged.to_string());
        out.push_str(&f.join(","));
        out.push('\n');
    }
    out
}

fn yes(b: Option<bool>) -> &'static str {
    match b {
        Some(true) => "pass",
        Some(false) => "fail",
        None => "not_evaluable",
    }
}

fn fmt_p(d: &crate::error_gated::Paired) -> String {
    format!(
        "{:+.3}±{:.3}[{:+.3},{:+.3}]{}/{}/{}",
        d.mean, d.sd, d.ci_lo, d.ci_hi, d.better, d.ties, d.worse
    )
}

/// Human-readable report (deterministic; printed by the CLI).
#[must_use]
pub fn report(rows: &[Row], flips: &[u32], budgets: &[Budget]) -> String {
    use std::fmt::Write;
    let mut o = String::new();
    let (a, c, ic) = (
        Part::Class(Slice::All),
        Part::Class(Slice::Cn),
        Part::Class(Slice::Icn),
    );
    let _ = writeln!(
        o,
        "setting_summary(k=4): flip,arm,budget,acc_all,acc_cn,acc_icn,acc_seen,acc_ue,learn_all,learn_cn,fgt_all,fgt_cn,last_all,last_cn,writes_total,cue_writes_total,cue_write_share,refused_total,cue_refused_total,cue_nonzero_end_task4,cue_mass_end_task4,saturated_end"
    );
    for &flip in flips {
        for s in ARMS {
            for &b in budgets {
                let x = (s, b);
                if rows_of(rows, flip, x).next().is_none() {
                    continue;
                }
                let m = |f: &dyn Fn(&Row) -> f64| setting_mean(rows, flip, x, f);
                let _ = writeln!(
                    o,
                    "{flip},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{:.1},{:.3},{:.1},{:.1},{:.1},{:.1},{:.1}",
                    s.name(),
                    b.name(),
                    m(&|r| r.inner.acc(a, K_MAX)),
                    m(&|r| r.inner.acc(c, K_MAX)),
                    m(&|r| r.inner.acc(ic, K_MAX)),
                    m(&|r| r.inner.acc(Part::Class(Slice::Seen), K_MAX)),
                    m(&|r| r.inner.acc(Part::Class(Slice::Ue), K_MAX)),
                    m(&|r| r.inner.learn(a, K_MAX)),
                    m(&|r| r.inner.learn(c, K_MAX)),
                    m(&|r| r.inner.forgetting(a, K_MAX)),
                    m(&|r| r.inner.forgetting(c, K_MAX)),
                    m(&|r| r.inner.last(a, K_MAX)),
                    m(&|r| r.inner.last(c, K_MAX)),
                    m(&|r| total_writes(r) as f64),
                    m(&|r| cue_writes(r) as f64),
                    m(&cue_share),
                    m(&|r| r.ledger.iter().map(|l| l.refused).sum::<u64>() as f64),
                    m(&|r| r.stats.iter().map(|t| t.cue_refused).sum::<u64>() as f64),
                    m(&|r| r.cue[K_MAX - 1].cue_nonzero_end as f64),
                    m(&|r| r.cue[K_MAX - 1].cue_mass_end as f64),
                    m(&|r| r.inner.saturated_end as f64),
                );
            }
        }
    }
    let _ = writeln!(
        o,
        "comparison(k): flip,k,slice,x,base,d_acc,d_learn,d_last,d_fgt(+=forgets_less),passes_sequence_rule"
    );
    let base_pairs = [
        (Z1, EG),
        (C1, EG),
        (ZP1, EG),
        (Z1, C1),
        (ZP1, Z1),
        (Z1, S1),
        (C1, S1),
        (EG, S1),
        (R1, S1),
        (Z_FREE, EG),
        (C_FREE, EG),
        (Z_FREE, C_FREE),
    ];
    for &flip in flips {
        for k in 2..=K_MAX {
            for s in [Slice::All, Slice::Seen, Slice::Cn, Slice::Icn, Slice::Ue] {
                for &(x, base) in &base_pairs {
                    let cmp = compare(rows, flip, k, s, x, base);
                    if cmp.acc.n == 0 {
                        continue;
                    }
                    let _ = writeln!(
                        o,
                        "{flip},{k},{},{}@{},{}@{},{},{},{},{},{}",
                        s.name(),
                        x.0.name(),
                        x.1.name(),
                        base.0.name(),
                        base.1.name(),
                        fmt_p(&cmp.acc),
                        fmt_p(&cmp.learn),
                        fmt_p(&cmp.last),
                        fmt_p(&cmp.forget),
                        cmp.passes()
                    );
                }
            }
        }
        let (below, n) = cue_cost_below_budget(rows, flip);
        let _ = writeln!(
            o,
            "cue_cost: flip={flip} seeds(cue_writes(E/E-cue0@free) < writes(error-gated@free))={below}/{n} mean_cue_writes(E/E-cue0@free)={:.1} mean_cue_writes(E/E-cue@free)={:.1} mean_writes(error-gated@free)={:.1} differ_frac(E/E-cue0 vs E/E-cue @paced100)={:.3} differ_frac(E/E-cue0-pattern-first vs E/E-cue0 @paced100)={:.3} median_cn={}",
            setting_mean(rows, flip, Z_FREE, &|r| cue_writes(r) as f64),
            setting_mean(rows, flip, C_FREE, &|r| cue_writes(r) as f64),
            setting_mean(rows, flip, EG, &|r| total_writes(r) as f64),
            differ_frac(rows, flip, Z1, C1),
            differ_frac(rows, flip, ZP1, Z1),
            median_cn(rows, flip)
        );
    }
    let (g, h) = readings(rows, flips);
    let _ = writeln!(
        o,
        "readings: G1_integrity={} G2_zero_init_changes_run={} G3_cn_pairs={} H1={} H2={} H3={} H4={} H5={} H6={} H7={}",
        g.0,
        g.1,
        g.2,
        yes(h[0]),
        yes(h[1]),
        yes(h[2]),
        yes(h[3]),
        yes(h[4]),
        yes(h[5]),
        yes(h[6])
    );
    o
}

/// One pilot census line: (seed, arm, budget, writes, cue writes, refused,
/// cue refused, cue edges non-zero after task 4).
pub type CensusLine = (u64, Arm, Budget, u64, u64, u64, u64, u64);

/// Pilot census: writes, cue writes, refusals and cue state only; no
/// accuracy is computed.
pub fn census(
    graph: Option<&Connectome>,
    flip: u32,
    seeds: &[u64],
    budgets: &[Budget],
) -> io::Result<Vec<CensusLine>> {
    let spec = GenSpec {
        flip_pct: flip,
        ..DESIGN.spec
    };
    let mut out = Vec::new();
    for &seed in seeds {
        let fx = ns::fixture(graph, &DESIGN, flip, seed)?;
        let cue = cue_edges(&fx)?;
        let mut curves: Vec<Vec<u64>> = Vec::new();
        {
            let mut c = initial_circuit(&fx, &spec, ERR)?;
            for i in 0..K_MAX {
                let cp = train_task(&mut c, &fx, &spec, seed, ERR, i, &Limit::default())?;
                curves.push(cp.ledger.cum);
            }
        }
        for a in ARMS {
            for &b in budgets {
                let lim = wb::limits(&curves, b);
                let mut c = initial_circuit(&fx, &spec, a)?;
                let mut t = [0_u64; 4];
                for (i, l) in lim.iter().enumerate() {
                    let cp = train_task(&mut c, &fx, &spec, seed, a, i, l)?;
                    t[0] += cp.ledger.writes;
                    t[1] += cp.stats.cue_writes;
                    t[2] += cp.ledger.refused;
                    t[3] += cp.stats.cue_refused;
                }
                let nz = cue_state(c.weights(), &cue).cue_nonzero_end;
                out.push((seed, a, b, t[0], t[1], t[2], t[3], nz));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error_gated::drive;
    use crate::generalization_bench::GraphShape;

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
        std::env::temp_dir().join(format!("v888-cue-init-{tag}-{}", std::process::id()))
    }

    fn curves_of(fx: &Fixture, d: &Design, seed: u64) -> Vec<Vec<u64>> {
        let (_, eg, _, _, _) = run_arm(fx, &d.spec, seed, ERR, &wb::free()).unwrap();
        eg.iter().map(|l| l.cum.clone()).collect()
    }

    #[test]
    fn reference_arms_match_write_allocation_bit_exactly() {
        let d = small();
        for seed in [1, 2, 3] {
            let fx = ns::fixture(None, &d, 50, seed).unwrap();
            let curves = curves_of(&fx, &d, seed);
            for b in [Budget::Free, Budget::Paced(100), Budget::Paced(50)] {
                let lim = wb::limits(&curves, b);
                for a in [SUP, ERR, REW, CUE1] {
                    let (r, l, s, _, w) = run_arm(&fx, &d.spec, seed, a, &lim).unwrap();
                    let (r0, l0, s0, w0) = wa::run_arm(&fx, &d.spec, seed, a.rule, &lim).unwrap();
                    assert_eq!(w, w0, "{} {}", a.name(), b.name());
                    assert_eq!(r, r0, "{} {}", a.name(), b.name());
                    assert_eq!(l, l0, "{} {}", a.name(), b.name());
                    assert_eq!(s, s0, "{} {}", a.name(), b.name());
                }
            }
        }
    }

    #[test]
    fn zero_init_touches_only_cue_edges_and_silences_the_cue() {
        let d = small();
        for seed in [1, 2, 3] {
            let fx = ns::fixture(None, &d, 50, seed).unwrap();
            let cue = cue_edges(&fx).unwrap();
            assert!(cue.iter().any(|&b| b) && cue.iter().any(|&b| !b));
            let one = initial_circuit(&fx, &d.spec, CUE1).unwrap();
            let zero = initial_circuit(&fx, &d.spec, CUE0).unwrap();
            assert!(init_matches(&fx, &d.spec, CUE1, one.weights()).unwrap());
            assert!(init_matches(&fx, &d.spec, CUE0, zero.weights()).unwrap());
            assert!(!init_matches(&fx, &d.spec, CUE0, one.weights()).unwrap());
            for (e, &on) in cue.iter().enumerate() {
                let (a, b) = (one.weights()[e], zero.weights()[e]);
                assert_eq!(a, d.spec.init_weight);
                assert_eq!(b, if on { 0 } else { a });
            }
            assert_eq!(zero.topology_fingerprint(), one.topology_fingerprint());
            // Uncued rules ignore the flag (cue lines are never active).
            let flagged = arm(Strategy::ErrorGated, true);
            assert!(!flagged.zero_applied());
            assert_eq!(
                initial_circuit(&fx, &d.spec, flagged).unwrap().weights(),
                one.weights()
            );
            // A silent cue adds no drive: with the cue on, every target's
            // drive equals its drive without the cue.
            for (j, c) in fx.cues.iter().enumerate() {
                for p in &fx.task.train[j] {
                    let v = &fx.task.valence[j];
                    let (with, _, _) = crate::context_cue::encode_ctx(&fx.graph, p, v, c).unwrap();
                    let (without, _, _) =
                        crate::context_cue::encode_ctx(&fx.graph, p, v, &[]).unwrap();
                    let dw = drive(&zero, &fx.graph, &with).unwrap();
                    let d0 = drive(&zero, &fx.graph, &without).unwrap();
                    assert_eq!(dw, d0);
                    let d1 = drive(&one, &fx.graph, &with).unwrap();
                    assert!(d1.iter().zip(&d0).all(|(a, b)| a >= b));
                }
            }
        }
    }

    #[test]
    fn zero_init_allocations_identical_uncapped_and_cue_writes_drop() {
        let d = small();
        let (mut fewer, mut differ) = (0, 0);
        for seed in 1..=4 {
            let fx = ns::fixture(None, &d, 50, seed).unwrap();
            let z = run_arm(&fx, &d.spec, seed, CUE0, &wb::free()).unwrap();
            let zp = run_arm(&fx, &d.spec, seed, CUE0_PF, &wb::free()).unwrap();
            assert_eq!(z.4, zp.4);
            assert_eq!(z.0, zp.0);
            assert_eq!(z.1, zp.1);
            let c = run_arm(&fx, &d.spec, seed, CUE1, &wb::free()).unwrap();
            let zc: u64 = z.2.iter().map(|s| s.cue_writes).sum();
            let cc: u64 = c.2.iter().map(|s| s.cue_writes).sum();
            if zc < cc {
                fewer += 1;
            }
            let lim = wb::limits(&curves_of(&fx, &d, seed), Budget::Paced(100));
            let zl = run_arm(&fx, &d.spec, seed, CUE0, &lim).unwrap();
            let cl = run_arm(&fx, &d.spec, seed, CUE1, &lim).unwrap();
            for x in [&zl, &cl] {
                for j in 0..K_MAX {
                    assert!(x.1[j].writes <= x.1[j].cap);
                    assert!(x.2[j].cue_writes <= x.1[j].writes);
                }
            }
            if zl.4 != cl.4 {
                differ += 1;
            }
            // Cue state is consistent with the final weights.
            let cue = cue_edges(&fx).unwrap();
            assert_eq!(z.3[K_MAX - 1], cue_state(&z.4, &cue));
        }
        assert!(fewer > 0);
        assert!(differ > 0);
    }

    #[test]
    fn resumes_bit_exactly_from_checkpoints() {
        let d = small();
        let seed = 4;
        let fx = ns::fixture(None, &d, 50, seed).unwrap();
        let g = &fx.graph;
        let curves = curves_of(&fx, &d, seed);
        let lim = wb::limits(&curves, Budget::Paced(100));
        for a in ARMS {
            let (_, lf, tf, _, wf) = run_arm(&fx, &d.spec, seed, a, &lim).unwrap();
            let mut c = initial_circuit(&fx, &d.spec, a).unwrap();
            for i in 0..K_MAX {
                let n = ts::task_order(&d.spec, seed, i).len();
                let mut cp = Checkpoint::start(i, g.edges(), &lim[i]);
                for end in [n / 3, n / 2, n] {
                    wa::train_range(&mut c, &fx, &d.spec, seed, a.rule, &lim[i], &mut cp, end)
                        .unwrap();
                    let (pw, pc) = (tmp("w"), tmp("c"));
                    c.save(&pw).unwrap();
                    cp.save(&pc).unwrap();
                    c = PlasticCircuit::load(&pw, g).unwrap();
                    cp = Checkpoint::load(&pc, g.edges()).unwrap();
                    std::fs::remove_file(&pw).unwrap();
                    std::fs::remove_file(&pc).unwrap();
                }
                assert_eq!(cp.ledger, lf[i], "{} task {i}", a.name());
                assert_eq!(cp.stats, tf[i], "{} task {i}", a.name());
            }
            assert_eq!(c.weights(), &wf[..], "{}", a.name());
        }
    }

    #[test]
    fn grid_is_deterministic_csv_complete_and_readings_gate() {
        let d = small();
        let flips = [50, 100];
        let run = || run_grid_design(None, &d, &flips, &[1, 2, 3], &BUDGETS, &ARMS).unwrap();
        let a = run();
        let b = run();
        assert_eq!(a, b);
        assert_eq!(a.len(), flips.len() * 3 * ARMS.len() * BUDGETS.len());
        assert!(a
            .iter()
            .all(|r| r.inner.topology_unchanged && r.free_alloc_invariant && r.init_ok));
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        let cols = csv_header().split(',').count();
        assert!(csv.lines().all(|l| l.split(',').count() == cols));
        let id = compare(&a, 50, K_MAX, Slice::All, EG, EG);
        assert_eq!(id.acc.ties, 3);
        let (g, h) = readings(&a, &flips);
        assert!(g.0);
        if !(g.0 && g.1 && g.2) {
            assert_eq!(h, [None; N_HYP]);
        } else {
            assert!(h.iter().all(Option::is_some));
        }
        let rep = report(&a, &flips, &BUDGETS);
        assert!(rep.contains("readings: G1_integrity=true"));
        assert_eq!(rep, report(&b, &flips, &BUDGETS));
        assert!(run_grid_design(None, &d, &[50], &[], &BUDGETS, &ARMS).is_err());
        assert!(run_grid_design(None, &d, &[50], &[1], &[Budget::Paced(0)], &ARMS).is_err());
        let tagged = [arm(Strategy::Tagged, false)];
        assert!(run_grid_design(None, &d, &[50], &[1], &BUDGETS, &tagged).is_err());
        assert_eq!(Arm::parse("E/E-cue0").unwrap(), CUE0);
        assert_eq!(Arm::parse("E/E-cue").unwrap(), CUE1);
        assert!(Arm::parse("E-tag").is_err());
    }

    #[test]
    fn readings_on_empty_grid_are_not_evaluable() {
        let (g, h) = readings(&[], &FLIPS);
        assert!(!g.0 && !g.1);
        assert_eq!(h, [None; N_HYP]);
        assert_eq!(cue_cost_below_budget(&[], 50), (0, 0));
    }
}

//! B888-4 cycle 20: **does the silent cue + last-resort win of cycle 19 hold
//! on fresh seeds, more held-out task orders and larger k, at equal write
//! budget against the supervised control?**
//!
//! Cycle 19 (#132, H7) found that the silent cue with the last-resort guard
//! (`E/E-cue0+last-resort@paced100`, Zg) is the best setting at equal
//! writes at k = 8 tasks (ACC_all 0.644, +0.097 over the supervised
//! control S1) on seeds 361..400 and two random held-out orders per seed.
//! This module re-tests that win, unchanged, on:
//!
//! * fresh seeds 401..440;
//! * per seed, the **reversed** task order plus three new seeded
//!   permutations (new salt);
//! * k = 8, 12 and 16 tasks, where the 64 reserved cue lines give each task
//!   8, 5 and 4 private cue lines (the guard's free capacity shrinks).
//!
//! Every setting is an unchanged `consolidation_guard::run_setting` code
//! path (bit-identity is tested), so the engine's checkpoint/resume
//! (`consolidation_guard::Checkpoint` + `PlasticCircuit::save`) applies
//! as-is; it is re-tested bit-exact for Zg at k = 16 on the reversed
//! order. New instrumentation: cue-capacity use per row (cue lines per task,
//! cue-line edges, cue-line edges non-zero / at weight 15 at the end).
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph with 4-bit
//! weights; "cue", "guard", "task order" and "forgetting" name parts of an
//! artificial benchmark, not claims about synapses, biology, insect memory
//! or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/CUE_ROBUSTNESS.md`
//! (committed and pushed before any learning run on the hypothesis seeds).

use crate::connectome::Connectome;
use crate::consolidation_guard::{self as cgd, Guard, Setting};
use crate::cue_init as ci;
use crate::cue_reach as cr;
use crate::error_gated::{paired, Paired, EFFECT};
use crate::generalization_bench::GenSpec;
use crate::novel_split::{self as ns, Design, Fixture, Slice};
use crate::seq_generalisation as sg;
use crate::write_allocation as wa;
use std::io;

fn bad(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Run lengths (each on its own fixture).
pub const KS: [usize; 3] = [8, 12, 16];
pub const K_MAIN: usize = 8;
pub const K_MID: usize = 12;
pub const K_BIG: usize = 16;
/// Flip levels; 50 is primary.
pub const FLIPS: [u32; 3] = [25, 50, 100];
pub const PRIMARY_FLIP: u32 = 50;
/// Hypothesis seeds (never used by an earlier B888-4 cycle).
pub const SEEDS: std::ops::RangeInclusive<u64> = 401..=440;
/// Pilot seeds (writes and guard counters only, identity order).
pub const PILOT_SEEDS: std::ops::RangeInclusive<u64> = 1..=5;
/// Seeded random held-out orders per seed (in addition to the reversal).
pub const RANDOM_ORDERS: usize = 3;
/// Held-out orders per seed: reversed + `RANDOM_ORDERS`.
pub const HELDOUT_ORDERS: usize = 1 + RANDOM_ORDERS;
/// Salt of this cycle's order generator (cycle 18: `..4C18..`, 19: `..4C19..`).
pub const ORDER_SALT: u64 = 0x0DE5_B888_4C20_0A1D;
/// Same design as cycles 18/19.
pub const DESIGN: Design = cgd::DESIGN;

/// The pre-registered settings, in CSV order.
pub const SETTINGS: [Setting; 5] = [
    Setting::Sup,
    Setting::Rew,
    Setting::Err,
    Setting::Z,
    Setting::Zg,
];

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(0x5851_F42D_4C95_7F2D)
            .wrapping_add(0x1405_7B7E_F767_814F);
        self.0 >> 17
    }
}

/// Task orders of one seed: 0 = identity, 1 = reversed, `2..2 + n_random`
/// = distinct seeded Fisher–Yates permutations ≠ identity and ≠ reversed
/// (the cycle 18 LCG keyed by `seed ^ ORDER_SALT`). Independent of the flip.
pub fn orders(seed: u64, k: usize, n_random: usize) -> io::Result<Vec<Vec<usize>>> {
    let fact: u128 = (1..=k as u128).product();
    if k < 3 || (n_random as u128) + 2 > fact {
        return Err(bad("not enough distinct task orders"));
    }
    let mut out = vec![(0..k).collect::<Vec<usize>>(), (0..k).rev().collect()];
    let mut rng = Lcg(seed ^ ORDER_SALT);
    let mut tries = 0;
    while out.len() < 2 + n_random {
        tries += 1;
        if tries > 10_000 {
            return Err(bad("could not draw held-out orders"));
        }
        let mut p: Vec<usize> = (0..k).collect();
        for i in (1..k).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            p.swap(i, j);
        }
        if !out.contains(&p) {
            out.push(p);
        }
    }
    Ok(out)
}

/// Kind of order index `ix`.
#[must_use]
pub fn order_kind(ix: usize) -> &'static str {
    match ix {
        0 => "identity",
        1 => "reversed",
        _ => "random",
    }
}

/// Cue-capacity use of one weight vector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CueUse {
    /// Cue lines per task (64 reserved lines split over k cues).
    pub lines_per_task: usize,
    /// Edges leaving any cue line.
    pub edges: u64,
    /// Cue-line edges with a non-zero weight.
    pub live: u64,
    /// Cue-line edges at the maximum weight 15.
    pub full: u64,
}

/// Cue-capacity use of `weights` on the fixture's cue lines.
pub fn cue_use(fx: &Fixture, weights: &[u8]) -> io::Result<CueUse> {
    let cue = ci::cue_edges(fx)?;
    if cue.len() != weights.len() {
        return Err(bad("weight vector does not match the graph"));
    }
    let lines_per_task = fx.cues.first().map_or(0, Vec::len);
    if fx.cues.iter().any(|c| c.len() != lines_per_task) {
        return Err(bad("cues of unequal size"));
    }
    let mut u = CueUse {
        lines_per_task,
        ..CueUse::default()
    };
    for (&on, &w) in cue.iter().zip(weights) {
        if on {
            u.edges += 1;
            u.live += u64::from(w != 0);
            u.full += u64::from(w == 15);
        }
    }
    Ok(u)
}

/// One measurement: the cycle 19 row plus cue-capacity use at the end.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub inner: cgd::Row,
    pub cue: CueUse,
}

impl Row {
    #[must_use]
    pub fn k(&self) -> usize {
        self.inner.k
    }
    #[must_use]
    pub fn acc(&self, s: Slice, kk: usize) -> f64 {
        self.inner.acc(s, kk)
    }
}

/// Paced limits from per-stage cumulative write curves (`paced100`).
#[must_use]
pub fn paced_limits(curves: &[Vec<u64>]) -> Vec<crate::write_budget::Limit> {
    cgd::paced_limits(curves)
}

/// All rows of one (k, flip, seed): per order, uncapped `error-gated` first
/// (its per-stage write curves set the `paced100` caps of every other
/// setting).
pub fn run_seed(
    graph: Option<&Connectome>,
    design: &Design,
    k: usize,
    flip: u32,
    seed: u64,
    n_random: usize,
) -> io::Result<Vec<Row>> {
    let spec = GenSpec {
        flip_pct: flip,
        ..design.spec
    };
    let fx = sg::fixture(graph, design, k, flip, seed)?;
    let mut rows = Vec::new();
    for (o, order) in orders(seed, k, n_random)?.into_iter().enumerate() {
        let e = cgd::run_setting(&fx, &spec, seed, Setting::Err, &order, &sg::free(k))?;
        let curves: Vec<Vec<u64>> = e.logs.iter().map(|t| t.ledger.cum.clone()).collect();
        let paced = paced_limits(&curves);
        let e_paced = cgd::run_setting(&fx, &spec, seed, Setting::Err, &order, &paced)?;
        let identity =
            e_paced.weights == e.weights && e_paced.logs.iter().all(|t| t.ledger.refused == 0);
        let cn_pairs: Vec<u64> = order.iter().map(|&t| sg::pair_counts(&fx, t)[2]).collect();
        let mut e = Some(e);
        for s in SETTINGS {
            let run = match s {
                Setting::Err => e.take().ok_or_else(|| bad("error-gated run missing"))?,
                _ => cgd::run_setting(&fx, &spec, seed, s, &order, &paced)?,
            };
            let init = ci::initial_circuit(&fx, &spec, s.base().arm())?;
            let cue = cue_use(&fx, &run.weights)?;
            rows.push(Row {
                inner: cgd::Row {
                    setting: s,
                    k,
                    flip,
                    seed,
                    order_ix: o,
                    order: order.clone(),
                    cn_pairs: cn_pairs.clone(),
                    saturated_end: crate::error_gated::saturated(&run.weights),
                    topology_unchanged: run.topology_unchanged,
                    init_ok: ci::init_matches(&fx, &spec, s.base().arm(), init.weights())?,
                    paced_identity: s != Setting::Err || identity,
                    digest: wa::digest(&run.weights),
                    ev: run.ev,
                    logs: run.logs,
                    glogs: run.glogs,
                    cost: run.cost,
                },
                cue,
            });
        }
    }
    Ok(rows)
}

/// Full grid, (k, flip, seed) jobs spread over `threads` worker threads; the
/// output order (k, flip, seed, order, setting) and content do not depend on
/// the thread count (tested).
#[allow(clippy::too_many_arguments)]
pub fn run_grid_threads(
    graph: Option<&Connectome>,
    design: &Design,
    ks: &[usize],
    flips: &[u32],
    seeds: &[u64],
    n_random: usize,
    threads: usize,
) -> io::Result<Vec<Row>> {
    if seeds.is_empty() || flips.is_empty() || ks.is_empty() {
        return Err(bad("need at least one k, seed and flip"));
    }
    let jobs: Vec<(usize, u32, u64)> = ks
        .iter()
        .flat_map(|&k| {
            flips
                .iter()
                .flat_map(move |&f| seeds.iter().map(move |&s| (k, f, s)))
        })
        .collect();
    let threads = threads.clamp(1, jobs.len());
    let mut results: Vec<Option<io::Result<Vec<Row>>>> = (0..jobs.len()).map(|_| None).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|w| {
                let jobs = &jobs;
                scope.spawn(move || {
                    (w..jobs.len())
                        .step_by(threads)
                        .map(|ix| {
                            let (k, f, s) = jobs[ix];
                            (ix, run_seed(graph, design, k, f, s, n_random))
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            if let Ok(v) = h.join() {
                for (ix, r) in v {
                    results[ix] = Some(r);
                }
            }
        }
    });
    let mut rows = Vec::new();
    for r in results {
        rows.extend(r.ok_or_else(|| bad("worker thread failed"))??);
    }
    Ok(rows)
}

/// Pre-registered grid.
pub fn run_grid(graph: Option<&Connectome>, flips: &[u32], seeds: &[u64]) -> io::Result<Vec<Row>> {
    run_grid_threads(
        graph,
        &DESIGN,
        &KS,
        flips,
        seeds,
        RANDOM_ORDERS,
        sg::default_threads(),
    )
}

/// Which task orders a statistic uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orders {
    Identity,
    /// All held-out orders (index ≥ 1); per seed the mean over them.
    Heldout,
    /// One order index only.
    Only(usize),
}

impl Orders {
    #[must_use]
    pub fn name(self) -> String {
        match self {
            Self::Identity => "identity".to_string(),
            Self::Heldout => "heldout".to_string(),
            Self::Only(ix) => format!("order{ix}_{}", order_kind(ix)),
        }
    }
    fn has(self, r: &Row) -> bool {
        match self {
            Self::Identity => r.inner.order_ix == 0,
            Self::Heldout => r.inner.order_ix > 0,
            Self::Only(ix) => r.inner.order_ix == ix,
        }
    }
}

/// Selection of rows: (run length, flip, orders).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sel {
    pub k: usize,
    pub flip: u32,
    pub orders: Orders,
}

fn rows_of(rows: &[Row], q: Sel, x: Setting) -> impl Iterator<Item = &Row> {
    rows.iter().filter(move |r| {
        r.inner.k == q.k && r.inner.flip == q.flip && r.inner.setting == x && q.orders.has(r)
    })
}

fn seeds_of(rows: &[Row], q: Sel) -> Vec<u64> {
    let mut s: Vec<u64> = rows
        .iter()
        .filter(|r| r.inner.k == q.k && r.inner.flip == q.flip)
        .map(|r| r.inner.seed)
        .collect();
    s.sort_unstable();
    s.dedup();
    s
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn seed_value(rows: &[Row], q: Sel, seed: u64, x: Setting, f: &dyn Fn(&Row) -> f64) -> Option<f64> {
    let v: Vec<f64> = rows_of(rows, q, x)
        .filter(|r| r.inner.seed == seed)
        .map(f)
        .collect();
    (!v.is_empty()).then(|| mean(&v))
}

fn diff(rows: &[Row], q: Sel, x: Setting, base: Setting, f: &dyn Fn(&Row) -> f64) -> Vec<f64> {
    seeds_of(rows, q)
        .into_iter()
        .filter_map(|s| Some(seed_value(rows, q, s, x, f)? - seed_value(rows, q, s, base, f)?))
        .collect()
}

/// x vs base paired by seed, metrics at prefix length `kk` (≤ q.k); fields
/// oriented so that > 0 favours x (`forget` = FGT(base) − FGT(x)).
#[must_use]
pub fn compare(
    rows: &[Row],
    q: Sel,
    kk: usize,
    s: Slice,
    x: Setting,
    base: Setting,
) -> cr::Comparison {
    let d = |f: &dyn Fn(&Row) -> f64| paired(&diff(rows, q, x, base, f));
    cr::Comparison {
        acc: d(&|r| r.inner.acc(s, kk)),
        learn: d(&|r| r.inner.learn(s, kk)),
        last: d(&|r| r.inner.last(s, kk)),
        forget: d(&|r| -r.inner.forgetting(s, kk)),
    }
}

/// Per-seed worst held-out order: min over the held-out orders of
/// ACC_all(x) − ACC_all(base) in that order (paired within the order).
#[must_use]
pub fn worst_order(rows: &[Row], k: usize, flip: u32, x: Setting, base: Setting) -> Paired {
    let mut ixs: Vec<usize> = rows
        .iter()
        .filter(|r| r.inner.k == k && r.inner.flip == flip && r.inner.order_ix > 0)
        .map(|r| r.inner.order_ix)
        .collect();
    ixs.sort_unstable();
    ixs.dedup();
    let q0 = Sel {
        k,
        flip,
        orders: Orders::Heldout,
    };
    let a = Slice::All;
    let f = |r: &Row| r.inner.acc(a, k);
    let d: Vec<f64> = seeds_of(rows, q0)
        .into_iter()
        .filter_map(|s| {
            let per: Vec<f64> = ixs
                .iter()
                .filter_map(|&ix| {
                    let q = Sel {
                        orders: Orders::Only(ix),
                        ..q0
                    };
                    Some(seed_value(rows, q, s, x, &f)? - seed_value(rows, q, s, base, &f)?)
                })
                .collect();
            per.into_iter().reduce(f64::min)
        })
        .collect();
    paired(&d)
}

/// Growth: per seed [ACC_all(x) − ACC_all(base)] at q.k minus the same at
/// prefix `k_lo` of the same runs.
#[must_use]
pub fn growth(rows: &[Row], q: Sel, x: Setting, base: Setting, k_lo: usize) -> Paired {
    let a = Slice::All;
    let hi = diff(rows, q, x, base, &|r| r.inner.acc(a, q.k));
    let lo = diff(rows, q, x, base, &|r| r.inner.acc(a, k_lo));
    paired(&hi.iter().zip(&lo).map(|(h, l)| h - l).collect::<Vec<_>>())
}

/// Positive: mean > 0 and bootstrap CI lower bound > 0.
#[must_use]
pub fn positive(d: &Paired) -> bool {
    d.n > 0 && d.mean > 0.0 && d.ci_lo > 0.0
}

/// The #132 H7 criterion: FGT cut mean ≥ EFFECT with CI lower bound > 0 and
/// positive ACC difference.
#[must_use]
pub fn guard_on_cue(c: &cr::Comparison) -> bool {
    c.forget.n > 0 && c.forget.mean >= EFFECT && c.forget.ci_lo > 0.0 && positive(&c.acc)
}

#[must_use]
pub fn setting_mean(rows: &[Row], q: Sel, x: Setting, f: &dyn Fn(&Row) -> f64) -> f64 {
    mean(&rows_of(rows, q, x).map(f).collect::<Vec<_>>())
}

/// Fraction of rows of x with `f(row) > 0` (`None` counts as 0).
#[must_use]
pub fn frac(rows: &[Row], q: Sel, x: Setting, f: &dyn Fn(&Row) -> Option<u64>) -> f64 {
    let v: Vec<&Row> = rows_of(rows, q, x).collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().filter(|r| f(r).unwrap_or(0) > 0).count() as f64 / v.len() as f64
    }
}

fn ks_of(rows: &[Row]) -> Vec<usize> {
    let mut ks: Vec<usize> = rows.iter().map(|r| r.inner.k).collect();
    ks.sort_unstable();
    ks.dedup();
    ks
}

fn flips_of(rows: &[Row]) -> Vec<u32> {
    let mut f: Vec<u32> = rows.iter().map(|r| r.inner.flip).collect();
    f.sort_unstable();
    f.dedup();
    f
}

/// Median CN pairs per (seed, task) at (k, flip), identity order.
#[must_use]
pub fn median_cn(rows: &[Row], k: usize, flip: u32) -> u64 {
    let q = Sel {
        k,
        flip,
        orders: Orders::Identity,
    };
    let mut v: Vec<u64> = rows_of(rows, q, Setting::Err)
        .flat_map(|r| r.inner.cn_pairs.iter().copied())
        .collect();
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

/// Orders valid: permutations, index 0 identity, index 1 reversed, indices
/// pairwise distinct per (k, flip, seed) and shared by all settings.
#[must_use]
pub fn orders_valid(rows: &[Row]) -> bool {
    rows.iter().all(|r| {
        let (k, o) = (r.inner.k, &r.inner.order);
        sg::is_permutation(o, k)
            && (r.inner.order_ix != 0 || o.iter().enumerate().all(|(i, &t)| i == t))
            && (r.inner.order_ix != 1 || o.iter().enumerate().all(|(i, &t)| i + t + 1 == k))
            && rows
                .iter()
                .filter(|y| {
                    y.inner.k == k && y.inner.flip == r.inner.flip && y.inner.seed == r.inner.seed
                })
                .all(|y| (y.inner.order_ix == r.inner.order_ix) == (y.inner.order == *o))
    })
}

/// Guard counters consistent with each setting's guard and its ledger.
#[must_use]
pub fn guards_consistent(rows: &[Row]) -> bool {
    rows.iter().all(|r| match r.inner.setting.guard() {
        None => r.inner.glogs.iter().all(|g| g.stats.is_none()),
        Some(gd) => r.inner.glogs.iter().zip(&r.inner.logs).all(|(g, t)| {
            g.stats.is_some_and(|s| {
                s.blocked == 0
                    && (gd == Guard::LastResort || s.deferred == 0)
                    && s.charged(gd) == t.ledger.writes
                    && s.effective == t.ledger.writes
            })
        }),
    })
}

/// Design gates: (G1 integrity, G2 manipulation at every k, G3 CN pairs at
/// the largest k).
#[must_use]
pub fn gates(rows: &[Row]) -> (bool, bool, bool) {
    let g1 = !rows.is_empty()
        && rows.iter().all(|r| {
            r.inner.topology_unchanged
                && r.inner.init_ok
                && r.inner.paced_identity
                && r.inner.within_cap()
                && r.cue.live <= r.cue.edges
                && r.cue.full <= r.cue.live
        })
        && orders_valid(rows)
        && guards_consistent(rows);
    let ks = ks_of(rows);
    let g2 = !ks.is_empty()
        && ks.iter().all(|&k| {
            let q = Sel {
                k,
                flip: PRIMARY_FLIP,
                orders: Orders::Heldout,
            };
            frac(rows, q, Setting::Zg, &|r: &Row| r.inner.deferred()) >= 0.5
                && frac(rows, q, Setting::Z, &|r: &Row| Some(r.inner.cue_writes())) >= 0.5
        });
    let g3 = ks
        .last()
        .is_some_and(|&k| median_cn(rows, k, PRIMARY_FLIP) >= ns::MIN_NOVEL_PAIRS);
    (g1, g2, g3)
}

/// Number of pre-registered hypotheses.
pub const N_HYP: usize = 10;

/// Pre-registered readings: (gates, [H1..H10]); `None` if a gate fails or
/// the needed (k, flip) is missing. All hypotheses use held-out orders.
#[must_use]
pub fn readings(rows: &[Row]) -> ((bool, bool, bool), [Option<bool>; N_HYP]) {
    use Setting::{Err, Rew, Sup, Zg, Z};
    let g = gates(rows);
    let mut h = [None; N_HYP];
    if !(g.0 && g.1 && g.2) {
        return (g, h);
    }
    let has = |k: usize| {
        rows.iter()
            .any(|r| r.inner.k == k && r.inner.flip == PRIMARY_FLIP && r.inner.order_ix > 0)
    };
    let all = Slice::All;
    let sel = |k| Sel {
        k,
        flip: PRIMARY_FLIP,
        orders: Orders::Heldout,
    };
    if has(K_MAIN) {
        let c = |x, b| compare(rows, sel(K_MAIN), K_MAIN, all, x, b);
        h[0] = Some(c(Zg, Sup).passes());
        h[1] = Some(guard_on_cue(&c(Zg, Z)));
        let per: Vec<bool> = (1..=HELDOUT_ORDERS)
            .map(|ix| {
                let q = Sel {
                    orders: Orders::Only(ix),
                    ..sel(K_MAIN)
                };
                positive(&compare(rows, q, K_MAIN, all, Zg, Sup).acc)
            })
            .collect();
        h[5] = Some(per.iter().all(|&b| b));
    }
    if has(K_MID) {
        h[2] = Some(compare(rows, sel(K_MID), K_MID, all, Zg, Sup).passes());
    }
    if has(K_BIG) {
        let c = |x, b| compare(rows, sel(K_BIG), K_BIG, all, x, b);
        h[3] = Some(c(Zg, Sup).passes());
        h[4] = Some(positive(&c(Zg, Z).acc));
        h[6] = Some(positive(&worst_order(rows, K_BIG, PRIMARY_FLIP, Zg, Sup)));
        h[7] = Some(positive(&c(Zg, Err).acc));
        h[8] = Some(positive(&c(Zg, Rew).acc));
        let gr = growth(rows, sel(K_BIG), Zg, Sup, K_MAIN);
        h[9] = Some(gr.n > 0 && gr.ci_lo > -EFFECT);
    }
    (g, h)
}

fn dash<T: ToString>(v: &[T]) -> String {
    if v.is_empty() {
        return "-".to_string();
    }
    v.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("-")
}

fn opt(v: Option<u64>) -> String {
    v.map_or_else(|| "-".to_string(), |x| x.to_string())
}

/// CSV header.
#[must_use]
pub fn csv_header() -> String {
    [
        "schema",
        "graph",
        "k",
        "flip_pct",
        "setting",
        "seed",
        "order_ix",
        "order_kind",
        "order",
        "cue_lines_per_task",
        "cn_pairs_s",
        "acc_all",
        "acc_cn",
        "learn_all",
        "learn_cn",
        "fgt_all",
        "fgt_cn",
        "last_all",
        "last_cn",
        "acc_all_k8",
        "fgt_all_k8",
        "r_all_diag_s",
        "r_all_final_s",
        "cap_total",
        "charged_writes",
        "effective_writes",
        "cue_writes",
        "refused",
        "overwrites",
        "deferred",
        "overwrites_s",
        "net_reversals",
        "consolidated_end",
        "cue_edges",
        "cue_live_end",
        "cue_full_end",
        "saturated_end",
        "init_ok",
        "paced_identity",
        "within_cap",
        "weights_digest",
        "topology_unchanged",
    ]
    .join(",")
}

/// Deterministic CSV (no timing column).
#[must_use]
pub fn to_csv(rows: &[Row], graph_label: &str) -> String {
    let mut out = csv_header();
    out.push('\n');
    let f6 = |x: f64| format!("{x:.6}");
    let (a, c) = (Slice::All, Slice::Cn);
    for row in rows {
        let r = &row.inner;
        let k = r.k;
        let k8 = K_MAIN.min(k);
        let row_of = |last: bool| {
            (0..k)
                .map(|j| format!("{:.4}", r.r(a, if last { k - 1 } else { j }, j)))
                .collect::<Vec<_>>()
                .join("-")
        };
        let ovs: Vec<String> = r
            .glogs
            .iter()
            .map(|g| opt(g.stats.map(|s| s.overwrites)))
            .collect();
        let f: Vec<String> = vec![
            "b888-4-cue-robustness-v1".to_string(),
            graph_label.to_string(),
            k.to_string(),
            r.flip.to_string(),
            r.setting.name().to_string(),
            r.seed.to_string(),
            r.order_ix.to_string(),
            order_kind(r.order_ix).to_string(),
            dash(&r.order),
            row.cue.lines_per_task.to_string(),
            dash(&r.cn_pairs),
            f6(r.acc(a, k)),
            f6(r.acc(c, k)),
            f6(r.learn(a, k)),
            f6(r.learn(c, k)),
            f6(r.forgetting(a, k)),
            f6(r.forgetting(c, k)),
            f6(r.last(a, k)),
            f6(r.last(c, k)),
            f6(r.acc(a, k8)),
            f6(r.forgetting(a, k8)),
            row_of(false),
            row_of(true),
            opt(r.cap_total()),
            r.charged_writes().to_string(),
            r.effective_writes().to_string(),
            r.cue_writes().to_string(),
            r.refused().to_string(),
            opt(r.overwrites()),
            opt(r.deferred()),
            ovs.join("-"),
            r.net_reversals().to_string(),
            r.consolidated_end().to_string(),
            row.cue.edges.to_string(),
            row.cue.live.to_string(),
            row.cue.full.to_string(),
            r.saturated_end.to_string(),
            r.init_ok.to_string(),
            r.paced_identity.to_string(),
            r.within_cap().to_string(),
            format!("{:016x}", r.digest),
            r.topology_unchanged.to_string(),
        ];
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

fn fmt_p(d: &Paired) -> String {
    format!(
        "{:+.3}±{:.3}[{:+.3},{:+.3}]{}/{}/{}",
        d.mean, d.sd, d.ci_lo, d.ci_hi, d.better, d.ties, d.worse
    )
}

/// Human-readable report (deterministic; printed by the CLI).
#[must_use]
pub fn report(rows: &[Row]) -> String {
    use std::fmt::Write;
    use Setting::{Err, Rew, Sup, Zg, Z};
    let mut o = String::new();
    let ks = ks_of(rows);
    let flips = flips_of(rows);
    let (a, c) = (Slice::All, Slice::Cn);
    let og = |v: Option<u64>| v.map_or(f64::NAN, |x| x as f64);
    let mut sels = vec![Orders::Identity, Orders::Heldout];
    sels.extend((1..=HELDOUT_ORDERS).map(Orders::Only));
    let _ = writeln!(
        o,
        "setting_summary: k,flip,orders,setting,acc_all,acc_cn,learn_all,learn_cn,fgt_all,fgt_cn,last_all,charged_writes,effective_writes,cue_writes,refused,overwrites,deferred,net_reversals,consolidated_end,saturated_end,cue_lines_per_task,cue_edges,cue_live_end,cue_full_end"
    );
    for &k in &ks {
        for &flip in &flips {
            for &ord in &sels {
                let q = Sel {
                    k,
                    flip,
                    orders: ord,
                };
                for s in SETTINGS {
                    if rows_of(rows, q, s).next().is_none() {
                        continue;
                    }
                    let m = |f: &dyn Fn(&Row) -> f64| setting_mean(rows, q, s, f);
                    let _ = writeln!(
                        o,
                        "{k},{flip},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
                        ord.name(),
                        s.name(),
                        m(&|r| r.inner.acc(a, k)),
                        m(&|r| r.inner.acc(c, k)),
                        m(&|r| r.inner.learn(a, k)),
                        m(&|r| r.inner.learn(c, k)),
                        m(&|r| r.inner.forgetting(a, k)),
                        m(&|r| r.inner.forgetting(c, k)),
                        m(&|r| r.inner.last(a, k)),
                        m(&|r| r.inner.charged_writes() as f64),
                        m(&|r| r.inner.effective_writes() as f64),
                        m(&|r| r.inner.cue_writes() as f64),
                        m(&|r| r.inner.refused() as f64),
                        m(&|r| og(r.inner.overwrites())),
                        m(&|r| og(r.inner.deferred())),
                        m(&|r| r.inner.net_reversals() as f64),
                        m(&|r| r.inner.consolidated_end() as f64),
                        m(&|r| r.inner.saturated_end as f64),
                        m(&|r| r.cue.lines_per_task as f64),
                        m(&|r| r.cue.edges as f64),
                        m(&|r| r.cue.live as f64),
                        m(&|r| r.cue.full as f64),
                    );
                }
            }
        }
    }
    let _ = writeln!(
        o,
        "comparison: k,flip,orders,slice,x,base,d_acc,d_learn,d_last,d_fgt(+=forgets_less),passes_sequence_rule"
    );
    let pairs = [
        (Zg, Sup),
        (Zg, Z),
        (Zg, Err),
        (Zg, Rew),
        (Z, Sup),
        (Err, Sup),
        (Rew, Sup),
        (Z, Err),
    ];
    for &k in &ks {
        for &flip in &flips {
            for &ord in &sels {
                let q = Sel {
                    k,
                    flip,
                    orders: ord,
                };
                for s in [a, c] {
                    for &(x, base) in &pairs {
                        // Per-order rows only for the primary pair on all pairs.
                        if matches!(ord, Orders::Only(_)) && (s != a || x != Zg) {
                            continue;
                        }
                        let cmp = compare(rows, q, k, s, x, base);
                        if cmp.acc.n == 0 {
                            continue;
                        }
                        let _ = writeln!(
                            o,
                            "{k},{flip},{},{},{},{},{},{},{},{},{}",
                            ord.name(),
                            s.name(),
                            x.name(),
                            base.name(),
                            fmt_p(&cmp.acc),
                            fmt_p(&cmp.learn),
                            fmt_p(&cmp.last),
                            fmt_p(&cmp.forget),
                            cmp.passes()
                        );
                    }
                }
            }
        }
    }
    let _ = writeln!(
        o,
        "worst_heldout_order(acc_all, per seed min over held-out orders): k,flip,x,base,d"
    );
    for &k in &ks {
        for &flip in &flips {
            for &(x, base) in &[(Zg, Sup), (Zg, Z), (Zg, Err), (Z, Sup), (Err, Sup)] {
                let w = worst_order(rows, k, flip, x, base);
                if w.n > 0 {
                    let _ = writeln!(o, "{k},{flip},{},{},{}", x.name(), base.name(), fmt_p(&w));
                }
            }
        }
    }
    let _ = writeln!(
        o,
        "growth(acc_all, k vs prefix k_lo of the same runs): k,k_lo,flip,orders,x,base,d(k)-d(k_lo)"
    );
    for &k in &ks {
        for k_lo in [K_MAIN, K_MID].into_iter().filter(|&p| p < k) {
            for &flip in &flips {
                for ord in [Orders::Identity, Orders::Heldout] {
                    let q = Sel {
                        k,
                        flip,
                        orders: ord,
                    };
                    for &(x, base) in &[(Zg, Sup), (Zg, Z), (Z, Sup), (Err, Sup), (Zg, Err)] {
                        let g = growth(rows, q, x, base, k_lo);
                        if g.n > 0 {
                            let _ = writeln!(
                                o,
                                "{k},{k_lo},{flip},{},{},{},{}",
                                ord.name(),
                                x.name(),
                                base.name(),
                                fmt_p(&g)
                            );
                        }
                    }
                }
            }
        }
    }
    for &k in &ks {
        for &flip in &flips {
            let q = Sel {
                k,
                flip,
                orders: Orders::Heldout,
            };
            let _ = writeln!(
                o,
                "manipulation: k={k} flip={flip} deferred_frac(Zg)={:.3} cue_write_frac(Z)={:.3} mean_overwrites(E,Z,Zg)=({:.1},{:.1},{:.1}) mean_deferred(Zg)={:.1} mean_cue_writes(Z,Zg)=({:.1},{:.1}) mean_effective_writes(S1,R1,E,Z,Zg)=({:.1},{:.1},{:.1},{:.1},{:.1})",
                frac(rows, q, Zg, &|r: &Row| r.inner.deferred()),
                frac(rows, q, Z, &|r: &Row| Some(r.inner.cue_writes())),
                setting_mean(rows, q, Err, &|r| og(r.inner.overwrites())),
                setting_mean(rows, q, Z, &|r| og(r.inner.overwrites())),
                setting_mean(rows, q, Zg, &|r| og(r.inner.overwrites())),
                setting_mean(rows, q, Zg, &|r| og(r.inner.deferred())),
                setting_mean(rows, q, Z, &|r| r.inner.cue_writes() as f64),
                setting_mean(rows, q, Zg, &|r| r.inner.cue_writes() as f64),
                setting_mean(rows, q, Sup, &|r| r.inner.effective_writes() as f64),
                setting_mean(rows, q, Rew, &|r| r.inner.effective_writes() as f64),
                setting_mean(rows, q, Err, &|r| r.inner.effective_writes() as f64),
                setting_mean(rows, q, Z, &|r| r.inner.effective_writes() as f64),
                setting_mean(rows, q, Zg, &|r| r.inner.effective_writes() as f64),
            );
        }
    }
    for &k in &ks {
        let _ = writeln!(
            o,
            "median_cn(k={k},flip={PRIMARY_FLIP})={}",
            median_cn(rows, k, PRIMARY_FLIP)
        );
    }
    let (g, h) = readings(rows);
    let mut line = format!(
        "readings: G1_integrity={} G2_manipulation={} G3_cn_pairs={}",
        g.0, g.1, g.2
    );
    for (i, x) in h.iter().enumerate() {
        let _ = write!(line, " H{}={}", i + 1, yes(*x));
    }
    let _ = writeln!(o, "{line}");
    o
}

/// One pilot census line: (k, cue lines per task, seed, setting, [charged
/// writes, effective writes, cue writes, refused, overwrites, deferred]).
pub type CensusLine = (usize, usize, u64, Setting, [Option<u64>; 6]);

/// Pilot census on the identity order: writes and guard counters only (the
/// #132 census restricted to this cycle's settings). No accuracy is computed
/// and no held-out order is drawn.
pub fn census(
    graph: Option<&Connectome>,
    design: &Design,
    k: usize,
    flip: u32,
    seeds: &[u64],
) -> io::Result<Vec<CensusLine>> {
    let lines = sg::cue_cap(design, k)?;
    Ok(cgd::census(graph, design, k, flip, seeds)?
        .into_iter()
        .filter(|(_, _, s, _)| SETTINGS.contains(s))
        .map(|(k, seed, s, v)| (k, lines, seed, s, [v[0], v[1], v[2], v[3], v[4], v[6]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generalization_bench::GraphShape;
    use crate::plasticity::PlasticCircuit;
    use crate::task_sequence as ts;

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
        std::env::temp_dir().join(format!("v888-cue-robust-{tag}-{}", std::process::id()))
    }

    #[test]
    fn orders_are_new_valid_distinct_and_include_the_reversal() {
        for seed in [1, 401, 440] {
            for k in KS {
                let o = orders(seed, k, RANDOM_ORDERS).unwrap();
                assert_eq!(o, orders(seed, k, RANDOM_ORDERS).unwrap());
                assert_eq!(o.len(), 1 + HELDOUT_ORDERS);
                assert_eq!(o[0], (0..k).collect::<Vec<_>>());
                assert_eq!(o[1], (0..k).rev().collect::<Vec<_>>());
                assert!(o.iter().all(|p| sg::is_permutation(p, k)));
                for i in 0..o.len() {
                    for j in 0..i {
                        assert_ne!(o[i], o[j]);
                    }
                }
                // New salt: none of this cycle's random orders is a cycle 18
                // or cycle 19 held-out order of the same seed.
                let old: Vec<Vec<usize>> = sg::orders(seed, k, 2)
                    .unwrap()
                    .into_iter()
                    .chain(cgd::orders(seed, k, 2).unwrap())
                    .filter(|p| p != &o[0])
                    .collect();
                assert!(o[2..].iter().all(|p| !old.contains(p)));
            }
        }
        assert!(orders(1, 2, 1).is_err());
        assert!(orders(1, 3, 4).is_ok());
        assert!(orders(1, 3, 5).is_err());
        assert_eq!(order_kind(0), "identity");
        assert_eq!(order_kind(1), "reversed");
        assert_eq!(order_kind(3), "random");
    }

    #[test]
    fn rows_are_the_cycle19_engine_bit_exactly() {
        let d = small();
        let (k, flip, seed) = (6, 50, 2);
        let rows = run_seed(None, &d, k, flip, seed, 2).unwrap();
        assert_eq!(rows.len(), 4 * SETTINGS.len());
        let spec = GenSpec {
            flip_pct: flip,
            ..d.spec
        };
        let fx = sg::fixture(None, &d, k, flip, seed).unwrap();
        for r in &rows {
            let order = &r.inner.order;
            let e = cgd::run_setting(&fx, &spec, seed, Setting::Err, order, &sg::free(k)).unwrap();
            let curves: Vec<Vec<u64>> = e.logs.iter().map(|t| t.ledger.cum.clone()).collect();
            let lim = if r.inner.setting == Setting::Err {
                sg::free(k)
            } else {
                paced_limits(&curves)
            };
            let x = cgd::run_setting(&fx, &spec, seed, r.inner.setting, order, &lim).unwrap();
            assert_eq!(r.inner.ev, x.ev);
            assert_eq!(r.inner.logs, x.logs);
            assert_eq!(r.inner.glogs, x.glogs);
            assert_eq!(r.inner.cost, x.cost);
            assert_eq!(r.inner.digest, wa::digest(&x.weights));
            assert_eq!(r.cue, cue_use(&fx, &x.weights).unwrap());
            assert!(r.inner.topology_unchanged && r.inner.init_ok && r.inner.paced_identity);
        }
    }

    #[test]
    fn cue_use_counts_capacity() {
        let d = small();
        let fx = sg::fixture(None, &d, 16, 50, 3).unwrap();
        let spec = d.spec;
        let z = ci::initial_circuit(&fx, &spec, Setting::Z.base().arm()).unwrap();
        let s = ci::initial_circuit(&fx, &spec, Setting::Sup.base().arm()).unwrap();
        let uz = cue_use(&fx, z.weights()).unwrap();
        let us = cue_use(&fx, s.weights()).unwrap();
        assert_eq!(uz.lines_per_task, sg::cue_cap(&d, 16).unwrap());
        assert!(uz.edges > 0);
        assert_eq!((uz.live, uz.full), (0, 0));
        assert_eq!((us.edges, us.live, us.full), (uz.edges, uz.edges, 0));
        let mut w = s.weights().to_vec();
        let cue = ci::cue_edges(&fx).unwrap();
        let e = cue.iter().position(|&b| b).unwrap();
        w[e] = 15;
        assert_eq!(cue_use(&fx, &w).unwrap().full, 1);
        assert!(cue_use(&fx, &w[1..]).is_err());
    }

    #[test]
    fn zg_resumes_bit_exactly_at_k16_on_the_reversed_order() {
        let d = small();
        let (k, seed) = (16, 5);
        let fx = sg::fixture(None, &d, k, 50, seed).unwrap();
        let g = &fx.graph;
        let order = orders(seed, k, 1).unwrap().remove(1);
        let e = cgd::run_setting(&fx, &d.spec, seed, Setting::Err, &order, &sg::free(k)).unwrap();
        let curves: Vec<Vec<u64>> = e.logs.iter().map(|t| t.ledger.cum.clone()).collect();
        let lim = paced_limits(&curves);
        let s = Setting::Zg;
        let full = cgd::run_setting(&fx, &d.spec, seed, s, &order, &lim).unwrap();
        let mut c = ci::initial_circuit(&fx, &d.spec, s.base().arm()).unwrap();
        let mut dir = vec![0_i8; g.edges()];
        let mut stamp = vec![0_u8; g.edges()];
        for (i, &t) in order.iter().enumerate() {
            let n = ts::task_order(&d.spec, seed, t).len();
            let start = c.weights().to_vec();
            let mut cp =
                cgd::Checkpoint::start(&order, i, &lim[i], Guard::LastResort, &dir, &stamp, &start)
                    .unwrap();
            for end in [n / 4, n / 2, n] {
                cgd::train_range(&mut c, &fx, &d.spec, seed, s, &lim[i], &mut cp, end).unwrap();
                let (pw, pc) = (tmp("w"), tmp("c"));
                c.save(&pw).unwrap();
                cp.save(&pc).unwrap();
                c = PlasticCircuit::load(&pw, g).unwrap();
                cp = cgd::Checkpoint::load(&pc, g.edges()).unwrap();
                std::fs::remove_file(&pw).unwrap();
                std::fs::remove_file(&pc).unwrap();
            }
            assert_eq!(Some(cp.g), full.glogs[i].stats, "stage {i}");
            assert_eq!(cp.inner.inner.inner.ledger, full.logs[i].ledger);
            let (rev, moved) =
                cgd::consolidate(&mut dir, &mut stamp, i, &cp.start, c.weights()).unwrap();
            assert_eq!(rev, full.glogs[i].net_reversals);
            assert_eq!(moved, full.glogs[i].moved);
        }
        assert_eq!(c.weights(), &full.weights[..]);
        // A checkpoint written for Zg is refused by a setting with another
        // guard.
        let cp = cgd::Checkpoint::start(
            &order,
            0,
            &lim[0],
            Guard::LastResort,
            &vec![0; g.edges()],
            &vec![0; g.edges()],
            full.weights.as_slice(),
        )
        .unwrap();
        let mut cp2 = cp.clone();
        let mut c2 = ci::initial_circuit(&fx, &d.spec, Setting::Z.base().arm()).unwrap();
        assert!(cgd::train_range(
            &mut c2,
            &fx,
            &d.spec,
            seed,
            Setting::Z,
            &lim[0],
            &mut cp2,
            1
        )
        .is_err());
    }

    #[test]
    fn grid_is_deterministic_thread_independent_and_gates_detect_tampering() {
        let d = small();
        let ks = [6, 8];
        let flips = [50];
        let seeds = [1, 2, 3];
        let a = run_grid_threads(None, &d, &ks, &flips, &seeds, 2, 1).unwrap();
        let b = run_grid_threads(None, &d, &ks, &flips, &seeds, 2, 4).unwrap();
        assert_eq!(a, b);
        assert_eq!(
            a.len(),
            ks.len() * flips.len() * seeds.len() * 4 * SETTINGS.len()
        );
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        let cols = csv_header().split(',').count();
        assert!(csv.lines().all(|l| l.split(',').count() == cols));
        let rep = report(&a);
        assert_eq!(rep, report(&b));
        assert!(rep.contains("readings: G1_integrity=true"));
        let g = gates(&a);
        assert!(g.0);
        let (g2, h) = readings(&a);
        assert_eq!(g, g2);
        // k = 12 / 16 absent: those hypotheses stay not evaluable.
        assert_eq!(h[2], None);
        assert_eq!(h[3], None);
        if !(g.0 && g.1 && g.2) {
            assert_eq!(h, [None; N_HYP]);
        }
        // Identity comparison: all ties; worst order ≤ held-out mean.
        let q = Sel {
            k: 8,
            flip: 50,
            orders: Orders::Heldout,
        };
        let id = compare(&a, q, 8, Slice::All, Setting::Zg, Setting::Zg);
        assert_eq!(id.acc.ties, 3);
        let w = worst_order(&a, 8, 50, Setting::Zg, Setting::Sup);
        let m = compare(&a, q, 8, Slice::All, Setting::Zg, Setting::Sup);
        assert_eq!(w.n, 3);
        assert!(w.mean <= m.acc.mean + 1e-12);
        let gr = growth(&a, q, Setting::Zg, Setting::Zg, 6);
        assert_eq!(gr.ties, 3);
        assert!(run_grid_threads(None, &d, &ks, &[50], &[], 2, 1).is_err());
        // Tampering: the reversal swapped, Zg with a blocked step, an
        // over-cap run and a cue counter beyond its edges all fail G1.
        let mut t = a.clone();
        let ix = t.iter().position(|r| r.inner.order_ix == 1).unwrap();
        t[ix].inner.order.swap(0, 1);
        assert!(!gates(&t).0);
        let mut t = a.clone();
        let ix = t
            .iter()
            .position(|r| r.inner.setting == Setting::Zg)
            .unwrap();
        t[ix].inner.glogs[0].stats.as_mut().unwrap().blocked = 1;
        assert!(!guards_consistent(&t));
        assert!(!gates(&t).0);
        assert_eq!(readings(&t).1, [None; N_HYP]);
        let mut t = a.clone();
        t[0].cue.live = t[0].cue.edges + 1;
        assert!(!gates(&t).0);
        let mut t = a;
        let ix = t
            .iter()
            .position(|r| r.inner.setting == Setting::Sup)
            .unwrap();
        t[ix].inner.logs[0].ledger.writes = t[ix].inner.logs[0].ledger.cap + 1;
        assert!(!gates(&t).0);
    }

    #[test]
    fn readings_are_evaluated_on_the_preregistered_ks() {
        // Small design at the pre-registered k = 8 / 12 / 16 (cue cap 4 gives
        // 2 / 1 / 1 lines per cue): gates are computed and, if they pass,
        // every hypothesis is evaluated.
        let d = small();
        let rows = run_grid_threads(None, &d, &KS, &[PRIMARY_FLIP], &[1, 2], 2, 3).unwrap();
        let (g, h) = readings(&rows);
        assert!(g.0);
        if g.0 && g.1 && g.2 {
            assert!(h.iter().all(Option::is_some));
        } else {
            assert_eq!(h, [None; N_HYP]);
        }
        assert!(rows
            .iter()
            .all(|r| r.cue.lines_per_task == sg::cue_cap(&d, r.inner.k).unwrap()));
    }

    #[test]
    fn census_counts_writes_without_accuracy() {
        let d = small();
        let lines = census(None, &d, 12, 50, &[1, 2]).unwrap();
        assert_eq!(lines.len(), 2 * SETTINGS.len());
        for (k, cue, _, s, v) in &lines {
            assert_eq!(*k, 12);
            assert_eq!(*cue, sg::cue_cap(&d, 12).unwrap());
            assert!(SETTINGS.contains(s));
            assert_eq!(v[4].is_some(), s.guard().is_some());
            assert_eq!(v[5].is_some(), s.guard().is_some());
            if *s != Setting::Zg {
                assert_eq!(v[5].unwrap_or(0), 0);
            }
        }
    }
}

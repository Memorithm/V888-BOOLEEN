//! B888-4 cycle 21: **cue capacity by line sharing**. Does giving each task
//! more cue lines — by sharing the *same* reserved lines between tasks —
//! restore the silent cue + last-resort lead over the supervised control at
//! k = 16 and k = 24, at equal lines and equal write budget?
//!
//! Cycle 20 (#133) found that the silent cue + last-resort guard (Zg) beats
//! the supervised control S1 at equal writes on fresh seeds and held-out
//! orders, but at k = 16 by only +0.042 (< 0.05, its H4 failed), and that
//! the decline tracks cue capacity per task: the 64 reserved cue lines are
//! split into k disjoint cues (4 lines per task at k = 16).
//!
//! This module keeps the graph, the reserved lines, the pattern pool, the
//! tasks, the initial weights and the `paced100` write budget fixed and only
//! changes which reserved lines each task's cue uses: the share-r code of
//! task j is the union of the cycle 20 blocks `B_j … B_{j+r−1}` (cyclic), so
//! every reserved line is shared by r tasks and each task sees r × as many
//! cue lines. r = 1 is the cycle 20 cue (bit-identical, tested). Every rule
//! is an unchanged `consolidation_guard::run_setting` code path, so the
//! engine's checkpoint/resume applies as-is (re-tested bit-exact for Zg at
//! share 2).
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph with 4-bit
//! weights; "cue", "code", "sharing", "guard" and "forgetting" name parts of
//! an artificial benchmark, not claims about synapses, biology, insect
//! memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/CUE_CAPACITY.md`
//! (committed and pushed before any learning run on the hypothesis seeds).

use crate::connectome::Connectome;
use crate::consolidation_guard::{self as cgd, Setting};
use crate::cue_displacement::TaskLog;
use crate::cue_init as ci;
use crate::cue_reach as cr;
use crate::cue_robustness as crob;
use crate::error_gated::{paired, Paired, EFFECT};
use crate::generalization_bench::GenSpec;
use crate::novel_split::{self as ns, Design, Fixture, Slice};
use crate::seq_generalisation as sg;
use crate::write_allocation as wa;
use crate::write_budget::Limit;
use std::io;

fn bad(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Run lengths (each on its own fixture).
pub const KS: [usize; 3] = [8, 16, 24];
pub const K_SMALL: usize = 8;
pub const K_MAIN: usize = 16;
pub const K_BIG: usize = 24;
/// Sharing factors (tasks per reserved line); 1 = the cycle 20 cue.
pub const SHARES: [usize; 3] = [1, 2, 4];
/// Flip levels; 50 is primary.
pub const FLIPS: [u32; 3] = [25, 50, 100];
pub const PRIMARY_FLIP: u32 = 50;
/// Hypothesis seeds (never used by an earlier B888-4 cycle).
pub const SEEDS: std::ops::RangeInclusive<u64> = 441..=480;
/// Pilot seeds (writes, guard counters and reach only, identity order).
pub const PILOT_SEEDS: std::ops::RangeInclusive<u64> = 1..=5;
/// Seeded random held-out orders per seed (in addition to the reversal).
pub const RANDOM_ORDERS: usize = 3;
/// Held-out orders per seed: reversed + `RANDOM_ORDERS`.
pub const HELDOUT_ORDERS: usize = 1 + RANDOM_ORDERS;
/// Salt of this cycle's order generator (cycle 20: `..4C20..`).
pub const ORDER_SALT: u64 = 0x0DE5_B888_4C21_0A1D;
/// H10 non-inferiority margin (k = 8).
pub const NO_HARM: f64 = 0.01;
/// Same design as cycles 18–20.
pub const DESIGN: Design = crob::DESIGN;

/// One compared arm: a #132 setting and, for cued settings, its share.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Arm {
    pub setting: Setting,
    /// Tasks per reserved cue line (0 for uncued settings).
    pub share: usize,
}

const fn arm(setting: Setting, share: usize) -> Arm {
    Arm { setting, share }
}

pub const S1: Arm = arm(Setting::Sup, 0);
pub const R1: Arm = arm(Setting::Rew, 0);
pub const E: Arm = arm(Setting::Err, 0);
pub const Z1: Arm = arm(Setting::Z, 1);
pub const ZG1: Arm = arm(Setting::Zg, 1);
pub const Z2: Arm = arm(Setting::Z, 2);
pub const ZG2: Arm = arm(Setting::Zg, 2);
pub const Z4: Arm = arm(Setting::Z, 4);
pub const ZG4: Arm = arm(Setting::Zg, 4);

/// The pre-registered arms, in CSV order.
pub const ARMS: [Arm; 9] = [S1, R1, E, Z1, ZG1, Z2, ZG2, Z4, ZG4];

impl Arm {
    #[must_use]
    pub fn name(self) -> String {
        if self.share == 0 {
            self.setting.name().to_string()
        } else {
            format!("{}/share{}", self.setting.name(), self.share)
        }
    }
    #[must_use]
    pub fn short(self) -> String {
        match self.setting {
            Setting::Sup => "S1".to_string(),
            Setting::Rew => "R1".to_string(),
            Setting::Err => "E".to_string(),
            Setting::Z => format!("Z{}", self.share),
            Setting::Zg => format!("Zg{}", self.share),
            _ => self.setting.name().to_string(),
        }
    }
}

/// Share-r codes of `blocks`: code j = sorted union of blocks
/// `(j + i) mod k`, i < r.
pub fn share_codes(blocks: &[Vec<usize>], r: usize) -> io::Result<Vec<Vec<usize>>> {
    let k = blocks.len();
    if r == 0 || r > k || blocks.iter().any(Vec::is_empty) {
        return Err(bad("invalid share or empty block"));
    }
    Ok((0..k)
        .map(|j| {
            let mut c: Vec<usize> = (0..r)
                .flat_map(|i| blocks[(j + i) % k].iter().copied())
                .collect();
            c.sort_unstable();
            c.dedup();
            c
        })
        .collect())
}

/// The k-task fixture with share-r cues (graph, pool, tasks, splits
/// unchanged; only the per-task cue lists differ).
pub fn shared_fixture(fx: &Fixture, r: usize) -> io::Result<Fixture> {
    let mut out = fx.clone();
    out.cues = share_codes(&fx.cues, r)?;
    Ok(out)
}

/// Codes valid against `blocks`: r = 1 is the blocks, every line of the
/// blocks is in exactly r codes, codes pairwise distinct (r < k), each code
/// has exactly the lines of its r blocks.
#[must_use]
pub fn codes_valid(blocks: &[Vec<usize>], codes: &[Vec<usize>], r: usize) -> bool {
    let k = blocks.len();
    if codes.len() != k || r == 0 || r > k {
        return false;
    }
    let lines: Vec<usize> = blocks.iter().flatten().copied().collect();
    let mut uniq = lines.clone();
    uniq.sort_unstable();
    uniq.dedup();
    if uniq.len() != lines.len() {
        return false;
    }
    let exact = lines
        .iter()
        .all(|l| codes.iter().filter(|c| c.contains(l)).count() == r);
    let only = codes.iter().flatten().all(|l| uniq.contains(l));
    let sizes =
        (0..k).all(|j| codes[j].len() == (0..r).map(|i| blocks[(j + i) % k].len()).sum::<usize>());
    let distinct = r == k || (0..k).all(|i| (0..i).all(|j| codes[i] != codes[j]));
    let same = r != 1
        || codes.iter().zip(blocks).all(|(c, b)| {
            let mut b = b.clone();
            b.sort_unstable();
            *c == b
        });
    exact && only && sizes && distinct && same
}

/// Distinct targets reached by the edges of `lines`.
#[must_use]
pub fn reach(graph: &Connectome, lines: &[usize]) -> usize {
    let mut t: Vec<u32> = lines
        .iter()
        .filter(|&&s| s + 1 < graph.offsets.len())
        .flat_map(|&s| (graph.offsets[s]..graph.offsets[s + 1]).map(|e| graph.targets[e as usize]))
        .collect();
    t.sort_unstable();
    t.dedup();
    t.len()
}

/// Mean reach per task code.
#[must_use]
pub fn mean_reach(fx: &Fixture) -> f64 {
    if fx.cues.is_empty() {
        return 0.0;
    }
    fx.cues
        .iter()
        .map(|c| reach(&fx.graph, c) as f64)
        .sum::<f64>()
        / fx.cues.len() as f64
}

/// Cue-capacity use of one row.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CueUse {
    /// Cue lines per task (r × the cycle 20 block size; 0 if uncued).
    pub lines_per_task: usize,
    /// Mean distinct targets reached per task code (0 if uncued).
    pub reach: f64,
    /// Edges leaving any reserved cue line (same for every r).
    pub edges: u64,
    /// Cue-line edges with a non-zero weight at the end.
    pub live: u64,
    /// Cue-line edges at weight 15 at the end.
    pub full: u64,
}

/// Cue use of `weights` on fixture `fx` (its cues = the row's codes).
pub fn cue_use(fx: &Fixture, weights: &[u8], cued: bool) -> io::Result<CueUse> {
    let cue = ci::cue_edges(fx)?;
    if cue.len() != weights.len() {
        return Err(bad("weight vector does not match the graph"));
    }
    let lines = fx.cues.first().map_or(0, Vec::len);
    if fx.cues.iter().any(|c| c.len() != lines) {
        return Err(bad("cues of unequal size"));
    }
    let mut u = CueUse::default();
    if cued {
        u.lines_per_task = lines;
        u.reach = mean_reach(fx);
    }
    for (&on, &w) in cue.iter().zip(weights) {
        if on {
            u.edges += 1;
            u.live += u64::from(w != 0);
            u.full += u64::from(w == 15);
        }
    }
    Ok(u)
}

/// One measurement: the #132 row (its `setting` field), the share and cue
/// use. `codes_ok` = the row's codes passed [`codes_valid`].
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub inner: cgd::Row,
    pub share: usize,
    pub cue: CueUse,
    pub codes_ok: bool,
}

impl Row {
    #[must_use]
    pub fn arm(&self) -> Arm {
        arm(self.inner.setting, self.share)
    }
}

/// Task orders of one seed (the #133 generator with this cycle's salt):
/// 0 = identity, 1 = reversed, then distinct seeded permutations.
pub fn orders(seed: u64, k: usize, n_random: usize) -> io::Result<Vec<Vec<usize>>> {
    crob_orders(seed, k, n_random, ORDER_SALT)
}

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

fn crob_orders(seed: u64, k: usize, n_random: usize, salt: u64) -> io::Result<Vec<Vec<usize>>> {
    let enough = (1..=k as u128)
        .try_fold(1_u128, u128::checked_mul)
        .is_none_or(|f| (n_random as u128) + 2 <= f);
    if k < 3 || !enough {
        return Err(bad("not enough distinct task orders"));
    }
    let mut out = vec![(0..k).collect::<Vec<usize>>(), (0..k).rev().collect()];
    let mut rng = Lcg(seed ^ salt);
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

/// All rows of one (k, flip, seed): per order, uncapped E first (its
/// per-stage write curves set the `paced100` caps of every other arm), then
/// S1, R1 and the cued arms on their share-r fixtures.
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
    let mut fxs = Vec::with_capacity(SHARES.len());
    for r in SHARES {
        let f = shared_fixture(&fx, r)?;
        let ok = codes_valid(&fx.cues, &f.cues, r);
        fxs.push((r, f, ok));
    }
    let mut rows = Vec::new();
    for (o, order) in orders(seed, k, n_random)?.into_iter().enumerate() {
        let e = cgd::run_setting(&fx, &spec, seed, Setting::Err, &order, &sg::free(k))?;
        let curves: Vec<Vec<u64>> = e.logs.iter().map(|t| t.ledger.cum.clone()).collect();
        let paced = cgd::paced_limits(&curves);
        let e_paced = cgd::run_setting(&fx, &spec, seed, Setting::Err, &order, &paced)?;
        let identity =
            e_paced.weights == e.weights && e_paced.logs.iter().all(|t| t.ledger.refused == 0);
        let cn_pairs: Vec<u64> = order.iter().map(|&t| sg::pair_counts(&fx, t)[2]).collect();
        let mut e = Some(e);
        for a in ARMS {
            let (f, ok) = if a.share == 0 {
                (&fx, true)
            } else {
                let (_, f, ok) = fxs
                    .iter()
                    .find(|(r, _, _)| *r == a.share)
                    .ok_or_else(|| bad("missing share fixture"))?;
                (f, *ok)
            };
            let run = match a.setting {
                Setting::Err => e.take().ok_or_else(|| bad("error-gated run missing"))?,
                s => cgd::run_setting(f, &spec, seed, s, &order, &paced)?,
            };
            let s = a.setting;
            let init = ci::initial_circuit(f, &spec, s.base().arm())?;
            let cue = cue_use(f, &run.weights, s.cued())?;
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
                    init_ok: ci::init_matches(f, &spec, s.base().arm(), init.weights())?,
                    paced_identity: s != Setting::Err || identity,
                    digest: wa::digest(&run.weights),
                    ev: run.ev,
                    logs: run.logs,
                    glogs: run.glogs,
                    cost: run.cost,
                },
                share: a.share,
                cue,
                codes_ok: ok,
            });
        }
    }
    Ok(rows)
}

/// Full grid, (k, flip, seed) jobs spread over `threads` worker threads; the
/// output order (k, flip, seed, order, arm) and content do not depend on
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
    // Largest k first so long jobs start early (output order is restored).
    let mut jobs: Vec<(usize, usize, u32, u64)> = Vec::new();
    for &k in ks {
        for &f in flips {
            for &s in seeds {
                jobs.push((jobs.len(), k, f, s));
            }
        }
    }
    let mut sched = jobs.clone();
    sched.sort_by_key(|&(ix, k, _, _)| (std::cmp::Reverse(k), ix));
    let threads = threads.clamp(1, jobs.len());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<Option<io::Result<Vec<Row>>>> = (0..jobs.len()).map(|_| None).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let (sched, next) = (&sched, &next);
                scope.spawn(move || {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&(ix, k, f, s)) = sched.get(i) else {
                            break;
                        };
                        out.push((ix, run_seed(graph, design, k, f, s, n_random)));
                    }
                    out
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
            Self::Only(ix) => format!("order{ix}_{}", crob::order_kind(ix)),
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

fn rows_of(rows: &[Row], q: Sel, x: Arm) -> impl Iterator<Item = &Row> {
    rows.iter().filter(move |r| {
        r.inner.k == q.k && r.inner.flip == q.flip && r.arm() == x && q.orders.has(r)
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

fn seed_value(rows: &[Row], q: Sel, seed: u64, x: Arm, f: &dyn Fn(&Row) -> f64) -> Option<f64> {
    let v: Vec<f64> = rows_of(rows, q, x)
        .filter(|r| r.inner.seed == seed)
        .map(f)
        .collect();
    (!v.is_empty()).then(|| mean(&v))
}

fn diff(rows: &[Row], q: Sel, x: Arm, base: Arm, f: &dyn Fn(&Row) -> f64) -> Vec<f64> {
    seeds_of(rows, q)
        .into_iter()
        .filter_map(|s| Some(seed_value(rows, q, s, x, f)? - seed_value(rows, q, s, base, f)?))
        .collect()
}

/// x vs base paired by seed, metrics at prefix length `kk` (≤ q.k); fields
/// oriented so that > 0 favours x (`forget` = FGT(base) − FGT(x)).
#[must_use]
pub fn compare(rows: &[Row], q: Sel, kk: usize, s: Slice, x: Arm, base: Arm) -> cr::Comparison {
    let d = |f: &dyn Fn(&Row) -> f64| paired(&diff(rows, q, x, base, f));
    cr::Comparison {
        acc: d(&|r| r.inner.acc(s, kk)),
        learn: d(&|r| r.inner.learn(s, kk)),
        last: d(&|r| r.inner.last(s, kk)),
        forget: d(&|r| -r.inner.forgetting(s, kk)),
    }
}

/// Per-seed worst held-out order: min over held-out orders of
/// ACC_all(x) − ACC_all(base) in that order.
#[must_use]
pub fn worst_order(rows: &[Row], k: usize, flip: u32, x: Arm, base: Arm) -> Paired {
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
    let f = |r: &Row| r.inner.acc(Slice::All, k);
    let d: Vec<f64> = seeds_of(rows, q0)
        .into_iter()
        .filter_map(|s| {
            ixs.iter()
                .filter_map(|&ix| {
                    let q = Sel {
                        orders: Orders::Only(ix),
                        ..q0
                    };
                    Some(seed_value(rows, q, s, x, &f)? - seed_value(rows, q, s, base, &f)?)
                })
                .reduce(f64::min)
        })
        .collect();
    paired(&d)
}

/// Growth: per seed [ACC_all(x) − ACC_all(base)] at q.k minus the same at
/// prefix `k_lo` of the same runs.
#[must_use]
pub fn growth(rows: &[Row], q: Sel, x: Arm, base: Arm, k_lo: usize) -> Paired {
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

#[must_use]
pub fn setting_mean(rows: &[Row], q: Sel, x: Arm, f: &dyn Fn(&Row) -> f64) -> f64 {
    mean(&rows_of(rows, q, x).map(f).collect::<Vec<_>>())
}

/// Fraction of rows of x with `f(row) > 0` (`None` counts as 0).
#[must_use]
pub fn frac(rows: &[Row], q: Sel, x: Arm, f: &dyn Fn(&Row) -> Option<u64>) -> f64 {
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
    let mut v: Vec<u64> = rows_of(rows, q, E)
        .flat_map(|r| r.inner.cn_pairs.iter().copied())
        .collect();
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

/// Orders valid: permutations, 0 identity, 1 reversed, distinct per (k,
/// flip, seed) and shared by all arms.
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

/// Guard counters consistent with each arm's guard and its ledger.
#[must_use]
pub fn guards_consistent(rows: &[Row]) -> bool {
    rows.iter().all(|r| match r.inner.setting.guard() {
        None => r.inner.glogs.iter().all(|g| g.stats.is_none()),
        Some(gd) => r.inner.glogs.iter().zip(&r.inner.logs).all(|(g, t)| {
            g.stats.is_some_and(|s| {
                s.blocked == 0
                    && (gd == cgd::Guard::LastResort || s.deferred == 0)
                    && s.charged(gd) == t.ledger.writes
                    && s.effective == t.ledger.writes
            })
        }),
    })
}

/// Share and lines consistent: cued arms carry their pre-registered share
/// and r × the share-1 lines of the same (k, flip, seed); uncued arms 0.
#[must_use]
pub fn shares_consistent(rows: &[Row]) -> bool {
    rows.iter().all(|r| {
        if !r.inner.setting.cued() {
            return r.share == 0 && r.cue.lines_per_task == 0;
        }
        let base = rows.iter().find(|y| {
            y.share == 1
                && y.inner.k == r.inner.k
                && y.inner.flip == r.inner.flip
                && y.inner.seed == r.inner.seed
        });
        SHARES.contains(&r.share)
            && base.is_some_and(|b| r.cue.lines_per_task == r.share * b.cue.lines_per_task)
            && rows
                .iter()
                .filter(|y| {
                    y.inner.k == r.inner.k
                        && y.inner.flip == r.inner.flip
                        && y.inner.seed == r.inner.seed
                })
                .all(|y| y.cue.edges == r.cue.edges)
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
                && r.codes_ok
                && r.cue.live <= r.cue.edges
                && r.cue.full <= r.cue.live
        })
        && orders_valid(rows)
        && guards_consistent(rows)
        && shares_consistent(rows);
    let ks = ks_of(rows);
    let g2 = !ks.is_empty()
        && ks.iter().all(|&k| {
            let q = Sel {
                k,
                flip: PRIMARY_FLIP,
                orders: Orders::Heldout,
            };
            frac(rows, q, ZG2, &|r: &Row| r.inner.deferred()) >= 0.5
                && frac(rows, q, Z2, &|r: &Row| Some(r.inner.cue_writes())) >= 0.5
                && setting_mean(rows, q, ZG2, &|r| r.cue.reach)
                    > setting_mean(rows, q, ZG1, &|r| r.cue.reach)
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
        h[0] = Some(c(ZG2, S1).passes());
        h[1] = Some(positive(&c(ZG2, ZG1).acc));
        h[2] = Some(positive(&c(ZG2, ZG1).forget));
        h[3] = Some(positive(&c(Z2, Z1).acc));
        h[4] = Some(positive(&c(ZG4, ZG2).acc));
        h[5] = Some(positive(&worst_order(rows, K_MAIN, PRIMARY_FLIP, ZG2, ZG1)));
    }
    if has(K_BIG) {
        let c = |x, b| compare(rows, sel(K_BIG), K_BIG, all, x, b);
        h[6] = Some(c(ZG2, S1).passes());
        h[7] = Some(positive(&c(ZG2, ZG1).acc));
        let gr = growth(rows, sel(K_BIG), ZG2, S1, K_SMALL);
        h[8] = Some(gr.n > 0 && gr.ci_lo > -EFFECT);
    }
    if has(K_SMALL) {
        let d = compare(rows, sel(K_SMALL), K_SMALL, all, ZG2, ZG1).acc;
        h[9] = Some(d.n > 0 && d.ci_lo > -NO_HARM);
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
        "share",
        "seed",
        "order_ix",
        "order_kind",
        "order",
        "cue_lines_per_task",
        "cue_reach_mean",
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
        "acc_all_k16",
        "r_all_diag_s",
        "r_all_final_s",
        "cap_total",
        "charged_writes",
        "effective_writes",
        "cue_writes",
        "refused",
        "overwrites",
        "deferred",
        "net_reversals",
        "consolidated_end",
        "cue_edges",
        "cue_live_end",
        "cue_full_end",
        "saturated_end",
        "init_ok",
        "paced_identity",
        "within_cap",
        "codes_ok",
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
        let row_of = |last: bool| {
            (0..k)
                .map(|j| format!("{:.4}", r.r(a, if last { k - 1 } else { j }, j)))
                .collect::<Vec<_>>()
                .join("-")
        };
        let f: Vec<String> = vec![
            "b888-4-cue-capacity-v1".to_string(),
            graph_label.to_string(),
            k.to_string(),
            r.flip.to_string(),
            r.setting.name().to_string(),
            row.share.to_string(),
            r.seed.to_string(),
            r.order_ix.to_string(),
            crob::order_kind(r.order_ix).to_string(),
            dash(&r.order),
            row.cue.lines_per_task.to_string(),
            format!("{:.3}", row.cue.reach),
            dash(&r.cn_pairs),
            f6(r.acc(a, k)),
            f6(r.acc(c, k)),
            f6(r.learn(a, k)),
            f6(r.learn(c, k)),
            f6(r.forgetting(a, k)),
            f6(r.forgetting(c, k)),
            f6(r.last(a, k)),
            f6(r.last(c, k)),
            f6(r.acc(a, K_SMALL.min(k))),
            f6(r.acc(a, K_MAIN.min(k))),
            row_of(false),
            row_of(true),
            opt(r.cap_total()),
            r.charged_writes().to_string(),
            r.effective_writes().to_string(),
            r.cue_writes().to_string(),
            r.refused().to_string(),
            opt(r.overwrites()),
            opt(r.deferred()),
            r.net_reversals().to_string(),
            r.consolidated_end().to_string(),
            row.cue.edges.to_string(),
            row.cue.live.to_string(),
            row.cue.full.to_string(),
            r.saturated_end.to_string(),
            r.init_ok.to_string(),
            r.paced_identity.to_string(),
            r.within_cap().to_string(),
            row.codes_ok.to_string(),
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
    let mut o = String::new();
    let ks = ks_of(rows);
    let flips = flips_of(rows);
    let (a, c) = (Slice::All, Slice::Cn);
    let og = |v: Option<u64>| v.map_or(f64::NAN, |x| x as f64);
    let mut sels = vec![Orders::Identity, Orders::Heldout];
    sels.extend((1..=HELDOUT_ORDERS).map(Orders::Only));
    let _ = writeln!(
        o,
        "setting_summary: k,flip,orders,arm,acc_all,acc_cn,learn_all,learn_cn,fgt_all,fgt_cn,last_all,charged_writes,effective_writes,cue_writes,refused,overwrites,deferred,net_reversals,consolidated_end,saturated_end,cue_lines_per_task,cue_reach_mean,cue_edges,cue_live_end,cue_full_end"
    );
    for &k in &ks {
        for &flip in &flips {
            for &ord in &sels {
                let q = Sel {
                    k,
                    flip,
                    orders: ord,
                };
                for x in ARMS {
                    if rows_of(rows, q, x).next().is_none() {
                        continue;
                    }
                    let m = |f: &dyn Fn(&Row) -> f64| setting_mean(rows, q, x, f);
                    let _ = writeln!(
                        o,
                        "{k},{flip},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.2},{:.1},{:.1},{:.1}",
                        ord.name(),
                        x.short(),
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
                        m(&|r| r.cue.reach),
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
        (ZG2, S1),
        (ZG1, S1),
        (ZG4, S1),
        (Z2, S1),
        (ZG2, ZG1),
        (ZG4, ZG1),
        (ZG4, ZG2),
        (Z2, Z1),
        (Z4, Z1),
        (ZG2, Z2),
        (ZG2, E),
        (ZG2, R1),
        (E, S1),
        (R1, S1),
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
                        if matches!(ord, Orders::Only(_))
                            && (s != a || !(x == ZG2 && (base == S1 || base == ZG1)))
                        {
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
                            x.short(),
                            base.short(),
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
            for &(x, base) in &[(ZG2, S1), (ZG2, ZG1), (ZG4, ZG1), (ZG1, S1), (Z2, Z1)] {
                let w = worst_order(rows, k, flip, x, base);
                if w.n > 0 {
                    let _ = writeln!(o, "{k},{flip},{},{},{}", x.short(), base.short(), fmt_p(&w));
                }
            }
        }
    }
    let _ = writeln!(
        o,
        "growth(acc_all, k vs prefix k_lo of the same runs): k,k_lo,flip,orders,x,base,d(k)-d(k_lo)"
    );
    for &k in &ks {
        for k_lo in [K_SMALL, K_MAIN].into_iter().filter(|&p| p < k) {
            for &flip in &flips {
                for ord in [Orders::Identity, Orders::Heldout] {
                    let q = Sel {
                        k,
                        flip,
                        orders: ord,
                    };
                    for &(x, base) in &[(ZG2, S1), (ZG1, S1), (ZG4, S1), (ZG2, ZG1), (E, S1)] {
                        let g = growth(rows, q, x, base, k_lo);
                        if g.n > 0 {
                            let _ = writeln!(
                                o,
                                "{k},{k_lo},{flip},{},{},{},{}",
                                ord.name(),
                                x.short(),
                                base.short(),
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
                "manipulation: k={k} flip={flip} deferred_frac(Zg2)={:.3} cue_write_frac(Z2)={:.3} reach(Zg1,Zg2,Zg4)=({:.2},{:.2},{:.2}) lines(Zg1,Zg2,Zg4)=({:.1},{:.1},{:.1}) live_cue_edges(Zg1,Zg2,Zg4)=({:.1},{:.1},{:.1}) cue_writes(Zg1,Zg2,Zg4)=({:.1},{:.1},{:.1}) overwrites(Zg1,Zg2,Zg4)=({:.1},{:.1},{:.1}) deferred(Zg1,Zg2,Zg4)=({:.1},{:.1},{:.1}) effective_writes(S1,R1,E,Zg1,Zg2,Zg4)=({:.1},{:.1},{:.1},{:.1},{:.1},{:.1})",
                frac(rows, q, ZG2, &|r: &Row| r.inner.deferred()),
                frac(rows, q, Z2, &|r: &Row| Some(r.inner.cue_writes())),
                setting_mean(rows, q, ZG1, &|r| r.cue.reach),
                setting_mean(rows, q, ZG2, &|r| r.cue.reach),
                setting_mean(rows, q, ZG4, &|r| r.cue.reach),
                setting_mean(rows, q, ZG1, &|r| r.cue.lines_per_task as f64),
                setting_mean(rows, q, ZG2, &|r| r.cue.lines_per_task as f64),
                setting_mean(rows, q, ZG4, &|r| r.cue.lines_per_task as f64),
                setting_mean(rows, q, ZG1, &|r| r.cue.live as f64),
                setting_mean(rows, q, ZG2, &|r| r.cue.live as f64),
                setting_mean(rows, q, ZG4, &|r| r.cue.live as f64),
                setting_mean(rows, q, ZG1, &|r| r.inner.cue_writes() as f64),
                setting_mean(rows, q, ZG2, &|r| r.inner.cue_writes() as f64),
                setting_mean(rows, q, ZG4, &|r| r.inner.cue_writes() as f64),
                setting_mean(rows, q, ZG1, &|r| og(r.inner.overwrites())),
                setting_mean(rows, q, ZG2, &|r| og(r.inner.overwrites())),
                setting_mean(rows, q, ZG4, &|r| og(r.inner.overwrites())),
                setting_mean(rows, q, ZG1, &|r| og(r.inner.deferred())),
                setting_mean(rows, q, ZG2, &|r| og(r.inner.deferred())),
                setting_mean(rows, q, ZG4, &|r| og(r.inner.deferred())),
                setting_mean(rows, q, S1, &|r| r.inner.effective_writes() as f64),
                setting_mean(rows, q, R1, &|r| r.inner.effective_writes() as f64),
                setting_mean(rows, q, E, &|r| r.inner.effective_writes() as f64),
                setting_mean(rows, q, ZG1, &|r| r.inner.effective_writes() as f64),
                setting_mean(rows, q, ZG2, &|r| r.inner.effective_writes() as f64),
                setting_mean(rows, q, ZG4, &|r| r.inner.effective_writes() as f64),
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

/// One pilot census line: (k, arm, seed, lines per task, mean reach,
/// [charged writes, effective writes, cue writes, refused, overwrites,
/// deferred]).
pub type CensusLine = (usize, Arm, u64, usize, f64, [Option<u64>; 6]);

fn train_only(
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    s: Setting,
    order: &[usize],
    lim: &[Limit],
) -> io::Result<(Vec<TaskLog>, Vec<cgd::GuardLog>)> {
    let mut c = ci::initial_circuit(fx, spec, s.base().arm())?;
    let mut dir = vec![0_i8; fx.graph.edges()];
    let mut stamp = vec![0_u8; fx.graph.edges()];
    let (mut logs, mut glogs) = (Vec::new(), Vec::new());
    for (i, l) in lim.iter().enumerate() {
        let (inner, gl) =
            cgd::train_stage(&mut c, fx, spec, seed, s, order, i, l, &mut dir, &mut stamp)?;
        glogs.push(gl);
        logs.push(TaskLog {
            ledger: inner.inner.inner.ledger,
            stats: inner.inner.inner.stats,
            cue_cum: inner.inner.cue_cum,
            ..TaskLog::default()
        });
    }
    Ok((logs, glogs))
}

/// Pilot census on the identity order: writes, guard counters and cue reach
/// only. No accuracy is computed and no held-out order is drawn.
pub fn census(
    graph: Option<&Connectome>,
    design: &Design,
    k: usize,
    flip: u32,
    seeds: &[u64],
) -> io::Result<Vec<CensusLine>> {
    let spec = GenSpec {
        flip_pct: flip,
        ..design.spec
    };
    let mut out = Vec::new();
    for &seed in seeds {
        let fx = sg::fixture(graph, design, k, flip, seed)?;
        let order: Vec<usize> = (0..k).collect();
        let (e_logs, _) = train_only(&fx, &spec, seed, Setting::Err, &order, &sg::free(k))?;
        let curves: Vec<Vec<u64>> = e_logs.iter().map(|t| t.ledger.cum.clone()).collect();
        let paced = cgd::paced_limits(&curves);
        for a in ARMS {
            let f = if a.share == 0 {
                fx.clone()
            } else {
                shared_fixture(&fx, a.share)?
            };
            let lim = if a.setting == Setting::Err {
                sg::free(k)
            } else {
                paced.clone()
            };
            let (logs, glogs) = train_only(&f, &spec, seed, a.setting, &order, &lim)?;
            let sum = |g: &dyn Fn(&TaskLog) -> u64| logs.iter().map(g).sum::<u64>();
            let gsum = |g: &dyn Fn(&cgd::GuardStats) -> u64| -> Option<u64> {
                glogs.iter().map(|x| x.stats.map(|s| g(&s))).sum()
            };
            let charged = sum(&|t| t.ledger.writes);
            let effective = gsum(&|s| s.effective).unwrap_or(charged);
            let (lines, rch) = if a.share == 0 {
                (0, 0.0)
            } else {
                (f.cues[0].len(), mean_reach(&f))
            };
            out.push((
                k,
                a,
                seed,
                lines,
                rch,
                [
                    Some(charged),
                    Some(effective),
                    Some(sum(&|t| t.stats.cue_writes)),
                    Some(sum(&|t| t.ledger.refused)),
                    gsum(&|s| s.overwrites),
                    gsum(&|s| s.deferred),
                ],
            ));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cue_robustness as crob;
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
        std::env::temp_dir().join(format!("v888-cue-capacity-{tag}-{}", std::process::id()))
    }

    #[test]
    fn share_codes_are_cyclic_unions_with_exact_multiplicity() {
        let blocks = vec![vec![5, 1], vec![2, 9], vec![7, 3], vec![4, 8]];
        let c1 = share_codes(&blocks, 1).unwrap();
        assert_eq!(c1, vec![vec![1, 5], vec![2, 9], vec![3, 7], vec![4, 8]]);
        let c2 = share_codes(&blocks, 2).unwrap();
        assert_eq!(c2[0], vec![1, 2, 5, 9]);
        assert_eq!(c2[3], vec![1, 4, 5, 8]);
        for r in 1..=4 {
            let c = share_codes(&blocks, r).unwrap();
            assert!(codes_valid(&blocks, &c, r), "r={r}");
            assert!(c.iter().all(|x| x.len() == 2 * r));
        }
        // Wrong multiplicity / foreign line / duplicate codes are rejected.
        assert!(!codes_valid(&blocks, &c2, 1));
        let mut bad_c = c2.clone();
        bad_c[0][0] = 99;
        assert!(!codes_valid(&blocks, &bad_c, 2));
        let mut dup = c1.clone();
        dup[1] = dup[0].clone();
        assert!(!codes_valid(&blocks, &dup, 1));
        assert!(share_codes(&blocks, 0).is_err());
        assert!(share_codes(&blocks, 5).is_err());
        assert!(share_codes(&[vec![], vec![1]], 1).is_err());
    }

    #[test]
    fn shared_fixture_changes_only_the_cue_assignment() {
        let d = small();
        let fx = sg::fixture(None, &d, 6, 50, 3).unwrap();
        let f2 = shared_fixture(&fx, 2).unwrap();
        assert_eq!(f2.graph, fx.graph);
        assert_eq!(f2.task.train, fx.task.train);
        assert_eq!(f2.task.test, fx.task.test);
        assert_eq!(f2.task.valence, fx.task.valence);
        assert_eq!(ci::cue_edges(&f2).unwrap(), ci::cue_edges(&fx).unwrap());
        assert!(codes_valid(&fx.cues, &f2.cues, 2));
        assert!(mean_reach(&f2) >= mean_reach(&fx));
        // Zero init applies to the same edges at every share.
        let spec = d.spec;
        let a = ci::initial_circuit(&fx, &spec, Setting::Z.base().arm()).unwrap();
        let b = ci::initial_circuit(&f2, &spec, Setting::Z.base().arm()).unwrap();
        assert_eq!(a.weights(), b.weights());
    }

    #[test]
    fn orders_use_the_new_salt_and_include_the_reversal() {
        for seed in [1, 441, 480] {
            for k in KS {
                let o = orders(seed, k, RANDOM_ORDERS).unwrap();
                assert_eq!(o.len(), 1 + HELDOUT_ORDERS);
                assert_eq!(o[0], (0..k).collect::<Vec<_>>());
                assert_eq!(o[1], (0..k).rev().collect::<Vec<_>>());
                assert!(o.iter().all(|p| sg::is_permutation(p, k)));
                for i in 0..o.len() {
                    for j in 0..i {
                        assert_ne!(o[i], o[j]);
                    }
                }
                let old = crob::orders(seed, k, RANDOM_ORDERS).unwrap();
                assert!(o[2..].iter().all(|p| !old[2..].contains(p)));
            }
        }
        // Same generator as cycle 20 under its salt.
        assert_eq!(
            crob_orders(7, 8, 3, crob::ORDER_SALT).unwrap(),
            crob::orders(7, 8, 3).unwrap()
        );
        assert!(orders(1, 2, 1).is_err());
        assert!(orders(1, 3, 5).is_err());
        assert!(orders(1, 40, 3).is_ok());
    }

    #[test]
    fn rows_are_the_cycle19_engine_and_share1_is_the_cycle20_cue() {
        let d = small();
        let (k, flip, seed) = (6, 50, 2);
        let rows = run_seed(None, &d, k, flip, seed, 2).unwrap();
        assert_eq!(rows.len(), 4 * ARMS.len());
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
                cgd::paced_limits(&curves)
            };
            // Share 1 = the unmodified cycle 20 fixture.
            let f = if r.share <= 1 {
                fx.clone()
            } else {
                shared_fixture(&fx, r.share).unwrap()
            };
            let x = cgd::run_setting(&f, &spec, seed, r.inner.setting, order, &lim).unwrap();
            assert_eq!(r.inner.ev, x.ev);
            assert_eq!(r.inner.logs, x.logs);
            assert_eq!(r.inner.glogs, x.glogs);
            assert_eq!(r.inner.digest, wa::digest(&x.weights));
            assert!(r.codes_ok);
        }
        // The cycle 20 harness gives the same Zg / S1 rows on the same order.
        let o = rows[ARMS.len()].inner.order.clone();
        assert_eq!(o, (0..k).rev().collect::<Vec<_>>());
        let old = crob::run_seed(None, &d, k, flip, seed, 2).unwrap();
        for (mine, theirs) in [(ZG1, Setting::Zg), (S1, Setting::Sup), (Z1, Setting::Z)] {
            let a = rows
                .iter()
                .find(|r| r.arm() == mine && r.inner.order_ix == 1)
                .unwrap();
            let b = old
                .iter()
                .find(|r| r.inner.setting == theirs && r.inner.order_ix == 1)
                .unwrap();
            assert_eq!(a.inner.ev, b.inner.ev);
            assert_eq!(a.inner.digest, b.inner.digest);
        }
    }

    #[test]
    fn checkpoint_resume_is_bit_exact_for_zg_share2() {
        let d = small();
        let (k, flip, seed) = (6, 50, 4);
        let spec = GenSpec {
            flip_pct: flip,
            ..d.spec
        };
        let base = sg::fixture(None, &d, k, flip, seed).unwrap();
        let fx = shared_fixture(&base, 2).unwrap();
        let g = &fx.graph;
        let order: Vec<usize> = (0..k).rev().collect();
        let e = cgd::run_setting(&fx, &spec, seed, Setting::Err, &order, &sg::free(k)).unwrap();
        let curves: Vec<Vec<u64>> = e.logs.iter().map(|t| t.ledger.cum.clone()).collect();
        let lim = cgd::paced_limits(&curves);
        let s = Setting::Zg;
        let full = cgd::run_setting(&fx, &spec, seed, s, &order, &lim).unwrap();
        let mut c = ci::initial_circuit(&fx, &spec, s.base().arm()).unwrap();
        let mut dir = vec![0_i8; g.edges()];
        let mut stamp = vec![0_u8; g.edges()];
        for (i, &t) in order.iter().enumerate() {
            let n = ts::task_order(&spec, seed, t).len();
            let start = c.weights().to_vec();
            let mut cp = cgd::Checkpoint::start(
                &order,
                i,
                &lim[i],
                cgd::Guard::LastResort,
                &dir,
                &stamp,
                &start,
            )
            .unwrap();
            for end in [n / 4, n / 2, n] {
                cgd::train_range(&mut c, &fx, &spec, seed, s, &lim[i], &mut cp, end).unwrap();
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
            cgd::consolidate(&mut dir, &mut stamp, i, &cp.start, c.weights()).unwrap();
        }
        assert_eq!(c.weights(), &full.weights[..]);
    }

    #[test]
    fn grid_is_deterministic_thread_independent_and_gates_detect_tampering() {
        let d = small();
        let ks = [6, 8];
        let a = run_grid_threads(None, &d, &ks, &[50], &[1, 2, 3], 2, 1).unwrap();
        let b = run_grid_threads(None, &d, &ks, &[50], &[1, 2, 3], 2, 4).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), ks.len() * 3 * 4 * ARMS.len());
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        let cols = csv_header().split(',').count();
        assert!(csv.lines().all(|l| l.split(',').count() == cols));
        let rep = report(&a);
        assert_eq!(rep, report(&b));
        assert!(rep.contains("readings: G1_integrity=true"));
        assert!(gates(&a).0);
        let (g, h) = readings(&a);
        // k = 16 / 24 absent here: those hypotheses stay not evaluable.
        assert_eq!(h[0], None);
        assert_eq!(h[6], None);
        if !(g.0 && g.1 && g.2) {
            assert_eq!(h, [None; N_HYP]);
        }
        let q = Sel {
            k: 8,
            flip: 50,
            orders: Orders::Heldout,
        };
        assert_eq!(compare(&a, q, 8, Slice::All, ZG2, ZG2).acc.ties, 3);
        let w = worst_order(&a, 8, 50, ZG2, S1);
        assert!(w.mean <= compare(&a, q, 8, Slice::All, ZG2, S1).acc.mean + 1e-12);
        assert_eq!(growth(&a, q, ZG2, ZG2, 6).ties, 3);
        assert!(run_grid_threads(None, &d, &ks, &[50], &[], 2, 1).is_err());
        // Tampering fails G1: a bad code flag, a wrong share, a swapped
        // reversal and an over-cap run.
        let mut t = a.clone();
        let ix = t.iter().position(|r| r.arm() == ZG2).unwrap();
        t[ix].codes_ok = false;
        assert!(!gates(&t).0);
        let mut t = a.clone();
        t[ix].cue.lines_per_task += 1;
        assert!(!gates(&t).0);
        let mut t = a.clone();
        let ix = t.iter().position(|r| r.inner.order_ix == 1).unwrap();
        t[ix].inner.order.swap(0, 1);
        assert!(!gates(&t).0);
        let mut t = a;
        let ix = t.iter().position(|r| r.arm() == S1).unwrap();
        t[ix].inner.logs[0].ledger.writes = t[ix].inner.logs[0].ledger.cap + 1;
        assert!(!gates(&t).0);
    }

    #[test]
    fn readings_are_evaluated_on_the_preregistered_ks() {
        // Small design at k = 8 / 16 / 24 (cue cap 4 → 2 / 1 / 0 lines: 24
        // is too many tasks for 16 reserved lines, so use 8 and 16 only).
        let d = small();
        let rows = run_grid_threads(None, &d, &[8, 16], &[PRIMARY_FLIP], &[1, 2], 2, 3).unwrap();
        let (g, h) = readings(&rows);
        assert!(g.0);
        if g.0 && g.1 && g.2 {
            assert!(h[..6].iter().all(Option::is_some));
            assert!(h[9].is_some());
            assert!(h[6..9].iter().all(Option::is_none));
        } else {
            assert_eq!(h, [None; N_HYP]);
        }
    }

    #[test]
    fn census_counts_writes_and_reach_without_accuracy() {
        let d = small();
        let lines = census(None, &d, 6, 50, &[1, 2]).unwrap();
        assert_eq!(lines.len(), 2 * ARMS.len());
        let base = sg::cue_cap(&d, 6).unwrap();
        for (k, a, _, l, rch, v) in &lines {
            assert_eq!(*k, 6);
            assert_eq!(*l, a.share * base);
            assert_eq!(*rch > 0.0, a.share > 0);
            assert_eq!(v[4].is_some(), a.setting.guard().is_some());
            if a.setting != Setting::Zg {
                assert_eq!(v[5].unwrap_or(0), 0);
            }
        }
    }
}

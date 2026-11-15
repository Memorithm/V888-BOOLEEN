//! B888-4 cycle 22: **cue sharing by task agreement**. Cycle 21 (#134) found
//! that sharing reserved cue lines between tasks (cyclic share-2 codes) cuts
//! forgetting but pays ~1:1 in learning, so the silent cue + last-resort
//! guard (Zg) lead over the supervised control (S1) does not grow. A
//! secondary reading was that the cost tracks label conflict between the
//! tasks that share a line. This module tests that reading causally.
//!
//! Everything is held fixed — graph, reserved lines, blocks, pattern pool,
//! tasks, initial weights, the `paced100` write budget and the line
//! multiplicity (every reserved line in exactly 2 codes) — and only the
//! **partner** whose block a task borrows changes: chosen along a greedy
//! tour that maximises (matched) or minimises (anti) the agreement of the
//! task rules, or by task id (cyclic, = the #134 share-2 code, bit-identical,
//! tested). The agreement uses the task rules (edge valences) before
//! training: an *oracle task-similarity descriptor*, not a learner estimate.
//! Every rule is an unchanged `consolidation_guard::run_setting` code path,
//! so the engine's checkpoint/resume applies as-is (re-tested bit-exact for
//! Zg on the matched code).
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph with 4-bit
//! weights; "cue", "code", "partner", "agreement", "guard" and "forgetting"
//! name parts of an artificial benchmark, not claims about synapses,
//! biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/CUE_SIMILARITY.md`
//! (committed and pushed before any learning run on the hypothesis seeds).

use crate::connectome::Connectome;
use crate::consolidation_guard::{self as cgd, Setting};
use crate::cue_capacity::{self as ccap, CueUse};
use crate::cue_displacement::TaskLog;
use crate::cue_init as ci;
use crate::cue_reach as cr;
use crate::cue_robustness as crob;
use crate::error_gated::{paired, Paired};
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
/// Flip levels: 25 and 100 carry hypotheses, 50 is the null condition.
pub const FLIPS: [u32; 3] = [25, 50, 100];
pub const FLIP_SIMILAR: u32 = 25;
pub const FLIP_NULL: u32 = 50;
pub const FLIP_OPPOSITE: u32 = 100;
/// Hypothesis seeds (never used by an earlier B888-4 cycle).
pub const SEEDS: std::ops::RangeInclusive<u64> = 481..=520;
/// Pilot seeds (agreement, writes, guard counters and reach only).
pub const PILOT_SEEDS: std::ops::RangeInclusive<u64> = 1..=5;
/// Seeded random held-out orders per seed (in addition to the reversal).
pub const RANDOM_ORDERS: usize = 3;
/// Held-out orders per seed: reversed + `RANDOM_ORDERS`.
pub const HELDOUT_ORDERS: usize = 1 + RANDOM_ORDERS;
/// Salt of this cycle's order generator (cycle 21: `..4C21..`).
pub const ORDER_SALT: u64 = 0x0DE5_B888_4C22_0A1D;
/// H4: learning-loss bound (Zg2m vs Zg1).
pub const LEARN_BOUND: f64 = 0.02;
/// H7: null band for Zg2m − Zg2c at flip 50.
pub const NULL_BAND: f64 = 0.01;
/// G2: minimum matched − anti partner agreement at flips 100 and 25.
pub const G2_GAP_OPPOSITE: f64 = 0.20;
pub const G2_GAP_SIMILAR: f64 = 0.05;
/// Same design as cycles 18–21.
pub const DESIGN: Design = crob::DESIGN;

/// Which cue code an arm uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Code {
    /// Uncued setting.
    None,
    /// Private blocks (the #133 cue).
    Own,
    /// Share-2, partner = next task id (= #134 share 2).
    Cyclic,
    /// Share-2, partner along the max-agreement greedy tour.
    Matched,
    /// Share-2, partner along the min-agreement greedy tour.
    Anti,
}

impl Code {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "-",
            Self::Own => "own",
            Self::Cyclic => "cyclic",
            Self::Matched => "matched",
            Self::Anti => "anti",
        }
    }
    fn tag(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Own => "1",
            Self::Cyclic => "2c",
            Self::Matched => "2m",
            Self::Anti => "2a",
        }
    }
    /// Is this a share-2 code?
    #[must_use]
    pub fn shared(self) -> bool {
        matches!(self, Self::Cyclic | Self::Matched | Self::Anti)
    }
}

/// One compared arm: a #132 setting and its cue code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Arm {
    pub setting: Setting,
    pub code: Code,
}

const fn arm(setting: Setting, code: Code) -> Arm {
    Arm { setting, code }
}

pub const S1: Arm = arm(Setting::Sup, Code::None);
pub const R1: Arm = arm(Setting::Rew, Code::None);
pub const E: Arm = arm(Setting::Err, Code::None);
pub const Z1: Arm = arm(Setting::Z, Code::Own);
pub const ZG1: Arm = arm(Setting::Zg, Code::Own);
pub const ZG2C: Arm = arm(Setting::Zg, Code::Cyclic);
pub const ZG2M: Arm = arm(Setting::Zg, Code::Matched);
pub const ZG2A: Arm = arm(Setting::Zg, Code::Anti);
pub const Z2M: Arm = arm(Setting::Z, Code::Matched);
/// The pre-registered arms, in CSV order (E first in each order: its write
/// curves set every other arm's `paced100` caps).
pub const ARMS: [Arm; 9] = [S1, R1, E, Z1, ZG1, ZG2C, ZG2M, ZG2A, Z2M];

impl Arm {
    #[must_use]
    pub fn name(self) -> String {
        if self.code == Code::None {
            self.setting.name().to_string()
        } else {
            format!("{}/{}", self.setting.name(), self.code.name())
        }
    }
    #[must_use]
    pub fn short(self) -> String {
        match self.setting {
            Setting::Sup => "S1".to_string(),
            Setting::Rew => "R1".to_string(),
            Setting::Err => "E".to_string(),
            Setting::Z => format!("Z{}", self.code.tag()),
            Setting::Zg => format!("Zg{}", self.code.tag()),
            _ => self.setting.name().to_string(),
        }
    }
}

/// Task agreement matrix: `a[i][j]` = fraction of usable edges leaving
/// pattern-pool lines on which tasks i and j have the same valence.
pub fn agreement(fx: &Fixture) -> io::Result<Vec<Vec<f64>>> {
    let g = &fx.graph;
    let v = &fx.task.valence;
    let k = v.len();
    let edges: Vec<usize> = fx
        .task
        .pool
        .iter()
        .filter(|&&s| s + 1 < g.offsets.len())
        .flat_map(|&s| (g.offsets[s] as usize)..(g.offsets[s + 1] as usize))
        .filter(|&e| e < g.counts.len() && g.counts[e] != 0)
        .collect();
    if k == 0 || edges.is_empty() || v.iter().any(|x| x.len() != g.edges()) {
        return Err(bad("no usable pool edges or valence of the wrong size"));
    }
    let n = edges.len() as f64;
    Ok((0..k)
        .map(|i| {
            (0..k)
                .map(|j| edges.iter().filter(|&&e| v[i][e] == v[j][e]).count() as f64 / n)
                .collect()
        })
        .collect())
}

/// Greedy tour from task 0: next = the unvisited task with the highest
/// (`matched`) or lowest agreement with the current one, ties to the
/// lowest id.
#[must_use]
pub fn tour(a: &[Vec<f64>], matched: bool) -> Vec<usize> {
    let k = a.len();
    if k == 0 {
        return Vec::new();
    }
    let mut seen = vec![false; k];
    let mut t = vec![0];
    seen[0] = true;
    while t.len() < k {
        let cur = t[t.len() - 1];
        let mut best: Option<usize> = None;
        for j in (0..k).filter(|&j| !seen[j]) {
            best = match best {
                None => Some(j),
                Some(b) => {
                    let better = if matched {
                        a[cur][j] > a[cur][b]
                    } else {
                        a[cur][j] < a[cur][b]
                    };
                    Some(if better { j } else { b })
                }
            };
        }
        if let Some(b) = best {
            seen[b] = true;
            t.push(b);
        }
    }
    t
}

/// Partner map of a tour: `σ(tour[p]) = tour[(p + 1) mod k]`.
#[must_use]
pub fn partners(tour: &[usize]) -> Vec<usize> {
    let k = tour.len();
    let mut s = vec![0; k];
    for p in 0..k {
        s[tour[p]] = tour[(p + 1) % k];
    }
    s
}

/// Partner map of a code (identity map for `Own`; empty for `None`).
pub fn partner_map(code: Code, a: &[Vec<f64>]) -> Vec<usize> {
    let k = a.len();
    match code {
        Code::None => Vec::new(),
        Code::Own => (0..k).collect(),
        Code::Cyclic => (0..k).map(|j| (j + 1) % k).collect(),
        Code::Matched => partners(&tour(a, true)),
        Code::Anti => partners(&tour(a, false)),
    }
}

/// Codes from blocks and a partner map: code j = sorted(B_j ∪ B_σ(j)).
pub fn codes_from(blocks: &[Vec<usize>], sigma: &[usize]) -> io::Result<Vec<Vec<usize>>> {
    if sigma.len() != blocks.len() || sigma.iter().any(|&s| s >= blocks.len()) {
        return Err(bad("partner map does not match the blocks"));
    }
    Ok((0..blocks.len())
        .map(|j| {
            let mut c: Vec<usize> = blocks[j].iter().chain(&blocks[sigma[j]]).copied().collect();
            c.sort_unstable();
            c.dedup();
            c
        })
        .collect())
}

/// Is `sigma` one cycle through all k tasks (k ≥ 3)?
#[must_use]
pub fn single_cycle(sigma: &[usize]) -> bool {
    let k = sigma.len();
    if k < 3 || sigma.iter().any(|&s| s >= k) {
        return false;
    }
    let (mut x, mut n) = (0, 0);
    loop {
        x = sigma[x];
        n += 1;
        if x == 0 || n > k {
            break;
        }
    }
    x == 0 && n == k
}

/// Codes valid against `blocks` for `code` with partner map `sigma`:
/// `Own` = the blocks; share-2 = σ one k-cycle, code j = B_j ∪ B_σ(j), every
/// line in exactly 2 codes, codes distinct (the #134 share-2 checks), and
/// the cyclic code equals the #134 share-2 code.
#[must_use]
pub fn codes_valid(
    blocks: &[Vec<usize>],
    codes: &[Vec<usize>],
    code: Code,
    sigma: &[usize],
) -> bool {
    match code {
        Code::None => true,
        Code::Own => ccap::codes_valid(blocks, codes, 1),
        _ => {
            let k = blocks.len();
            let lines: Vec<usize> = blocks.iter().flatten().copied().collect();
            let mut uniq = lines.clone();
            uniq.sort_unstable();
            uniq.dedup();
            let cyc = (0..k).all(|j| sigma.get(j) == Some(&((j + 1) % k)));
            single_cycle(sigma)
                && uniq.len() == lines.len()
                && codes_from(blocks, sigma).is_ok_and(|c| c == codes)
                && lines
                    .iter()
                    .all(|l| codes.iter().filter(|c| c.contains(l)).count() == 2)
                && (0..k).all(|i| (0..i).all(|j| codes[i] != codes[j]))
                && (code != Code::Cyclic
                    || (cyc && ccap::share_codes(blocks, 2).is_ok_and(|c| c == codes)))
        }
    }
}

/// Mean partner agreement mean_j a[j][σ(j)] and conflicting links (a < 0.5).
#[must_use]
pub fn partner_agreement(a: &[Vec<f64>], sigma: &[usize]) -> (f64, usize) {
    if sigma.is_empty() {
        return (0.0, 0);
    }
    let v: Vec<f64> = sigma.iter().enumerate().map(|(j, &s)| a[j][s]).collect();
    (
        v.iter().sum::<f64>() / v.len() as f64,
        v.iter().filter(|&&x| x < 0.5).count(),
    )
}

/// The fixture of one code (graph, pool, tasks, splits unchanged; only the
/// per-task cue lists differ), its partner map and validity flag.
pub fn code_fixture(
    fx: &Fixture,
    code: Code,
    a: &[Vec<f64>],
) -> io::Result<(Fixture, Vec<usize>, bool)> {
    let sigma = partner_map(code, a);
    let mut out = fx.clone();
    if code.shared() {
        out.cues = codes_from(&fx.cues, &sigma)?;
    }
    let ok = codes_valid(&fx.cues, &out.cues, code, &sigma);
    Ok((out, sigma, ok))
}

/// One measurement: the #132 row, the code, cue use and partner agreement.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub inner: cgd::Row,
    pub code: Code,
    pub cue: CueUse,
    pub codes_ok: bool,
    /// Mean agreement of each task with its partner (1 for `Own`, 0 uncued).
    pub agree: f64,
    /// Partner links with agreement < 0.5.
    pub conflicts: usize,
    /// Partner map (empty for uncued arms).
    pub sigma: Vec<usize>,
}

impl Row {
    #[must_use]
    pub fn arm(&self) -> Arm {
        arm(self.inner.setting, self.code)
    }
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

/// Task orders of one seed (the #133 generator with this cycle's salt):
/// 0 = identity, 1 = reversed, then distinct seeded permutations.
pub fn orders(seed: u64, k: usize, n_random: usize) -> io::Result<Vec<Vec<usize>>> {
    orders_salted(seed, k, n_random, ORDER_SALT)
}

fn orders_salted(seed: u64, k: usize, n_random: usize, salt: u64) -> io::Result<Vec<Vec<usize>>> {
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

type CodeFx = (Code, Fixture, Vec<usize>, bool);

fn code_fixtures(fx: &Fixture, a: &[Vec<f64>]) -> io::Result<Vec<CodeFx>> {
    [
        Code::None,
        Code::Own,
        Code::Cyclic,
        Code::Matched,
        Code::Anti,
    ]
    .into_iter()
    .map(|c| {
        let (f, s, ok) = code_fixture(fx, c, a)?;
        Ok((c, f, s, ok))
    })
    .collect()
}

/// All rows of one (k, flip, seed): per order, uncapped E first (its
/// per-stage write curves set the `paced100` caps of every other arm), then
/// the other arms on their code fixtures.
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
    let a = agreement(&fx)?;
    let fxs = code_fixtures(&fx, &a)?;
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
        for x in ARMS {
            let (_, f, sigma, ok) = fxs
                .iter()
                .find(|(c, ..)| *c == x.code)
                .ok_or_else(|| bad("missing code fixture"))?;
            let run = match x.setting {
                Setting::Err => e.take().ok_or_else(|| bad("error-gated run missing"))?,
                s => cgd::run_setting(f, &spec, seed, s, &order, &paced)?,
            };
            let s = x.setting;
            let init = ci::initial_circuit(f, &spec, s.base().arm())?;
            let cue = ccap::cue_use(f, &run.weights, s.cued())?;
            let (agree, conflicts) = partner_agreement(&a, sigma);
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
                code: x.code,
                cue,
                codes_ok: *ok,
                agree,
                conflicts,
                sigma: sigma.clone(),
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

/// Paired per-seed difference of a per-(k, flip, seed) row quantity
/// (e.g. partner agreement) between two arms, identity order.
#[must_use]
pub fn row_diff(
    rows: &[Row],
    k: usize,
    flip: u32,
    x: Arm,
    base: Arm,
    f: &dyn Fn(&Row) -> f64,
) -> Vec<f64> {
    let q = Sel {
        k,
        flip,
        orders: Orders::Identity,
    };
    diff(rows, q, x, base, f)
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

/// Codes consistent: uncued arms carry no code and no lines; cued arms have
/// a pre-registered code, 1 × (own) or 2 × (shared) the private lines of the
/// same (k, flip, seed), the same cue-line edges, and a partner map that is
/// identical across orders and arms of the same code.
#[must_use]
pub fn codes_consistent(rows: &[Row]) -> bool {
    rows.iter().all(|r| {
        if !r.inner.setting.cued() {
            return r.code == Code::None && r.cue.lines_per_task == 0 && r.sigma.is_empty();
        }
        let same = |y: &&Row| {
            y.inner.k == r.inner.k && y.inner.flip == r.inner.flip && y.inner.seed == r.inner.seed
        };
        let base = rows.iter().filter(same).find(|y| y.code == Code::Own);
        let mult = if r.code.shared() { 2 } else { 1 };
        r.code != Code::None
            && base.is_some_and(|b| r.cue.lines_per_task == mult * b.cue.lines_per_task)
            && rows
                .iter()
                .filter(same)
                .all(|y| !y.inner.setting.cued() || y.cue.edges == r.cue.edges)
            && rows
                .iter()
                .filter(same)
                .filter(|y| y.code == r.code)
                .all(|y| y.sigma == r.sigma && y.agree == r.agree)
    })
}

/// Design gates: (G1 integrity, G2 manipulation, G3 CN pairs).
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
        && codes_consistent(rows);
    let ks = ks_of(rows);
    let flips = flips_of(rows);
    let g2 = !ks.is_empty()
        && ks.iter().all(|&k| {
            flips.iter().all(|&flip| {
                let q = Sel {
                    k,
                    flip,
                    orders: Orders::Heldout,
                };
                let deferral = frac(rows, q, ZG2M, &|r: &Row| r.inner.deferred()) >= 0.5;
                let gap = match flip {
                    FLIP_OPPOSITE => Some(G2_GAP_OPPOSITE),
                    FLIP_SIMILAR => Some(G2_GAP_SIMILAR),
                    _ => None,
                };
                let agree_ok = k < K_MAIN
                    || gap.is_none_or(|g| {
                        let ma = row_diff(rows, k, flip, ZG2M, ZG2A, &|r| r.agree);
                        let mc = row_diff(rows, k, flip, ZG2M, ZG2C, &|r| r.agree);
                        !ma.is_empty() && mean(&ma) >= g && mc.iter().all(|&d| d >= 0.0)
                    });
                deferral && agree_ok
            })
        });
    let g3 = ks
        .last()
        .is_some_and(|&k| median_cn(rows, k, FLIP_NULL) >= ns::MIN_NOVEL_PAIRS);
    (g1, g2, g3)
}

/// Number of pre-registered hypotheses.
pub const N_HYP: usize = 11;

/// Pre-registered readings: (gates, [H1..H11]); `None` if a gate fails or
/// the needed (k, flip) is missing. All hypotheses use held-out orders.
#[must_use]
pub fn readings(rows: &[Row]) -> ((bool, bool, bool), [Option<bool>; N_HYP]) {
    let g = gates(rows);
    let mut h = [None; N_HYP];
    if !(g.0 && g.1 && g.2) {
        return (g, h);
    }
    let has = |k: usize, flip: u32| {
        rows.iter()
            .any(|r| r.inner.k == k && r.inner.flip == flip && r.inner.order_ix > 0)
    };
    let all = Slice::All;
    let c = |k: usize, flip: u32, x, b| {
        compare(
            rows,
            Sel {
                k,
                flip,
                orders: Orders::Heldout,
            },
            k,
            all,
            x,
            b,
        )
    };
    let (km, fo, fs, fnull) = (K_MAIN, FLIP_OPPOSITE, FLIP_SIMILAR, FLIP_NULL);
    if has(km, fo) {
        h[0] = Some(positive(&c(km, fo, ZG2M, ZG2C).acc));
        h[1] = Some(c(km, fo, ZG2M, S1).passes());
        h[2] = Some(positive(&c(km, fo, ZG2M, ZG1).acc));
        let d = c(km, fo, ZG2M, ZG1);
        h[3] = Some(positive(&d.forget) && d.learn.n > 0 && d.learn.ci_lo > -LEARN_BOUND);
        h[9] = Some(positive(&worst_order(rows, km, fo, ZG2M, ZG2C)));
        h[10] = Some(positive(&c(km, fo, Z2M, Z1).acc));
    }
    if has(km, fs) {
        h[4] = Some(positive(&c(km, fs, ZG2M, ZG2A).acc));
        h[5] = Some(positive(&c(km, fs, ZG2M, ZG1).acc));
    }
    if has(km, fnull) {
        let d = c(km, fnull, ZG2M, ZG2C).acc;
        h[6] = Some(d.n > 0 && d.ci_lo >= -NULL_BAND && d.ci_hi <= NULL_BAND);
        h[7] = Some(c(km, fnull, ZG2M, S1).passes());
    }
    if has(K_BIG, fo) {
        h[8] = Some(positive(&c(K_BIG, fo, ZG2M, ZG1).acc));
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
        "code",
        "seed",
        "order_ix",
        "order_kind",
        "order",
        "partner_map",
        "partner_agree_mean",
        "conflict_links",
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
            "b888-4-cue-similarity-v1".to_string(),
            graph_label.to_string(),
            k.to_string(),
            r.flip.to_string(),
            r.setting.name().to_string(),
            row.code.name().to_string(),
            r.seed.to_string(),
            r.order_ix.to_string(),
            crob::order_kind(r.order_ix).to_string(),
            dash(&r.order),
            dash(&row.sigma),
            f6(row.agree),
            row.conflicts.to_string(),
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
        "setting_summary: k,flip,orders,arm,acc_all,acc_cn,learn_all,learn_cn,fgt_all,fgt_cn,last_all,charged_writes,effective_writes,cue_writes,refused,overwrites,deferred,net_reversals,consolidated_end,saturated_end,cue_lines_per_task,cue_reach_mean,cue_edges,cue_live_end,cue_full_end,partner_agree_mean,conflict_links"
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
                        "{k},{flip},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.2},{:.1},{:.1},{:.1},{:.3},{:.2}",
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
                        m(&|r| r.agree),
                        m(&|r| r.conflicts as f64),
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
        (ZG2M, S1),
        (ZG2M, ZG1),
        (ZG2M, ZG2C),
        (ZG2M, ZG2A),
        (ZG2C, ZG1),
        (ZG2A, ZG1),
        (ZG2C, ZG2A),
        (ZG1, S1),
        (ZG2C, S1),
        (Z2M, Z1),
        (Z2M, S1),
        (ZG2M, Z2M),
        (ZG2M, E),
        (ZG2M, R1),
        (E, S1),
        (R1, S1),
    ];
    let key = |x: Arm, base: Arm| x == ZG2M && (base == S1 || base == ZG1 || base == ZG2C);
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
                        if matches!(ord, Orders::Only(_)) && (s != a || !key(x, base)) {
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
            for &(x, base) in &[
                (ZG2M, ZG2C),
                (ZG2M, ZG1),
                (ZG2M, S1),
                (ZG2M, ZG2A),
                (ZG1, S1),
            ] {
                let w = worst_order(rows, k, flip, x, base);
                if w.n > 0 {
                    let _ = writeln!(o, "{k},{flip},{},{},{}", x.short(), base.short(), fmt_p(&w));
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
            let ag = |x: Arm| setting_mean(rows, q, x, &|r| r.agree);
            let cf = |x: Arm| setting_mean(rows, q, x, &|r| r.conflicts as f64);
            let ma = row_diff(rows, k, flip, ZG2M, ZG2A, &|r| r.agree);
            let mc = row_diff(rows, k, flip, ZG2M, ZG2C, &|r| r.agree);
            let sm = |x: Arm, f: &dyn Fn(&Row) -> f64| setting_mean(rows, q, x, f);
            let _ = writeln!(
                o,
                "manipulation: k={k} flip={flip} partner_agree(Zg2c,Zg2m,Zg2a)=({:.3},{:.3},{:.3}) conflict_links(Zg2c,Zg2m,Zg2a)=({:.2},{:.2},{:.2}) agree_gap(m-a)={:.3} seeds_matched_ge_cyclic={}/{} deferred_frac(Zg2m)={:.3} reach(Zg1,Zg2c,Zg2m,Zg2a)=({:.2},{:.2},{:.2},{:.2}) live_cue_edges(Zg1,Zg2c,Zg2m,Zg2a)=({:.1},{:.1},{:.1},{:.1}) cue_writes(Zg1,Zg2c,Zg2m,Zg2a)=({:.1},{:.1},{:.1},{:.1}) overwrites(Zg1,Zg2c,Zg2m,Zg2a)=({:.1},{:.1},{:.1},{:.1}) effective_writes(S1,E,Zg1,Zg2c,Zg2m,Zg2a)=({:.1},{:.1},{:.1},{:.1},{:.1},{:.1})",
                ag(ZG2C),
                ag(ZG2M),
                ag(ZG2A),
                cf(ZG2C),
                cf(ZG2M),
                cf(ZG2A),
                mean(&ma),
                mc.iter().filter(|&&d| d >= 0.0).count(),
                mc.len(),
                frac(rows, q, ZG2M, &|r: &Row| r.inner.deferred()),
                sm(ZG1, &|r| r.cue.reach),
                sm(ZG2C, &|r| r.cue.reach),
                sm(ZG2M, &|r| r.cue.reach),
                sm(ZG2A, &|r| r.cue.reach),
                sm(ZG1, &|r| r.cue.live as f64),
                sm(ZG2C, &|r| r.cue.live as f64),
                sm(ZG2M, &|r| r.cue.live as f64),
                sm(ZG2A, &|r| r.cue.live as f64),
                sm(ZG1, &|r| r.inner.cue_writes() as f64),
                sm(ZG2C, &|r| r.inner.cue_writes() as f64),
                sm(ZG2M, &|r| r.inner.cue_writes() as f64),
                sm(ZG2A, &|r| r.inner.cue_writes() as f64),
                sm(ZG1, &|r| og(r.inner.overwrites())),
                sm(ZG2C, &|r| og(r.inner.overwrites())),
                sm(ZG2M, &|r| og(r.inner.overwrites())),
                sm(ZG2A, &|r| og(r.inner.overwrites())),
                sm(S1, &|r| r.inner.effective_writes() as f64),
                sm(E, &|r| r.inner.effective_writes() as f64),
                sm(ZG1, &|r| r.inner.effective_writes() as f64),
                sm(ZG2C, &|r| r.inner.effective_writes() as f64),
                sm(ZG2M, &|r| r.inner.effective_writes() as f64),
                sm(ZG2A, &|r| r.inner.effective_writes() as f64),
            );
        }
    }
    for &k in &ks {
        let _ = writeln!(
            o,
            "median_cn(k={k},flip={FLIP_NULL})={}",
            median_cn(rows, k, FLIP_NULL)
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

/// One pilot census line: (k, flip, arm, seed, partner agreement,
/// conflicting links, mean reach, [charged writes, effective writes, cue
/// writes, refused, overwrites, deferred]).
pub type CensusLine = (usize, u32, Arm, u64, f64, usize, f64, [Option<u64>; 6]);

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

/// Pilot census on the identity order: partner agreement, writes, guard
/// counters and cue reach only. No accuracy is computed and no held-out
/// order is drawn.
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
        let ag = agreement(&fx)?;
        let fxs = code_fixtures(&fx, &ag)?;
        let order: Vec<usize> = (0..k).collect();
        let (e_logs, _) = train_only(&fx, &spec, seed, Setting::Err, &order, &sg::free(k))?;
        let curves: Vec<Vec<u64>> = e_logs.iter().map(|t| t.ledger.cum.clone()).collect();
        let paced = cgd::paced_limits(&curves);
        for x in ARMS {
            let (_, f, sigma, _) = fxs
                .iter()
                .find(|(c, ..)| *c == x.code)
                .ok_or_else(|| bad("missing code fixture"))?;
            let lim = if x.setting == Setting::Err {
                sg::free(k)
            } else {
                paced.clone()
            };
            let (logs, glogs) = train_only(f, &spec, seed, x.setting, &order, &lim)?;
            let sum = |g: &dyn Fn(&TaskLog) -> u64| logs.iter().map(g).sum::<u64>();
            let gsum = |g: &dyn Fn(&cgd::GuardStats) -> u64| -> Option<u64> {
                glogs.iter().map(|y| y.stats.map(|s| g(&s))).sum()
            };
            let charged = sum(&|t| t.ledger.writes);
            let effective = gsum(&|s| s.effective).unwrap_or(charged);
            let (agree, conflicts) = partner_agreement(&ag, sigma);
            let rch = if x.code == Code::None {
                0.0
            } else {
                ccap::mean_reach(f)
            };
            out.push((
                k,
                flip,
                x,
                seed,
                agree,
                conflicts,
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

//! B888-4 cycle 24: **cue ownership** — does the learning cost of sharing
//! (#136 H7) disappear if each task may only *write* its own block of a
//! shared cue code?
//!
//! Cycle 23 (#136) found that a share-2 cue code whose partner task agrees
//! most (matched) forgets less than private cues but also learns less
//! (LEARN −0.055 at k = 16, flip 100). An exploratory re-analysis of the
//! published #136 CSV tied that cost to where the write budget goes (every
//! share-2 code spends ≈ 53 % of the equal `paced100` budget on cue-line
//! edges vs 33 % for private cues) rather than to interference from a
//! partner's earlier cue weights. This cycle keeps the matched share-2 code
//! (same lines per task, same partner map, same cue drive at evaluation) but
//! freezes the partner's block while a task is trained: its lines stay
//! active and drive the targets, their edges are never written by that task
//! (`consolidation_guard::run_setting_masked`). Owned-write cyclic and
//! random codes and an unguarded owned matched arm are the controls.
//!
//! Engine state for checkpoint/resume is the #132 checkpoint; the mask is a
//! pure function of the fixture and the partner map (re-tested bit-exact
//! mid-stage for the owned arms).
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph with 4-bit
//! weights; "cue", "code", "block", "partner", "ownership", "frozen", "guard"
//! and "forgetting" name parts of an artificial benchmark, not claims about
//! synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/CUE_OWNERSHIP.md`
//! (committed and pushed before any learning run on the hypothesis seeds).

use crate::connectome::Connectome;
use crate::consolidation_guard::{self as cgd, Setting};
use crate::cue_capacity::{self as ccap, CueUse};
use crate::cue_displacement::TaskLog;
use crate::cue_init as ci;
use crate::cue_partner as cp;
use crate::cue_reach as cr;
use crate::cue_robustness as crob;
use crate::cue_similarity as csim;
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
/// Flip levels: 100 and 50 carry the primary hypotheses, 25 is secondary.
pub const FLIPS: [u32; 3] = [25, 50, 100];
pub const FLIP_SIMILAR: u32 = 25;
pub const FLIP_MID: u32 = 50;
pub const FLIP_OPPOSITE: u32 = 100;
/// Hypothesis seeds (never used by an earlier B888-4 cycle).
pub const SEEDS: std::ops::RangeInclusive<u64> = 561..=600;
/// Pilot seeds (agreement, writes, guard counters, frozen skips and reach).
pub const PILOT_SEEDS: std::ops::RangeInclusive<u64> = 1..=5;
/// Seeded random held-out orders per seed (in addition to the reversal).
pub const RANDOM_ORDERS: usize = 3;
/// Held-out orders per seed: reversed + `RANDOM_ORDERS`.
pub const HELDOUT_ORDERS: usize = 1 + RANDOM_ORDERS;
/// Salt of this cycle's order generator (cycle 23: `..4C23..`).
pub const ORDER_SALT: u64 = 0x0DE5_B888_4C24_0A1D;
/// Salt of the random partner cycle (cycle 23: `..4C23_5A77`).
pub const PARTNER_SALT: u64 = 0x0DE5_B888_4C24_5A77;
/// H1 / H13: learning-loss bound (Zgo2m vs Zg1).
pub const LEARN_BOUND: f64 = 0.02;
/// G2 (a): owned cue writes ≤ this fraction of shared-write cue writes.
pub const G2_CUE_RATIO: f64 = 0.80;
/// G2 (b): minimum fraction of held-out Zgo2m rows with frozen skips.
pub const G2_SKIP_FRAC: f64 = 0.5;
/// G2 (c): minimum mean matched − random partner agreement.
pub const G2_RANDOM_OPPOSITE: f64 = 0.20;
pub const G2_RANDOM_MID: f64 = 0.02;
/// Same design as cycles 18–23.
pub const DESIGN: Design = crob::DESIGN;

/// Which cue code (partner map) an arm uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Code {
    /// Uncued setting.
    None,
    /// Private blocks (the #133 cue).
    Own,
    /// Share-2, partner along the #136 best candidate tour.
    Matched,
    /// Share-2, partner = next task id (= #134 share 2).
    Cyclic,
    /// Share-2, partner along a seeded random k-cycle (new salt).
    Random,
}

impl Code {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "-",
            Self::Own => "own",
            Self::Matched => "matched",
            Self::Cyclic => "cyclic",
            Self::Random => "random",
        }
    }
    fn tag(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Own => "1",
            Self::Matched => "2m",
            Self::Cyclic => "2c",
            Self::Random => "2r",
        }
    }
    /// Is this a share-2 code?
    #[must_use]
    pub fn shared(self) -> bool {
        matches!(self, Self::Matched | Self::Cyclic | Self::Random)
    }
}

/// One compared arm: a #132 setting, its cue code and its write mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Arm {
    pub setting: Setting,
    pub code: Code,
    /// Owned write: during task j only the lines of its own block are
    /// writable; the partner block is active but frozen.
    pub owned: bool,
}

const fn arm(setting: Setting, code: Code, owned: bool) -> Arm {
    Arm {
        setting,
        code,
        owned,
    }
}

pub const S1: Arm = arm(Setting::Sup, Code::None, false);
pub const R1: Arm = arm(Setting::Rew, Code::None, false);
pub const E: Arm = arm(Setting::Err, Code::None, false);
pub const Z1: Arm = arm(Setting::Z, Code::Own, false);
pub const ZG1: Arm = arm(Setting::Zg, Code::Own, false);
pub const ZG2M: Arm = arm(Setting::Zg, Code::Matched, false);
pub const ZGO2M: Arm = arm(Setting::Zg, Code::Matched, true);
pub const ZGO2C: Arm = arm(Setting::Zg, Code::Cyclic, true);
pub const ZGO2R: Arm = arm(Setting::Zg, Code::Random, true);
pub const ZO2M: Arm = arm(Setting::Z, Code::Matched, true);

/// The pre-registered arms, in CSV order (E first in each order: its write
/// curves set every other arm's `paced100` caps).
pub const ARMS: [Arm; 10] = [S1, R1, E, Z1, ZG1, ZG2M, ZGO2M, ZGO2C, ZGO2R, ZO2M];

impl Arm {
    #[must_use]
    pub fn name(self) -> String {
        if self.code == Code::None {
            self.setting.name().to_string()
        } else {
            format!(
                "{}/{}{}",
                self.setting.name(),
                self.code.name(),
                if self.owned { "+owned" } else { "" }
            )
        }
    }
    #[must_use]
    pub fn short(self) -> String {
        let o = if self.owned { "o" } else { "" };
        match self.setting {
            Setting::Sup => "S1".to_string(),
            Setting::Rew => "R1".to_string(),
            Setting::Err => "E".to_string(),
            Setting::Z => format!("Z{o}{}", self.code.tag()),
            Setting::Zg => format!("Zg{o}{}", self.code.tag()),
            _ => self.setting.name().to_string(),
        }
    }
    /// Write mode name.
    #[must_use]
    pub fn mode(self) -> &'static str {
        if self.code == Code::None {
            "-"
        } else if self.owned {
            "owned"
        } else {
            "share"
        }
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

/// Seeded random k-cycle (Sattolo's algorithm, the #136 construction with
/// this cycle's salt), as a partner map.
#[must_use]
pub fn random_cycle(seed: u64, k: usize) -> Vec<usize> {
    let mut p: Vec<usize> = (0..k).collect();
    let mut rng = Lcg(seed ^ PARTNER_SALT ^ k as u64);
    for i in (1..k).rev() {
        let j = (rng.next() % i as u64) as usize;
        p.swap(i, j);
    }
    p
}

/// Partner map of a code (identity map for `Own`; empty for `None`).
#[must_use]
pub fn partner_map(code: Code, a: &[Vec<f64>], seed: u64) -> Vec<usize> {
    let k = a.len();
    match code {
        Code::None => Vec::new(),
        Code::Own => (0..k).collect(),
        Code::Cyclic => cp::partner_map(cp::Code::Cyclic, a, seed),
        Code::Matched => cp::partner_map(cp::Code::Matched, a, seed),
        Code::Random => random_cycle(seed, k),
    }
}

/// Frozen lines of each task under owned write: `B_σ(j) \ B_j`, sorted.
#[must_use]
pub fn frozen_lines(blocks: &[Vec<usize>], sigma: &[usize]) -> Vec<Vec<usize>> {
    sigma
        .iter()
        .enumerate()
        .map(|(j, &s)| {
            let mut v: Vec<usize> = blocks[s]
                .iter()
                .copied()
                .filter(|l| !blocks[j].contains(l))
                .collect();
            v.sort_unstable();
            v
        })
        .collect()
}

/// Codes valid against `blocks`: Own = the blocks; share-2 = the #135/#136
/// checks (σ one k-cycle, code j = B_j ∪ B_σ(j), every line in exactly 2
/// codes, codes distinct; cyclic = the #134 share-2 code) with σ the
/// pre-registered partner map; owned masks = `B_σ(j) \ B_j` (empty for
/// share-write and private codes).
#[must_use]
pub fn codes_valid(
    blocks: &[Vec<usize>],
    codes: &[Vec<usize>],
    x: Arm,
    sigma: &[usize],
    frozen: &[Vec<usize>],
    a: &[Vec<f64>],
    seed: u64,
) -> bool {
    let base = match x.code {
        Code::None => return frozen.is_empty() && !x.owned,
        Code::Own => csim::Code::Own,
        Code::Cyclic => csim::Code::Cyclic,
        Code::Matched | Code::Random => csim::Code::Matched,
    };
    let mask_ok = if x.owned {
        x.code.shared()
            && frozen == frozen_lines(blocks, sigma).as_slice()
            && frozen
                .iter()
                .zip(codes)
                .zip(blocks)
                .all(|((f, c), b)| f.iter().all(|l| c.contains(l) && !b.contains(l)))
    } else {
        frozen.is_empty()
    };
    csim::codes_valid(blocks, codes, base, sigma)
        && (x.code == Code::Own || sigma == partner_map(x.code, a, seed).as_slice())
        && mask_ok
}

/// One arm's fixture (graph, pool, tasks, splits unchanged; only the
/// per-task cue lists differ), partner map, frozen lines and validity flag.
pub type ArmFx = (Fixture, Vec<usize>, Vec<Vec<usize>>, bool);

/// Fixture of arm `x`.
pub fn arm_fixture(fx: &Fixture, x: Arm, a: &[Vec<f64>], seed: u64) -> io::Result<ArmFx> {
    let sigma = partner_map(x.code, a, seed);
    let mut out = fx.clone();
    if x.code.shared() {
        out.cues = csim::codes_from(&fx.cues, &sigma)?;
    } else if x.code == Code::Own {
        // Sorted blocks, as the cycle 21 share-1 code (bit-identical rows).
        out.cues = ccap::share_codes(&fx.cues, 1)?;
    }
    let frozen = if x.owned {
        frozen_lines(&fx.cues, &sigma)
    } else {
        Vec::new()
    };
    let ok = codes_valid(&fx.cues, &out.cues, x, &sigma, &frozen, a, seed);
    Ok((out, sigma, frozen, ok))
}

/// One measurement: the #132 row, code, write mode, cue use, partner
/// agreement and the mask record.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub inner: cgd::Row,
    pub code: Code,
    pub owned: bool,
    pub cue: CueUse,
    pub codes_ok: bool,
    /// Mean agreement of each task with its partner (1 for `Own`, 0 uncued).
    pub agree: f64,
    /// Partner links with agreement < 0.5.
    pub conflicts: usize,
    /// Partner map (empty for uncued arms).
    pub sigma: Vec<usize>,
    /// Mean writable cue lines per task (lines per task − frozen lines).
    pub writable: f64,
    /// Edge steps removed by the mask, summed over stages.
    pub frozen_skipped: u64,
    /// Frozen edges moved during a stage in which they were frozen.
    pub frozen_moved: u64,
}

impl Row {
    #[must_use]
    pub fn arm(&self) -> Arm {
        arm(self.inner.setting, self.code, self.owned)
    }
    /// Cue writes as a share of effective writes.
    #[must_use]
    pub fn cue_share(&self) -> f64 {
        let e = self.inner.effective_writes();
        if e == 0 {
            0.0
        } else {
            self.inner.cue_writes() as f64 / e as f64
        }
    }
}

/// Task orders of one seed (the #133 generator with this cycle's salt):
/// 0 = identity, 1 = reversed, then distinct seeded permutations.
pub fn orders(seed: u64, k: usize, n_random: usize) -> io::Result<Vec<Vec<usize>>> {
    let enough = (1..=k as u128)
        .try_fold(1_u128, u128::checked_mul)
        .is_none_or(|f| (n_random as u128) + 2 <= f);
    if k < 3 || !enough {
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

fn arm_fixtures(fx: &Fixture, a: &[Vec<f64>], seed: u64) -> io::Result<Vec<(Arm, ArmFx)>> {
    ARMS.into_iter()
        .map(|x| Ok((x, arm_fixture(fx, x, a, seed)?)))
        .collect()
}

/// All rows of one (k, flip, seed): per order, uncapped E first (its
/// per-stage write curves set the `paced100` caps of every other arm), then
/// the other arms on their code fixtures and masks.
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
    let a = cp::agreement(&fx)?;
    let fxs = arm_fixtures(&fx, &a, seed)?;
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
        for (x, (f, sigma, frozen, ok)) in &fxs {
            let x = *x;
            let s = x.setting;
            let (run, mask) = match s {
                Setting::Err => (
                    e.take().ok_or_else(|| bad("error-gated run missing"))?,
                    cgd::MaskLog::default(),
                ),
                s => cgd::run_setting_masked(f, &spec, seed, s, &order, &paced, frozen)?,
            };
            let init = ci::initial_circuit(f, &spec, s.base().arm())?;
            let cue = ccap::cue_use(f, &run.weights, s.cued())?;
            let (agree, conflicts) = cp::partner_agreement(&a, sigma);
            let writable = if s.cued() && k > 0 {
                cue.lines_per_task as f64
                    - frozen.iter().map(Vec::len).sum::<usize>() as f64 / k as f64
            } else {
                0.0
            };
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
                owned: x.owned,
                cue,
                codes_ok: *ok,
                agree,
                conflicts,
                sigma: sigma.clone(),
                writable,
                frozen_skipped: mask.skipped.iter().sum(),
                frozen_moved: mask.moved.iter().sum(),
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

/// Per-stage LEARN difference x − base (all pairs, held-out orders) split
/// by stage class: (first stage, partner trained before, partner not yet
/// trained); per seed the mean over its rows and stages of that class.
#[must_use]
pub fn stage_learn(rows: &[Row], k: usize, flip: u32, x: Arm, base: Arm) -> [Paired; 3] {
    let q = Sel {
        k,
        flip,
        orders: Orders::Heldout,
    };
    let mut per: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for s in seeds_of(rows, q) {
        let mut acc: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for r in rows_of(rows, q, x).filter(|r| r.inner.seed == s) {
            let Some(b) = rows_of(rows, q, base)
                .find(|b| b.inner.seed == s && b.inner.order_ix == r.inner.order_ix)
            else {
                continue;
            };
            let order = &r.inner.order;
            let pos: Vec<usize> = {
                let mut p = vec![0; k];
                for (i, &t) in order.iter().enumerate() {
                    p[t] = i;
                }
                p
            };
            for (i, &t) in order.iter().enumerate() {
                let d = r.inner.r(Slice::All, i, i) - b.inner.r(Slice::All, i, i);
                let class = if i == 0 {
                    0
                } else if r.sigma.get(t).is_some_and(|&p| pos[p] < i) {
                    1
                } else {
                    2
                };
                acc[class].push(d);
            }
        }
        for (c, v) in acc.iter().enumerate() {
            if !v.is_empty() {
                per[c].push(mean(v));
            }
        }
    }
    [paired(&per[0]), paired(&per[1]), paired(&per[2])]
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

/// Fraction of rows of x with `f(row) > 0`.
#[must_use]
pub fn frac(rows: &[Row], q: Sel, x: Arm, f: &dyn Fn(&Row) -> u64) -> f64 {
    let v: Vec<&Row> = rows_of(rows, q, x).collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().filter(|r| f(r) > 0).count() as f64 / v.len() as f64
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

/// Codes consistent: uncued arms carry no code, no lines and no mask; cued
/// arms have 1 × (own) or 2 × (share-2) the private lines of the same (k,
/// flip, seed) and the same cue-line edges; arms with the same code share
/// the partner map; Zgo2m / Zo2m have the cue lists (lines, partner map) of
/// Zg2m; frozen moves are 0 everywhere and frozen skips are 0 in non-owned
/// rows.
#[must_use]
pub fn codes_consistent(rows: &[Row]) -> bool {
    rows.iter().all(|r| {
        if r.frozen_moved != 0 || (!r.owned && r.frozen_skipped != 0) {
            return false;
        }
        if !r.inner.setting.cued() {
            return r.code == Code::None
                && !r.owned
                && r.cue.lines_per_task == 0
                && r.sigma.is_empty();
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
                .all(|y| y.sigma == r.sigma && y.agree == r.agree && y.cue.reach == r.cue.reach)
            && (!r.owned || (r.writable - (r.cue.lines_per_task as f64) / 2.0).abs() < 1e-9)
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
    let main: Vec<usize> = ks.iter().copied().filter(|&k| k >= K_MAIN).collect();
    let g2 = !main.is_empty()
        && main.iter().all(|&k| {
            flips
                .iter()
                .filter(|&&f| f == FLIP_MID || f == FLIP_OPPOSITE)
                .all(|&flip| {
                    let q = Sel {
                        k,
                        flip,
                        orders: Orders::Heldout,
                    };
                    let cw = |x: Arm| setting_mean(rows, q, x, &|r| r.inner.cue_writes() as f64);
                    let cue_ok = cw(ZG2M) > 0.0 && cw(ZGO2M) <= G2_CUE_RATIO * cw(ZG2M);
                    let skip_ok = frac(rows, q, ZGO2M, &|r: &Row| r.frozen_skipped) >= G2_SKIP_FRAC;
                    let agree = |x: Arm, b: Arm| row_diff(rows, k, flip, x, b, &|r| r.agree);
                    let mc = agree(ZGO2M, ZGO2C);
                    let mr = agree(ZGO2M, ZGO2R);
                    let gr = if flip == FLIP_OPPOSITE {
                        G2_RANDOM_OPPOSITE
                    } else {
                        G2_RANDOM_MID
                    };
                    let agree_ok = !mc.is_empty()
                        && mc.iter().all(|&d| d >= 0.0)
                        && !mr.is_empty()
                        && mean(&mr) >= gr;
                    cue_ok && skip_ok && agree_ok
                })
        });
    let g3 = ks
        .last()
        .is_some_and(|&k| median_cn(rows, k, FLIP_MID) >= ns::MIN_NOVEL_PAIRS);
    (g1, g2, g3)
}

/// Number of pre-registered hypotheses.
pub const N_HYP: usize = 13;

fn traded(d: &cr::Comparison) -> bool {
    positive(&d.forget) && d.learn.n > 0 && d.learn.ci_lo > -LEARN_BOUND
}

/// Pre-registered readings: (gates, [H1..H13]); `None` if a gate fails or
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
    let c = |k: usize, flip: u32, x, b| {
        compare(
            rows,
            Sel {
                k,
                flip,
                orders: Orders::Heldout,
            },
            k,
            Slice::All,
            x,
            b,
        )
    };
    let (km, fo, fm) = (K_MAIN, FLIP_OPPOSITE, FLIP_MID);
    if has(km, fo) {
        h[0] = Some(traded(&c(km, fo, ZGO2M, ZG1)));
        let d = c(km, fo, ZGO2M, ZG2M);
        h[1] = Some(positive(&d.learn));
        h[2] = Some(positive(&d.acc));
        h[3] = Some(c(km, fo, ZGO2M, S1).passes());
        h[6] = Some(positive(&c(km, fo, ZGO2M, ZG1).acc));
        h[8] = Some(positive(&c(km, fo, ZGO2M, ZGO2C).acc));
        h[9] = Some(positive(&c(km, fo, ZGO2M, ZGO2R).acc));
        h[11] = Some(positive(&c(km, fo, ZO2M, Z1).acc));
    }
    if has(km, fm) {
        h[4] = Some(positive(&c(km, fm, ZGO2M, ZG2M).acc));
        h[5] = Some(c(km, fm, ZGO2M, S1).passes());
        h[7] = Some(positive(&c(km, fm, ZGO2M, ZG1).acc));
        h[12] = Some(traded(&c(km, fm, ZGO2M, ZG1)));
    }
    if has(K_BIG, fo) {
        h[10] = Some(positive(&c(K_BIG, fo, ZGO2M, ZG2M).acc));
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
        "write_mode",
        "seed",
        "order_ix",
        "order_kind",
        "order",
        "partner_map",
        "partner_agree_mean",
        "conflict_links",
        "cue_lines_per_task",
        "writable_lines_per_task",
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
        "cue_write_share",
        "frozen_skipped",
        "frozen_moved",
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
            "b888-4-cue-ownership-v1".to_string(),
            graph_label.to_string(),
            k.to_string(),
            r.flip.to_string(),
            r.setting.name().to_string(),
            row.code.name().to_string(),
            row.arm().mode().to_string(),
            r.seed.to_string(),
            r.order_ix.to_string(),
            crob::order_kind(r.order_ix).to_string(),
            dash(&r.order),
            dash(&row.sigma),
            f6(row.agree),
            row.conflicts.to_string(),
            row.cue.lines_per_task.to_string(),
            format!("{:.3}", row.writable),
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
            f6(row.cue_share()),
            row.frozen_skipped.to_string(),
            row.frozen_moved.to_string(),
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
        "setting_summary: k,flip,orders,arm,acc_all,acc_cn,learn_all,learn_cn,fgt_all,fgt_cn,last_all,charged_writes,effective_writes,cue_writes,cue_write_share,frozen_skipped,refused,overwrites,deferred,net_reversals,consolidated_end,saturated_end,cue_lines_per_task,writable_lines_per_task,cue_reach_mean,cue_edges,cue_live_end,cue_full_end,partner_agree_mean,conflict_links"
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
                        "{k},{flip},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{:.1},{:.1},{:.3},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.2},{:.2},{:.1},{:.1},{:.1},{:.3},{:.2}",
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
                        m(&Row::cue_share),
                        m(&|r| r.frozen_skipped as f64),
                        m(&|r| r.inner.refused() as f64),
                        m(&|r| og(r.inner.overwrites())),
                        m(&|r| og(r.inner.deferred())),
                        m(&|r| r.inner.net_reversals() as f64),
                        m(&|r| r.inner.consolidated_end() as f64),
                        m(&|r| r.inner.saturated_end as f64),
                        m(&|r| r.cue.lines_per_task as f64),
                        m(&|r| r.writable),
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
        (ZGO2M, S1),
        (ZGO2M, ZG1),
        (ZGO2M, ZG2M),
        (ZGO2M, ZGO2C),
        (ZGO2M, ZGO2R),
        (ZG2M, ZG1),
        (ZG2M, S1),
        (ZGO2C, ZG1),
        (ZGO2R, ZG1),
        (ZGO2R, ZGO2C),
        (ZGO2C, S1),
        (ZGO2R, S1),
        (ZG1, S1),
        (ZO2M, Z1),
        (ZO2M, S1),
        (ZGO2M, ZO2M),
        (ZGO2M, E),
        (ZGO2M, R1),
        (Z1, S1),
        (E, S1),
        (R1, S1),
    ];
    let key = |x: Arm, base: Arm| {
        x == ZGO2M && (base == S1 || base == ZG1 || base == ZG2M || base == ZGO2C)
    };
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
                (ZGO2M, ZG2M),
                (ZGO2M, ZG1),
                (ZGO2M, S1),
                (ZGO2M, ZGO2C),
                (ZG2M, ZG1),
                (ZG1, S1),
            ] {
                let w = worst_order(rows, k, flip, x, base);
                if w.n > 0 {
                    let _ = writeln!(o, "{k},{flip},{},{},{}", x.short(), base.short(), fmt_p(&w));
                }
            }
        }
    }
    let _ = writeln!(
        o,
        "stage_learn(learn_all per stage, held-out): k,flip,x,base,first_stage,partner_trained_before,partner_not_yet_trained"
    );
    for &k in &ks {
        for &flip in &flips {
            for &(x, base) in &[(ZGO2M, ZG1), (ZG2M, ZG1), (ZGO2M, ZG2M)] {
                let [f0, f1, f2] = stage_learn(rows, k, flip, x, base);
                if f0.n > 0 {
                    let _ = writeln!(
                        o,
                        "{k},{flip},{},{},{},{},{}",
                        x.short(),
                        base.short(),
                        fmt_p(&f0),
                        fmt_p(&f1),
                        fmt_p(&f2)
                    );
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
            let rd = |x: Arm, b: Arm| row_diff(rows, k, flip, x, b, &|r| r.agree);
            let (mc, mr) = (rd(ZGO2M, ZGO2C), rd(ZGO2M, ZGO2R));
            let sm = |x: Arm, f: &dyn Fn(&Row) -> f64| setting_mean(rows, q, x, f);
            let six = [ZG1, ZG2M, ZGO2M, ZGO2C, ZGO2R, ZO2M];
            let list = |f: &dyn Fn(&Row) -> f64, p: usize| {
                six.iter()
                    .map(|&x| format!("{:.*}", p, sm(x, f)))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let cw = |x: Arm| sm(x, &|r| r.inner.cue_writes() as f64);
            let _ = writeln!(
                o,
                "manipulation: k={k} flip={flip} partner_agree(Zgo2c,Zgo2m,Zgo2r)=({:.3},{:.3},{:.3}) agree_gap(m-r)={:.3} seeds_matched_ge_cyclic={}/{} cue_writes_ratio(Zgo2m/Zg2m)={:.3} skip_frac(Zgo2m)={:.3} cue_writes(Zg1,Zg2m,Zgo2m,Zgo2c,Zgo2r,Zo2m)=({}) cue_share(...)=({}) frozen_skipped(...)=({}) deferred(...)=({}) overwrites(...)=({}) live_cue_edges(...)=({}) effective_writes(S1,E,...)=({:.1},{:.1},{})",
                sm(ZGO2C, &|r| r.agree),
                sm(ZGO2M, &|r| r.agree),
                sm(ZGO2R, &|r| r.agree),
                mean(&mr),
                mc.iter().filter(|&&d| d >= 0.0).count(),
                mc.len(),
                if cw(ZG2M) > 0.0 { cw(ZGO2M) / cw(ZG2M) } else { f64::NAN },
                frac(rows, q, ZGO2M, &|r: &Row| r.frozen_skipped),
                list(&|r| r.inner.cue_writes() as f64, 1),
                list(&Row::cue_share, 3),
                list(&|r| r.frozen_skipped as f64, 1),
                list(&|r| og(r.inner.deferred()), 1),
                list(&|r| og(r.inner.overwrites()), 1),
                list(&|r| r.cue.live as f64, 1),
                sm(S1, &|r| r.inner.effective_writes() as f64),
                sm(E, &|r| r.inner.effective_writes() as f64),
                list(&|r| r.inner.effective_writes() as f64, 1),
            );
        }
    }
    for &k in &ks {
        let _ = writeln!(
            o,
            "median_cn(k={k},flip={FLIP_MID})={}",
            median_cn(rows, k, FLIP_MID)
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

/// One pilot census line: (k, flip, arm, seed, partner agreement, mean
/// reach, [charged writes, effective writes, cue writes, refused,
/// overwrites, deferred, frozen skips, frozen moves]).
pub type CensusLine = (usize, u32, Arm, u64, f64, f64, [Option<u64>; 8]);

type TrainOnly = (Vec<TaskLog>, Vec<cgd::GuardLog>, u64, u64);

fn train_only(
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    s: Setting,
    order: &[usize],
    lim: &[Limit],
    frozen: &[Vec<usize>],
) -> io::Result<TrainOnly> {
    let mut c = ci::initial_circuit(fx, spec, s.base().arm())?;
    let mut dir = vec![0_i8; fx.graph.edges()];
    let mut stamp = vec![0_u8; fx.graph.edges()];
    let (mut logs, mut glogs) = (Vec::new(), Vec::new());
    let (mut skipped, mut moved) = (0, 0);
    for (i, l) in lim.iter().enumerate() {
        let fz: &[usize] = frozen.get(order[i]).map_or(&[], Vec::as_slice);
        let before = c.weights().to_vec();
        let (inner, gl) = cgd::train_stage_masked(
            &mut c,
            fx,
            spec,
            seed,
            s,
            order,
            i,
            l,
            &mut dir,
            &mut stamp,
            fz,
            &mut skipped,
        )?;
        moved += cgd::moved_on_lines(&fx.graph, fz, &before, c.weights());
        glogs.push(gl);
        logs.push(TaskLog {
            ledger: inner.inner.inner.ledger,
            stats: inner.inner.inner.stats,
            cue_cum: inner.inner.cue_cum,
            ..TaskLog::default()
        });
    }
    Ok((logs, glogs, skipped, moved))
}

/// Pilot census on the identity order: partner agreement, writes, guard
/// counters, frozen skips and cue reach only. No accuracy is computed and no
/// held-out order is drawn.
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
        let ag = cp::agreement(&fx)?;
        let fxs = arm_fixtures(&fx, &ag, seed)?;
        let order: Vec<usize> = (0..k).collect();
        let (e_logs, ..) = train_only(&fx, &spec, seed, Setting::Err, &order, &sg::free(k), &[])?;
        let curves: Vec<Vec<u64>> = e_logs.iter().map(|t| t.ledger.cum.clone()).collect();
        let paced = cgd::paced_limits(&curves);
        for (x, (f, sigma, frozen, _)) in &fxs {
            let x = *x;
            let lim = if x.setting == Setting::Err {
                sg::free(k)
            } else {
                paced.clone()
            };
            let (logs, glogs, skipped, moved) =
                train_only(f, &spec, seed, x.setting, &order, &lim, frozen)?;
            let sum = |g: &dyn Fn(&TaskLog) -> u64| logs.iter().map(g).sum::<u64>();
            let gsum = |g: &dyn Fn(&cgd::GuardStats) -> u64| -> Option<u64> {
                glogs.iter().map(|y| y.stats.map(|s| g(&s))).sum()
            };
            let charged = sum(&|t| t.ledger.writes);
            let effective = gsum(&|s| s.effective).unwrap_or(charged);
            let (agree, _) = cp::partner_agreement(&ag, sigma);
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
                rch,
                [
                    Some(charged),
                    Some(effective),
                    Some(sum(&|t| t.stats.cue_writes)),
                    Some(sum(&|t| t.ledger.refused)),
                    gsum(&|s| s.overwrites),
                    gsum(&|s| s.deferred),
                    Some(skipped),
                    Some(moved),
                ],
            ));
        }
    }
    Ok(out)
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
        std::env::temp_dir().join(format!("v888-cue-ownership-{tag}-{}", std::process::id()))
    }

    fn paced(fx: &Fixture, spec: &GenSpec, seed: u64, order: &[usize]) -> Vec<Limit> {
        let k = fx.task.train.len();
        let e = cgd::run_setting(fx, spec, seed, Setting::Err, order, &sg::free(k)).unwrap();
        let curves: Vec<Vec<u64>> = e.logs.iter().map(|t| t.ledger.cum.clone()).collect();
        cgd::paced_limits(&curves)
    }

    #[test]
    fn frozen_lines_are_the_partner_block_and_tampered_masks_are_invalid() {
        let blocks = vec![vec![10, 11], vec![12, 13], vec![14, 15], vec![16, 17]];
        let sigma = vec![2, 0, 3, 1];
        let f = frozen_lines(&blocks, &sigma);
        assert_eq!(
            f,
            vec![vec![14, 15], vec![10, 11], vec![16, 17], vec![12, 13]]
        );
        let d = small();
        let (k, flip, seed) = (6, 100, 3);
        let fx = sg::fixture(None, &d, k, flip, seed).unwrap();
        let a = cp::agreement(&fx).unwrap();
        for x in ARMS {
            let (f, sigma, frozen, ok) = arm_fixture(&fx, x, &a, seed).unwrap();
            assert!(ok, "{x:?}");
            assert_eq!(f.graph, fx.graph);
            assert_eq!(f.task, fx.task);
            assert_eq!(frozen.is_empty(), !x.owned);
            for (j, fz) in frozen.iter().enumerate() {
                assert_eq!(fz.len(), fx.cues[sigma[j]].len());
                assert!(fz.iter().all(|l| f.cues[j].contains(l)));
                assert!(fz.iter().all(|l| !fx.cues[j].contains(l)));
            }
            if x.owned {
                let mut bad_mask = frozen.clone();
                bad_mask[0].pop();
                assert!(!codes_valid(
                    &fx.cues, &f.cues, x, &sigma, &bad_mask, &a, seed
                ));
                assert!(!codes_valid(&fx.cues, &f.cues, x, &sigma, &[], &a, seed));
            } else if x.code != Code::None {
                let fl = frozen_lines(&fx.cues, &sigma);
                assert!(!codes_valid(&fx.cues, &f.cues, x, &sigma, &fl, &a, seed));
            }
        }
        // The matched owned code has the cue lists and partner map of the
        // #136 matched code; the cyclic one those of #136 cyclic.
        let (m, sm, _, _) = arm_fixture(&fx, ZGO2M, &a, seed).unwrap();
        let (pm, psm, _) = cp::code_fixture(&fx, cp::Code::Matched, &a, seed).unwrap();
        assert_eq!((m.cues, sm), (pm.cues, psm));
        let (c, sc, _, _) = arm_fixture(&fx, ZGO2C, &a, seed).unwrap();
        let (pc, psc, _) = cp::code_fixture(&fx, cp::Code::Cyclic, &a, seed).unwrap();
        assert_eq!((c.cues, sc), (pc.cues, psc));
        // New random salt: a single cycle, seeded, and not the #136 one.
        for kk in [3, 8, 16, 24] {
            let r = random_cycle(561, kk);
            assert!(csim::single_cycle(&r));
            assert_eq!(r, random_cycle(561, kk));
        }
        assert_ne!(random_cycle(561, 16), cp::random_cycle(561, 16));
    }

    #[test]
    fn empty_mask_is_bit_identical_and_masks_need_a_guarded_step() {
        let d = small();
        let (k, flip, seed) = (6, 50, 2);
        let spec = GenSpec {
            flip_pct: flip,
            ..d.spec
        };
        let fx = sg::fixture(None, &d, k, flip, seed).unwrap();
        let a = cp::agreement(&fx).unwrap();
        let (f, _, _, _) = arm_fixture(&fx, ZG2M, &a, seed).unwrap();
        let order: Vec<usize> = (0..k).rev().collect();
        let lim = paced(&fx, &spec, seed, &order);
        for s in [
            Setting::Sup,
            Setting::Rew,
            Setting::Err,
            Setting::Z,
            Setting::Zg,
        ] {
            let x = cgd::run_setting(&f, &spec, seed, s, &order, &lim).unwrap();
            let (y, m) = cgd::run_setting_masked(&f, &spec, seed, s, &order, &lim, &[]).unwrap();
            assert_eq!(x.ev, y.ev, "{s:?}");
            assert_eq!(x.logs, y.logs);
            assert_eq!(x.glogs, y.glogs);
            assert_eq!(x.weights, y.weights);
            assert_eq!(m.skipped, vec![0; k]);
            assert_eq!(m.moved, vec![0; k]);
        }
        let (_, sigma, frozen, _) = arm_fixture(&fx, ZGO2M, &a, seed).unwrap();
        assert!(!sigma.is_empty());
        assert!(
            cgd::run_setting_masked(&f, &spec, seed, Setting::Sup, &order, &lim, &frozen).is_err()
        );
        assert!(
            cgd::run_setting_masked(&f, &spec, seed, Setting::Zg, &order, &lim, &frozen[1..])
                .is_err()
        );
    }

    #[test]
    fn owned_code_equals_private_until_the_partner_is_trained_and_frozen_edges_never_move() {
        let d = small();
        let (k, flip, seed) = (6, 100, 5);
        let spec = GenSpec {
            flip_pct: flip,
            ..d.spec
        };
        let fx = sg::fixture(None, &d, k, flip, seed).unwrap();
        let a = cp::agreement(&fx).unwrap();
        // Identity order + cyclic partner (j → j + 1): every partner is
        // trained after its task except the last task's (task 0), so the
        // owned cyclic code and private cues give identical weights for
        // stages 0..k−2.
        let order: Vec<usize> = (0..k).collect();
        let lim = paced(&fx, &spec, seed, &order);
        let (fo, _, frozen, _) = arm_fixture(&fx, ZGO2C, &a, seed).unwrap();
        let (fp, ..) = arm_fixture(&fx, ZG1, &a, seed).unwrap();
        let s = Setting::Zg;
        let mut co = ci::initial_circuit(&fo, &spec, s.base().arm()).unwrap();
        let mut cpv = ci::initial_circuit(&fp, &spec, s.base().arm()).unwrap();
        assert_eq!(co.weights(), cpv.weights());
        let n = fx.graph.edges();
        let (mut d1, mut s1, mut d2, mut s2) = (vec![0; n], vec![0; n], vec![0; n], vec![0; n]);
        let mut total_skipped = 0;
        for i in 0..k - 1 {
            let mut sk = 0;
            let (x, gx) = cgd::train_stage_masked(
                &mut co,
                &fo,
                &spec,
                seed,
                s,
                &order,
                i,
                &lim[i],
                &mut d1,
                &mut s1,
                &frozen[order[i]],
                &mut sk,
            )
            .unwrap();
            let (y, gy) = cgd::train_stage(
                &mut cpv, &fp, &spec, seed, s, &order, i, &lim[i], &mut d2, &mut s2,
            )
            .unwrap();
            assert_eq!(co.weights(), cpv.weights(), "stage {i}");
            assert_eq!(x.inner.inner.ledger, y.inner.inner.ledger);
            assert_eq!(gx, gy);
            total_skipped += sk;
        }
        // Silent frozen lines are still masked (and counted).
        assert!(total_skipped > 0);
        // Matched owned arm on the reversed order: frozen edges never move
        // in their frozen stage, the mask binds, cue writes drop vs share.
        let order: Vec<usize> = (0..k).rev().collect();
        let lim = paced(&fx, &spec, seed, &order);
        let (fm, _, fzm, _) = arm_fixture(&fx, ZGO2M, &a, seed).unwrap();
        let (run, mask) = cgd::run_setting_masked(&fm, &spec, seed, s, &order, &lim, &fzm).unwrap();
        assert_eq!(mask.moved, vec![0; k]);
        assert!(mask.skipped.iter().sum::<u64>() > 0);
        let share = cgd::run_setting(&fm, &spec, seed, s, &order, &lim).unwrap();
        let cw = |r: &cgd::Run| r.logs.iter().map(|t| t.stats.cue_writes).sum::<u64>();
        assert!(cw(&run) < cw(&share));
        assert!(run.topology_unchanged);
    }

    #[test]
    fn rows_are_the_engine_and_share_arms_are_the_cycle23_rows() {
        let d = small();
        let (k, flip, seed) = (6, 100, 2);
        let rows = run_seed(None, &d, k, flip, seed, 2).unwrap();
        assert_eq!(rows.len(), 4 * ARMS.len());
        let spec = GenSpec {
            flip_pct: flip,
            ..d.spec
        };
        let fx = sg::fixture(None, &d, k, flip, seed).unwrap();
        let a = cp::agreement(&fx).unwrap();
        for r in &rows {
            let order = &r.inner.order;
            let lim = if r.inner.setting == Setting::Err {
                sg::free(k)
            } else {
                paced(&fx, &spec, seed, order)
            };
            let (f, _, frozen, ok) = arm_fixture(&fx, r.arm(), &a, seed).unwrap();
            let (x, m) =
                cgd::run_setting_masked(&f, &spec, seed, r.inner.setting, order, &lim, &frozen)
                    .unwrap();
            assert_eq!(r.inner.ev, x.ev);
            assert_eq!(r.inner.logs, x.logs);
            assert_eq!(r.inner.digest, wa::digest(&x.weights));
            assert_eq!(r.frozen_skipped, m.skipped.iter().sum::<u64>());
            assert_eq!(r.frozen_moved, 0);
            assert!(ok && r.codes_ok);
        }
        // Order salts differ from #136, so compare the share arms with the
        // #136 engine on the same order and code directly: S1, Z1, Zg1 and
        // Zg2m are the #136 rows (reversal, order index 1, same in both).
        let old = cp::run_seed(None, &d, k, flip, seed, 2).unwrap();
        for (mine, theirs) in [(ZG1, cp::ZG1), (S1, cp::S1), (ZG2M, cp::ZG2M), (Z1, cp::Z1)] {
            let x = rows
                .iter()
                .find(|r| r.arm() == mine && r.inner.order_ix == 1)
                .unwrap();
            let y = old
                .iter()
                .find(|r| r.arm() == theirs && r.inner.order_ix == 1)
                .unwrap();
            assert_eq!(x.inner.ev, y.inner.ev);
            assert_eq!(x.inner.digest, y.inner.digest);
        }
        let om = rows.iter().find(|r| r.arm() == ZGO2M).unwrap();
        let sm = rows.iter().find(|r| r.arm() == ZG2M).unwrap();
        assert_eq!(om.sigma, sm.sigma);
        assert_eq!(om.cue.lines_per_task, sm.cue.lines_per_task);
        assert!((om.writable - sm.cue.lines_per_task as f64 / 2.0).abs() < 1e-12);
        assert!(om.frozen_skipped > 0 && sm.frozen_skipped == 0);
    }

    #[test]
    fn checkpoint_resume_is_bit_exact_for_owned_arms() {
        let d = small();
        let (k, flip, seed) = (6, 100, 4);
        let spec = GenSpec {
            flip_pct: flip,
            ..d.spec
        };
        let base = sg::fixture(None, &d, k, flip, seed).unwrap();
        let a = cp::agreement(&base).unwrap();
        for x in [ZGO2M, ZO2M] {
            let (fx, _, frozen, ok) = arm_fixture(&base, x, &a, seed).unwrap();
            assert!(ok);
            let g = &fx.graph;
            let order: Vec<usize> = (0..k).rev().collect();
            let lim = paced(&base, &spec, seed, &order);
            let s = x.setting;
            let gd = s.guard().unwrap();
            let (full, mask) =
                cgd::run_setting_masked(&fx, &spec, seed, s, &order, &lim, &frozen).unwrap();
            let mut c = ci::initial_circuit(&fx, &spec, s.base().arm()).unwrap();
            let mut dir = vec![0_i8; g.edges()];
            let mut stamp = vec![0_u8; g.edges()];
            for (i, &t) in order.iter().enumerate() {
                let n = ts::task_order(&spec, seed, t).len();
                let start = c.weights().to_vec();
                let mut cpt =
                    cgd::Checkpoint::start(&order, i, &lim[i], gd, &dir, &stamp, &start).unwrap();
                let mut sk = 0;
                for end in [n / 4, n / 2, n] {
                    cgd::train_range_masked(
                        &mut c, &fx, &spec, seed, s, &lim[i], &mut cpt, end, &frozen[t], &mut sk,
                    )
                    .unwrap();
                    let (pw, pc) = (tmp("w"), tmp("c"));
                    c.save(&pw).unwrap();
                    cpt.save(&pc).unwrap();
                    c = PlasticCircuit::load(&pw, g).unwrap();
                    cpt = cgd::Checkpoint::load(&pc, g.edges()).unwrap();
                    std::fs::remove_file(&pw).unwrap();
                    std::fs::remove_file(&pc).unwrap();
                }
                assert_eq!(Some(cpt.g), full.glogs[i].stats, "{x:?} stage {i}");
                assert_eq!(cpt.inner.inner.inner.ledger, full.logs[i].ledger);
                assert_eq!(sk, mask.skipped[i]);
                cgd::consolidate(&mut dir, &mut stamp, i, &cpt.start, c.weights()).unwrap();
            }
            assert_eq!(c.weights(), &full.weights[..]);
        }
    }

    #[test]
    fn grid_is_deterministic_thread_independent_and_gates_detect_tampering() {
        let d = small();
        let ks = [6, 16];
        let a = run_grid_threads(None, &d, &ks, &[50, 100], &[1, 2, 3], 2, 1).unwrap();
        let b = run_grid_threads(None, &d, &ks, &[50, 100], &[1, 2, 3], 2, 4).unwrap();
        assert_eq!(a, b);
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        let cols = csv_header().split(',').count();
        assert!(csv.lines().all(|l| l.split(',').count() == cols));
        let rep = report(&a);
        assert_eq!(rep, report(&b));
        assert!(rep.contains("readings: G1_integrity=true"));
        assert!(rep.contains("Zgo2r") && rep.contains("stage_learn"));
        assert!(gates(&a).0);
        let q = Sel {
            k: 6,
            flip: 100,
            orders: Orders::Heldout,
        };
        assert_eq!(compare(&a, q, 6, Slice::All, ZGO2M, ZGO2M).acc.ties, 3);
        let w = worst_order(&a, 6, 100, ZGO2M, S1);
        assert!(w.mean <= compare(&a, q, 6, Slice::All, ZGO2M, S1).acc.mean + 1e-12);
        assert!(run_grid_threads(None, &d, &ks, &[50], &[], 2, 1).is_err());
        // G1 tampering: code flag, lines, partner map, a frozen edge that
        // moved, frozen skips in a non-owned row, a swapped reversal and an
        // over-cap run.
        for arm in [ZGO2M, ZGO2R, ZG2M] {
            let ix = a.iter().position(|r| r.arm() == arm).unwrap();
            let mut t = a.clone();
            t[ix].codes_ok = false;
            assert!(!gates(&t).0);
            let mut t = a.clone();
            t[ix].cue.lines_per_task += 1;
            assert!(!gates(&t).0);
            let mut t = a.clone();
            t[ix].sigma.swap(0, 1);
            assert!(!gates(&t).0);
            let mut t = a.clone();
            t[ix].frozen_moved = 1;
            assert!(!gates(&t).0);
        }
        let mut t = a.clone();
        let j = t.iter().position(|r| r.arm() == ZG2M).unwrap();
        t[j].frozen_skipped = 1;
        assert!(!gates(&t).0);
        let mut t = a.clone();
        let j = t.iter().position(|r| r.inner.order_ix == 1).unwrap();
        t[j].inner.order.swap(0, 1);
        assert!(!gates(&t).0);
        let mut t = a.clone();
        let j = t.iter().position(|r| r.arm() == S1).unwrap();
        t[j].inner.logs[0].ledger.writes = t[j].inner.logs[0].ledger.cap + 1;
        assert!(!gates(&t).0);
        // G2 tampering at k = 16: no frozen skips, cue writes of the owned
        // arm inflated to the share level, matched below cyclic in a seed.
        let g2 = gates(&a).1;
        if g2 {
            let mut t = a.clone();
            for r in t.iter_mut().filter(|r| r.arm() == ZGO2M) {
                r.frozen_skipped = 0;
            }
            assert!(!gates(&t).1);
            let mut t = a.clone();
            for r in t.iter_mut().filter(|r| r.arm() == ZGO2M && r.inner.k == 16) {
                r.inner.logs[0].stats.cue_writes += 10_000;
            }
            assert!(!gates(&t).1);
            let mut t = a.clone();
            for r in t.iter_mut() {
                if r.arm() == ZGO2C && r.inner.seed == 1 && r.inner.flip == 100 {
                    r.agree = 2.0;
                }
            }
            assert!(!gates(&t).1);
        }
    }

    #[test]
    fn readings_are_evaluated_on_the_preregistered_ks_and_flips() {
        let d = small();
        let rows = run_grid_threads(None, &d, &[8, 16], &[25, 50, 100], &[1, 2], 2, 4).unwrap();
        let (g, h) = readings(&rows);
        assert!(g.0);
        if g.0 && g.1 && g.2 {
            assert!(h[..10].iter().all(Option::is_some));
            assert!(h[10].is_none(), "k = 24 absent");
            assert!(h[11..].iter().all(Option::is_some));
        } else {
            assert_eq!(h, [None; N_HYP]);
        }
        // No k ≥ 16: G2 cannot pass, nothing is evaluable.
        let small_rows = run_grid_threads(None, &d, &[6], &[50, 100], &[1], 1, 2).unwrap();
        assert!(!gates(&small_rows).1);
        assert_eq!(readings(&small_rows).1, [None; N_HYP]);
    }

    #[test]
    fn census_counts_writes_and_frozen_skips_without_accuracy() {
        let d = small();
        let lines = census(None, &d, 6, 100, &[1, 2]).unwrap();
        assert_eq!(lines.len(), 2 * ARMS.len());
        for (k, flip, x, _, ag, rch, v) in &lines {
            assert_eq!((*k, *flip), (6, 100));
            assert_eq!(*rch > 0.0, x.code != Code::None);
            assert_eq!(v[4].is_some(), x.setting.guard().is_some());
            assert_eq!(v[6].is_some_and(|s| s > 0), x.owned);
            assert_eq!(v[7], Some(0));
            if x.code == Code::Own {
                assert!((ag - 1.0).abs() < 1e-12);
            }
            if x.code == Code::Cyclic {
                assert!(ag.abs() < 1e-12);
            }
        }
    }
}

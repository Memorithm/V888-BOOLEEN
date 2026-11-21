//! B888-4 cycle 27: **partner-write dose** — does the number of writable
//! partner lines of the matched share-2 cue code (a count, not a time
//! window) rank generalisation and forgetting at equal write budget?
//!
//! Cycles 25 (#184) and 26 (#185) tried to split the #139 shared-vs-owned
//! trade-off in time and both failed their manipulation check G2, so their
//! hypotheses were not evaluable. The cycle 27 diagnosis
//! ([`idle_diagnosis`], `docs/CUE_WRITE_DOSE.md`) shows a common root: the
//! uncapped error-gated write curve that sets every arm's `paced100` budget
//! is extremely front-loaded (half of a stage's writes by episode ≈ 6 of
//! 256), and at flip 100 a stage whose task has the same valence vector as
//! an earlier-trained task can be **idle** (zero uncapped writes, so zero
//! budget for every capped arm). Any within-stage clock is degenerate on
//! those stages (`budget_split` returns `n`), which is exactly G2 (e) of
//! cycle 26. Both cycles' descriptive readings also said that what ranks
//! the arms is how many partner-block writes they make, not when.
//!
//! This cycle drops timing and manipulates the **count** directly: for each
//! task j a nested, seeded subset of its partner lines `F_j = B_σ(j) \ B_j`
//! is writable for the whole stage (dose 0/4 … 4/4 of `|F_j|`; 0 = the #139
//! owned write, 4/4 = the #136 shared write), plus a same-count control on
//! the complementary half of `F_j`. Masks are constant within a stage, so
//! idle stages are idle for every arm alike and no gate depends on a clock.
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph with 4-bit
//! weights; "cue", "code", "block", "partner", "dose", "frozen", "guard",
//! "idle" and "forgetting" name parts of an artificial benchmark, not claims
//! about synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/CUE_WRITE_DOSE.md`
//! (committed and pushed before any learning run on the hypothesis seeds).

use crate::connectome::Connectome;
use crate::consolidation_guard::{self as cgd, PhaseCount, Setting, StageMask};
use crate::cue_capacity::{self as ccap, CueUse};
use crate::cue_displacement::TaskLog;
use crate::cue_init as ci;
use crate::cue_ownership as cown;
use crate::cue_partner as cp;
use crate::cue_reach as cr;
use crate::cue_robustness as crob;
use crate::cue_similarity as csim;
use crate::error_gated::{paired, Paired};
use crate::generalization_bench::GenSpec;
use crate::novel_split::{self as ns, Design, Fixture, Slice};
use crate::seq_generalisation as sg;
use crate::task_sequence as ts;
use crate::write_allocation as wa;
use crate::write_budget::Limit;
use std::io;

fn bad(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Run lengths (each on its own fixture): `|F_j|` = 8 at k = 8 and 4 at
/// k = 16, so every quarter dose is a whole number of lines.
pub const KS: [usize; 2] = [8, 16];
pub const K_SMALL: usize = 8;
pub const K_MAIN: usize = 16;
/// Flip levels: 100 and 50 carry the primary hypotheses, 25 is secondary.
pub const FLIPS: [u32; 3] = [25, 50, 100];
pub const FLIP_SIMILAR: u32 = 25;
pub const FLIP_MID: u32 = 50;
pub const FLIP_OPPOSITE: u32 = 100;
/// Hypothesis seeds (never used by an earlier B888-4 cycle).
pub const SEEDS: std::ops::RangeInclusive<u64> = 681..=720;
/// Pilot seeds (label-free census).
pub const PILOT_SEEDS: std::ops::RangeInclusive<u64> = 1..=5;
/// Seeded random held-out orders per seed (in addition to the reversal).
pub const RANDOM_ORDERS: usize = 3;
/// Held-out orders per seed: reversed + `RANDOM_ORDERS`.
pub const HELDOUT_ORDERS: usize = 1 + RANDOM_ORDERS;
/// Salt of this cycle's order generator (cycle 26: `..4C26..`).
pub const ORDER_SALT: u64 = 0x0DE5_B888_4C27_0A1D;
/// Salt of the per-task writable-line permutation.
pub const DOSE_SALT: u64 = 0x0DE5_B888_4C27_D05E;
/// Number of dose steps (dose d writes `⌊d · |F_j| / DOSE_STEPS⌋` lines).
pub const DOSE_STEPS: u8 = 4;
/// G2 (b), (c): minimum fraction of held-out rows where the manipulation acts.
pub const G2_ACT_FRAC: f64 = 0.5;
/// G2 (d): mean partner moves of the complement arm / the nested half arm
/// must lie in `[1 / G2_RATIO, G2_RATIO]`.
pub const G2_RATIO: f64 = 2.0;
/// H5 equivalence margin on Δ ACC (95 % CI inside ±margin).
pub const EQUIV_MARGIN: f64 = 0.01;
/// Same design as cycles 18–26.
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
}

impl Code {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "-",
            Self::Own => "own",
            Self::Matched => "matched",
        }
    }
    /// Is this a share-2 code?
    #[must_use]
    pub fn shared(self) -> bool {
        self == Self::Matched
    }
}

/// Which partner lines of a share-2 code are writable (whole stage).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dose {
    /// Not a share-2 arm.
    None,
    /// The first `⌊d · |F_j| / 4⌋` lines of the seeded permutation of
    /// `F_j` (nested in d; 0 = owned, 4 = shared).
    Nested(u8),
    /// `F_j` minus the `Nested(2)` lines (same count when `|F_j|` is even).
    Complement,
}

impl Dose {
    #[must_use]
    pub fn name(self) -> String {
        match self {
            Self::None => "-".to_string(),
            Self::Nested(d) => format!("{d}/{DOSE_STEPS}"),
            Self::Complement => format!("{}/{DOSE_STEPS}c", DOSE_STEPS / 2),
        }
    }
    fn tag(self) -> String {
        match self {
            Self::None => String::new(),
            Self::Nested(d) => format!("d{d}"),
            Self::Complement => format!("d{}c", DOSE_STEPS / 2),
        }
    }
    /// Nominal writable fraction of `F_j` (0 for non-share-2 arms).
    #[must_use]
    pub fn fraction(self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Nested(d) => f64::from(d) / f64::from(DOSE_STEPS),
            Self::Complement => 0.5,
        }
    }
}

/// One compared arm: a #132 setting, its cue code and its partner dose.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Arm {
    pub setting: Setting,
    pub code: Code,
    pub dose: Dose,
}

const fn arm(setting: Setting, code: Code, dose: Dose) -> Arm {
    Arm {
        setting,
        code,
        dose,
    }
}

pub const S1: Arm = arm(Setting::Sup, Code::None, Dose::None);
pub const R1: Arm = arm(Setting::Rew, Code::None, Dose::None);
pub const E: Arm = arm(Setting::Err, Code::None, Dose::None);
pub const ZG1: Arm = arm(Setting::Zg, Code::Own, Dose::None);
pub const ZGD0: Arm = arm(Setting::Zg, Code::Matched, Dose::Nested(0));
pub const ZGD1: Arm = arm(Setting::Zg, Code::Matched, Dose::Nested(1));
pub const ZGD2: Arm = arm(Setting::Zg, Code::Matched, Dose::Nested(2));
pub const ZGD3: Arm = arm(Setting::Zg, Code::Matched, Dose::Nested(3));
pub const ZGD4: Arm = arm(Setting::Zg, Code::Matched, Dose::Nested(4));
pub const ZGD2C: Arm = arm(Setting::Zg, Code::Matched, Dose::Complement);

/// The nested dose arms, in dose order (the slope is fitted on these).
pub const DOSES: [Arm; 5] = [ZGD0, ZGD1, ZGD2, ZGD3, ZGD4];

/// The pre-registered arms, in CSV order (E first among the learners: its
/// write curves set every other arm's `paced100` caps).
pub const ARMS: [Arm; 10] = [S1, R1, E, ZG1, ZGD0, ZGD1, ZGD2, ZGD3, ZGD4, ZGD2C];

impl Arm {
    #[must_use]
    pub fn name(self) -> String {
        match self.code {
            Code::None => self.setting.name().to_string(),
            Code::Own => format!("{}/{}", self.setting.name(), self.code.name()),
            Code::Matched => format!(
                "{}/{}+dose{}",
                self.setting.name(),
                self.code.name(),
                self.dose.name()
            ),
        }
    }
    #[must_use]
    pub fn short(self) -> String {
        match (self.setting, self.code) {
            (Setting::Sup, _) => "S1".to_string(),
            (Setting::Rew, _) => "R1".to_string(),
            (Setting::Err, _) => "E".to_string(),
            (Setting::Zg, Code::Own) => "Zg1".to_string(),
            (Setting::Zg, Code::Matched) => format!("Zg{}", self.dose.tag()),
            _ => self.setting.name().to_string(),
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

/// Seeded permutation of the partner lines `f` of task `j` (Fisher–Yates on
/// an LCG keyed by `seed ^ DOSE_SALT` and the task index). Independent of
/// the flip, the order and the arm.
#[must_use]
pub fn line_permutation(seed: u64, j: usize, f: &[usize]) -> Vec<usize> {
    let mut p = f.to_vec();
    let mut rng = Lcg(seed ^ DOSE_SALT ^ (j as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    for i in (1..p.len()).rev() {
        let r = (rng.next() % (i as u64 + 1)) as usize;
        p.swap(i, r);
    }
    p
}

/// Writable partner lines of every task under `dose` (sorted).
#[must_use]
pub fn writable(dose: Dose, f: &[Vec<usize>], seed: u64) -> Vec<Vec<usize>> {
    f.iter()
        .enumerate()
        .map(|(j, l)| {
            let p = line_permutation(seed, j, l);
            let take = |d: u8| (usize::from(d) * l.len()) / usize::from(DOSE_STEPS);
            let mut w: Vec<usize> = match dose {
                Dose::None => Vec::new(),
                Dose::Nested(d) => p[..take(d.min(DOSE_STEPS))].to_vec(),
                Dose::Complement => p[take(DOSE_STEPS / 2)..].to_vec(),
            };
            w.sort_unstable();
            w
        })
        .collect()
}

/// Constant per-stage masks: `F_j` minus its writable lines, frozen in the
/// whole stage.
#[must_use]
pub fn schedule(dose: Dose, f: &[Vec<usize>], seed: u64) -> Vec<StageMask> {
    f.iter()
        .zip(writable(dose, f, seed))
        .map(|(l, w)| {
            let frozen: Vec<usize> = l.iter().copied().filter(|x| !w.contains(x)).collect();
            StageMask::constant(&frozen)
        })
        .collect()
}

/// Codes valid against `blocks`: Own = the blocks; share-2 = the #135 /
/// #136 checks with σ the matched partner map; watched lines = `F_j` and
/// masks = the dose schedule for share-2 codes (both empty otherwise).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn codes_valid(
    blocks: &[Vec<usize>],
    codes: &[Vec<usize>],
    x: Arm,
    sigma: &[usize],
    masks: &[StageMask],
    watch: &[Vec<usize>],
    a: &[Vec<f64>],
    seed: u64,
) -> bool {
    let base = match x.code {
        Code::None => return masks.is_empty() && watch.is_empty() && x.dose == Dose::None,
        Code::Own => csim::Code::Own,
        Code::Matched => csim::Code::Matched,
    };
    let sched_ok = if x.code.shared() {
        let f = cown::frozen_lines(blocks, sigma);
        watch == f.as_slice()
            && x.dose != Dose::None
            && masks == schedule(x.dose, &f, seed).as_slice()
            && f.iter()
                .zip(codes)
                .zip(blocks)
                .all(|((f, c), b)| f.iter().all(|l| c.contains(l) && !b.contains(l)))
    } else {
        masks.is_empty() && watch.is_empty() && x.dose == Dose::None
    };
    csim::codes_valid(blocks, codes, base, sigma)
        && (x.code == Code::Own || sigma == cp::partner_map(cp::Code::Matched, a, seed).as_slice())
        && sched_ok
}

/// One arm's fixture (graph, pool, tasks, splits unchanged; only the
/// per-task cue lists differ), partner map, masks, watched lines, mean
/// writable lines per task and validity flag.
pub type ArmFx = (
    Fixture,
    Vec<usize>,
    Vec<StageMask>,
    Vec<Vec<usize>>,
    f64,
    bool,
);

/// Fixture of arm `x`.
pub fn arm_fixture(fx: &Fixture, x: Arm, a: &[Vec<f64>], seed: u64) -> io::Result<ArmFx> {
    let k = fx.cues.len();
    let mut out = fx.clone();
    let (sigma, masks, watch, w) = match x.code {
        Code::Matched => {
            let sigma = cp::partner_map(cp::Code::Matched, a, seed);
            out.cues = csim::codes_from(&fx.cues, &sigma)?;
            let f = cown::frozen_lines(&fx.cues, &sigma);
            let w = writable(x.dose, &f, seed)
                .iter()
                .map(Vec::len)
                .sum::<usize>() as f64
                / k.max(1) as f64;
            (sigma, schedule(x.dose, &f, seed), f, w)
        }
        Code::Own => {
            // Sorted blocks, as the cycle 21 share-1 code (bit-identical rows).
            out.cues = ccap::share_codes(&fx.cues, 1)?;
            ((0..k).collect(), Vec::new(), Vec::new(), 0.0)
        }
        Code::None => (Vec::new(), Vec::new(), Vec::new(), 0.0),
    };
    let ok = codes_valid(&fx.cues, &out.cues, x, &sigma, &masks, &watch, a, seed);
    Ok((out, sigma, masks, watch, w, ok))
}

/// One measurement: the #132 row, code, dose, cue use, partner agreement
/// and the mask record.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub inner: cgd::Row,
    pub code: Code,
    pub dose: Dose,
    pub cue: CueUse,
    pub codes_ok: bool,
    /// Mean agreement of each task with its partner (1 for `Own`, 0 uncued).
    pub agree: f64,
    /// Partner links with agreement < 0.5.
    pub conflicts: usize,
    /// Partner map (empty for uncued arms).
    pub sigma: Vec<usize>,
    /// Mean partner lines per task (`|B_σ(j) \ B_j|`; 0 if not share-2).
    pub partner_lines: f64,
    /// Mean writable partner lines per task (0 if not share-2).
    pub writable_lines: f64,
    /// Edge steps removed by the mask, summed over stages.
    pub skipped: u64,
    /// Frozen edges whose weight moved while frozen (must be 0).
    pub frozen_moved: u64,
    /// Partner-line (`F_j`) edges whose weight changed during stage j,
    /// summed over stages (share-2 arms only).
    pub partner_moved: u64,
    /// Uncapped error-gated writes per stage of this (k, flip, seed,
    /// order), in training order (the `paced100` stage totals).
    pub e_stage_writes: Vec<u64>,
}

impl Row {
    #[must_use]
    pub fn arm(&self) -> Arm {
        arm(self.inner.setting, self.code, self.dose)
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
    /// Stages with zero uncapped error-gated writes (zero budget for every
    /// capped arm).
    #[must_use]
    pub fn idle_stages(&self) -> usize {
        self.e_stage_writes.iter().filter(|&&w| w == 0).count()
    }
}

/// Task orders of one seed with an explicit salt (the #133 generator):
/// 0 = identity, 1 = reversed, then distinct seeded permutations.
pub fn orders_salted(
    seed: u64,
    k: usize,
    n_random: usize,
    salt: u64,
) -> io::Result<Vec<Vec<usize>>> {
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

/// Task orders of one seed with this cycle's salt.
pub fn orders(seed: u64, k: usize, n_random: usize) -> io::Result<Vec<Vec<usize>>> {
    orders_salted(seed, k, n_random, ORDER_SALT)
}

fn arm_fixtures(fx: &Fixture, a: &[Vec<f64>], seed: u64) -> io::Result<Vec<(Arm, ArmFx)>> {
    ARMS.into_iter()
        .map(|x| Ok((x, arm_fixture(fx, x, a, seed)?)))
        .collect()
}

fn sum_phases(p: &[PhaseCount]) -> (u64, u64, u64) {
    let mut sk = 0;
    let mut fm = 0;
    let mut wm = 0;
    for c in p {
        for h in 0..2 {
            sk += c.skipped[h];
            fm += c.frozen_moved[h];
            wm += c.watch_moved[h];
        }
    }
    (sk, fm, wm)
}

/// All rows of one (k, flip, seed): per order, uncapped E first (its
/// per-stage write curves set the `paced100` caps of every other arm), then
/// the other arms on their code fixtures and constant dose masks.
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
        let stage_writes: Vec<u64> = curves
            .iter()
            .map(|c| c.last().copied().unwrap_or(0))
            .collect();
        let e_paced = cgd::run_setting(&fx, &spec, seed, Setting::Err, &order, &paced)?;
        let identity =
            e_paced.weights == e.weights && e_paced.logs.iter().all(|t| t.ledger.refused == 0);
        let cn_pairs: Vec<u64> = order.iter().map(|&t| sg::pair_counts(&fx, t)[2]).collect();
        let mut e = Some(e);
        for (x, (f, sigma, masks, watch, w, ok)) in &fxs {
            let x = *x;
            let s = x.setting;
            let (run, phases) = match s {
                Setting::Err => (
                    e.take().ok_or_else(|| bad("error-gated run missing"))?,
                    Vec::new(),
                ),
                s => cgd::run_setting_scheduled_at(
                    f,
                    &spec,
                    seed,
                    s,
                    &order,
                    &paced,
                    masks,
                    watch,
                    &[],
                )?,
            };
            let (skipped, frozen_moved, partner_moved) = sum_phases(&phases);
            let init = ci::initial_circuit(f, &spec, s.base().arm())?;
            let cue = ccap::cue_use(f, &run.weights, s.cued())?;
            let (agree, conflicts) = cp::partner_agreement(&a, sigma);
            let partner_lines = if k > 0 {
                watch.iter().map(Vec::len).sum::<usize>() as f64 / k as f64
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
                dose: x.dose,
                cue,
                codes_ok: *ok,
                agree,
                conflicts,
                sigma: sigma.clone(),
                partner_lines,
                writable_lines: *w,
                skipped,
                frozen_moved,
                partner_moved,
                e_stage_writes: stage_writes.clone(),
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

/// Least-squares slope of `y` on the nominal dose fractions of [`DOSES`]
/// (units: change of `y` from dose 0 to the full shared write).
#[must_use]
pub fn dose_slope(y: &[f64; 5]) -> f64 {
    let x: Vec<f64> = DOSES.iter().map(|a| a.dose.fraction()).collect();
    let (mx, my) = (mean(&x), mean(y));
    let num: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let den: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    num / den
}

/// Per-seed values of `f` (mean over the selected orders) for the five
/// nested dose arms.
fn dose_values(rows: &[Row], q: Sel, f: &dyn Fn(&Row) -> f64) -> Vec<[f64; 5]> {
    seeds_of(rows, q)
        .into_iter()
        .filter_map(|s| {
            let mut v = [0.0; 5];
            for (i, &x) in DOSES.iter().enumerate() {
                v[i] = seed_value(rows, q, s, x, f)?;
            }
            Some(v)
        })
        .collect()
}

/// Per-seed dose slope of `f`, paired over seeds (sign × slope, so that a
/// test of "slope < 0" uses `sign = −1`).
#[must_use]
pub fn slope(rows: &[Row], q: Sel, sign: f64, f: &dyn Fn(&Row) -> f64) -> Paired {
    let v: Vec<f64> = dose_values(rows, q, f)
        .iter()
        .map(|y| sign * dose_slope(y))
        .collect();
    paired(&v)
}

/// Per-seed concavity of `f` along the dose: f(dose 2/4) − (f(0) + f(4/4)) / 2.
#[must_use]
pub fn concavity(rows: &[Row], q: Sel, f: &dyn Fn(&Row) -> f64) -> Paired {
    let v: Vec<f64> = dose_values(rows, q, f)
        .iter()
        .map(|y| y[2] - 0.5 * (y[0] + y[4]))
        .collect();
    paired(&v)
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

/// Positive: mean > 0 and bootstrap CI lower bound > 0.
#[must_use]
pub fn positive(d: &Paired) -> bool {
    d.n > 0 && d.mean > 0.0 && d.ci_lo > 0.0
}

/// Equivalent: the 95 % CI lies strictly inside ±`margin`.
#[must_use]
pub fn equivalent(d: &Paired, margin: f64) -> bool {
    d.n > 0 && d.ci_lo > -margin && d.ci_hi < margin
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

/// Expected writable lines per task of a dose given `|F_j|`.
#[must_use]
pub fn expected_writable(dose: Dose, partner_lines: f64) -> f64 {
    let n = partner_lines.round() as usize;
    let take = |d: u8| ((usize::from(d) * n) / usize::from(DOSE_STEPS)) as f64;
    match dose {
        Dose::None => 0.0,
        Dose::Nested(d) => take(d),
        Dose::Complement => n as f64 - take(DOSE_STEPS / 2),
    }
}

/// Codes consistent: uncued arms carry no code, no lines and no mask; cued
/// arms have 1 × (own) or 2 × (share-2) the private lines of the same
/// (k, flip, seed) and the same cue-line edges; all share-2 arms share the
/// partner map and partner lines (= half their lines per task); writable
/// lines follow the dose (the complement has the same count as the nested
/// half); frozen moves are 0 everywhere; skips only where a line is frozen;
/// partner moves only for share-2 arms with a writable line.
#[must_use]
pub fn codes_consistent(rows: &[Row]) -> bool {
    rows.iter().all(|r| {
        if r.frozen_moved != 0 {
            return false;
        }
        let shared = r.code.shared();
        let frozen_any = shared && r.writable_lines + 1e-9 < r.partner_lines;
        if !frozen_any && r.skipped != 0 {
            return false;
        }
        if (!shared || r.writable_lines == 0.0) && r.partner_moved != 0 {
            return false;
        }
        if !r.inner.setting.cued() {
            return r.code == Code::None
                && r.dose == Dose::None
                && r.cue.lines_per_task == 0
                && r.sigma.is_empty()
                && r.partner_lines == 0.0
                && r.writable_lines == 0.0;
        }
        if !shared && (r.dose != Dose::None || r.partner_lines != 0.0 || r.writable_lines != 0.0) {
            return false;
        }
        if shared
            && (r.dose == Dose::None
                || (r.writable_lines - expected_writable(r.dose, r.partner_lines)).abs() > 1e-9)
        {
            return false;
        }
        if r.dose == Dose::Complement
            && (expected_writable(Dose::Complement, r.partner_lines)
                - expected_writable(Dose::Nested(DOSE_STEPS / 2), r.partner_lines))
            .abs()
                > 1e-9
        {
            return false;
        }
        let same = |y: &&Row| {
            y.inner.k == r.inner.k && y.inner.flip == r.inner.flip && y.inner.seed == r.inner.seed
        };
        let base = rows.iter().filter(same).find(|y| y.code == Code::Own);
        let mult = if shared { 2 } else { 1 };
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
                .all(|y| {
                    y.sigma == r.sigma
                        && y.agree == r.agree
                        && y.cue.reach == r.cue.reach
                        && y.partner_lines == r.partner_lines
                })
            && (!shared || (r.partner_lines - (r.cue.lines_per_task as f64) / 2.0).abs() < 1e-9)
    })
}

/// Every arm of a (k, flip, seed, order) carries the same uncapped E stage
/// totals, one per stage, and every capped arm's per-stage cap equals them.
#[must_use]
pub fn budgets_consistent(rows: &[Row]) -> bool {
    rows.iter().all(|r| {
        r.e_stage_writes.len() == r.inner.order.len()
            && (r.inner.setting == Setting::Err
                || (r.inner.logs.len() == r.e_stage_writes.len()
                    && r.inner
                        .logs
                        .iter()
                        .zip(&r.e_stage_writes)
                        .all(|(t, &w)| t.ledger.cap == w)))
            && rows.iter().all(|y| {
                y.inner.k != r.inner.k
                    || y.inner.flip != r.inner.flip
                    || y.inner.seed != r.inner.seed
                    || y.inner.order_ix != r.inner.order_ix
                    || y.e_stage_writes == r.e_stage_writes
            })
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
        && codes_consistent(rows)
        && budgets_consistent(rows);
    let ks = ks_of(rows);
    let flips = flips_of(rows);
    let main: Vec<usize> = ks.iter().copied().filter(|k| KS.contains(k)).collect();
    let g2 = main.contains(&K_MAIN)
        && main.iter().all(|&k| {
            flips
                .iter()
                .filter(|&&f| f == FLIP_MID || f == FLIP_OPPOSITE)
                .all(|&flip| g2_at(rows, k, flip).iter().all(|&b| b))
        });
    let g3 = ks
        .last()
        .is_some_and(|&k| median_cn(rows, k, FLIP_MID) >= ns::MIN_NOVEL_PAIRS);
    (g1, g2, g3)
}

/// G2 clauses (a)–(d) at one (k, flip), held-out rows: (a) mean
/// partner-line moves strictly increase along the five nested doses; (b)
/// the 1/4 dose moves a partner line in ≥ 50 % of rows; (c) the 1/4, 1/2,
/// 3/4 and complement doses skip a frozen edge step in ≥ 50 % of rows; (d)
/// the complement's mean partner moves are within a factor 2 of the nested
/// half's.
#[must_use]
pub fn g2_at(rows: &[Row], k: usize, flip: u32) -> [bool; 4] {
    let q = Sel {
        k,
        flip,
        orders: Orders::Heldout,
    };
    let pm = |x: Arm| setting_mean(rows, q, x, &|r| r.partner_moved as f64);
    let present = rows_of(rows, q, ZGD0).next().is_some();
    let m: Vec<f64> = DOSES.iter().map(|&x| pm(x)).collect();
    let a = present && m.windows(2).all(|w| w[1] > w[0]);
    let b = frac(rows, q, ZGD1, &|r: &Row| r.partner_moved) >= G2_ACT_FRAC;
    let c = [ZGD1, ZGD2, ZGD3, ZGD2C]
        .iter()
        .all(|&x| frac(rows, q, x, &|r: &Row| r.skipped) >= G2_ACT_FRAC);
    let (h, hc) = (pm(ZGD2), pm(ZGD2C));
    let d = h > 0.0 && hc > 0.0 && hc / h >= 1.0 / G2_RATIO && hc / h <= G2_RATIO;
    [a, b, c, d]
}

/// Number of pre-registered hypotheses.
pub const N_HYP: usize = 12;

/// Pre-registered readings: (gates, [H1..H12]); `None` if a gate fails or
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
    let sel = |k: usize, flip: u32| Sel {
        k,
        flip,
        orders: Orders::Heldout,
    };
    let c = |k: usize, flip: u32, s: Slice, x, b| compare(rows, sel(k, flip), k, s, x, b);
    let (km, fo, fm, a) = (K_MAIN, FLIP_OPPOSITE, FLIP_MID, Slice::All);
    if has(km, fo) {
        let q = sel(km, fo);
        h[0] = Some(positive(&slope(rows, q, 1.0, &|r| r.inner.acc(a, km))));
        h[1] = Some(positive(&slope(rows, q, -1.0, &|r| r.inner.learn(a, km))));
        h[2] = Some(positive(&slope(rows, q, -1.0, &|r| {
            r.inner.forgetting(a, km)
        })));
        h[3] = Some(positive(&concavity(rows, q, &|r| r.inner.acc(a, km))));
        h[4] = Some(equivalent(&c(km, fo, a, ZGD2, ZGD2C).acc, EQUIV_MARGIN));
        h[5] = Some(c(km, fo, a, ZGD4, S1).passes());
        h[6] = Some(c(km, fo, a, ZGD2, S1).passes());
        h[10] = Some(positive(&slope(rows, q, 1.0, &|r| {
            r.inner.acc(Slice::Cn, km)
        })));
        h[11] = Some(positive(&c(km, fo, Slice::Cn, ZGD2, S1).acc));
    }
    if has(km, fm) {
        h[7] = Some(positive(&slope(rows, sel(km, fm), 1.0, &|r| {
            r.inner.acc(a, km)
        })));
        h[8] = Some(c(km, fm, a, ZGD2, S1).passes());
    }
    if has(K_SMALL, fo) {
        h[9] = Some(positive(&slope(rows, sel(K_SMALL, fo), 1.0, &|r| {
            r.inner.acc(a, K_SMALL)
        })));
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
        "dose",
        "seed",
        "order_ix",
        "order_kind",
        "order",
        "partner_map",
        "partner_agree_mean",
        "conflict_links",
        "cue_lines_per_task",
        "partner_lines_per_task",
        "writable_partner_lines",
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
        "e_stage_writes_s",
        "idle_stages",
        "frozen_skipped",
        "frozen_moved",
        "partner_moved",
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
            "b888-4-cue-write-dose-v1".to_string(),
            graph_label.to_string(),
            k.to_string(),
            r.flip.to_string(),
            r.setting.name().to_string(),
            row.code.name().to_string(),
            row.dose.name(),
            r.seed.to_string(),
            r.order_ix.to_string(),
            crob::order_kind(r.order_ix).to_string(),
            dash(&r.order),
            dash(&row.sigma),
            f6(row.agree),
            row.conflicts.to_string(),
            row.cue.lines_per_task.to_string(),
            format!("{:.3}", row.partner_lines),
            format!("{:.3}", row.writable_lines),
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
            dash(&row.e_stage_writes),
            row.idle_stages().to_string(),
            row.skipped.to_string(),
            row.frozen_moved.to_string(),
            row.partner_moved.to_string(),
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

type Metric<'a> = (&'static str, Box<dyn Fn(&Row) -> f64 + 'a>);

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
        "setting_summary: k,flip,orders,arm,acc_all,acc_cn,learn_all,learn_cn,fgt_all,fgt_cn,last_all,charged_writes,effective_writes,cue_writes,cue_write_share,idle_stages,skipped,partner_moved,refused,overwrites,deferred,net_reversals,consolidated_end,saturated_end,cue_lines_per_task,partner_lines_per_task,writable_partner_lines,cue_reach_mean,cue_edges,cue_live_end,cue_full_end,partner_agree_mean,conflict_links"
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
                        "{k},{flip},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{:.1},{:.1},{:.3},{:.2},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.2},{:.2},{:.1},{:.1},{:.1},{:.1},{:.3},{:.2}",
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
                        m(&|r| r.idle_stages() as f64),
                        m(&|r| r.skipped as f64),
                        m(&|r| r.partner_moved as f64),
                        m(&|r| r.inner.refused() as f64),
                        m(&|r| og(r.inner.overwrites())),
                        m(&|r| og(r.inner.deferred())),
                        m(&|r| r.inner.net_reversals() as f64),
                        m(&|r| r.inner.consolidated_end() as f64),
                        m(&|r| r.inner.saturated_end as f64),
                        m(&|r| r.cue.lines_per_task as f64),
                        m(&|r| r.partner_lines),
                        m(&|r| r.writable_lines),
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
        (ZGD4, ZGD0),
        (ZGD1, ZGD0),
        (ZGD2, ZGD0),
        (ZGD3, ZGD0),
        (ZGD4, ZGD3),
        (ZGD4, ZGD2),
        (ZGD2, ZGD2C),
        (ZGD0, S1),
        (ZGD1, S1),
        (ZGD2, S1),
        (ZGD3, S1),
        (ZGD4, S1),
        (ZGD2C, S1),
        (ZGD2, ZG1),
        (ZGD4, ZG1),
        (ZGD0, ZG1),
        (ZG1, S1),
        (E, S1),
        (R1, S1),
    ];
    let key = |x: Arm, base: Arm| (base == S1 && (x == ZGD2 || x == ZGD4)) || base == ZGD2C;
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
        "dose_response(held-out; per-seed least-squares slope over doses 0,1/4,1/2,3/4,1 = change from owned to shared; concavity = f(1/2) - (f(0)+f(1))/2): k,flip,metric,slope,concavity"
    );
    for &k in &ks {
        for &flip in &flips {
            let q = Sel {
                k,
                flip,
                orders: Orders::Heldout,
            };
            if dose_values(rows, q, &|r| r.inner.acc(a, k)).is_empty() {
                continue;
            }
            let metrics: [Metric; 7] = [
                ("acc_all", Box::new(move |r: &Row| r.inner.acc(a, k))),
                ("acc_cn", Box::new(move |r: &Row| r.inner.acc(c, k))),
                ("learn_all", Box::new(move |r: &Row| r.inner.learn(a, k))),
                ("fgt_all", Box::new(move |r: &Row| r.inner.forgetting(a, k))),
                ("last_all", Box::new(move |r: &Row| r.inner.last(a, k))),
                ("partner_moved", Box::new(|r: &Row| r.partner_moved as f64)),
                (
                    "cue_writes",
                    Box::new(|r: &Row| r.inner.cue_writes() as f64),
                ),
            ];
            for (name, f) in &metrics {
                let _ = writeln!(
                    o,
                    "{k},{flip},{name},{},{}",
                    fmt_p(&slope(rows, q, 1.0, f.as_ref())),
                    fmt_p(&concavity(rows, q, f.as_ref()))
                );
            }
        }
    }
    let _ = writeln!(
        o,
        "worst_heldout_order(acc_all, per seed min over held-out orders): k,flip,x,base,d"
    );
    for &k in &ks {
        for &flip in &flips {
            for &(x, base) in &[(ZGD4, ZGD0), (ZGD2, ZGD2C), (ZGD2, S1), (ZGD4, S1)] {
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
            let sm = |x: Arm, f: &dyn Fn(&Row) -> f64| setting_mean(rows, q, x, f);
            let seven = [ZG1, ZGD0, ZGD1, ZGD2, ZGD3, ZGD4, ZGD2C];
            let list = |f: &dyn Fn(&Row) -> f64, p: usize| {
                seven
                    .iter()
                    .map(|&x| format!("{:.*}", p, sm(x, f)))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let g = g2_at(rows, k, flip);
            let _ = writeln!(
                o,
                "manipulation: k={k} flip={flip} idle_stages_mean={:.2} rows_with_idle_frac={:.3} g2(a_monotone,b_dose1_moves,c_skips,d_complement_ratio)=({},{},{},{}) frac_moves(Zgd1)={:.3} frac_skips(Zgd1,Zgd2,Zgd3,Zgd2c)=({:.3},{:.3},{:.3},{:.3}) writable_lines(Zg1,Zgd0..Zgd4,Zgd2c)=({}) partner_moved(...)=({}) skipped(...)=({}) cue_writes(...)=({}) cue_share(...)=({}) deferred(...)=({}) overwrites(...)=({}) live_cue_edges(...)=({}) effective_writes(S1,E,...)=({:.1},{:.1},{})",
                sm(E, &|r| r.idle_stages() as f64),
                frac(rows, q, E, &|r: &Row| r.idle_stages() as u64),
                g[0],
                g[1],
                g[2],
                g[3],
                frac(rows, q, ZGD1, &|r: &Row| r.partner_moved),
                frac(rows, q, ZGD1, &|r: &Row| r.skipped),
                frac(rows, q, ZGD2, &|r: &Row| r.skipped),
                frac(rows, q, ZGD3, &|r: &Row| r.skipped),
                frac(rows, q, ZGD2C, &|r: &Row| r.skipped),
                list(&|r| r.writable_lines, 2),
                list(&|r| r.partner_moved as f64, 1),
                list(&|r| r.skipped as f64, 1),
                list(&|r| r.inner.cue_writes() as f64, 1),
                list(&Row::cue_share, 3),
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
/// reach, writable lines per task, [charged writes, effective writes, cue
/// writes, refused, overwrites, deferred, skipped, frozen moves, partner
/// moves, idle stages]).
pub type CensusLine = (usize, u32, Arm, u64, f64, f64, f64, [Option<u64>; 10]);

type TrainOnly = (Vec<TaskLog>, Vec<cgd::GuardLog>, Vec<PhaseCount>);

#[allow(clippy::too_many_arguments)]
fn train_only(
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    s: Setting,
    order: &[usize],
    lim: &[Limit],
    masks: &[StageMask],
    watch: &[Vec<usize>],
) -> io::Result<TrainOnly> {
    let mut c = ci::initial_circuit(fx, spec, s.base().arm())?;
    let mut dir = vec![0_i8; fx.graph.edges()];
    let mut stamp = vec![0_u8; fx.graph.edges()];
    let (mut logs, mut glogs, mut phases) = (Vec::new(), Vec::new(), Vec::new());
    let none = StageMask::default();
    for (i, l) in lim.iter().enumerate() {
        let m = masks.get(order[i]).unwrap_or(&none);
        let w: &[usize] = watch.get(order[i]).map_or(&[], Vec::as_slice);
        let n = ts::task_order(spec, seed, order[i]).len();
        let (inner, gl, pc) = cgd::train_stage_scheduled_at(
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
            m,
            w,
            cgd::phase_split(n),
        )?;
        phases.push(pc);
        glogs.push(gl);
        logs.push(TaskLog {
            ledger: inner.inner.inner.ledger,
            stats: inner.inner.inner.stats,
            cue_cum: inner.inner.cue_cum,
            ..TaskLog::default()
        });
    }
    Ok((logs, glogs, phases))
}

/// Pilot census on the identity order: partner agreement, writes, guard
/// counters, skips, partner-line moves, idle stages and cue reach only. No
/// accuracy is computed and no held-out order is drawn.
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
        let (e_logs, ..) = train_only(
            &fx,
            &spec,
            seed,
            Setting::Err,
            &order,
            &sg::free(k),
            &[],
            &[],
        )?;
        let curves: Vec<Vec<u64>> = e_logs.iter().map(|t| t.ledger.cum.clone()).collect();
        let paced = cgd::paced_limits(&curves);
        let idle = curves
            .iter()
            .filter(|c| c.last().copied().unwrap_or(0) == 0)
            .count() as u64;
        for (x, (f, sigma, masks, watch, w, _)) in &fxs {
            let x = *x;
            let lim = if x.setting == Setting::Err {
                sg::free(k)
            } else {
                paced.clone()
            };
            let (logs, glogs, phases) =
                train_only(f, &spec, seed, x.setting, &order, &lim, masks, watch)?;
            let (sk, fm, pm) = sum_phases(&phases);
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
                *w,
                [
                    Some(charged),
                    Some(effective),
                    Some(sum(&|t| t.stats.cue_writes)),
                    Some(sum(&|t| t.ledger.refused)),
                    gsum(&|s| s.overwrites),
                    gsum(&|s| s.deferred),
                    Some(sk),
                    Some(fm),
                    Some(pm),
                    Some(idle),
                ],
            ));
        }
    }
    Ok(out)
}

/// One idle-stage diagnosis line (uncapped error-gated only; no accuracy).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IdleLine {
    pub k: usize,
    pub flip: u32,
    pub seed: u64,
    pub order_ix: usize,
    /// Uncapped E writes per stage, in training order.
    pub stage_writes: Vec<u64>,
    /// Stage task has the same valence vector as the previous stage's task.
    pub prev_same: Vec<bool>,
    /// Stage task has the same valence vector as some earlier stage's task.
    pub seen_same: Vec<bool>,
    /// `budget_split` of each stage (the cycle 26 write clock).
    pub splits: Vec<usize>,
    pub episodes: usize,
}

impl IdleLine {
    /// Mean `budget_split` over all stages (the cycle 26 G2 (e) statistic).
    #[must_use]
    pub fn split_mean_all(&self) -> f64 {
        mean(&self.splits.iter().map(|&s| s as f64).collect::<Vec<_>>())
    }
    /// Mean `budget_split` over stages with at least one uncapped write.
    #[must_use]
    pub fn split_mean_active(&self) -> f64 {
        mean(
            &self
                .splits
                .iter()
                .zip(&self.stage_writes)
                .filter(|(_, &w)| w > 0)
                .map(|(&s, _)| s as f64)
                .collect::<Vec<_>>(),
        )
    }
    /// Stages with zero uncapped writes.
    #[must_use]
    pub fn idle(&self) -> usize {
        self.stage_writes.iter().filter(|&&w| w == 0).count()
    }
}

/// Idle-stage diagnosis of one (k, flip, seed) on the orders of
/// `order_salt` (cycle 26: `cue_write_clock::ORDER_SALT`): uncapped
/// error-gated runs only, i.e. the stage totals that set every `paced100`
/// cap. No accuracy is computed.
pub fn idle_diagnosis(
    graph: Option<&Connectome>,
    design: &Design,
    k: usize,
    flip: u32,
    seed: u64,
    n_random: usize,
    order_salt: u64,
) -> io::Result<Vec<IdleLine>> {
    let spec = GenSpec {
        flip_pct: flip,
        ..design.spec
    };
    let fx = sg::fixture(graph, design, k, flip, seed)?;
    let mut out = Vec::new();
    for (o, order) in orders_salted(seed, k, n_random, order_salt)?
        .into_iter()
        .enumerate()
    {
        let e = cgd::run_setting(&fx, &spec, seed, Setting::Err, &order, &sg::free(k))?;
        let curves: Vec<Vec<u64>> = e.logs.iter().map(|t| t.ledger.cum.clone()).collect();
        let v = &fx.task.valence;
        let prev_same = (0..k)
            .map(|i| i > 0 && v[order[i]] == v[order[i - 1]])
            .collect();
        let seen_same = (0..k)
            .map(|i| (0..i).any(|j| v[order[i]] == v[order[j]]))
            .collect();
        out.push(IdleLine {
            k,
            flip,
            seed,
            order_ix: o,
            stage_writes: curves
                .iter()
                .map(|c| c.last().copied().unwrap_or(0))
                .collect(),
            prev_same,
            seen_same,
            splits: cgd::budget_splits(&curves),
            episodes: curves.first().map(Vec::len).unwrap_or(0),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Design {
        Design {
            // The main graph shape and cue reservation (|F_j| = 8 at k = 8,
            // 4 at k = 16) with a shorter task.
            shape: DESIGN.shape,
            spec: GenSpec {
                active: 4,
                train_patterns: 8,
                test_patterns: 24,
                rounds_a: 2,
                rounds_b: 2,
                ..ns::DESIGN.spec
            },
            cue_cap: DESIGN.cue_cap,
        }
    }

    /// One small grid (k 8 / 16, flips 50 / 100, seeds 1..2, 4 workers)
    /// shared by the grid and readings tests, computed once per test
    /// binary so the qualification stays inside its time budget.
    fn base_grid() -> &'static [Row] {
        static GRID: std::sync::OnceLock<Vec<Row>> = std::sync::OnceLock::new();
        GRID.get_or_init(|| {
            run_grid_threads(None, &small(), &[8, 16], &[50, 100], &[1, 2], 2, 4).unwrap()
        })
    }

    #[test]
    fn doses_are_nested_seeded_and_the_complement_is_disjoint() {
        let f = vec![vec![10, 11, 12, 13], vec![20, 21, 22, 23, 24, 25, 26, 27]];
        for seed in [1_u64, 2, 681] {
            let w: Vec<Vec<Vec<usize>>> = (0..=DOSE_STEPS)
                .map(|d| writable(Dose::Nested(d), &f, seed))
                .collect();
            for j in 0..2 {
                for d in 0..=usize::from(DOSE_STEPS) {
                    assert_eq!(w[d][j].len(), d * f[j].len() / 4);
                    assert!(w[d][j].iter().all(|l| f[j].contains(l)));
                    if d > 0 {
                        assert!(w[d - 1][j].iter().all(|l| w[d][j].contains(l)));
                    }
                }
                assert_eq!(w[4][j], f[j]);
                let c = &writable(Dose::Complement, &f, seed)[j];
                assert_eq!(c.len(), w[2][j].len());
                assert!(c.iter().all(|l| !w[2][j].contains(l)));
            }
            assert_eq!(
                schedule(Dose::Nested(0), &f, seed)[0],
                StageMask::constant(&f[0])
            );
            assert_eq!(
                schedule(Dose::Nested(4), &f, seed)[1],
                StageMask::constant(&[])
            );
            assert!(writable(Dose::None, &f, seed).iter().all(Vec::is_empty));
        }
        assert_ne!(
            line_permutation(1, 1, &f[1]),
            line_permutation(2, 1, &f[1]),
            "seeded"
        );
        assert!((dose_slope(&[0.0, 0.25, 0.5, 0.75, 1.0]) - 1.0).abs() < 1e-12);
        assert!((dose_slope(&[1.0, 0.75, 0.5, 0.25, 0.0]) + 1.0).abs() < 1e-12);
        assert!(dose_slope(&[1.0; 5]).abs() < 1e-12);
        assert!((expected_writable(Dose::Nested(3), 4.0) - 3.0).abs() < 1e-12);
        assert!((expected_writable(Dose::Complement, 8.0) - 4.0).abs() < 1e-12);
        assert!(equivalent(&paired(&[0.001, -0.002, 0.0005, 0.001]), 0.01));
        assert!(!equivalent(&paired(&[0.05, 0.04, 0.06]), 0.01));
        assert_eq!(ZGD2C.short(), "Zgd2c");
        assert_eq!(ZGD0.short(), "Zgd0");
        assert_eq!(ZG1.short(), "Zg1");
    }

    #[test]
    fn full_dose_is_the_shared_write_and_zero_dose_the_owned_write() {
        let d = small();
        let (k, flip, seed) = (6, 100_u32, 3_u64);
        let spec = GenSpec {
            flip_pct: flip,
            ..d.spec
        };
        let fx = sg::fixture(None, &d, k, flip, seed).unwrap();
        let a = cp::agreement(&fx).unwrap();
        let order: Vec<usize> = (0..k).rev().collect();
        let e = cgd::run_setting(&fx, &spec, seed, Setting::Err, &order, &sg::free(k)).unwrap();
        let curves: Vec<Vec<u64>> = e.logs.iter().map(|t| t.ledger.cum.clone()).collect();
        let paced = cgd::paced_limits(&curves);
        let (f4, _, m4, w4, _, ok4) = arm_fixture(&fx, ZGD4, &a, seed).unwrap();
        assert!(ok4);
        let (r4, _) = cgd::run_setting_scheduled_at(
            &f4,
            &spec,
            seed,
            Setting::Zg,
            &order,
            &paced,
            &m4,
            &w4,
            &[],
        )
        .unwrap();
        let plain = cgd::run_setting(&f4, &spec, seed, Setting::Zg, &order, &paced).unwrap();
        assert_eq!(r4.weights, plain.weights);
        let (f0, sigma, m0, w0, wl, ok0) = arm_fixture(&fx, ZGD0, &a, seed).unwrap();
        assert!(ok0 && wl == 0.0);
        let frozen = cown::frozen_lines(&fx.cues, &sigma);
        let (r0, ph) = cgd::run_setting_scheduled_at(
            &f0,
            &spec,
            seed,
            Setting::Zg,
            &order,
            &paced,
            &m0,
            &w0,
            &[],
        )
        .unwrap();
        let (masked, _) =
            cgd::run_setting_masked(&f0, &spec, seed, Setting::Zg, &order, &paced, &frozen)
                .unwrap();
        assert_eq!(r0.weights, masked.weights);
        let (_, fm, pm) = sum_phases(&ph);
        assert_eq!((fm, pm), (0, 0), "no frozen or partner move at dose 0");
        // A tampered schedule or partner map is invalid.
        let (f2, s2, m2, w2, _, ok2) = arm_fixture(&fx, ZGD2, &a, seed).unwrap();
        assert!(ok2);
        let mut bad_m = m2.clone();
        bad_m[0] = StageMask::constant(&w2[0]);
        assert!(!codes_valid(
            &fx.cues, &f2.cues, ZGD2, &s2, &bad_m, &w2, &a, seed
        ));
        assert!(!codes_valid(
            &fx.cues, &f2.cues, ZGD2C, &s2, &m2, &w2, &a, seed
        ));
        assert!(codes_valid(
            &fx.cues, &f2.cues, ZGD2, &s2, &m2, &w2, &a, seed
        ));
    }

    #[test]
    fn grid_is_deterministic_thread_independent_and_gates_detect_tampering() {
        let d = small();
        let ks = [8, 16];
        let a: Vec<Row> = base_grid().to_vec();
        assert_eq!(a.len(), 2 * 2 * 2 * 4 * ARMS.len());
        // Same rows from one worker on a sub-grid (thread count and grid
        // composition do not change a row).
        let b = run_grid_threads(None, &d, &ks, &[100], &[1], 2, 1).unwrap();
        let a_sub: Vec<Row> = a
            .iter()
            .filter(|r| r.inner.flip == 100 && r.inner.seed == 1)
            .cloned()
            .collect();
        assert_eq!(a_sub, b);
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&a, "GENERATED_SYNTHETIC"));
        assert_eq!(
            to_csv(&a_sub, "GENERATED_SYNTHETIC"),
            to_csv(&b, "GENERATED_SYNTHETIC")
        );
        assert_eq!(csv.lines().count(), 1 + a.len());
        let cols = csv_header().split(',').count();
        assert!(csv.lines().all(|l| l.split(',').count() == cols));
        let rep = report(&a);
        assert_eq!(rep, report(&a.clone()));
        assert_eq!(report(&a_sub), report(&b));
        assert!(rep.contains("readings: G1_integrity=true"), "{rep}");
        assert!(rep.contains("Zgd2c") && rep.contains("dose_response"));
        assert!(gates(&a).0);
        let q = Sel {
            k: 8,
            flip: 100,
            orders: Orders::Heldout,
        };
        assert_eq!(compare(&a, q, 8, Slice::All, ZGD2, ZGD2).acc.ties, 2);
        assert!(run_grid_threads(None, &d, &ks, &[50], &[], 2, 1).is_err());
        let tamper = |x: Arm, f: &dyn Fn(&mut Row)| {
            let mut t = a.clone();
            let j = t.iter().position(|r| r.arm() == x).unwrap();
            f(&mut t[j]);
            gates(&t).0
        };
        for x in [ZGD0, ZGD1, ZGD2, ZGD3, ZGD4, ZGD2C] {
            assert!(!tamper(x, &|r| r.codes_ok = false));
            assert!(!tamper(x, &|r| r.cue.lines_per_task += 1));
            assert!(!tamper(x, &|r| r.sigma.swap(0, 1)));
            assert!(!tamper(x, &|r| r.frozen_moved = 1));
            assert!(!tamper(x, &|r| r.writable_lines += 1.0));
            assert!(!tamper(x, &|r| r.e_stage_writes[0] += 1));
        }
        assert!(!tamper(ZGD4, &|r| r.skipped = 1));
        assert!(!tamper(ZG1, &|r| r.skipped = 1));
        assert!(!tamper(ZGD0, &|r| r.partner_moved = 1));
        assert!(!tamper(ZG1, &|r| r.partner_moved = 1));
        assert!(!tamper(S1, &|r| r.dose = Dose::Nested(1)));
        let mut t = a.clone();
        let j = t.iter().position(|r| r.inner.order_ix == 1).unwrap();
        t[j].inner.order.swap(0, 1);
        assert!(!gates(&t).0);
        let mut t = a.clone();
        let j = t.iter().position(|r| r.arm() == S1).unwrap();
        t[j].inner.logs[0].ledger.writes = t[j].inner.logs[0].ledger.cap + 1;
        assert!(!gates(&t).0);
        // G2 tampering (each clause can fail on its own).
        assert!(gates(&a).1, "G2 binds on the small design");
        let g2 = |x: Arm, f: &dyn Fn(&mut Row)| {
            let mut t = a.clone();
            for r in t.iter_mut().filter(|r| r.arm() == x) {
                f(r);
            }
            gates(&t).1
        };
        assert!(!g2(ZGD3, &|r| r.partner_moved = 0));
        assert!(!g2(ZGD1, &|r| r.partner_moved = 0));
        assert!(!g2(ZGD2, &|r| r.skipped = 0));
        assert!(!g2(ZGD2C, &|r| r.partner_moved *= 100));
        assert!(!g2(ZGD2C, &|r| r.partner_moved = 0));
    }

    #[test]
    fn readings_are_evaluated_on_the_preregistered_ks_and_flips() {
        let d = small();
        let (g, h) = readings(base_grid());
        assert!(g.0);
        if g.0 && g.1 && g.2 {
            assert!(h.iter().all(Option::is_some));
        } else {
            assert_eq!(h, [None; N_HYP]);
        }
        // No k = 16: G2 cannot pass, nothing is evaluable.
        let small_rows = run_grid_threads(None, &d, &[8], &[50, 100], &[1], 1, 2).unwrap();
        assert!(!gates(&small_rows).1);
        assert_eq!(readings(&small_rows).1, [None; N_HYP]);
    }

    #[test]
    fn census_counts_writes_skips_and_partner_moves_without_accuracy() {
        let d = small();
        let lines = census(None, &d, 8, 100, &[1, 2]).unwrap();
        assert_eq!(lines.len(), 2 * ARMS.len());
        for (k, flip, x, _, ag, rch, w, v) in &lines {
            assert_eq!((*k, *flip), (8, 100));
            assert_eq!(*rch > 0.0, x.code != Code::None);
            assert_eq!(v[4].is_some(), x.setting.guard().is_some());
            assert_eq!(v[7], Some(0), "frozen moves");
            assert_eq!(
                v[9],
                Some(0),
                "identity order at flip 100 has no idle stage"
            );
            if !x.code.shared() || x.dose == Dose::Nested(DOSE_STEPS) {
                assert_eq!(v[6], Some(0), "{x:?}");
            }
            if !x.code.shared() || x.dose == Dose::Nested(0) {
                assert_eq!(v[8], Some(0), "{x:?}");
                assert!(*w == 0.0);
            }
            if x.code == Code::Own {
                assert!((ag - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn idle_diagnosis_matches_the_grid_budget_and_flags_repeated_valence() {
        let d = small();
        let lines = idle_diagnosis(None, &d, 8, 100, 2, 2, ORDER_SALT).unwrap();
        let rows = run_seed(None, &d, 8, 100, 2, 2).unwrap();
        assert_eq!(lines.len(), 4);
        for l in &lines {
            let r = rows
                .iter()
                .find(|r| r.inner.order_ix == l.order_ix)
                .unwrap();
            assert_eq!(l.stage_writes, r.e_stage_writes);
            assert!(!l.prev_same[0] && !l.seen_same[0]);
            assert!(l.prev_same.iter().zip(&l.seen_same).all(|(p, s)| !p || *s));
            // Flip 100: tasks alternate between two valence vectors.
            for (i, &t) in r.inner.order.iter().enumerate().skip(1) {
                assert_eq!(l.prev_same[i], (t % 2) == (r.inner.order[i - 1] % 2));
            }
            assert_eq!(l.splits.len(), 8);
        }
        // Identity order at flip 100 alternates the valence: no repeat.
        assert!(lines[0].prev_same.iter().all(|&b| !b));
        // The cycle 26 orders are reproduced with the cycle 26 salt.
        let c26 =
            idle_diagnosis(None, &d, 8, 100, 2, 2, crate::cue_write_clock::ORDER_SALT).unwrap();
        let o26 = crate::cue_write_clock::orders(2, 8, 2).unwrap();
        assert_eq!(c26.len(), o26.len());
        let idle_line = IdleLine {
            k: 3,
            flip: 100,
            seed: 0,
            order_ix: 2,
            stage_writes: vec![10, 0, 4],
            prev_same: vec![false, true, false],
            seen_same: vec![false, true, true],
            splits: vec![2, 8, 4],
            episodes: 8,
        };
        assert_eq!(idle_line.idle(), 1);
        assert!((idle_line.split_mean_all() - 14.0 / 3.0).abs() < 1e-12);
        assert!((idle_line.split_mean_active() - 3.0).abs() < 1e-12);
    }
}

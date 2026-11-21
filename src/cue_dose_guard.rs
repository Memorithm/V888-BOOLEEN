//! B888-4 cycle 29: **dose guard** — why does the dosed share-2 cue code
//! lose to the supervised control S1 on *raw* forgetting while it wins on
//! *relative* forgetting, and does the raw gap close with more graph
//! sources or with a changed write policy (the `block` consolidation guard
//! instead of `last-resort`), at equal `paced100` write budget and without
//! changing the linear partner-write dose effect of cycles 27–28?
//!
//! Cycle 28 (#187, `docs/CUE_DOSE_TRANSFER.md`) found at k = 16, flip 100
//! that Zgd4 forgets about 0.11 more than S1 in raw terms (H5 failed) while
//! it loses a smaller share of its above-chance learning (H6/H7 passed). By
//! definition FGT = LEARN_head − RET_head (RET_head = final accuracy on the
//! first k − 1 tasks), so the raw gap splits exactly into a retention term
//! and a learning term; this cycle measures both, on fresh seeds 761..800.
//! It re-runs the **unchanged** cycle 27 engine
//! ([`crate::cue_write_dose::run_seed`], ten arms) and adds five arms that
//! use the same matched share-2 code and the same constant dose masks under
//! the `block` guard ([`Setting::Zb`] = `E/E-cue0+block@paced100`), with the
//! same per-stage `paced100` caps (the uncapped error-gated write curve of
//! the same (seed, order)).
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026,
//! BANC/FlyWire consortium, see `docs/SOURCES.md`) is present on the build
//! machine; every condition uses the generated graph of
//! `consolidation_scale::generate_graph` and artificial labels. "Cue",
//! "dose", "guard", "block", "partner", "retention" and "forgetting" name
//! parts of an artificial benchmark on a Boolean graph with 4-bit weights,
//! not claims about synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/CUE_DOSE_GUARD.md`
//! (committed and pushed before any learning run on the hypothesis seeds).

use crate::consolidation_guard::{self as cgd, PhaseCount, Setting, StageMask};
use crate::cue_capacity as ccap;
use crate::cue_displacement::TaskLog;
use crate::cue_dose_transfer as cdt;
use crate::cue_init as ci;
use crate::cue_partner as cp;
use crate::cue_write_dose::{self as cwd, Arm, Code, Dose, Orders, Row, Sel, S1, ZGD0, ZGD4};
use crate::error_gated::{paired, Paired};
use crate::generalization_bench::{GenSpec, GraphShape};
use crate::novel_split::{self as ns, Design, Fixture, Slice};
use crate::seq_generalisation as sg;
use crate::task_sequence as ts;
use crate::write_allocation as wa;
use crate::write_budget::Limit;
use std::collections::HashSet;
use std::io;

fn bad(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Run length (cycles 27–28 primary k; `|F_j|` = 4).
pub const K: usize = cdt::K;
/// Label flip between consecutive tasks (cycles 27–28 primary flip).
pub const FLIP: u32 = cdt::FLIP;
/// Hypothesis seeds (never used by an earlier B888-4 cycle).
pub const SEEDS: std::ops::RangeInclusive<u64> = 761..=800;
/// Label-free pilot seeds.
pub const PILOT_SEEDS: std::ops::RangeInclusive<u64> = 1..=5;
/// Seeded random held-out orders (plus the reversed order).
pub const RANDOM_ORDERS: usize = 3;
/// Equivalence margin (H10, H12).
pub const EQUIV_MARGIN: f64 = 0.01;
/// Non-inferiority / replication margin on raw forgetting and accuracy
/// (H2, H4, H6, H8).
pub const NONINF_MARGIN: f64 = 0.02;
/// Share of held-out rows of every block arm that must refuse at least one
/// overwrite (gate G2b).
pub const BLOCK_ACT_FRAC: f64 = 0.5;

/// `E/E-cue0+block@paced100` with the matched share-2 code at dose d/4.
const fn zb(d: u8) -> Arm {
    Arm {
        setting: Setting::Zb,
        code: Code::Matched,
        dose: Dose::Nested(d),
    }
}

pub const ZBD0: Arm = zb(0);
pub const ZBD1: Arm = zb(1);
pub const ZBD2: Arm = zb(2);
pub const ZBD3: Arm = zb(3);
pub const ZBD4: Arm = zb(4);
/// The five block-guard dose arms, in dose order.
pub const ZB_DOSES: [Arm; 5] = [ZBD0, ZBD1, ZBD2, ZBD3, ZBD4];
/// The five last-resort dose arms of the unchanged engine.
pub const ZG_DOSES: [Arm; 5] = cwd::DOSES;

/// Short arm label (block arms `Zbd0`..`Zbd4`, others as cycle 27).
#[must_use]
pub fn short(x: Arm) -> String {
    match (x.setting, x.dose) {
        (Setting::Zb, Dose::Nested(d)) => format!("Zbd{d}"),
        _ => x.short(),
    }
}

/// Pre-registered fixture conditions (only the number of graph sources
/// differs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cond {
    /// The cycle 27 design (128 sources).
    Base,
    /// 256 graph sources (targets 32, fan-in 8 unchanged).
    Src256,
    /// 512 graph sources (targets 32, fan-in 8 unchanged).
    Src512,
    /// B888-4 cycle 30 (`docs/CUE_DOSE_DENSITY.md`): 256 graph sources with
    /// fan-in 16 (the edge density per source of `Base`). Not in [`CONDS`].
    D256,
    /// B888-4 cycle 30 (`docs/CUE_DOSE_DENSITY.md`): 512 graph sources with
    /// fan-in 32, i.e. the edge density per source of `Base` (2 edges per
    /// source). Not in [`CONDS`]: the cycle 29 grid is unchanged.
    D512,
    /// B888-4 cycle 31 (`docs/CUE_FANIN.md`): 128 graph sources (as `Base`)
    /// with fan-in 16. Not in [`CONDS`].
    F16,
    /// B888-4 cycle 31 (`docs/CUE_FANIN.md`): 128 graph sources (as `Base`)
    /// with fan-in 32. Not in [`CONDS`].
    F32,
}

/// Conditions in CSV / report order.
pub const CONDS: [Cond; 3] = [Cond::Base, Cond::Src256, Cond::Src512];

impl Cond {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Src256 => "src256",
            Self::Src512 => "src512",
            Self::D256 => "d256",
            Self::D512 => "d512",
            Self::F16 => "f16",
            Self::F32 => "f32",
        }
    }
    /// Parse a condition name.
    pub fn parse(s: &str) -> io::Result<Self> {
        CONDS
            .into_iter()
            .find(|c| c.name() == s)
            .ok_or_else(|| bad("unknown condition"))
    }
    /// Graph sources of the condition.
    #[must_use]
    pub fn sources(self) -> usize {
        match self {
            Self::Base | Self::F16 | Self::F32 => cwd::DESIGN.shape.sources,
            Self::Src256 | Self::D256 => 256,
            Self::Src512 | Self::D512 => 512,
        }
    }
    /// Fan-in of the condition (8 = cycle 27, except [`Cond::D256`] and
    /// [`Cond::D512`]).
    #[must_use]
    pub fn fan_in(self) -> usize {
        match self {
            Self::D256 | Self::F16 => 16,
            Self::D512 | Self::F32 => 32,
            _ => cwd::DESIGN.shape.fan_in,
        }
    }
    /// Design of the condition (only `shape.sources`, and for
    /// [`Cond::D256`] / [`Cond::D512`] `shape.fan_in`, differ from cycle 27).
    #[must_use]
    pub fn design(self) -> Design {
        let d = cwd::DESIGN;
        Design {
            shape: GraphShape {
                sources: self.sources(),
                fan_in: self.fan_in(),
                ..d.shape
            },
            ..d
        }
    }
}

/// Rows of one condition.
#[derive(Clone, Debug, PartialEq)]
pub struct CondRows {
    pub cond: Cond,
    /// Rows of the unchanged cycle 27 engine (ten arms).
    pub rows: Vec<Row>,
    /// Rows of the five block-guard dose arms (same seeds, orders, caps).
    pub guard_rows: Vec<Row>,
    /// Per seed: leakage-free fixture (gate G4).
    pub leakage: Vec<(u64, bool)>,
}

impl CondRows {
    /// Both families together (for paired contrasts across families).
    #[must_use]
    pub fn all_rows(&self) -> Vec<Row> {
        let mut v = self.rows.clone();
        v.extend(self.guard_rows.iter().cloned());
        v
    }
}

/// Leakage check of one fixture: no TEST pattern (as a set of active
/// inputs) equals any training pattern of any task.
pub fn leakage_free(design: &Design, seed: u64) -> io::Result<bool> {
    let fx = sg::fixture(None, design, K, FLIP, seed)?;
    let norm = |p: &Vec<usize>| {
        let mut v = p.clone();
        v.sort_unstable();
        v
    };
    let test: HashSet<Vec<usize>> = fx.task.test.iter().map(norm).collect();
    Ok(!test.is_empty()
        && fx
            .task
            .train
            .iter()
            .flatten()
            .all(|p| !test.contains(&norm(p))))
}

fn sum_phases(p: &[PhaseCount]) -> (u64, u64, u64) {
    let (mut sk, mut fm, mut wm) = (0, 0, 0);
    for c in p {
        for h in 0..2 {
            sk += c.skipped[h];
            fm += c.frozen_moved[h];
            wm += c.watch_moved[h];
        }
    }
    (sk, fm, wm)
}

/// Rows of the block-guard dose arms of one (seed): per order, the uncapped
/// error-gated run sets the `paced100` caps exactly as in
/// [`cwd::run_seed`]; each block arm then runs on its code fixture and
/// constant dose masks.
pub fn run_seed_guard(design: &Design, seed: u64, n_random: usize) -> io::Result<Vec<Row>> {
    let spec = GenSpec {
        flip_pct: FLIP,
        ..design.spec
    };
    let fx = sg::fixture(None, design, K, FLIP, seed)?;
    let a = cp::agreement(&fx)?;
    let fxs = ZB_DOSES
        .iter()
        .map(|&x| Ok((x, cwd::arm_fixture(&fx, x, &a, seed)?)))
        .collect::<io::Result<Vec<_>>>()?;
    let mut rows = Vec::new();
    for (o, order) in cwd::orders(seed, K, n_random)?.into_iter().enumerate() {
        let e = cgd::run_setting(&fx, &spec, seed, Setting::Err, &order, &sg::free(K))?;
        let curves: Vec<Vec<u64>> = e.logs.iter().map(|t| t.ledger.cum.clone()).collect();
        let paced = cgd::paced_limits(&curves);
        let stage_writes: Vec<u64> = curves
            .iter()
            .map(|c| c.last().copied().unwrap_or(0))
            .collect();
        let cn_pairs: Vec<u64> = order.iter().map(|&t| sg::pair_counts(&fx, t)[2]).collect();
        for (x, (f, sigma, masks, watch, w, ok)) in &fxs {
            let s = x.setting;
            let (run, phases) = cgd::run_setting_scheduled_at(
                f,
                &spec,
                seed,
                s,
                &order,
                &paced,
                masks,
                watch,
                &[],
            )?;
            let (skipped, frozen_moved, partner_moved) = sum_phases(&phases);
            let init = ci::initial_circuit(f, &spec, s.base().arm())?;
            let cue = ccap::cue_use(f, &run.weights, s.cued())?;
            let (agree, conflicts) = cp::partner_agreement(&a, sigma);
            let partner_lines = watch.iter().map(Vec::len).sum::<usize>() as f64 / K as f64;
            rows.push(Row {
                inner: cgd::Row {
                    setting: s,
                    k: K,
                    flip: FLIP,
                    seed,
                    order_ix: o,
                    order: order.clone(),
                    cn_pairs: cn_pairs.clone(),
                    saturated_end: crate::error_gated::saturated(&run.weights),
                    topology_unchanged: run.topology_unchanged,
                    init_ok: ci::init_matches(f, &spec, s.base().arm(), init.weights())?,
                    paced_identity: true,
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

/// Both families of one seed: (unchanged cycle 27 rows, block-guard rows).
pub fn run_seed(design: &Design, seed: u64, n_random: usize) -> io::Result<(Vec<Row>, Vec<Row>)> {
    Ok((
        cwd::run_seed(None, design, K, FLIP, seed, n_random)?,
        run_seed_guard(design, seed, n_random)?,
    ))
}

type SeedRows = io::Result<(Vec<Row>, Vec<Row>)>;

/// One condition over `seeds` on `threads` worker threads; the output order
/// (seed, order, arm) and content do not depend on the thread count.
pub fn run_cond(
    design: &Design,
    seeds: &[u64],
    n_random: usize,
    threads: usize,
) -> io::Result<(Vec<Row>, Vec<Row>)> {
    if seeds.is_empty() {
        return Err(bad("need at least one seed"));
    }
    let threads = threads.clamp(1, seeds.len());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<Option<SeedRows>> = (0..seeds.len()).map(|_| None).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let next = &next;
                scope.spawn(move || {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&s) = seeds.get(i) else {
                            break;
                        };
                        out.push((i, run_seed(design, s, n_random)));
                    }
                    out
                })
            })
            .collect();
        for h in handles {
            if let Ok(v) = h.join() {
                for (i, r) in v {
                    results[i] = Some(r);
                }
            }
        }
    });
    let (mut rows, mut guard_rows) = (Vec::new(), Vec::new());
    for r in results {
        let (a, b) = r.ok_or_else(|| bad("worker thread failed"))??;
        rows.extend(a);
        guard_rows.extend(b);
    }
    Ok((rows, guard_rows))
}

/// Run the pre-registered grid of the given conditions.
pub fn run(conds: &[Cond], seeds: &[u64], threads: usize) -> io::Result<Vec<CondRows>> {
    run_with(conds, seeds, threads, RANDOM_ORDERS, Cond::design)
}

/// [`run`] with an explicit design per condition and order count (tests use
/// a shorter task).
pub fn run_with(
    conds: &[Cond],
    seeds: &[u64],
    threads: usize,
    n_random: usize,
    design: fn(Cond) -> Design,
) -> io::Result<Vec<CondRows>> {
    conds
        .iter()
        .map(|&cond| {
            let d = design(cond);
            let (rows, guard_rows) = run_cond(&d, seeds, n_random, threads)?;
            let leakage = seeds
                .iter()
                .map(|&s| Ok((s, leakage_free(&d, s)?)))
                .collect::<io::Result<Vec<_>>>()?;
            Ok(CondRows {
                cond,
                rows,
                guard_rows,
                leakage,
            })
        })
        .collect()
}

fn sel() -> Sel {
    Sel {
        k: K,
        flip: FLIP,
        orders: Orders::Heldout,
    }
}

pub fn seeds_of(rows: &[Row]) -> Vec<u64> {
    let mut s: Vec<u64> = rows.iter().map(|r| r.inner.seed).collect();
    s.sort_unstable();
    s.dedup();
    s
}

/// Mean of `f` over the held-out rows of arm `x` and `seed`.
pub fn seed_mean(rows: &[Row], seed: u64, x: Arm, f: &dyn Fn(&Row) -> f64) -> Option<f64> {
    let v: Vec<f64> = rows
        .iter()
        .filter(|r| {
            r.inner.k == K
                && r.inner.flip == FLIP
                && r.inner.order_ix > 0
                && r.inner.seed == seed
                && r.arm() == x
        })
        .map(f)
        .collect();
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

/// ACC_all of a row (final balanced accuracy over all k tasks).
#[must_use]
pub fn acc(r: &Row) -> f64 {
    r.inner.acc(Slice::All, K)
}

/// Retained accuracy on the first k − 1 tasks at the end of the sequence:
/// mean over j < k − 1 of R[k−1][j]. Exactly FGT_all = LEARN_head − RET_head.
#[must_use]
pub fn ret_head(r: &Row) -> f64 {
    let k = K as f64;
    (k * r.inner.acc(Slice::All, K) - r.inner.last(Slice::All, K)) / (k - 1.0)
}

/// Per seed `f(x) − f(base)` (mean over held-out orders), paired.
#[must_use]
pub fn delta(rows: &[Row], x: Arm, base: Arm, f: &dyn Fn(&Row) -> f64) -> Paired {
    paired(&delta_by_seed(rows, x, base, f))
}

fn delta_by_seed(rows: &[Row], x: Arm, base: Arm, f: &dyn Fn(&Row) -> f64) -> Vec<f64> {
    delta_seeded(rows, x, base, f)
        .into_iter()
        .map(|(_, v)| v)
        .collect()
}

pub fn delta_seeded(rows: &[Row], x: Arm, base: Arm, f: &dyn Fn(&Row) -> f64) -> Vec<(u64, f64)> {
    seeds_of(rows)
        .into_iter()
        .filter_map(|s| Some((s, seed_mean(rows, s, x, f)? - seed_mean(rows, s, base, f)?)))
        .collect()
}

/// Raw forgetting gap of x against S1 per seed: FGT(S1) − FGT(x) (> 0: x
/// forgets less).
#[must_use]
pub fn raw_gap(rows: &[Row], x: Arm) -> Paired {
    delta(rows, S1, x, &cdt::fgt)
}

/// Per seed (seed index paired across conditions) change of the raw gap
/// FGT(S1) − FGT(Zgd4) from condition `a` to condition `b` (> 0: the gap
/// closes in `b`).
#[must_use]
pub fn gap_change(a: &[Row], b: &[Row]) -> Paired {
    let ga = delta_seeded(a, S1, ZGD4, &cdt::fgt);
    let gb = delta_seeded(b, S1, ZGD4, &cdt::fgt);
    let d: Vec<f64> = gb
        .iter()
        .filter_map(|&(s, vb)| ga.iter().find(|x| x.0 == s).map(|&(_, va)| vb - va))
        .collect();
    paired(&d)
}

/// Per-seed dose slope (sign × slope) of `f` along `arms` (five nested
/// doses 0/4 … 4/4).
#[must_use]
pub fn slope(rows: &[Row], arms: &[Arm; 5], sign: f64, f: &dyn Fn(&Row) -> f64) -> Paired {
    let v: Vec<f64> = dose_values(rows, arms, f)
        .iter()
        .map(|y| sign * cwd::dose_slope(y))
        .collect();
    paired(&v)
}

/// Per-seed concavity f(2/4) − (f(0) + f(4/4)) / 2 along `arms`.
#[must_use]
pub fn concavity(rows: &[Row], arms: &[Arm; 5], f: &dyn Fn(&Row) -> f64) -> Paired {
    let v: Vec<f64> = dose_values(rows, arms, f)
        .iter()
        .map(|y| y[2] - 0.5 * (y[0] + y[4]))
        .collect();
    paired(&v)
}

fn dose_values(rows: &[Row], arms: &[Arm; 5], f: &dyn Fn(&Row) -> f64) -> Vec<[f64; 5]> {
    seeds_of(rows)
        .into_iter()
        .filter_map(|s| {
            let mut v = [0.0; 5];
            for (i, &x) in arms.iter().enumerate() {
                v[i] = seed_mean(rows, s, x, f)?;
            }
            Some(v)
        })
        .collect()
}

pub fn arm_mean(rows: &[Row], x: Arm, f: &dyn Fn(&Row) -> f64) -> f64 {
    cwd::setting_mean(rows, sel(), x, f)
}

/// Integrity of the block-guard rows against the unchanged rows of the same
/// condition: every row valid (topology, init, cap, codes, no frozen move,
/// writable lines = dose), the `block` guard consistent with its ledger (no
/// effective overwrite, no deferral), the same orders, the same uncapped
/// stage totals and caps as the unchanged rows of the same (seed, order),
/// and the same partner map, partner lines and cue edges as the last-resort
/// share-2 arms.
#[must_use]
pub fn guard_rows_consistent(rows: &[Row], guard_rows: &[Row]) -> bool {
    !guard_rows.is_empty()
        && guard_rows.iter().all(|r| {
            let same =
                |y: &&Row| y.inner.seed == r.inner.seed && y.inner.order_ix == r.inner.order_ix;
            let peer = rows.iter().filter(same).find(|y| y.arm() == ZGD0);
            r.inner.setting == Setting::Zb
                && ZB_DOSES.contains(&r.arm())
                && r.inner.topology_unchanged
                && r.inner.init_ok
                && r.inner.within_cap()
                && r.codes_ok
                && r.cue.live <= r.cue.edges
                && r.cue.full <= r.cue.live
                && r.frozen_moved == 0
                && (r.writable_lines - cwd::expected_writable(r.dose, r.partner_lines)).abs() < 1e-9
                && (r.writable_lines > 0.0 || r.partner_moved == 0)
                && (r.writable_lines + 1e-9 < r.partner_lines || r.skipped == 0)
                && r.inner.glogs.len() == K
                && r.inner.glogs.iter().zip(&r.inner.logs).all(|(g, t)| {
                    g.stats.is_some_and(|s| {
                        s.overwrites == 0 && s.deferred == 0 && s.effective == t.ledger.writes
                    })
                })
                && r.inner.logs.len() == r.e_stage_writes.len()
                && r.inner
                    .logs
                    .iter()
                    .zip(&r.e_stage_writes)
                    .all(|(t, &w)| t.ledger.cap == w)
                && peer.is_some_and(|p| {
                    p.inner.order == r.inner.order
                        && p.e_stage_writes == r.e_stage_writes
                        && p.sigma == r.sigma
                        && p.partner_lines == r.partner_lines
                        && p.cue.edges == r.cue.edges
                        && p.cue.lines_per_task == r.cue.lines_per_task
                })
        })
        && seeds_of(rows) == seeds_of(guard_rows)
}

/// Blocked (refused) overwrites of a row, summed over stages.
#[must_use]
pub fn blocked(r: &Row) -> u64 {
    r.inner.blocked().unwrap_or(0)
}

/// Number of gates.
pub const N_GATES: usize = 7;

/// Design gates of one condition: G1 integrity (unchanged rows as cycle 27
/// and block rows by [`guard_rows_consistent`]); G2a the cycle 27 dose
/// manipulation clauses on the last-resort doses; G2b every block arm
/// refuses at least one overwrite in ≥ 50 % of held-out rows; G2c the block
/// doses bind (mean partner-line moves strictly increase along Zbd0..4);
/// G3 novel pairs; G4 leakage; G5 relative-forgetting denominators for S1,
/// Zgd4 and Zbd4.
#[must_use]
pub fn gates(c: &CondRows) -> [bool; N_GATES] {
    let (rows, gr) = (&c.rows, &c.guard_rows);
    let q = sel();
    let g1 = cwd::gates(rows).0 && guard_rows_consistent(rows, gr);
    let g2a =
        rows.iter().any(|r| r.inner.order_ix > 0) && cwd::g2_at(rows, K, FLIP).iter().all(|&b| b);
    let g2b = ZB_DOSES
        .iter()
        .all(|&x| cwd::frac(gr, q, x, &blocked) >= BLOCK_ACT_FRAC);
    let pm: Vec<f64> = ZB_DOSES
        .iter()
        .map(|&x| arm_mean(gr, x, &|r| r.partner_moved as f64))
        .collect();
    let g2c = !gr.is_empty() && pm.windows(2).all(|w| w[1] > w[0]);
    let g3 = cwd::median_cn(rows, K, FLIP) >= ns::MIN_NOVEL_PAIRS;
    let seeds = seeds_of(rows);
    let g4 = !seeds.is_empty()
        && seeds
            .iter()
            .all(|s| c.leakage.iter().any(|&(t, ok)| t == *s && ok));
    let g5 = cdt::rf_defined(&c.all_rows(), &[S1, ZGD4, ZBD4]);
    [g1, g2a, g2b, g2c, g3, g4, g5]
}

/// Is a condition evaluable (G1, G2a, G2b, G3, G4 pass)?
#[must_use]
pub fn evaluable(c: &CondRows) -> bool {
    let g = gates(c);
    g[0] && g[1] && g[2] && g[4] && g[5]
}

/// Number of pre-registered hypotheses.
pub const N_HYP: usize = 13;

fn find(all: &[CondRows], c: Cond) -> Option<&CondRows> {
    all.iter().find(|x| x.cond == c)
}

/// Raw forgetting non-inferior to S1: CI lower bound of FGT(S1) − FGT(x)
/// above −[`NONINF_MARGIN`].
fn noninf(d: &Paired) -> bool {
    d.n > 0 && d.ci_lo > -NONINF_MARGIN
}

/// Pre-registered readings H1..H13 (`None` = not evaluable).
#[must_use]
pub fn readings(all: &[CondRows]) -> [Option<bool>; N_HYP] {
    let ok = |c: Cond| find(all, c).filter(|x| evaluable(x));
    let ok_dose = |c: Cond| ok(c).filter(|x| gates(x)[3]);
    let ok_rf = |c: Cond| ok(c).filter(|x| gates(x)[6]);
    let rows = |c: &CondRows| c.all_rows();
    let (b, s2, s5) = (ok(Cond::Base), ok(Cond::Src256), ok(Cond::Src512));
    let primary = |c: &CondRows| {
        let r = rows(c);
        noninf(&raw_gap(&r, ZBD4)) && cwd::positive(&delta(&r, ZBD4, S1, &acc))
    };
    let zg_slope = |c: &CondRows| cwd::positive(&slope(&c.rows, &ZG_DOSES, 1.0, &acc));
    let mut h = [None; N_HYP];
    h[0] = b.map(|c| cwd::positive(&delta(&c.rows, ZGD4, S1, &ret_head)));
    h[1] = b.map(|c| {
        let d = raw_gap(&c.rows, ZGD4);
        d.n > 0 && d.ci_hi < -NONINF_MARGIN
    });
    h[2] = match (b, s5) {
        (Some(x), Some(y)) => Some(cwd::positive(&gap_change(&x.rows, &y.rows))),
        _ => None,
    };
    h[3] = b.map(|c| noninf(&raw_gap(&rows(c), ZBD4)));
    h[4] = b.map(|c| cwd::positive(&delta(&rows(c), ZBD4, S1, &acc)));
    h[5] = match (s2, s5) {
        (Some(x), Some(y)) => Some(primary(x) && primary(y)),
        _ => None,
    };
    h[6] = b.map(|c| cwd::positive(&delta(&rows(c), ZBD4, ZGD4, &ret_head)));
    h[7] = b.map(|c| {
        let d = delta(&rows(c), ZBD4, ZGD4, &acc);
        d.n > 0 && d.ci_lo > -NONINF_MARGIN
    });
    h[8] = match (b, s2, s5) {
        (Some(x), Some(y), Some(z)) => Some(zg_slope(x) && zg_slope(y) && zg_slope(z)),
        _ => None,
    };
    h[9] = b.map(|c| cwd::equivalent(&concavity(&c.rows, &ZG_DOSES, &acc), EQUIV_MARGIN));
    h[10] = ok_dose(Cond::Base).map(|c| cwd::positive(&slope(&c.guard_rows, &ZB_DOSES, 1.0, &acc)));
    h[11] = ok_dose(Cond::Base)
        .map(|c| cwd::equivalent(&concavity(&c.guard_rows, &ZB_DOSES, &acc), EQUIV_MARGIN));
    h[12] = ok_rf(Cond::Base).map(|c| cwd::positive(&cdt::rf_diff(&rows(c), ZBD4, S1)));
    h
}

/// CSV of every condition: this cycle's schema, the condition, the family,
/// the cycle 27 row schema and two extra columns (blocked overwrites,
/// RET_head).
#[must_use]
pub fn to_csv(all: &[CondRows]) -> String {
    let mut out = String::new();
    let mut header = false;
    for c in all {
        for (fam, rows) in [("lastresort", &c.rows), ("block", &c.guard_rows)] {
            let body = cwd::to_csv(rows, "GENERATED_SYNTHETIC");
            let mut lines = body.lines();
            if let Some(h) = lines.next() {
                if !header {
                    out.push_str("guard_schema,cond,family,");
                    out.push_str(h);
                    out.push_str(",blocked,ret_head\n");
                    header = true;
                }
            }
            for (line, r) in lines.zip(rows.iter()) {
                out.push_str("b888-4-cue-dose-guard-v1,");
                out.push_str(c.cond.name());
                out.push(',');
                out.push_str(fam);
                out.push(',');
                out.push_str(line);
                out.push_str(&format!(",{},{:.6}\n", blocked(r), ret_head(r)));
            }
        }
    }
    out
}

fn fmt_p(d: &Paired) -> String {
    format!(
        "{:+.3}±{:.3}[{:+.3},{:+.3}]{}/{}/{}",
        d.mean, d.sd, d.ci_lo, d.ci_hi, d.better, d.ties, d.worse
    )
}

fn yes(b: Option<bool>) -> &'static str {
    match b {
        Some(true) => "PASS",
        Some(false) => "FAIL",
        None => "NOT_EVALUABLE",
    }
}

/// Hypothesis statements (report order).
pub const HYP_NAMES: [&str; N_HYP] = [
    "H1 base RET_head(Zgd4)-RET_head(S1) > 0",
    "H2 base raw gap FGT(S1)-FGT(Zgd4) replicates (CI hi < -0.02)",
    "H3 raw gap closes with sources: gap(src512)-gap(base) > 0",
    "H4 base FGT(S1)-FGT(Zbd4) non-inferior (CI lo > -0.02) [primary]",
    "H5 base ACC_all(Zbd4)-ACC_all(S1) > 0 [primary]",
    "H6 H4 and H5 both hold in src256 and src512",
    "H7 base RET_head(Zbd4)-RET_head(Zgd4) > 0",
    "H8 base ACC_all(Zbd4)-ACC_all(Zgd4) non-inferior (CI lo > -0.02)",
    "H9 Zgd0..4 dose slope ACC_all > 0 in base, src256 and src512",
    "H10 base Zgd0..4 concavity ACC_all equivalent to 0 (+-0.01)",
    "H11 base Zbd0..4 dose slope ACC_all > 0",
    "H12 base Zbd0..4 concavity ACC_all equivalent to 0 (+-0.01)",
    "H13 base RF(S1)-RF(Zbd4) > 0",
];

/// Text report: gates, summaries, hypotheses, decomposition, secondary.
#[must_use]
pub fn report(all: &[CondRows]) -> String {
    use std::fmt::Write as _;
    let mut o = String::new();
    let (a, cn) = (Slice::All, Slice::Cn);
    let _ = writeln!(o, "gates: cond,G1_integrity,G2a_lastresort_dose,G2b_block_binds,G2c_block_dose_binds,G3_novel_pairs,G4_leakage,G5_rf_denominator,median_cn,partner_moves_Zgd0..4,partner_moves_Zbd0..4,blocked_frac_Zbd0..4,leak_free_seeds");
    for c in all {
        let g = gates(c);
        let pm = |rows: &[Row], arms: &[Arm; 5]| {
            arms.iter()
                .map(|&x| format!("{:.1}", arm_mean(rows, x, &|r| r.partner_moved as f64)))
                .collect::<Vec<_>>()
                .join("/")
        };
        let bf: Vec<String> = ZB_DOSES
            .iter()
            .map(|&x| format!("{:.2}", cwd::frac(&c.guard_rows, sel(), x, &blocked)))
            .collect();
        let _ = writeln!(
            o,
            "{},{},{},{},{},{},{},{},{},{},{},{},{}/{}",
            c.cond.name(),
            g[0],
            g[1],
            g[2],
            g[3],
            g[4],
            g[5],
            g[6],
            cwd::median_cn(&c.rows, K, FLIP),
            pm(&c.rows, &ZG_DOSES),
            pm(&c.guard_rows, &ZB_DOSES),
            bf.join("/"),
            c.leakage.iter().filter(|x| x.1).count(),
            c.leakage.len()
        );
    }
    let _ = writeln!(o, "summary(heldout orders, mean over rows): cond,arm,acc_all,acc_cn,learn_all,learn_head,ret_head,fgt_all,last_all,rel_forgetting_mean(seeds_undefined),charged_writes,cue_writes,blocked,partner_moved,skipped");
    for c in all {
        let r = c.all_rows();
        let mut arms: Vec<Arm> = cwd::ARMS.to_vec();
        arms.extend(ZB_DOSES);
        for x in arms {
            let v: Vec<f64> = seeds_of(&r)
                .into_iter()
                .filter_map(|s| cdt::rel_forgetting(&r, s, x))
                .collect();
            let und = seeds_of(&r).len() - v.len();
            let rfm = if v.is_empty() {
                f64::NAN
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            };
            let _ = writeln!(
                o,
                "{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}({}),{:.1},{:.1},{:.1},{:.1},{:.1}",
                c.cond.name(),
                short(x),
                arm_mean(&r, x, &acc),
                arm_mean(&r, x, &|y| y.inner.acc(cn, K)),
                arm_mean(&r, x, &|y| y.inner.learn(a, K)),
                arm_mean(&r, x, &cdt::learn_head),
                arm_mean(&r, x, &ret_head),
                arm_mean(&r, x, &cdt::fgt),
                arm_mean(&r, x, &|y| y.inner.last(a, K)),
                rfm,
                und,
                arm_mean(&r, x, &|y| y.inner.charged_writes() as f64),
                arm_mean(&r, x, &|y| y.inner.cue_writes() as f64),
                arm_mean(&r, x, &|y| blocked(y) as f64),
                arm_mean(&r, x, &|y| y.partner_moved as f64),
                arm_mean(&r, x, &|y| y.skipped as f64),
            );
        }
    }
    let h = readings(all);
    let line = |o: &mut String, id: &str, c: Cond, what: &str, d: &Paired| {
        let _ = writeln!(o, "  {id},{},{what},{}", c.name(), fmt_p(d));
    };
    let _ = writeln!(
        o,
        "hypotheses (k={K}, flip {FLIP}, held-out orders; per seed the mean over its held-out orders; mean±sd[95% bootstrap CI]better/ties/worse):"
    );
    let base = find(all, Cond::Base);
    for (i, n) in HYP_NAMES.iter().enumerate() {
        let _ = writeln!(o, "{n}: {}", yes(h[i]));
        for cr in all {
            let (c, r) = (cr.cond, cr.all_rows());
            let is_base = c == Cond::Base;
            match i {
                0 if is_base => line(
                    &mut o,
                    "dRET_head",
                    c,
                    "Zgd4-S1",
                    &delta(&r, ZGD4, S1, &ret_head),
                ),
                1 if is_base => line(
                    &mut o,
                    "fgt(S1)-fgt(Zgd4)",
                    c,
                    "Zgd4-S1",
                    &raw_gap(&r, ZGD4),
                ),
                2 if c == Cond::Src512 => {
                    if let Some(bc) = base {
                        line(
                            &mut o,
                            "gap(src512)-gap(base)",
                            c,
                            "Zgd4-S1",
                            &gap_change(&bc.rows, &cr.rows),
                        );
                    }
                }
                3 if is_base => line(
                    &mut o,
                    "fgt(S1)-fgt(Zbd4)",
                    c,
                    "Zbd4-S1",
                    &raw_gap(&r, ZBD4),
                ),
                4 if is_base => line(&mut o, "dACC_all", c, "Zbd4-S1", &delta(&r, ZBD4, S1, &acc)),
                5 if !is_base => {
                    line(
                        &mut o,
                        "fgt(S1)-fgt(Zbd4)",
                        c,
                        "Zbd4-S1",
                        &raw_gap(&r, ZBD4),
                    );
                    line(&mut o, "dACC_all", c, "Zbd4-S1", &delta(&r, ZBD4, S1, &acc));
                }
                6 if is_base => line(
                    &mut o,
                    "dRET_head",
                    c,
                    "Zbd4-Zgd4",
                    &delta(&r, ZBD4, ZGD4, &ret_head),
                ),
                7 if is_base => line(
                    &mut o,
                    "dACC_all",
                    c,
                    "Zbd4-Zgd4",
                    &delta(&r, ZBD4, ZGD4, &acc),
                ),
                8 => line(
                    &mut o,
                    "slope_acc_all",
                    c,
                    "Zgd0..4",
                    &slope(&cr.rows, &ZG_DOSES, 1.0, &acc),
                ),
                9 if is_base => line(
                    &mut o,
                    "concavity_acc_all",
                    c,
                    "Zgd0,2,4",
                    &concavity(&cr.rows, &ZG_DOSES, &acc),
                ),
                10 if is_base => line(
                    &mut o,
                    "slope_acc_all",
                    c,
                    "Zbd0..4",
                    &slope(&cr.guard_rows, &ZB_DOSES, 1.0, &acc),
                ),
                11 if is_base => line(
                    &mut o,
                    "concavity_acc_all",
                    c,
                    "Zbd0,2,4",
                    &concavity(&cr.guard_rows, &ZB_DOSES, &acc),
                ),
                12 if is_base => line(
                    &mut o,
                    "rf(S1)-rf(Zbd4)",
                    c,
                    "Zbd4-S1",
                    &cdt::rf_diff(&r, ZBD4, S1),
                ),
                _ => {}
            }
        }
    }
    let _ = writeln!(
        o,
        "decomposition (exact: fgt(S1)-fgt(x) = dRET_head - dLEARN_head; descriptive): cond,arm,dLEARN_head,dRET_head,fgt(S1)-fgt(x)"
    );
    for cr in all {
        let r = cr.all_rows();
        for x in [ZGD0, ZGD4, ZBD0, ZBD4] {
            let _ = writeln!(
                o,
                "  {},{},{},{},{}",
                cr.cond.name(),
                short(x),
                fmt_p(&delta(&r, x, S1, &cdt::learn_head)),
                fmt_p(&delta(&r, x, S1, &ret_head)),
                fmt_p(&raw_gap(&r, x))
            );
        }
    }
    let _ = writeln!(
        o,
        "secondary (descriptive, no pass/fail): id,cond,contrast,value"
    );
    for cr in all {
        let (c, r) = (cr.cond, cr.all_rows());
        line(
            &mut o,
            "fgt(S1)-fgt(Zgd4)",
            c,
            "Zgd4-S1",
            &raw_gap(&r, ZGD4),
        );
        line(
            &mut o,
            "fgt(S1)-fgt(Zbd4)",
            c,
            "Zbd4-S1",
            &raw_gap(&r, ZBD4),
        );
        line(&mut o, "dACC_all", c, "Zbd4-S1", &delta(&r, ZBD4, S1, &acc));
        line(&mut o, "dACC_all", c, "Zgd4-S1", &delta(&r, ZGD4, S1, &acc));
        line(
            &mut o,
            "dACC_all",
            c,
            "Zbd4-Zgd4",
            &delta(&r, ZBD4, ZGD4, &acc),
        );
        line(
            &mut o,
            "dLEARN_all",
            c,
            "Zbd4-Zgd4",
            &delta(&r, ZBD4, ZGD4, &|y| y.inner.learn(a, K)),
        );
        line(
            &mut o,
            "dRET_head",
            c,
            "Zbd4-Zgd4",
            &delta(&r, ZBD4, ZGD4, &ret_head),
        );
        line(
            &mut o,
            "dACC_cn",
            c,
            "Zbd4-S1",
            &delta(&r, ZBD4, S1, &|y| y.inner.acc(cn, K)),
        );
        line(
            &mut o,
            "rf(S1)-rf(Zgd4)",
            c,
            "Zgd4-S1",
            &cdt::rf_diff(&r, ZGD4, S1),
        );
        line(
            &mut o,
            "rf(S1)-rf(Zbd4)",
            c,
            "Zbd4-S1",
            &cdt::rf_diff(&r, ZBD4, S1),
        );
        line(
            &mut o,
            "slope_acc_all",
            c,
            "Zgd0..4",
            &slope(&cr.rows, &ZG_DOSES, 1.0, &acc),
        );
        line(
            &mut o,
            "slope_acc_all",
            c,
            "Zbd0..4",
            &slope(&cr.guard_rows, &ZB_DOSES, 1.0, &acc),
        );
        line(
            &mut o,
            "-slope_fgt_all",
            c,
            "Zbd0..4",
            &slope(&cr.guard_rows, &ZB_DOSES, -1.0, &cdt::fgt),
        );
        line(
            &mut o,
            "slope_acc_cn",
            c,
            "Zbd0..4",
            &slope(&cr.guard_rows, &ZB_DOSES, 1.0, &|y| y.inner.acc(cn, K)),
        );
        line(
            &mut o,
            "concavity_acc_all",
            c,
            "Zgd0,2,4",
            &concavity(&cr.rows, &ZG_DOSES, &acc),
        );
        line(
            &mut o,
            "concavity_acc_all",
            c,
            "Zbd0,2,4",
            &concavity(&cr.guard_rows, &ZB_DOSES, &acc),
        );
        if let Some(bc) = base.filter(|_| c != Cond::Base) {
            line(
                &mut o,
                "gap(cond)-gap(base)",
                c,
                "Zgd4-S1",
                &gap_change(&bc.rows, &cr.rows),
            );
        }
    }
    o
}

/// One label-free census line of a block arm (identity order): (arm, seed,
/// writable lines, charged writes, cue writes, blocked, skipped, frozen
/// moves, partner moves).
pub type GuardCensus = (Arm, u64, f64, [u64; 7]);

/// Label-free pilot of the block arms on the identity order: writes,
/// blocked overwrites, skips, frozen and partner-line moves. Training only:
/// no evaluation, no accuracy, no held-out order.
pub fn guard_census(cond: Cond, seeds: &[u64]) -> io::Result<Vec<GuardCensus>> {
    let design = cond.design();
    let spec = GenSpec {
        flip_pct: FLIP,
        ..design.spec
    };
    let mut out = Vec::new();
    for &seed in seeds {
        let fx = sg::fixture(None, &design, K, FLIP, seed)?;
        let a = cp::agreement(&fx)?;
        let order: Vec<usize> = (0..K).collect();
        let (elogs, _, _) = train_only(
            &fx,
            &spec,
            seed,
            Setting::Err,
            &order,
            &sg::free(K),
            &[],
            &[],
        )?;
        let curves: Vec<Vec<u64>> = elogs.iter().map(|t| t.ledger.cum.clone()).collect();
        let paced = cgd::paced_limits(&curves);
        for x in ZB_DOSES {
            let (f, _, masks, watch, w, _) = cwd::arm_fixture(&fx, x, &a, seed)?;
            let (logs, glogs, phases) =
                train_only(&f, &spec, seed, x.setting, &order, &paced, &masks, &watch)?;
            let (sk, fm, pm) = sum_phases(&phases);
            let sum = |g: &dyn Fn(&TaskLog) -> u64| logs.iter().map(g).sum::<u64>();
            let bl: u64 = glogs
                .iter()
                .filter_map(|g| g.stats.map(|s| s.blocked))
                .sum();
            out.push((
                x,
                seed,
                w,
                [
                    sum(&|t| t.ledger.writes),
                    sum(&|t| t.stats.cue_writes),
                    bl,
                    sk,
                    fm,
                    pm,
                    sum(&|t| t.ledger.refused),
                ],
            ));
        }
    }
    Ok(out)
}

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

/// Label-free census of the unchanged cycle 27 arms (identity order).
pub fn census(cond: Cond, seeds: &[u64]) -> io::Result<Vec<cwd::CensusLine>> {
    cwd::census(None, &cond.design(), K, FLIP, seeds)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The condition's graph shape and cue reservation with a short task.
    fn small(c: Cond) -> Design {
        let d = c.design();
        Design {
            spec: GenSpec {
                active: 4,
                train_patterns: 8,
                test_patterns: 24,
                rounds_a: 2,
                rounds_b: 2,
                ..d.spec
            },
            ..d
        }
    }

    /// One small grid (base and src512, seed 2, identity + reversed + 1
    /// random order), computed
    /// once per test binary.
    fn grid() -> &'static [CondRows] {
        static GRID: std::sync::OnceLock<Vec<CondRows>> = std::sync::OnceLock::new();
        GRID.get_or_init(|| run_with(&[Cond::Base, Cond::Src512], &[2], 2, 1, small).unwrap())
    }

    #[test]
    fn block_setting_is_new_and_outside_every_earlier_grid() {
        assert!(!cgd::SETTINGS.contains(&Setting::Zb));
        assert_eq!(Setting::Zb.guard(), Some(cgd::Guard::Block));
        assert_eq!(Setting::Zb.base(), Setting::Zg.base());
        assert!(Setting::Zb.cued());
        assert_eq!(Setting::Zb.name(), "E/E-cue0+block@paced100");
        assert!(!cwd::ARMS.contains(&ZBD4));
        assert_eq!(short(ZBD3), "Zbd3");
        assert_eq!(short(ZGD4), "Zgd4");
        for (z, g) in ZB_DOSES.iter().zip(ZG_DOSES) {
            assert_eq!((z.code, z.dose), (g.code, g.dose));
        }
    }

    #[test]
    fn conditions_change_only_the_number_of_sources() {
        let b = Cond::Base.design();
        assert_eq!(b, cwd::DESIGN);
        for (c, n) in [(Cond::Src256, 256), (Cond::Src512, 512)] {
            let d = c.design();
            assert_eq!(d.shape.sources, n);
            assert_eq!(
                Design {
                    shape: b.shape,
                    ..d
                },
                b
            );
        }
        assert_eq!(Cond::Src256.design(), cdt::Cond::Src256.design());
        for c in CONDS {
            assert_eq!(Cond::parse(c.name()).unwrap(), c);
        }
        assert!(Cond::parse("train64").is_err());
    }

    #[test]
    fn every_condition_builds_a_leakage_free_fixture_with_whole_quarter_doses() {
        // Fixture construction only (no learning, no accuracy).
        for c in CONDS {
            assert!(leakage_free(&c.design(), 1).unwrap(), "{}", c.name());
            let fx = sg::fixture(None, &c.design(), K, FLIP, 1).unwrap();
            let a = cp::agreement(&fx).unwrap();
            let sigma = cp::partner_map(cp::Code::Matched, &a, 1);
            let f = crate::cue_ownership::frozen_lines(&fx.cues, &sigma);
            assert!(f.iter().all(|l| l.len() == 4), "{}", c.name());
        }
    }

    #[test]
    fn block_rows_share_caps_orders_and_codes_with_the_unchanged_engine() {
        let all = grid();
        for c in all {
            assert_eq!(c.rows.len(), 3 * cwd::ARMS.len());
            assert_eq!(c.guard_rows.len(), 3 * ZB_DOSES.len());
            assert!(
                guard_rows_consistent(&c.rows, &c.guard_rows),
                "{}",
                c.cond.name()
            );
            assert!(gates(c)[0], "G1 {}", c.cond.name());
            assert!(gates(c)[5], "G4 {}", c.cond.name());
            // The unchanged rows are exactly the cycle 27 engine's rows.
            let d = small(c.cond);
            assert_eq!(c.rows, cwd::run_seed(None, &d, K, FLIP, 2, 1).unwrap());
            // Tampering breaks integrity.
            let mut t = c.guard_rows.clone();
            t[0].e_stage_writes[0] += 1;
            assert!(!guard_rows_consistent(&c.rows, &t));
            let mut t = c.guard_rows.clone();
            t[1].frozen_moved = 1;
            assert!(!guard_rows_consistent(&c.rows, &t));
            let mut t = c.guard_rows.clone();
            if let Some(s) = t[2].inner.glogs[0].stats.as_mut() {
                s.overwrites += 1;
            }
            assert!(!guard_rows_consistent(&c.rows, &t));
        }
    }

    #[test]
    fn raw_forgetting_splits_exactly_into_learning_and_retention() {
        for c in grid() {
            for r in c.all_rows() {
                let d = cdt::fgt(&r) - (cdt::learn_head(&r) - ret_head(&r));
                assert!(d.abs() < 1e-12, "{d}");
            }
            let r = c.all_rows();
            for x in [ZGD4, ZBD4] {
                let g = raw_gap(&r, x).mean;
                let dl = delta(&r, x, S1, &cdt::learn_head).mean;
                let dr = delta(&r, x, S1, &ret_head).mean;
                assert!((g - (dr - dl)).abs() < 1e-12);
            }
            assert_eq!(delta(&r, ZBD4, ZBD4, &acc).mean, 0.0);
        }
        let all = grid();
        assert_eq!(gap_change(&all[0].rows, &all[0].rows).mean, 0.0);
    }

    #[test]
    fn readings_need_their_conditions_and_csv_has_both_families() {
        let all = grid();
        let h = readings(all);
        // src256 is missing: H6 and H9 are not evaluable.
        assert!(h[5].is_none() && h[8].is_none());
        let only_src = &all[1..];
        assert!(readings(only_src).iter().all(Option::is_none));
        let csv = to_csv(all);
        let mut lines = csv.lines();
        assert!(lines
            .next()
            .unwrap()
            .starts_with("guard_schema,cond,family,"));
        let n = 2 * 3 * (cwd::ARMS.len() + ZB_DOSES.len());
        assert_eq!(lines.count(), n);
        assert!(csv.contains(",src512,block,"));
        assert!(report(all).contains("H13 base RF(S1)-RF(Zbd4) > 0"));
    }

    #[test]
    fn output_does_not_depend_on_the_thread_count() {
        let d = small(Cond::Base);
        let a = run_cond(&d, &[3, 4], 0, 1).unwrap();
        let b = run_cond(&d, &[3, 4], 0, 2).unwrap();
        assert_eq!(a, b);
    }
}

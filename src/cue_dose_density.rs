//! B888-4 cycle 30: **dose density** — is the cycle 29 H3 failure (the raw
//! forgetting gap FGT(S1) − FGT(Zgd4) *widens* when graph sources go
//! 128 → 256 → 512) caused by **edge dilution**? In cycle 29 the number of
//! sources grew while targets (32) and fan-in (8) stayed fixed, so the
//! edge density fell from 2 to 0.5 edges per source and most active inputs
//! of a pattern reached no target at all. This cycle re-runs the
//! **unchanged** cycle 29 engine ([`crate::cue_dose_guard::run_cond`]:
//! the cycle 27 last-resort arms plus the five `block`-guard dose arms) on
//! four conditions: `base` (128 sources, fan-in 8), `src512` (512, fan-in 8;
//! the cycle 29 dilution condition, replicated on new seeds), and two
//! density-matched conditions `d256` (256, fan-in 16) and `d512` (512,
//! fan-in 32) that keep 2 edges per source.
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026,
//! BANC/FlyWire consortium, see `docs/SOURCES.md`) is present on the build
//! machine; every condition uses the generated graph of
//! `consolidation_scale::generate_graph` and artificial labels. "Cue",
//! "dose", "guard", "density", "retention" and "forgetting" name parts of an
//! artificial benchmark on a Boolean graph with 4-bit weights, not claims
//! about synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/CUE_DOSE_DENSITY.md`
//! (committed and pushed before any learning run on the hypothesis seeds).

use crate::cue_dose_guard::{
    self as cdg, acc, delta, delta_seeded, raw_gap, ret_head, short, Cond, CondRows, ZBD4,
    ZB_DOSES, ZG_DOSES,
};
use crate::cue_dose_transfer as cdt;
use crate::cue_write_dose::{self as cwd, Arm, Row, S1, ZGD0, ZGD4};
use crate::error_gated::{paired, Paired};
use crate::novel_split::{Design, Slice};
use crate::seq_generalisation as sg;
use std::fmt::Write as _;
use std::io;

/// Hypothesis seeds (never used by an earlier B888-4 cycle).
pub const SEEDS: std::ops::RangeInclusive<u64> = 801..=840;
/// Label-free pilot seeds.
pub const PILOT_SEEDS: std::ops::RangeInclusive<u64> = 1..=5;
/// Pre-registered conditions, in CSV / report order.
pub const CONDS: [Cond; 4] = [Cond::Base, Cond::Src512, Cond::D256, Cond::D512];
/// Equivalence margin (concavity).
pub const EQUIV_MARGIN: f64 = cdg::EQUIV_MARGIN;
/// Non-inferiority margin (raw forgetting, gap change).
pub const NONINF_MARGIN: f64 = cdg::NONINF_MARGIN;
/// CSV schema tag.
pub const SCHEMA: &str = "b888-4-cue-dose-density-v1";

/// Parse a condition name of this cycle.
pub fn parse_cond(s: &str) -> io::Result<Cond> {
    CONDS
        .into_iter()
        .find(|c| c.name() == s)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "unknown condition"))
}

/// Edges per graph source of a condition (targets × fan-in / sources).
#[must_use]
pub fn edges_per_source(c: Cond) -> f64 {
    let d = c.design();
    (d.shape.targets * d.shape.fan_in) as f64 / d.shape.sources as f64
}

/// Run the pre-registered grid of the given conditions (the unchanged
/// cycle 29 engine per condition).
pub fn run(conds: &[Cond], seeds: &[u64], threads: usize) -> io::Result<Vec<CondRows>> {
    cdg::run_with(conds, seeds, threads, cdg::RANDOM_ORDERS, Cond::design)
}

/// Label-free coverage of one fixture: (share of graph sources with at
/// least one outgoing edge, mean out-degree of the active training-pattern
/// inputs, i.e. targets reached per active input). Uses the graph and the pattern inputs only (no
/// label, no learning, no accuracy).
pub fn coverage(design: &Design, seed: u64) -> io::Result<(f64, f64)> {
    let fx = sg::fixture(None, design, cdg::K, cdg::FLIP, seed)?;
    let g = &fx.graph;
    let deg = |i: usize| -> u64 {
        if i + 1 < g.offsets.len() {
            g.offsets[i + 1] - g.offsets[i]
        } else {
            0
        }
    };
    let n = design.shape.sources;
    let src = (0..n).filter(|&i| deg(i) > 0).count() as f64 / n as f64;
    let (mut hit, mut tot) = (0_u64, 0_u64);
    for p in fx.task.train.iter().flatten() {
        for &i in p {
            tot += 1;
            hit += deg(i);
        }
    }
    Ok((
        src,
        if tot == 0 {
            0.0
        } else {
            hit as f64 / tot as f64
        },
    ))
}

/// Per seed value (shared across conditions) `fb(seed) − fa(seed)` where
/// `fa`/`fb` are per-seed values of conditions `a`/`b`. The two conditions
/// are different fixtures; pairing by seed value is a fixed procedure.
fn cross(a: &[(u64, f64)], b: &[(u64, f64)]) -> Paired {
    let d: Vec<f64> = b
        .iter()
        .filter_map(|&(s, vb)| a.iter().find(|x| x.0 == s).map(|&(_, va)| vb - va))
        .collect();
    paired(&d)
}

/// Per seed raw gap FGT(S1) − FGT(Zgd4).
fn gap_seeded(rows: &[Row]) -> Vec<(u64, f64)> {
    delta_seeded(rows, S1, ZGD4, &cdt::fgt)
}

/// Per seed mean of `f` for arm `x` (held-out orders).
fn arm_seeded(rows: &[Row], x: Arm, f: &dyn Fn(&Row) -> f64) -> Vec<(u64, f64)> {
    cdg::seeds_of(rows)
        .into_iter()
        .filter_map(|s| Some((s, cdg::seed_mean(rows, s, x, f)?)))
        .collect()
}

/// Change of the raw gap from condition `a` to `b` (> 0: closes in `b`).
#[must_use]
pub fn gap_change(a: &CondRows, b: &CondRows) -> Paired {
    cross(&gap_seeded(&a.rows), &gap_seeded(&b.rows))
}

/// Change of the retention term RET_head(Zgd4) − RET_head(S1) from `a` to `b`.
#[must_use]
pub fn ret_term_change(a: &CondRows, b: &CondRows) -> Paired {
    cross(
        &delta_seeded(&a.rows, ZGD4, S1, &ret_head),
        &delta_seeded(&b.rows, ZGD4, S1, &ret_head),
    )
}

/// Change of LEARN_head of arm `x` from `a` to `b`.
#[must_use]
pub fn learn_change(a: &CondRows, b: &CondRows, x: Arm) -> Paired {
    cross(
        &arm_seeded(&a.rows, x, &cdt::learn_head),
        &arm_seeded(&b.rows, x, &cdt::learn_head),
    )
}

fn acc_cn(r: &Row) -> f64 {
    r.inner.acc(Slice::Cn, cdg::K)
}

fn find(all: &[CondRows], c: Cond) -> Option<&CondRows> {
    all.iter().find(|x| x.cond == c)
}

fn noninf(d: &Paired) -> bool {
    d.n > 0 && d.ci_lo > -NONINF_MARGIN
}

/// Number of pre-registered hypotheses.
pub const N_HYP: usize = 12;

/// Hypothesis statements (report order).
pub const HYP_NAMES: [&str; N_HYP] = [
    "H1 manipulation: LEARN_head(E) d512 - src512 > 0",
    "H2 replication: gap(src512)-gap(base) < 0 (CI hi < 0)",
    "H3 dilution [primary]: gap(d512)-gap(src512) > 0",
    "H4 closure at matched density: gap(d512)-gap(base) > 0",
    "H5 no widening at matched density: gap(d512)-gap(base) non-inferior (CI lo > -0.02)",
    "H6 retention term restored: dRET_head(Zgd4-S1) d512 - src512 > 0",
    "H7 supervised control: FGT(S1)-FGT(Zbd4) non-inferior and ACC_all(Zbd4)-ACC_all(S1) > 0 in base, src512, d256 and d512",
    "H8 novel pairs: ACC_cn(Zgd4)-ACC_cn(S1) > 0 in d512",
    "H9 novel pairs: ACC_cn(Zbd4)-ACC_cn(S1) > 0 in d512",
    "H10 Zgd0..4 dose slope ACC_all > 0 in base, src512, d256 and d512",
    "H11 d512 Zgd0..4 concavity ACC_all equivalent to 0 (+-0.01)",
    "H12 d512 Zbd0..4 dose slope ACC_all > 0",
];

/// Pre-registered readings H1..H12 (`None` = not evaluable: a needed
/// condition is missing or fails a gate it needs).
#[must_use]
pub fn readings(all: &[CondRows]) -> [Option<bool>; N_HYP] {
    let ok = |c: Cond| find(all, c).filter(|x| cdg::evaluable(x));
    let ok_dose = |c: Cond| ok(c).filter(|x| cdg::gates(x)[3]);
    let (b, s5, d2, d5) = (
        ok(Cond::Base),
        ok(Cond::Src512),
        ok(Cond::D256),
        ok(Cond::D512),
    );
    let all4 = [b, s5, d2, d5];
    let every = |f: &dyn Fn(&CondRows) -> bool| -> Option<bool> {
        all4.iter()
            .map(|c| c.map(f))
            .collect::<Option<Vec<bool>>>()
            .map(|v| v.iter().all(|&x| x))
    };
    let mut h = [None; N_HYP];
    h[0] = s5
        .zip(d5)
        .map(|(x, y)| cwd::positive(&learn_change(x, y, cwd::E)));
    h[1] = b.zip(s5).map(|(x, y)| {
        let d = gap_change(x, y);
        d.n > 0 && d.ci_hi < 0.0
    });
    h[2] = s5.zip(d5).map(|(x, y)| cwd::positive(&gap_change(x, y)));
    h[3] = b.zip(d5).map(|(x, y)| cwd::positive(&gap_change(x, y)));
    h[4] = b.zip(d5).map(|(x, y)| noninf(&gap_change(x, y)));
    h[5] = s5
        .zip(d5)
        .map(|(x, y)| cwd::positive(&ret_term_change(x, y)));
    h[6] = every(&|c| {
        let r = c.all_rows();
        noninf(&raw_gap(&r, ZBD4)) && cwd::positive(&delta(&r, ZBD4, S1, &acc))
    });
    h[7] = d5.map(|c| cwd::positive(&delta(&c.rows, ZGD4, S1, &acc_cn)));
    h[8] = d5.map(|c| cwd::positive(&delta(&c.all_rows(), ZBD4, S1, &acc_cn)));
    h[9] = every(&|c| cwd::positive(&cdg::slope(&c.rows, &ZG_DOSES, 1.0, &acc)));
    h[10] = d5.map(|c| cwd::equivalent(&cdg::concavity(&c.rows, &ZG_DOSES, &acc), EQUIV_MARGIN));
    h[11] = ok_dose(Cond::D512)
        .map(|c| cwd::positive(&cdg::slope(&c.guard_rows, &ZB_DOSES, 1.0, &acc)));
    h
}

/// CSV of every condition: this cycle's schema, the condition, the family,
/// the cycle 27 row schema, blocked overwrites and RET_head (same columns
/// as cycle 29 apart from the schema tag).
#[must_use]
pub fn to_csv(all: &[CondRows]) -> String {
    cdg::to_csv(all).replace("b888-4-cue-dose-guard-v1,", &format!("{SCHEMA},"))
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

/// Text report: gates, summaries, hypotheses, decomposition, secondary.
#[must_use]
pub fn report(all: &[CondRows]) -> String {
    let mut o = String::new();
    let (a, cn) = (Slice::All, Slice::Cn);
    let q = cwd::Sel {
        k: cdg::K,
        flip: cdg::FLIP,
        orders: cwd::Orders::Heldout,
    };
    let _ = writeln!(o, "gates: cond,G1_integrity,G2a_lastresort_dose,G2b_block_binds,G2c_block_dose_binds,G3_novel_pairs,G4_leakage,G5_rf_denominator,median_cn,partner_moves_Zgd0..4,partner_moves_Zbd0..4,blocked_frac_Zbd0..4,leak_free_seeds");
    for c in all {
        let g = cdg::gates(c);
        let pm = |rows: &[Row], arms: &[Arm; 5]| {
            arms.iter()
                .map(|&x| format!("{:.1}", cdg::arm_mean(rows, x, &|r| r.partner_moved as f64)))
                .collect::<Vec<_>>()
                .join("/")
        };
        let bf: Vec<String> = ZB_DOSES
            .iter()
            .map(|&x| format!("{:.2}", cwd::frac(&c.guard_rows, q, x, &cdg::blocked)))
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
            cwd::median_cn(&c.rows, cdg::K, cdg::FLIP),
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
        let seeds = cdg::seeds_of(&r);
        for x in arms {
            let v: Vec<f64> = seeds
                .iter()
                .filter_map(|&s| cdt::rel_forgetting(&r, s, x))
                .collect();
            let rfm = if v.is_empty() {
                f64::NAN
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            };
            let m = |f: &dyn Fn(&Row) -> f64| cdg::arm_mean(&r, x, f);
            let _ = writeln!(
                o,
                "{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}({}),{:.1},{:.1},{:.1},{:.1},{:.1}",
                c.cond.name(),
                short(x),
                m(&acc),
                m(&|y| y.inner.acc(cn, cdg::K)),
                m(&|y| y.inner.learn(a, cdg::K)),
                m(&cdt::learn_head),
                m(&ret_head),
                m(&cdt::fgt),
                m(&|y| y.inner.last(a, cdg::K)),
                rfm,
                seeds.len() - v.len(),
                m(&|y| y.inner.charged_writes() as f64),
                m(&|y| y.inner.cue_writes() as f64),
                m(&|y| cdg::blocked(y) as f64),
                m(&|y| y.partner_moved as f64),
                m(&|y| y.skipped as f64),
            );
        }
    }
    let h = readings(all);
    let get = |c: Cond| find(all, c);
    let line = |o: &mut String, id: &str, c: &str, what: &str, d: &Paired| {
        let _ = writeln!(o, "  {id},{c},{what},{}", fmt_p(d));
    };
    let _ = writeln!(
        o,
        "hypotheses (k={}, flip {}, held-out orders; per seed the mean over its held-out orders; mean±sd[95% bootstrap CI]better/ties/worse):",
        cdg::K,
        cdg::FLIP
    );
    let pair = |x: Cond, y: Cond| get(x).zip(get(y));
    for (i, n) in HYP_NAMES.iter().enumerate() {
        let _ = writeln!(o, "{n}: {}", yes(h[i]));
        match i {
            0 => {
                if let Some((x, y)) = pair(Cond::Src512, Cond::D512) {
                    line(
                        &mut o,
                        "learn_head(E)",
                        "d512-src512",
                        "E",
                        &learn_change(x, y, cwd::E),
                    );
                }
            }
            1 => {
                if let Some((x, y)) = pair(Cond::Base, Cond::Src512) {
                    line(
                        &mut o,
                        "gap(src512)-gap(base)",
                        "src512-base",
                        "Zgd4-S1",
                        &gap_change(x, y),
                    );
                }
            }
            2 => {
                if let Some((x, y)) = pair(Cond::Src512, Cond::D512) {
                    line(
                        &mut o,
                        "gap(d512)-gap(src512)",
                        "d512-src512",
                        "Zgd4-S1",
                        &gap_change(x, y),
                    );
                }
            }
            3 | 4 => {
                if let Some((x, y)) = pair(Cond::Base, Cond::D512) {
                    line(
                        &mut o,
                        "gap(d512)-gap(base)",
                        "d512-base",
                        "Zgd4-S1",
                        &gap_change(x, y),
                    );
                }
            }
            5 => {
                if let Some((x, y)) = pair(Cond::Src512, Cond::D512) {
                    line(
                        &mut o,
                        "dRET_head(Zgd4-S1)",
                        "d512-src512",
                        "Zgd4-S1",
                        &ret_term_change(x, y),
                    );
                }
            }
            6 => {
                for c in all {
                    let r = c.all_rows();
                    line(
                        &mut o,
                        "fgt(S1)-fgt(Zbd4)",
                        c.cond.name(),
                        "Zbd4-S1",
                        &raw_gap(&r, ZBD4),
                    );
                    line(
                        &mut o,
                        "dACC_all",
                        c.cond.name(),
                        "Zbd4-S1",
                        &delta(&r, ZBD4, S1, &acc),
                    );
                }
            }
            7 => {
                if let Some(c) = get(Cond::D512) {
                    line(
                        &mut o,
                        "dACC_cn",
                        "d512",
                        "Zgd4-S1",
                        &delta(&c.rows, ZGD4, S1, &acc_cn),
                    );
                }
            }
            8 => {
                if let Some(c) = get(Cond::D512) {
                    line(
                        &mut o,
                        "dACC_cn",
                        "d512",
                        "Zbd4-S1",
                        &delta(&c.all_rows(), ZBD4, S1, &acc_cn),
                    );
                }
            }
            9 => {
                for c in all {
                    line(
                        &mut o,
                        "slope_acc_all",
                        c.cond.name(),
                        "Zgd0..4",
                        &cdg::slope(&c.rows, &ZG_DOSES, 1.0, &acc),
                    );
                }
            }
            10 => {
                if let Some(c) = get(Cond::D512) {
                    line(
                        &mut o,
                        "concavity_acc_all",
                        "d512",
                        "Zgd0,2,4",
                        &cdg::concavity(&c.rows, &ZG_DOSES, &acc),
                    );
                }
            }
            11 => {
                if let Some(c) = get(Cond::D512) {
                    line(
                        &mut o,
                        "slope_acc_all",
                        "d512",
                        "Zbd0..4",
                        &cdg::slope(&c.guard_rows, &ZB_DOSES, 1.0, &acc),
                    );
                }
            }
            _ => {}
        }
    }
    let _ = writeln!(
        o,
        "decomposition (exact: fgt(S1)-fgt(x) = dRET_head - dLEARN_head; descriptive): cond,arm,dLEARN_head,dRET_head,fgt(S1)-fgt(x)"
    );
    for cr in all {
        let r = cr.all_rows();
        for x in [ZGD0, ZGD4, cdg::ZBD0, ZBD4] {
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
        let (c, r) = (cr.cond.name(), cr.all_rows());
        line(
            &mut o,
            "fgt(S1)-fgt(Zgd4)",
            c,
            "Zgd4-S1",
            &raw_gap(&r, ZGD4),
        );
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
            "dACC_cn",
            c,
            "Zgd4-S1",
            &delta(&r, ZGD4, S1, &acc_cn),
        );
        line(
            &mut o,
            "dACC_cn",
            c,
            "Zbd4-S1",
            &delta(&r, ZBD4, S1, &acc_cn),
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
            "Zbd0..4",
            &cdg::slope(&cr.guard_rows, &ZB_DOSES, 1.0, &acc),
        );
        line(
            &mut o,
            "concavity_acc_all",
            c,
            "Zgd0,2,4",
            &cdg::concavity(&cr.rows, &ZG_DOSES, &acc),
        );
    }
    for (x, y) in [
        (Cond::Base, Cond::D256),
        (Cond::Base, Cond::Src512),
        (Cond::D256, Cond::D512),
    ] {
        if let Some((a1, b1)) = pair(x, y) {
            let id = format!("{}-{}", y.name(), x.name());
            line(&mut o, "gap change", &id, "Zgd4-S1", &gap_change(a1, b1));
            line(
                &mut o,
                "dRET_head(Zgd4-S1) change",
                &id,
                "Zgd4-S1",
                &ret_term_change(a1, b1),
            );
        }
    }
    for (x, y) in [(Cond::Base, Cond::D512), (Cond::Base, Cond::Src512)] {
        if let Some((a1, b1)) = pair(x, y) {
            let id = format!("{}-{}", y.name(), x.name());
            line(
                &mut o,
                "learn_head(E) change",
                &id,
                "E",
                &learn_change(a1, b1, cwd::E),
            );
            line(
                &mut o,
                "dRET_head(Zgd4-S1) change",
                &id,
                "Zgd4-S1",
                &ret_term_change(a1, b1),
            );
        }
    }
    o
}

/// Label-free pilot text (identity order; training only, no accuracy, no
/// held-out order): coverage, last-resort and block census, leakage.
pub fn census_report(conds: &[Cond], seeds: &[u64]) -> io::Result<String> {
    let mut o = String::new();
    let _ = writeln!(o, "coverage(label-free; graph and pattern inputs only): cond,sources,fan_in,edges_per_source,seeds,share_sources_with_out_edge,mean_out_degree_of_active_train_inputs");
    for &c in conds {
        let d = c.design();
        let v = seeds
            .iter()
            .map(|&s| coverage(&d, s))
            .collect::<io::Result<Vec<_>>>()?;
        let n = v.len().max(1) as f64;
        let _ = writeln!(
            o,
            "{},{},{},{:.2},{},{:.3},{:.3}",
            c.name(),
            d.shape.sources,
            d.shape.fan_in,
            edges_per_source(c),
            v.len(),
            v.iter().map(|x| x.0).sum::<f64>() / n,
            v.iter().map(|x| x.1).sum::<f64>() / n
        );
    }
    let _ = writeln!(o, "census(identity order; training only, no accuracy computed, no held-out order drawn): cond,arm,seeds,writable_partner_lines,charged_writes,cue_writes,blocked,skipped,frozen_moved,partner_moved,refused");
    for &c in conds {
        let lines = cdg::census(c, seeds)?;
        for x in cwd::ARMS {
            let v: Vec<&cwd::CensusLine> = lines.iter().filter(|l| l.2 == x).collect();
            let n = v.len().max(1) as f64;
            let m = |i: usize| v.iter().map(|l| l.7[i].unwrap_or(0) as f64).sum::<f64>() / n;
            let _ = writeln!(
                o,
                "{},{},{},{:.2},{:.1},{:.1},-,{:.1},{:.1},{:.1},{:.1}",
                c.name(),
                x.short(),
                v.len(),
                v.iter().map(|l| l.6).sum::<f64>() / n,
                m(0),
                m(2),
                m(6),
                m(7),
                m(8),
                m(3),
            );
        }
        let gl = cdg::guard_census(c, seeds)?;
        for x in ZB_DOSES {
            let v: Vec<&cdg::GuardCensus> = gl.iter().filter(|l| l.0 == x).collect();
            let n = v.len().max(1) as f64;
            let m = |i: usize| v.iter().map(|l| l.3[i] as f64).sum::<f64>() / n;
            let _ = writeln!(
                o,
                "{},{},{},{:.2},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
                c.name(),
                short(x),
                v.len(),
                v.iter().map(|l| l.2).sum::<f64>() / n,
                m(0),
                m(1),
                m(2),
                m(3),
                m(4),
                m(5),
                m(6),
            );
        }
        let leak = seeds
            .iter()
            .map(|&s| cdg::leakage_free(&c.design(), s))
            .collect::<io::Result<Vec<bool>>>()?;
        let _ = writeln!(
            o,
            "leakage({}): leak_free_fixtures={}/{}",
            c.name(),
            leak.iter().filter(|&&b| b).count(),
            leak.len()
        );
    }
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consolidation_guard as cgd;
    use crate::cue_init as ci;
    use crate::cue_partner as cp;
    use crate::generalization_bench::GenSpec;
    use crate::plasticity::PlasticCircuit;
    use crate::task_sequence as ts;

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

    fn grid() -> &'static [CondRows] {
        static GRID: std::sync::OnceLock<Vec<CondRows>> = std::sync::OnceLock::new();
        GRID.get_or_init(|| cdg::run_with(&[Cond::Src512, Cond::D512], &[2], 2, 0, small).unwrap())
    }

    #[test]
    fn conditions_change_only_sources_and_fan_in_and_cycle29_grid_is_unchanged() {
        assert_eq!(cdg::CONDS, [Cond::Base, Cond::Src256, Cond::Src512]);
        assert!(cdg::Cond::parse("d512").is_err());
        let b = Cond::Base.design();
        assert_eq!(b, cwd::DESIGN);
        for (c, n, f) in [
            (Cond::Src512, 512, 8),
            (Cond::D256, 256, 16),
            (Cond::D512, 512, 32),
        ] {
            let d = c.design();
            assert_eq!(
                (d.shape.sources, d.shape.fan_in, d.shape.targets),
                (n, f, 32)
            );
            assert_eq!(
                Design {
                    shape: b.shape,
                    ..d
                },
                b
            );
        }
        assert_eq!(edges_per_source(Cond::Base), 2.0);
        assert_eq!(edges_per_source(Cond::D256), 2.0);
        assert_eq!(edges_per_source(Cond::D512), 2.0);
        assert_eq!(edges_per_source(Cond::Src512), 0.5);
        for c in CONDS {
            assert_eq!(parse_cond(c.name()).unwrap(), c);
        }
        assert!(parse_cond("src256").is_err());
    }

    #[test]
    fn every_condition_builds_a_leakage_free_fixture_with_four_frozen_lines() {
        // Fixture construction only (no learning, no accuracy).
        for c in CONDS {
            assert!(cdg::leakage_free(&c.design(), 1).unwrap(), "{}", c.name());
            let fx = sg::fixture(None, &c.design(), cdg::K, cdg::FLIP, 1).unwrap();
            let a = cp::agreement(&fx).unwrap();
            let sigma = cp::partner_map(cp::Code::Matched, &a, 1);
            let f = crate::cue_ownership::frozen_lines(&fx.cues, &sigma);
            assert!(f.iter().all(|l| l.len() == 4), "{}", c.name());
        }
    }

    #[test]
    fn density_matching_restores_input_coverage() {
        // Label-free structure only.
        let (s5, i5) = coverage(&Cond::Src512.design(), 1).unwrap();
        let (d5, j5) = coverage(&Cond::D512.design(), 1).unwrap();
        let (bs, bi) = coverage(&Cond::Base.design(), 1).unwrap();
        assert!(s5 < d5 && i5 < j5, "{s5} {d5} {i5} {j5}");
        assert!((0.0..=1.0).contains(&bs) && bi >= 1.0);
    }

    #[test]
    fn rows_come_from_the_unchanged_cycle29_engine_and_gates_hold() {
        // `run` is `cue_dose_guard::run_with` (whose rows are pinned to
        // `cue_write_dose::run_seed` and thread-count independence by the
        // cycle 29 tests); here only shape and integrity at fan-in 32.
        for c in grid() {
            assert_eq!(c.rows.len(), 2 * cwd::ARMS.len());
            assert_eq!(c.guard_rows.len(), 2 * ZB_DOSES.len());
            let gt = cdg::gates(c);
            assert!(gt[0] && gt[5], "{}", c.cond.name());
        }
    }

    #[test]
    fn cross_condition_contrasts_and_readings() {
        let all = grid();
        let (s5, d5) = (&all[0], &all[1]);
        assert_eq!(gap_change(s5, s5).mean, 0.0);
        assert_eq!(ret_term_change(d5, d5).mean, 0.0);
        let g = gap_change(s5, d5).mean;
        let ga = raw_gap(&s5.rows, ZGD4).mean;
        let gb = raw_gap(&d5.rows, ZGD4).mean;
        assert!((g - (gb - ga)).abs() < 1e-12);
        let h = readings(all);
        // base and d256 missing: H2, H4, H5, H7, H10 not evaluable.
        for i in [1, 3, 4, 6, 9] {
            assert!(h[i].is_none(), "H{}", i + 1);
        }
        let csv = to_csv(all);
        assert!(csv.lines().skip(1).all(|l| l.starts_with(SCHEMA)));
        assert_eq!(
            csv.lines().count(),
            1 + 2 * 2 * (cwd::ARMS.len() + ZB_DOSES.len())
        );
        assert!(csv.contains(",d512,block,"));
        let rep = report(all);
        assert!(rep.contains("H3 dilution [primary]"));
        assert!(rep.contains("d512,Zbd4,"));
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "v888-cue-dose-density-{tag}-{}",
            std::process::id()
        ))
    }

    #[test]
    fn checkpoint_resume_is_bit_exact_for_a_block_dose_arm_at_fan_in_32() {
        let d = small(Cond::D512);
        let (k, seed) = (cdg::K, 4);
        let spec = GenSpec {
            flip_pct: cdg::FLIP,
            ..d.spec
        };
        let base = sg::fixture(None, &d, k, cdg::FLIP, seed).unwrap();
        let a = cp::agreement(&base).unwrap();
        let order: Vec<usize> = (0..k).rev().collect();
        let e =
            cgd::run_setting(&base, &spec, seed, cgd::Setting::Err, &order, &sg::free(k)).unwrap();
        let curves: Vec<Vec<u64>> = e.logs.iter().map(|t| t.ledger.cum.clone()).collect();
        let lim = cgd::paced_limits(&curves);
        for x in [cdg::ZBD2] {
            let (fx, _, masks, watch, _, ok) = cwd::arm_fixture(&base, x, &a, seed).unwrap();
            assert!(ok);
            let g = &fx.graph;
            let s = x.setting;
            let gd = s.guard().unwrap();
            let (full, ph) = cgd::run_setting_scheduled_at(
                &fx,
                &spec,
                seed,
                s,
                &order,
                &lim,
                &masks,
                &watch,
                &[],
            )
            .unwrap();
            let mut c = ci::initial_circuit(&fx, &spec, s.base().arm()).unwrap();
            let mut dir = vec![0_i8; g.edges()];
            let mut stamp = vec![0_u8; g.edges()];
            for (i, &t) in order.iter().enumerate() {
                let n = ts::task_order(&spec, seed, t).len();
                let split = cgd::phase_split(n);
                let start = c.weights().to_vec();
                let mut cpt =
                    cgd::Checkpoint::start(&order, i, &lim[i], gd, &dir, &stamp, &start).unwrap();
                let mut sk = [0; 2];
                let mut ends = vec![split.max(1) / 2, split, split.saturating_add(1).min(n), n];
                ends.sort_unstable();
                ends.dedup();
                ends.retain(|&e| e > 0);
                for end in ends {
                    cgd::train_range_scheduled_at(
                        &mut c, &fx, &spec, seed, s, &lim[i], &mut cpt, end, &masks[t], split,
                        &mut sk,
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
                assert_eq!(sk, ph[i].skipped);
                cgd::consolidate(&mut dir, &mut stamp, i, &cpt.start, c.weights()).unwrap();
            }
            assert_eq!(c.weights(), &full.weights[..]);
        }
    }
}

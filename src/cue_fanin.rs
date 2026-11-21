//! B888-4 cycle 31: **fan-in sweep** — does raising the fan-in at a
//! **fixed** number of graph sources lower learning by itself? Cycle 30
//! (`docs/CUE_DOSE_DENSITY.md`, #189) tried to undo the cycle 29 dilution
//! by scaling fan-in with sources (d512: 512 sources, fan-in 32) and found
//! the opposite of the expected manipulation: learning fell further
//! (LEARN_head(E) d512 − src512 = −0.193), and the raw forgetting gap
//! FGT(S1) − FGT(Zgd4) "closed" only because every arm stopped learning.
//! Cycle 30 changed fan-in and sources together. This cycle separates them
//! with a 2 × 2 (sources 128 / 512 × fan-in 8 / 32) plus a fan-in midpoint
//! at 128 sources (fan-in 16), on the **unchanged** cycle 29 engine
//! ([`crate::cue_dose_guard::run_with`]: the cycle 27 last-resort arms plus
//! the five `block`-guard dose arms).
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026,
//! BANC/FlyWire consortium, see `docs/SOURCES.md`) is present on the build
//! machine; every condition uses the generated graph of
//! `consolidation_scale::generate_graph` and artificial labels. "Cue",
//! "dose", "guard", "fan-in", "load", "retention" and "forgetting" name
//! parts of an artificial benchmark on a Boolean graph with 4-bit weights,
//! not claims about synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/CUE_FANIN.md`
//! (committed and pushed before any learning run on the hypothesis seeds).

use crate::cue_dose_density as cdd;
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
pub const SEEDS: std::ops::RangeInclusive<u64> = 841..=880;
/// Label-free pilot seeds.
pub const PILOT_SEEDS: std::ops::RangeInclusive<u64> = 1..=5;
/// Pre-registered conditions, in CSV / report order.
pub const CONDS: [Cond; 5] = [Cond::Base, Cond::F16, Cond::F32, Cond::Src512, Cond::D512];
/// Equivalence margin (cycle 29/30 value; not used by a hypothesis here).
pub const EQUIV_MARGIN: f64 = cdg::EQUIV_MARGIN;
/// Non-inferiority margin (H5) and the two-sided equivalence margin of the
/// interaction contrast (H6).
pub const NONINF_MARGIN: f64 = cdg::NONINF_MARGIN;
/// CSV schema tag.
pub const SCHEMA: &str = "b888-4-cue-fanin-v1";

/// Parse a condition name of this cycle.
pub fn parse_cond(s: &str) -> io::Result<Cond> {
    CONDS
        .into_iter()
        .find(|c| c.name() == s)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "unknown condition"))
}

/// Run the pre-registered grid of the given conditions (the unchanged
/// cycle 29 engine per condition).
pub fn run(conds: &[Cond], seeds: &[u64], threads: usize) -> io::Result<Vec<CondRows>> {
    cdg::run_with(conds, seeds, threads, cdg::RANDOM_ORDERS, Cond::design)
}

/// Label-free target load of one fixture, over every training pattern of
/// every task: (mean number of (active input, target) edges per target,
/// mean share of targets reached by at least one active input, mean
/// Jaccard overlap of the reached-target sets of two training patterns of
/// the same task; pairs whose union is empty are skipped). Graph and
/// pattern inputs only: no label, no learning, no accuracy.
pub fn target_load(design: &Design, seed: u64) -> io::Result<(f64, f64, f64)> {
    let fx = sg::fixture(None, design, cdg::K, cdg::FLIP, seed)?;
    let g = &fx.graph;
    let n_src = design.shape.sources;
    let n_tgt = design.shape.targets;
    let reached = |p: &[usize]| -> (u64, Vec<bool>) {
        let mut hit = vec![false; n_tgt];
        let mut e = 0_u64;
        for &i in p {
            if i + 1 >= g.offsets.len() {
                continue;
            }
            for k in g.offsets[i]..g.offsets[i + 1] {
                let t = g.targets[k as usize] as usize;
                if let Some(h) = t.checked_sub(n_src).and_then(|j| hit.get_mut(j)) {
                    *h = true;
                    e += 1;
                }
            }
        }
        (e, hit)
    };
    let (mut load, mut share, mut np) = (0.0, 0.0, 0_u64);
    let (mut jac, mut nj) = (0.0, 0_u64);
    for task in &fx.task.train {
        let sets: Vec<Vec<bool>> = task
            .iter()
            .map(|p| {
                let (e, h) = reached(p);
                load += e as f64 / n_tgt as f64;
                share += h.iter().filter(|&&b| b).count() as f64 / n_tgt as f64;
                np += 1;
                h
            })
            .collect();
        for a in 0..sets.len() {
            for b in a + 1..sets.len() {
                let (mut i, mut u) = (0_u32, 0_u32);
                for (x, y) in sets[a].iter().zip(&sets[b]) {
                    i += u32::from(*x && *y);
                    u += u32::from(*x || *y);
                }
                if u > 0 {
                    jac += f64::from(i) / f64::from(u);
                    nj += 1;
                }
            }
        }
    }
    let div = |a: f64, n: u64| if n == 0 { 0.0 } else { a / n as f64 };
    Ok((div(load, np), div(share, np), div(jac, nj)))
}

/// Per seed value `fb(seed) − fa(seed)` (different fixtures; pairing by
/// seed value is a fixed procedure, not a claim that they are matched).
fn cross_seeded(a: &[(u64, f64)], b: &[(u64, f64)]) -> Vec<(u64, f64)> {
    b.iter()
        .filter_map(|&(s, vb)| a.iter().find(|x| x.0 == s).map(|&(_, va)| (s, vb - va)))
        .collect()
}

fn vals(v: &[(u64, f64)]) -> Vec<f64> {
    v.iter().map(|x| x.1).collect()
}

fn arm_seeded(rows: &[Row], x: Arm, f: &dyn Fn(&Row) -> f64) -> Vec<(u64, f64)> {
    cdg::seeds_of(rows)
        .into_iter()
        .filter_map(|s| Some((s, cdg::seed_mean(rows, s, x, f)?)))
        .collect()
}

fn learn_e(c: &CondRows) -> Vec<(u64, f64)> {
    arm_seeded(&c.rows, cwd::E, &cdt::learn_head)
}

/// Change of the learning term LEARN_head(Zgd4) − LEARN_head(S1) from
/// condition `a` to `b`.
#[must_use]
pub fn learn_term_change(a: &CondRows, b: &CondRows) -> Paired {
    let f = |c: &CondRows| delta_seeded(&c.rows, ZGD4, S1, &cdt::learn_head);
    paired(&vals(&cross_seeded(&f(a), &f(b))))
}

/// Fan-in × sources interaction on LEARN_head(E), per seed:
/// [LEARN(d512) − LEARN(src512)] − [LEARN(f32) − LEARN(base)].
#[must_use]
pub fn interaction(base: &CondRows, f32c: &CondRows, s512: &CondRows, d512: &CondRows) -> Paired {
    let hi = cross_seeded(&learn_e(s512), &learn_e(d512));
    let lo = cross_seeded(&learn_e(base), &learn_e(f32c));
    paired(&vals(&cross_seeded(&lo, &hi)))
}

fn neg(d: &Paired) -> bool {
    d.n > 0 && d.ci_hi < 0.0
}

fn noninf(d: &Paired) -> bool {
    d.n > 0 && d.ci_lo > -NONINF_MARGIN
}

fn acc_cn(r: &Row) -> f64 {
    r.inner.acc(Slice::Cn, cdg::K)
}

fn find(all: &[CondRows], c: Cond) -> Option<&CondRows> {
    all.iter().find(|x| x.cond == c)
}

/// Number of pre-registered hypotheses.
pub const N_HYP: usize = 12;

/// Hypothesis statements (report order).
pub const HYP_NAMES: [&str; N_HYP] = [
    "H1 fan-in alone [primary]: LEARN_head(E) f32 - base < 0 (CI hi < 0)",
    "H2 monotone in fan-in: LEARN_head(E) f16 - base and f32 - f16 both CI hi < 0",
    "H3 cycle 30 H1 replicates: LEARN_head(E) d512 - src512 CI hi < 0",
    "H4 sources at fan-in 8: LEARN_head(E) src512 - base CI hi < 0",
    "H5 sources at fan-in 32: LEARN_head(E) d512 - f32 non-inferior (CI lo > -0.02)",
    "H6 additivity: [(d512-src512)-(f32-base)] LEARN_head(E) CI inside +-0.02",
    "H7 raw gap closes with fan-in: gap(f32)-gap(base) > 0",
    "H8 closure is a learning collapse: dLEARN_head(Zgd4-S1) f32 - base CI hi < 0",
    "H9 retention term improves with fan-in: dRET_head(Zgd4-S1) f32 - base > 0",
    "H10 supervised control: FGT(S1)-FGT(Zbd4) non-inferior and ACC_all(Zbd4)-ACC_all(S1) > 0 in base, f16, f32, src512 and d512",
    "H11 generalisation at f32: ACC_all(Zgd4)-ACC_all(S1) > 0 and ACC_cn(Zgd4)-ACC_cn(S1) > 0",
    "H12 Zgd0..4 dose slope ACC_all > 0 in base, f16, f32, src512 and d512",
];

/// Pre-registered readings H1..H12 (`None` = not evaluable: a needed
/// condition is missing or fails a gate it needs).
#[must_use]
pub fn readings(all: &[CondRows]) -> [Option<bool>; N_HYP] {
    let ok = |c: Cond| find(all, c).filter(|x| cdg::evaluable(x));
    let (b, f1, f3, s5, d5) = (
        ok(Cond::Base),
        ok(Cond::F16),
        ok(Cond::F32),
        ok(Cond::Src512),
        ok(Cond::D512),
    );
    let le = |x: &CondRows, y: &CondRows| cdd::learn_change(x, y, cwd::E);
    let all5 = [b, f1, f3, s5, d5];
    let every = |f: &dyn Fn(&CondRows) -> bool| -> Option<bool> {
        all5.iter()
            .map(|c| c.map(f))
            .collect::<Option<Vec<bool>>>()
            .map(|v| v.iter().all(|&x| x))
    };
    let mut h = [None; N_HYP];
    h[0] = b.zip(f3).map(|(x, y)| neg(&le(x, y)));
    h[1] = b
        .zip(f1)
        .zip(f3)
        .map(|((x, y), z)| neg(&le(x, y)) && neg(&le(y, z)));
    h[2] = s5.zip(d5).map(|(x, y)| neg(&le(x, y)));
    h[3] = b.zip(s5).map(|(x, y)| neg(&le(x, y)));
    h[4] = f3.zip(d5).map(|(x, y)| noninf(&le(x, y)));
    h[5] = b.zip(f3).zip(s5).zip(d5).map(|(((w, x), y), z)| {
        let d = interaction(w, x, y, z);
        d.n > 0 && d.ci_lo > -NONINF_MARGIN && d.ci_hi < NONINF_MARGIN
    });
    h[6] = b
        .zip(f3)
        .map(|(x, y)| cwd::positive(&cdd::gap_change(x, y)));
    h[7] = b.zip(f3).map(|(x, y)| neg(&learn_term_change(x, y)));
    h[8] = b
        .zip(f3)
        .map(|(x, y)| cwd::positive(&cdd::ret_term_change(x, y)));
    h[9] = every(&|c| {
        let r = c.all_rows();
        noninf(&raw_gap(&r, ZBD4)) && cwd::positive(&delta(&r, ZBD4, S1, &acc))
    });
    h[10] = f3.map(|c| {
        cwd::positive(&delta(&c.rows, ZGD4, S1, &acc))
            && cwd::positive(&delta(&c.rows, ZGD4, S1, &acc_cn))
    });
    h[11] = every(&|c| cwd::positive(&cdg::slope(&c.rows, &ZG_DOSES, 1.0, &acc)));
    h
}

/// CSV of every condition: this cycle's schema, the condition, the family,
/// the cycle 27 row schema, blocked overwrites and RET_head (same columns
/// as cycles 29–30 apart from the schema tag).
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
    let _ = writeln!(o, "gates: cond,G1_integrity,G2a_lastresort_dose,G2b_block_binds,G2c_block_dose_binds,G3_novel_pairs,G4_leakage,G5_rf_denominator,median_cn,blocked_frac_Zbd0..4,leak_free_seeds");
    for c in all {
        let g = cdg::gates(c);
        let bf: Vec<String> = ZB_DOSES
            .iter()
            .map(|&x| format!("{:.2}", cwd::frac(&c.guard_rows, q, x, &cdg::blocked)))
            .collect();
        let _ = writeln!(
            o,
            "{},{},{},{},{},{},{},{},{},{},{}/{}",
            c.cond.name(),
            g[0],
            g[1],
            g[2],
            g[3],
            g[4],
            g[5],
            g[6],
            cwd::median_cn(&c.rows, cdg::K, cdg::FLIP),
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
    let pair = |x: Cond, y: Cond| get(x).zip(get(y));
    let line = |o: &mut String, id: &str, c: &str, what: &str, d: &Paired| {
        let _ = writeln!(o, "  {id},{c},{what},{}", fmt_p(d));
    };
    let le_line = |o: &mut String, x: Cond, y: Cond| {
        if let Some((p, q)) = pair(x, y) {
            line(
                o,
                "learn_head(E)",
                &format!("{}-{}", y.name(), x.name()),
                "E",
                &cdd::learn_change(p, q, cwd::E),
            );
        }
    };
    let _ = writeln!(
        o,
        "hypotheses (k={}, flip {}, held-out orders; per seed the mean over its held-out orders; mean±sd[95% bootstrap CI]better/ties/worse):",
        cdg::K,
        cdg::FLIP
    );
    for (i, n) in HYP_NAMES.iter().enumerate() {
        let _ = writeln!(o, "{n}: {}", yes(h[i]));
        match i {
            0 => le_line(&mut o, Cond::Base, Cond::F32),
            1 => {
                le_line(&mut o, Cond::Base, Cond::F16);
                le_line(&mut o, Cond::F16, Cond::F32);
            }
            2 => le_line(&mut o, Cond::Src512, Cond::D512),
            3 => le_line(&mut o, Cond::Base, Cond::Src512),
            4 => le_line(&mut o, Cond::F32, Cond::D512),
            5 => {
                if let (Some(w), Some(x), Some(y), Some(z)) = (
                    get(Cond::Base),
                    get(Cond::F32),
                    get(Cond::Src512),
                    get(Cond::D512),
                ) {
                    line(
                        &mut o,
                        "learn_head(E) interaction",
                        "(d512-src512)-(f32-base)",
                        "E",
                        &interaction(w, x, y, z),
                    );
                }
            }
            6 => {
                if let Some((x, y)) = pair(Cond::Base, Cond::F32) {
                    line(
                        &mut o,
                        "gap(f32)-gap(base)",
                        "f32-base",
                        "Zgd4-S1",
                        &cdd::gap_change(x, y),
                    );
                }
            }
            7 => {
                if let Some((x, y)) = pair(Cond::Base, Cond::F32) {
                    line(
                        &mut o,
                        "dLEARN_head(Zgd4-S1) change",
                        "f32-base",
                        "Zgd4-S1",
                        &learn_term_change(x, y),
                    );
                }
            }
            8 => {
                if let Some((x, y)) = pair(Cond::Base, Cond::F32) {
                    line(
                        &mut o,
                        "dRET_head(Zgd4-S1) change",
                        "f32-base",
                        "Zgd4-S1",
                        &cdd::ret_term_change(x, y),
                    );
                }
            }
            9 => {
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
            10 => {
                if let Some(c) = get(Cond::F32) {
                    line(
                        &mut o,
                        "dACC_all",
                        "f32",
                        "Zgd4-S1",
                        &delta(&c.rows, ZGD4, S1, &acc),
                    );
                    line(
                        &mut o,
                        "dACC_cn",
                        "f32",
                        "Zgd4-S1",
                        &delta(&c.rows, ZGD4, S1, &acc_cn),
                    );
                }
            }
            11 => {
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
            "rf(S1)-rf(Zgd4)",
            c,
            "Zgd4-S1",
            &cdt::rf_diff(&r, ZGD4, S1),
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
        (Cond::Base, Cond::F16),
        (Cond::F16, Cond::F32),
        (Cond::Src512, Cond::D512),
        (Cond::F32, Cond::D512),
    ] {
        if let Some((p, q)) = pair(x, y) {
            let id = format!("{}-{}", y.name(), x.name());
            line(&mut o, "gap change", &id, "Zgd4-S1", &cdd::gap_change(p, q));
            line(
                &mut o,
                "dLEARN_head(Zgd4-S1) change",
                &id,
                "Zgd4-S1",
                &learn_term_change(p, q),
            );
            line(
                &mut o,
                "dRET_head(Zgd4-S1) change",
                &id,
                "Zgd4-S1",
                &cdd::ret_term_change(p, q),
            );
        }
    }
    o
}

/// Label-free pilot text (identity order; training only, no accuracy, no
/// held-out order): coverage, target load, last-resort and block census,
/// leakage.
pub fn census_report(conds: &[Cond], seeds: &[u64]) -> io::Result<String> {
    let mut o = String::new();
    let _ = writeln!(o, "coverage(label-free; graph and pattern inputs only): cond,sources,fan_in,edges_per_source,seeds,share_sources_with_out_edge,mean_out_degree_of_active_train_inputs,active_edges_per_target,share_targets_reached,jaccard_reached_targets_same_task");
    for &c in conds {
        let d = c.design();
        let cov = seeds
            .iter()
            .map(|&s| cdd::coverage(&d, s))
            .collect::<io::Result<Vec<_>>>()?;
        let tl = seeds
            .iter()
            .map(|&s| target_load(&d, s))
            .collect::<io::Result<Vec<_>>>()?;
        let n = seeds.len().max(1) as f64;
        let _ = writeln!(
            o,
            "{},{},{},{:.2},{},{:.3},{:.3},{:.3},{:.3},{:.3}",
            c.name(),
            d.shape.sources,
            d.shape.fan_in,
            cdd::edges_per_source(c),
            seeds.len(),
            cov.iter().map(|x| x.0).sum::<f64>() / n,
            cov.iter().map(|x| x.1).sum::<f64>() / n,
            tl.iter().map(|x| x.0).sum::<f64>() / n,
            tl.iter().map(|x| x.1).sum::<f64>() / n,
            tl.iter().map(|x| x.2).sum::<f64>() / n,
        );
    }
    // The cycle 30 census without its coverage block (printed above).
    let rest = cdd::census_report(conds, seeds)?;
    o.push_str(rest.find("census(").map_or("", |i| &rest[i..]));
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generalization_bench::GenSpec;

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

    #[test]
    fn fan_in_conditions_change_only_fan_in_and_earlier_grids_are_unchanged() {
        assert_eq!(cdg::CONDS, [Cond::Base, Cond::Src256, Cond::Src512]);
        assert_eq!(
            cdd::CONDS,
            [Cond::Base, Cond::Src512, Cond::D256, Cond::D512]
        );
        assert!(cdg::Cond::parse("f32").is_err());
        assert!(cdd::parse_cond("f16").is_err());
        let b = Cond::Base.design();
        for (c, f, eps) in [(Cond::F16, 16, 4.0), (Cond::F32, 32, 8.0)] {
            let d = c.design();
            assert_eq!(
                (d.shape.sources, d.shape.fan_in, d.shape.targets),
                (128, f, 32)
            );
            assert_eq!(
                Design {
                    shape: b.shape,
                    ..d
                },
                b
            );
            assert_eq!(cdd::edges_per_source(c), eps);
        }
        for c in CONDS {
            assert_eq!(parse_cond(c.name()).unwrap(), c);
            assert!(cdg::leakage_free(&c.design(), 1).unwrap(), "{}", c.name());
        }
        assert!(parse_cond("d256").is_err());
    }

    #[test]
    fn target_load_rises_with_fan_in_at_fixed_sources() {
        // Label-free structure only (graph and pattern inputs).
        let (l8, s8, j8) = target_load(&Cond::Base.design(), 1).unwrap();
        let (l32, s32, j32) = target_load(&Cond::F32.design(), 1).unwrap();
        assert!(
            l8 < l32 && s8 < s32 && j8 < j32,
            "{l8} {l32} {s8} {s32} {j8} {j32}"
        );
        assert!((0.0..=1.0).contains(&s32) && (0.0..=1.0).contains(&j32));
    }

    #[test]
    fn readings_and_report_on_a_tiny_grid() {
        let all = cdg::run_with(&[Cond::Base, Cond::F32], &[2], 2, 0, small).unwrap();
        for c in &all {
            assert!(cdg::gates(c)[0], "{}", c.cond.name());
        }
        let (b, f) = (&all[0], &all[1]);
        assert_eq!(learn_term_change(b, b).mean, 0.0);
        let d = interaction(b, f, b, f);
        assert_eq!(d.mean, 0.0);
        let h = readings(&all);
        // f16, src512, d512 missing.
        for i in [1, 2, 3, 4, 5, 9, 11] {
            assert!(h[i].is_none(), "H{}", i + 1);
        }
        let csv = to_csv(&all);
        assert!(csv.lines().skip(1).all(|l| l.starts_with(SCHEMA)));
        assert!(csv.contains(",f32,block,"));
        let rep = report(&all);
        assert!(rep.contains("H1 fan-in alone [primary]"));
        assert!(rep.contains("f32,Zbd4,"));
    }
}

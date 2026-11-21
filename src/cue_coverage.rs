//! B888-4 cycle 32: **edge coverage** — is the fan-in learning loss of
//! cycle 31 (`docs/CUE_FANIN.md`, #190: uncapped E LEARN_head 0.987 →
//! 0.909 when the fan-in goes 8 → 32 at 128 sources) caused by TEST pairs
//! whose active edges were never trained in the current task? Under the OR
//! rule a target with `m` active edges is negative only if all `m` edges
//! are negative, so a correct negative needs every one of its `m` edges to
//! have been written in the task; higher fan-in raises `m`. This cycle
//! raises edge coverage without changing the graph or the number of
//! presentations: 4× the distinct training patterns per task with 1/4 of
//! the rounds (`Cond::BaseC`, `Cond::F32C`), on the **unchanged** cycle 29
//! engine ([`crate::cue_dose_guard::run_with`]).
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026,
//! BANC/FlyWire consortium, see `docs/SOURCES.md`) is present on the build
//! machine; every condition uses the generated graph of
//! `consolidation_scale::generate_graph` and artificial labels. "Cue",
//! "dose", "guard", "coverage", "retention" and "forgetting" name parts of
//! an artificial benchmark on a Boolean graph with 4-bit weights, not claims
//! about synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/CUE_COVERAGE.md`
//! (committed and pushed before any learning run on the hypothesis seeds).

use crate::cue_dose_density as cdd;
use crate::cue_dose_guard::{
    self as cdg, acc, delta, raw_gap, ret_head, short, Cond, CondRows, ZBD4, ZB_DOSES, ZG_DOSES,
};
use crate::cue_dose_transfer as cdt;
use crate::cue_fanin as cfi;
use crate::cue_write_dose::{self as cwd, Arm, Row, S1, ZGD0, ZGD4};
use crate::error_gated::{paired, Paired};
use crate::novel_split::{Class, Design, Slice};
use crate::seq_generalisation as sg;
use crate::task_sequence as ts;
use std::fmt::Write as _;
use std::io;

/// Hypothesis seeds (never used by an earlier B888-4 cycle).
pub const SEEDS: std::ops::RangeInclusive<u64> = 881..=920;
/// Label-free pilot seeds.
pub const PILOT_SEEDS: std::ops::RangeInclusive<u64> = 1..=5;
/// Pre-registered conditions, in CSV / report order.
pub const CONDS: [Cond; 4] = [Cond::Base, Cond::F32, Cond::BaseC, Cond::F32C];
/// Non-inferiority margin (H2, H8; cycle 29–31 value).
pub const NONINF_MARGIN: f64 = cdg::NONINF_MARGIN;
/// CSV schema tag.
pub const SCHEMA: &str = "b888-4-cue-coverage-v1";

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

/// Label-free edge coverage of one fixture, over every task `j` and every
/// TEST (pattern, reached target) pair classified against task `j`'s
/// training set: (share of pairs all of whose active edges occur in some
/// training pattern of task `j` — classes Seen and Compositional, i.e.
/// not UnseenEdge —, mean number of active edges per pair, share of pairs
/// with ≥ 2 active edges, share of pairs whose exact configuration was
/// trained — class Seen). Graph and pattern inputs only: the task split's
/// valence is used by nothing here (only `classify`, which reads edges and
/// configurations); no label, no learning, no accuracy.
pub fn edge_coverage(design: &Design, seed: u64) -> io::Result<(f64, f64, f64, f64)> {
    let fx = sg::fixture(None, design, cdg::K, cdg::FLIP, seed)?;
    let g = &fx.graph;
    let (mut cov, mut m, mut multi, mut seen, mut n) = (0_u64, 0_u64, 0_u64, 0_u64, 0_u64);
    for split in &fx.splits {
        for p in &fx.task.test {
            for (t, conf) in ts::configurations(g, p) {
                n += 1;
                m += conf.len() as u64;
                multi += u64::from(conf.len() >= 2);
                match split.classify(t, &conf) {
                    Class::Seen => {
                        cov += 1;
                        seen += 1;
                    }
                    Class::Compositional => cov += 1,
                    Class::UnseenEdge => {}
                }
            }
        }
    }
    let div = |a: u64| if n == 0 { 0.0 } else { a as f64 / n as f64 };
    Ok((div(cov), div(m), div(multi), div(seen)))
}

/// Per seed covered share of a condition.
pub fn covered_seeded(c: Cond, seeds: &[u64]) -> io::Result<Vec<(u64, f64)>> {
    let d = c.design();
    seeds
        .iter()
        .map(|&s| Ok((s, edge_coverage(&d, s)?.0)))
        .collect()
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

/// Coverage × fan-in interaction on LEARN_head(E), per seed:
/// [LEARN(f32c) − LEARN(f32)] − [LEARN(basec) − LEARN(base)].
#[must_use]
pub fn interaction(base: &CondRows, f32: &CondRows, basec: &CondRows, f32c: &CondRows) -> Paired {
    let hi = cross_seeded(&learn_e(f32), &learn_e(f32c));
    let lo = cross_seeded(&learn_e(base), &learn_e(basec));
    paired(&vals(&cross_seeded(&lo, &hi)))
}

/// Per seed covered-share change from `a` to `b` (label-free).
pub fn coverage_change(a: Cond, b: Cond, seeds: &[u64]) -> io::Result<Paired> {
    Ok(paired(&vals(&cross_seeded(
        &covered_seeded(a, seeds)?,
        &covered_seeded(b, seeds)?,
    ))))
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
pub const N_HYP: usize = 11;

/// Hypothesis statements (report order).
pub const HYP_NAMES: [&str; N_HYP] = [
    "H1 coverage restores learning at fan-in 32 [primary]: LEARN_head(E) f32c - f32 > 0",
    "H2 coverage removes the fan-in loss: LEARN_head(E) f32c - basec non-inferior (CI lo > -0.02)",
    "H3 coverage helps more at fan-in 32: [(f32c-f32)-(basec-base)] LEARN_head(E) > 0",
    "H4 cycle 31 H1 replicates: LEARN_head(E) f32 - base CI hi < 0",
    "H5 manipulation check (label-free): covered share f32c - f32 > 0 and basec - base > 0",
    "H6 learning term recovers: dLEARN_head(Zgd4-S1) f32c - f32 > 0",
    "H7 retention term recovers: dRET_head(Zgd4-S1) f32c - f32 > 0",
    "H8 supervised control at f32c: FGT(S1)-FGT(Zbd4) non-inferior and ACC_all(Zbd4)-ACC_all(S1) > 0",
    "H9 generalisation at f32c: ACC_all(Zgd4)-ACC_all(S1) > 0 and ACC_cn(Zgd4)-ACC_cn(S1) > 0",
    "H10 cycle 31 H10 failure replicates at f32: FGT(S1)-FGT(Zbd4) CI hi < 0",
    "H11 Zgd0..4 dose slope ACC_all > 0 at f32c",
];

/// Pre-registered readings H1..H11 (`None` = not evaluable: a needed
/// condition is missing or fails a gate it needs; H1–H3 are also not
/// evaluable if the H5 manipulation check fails). `cov5` is the H5 reading
/// (computed label-free from the same seeds; `None` if a condition is
/// missing).
#[must_use]
pub fn readings_with(all: &[CondRows], cov5: Option<bool>) -> [Option<bool>; N_HYP] {
    let ok = |c: Cond| find(all, c).filter(|x| cdg::evaluable(x));
    let (b, f, bc, fc) = (
        ok(Cond::Base),
        ok(Cond::F32),
        ok(Cond::BaseC),
        ok(Cond::F32C),
    );
    let le = |x: &CondRows, y: &CondRows| cdd::learn_change(x, y, cwd::E);
    let mc = cov5 == Some(true);
    let mut h = [None; N_HYP];
    h[0] = f
        .zip(fc)
        .filter(|_| mc)
        .map(|(x, y)| cwd::positive(&le(x, y)));
    h[1] = bc.zip(fc).filter(|_| mc).map(|(x, y)| noninf(&le(x, y)));
    h[2] = b
        .zip(f)
        .zip(bc)
        .zip(fc)
        .filter(|_| mc)
        .map(|(((w, x), y), z)| cwd::positive(&interaction(w, x, y, z)));
    h[3] = b.zip(f).map(|(x, y)| neg(&le(x, y)));
    h[4] = cov5;
    h[5] = f
        .zip(fc)
        .map(|(x, y)| cwd::positive(&cfi::learn_term_change(x, y)));
    h[6] = f
        .zip(fc)
        .map(|(x, y)| cwd::positive(&cdd::ret_term_change(x, y)));
    h[7] = fc.map(|c| {
        let r = c.all_rows();
        noninf(&raw_gap(&r, ZBD4)) && cwd::positive(&delta(&r, ZBD4, S1, &acc))
    });
    h[8] = fc.map(|c| {
        cwd::positive(&delta(&c.rows, ZGD4, S1, &acc))
            && cwd::positive(&delta(&c.rows, ZGD4, S1, &acc_cn))
    });
    h[9] = f.map(|c| neg(&raw_gap(&c.all_rows(), ZBD4)));
    h[10] = fc.map(|c| cwd::positive(&cdg::slope(&c.rows, &ZG_DOSES, 1.0, &acc)));
    h
}

/// H5 manipulation check on the seeds of the run (label-free).
pub fn manipulation_check(all: &[CondRows]) -> io::Result<Option<bool>> {
    let has = |c: Cond| find(all, c).is_some();
    if !CONDS.iter().all(|&c| has(c)) {
        return Ok(None);
    }
    let seeds = find(all, Cond::Base).map_or_else(Vec::new, |c| cdg::seeds_of(&c.rows));
    Ok(Some(
        cwd::positive(&coverage_change(Cond::F32, Cond::F32C, &seeds)?)
            && cwd::positive(&coverage_change(Cond::Base, Cond::BaseC, &seeds)?),
    ))
}

/// Pre-registered readings with the H5 check computed from the run's seeds.
pub fn readings(all: &[CondRows]) -> io::Result<[Option<bool>; N_HYP]> {
    Ok(readings_with(all, manipulation_check(all)?))
}

/// CSV of every condition: this cycle's schema, the condition, the family,
/// the cycle 27 row schema, blocked overwrites and RET_head (same columns
/// as cycles 29–31 apart from the schema tag).
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

/// Text report: gates, label-free coverage, summaries, hypotheses,
/// decomposition.
pub fn report(all: &[CondRows]) -> io::Result<String> {
    let mut o = String::new();
    let a = Slice::All;
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
    let _ = writeln!(o, "edge coverage (label-free, run seeds): cond,train_patterns,rounds,covered_share,active_edges_per_pair,share_pairs_m>=2,seen_share");
    for c in all {
        let d = c.cond.design();
        let seeds = cdg::seeds_of(&c.rows);
        let v = seeds
            .iter()
            .map(|&s| edge_coverage(&d, s))
            .collect::<io::Result<Vec<_>>>()?;
        let n = v.len().max(1) as f64;
        let _ = writeln!(
            o,
            "{},{},{},{:.3},{:.3},{:.3},{:.3}",
            c.cond.name(),
            d.spec.train_patterns,
            d.spec.rounds_a,
            v.iter().map(|x| x.0).sum::<f64>() / n,
            v.iter().map(|x| x.1).sum::<f64>() / n,
            v.iter().map(|x| x.2).sum::<f64>() / n,
            v.iter().map(|x| x.3).sum::<f64>() / n,
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
                m(&acc_cn),
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
    let h = readings(all)?;
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
    let seeds = get(Cond::Base).map_or_else(Vec::new, |c| cdg::seeds_of(&c.rows));
    let _ = writeln!(
        o,
        "hypotheses (k={}, flip {}, held-out orders; per seed the mean over its held-out orders; mean±sd[95% bootstrap CI]better/ties/worse):",
        cdg::K,
        cdg::FLIP
    );
    for (i, n) in HYP_NAMES.iter().enumerate() {
        let _ = writeln!(o, "{n}: {}", yes(h[i]));
        match i {
            0 => le_line(&mut o, Cond::F32, Cond::F32C),
            1 => le_line(&mut o, Cond::BaseC, Cond::F32C),
            2 => {
                if let (Some(w), Some(x), Some(y), Some(z)) = (
                    get(Cond::Base),
                    get(Cond::F32),
                    get(Cond::BaseC),
                    get(Cond::F32C),
                ) {
                    line(
                        &mut o,
                        "learn_head(E) interaction",
                        "(f32c-f32)-(basec-base)",
                        "E",
                        &interaction(w, x, y, z),
                    );
                }
            }
            3 => le_line(&mut o, Cond::Base, Cond::F32),
            4 => {
                if h[4].is_some() {
                    for (x, y) in [(Cond::F32, Cond::F32C), (Cond::Base, Cond::BaseC)] {
                        line(
                            &mut o,
                            "covered_share",
                            &format!("{}-{}", y.name(), x.name()),
                            "-",
                            &coverage_change(x, y, &seeds)?,
                        );
                    }
                }
            }
            5 => {
                if let Some((x, y)) = pair(Cond::F32, Cond::F32C) {
                    line(
                        &mut o,
                        "dLEARN_head(Zgd4-S1) change",
                        "f32c-f32",
                        "Zgd4-S1",
                        &cfi::learn_term_change(x, y),
                    );
                }
            }
            6 => {
                if let Some((x, y)) = pair(Cond::F32, Cond::F32C) {
                    line(
                        &mut o,
                        "dRET_head(Zgd4-S1) change",
                        "f32c-f32",
                        "Zgd4-S1",
                        &cdd::ret_term_change(x, y),
                    );
                }
            }
            7 | 9 => {
                let c = if i == 7 { Cond::F32C } else { Cond::F32 };
                if let Some(cr) = get(c) {
                    let r = cr.all_rows();
                    line(
                        &mut o,
                        "fgt(S1)-fgt(Zbd4)",
                        c.name(),
                        "Zbd4-S1",
                        &raw_gap(&r, ZBD4),
                    );
                    line(
                        &mut o,
                        "dACC_all",
                        c.name(),
                        "Zbd4-S1",
                        &delta(&r, ZBD4, S1, &acc),
                    );
                }
            }
            8 => {
                if let Some(c) = get(Cond::F32C) {
                    line(
                        &mut o,
                        "dACC_all",
                        "f32c",
                        "Zgd4-S1",
                        &delta(&c.rows, ZGD4, S1, &acc),
                    );
                    line(
                        &mut o,
                        "dACC_cn",
                        "f32c",
                        "Zgd4-S1",
                        &delta(&c.rows, ZGD4, S1, &acc_cn),
                    );
                }
            }
            10 => {
                if let Some(c) = get(Cond::F32C) {
                    line(
                        &mut o,
                        "slope_acc_all",
                        "f32c",
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
        "secondary (descriptive, every condition): id,cond,arms,mean±sd[CI]b/t/w"
    );
    for cr in all {
        let r = cr.all_rows();
        let c = cr.cond.name();
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
            "dACC_cn",
            c,
            "Zgd4-S1",
            &delta(&r, ZGD4, S1, &acc_cn),
        );
        line(
            &mut o,
            "slope_acc_all",
            c,
            "Zgd0..4",
            &cdg::slope(&cr.rows, &ZG_DOSES, 1.0, &acc),
        );
        line(&mut o, "dACC_all", c, "E-S1", &delta(&r, cwd::E, S1, &acc));
    }
    le_line(&mut o, Cond::Base, Cond::BaseC);
    if let Some((x, y)) = pair(Cond::Base, Cond::F32) {
        line(
            &mut o,
            "gap change",
            "f32-base",
            "Zgd4-S1",
            &cdd::gap_change(x, y),
        );
    }
    if let Some((x, y)) = pair(Cond::F32, Cond::F32C) {
        line(
            &mut o,
            "gap change",
            "f32c-f32",
            "Zgd4-S1",
            &cdd::gap_change(x, y),
        );
    }
    Ok(o)
}

/// Label-free pilot text (identity order; training only, no accuracy, no
/// held-out order): edge coverage, then the cycle 31 census (coverage,
/// target load, last-resort and block census, leakage).
pub fn census_report(conds: &[Cond], seeds: &[u64]) -> io::Result<String> {
    let mut o = String::new();
    let _ = writeln!(o, "edge coverage(label-free; graph and pattern inputs only): cond,sources,fan_in,train_patterns,rounds,presentations_per_task,seeds,covered_share,active_edges_per_pair,share_pairs_m>=2,seen_share");
    for &c in conds {
        let d = c.design();
        let v = seeds
            .iter()
            .map(|&s| edge_coverage(&d, s))
            .collect::<io::Result<Vec<_>>>()?;
        let n = v.len().max(1) as f64;
        let _ = writeln!(
            o,
            "{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3}",
            c.name(),
            d.shape.sources,
            d.shape.fan_in,
            d.spec.train_patterns,
            d.spec.rounds_a,
            d.spec.train_patterns * d.spec.rounds_a as usize,
            seeds.len(),
            v.iter().map(|x| x.0).sum::<f64>() / n,
            v.iter().map(|x| x.1).sum::<f64>() / n,
            v.iter().map(|x| x.2).sum::<f64>() / n,
            v.iter().map(|x| x.3).sum::<f64>() / n,
        );
    }
    for (x, y) in [(Cond::F32, Cond::F32C), (Cond::Base, Cond::BaseC)] {
        if conds.contains(&x) && conds.contains(&y) {
            let _ = writeln!(
                o,
                "covered_share {}-{} (pilot seeds, label-free): {}",
                y.name(),
                x.name(),
                fmt_p(&coverage_change(x, y, seeds)?)
            );
        }
    }
    o.push_str(&cfi::census_report(conds, seeds)?);
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generalization_bench::GenSpec;

    fn small(c: Cond) -> Design {
        let d = c.design();
        let f = if c.more_patterns() { 2 } else { 1 };
        Design {
            spec: GenSpec {
                active: 4,
                train_patterns: 4 * f,
                test_patterns: 16,
                rounds_a: 2 / f as u32,
                rounds_b: 2 / f as u32,
                ..d.spec
            },
            ..d
        }
    }

    #[test]
    fn coverage_conditions_change_only_patterns_and_rounds() {
        assert_eq!(cdg::CONDS, [Cond::Base, Cond::Src256, Cond::Src512]);
        assert!(cdg::Cond::parse("f32c").is_err());
        assert!(cfi::parse_cond("basec").is_err());
        for (c, p) in [(Cond::BaseC, Cond::Base), (Cond::F32C, Cond::F32)] {
            let (d, e) = (c.design(), p.design());
            assert_eq!(d.shape, e.shape);
            assert_eq!(
                (d.spec.train_patterns, d.spec.rounds_a, d.spec.rounds_b),
                (128, 2, 2)
            );
            assert_eq!(
                d.spec.train_patterns * d.spec.rounds_a as usize,
                e.spec.train_patterns * e.spec.rounds_a as usize
            );
            assert_eq!(
                GenSpec {
                    train_patterns: e.spec.train_patterns,
                    rounds_a: e.spec.rounds_a,
                    rounds_b: e.spec.rounds_b,
                    ..d.spec
                },
                e.spec
            );
            assert!(cdg::leakage_free(&d, 1).unwrap(), "{}", c.name());
        }
        for c in CONDS {
            assert_eq!(parse_cond(c.name()).unwrap(), c);
        }
        assert!(parse_cond("f16").is_err());
    }

    #[test]
    fn more_patterns_raise_label_free_edge_coverage() {
        let (cb, mb, _, _) = edge_coverage(&Cond::F32.design(), 1).unwrap();
        let (cc, mc, _, _) = edge_coverage(&Cond::F32C.design(), 1).unwrap();
        assert!(cc > cb, "{cb} {cc}");
        // Same graph; the TEST set is redrawn (it follows the training
        // patterns in the pattern stream), so edges per pair only match
        // approximately.
        assert!((mb - mc).abs() < 0.2, "{mb} {mc}");
        let (_, m8, _, _) = edge_coverage(&Cond::Base.design(), 1).unwrap();
        assert!(m8 < mb && m8 < mc);
    }

    #[test]
    fn readings_and_report_on_a_tiny_grid() {
        let all = cdg::run_with(&CONDS, &[2], 4, 0, small).unwrap();
        for c in &all {
            assert!(cdg::gates(c)[0], "{}", c.cond.name());
        }
        let (b, f) = (&all[0], &all[1]);
        assert_eq!(interaction(b, f, b, f).mean, 0.0);
        let h = readings_with(&all[..2], None);
        for i in [0, 1, 2, 4, 5, 6, 7, 8, 10] {
            assert!(h[i].is_none(), "H{}", i + 1);
        }
        let h = readings_with(&all, Some(false));
        assert!(h[0].is_none() && h[1].is_none() && h[2].is_none());
        let csv = to_csv(&all);
        assert!(csv.lines().skip(1).all(|l| l.starts_with(SCHEMA)));
        assert!(csv.contains(",f32c,block,"));
        let rep = report(&all).unwrap();
        assert!(rep.contains("H1 coverage restores learning"));
        assert!(rep.contains("f32c,Zbd4,"));
    }
}

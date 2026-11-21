//! B888-4 cycle 28: **dose transfer** — does the cycle 27 partner-write
//! dose effect generalise beyond its fixture, does it hold up on forgetting
//! against the supervised control S1 at equal write budget, and why was the
//! cycle 27 dose curve linear (its H4 failed)?
//!
//! Cycle 27 (#186, `docs/CUE_WRITE_DOSE.md`) found at k = 16, flip 100 that
//! held-out accuracy rises **linearly** with the number of writable partner
//! lines of the matched share-2 cue code (H1 slope +0.026, H4 concavity
//! +0.000), and that every dosed arm beats S1 on ACC but forgets more in raw
//! terms. This cycle re-runs the **unchanged** cycle 27 engine
//! ([`crate::cue_write_dose::run_grid_threads`], same ten arms, same
//! constant dose masks, same `paced100` equal budget) on **fresh seeds**
//! 721..760 in three pre-registered fixture conditions (the cycle 27
//! design, twice the training patterns per task, twice the graph sources)
//! and evaluates pre-registered hypotheses on generalisation of the dose
//! effect, raw and relative forgetting against S1, and the additivity of
//! disjoint partner-line halves (the diagnosis of the linear curve).
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026,
//! BANC/FlyWire consortium, see `docs/SOURCES.md`) is present on the build
//! machine; every condition uses the generated graph of
//! `consolidation_scale::generate_graph` and artificial labels. "Cue",
//! "dose", "partner", "forgetting" and "transfer" name parts of an
//! artificial benchmark on a Boolean graph with 4-bit weights, not claims
//! about synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/CUE_DOSE_TRANSFER.md`
//! (committed and pushed before any learning run on the hypothesis seeds).

use crate::cue_write_dose::{
    self as cwd, Arm, Orders, Row, Sel, DOSES, S1, ZGD0, ZGD2, ZGD2C, ZGD4,
};
use crate::error_gated::{paired, Paired};
use crate::generalization_bench::{GenSpec, GraphShape};
use crate::novel_split::{self as ns, Design, Slice};
use crate::seq_generalisation as sg;
use std::collections::HashSet;
use std::io;

fn bad(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Run length (the cycle 27 primary k; `|F_j|` = 4, so quarter doses are
/// whole lines).
pub const K: usize = 16;
/// Label flip between consecutive tasks (the cycle 27 primary flip).
pub const FLIP: u32 = 100;
/// Hypothesis seeds (never used by an earlier B888-4 cycle).
pub const SEEDS: std::ops::RangeInclusive<u64> = 721..=760;
/// Label-free pilot seeds.
pub const PILOT_SEEDS: std::ops::RangeInclusive<u64> = 1..=5;
/// Seeded random held-out orders (plus the reversed order).
pub const RANDOM_ORDERS: usize = 3;
/// Equivalence margin (H10, H11, H12).
pub const EQUIV_MARGIN: f64 = 0.01;
/// Non-inferiority margin on raw forgetting (H5).
pub const NONINF_MARGIN: f64 = 0.02;
/// Minimum above-chance learning (balanced accuracy − 0.5) per seed for the
/// relative-forgetting ratio to be defined (gate G5).
pub const MIN_RF_DEN: f64 = 0.05;
/// Chance level of the balanced accuracy.
pub const CHANCE: f64 = 0.5;

/// Pre-registered fixture conditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cond {
    /// The cycle 27 design (128 sources, 32 training patterns per task).
    Base,
    /// Twice the training patterns per task (64).
    Train64,
    /// Twice the graph sources (256; targets and fan-in unchanged).
    Src256,
}

/// Conditions in CSV / report order.
pub const CONDS: [Cond; 3] = [Cond::Base, Cond::Train64, Cond::Src256];

impl Cond {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Train64 => "train64",
            Self::Src256 => "src256",
        }
    }
    /// Parse a condition name.
    pub fn parse(s: &str) -> io::Result<Self> {
        CONDS
            .into_iter()
            .find(|c| c.name() == s)
            .ok_or_else(|| bad("unknown condition"))
    }
    /// Design of the condition (only one field differs from cycle 27).
    #[must_use]
    pub fn design(self) -> Design {
        let d = cwd::DESIGN;
        match self {
            Self::Base => d,
            Self::Train64 => Design {
                spec: GenSpec {
                    train_patterns: 64,
                    ..d.spec
                },
                ..d
            },
            Self::Src256 => Design {
                shape: GraphShape {
                    sources: 256,
                    ..d.shape
                },
                ..d
            },
        }
    }
}

/// Rows of one condition.
#[derive(Clone, Debug, PartialEq)]
pub struct CondRows {
    pub cond: Cond,
    pub rows: Vec<Row>,
    /// Seeds whose fixture has no TEST pattern equal to a training pattern
    /// of any task (gate G4), with the check result.
    pub leakage: Vec<(u64, bool)>,
}

/// Leakage check of one fixture: no TEST pattern (as a set of active
/// inputs) equals any training pattern of any task.
pub fn leakage_free(cond: Cond, seed: u64) -> io::Result<bool> {
    leakage_free_in(&cond.design(), seed)
}

fn leakage_free_in(design: &Design, seed: u64) -> io::Result<bool> {
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

/// Run the pre-registered grid of the given conditions (unchanged cycle 27
/// engine, k = 16, flip 100, identity + reversed + 3 seeded orders).
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
            let rows = cwd::run_grid_threads(None, &d, &[K], &[FLIP], seeds, n_random, threads)?;
            let leakage = seeds
                .iter()
                .map(|&s| Ok((s, leakage_free_in(&d, s)?)))
                .collect::<io::Result<Vec<_>>>()?;
            Ok(CondRows {
                cond,
                rows,
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

fn seeds_of(rows: &[Row]) -> Vec<u64> {
    let mut s: Vec<u64> = rows.iter().map(|r| r.inner.seed).collect();
    s.sort_unstable();
    s.dedup();
    s
}

/// Mean of `f` over the held-out rows of arm `x` and `seed`.
fn seed_mean(rows: &[Row], seed: u64, x: Arm, f: &dyn Fn(&Row) -> f64) -> Option<f64> {
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

/// FGT_all of a row (mean over tasks 1..k−1 of R[j][j] − R[k][j]).
#[must_use]
pub fn fgt(r: &Row) -> f64 {
    r.inner.forgetting(Slice::All, K)
}

/// Learning of the first k − 1 tasks (the tasks FGT is measured on):
/// mean over j < k − 1 of R[j][j].
#[must_use]
pub fn learn_head(r: &Row) -> f64 {
    let k = K as f64;
    (k * r.inner.learn(Slice::All, K) - r.inner.last(Slice::All, K)) / (k - 1.0)
}

/// Relative forgetting of arm `x` on one seed: mean FGT_all over its
/// held-out orders divided by the mean above-chance learning of the same
/// tasks (`learn_head` − 0.5). `None` if the arm is missing or the
/// denominator is below [`MIN_RF_DEN`].
#[must_use]
pub fn rel_forgetting(rows: &[Row], seed: u64, x: Arm) -> Option<f64> {
    let f = seed_mean(rows, seed, x, &fgt)?;
    let den = seed_mean(rows, seed, x, &learn_head)? - CHANCE;
    (den >= MIN_RF_DEN).then(|| f / den)
}

/// Gate G5: the relative-forgetting denominator is defined for every seed
/// of the given arms.
#[must_use]
pub fn rf_defined(rows: &[Row], arms: &[Arm]) -> bool {
    let seeds = seeds_of(rows);
    !seeds.is_empty()
        && seeds
            .iter()
            .all(|&s| arms.iter().all(|&x| rel_forgetting(rows, s, x).is_some()))
}

/// Per seed RF(base) − RF(x) (> 0: x forgets a smaller share of what it
/// learned).
#[must_use]
pub fn rf_diff(rows: &[Row], x: Arm, base: Arm) -> Paired {
    let d: Vec<f64> = seeds_of(rows)
        .into_iter()
        .filter_map(|s| Some(rel_forgetting(rows, s, base)? - rel_forgetting(rows, s, x)?))
        .collect();
    paired(&d)
}

/// Per seed interaction of the two disjoint partner-line halves:
/// f(Zgd4) − f(Zgd2) − f(Zgd2c) + f(Zgd0) (0 if the halves add up).
#[must_use]
pub fn interaction(rows: &[Row], f: &dyn Fn(&Row) -> f64) -> Paired {
    let d: Vec<f64> = seeds_of(rows)
        .into_iter()
        .filter_map(|s| {
            Some(
                seed_mean(rows, s, ZGD4, f)?
                    - seed_mean(rows, s, ZGD2, f)?
                    - seed_mean(rows, s, ZGD2C, f)?
                    + seed_mean(rows, s, ZGD0, f)?,
            )
        })
        .collect();
    paired(&d)
}

/// Design gates of one condition: (G1 integrity, G2 manipulation, G3 novel
/// pairs, G4 leakage, G5 relative-forgetting denominator for S1 and Zgd4).
#[must_use]
pub fn gates(c: &CondRows) -> [bool; 5] {
    let rows = &c.rows;
    let g1 = cwd::gates(rows).0;
    let g2 =
        rows.iter().any(|r| r.inner.order_ix > 0) && cwd::g2_at(rows, K, FLIP).iter().all(|&b| b);
    let g3 = cwd::median_cn(rows, K, FLIP) >= ns::MIN_NOVEL_PAIRS;
    let seeds = seeds_of(rows);
    let g4 = !seeds.is_empty()
        && seeds
            .iter()
            .all(|s| c.leakage.iter().any(|&(t, ok)| t == *s && ok));
    let g5 = rf_defined(rows, &[S1, ZGD4]);
    [g1, g2, g3, g4, g5]
}

/// Number of pre-registered hypotheses.
pub const N_HYP: usize = 12;

fn find(all: &[CondRows], c: Cond) -> Option<&CondRows> {
    all.iter().find(|x| x.cond == c)
}

/// Is a condition evaluable (G1–G4 pass)?
#[must_use]
pub fn evaluable(c: &CondRows) -> bool {
    gates(c)[..4].iter().all(|&b| b)
}

/// Pre-registered readings H1..H12 (`None` = not evaluable: a needed
/// condition is missing or fails G1–G4; H6 / H7 also need G5).
#[must_use]
pub fn readings(all: &[CondRows]) -> [Option<bool>; N_HYP] {
    let q = sel();
    let ok = |c: Cond| find(all, c).filter(|x| evaluable(x));
    let rf_ok = |c: Cond| ok(c).filter(|x| gates(x)[4]);
    let acc = |r: &Row| r.inner.acc(Slice::All, K);
    let slope_acc = |c: &CondRows| cwd::positive(&cwd::slope(&c.rows, q, 1.0, &acc));
    let slope_fgt = |c: &CondRows| cwd::positive(&cwd::slope(&c.rows, q, -1.0, &fgt));
    let cmp = |c: &CondRows, s: Slice| cwd::compare(&c.rows, q, K, s, ZGD4, S1);
    let rf = |c: &CondRows| cwd::positive(&rf_diff(&c.rows, ZGD4, S1));
    let inter = |c: &CondRows| cwd::equivalent(&interaction(&c.rows, &acc), EQUIV_MARGIN);
    let both = |a: Option<&CondRows>, b: Option<&CondRows>, f: &dyn Fn(&CondRows) -> bool| {
        Some(f(a?) && f(b?))
    };
    let (b, t, s) = (ok(Cond::Base), ok(Cond::Train64), ok(Cond::Src256));
    let mut h = [None; N_HYP];
    h[0] = b.map(slope_acc);
    h[1] = t.map(slope_acc);
    h[2] = s.map(slope_acc);
    h[3] = both(b, t, &slope_fgt).and_then(|x| Some(x && slope_fgt(s?)));
    h[4] = b.map(|c| {
        let d = cmp(c, Slice::All).forget;
        d.n > 0 && d.ci_lo > -NONINF_MARGIN
    });
    h[5] = rf_ok(Cond::Base).map(rf);
    h[6] = both(rf_ok(Cond::Train64), rf_ok(Cond::Src256), &rf);
    h[7] = both(t, s, &|c| cmp(c, Slice::All).passes());
    let cn = |c: &CondRows| cwd::positive(&cmp(c, Slice::Cn).acc);
    h[8] = both(b, t, &cn).and_then(|x| Some(x && cn(s?)));
    h[9] = b.map(|c| cwd::equivalent(&cwd::concavity(&c.rows, q, &acc), EQUIV_MARGIN));
    h[10] = b.map(inter);
    h[11] = both(t, s, &inter);
    h
}

/// CSV of every condition: the cycle 27 row schema prefixed by this
/// cycle's schema and the condition.
#[must_use]
pub fn to_csv(all: &[CondRows]) -> String {
    let mut out = String::new();
    for (i, c) in all.iter().enumerate() {
        let body = cwd::to_csv(&c.rows, "GENERATED_SYNTHETIC");
        for (j, line) in body.lines().enumerate() {
            if j == 0 {
                if i == 0 {
                    out.push_str("transfer_schema,cond,");
                    out.push_str(line);
                    out.push('\n');
                }
                continue;
            }
            out.push_str("b888-4-cue-dose-transfer-v1,");
            out.push_str(c.cond.name());
            out.push(',');
            out.push_str(line);
            out.push('\n');
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

fn arm_mean(rows: &[Row], x: Arm, f: &dyn Fn(&Row) -> f64) -> f64 {
    cwd::setting_mean(rows, sel(), x, f)
}

/// Mean over seeds of the relative forgetting of arm `x` (seeds with an
/// undefined ratio are skipped and counted).
fn rf_mean(rows: &[Row], x: Arm) -> (f64, usize) {
    let v: Vec<f64> = seeds_of(rows)
        .into_iter()
        .filter_map(|s| rel_forgetting(rows, s, x))
        .collect();
    let m = if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    };
    (m, seeds_of(rows).len() - v.len())
}

/// Text report: gates, per-condition summaries, hypotheses, secondary.
#[must_use]
pub fn report(all: &[CondRows]) -> String {
    use std::fmt::Write as _;
    let mut o = String::new();
    let q = sel();
    let (a, cn) = (Slice::All, Slice::Cn);
    let acc = |r: &Row| r.inner.acc(a, K);
    let _ = writeln!(o, "gates: cond,G1_integrity,G2_manipulation,G3_novel_pairs,G4_leakage,G5_rf_denominator,median_cn,g2_clauses,partner_moves_Zgd0..4,partner_moves_Zgd2c,idle_stages_mean,leak_free_seeds");
    for c in all {
        let g = gates(c);
        let pm: Vec<String> = DOSES
            .iter()
            .map(|&x| format!("{:.1}", arm_mean(&c.rows, x, &|r| r.partner_moved as f64)))
            .collect();
        let idle = arm_mean(&c.rows, S1, &|r| r.idle_stages() as f64);
        let g2 = cwd::g2_at(&c.rows, K, FLIP);
        let _ = writeln!(
            o,
            "{},{},{},{},{},{},{},{}{}{}{},{},{:.1},{:.2},{}/{}",
            c.cond.name(),
            g[0],
            g[1],
            g[2],
            g[3],
            g[4],
            cwd::median_cn(&c.rows, K, FLIP),
            u8::from(g2[0]),
            u8::from(g2[1]),
            u8::from(g2[2]),
            u8::from(g2[3]),
            pm.join("/"),
            arm_mean(&c.rows, ZGD2C, &|r| r.partner_moved as f64),
            idle,
            c.leakage.iter().filter(|x| x.1).count(),
            c.leakage.len()
        );
    }
    let _ = writeln!(o, "summary(heldout orders, mean over rows): cond,arm,acc_all,acc_cn,learn_all,learn_head,fgt_all,last_all,rel_forgetting_mean(seeds_undefined),charged_writes,cue_writes,partner_moved,skipped");
    for c in all {
        for x in cwd::ARMS {
            let (rfm, und) = rf_mean(&c.rows, x);
            let _ = writeln!(
                o,
                "{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}({}),{:.1},{:.1},{:.1},{:.1}",
                c.cond.name(),
                x.short(),
                arm_mean(&c.rows, x, &acc),
                arm_mean(&c.rows, x, &|r| r.inner.acc(cn, K)),
                arm_mean(&c.rows, x, &|r| r.inner.learn(a, K)),
                arm_mean(&c.rows, x, &learn_head),
                arm_mean(&c.rows, x, &fgt),
                arm_mean(&c.rows, x, &|r| r.inner.last(a, K)),
                rfm,
                und,
                arm_mean(&c.rows, x, &|r| r.inner.charged_writes() as f64),
                arm_mean(&c.rows, x, &|r| r.inner.cue_writes() as f64),
                arm_mean(&c.rows, x, &|r| r.partner_moved as f64),
                arm_mean(&c.rows, x, &|r| r.skipped as f64),
            );
        }
    }
    let h = readings(all);
    let get = |c: Cond| find(all, c);
    let line = |o: &mut String, id: &str, c: Cond, what: &str, d: Option<Paired>| {
        if let Some(d) = d {
            let _ = writeln!(o, "  {id},{},{what},{}", c.name(), fmt_p(&d));
        }
    };
    let _ = writeln!(
        o,
        "hypotheses (k={K}, flip {FLIP}, held-out orders; per seed the mean over its held-out orders; mean±sd[95% bootstrap CI]better/ties/worse):"
    );
    let names = [
        "H1 base dose slope ACC_all > 0",
        "H2 train64 dose slope ACC_all > 0",
        "H3 src256 dose slope ACC_all > 0",
        "H4 -dose slope FGT_all > 0 in base, train64 and src256",
        "H5 base FGT(S1)-FGT(Zgd4) non-inferior (CI lo > -0.02)",
        "H6 base RF(S1)-RF(Zgd4) > 0",
        "H7 train64 and src256 RF(S1)-RF(Zgd4) > 0",
        "H8 train64 and src256 Zgd4 vs S1 sequence rule",
        "H9 ACC_cn(Zgd4)-ACC_cn(S1) > 0 in base, train64 and src256",
        "H10 base concavity ACC_all equivalent to 0 (+-0.01)",
        "H11 base interaction ACC_all equivalent to 0 (+-0.01)",
        "H12 train64 and src256 interaction ACC_all equivalent to 0 (+-0.01)",
    ];
    for (i, n) in names.iter().enumerate() {
        let _ = writeln!(o, "{n}: {}", yes(h[i]));
        for c in CONDS {
            let Some(cr) = get(c) else { continue };
            let rows = &cr.rows;
            let cmp = |s: Slice| cwd::compare(rows, q, K, s, ZGD4, S1);
            match i {
                0..=2 if c == CONDS[i] => line(
                    &mut o,
                    "slope_acc_all",
                    c,
                    "Zgd0..4",
                    Some(cwd::slope(rows, q, 1.0, &acc)),
                ),
                3 => line(
                    &mut o,
                    "-slope_fgt_all",
                    c,
                    "Zgd0..4",
                    Some(cwd::slope(rows, q, -1.0, &fgt)),
                ),
                4 if c == Cond::Base => line(
                    &mut o,
                    "fgt(S1)-fgt(Zgd4)",
                    c,
                    "Zgd4-S1",
                    Some(cmp(a).forget),
                ),
                5 if c == Cond::Base => line(
                    &mut o,
                    "rf(S1)-rf(Zgd4)",
                    c,
                    "Zgd4-S1",
                    Some(rf_diff(rows, ZGD4, S1)),
                ),
                6 if c != Cond::Base => line(
                    &mut o,
                    "rf(S1)-rf(Zgd4)",
                    c,
                    "Zgd4-S1",
                    Some(rf_diff(rows, ZGD4, S1)),
                ),
                7 if c != Cond::Base => {
                    let m = cmp(a);
                    line(&mut o, "dACC", c, "Zgd4-S1", Some(m.acc));
                    line(&mut o, "dLEARN", c, "Zgd4-S1", Some(m.learn));
                    line(&mut o, "dLAST", c, "Zgd4-S1", Some(m.last));
                }
                8 => line(&mut o, "dACC_cn", c, "Zgd4-S1", Some(cmp(cn).acc)),
                9 if c == Cond::Base => line(
                    &mut o,
                    "concavity_acc_all",
                    c,
                    "Zgd0,2,4",
                    Some(cwd::concavity(rows, q, &acc)),
                ),
                10 if c == Cond::Base => line(
                    &mut o,
                    "interaction_acc_all",
                    c,
                    "Zgd4-Zgd2-Zgd2c+Zgd0",
                    Some(interaction(rows, &acc)),
                ),
                11 if c != Cond::Base => line(
                    &mut o,
                    "interaction_acc_all",
                    c,
                    "Zgd4-Zgd2-Zgd2c+Zgd0",
                    Some(interaction(rows, &acc)),
                ),
                _ => {}
            }
        }
    }
    let _ = writeln!(
        o,
        "secondary (descriptive, no pass/fail): id,cond,contrast,value"
    );
    for cr in all {
        let (c, rows) = (cr.cond, &cr.rows);
        let cmp = |x: Arm, b: Arm, s: Slice| cwd::compare(rows, q, K, s, x, b);
        line(
            &mut o,
            "slope_acc_all",
            c,
            "Zgd0..4",
            Some(cwd::slope(rows, q, 1.0, &acc)),
        );
        line(
            &mut o,
            "-slope_learn_all",
            c,
            "Zgd0..4",
            Some(cwd::slope(rows, q, -1.0, &|r| r.inner.learn(a, K))),
        );
        line(
            &mut o,
            "slope_acc_cn",
            c,
            "Zgd0..4",
            Some(cwd::slope(rows, q, 1.0, &|r| r.inner.acc(cn, K))),
        );
        line(
            &mut o,
            "slope_partner_moved",
            c,
            "Zgd0..4",
            Some(cwd::slope(rows, q, 1.0, &|r| r.partner_moved as f64)),
        );
        line(
            &mut o,
            "concavity_acc_all",
            c,
            "Zgd0,2,4",
            Some(cwd::concavity(rows, q, &acc)),
        );
        line(
            &mut o,
            "concavity_fgt_all",
            c,
            "Zgd0,2,4",
            Some(cwd::concavity(rows, q, &fgt)),
        );
        line(
            &mut o,
            "concavity_learn_all",
            c,
            "Zgd0,2,4",
            Some(cwd::concavity(rows, q, &|r| r.inner.learn(a, K))),
        );
        line(
            &mut o,
            "concavity_partner_moved",
            c,
            "Zgd0,2,4",
            Some(cwd::concavity(rows, q, &|r| r.partner_moved as f64)),
        );
        line(
            &mut o,
            "interaction_acc_all",
            c,
            "Zgd4-Zgd2-Zgd2c+Zgd0",
            Some(interaction(rows, &acc)),
        );
        line(
            &mut o,
            "interaction_fgt_all",
            c,
            "Zgd4-Zgd2-Zgd2c+Zgd0",
            Some(interaction(rows, &fgt)),
        );
        line(
            &mut o,
            "interaction_learn_all",
            c,
            "Zgd4-Zgd2-Zgd2c+Zgd0",
            Some(interaction(rows, &|r| r.inner.learn(a, K))),
        );
        line(
            &mut o,
            "interaction_partner_moved",
            c,
            "Zgd4-Zgd2-Zgd2c+Zgd0",
            Some(interaction(rows, &|r| r.partner_moved as f64)),
        );
        line(
            &mut o,
            "dACC_all",
            c,
            "Zgd2-Zgd2c",
            Some(cmp(ZGD2, ZGD2C, a).acc),
        );
        for x in [ZGD0, ZGD2, ZGD4] {
            let m = cmp(x, S1, a);
            let n = format!("{}-S1", x.short());
            line(&mut o, "dACC_all", c, &n, Some(m.acc));
            line(&mut o, "fgt(S1)-fgt(x)", c, &n, Some(m.forget));
            line(&mut o, "dLEARN_all", c, &n, Some(m.learn));
            line(&mut o, "dLAST_all", c, &n, Some(m.last));
            if rf_defined(rows, &[x, S1]) {
                line(&mut o, "rf(S1)-rf(x)", c, &n, Some(rf_diff(rows, x, S1)));
            }
        }
        line(
            &mut o,
            "worst_heldout_order_dACC_all",
            c,
            "Zgd4-S1",
            Some(cwd::worst_order(rows, K, FLIP, ZGD4, S1)),
        );
        line(
            &mut o,
            "worst_heldout_order_dACC_all",
            c,
            "Zgd2-Zgd2c",
            Some(cwd::worst_order(rows, K, FLIP, ZGD2, ZGD2C)),
        );
    }
    o
}

/// Label-free pilot of one condition (the cycle 27 census on the identity
/// order: writes, skips, partner-line moves, idle stages; no accuracy).
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
                train_patterns: if c == Cond::Train64 { 16 } else { 8 },
                test_patterns: 24,
                rounds_a: 2,
                rounds_b: 2,
                ..d.spec
            },
            ..d
        }
    }

    /// One small grid (all conditions, seed 2, 1 random order),
    /// computed once per test binary.
    fn grid() -> &'static [CondRows] {
        static GRID: std::sync::OnceLock<Vec<CondRows>> = std::sync::OnceLock::new();
        GRID.get_or_init(|| run_with(&CONDS, &[2], 3, 1, small).unwrap())
    }

    #[test]
    fn conditions_change_exactly_one_design_field() {
        let b = Cond::Base.design();
        assert_eq!(b, cwd::DESIGN);
        let t = Cond::Train64.design();
        assert_eq!(t.spec.train_patterns, 64);
        assert_eq!(Design { spec: b.spec, ..t }, b);
        let s = Cond::Src256.design();
        assert_eq!(s.shape.sources, 256);
        assert_eq!(
            Design {
                shape: b.shape,
                ..s
            },
            b
        );
        for c in CONDS {
            assert_eq!(Cond::parse(c.name()).unwrap(), c);
        }
        assert!(Cond::parse("x").is_err());
    }

    #[test]
    fn every_condition_builds_a_leakage_free_fixture_with_whole_quarter_doses() {
        for c in CONDS {
            assert!(leakage_free(c, 1).unwrap(), "{}", c.name());
            let fx = sg::fixture(None, &c.design(), K, FLIP, 1).unwrap();
            let a = crate::cue_partner::agreement(&fx).unwrap();
            let sigma = crate::cue_partner::partner_map(crate::cue_partner::Code::Matched, &a, 1);
            let f = crate::cue_ownership::frozen_lines(&fx.cues, &sigma);
            assert!(f.iter().all(|l| l.len() == 4), "{}", c.name());
        }
    }

    #[test]
    fn interaction_and_relative_forgetting_are_paired_by_seed() {
        // A tiny grid on a pilot seed: the interaction of an arm with itself
        // and the RF difference of an arm with itself are exactly zero.
        let all = &grid()[..1];
        let rows = &all[0].rows;
        let acc = |r: &Row| r.inner.acc(Slice::All, K);
        let z = cwd::compare(rows, sel(), K, Slice::All, ZGD4, ZGD4).acc;
        assert_eq!(z.mean, 0.0);
        let i = interaction(rows, &acc);
        assert_eq!(i.n, 1);
        let manual: f64 = [2_u64]
            .iter()
            .map(|&sd| {
                seed_mean(rows, sd, ZGD4, &acc).unwrap()
                    - seed_mean(rows, sd, ZGD2, &acc).unwrap()
                    - seed_mean(rows, sd, ZGD2C, &acc).unwrap()
                    + seed_mean(rows, sd, ZGD0, &acc).unwrap()
            })
            .sum::<f64>();
        assert!((i.mean - manual).abs() < 1e-12);
        if rf_defined(rows, &[ZGD4]) {
            assert_eq!(rf_diff(rows, ZGD4, ZGD4).mean, 0.0);
        }
        let g = gates(&all[0]);
        assert!(g[0] && g[3], "integrity and leakage gates on a pilot seed");
        // Tampering with leakage flags fails G4; a missing condition is not evaluable.
        let mut t = all[0].clone();
        t.leakage[0].1 = false;
        assert!(!gates(&t)[3]);
        let h = readings(all);
        assert!(h[1].is_none() && h[2].is_none() && h[11].is_none());
        let csv = to_csv(all);
        assert!(csv.starts_with("transfer_schema,cond,"));
        assert_eq!(csv.lines().count(), 1 + rows.len());
        assert!(report(all).contains("H11 base interaction"));
        // All three conditions: integrity and leakage hold in every one.
        let full = grid();
        let csv = to_csv(full);
        assert_eq!(
            csv.lines().count(),
            1 + full.iter().map(|c| c.rows.len()).sum::<usize>()
        );
        for c in full {
            let g = gates(c);
            assert!(g[0] && g[3], "{}", c.cond.name());
        }
        // (Thread independence of the engine is tested in `cue_write_dose`.)
    }

    #[test]
    fn learn_head_excludes_the_last_task() {
        let r = grid()[0]
            .rows
            .iter()
            .find(|r| r.inner.order_ix > 0)
            .unwrap();
        let m = (0..K - 1).map(|j| r.inner.r(Slice::All, j, j)).sum::<f64>() / (K - 1) as f64;
        assert!((learn_head(r) - m).abs() < 1e-9);
    }
}

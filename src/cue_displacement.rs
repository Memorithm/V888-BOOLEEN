//! B888-4 cycle 17: **where does the rest of the silent cue's learning loss
//! come from?** A write-displacement decomposition at an equal write budget.
//!
//! Cycle 16 (#128) started the cue-line edges at weight 0 (`E/E-cue0`, "Z")
//! and recovered about half of the cued arm's learning loss at
//! `error-gated`'s write schedule (`paced100`), but Z still learned ≈ 0.15
//! less than `error-gated` while spending ≈ 37 % of the allowance on cue
//! edges. Two readings remain: (a) **displacement** — every cue write is a
//! write the pattern edges no longer get; (b) **interference** — the cue
//! edges that do gain weight change which targets fire and so which updates
//! the error gate offers. This module separates them with two
//! counter-factual budgets on the unchanged rules:
//!
//! * `E/E-cue0-exempt` ("Zx"): Z, but cue-edge writes are **not charged**:
//!   only pattern-edge writes count against `error-gated`'s paced curve, so
//!   Zx gets exactly `error-gated`'s pattern-write schedule plus free cue
//!   writes (unequal *total* writes, reported as such);
//! * `error-gated@Z-pattern` ("Em"): uncued `error-gated` capped at the
//!   per-episode cumulative **pattern**-write curve that Z@paced100 actually
//!   made in the same (flip, seed) — the pattern budget the cue leaves.
//!
//! Everything else (graph, task, pool, cues, episodes, update rules,
//! `paced100` budget) is the #124–#128 design. With charging on, the
//! module's cued step is bit-identical to `write_allocation::step`
//! (source-major, tested); uncharged and uncapped it is bit-identical to the
//! charged uncapped run (tested).
//!
//! Engine state for checkpoint/resume = circuit weights
//! (`PlasticCircuit::save`) + [`Checkpoint`] (the #127 text checkpoint plus
//! the per-episode cumulative cue-write curve); resuming mid-task is
//! bit-exact for every setting (tested).
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph; "cue",
//! "displacement" and "interference" name parts of an update rule and its
//! write accounting, not claims about synapses, biology, insect memory or
//! BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/CUE_DISPLACEMENT.md`
//! (committed before any learning run on the hypothesis seeds).
use crate::connectome::Connectome;
use crate::context_cue as cc;
use crate::cue_init::{self as ci, Arm};
use crate::cue_reach::{self as cr, Family, Part};
use crate::error_gated::{drive, paired, Paired, PhaseCost, EFFECT, THRESHOLD};
use crate::generalization_bench::GenSpec;
use crate::novel_split::{self as ns, Design, Eval, Fixture, Slice, SLICES};
use crate::plasticity::{decrement_four, increment_four, PlasticCircuit};
use crate::task_sequence::{self as ts, K_MAX};
use crate::write_allocation::{self as wa, Alloc, Stats, Strategy, Tags};
use crate::write_budget::{self as wb, Budget, Limit, WriteLedger};
use std::io::{self, ErrorKind};
use std::path::Path;

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Pre-registered flips (50 primary, 100 high conflict, 25 secondary).
pub const FLIPS: [u32; 3] = [25, 50, 100];
pub const PRIMARY_FLIP: u32 = 50;
pub const HIGH_FLIP: u32 = 100;
/// Pre-registered seeds: 281..=320, never used by an earlier B888-4 cycle.
pub const SEEDS: std::ops::RangeInclusive<u64> = 281..=320;
/// The equal-write budget of every capped setting.
pub const BUDGET: Budget = Budget::Paced(100);
/// Fixed-pool design of #124–#128.
pub const DESIGN: Design = wb::DESIGN;
/// Pilot seeds (writes only, no accuracy).
pub const PILOT_SEEDS: std::ops::RangeInclusive<u64> = 1..=5;

/// One pre-registered setting (rule + init + write accounting + cap).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Setting {
    /// `supervised@paced100` (control).
    Sup,
    /// `reward@paced100` (clean per-target reward-modulated).
    Rew,
    /// Uncapped `error-gated` (= its `paced100` run, gated): base.
    Err,
    /// `E/E-cue0@paced100` (#128 primary arm, every write charged).
    Z,
    /// `E/E-cue0@paced100` with cue writes not charged.
    Zx,
    /// `error-gated` capped at Z's per-episode pattern-write curve.
    Em,
    /// Uncapped `E/E-cue0`.
    Zfree,
}

/// The pre-registered settings, in CSV order (Z before Em: Em's cap is
/// read from Z's run).
pub const SETTINGS: [Setting; 7] = [
    Setting::Sup,
    Setting::Rew,
    Setting::Err,
    Setting::Z,
    Setting::Zx,
    Setting::Em,
    Setting::Zfree,
];

impl Setting {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Sup => "supervised@paced100",
            Self::Rew => "reward@paced100",
            Self::Err => "error-gated@free",
            Self::Z => "E/E-cue0@paced100",
            Self::Zx => "E/E-cue0-exempt@paced100",
            Self::Em => "error-gated@Z-pattern",
            Self::Zfree => "E/E-cue0@free",
        }
    }
    /// Parse a setting name.
    pub fn parse(s: &str) -> io::Result<Self> {
        SETTINGS
            .iter()
            .copied()
            .find(|x| x.name() == s)
            .ok_or_else(|| bad("unknown setting"))
    }
    /// The `cue_init` arm (rule + cue-edge init).
    #[must_use]
    pub fn arm(self) -> Arm {
        match self {
            Self::Sup => ci::SUP,
            Self::Rew => ci::REW,
            Self::Err | Self::Em => ci::ERR,
            Self::Z | Self::Zx | Self::Zfree => ci::CUE0,
        }
    }
    #[must_use]
    pub fn cued(self) -> bool {
        self.arm().cued()
    }
    /// Are cue writes exempt from the cap?
    #[must_use]
    pub fn exempt(self) -> bool {
        self == Self::Zx
    }
}

/// One episode. Uncued rules delegate to `write_allocation::step` (no
/// tagging); the cued rule is source-major `E/E-cue` with an optional
/// exemption of cue-edge writes from the ledger: with `exempt = false` it is
/// bit-identical to `write_allocation::step` (tested). With `exempt = true`
/// `ledger.writes` counts **pattern**-edge writes only and the cap refuses
/// pattern steps only; cue writes are counted in `stats.cue_writes`.
#[allow(clippy::too_many_arguments)]
pub fn step(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    rule: Strategy,
    exempt: bool,
    pattern: &[usize],
    valence: &[bool],
    cue: &[usize],
    stats: &mut Stats,
    ledger: &mut WriteLedger,
    cost: &mut PhaseCost,
) -> io::Result<()> {
    if rule != Strategy::EeCue(Alloc::SourceMajor) {
        if exempt || rule.cued() || rule == Strategy::Tagged {
            return Err(bad("only source-major E/E-cue is cued in this cycle"));
        }
        let mut tags = Tags::new(graph.edges());
        return wa::step(
            circuit, graph, rule, pattern, valence, cue, &mut tags, stats, ledger, cost,
        );
    }
    let (input, reached, teacher) = cc::encode_ctx(graph, pattern, valence, cue)?;
    let n = graph.ids.len();
    let mut want = vec![false; n];
    for (t, w) in want.iter_mut().enumerate() {
        *w = teacher.get(t).map_err(|_| bad("bad target"))?;
    }
    let d = drive(circuit, graph, &input)?;
    let mut offer = vec![false; n];
    for &t in &reached {
        offer[t] = crate::gates::ge(d[t], THRESHOLD) != want[t];
    }
    cost.episodes += 1;
    cost.targets_seen += reached.len() as u64;
    cost.targets_updated += reached.iter().filter(|&&t| offer[t]).count() as u64;
    let sources: Vec<usize> = input.active_indices().collect();
    for &source in &sources {
        let on_cue = cue.contains(&source);
        let charged = !(exempt && on_cue);
        for edge in graph.offsets[source]..graph.offsets[source + 1] {
            let edge = edge as usize;
            let t = graph.targets[edge] as usize;
            if !offer[t] {
                continue;
            }
            let up = want[t];
            let w = circuit.weights()[edge];
            let next = if up {
                increment_four(w)
            } else {
                decrement_four(w)
            };
            if charged && ledger.writes >= ledger.allow && next != w {
                ledger.refused += 1;
                if on_cue {
                    stats.cue_refused += 1;
                }
                if ledger.first_refusal == 0 {
                    ledger.first_refusal = cost.episodes;
                }
                continue;
            }
            cost.attempts += 1;
            if circuit.apply_edge_step(edge, up)? {
                cost.weights_changed += 1;
                if charged {
                    ledger.writes += 1;
                }
                if on_cue {
                    stats.cue_writes += 1;
                }
            }
        }
    }
    ledger.cum.push(ledger.writes);
    Ok(())
}

/// Resumable state of one task: the #127 checkpoint plus the per-episode
/// cumulative cue-write curve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub inner: wa::Checkpoint,
    pub cue_cum: Vec<u64>,
}

const MAGIC: &str = "v888-cue-displacement-checkpoint-v1";

impl Checkpoint {
    #[must_use]
    pub fn start(task: usize, edges: usize, limit: &Limit) -> Self {
        Self {
            inner: wa::Checkpoint::start(task, edges, limit),
            cue_cum: Vec::new(),
        }
    }
    /// Deterministic text serialisation.
    #[must_use]
    pub fn encode(&self) -> String {
        let cum = if self.cue_cum.is_empty() {
            "-".to_string()
        } else {
            self.cue_cum
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        };
        format!("{MAGIC}\ncuecum {cum}\n{}", self.inner.encode())
    }
    /// Parse [`Checkpoint::encode`]; `edges` must match the graph.
    pub fn decode(s: &str, edges: usize) -> io::Result<Self> {
        let mut it = s.splitn(3, '\n');
        if it.next() != Some(MAGIC) {
            return Err(bad("not a cue-displacement checkpoint"));
        }
        let line = it.next().ok_or_else(|| bad("truncated checkpoint"))?;
        let rest = it.next().ok_or_else(|| bad("truncated checkpoint"))?;
        let v = line
            .strip_prefix("cuecum ")
            .ok_or_else(|| bad("bad checkpoint field"))?;
        let cue_cum: Vec<u64> = if v == "-" {
            Vec::new()
        } else {
            v.split(' ')
                .map(|x| x.parse::<u64>().map_err(|_| bad("bad number")))
                .collect::<io::Result<_>>()?
        };
        let inner = wa::Checkpoint::decode(rest, edges)?;
        if cue_cum.len() != inner.episode {
            return Err(bad("checkpoint cue curve mismatch"));
        }
        Ok(Self { inner, cue_cum })
    }
    pub fn save(&self, path: &Path) -> io::Result<()> {
        std::fs::write(path, self.encode())
    }
    pub fn load(path: &Path, edges: usize) -> io::Result<Self> {
        Self::decode(&std::fs::read_to_string(path)?, edges)
    }
}

fn limit_at(limit: &Limit, e: usize) -> u64 {
    limit
        .0
        .as_ref()
        .map_or(u64::MAX, |a| a.get(e).copied().unwrap_or(u64::MAX))
}

/// Train episodes `cp.inner.episode..end` of task `cp.inner.task`.
#[allow(clippy::too_many_arguments)]
pub fn train_range(
    circuit: &mut PlasticCircuit,
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    s: Setting,
    limit: &Limit,
    cp: &mut Checkpoint,
    end: usize,
) -> io::Result<()> {
    let i = cp.inner.task;
    let order = ts::task_order(spec, seed, i);
    if cp.inner.episode > end || end > order.len() {
        return Err(bad("bad episode range"));
    }
    let cue: &[usize] = if s.cued() {
        fx.cues.get(i).ok_or_else(|| bad("missing cue"))?
    } else {
        &[]
    };
    for (e, &ix) in order.iter().enumerate().take(end).skip(cp.inner.episode) {
        let c = &mut cp.inner;
        c.ledger.allow = limit_at(limit, e);
        step(
            circuit,
            &fx.graph,
            s.arm().rule,
            s.exempt(),
            &fx.task.train[i][ix],
            &fx.task.valence[i],
            cue,
            &mut c.stats,
            &mut c.ledger,
            &mut c.cost,
        )?;
        cp.cue_cum.push(c.stats.cue_writes);
    }
    cp.inner.episode = end;
    if end == order.len() {
        cp.inner.ledger.allow = cp.inner.ledger.cap;
    }
    Ok(())
}

/// Per-task record of one run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TaskLog {
    pub ledger: WriteLedger,
    pub stats: Stats,
    /// Cumulative cue writes after each episode.
    pub cue_cum: Vec<u64>,
    /// Cue-line edges non-zero / their weight mass at the end of the task.
    pub cue_nonzero_end: u64,
    pub cue_mass_end: u64,
}

impl TaskLog {
    /// Pattern-edge writes (total minus cue).
    #[must_use]
    pub fn pattern_writes(&self, exempt: bool) -> u64 {
        if exempt {
            self.ledger.writes
        } else {
            self.ledger.writes - self.stats.cue_writes
        }
    }
    /// Total effective writes (pattern + cue).
    #[must_use]
    pub fn total_writes(&self, exempt: bool) -> u64 {
        if exempt {
            self.ledger.writes + self.stats.cue_writes
        } else {
            self.ledger.writes
        }
    }
    /// Per-episode cumulative pattern-write curve (Em's cap source).
    #[must_use]
    pub fn pattern_curve(&self, exempt: bool) -> Vec<u64> {
        self.ledger
            .cum
            .iter()
            .zip(&self.cue_cum)
            .map(|(&w, &c)| if exempt { w } else { w - c })
            .collect()
    }
}

/// Output of [`run_setting`]: metrics row, per-task logs, final weights.
pub type Run = (cr::Row, [TaskLog; K_MAX], Vec<u8>);

/// Run one setting over the K_MAX tasks with per-task write limits.
pub fn run_setting(
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    s: Setting,
    limits: &[Limit; K_MAX],
) -> io::Result<Run> {
    let graph = &fx.graph;
    if fx.task.train.len() != K_MAX || fx.cues.len() != K_MAX || fx.splits.len() != K_MAX {
        return Err(bad("need K_MAX tasks, cues and splits"));
    }
    graph.validate()?;
    let original = graph.clone();
    let n = graph.ids.len();
    let cue = ci::cue_edges(fx)?;
    let mut circuit = ci::initial_circuit(fx, spec, s.arm())?;
    let fingerprint = circuit.topology_fingerprint();
    let mut ev = [[Eval::default(); K_MAX]; K_MAX];
    let mut tgt = vec![crate::generalization_bench::Confusion::default(); K_MAX * K_MAX * n];
    let mut cost = [PhaseCost::default(); K_MAX];
    let mut logs: [TaskLog; K_MAX] = Default::default();
    for i in 0..K_MAX {
        let episodes = ts::task_order(spec, seed, i).len();
        let mut cp = Checkpoint::start(i, graph.edges(), &limits[i]);
        train_range(
            &mut circuit,
            fx,
            spec,
            seed,
            s,
            &limits[i],
            &mut cp,
            episodes,
        )?;
        let (mut nz, mut mass) = (0, 0);
        for (&x, &on) in circuit.weights().iter().zip(&cue) {
            if on && x > 0 {
                nz += 1;
                mass += u64::from(x);
            }
        }
        cost[i] = cp.inner.cost;
        logs[i] = TaskLog {
            ledger: cp.inner.ledger,
            stats: cp.inner.stats,
            cue_cum: cp.cue_cum,
            cue_nonzero_end: nz,
            cue_mass_end: mass,
        };
        for j in 0..K_MAX {
            let c: &[usize] = if s.cued() { &fx.cues[j] } else { &[] };
            let (e, per) = cr::evaluate(
                &circuit,
                graph,
                &fx.task.test,
                &fx.task.valence[j],
                c,
                &fx.splits[j],
            )?;
            ev[i][j] = e;
            tgt[(i * K_MAX + j) * n..(i * K_MAX + j + 1) * n].copy_from_slice(&per);
        }
    }
    let mut cue_lines = [0; K_MAX];
    if s.cued() {
        for (l, c) in cue_lines.iter_mut().zip(&fx.cues) {
            *l = c.len();
        }
    }
    let row = cr::Row {
        family: Family::Fixed,
        level: if s.cued() { ci::CUE_LEVEL } else { 0 },
        arm: match s.arm().rule {
            Strategy::Supervised | Strategy::Reward => cr::SUPERVISED,
            Strategy::EeCue(_) => cr::EE,
            _ => cr::ERROR_GATED,
        },
        flip: spec.flip_pct,
        seed,
        pool: fx.task.pool.len(),
        reachable: cr::reachable_targets(graph),
        cue_lines,
        cue_targets: fx.cues.iter().map(|c| cr::reach_set(graph, c)).collect(),
        ev,
        tgt,
        nodes: n,
        cost,
        saturated_end: crate::error_gated::saturated(circuit.weights()),
        topology_unchanged: *graph == original
            && circuit.topology_fingerprint() == fingerprint
            && circuit.weights().len() == graph.edges(),
    };
    Ok((row, logs, circuit.weights().to_vec()))
}

/// Em's per-task limits: Z's per-episode cumulative pattern-write curves.
#[must_use]
pub fn pattern_limits(z: &[TaskLog; K_MAX]) -> [Limit; K_MAX] {
    let mut out: [Limit; K_MAX] = Default::default();
    for (l, t) in out.iter_mut().zip(z) {
        *l = Limit(Some(t.pattern_curve(false)));
    }
    out
}

/// One measurement: (flip, seed, setting) on a K_MAX-task run.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub setting: Setting,
    pub inner: cr::Row,
    pub logs: [TaskLog; K_MAX],
    /// Initial weights are the documented init of the setting's arm.
    pub init_ok: bool,
    /// `error-gated@free` only: its `paced100` run is bit-identical with no
    /// refusal (true for other settings).
    pub paced_identity: bool,
    /// Digest of the final weights.
    pub digest: u64,
}

impl Row {
    #[must_use]
    pub fn total_writes(&self) -> u64 {
        self.logs
            .iter()
            .map(|t| t.total_writes(self.setting.exempt()))
            .sum()
    }
    #[must_use]
    pub fn pattern_writes(&self) -> u64 {
        self.logs
            .iter()
            .map(|t| t.pattern_writes(self.setting.exempt()))
            .sum()
    }
    #[must_use]
    pub fn cue_writes(&self) -> u64 {
        self.logs.iter().map(|t| t.stats.cue_writes).sum()
    }
    #[must_use]
    pub fn refused(&self) -> u64 {
        self.logs.iter().map(|t| t.ledger.refused).sum()
    }
    /// Charged writes never exceed the allowance in any task.
    #[must_use]
    pub fn within_cap(&self) -> bool {
        self.logs.iter().all(|t| t.ledger.writes <= t.ledger.cap)
    }
}

/// Full grid: per (flip, seed) uncapped `error-gated` first (its write
/// curves set the `paced100` caps), Z before Em (Z's pattern curve sets
/// Em's cap).
pub fn run_grid_design(
    graph: Option<&Connectome>,
    design: &Design,
    flips: &[u32],
    seeds: &[u64],
) -> io::Result<Vec<Row>> {
    if seeds.is_empty() || flips.is_empty() {
        return Err(bad("need at least one seed and flip"));
    }
    let mut rows = Vec::new();
    for &flip in flips {
        let spec = GenSpec {
            flip_pct: flip,
            ..design.spec
        };
        for &seed in seeds {
            let fx = ns::fixture(graph, design, flip, seed)?;
            let e = run_setting(&fx, &spec, seed, Setting::Err, &wb::free())?;
            let curves: Vec<Vec<u64>> = e.1.iter().map(|t| t.ledger.cum.clone()).collect();
            let paced = wb::limits(&curves, BUDGET);
            let e_paced = run_setting(&fx, &spec, seed, Setting::Err, &paced)?;
            let identity = e_paced.2 == e.2 && e_paced.1.iter().all(|t| t.ledger.refused == 0);
            let mut z_logs: Option<[TaskLog; K_MAX]> = None;
            for s in SETTINGS {
                let run = match s {
                    Setting::Err => e.clone(),
                    Setting::Zfree => run_setting(&fx, &spec, seed, s, &wb::free())?,
                    Setting::Em => {
                        let z = z_logs.as_ref().ok_or_else(|| bad("Em needs Z first"))?;
                        run_setting(&fx, &spec, seed, s, &pattern_limits(z))?
                    }
                    _ => run_setting(&fx, &spec, seed, s, &paced)?,
                };
                if s == Setting::Z {
                    z_logs = Some(run.1.clone());
                }
                let init = ci::initial_circuit(&fx, &spec, s.arm())?;
                rows.push(Row {
                    setting: s,
                    init_ok: ci::init_matches(&fx, &spec, s.arm(), init.weights())?,
                    paced_identity: s != Setting::Err || identity,
                    digest: wa::digest(&run.2),
                    inner: run.0,
                    logs: run.1,
                });
            }
        }
    }
    Ok(rows)
}

/// Pre-registered grid on the fixed-pool design.
pub fn run_grid(graph: Option<&Connectome>, flips: &[u32], seeds: &[u64]) -> io::Result<Vec<Row>> {
    run_grid_design(graph, &DESIGN, flips, seeds)
}

fn find(rows: &[Row], flip: u32, seed: u64, x: Setting) -> Option<&Row> {
    rows.iter()
        .find(|r| r.inner.flip == flip && r.inner.seed == seed && r.setting == x)
}

fn rows_of(rows: &[Row], flip: u32, x: Setting) -> impl Iterator<Item = &Row> {
    rows.iter()
        .filter(move |r| r.inner.flip == flip && r.setting == x)
}

fn pairs_of(rows: &[Row], flip: u32, x: Setting, base: Setting) -> Vec<(&Row, &Row)> {
    rows_of(rows, flip, x)
        .filter_map(|r| find(rows, flip, r.inner.seed, base).map(|c| (r, c)))
        .collect()
}

/// x vs base paired by seed at k (same oriented fields as `cue_reach`).
#[must_use]
pub fn compare(
    rows: &[Row],
    flip: u32,
    k: usize,
    s: Slice,
    x: Setting,
    base: Setting,
) -> cr::Comparison {
    let p = Part::Class(s);
    let pairs = pairs_of(rows, flip, x, base);
    let d = |f: &dyn Fn(&cr::Row, &cr::Row) -> f64| {
        paired(
            &pairs
                .iter()
                .map(|(r, c)| f(&r.inner, &c.inner))
                .collect::<Vec<_>>(),
        )
    };
    cr::Comparison {
        acc: d(&|r, c| r.acc(p, k) - c.acc(p, k)),
        learn: d(&|r, c| r.learn(p, k) - c.learn(p, k)),
        last: d(&|r, c| r.last(p, k) - c.last(p, k)),
        forget: d(&|r, c| c.forgetting(p, k) - r.forgetting(p, k)),
    }
}

fn mean(v: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = v.collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Mean of `f` over the rows of one (flip, setting).
#[must_use]
pub fn setting_mean(rows: &[Row], flip: u32, x: Setting, f: &dyn Fn(&Row) -> f64) -> f64 {
    mean(rows_of(rows, flip, x).map(f))
}

/// Fraction of seeds where x and y end with different weights.
#[must_use]
pub fn differ_frac(rows: &[Row], flip: u32, x: Setting, y: Setting) -> f64 {
    let p = pairs_of(rows, flip, x, y);
    if p.is_empty() {
        0.0
    } else {
        p.iter().filter(|(a, b)| a.digest != b.digest).count() as f64 / p.len() as f64
    }
}

/// Fraction of seeds where `x` refused at least one step.
#[must_use]
pub fn binding_frac(rows: &[Row], flip: u32, x: Setting) -> f64 {
    let v: Vec<&Row> = rows_of(rows, flip, x).collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().filter(|r| r.refused() > 0).count() as f64 / v.len() as f64
    }
}

fn median(mut v: Vec<u64>) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

/// Median CN pairs per (seed, task) at a flip.
#[must_use]
pub fn median_cn(rows: &[Row], flip: u32) -> u64 {
    median(
        rows_of(rows, flip, Setting::Err)
            .flat_map(|r| (0..K_MAX).map(move |j| r.inner.pairs(Part::Class(Slice::Cn), j)))
            .collect(),
    )
}

/// Em's pattern budget matches Z's pattern writes: per (flip, seed, task)
/// Em's cap equals Z's pattern writes.
#[must_use]
pub fn em_matched(rows: &[Row]) -> bool {
    rows.iter().filter(|r| r.setting == Setting::Em).all(|em| {
        find(rows, em.inner.flip, em.inner.seed, Setting::Z).is_some_and(|z| {
            em.logs
                .iter()
                .zip(&z.logs)
                .all(|(a, b)| a.ledger.cap == b.pattern_writes(false))
        })
    })
}

/// Zx's pattern writes are capped at `error-gated`'s per-task writes.
#[must_use]
pub fn zx_matched(rows: &[Row]) -> bool {
    rows.iter().filter(|r| r.setting == Setting::Zx).all(|zx| {
        find(rows, zx.inner.flip, zx.inner.seed, Setting::Err).is_some_and(|e| {
            zx.logs
                .iter()
                .zip(&e.logs)
                .all(|(a, b)| a.ledger.cap == b.ledger.writes && a.ledger.writes <= b.ledger.writes)
        })
    })
}

/// Design gates: (G1 integrity, G2 manipulation, G3 CN pairs).
#[must_use]
pub fn gates(rows: &[Row]) -> (bool, bool, bool) {
    let g1 = !rows.is_empty()
        && rows
            .iter()
            .all(|r| r.inner.topology_unchanged && r.init_ok && r.paced_identity && r.within_cap())
        && em_matched(rows)
        && zx_matched(rows);
    let f = PRIMARY_FLIP;
    let g2 = differ_frac(rows, f, Setting::Zx, Setting::Z) >= 0.5
        && binding_frac(rows, f, Setting::Em) >= 0.5;
    let g3 = median_cn(rows, f) >= ns::MIN_NOVEL_PAIRS;
    (g1, g2, g3)
}

/// Number of pre-registered hypotheses.
pub const N_HYP: usize = 8;

/// Equivalence within ±EFFECT: 95 % CI inside (−0.05, +0.05).
#[must_use]
pub fn equivalent(d: &Paired) -> bool {
    d.n > 0 && d.ci_lo > -EFFECT && d.ci_hi < EFFECT
}

/// Non-inferiority at −EFFECT: 95 % CI lower bound > −0.05.
#[must_use]
pub fn noninferior(d: &Paired) -> bool {
    d.n > 0 && d.ci_lo > -EFFECT
}

/// Pre-registered readings: (gates, [H1..H8]); `None` if a gate fails or
/// the hypothesis' flip is missing.
#[must_use]
pub fn readings(rows: &[Row], flips: &[u32]) -> ((bool, bool, bool), [Option<bool>; N_HYP]) {
    use Setting::{Em, Err, Sup, Zx, Z};
    let g = gates(rows);
    let mut h = [None; N_HYP];
    if !(g.0 && g.1 && g.2) {
        return (g, h);
    }
    let (all, cn) = (Slice::All, Slice::Cn);
    let f = PRIMARY_FLIP;
    if flips.contains(&f) {
        h[0] = Some(compare(rows, f, K_MAX, all, Zx, Z).learn.strong());
        h[1] = Some(noninferior(&compare(rows, f, K_MAX, all, Zx, Err).learn));
        h[2] = Some(equivalent(&compare(rows, f, K_MAX, all, Em, Z).learn));
        h[3] = Some(compare(rows, f, K_MAX, all, Z, Em).acc.strong());
        h[4] = Some(compare(rows, f, K_MAX, all, Zx, Err).passes());
        h[5] = Some(compare(rows, f, K_MAX, cn, Zx, Err).passes());
        h[7] = Some(compare(rows, f, K_MAX, cn, Z, Sup).passes());
    }
    if flips.contains(&HIGH_FLIP) {
        let c = compare(rows, HIGH_FLIP, K_MAX, cn, Zx, Err);
        h[6] = Some(c.forget.strong() && c.learn.mean >= -EFFECT);
    }
    (g, h)
}

/// CSV header (generated; see `to_csv`).
#[must_use]
pub fn csv_header() -> String {
    let mut h: Vec<String> = [
        "schema",
        "graph",
        "flip_pct",
        "setting",
        "seed",
        "pool",
        "reachable_targets",
        "cue_lines_total",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    for s in SLICES {
        for j in 1..=K_MAX {
            h.push(format!("pairs_{}_{j}", s.name()));
        }
    }
    for s in SLICES {
        for i in 1..=K_MAX {
            for j in 1..=K_MAX {
                h.push(format!("r_{}_{i}_{j}", s.name()));
            }
        }
    }
    for s in SLICES {
        for name in ["acc", "learn", "forgetting", "last"] {
            for k in 2..=K_MAX {
                h.push(format!("{name}_{}_k{k}", s.name()));
            }
        }
    }
    for name in [
        "cap",
        "charged_writes",
        "pattern_writes",
        "cue_writes",
        "total_writes",
        "refused",
        "cue_refused",
        "first_refusal",
        "attempts",
        "cue_nonzero_end",
        "cue_mass_end",
    ] {
        for j in 1..=K_MAX {
            h.push(format!("{name}_{j}"));
        }
    }
    for name in [
        "saturated_end",
        "init_ok",
        "paced_identity",
        "weights_digest",
        "topology_unchanged",
    ] {
        h.push(name.to_string());
    }
    h.join(",")
}

/// Deterministic CSV (no timing column).
#[must_use]
pub fn to_csv(rows: &[Row], graph_label: &str) -> String {
    let mut out = csv_header();
    out.push('\n');
    for r in rows {
        let x = &r.inner;
        let ex = r.setting.exempt();
        let mut f: Vec<String> = vec![
            "b888-4-cue-displacement-v1".to_string(),
            graph_label.to_string(),
            x.flip.to_string(),
            r.setting.name().to_string(),
            x.seed.to_string(),
            x.pool.to_string(),
            x.reachable.to_string(),
            x.cue_lines.iter().sum::<usize>().to_string(),
        ];
        for s in SLICES {
            f.extend((0..K_MAX).map(|j| x.pairs(Part::Class(s), j).to_string()));
        }
        for s in SLICES {
            for i in 0..K_MAX {
                f.extend((0..K_MAX).map(|j| format!("{:.6}", x.r(Part::Class(s), i, j))));
            }
        }
        for s in SLICES {
            let p = Part::Class(s);
            for k in 2..=K_MAX {
                f.push(format!("{:.6}", x.acc(p, k)));
            }
            for k in 2..=K_MAX {
                f.push(format!("{:.6}", x.learn(p, k)));
            }
            for k in 2..=K_MAX {
                f.push(format!("{:.6}", x.forgetting(p, k)));
            }
            for k in 2..=K_MAX {
                f.push(format!("{:.6}", x.last(p, k)));
            }
        }
        f.extend(r.logs.iter().map(|t| {
            if t.ledger.cap == u64::MAX {
                "-".to_string()
            } else {
                t.ledger.cap.to_string()
            }
        }));
        f.extend(r.logs.iter().map(|t| t.ledger.writes.to_string()));
        f.extend(r.logs.iter().map(|t| t.pattern_writes(ex).to_string()));
        f.extend(r.logs.iter().map(|t| t.stats.cue_writes.to_string()));
        f.extend(r.logs.iter().map(|t| t.total_writes(ex).to_string()));
        f.extend(r.logs.iter().map(|t| t.ledger.refused.to_string()));
        f.extend(r.logs.iter().map(|t| t.stats.cue_refused.to_string()));
        f.extend(r.logs.iter().map(|t| t.ledger.first_refusal.to_string()));
        f.extend(x.cost.iter().map(|c| c.attempts.to_string()));
        f.extend(r.logs.iter().map(|t| t.cue_nonzero_end.to_string()));
        f.extend(r.logs.iter().map(|t| t.cue_mass_end.to_string()));
        f.push(x.saturated_end.to_string());
        f.push(r.init_ok.to_string());
        f.push(r.paced_identity.to_string());
        f.push(format!("{:016x}", r.digest));
        f.push(x.topology_unchanged.to_string());
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
pub fn report(rows: &[Row], flips: &[u32]) -> String {
    use std::fmt::Write;
    use Setting::{Em, Err, Rew, Sup, Zfree, Zx, Z};
    let mut o = String::new();
    let (a, c) = (Part::Class(Slice::All), Part::Class(Slice::Cn));
    let _ = writeln!(
        o,
        "setting_summary(k=4): flip,setting,acc_all,acc_cn,acc_icn,acc_seen,acc_ue,learn_all,learn_cn,fgt_all,fgt_cn,last_all,last_cn,pattern_writes,cue_writes,total_writes,refused,cue_nonzero_end_task4,cue_mass_end_task4,saturated_end"
    );
    for &flip in flips {
        for s in SETTINGS {
            if rows_of(rows, flip, s).next().is_none() {
                continue;
            }
            let m = |f: &dyn Fn(&Row) -> f64| setting_mean(rows, flip, s, f);
            let _ = writeln!(
                o,
                "{flip},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
                s.name(),
                m(&|r| r.inner.acc(a, K_MAX)),
                m(&|r| r.inner.acc(c, K_MAX)),
                m(&|r| r.inner.acc(Part::Class(Slice::Icn), K_MAX)),
                m(&|r| r.inner.acc(Part::Class(Slice::Seen), K_MAX)),
                m(&|r| r.inner.acc(Part::Class(Slice::Ue), K_MAX)),
                m(&|r| r.inner.learn(a, K_MAX)),
                m(&|r| r.inner.learn(c, K_MAX)),
                m(&|r| r.inner.forgetting(a, K_MAX)),
                m(&|r| r.inner.forgetting(c, K_MAX)),
                m(&|r| r.inner.last(a, K_MAX)),
                m(&|r| r.inner.last(c, K_MAX)),
                m(&|r| r.pattern_writes() as f64),
                m(&|r| r.cue_writes() as f64),
                m(&|r| r.total_writes() as f64),
                m(&|r| r.refused() as f64),
                m(&|r| r.logs[K_MAX - 1].cue_nonzero_end as f64),
                m(&|r| r.logs[K_MAX - 1].cue_mass_end as f64),
                m(&|r| r.inner.saturated_end as f64),
            );
        }
    }
    let _ = writeln!(
        o,
        "comparison(k): flip,k,slice,x,base,d_acc,d_learn,d_last,d_fgt(+=forgets_less),passes_sequence_rule"
    );
    let base_pairs = [
        (Zx, Z),
        (Zx, Err),
        (Em, Z),
        (Z, Em),
        (Em, Err),
        (Z, Err),
        (Zfree, Err),
        (Zx, Zfree),
        (Z, Sup),
        (Em, Sup),
        (Err, Sup),
        (Rew, Sup),
    ];
    for &flip in flips {
        for k in 2..=K_MAX {
            for s in [Slice::All, Slice::Seen, Slice::Cn, Slice::Icn, Slice::Ue] {
                for &(x, base) in &base_pairs {
                    let cmp = compare(rows, flip, k, s, x, base);
                    if cmp.acc.n == 0 {
                        continue;
                    }
                    let _ = writeln!(
                        o,
                        "{flip},{k},{},{},{},{},{},{},{},{}",
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
        let _ = writeln!(
            o,
            "manipulation: flip={flip} differ_frac(Zx vs Z)={:.3} differ_frac(Zx vs Zfree)={:.3} binding_frac(Em)={:.3} mean_pattern_writes(Z)={:.1} mean_pattern_writes(Zx)={:.1} mean_writes(error-gated)={:.1} mean_cue_writes(Zx)={:.1} median_cn={}",
            differ_frac(rows, flip, Zx, Z),
            differ_frac(rows, flip, Zx, Zfree),
            binding_frac(rows, flip, Em),
            setting_mean(rows, flip, Z, &|r| r.pattern_writes() as f64),
            setting_mean(rows, flip, Zx, &|r| r.pattern_writes() as f64),
            setting_mean(rows, flip, Err, &|r| r.total_writes() as f64),
            setting_mean(rows, flip, Zx, &|r| r.cue_writes() as f64),
            median_cn(rows, flip)
        );
    }
    let (g, h) = readings(rows, flips);
    let _ = writeln!(
        o,
        "readings: G1_integrity={} G2_manipulation={} G3_cn_pairs={} H1={} H2={} H3={} H4={} H5={} H6={} H7={} H8={}",
        g.0,
        g.1,
        g.2,
        yes(h[0]),
        yes(h[1]),
        yes(h[2]),
        yes(h[3]),
        yes(h[4]),
        yes(h[5]),
        yes(h[6]),
        yes(h[7])
    );
    o
}

/// One pilot census line: (seed, setting, pattern writes, cue writes,
/// total writes, refused).
pub type CensusLine = (u64, Setting, u64, u64, u64, u64);

/// Pilot census: writes and refusals only; no accuracy is computed.
pub fn census(graph: Option<&Connectome>, flip: u32, seeds: &[u64]) -> io::Result<Vec<CensusLine>> {
    let spec = GenSpec {
        flip_pct: flip,
        ..DESIGN.spec
    };
    let mut out = Vec::new();
    for &seed in seeds {
        let fx = ns::fixture(graph, &DESIGN, flip, seed)?;
        let e = run_setting(&fx, &spec, seed, Setting::Err, &wb::free())?;
        let curves: Vec<Vec<u64>> = e.1.iter().map(|t| t.ledger.cum.clone()).collect();
        let paced = wb::limits(&curves, BUDGET);
        let mut z: Option<[TaskLog; K_MAX]> = None;
        for s in SETTINGS {
            let lim = match s {
                Setting::Err | Setting::Zfree => wb::free(),
                Setting::Em => pattern_limits(z.as_ref().ok_or_else(|| bad("Em needs Z"))?),
                _ => paced.clone(),
            };
            // Training only (evaluation is skipped: no accuracy computed).
            let mut c = ci::initial_circuit(&fx, &spec, s.arm())?;
            let mut logs: [TaskLog; K_MAX] = Default::default();
            for (i, l) in lim.iter().enumerate() {
                let n = ts::task_order(&spec, seed, i).len();
                let mut cp = Checkpoint::start(i, fx.graph.edges(), l);
                train_range(&mut c, &fx, &spec, seed, s, l, &mut cp, n)?;
                logs[i] = TaskLog {
                    ledger: cp.inner.ledger,
                    stats: cp.inner.stats,
                    cue_cum: cp.cue_cum,
                    ..TaskLog::default()
                };
            }
            let ex = s.exempt();
            out.push((
                seed,
                s,
                logs.iter().map(|t| t.pattern_writes(ex)).sum(),
                logs.iter().map(|t| t.stats.cue_writes).sum(),
                logs.iter().map(|t| t.total_writes(ex)).sum(),
                logs.iter().map(|t| t.ledger.refused).sum(),
            ));
            if s == Setting::Z {
                z = Some(logs);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generalization_bench::GraphShape;

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
        std::env::temp_dir().join(format!("v888-cue-disp-{tag}-{}", std::process::id()))
    }

    fn paced_of(fx: &Fixture, d: &Design, seed: u64, b: Budget) -> [Limit; K_MAX] {
        let (_, eg, _) = run_setting(fx, &d.spec, seed, Setting::Err, &wb::free()).unwrap();
        let curves: Vec<Vec<u64>> = eg.iter().map(|t| t.ledger.cum.clone()).collect();
        wb::limits(&curves, b)
    }

    #[test]
    fn charged_settings_match_cue_init_bit_exactly() {
        let d = small();
        for seed in [1, 2, 3] {
            let fx = ns::fixture(None, &d, 50, seed).unwrap();
            for b in [Budget::Free, Budget::Paced(100), Budget::Paced(50)] {
                let lim = paced_of(&fx, &d, seed, b);
                for s in [Setting::Sup, Setting::Rew, Setting::Err, Setting::Z] {
                    let (r, logs, w) = run_setting(&fx, &d.spec, seed, s, &lim).unwrap();
                    let (r0, l0, s0, c0, w0) =
                        ci::run_arm(&fx, &d.spec, seed, s.arm(), &lim).unwrap();
                    assert_eq!(w, w0, "{} {}", s.name(), b.name());
                    assert_eq!(r, r0, "{} {}", s.name(), b.name());
                    for i in 0..K_MAX {
                        assert_eq!(logs[i].ledger, l0[i]);
                        assert_eq!(logs[i].stats, s0[i]);
                        assert_eq!(logs[i].cue_nonzero_end, c0[i].cue_nonzero_end);
                        assert_eq!(logs[i].cue_mass_end, c0[i].cue_mass_end);
                        assert_eq!(logs[i].cue_cum.len(), logs[i].ledger.cum.len());
                        assert_eq!(
                            logs[i].cue_cum.last().copied().unwrap_or(0),
                            s0[i].cue_writes
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn exemption_is_inert_uncapped_and_frees_only_cue_writes_when_capped() {
        let d = small();
        let mut differ = 0;
        for seed in [1, 2, 3, 4] {
            let fx = ns::fixture(None, &d, 50, seed).unwrap();
            let free = wb::free();
            let z = run_setting(&fx, &d.spec, seed, Setting::Zfree, &free).unwrap();
            let zx = run_setting(&fx, &d.spec, seed, Setting::Zx, &free).unwrap();
            assert_eq!(z.2, zx.2);
            assert_eq!(z.0, zx.0);
            for i in 0..K_MAX {
                assert_eq!(z.1[i].total_writes(false), zx.1[i].total_writes(true));
                assert_eq!(z.1[i].pattern_writes(false), zx.1[i].pattern_writes(true));
            }
            let lim = paced_of(&fx, &d, seed, Budget::Paced(100));
            let zc = run_setting(&fx, &d.spec, seed, Setting::Z, &lim).unwrap();
            let zxc = run_setting(&fx, &d.spec, seed, Setting::Zx, &lim).unwrap();
            for i in 0..K_MAX {
                // Z: everything charged; Zx: only pattern writes charged.
                assert!(zc.1[i].total_writes(false) <= zc.1[i].ledger.cap);
                assert!(zxc.1[i].pattern_writes(true) <= zxc.1[i].ledger.cap);
                assert_eq!(zxc.1[i].stats.cue_refused, 0);
                assert_eq!(zxc.1[i].ledger.cap, zc.1[i].ledger.cap);
            }
            if zc.2 != zxc.2 {
                differ += 1;
            }
            // Em is capped at Z's pattern-write curve.
            let em_lim = pattern_limits(&zc.1);
            let em = run_setting(&fx, &d.spec, seed, Setting::Em, &em_lim).unwrap();
            for i in 0..K_MAX {
                assert_eq!(em.1[i].ledger.cap, zc.1[i].pattern_writes(false));
                assert!(em.1[i].ledger.writes <= em.1[i].ledger.cap);
                assert_eq!(em.1[i].stats.cue_writes, 0);
                let curve = zc.1[i].pattern_curve(false);
                assert!(curve.windows(2).all(|w| w[0] <= w[1]));
                assert!(em.1[i].ledger.cum.iter().zip(&curve).all(|(&w, &c)| w <= c));
            }
        }
        assert!(differ > 0);
    }

    #[test]
    fn uncued_rules_reject_exemption() {
        let d = small();
        let fx = ns::fixture(None, &d, 50, 1).unwrap();
        let mut c = PlasticCircuit::new(&fx.graph, d.spec.init_weight).unwrap();
        let mut st = Stats::default();
        let mut l = WriteLedger::new(&Limit::default());
        let mut cost = PhaseCost::default();
        let p = &fx.task.train[0][0];
        let v = &fx.task.valence[0];
        assert!(step(
            &mut c,
            &fx.graph,
            Strategy::ErrorGated,
            true,
            p,
            v,
            &[],
            &mut st,
            &mut l,
            &mut cost
        )
        .is_err());
        assert!(step(
            &mut c,
            &fx.graph,
            Strategy::EeCue(Alloc::PatternFirst),
            false,
            p,
            v,
            &[],
            &mut st,
            &mut l,
            &mut cost
        )
        .is_err());
    }

    #[test]
    fn resumes_bit_exactly_from_checkpoints() {
        let d = small();
        let seed = 4;
        let fx = ns::fixture(None, &d, 50, seed).unwrap();
        let g = &fx.graph;
        let paced = paced_of(&fx, &d, seed, Budget::Paced(100));
        let (_, zl, _) = run_setting(&fx, &d.spec, seed, Setting::Z, &paced).unwrap();
        for s in SETTINGS {
            let lim = match s {
                Setting::Err | Setting::Zfree => wb::free(),
                Setting::Em => pattern_limits(&zl),
                _ => paced.clone(),
            };
            let (_, logs, wf) = run_setting(&fx, &d.spec, seed, s, &lim).unwrap();
            let mut c = ci::initial_circuit(&fx, &d.spec, s.arm()).unwrap();
            for i in 0..K_MAX {
                let n = ts::task_order(&d.spec, seed, i).len();
                let mut cp = Checkpoint::start(i, g.edges(), &lim[i]);
                for end in [n / 3, n / 2, n] {
                    train_range(&mut c, &fx, &d.spec, seed, s, &lim[i], &mut cp, end).unwrap();
                    let (pw, pc) = (tmp("w"), tmp("c"));
                    c.save(&pw).unwrap();
                    cp.save(&pc).unwrap();
                    c = PlasticCircuit::load(&pw, g).unwrap();
                    cp = Checkpoint::load(&pc, g.edges()).unwrap();
                    std::fs::remove_file(&pw).unwrap();
                    std::fs::remove_file(&pc).unwrap();
                }
                assert_eq!(cp.inner.ledger, logs[i].ledger, "{} task {i}", s.name());
                assert_eq!(cp.inner.stats, logs[i].stats, "{} task {i}", s.name());
                assert_eq!(cp.cue_cum, logs[i].cue_cum, "{} task {i}", s.name());
            }
            assert_eq!(c.weights(), &wf[..], "{}", s.name());
        }
        // Corrupt checkpoints are rejected.
        let cp = Checkpoint::start(0, g.edges(), &Limit::default());
        let enc = cp.encode();
        assert_eq!(Checkpoint::decode(&enc, g.edges()).unwrap(), cp);
        assert!(Checkpoint::decode(&enc.replace("cuecum -", "cuecum 3"), g.edges()).is_err());
        assert!(Checkpoint::decode(&enc.replacen("v888", "x", 1), g.edges()).is_err());
        assert!(Checkpoint::decode(&enc, g.edges() + 1).is_err());
    }

    #[test]
    fn grid_is_deterministic_csv_complete_and_readings_gate() {
        let d = small();
        let flips = [50, 100];
        let run = || run_grid_design(None, &d, &flips, &[1, 2, 3]).unwrap();
        let a = run();
        let b = run();
        assert_eq!(a, b);
        assert_eq!(a.len(), flips.len() * 3 * SETTINGS.len());
        assert!(a
            .iter()
            .all(|r| r.inner.topology_unchanged && r.init_ok && r.within_cap()));
        assert!(em_matched(&a) && zx_matched(&a));
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        let cols = csv_header().split(',').count();
        assert!(csv.lines().all(|l| l.split(',').count() == cols));
        let id = compare(&a, 50, K_MAX, Slice::All, Setting::Err, Setting::Err);
        assert_eq!(id.acc.ties, 3);
        let (g, h) = readings(&a, &flips);
        assert!(g.0);
        if !(g.0 && g.1 && g.2) {
            assert_eq!(h, [None; N_HYP]);
        } else {
            assert!(h.iter().all(Option::is_some));
        }
        let rep = report(&a, &flips);
        assert!(rep.contains("readings: G1_integrity=true"));
        assert_eq!(rep, report(&b, &flips));
        assert!(run_grid_design(None, &d, &[50], &[]).is_err());
        assert_eq!(
            Setting::parse("E/E-cue0-exempt@paced100").unwrap(),
            Setting::Zx
        );
        assert!(Setting::parse("E-tag").is_err());
    }

    #[test]
    fn equivalence_and_noninferiority_rules() {
        let mk = |lo: f64, hi: f64| Paired {
            n: 10,
            mean: (lo + hi) / 2.0,
            sd: 0.0,
            better: 5,
            ties: 0,
            worse: 5,
            ci_lo: lo,
            ci_hi: hi,
        };
        assert!(equivalent(&mk(-0.04, 0.04)));
        assert!(!equivalent(&mk(-0.06, 0.0)));
        assert!(!equivalent(&mk(0.0, 0.05)));
        assert!(noninferior(&mk(-0.049, 0.2)));
        assert!(!noninferior(&mk(-0.05, 0.2)));
        let (g, h) = readings(&[], &FLIPS);
        assert!(!g.0);
        assert_eq!(h, [None; N_HYP]);
    }
}

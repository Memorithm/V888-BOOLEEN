//! B888-4 cycle 19: **can a consolidation guard cut forgetting at large k
//! without losing the error gate's learning, at equal write budget?**
//!
//! Cycle 18 (#131) found that at k = 8 tasks on held-out orders the plain
//! error gate (`error-gated`, E) learns almost everything (LEARN 0.99) but
//! forgets about half of it (FGT 0.48), so its lead over the supervised
//! control shrinks with k (+0.123 at k = 2, +0.054 at k = 4, +0.020 at
//! k = 8). This module adds a **consolidation guard** on the original
//! connections: at the end of every stage, each edge whose weight moved
//! during that stage is *consolidated* with the sign of its net change. In
//! later stages a weight-changing step **against** an edge's consolidated
//! sign is an *overwrite*. Guards (constants fixed a priori):
//!
//! * `block` (Gb): every overwrite is refused (counted, not retried).
//! * `block-recent2` (Gr): an overwrite is refused only if the edge was
//!   consolidated at the end of one of the last `RECENT_WINDOW` = 2 stages;
//!   older consolidations are plastic again (a stability/plasticity window
//!   between `off` and `block`).
//! * `charge4` (Gc): an overwrite is allowed but charged `GUARD_COST` = 4
//!   writes against the same `paced100` budget.
//! * `last-resort` (Gl, a stability/plasticity gate): an overwrite on edge
//!   e → t is deferred while target t has at least one offered edge that
//!   would change its weight **without** an overwrite (a "free" edge); if
//!   none exists, **one** overwrite per (episode, target) is allowed (the
//!   first in source order) and the others are deferred. Learning can
//!   always proceed; free capacity is used first and overwrites are
//!   rationed.
//! * `off`: no guard; bit-identical to the cycle 18 code path
//!   (`seq_generalisation::run_setting`) for `error-gated` and
//!   `E/E-cue0` (tested).
//!
//! Only weights of original edges change; there is no rehearsal buffer and
//! no extra memory of patterns (the guard state is one sign per edge).
//!
//! Engine state for checkpoint/resume = circuit weights
//! (`PlasticCircuit::save`) + [`Checkpoint`] (guard, per-edge consolidated
//! signs, stage-start weights, guard counters and the cycle 18 checkpoint);
//! resuming mid-stage is bit-exact for every setting and corrupt checkpoints
//! are rejected (tested).
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph with 4-bit
//! weights; "consolidation", "guard" and "forgetting" name parts of an
//! artificial benchmark, not claims about synapses, biology, insect memory
//! or BANC physiology.
//!
//! Protocol and constants are pre-registered in
//! `docs/CONSOLIDATION_GUARD.md` (committed and pushed before any learning
//! run on the hypothesis seeds).
use crate::connectome::Connectome;
use crate::context_cue as cc;
use crate::cue_displacement::{self as cd, TaskLog};
use crate::cue_init as ci;
use crate::cue_reach as cr;
use crate::error_gated::{drive, paired, Paired, PhaseCost, EFFECT, THRESHOLD};
use crate::generalization_bench::GenSpec;
use crate::novel_split::{self as ns, Design, Eval, Fixture, Slice};
use crate::plasticity::{decrement_four, increment_four, PlasticCircuit};
use crate::seq_generalisation as sg;
use crate::task_sequence as ts;
use crate::write_allocation::{self as wa, Stats};
use crate::write_budget::{Limit, WriteLedger};
use std::io::{self, ErrorKind};
use std::path::Path;

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Pre-registered sequence lengths (k = 4 reference fixture, k = 8 main).
pub const KS: [usize; 2] = [4, 8];
pub const K_MAIN: usize = 8;
pub const K_SMALL: usize = 4;
/// Pre-registered flips (50 primary, 100 high conflict, 25 secondary).
pub const FLIPS: [u32; 3] = [25, 50, 100];
pub const PRIMARY_FLIP: u32 = 50;
pub const HIGH_FLIP: u32 = 100;
/// Pre-registered seeds: 361..=400, never used by an earlier B888-4 cycle.
pub const SEEDS: std::ops::RangeInclusive<u64> = 361..=400;
/// Pilot seeds (writes and guard counters only, identity order only).
pub const PILOT_SEEDS: std::ops::RangeInclusive<u64> = 1..=5;
/// Held-out task orders per seed (in addition to the identity order).
pub const HELDOUT_ORDERS: usize = 2;
/// Salt of the held-out order generator (new for this cycle).
pub const ORDER_SALT: u64 = 0x0DE5_B888_4C19_0A1D;
/// The #124–#131 fixed-pool design.
pub const DESIGN: Design = sg::DESIGN;
/// Paced budget of every capped setting (percent of `error-gated`).
pub const PACED_PCT: u64 = 100;
/// Charged writes per overwrite under `charge4`.
pub const GUARD_COST: u64 = 4;
/// Protection window (stages) of `block-recent2`.
pub const RECENT_WINDOW: usize = 2;
/// H1 bound on the forgetting cut; H1 / H9 / H10 bound on the learning
/// loss.
pub const FGT_CUT: f64 = 0.10;
pub const LEARN_LOSS: f64 = 0.10;

/// Consolidation guard.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Guard {
    Off,
    Block,
    Recent,
    Charge,
    LastResort,
}

/// All guards.
pub const GUARDS: [Guard; 5] = [
    Guard::Off,
    Guard::Block,
    Guard::Recent,
    Guard::Charge,
    Guard::LastResort,
];

impl Guard {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Block => "block",
            Self::Recent => "block-recent2",
            Self::Charge => "charge4",
            Self::LastResort => "last-resort",
        }
    }
    pub fn parse(s: &str) -> io::Result<Self> {
        GUARDS
            .into_iter()
            .find(|g| g.name() == s)
            .ok_or_else(|| bad("unknown guard"))
    }
}

/// One pre-registered setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Setting {
    /// `supervised@paced100` (control).
    Sup,
    /// `reward@paced100`.
    Rew,
    /// Uncapped `error-gated` (= its `paced100` run).
    Err,
    /// `E/E-cue0@paced100` (silent cue).
    Z,
    /// `error-gated+block@paced100`.
    Gb,
    /// `error-gated+block-recent2@paced100`.
    Gr,
    /// `error-gated+charge4@paced100`.
    Gc,
    /// `error-gated+last-resort@paced100`.
    Gl,
    /// `E/E-cue0+last-resort@paced100`.
    Zg,
}

/// The pre-registered settings, in CSV order.
pub const SETTINGS: [Setting; 9] = [
    Setting::Sup,
    Setting::Rew,
    Setting::Err,
    Setting::Z,
    Setting::Gb,
    Setting::Gr,
    Setting::Gc,
    Setting::Gl,
    Setting::Zg,
];

impl Setting {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Sup => "supervised@paced100",
            Self::Rew => "reward@paced100",
            Self::Err => "error-gated@free",
            Self::Z => "E/E-cue0@paced100",
            Self::Gb => "error-gated+block@paced100",
            Self::Gr => "error-gated+block-recent2@paced100",
            Self::Gc => "error-gated+charge4@paced100",
            Self::Gl => "error-gated+last-resort@paced100",
            Self::Zg => "E/E-cue0+last-resort@paced100",
        }
    }
    pub fn parse(s: &str) -> io::Result<Self> {
        SETTINGS
            .iter()
            .copied()
            .find(|x| x.name() == s)
            .ok_or_else(|| bad("unknown setting"))
    }
    /// The cycle 18 setting whose rule and initial weights this one uses.
    #[must_use]
    pub fn base(self) -> cd::Setting {
        match self {
            Self::Sup => cd::Setting::Sup,
            Self::Rew => cd::Setting::Rew,
            Self::Err | Self::Gb | Self::Gr | Self::Gc | Self::Gl => cd::Setting::Err,
            Self::Z | Self::Zg => cd::Setting::Z,
        }
    }
    /// Guard of the guarded step (`None` = the cycle 18 step, uninstrumented).
    #[must_use]
    pub fn guard(self) -> Option<Guard> {
        match self {
            Self::Sup | Self::Rew => None,
            Self::Err | Self::Z => Some(Guard::Off),
            Self::Gb => Some(Guard::Block),
            Self::Gr => Some(Guard::Recent),
            Self::Gc => Some(Guard::Charge),
            Self::Gl | Self::Zg => Some(Guard::LastResort),
        }
    }
    #[must_use]
    pub fn cued(self) -> bool {
        self.base().cued()
    }
}

/// Guard counters of one stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GuardStats {
    /// Effective (weight-changing) writes.
    pub effective: u64,
    /// Effective writes that were overwrites of a consolidated edge.
    pub overwrites: u64,
    /// Overwrites refused by `block` / `block-recent2`.
    pub blocked: u64,
    /// Overwrites deferred by `last-resort`.
    pub deferred: u64,
}

impl GuardStats {
    /// Charged writes these counters imply under `guard`.
    #[must_use]
    pub fn charged(&self, guard: Guard) -> u64 {
        if guard == Guard::Charge {
            self.effective + (GUARD_COST - 1) * self.overwrites
        } else {
            self.effective
        }
    }
}

/// Per-stage guard record (all settings; per-episode counters only for
/// guarded-step settings, `None` otherwise).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GuardLog {
    pub stats: Option<GuardStats>,
    /// Edges whose net change in this stage reversed their consolidated sign
    /// (computed from stage-start / stage-end weights; every setting).
    pub net_reversals: u64,
    /// Edges whose weight moved in this stage.
    pub moved: u64,
    /// Consolidated edges after this stage.
    pub consolidated: u64,
}

/// One episode of stage `stage` of the guarded error gate (uncued, or
/// source-major `E/E-cue` when `cue` is non-empty). `dir[e]` = consolidated
/// sign of edge e (+1 up, −1 down, 0 none), `stamp[e]` = 1 + the stage at
/// whose end it was last consolidated (0 = never). With `Guard::Off` the weights, ledger, stats
/// and cost are bit-identical to `cue_displacement::step` without exemption
/// (tested through the runs).
#[allow(clippy::too_many_arguments)]
pub fn step(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    guard: Guard,
    dir: &[i8],
    stamp: &[u8],
    stage: usize,
    pattern: &[usize],
    valence: &[bool],
    cue: &[usize],
    stats: &mut Stats,
    g: &mut GuardStats,
    ledger: &mut WriteLedger,
    cost: &mut PhaseCost,
) -> io::Result<()> {
    let mut skipped = 0;
    step_masked(
        circuit,
        graph,
        guard,
        dir,
        stamp,
        stage,
        pattern,
        valence,
        cue,
        &[],
        &mut skipped,
        stats,
        g,
        ledger,
        cost,
    )
}

/// [`step`] with a set of **frozen** source lines (cycle 24, owned-write
/// cue codes): frozen lines stay active in the drive, but their edges are
/// never written, never offered to the budget and never count as a free
/// edge for `last-resort`. `skipped` counts the edge steps the mask removed
/// (offered target, weight would change). With `frozen` empty this is
/// exactly [`step`] (tested).
#[allow(clippy::too_many_arguments)]
pub fn step_masked(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    guard: Guard,
    dir: &[i8],
    stamp: &[u8],
    stage: usize,
    pattern: &[usize],
    valence: &[bool],
    cue: &[usize],
    frozen: &[usize],
    skipped: &mut u64,
    stats: &mut Stats,
    g: &mut GuardStats,
    ledger: &mut WriteLedger,
    cost: &mut PhaseCost,
) -> io::Result<()> {
    if dir.len() != graph.edges()
        || stamp.len() != graph.edges()
        || circuit.weights().len() != graph.edges()
    {
        return Err(bad("guard shape mismatch"));
    }
    let recent = |e: usize| {
        let s = usize::from(stamp[e]);
        s > 0 && s - 1 + RECENT_WINDOW >= stage
    };
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
    let next_of = |w: u8, up: bool| {
        if up {
            increment_four(w)
        } else {
            decrement_four(w)
        }
    };
    let against = |e: usize, up: bool| dir[e] == if up { -1 } else { 1 };
    // last-resort: does target t have a free (changing, non-overwrite) edge
    // among the active sources (weights before the update; budget ignored)?
    let mut has_free = vec![false; n];
    if guard == Guard::LastResort {
        for &source in sources.iter().filter(|s| !frozen.contains(s)) {
            for edge in graph.offsets[source]..graph.offsets[source + 1] {
                let edge = edge as usize;
                let t = graph.targets[edge] as usize;
                if offer[t] {
                    let w = circuit.weights()[edge];
                    if next_of(w, want[t]) != w && !against(edge, want[t]) {
                        has_free[t] = true;
                    }
                }
            }
        }
    }
    // last-resort: at most one overwrite per (episode, target).
    let mut over_used = vec![false; n];
    for &source in &sources {
        let on_cue = cue.contains(&source);
        let is_frozen = frozen.contains(&source);
        for edge in graph.offsets[source]..graph.offsets[source + 1] {
            let edge = edge as usize;
            let t = graph.targets[edge] as usize;
            if !offer[t] {
                continue;
            }
            let up = want[t];
            let w = circuit.weights()[edge];
            let next = next_of(w, up);
            if is_frozen {
                if next != w {
                    *skipped += 1;
                }
                continue;
            }
            let over = next != w && against(edge, up);
            if over {
                match guard {
                    Guard::Block => {
                        g.blocked += 1;
                        continue;
                    }
                    Guard::Recent if recent(edge) => {
                        g.blocked += 1;
                        continue;
                    }
                    Guard::LastResort if has_free[t] || over_used[t] => {
                        g.deferred += 1;
                        continue;
                    }
                    _ => {}
                }
            }
            let units = if over && guard == Guard::Charge {
                GUARD_COST
            } else {
                1
            };
            if next != w && ledger.writes.saturating_add(units) > ledger.allow {
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
                ledger.writes += units;
                g.effective += 1;
                if over {
                    g.overwrites += 1;
                    over_used[t] = true;
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

/// Resumable state of one run: guard, consolidated signs, stage-start
/// weights, guard counters of the stage and the cycle 18 checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub guard: Guard,
    pub dir: Vec<i8>,
    pub stamp: Vec<u8>,
    pub start: Vec<u8>,
    pub g: GuardStats,
    pub inner: sg::Checkpoint,
}

const MAGIC: &str = "v888-consolidation-guard-checkpoint-v1";

impl Checkpoint {
    /// Start of `stage` with the consolidation state (`dir`, `stamp`) and
    /// the current weights `start`.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        order: &[usize],
        stage: usize,
        limit: &Limit,
        guard: Guard,
        dir: &[i8],
        stamp: &[u8],
        start: &[u8],
    ) -> io::Result<Self> {
        if dir.len() != start.len() || stamp.len() != start.len() {
            return Err(bad("guard state shape mismatch"));
        }
        Ok(Self {
            guard,
            dir: dir.to_vec(),
            stamp: stamp.to_vec(),
            start: start.to_vec(),
            g: GuardStats::default(),
            inner: sg::Checkpoint::start(order, stage, start.len(), limit)?,
        })
    }
    /// Deterministic text serialisation.
    #[must_use]
    pub fn encode(&self) -> String {
        let dir: String = self
            .dir
            .iter()
            .map(|&d| match d {
                1 => '+',
                -1 => '-',
                _ => '0',
            })
            .collect();
        let start: String = self
            .start
            .iter()
            .map(|&w| char::from_digit(u32::from(w & 15), 16).unwrap_or('0'))
            .collect();
        let stamp = if self.stamp.is_empty() {
            "-".to_string()
        } else {
            self.stamp
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        };
        let g = &self.g;
        format!(
            "{MAGIC}\nguard {}\ndir {}\nstamp {stamp}\nstart {}\ngstats {} {} {} {}\n{}",
            self.guard.name(),
            if dir.is_empty() { "-".to_string() } else { dir },
            if start.is_empty() {
                "-".to_string()
            } else {
                start
            },
            g.effective,
            g.overwrites,
            g.blocked,
            g.deferred,
            self.inner.encode()
        )
    }
    /// Parse [`Checkpoint::encode`]; `edges` must match the graph.
    pub fn decode(s: &str, edges: usize) -> io::Result<Self> {
        let mut it = s.splitn(7, '\n');
        if it.next() != Some(MAGIC) {
            return Err(bad("not a consolidation-guard checkpoint"));
        }
        let mut field = |name: &str| -> io::Result<String> {
            let line = it.next().ok_or_else(|| bad("truncated checkpoint"))?;
            line.strip_prefix(name)
                .and_then(|v| v.strip_prefix(' '))
                .map(str::to_string)
                .ok_or_else(|| bad("bad checkpoint field"))
        };
        let guard = Guard::parse(&field("guard")?)?;
        let ds = field("dir")?;
        let ts_ = field("stamp")?;
        let ss = field("start")?;
        let gs = field("gstats")?;
        let dir: Vec<i8> = if ds == "-" {
            Vec::new()
        } else {
            ds.chars()
                .map(|c| match c {
                    '+' => Ok(1),
                    '-' => Ok(-1),
                    '0' => Ok(0),
                    _ => Err(bad("bad consolidated sign")),
                })
                .collect::<io::Result<_>>()?
        };
        let start: Vec<u8> = if ss == "-" {
            Vec::new()
        } else {
            ss.chars()
                .map(|c| {
                    c.to_digit(16)
                        .and_then(|v| u8::try_from(v).ok())
                        .ok_or_else(|| bad("bad start weight"))
                })
                .collect::<io::Result<_>>()?
        };
        let stamp: Vec<u8> = if ts_ == "-" {
            Vec::new()
        } else {
            ts_.split(' ')
                .map(|x| x.parse::<u8>().map_err(|_| bad("bad stamp")))
                .collect::<io::Result<_>>()?
        };
        if dir.len() != edges || start.len() != edges || stamp.len() != edges {
            return Err(bad("checkpoint edge count mismatch"));
        }
        if dir.iter().zip(&stamp).any(|(&d, &t)| (d == 0) != (t == 0)) {
            return Err(bad("consolidated signs and stamps disagree"));
        }
        let v: Vec<u64> = gs
            .split(' ')
            .map(|x| x.parse::<u64>().map_err(|_| bad("bad number")))
            .collect::<io::Result<_>>()?;
        if v.len() != 4 || v[1] > v[0] {
            return Err(bad("bad guard counters"));
        }
        let g = GuardStats {
            effective: v[0],
            overwrites: v[1],
            blocked: v[2],
            deferred: v[3],
        };
        if (!matches!(guard, Guard::Block | Guard::Recent) && g.blocked != 0)
            || (guard != Guard::LastResort && g.deferred != 0)
            || (guard == Guard::Block && g.overwrites != 0)
        {
            return Err(bad("guard counters inconsistent with the guard"));
        }
        let rest = it.next().ok_or_else(|| bad("truncated checkpoint"))?;
        let inner = sg::Checkpoint::decode(rest, edges)?;
        if inner.inner.inner.ledger.writes != g.charged(guard) {
            return Err(bad("guard counters do not match the ledger"));
        }
        if stamp.iter().any(|&t| usize::from(t) > inner.stage) {
            return Err(bad("stamp from a stage not yet trained"));
        }
        Ok(Self {
            guard,
            dir,
            stamp,
            start,
            g,
            inner,
        })
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

/// Train episodes `episode..end` of the current stage with the guarded
/// step (same episode loop as `cue_displacement::train_range`).
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
    let mut skipped = 0;
    train_range_masked(
        circuit,
        fx,
        spec,
        seed,
        s,
        limit,
        cp,
        end,
        &[],
        &mut skipped,
    )
}

/// [`train_range`] with the stage's frozen lines (see [`step_masked`]);
/// adds the mask's skipped edge steps to `skipped`. The mask is a pure
/// function of the fixture and code, not engine state, so the
/// [`Checkpoint`] is unchanged and resuming with the same mask is bit-exact.
#[allow(clippy::too_many_arguments)]
pub fn train_range_masked(
    circuit: &mut PlasticCircuit,
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    s: Setting,
    limit: &Limit,
    cp: &mut Checkpoint,
    end: usize,
    frozen: &[usize],
    skipped: &mut u64,
) -> io::Result<()> {
    if s.guard() != Some(cp.guard) {
        return Err(bad("checkpoint guard does not match the setting"));
    }
    let o = &mut cp.inner;
    if o.k != fx.task.train.len() || o.inner.inner.task != o.order[o.stage] {
        return Err(bad("checkpoint does not match the fixture"));
    }
    if cp.dir.len() != fx.graph.edges() || cp.stamp.len() != fx.graph.edges() {
        return Err(bad("guard state shape mismatch"));
    }
    let i = o.inner.inner.task;
    let eps = ts::task_order(spec, seed, i);
    if o.inner.inner.episode > end || end > eps.len() {
        return Err(bad("bad episode range"));
    }
    let cue: &[usize] = if s.cued() {
        fx.cues.get(i).ok_or_else(|| bad("missing cue"))?
    } else {
        &[]
    };
    for (e, &ix) in eps.iter().enumerate().take(end).skip(o.inner.inner.episode) {
        let c = &mut o.inner.inner;
        c.ledger.allow = limit_at(limit, e);
        step_masked(
            circuit,
            &fx.graph,
            cp.guard,
            &cp.dir,
            &cp.stamp,
            o.stage,
            &fx.task.train[i][ix],
            &fx.task.valence[i],
            cue,
            frozen,
            skipped,
            &mut c.stats,
            &mut cp.g,
            &mut c.ledger,
            &mut c.cost,
        )?;
        o.inner.cue_cum.push(c.stats.cue_writes);
    }
    o.inner.inner.episode = end;
    if end == eps.len() {
        o.inner.inner.ledger.allow = o.inner.inner.ledger.cap;
    }
    Ok(())
}

/// Consolidate after stage `stage`: returns (net reversals, moved), sets
/// `dir` to the sign of every moved edge's net change and its `stamp` to
/// `stage + 1`.
pub fn consolidate(
    dir: &mut [i8],
    stamp: &mut [u8],
    stage: usize,
    start: &[u8],
    end: &[u8],
) -> io::Result<(u64, u64)> {
    if dir.len() != start.len() || stamp.len() != start.len() || start.len() != end.len() {
        return Err(bad("consolidation shape mismatch"));
    }
    let st = u8::try_from(stage + 1).map_err(|_| bad("too many stages"))?;
    let (mut rev, mut moved) = (0, 0);
    for (((d, t), &a), &b) in dir.iter_mut().zip(stamp.iter_mut()).zip(start).zip(end) {
        if a != b {
            let s: i8 = if b > a { 1 } else { -1 };
            moved += 1;
            if *d == -s {
                rev += 1;
            }
            *d = s;
            *t = st;
        }
    }
    Ok((rev, moved))
}

/// Train all of stage `i` (task `order[i]`) and consolidate. Settings
/// without a guard train through `seq_generalisation::train_range` (the
/// cycle 18 code path). Returns the final cycle 18 checkpoint of the stage
/// and its guard record.
#[allow(clippy::too_many_arguments)]
pub fn train_stage(
    circuit: &mut PlasticCircuit,
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    s: Setting,
    order: &[usize],
    i: usize,
    limit: &Limit,
    dir: &mut [i8],
    stamp: &mut [u8],
) -> io::Result<(sg::Checkpoint, GuardLog)> {
    let mut skipped = 0;
    train_stage_masked(
        circuit,
        fx,
        spec,
        seed,
        s,
        order,
        i,
        limit,
        dir,
        stamp,
        &[],
        &mut skipped,
    )
}

/// [`train_stage`] with the stage's frozen lines (guarded-step settings
/// only; a non-empty mask on a setting without a guarded step is an error).
#[allow(clippy::too_many_arguments)]
pub fn train_stage_masked(
    circuit: &mut PlasticCircuit,
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    s: Setting,
    order: &[usize],
    i: usize,
    limit: &Limit,
    dir: &mut [i8],
    stamp: &mut [u8],
    frozen: &[usize],
    skipped: &mut u64,
) -> io::Result<(sg::Checkpoint, GuardLog)> {
    let t = *order.get(i).ok_or_else(|| bad("stage out of range"))?;
    let episodes = ts::task_order(spec, seed, t).len();
    let start = circuit.weights().to_vec();
    let (inner, stats) = match s.guard() {
        None if !frozen.is_empty() => {
            return Err(bad("frozen lines need a guarded-step setting"));
        }
        None => {
            let mut cp = sg::Checkpoint::start(order, i, fx.graph.edges(), limit)?;
            sg::train_range(circuit, fx, spec, seed, s.base(), limit, &mut cp, episodes)?;
            (cp, None)
        }
        Some(gd) => {
            let mut cp = Checkpoint::start(order, i, limit, gd, dir, stamp, &start)?;
            train_range_masked(
                circuit, fx, spec, seed, s, limit, &mut cp, episodes, frozen, skipped,
            )?;
            (cp.inner, Some(cp.g))
        }
    };
    let (net_reversals, moved) = consolidate(dir, stamp, i, &start, circuit.weights())?;
    Ok((
        inner,
        GuardLog {
            stats,
            net_reversals,
            moved,
            consolidated: dir.iter().filter(|&&d| d != 0).count() as u64,
        },
    ))
}

/// Output of [`run_setting`].
pub struct Run {
    pub ev: Vec<Vec<Eval>>,
    pub logs: Vec<TaskLog>,
    pub glogs: Vec<GuardLog>,
    pub cost: Vec<PhaseCost>,
    pub weights: Vec<u8>,
    pub topology_unchanged: bool,
}

/// Run one setting over the fixture's tasks in `order` with per-stage write
/// limits; same evaluation as `seq_generalisation::run_setting`.
pub fn run_setting(
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    s: Setting,
    order: &[usize],
    limits: &[Limit],
) -> io::Result<Run> {
    Ok(run_setting_masked(fx, spec, seed, s, order, limits, &[])?.0)
}

/// Per-stage record of the frozen-line mask of [`run_setting_masked`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MaskLog {
    /// Edge steps removed by the mask, per stage.
    pub skipped: Vec<u64>,
    /// Edges of the stage's frozen lines whose weight changed during that
    /// stage, per stage (0 by construction; checked, not assumed).
    pub moved: Vec<u64>,
}

/// Edges leaving `lines` whose weight differs between `before` and `after`.
#[must_use]
pub fn moved_on_lines(graph: &Connectome, lines: &[usize], before: &[u8], after: &[u8]) -> u64 {
    lines
        .iter()
        .map(|&src| {
            let (lo, hi) = (graph.offsets[src] as usize, graph.offsets[src + 1] as usize);
            before[lo..hi]
                .iter()
                .zip(&after[lo..hi])
                .filter(|(a, b)| a != b)
                .count() as u64
        })
        .sum()
}

/// [`run_setting`] with per-task frozen lines: `frozen` is empty (no mask,
/// bit-identical to [`run_setting`], tested) or has one line list per task
/// (indexed by task id, applied while that task is trained).
pub fn run_setting_masked(
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    s: Setting,
    order: &[usize],
    limits: &[Limit],
    frozen: &[Vec<usize>],
) -> io::Result<(Run, MaskLog)> {
    let graph = &fx.graph;
    let k = fx.task.train.len();
    if fx.cues.len() != k || fx.splits.len() != k || limits.len() != k {
        return Err(bad("need k tasks, cues, splits and limits"));
    }
    if !frozen.is_empty() && frozen.len() != k {
        return Err(bad("need no mask or one frozen-line list per task"));
    }
    if frozen
        .iter()
        .flatten()
        .any(|&src| src + 1 >= graph.offsets.len())
    {
        return Err(bad("frozen line out of range"));
    }
    let mut mask = MaskLog::default();
    if !sg::is_permutation(order, k) {
        return Err(bad("order is not a permutation of the tasks"));
    }
    graph.validate()?;
    let original = graph.clone();
    let cue = ci::cue_edges(fx)?;
    let mut circuit = ci::initial_circuit(fx, spec, s.base().arm())?;
    let fingerprint = circuit.topology_fingerprint();
    let mut dir = vec![0_i8; graph.edges()];
    let mut stamp = vec![0_u8; graph.edges()];
    let mut ev = Vec::with_capacity(k);
    let mut logs = Vec::with_capacity(k);
    let mut glogs = Vec::with_capacity(k);
    let mut cost = Vec::with_capacity(k);
    for i in 0..k {
        let fz: &[usize] = frozen.get(order[i]).map_or(&[], Vec::as_slice);
        let before = circuit.weights().to_vec();
        let mut skipped = 0;
        let (inner, gl) = train_stage_masked(
            &mut circuit,
            fx,
            spec,
            seed,
            s,
            order,
            i,
            &limits[i],
            &mut dir,
            &mut stamp,
            fz,
            &mut skipped,
        )?;
        let moved = moved_on_lines(graph, fz, &before, circuit.weights());
        mask.skipped.push(skipped);
        mask.moved.push(moved);
        glogs.push(gl);
        let (mut nz, mut mass) = (0, 0);
        for (&x, &on) in circuit.weights().iter().zip(&cue) {
            if on && x > 0 {
                nz += 1;
                mass += u64::from(x);
            }
        }
        cost.push(inner.inner.inner.cost);
        logs.push(TaskLog {
            ledger: inner.inner.inner.ledger,
            stats: inner.inner.inner.stats,
            cue_cum: inner.inner.cue_cum,
            cue_nonzero_end: nz,
            cue_mass_end: mass,
        });
        let mut row = Vec::with_capacity(i + 1);
        for &u in &order[..=i] {
            let c: &[usize] = if s.cued() { &fx.cues[u] } else { &[] };
            row.push(ns::evaluate(
                &circuit,
                graph,
                &fx.task.test,
                &fx.task.valence[u],
                c,
                &fx.splits[u],
            )?);
        }
        ev.push(row);
    }
    Ok((
        Run {
            ev,
            logs,
            glogs,
            cost,
            topology_unchanged: *graph == original
                && circuit.topology_fingerprint() == fingerprint
                && circuit.weights().len() == graph.edges(),
            weights: circuit.weights().to_vec(),
        },
        mask,
    ))
}

/// Two-phase frozen-line schedule of one task (cycle 25, deferred partner
/// writes): `early` is frozen during episodes `0..split`, `late` during
/// `split..n`, with `split = phase_split(n)`. Same masks in both halves is
/// exactly [`train_range_masked`] with that mask (tested).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StageMask {
    pub early: Vec<usize>,
    pub late: Vec<usize>,
}

impl StageMask {
    /// The same frozen lines in both halves.
    #[must_use]
    pub fn constant(lines: &[usize]) -> Self {
        Self {
            early: lines.to_vec(),
            late: lines.to_vec(),
        }
    }
    /// No frozen line in either half.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.early.is_empty() && self.late.is_empty()
    }
}

/// Episode at which a stage of `episodes` episodes switches from the early
/// to the late mask (`⌊episodes / 2⌋`).
#[must_use]
pub const fn phase_split(episodes: usize) -> usize {
    episodes / 2
}

/// Per-half counters of one scheduled stage (index 0 = early half,
/// 1 = late half).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseCount {
    /// Edge steps removed by the mask.
    pub skipped: [u64; 2],
    /// Edges of the half's frozen lines whose weight changed during that
    /// half (0 by construction; checked, not assumed).
    pub frozen_moved: [u64; 2],
    /// Edges of the watched lines whose weight changed during that half.
    pub watch_moved: [u64; 2],
}

/// [`train_range_masked`] under a two-phase schedule: episodes before
/// `split` use `mask.early`, the others `mask.late`; resumes from the
/// checkpoint's episode (the schedule is a pure function of the episode
/// index, so save/load anywhere is bit-exact). Skips are added per half.
#[allow(clippy::too_many_arguments)]
pub fn train_range_scheduled(
    circuit: &mut PlasticCircuit,
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    s: Setting,
    limit: &Limit,
    cp: &mut Checkpoint,
    end: usize,
    mask: &StageMask,
    skipped: &mut [u64; 2],
) -> io::Result<()> {
    let task = cp.inner.inner.inner.task;
    let split = phase_split(ts::task_order(spec, seed, task).len());
    let at = cp.inner.inner.inner.episode;
    if at < split {
        let e = end.min(split);
        train_range_masked(
            circuit,
            fx,
            spec,
            seed,
            s,
            limit,
            cp,
            e,
            &mask.early,
            &mut skipped[0],
        )?;
    }
    if end > split || at >= split {
        train_range_masked(
            circuit,
            fx,
            spec,
            seed,
            s,
            limit,
            cp,
            end,
            &mask.late,
            &mut skipped[1],
        )?;
    }
    Ok(())
}

/// [`train_stage_masked`] under a two-phase schedule, recording per-half
/// skips, frozen moves and moves on the `watch` lines (any setting may
/// watch; a non-empty mask needs a guarded-step setting).
#[allow(clippy::too_many_arguments)]
pub fn train_stage_scheduled(
    circuit: &mut PlasticCircuit,
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    s: Setting,
    order: &[usize],
    i: usize,
    limit: &Limit,
    dir: &mut [i8],
    stamp: &mut [u8],
    mask: &StageMask,
    watch: &[usize],
) -> io::Result<(sg::Checkpoint, GuardLog, PhaseCount)> {
    let t = *order.get(i).ok_or_else(|| bad("stage out of range"))?;
    let episodes = ts::task_order(spec, seed, t).len();
    let split = phase_split(episodes);
    let start = circuit.weights().to_vec();
    let mut pc = PhaseCount::default();
    let (inner, stats, mid) = match s.guard() {
        None if !mask.is_empty() => {
            return Err(bad("frozen lines need a guarded-step setting"));
        }
        None => {
            let mut cp = sg::Checkpoint::start(order, i, fx.graph.edges(), limit)?;
            sg::train_range(circuit, fx, spec, seed, s.base(), limit, &mut cp, split)?;
            let mid = circuit.weights().to_vec();
            sg::train_range(circuit, fx, spec, seed, s.base(), limit, &mut cp, episodes)?;
            (cp, None, mid)
        }
        Some(gd) => {
            let mut cp = Checkpoint::start(order, i, limit, gd, dir, stamp, &start)?;
            train_range_scheduled(
                circuit,
                fx,
                spec,
                seed,
                s,
                limit,
                &mut cp,
                split,
                mask,
                &mut pc.skipped,
            )?;
            let mid = circuit.weights().to_vec();
            train_range_scheduled(
                circuit,
                fx,
                spec,
                seed,
                s,
                limit,
                &mut cp,
                episodes,
                mask,
                &mut pc.skipped,
            )?;
            (cp.inner, Some(cp.g), mid)
        }
    };
    let end = circuit.weights();
    pc.frozen_moved = [
        moved_on_lines(&fx.graph, &mask.early, &start, &mid),
        moved_on_lines(&fx.graph, &mask.late, &mid, end),
    ];
    pc.watch_moved = [
        moved_on_lines(&fx.graph, watch, &start, &mid),
        moved_on_lines(&fx.graph, watch, &mid, end),
    ];
    let (net_reversals, moved) = consolidate(dir, stamp, i, &start, end)?;
    Ok((
        inner,
        GuardLog {
            stats,
            net_reversals,
            moved,
            consolidated: dir.iter().filter(|&&d| d != 0).count() as u64,
        },
        pc,
    ))
}

/// [`run_setting`] with a two-phase frozen-line schedule per task and
/// watched lines per task (cycle 25). `masks` and `watch` are each empty or
/// hold one entry per task (indexed by task id). With every mask
/// [`StageMask::constant`] this is bit-identical to [`run_setting_masked`]
/// and with no mask to [`run_setting`] (tested).
#[allow(clippy::too_many_arguments)]
pub fn run_setting_scheduled(
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    s: Setting,
    order: &[usize],
    limits: &[Limit],
    masks: &[StageMask],
    watch: &[Vec<usize>],
) -> io::Result<(Run, Vec<PhaseCount>)> {
    let graph = &fx.graph;
    let k = fx.task.train.len();
    if fx.cues.len() != k || fx.splits.len() != k || limits.len() != k {
        return Err(bad("need k tasks, cues, splits and limits"));
    }
    if (!masks.is_empty() && masks.len() != k) || (!watch.is_empty() && watch.len() != k) {
        return Err(bad("need no schedule or one entry per task"));
    }
    let lines = masks
        .iter()
        .flat_map(|m| m.early.iter().chain(&m.late))
        .chain(watch.iter().flatten());
    if lines.into_iter().any(|&src| src + 1 >= graph.offsets.len()) {
        return Err(bad("frozen or watched line out of range"));
    }
    if !sg::is_permutation(order, k) {
        return Err(bad("order is not a permutation of the tasks"));
    }
    graph.validate()?;
    let original = graph.clone();
    let cue = ci::cue_edges(fx)?;
    let mut circuit = ci::initial_circuit(fx, spec, s.base().arm())?;
    let fingerprint = circuit.topology_fingerprint();
    let mut dir = vec![0_i8; graph.edges()];
    let mut stamp = vec![0_u8; graph.edges()];
    let mut ev = Vec::with_capacity(k);
    let mut logs = Vec::with_capacity(k);
    let mut glogs = Vec::with_capacity(k);
    let mut cost = Vec::with_capacity(k);
    let mut phases = Vec::with_capacity(k);
    let none = StageMask::default();
    for i in 0..k {
        let m = masks.get(order[i]).unwrap_or(&none);
        let w: &[usize] = watch.get(order[i]).map_or(&[], Vec::as_slice);
        let (inner, gl, pc) = train_stage_scheduled(
            &mut circuit,
            fx,
            spec,
            seed,
            s,
            order,
            i,
            &limits[i],
            &mut dir,
            &mut stamp,
            m,
            w,
        )?;
        phases.push(pc);
        glogs.push(gl);
        let (mut nz, mut mass) = (0, 0);
        for (&x, &on) in circuit.weights().iter().zip(&cue) {
            if on && x > 0 {
                nz += 1;
                mass += u64::from(x);
            }
        }
        cost.push(inner.inner.inner.cost);
        logs.push(TaskLog {
            ledger: inner.inner.inner.ledger,
            stats: inner.inner.inner.stats,
            cue_cum: inner.inner.cue_cum,
            cue_nonzero_end: nz,
            cue_mass_end: mass,
        });
        let mut row = Vec::with_capacity(i + 1);
        for &u in &order[..=i] {
            let c: &[usize] = if s.cued() { &fx.cues[u] } else { &[] };
            row.push(ns::evaluate(
                &circuit,
                graph,
                &fx.task.test,
                &fx.task.valence[u],
                c,
                &fx.splits[u],
            )?);
        }
        ev.push(row);
    }
    Ok((
        Run {
            ev,
            logs,
            glogs,
            cost,
            topology_unchanged: *graph == original
                && circuit.topology_fingerprint() == fingerprint
                && circuit.weights().len() == graph.edges(),
            weights: circuit.weights().to_vec(),
        },
        phases,
    ))
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

/// Task orders of one seed: index 0 = identity, `1..=n_heldout` = distinct
/// seeded permutations ≠ identity (the cycle 18 generator with this cycle's
/// salt). Independent of the flip.
pub fn orders(seed: u64, k: usize, n_heldout: usize) -> io::Result<Vec<Vec<usize>>> {
    let fact: u128 = (1..=k as u128).product();
    if k < 2 || (n_heldout as u128) >= fact {
        return Err(bad("not enough distinct task orders"));
    }
    let mut out = vec![(0..k).collect::<Vec<usize>>()];
    let mut rng = Lcg(seed ^ ORDER_SALT);
    let mut tries = 0;
    while out.len() <= n_heldout {
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

/// One measurement: (k, flip, seed, order, setting).
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub setting: Setting,
    pub k: usize,
    pub flip: u32,
    pub seed: u64,
    pub order_ix: usize,
    pub order: Vec<usize>,
    pub ev: Vec<Vec<Eval>>,
    pub logs: Vec<TaskLog>,
    pub glogs: Vec<GuardLog>,
    pub cost: Vec<PhaseCost>,
    /// TEST CN pairs of each stage's task.
    pub cn_pairs: Vec<u64>,
    pub saturated_end: u64,
    pub topology_unchanged: bool,
    pub init_ok: bool,
    /// `error-gated@free` only: its `paced100` run is bit-identical with no
    /// refusal (true for other settings).
    pub paced_identity: bool,
    pub digest: u64,
}

fn mean(v: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = v.collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

impl Row {
    #[must_use]
    pub fn r(&self, s: Slice, i: usize, j: usize) -> f64 {
        s.pick(&self.ev[i][j]).balanced()
    }
    #[must_use]
    pub fn acc(&self, s: Slice, k: usize) -> f64 {
        mean((0..k).map(|j| self.r(s, k - 1, j)))
    }
    #[must_use]
    pub fn learn(&self, s: Slice, k: usize) -> f64 {
        mean((0..k).map(|j| self.r(s, j, j)))
    }
    #[must_use]
    pub fn forgetting(&self, s: Slice, k: usize) -> f64 {
        mean((0..k.saturating_sub(1)).map(|j| self.r(s, j, j) - self.r(s, k - 1, j)))
    }
    #[must_use]
    pub fn last(&self, s: Slice, k: usize) -> f64 {
        self.r(s, k - 1, k - 1)
    }
    #[must_use]
    pub fn charged_writes(&self) -> u64 {
        self.logs.iter().map(|t| t.ledger.writes).sum()
    }
    #[must_use]
    pub fn cap_total(&self) -> Option<u64> {
        self.logs.iter().try_fold(0_u64, |a, t| {
            (t.ledger.cap != u64::MAX).then(|| a + t.ledger.cap)
        })
    }
    /// Effective writes (= charged writes for settings without guard
    /// counters, where every charged write is effective).
    #[must_use]
    pub fn effective_writes(&self) -> u64 {
        self.logs
            .iter()
            .zip(&self.glogs)
            .map(|(t, g)| g.stats.map_or(t.ledger.writes, |s| s.effective))
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
    fn gsum(&self, f: fn(&GuardStats) -> u64) -> Option<u64> {
        self.glogs
            .iter()
            .map(|g| g.stats.as_ref().map(f))
            .sum::<Option<u64>>()
    }
    #[must_use]
    pub fn overwrites(&self) -> Option<u64> {
        self.gsum(|g| g.overwrites)
    }
    #[must_use]
    pub fn blocked(&self) -> Option<u64> {
        self.gsum(|g| g.blocked)
    }
    #[must_use]
    pub fn deferred(&self) -> Option<u64> {
        self.gsum(|g| g.deferred)
    }
    #[must_use]
    pub fn net_reversals(&self) -> u64 {
        self.glogs.iter().map(|g| g.net_reversals).sum()
    }
    #[must_use]
    pub fn consolidated_end(&self) -> u64 {
        self.glogs.last().map_or(0, |g| g.consolidated)
    }
    #[must_use]
    pub fn within_cap(&self) -> bool {
        self.logs.iter().all(|t| t.ledger.writes <= t.ledger.cap)
    }
    #[must_use]
    pub fn heldout(&self) -> bool {
        self.order_ix > 0
    }
}

/// Paced limits from per-stage cumulative write curves.
#[must_use]
pub fn paced_limits(curves: &[Vec<u64>]) -> Vec<Limit> {
    sg::paced_limits(curves, PACED_PCT)
}

/// All rows of one (k, flip, seed): per order, uncapped `error-gated`
/// first (its per-stage write curves set the `paced100` caps of every
/// other setting).
pub fn run_seed(
    graph: Option<&Connectome>,
    design: &Design,
    k: usize,
    flip: u32,
    seed: u64,
    n_heldout: usize,
) -> io::Result<Vec<Row>> {
    let spec = GenSpec {
        flip_pct: flip,
        ..design.spec
    };
    let fx = sg::fixture(graph, design, k, flip, seed)?;
    let mut rows = Vec::new();
    for (o, order) in orders(seed, k, n_heldout)?.into_iter().enumerate() {
        let e = run_setting(&fx, &spec, seed, Setting::Err, &order, &sg::free(k))?;
        let curves: Vec<Vec<u64>> = e.logs.iter().map(|t| t.ledger.cum.clone()).collect();
        let paced = paced_limits(&curves);
        let e_paced = run_setting(&fx, &spec, seed, Setting::Err, &order, &paced)?;
        let identity =
            e_paced.weights == e.weights && e_paced.logs.iter().all(|t| t.ledger.refused == 0);
        let cn_pairs: Vec<u64> = order.iter().map(|&t| sg::pair_counts(&fx, t)[2]).collect();
        let mut e = Some(e);
        for s in SETTINGS {
            let run = match s {
                Setting::Err => e.take().ok_or_else(|| bad("error-gated run missing"))?,
                _ => run_setting(&fx, &spec, seed, s, &order, &paced)?,
            };
            let init = ci::initial_circuit(&fx, &spec, s.base().arm())?;
            rows.push(Row {
                setting: s,
                k,
                flip,
                seed,
                order_ix: o,
                order: order.clone(),
                cn_pairs: cn_pairs.clone(),
                saturated_end: crate::error_gated::saturated(&run.weights),
                topology_unchanged: run.topology_unchanged,
                init_ok: ci::init_matches(&fx, &spec, s.base().arm(), init.weights())?,
                paced_identity: s != Setting::Err || identity,
                digest: wa::digest(&run.weights),
                ev: run.ev,
                logs: run.logs,
                glogs: run.glogs,
                cost: run.cost,
            });
        }
    }
    Ok(rows)
}

/// Full grid, (k, flip, seed) jobs spread over `threads` worker threads; the
/// output order (k, flip, seed, order, setting) and content do not depend on
/// the thread count (tested).
#[allow(clippy::too_many_arguments)]
pub fn run_grid_threads(
    graph: Option<&Connectome>,
    design: &Design,
    ks: &[usize],
    flips: &[u32],
    seeds: &[u64],
    n_heldout: usize,
    threads: usize,
) -> io::Result<Vec<Row>> {
    if seeds.is_empty() || flips.is_empty() || ks.is_empty() {
        return Err(bad("need at least one k, seed and flip"));
    }
    let jobs: Vec<(usize, u32, u64)> = ks
        .iter()
        .flat_map(|&k| {
            flips
                .iter()
                .flat_map(move |&f| seeds.iter().map(move |&s| (k, f, s)))
        })
        .collect();
    let threads = threads.clamp(1, jobs.len());
    let mut results: Vec<Option<io::Result<Vec<Row>>>> = (0..jobs.len()).map(|_| None).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|w| {
                let jobs = &jobs;
                scope.spawn(move || {
                    (w..jobs.len())
                        .step_by(threads)
                        .map(|ix| {
                            let (k, f, s) = jobs[ix];
                            (ix, run_seed(graph, design, k, f, s, n_heldout))
                        })
                        .collect::<Vec<_>>()
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
        HELDOUT_ORDERS,
        sg::default_threads(),
    )
}

/// Which task orders a statistic uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orders {
    Identity,
    Heldout,
}

impl Orders {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Heldout => "heldout",
        }
    }
    fn has(self, r: &Row) -> bool {
        r.heldout() == (self == Self::Heldout)
    }
}

/// Selection of rows: (run length, flip, orders).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sel {
    pub k: usize,
    pub flip: u32,
    pub orders: Orders,
}

fn rows_of(rows: &[Row], q: Sel, x: Setting) -> impl Iterator<Item = &Row> {
    rows.iter()
        .filter(move |r| r.k == q.k && r.flip == q.flip && r.setting == x && q.orders.has(r))
}

fn seeds_of(rows: &[Row], q: Sel) -> Vec<u64> {
    let mut s: Vec<u64> = rows
        .iter()
        .filter(|r| r.k == q.k && r.flip == q.flip)
        .map(|r| r.seed)
        .collect();
    s.sort_unstable();
    s.dedup();
    s
}

/// Per-seed mean of `f` over the rows of (q, seed, setting).
fn seed_value(rows: &[Row], q: Sel, seed: u64, x: Setting, f: &dyn Fn(&Row) -> f64) -> Option<f64> {
    let v: Vec<f64> = rows_of(rows, q, x)
        .filter(|r| r.seed == seed)
        .map(f)
        .collect();
    if v.is_empty() {
        None
    } else {
        Some(v.iter().sum::<f64>() / v.len() as f64)
    }
}

fn diff(rows: &[Row], q: Sel, x: Setting, base: Setting, f: &dyn Fn(&Row) -> f64) -> Vec<f64> {
    seeds_of(rows, q)
        .into_iter()
        .filter_map(|s| Some(seed_value(rows, q, s, x, f)? - seed_value(rows, q, s, base, f)?))
        .collect()
}

/// x vs base paired by seed, metrics at prefix length `kk` (≤ q.k); fields
/// oriented so that > 0 favours x (`forget` = FGT(base) − FGT(x)).
#[must_use]
pub fn compare(
    rows: &[Row],
    q: Sel,
    kk: usize,
    s: Slice,
    x: Setting,
    base: Setting,
) -> cr::Comparison {
    let d = |f: &dyn Fn(&Row) -> f64| paired(&diff(rows, q, x, base, f));
    cr::Comparison {
        acc: d(&|r| r.acc(s, kk)),
        learn: d(&|r| r.learn(s, kk)),
        last: d(&|r| r.last(s, kk)),
        forget: d(&|r| -r.forgetting(s, kk)),
    }
}

/// Growth: per seed [ACC_all(x) − ACC_all(base)] at q.k minus the same at
/// prefix `k_lo` of the same runs.
#[must_use]
pub fn growth(rows: &[Row], q: Sel, x: Setting, base: Setting, k_lo: usize) -> Paired {
    let a = Slice::All;
    let hi = diff(rows, q, x, base, &|r| r.acc(a, q.k));
    let lo = diff(rows, q, x, base, &|r| r.acc(a, k_lo));
    paired(&hi.iter().zip(&lo).map(|(h, l)| h - l).collect::<Vec<_>>())
}

/// Positive: mean > 0 and bootstrap CI lower bound > 0.
#[must_use]
pub fn positive(d: &Paired) -> bool {
    d.n > 0 && d.mean > 0.0 && d.ci_lo > 0.0
}

/// Pre-registered forgetting criterion: FGT cut mean ≥ `cut` with CI lower
/// bound > 0, and learning loss mean ≤ `LEARN_LOSS`.
#[must_use]
pub fn guard_criterion(c: &cr::Comparison, cut: f64) -> bool {
    c.forget.n > 0 && c.forget.mean >= cut && c.forget.ci_lo > 0.0 && c.learn.mean >= -LEARN_LOSS
}

#[must_use]
pub fn setting_mean(rows: &[Row], q: Sel, x: Setting, f: &dyn Fn(&Row) -> f64) -> f64 {
    mean(rows_of(rows, q, x).map(f))
}

/// Fraction of rows of x with `f(row) > 0` (`None` counts as 0).
#[must_use]
pub fn frac(rows: &[Row], q: Sel, x: Setting, f: &dyn Fn(&Row) -> Option<u64>) -> f64 {
    let v: Vec<&Row> = rows_of(rows, q, x).collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().filter(|r| f(r).unwrap_or(0) > 0).count() as f64 / v.len() as f64
    }
}

fn median(mut v: Vec<u64>) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

/// Median CN pairs per (seed, task) at (K_MAIN, flip), identity order.
#[must_use]
pub fn median_cn(rows: &[Row], flip: u32) -> u64 {
    let q = Sel {
        k: K_MAIN,
        flip,
        orders: Orders::Identity,
    };
    median(
        rows_of(rows, q, Setting::Err)
            .flat_map(|r| r.cn_pairs.iter().copied())
            .collect(),
    )
}

/// Orders valid and shared by all settings of a (k, flip, seed).
#[must_use]
pub fn orders_valid(rows: &[Row]) -> bool {
    rows.iter().all(|r| {
        sg::is_permutation(&r.order, r.k)
            && (r.order_ix == 0) == r.order.iter().enumerate().all(|(i, &t)| i == t)
            && rows
                .iter()
                .filter(|y| y.k == r.k && y.flip == r.flip && y.seed == r.seed)
                .all(|y| (y.order_ix == r.order_ix) == (y.order == r.order))
    })
}

/// Guard counters consistent with each setting's guard and its ledger.
#[must_use]
pub fn guards_consistent(rows: &[Row]) -> bool {
    rows.iter().all(|r| match r.setting.guard() {
        None => r.glogs.iter().all(|g| g.stats.is_none()),
        Some(gd) => r.glogs.iter().zip(&r.logs).all(|(g, t)| {
            g.stats.is_some_and(|s| {
                (matches!(gd, Guard::Block | Guard::Recent) || s.blocked == 0)
                    && (gd == Guard::LastResort || s.deferred == 0)
                    && (gd != Guard::Block || s.overwrites == 0)
                    && s.charged(gd) == t.ledger.writes
            })
        }),
    })
}

/// Design gates: (G1 integrity, G2 manipulation, G3 CN pairs).
#[must_use]
pub fn gates(rows: &[Row]) -> (bool, bool, bool) {
    let g1 = !rows.is_empty()
        && rows
            .iter()
            .all(|r| r.topology_unchanged && r.init_ok && r.paced_identity && r.within_cap())
        && orders_valid(rows)
        && guards_consistent(rows);
    let q = Sel {
        k: K_MAIN,
        flip: PRIMARY_FLIP,
        orders: Orders::Heldout,
    };
    let g2 = frac(rows, q, Setting::Err, &Row::overwrites) >= 0.5
        && frac(rows, q, Setting::Gb, &Row::blocked) >= 0.5
        && frac(rows, q, Setting::Gr, &Row::blocked) >= 0.5
        && frac(rows, q, Setting::Gc, &Row::overwrites) >= 0.5
        && frac(rows, q, Setting::Zg, &Row::deferred) >= 0.5;
    let g3 = median_cn(rows, PRIMARY_FLIP) >= ns::MIN_NOVEL_PAIRS;
    (g1, g2, g3)
}

/// Equivalence within ±EFFECT: 95 % CI inside (−0.05, +0.05).
#[must_use]
pub fn equivalent(d: &Paired) -> bool {
    d.n > 0 && d.ci_lo > -EFFECT && d.ci_hi < EFFECT
}

/// Number of pre-registered hypotheses.
pub const N_HYP: usize = 10;

/// Pre-registered readings: (gates, [H1..H10]); `None` if a gate fails or
/// the needed (k, flip) is missing. All hypotheses use held-out orders.
#[must_use]
pub fn readings(rows: &[Row]) -> ((bool, bool, bool), [Option<bool>; N_HYP]) {
    use Setting::{Err, Gb, Gc, Gl, Gr, Rew, Sup, Zg, Z};
    let g = gates(rows);
    let mut h = [None; N_HYP];
    if !(g.0 && g.1 && g.2) {
        return (g, h);
    }
    let has = |k: usize, f: u32| rows.iter().any(|r| r.k == k && r.flip == f);
    let all = Slice::All;
    let sel = |k, flip| Sel {
        k,
        flip,
        orders: Orders::Heldout,
    };
    if has(K_MAIN, PRIMARY_FLIP) {
        let q = sel(K_MAIN, PRIMARY_FLIP);
        let c = |x, b| compare(rows, q, K_MAIN, all, x, b);
        h[0] = Some(guard_criterion(&c(Gr, Err), FGT_CUT));
        h[1] = Some(guard_criterion(&c(Gb, Err), FGT_CUT));
        h[2] = Some(guard_criterion(&c(Gc, Err), FGT_CUT));
        h[3] = Some(positive(&c(Gr, Err).acc));
        h[4] = Some(c(Gr, Sup).passes());
        h[5] = Some(positive(&c(Gr, Rew).acc));
        let zg = c(Zg, Z);
        h[6] = Some(zg.forget.mean >= EFFECT && zg.forget.ci_lo > 0.0 && positive(&zg.acc));
        h[7] = Some(
            positive(&c(Gr, Err).forget)
                && positive(&c(Gb, Gr).forget)
                && positive(&c(Err, Gr).learn)
                && positive(&c(Gr, Gb).learn),
        );
        h[8] = Some(equivalent(&c(Gl, Err).acc));
    }
    if has(K_SMALL, PRIMARY_FLIP) {
        let q = sel(K_SMALL, PRIMARY_FLIP);
        let c = compare(rows, q, K_SMALL, all, Gr, Err);
        h[9] = Some(positive(&c.forget) && c.learn.mean >= -LEARN_LOSS);
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

/// CSV header (one row per (k, flip, seed, order, setting); summary
/// columns, per-stage values dash-joined in stage order).
#[must_use]
pub fn csv_header() -> String {
    [
        "schema",
        "graph",
        "k",
        "flip_pct",
        "setting",
        "seed",
        "order_ix",
        "order_kind",
        "order",
        "cn_pairs_s",
        "acc_all",
        "acc_cn",
        "learn_all",
        "learn_cn",
        "fgt_all",
        "fgt_cn",
        "last_all",
        "last_cn",
        "acc_all_k4",
        "learn_all_k4",
        "fgt_all_k4",
        "r_all_diag_s",
        "r_all_final_s",
        "r_cn_diag_s",
        "r_cn_final_s",
        "cap_total",
        "charged_writes",
        "effective_writes",
        "cue_writes",
        "refused",
        "overwrites",
        "blocked",
        "deferred",
        "overwrites_s",
        "net_reversals",
        "net_reversals_s",
        "moved_s",
        "consolidated_s",
        "saturated_end",
        "init_ok",
        "paced_identity",
        "within_cap",
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
    for r in rows {
        let k = r.k;
        let k4 = K_SMALL.min(k);
        let row_of = |s: Slice, last: bool| {
            (0..k)
                .map(|j| format!("{:.4}", r.r(s, if last { k - 1 } else { j }, j)))
                .collect::<Vec<_>>()
                .join("-")
        };
        let ovs: Vec<String> = r
            .glogs
            .iter()
            .map(|g| opt(g.stats.map(|s| s.overwrites)))
            .collect();
        let f: Vec<String> = vec![
            "b888-4-consolidation-guard-v1".to_string(),
            graph_label.to_string(),
            k.to_string(),
            r.flip.to_string(),
            r.setting.name().to_string(),
            r.seed.to_string(),
            r.order_ix.to_string(),
            if r.heldout() { "heldout" } else { "identity" }.to_string(),
            dash(&r.order),
            dash(&r.cn_pairs),
            f6(r.acc(a, k)),
            f6(r.acc(c, k)),
            f6(r.learn(a, k)),
            f6(r.learn(c, k)),
            f6(r.forgetting(a, k)),
            f6(r.forgetting(c, k)),
            f6(r.last(a, k)),
            f6(r.last(c, k)),
            f6(r.acc(a, k4)),
            f6(r.learn(a, k4)),
            f6(r.forgetting(a, k4)),
            row_of(a, false),
            row_of(a, true),
            row_of(c, false),
            row_of(c, true),
            opt(r.cap_total()),
            r.charged_writes().to_string(),
            r.effective_writes().to_string(),
            r.cue_writes().to_string(),
            r.refused().to_string(),
            opt(r.overwrites()),
            opt(r.blocked()),
            opt(r.deferred()),
            ovs.join("-"),
            r.net_reversals().to_string(),
            dash(&r.glogs.iter().map(|g| g.net_reversals).collect::<Vec<_>>()),
            dash(&r.glogs.iter().map(|g| g.moved).collect::<Vec<_>>()),
            dash(&r.glogs.iter().map(|g| g.consolidated).collect::<Vec<_>>()),
            r.saturated_end.to_string(),
            r.init_ok.to_string(),
            r.paced_identity.to_string(),
            r.within_cap().to_string(),
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
    use Setting::{Err, Gb, Gc, Gl, Gr, Rew, Sup, Zg, Z};
    let mut o = String::new();
    let mut ks: Vec<usize> = rows.iter().map(|r| r.k).collect();
    ks.sort_unstable();
    ks.dedup();
    let mut flips: Vec<u32> = rows.iter().map(|r| r.flip).collect();
    flips.sort_unstable();
    flips.dedup();
    let (a, c) = (Slice::All, Slice::Cn);
    let og = |v: Option<u64>| v.map_or(f64::NAN, |x| x as f64);
    let _ = writeln!(
        o,
        "setting_summary: k,flip,orders,setting,acc_all,acc_cn,learn_all,learn_cn,fgt_all,fgt_cn,last_all,charged_writes,effective_writes,cue_writes,refused,overwrites,blocked,deferred,net_reversals,consolidated_end,saturated_end"
    );
    for &k in &ks {
        for &flip in &flips {
            for ord in [Orders::Identity, Orders::Heldout] {
                let q = Sel {
                    k,
                    flip,
                    orders: ord,
                };
                for s in SETTINGS {
                    if rows_of(rows, q, s).next().is_none() {
                        continue;
                    }
                    let m = |f: &dyn Fn(&Row) -> f64| setting_mean(rows, q, s, f);
                    let _ = writeln!(
                        o,
                        "{k},{flip},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
                        ord.name(),
                        s.name(),
                        m(&|r| r.acc(a, k)),
                        m(&|r| r.acc(c, k)),
                        m(&|r| r.learn(a, k)),
                        m(&|r| r.learn(c, k)),
                        m(&|r| r.forgetting(a, k)),
                        m(&|r| r.forgetting(c, k)),
                        m(&|r| r.last(a, k)),
                        m(&|r| r.charged_writes() as f64),
                        m(&|r| r.effective_writes() as f64),
                        m(&|r| r.cue_writes() as f64),
                        m(&|r| r.refused() as f64),
                        m(&|r| og(r.overwrites())),
                        m(&|r| og(r.blocked())),
                        m(&|r| og(r.deferred())),
                        m(&|r| r.net_reversals() as f64),
                        m(&|r| r.consolidated_end() as f64),
                        m(&|r| r.saturated_end as f64),
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
        (Gr, Err),
        (Gb, Err),
        (Gc, Err),
        (Gl, Err),
        (Gr, Sup),
        (Gb, Sup),
        (Gc, Sup),
        (Err, Sup),
        (Z, Sup),
        (Zg, Sup),
        (Gr, Rew),
        (Rew, Sup),
        (Gr, Z),
        (Zg, Z),
        (Zg, Err),
        (Zg, Rew),
        (Gb, Gr),
        (Err, Gr),
        (Z, Err),
    ];
    for &k in &ks {
        for &flip in &flips {
            for ord in [Orders::Identity, Orders::Heldout] {
                let q = Sel {
                    k,
                    flip,
                    orders: ord,
                };
                for s in [a, c] {
                    for &(x, base) in &pairs {
                        let cmp = compare(rows, q, k, s, x, base);
                        if cmp.acc.n == 0 {
                            continue;
                        }
                        let _ = writeln!(
                            o,
                            "{k},{flip},{},{},{},{},{},{},{},{},{}",
                            ord.name(),
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
        }
    }
    let _ = writeln!(
        o,
        "growth(acc_all, k vs prefix {K_SMALL} of the same runs): k,flip,orders,x,base,d(k)-d({K_SMALL})"
    );
    for &k in ks.iter().filter(|&&k| k > K_SMALL) {
        for &flip in &flips {
            for ord in [Orders::Identity, Orders::Heldout] {
                let q = Sel {
                    k,
                    flip,
                    orders: ord,
                };
                for &(x, base) in &[(Gr, Err), (Gr, Sup), (Err, Sup), (Zg, Z), (Gb, Err)] {
                    let g = growth(rows, q, x, base, K_SMALL);
                    if g.n > 0 {
                        let _ = writeln!(
                            o,
                            "{k},{flip},{},{},{},{}",
                            ord.name(),
                            x.name(),
                            base.name(),
                            fmt_p(&g)
                        );
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
                "manipulation: k={k} flip={flip} overwrite_frac(E)={:.3} blocked_frac(Gb)={:.3} blocked_frac(Gr)={:.3} overwrite_frac(Gc)={:.3} deferred_frac(Zg)={:.3} deferred_frac(Gl)={:.3} mean_overwrites(E,Gr,Gc,Gl,Z,Zg)=({:.1},{:.1},{:.1},{:.1},{:.1},{:.1}) mean_blocked(Gb,Gr)=({:.1},{:.1}) mean_deferred(Gl,Zg)=({:.1},{:.1}) mean_effective_writes(E,Gb,Gr,Gc,Gl,Z,Zg)=({:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1})",
                frac(rows, q, Err, &Row::overwrites),
                frac(rows, q, Gb, &Row::blocked),
                frac(rows, q, Gr, &Row::blocked),
                frac(rows, q, Gc, &Row::overwrites),
                frac(rows, q, Zg, &Row::deferred),
                frac(rows, q, Gl, &Row::deferred),
                setting_mean(rows, q, Err, &|r| og(r.overwrites())),
                setting_mean(rows, q, Gr, &|r| og(r.overwrites())),
                setting_mean(rows, q, Gc, &|r| og(r.overwrites())),
                setting_mean(rows, q, Gl, &|r| og(r.overwrites())),
                setting_mean(rows, q, Z, &|r| og(r.overwrites())),
                setting_mean(rows, q, Zg, &|r| og(r.overwrites())),
                setting_mean(rows, q, Gb, &|r| og(r.blocked())),
                setting_mean(rows, q, Gr, &|r| og(r.blocked())),
                setting_mean(rows, q, Gl, &|r| og(r.deferred())),
                setting_mean(rows, q, Zg, &|r| og(r.deferred())),
                setting_mean(rows, q, Err, &|r| r.effective_writes() as f64),
                setting_mean(rows, q, Gb, &|r| r.effective_writes() as f64),
                setting_mean(rows, q, Gr, &|r| r.effective_writes() as f64),
                setting_mean(rows, q, Gc, &|r| r.effective_writes() as f64),
                setting_mean(rows, q, Gl, &|r| r.effective_writes() as f64),
                setting_mean(rows, q, Z, &|r| r.effective_writes() as f64),
                setting_mean(rows, q, Zg, &|r| r.effective_writes() as f64),
            );
        }
    }
    let _ = writeln!(
        o,
        "median_cn(k={K_MAIN},flip={PRIMARY_FLIP})={}",
        median_cn(rows, PRIMARY_FLIP)
    );
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

/// One pilot census line (identity order only; no accuracy computed):
/// (k, seed, setting, [charged writes, effective writes, cue writes,
/// refused, overwrites, blocked, deferred, net reversals, consolidated at
/// the end]).
pub type CensusLine = (usize, u64, Setting, [Option<u64>; 9]);

/// Pilot census on the identity order: writes and guard counters only. No
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
        let order: Vec<usize> = (0..k).collect();
        let train_only = |s: Setting, lim: &[Limit]| -> io::Result<Row> {
            let mut c = ci::initial_circuit(&fx, &spec, s.base().arm())?;
            let mut dir = vec![0_i8; fx.graph.edges()];
            let mut stamp = vec![0_u8; fx.graph.edges()];
            let (mut logs, mut glogs) = (Vec::new(), Vec::new());
            for (i, l) in lim.iter().enumerate() {
                let (inner, gl) = train_stage(
                    &mut c, &fx, &spec, seed, s, &order, i, l, &mut dir, &mut stamp,
                )?;
                glogs.push(gl);
                logs.push(TaskLog {
                    ledger: inner.inner.inner.ledger,
                    stats: inner.inner.inner.stats,
                    cue_cum: inner.inner.cue_cum,
                    ..TaskLog::default()
                });
            }
            Ok(Row {
                setting: s,
                k,
                flip,
                seed,
                order_ix: 0,
                order: order.clone(),
                ev: Vec::new(),
                logs,
                glogs,
                cost: Vec::new(),
                cn_pairs: Vec::new(),
                saturated_end: 0,
                topology_unchanged: true,
                init_ok: true,
                paced_identity: true,
                digest: 0,
            })
        };
        let e = train_only(Setting::Err, &sg::free(k))?;
        let curves: Vec<Vec<u64>> = e.logs.iter().map(|t| t.ledger.cum.clone()).collect();
        let paced = paced_limits(&curves);
        for s in SETTINGS {
            let r = if s == Setting::Err {
                e.clone()
            } else {
                train_only(s, &paced)?
            };
            out.push((
                k,
                seed,
                s,
                [
                    Some(r.charged_writes()),
                    Some(r.effective_writes()),
                    Some(r.cue_writes()),
                    Some(r.refused()),
                    r.overwrites(),
                    r.blocked(),
                    r.deferred(),
                    Some(r.net_reversals()),
                    Some(r.consolidated_end()),
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
        std::env::temp_dir().join(format!("v888-cons-guard-{tag}-{}", std::process::id()))
    }

    fn limits_for(fx: &Fixture, spec: &GenSpec, seed: u64, order: &[usize]) -> Vec<Limit> {
        let k = order.len();
        let e = run_setting(fx, spec, seed, Setting::Err, order, &sg::free(k)).unwrap();
        let curves: Vec<Vec<u64>> = e.logs.iter().map(|t| t.ledger.cum.clone()).collect();
        paced_limits(&curves)
    }

    #[test]
    fn guard_off_and_unguarded_settings_match_cycle18_bit_exactly() {
        let d = small();
        for (k, seed) in [(4, 1), (6, 2), (8, 3)] {
            let fx = sg::fixture(None, &d, k, 50, seed).unwrap();
            for order in orders(seed, k, 1).unwrap() {
                let paced = limits_for(&fx, &d.spec, seed, &order);
                for (s, lim) in [
                    (Setting::Err, sg::free(k)),
                    (Setting::Err, paced.clone()),
                    (Setting::Z, paced.clone()),
                    (Setting::Z, sg::free(k)),
                    (Setting::Sup, paced.clone()),
                    (Setting::Rew, paced.clone()),
                ] {
                    let old = sg::run_setting(&fx, &d.spec, seed, s.base(), &order, &lim).unwrap();
                    let new = run_setting(&fx, &d.spec, seed, s, &order, &lim).unwrap();
                    assert_eq!(new.weights, old.weights, "{}", s.name());
                    assert_eq!(new.logs, old.logs, "{}", s.name());
                    assert_eq!(new.cost, old.cost, "{}", s.name());
                    assert_eq!(new.ev, old.ev, "{}", s.name());
                    assert_eq!(new.topology_unchanged, old.topology_unchanged);
                    // Off counts overwrites but never blocks or defers.
                    for (g, t) in new.glogs.iter().zip(&new.logs) {
                        assert_eq!(g.stats.is_some(), s.guard().is_some());
                        if let Some(x) = g.stats {
                            assert_eq!(x.blocked + x.deferred, 0);
                            assert_eq!(x.effective, t.ledger.writes);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn guards_change_behaviour_and_keep_their_invariants() {
        let d = small();
        let k = 8;
        let mut changed = [false; 5];
        let mut acted = [false; 5];
        for seed in [1, 2, 3, 4] {
            let fx = sg::fixture(None, &d, k, 50, seed).unwrap();
            let order = orders(seed, k, 1).unwrap().remove(1);
            let paced = limits_for(&fx, &d.spec, seed, &order);
            let e = run_setting(&fx, &d.spec, seed, Setting::Err, &order, &paced).unwrap();
            let z = run_setting(&fx, &d.spec, seed, Setting::Z, &order, &paced).unwrap();
            for (ix, s) in [
                Setting::Gb,
                Setting::Gr,
                Setting::Gc,
                Setting::Gl,
                Setting::Zg,
            ]
            .into_iter()
            .enumerate()
            {
                let r = run_setting(&fx, &d.spec, seed, s, &order, &paced).unwrap();
                let base = if s.cued() { &z } else { &e };
                changed[ix] |= r.weights != base.weights;
                assert!(r.topology_unchanged);
                for (t, g) in r.logs.iter().zip(&r.glogs) {
                    let st = g.stats.unwrap();
                    assert!(t.ledger.writes <= t.ledger.cap);
                    let gd = s.guard().unwrap();
                    assert_eq!(st.charged(gd), t.ledger.writes);
                    match gd {
                        Guard::Block => {
                            assert_eq!(st.overwrites + st.deferred, 0);
                            acted[ix] |= st.blocked > 0;
                        }
                        Guard::Recent => {
                            assert_eq!(st.deferred, 0);
                            acted[ix] |= st.blocked > 0;
                        }
                        Guard::Charge => {
                            assert_eq!(st.blocked + st.deferred, 0);
                            acted[ix] |= st.overwrites > 0;
                        }
                        Guard::LastResort => {
                            assert_eq!(st.blocked, 0);
                            acted[ix] |= st.deferred > 0;
                        }
                        Guard::Off => unreachable!(),
                    }
                }
            }
        }
        assert!(changed.iter().all(|&c| c), "{changed:?}");
        assert!(acted.iter().all(|&c| c), "{acted:?}");
    }

    #[test]
    fn consolidation_tracks_net_signs() {
        let mut dir = vec![0_i8, 1, -1, 1, 0];
        let mut stamp = vec![0_u8, 1, 2, 1, 0];
        let (r, m) =
            consolidate(&mut dir, &mut stamp, 2, &[1, 1, 0, 2, 3], &[0, 2, 1, 1, 3]).unwrap();
        assert_eq!(dir, vec![-1, 1, 1, -1, 0]);
        assert_eq!(stamp, vec![3, 3, 3, 3, 0]);
        assert_eq!((r, m), (2, 4));
        assert!(consolidate(&mut dir, &mut stamp, 3, &[1], &[1, 2]).is_err());
        assert!(consolidate(&mut dir, &mut stamp, 255, &[1; 5], &[2; 5]).is_err());
    }

    #[test]
    fn resumes_bit_exactly_from_mid_stage_checkpoints() {
        let d = small();
        let seed = 4;
        let k = 6;
        let fx = sg::fixture(None, &d, k, 50, seed).unwrap();
        let g = &fx.graph;
        let order = orders(seed, k, 1).unwrap().remove(1);
        let paced = limits_for(&fx, &d.spec, seed, &order);
        for s in SETTINGS.into_iter().filter(|s| s.guard().is_some()) {
            let lim = if s == Setting::Err {
                sg::free(k)
            } else {
                paced.clone()
            };
            let full = run_setting(&fx, &d.spec, seed, s, &order, &lim).unwrap();
            let mut c = ci::initial_circuit(&fx, &d.spec, s.base().arm()).unwrap();
            let mut dir = vec![0_i8; g.edges()];
            let mut stamp = vec![0_u8; g.edges()];
            for (i, &t) in order.iter().enumerate() {
                let n = ts::task_order(&d.spec, seed, t).len();
                let start = c.weights().to_vec();
                let gd = s.guard().unwrap();
                let mut cp =
                    Checkpoint::start(&order, i, &lim[i], gd, &dir, &stamp, &start).unwrap();
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
                assert_eq!(Some(cp.g), full.glogs[i].stats, "{} {i}", s.name());
                assert_eq!(cp.inner.inner.inner.ledger, full.logs[i].ledger);
                assert_eq!(cp.inner.inner.cue_cum, full.logs[i].cue_cum);
                assert_eq!(cp.inner.inner.inner.cost, full.cost[i]);
                let (rev, moved) =
                    consolidate(&mut dir, &mut stamp, i, &cp.start, c.weights()).unwrap();
                assert_eq!(rev, full.glogs[i].net_reversals);
                assert_eq!(moved, full.glogs[i].moved);
            }
            assert_eq!(c.weights(), &full.weights[..], "{}", s.name());
        }
    }

    #[test]
    fn corrupt_checkpoints_are_rejected() {
        let d = small();
        let fx = sg::fixture(None, &d, 6, 50, 1).unwrap();
        let e = fx.graph.edges();
        let order = vec![2, 0, 1, 5, 4, 3];
        let mut dir = vec![0_i8; e];
        dir[0] = 1;
        dir[1] = -1;
        let mut stamp = vec![0_u8; e];
        stamp[0] = 3;
        stamp[1] = 1;
        let start = vec![1_u8; e];
        let lim = Limit::default();
        let cp =
            Checkpoint::start(&order, 3, &lim, Guard::LastResort, &dir, &stamp, &start).unwrap();
        let enc = cp.encode();
        assert_eq!(Checkpoint::decode(&enc, e).unwrap(), cp);
        assert!(Checkpoint::decode(&enc.replacen("v888", "x", 1), e).is_err());
        assert!(Checkpoint::decode(&enc, e + 1).is_err());
        assert!(Checkpoint::decode(&enc.replace("guard last-resort", "guard maybe"), e).is_err());
        assert!(Checkpoint::decode(&enc.replace("dir +-", "dir +x"), e).is_err());
        assert!(Checkpoint::decode(&enc.replace("dir +-", "dir +"), e).is_err());
        // Stamps: unknown value, from a stage not yet trained, or without a
        // consolidated sign.
        assert!(Checkpoint::decode(&enc.replace("stamp 3 1 0", "stamp x 1 0"), e).is_err());
        assert!(Checkpoint::decode(&enc.replace("stamp 3 1 0", "stamp 4 1 0"), e).is_err());
        assert!(Checkpoint::decode(&enc.replace("stamp 3 1 0", "stamp 3 0 0"), e).is_err());
        assert!(Checkpoint::decode(&enc.replace("stamp 3 1 0", "stamp 3 1 2"), e).is_err());
        assert!(Checkpoint::decode(&enc.replace("stamp 3 1 0", "stamp 3 1"), e).is_err());
        let rec = enc.replace("guard last-resort", "guard block-recent2");
        assert!(Checkpoint::decode(&rec.replace("gstats 0 0 0 0", "gstats 0 0 4 0"), e).is_ok());
        assert!(Checkpoint::decode(&enc.replacen("start 1", "start g", 1), e).is_err());
        assert!(Checkpoint::decode(&enc.replacen("start 1", "start ", 1), e).is_err());
        assert!(Checkpoint::decode(&enc.replace("gstats 0 0 0 0", "gstats 0 1 0 0"), e).is_err());
        assert!(Checkpoint::decode(&enc.replace("gstats 0 0 0 0", "gstats 0 0 0"), e).is_err());
        // Counters that do not match the ledger or the guard.
        assert!(Checkpoint::decode(&enc.replace("gstats 0 0 0 0", "gstats 3 0 0 0"), e).is_err());
        assert!(Checkpoint::decode(&enc.replace("gstats 0 0 0 0", "gstats 0 0 2 0"), e).is_err());
        let blk = enc.replace("guard last-resort", "guard block");
        assert!(Checkpoint::decode(&blk, e).is_ok());
        assert!(Checkpoint::decode(&blk.replace("gstats 0 0 0 0", "gstats 0 0 0 5"), e).is_err());
        // Inner (cycle 18) corruption is rejected too.
        assert!(Checkpoint::decode(&enc.replace("stage 3", "stage 6"), e).is_err());
        assert!(Checkpoint::decode(&enc.replace("task 5", "task 4"), e).is_err());
        let cut: String = enc.lines().take(6).collect::<Vec<_>>().join("\n");
        assert!(Checkpoint::decode(&cut, e).is_err());
        assert!(Checkpoint::start(&order, 3, &lim, Guard::Off, &dir, &stamp, &start[1..]).is_err());
        assert!(Checkpoint::start(&order, 3, &lim, Guard::Off, &dir, &stamp[1..], &start).is_err());
        // A checkpoint of another guard is refused by train_range.
        let mut c = PlasticCircuit::new(&fx.graph, 1).unwrap();
        let mut cp = cp;
        assert!(train_range(&mut c, &fx, &d.spec, 1, Setting::Gb, &lim, &mut cp, 1).is_err());
        // A checkpoint of another run length is refused too.
        let fx4 = sg::fixture(None, &d, 4, 50, 1).unwrap();
        let mut c4 = PlasticCircuit::new(&fx4.graph, 1).unwrap();
        assert!(train_range(&mut c4, &fx4, &d.spec, 1, Setting::Gl, &lim, &mut cp, 1).is_err());
    }

    #[test]
    fn grid_is_deterministic_thread_independent_and_readings_gate() {
        let d = small();
        let ks = [4, 6];
        let flips = [50, 100];
        let seeds = [1, 2, 3];
        let a = run_grid_threads(None, &d, &ks, &flips, &seeds, 2, 1).unwrap();
        let b = run_grid_threads(None, &d, &ks, &flips, &seeds, 2, 5).unwrap();
        assert_eq!(a, b);
        assert_eq!(
            a.len(),
            ks.len() * flips.len() * seeds.len() * 3 * SETTINGS.len()
        );
        assert!(a
            .iter()
            .all(|r| r.topology_unchanged && r.init_ok && r.within_cap() && r.paced_identity));
        assert!(orders_valid(&a) && guards_consistent(&a));
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        let cols = csv_header().split(',').count();
        assert!(csv.lines().all(|l| l.split(',').count() == cols));
        let q = Sel {
            k: 6,
            flip: 50,
            orders: Orders::Heldout,
        };
        let id = compare(&a, q, 6, Slice::All, Setting::Gl, Setting::Gl);
        assert_eq!(id.acc.ties, 3);
        let rep = report(&a);
        assert!(rep.contains("readings: G1_integrity=true"));
        assert_eq!(rep, report(&b));
        let (g, h) = readings(&a);
        assert!(g.0);
        if !(g.0 && g.1 && g.2) {
            assert_eq!(h, [None; N_HYP]);
        }
        assert!(run_grid_threads(None, &d, &ks, &[50], &[], 2, 1).is_err());
        let mut t = a.clone();
        t[0].order.swap(0, 1);
        assert!(!gates(&t).0);
        let mut t = a;
        let ix = t.iter().position(|r| r.setting == Setting::Gb).unwrap();
        t[ix].glogs[0].stats.as_mut().unwrap().deferred = 1;
        assert!(!guards_consistent(&t));
        assert!(!gates(&t).0);
    }

    #[test]
    fn heldout_orders_are_new_valid_and_distinct() {
        for seed in [1, 361, 400] {
            let o = orders(seed, 8, HELDOUT_ORDERS).unwrap();
            assert_eq!(o, orders(seed, 8, HELDOUT_ORDERS).unwrap());
            assert_eq!(o[0], (0..8).collect::<Vec<_>>());
            assert!(o.iter().all(|p| sg::is_permutation(p, 8)));
            assert!(o[1] != o[0] && o[2] != o[0] && o[1] != o[2]);
            // Different salt from cycle 18.
            assert_ne!(o, sg::orders(seed, 8, HELDOUT_ORDERS).unwrap());
        }
        assert!(orders(1, 2, 2).is_err());
    }

    #[test]
    fn names_parse_and_criteria() {
        for s in SETTINGS {
            assert_eq!(Setting::parse(s.name()).unwrap(), s);
        }
        assert!(Setting::parse("nope").is_err());
        for g in GUARDS {
            assert_eq!(Guard::parse(g.name()).unwrap(), g);
        }
        assert!(Guard::parse("never").is_err());
        let mk = |mean: f64, lo: f64| Paired {
            n: 10,
            mean,
            sd: 0.0,
            better: 10,
            ties: 0,
            worse: 0,
            ci_lo: lo,
            ci_hi: mean + 0.01,
        };
        let c = |f: Paired, l: Paired| cr::Comparison {
            acc: mk(0.0, 0.0),
            learn: l,
            last: mk(0.0, 0.0),
            forget: f,
        };
        assert!(guard_criterion(
            &c(mk(0.12, 0.05), mk(-0.08, -0.1)),
            FGT_CUT
        ));
        assert!(!guard_criterion(&c(mk(0.09, 0.05), mk(0.0, 0.0)), FGT_CUT));
        assert!(!guard_criterion(&c(mk(0.12, -0.01), mk(0.0, 0.0)), FGT_CUT));
        assert!(!guard_criterion(
            &c(mk(0.12, 0.05), mk(-0.11, -0.2)),
            FGT_CUT
        ));
        assert!(positive(&mk(0.01, 0.001)));
        assert!(!positive(&mk(0.01, -0.001)));
        let (g, h) = readings(&[]);
        assert!(!g.0);
        assert_eq!(h, [None; N_HYP]);
    }

    #[test]
    fn census_counts_writes_without_accuracy() {
        let d = small();
        let lines = census(None, &d, 6, 50, &[1, 2]).unwrap();
        assert_eq!(lines.len(), 2 * SETTINGS.len());
        for (_, _, s, v) in &lines {
            assert_eq!(v[4].is_some(), s.guard().is_some());
            if !s.cued() {
                assert_eq!(v[2], Some(0));
            }
        }
    }
}

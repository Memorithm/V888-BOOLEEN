//! B888-4 cycle 15: **write allocation** of the cued error-gated arm under
//! an equal weight-write budget, plus a secondary task-local negative-tag
//! rule.
//!
//! Cycle 14 (#126) gave every arm uncapped `error-gated`'s per-episode write
//! schedule (`paced100`). `E/E-cue` then kept only +0.011 ACC_all at flip 50
//! because its cue edges used the allowance first and it learned each task
//! much less (Δ LEARN −0.28). Inside an episode, #126 visited edges
//! source-major (cue and pattern lines mixed by index), so *which* writes a
//! capped arm made was an accident of line numbering. #126 listed other
//! allocation rules as untested.
//!
//! This module keeps the `E/E-cue` update rule and changes only the order in
//! which the offered edge steps of an episode are funded ([`Alloc`]):
//! * [`Alloc::SourceMajor`]: #126 order (ascending source index);
//! * [`Alloc::PatternFirst`]: every pattern-line edge, then cue-line edges;
//! * [`Alloc::CueFirst`]: every cue-line edge, then pattern-line edges.
//!
//! Gate decisions use the drive before the update and each step touches a
//! single edge, so with no cap the three orders are bit-identical (tested
//! and gated). Under a cap only the allocation differs.
//!
//! Secondary arm `E-tag` ([`Strategy::Tagged`]): uncued error gate with one
//! task-local bit per edge. Every active edge into a target labelled *should
//! not fire* is tagged negative (no write needed; under the OR rule every
//! such edge is negative), tags are cleared at each task boundary, and a
//! false negative raises only untagged edges. With tagging disabled it is
//! bit-identical to `error-gated` (tested). A writes-only pilot found it
//! nearly inactive on this design (see `docs/WRITE_ALLOCATION.md`).
//!
//! The supervised (S), error-gated (E) and per-target reward (R) arms are the
//! unchanged `write_budget::step` code paths of #126; `E/E-cue` source-major
//! is bit-identical to #126's `E/E-cue` (tested). Budgets reuse
//! `write_budget::{Budget, Limit, WriteLedger, limits}`.
//!
//! Engine state for checkpoint/resume = circuit weights
//! (`PlasticCircuit::save`) + [`Checkpoint`] (tags, stats, ledger, cost,
//! task, next episode); resuming is bit-exact (tested).
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph; "cue", "tag"
//! and "allocation" name parts of an update rule, not claims about
//! synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/WRITE_ALLOCATION.md`
//! (committed before any learning run on the hypothesis seeds).
use crate::connectome::Connectome;
use crate::context_cue as cc;
use crate::cue_reach::{self as cr, Family, Part};
use crate::error_gated::{drive, paired, Paired, PhaseCost, EFFECT, THRESHOLD};
use crate::generalization_bench::GenSpec;
use crate::novel_split::{self as ns, Design, Eval, Fixture, Slice, SLICES};
use crate::plasticity::{decrement_four, increment_four, PlasticCircuit};
use crate::task_sequence::{self as ts, K_MAX};
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
/// Pre-registered seeds: 201..=240, never used by an earlier B888-4 cycle.
pub const SEEDS: std::ops::RangeInclusive<u64> = 201..=240;
/// Pre-registered budgets. Primary: `Paced(100)`.
pub const BUDGETS: [Budget; 2] = [Budget::Free, Budget::Paced(100)];
pub const PRIMARY_BUDGET: Budget = Budget::Paced(100);
/// Cue level of the cued arms (full reach on the #124 pool).
pub const CUE_LEVEL: usize = wb::CUE_LEVEL;
/// Fixed-pool design of #124/#126.
pub const DESIGN: Design = wb::DESIGN;

/// Funding order of the offered edge steps inside one episode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Alloc {
    SourceMajor,
    PatternFirst,
    CueFirst,
}

/// Update rule of one arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Strategy {
    Supervised,
    ErrorGated,
    Reward,
    /// Error gate on pattern and cue edges, task cue on, given allocation.
    EeCue(Alloc),
    /// Uncued error gate + task-local negative tags (secondary).
    Tagged,
}

/// The pre-registered strategies, in CSV order.
pub const STRATEGIES: [Strategy; 7] = [
    Strategy::Supervised,
    Strategy::ErrorGated,
    Strategy::Reward,
    Strategy::EeCue(Alloc::SourceMajor),
    Strategy::EeCue(Alloc::PatternFirst),
    Strategy::EeCue(Alloc::CueFirst),
    Strategy::Tagged,
];

impl Strategy {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Supervised => "supervised",
            Self::ErrorGated => "error-gated",
            Self::Reward => "reward",
            Self::EeCue(Alloc::SourceMajor) => "E/E-cue",
            Self::EeCue(Alloc::PatternFirst) => "E/E-cue-pattern-first",
            Self::EeCue(Alloc::CueFirst) => "E/E-cue-cue-first",
            Self::Tagged => "E-tag",
        }
    }
    /// Parse a strategy name.
    pub fn parse(s: &str) -> io::Result<Self> {
        STRATEGIES
            .iter()
            .copied()
            .find(|x| x.name() == s)
            .ok_or_else(|| bad("unknown strategy"))
    }
    #[must_use]
    pub fn cued(self) -> bool {
        matches!(self, Self::EeCue(_))
    }
    fn base(self) -> Option<wb::Strategy> {
        match self {
            Self::Supervised => Some(wb::Strategy::Supervised),
            Self::ErrorGated => Some(wb::Strategy::ErrorGated),
            Self::Reward => Some(wb::Strategy::Reward),
            Self::EeCue(_) | Self::Tagged => None,
        }
    }
    fn arm(self) -> ts::Arm {
        match self {
            Self::Supervised | Self::Reward => cr::SUPERVISED,
            Self::ErrorGated | Self::Tagged => cr::ERROR_GATED,
            Self::EeCue(_) => cr::EE,
        }
    }
}

/// Task-local negative tags: one bit per original edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tags {
    bits: Vec<bool>,
    /// Tagging on? (`false` = plain error gate, used by the identity test.)
    pub enabled: bool,
}

impl Tags {
    #[must_use]
    pub fn new(edges: usize) -> Self {
        Self {
            bits: vec![false; edges],
            enabled: true,
        }
    }
    #[must_use]
    pub fn get(&self, edge: usize) -> bool {
        self.bits.get(edge).copied().unwrap_or(false)
    }
    #[must_use]
    pub fn count(&self) -> u64 {
        self.bits.iter().filter(|&&b| b).count() as u64
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.bits.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }
}

/// Per-task instrumentation beyond the write ledger.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Effective writes on cue-line edges.
    pub cue_writes: u64,
    /// Refused steps on cue-line edges.
    pub cue_refused: u64,
    /// E-tag: raises skipped because the edge was tagged (would have
    /// changed a weight).
    pub blocked: u64,
    /// E-tag: tagged edges at the end of the task.
    pub tagged_end: u64,
    /// E-tag: false negatives whose every active edge was tagged (cannot
    /// happen with consistent OR labels; gated).
    pub all_tagged: u64,
}

/// One episode of `strategy` with task cue `cue` (ignored by uncued arms).
/// S / E / R delegate to `write_budget::step` (bit-identical to #126).
#[allow(clippy::too_many_arguments)]
pub fn step(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    strategy: Strategy,
    pattern: &[usize],
    valence: &[bool],
    cue: &[usize],
    tags: &mut Tags,
    stats: &mut Stats,
    ledger: &mut WriteLedger,
    cost: &mut PhaseCost,
) -> io::Result<()> {
    if let Some(b) = strategy.base() {
        return wb::step(circuit, graph, b, pattern, valence, &[], ledger, cost);
    }
    if tags.len() != graph.edges() {
        return Err(bad("tag shape mismatch"));
    }
    let cue: &[usize] = if strategy.cued() { cue } else { &[] };
    let tagging = strategy == Strategy::Tagged && tags.enabled;
    let (input, reached, teacher) = cc::encode_ctx(graph, pattern, valence, cue)?;
    let n = graph.ids.len();
    let mut want = vec![false; n];
    for (t, w) in want.iter_mut().enumerate() {
        *w = teacher.get(t).map_err(|_| bad("bad target"))?;
    }
    let d = drive(circuit, graph, &input)?;
    let mut offer = vec![false; n];
    let mut is_reached = vec![false; n];
    for &t in &reached {
        is_reached[t] = true;
        offer[t] = crate::gates::ge(d[t], THRESHOLD) != want[t];
    }
    cost.episodes += 1;
    cost.targets_seen += reached.len() as u64;
    cost.targets_updated += reached.iter().filter(|&&t| offer[t]).count() as u64;
    let mut sources: Vec<usize> = input.active_indices().collect();
    let is_cue = |s: &usize| cue.contains(s);
    match strategy {
        Strategy::EeCue(Alloc::PatternFirst) => sources.sort_by_key(|s| (is_cue(s), *s)),
        Strategy::EeCue(Alloc::CueFirst) => sources.sort_by_key(|s| (!is_cue(s), *s)),
        _ => {}
    }
    if tagging {
        for &source in &sources {
            for edge in graph.offsets[source]..graph.offsets[source + 1] {
                let edge = edge as usize;
                let t = graph.targets[edge] as usize;
                if is_reached[t] && !want[t] {
                    tags.bits[edge] = true;
                }
            }
        }
    }
    let mut untagged_seen = vec![false; n];
    for &source in &sources {
        let on_cue = is_cue(&source);
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
            if up && tagging {
                if tags.get(edge) {
                    if next != w {
                        stats.blocked += 1;
                    }
                    continue;
                }
                untagged_seen[t] = true;
            }
            if ledger.writes >= ledger.allow && next != w {
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
                ledger.writes += 1;
                if on_cue {
                    stats.cue_writes += 1;
                }
            }
        }
    }
    if tagging {
        stats.all_tagged += reached
            .iter()
            .filter(|&&t| offer[t] && want[t] && !untagged_seen[t])
            .count() as u64;
    }
    ledger.cum.push(ledger.writes);
    Ok(())
}

/// Resumable training state of one task (everything but the weights).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub task: usize,
    /// Next episode index (0-based) within `task`.
    pub episode: usize,
    pub tags: Tags,
    pub stats: Stats,
    pub ledger: WriteLedger,
    pub cost: PhaseCost,
}

const MAGIC: &str = "v888-write-allocation-checkpoint-v1";

fn nums(v: &[u64]) -> String {
    if v.is_empty() {
        return "-".to_string();
    }
    v.iter().map(u64::to_string).collect::<Vec<_>>().join(" ")
}

impl Checkpoint {
    /// Fresh state at the start of `task` (tags cleared: task boundary).
    #[must_use]
    pub fn start(task: usize, edges: usize, limit: &Limit) -> Self {
        Self {
            task,
            episode: 0,
            tags: Tags::new(edges),
            stats: Stats::default(),
            ledger: WriteLedger::new(limit),
            cost: PhaseCost::default(),
        }
    }
    /// Deterministic text serialisation.
    #[must_use]
    pub fn encode(&self) -> String {
        let (l, c, s) = (&self.ledger, &self.cost, &self.stats);
        let bits: String = self
            .tags
            .bits
            .iter()
            .map(|&b| if b { '1' } else { '0' })
            .collect();
        format!(
            "{MAGIC}\ntask {}\nepisode {}\nenabled {}\ntags {}\nstats {}\nledger {}\ncum {}\ncost {}\n",
            self.task,
            self.episode,
            u8::from(self.tags.enabled),
            if bits.is_empty() { "-".to_string() } else { bits },
            nums(&[s.cue_writes, s.cue_refused, s.blocked, s.tagged_end, s.all_tagged]),
            nums(&[l.cap, l.writes, l.refused, l.first_refusal, l.allow]),
            nums(&l.cum),
            nums(&[c.episodes, c.targets_seen, c.targets_updated, c.attempts, c.weights_changed]),
        )
    }
    /// Parse [`Checkpoint::encode`]; `edges` must match the graph.
    pub fn decode(s: &str, edges: usize) -> io::Result<Self> {
        let mut lines = s.lines();
        if lines.next() != Some(MAGIC) {
            return Err(bad("not a write-allocation checkpoint"));
        }
        let mut field = |name: &str| -> io::Result<String> {
            let line = lines.next().ok_or_else(|| bad("truncated checkpoint"))?;
            line.strip_prefix(name)
                .and_then(|r| r.strip_prefix(' '))
                .map(str::to_string)
                .ok_or_else(|| bad("bad checkpoint field"))
        };
        let list = |s: &str| -> io::Result<Vec<u64>> {
            if s == "-" {
                return Ok(Vec::new());
            }
            s.split(' ')
                .map(|x| x.parse::<u64>().map_err(|_| bad("bad number")))
                .collect()
        };
        let one = |s: String| s.parse::<usize>().map_err(|_| bad("bad number"));
        let task = one(field("task")?)?;
        let episode = one(field("episode")?)?;
        let enabled = match field("enabled")?.as_str() {
            "0" => false,
            "1" => true,
            _ => return Err(bad("bad enabled flag")),
        };
        let t = field("tags")?;
        let bits: Vec<bool> = if t == "-" {
            Vec::new()
        } else {
            t.chars()
                .map(|c| match c {
                    '0' => Ok(false),
                    '1' => Ok(true),
                    _ => Err(bad("bad tag bit")),
                })
                .collect::<io::Result<_>>()?
        };
        if bits.len() != edges {
            return Err(bad("checkpoint edge count mismatch"));
        }
        let st = list(&field("stats")?)?;
        let le = list(&field("ledger")?)?;
        let cum = list(&field("cum")?)?;
        let co = list(&field("cost")?)?;
        if st.len() != 5 || le.len() != 5 || co.len() != 5 || task >= K_MAX {
            return Err(bad("bad checkpoint arity"));
        }
        if cum.len() != episode {
            return Err(bad("checkpoint episode/curve mismatch"));
        }
        Ok(Self {
            task,
            episode,
            tags: Tags { bits, enabled },
            stats: Stats {
                cue_writes: st[0],
                cue_refused: st[1],
                blocked: st[2],
                tagged_end: st[3],
                all_tagged: st[4],
            },
            ledger: WriteLedger {
                cap: le[0],
                writes: le[1],
                refused: le[2],
                first_refusal: le[3],
                cum,
                allow: le[4],
            },
            cost: PhaseCost {
                episodes: co[0],
                targets_seen: co[1],
                targets_updated: co[2],
                attempts: co[3],
                weights_changed: co[4],
            },
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

/// Train episodes `cp.episode..end` of task `cp.task` from state `cp`.
#[allow(clippy::too_many_arguments)]
pub fn train_range(
    circuit: &mut PlasticCircuit,
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    strategy: Strategy,
    limit: &Limit,
    cp: &mut Checkpoint,
    end: usize,
) -> io::Result<()> {
    let i = cp.task;
    let order = ts::task_order(spec, seed, i);
    if cp.episode > end || end > order.len() {
        return Err(bad("bad episode range"));
    }
    let cue = fx.cues.get(i).ok_or_else(|| bad("missing cue"))?;
    for (e, &ix) in order.iter().enumerate().take(end).skip(cp.episode) {
        cp.ledger.allow = limit_at(limit, e);
        step(
            circuit,
            &fx.graph,
            strategy,
            &fx.task.train[i][ix],
            &fx.task.valence[i],
            cue,
            &mut cp.tags,
            &mut cp.stats,
            &mut cp.ledger,
            &mut cp.cost,
        )?;
    }
    cp.episode = end;
    if end == order.len() {
        cp.ledger.allow = cp.ledger.cap;
        cp.stats.tagged_end = cp.tags.count();
    }
    Ok(())
}

/// FNV-1a digest of a weight vector.
#[must_use]
pub fn digest(w: &[u8]) -> u64 {
    w.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// One measurement: (flip, seed, strategy, budget) on a K_MAX-task run.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub strategy: Strategy,
    pub budget: Budget,
    pub inner: cr::Row,
    pub ledger: [WriteLedger; K_MAX],
    pub stats: [Stats; K_MAX],
    /// Final weights equal to the uncapped run of the same strategy.
    pub same_as_free: bool,
    /// Uncapped final weights equal to uncapped source-major `E/E-cue`
    /// (always true for uncued arms).
    pub free_alloc_invariant: bool,
    /// Digest of the final weights.
    pub digest: u64,
}

/// Output of [`run_arm`]: row, per-task ledgers and stats, final weights.
pub type ArmRun = (cr::Row, [WriteLedger; K_MAX], [Stats; K_MAX], Vec<u8>);

/// Run one strategy over the K_MAX tasks with per-task write limits.
pub fn run_arm(
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    strategy: Strategy,
    limits: &[Limit; K_MAX],
) -> io::Result<ArmRun> {
    run_arm_with(fx, spec, seed, strategy, limits, true)
}

fn run_arm_with(
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    strategy: Strategy,
    limits: &[Limit; K_MAX],
    tagging: bool,
) -> io::Result<ArmRun> {
    let graph = &fx.graph;
    if fx.task.train.len() != K_MAX || fx.cues.len() != K_MAX || fx.splits.len() != K_MAX {
        return Err(bad("need K_MAX tasks, cues and splits"));
    }
    graph.validate()?;
    let original = graph.clone();
    let n = graph.ids.len();
    let mut circuit = PlasticCircuit::new(graph, spec.init_weight)?;
    let fingerprint = circuit.topology_fingerprint();
    let mut ev = [[Eval::default(); K_MAX]; K_MAX];
    let mut tgt = vec![crate::generalization_bench::Confusion::default(); K_MAX * K_MAX * n];
    let mut cost = [PhaseCost::default(); K_MAX];
    let mut ledger: [WriteLedger; K_MAX] = Default::default();
    let mut stats = [Stats::default(); K_MAX];
    for i in 0..K_MAX {
        let episodes = ts::task_order(spec, seed, i).len();
        let mut cp = Checkpoint::start(i, graph.edges(), &limits[i]);
        cp.tags.enabled = tagging;
        train_range(
            &mut circuit,
            fx,
            spec,
            seed,
            strategy,
            &limits[i],
            &mut cp,
            episodes,
        )?;
        ledger[i] = cp.ledger;
        stats[i] = cp.stats;
        cost[i] = cp.cost;
        for j in 0..K_MAX {
            let cue: &[usize] = if strategy.cued() { &fx.cues[j] } else { &[] };
            let (e, per) = cr::evaluate(
                &circuit,
                graph,
                &fx.task.test,
                &fx.task.valence[j],
                cue,
                &fx.splits[j],
            )?;
            ev[i][j] = e;
            tgt[(i * K_MAX + j) * n..(i * K_MAX + j + 1) * n].copy_from_slice(&per);
        }
    }
    let mut cue_lines = [0; K_MAX];
    if strategy.cued() {
        for (l, c) in cue_lines.iter_mut().zip(&fx.cues) {
            *l = c.len();
        }
    }
    let row = cr::Row {
        family: Family::Fixed,
        level: if strategy.cued() { CUE_LEVEL } else { 0 },
        arm: strategy.arm(),
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
    Ok((row, ledger, stats, circuit.weights().to_vec()))
}

/// Full grid: per (flip, seed) uncapped `error-gated` first (its write
/// curves set the caps), then every (strategy, budget).
pub fn run_grid_design(
    graph: Option<&Connectome>,
    design: &Design,
    flips: &[u32],
    seeds: &[u64],
    budgets: &[Budget],
    strategies: &[Strategy],
) -> io::Result<Vec<Row>> {
    if seeds.is_empty() || flips.is_empty() || budgets.is_empty() || strategies.is_empty() {
        return Err(bad("need at least one seed, flip, budget and strategy"));
    }
    if budgets.iter().any(|b| match b {
        Budget::Free => false,
        Budget::Paced(p) | Budget::Total(p) => *p == 0 || *p > 100,
    }) {
        return Err(bad("budget must be in 1..=100 % or uncapped"));
    }
    let mut rows = Vec::new();
    for &flip in flips {
        let spec = GenSpec {
            flip_pct: flip,
            ..design.spec
        };
        for &seed in seeds {
            let fx = ns::fixture(graph, design, flip, seed)?;
            let (_, eg, _, _) = run_arm(&fx, &spec, seed, Strategy::ErrorGated, &wb::free())?;
            let curves: Vec<Vec<u64>> = eg.iter().map(|l| l.cum.clone()).collect();
            let (_, _, _, ee_free) = run_arm(
                &fx,
                &spec,
                seed,
                Strategy::EeCue(Alloc::SourceMajor),
                &wb::free(),
            )?;
            for &s in strategies {
                let (_, _, _, free_w) = run_arm(&fx, &spec, seed, s, &wb::free())?;
                let invariant = !s.cued() || free_w == ee_free;
                for &b in budgets {
                    let (inner, ledger, stats, w) =
                        run_arm(&fx, &spec, seed, s, &wb::limits(&curves, b))?;
                    rows.push(Row {
                        strategy: s,
                        budget: b,
                        inner,
                        ledger,
                        stats,
                        same_as_free: w == free_w,
                        free_alloc_invariant: invariant,
                        digest: digest(&w),
                    });
                }
            }
        }
    }
    Ok(rows)
}

/// Pre-registered grid on the #124/#126 fixed-pool design.
pub fn run_grid(
    graph: Option<&Connectome>,
    flips: &[u32],
    seeds: &[u64],
    budgets: &[Budget],
) -> io::Result<Vec<Row>> {
    run_grid_design(graph, &DESIGN, flips, seeds, budgets, &STRATEGIES)
}

/// A (strategy, budget) setting.
pub type Setting = (Strategy, Budget);
/// The base of every gain: uncapped `error-gated`.
pub const EG: Setting = (Strategy::ErrorGated, Budget::Free);
/// Primary arm: pattern-first `E/E-cue` at `paced100`.
pub const PF: Setting = (Strategy::EeCue(Alloc::PatternFirst), PRIMARY_BUDGET);
/// #126 source-major `E/E-cue` at `paced100`.
pub const SM: Setting = (Strategy::EeCue(Alloc::SourceMajor), PRIMARY_BUDGET);
pub const CF: Setting = (Strategy::EeCue(Alloc::CueFirst), PRIMARY_BUDGET);
pub const S1: Setting = (Strategy::Supervised, PRIMARY_BUDGET);
pub const R1: Setting = (Strategy::Reward, PRIMARY_BUDGET);
pub const TAG: Setting = (Strategy::Tagged, PRIMARY_BUDGET);
pub const EE_FREE: Setting = (Strategy::EeCue(Alloc::SourceMajor), Budget::Free);

fn find(rows: &[Row], flip: u32, seed: u64, x: Setting) -> Option<&Row> {
    rows.iter().find(|r| {
        r.inner.flip == flip && r.inner.seed == seed && r.strategy == x.0 && r.budget == x.1
    })
}

fn rows_of(rows: &[Row], flip: u32, x: Setting) -> impl Iterator<Item = &Row> {
    rows.iter()
        .filter(move |r| r.inner.flip == flip && r.strategy == x.0 && r.budget == x.1)
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

fn total_writes(r: &Row) -> u64 {
    r.ledger.iter().map(|l| l.writes).sum()
}

fn cue_share(r: &Row) -> f64 {
    let w = total_writes(r);
    if w == 0 {
        0.0
    } else {
        r.stats.iter().map(|s| s.cue_writes).sum::<u64>() as f64 / w as f64
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
        rows_of(rows, flip, EG)
            .flat_map(|r| (0..K_MAX).map(move |j| r.inner.pairs(Part::Class(Slice::Cn), j)))
            .collect(),
    )
}

/// Design gates: (G1 integrity, G2 allocation changes the run, G3 CN).
/// G1: every row keeps its topology; writes ≤ allowance; `error-gated` at
/// 100 % budgets is bit-identical to its uncapped run with no refusal; the
/// three `E/E-cue` allocations are bit-identical uncapped; no E-tag false
/// negative ever had every edge tagged.
#[must_use]
pub fn gates(rows: &[Row]) -> (bool, bool, bool) {
    let g1 = !rows.is_empty()
        && rows.iter().all(|r| {
            r.inner.topology_unchanged
                && r.free_alloc_invariant
                && r.ledger.iter().all(|l| l.writes <= l.cap)
                && r.stats.iter().all(|t| t.all_tagged == 0)
                && (r.strategy != Strategy::ErrorGated
                    || matches!(r.budget, Budget::Paced(p) | Budget::Total(p) if p < 100)
                    || (r.same_as_free && r.ledger.iter().all(|l| l.refused == 0)))
        });
    let g2 = differ_frac(rows, PRIMARY_FLIP, PF, SM) >= 0.5;
    let g3 = median_cn(rows, PRIMARY_FLIP) >= ns::MIN_NOVEL_PAIRS;
    (g1, g2, g3)
}

/// Number of pre-registered hypotheses.
pub const N_HYP: usize = 7;

/// Pre-registered readings: (gates, [H1..H7]); `None` if a gate fails or
/// the hypothesis' flip is missing.
#[must_use]
pub fn readings(rows: &[Row], flips: &[u32]) -> ((bool, bool, bool), [Option<bool>; N_HYP]) {
    let g = gates(rows);
    let mut h = [None; N_HYP];
    if !(g.0 && g.1 && g.2) {
        return (g, h);
    }
    let (all, cn) = (Slice::All, Slice::Cn);
    let f = PRIMARY_FLIP;
    if flips.contains(&f) {
        h[0] = Some(compare(rows, f, K_MAX, all, PF, EG).passes());
        h[1] = Some(compare(rows, f, K_MAX, all, PF, SM).learn.strong());
        h[2] = Some(compare(rows, f, K_MAX, cn, PF, EG).passes());
        h[4] = Some(compare(rows, f, K_MAX, cn, PF, S1).passes());
        let r = compare(rows, f, K_MAX, all, R1, S1);
        h[6] = Some(r.acc.n > 0 && r.acc.ci_lo > 0.0);
    }
    if flips.contains(&HIGH_FLIP) {
        let c = compare(rows, HIGH_FLIP, K_MAX, cn, PF, EG);
        h[3] = Some(c.passes());
        h[5] = Some(c.forget.strong() && c.learn.mean >= -EFFECT);
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
        "strategy",
        "budget",
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
        "writes",
        "refused",
        "first_refusal",
        "attempts",
        "cue_writes",
        "cue_refused",
        "blocked",
        "tagged_end",
    ] {
        for j in 1..=K_MAX {
            h.push(format!("{name}_{j}"));
        }
    }
    for name in [
        "saturated_end",
        "same_as_free",
        "free_alloc_invariant",
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
        let mut f: Vec<String> = vec![
            "b888-4-write-allocation-v1".to_string(),
            graph_label.to_string(),
            x.flip.to_string(),
            r.strategy.name().to_string(),
            r.budget.name(),
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
        f.extend(r.ledger.iter().map(|l| {
            if l.cap == u64::MAX {
                "-".to_string()
            } else {
                l.cap.to_string()
            }
        }));
        f.extend(r.ledger.iter().map(|l| l.writes.to_string()));
        f.extend(r.ledger.iter().map(|l| l.refused.to_string()));
        f.extend(r.ledger.iter().map(|l| l.first_refusal.to_string()));
        f.extend(x.cost.iter().map(|c| c.attempts.to_string()));
        f.extend(r.stats.iter().map(|t| t.cue_writes.to_string()));
        f.extend(r.stats.iter().map(|t| t.cue_refused.to_string()));
        f.extend(r.stats.iter().map(|t| t.blocked.to_string()));
        f.extend(r.stats.iter().map(|t| t.tagged_end.to_string()));
        f.push(x.saturated_end.to_string());
        f.push(r.same_as_free.to_string());
        f.push(r.free_alloc_invariant.to_string());
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
pub fn report(rows: &[Row], flips: &[u32], budgets: &[Budget]) -> String {
    use std::fmt::Write;
    let mut o = String::new();
    let (a, c, ic) = (
        Part::Class(Slice::All),
        Part::Class(Slice::Cn),
        Part::Class(Slice::Icn),
    );
    let _ = writeln!(
        o,
        "setting_summary(k=4): flip,strategy,budget,acc_all,acc_cn,acc_icn,acc_seen,acc_ue,learn_all,learn_cn,fgt_all,fgt_cn,last_all,last_cn,writes_total,cue_write_share,refused_total,cue_refused_total,blocked_total,tagged_end_mean,saturated_end"
    );
    for &flip in flips {
        for s in STRATEGIES {
            for &b in budgets {
                let x = (s, b);
                if rows_of(rows, flip, x).next().is_none() {
                    continue;
                }
                let m = |f: &dyn Fn(&Row) -> f64| setting_mean(rows, flip, x, f);
                let _ = writeln!(
                    o,
                    "{flip},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{:.3},{:.1},{:.1},{:.1},{:.1},{:.1}",
                    s.name(),
                    b.name(),
                    m(&|r| r.inner.acc(a, K_MAX)),
                    m(&|r| r.inner.acc(c, K_MAX)),
                    m(&|r| r.inner.acc(ic, K_MAX)),
                    m(&|r| r.inner.acc(Part::Class(Slice::Seen), K_MAX)),
                    m(&|r| r.inner.acc(Part::Class(Slice::Ue), K_MAX)),
                    m(&|r| r.inner.learn(a, K_MAX)),
                    m(&|r| r.inner.learn(c, K_MAX)),
                    m(&|r| r.inner.forgetting(a, K_MAX)),
                    m(&|r| r.inner.forgetting(c, K_MAX)),
                    m(&|r| r.inner.last(a, K_MAX)),
                    m(&|r| r.inner.last(c, K_MAX)),
                    m(&|r| total_writes(r) as f64),
                    m(&cue_share),
                    m(&|r| r.ledger.iter().map(|l| l.refused).sum::<u64>() as f64),
                    m(&|r| r.stats.iter().map(|t| t.cue_refused).sum::<u64>() as f64),
                    m(&|r| r.stats.iter().map(|t| t.blocked).sum::<u64>() as f64),
                    m(&|r| r.stats.iter().map(|t| t.tagged_end).sum::<u64>() as f64 / K_MAX as f64),
                    m(&|r| r.inner.saturated_end as f64),
                );
            }
        }
    }
    let _ = writeln!(
        o,
        "comparison(k): flip,k,slice,x,base,d_acc,d_learn,d_last,d_fgt(+=forgets_less),passes_sequence_rule"
    );
    let base_pairs = [
        (PF, EG),
        (SM, EG),
        (CF, EG),
        (PF, SM),
        (CF, SM),
        (PF, S1),
        (EG, S1),
        (R1, S1),
        (TAG, EG),
        (EE_FREE, EG),
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
                        "{flip},{k},{},{}@{},{}@{},{},{},{},{},{}",
                        s.name(),
                        x.0.name(),
                        x.1.name(),
                        base.0.name(),
                        base.1.name(),
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
            "allocation: flip={flip} differ_frac(pattern-first vs source-major @paced100)={:.3} differ_frac(cue-first vs source-major @paced100)={:.3} median_cn={}",
            differ_frac(rows, flip, PF, SM),
            differ_frac(rows, flip, CF, SM),
            median_cn(rows, flip)
        );
    }
    let (g, h) = readings(rows, flips);
    let _ = writeln!(
        o,
        "readings: G1_integrity={} G2_allocation_changes_run={} G3_cn_pairs={} H1={} H2={} H3={} H4={} H5={} H6={} H7={}",
        g.0,
        g.1,
        g.2,
        yes(h[0]),
        yes(h[1]),
        yes(h[2]),
        yes(h[3]),
        yes(h[4]),
        yes(h[5]),
        yes(h[6])
    );
    o
}

/// One pilot census line: (seed, strategy, budget, writes, cue writes,
/// refused, cue refused, blocked).
pub type CensusLine = (u64, Strategy, Budget, u64, u64, u64, u64, u64);

/// Pilot census: writes, cue writes, refusals and blocked raises only; no
/// accuracy is computed.
pub fn census(
    graph: Option<&Connectome>,
    flip: u32,
    seeds: &[u64],
    budgets: &[Budget],
) -> io::Result<Vec<CensusLine>> {
    let spec = GenSpec {
        flip_pct: flip,
        ..DESIGN.spec
    };
    let mut out = Vec::new();
    for &seed in seeds {
        let fx = ns::fixture(graph, &DESIGN, flip, seed)?;
        let mut curves: Vec<Vec<u64>> = Vec::new();
        {
            let mut c = PlasticCircuit::new(&fx.graph, spec.init_weight)?;
            for i in 0..K_MAX {
                let n = ts::task_order(&spec, seed, i).len();
                let mut cp = Checkpoint::start(i, fx.graph.edges(), &Limit::default());
                let free = Limit::default();
                train_range(
                    &mut c,
                    &fx,
                    &spec,
                    seed,
                    Strategy::ErrorGated,
                    &free,
                    &mut cp,
                    n,
                )?;
                curves.push(cp.ledger.cum);
            }
        }
        for s in STRATEGIES {
            for &b in budgets {
                let lim = wb::limits(&curves, b);
                let mut c = PlasticCircuit::new(&fx.graph, spec.init_weight)?;
                let mut t = [0_u64; 5];
                for (i, l) in lim.iter().enumerate() {
                    let n = ts::task_order(&spec, seed, i).len();
                    let mut cp = Checkpoint::start(i, fx.graph.edges(), l);
                    train_range(&mut c, &fx, &spec, seed, s, l, &mut cp, n)?;
                    t[0] += cp.ledger.writes;
                    t[1] += cp.stats.cue_writes;
                    t[2] += cp.ledger.refused;
                    t[3] += cp.stats.cue_refused;
                    t[4] += cp.stats.blocked;
                }
                out.push((seed, s, b, t[0], t[1], t[2], t[3], t[4]));
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
        std::env::temp_dir().join(format!(
            "v888-write-allocation-{tag}-{}",
            std::process::id()
        ))
    }

    fn curves_of(fx: &Fixture, d: &Design, seed: u64) -> Vec<Vec<u64>> {
        let (_, eg, _, _) = run_arm(fx, &d.spec, seed, Strategy::ErrorGated, &wb::free()).unwrap();
        eg.iter().map(|l| l.cum.clone()).collect()
    }

    #[test]
    fn base_arms_and_source_major_match_write_budget_bit_exactly() {
        let d = small();
        for seed in [1, 2, 3] {
            let fx = ns::fixture(None, &d, 50, seed).unwrap();
            let curves = curves_of(&fx, &d, seed);
            for b in [Budget::Free, Budget::Paced(100), Budget::Paced(50)] {
                let lim = wb::limits(&curves, b);
                for (s, old) in [
                    (Strategy::Supervised, wb::Strategy::Supervised),
                    (Strategy::ErrorGated, wb::Strategy::ErrorGated),
                    (Strategy::Reward, wb::Strategy::Reward),
                    (Strategy::EeCue(Alloc::SourceMajor), wb::Strategy::EeCue),
                ] {
                    let (r, l, _, w) = run_arm(&fx, &d.spec, seed, s, &lim).unwrap();
                    let (r0, l0, w0) = wb::run_arm(&fx, &d.spec, seed, old, &lim).unwrap();
                    assert_eq!(w, w0, "{} {}", s.name(), b.name());
                    assert_eq!(r.ev, r0.ev, "{} {}", s.name(), b.name());
                    assert_eq!(r.cost, r0.cost, "{} {}", s.name(), b.name());
                    assert_eq!(l, l0, "{} {}", s.name(), b.name());
                }
            }
        }
    }

    #[test]
    fn allocations_are_identical_uncapped_and_differ_capped() {
        let d = small();
        let mut differ = 0;
        for seed in 1..=4 {
            let fx = ns::fixture(None, &d, 50, seed).unwrap();
            let curves = curves_of(&fx, &d, seed);
            let free: Vec<_> = [Alloc::SourceMajor, Alloc::PatternFirst, Alloc::CueFirst]
                .iter()
                .map(|&a| run_arm(&fx, &d.spec, seed, Strategy::EeCue(a), &wb::free()).unwrap())
                .collect();
            assert_eq!(free[0].3, free[1].3);
            assert_eq!(free[0].3, free[2].3);
            assert_eq!(free[0].0.ev, free[1].0.ev);
            assert_eq!(free[0].1, free[2].1);
            let lim = wb::limits(&curves, Budget::Paced(100));
            let sm = run_arm(
                &fx,
                &d.spec,
                seed,
                Strategy::EeCue(Alloc::SourceMajor),
                &lim,
            )
            .unwrap();
            let pf = run_arm(
                &fx,
                &d.spec,
                seed,
                Strategy::EeCue(Alloc::PatternFirst),
                &lim,
            )
            .unwrap();
            let cf = run_arm(&fx, &d.spec, seed, Strategy::EeCue(Alloc::CueFirst), &lim).unwrap();
            for x in [&sm, &pf, &cf] {
                for j in 0..K_MAX {
                    assert!(x.1[j].writes <= x.1[j].cap);
                    assert!(x.2[j].cue_writes <= x.1[j].writes);
                }
            }
            // Funding pattern edges first never spends more on cue edges in
            // the first task than funding cue edges first.
            assert!(pf.2[0].cue_writes <= cf.2[0].cue_writes);
            if pf.3 != sm.3 {
                differ += 1;
            }
        }
        assert!(differ > 0);
    }

    #[test]
    fn untagged_rule_is_bit_identical_to_error_gated() {
        let d = small();
        for seed in [1, 2, 3] {
            let fx = ns::fixture(None, &d, 50, seed).unwrap();
            let (re, le, _, we) =
                run_arm(&fx, &d.spec, seed, Strategy::ErrorGated, &wb::free()).unwrap();
            let (rx, lx, tx, wx) =
                run_arm_with(&fx, &d.spec, seed, Strategy::Tagged, &wb::free(), false).unwrap();
            assert_eq!(we, wx);
            assert_eq!(re.ev, rx.ev);
            assert_eq!(re.cost, rx.cost);
            assert_eq!(le, lx);
            assert!(tx.iter().all(|t| *t == Stats::default()));
        }
    }

    #[test]
    fn tags_only_mark_negative_edges_and_block_raises() {
        let d = small();
        let mut blocked = 0;
        for seed in 1..=6 {
            let fx = ns::fixture(None, &d, 50, seed).unwrap();
            let g = &fx.graph;
            let mut c = PlasticCircuit::new(g, 1).unwrap();
            for i in 0..K_MAX {
                let mut cp = Checkpoint::start(i, g.edges(), &Limit::default());
                let n = ts::task_order(&d.spec, seed, i).len();
                let free = Limit::default();
                train_range(
                    &mut c,
                    &fx,
                    &d.spec,
                    seed,
                    Strategy::Tagged,
                    &free,
                    &mut cp,
                    n,
                )
                .unwrap();
                for e in 0..g.edges() {
                    if cp.tags.get(e) {
                        assert!(!fx.task.valence[i][e], "seed {seed} task {i} edge {e}");
                    }
                }
                assert_eq!(cp.stats.all_tagged, 0);
                assert_eq!(cp.stats.tagged_end, cp.tags.count());
                assert_eq!(cp.stats.cue_writes, 0);
                blocked += cp.stats.blocked;
            }
        }
        assert!(blocked > 0);
    }

    #[test]
    fn resumes_bit_exactly_from_checkpoints() {
        let d = small();
        let seed = 4;
        let fx = ns::fixture(None, &d, 50, seed).unwrap();
        let g = &fx.graph;
        let curves = curves_of(&fx, &d, seed);
        let lim = wb::limits(&curves, Budget::Paced(100));
        for s in STRATEGIES {
            let (_, lf, tf, wf) = run_arm(&fx, &d.spec, seed, s, &lim).unwrap();
            let mut c = PlasticCircuit::new(g, d.spec.init_weight).unwrap();
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
                assert_eq!(cp.ledger, lf[i], "{} task {i}", s.name());
                assert_eq!(cp.stats, tf[i], "{} task {i}", s.name());
            }
            assert_eq!(c.weights(), &wf[..], "{}", s.name());
        }
        let cp = Checkpoint::start(0, g.edges(), &Limit::default());
        assert!(Checkpoint::decode(&cp.encode(), g.edges() + 1).is_err());
        assert!(Checkpoint::decode("nope", g.edges()).is_err());
        assert_eq!(Checkpoint::decode(&cp.encode(), g.edges()).unwrap(), cp);
        let mut bad_range = Checkpoint::start(0, g.edges(), &Limit::default());
        bad_range.episode = 3;
        let mut c = PlasticCircuit::new(g, 1).unwrap();
        let free = Limit::default();
        let s = Strategy::ErrorGated;
        assert!(train_range(&mut c, &fx, &d.spec, seed, s, &free, &mut bad_range, 2).is_err());
    }

    #[test]
    fn grid_is_deterministic_csv_complete_and_readings_gate() {
        let d = small();
        let flips = [50, 100];
        let run = || run_grid_design(None, &d, &flips, &[1, 2, 3], &BUDGETS, &STRATEGIES).unwrap();
        let a = run();
        let b = run();
        assert_eq!(a, b);
        assert_eq!(a.len(), flips.len() * 3 * STRATEGIES.len() * BUDGETS.len());
        assert!(a
            .iter()
            .all(|r| r.inner.topology_unchanged && r.free_alloc_invariant));
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        let cols = csv_header().split(',').count();
        assert!(csv.lines().all(|l| l.split(',').count() == cols));
        let id = compare(&a, 50, K_MAX, Slice::All, EG, EG);
        assert_eq!(id.acc.ties, 3);
        let (g, h) = readings(&a, &flips);
        assert!(g.0);
        if !(g.0 && g.1 && g.2) {
            assert_eq!(h, [None; N_HYP]);
        } else {
            assert!(h.iter().all(Option::is_some));
        }
        let rep = report(&a, &flips, &BUDGETS);
        assert!(rep.contains("readings: G1_integrity=true"));
        assert_eq!(rep, report(&b, &flips, &BUDGETS));
        assert!(run_grid_design(None, &d, &[50], &[], &BUDGETS, &STRATEGIES).is_err());
        assert!(run_grid_design(None, &d, &[50], &[1], &[Budget::Paced(0)], &STRATEGIES).is_err());
        assert_eq!(
            Strategy::parse("E/E-cue-pattern-first").unwrap(),
            Strategy::EeCue(Alloc::PatternFirst)
        );
        assert!(Strategy::parse("x").is_err());
    }

    #[test]
    fn readings_on_empty_grid_are_not_evaluable() {
        let (g, h) = readings(&[], &FLIPS);
        assert!(!g.0 && !g.1);
        assert_eq!(h, [None; N_HYP]);
    }
}

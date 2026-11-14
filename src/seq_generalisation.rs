//! B888-4 cycle 18: **do the cycle 13–17 findings generalise to longer task
//! sequences (k = 8) and to held-out task orders?**
//!
//! Cycles 13–17 (#125–#129) were all measured on one four-task sequence
//! T1→T2→T3→T4 in its fixed order. Cycle 17 closed the silent cue's
//! learning-gap question at k = 4 (the remaining loss is write displacement)
//! and found that the cue's writes buy little at flip 50. This module reruns
//! the equal-budget comparison on **eight tasks** and on **task orders that
//! were never used before the main run** (two seeded permutations per seed,
//! in addition to the identity order of every earlier cycle).
//!
//! * Settings (re-used from `cue_displacement`, same code paths):
//!   `supervised@paced100` (control; on this teacher-clamped task the
//!   associative co-activation rule is the same update, see
//!   `docs/CONSOLIDATION.md`), `reward@paced100`, uncapped `error-gated`,
//!   `E/E-cue0@paced100` ("Z"), `error-gated@Z-pattern` ("Em") and uncapped
//!   `E/E-cue0` ("Zfree").
//! * Fixture: the #124 fixed-pool design generalised to `k` tasks with the
//!   **same total of reserved cue lines** (`design.cue_cap × 4`, 64 for the
//!   primary design), so each of the `k` cues gets `64 / k` lines and the
//!   pattern pool keeps its size. At `k = 4` the fixture is exactly
//!   `novel_split::fixture` of the #124–#129 design (tested).
//! * Order: a run trains the `k` tasks in `order` (a permutation of task
//!   ids). Each task keeps its own valence, cue, training set and episode
//!   order, so the identity order is bit-identical to
//!   `cue_displacement::run_setting` at `k = 4` (tested). Metrics are taken
//!   in **stage** coordinates: `R[i][j]` is the task trained at stage `j`
//!   measured after stage `i`.
//!
//! Engine state for checkpoint/resume = circuit weights
//! (`PlasticCircuit::save`) + [`Checkpoint`] (k, order, stage and the #129
//! per-task checkpoint); resuming mid-stage is bit-exact for every setting
//! and corrupt checkpoints are rejected (tested).
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph; "task", "cue"
//! and "order" name parts of an artificial benchmark, not claims about
//! synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/SEQ_GENERALISATION.md`
//! (committed and pushed before any learning run on the hypothesis seeds).
use crate::connectome::Connectome;
use crate::context_cue as cc;
use crate::cue_displacement::{self as cd, Setting, TaskLog};
use crate::cue_init as ci;
use crate::cue_reach as cr;
use crate::error_gated::{paired, Paired, PhaseCost, EFFECT};
use crate::generalization_bench::GenSpec;
use crate::novel_split::{self as ns, Class, Design, Eval, Fixture, Slice, TaskSplit, SLICES};
use crate::plasticity::PlasticCircuit;
use crate::task_sequence as ts;
use crate::write_allocation as wa;
use crate::write_budget::Limit;
use std::io::{self, ErrorKind};
use std::path::Path;

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Pre-registered sequence length of the main run.
pub const K_RUN: usize = 8;
/// Reference length (the #123–#129 sequence length; prefix of the run).
pub const K_REF: usize = 4;
/// Pre-registered flips (50 primary, 100 high conflict, 25 secondary).
pub const FLIPS: [u32; 3] = [25, 50, 100];
pub const PRIMARY_FLIP: u32 = 50;
pub const HIGH_FLIP: u32 = 100;
/// Pre-registered seeds: 321..=360, never used by an earlier B888-4 cycle.
pub const SEEDS: std::ops::RangeInclusive<u64> = 321..=360;
/// Pilot seeds (writes and pair counts only, identity order only).
pub const PILOT_SEEDS: std::ops::RangeInclusive<u64> = 1..=5;
/// Held-out task orders per seed (in addition to the identity order).
pub const HELDOUT_ORDERS: usize = 2;
/// Salt of the held-out order generator.
pub const ORDER_SALT: u64 = 0x0DE5_B888_4C18_0A1D;
/// The #124–#129 fixed-pool design (`cue_cap` = lines per cue at `K_REF`).
pub const DESIGN: Design = cd::DESIGN;
/// The pre-registered settings, in CSV order (Z before Em: Em's cap is
/// read from Z's run).
pub const SETTINGS: [Setting; 6] = [
    Setting::Sup,
    Setting::Rew,
    Setting::Err,
    Setting::Z,
    Setting::Em,
    Setting::Zfree,
];
/// Paced budget of every capped setting (percent of `error-gated`).
pub const PACED_PCT: u64 = 100;

/// Lines per cue for a `k`-task fixture: the reserved total
/// `design.cue_cap × K_REF` split evenly.
pub fn cue_cap(design: &Design, k: usize) -> io::Result<usize> {
    if k < 2 {
        return Err(bad("need at least two tasks"));
    }
    let cap = design.cue_cap * K_REF / k;
    if cap == 0 {
        return Err(bad("too many tasks for the reserved cue lines"));
    }
    Ok(cap)
}

/// `k`-task fixture of one (flip, seed). At `k = K_REF` it equals
/// `novel_split::fixture(graph, design, flip, seed)` (tested).
pub fn fixture(
    graph: Option<&Connectome>,
    design: &Design,
    k: usize,
    flip: u32,
    seed: u64,
) -> io::Result<Fixture> {
    let cap = cue_cap(design, k)?;
    let spec = GenSpec {
        flip_pct: flip,
        ..design.spec
    };
    let g = crate::reward_limits::graph_for(graph, &design.shape, seed)?;
    let cues = cc::choose_n_cues_capped(&g, k, cap)?;
    let task = ts::build_sequence(&g, &spec, seed, &cues.concat(), k)?;
    let splits = (0..k)
        .map(|j| TaskSplit::new(&g, &task.train[j], &task.valence[j]))
        .collect();
    Ok(Fixture {
        graph: g,
        cues,
        task,
        splits,
    })
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

/// Is `order` a permutation of `0..k`?
#[must_use]
pub fn is_permutation(order: &[usize], k: usize) -> bool {
    let mut seen = vec![false; k];
    order.len() == k
        && order
            .iter()
            .all(|&t| t < k && !std::mem::replace(&mut seen[t], true))
}

/// The task orders of one seed: index 0 is the identity order (every
/// earlier cycle, the pilot), indices `1..=n_heldout` are distinct seeded
/// permutations different from the identity (Fisher–Yates on an LCG keyed
/// by `seed ^ ORDER_SALT`). Independent of the flip.
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

/// Paced limits from per-stage cumulative write curves (the
/// `write_budget::limits` formula, any number of tasks).
#[must_use]
pub fn paced_limits(curves: &[Vec<u64>], pct: u64) -> Vec<Limit> {
    curves
        .iter()
        .map(|c| Limit(Some(c.iter().map(|&w| w * pct / 100).collect())))
        .collect()
}

/// Em's per-stage limits: Z's per-episode cumulative pattern-write curves.
#[must_use]
pub fn pattern_limits(z: &[TaskLog]) -> Vec<Limit> {
    z.iter()
        .map(|t| Limit(Some(t.pattern_curve(false))))
        .collect()
}

/// Unlimited limits for `k` stages.
#[must_use]
pub fn free(k: usize) -> Vec<Limit> {
    vec![Limit::default(); k]
}

/// Resumable state of one run: sequence length, order, stage and the
/// #129 per-task checkpoint of the task trained at that stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub k: usize,
    pub order: Vec<usize>,
    pub stage: usize,
    pub inner: cd::Checkpoint,
}

const MAGIC: &str = "v888-seq-generalisation-checkpoint-v1";

impl Checkpoint {
    /// Start of `stage` of a run in `order`.
    pub fn start(order: &[usize], stage: usize, edges: usize, limit: &Limit) -> io::Result<Self> {
        let k = order.len();
        if !is_permutation(order, k) || stage >= k {
            return Err(bad("bad order or stage"));
        }
        Ok(Self {
            k,
            order: order.to_vec(),
            stage,
            inner: cd::Checkpoint::start(order[stage], edges, limit),
        })
    }
    /// Deterministic text serialisation.
    #[must_use]
    pub fn encode(&self) -> String {
        let order: Vec<String> = self.order.iter().map(usize::to_string).collect();
        // The #127 inner format only admits task ids < 4; the task id is
        // stored here and the inner record carries 0.
        let mut inner = self.inner.clone();
        inner.inner.task = 0;
        format!(
            "{MAGIC}\nk {}\norder {}\nstage {}\ntask {}\n{}",
            self.k,
            order.join(" "),
            self.stage,
            self.inner.inner.task,
            inner.encode()
        )
    }
    /// Parse [`Checkpoint::encode`]; `edges` must match the graph.
    pub fn decode(s: &str, edges: usize) -> io::Result<Self> {
        let mut it = s.splitn(6, '\n');
        if it.next() != Some(MAGIC) {
            return Err(bad("not a seq-generalisation checkpoint"));
        }
        let mut field = |name: &str| -> io::Result<String> {
            let line = it.next().ok_or_else(|| bad("truncated checkpoint"))?;
            line.strip_prefix(name)
                .and_then(|v| v.strip_prefix(' '))
                .map(str::to_string)
                .ok_or_else(|| bad("bad checkpoint field"))
        };
        let num = |v: &str| v.parse::<usize>().map_err(|_| bad("bad number"));
        let k = num(&field("k")?)?;
        let order = field("order")?
            .split(' ')
            .map(num)
            .collect::<io::Result<Vec<usize>>>()?;
        let stage = num(&field("stage")?)?;
        let task = num(&field("task")?)?;
        let rest = it.next().ok_or_else(|| bad("truncated checkpoint"))?;
        if !is_permutation(&order, k) || stage >= k {
            return Err(bad("checkpoint order or stage invalid"));
        }
        if task != order[stage] {
            return Err(bad("checkpoint task does not match its stage"));
        }
        let mut inner = cd::Checkpoint::decode(rest, edges)?;
        if inner.inner.task != 0 {
            return Err(bad("bad inner checkpoint task"));
        }
        inner.inner.task = task;
        Ok(Self {
            k,
            order,
            stage,
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

/// Train episodes `cp.inner.inner.episode..end` of stage `cp.stage`.
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
    if cp.k != fx.task.train.len() || cp.inner.inner.task != cp.order[cp.stage] {
        return Err(bad("checkpoint does not match the fixture"));
    }
    cd::train_range(circuit, fx, spec, seed, s, limit, &mut cp.inner, end)
}

/// One measurement: (flip, seed, order, setting) on a k-task run.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub setting: Setting,
    pub flip: u32,
    pub seed: u64,
    pub k: usize,
    /// 0 = identity order; ≥ 1 = held-out order.
    pub order_ix: usize,
    pub order: Vec<usize>,
    pub pool: usize,
    pub reachable: usize,
    /// Lines of each task's cue (by task id; filled for uncued rows too).
    pub cue_lines: Vec<usize>,
    /// `ev[i][j]` (`j ≤ i`): task `order[j]` on TEST after stage `i`.
    pub ev: Vec<Vec<Eval>>,
    /// Per-stage write logs.
    pub logs: Vec<TaskLog>,
    pub cost: Vec<PhaseCost>,
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
    /// Balanced accuracy of stage-j's task after stage i (`j ≤ i`).
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
    /// TEST pairs of slice `s` of the task trained at stage `j`.
    #[must_use]
    pub fn pairs(&self, s: Slice, j: usize) -> u64 {
        s.pick(&self.ev[j][j]).total()
    }
    #[must_use]
    pub fn total_writes(&self) -> u64 {
        self.logs.iter().map(|t| t.total_writes(false)).sum()
    }
    #[must_use]
    pub fn pattern_writes(&self) -> u64 {
        self.logs.iter().map(|t| t.pattern_writes(false)).sum()
    }
    #[must_use]
    pub fn cue_writes(&self) -> u64 {
        self.logs.iter().map(|t| t.stats.cue_writes).sum()
    }
    #[must_use]
    pub fn refused(&self) -> u64 {
        self.logs.iter().map(|t| t.ledger.refused).sum()
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

/// Output of [`run_setting`]: per-stage evaluations (lower triangle),
/// per-stage logs and costs, final weights.
pub struct Run {
    pub ev: Vec<Vec<Eval>>,
    pub logs: Vec<TaskLog>,
    pub cost: Vec<PhaseCost>,
    pub weights: Vec<u8>,
    pub topology_unchanged: bool,
}

/// Run one setting over the fixture's tasks in `order` with per-stage write
/// limits (Zx, the exempt accounting of #129, is not a setting here).
pub fn run_setting(
    fx: &Fixture,
    spec: &GenSpec,
    seed: u64,
    s: Setting,
    order: &[usize],
    limits: &[Limit],
) -> io::Result<Run> {
    let graph = &fx.graph;
    let k = fx.task.train.len();
    if s.exempt() {
        return Err(bad("the exempt accounting is not part of this cycle"));
    }
    if fx.cues.len() != k || fx.splits.len() != k || limits.len() != k {
        return Err(bad("need k tasks, cues, splits and limits"));
    }
    if !is_permutation(order, k) {
        return Err(bad("order is not a permutation of the tasks"));
    }
    graph.validate()?;
    let original = graph.clone();
    let cue = ci::cue_edges(fx)?;
    let mut circuit = ci::initial_circuit(fx, spec, s.arm())?;
    let fingerprint = circuit.topology_fingerprint();
    let mut ev = Vec::with_capacity(k);
    let mut logs = Vec::with_capacity(k);
    let mut cost = Vec::with_capacity(k);
    for (i, &t) in order.iter().enumerate() {
        let episodes = ts::task_order(spec, seed, t).len();
        let mut cp = Checkpoint::start(order, i, graph.edges(), &limits[i])?;
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
        cost.push(cp.inner.inner.cost);
        logs.push(TaskLog {
            ledger: cp.inner.inner.ledger,
            stats: cp.inner.inner.stats,
            cue_cum: cp.inner.cue_cum,
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
    Ok(Run {
        ev,
        logs,
        cost,
        topology_unchanged: *graph == original
            && circuit.topology_fingerprint() == fingerprint
            && circuit.weights().len() == graph.edges(),
        weights: circuit.weights().to_vec(),
    })
}

/// All rows of one (flip, seed): per order, uncapped `error-gated` first
/// (its per-stage write curves set the `paced100` caps), Z before Em.
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
    let fx = fixture(graph, design, k, flip, seed)?;
    let reachable = cr::reachable_targets(&fx.graph);
    let cue_lines: Vec<usize> = fx.cues.iter().map(Vec::len).collect();
    let mut rows = Vec::new();
    for (o, order) in orders(seed, k, n_heldout)?.into_iter().enumerate() {
        let e = run_setting(&fx, &spec, seed, Setting::Err, &order, &free(k))?;
        let curves: Vec<Vec<u64>> = e.logs.iter().map(|t| t.ledger.cum.clone()).collect();
        let paced = paced_limits(&curves, PACED_PCT);
        let e_paced = run_setting(&fx, &spec, seed, Setting::Err, &order, &paced)?;
        let identity =
            e_paced.weights == e.weights && e_paced.logs.iter().all(|t| t.ledger.refused == 0);
        let mut z_logs: Option<Vec<TaskLog>> = None;
        for s in SETTINGS {
            let run = match s {
                Setting::Err => Run {
                    ev: e.ev.clone(),
                    logs: e.logs.clone(),
                    cost: e.cost.clone(),
                    weights: e.weights.clone(),
                    topology_unchanged: e.topology_unchanged,
                },
                Setting::Zfree => run_setting(&fx, &spec, seed, s, &order, &free(k))?,
                Setting::Em => {
                    let z = z_logs.as_ref().ok_or_else(|| bad("Em needs Z first"))?;
                    run_setting(&fx, &spec, seed, s, &order, &pattern_limits(z))?
                }
                _ => run_setting(&fx, &spec, seed, s, &order, &paced)?,
            };
            if s == Setting::Z {
                z_logs = Some(run.logs.clone());
            }
            let init = ci::initial_circuit(&fx, &spec, s.arm())?;
            rows.push(Row {
                setting: s,
                flip,
                seed,
                k,
                order_ix: o,
                order: order.clone(),
                pool: fx.task.pool.len(),
                reachable,
                cue_lines: cue_lines.clone(),
                saturated_end: crate::error_gated::saturated(&run.weights),
                topology_unchanged: run.topology_unchanged,
                init_ok: ci::init_matches(&fx, &spec, s.arm(), init.weights())?,
                paced_identity: s != Setting::Err || identity,
                digest: wa::digest(&run.weights),
                ev: run.ev,
                logs: run.logs,
                cost: run.cost,
            });
        }
    }
    Ok(rows)
}

/// Full grid, (flip, seed) jobs spread over `threads` worker threads; the
/// output order (flip, seed, order, setting) and content do not depend on
/// the thread count (tested).
#[allow(clippy::too_many_arguments)]
pub fn run_grid_threads(
    graph: Option<&Connectome>,
    design: &Design,
    k: usize,
    flips: &[u32],
    seeds: &[u64],
    n_heldout: usize,
    threads: usize,
) -> io::Result<Vec<Row>> {
    if seeds.is_empty() || flips.is_empty() {
        return Err(bad("need at least one seed and flip"));
    }
    let jobs: Vec<(u32, u64)> = flips
        .iter()
        .flat_map(|&f| seeds.iter().map(move |&s| (f, s)))
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
                            let (f, s) = jobs[ix];
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

/// Default worker count (does not change any output).
#[must_use]
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

/// Pre-registered grid on the fixed-pool design at `K_RUN`.
pub fn run_grid(graph: Option<&Connectome>, flips: &[u32], seeds: &[u64]) -> io::Result<Vec<Row>> {
    run_grid_threads(
        graph,
        &DESIGN,
        K_RUN,
        flips,
        seeds,
        HELDOUT_ORDERS,
        default_threads(),
    )
}

/// Which task orders a statistic uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orders {
    /// The identity order of every earlier cycle.
    Identity,
    /// The held-out orders (per-seed mean over them).
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

fn rows_of(rows: &[Row], flip: u32, x: Setting, o: Orders) -> impl Iterator<Item = &Row> {
    rows.iter()
        .filter(move |r| r.flip == flip && r.setting == x && o.has(r))
}

fn seeds_of(rows: &[Row], flip: u32) -> Vec<u64> {
    let mut s: Vec<u64> = rows
        .iter()
        .filter(|r| r.flip == flip)
        .map(|r| r.seed)
        .collect();
    s.sort_unstable();
    s.dedup();
    s
}

/// Per-seed mean of `f` over the rows of (flip, seed, setting, orders).
fn seed_value(
    rows: &[Row],
    flip: u32,
    seed: u64,
    x: Setting,
    o: Orders,
    f: &dyn Fn(&Row) -> f64,
) -> Option<f64> {
    let v: Vec<f64> = rows_of(rows, flip, x, o)
        .filter(|r| r.seed == seed)
        .map(f)
        .collect();
    if v.is_empty() {
        None
    } else {
        Some(v.iter().sum::<f64>() / v.len() as f64)
    }
}

/// Per-seed differences `g(seed)` over seeds where it is defined.
fn per_seed(rows: &[Row], flip: u32, g: &dyn Fn(u64) -> Option<f64>) -> Vec<f64> {
    seeds_of(rows, flip).into_iter().filter_map(g).collect()
}

/// Per-seed difference f(x) − f(base) on the given orders.
fn diff(
    rows: &[Row],
    flip: u32,
    o: Orders,
    x: Setting,
    base: Setting,
    f: &dyn Fn(&Row) -> f64,
) -> Vec<f64> {
    per_seed(rows, flip, &|s| {
        Some(seed_value(rows, flip, s, x, o, f)? - seed_value(rows, flip, s, base, o, f)?)
    })
}

/// x vs base paired by seed at k (fields oriented so that > 0 favours x).
#[must_use]
pub fn compare(
    rows: &[Row],
    flip: u32,
    k: usize,
    s: Slice,
    o: Orders,
    x: Setting,
    base: Setting,
) -> cr::Comparison {
    let d = |f: &dyn Fn(&Row) -> f64| paired(&diff(rows, flip, o, x, base, f));
    cr::Comparison {
        acc: d(&|r| r.acc(s, k)),
        learn: d(&|r| r.learn(s, k)),
        last: d(&|r| r.last(s, k)),
        forget: d(&|r| -r.forgetting(s, k)),
    }
}

/// Growth with k: per seed [ACC(x) − ACC(base)] at `k_hi` minus the same at
/// `k_lo` (all pairs, given orders).
#[must_use]
pub fn growth(
    rows: &[Row],
    flip: u32,
    o: Orders,
    x: Setting,
    base: Setting,
    k_lo: usize,
    k_hi: usize,
) -> Paired {
    let a = Slice::All;
    let hi = diff(rows, flip, o, x, base, &|r| r.acc(a, k_hi));
    let lo = diff(rows, flip, o, x, base, &|r| r.acc(a, k_lo));
    paired(&hi.iter().zip(&lo).map(|(h, l)| h - l).collect::<Vec<_>>())
}

/// Order shift: per seed [ACC_s(x) − ACC_s(base)] on held-out orders minus
/// the same on the identity order.
#[must_use]
pub fn order_shift(
    rows: &[Row],
    flip: u32,
    k: usize,
    s: Slice,
    x: Setting,
    base: Setting,
) -> Paired {
    let f = |r: &Row| r.acc(s, k);
    let h = diff(rows, flip, Orders::Heldout, x, base, &f);
    let i = diff(rows, flip, Orders::Identity, x, base, &f);
    paired(&h.iter().zip(&i).map(|(a, b)| a - b).collect::<Vec<_>>())
}

/// Equivalence within ±EFFECT: 95 % CI inside (−0.05, +0.05).
#[must_use]
pub fn equivalent(d: &Paired) -> bool {
    d.n > 0 && d.ci_lo > -EFFECT && d.ci_hi < EFFECT
}

/// Positive: mean > 0 and bootstrap CI lower bound > 0.
#[must_use]
pub fn positive(d: &Paired) -> bool {
    d.n > 0 && d.mean > 0.0 && d.ci_lo > 0.0
}

/// Mean of `f` over the rows of one (flip, setting, orders).
#[must_use]
pub fn setting_mean(
    rows: &[Row],
    flip: u32,
    x: Setting,
    o: Orders,
    f: &dyn Fn(&Row) -> f64,
) -> f64 {
    mean(rows_of(rows, flip, x, o).map(f))
}

/// Fraction of (seed, order) rows of x that refused at least one step.
#[must_use]
pub fn binding_frac(rows: &[Row], flip: u32, x: Setting, o: Orders) -> f64 {
    let v: Vec<&Row> = rows_of(rows, flip, x, o).collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().filter(|r| r.refused() > 0).count() as f64 / v.len() as f64
    }
}

/// Fraction of (seed, order) rows of x with at least one cue write.
#[must_use]
pub fn cue_used_frac(rows: &[Row], flip: u32, x: Setting, o: Orders) -> f64 {
    let v: Vec<&Row> = rows_of(rows, flip, x, o).collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().filter(|r| r.cue_writes() > 0).count() as f64 / v.len() as f64
    }
}

fn median(mut v: Vec<u64>) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

/// Median CN pairs per (seed, task) at a flip (identity-order
/// `error-gated` rows; counts do not depend on the setting or order).
#[must_use]
pub fn median_cn(rows: &[Row], flip: u32) -> u64 {
    median(
        rows_of(rows, flip, Setting::Err, Orders::Identity)
            .flat_map(|r| (0..r.k).map(move |j| r.pairs(Slice::Cn, j)))
            .collect(),
    )
}

fn find<'a>(rows: &'a [Row], r: &Row, x: Setting) -> Option<&'a Row> {
    rows.iter().find(|y| {
        y.flip == r.flip && y.seed == r.seed && y.order_ix == r.order_ix && y.setting == x
    })
}

/// Em's per-stage cap equals Z's pattern writes in the same (flip, seed,
/// order).
#[must_use]
pub fn em_matched(rows: &[Row]) -> bool {
    rows.iter().filter(|r| r.setting == Setting::Em).all(|em| {
        find(rows, em, Setting::Z).is_some_and(|z| {
            em.logs.len() == z.logs.len()
                && em
                    .logs
                    .iter()
                    .zip(&z.logs)
                    .all(|(a, b)| a.ledger.cap == b.pattern_writes(false))
        })
    })
}

/// Every order is a permutation, order 0 is the identity, held-out orders
/// differ from it and from each other, and every setting of a (flip, seed)
/// used the same orders.
#[must_use]
pub fn orders_valid(rows: &[Row]) -> bool {
    rows.iter().all(|r| {
        is_permutation(&r.order, r.k)
            && (r.order_ix == 0) == r.order.iter().enumerate().all(|(i, &t)| i == t)
            && rows
                .iter()
                .filter(|y| y.flip == r.flip && y.seed == r.seed)
                .all(|y| (y.order_ix == r.order_ix) == (y.order == r.order))
    })
}

/// Design gates: (G1 integrity, G2 manipulation, G3 CN pairs).
#[must_use]
pub fn gates(rows: &[Row]) -> (bool, bool, bool) {
    let g1 = !rows.is_empty()
        && rows
            .iter()
            .all(|r| r.topology_unchanged && r.init_ok && r.paced_identity && r.within_cap())
        && em_matched(rows)
        && orders_valid(rows);
    let f = PRIMARY_FLIP;
    let h = Orders::Heldout;
    let g2 = binding_frac(rows, f, Setting::Em, h) >= 0.5
        && cue_used_frac(rows, f, Setting::Z, h) >= 0.5;
    let g3 = median_cn(rows, f) >= ns::MIN_NOVEL_PAIRS;
    (g1, g2, g3)
}

/// Number of pre-registered hypotheses.
pub const N_HYP: usize = 8;

/// Pre-registered readings: (gates, [H1..H8]); `None` if a gate fails or
/// the hypothesis' flip is missing. Hypotheses use the run length of the
/// rows (`K_RUN` in the main run) and `K_REF` for H5.
#[must_use]
pub fn readings(rows: &[Row], flips: &[u32]) -> ((bool, bool, bool), [Option<bool>; N_HYP]) {
    use Setting::{Em, Err, Rew, Sup, Z};
    let g = gates(rows);
    let mut h = [None; N_HYP];
    if !(g.0 && g.1 && g.2) {
        return (g, h);
    }
    let k = rows.iter().map(|r| r.k).min().unwrap_or(0);
    let (all, cn, ho) = (Slice::All, Slice::Cn, Orders::Heldout);
    let f = PRIMARY_FLIP;
    if flips.contains(&f) {
        h[0] = Some(compare(rows, f, k, all, ho, Err, Sup).acc.strong());
        h[1] = Some(compare(rows, f, k, cn, ho, Z, Sup).passes());
        h[2] = Some(equivalent(&compare(rows, f, k, all, ho, Em, Z).learn));
        h[3] = Some(compare(rows, f, k, all, ho, Z, Em).acc.strong());
        if k > K_REF {
            h[4] = Some(positive(&growth(rows, f, ho, Z, Em, K_REF, k)));
        }
        h[6] = Some(equivalent(&order_shift(rows, f, k, cn, Z, Sup)));
        h[7] = Some(
            positive(&compare(rows, f, k, all, ho, Err, Rew).acc)
                && positive(&compare(rows, f, k, all, ho, Rew, Sup).acc),
        );
    }
    if flips.contains(&HIGH_FLIP) {
        h[5] = Some(compare(rows, HIGH_FLIP, k, all, ho, Z, Em).passes());
    }
    (g, h)
}

/// Stage-trained task pair counts of one fixture task (no accuracy):
/// (all, seen, cn, icn, ue).
#[must_use]
pub fn pair_counts(fx: &Fixture, j: usize) -> [u64; 5] {
    let mut c = [0_u64; 5];
    for p in &fx.task.test {
        for (t, conf) in ts::configurations(&fx.graph, p) {
            c[0] += 1;
            match fx.splits[j].classify(t, &conf) {
                Class::Seen => c[1] += 1,
                Class::Compositional => {
                    c[2] += 1;
                    if fx.splits[j].inferable(&conf, &fx.task.valence[j]) {
                        c[3] += 1;
                    }
                }
                Class::UnseenEdge => c[4] += 1,
            }
        }
    }
    c
}

fn dash<T: ToString>(v: &[T]) -> String {
    v.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("-")
}

/// CSV header for k-task rows (generated; see `to_csv`).
#[must_use]
pub fn csv_header(k: usize) -> String {
    let mut h: Vec<String> = [
        "schema",
        "graph",
        "k",
        "flip_pct",
        "setting",
        "seed",
        "order_ix",
        "order_kind",
        "order",
        "pool",
        "reachable_targets",
        "cue_lines",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    for s in SLICES {
        for j in 1..=k {
            h.push(format!("pairs_{}_s{j}", s.name()));
        }
    }
    for s in [Slice::All, Slice::Cn] {
        for i in 1..=k {
            for j in 1..=i {
                h.push(format!("r_{}_{i}_{j}", s.name()));
            }
        }
    }
    for s in SLICES {
        for name in ["acc", "learn", "forgetting", "last"] {
            for kk in 2..=k {
                h.push(format!("{name}_{}_k{kk}", s.name()));
            }
        }
    }
    for name in [
        "cap",
        "charged_writes",
        "pattern_writes",
        "cue_writes",
        "refused",
        "cue_refused",
        "first_refusal",
        "attempts",
        "cue_nonzero_end",
        "cue_mass_end",
    ] {
        for j in 1..=k {
            h.push(format!("{name}_s{j}"));
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

/// Deterministic CSV (no timing column). All rows must share one k.
#[must_use]
pub fn to_csv(rows: &[Row], graph_label: &str) -> String {
    let k = rows.first().map_or(K_RUN, |r| r.k);
    let mut out = csv_header(k);
    out.push('\n');
    for r in rows.iter().filter(|r| r.k == k) {
        let mut f: Vec<String> = vec![
            "b888-4-seq-generalisation-v1".to_string(),
            graph_label.to_string(),
            r.k.to_string(),
            r.flip.to_string(),
            r.setting.name().to_string(),
            r.seed.to_string(),
            r.order_ix.to_string(),
            if r.heldout() { "heldout" } else { "identity" }.to_string(),
            dash(&r.order),
            r.pool.to_string(),
            r.reachable.to_string(),
            dash(&r.cue_lines),
        ];
        for s in SLICES {
            f.extend((0..k).map(|j| r.pairs(s, j).to_string()));
        }
        for s in [Slice::All, Slice::Cn] {
            for i in 0..k {
                f.extend((0..=i).map(|j| format!("{:.6}", r.r(s, i, j))));
            }
        }
        for s in SLICES {
            for kk in 2..=k {
                f.push(format!("{:.6}", r.acc(s, kk)));
            }
            for kk in 2..=k {
                f.push(format!("{:.6}", r.learn(s, kk)));
            }
            for kk in 2..=k {
                f.push(format!("{:.6}", r.forgetting(s, kk)));
            }
            for kk in 2..=k {
                f.push(format!("{:.6}", r.last(s, kk)));
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
        f.extend(r.logs.iter().map(|t| t.pattern_writes(false).to_string()));
        f.extend(r.logs.iter().map(|t| t.stats.cue_writes.to_string()));
        f.extend(r.logs.iter().map(|t| t.ledger.refused.to_string()));
        f.extend(r.logs.iter().map(|t| t.stats.cue_refused.to_string()));
        f.extend(r.logs.iter().map(|t| t.ledger.first_refusal.to_string()));
        f.extend(r.cost.iter().map(|c| c.attempts.to_string()));
        f.extend(r.logs.iter().map(|t| t.cue_nonzero_end.to_string()));
        f.extend(r.logs.iter().map(|t| t.cue_mass_end.to_string()));
        f.push(r.saturated_end.to_string());
        f.push(r.init_ok.to_string());
        f.push(r.paced_identity.to_string());
        f.push(format!("{:016x}", r.digest));
        f.push(r.topology_unchanged.to_string());
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
    use Setting::{Em, Err, Rew, Sup, Zfree, Z};
    let mut o = String::new();
    let k = rows.iter().map(|r| r.k).min().unwrap_or(0);
    let (a, c) = (Slice::All, Slice::Cn);
    let mut ks: Vec<usize> = vec![2, K_REF, 6, k];
    ks.retain(|&x| x >= 2 && x <= k);
    ks.sort_unstable();
    ks.dedup();
    let _ = writeln!(
        o,
        "setting_summary: flip,orders,k,setting,acc_all,acc_cn,acc_icn,acc_seen,acc_ue,learn_all,learn_cn,fgt_all,fgt_cn,last_all,last_cn,pattern_writes,cue_writes,total_writes,refused,cue_nonzero_end_last,cue_mass_end_last,saturated_end"
    );
    for &flip in flips {
        for ord in [Orders::Identity, Orders::Heldout] {
            for &kk in &[K_REF.min(k), k] {
                for s in SETTINGS {
                    if rows_of(rows, flip, s, ord).next().is_none() {
                        continue;
                    }
                    let m = |f: &dyn Fn(&Row) -> f64| setting_mean(rows, flip, s, ord, f);
                    let _ = writeln!(
                        o,
                        "{flip},{},{kk},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
                        ord.name(),
                        s.name(),
                        m(&|r| r.acc(a, kk)),
                        m(&|r| r.acc(c, kk)),
                        m(&|r| r.acc(Slice::Icn, kk)),
                        m(&|r| r.acc(Slice::Seen, kk)),
                        m(&|r| r.acc(Slice::Ue, kk)),
                        m(&|r| r.learn(a, kk)),
                        m(&|r| r.learn(c, kk)),
                        m(&|r| r.forgetting(a, kk)),
                        m(&|r| r.forgetting(c, kk)),
                        m(&|r| r.last(a, kk)),
                        m(&|r| r.last(c, kk)),
                        m(&|r| r.logs[..kk].iter().map(|t| t.pattern_writes(false)).sum::<u64>() as f64),
                        m(&|r| r.logs[..kk].iter().map(|t| t.stats.cue_writes).sum::<u64>() as f64),
                        m(&|r| r.logs[..kk].iter().map(|t| t.total_writes(false)).sum::<u64>() as f64),
                        m(&|r| r.logs[..kk].iter().map(|t| t.ledger.refused).sum::<u64>() as f64),
                        m(&|r| r.logs[kk - 1].cue_nonzero_end as f64),
                        m(&|r| r.logs[kk - 1].cue_mass_end as f64),
                        m(&|r| r.saturated_end as f64),
                    );
                }
            }
        }
    }
    let _ = writeln!(
        o,
        "comparison(k): flip,orders,k,slice,x,base,d_acc,d_learn,d_last,d_fgt(+=forgets_less),passes_sequence_rule"
    );
    let base_pairs = [
        (Err, Sup),
        (Z, Sup),
        (Em, Sup),
        (Rew, Sup),
        (Err, Rew),
        (Em, Z),
        (Z, Em),
        (Em, Err),
        (Z, Err),
        (Zfree, Err),
    ];
    for &flip in flips {
        for ord in [Orders::Identity, Orders::Heldout] {
            for &kk in &ks {
                for s in SLICES {
                    if kk != k && !matches!(s, Slice::All | Slice::Cn) {
                        continue;
                    }
                    for &(x, base) in &base_pairs {
                        let cmp = compare(rows, flip, kk, s, ord, x, base);
                        if cmp.acc.n == 0 {
                            continue;
                        }
                        let _ = writeln!(
                            o,
                            "{flip},{},{kk},{},{},{},{},{},{},{},{}",
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
        "growth(acc_all, k vs {K_REF}): flip,orders,x,base,d(k)-d({K_REF})"
    );
    for &flip in flips {
        for ord in [Orders::Identity, Orders::Heldout] {
            for &(x, base) in &[(Z, Em), (Z, Sup), (Err, Sup), (Z, Err)] {
                let g = growth(rows, flip, ord, x, base, K_REF.min(k), k);
                if g.n > 0 {
                    let _ = writeln!(
                        o,
                        "{flip},{},{},{},{}",
                        ord.name(),
                        x.name(),
                        base.name(),
                        fmt_p(&g)
                    );
                }
            }
        }
    }
    let _ = writeln!(
        o,
        "order_shift(k, heldout - identity): flip,slice,x,base,shift_acc"
    );
    for &flip in flips {
        for s in [Slice::All, Slice::Cn] {
            for &(x, base) in &[(Z, Sup), (Err, Sup), (Z, Em), (Rew, Sup)] {
                let d = order_shift(rows, flip, k, s, x, base);
                if d.n > 0 {
                    let _ = writeln!(
                        o,
                        "{flip},{},{},{},{}",
                        s.name(),
                        x.name(),
                        base.name(),
                        fmt_p(&d)
                    );
                }
            }
        }
        let _ = writeln!(
            o,
            "manipulation: flip={flip} binding_frac(Em,heldout)={:.3} binding_frac(Em,identity)={:.3} cue_used_frac(Z,heldout)={:.3} mean_pattern_writes(Z,heldout)={:.1} mean_writes(error-gated,heldout)={:.1} median_cn={}",
            binding_frac(rows, flip, Em, Orders::Heldout),
            binding_frac(rows, flip, Em, Orders::Identity),
            cue_used_frac(rows, flip, Z, Orders::Heldout),
            setting_mean(rows, flip, Z, Orders::Heldout, &|r| r.pattern_writes() as f64),
            setting_mean(rows, flip, Err, Orders::Heldout, &|r| r.total_writes() as f64),
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

/// One pilot census line (identity order only; no accuracy computed):
/// seed, setting, pattern writes, cue writes, total writes, refused.
pub type CensusLine = (u64, Setting, u64, u64, u64, u64);

/// Fixture facts of one pilot seed: pool, cue lines per cue, cue reach
/// per cue, CN pairs per task.
pub type FixtureLine = (u64, usize, Vec<usize>, Vec<usize>, Vec<u64>);

/// Pilot census on the identity order: fixture facts and writes /
/// refusals only. No accuracy is computed and no held-out order is drawn.
pub fn census(
    graph: Option<&Connectome>,
    design: &Design,
    k: usize,
    flip: u32,
    seeds: &[u64],
) -> io::Result<(Vec<FixtureLine>, Vec<CensusLine>)> {
    let spec = GenSpec {
        flip_pct: flip,
        ..design.spec
    };
    let mut facts = Vec::new();
    let mut out = Vec::new();
    for &seed in seeds {
        let fx = fixture(graph, design, k, flip, seed)?;
        facts.push((
            seed,
            fx.task.pool.len(),
            fx.cues.iter().map(Vec::len).collect(),
            fx.cues
                .iter()
                .map(|c| cr::reach_set(&fx.graph, c).len())
                .collect(),
            (0..k).map(|j| pair_counts(&fx, j)[2]).collect(),
        ));
        let order: Vec<usize> = (0..k).collect();
        let train_only = |s: Setting, lim: &[Limit]| -> io::Result<Vec<TaskLog>> {
            let mut c = ci::initial_circuit(&fx, &spec, s.arm())?;
            let mut logs = Vec::with_capacity(k);
            for (i, l) in lim.iter().enumerate() {
                let n = ts::task_order(&spec, seed, order[i]).len();
                let mut cp = Checkpoint::start(&order, i, fx.graph.edges(), l)?;
                train_range(&mut c, &fx, &spec, seed, s, l, &mut cp, n)?;
                logs.push(TaskLog {
                    ledger: cp.inner.inner.ledger,
                    stats: cp.inner.inner.stats,
                    cue_cum: cp.inner.cue_cum,
                    ..TaskLog::default()
                });
            }
            Ok(logs)
        };
        let e = train_only(Setting::Err, &free(k))?;
        let curves: Vec<Vec<u64>> = e.iter().map(|t| t.ledger.cum.clone()).collect();
        let paced = paced_limits(&curves, PACED_PCT);
        let mut z: Option<Vec<TaskLog>> = None;
        for s in SETTINGS {
            let logs = match s {
                Setting::Err => e.clone(),
                Setting::Zfree => train_only(s, &free(k))?,
                Setting::Em => train_only(
                    s,
                    &pattern_limits(z.as_ref().ok_or_else(|| bad("Em needs Z"))?),
                )?,
                _ => train_only(s, &paced)?,
            };
            out.push((
                seed,
                s,
                logs.iter().map(|t| t.pattern_writes(false)).sum(),
                logs.iter().map(|t| t.stats.cue_writes).sum(),
                logs.iter().map(|t| t.total_writes(false)).sum(),
                logs.iter().map(|t| t.ledger.refused).sum(),
            ));
            if s == Setting::Z {
                z = Some(logs);
            }
        }
    }
    Ok((facts, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generalization_bench::GraphShape;
    use crate::task_sequence::K_MAX;
    use crate::write_budget::{self as wb, Budget};

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
        std::env::temp_dir().join(format!("v888-seq-gen-{tag}-{}", std::process::id()))
    }

    #[test]
    fn k4_fixture_is_the_cycle17_fixture_and_cue_lines_are_conserved() {
        let d = small();
        for seed in [1, 2, 3] {
            let a = fixture(None, &d, K_REF, 50, seed).unwrap();
            let b = ns::fixture(None, &d, 50, seed).unwrap();
            assert_eq!(a.graph, b.graph);
            assert_eq!(a.cues, b.cues);
            assert_eq!(a.task, b.task);
            assert_eq!(a.splits, b.splits);
            let f8 = fixture(None, &d, 8, 50, seed).unwrap();
            assert_eq!(f8.cues.len(), 8);
            assert!(f8.cues.iter().all(|c| !c.is_empty() && c.len() <= 2));
            assert_eq!(f8.task.train.len(), 8);
            assert_eq!(f8.task.valence.len(), 8);
        }
        assert_eq!(cue_cap(&DESIGN, K_RUN).unwrap(), 8);
        assert_eq!(cue_cap(&DESIGN, K_REF).unwrap(), cr::FULL);
        assert!(cue_cap(&DESIGN, 1).is_err());
        assert!(cue_cap(&small(), 100).is_err());
    }

    #[test]
    fn identity_order_matches_cue_displacement_bit_exactly() {
        let d = small();
        for seed in [1, 2] {
            let fx = ns::fixture(None, &d, 50, seed).unwrap();
            let order: Vec<usize> = (0..K_MAX).collect();
            let (_, eg, _) =
                cd::run_setting(&fx, &d.spec, seed, Setting::Err, &wb::free()).unwrap();
            let curves: Vec<Vec<u64>> = eg.iter().map(|t| t.ledger.cum.clone()).collect();
            let old = wb::limits(&curves, Budget::Paced(100));
            let new = paced_limits(&curves, PACED_PCT);
            assert_eq!(&old[..], &new[..]);
            let (_, zl, _) = cd::run_setting(&fx, &d.spec, seed, Setting::Z, &old).unwrap();
            for s in SETTINGS {
                let (lo, ln): ([Limit; K_MAX], Vec<Limit>) = match s {
                    Setting::Err | Setting::Zfree => (wb::free(), free(K_MAX)),
                    Setting::Em => (cd::pattern_limits(&zl), pattern_limits(&zl)),
                    _ => (old.clone(), new.clone()),
                };
                assert_eq!(&lo[..], &ln[..]);
                let (r0, l0, w0) = cd::run_setting(&fx, &d.spec, seed, s, &lo).unwrap();
                let run = run_setting(&fx, &d.spec, seed, s, &order, &ln).unwrap();
                assert_eq!(run.weights, w0, "{}", s.name());
                assert_eq!(&run.logs[..], &l0[..], "{}", s.name());
                assert_eq!(&run.cost[..], &r0.cost[..]);
                for i in 0..K_MAX {
                    assert_eq!(run.ev[i].len(), i + 1);
                    for j in 0..=i {
                        assert_eq!(run.ev[i][j], r0.ev[i][j], "{} {i} {j}", s.name());
                    }
                }
            }
        }
    }

    #[test]
    fn heldout_orders_are_deterministic_valid_and_distinct() {
        for seed in [1, 2, 321, 360] {
            let o = orders(seed, 8, HELDOUT_ORDERS).unwrap();
            assert_eq!(o, orders(seed, 8, HELDOUT_ORDERS).unwrap());
            assert_eq!(o.len(), 1 + HELDOUT_ORDERS);
            assert_eq!(o[0], (0..8).collect::<Vec<_>>());
            assert!(o.iter().all(|p| is_permutation(p, 8)));
            assert_ne!(o[1], o[0]);
            assert_ne!(o[2], o[0]);
            assert_ne!(o[1], o[2]);
        }
        assert_ne!(orders(1, 8, 2).unwrap(), orders(2, 8, 2).unwrap());
        assert!(orders(1, 2, 1).is_ok());
        assert!(orders(1, 2, 2).is_err());
        assert!(orders(1, 1, 0).is_err());
        assert!(is_permutation(&[2, 0, 1], 3));
        assert!(!is_permutation(&[2, 2, 1], 3));
        assert!(!is_permutation(&[0, 1], 3));
        assert!(!is_permutation(&[0, 1, 3], 3));
    }

    #[test]
    fn heldout_order_trains_tasks_in_that_order() {
        let d = small();
        let seed = 3;
        let fx = fixture(None, &d, 6, 50, seed).unwrap();
        let o = orders(seed, 6, 1).unwrap();
        let p = &o[1];
        let run = run_setting(&fx, &d.spec, seed, Setting::Err, p, &free(6)).unwrap();
        // Stage 0 of the permuted order equals the first task trained alone.
        let id: Vec<usize> = std::iter::once(p[0])
            .chain((0..6).filter(|&t| t != p[0]))
            .collect();
        let run2 = run_setting(&fx, &d.spec, seed, Setting::Err, &id, &free(6)).unwrap();
        assert_eq!(run.logs[0], run2.logs[0]);
        assert_eq!(run.ev[0], run2.ev[0]);
        assert!(run_setting(&fx, &d.spec, seed, Setting::Err, &[0, 1, 2], &free(6)).is_err());
        assert!(run_setting(&fx, &d.spec, seed, Setting::Zx, p, &free(6)).is_err());
        assert!(run_setting(&fx, &d.spec, seed, Setting::Err, p, &free(5)).is_err());
    }

    #[test]
    fn resumes_bit_exactly_from_checkpoints() {
        let d = small();
        let seed = 4;
        let k = 6;
        let fx = fixture(None, &d, k, 50, seed).unwrap();
        let g = &fx.graph;
        let order = orders(seed, k, 1).unwrap().remove(1);
        let e = run_setting(&fx, &d.spec, seed, Setting::Err, &order, &free(k)).unwrap();
        let curves: Vec<Vec<u64>> = e.logs.iter().map(|t| t.ledger.cum.clone()).collect();
        let paced = paced_limits(&curves, PACED_PCT);
        let z = run_setting(&fx, &d.spec, seed, Setting::Z, &order, &paced).unwrap();
        for s in SETTINGS {
            let lim = match s {
                Setting::Err | Setting::Zfree => free(k),
                Setting::Em => pattern_limits(&z.logs),
                _ => paced.clone(),
            };
            let full = run_setting(&fx, &d.spec, seed, s, &order, &lim).unwrap();
            let mut c = ci::initial_circuit(&fx, &d.spec, s.arm()).unwrap();
            for (i, &t) in order.iter().enumerate() {
                let n = ts::task_order(&d.spec, seed, t).len();
                let mut cp = Checkpoint::start(&order, i, g.edges(), &lim[i]).unwrap();
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
                assert_eq!(
                    cp.inner.inner.ledger,
                    full.logs[i].ledger,
                    "{} {i}",
                    s.name()
                );
                assert_eq!(cp.inner.inner.stats, full.logs[i].stats, "{} {i}", s.name());
                assert_eq!(cp.inner.cue_cum, full.logs[i].cue_cum, "{} {i}", s.name());
            }
            assert_eq!(c.weights(), &full.weights[..], "{}", s.name());
        }
    }

    #[test]
    fn corrupt_checkpoints_are_rejected() {
        let d = small();
        let fx = fixture(None, &d, 6, 50, 1).unwrap();
        let e = fx.graph.edges();
        let order = vec![2, 0, 1, 5, 4, 3];
        let cp = Checkpoint::start(&order, 3, e, &Limit::default()).unwrap();
        let enc = cp.encode();
        assert_eq!(Checkpoint::decode(&enc, e).unwrap(), cp);
        assert!(Checkpoint::decode(&enc.replacen("v888", "x", 1), e).is_err());
        assert!(Checkpoint::decode(&enc, e + 1).is_err());
        assert!(
            Checkpoint::decode(&enc.replace("order 2 0 1 5 4 3", "order 2 0 1 5 4 4"), e).is_err()
        );
        assert!(
            Checkpoint::decode(&enc.replace("order 2 0 1 5 4 3", "order 2 0 1 5 4"), e).is_err()
        );
        assert!(Checkpoint::decode(&enc.replace("stage 3", "stage 6"), e).is_err());
        assert!(Checkpoint::decode(&enc.replace("stage 3", "stage 2"), e).is_err());
        assert!(Checkpoint::decode(&enc.replace("k 6", "k x"), e).is_err());
        assert!(Checkpoint::decode(&enc.replace("cuecum -", "cuecum 3"), e).is_err());
        assert!(Checkpoint::decode(&enc.replace("task 5", "task 4"), e).is_err());
        assert!(Checkpoint::decode(&enc.replace("\ntask 0\n", "\ntask 1\n"), e).is_err());
        let cut: String = enc.lines().take(3).collect::<Vec<_>>().join("\n");
        assert!(Checkpoint::decode(&cut, e).is_err());
        assert!(Checkpoint::start(&order, 6, e, &Limit::default()).is_err());
        assert!(Checkpoint::start(&[0, 0], 0, e, &Limit::default()).is_err());
        // A checkpoint of another run length is refused by train_range.
        let fx4 = fixture(None, &d, 4, 50, 1).unwrap();
        let mut c = PlasticCircuit::new(&fx4.graph, 1).unwrap();
        let mut cp = cp;
        assert!(train_range(
            &mut c,
            &fx4,
            &d.spec,
            1,
            Setting::Err,
            &Limit::default(),
            &mut cp,
            1
        )
        .is_err());
    }

    #[test]
    fn grid_is_deterministic_thread_independent_and_readings_gate() {
        let d = small();
        let flips = [50, 100];
        let seeds = [1, 2, 3];
        let a = run_grid_threads(None, &d, 6, &flips, &seeds, 2, 1).unwrap();
        let b = run_grid_threads(None, &d, 6, &flips, &seeds, 2, 4).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), flips.len() * seeds.len() * 3 * SETTINGS.len());
        assert!(a
            .iter()
            .all(|r| r.topology_unchanged && r.init_ok && r.within_cap() && r.paced_identity));
        assert!(em_matched(&a) && orders_valid(&a));
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        let cols = csv_header(6).split(',').count();
        assert!(csv.lines().all(|l| l.split(',').count() == cols));
        let id = compare(
            &a,
            50,
            6,
            Slice::All,
            Orders::Heldout,
            Setting::Err,
            Setting::Err,
        );
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
        assert!(run_grid_threads(None, &d, 6, &[50], &[], 2, 1).is_err());
        // Tampering with an order breaks G1.
        let mut t = a.clone();
        t[0].order.swap(0, 1);
        assert!(!orders_valid(&t));
        assert!(!gates(&t).0);
    }

    #[test]
    fn equivalence_positive_and_growth_rules() {
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
        assert!(positive(&mk(0.001, 0.02)));
        assert!(!positive(&mk(-0.001, 0.02)));
        let (g, h) = readings(&[], &FLIPS);
        assert!(!g.0);
        assert_eq!(h, [None; N_HYP]);
    }

    #[test]
    fn census_counts_writes_without_accuracy() {
        let d = small();
        let (facts, lines) = census(None, &d, 6, 50, &[1, 2]).unwrap();
        assert_eq!(facts.len(), 2);
        assert!(facts
            .iter()
            .all(|f| f.2.len() == 6 && f.3.len() == 6 && f.4.len() == 6));
        assert_eq!(lines.len(), 2 * SETTINGS.len());
        for (_, s, pw, cw, tw, _) in &lines {
            assert_eq!(pw + cw, *tw);
            if !s.cued() {
                assert_eq!(*cw, 0);
            }
        }
    }
}

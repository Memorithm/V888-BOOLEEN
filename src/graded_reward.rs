//! B888-4: a **graded reward ladder** between the scalar "all predicted
//! spikes right" bit of [`crate::eligibility_trace`] and per-target reward,
//! all consumed by the unchanged #36 eligibility-trace + reward-prediction
//! error machinery, plus an **episode-stamped tag** arm for delayed scalars.
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of an update rule on a Boolean graph; nothing here is
//! a claim about biology, dopamine, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/GRADED_REWARD.md`
//! (committed before any run). Only weights of existing edges change.
use crate::connectome::Connectome;
use crate::eligibility_trace::{
    self as et, TraceArm, TraceCost, BASELINE_INIT, BASELINE_ONE, BASELINE_RATE, CREDIT_CLAMP,
    STEP_THRESHOLD, TAG, TRACE_MAX,
};
use crate::generalization_bench::{
    self as gb, build_task, encode, episode_order, evaluate, Confusion, GenSpec, GraphShape,
    Spread, Task, Verdict,
};
use crate::plasticity::PlasticCircuit;
use crate::reward_limits as rl;
use std::collections::VecDeque;
use std::io::{self, ErrorKind};

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Granularity of the reward signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grain {
    /// One bit: every firing target right (the #36 scalar).
    Bit,
    /// One graded value: fraction of firing targets right.
    Fraction,
    /// One bit per group (target index mod k) with a firing target.
    Groups(u32),
    /// One bit per firing target.
    PerTarget,
}

impl Grain {
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::Bit => "bit".to_string(),
            Self::Fraction => "fraction".to_string(),
            Self::Groups(k) => format!("group-{k}"),
            Self::PerTarget => "per-target".to_string(),
        }
    }

    fn group_of(self, target: usize) -> usize {
        match self {
            Self::Bit | Self::Fraction => 0,
            Self::Groups(k) => target % (k.max(1) as usize),
            Self::PerTarget => target,
        }
    }
}

/// Pre-registered ladder (delay 0), ordered by refinement.
pub const LADDER: [Grain; 6] = [
    Grain::Bit,
    Grain::Fraction,
    Grain::Groups(2),
    Grain::Groups(4),
    Grain::Groups(8),
    Grain::PerTarget,
];

/// Partition-refinement chain of H1.
pub const CHAIN: [Grain; 5] = [
    Grain::Bit,
    Grain::Groups(2),
    Grain::Groups(4),
    Grain::Groups(8),
    Grain::PerTarget,
];

/// Pre-registered delays of the stamped arm.
pub const STAMP_DELAYS: [u32; 2] = [1, 4];

/// Reward signals of one episode: (group, value in 0..=1024) plus nominal
/// bits.
type Signals = Vec<(usize, i32)>;

fn signals(grain: Grain, firing: &[(usize, bool)]) -> (Signals, f64) {
    if firing.is_empty() {
        return (Vec::new(), 0.0);
    }
    match grain {
        Grain::Bit => {
            let all = firing.iter().all(|&(_, ok)| ok);
            (vec![(0, if all { BASELINE_ONE } else { 0 })], 1.0)
        }
        Grain::Fraction => {
            let right = firing.iter().filter(|&&(_, ok)| ok).count() as i32;
            let n = firing.len() as i32;
            (vec![(0, BASELINE_ONE * right / n)], f64::from(n + 1).log2())
        }
        Grain::Groups(_) | Grain::PerTarget => {
            let mut out: Vec<(usize, bool)> = Vec::new();
            for &(t, ok) in firing {
                let g = grain.group_of(t);
                if let Some(slot) = out.iter_mut().find(|(h, _)| *h == g) {
                    slot.1 &= ok;
                } else {
                    out.push((g, ok));
                }
            }
            out.sort_unstable_by_key(|&(g, _)| g);
            let bits = out.len() as f64;
            (
                out.into_iter()
                    .map(|(g, ok)| (g, if ok { BASELINE_ONE } else { 0 }))
                    .collect(),
                bits,
            )
        }
    }
}

/// Instrumentation of one graded phase.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GradedCost {
    pub trace: TraceCost,
    /// Episodes with at least one signal.
    pub episodes_with_signal: u64,
    /// Individual (group, value) signals emitted / consumed.
    pub signals_emitted: u64,
    pub signals_consumed: u64,
    /// Nominal reward bits emitted (`Σ log2(levels)`).
    pub bits_emitted: f64,
}

/// Phase-local state: trace + credit per edge, one baseline per group.
struct GradedState {
    trace: Vec<u8>,
    credit: Vec<i32>,
    baseline: Vec<i32>,
    edge_group: Vec<usize>,
}

impl GradedState {
    /// Apply signals with eligibilities `elig(edge)`; returns (attempted,
    /// up, down).
    fn consume(
        &mut self,
        circuit: &mut PlasticCircuit,
        sigs: &[(usize, i32)],
        elig: &dyn Fn(usize, &[u8]) -> i32,
        allow: &mut dyn FnMut(usize) -> bool,
    ) -> io::Result<(u64, u64, u64)> {
        let mut rpe: Vec<Option<i32>> = vec![None; self.baseline.len()];
        for &(g, v) in sigs {
            let b = self.baseline[g];
            rpe[g] = Some(v - b);
            self.baseline[g] = b + (v - b) / BASELINE_RATE;
        }
        let (mut attempted, mut up, mut down) = (0, 0, 0);
        for edge in 0..self.credit.len() {
            let Some(r) = rpe[self.edge_group[edge]] else {
                continue;
            };
            let e = elig(edge, &self.trace);
            if e == 0 {
                continue;
            }
            let c = (self.credit[edge] + r * e / i32::from(TAG)).clamp(-CREDIT_CLAMP, CREDIT_CLAMP);
            self.credit[edge] = c;
            if c.abs() >= STEP_THRESHOLD {
                let reinforce = c > 0;
                self.credit[edge] = c - c.signum() * STEP_THRESHOLD;
                attempted += 1;
                if !allow(edge) {
                    continue;
                }
                if circuit.apply_edge_step(edge, reinforce)? {
                    if reinforce {
                        up += 1;
                    } else {
                        down += 1;
                    }
                }
            }
        }
        Ok((attempted, up, down))
    }
}

/// One phase of a graded arm. `stamped = false`: #36 decaying trace, signal
/// of episode k − d consumed with the current traces. `stamped = true`:
/// signal of episode k − d consumed with full eligibility on exactly the
/// edges tagged in episode k − d.
#[allow(clippy::too_many_arguments)]
pub fn train_graded_phase(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    patterns: &[Vec<usize>],
    valence: &[bool],
    budget: (u32, u64),
    grain: Grain,
    delay: u32,
    stamped: bool,
) -> io::Result<GradedCost> {
    let (rounds, order_seed) = budget;
    let items: Vec<(&[usize], &[bool])> = episode_order(patterns.len(), rounds, order_seed)
        .into_iter()
        .map(|idx| (patterns[idx].as_slice(), valence))
        .collect();
    train_graded_stream(circuit, graph, &items, grain, delay, stamped, &mut |_| true)
}

/// Like [`train_graded_phase`] but over an explicit episode stream of
/// (pattern, valence) items, with `allow(edge)` asked before each attempted
/// step (threshold crossing). A refused step consumes its credit and does
/// not change the weight. With `allow = |_| true` and the phase's own order
/// this is exactly [`train_graded_phase`].
pub(crate) fn train_graded_stream(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    items: &[(&[usize], &[bool])],
    grain: Grain,
    delay: u32,
    stamped: bool,
    allow: &mut dyn FnMut(usize) -> bool,
) -> io::Result<GradedCost> {
    if delay > 1024 || circuit.weights().len() != graph.edges() {
        return Err(bad("invalid graded phase"));
    }
    if let Grain::Groups(0) = grain {
        return Err(bad("group count must be positive"));
    }
    let n = graph.ids.len();
    let groups = match grain {
        Grain::Bit | Grain::Fraction => 1,
        Grain::Groups(k) => k as usize,
        Grain::PerTarget => n,
    };
    let edge_group: Vec<usize> = graph
        .targets
        .iter()
        .map(|&t| grain.group_of(t as usize))
        .collect();
    if edge_group.iter().any(|&g| g >= groups) {
        return Err(bad("bad group"));
    }
    let mut st = GradedState {
        trace: vec![0; graph.edges()],
        credit: vec![0; graph.edges()],
        baseline: vec![BASELINE_INIT; groups],
        edge_group,
    };
    let mut cost = GradedCost::default();
    let mut past: VecDeque<(Signals, Vec<usize>)> = VecDeque::new();
    for &(pattern, valence) in items {
        cost.trace.episodes += 1;
        for e in &mut st.trace {
            *e = e.saturating_sub(1);
        }
        let (input, reached, teacher) = encode(graph, pattern, valence)?;
        let prediction = circuit.predict(graph, &input, 1)?;
        let mut firing = Vec::new();
        for &t in &reached {
            if prediction.get(t).map_err(|_| bad("bad target"))? {
                firing.push((t, teacher.get(t).map_err(|_| bad("bad target"))?));
            }
        }
        let mut tagged = Vec::new();
        for source in input.active_indices() {
            for edge in graph.offsets[source]..graph.offsets[source + 1] {
                let edge = edge as usize;
                let target = graph.targets[edge] as usize;
                if prediction.get(target).map_err(|_| bad("bad target"))? {
                    st.trace[edge] = TAG.min(TRACE_MAX);
                    tagged.push(edge);
                }
            }
        }
        let (sigs, bits) = signals(grain, &firing);
        if !sigs.is_empty() {
            cost.episodes_with_signal += 1;
            cost.trace.scalars_emitted += 1;
            cost.signals_emitted += sigs.len() as u64;
            cost.bits_emitted += bits;
        }
        past.push_back((sigs, tagged));
        let due = if past.len() > delay as usize {
            past.pop_front()
        } else {
            None
        };
        if let Some((sigs, tagged)) = due {
            if sigs.is_empty() {
                continue;
            }
            cost.trace.scalars_consumed += 1;
            cost.signals_consumed += sigs.len() as u64;
            let (a, u, d) = if stamped {
                let mut mark = vec![false; graph.edges()];
                for &e in &tagged {
                    mark[e] = true;
                }
                st.consume(
                    circuit,
                    &sigs,
                    &|edge, _| {
                        if mark[edge] {
                            i32::from(TAG)
                        } else {
                            0
                        }
                    },
                    allow,
                )?
            } else {
                st.consume(circuit, &sigs, &|edge, tr| i32::from(tr[edge]), allow)?
            };
            cost.trace.update_calls += a;
            cost.trace.steps_up += u;
            cost.trace.steps_down += d;
            cost.trace.weights_changed += u + d;
        }
    }
    Ok(cost)
}

/// Arms of the graded study.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradedArm {
    /// A reference arm of #36 (control, #34 reward, #36 trace, clean ref).
    Reference(TraceArm),
    /// Ladder arm: graded signal through the #36 trace.
    Ladder(Grain),
    /// Episode-stamped tags, bit reward.
    Stamped,
}

impl GradedArm {
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::Reference(a) => a.as_label().to_string(),
            Self::Ladder(g) => format!("trace-{}", g.label()),
            Self::Stamped => "stamped-bit".to_string(),
        }
    }
}

/// One (arm, delay, seed) measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct GradedRow {
    pub arm: GradedArm,
    pub delay: u32,
    pub seed: u64,
    pub a_train_after_a: Confusion,
    pub a_test_after_a: Confusion,
    pub a_test_after_b: Confusion,
    pub b_test_after_b: Confusion,
    pub cost_a: GradedCost,
    pub cost_b: GradedCost,
    pub topology_unchanged: bool,
}

impl GradedRow {
    #[must_use]
    pub fn forgetting(&self) -> f64 {
        self.a_test_after_a.balanced() - self.a_test_after_b.balanced()
    }

    /// Reward signals emitted per episode (A+B).
    #[must_use]
    pub fn signals_per_episode(&self) -> f64 {
        let ep = (self.cost_a.trace.episodes + self.cost_b.trace.episodes).max(1) as f64;
        (self.cost_a.signals_emitted + self.cost_b.signals_emitted) as f64 / ep
    }

    /// Nominal reward bits emitted per episode (A+B).
    #[must_use]
    pub fn bits_per_episode(&self) -> f64 {
        let ep = (self.cost_a.trace.episodes + self.cost_b.trace.episodes).max(1) as f64;
        (self.cost_a.bits_emitted + self.cost_b.bits_emitted) / ep
    }
}

fn wrap(c: TraceCost) -> GradedCost {
    GradedCost {
        trace: c,
        episodes_with_signal: c.scalars_emitted,
        signals_emitted: c.scalars_emitted,
        signals_consumed: c.scalars_consumed,
        bits_emitted: c.scalars_emitted as f64,
    }
}

/// Run A then B for one arm (budgets and orders as in
/// `eligibility_trace::run_trace_arm`).
pub fn run_graded_arm(
    graph: &Connectome,
    task: &Task,
    spec: &GenSpec,
    delay: u32,
    arm: GradedArm,
    seed: u64,
) -> io::Result<GradedRow> {
    if let GradedArm::Reference(a) = arm {
        let r = et::run_trace_arm(graph, task, spec, delay, a, seed)?;
        let reward_like = matches!(a, TraceArm::Reward | TraceArm::Trace);
        let c = |x: TraceCost| {
            if reward_like {
                wrap(x)
            } else {
                GradedCost {
                    trace: x,
                    ..GradedCost::default()
                }
            }
        };
        return Ok(GradedRow {
            arm,
            delay,
            seed,
            a_train_after_a: r.a_train_after_a,
            a_test_after_a: r.a_test_after_a,
            a_test_after_b: r.a_test_after_b,
            b_test_after_b: r.b_test_after_b,
            cost_a: c(r.cost_a),
            cost_b: c(r.cost_b),
            topology_unchanged: r.topology_unchanged,
        });
    }
    let (grain, stamped) = match arm {
        GradedArm::Ladder(g) => (g, false),
        GradedArm::Stamped => (Grain::Bit, true),
        GradedArm::Reference(_) => return Err(bad("unreachable")),
    };
    let original = graph.clone();
    let mut circuit = PlasticCircuit::new(graph, spec.init_weight)?;
    let fingerprint = circuit.topology_fingerprint();
    let cost_a = train_graded_phase(
        &mut circuit,
        graph,
        &task.a_train,
        &task.valence_a,
        (spec.rounds_a, seed),
        grain,
        delay,
        stamped,
    )?;
    let a_train_after_a = evaluate(&circuit, graph, &task.a_train, &task.valence_a)?;
    let a_test_after_a = evaluate(&circuit, graph, &task.test, &task.valence_a)?;
    let cost_b = train_graded_phase(
        &mut circuit,
        graph,
        &task.b_train,
        &task.valence_b,
        (spec.rounds_b, seed ^ 0xB0B0_B0B0),
        grain,
        delay,
        stamped,
    )?;
    let a_test_after_b = evaluate(&circuit, graph, &task.test, &task.valence_a)?;
    let b_test_after_b = evaluate(&circuit, graph, &task.test, &task.valence_b)?;
    Ok(GradedRow {
        arm,
        delay,
        seed,
        a_train_after_a,
        a_test_after_a,
        a_test_after_b,
        b_test_after_b,
        cost_a,
        cost_b,
        topology_unchanged: *graph == original
            && circuit.topology_fingerprint() == fingerprint
            && circuit.weights().len() == graph.edges(),
    })
}

/// The pre-registered (arm, delay) list per seed (13 entries).
#[must_use]
pub fn preregistered_arms() -> Vec<(GradedArm, u32)> {
    let mut out = vec![
        (GradedArm::Reference(TraceArm::Control), 0),
        (GradedArm::Reference(TraceArm::RewardCleanReference), 0),
        (GradedArm::Reference(TraceArm::Reward), 0),
    ];
    for g in LADDER {
        out.push((GradedArm::Ladder(g), 0));
    }
    for d in STAMP_DELAYS {
        out.push((GradedArm::Reference(TraceArm::Trace), d));
        out.push((GradedArm::Stamped, d));
    }
    out
}

/// Grid: seed x pre-registered arms. Graph and task per seed as in #34/#36.
pub fn run_graded(
    graph: Option<&Connectome>,
    shape: &GraphShape,
    spec: &GenSpec,
    seeds: &[u64],
) -> io::Result<Vec<GradedRow>> {
    if seeds.is_empty() {
        return Err(bad("need at least one seed"));
    }
    let mut rows = Vec::new();
    for &seed in seeds {
        let g = rl::graph_for(graph, shape, seed)?;
        let task = build_task(&g, spec, seed)?;
        for (arm, d) in preregistered_arms() {
            rows.push(run_graded_arm(&g, &task, spec, d, arm, seed)?);
        }
    }
    Ok(rows)
}

pub const CSV_HEADER: &str = "schema,graph,arm,delay,seed,a_train_bal_after_a,a_test_bal_after_a,a_test_bal_after_b,b_test_bal_after_b,forgetting,signals_per_episode,bits_per_episode,a_episodes,a_episodes_with_signal,a_signals_emitted,a_signals_consumed,a_update_calls,a_weights_changed,a_steps_up,a_steps_down,b_episodes,b_episodes_with_signal,b_signals_emitted,b_signals_consumed,b_update_calls,b_weights_changed,b_steps_up,b_steps_down,topology_unchanged";

fn cost_cols(c: &GradedCost) -> String {
    format!(
        "{},{},{},{},{},{},{},{}",
        c.trace.episodes,
        c.episodes_with_signal,
        c.signals_emitted,
        c.signals_consumed,
        c.trace.update_calls,
        c.trace.weights_changed,
        c.trace.steps_up,
        c.trace.steps_down
    )
}

/// Deterministic CSV (no timing column).
#[must_use]
pub fn to_csv(rows: &[GradedRow], graph_label: &str) -> String {
    let mut out = String::from(CSV_HEADER);
    out.push('\n');
    for r in rows {
        out.push_str(&format!(
            "b888-4-graded-reward-v1,{graph_label},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{}\n",
            r.arm.label(),
            r.delay,
            r.seed,
            r.a_train_after_a.balanced(),
            r.a_test_after_a.balanced(),
            r.a_test_after_b.balanced(),
            r.b_test_after_b.balanced(),
            r.forgetting(),
            r.signals_per_episode(),
            r.bits_per_episode(),
            cost_cols(&r.cost_a),
            cost_cols(&r.cost_b),
            r.topology_unchanged,
        ));
    }
    out
}

/// Paired comparison of (x, dx) against (y, dy).
#[derive(Clone, Debug, PartialEq)]
pub struct Comparison {
    pub x: (GradedArm, u32),
    pub y: (GradedArm, u32),
    pub n: usize,
    pub x_a_test: Spread,
    pub y_a_test: Spread,
    pub gen: gb::Paired,
    /// y forgetting - x forgetting (+ = x forgets less).
    pub forget: gb::Paired,
    /// x B test - y B test.
    pub b: Spread,
    pub gen_verdict: Verdict,
    pub forget_verdict: Verdict,
}

#[must_use]
pub fn compare(rows: &[GradedRow], x: (GradedArm, u32), y: (GradedArm, u32)) -> Comparison {
    let pick = |k: (GradedArm, u32), seed: u64| {
        rows.iter()
            .find(|r| r.arm == k.0 && r.delay == k.1 && r.seed == seed)
    };
    let (mut xa, mut ya, mut dg, mut df, mut db) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for rx in rows.iter().filter(|r| r.arm == x.0 && r.delay == x.1) {
        if let Some(ry) = pick(y, rx.seed) {
            xa.push(rx.a_test_after_a.balanced());
            ya.push(ry.a_test_after_a.balanced());
            dg.push(rx.a_test_after_a.balanced() - ry.a_test_after_a.balanced());
            df.push(ry.forgetting() - rx.forgetting());
            db.push(rx.b_test_after_b.balanced() - ry.b_test_after_b.balanced());
        }
    }
    let n = dg.len();
    let gen = gb::paired(&dg);
    let forget = gb::paired(&df);
    let b = gb::spread(&db);
    Comparison {
        x,
        y,
        n,
        x_a_test: gb::spread(&xa),
        y_a_test: gb::spread(&ya),
        gen_verdict: gb::verdict(&gen, n, true),
        forget_verdict: gb::verdict(&forget, n, b.mean > -gb::MARGIN),
        gen,
        forget,
        b,
    }
}

const CONTROL: (GradedArm, u32) = (GradedArm::Reference(TraceArm::Control), 0);

/// Pre-registered comparisons.
#[must_use]
pub fn preregistered_comparisons(rows: &[GradedRow]) -> Vec<Comparison> {
    let bit = (GradedArm::Ladder(Grain::Bit), 0);
    let mut out = Vec::new();
    for g in LADDER.iter().skip(1) {
        out.push(compare(rows, (GradedArm::Ladder(*g), 0), bit));
    }
    for g in LADDER {
        out.push(compare(rows, (GradedArm::Ladder(g), 0), CONTROL));
    }
    out.push(compare(
        rows,
        (GradedArm::Ladder(Grain::PerTarget), 0),
        (GradedArm::Reference(TraceArm::RewardCleanReference), 0),
    ));
    for d in STAMP_DELAYS {
        out.push(compare(
            rows,
            (GradedArm::Stamped, d),
            (GradedArm::Reference(TraceArm::Trace), d),
        ));
        out.push(compare(rows, (GradedArm::Stamped, d), CONTROL));
    }
    out
}

/// Mean A-test of one (arm, delay).
#[must_use]
pub fn mean_a_test(rows: &[GradedRow], arm: GradedArm, delay: u32) -> f64 {
    let v: Vec<f64> = rows
        .iter()
        .filter(|r| r.arm == arm && r.delay == delay)
        .map(|r| r.a_test_after_a.balanced())
        .collect();
    gb::spread(&v).mean
}

/// Pre-stated hypothesis readings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H1 {
    Supported,
    PartiallySupported,
    NotSupported,
}

#[must_use]
pub fn read_h1(rows: &[GradedRow]) -> H1 {
    let c = compare(
        rows,
        (GradedArm::Ladder(Grain::PerTarget), 0),
        (GradedArm::Ladder(Grain::Bit), 0),
    );
    let means: Vec<f64> = CHAIN
        .iter()
        .map(|&g| mean_a_test(rows, GradedArm::Ladder(g), 0))
        .collect();
    let monotone = means.windows(2).all(|w| w[1] >= w[0]);
    match (c.gen_verdict == Verdict::Beats, monotone) {
        (true, true) => H1::Supported,
        (true, false) => H1::PartiallySupported,
        _ => H1::NotSupported,
    }
}

/// "Reaches the control" rule of #36.
#[must_use]
pub fn reaches_control(c: &Comparison) -> bool {
    c.y == CONTROL && c.gen.delta.mean > -gb::MARGIN
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn shape() -> GraphShape {
        GraphShape {
            sources: 40,
            targets: 12,
            fan_in: 6,
        }
    }

    fn spec() -> GenSpec {
        GenSpec {
            active: 3,
            train_patterns: 16,
            test_patterns: 16,
            rounds_a: 4,
            rounds_b: 4,
            positive_pct: 50,
            flip_pct: 50,
            init_weight: 1,
        }
    }

    fn setup(seed: u64) -> (Connectome, Task) {
        let g = rl::graph_for(None, &shape(), seed).unwrap();
        let t = build_task(&g, &spec(), seed).unwrap();
        (g, t)
    }

    #[test]
    fn signals_follow_preregistered_definitions() {
        let f = [(40, true), (41, false), (42, true), (44, true)];
        assert_eq!(signals(Grain::Bit, &f).0, vec![(0, 0)]);
        let (s, b) = signals(Grain::Fraction, &f);
        assert_eq!(s, vec![(0, 768)]);
        assert!((b - 5f64.log2()).abs() < 1e-12);
        // mod 2: {40,42,44} all right, {41} wrong
        assert_eq!(signals(Grain::Groups(2), &f).0, vec![(0, 1024), (1, 0)]);
        // mod 4: 40,44 -> 0 ; 41 -> 1 ; 42 -> 2
        let (s, b) = signals(Grain::Groups(4), &f);
        assert_eq!(s, vec![(0, 1024), (1, 0), (2, 1024)]);
        assert!((b - 3.0).abs() < 1e-12);
        assert_eq!(signals(Grain::PerTarget, &f).0.len(), 4);
        assert_eq!(signals(Grain::Bit, &[(1, true)]).0, vec![(0, 1024)]);
        assert!(signals(Grain::PerTarget, &[]).0.is_empty());
    }

    #[test]
    fn bit_ladder_arm_is_bit_identical_to_the_36_trace_arm() {
        for seed in [2, 3] {
            let (g, task) = setup(seed);
            for d in [0, 1, 4] {
                let ours = if d == 0 {
                    run_graded_arm(&g, &task, &spec(), 0, GradedArm::Ladder(Grain::Bit), seed)
                } else {
                    // non-stamped bit at delay d through the graded code path
                    let mut c = PlasticCircuit::new(&g, 1).unwrap();
                    let cost = train_graded_phase(
                        &mut c,
                        &g,
                        &task.a_train,
                        &task.valence_a,
                        (4, seed),
                        Grain::Bit,
                        d,
                        false,
                    )
                    .unwrap();
                    let mut c2 = PlasticCircuit::new(&g, 1).unwrap();
                    let theirs = et::train_trace_phase(
                        &mut c2,
                        &g,
                        &task.a_train,
                        &task.valence_a,
                        (4, seed),
                        d,
                        false,
                    )
                    .unwrap();
                    assert_eq!(cost.trace, theirs);
                    assert_eq!(c.weights(), c2.weights());
                    continue;
                }
                .unwrap();
                let theirs =
                    et::run_trace_arm(&g, &task, &spec(), 0, TraceArm::Trace, seed).unwrap();
                assert_eq!(ours.a_test_after_a, theirs.a_test_after_a);
                assert_eq!(ours.a_test_after_b, theirs.a_test_after_b);
                assert_eq!(ours.b_test_after_b, theirs.b_test_after_b);
                assert_eq!(ours.cost_a.trace, theirs.cost_a);
                assert_eq!(ours.cost_b.trace, theirs.cost_b);
            }
        }
    }

    #[test]
    fn group_signal_only_moves_edges_of_its_group() {
        let (g, _) = setup(1);
        let edge_group: Vec<usize> = g.targets.iter().map(|&t| t as usize % 2).collect();
        let mut st = GradedState {
            trace: vec![TAG; g.edges()],
            credit: vec![0; g.edges()],
            baseline: vec![BASELINE_INIT; 2],
            edge_group: edge_group.clone(),
        };
        let mut c = PlasticCircuit::new(&g, 1).unwrap();
        let (a, _, _) = st
            .consume(&mut c, &[(0, 1024)], &|e, tr| i32::from(tr[e]), &mut |_| {
                true
            })
            .unwrap();
        assert_eq!(a, 0);
        for (e, &grp) in edge_group.iter().enumerate() {
            assert_eq!(st.credit[e], if grp == 0 { 512 } else { 0 });
        }
        assert_eq!(st.baseline, vec![544, 512]);
    }

    #[test]
    fn stamped_credit_lands_on_the_stamped_episode_only() {
        let (g, task) = setup(4);
        for d in STAMP_DELAYS {
            let r = run_graded_arm(&g, &task, &spec(), d, GradedArm::Stamped, 4).unwrap();
            assert!(r.topology_unchanged);
            assert_eq!(
                r.cost_a.trace.weights_changed,
                r.cost_a.trace.steps_up + r.cost_a.trace.steps_down
            );
            assert!(r.cost_a.signals_emitted - r.cost_a.signals_consumed <= u64::from(d));
        }
        // delay longer than a phase: nothing consumed, nothing moves
        let mut c = PlasticCircuit::new(&g, 1).unwrap();
        let cost = train_graded_phase(
            &mut c,
            &g,
            &task.a_train,
            &task.valence_a,
            (4, 4),
            Grain::Bit,
            1000,
            true,
        )
        .unwrap();
        assert_eq!((cost.signals_consumed, cost.trace.weights_changed), (0, 0));
        assert!(train_graded_phase(
            &mut c,
            &g,
            &task.a_train,
            &task.valence_a,
            (4, 4),
            Grain::Groups(0),
            0,
            false
        )
        .is_err());
    }

    #[test]
    fn grid_is_deterministic_and_csv_complete() {
        let a = run_graded(None, &shape(), &spec(), &[1, 2]).unwrap();
        let b = run_graded(None, &shape(), &spec(), &[1, 2]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 2 * 13);
        assert!(a.iter().all(|r| r.topology_unchanged));
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        assert!(csv.contains(",trace-group-4,0,"));
        assert!(csv.contains(",stamped-bit,4,"));
        let cmp = preregistered_comparisons(&a);
        assert_eq!(cmp.len(), 5 + 6 + 1 + 4);
        assert!(cmp.iter().all(|c| c.n == 2));
        let self_cmp = compare(&a, CONTROL, CONTROL);
        assert!(reaches_control(&self_cmp));
        let _ = read_h1(&a);
        // per-target emits at least as many signals as any coarser grain
        let spe = |g: Grain| {
            a.iter()
                .filter(|r| r.arm == GradedArm::Ladder(g) && r.seed == 1)
                .map(GradedRow::signals_per_episode)
                .sum::<f64>()
        };
        assert!(spe(Grain::PerTarget) >= spe(Grain::Groups(8)));
        assert!(run_graded(None, &shape(), &spec(), &[]).is_err());
    }

    #[test]
    fn checkpoint_between_a_and_b_resumes_bit_exactly() -> io::Result<()> {
        let (g, task) = setup(7);
        let mut cont = PlasticCircuit::new(&g, 1)?;
        let a = (4, 7);
        train_graded_phase(
            &mut cont,
            &g,
            &task.a_train,
            &task.valence_a,
            a,
            Grain::Groups(4),
            0,
            false,
        )?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| bad("clock"))?
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("v888-gr-{}-{nonce}.v888pls", std::process::id()));
        cont.save(&path)?;
        let mut resumed = PlasticCircuit::load(&path, &g)?;
        std::fs::remove_file(&path)?;
        let b = (4, 7 ^ 0xB0B0_B0B0);
        let c1 = train_graded_phase(
            &mut cont,
            &g,
            &task.b_train,
            &task.valence_b,
            b,
            Grain::PerTarget,
            4,
            true,
        )?;
        let c2 = train_graded_phase(
            &mut resumed,
            &g,
            &task.b_train,
            &task.valence_b,
            b,
            Grain::PerTarget,
            4,
            true,
        )?;
        assert_eq!(c1, c2);
        assert_eq!(cont.weights(), resumed.weights());
        Ok(())
    }
}

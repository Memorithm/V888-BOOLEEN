//! B888-4: a concrete remedy for the **scalar credit-assignment** failure of
//! the reward-modulated strategy found in [`crate::reward_limits`]: a bounded
//! eligibility trace per original synapse, consumed by a scalar (possibly
//! delayed) reward through a reward-prediction error and a bounded credit
//! accumulator, plus an optional reactivation of silent targets.
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of an update rule on a Boolean graph; nothing here is
//! a claim about biology, dopamine, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/ELIGIBILITY_TRACES.md`
//! (committed before any run). Only weights of existing edges change.
use crate::connectome::Connectome;
use crate::generalization_bench::{
    self as gb, build_task, encode, episode_order, evaluate, Confusion, GenSpec, GraphShape,
    Spread, Task, Verdict,
};
use crate::plasticity::PlasticCircuit;
use crate::reward_limits::{self as rl, Credit, RewardCondition};
use std::collections::VecDeque;
use std::io::{self, ErrorKind};

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Pre-registered constants (not tuned on results).
pub const TAG: u8 = 8;
pub const TRACE_MAX: u8 = 15;
pub const BASELINE_ONE: i32 = 1024;
pub const BASELINE_INIT: i32 = 512;
pub const BASELINE_RATE: i32 = 16;
pub const STEP_THRESHOLD: i32 = 1024;
pub const CREDIT_CLAMP: i32 = 2048;
pub const REACTIVATE_AFTER: u8 = 8;

/// Arms of the eligibility-trace study.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceArm {
    /// Clean supervised control (credit modes do not apply to a teacher).
    Control,
    /// The #34 reward-modulated arm under the same scalar credit condition.
    Reward,
    /// Eligibility trace + reward-prediction error.
    Trace,
    /// `Trace` + reactivation of silent targets.
    TraceReact,
    /// Reference: #34 reward arm with clean per-target reward.
    RewardCleanReference,
}

impl TraceArm {
    #[must_use]
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Control => "supervised-matched",
            Self::Reward => "reward-modulated",
            Self::Trace => "trace",
            Self::TraceReact => "trace-react",
            Self::RewardCleanReference => "reward-clean-per-target-ref",
        }
    }
}

/// Pre-registered scalar delays: 0 (scalar per episode), 1 (previous
/// episode), 4 (delayed).
pub const DEFAULT_DELAYS: [u32; 3] = [0, 1, 4];

/// The #34 credit mode carrying the same scalar with delay `d`.
#[must_use]
pub fn credit_for_delay(d: u32) -> Credit {
    if d == 0 {
        Credit::Global
    } else {
        Credit::GlobalDelayedNoTrace(d)
    }
}

#[must_use]
pub fn delay_label(d: u32) -> String {
    if d == 0 {
        "global".to_string()
    } else {
        format!("global-delay-{d}")
    }
}

/// Instrumentation of one phase (all arms map onto it).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TraceCost {
    pub episodes: u64,
    /// Scalars produced (episodes with at least one predicted spike).
    pub scalars_emitted: u64,
    /// Scalars actually applied (excludes those still in flight at phase end).
    pub scalars_consumed: u64,
    /// Update calls (control / #34 reward arm) or weight steps attempted.
    pub update_calls: u64,
    pub weights_changed: u64,
    pub steps_up: u64,
    pub steps_down: u64,
    pub reactivations: u64,
}

/// Phase-local learning state of the trace arms (empty at phase start).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceState {
    pub trace: Vec<u8>,
    pub credit: Vec<i32>,
    pub baseline: i32,
    pub silent: Vec<u8>,
}

impl TraceState {
    #[must_use]
    pub fn new(edges: usize, neurons: usize) -> Self {
        Self {
            trace: vec![0; edges],
            credit: vec![0; edges],
            baseline: BASELINE_INIT,
            silent: vec![0; neurons],
        }
    }

    /// Consume one scalar `r` with the current traces. Returns
    /// (steps attempted, changed up, changed down).
    pub fn consume(
        &mut self,
        circuit: &mut PlasticCircuit,
        r: bool,
    ) -> io::Result<(u64, u64, u64)> {
        let target = if r { BASELINE_ONE } else { 0 };
        let rpe = target - self.baseline;
        self.baseline += (target - self.baseline) / BASELINE_RATE;
        let (mut attempted, mut up, mut down) = (0, 0, 0);
        for edge in 0..self.trace.len() {
            let e = i32::from(self.trace[edge]);
            if e == 0 {
                continue;
            }
            let c =
                (self.credit[edge] + rpe * e / i32::from(TAG)).clamp(-CREDIT_CLAMP, CREDIT_CLAMP);
            self.credit[edge] = c;
            if c.abs() >= STEP_THRESHOLD {
                let reinforce = c > 0;
                self.credit[edge] = c - c.signum() * STEP_THRESHOLD;
                attempted += 1;
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

/// Trace arm for one phase. `delay` = episodes between the scalar's episode
/// and its consumption; `react` enables reactivation of silent targets.
pub fn train_trace_phase(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    patterns: &[Vec<usize>],
    valence: &[bool],
    budget: (u32, u64),
    delay: u32,
    react: bool,
) -> io::Result<TraceCost> {
    if delay > 1024 || circuit.weights().len() != graph.edges() {
        return Err(bad("invalid trace phase"));
    }
    let (rounds, order_seed) = budget;
    let mut st = TraceState::new(graph.edges(), graph.ids.len());
    let mut cost = TraceCost::default();
    let mut past: VecDeque<Option<bool>> = VecDeque::new();
    for idx in episode_order(patterns.len(), rounds, order_seed) {
        cost.episodes += 1;
        for e in &mut st.trace {
            *e = e.saturating_sub(1);
        }
        let (input, reached, teacher) = encode(graph, &patterns[idx], valence)?;
        let prediction = circuit.predict(graph, &input, 1)?;
        let mut fired_any = false;
        let mut all_right = true;
        for &t in &reached {
            if prediction.get(t).map_err(|_| bad("bad target"))? {
                fired_any = true;
                all_right &= teacher.get(t).map_err(|_| bad("bad target"))?;
            }
        }
        // Pre/post co-activity tags on existing edges.
        for source in input.active_indices() {
            for edge in graph.offsets[source]..graph.offsets[source + 1] {
                let edge = edge as usize;
                let target = graph.targets[edge] as usize;
                if prediction.get(target).map_err(|_| bad("bad target"))? {
                    st.trace[edge] = TAG.min(TRACE_MAX);
                }
            }
        }
        let scalar = if fired_any {
            cost.scalars_emitted += 1;
            Some(all_right)
        } else {
            None
        };
        past.push_back(scalar);
        let due = if past.len() > delay as usize {
            past.pop_front().flatten()
        } else {
            None
        };
        if let Some(r) = due {
            cost.scalars_consumed += 1;
            let (a, u, d) = st.consume(circuit, r)?;
            cost.update_calls += a;
            cost.steps_up += u;
            cost.steps_down += d;
            cost.weights_changed += u + d;
        }
        if react {
            for &t in &reached {
                if prediction.get(t).map_err(|_| bad("bad target"))? {
                    st.silent[t] = 0;
                    continue;
                }
                st.silent[t] = st.silent[t].saturating_add(1);
                if st.silent[t] < REACTIVATE_AFTER {
                    continue;
                }
                st.silent[t] = 0;
                let mut chosen: Option<usize> = None;
                for source in input.active_indices() {
                    for edge in graph.offsets[source]..graph.offsets[source + 1] {
                        let edge = edge as usize;
                        if graph.targets[edge] as usize == t
                            && circuit.weights()[edge] == 0
                            && chosen.is_none_or(|c| edge < c)
                        {
                            chosen = Some(edge);
                        }
                    }
                }
                if let Some(edge) = chosen {
                    if circuit.apply_edge_step(edge, true)? {
                        cost.reactivations += 1;
                        cost.weights_changed += 1;
                        cost.steps_up += 1;
                    }
                }
            }
        }
    }
    Ok(cost)
}

fn from_rl(c: &rl::RewardCost) -> TraceCost {
    TraceCost {
        episodes: c.phase.episodes,
        scalars_emitted: c.rewards_emitted,
        scalars_consumed: 0,
        update_calls: c.phase.update_calls,
        weights_changed: c.phase.weights_changed,
        steps_up: 0,
        steps_down: 0,
        reactivations: 0,
    }
}

/// One (delay, seed, arm) measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceRow {
    pub delay: u32,
    pub arm: TraceArm,
    pub seed: u64,
    pub a_train_after_a: Confusion,
    pub a_test_after_a: Confusion,
    pub a_test_after_b: Confusion,
    pub b_test_after_b: Confusion,
    pub cost_a: TraceCost,
    pub cost_b: TraceCost,
    pub topology_unchanged: bool,
}

impl TraceRow {
    #[must_use]
    pub fn forgetting(&self) -> f64 {
        self.a_test_after_a.balanced() - self.a_test_after_b.balanced()
    }
}

const PHASE_A: u64 = 0xA;
const PHASE_B: u64 = 0xB;

fn train(
    arm: TraceArm,
    delay: u32,
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    set: (&[Vec<usize>], &[bool]),
    budget: (u32, u64),
    keys: (u64, u64),
) -> io::Result<TraceCost> {
    let (patterns, valence) = set;
    match arm {
        TraceArm::Control => Ok(from_rl(&rl::train_control_phase(
            circuit,
            graph,
            patterns,
            valence,
            budget,
            &RewardCondition::CLEAN,
            keys,
        )?)),
        TraceArm::Reward | TraceArm::RewardCleanReference => {
            let cond = if arm == TraceArm::Reward {
                RewardCondition {
                    credit: credit_for_delay(delay),
                    ..RewardCondition::CLEAN
                }
            } else {
                RewardCondition::CLEAN
            };
            Ok(from_rl(&rl::train_reward_phase(
                circuit, graph, patterns, valence, budget, &cond, keys,
            )?))
        }
        TraceArm::Trace | TraceArm::TraceReact => train_trace_phase(
            circuit,
            graph,
            patterns,
            valence,
            budget,
            delay,
            arm == TraceArm::TraceReact,
        ),
    }
}

/// Run A then B for one arm; phase budgets and orders as in
/// `generalization_bench::run_arm` and `reward_limits::run_limit_arm`.
pub fn run_trace_arm(
    graph: &Connectome,
    task: &Task,
    spec: &GenSpec,
    delay: u32,
    arm: TraceArm,
    seed: u64,
) -> io::Result<TraceRow> {
    let original = graph.clone();
    let mut circuit = PlasticCircuit::new(graph, spec.init_weight)?;
    let fingerprint = circuit.topology_fingerprint();
    let cost_a = train(
        arm,
        delay,
        &mut circuit,
        graph,
        (&task.a_train, &task.valence_a),
        (spec.rounds_a, seed),
        (seed, PHASE_A),
    )?;
    let a_train_after_a = evaluate(&circuit, graph, &task.a_train, &task.valence_a)?;
    let a_test_after_a = evaluate(&circuit, graph, &task.test, &task.valence_a)?;
    let cost_b = train(
        arm,
        delay,
        &mut circuit,
        graph,
        (&task.b_train, &task.valence_b),
        (spec.rounds_b, seed ^ 0xB0B0_B0B0),
        (seed, PHASE_B),
    )?;
    let a_test_after_b = evaluate(&circuit, graph, &task.test, &task.valence_a)?;
    let b_test_after_b = evaluate(&circuit, graph, &task.test, &task.valence_b)?;
    Ok(TraceRow {
        delay,
        arm,
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

/// Grid: seed x (clean reference + delay x {control, reward, trace,
/// trace-react}). Graph and task per seed as in `run_limits`.
pub fn run_traces(
    graph: Option<&Connectome>,
    shape: &GraphShape,
    spec: &GenSpec,
    seeds: &[u64],
    delays: &[u32],
) -> io::Result<Vec<TraceRow>> {
    if seeds.is_empty() || delays.is_empty() {
        return Err(bad("need at least one seed and one delay"));
    }
    let mut rows = Vec::new();
    for &seed in seeds {
        let g = rl::graph_for(graph, shape, seed)?;
        let task = build_task(&g, spec, seed)?;
        rows.push(run_trace_arm(
            &g,
            &task,
            spec,
            0,
            TraceArm::RewardCleanReference,
            seed,
        )?);
        for &d in delays {
            for arm in [
                TraceArm::Control,
                TraceArm::Reward,
                TraceArm::Trace,
                TraceArm::TraceReact,
            ] {
                rows.push(run_trace_arm(&g, &task, spec, d, arm, seed)?);
            }
        }
    }
    Ok(rows)
}

pub const CSV_HEADER: &str = "schema,graph,condition,delay,arm,seed,a_train_bal_after_a,a_test_bal_after_a,a_test_bal_after_b,b_test_bal_after_b,forgetting,a_episodes,a_scalars_emitted,a_scalars_consumed,a_update_calls,a_weights_changed,a_steps_up,a_steps_down,a_reactivations,b_episodes,b_scalars_emitted,b_scalars_consumed,b_update_calls,b_weights_changed,b_steps_up,b_steps_down,b_reactivations,topology_unchanged";

fn cost_cols(c: &TraceCost) -> String {
    format!(
        "{},{},{},{},{},{},{},{}",
        c.episodes,
        c.scalars_emitted,
        c.scalars_consumed,
        c.update_calls,
        c.weights_changed,
        c.steps_up,
        c.steps_down,
        c.reactivations
    )
}

/// Deterministic CSV (no timing column).
#[must_use]
pub fn to_csv(rows: &[TraceRow], graph_label: &str) -> String {
    let mut out = String::from(CSV_HEADER);
    out.push('\n');
    for r in rows {
        let cond = if r.arm == TraceArm::RewardCleanReference {
            "per-target-clean".to_string()
        } else {
            delay_label(r.delay)
        };
        out.push_str(&format!(
            "b888-4-eligibility-traces-v1,{graph_label},{cond},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{}\n",
            r.delay,
            r.arm.as_label(),
            r.seed,
            r.a_train_after_a.balanced(),
            r.a_test_after_a.balanced(),
            r.a_test_after_b.balanced(),
            r.b_test_after_b.balanced(),
            r.forgetting(),
            cost_cols(&r.cost_a),
            cost_cols(&r.cost_b),
            r.topology_unchanged,
        ));
    }
    out
}

/// Paired comparison of arm `x` against arm `y` on one delay.
#[derive(Clone, Debug, PartialEq)]
pub struct Comparison {
    pub delay: u32,
    pub x: TraceArm,
    pub y: TraceArm,
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
pub fn compare(rows: &[TraceRow], delay: u32, x: TraceArm, y: TraceArm) -> Comparison {
    let pick = |arm: TraceArm, seed: u64| {
        rows.iter().find(|r| {
            r.arm == arm
                && r.seed == seed
                && (arm == TraceArm::RewardCleanReference || r.delay == delay)
        })
    };
    let (mut xa, mut ya, mut dg, mut df, mut db) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for rx in rows
        .iter()
        .filter(|r| r.arm == x && (x == TraceArm::RewardCleanReference || r.delay == delay))
    {
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
        delay,
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

/// Pre-registered comparisons per delay.
#[must_use]
pub fn preregistered_comparisons(rows: &[TraceRow], delays: &[u32]) -> Vec<Comparison> {
    let mut out = Vec::new();
    for &d in delays {
        for (x, y) in [
            (TraceArm::Trace, TraceArm::Reward),
            (TraceArm::TraceReact, TraceArm::Reward),
            (TraceArm::TraceReact, TraceArm::Trace),
            (TraceArm::Trace, TraceArm::Control),
            (TraceArm::TraceReact, TraceArm::Control),
            (TraceArm::Reward, TraceArm::Control),
        ] {
            out.push(compare(rows, d, x, y));
        }
    }
    out
}

/// Whether arm `x` "reaches the control" under the pre-stated rule.
#[must_use]
pub fn reaches_control(c: &Comparison) -> bool {
    c.y == TraceArm::Control && c.gen.delta.mean > -gb::MARGIN
}

/// Label of a verdict of arm `x` against arm `y` (the shared `#27` rule).
#[must_use]
pub fn verdict_label(v: Verdict) -> &'static str {
    match v {
        Verdict::Beats => "x_beats_y",
        Verdict::Worse => "x_worse_than_y",
        Verdict::NoClearDifference => "no_clear_difference",
        Verdict::Control => "control",
    }
}

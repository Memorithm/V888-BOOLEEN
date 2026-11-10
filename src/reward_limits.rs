//! B888-4: **limits of the reward-modulated strategy** under degraded reward,
//! at equal episode budget, against the supervised control.
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. The
//! default run uses the deterministic generated graph and the artificial
//! OR-rule task of [`crate::generalization_bench`]. These are engineering
//! stress tests of an update rule on a Boolean graph; nothing here is a claim
//! about biology, dopamine, insect memory or BANC physiology.
//!
//! Protocol (also in `docs/REWARD_LIMITS.md`, fixed before any number was
//! looked at):
//! * Task, graph, pattern sets, episode order, initial weight (default 1) and
//!   metrics are exactly those of `generalization_bench` (A then B, flip 50 %,
//!   balanced accuracy, forgetting = bal(A test after A) - bal(A test after
//!   B)). Evaluation always uses the *clean* labels.
//! * A [`RewardCondition`] degrades the reward channel along one axis:
//!   - `noise_pct`: each reward sign is flipped with this probability;
//!   - `deliver_pct`: an episode delivers a reward at all with this
//!     probability (sparse reward; otherwise no update);
//!   - [`Credit`]: `PerTarget` (reference: one reward per predicted target),
//!     `Global` (one scalar per episode: +1 iff every predicted spike was
//!     right, applied to all eligible synapses), `DelayedTrace(d)` (per-target
//!     reward and its eligibility applied `d` episodes later; pending updates
//!     are flushed at phase end) and `GlobalDelayedNoTrace(d)` (the global
//!     scalar of episode k-d applied to the eligibility of episode k:
//!     misattributed credit; the first d episodes get none).
//! * Matched control: the supervised arm receives the *same* degradation
//!   where it has a meaning: the teacher bit of a reached target is flipped
//!   with the same (seed, phase, episode, target) hash as the reward flip, and
//!   it is skipped on exactly the episodes where the reward is not delivered.
//!   Credit modes do not apply to a teacher; the control stays clean there.
//! * Noise and delivery draws are a pure hash of (seed, phase, episode,
//!   target), so they are identical across arms and fully reproducible.
//! * Pre-stated reading: a condition "hurts" the reward arm if its mean
//!   paired drop in A-test balanced accuracy vs the clean reward arm (same
//!   seed) is >= 0.05; verdicts vs the matched control use the
//!   `generalization_bench` rule (margin 0.05, >= 80 % of seeds).
use crate::connectome::Connectome;
use crate::consolidation_scale::{generate_graph, ScaleSpec};
use crate::generalization_bench::{
    self as gb, build_task, encode, episode_order, evaluate, Confusion, GenSpec, GraphShape,
    PhaseCost, Spread, Task, Verdict,
};
use crate::learning_strategies::{apply_update, LearningStrategy, RewardSignal};
use crate::plasticity::PlasticCircuit;
use crate::BooleanState;
use std::collections::VecDeque;
use std::io::{self, ErrorKind};

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// How reward is attributed to synapses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Credit {
    PerTarget,
    Global,
    DelayedTrace(u32),
    GlobalDelayedNoTrace(u32),
}

/// One degradation of the reward channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RewardCondition {
    pub noise_pct: u32,
    pub deliver_pct: u32,
    pub credit: Credit,
}

impl RewardCondition {
    pub const CLEAN: Self = Self {
        noise_pct: 0,
        deliver_pct: 100,
        credit: Credit::PerTarget,
    };

    fn validate(&self) -> io::Result<()> {
        let delay = match self.credit {
            Credit::DelayedTrace(d) | Credit::GlobalDelayedNoTrace(d) => d,
            _ => 0,
        };
        if self.noise_pct > 100 || self.deliver_pct > 100 || delay > 1024 {
            return Err(bad("invalid reward condition"));
        }
        Ok(())
    }

    /// Stable label used in CSV and reports.
    #[must_use]
    pub fn label(&self) -> String {
        let credit = match self.credit {
            Credit::PerTarget => "per-target".to_string(),
            Credit::Global => "global".to_string(),
            Credit::DelayedTrace(d) => format!("delayed-trace-{d}"),
            Credit::GlobalDelayedNoTrace(d) => format!("global-delayed-no-trace-{d}"),
        };
        format!(
            "noise{}_deliver{}_{}",
            self.noise_pct, self.deliver_pct, credit
        )
    }

    /// Whether the matched supervised control differs from the clean control.
    #[must_use]
    pub fn affects_control(&self) -> bool {
        self.noise_pct > 0 || self.deliver_pct < 100
    }
}

/// Pre-registered condition grid (one axis varied at a time).
#[must_use]
pub fn default_conditions() -> Vec<RewardCondition> {
    let c = RewardCondition::CLEAN;
    let mut out = vec![c];
    for noise_pct in [10, 20, 30, 40] {
        out.push(RewardCondition { noise_pct, ..c });
    }
    for deliver_pct in [50, 25, 10] {
        out.push(RewardCondition { deliver_pct, ..c });
    }
    for credit in [
        Credit::Global,
        Credit::DelayedTrace(4),
        Credit::DelayedTrace(32),
        Credit::GlobalDelayedNoTrace(1),
    ] {
        out.push(RewardCondition { credit, ..c });
    }
    out
}

/// SplitMix64 finaliser: a pure, arm-independent noise source.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const EPISODE_TAG: u64 = u64::MAX;

/// Deterministic Bernoulli(pct %) draw keyed by (seed, phase, episode, slot).
fn draw(seed: u64, phase: u64, episode: u64, slot: u64, salt: u64, pct: u32) -> bool {
    let h = mix(mix(mix(mix(seed ^ salt) ^ phase) ^ episode) ^ slot);
    (h % 100) < u64::from(pct)
}

const NOISE_SALT: u64 = 0x4E01_5E00;
const DELIVER_SALT: u64 = 0xDE11_7E00;

/// Instrumentation of one degraded phase.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RewardCost {
    pub phase: PhaseCost,
    /// Reward events produced (per eligible target, or per episode if global).
    pub rewards_emitted: u64,
    /// Reward events whose sign was flipped by noise.
    pub rewards_flipped: u64,
    /// Episodes with no reward delivered (sparse).
    pub episodes_undelivered: u64,
    /// Reward events whose sign disagrees with the clean per-target truth of
    /// the synapse it is applied to (noise and/or credit errors).
    pub misattributed: u64,
}

fn sign(positive: bool) -> RewardSignal {
    if positive {
        RewardSignal::Positive
    } else {
        RewardSignal::Negative
    }
}

fn single(n: usize, t: usize) -> io::Result<BooleanState> {
    let mut s = BooleanState::new(n);
    s.set(t, true).map_err(|_| bad("bad target"))?;
    Ok(s)
}

type Pending = (BooleanState, usize, bool, bool);

fn apply_one(
    cost: &mut RewardCost,
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    item: &Pending,
) -> io::Result<()> {
    let (input, t, reward, truth) = item;
    let post = single(graph.ids.len(), *t)?;
    let r = apply_update(
        circuit,
        graph,
        LearningStrategy::RewardModulated,
        input,
        &post,
        sign(*reward),
    )?;
    cost.phase.update_calls += 1;
    cost.phase.weights_changed += r.weights_changed;
    cost.misattributed += u64::from(reward != truth);
    Ok(())
}

/// Reward-modulated phase under a degraded reward channel. `keys` =
/// (seed, phase id) for the noise / delivery hash.
pub fn train_reward_phase(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    patterns: &[Vec<usize>],
    valence: &[bool],
    budget: (u32, u64),
    cond: &RewardCondition,
    keys: (u64, u64),
) -> io::Result<RewardCost> {
    cond.validate()?;
    let (rounds, order_seed) = budget;
    let (seed, phase) = keys;
    let mut cost = RewardCost::default();
    let mut queue: VecDeque<Vec<Pending>> = VecDeque::new();
    let mut past: VecDeque<Option<bool>> = VecDeque::new();
    for (k, idx) in episode_order(patterns.len(), rounds, order_seed)
        .into_iter()
        .enumerate()
    {
        let k = k as u64;
        cost.phase.episodes += 1;
        let (input, reached, teacher) = encode(graph, &patterns[idx], valence)?;
        let prediction = circuit.predict(graph, &input, 1)?;
        let mut firing = Vec::new();
        for &t in &reached {
            if prediction.get(t).map_err(|_| bad("bad target"))? {
                firing.push((t, teacher.get(t).map_err(|_| bad("bad target"))?));
            }
        }
        let delivered = draw(seed, phase, k, EPISODE_TAG, DELIVER_SALT, cond.deliver_pct);
        cost.episodes_undelivered += u64::from(!delivered);
        match cond.credit {
            Credit::PerTarget | Credit::DelayedTrace(_) => {
                let mut now: Vec<Pending> = Vec::new();
                if delivered {
                    for &(t, truth) in &firing {
                        let flip = draw(seed, phase, k, t as u64, NOISE_SALT, cond.noise_pct);
                        cost.rewards_emitted += 1;
                        cost.rewards_flipped += u64::from(flip);
                        now.push((input.clone(), t, truth ^ flip, truth));
                    }
                }
                let due = if let Credit::DelayedTrace(d) = cond.credit {
                    queue.push_back(now);
                    if queue.len() > d as usize {
                        queue.pop_front().unwrap_or_default()
                    } else {
                        Vec::new()
                    }
                } else {
                    now
                };
                for item in &due {
                    apply_one(&mut cost, circuit, graph, item)?;
                }
            }
            Credit::Global | Credit::GlobalDelayedNoTrace(_) => {
                let scalar = if delivered && !firing.is_empty() {
                    let flip = draw(seed, phase, k, EPISODE_TAG, NOISE_SALT, cond.noise_pct);
                    cost.rewards_emitted += 1;
                    cost.rewards_flipped += u64::from(flip);
                    Some(firing.iter().all(|&(_, truth)| truth) ^ flip)
                } else {
                    None
                };
                let used = if let Credit::GlobalDelayedNoTrace(d) = cond.credit {
                    past.push_back(scalar);
                    if past.len() > d as usize {
                        past.pop_front().flatten()
                    } else {
                        None
                    }
                } else {
                    scalar
                };
                if let Some(reward) = used {
                    for &(t, truth) in &firing {
                        apply_one(
                            &mut cost,
                            circuit,
                            graph,
                            &(input.clone(), t, reward, truth),
                        )?;
                    }
                }
            }
        }
    }
    // Flush delayed-trace updates still in flight at phase end.
    for due in queue {
        for item in &due {
            apply_one(&mut cost, circuit, graph, item)?;
        }
    }
    Ok(cost)
}

/// Supervised control with the matched degradation (label noise with the
/// same hash as the reward flip; same undelivered episodes skipped).
pub fn train_control_phase(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    patterns: &[Vec<usize>],
    valence: &[bool],
    budget: (u32, u64),
    cond: &RewardCondition,
    keys: (u64, u64),
) -> io::Result<RewardCost> {
    cond.validate()?;
    let (rounds, order_seed) = budget;
    let (seed, phase) = keys;
    let mut cost = RewardCost::default();
    for (k, idx) in episode_order(patterns.len(), rounds, order_seed)
        .into_iter()
        .enumerate()
    {
        let k = k as u64;
        cost.phase.episodes += 1;
        if !draw(seed, phase, k, EPISODE_TAG, DELIVER_SALT, cond.deliver_pct) {
            cost.episodes_undelivered += 1;
            continue;
        }
        let (input, reached, mut teacher) = encode(graph, &patterns[idx], valence)?;
        for &t in &reached {
            cost.rewards_emitted += 1;
            if draw(seed, phase, k, t as u64, NOISE_SALT, cond.noise_pct) {
                cost.rewards_flipped += 1;
                cost.misattributed += 1;
                let v = teacher.get(t).map_err(|_| bad("bad target"))?;
                teacher.set(t, !v).map_err(|_| bad("bad target"))?;
            }
        }
        let r = apply_update(
            circuit,
            graph,
            LearningStrategy::Supervised,
            &input,
            &teacher,
            RewardSignal::None,
        )?;
        cost.phase.update_calls += 1;
        cost.phase.weights_changed += r.weights_changed;
    }
    Ok(cost)
}

/// Which arm of the limits study.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitArm {
    /// Supervised control under the matched degradation.
    Control,
    /// Reward-modulated under the degradation.
    Reward,
}

impl LimitArm {
    #[must_use]
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Control => "supervised-matched",
            Self::Reward => "reward-modulated",
        }
    }
}

/// One (condition, seed, arm) measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct LimitRow {
    pub condition: RewardCondition,
    pub arm: LimitArm,
    pub seed: u64,
    pub init_weight: u8,
    pub a_train_after_a: Confusion,
    pub a_test_after_a: Confusion,
    pub a_test_after_b: Confusion,
    pub b_test_after_b: Confusion,
    pub cost_a: RewardCost,
    pub cost_b: RewardCost,
    pub topology_unchanged: bool,
}

impl LimitRow {
    #[must_use]
    pub fn forgetting(&self) -> f64 {
        self.a_test_after_a.balanced() - self.a_test_after_b.balanced()
    }
}

const PHASE_A: u64 = 0xA;
const PHASE_B: u64 = 0xB;

/// Run A then B for one arm, one condition, one seed's task. Phase budgets
/// and episode orders are those of `generalization_bench::run_arm`.
pub fn run_limit_arm(
    graph: &Connectome,
    task: &Task,
    spec: &GenSpec,
    cond: &RewardCondition,
    arm: LimitArm,
    seed: u64,
) -> io::Result<LimitRow> {
    let original = graph.clone();
    let mut circuit = PlasticCircuit::new(graph, spec.init_weight)?;
    let fingerprint = circuit.topology_fingerprint();
    let train = match arm {
        LimitArm::Control => train_control_phase,
        LimitArm::Reward => train_reward_phase,
    };
    let cost_a = train(
        &mut circuit,
        graph,
        &task.a_train,
        &task.valence_a,
        (spec.rounds_a, seed),
        cond,
        (seed, PHASE_A),
    )?;
    let a_train_after_a = evaluate(&circuit, graph, &task.a_train, &task.valence_a)?;
    let a_test_after_a = evaluate(&circuit, graph, &task.test, &task.valence_a)?;
    let cost_b = train(
        &mut circuit,
        graph,
        &task.b_train,
        &task.valence_b,
        (spec.rounds_b, seed ^ 0xB0B0_B0B0),
        cond,
        (seed, PHASE_B),
    )?;
    let a_test_after_b = evaluate(&circuit, graph, &task.test, &task.valence_a)?;
    let b_test_after_b = evaluate(&circuit, graph, &task.test, &task.valence_b)?;
    Ok(LimitRow {
        condition: *cond,
        arm,
        seed,
        init_weight: spec.init_weight,
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

pub(crate) fn graph_for(
    graph: Option<&Connectome>,
    shape: &GraphShape,
    seed: u64,
) -> io::Result<Connectome> {
    match graph {
        Some(g) => Ok(g.clone()),
        None => Ok(generate_graph(
            &ScaleSpec {
                sources: shape.sources,
                targets: shape.targets,
                fan_in: shape.fan_in,
                conflict_pct: 0,
                rounds_a: 1,
                rounds_b: 1,
            },
            seed,
        )?
        .graph),
    }
}

/// Grid: condition x seed x {control, reward}. Graph and task per seed are
/// identical to `generalization_bench::run_grid` for the same seed.
pub fn run_limits(
    graph: Option<&Connectome>,
    shape: &GraphShape,
    spec: &GenSpec,
    seeds: &[u64],
    conditions: &[RewardCondition],
) -> io::Result<Vec<LimitRow>> {
    if seeds.is_empty() || conditions.is_empty() {
        return Err(bad("need at least one seed and one condition"));
    }
    let mut rows = Vec::new();
    for &seed in seeds {
        let g = graph_for(graph, shape, seed)?;
        let task = build_task(&g, spec, seed)?;
        for cond in conditions {
            for arm in [LimitArm::Control, LimitArm::Reward] {
                rows.push(run_limit_arm(&g, &task, spec, cond, arm, seed)?);
            }
        }
    }
    Ok(rows)
}

pub const CSV_HEADER: &str = "schema,graph,condition,noise_pct,deliver_pct,credit,arm,seed,init_weight,a_train_bal_after_a,a_test_bal_after_a,a_test_bal_after_b,b_test_bal_after_b,forgetting,a_episodes,a_update_calls,a_weights_changed,a_rewards_emitted,a_rewards_flipped,a_episodes_undelivered,a_misattributed,b_episodes,b_update_calls,b_weights_changed,b_rewards_emitted,b_rewards_flipped,b_episodes_undelivered,b_misattributed,topology_unchanged";

/// Deterministic CSV (no timing column).
#[must_use]
pub fn to_csv(rows: &[LimitRow], graph_label: &str) -> String {
    let mut out = String::from(CSV_HEADER);
    out.push('\n');
    for r in rows {
        let label = r.condition.label();
        let credit = label.splitn(3, '_').nth(2).unwrap_or("");
        out.push_str(&format!(
            "b888-4-reward-limits-v1,{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            graph_label,
            label,
            r.condition.noise_pct,
            r.condition.deliver_pct,
            credit,
            r.arm.as_label(),
            r.seed,
            r.init_weight,
            r.a_train_after_a.balanced(),
            r.a_test_after_a.balanced(),
            r.a_test_after_b.balanced(),
            r.b_test_after_b.balanced(),
            r.forgetting(),
            r.cost_a.phase.episodes,
            r.cost_a.phase.update_calls,
            r.cost_a.phase.weights_changed,
            r.cost_a.rewards_emitted,
            r.cost_a.rewards_flipped,
            r.cost_a.episodes_undelivered,
            r.cost_a.misattributed,
            r.cost_b.phase.episodes,
            r.cost_b.phase.update_calls,
            r.cost_b.phase.weights_changed,
            r.cost_b.rewards_emitted,
            r.cost_b.rewards_flipped,
            r.cost_b.episodes_undelivered,
            r.cost_b.misattributed,
            r.topology_unchanged,
        ));
    }
    out
}

/// Per-condition summary of the reward arm.
#[derive(Clone, Debug, PartialEq)]
pub struct LimitSummary {
    pub condition: RewardCondition,
    pub seeds: usize,
    pub reward_a_test: Spread,
    pub control_a_test: Spread,
    pub reward_forgetting: Spread,
    pub control_forgetting: Spread,
    pub reward_b_test: Spread,
    pub control_b_test: Spread,
    /// Reward arm A test vs clean reward arm (same seed); negative = drop.
    pub reward_vs_clean: Spread,
    /// Paired reward - matched control, A test (generalisation).
    pub gen_vs_control: gb::Paired,
    /// Paired control forgetting - reward forgetting (+ = reward forgets less).
    pub forget_vs_control: gb::Paired,
    pub b_vs_control: Spread,
    pub gen_verdict: Verdict,
    pub forget_verdict: Verdict,
    /// Pre-stated: mean drop vs clean reward >= 0.05.
    pub hurts_reward: bool,
    pub reward_update_calls: f64,
    pub control_update_calls: f64,
}

#[must_use]
pub fn summarize(rows: &[LimitRow]) -> Vec<LimitSummary> {
    let mut conds: Vec<RewardCondition> = Vec::new();
    for r in rows {
        if !conds.contains(&r.condition) {
            conds.push(r.condition);
        }
    }
    let find = |c: &RewardCondition, arm: LimitArm, seed: u64| {
        rows.iter()
            .find(|r| r.condition == *c && r.arm == arm && r.seed == seed)
    };
    let mut out = Vec::new();
    for cond in conds {
        let rew: Vec<&LimitRow> = rows
            .iter()
            .filter(|r| r.condition == cond && r.arm == LimitArm::Reward)
            .collect();
        let mut ctl = Vec::new();
        let (mut d_gen, mut d_forget, mut d_b, mut d_clean) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for r in &rew {
            if let Some(c) = find(&cond, LimitArm::Control, r.seed) {
                ctl.push(c);
                d_gen.push(r.a_test_after_a.balanced() - c.a_test_after_a.balanced());
                d_forget.push(c.forgetting() - r.forgetting());
                d_b.push(r.b_test_after_b.balanced() - c.b_test_after_b.balanced());
            }
            if let Some(clean) = find(&RewardCondition::CLEAN, LimitArm::Reward, r.seed) {
                d_clean.push(r.a_test_after_a.balanced() - clean.a_test_after_a.balanced());
            }
        }
        let col = |v: &[&LimitRow], f: &dyn Fn(&LimitRow) -> f64| -> Spread {
            gb::spread(&v.iter().map(|r| f(r)).collect::<Vec<_>>())
        };
        let gen_p = gb::paired(&d_gen);
        let forget_p = gb::paired(&d_forget);
        let b_spread = gb::spread(&d_b);
        let clean_spread = gb::spread(&d_clean);
        let n = d_gen.len();
        let calls = |v: &[&LimitRow]| {
            if v.is_empty() {
                0.0
            } else {
                v.iter()
                    .map(|r| (r.cost_a.phase.update_calls + r.cost_b.phase.update_calls) as f64)
                    .sum::<f64>()
                    / v.len() as f64
            }
        };
        out.push(LimitSummary {
            condition: cond,
            seeds: rew.len(),
            reward_a_test: col(&rew, &|r| r.a_test_after_a.balanced()),
            control_a_test: col(&ctl, &|r| r.a_test_after_a.balanced()),
            reward_forgetting: col(&rew, &|r| r.forgetting()),
            control_forgetting: col(&ctl, &|r| r.forgetting()),
            reward_b_test: col(&rew, &|r| r.b_test_after_b.balanced()),
            control_b_test: col(&ctl, &|r| r.b_test_after_b.balanced()),
            reward_vs_clean: clean_spread,
            gen_verdict: gb::verdict(&gen_p, n, true),
            forget_verdict: gb::verdict(&forget_p, n, b_spread.mean > -gb::MARGIN),
            gen_vs_control: gen_p,
            forget_vs_control: forget_p,
            b_vs_control: b_spread,
            hurts_reward: clean_spread.mean <= -gb::MARGIN,
            reward_update_calls: calls(&rew),
            control_update_calls: calls(&ctl),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generalization_bench::{run_arm, Arm};
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
        let g = graph_for(None, &shape(), seed).unwrap();
        let t = build_task(&g, &spec(), seed).unwrap();
        (g, t)
    }

    #[test]
    fn clean_condition_reproduces_generalization_bench_arms_exactly() {
        let (g, task) = setup(3);
        let c = RewardCondition::CLEAN;
        let rew = run_limit_arm(&g, &task, &spec(), &c, LimitArm::Reward, 3).unwrap();
        let ctl = run_limit_arm(&g, &task, &spec(), &c, LimitArm::Control, 3).unwrap();
        let rew0 = run_arm(
            &g,
            &task,
            &spec(),
            Arm::Learn(LearningStrategy::RewardModulated),
            3,
        )
        .unwrap();
        let ctl0 = run_arm(
            &g,
            &task,
            &spec(),
            Arm::Learn(LearningStrategy::Supervised),
            3,
        )
        .unwrap();
        assert_eq!(rew.a_test_after_a, rew0.a_test_after_a);
        assert_eq!(rew.a_test_after_b, rew0.a_test_after_b);
        assert_eq!(rew.b_test_after_b, rew0.b_test_after_b);
        assert_eq!(rew.cost_a.phase, rew0.cost_a);
        assert_eq!(ctl.a_test_after_b, ctl0.a_test_after_b);
        assert_eq!(ctl.cost_b.phase, ctl0.cost_b);
        assert_eq!(rew.cost_a.misattributed + rew.cost_a.rewards_flipped, 0);
    }

    #[test]
    fn noise_and_delivery_are_hash_matched_across_arms() {
        let (g, task) = setup(4);
        let c = RewardCondition {
            deliver_pct: 25,
            ..RewardCondition::CLEAN
        };
        let rew = run_limit_arm(&g, &task, &spec(), &c, LimitArm::Reward, 4).unwrap();
        let ctl = run_limit_arm(&g, &task, &spec(), &c, LimitArm::Control, 4).unwrap();
        assert_eq!(
            rew.cost_a.episodes_undelivered,
            ctl.cost_a.episodes_undelivered
        );
        assert!(rew.cost_a.episodes_undelivered > 0);
        assert_eq!(rew.cost_a.phase.episodes, ctl.cost_a.phase.episodes);
        // full noise inverts every reward sign: everything is misattributed
        let all = RewardCondition {
            noise_pct: 100,
            ..RewardCondition::CLEAN
        };
        let r = run_limit_arm(&g, &task, &spec(), &all, LimitArm::Reward, 4).unwrap();
        assert_eq!(r.cost_a.rewards_flipped, r.cost_a.rewards_emitted);
        assert_eq!(r.cost_a.misattributed, r.cost_a.phase.update_calls);
        let none = RewardCondition {
            deliver_pct: 0,
            ..RewardCondition::CLEAN
        };
        let r = run_limit_arm(&g, &task, &spec(), &none, LimitArm::Reward, 4).unwrap();
        assert_eq!(
            r.cost_a.phase.weights_changed + r.cost_b.phase.weights_changed,
            0
        );
        assert_eq!(r.cost_a.episodes_undelivered, r.cost_a.phase.episodes);
    }

    #[test]
    fn delayed_trace_zero_equals_per_target_and_flush_keeps_update_count() {
        let (g, task) = setup(5);
        let base = run_limit_arm(
            &g,
            &task,
            &spec(),
            &RewardCondition::CLEAN,
            LimitArm::Reward,
            5,
        )
        .unwrap();
        let d0 = RewardCondition {
            credit: Credit::DelayedTrace(0),
            ..RewardCondition::CLEAN
        };
        let r0 = run_limit_arm(&g, &task, &spec(), &d0, LimitArm::Reward, 5).unwrap();
        assert_eq!(base.a_test_after_b, r0.a_test_after_b);
        assert_eq!(base.cost_a, r0.cost_a);
        let d4 = RewardCondition {
            credit: Credit::DelayedTrace(4),
            ..RewardCondition::CLEAN
        };
        let r4 = run_limit_arm(&g, &task, &spec(), &d4, LimitArm::Reward, 5).unwrap();
        // every emitted reward is applied exactly once (flushed at phase end)
        assert_eq!(r4.cost_a.rewards_emitted, r4.cost_a.phase.update_calls);
        assert_eq!(r4.cost_a.misattributed, 0);
        assert!(r4.topology_unchanged);
    }

    #[test]
    fn global_credit_misattributes_and_no_trace_delay_skips_first_episodes() {
        let (g, task) = setup(6);
        let glob = RewardCondition {
            credit: Credit::Global,
            ..RewardCondition::CLEAN
        };
        let r = run_limit_arm(&g, &task, &spec(), &glob, LimitArm::Reward, 6).unwrap();
        assert!(r.cost_a.misattributed + r.cost_b.misattributed > 0);
        assert_eq!(r.cost_a.rewards_flipped, 0);
        let far = RewardCondition {
            credit: Credit::GlobalDelayedNoTrace(1000),
            ..RewardCondition::CLEAN
        };
        let r = run_limit_arm(&g, &task, &spec(), &far, LimitArm::Reward, 6).unwrap();
        // delay longer than a phase: no reward ever arrives
        assert_eq!(r.cost_a.phase.update_calls + r.cost_b.phase.update_calls, 0);
    }

    #[test]
    fn grid_is_deterministic_and_summary_is_consistent() {
        let conds = [
            RewardCondition::CLEAN,
            RewardCondition {
                noise_pct: 30,
                ..RewardCondition::CLEAN
            },
        ];
        let a = run_limits(None, &shape(), &spec(), &[1, 2], &conds).unwrap();
        let b = run_limits(None, &shape(), &spec(), &[1, 2], &conds).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 2 * 2 * 2);
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        assert!(csv.contains(",noise30_deliver100_per-target,30,100,per-target,"));
        let s = summarize(&a);
        assert_eq!(s.len(), 2);
        assert!(s[0].reward_vs_clean.mean.abs() < 1e-12);
        assert!(!s[0].hurts_reward);
        assert!(a.iter().all(|r| r.topology_unchanged));
        assert!(run_limits(None, &shape(), &spec(), &[], &conds).is_err());
        let bad_c = RewardCondition {
            noise_pct: 101,
            ..RewardCondition::CLEAN
        };
        assert!(run_limits(None, &shape(), &spec(), &[1], &[bad_c]).is_err());
    }

    #[test]
    fn checkpoint_between_a_and_b_resumes_bit_exactly() -> io::Result<()> {
        let (g, task) = setup(7);
        let cond = RewardCondition {
            noise_pct: 20,
            deliver_pct: 50,
            credit: Credit::DelayedTrace(4),
        };
        let mut cont = PlasticCircuit::new(&g, 1)?;
        train_reward_phase(
            &mut cont,
            &g,
            &task.a_train,
            &task.valence_a,
            (4, 7),
            &cond,
            (7, PHASE_A),
        )?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| bad("clock"))?
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("v888-rl-{}-{nonce}.v888pls", std::process::id()));
        cont.save(&path)?;
        let mut resumed = PlasticCircuit::load(&path, &g)?;
        std::fs::remove_file(&path)?;
        let b = (4, 7 ^ 0xB0B0_B0B0);
        let c1 = train_reward_phase(
            &mut cont,
            &g,
            &task.b_train,
            &task.valence_b,
            b,
            &cond,
            (7, PHASE_B),
        )?;
        let c2 = train_reward_phase(
            &mut resumed,
            &g,
            &task.b_train,
            &task.valence_b,
            b,
            &cond,
            (7, PHASE_B),
        )?;
        assert_eq!(c1, c2);
        assert_eq!(cont.weights(), resumed.weights());
        Ok(())
    }

    #[test]
    fn default_grid_varies_one_axis_at_a_time() {
        let conds = default_conditions();
        assert_eq!(conds[0], RewardCondition::CLEAN);
        assert_eq!(conds.len(), 12);
        for c in &conds[1..] {
            let axes = u32::from(c.noise_pct != 0)
                + u32::from(c.deliver_pct != 100)
                + u32::from(c.credit != Credit::PerTarget);
            assert_eq!(axes, 1, "{}", c.label());
        }
    }
}

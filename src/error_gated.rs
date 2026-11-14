//! B888-4 cycle 8: **error-gated** and **margin-gated** supervision. The
//! matched supervised control steps every edge from an active source toward
//! its target's teacher bit on every episode; the gated arms apply the very
//! same step only to targets whose prediction is wrong (or, for the margin
//! arm, whose positive drive is below a fixed margin). Equal episode budget,
//! no rehearsal, no protection. Test pairs are split into *agree* pairs (A
//! label = B label) and *conflict* pairs, so avoidable and unavoidable
//! forgetting are measured separately.
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph; nothing here is
//! a claim about synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in
//! `docs/ERROR_GATED_SUPERVISION.md` (committed before any run). Only weights
//! of existing edges change.
use crate::connectome::Connectome;
use crate::consolidated_graded::{b_stream, Item};
use crate::gates;
use crate::generalization_bench::{
    self as gb, build_task, encode, episode_order, Confusion, GenSpec, GraphShape,
};
use crate::importance_protection::mix;
use crate::learning_strategies::{apply_update_gated, LearningStrategy, RewardSignal};
use crate::plasticity::PlasticCircuit;
use crate::reward_limits as rl;
use crate::BooleanState;
use std::io::{self, ErrorKind};

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Pre-registered positive-drive margin of `margin-gated-4`.
pub const MARGIN: u128 = 4;
/// Prediction threshold (same as every earlier B888-4 bench).
pub const THRESHOLD: u128 = 1;
/// Bootstrap seed and resample count (pre-registered).
pub const BOOT_SEED: u64 = 0xB007_5EED;
pub const BOOT_RESAMPLES: usize = 2000;
/// Pre-registered effect margin.
pub const EFFECT: f64 = 0.05;

/// When the supervised step is applied to a reached target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    /// Always (the matched control).
    Supervised,
    /// Only if the prediction for the target is wrong.
    ErrorGated,
    /// Wrong, or should fire with drive below the margin.
    MarginGated(u128),
}

impl Gate {
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::Supervised => "supervised".to_string(),
            Self::ErrorGated => "error-gated".to_string(),
            Self::MarginGated(m) => format!("margin-gated-{m}"),
        }
    }
}

pub const ARMS: [Gate; 3] = [
    Gate::Supervised,
    Gate::ErrorGated,
    Gate::MarginGated(MARGIN),
];

/// Per-target drive Σ weight × contacts over active edges: the exact value
/// `PlasticCircuit::predict` compares to its threshold (same ripple adds).
pub fn drive(
    circuit: &PlasticCircuit,
    graph: &Connectome,
    input: &BooleanState,
) -> io::Result<Vec<u128>> {
    if circuit.weights().len() != graph.edges() || input.len() != graph.ids.len() {
        return Err(bad("drive shape mismatch"));
    }
    let mut out = vec![0_u128; graph.ids.len()];
    for source in input.active_indices() {
        for edge in graph.offsets[source]..graph.offsets[source + 1] {
            let edge = edge as usize;
            let weight = circuit.weights()[edge];
            if weight == 0 {
                continue;
            }
            let contacts = u128::from(graph.counts[edge]);
            let target = graph.targets[edge] as usize;
            for shift in 0..4 {
                if (weight >> shift) & 1 != 0 {
                    out[target] = gates::add(out[target], contacts << shift)?;
                }
            }
        }
    }
    Ok(out)
}

/// Per-phase instrumentation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseCost {
    pub episodes: u64,
    /// (episode, reached target) pairs seen.
    pub targets_seen: u64,
    /// Of those, pairs whose incoming edges were offered the update.
    pub targets_updated: u64,
    /// Edge update attempts (same counter as the control's `attempts`).
    pub attempts: u64,
    pub weights_changed: u64,
}

/// One gated supervised episode. The gate decision uses the weights before
/// the update.
pub fn step(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    gate: Gate,
    item: Item<'_>,
    cost: &mut PhaseCost,
) -> io::Result<()> {
    let (pattern, valence) = item;
    let (input, reached, teacher) = encode(graph, pattern, valence)?;
    let mut update = vec![gate == Gate::Supervised; graph.ids.len()];
    if gate != Gate::Supervised {
        let d = drive(circuit, graph, &input)?;
        for &t in &reached {
            let want = teacher.get(t).map_err(|_| bad("bad target"))?;
            let fires = gates::ge(d[t], THRESHOLD);
            update[t] = match gate {
                Gate::Supervised => true,
                Gate::ErrorGated => fires != want,
                Gate::MarginGated(m) => {
                    if want {
                        !gates::ge(d[t], m)
                    } else {
                        fires
                    }
                }
            };
        }
    }
    cost.episodes += 1;
    cost.targets_seen += reached.len() as u64;
    cost.targets_updated += reached.iter().filter(|&&t| update[t]).count() as u64;
    let r = apply_update_gated(
        circuit,
        graph,
        LearningStrategy::Supervised,
        &input,
        &teacher,
        RewardSignal::None,
        &mut |edge| update[graph.targets[edge] as usize],
    )?;
    cost.attempts += r.edges_visited;
    cost.weights_changed += r.weights_changed;
    Ok(())
}

/// Train a stream of episodes, accumulating into `cost`.
pub fn train(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    gate: Gate,
    items: &[Item<'_>],
    cost: &mut PhaseCost,
) -> io::Result<()> {
    for &item in items {
        step(circuit, graph, gate, item, cost)?;
    }
    Ok(())
}

/// Confusion over all test pairs and over agree / conflict pairs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Split {
    pub all: Confusion,
    pub agree: Confusion,
    pub conflict: Confusion,
}

fn tally(c: &mut Confusion, want: bool, got: bool) {
    match (want, got) {
        (true, true) => c.tp += 1,
        (false, false) => c.tn += 1,
        (false, true) => c.fp += 1,
        (true, false) => c.fn_ += 1,
    }
}

/// Score under `valence`; a pair is "agree" if its label under `other`
/// is the same. `all` equals `generalization_bench::evaluate` (tested).
pub fn evaluate_split(
    circuit: &PlasticCircuit,
    graph: &Connectome,
    patterns: &[Vec<usize>],
    valence: &[bool],
    other: &[bool],
) -> io::Result<Split> {
    let mut s = Split::default();
    for p in patterns {
        let (input, reached, teacher) = encode(graph, p, valence)?;
        let (_, _, teacher_other) = encode(graph, p, other)?;
        let out = circuit.predict(graph, &input, THRESHOLD)?;
        for t in reached {
            let want = teacher.get(t).map_err(|_| bad("bad target"))?;
            let alt = teacher_other.get(t).map_err(|_| bad("bad target"))?;
            let got = out.get(t).map_err(|_| bad("bad target"))?;
            tally(&mut s.all, want, got);
            tally(
                if want == alt {
                    &mut s.agree
                } else {
                    &mut s.conflict
                },
                want,
                got,
            );
        }
    }
    Ok(s)
}

/// One (gate, seed) measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub gate: Gate,
    pub seed: u64,
    pub a_train_after_a: Confusion,
    pub a_test_after_a: Split,
    pub a_test_after_b: Split,
    pub b_test_after_b: Split,
    pub cost_a: PhaseCost,
    pub cost_b: PhaseCost,
    pub saturated_after_a: u64,
    pub saturated_after_b: u64,
    pub topology_unchanged: bool,
}

impl Row {
    #[must_use]
    pub fn forgetting(&self) -> f64 {
        self.a_test_after_a.all.balanced() - self.a_test_after_b.all.balanced()
    }
    #[must_use]
    pub fn forgetting_agree(&self) -> f64 {
        self.a_test_after_a.agree.balanced() - self.a_test_after_b.agree.balanced()
    }
    #[must_use]
    pub fn forgetting_conflict(&self) -> f64 {
        self.a_test_after_a.conflict.balanced() - self.a_test_after_b.conflict.balanced()
    }
}

/// Weights on the 0 / 15 plateaus.
#[must_use]
pub fn saturated(weights: &[u8]) -> u64 {
    weights.iter().filter(|&&w| w == 0 || w == 15).count() as u64
}

/// The #39/#43 task-A episode order.
pub(crate) fn a_items<'a>(task: &'a gb::Task, spec: &GenSpec, seed: u64) -> Vec<Item<'a>> {
    episode_order(task.a_train.len(), spec.rounds_a, seed)
        .into_iter()
        .map(|i| (task.a_train[i].as_slice(), task.valence_a.as_slice()))
        .collect()
}

/// The #39/#43 task-B episode order (no rehearsal).
pub(crate) fn b_items<'a>(task: &'a gb::Task, spec: &GenSpec, seed: u64) -> Vec<Item<'a>> {
    b_stream(task, spec, seed, false)
}

/// Run A then B for one gate.
pub fn run_arm(
    graph: &Connectome,
    task: &gb::Task,
    spec: &GenSpec,
    gate: Gate,
    seed: u64,
) -> io::Result<Row> {
    let original = graph.clone();
    let mut circuit = PlasticCircuit::new(graph, spec.init_weight)?;
    let fingerprint = circuit.topology_fingerprint();
    let (va, vb) = (&task.valence_a, &task.valence_b);
    let mut cost_a = PhaseCost::default();
    train(
        &mut circuit,
        graph,
        gate,
        &a_items(task, spec, seed),
        &mut cost_a,
    )?;
    let a_train_after_a = gb::evaluate(&circuit, graph, &task.a_train, va)?;
    let a_test_after_a = evaluate_split(&circuit, graph, &task.test, va, vb)?;
    let saturated_after_a = saturated(circuit.weights());
    let mut cost_b = PhaseCost::default();
    train(
        &mut circuit,
        graph,
        gate,
        &b_items(task, spec, seed),
        &mut cost_b,
    )?;
    let a_test_after_b = evaluate_split(&circuit, graph, &task.test, va, vb)?;
    let b_test_after_b = evaluate_split(&circuit, graph, &task.test, vb, va)?;
    Ok(Row {
        gate,
        seed,
        a_train_after_a,
        a_test_after_a,
        a_test_after_b,
        b_test_after_b,
        cost_a,
        cost_b,
        saturated_after_a,
        saturated_after_b: saturated(circuit.weights()),
        topology_unchanged: *graph == original
            && circuit.topology_fingerprint() == fingerprint
            && circuit.weights().len() == graph.edges(),
    })
}

/// Grid: seed × gate (3 rows per seed).
pub fn run_grid(
    graph: Option<&Connectome>,
    shape: &GraphShape,
    spec: &GenSpec,
    seeds: &[u64],
) -> io::Result<Vec<Row>> {
    if seeds.is_empty() {
        return Err(bad("need at least one seed"));
    }
    let mut rows = Vec::new();
    for &seed in seeds {
        let g = rl::graph_for(graph, shape, seed)?;
        let task = build_task(&g, spec, seed)?;
        for gate in ARMS {
            rows.push(run_arm(&g, &task, spec, gate, seed)?);
        }
    }
    Ok(rows)
}

pub const CSV_HEADER: &str = "schema,graph,arm,seed,a_train_bal_after_a,a_test_bal_after_a,a_test_bal_after_b,b_test_bal_after_b,forgetting,agree_a_test_after_a,agree_a_test_after_b,agree_b_test,agree_forgetting,conflict_a_test_after_a,conflict_a_test_after_b,conflict_b_test,conflict_forgetting,test_pairs_agree,test_pairs_conflict,a_episodes,a_targets_seen,a_targets_updated,a_attempts,a_weights_changed,b_episodes,b_targets_seen,b_targets_updated,b_attempts,b_weights_changed,saturated_after_a,saturated_after_b,topology_unchanged";

/// Deterministic CSV (no timing column).
#[must_use]
pub fn to_csv(rows: &[Row], graph_label: &str) -> String {
    let mut out = String::from(CSV_HEADER);
    out.push('\n');
    for r in rows {
        let line = format!(
            "b888-4-error-gated-v1,{graph_label},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            r.gate.label(),
            r.seed,
            r.a_train_after_a.balanced(),
            r.a_test_after_a.all.balanced(),
            r.a_test_after_b.all.balanced(),
            r.b_test_after_b.all.balanced(),
            r.forgetting(),
            r.a_test_after_a.agree.balanced(),
            r.a_test_after_b.agree.balanced(),
            r.b_test_after_b.agree.balanced(),
            r.forgetting_agree(),
            r.a_test_after_a.conflict.balanced(),
            r.a_test_after_b.conflict.balanced(),
            r.b_test_after_b.conflict.balanced(),
            r.forgetting_conflict(),
            r.a_test_after_a.agree.total(),
            r.a_test_after_a.conflict.total(),
            r.cost_a.episodes,
            r.cost_a.targets_seen,
            r.cost_a.targets_updated,
            r.cost_a.attempts,
            r.cost_a.weights_changed,
            r.cost_b.episodes,
            r.cost_b.targets_seen,
            r.cost_b.targets_updated,
            r.cost_b.attempts,
            r.cost_b.weights_changed,
            r.saturated_after_a,
            r.saturated_after_b,
            r.topology_unchanged,
        );
        out.push_str(&line);
    }
    out
}

/// Summary of a vector of paired differences, oriented so that > 0 is
/// "better", with a deterministic percentile bootstrap CI of the mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Paired {
    pub n: usize,
    pub mean: f64,
    pub sd: f64,
    pub better: usize,
    pub ties: usize,
    pub worse: usize,
    pub ci_lo: f64,
    pub ci_hi: f64,
}

impl Paired {
    /// Pre-registered "strong" criterion: mean ≥ 0.05, > 0 in ≥ 80 % of
    /// seeds (16/20), bootstrap CI lower bound > 0.
    #[must_use]
    pub fn strong(&self) -> bool {
        self.n > 0 && self.mean >= EFFECT && self.better * 5 >= self.n * 4 && self.ci_lo > 0.0
    }
}

/// Percentile bootstrap 95 % CI of the mean (deterministic).
#[must_use]
pub fn bootstrap_ci(values: &[f64], resamples: usize, seed: u64) -> (f64, f64) {
    let n = values.len();
    if n == 0 || resamples == 0 {
        return (0.0, 0.0);
    }
    let mut counter = 0_u64;
    let mut means: Vec<f64> = (0..resamples)
        .map(|_| {
            let mut sum = 0.0;
            for _ in 0..n {
                counter = counter.wrapping_add(1);
                sum += values[(mix(seed ^ counter.wrapping_mul(0x9E37_79B9)) % n as u64) as usize];
            }
            sum / n as f64
        })
        .collect();
    means.sort_by(f64::total_cmp);
    let lo = (resamples as f64 * 0.025).floor() as usize;
    let hi = ((resamples as f64 * 0.975).ceil() as usize).saturating_sub(1);
    (means[lo.min(resamples - 1)], means[hi.min(resamples - 1)])
}

#[must_use]
pub fn paired(diffs: &[f64]) -> Paired {
    let s = gb::spread(diffs);
    let (ci_lo, ci_hi) = bootstrap_ci(diffs, BOOT_RESAMPLES, BOOT_SEED);
    Paired {
        n: diffs.len(),
        mean: s.mean,
        sd: s.sd,
        better: diffs.iter().filter(|&&d| d > 1e-12).count(),
        ties: diffs.iter().filter(|&&d| d.abs() <= 1e-12).count(),
        worse: diffs.iter().filter(|&&d| d < -1e-12).count(),
        ci_lo,
        ci_hi,
    }
}

/// Arm `x` paired by seed against the `supervised` control.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Comparison {
    pub x: Gate,
    /// control − x (positive = x forgets less).
    pub forget: Paired,
    pub forget_agree: Paired,
    pub forget_conflict: Paired,
    /// x − control.
    pub a_test: Paired,
    pub retained: Paired,
    pub b: Paired,
    pub trade: Paired,
}

impl Comparison {
    /// Pre-registered pass rule (H1/H2).
    #[must_use]
    pub fn passes(&self) -> bool {
        self.forget.strong() && self.a_test.mean >= -EFFECT && self.b.mean >= -EFFECT
    }
    /// Pre-registered generalisation rule (H3).
    #[must_use]
    pub fn generalises(&self) -> bool {
        (self.a_test.strong() && self.b.mean >= -EFFECT)
            || (self.b.strong() && self.a_test.mean >= -EFFECT)
    }
}

fn pairs(rows: &[Row], x: Gate) -> Vec<(&Row, &Row)> {
    rows.iter()
        .filter(|r| r.gate == x)
        .filter_map(|r| {
            rows.iter()
                .find(|c| c.gate == Gate::Supervised && c.seed == r.seed)
                .map(|c| (r, c))
        })
        .collect()
}

#[must_use]
pub fn compare(rows: &[Row], x: Gate) -> Comparison {
    let p = pairs(rows, x);
    let d =
        |f: &dyn Fn(&Row, &Row) -> f64| paired(&p.iter().map(|(r, c)| f(r, c)).collect::<Vec<_>>());
    Comparison {
        x,
        forget: d(&|r, c| c.forgetting() - r.forgetting()),
        forget_agree: d(&|r, c| c.forgetting_agree() - r.forgetting_agree()),
        forget_conflict: d(&|r, c| c.forgetting_conflict() - r.forgetting_conflict()),
        a_test: d(&|r, c| r.a_test_after_a.all.balanced() - c.a_test_after_a.all.balanced()),
        retained: d(&|r, c| r.a_test_after_b.all.balanced() - c.a_test_after_b.all.balanced()),
        b: d(&|r, c| r.b_test_after_b.all.balanced() - c.b_test_after_b.all.balanced()),
        trade: d(&|r, c| {
            (r.a_test_after_b.all.balanced() - c.a_test_after_b.all.balanced())
                + (r.b_test_after_b.all.balanced() - c.b_test_after_b.all.balanced())
        }),
    }
}

/// Comparisons of every gated arm vs the control.
#[must_use]
pub fn preregistered_comparisons(rows: &[Row]) -> Vec<Comparison> {
    ARMS.iter()
        .filter(|&&g| g != Gate::Supervised)
        .map(|&g| compare(rows, g))
        .collect()
}

/// Control's agree-pair forgetting (H4), one-sample, > 0 = forgets.
#[must_use]
pub fn control_agree_forgetting(rows: &[Row]) -> Paired {
    paired(
        &rows
            .iter()
            .filter(|r| r.gate == Gate::Supervised)
            .map(Row::forgetting_agree)
            .collect::<Vec<_>>(),
    )
}

/// (H1, H2, H3, H4) as pre-registered.
#[must_use]
pub fn readings(rows: &[Row]) -> (bool, bool, bool, bool) {
    let e = compare(rows, Gate::ErrorGated);
    let m = compare(rows, Gate::MarginGated(MARGIN));
    (
        e.passes(),
        m.passes(),
        e.generalises() || m.generalises(),
        control_agree_forgetting(rows).strong(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consolidated_graded as cg;
    use std::fs;
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

    fn setup(seed: u64) -> (Connectome, gb::Task) {
        let g = rl::graph_for(None, &shape(), seed).unwrap();
        let t = build_task(&g, &spec(), seed).unwrap();
        (g, t)
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("v888-eg-{tag}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn supervised_arm_is_bit_identical_to_the_matched_control() {
        for seed in [1, 2, 3] {
            let (g, task) = setup(seed);
            let ours = run_arm(&g, &task, &spec(), Gate::Supervised, seed).unwrap();
            let theirs = cg::run_arm(
                &g,
                &task,
                &spec(),
                cg::Learner::Control,
                cg::Mechanism::None,
                seed,
            )
            .unwrap();
            assert_eq!(ours.a_train_after_a, theirs.a_train_after_a);
            assert_eq!(ours.a_test_after_a.all, theirs.a_test_after_a);
            assert_eq!(ours.a_test_after_b.all, theirs.a_test_after_b);
            assert_eq!(ours.b_test_after_b.all, theirs.b_test_after_b);
            assert_eq!(ours.cost_a.attempts, theirs.cost_a.attempts);
            assert_eq!(ours.cost_b.weights_changed, theirs.cost_b.weights_changed);
            assert_eq!(ours.cost_a.targets_seen, ours.cost_a.targets_updated);
            assert!(ours.topology_unchanged);
        }
    }

    #[test]
    fn drive_thresholds_exactly_like_predict() {
        let (g, task) = setup(4);
        let mut c = PlasticCircuit::new(&g, 1).unwrap();
        let mut cost = PhaseCost::default();
        train(
            &mut c,
            &g,
            Gate::Supervised,
            &a_items(&task, &spec(), 4),
            &mut cost,
        )
        .unwrap();
        for p in task.test.iter().chain(&task.a_train) {
            let (input, _, _) = encode(&g, p, &task.valence_a).unwrap();
            let d = drive(&c, &g, &input).unwrap();
            let out = c.predict(&g, &input, THRESHOLD).unwrap();
            for (t, &v) in d.iter().enumerate() {
                assert_eq!(gates::ge(v, THRESHOLD), out.get(t).unwrap());
            }
        }
    }

    #[test]
    fn split_matches_evaluate_and_partitions_pairs() {
        let (g, task) = setup(5);
        let c = PlasticCircuit::new(&g, 1).unwrap();
        let (va, vb) = (&task.valence_a, &task.valence_b);
        let s = evaluate_split(&c, &g, &task.test, va, vb).unwrap();
        assert_eq!(s.all, gb::evaluate(&c, &g, &task.test, va).unwrap());
        assert_eq!(s.all.total(), s.agree.total() + s.conflict.total());
        assert!(s.conflict.total() > 0 && s.agree.total() > 0);
        let same = evaluate_split(&c, &g, &task.test, va, va).unwrap();
        assert_eq!(same.conflict.total(), 0);
        assert_eq!(same.agree, same.all);
        // Conflict pairs have opposite labels under A and B.
        let sb = evaluate_split(&c, &g, &task.test, vb, va).unwrap();
        assert_eq!(
            sb.conflict.tp + sb.conflict.fn_,
            s.conflict.tn + s.conflict.fp
        );
    }

    #[test]
    fn error_gate_leaves_correct_targets_untouched() {
        let (g, task) = setup(6);
        let mut c = PlasticCircuit::new(&g, 1).unwrap();
        let mut cost = PhaseCost::default();
        train(
            &mut c,
            &g,
            Gate::Supervised,
            &a_items(&task, &spec(), 6),
            &mut cost,
        )
        .unwrap();
        let mut touched_wrong = 0;
        for p in &task.b_train {
            let (input, reached, teacher) = encode(&g, p, &task.valence_b).unwrap();
            let before = c.weights().to_vec();
            let out = c.predict(&g, &input, THRESHOLD).unwrap();
            let mut cost = PhaseCost::default();
            step(
                &mut c,
                &g,
                Gate::ErrorGated,
                (p, &task.valence_b),
                &mut cost,
            )
            .unwrap();
            let wrong: Vec<usize> = reached
                .iter()
                .copied()
                .filter(|&t| out.get(t).unwrap() != teacher.get(t).unwrap())
                .collect();
            assert_eq!(cost.targets_updated, wrong.len() as u64);
            for (e, (&w0, &w1)) in before.iter().zip(c.weights()).enumerate() {
                if w0 != w1 {
                    assert!(wrong.contains(&(g.targets[e] as usize)), "edge {e}");
                    touched_wrong += 1;
                }
            }
        }
        assert!(touched_wrong > 0);
    }

    #[test]
    fn margin_one_equals_error_gating() {
        let (g, task) = setup(7);
        let e = run_arm(&g, &task, &spec(), Gate::ErrorGated, 7).unwrap();
        let mut m = run_arm(&g, &task, &spec(), Gate::MarginGated(1), 7).unwrap();
        m.gate = Gate::ErrorGated;
        assert_eq!(e, m);
        let m4 = run_arm(&g, &task, &spec(), Gate::MarginGated(4), 7).unwrap();
        assert!(m4.cost_a.targets_updated >= e.cost_a.targets_updated);
    }

    #[test]
    fn checkpoint_after_a_and_mid_b_resumes_bit_exactly() -> io::Result<()> {
        let (g, task) = setup(8);
        for gate in ARMS {
            let full = run_arm(&g, &task, &spec(), gate, 8)?;
            let mut c = PlasticCircuit::new(&g, 1)?;
            let mut cost_a = PhaseCost::default();
            train(&mut c, &g, gate, &a_items(&task, &spec(), 8), &mut cost_a)?;
            assert_eq!(cost_a, full.cost_a);
            let p = tmp("a");
            c.save(&p)?;
            let mut c = PlasticCircuit::load(&p, &g)?;
            fs::remove_file(&p)?;
            let b = b_items(&task, &spec(), 8);
            let (first, rest) = b.split_at(b.len() / 2);
            let mut cost_b = PhaseCost::default();
            train(&mut c, &g, gate, first, &mut cost_b)?;
            let p = tmp("b");
            c.save(&p)?;
            let mut c = PlasticCircuit::load(&p, &g)?;
            fs::remove_file(&p)?;
            train(&mut c, &g, gate, rest, &mut cost_b)?;
            assert_eq!(cost_b, full.cost_b);
            let (va, vb) = (&task.valence_a, &task.valence_b);
            assert_eq!(
                evaluate_split(&c, &g, &task.test, va, vb)?,
                full.a_test_after_b
            );
            assert_eq!(
                evaluate_split(&c, &g, &task.test, vb, va)?,
                full.b_test_after_b
            );
            assert_eq!(saturated(c.weights()), full.saturated_after_b);
        }
        Ok(())
    }

    #[test]
    fn bootstrap_is_deterministic_and_brackets_the_mean() {
        let v = [0.1, 0.2, -0.05, 0.3, 0.0, 0.12, 0.07];
        let a = bootstrap_ci(&v, 500, BOOT_SEED);
        assert_eq!(a, bootstrap_ci(&v, 500, BOOT_SEED));
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        assert!(a.0 <= mean && mean <= a.1 && a.0 < a.1);
        assert_eq!(bootstrap_ci(&[0.2; 5], 100, 1), (0.2, 0.2));
        assert_eq!(bootstrap_ci(&[], 100, 1), (0.0, 0.0));
        let p = paired(&[0.1; 20]);
        assert!(p.strong());
        assert!(!paired(&[0.04; 20]).strong());
        let mut mixed = vec![0.2; 15];
        mixed.extend([-0.01; 5]);
        assert!(!paired(&mixed).strong());
    }

    #[test]
    fn grid_is_deterministic_and_csv_complete() {
        let a = run_grid(None, &shape(), &spec(), &[1, 2]).unwrap();
        let b = run_grid(None, &shape(), &spec(), &[1, 2]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 2 * ARMS.len());
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        let cols = CSV_HEADER.split(',').count();
        assert!(csv.lines().all(|l| l.split(',').count() == cols));
        assert!(csv.contains(",margin-gated-4,2,"));
        let cmp = preregistered_comparisons(&a);
        assert_eq!(cmp.len(), 2);
        assert!(cmp.iter().all(|c| c.forget.n == 2));
        let id = compare(&a, Gate::Supervised);
        assert_eq!(id.forget.ties, 2);
        assert!(!id.passes() && !id.generalises());
        let _ = readings(&a);
        assert!(a.iter().all(|r| r.topology_unchanged));
        assert!(run_grid(None, &shape(), &spec(), &[]).is_err());
    }
}

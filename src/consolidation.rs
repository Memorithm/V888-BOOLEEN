//! Consolidation-vs-forgetting experiment on a frozen directed graph.
//!
//! Learn task A, then conflicting task B, and measure how much of A survives.
//! The consolidation hypothesis is an engineering rule, not a model of
//! biological memory: edges *visited* while learning A are marked important,
//! and during B each important edge accepts only one in `protect_period`
//! update attempts (`protect_period == 1` disables protection = control).
//! Only original edges are ever touched; anatomy is never modified.
use crate::connectome::Connectome;
use crate::learning_strategies::{
    apply_labeled_episode_gated, comparison_graph, find_four_sources, shuffle_with_seed,
    LearningStrategy,
};
use crate::plasticity::PlasticCircuit;
use crate::BooleanState;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::Path;
use std::time::{Duration, Instant};

const MAGIC: &[u8; 8] = b"V888CON1";
const VERSION: u64 = 1;

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsolidationConfig {
    pub seed: u64,
    pub rounds_a: u32,
    pub rounds_b: u32,
    /// 1 = no protection (supervised control). N>1: protected edges accept 1 of N attempts.
    pub protect_period: u32,
    /// Minimum visits during A for an edge to be protected.
    pub min_visits: u32,
    /// Update rule used for both A and B (same rule, same budget).
    pub strategy: LearningStrategy,
    /// Threshold variant: only edges whose 4-bit weight after A is at least
    /// this value are protected (0 = every visited edge, the original rule).
    pub min_weight: u8,
    pub scenario: Scenario,
}

/// How task B relates to task A.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scenario {
    /// B reuses A's negative source as its positive one (labels conflict).
    Conflict,
    /// B uses two other sources of the same target (no label conflict):
    /// protection should cost nothing here.
    Disjoint,
}

impl Scenario {
    #[must_use]
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Conflict => "conflict",
            Self::Disjoint => "disjoint",
        }
    }
}

impl ConsolidationConfig {
    /// Supervised, conflicting-task defaults (the PR #21 behavior).
    #[must_use]
    pub fn supervised(seed: u64, rounds_a: u32, rounds_b: u32, protect_period: u32) -> Self {
        Self {
            seed,
            rounds_a,
            rounds_b,
            protect_period,
            min_visits: 1,
            strategy: LearningStrategy::Supervised,
            min_weight: 0,
            scenario: Scenario::Conflict,
        }
    }
}

impl ConsolidationConfig {
    fn validate(&self) -> io::Result<()> {
        if self.rounds_a == 0
            || self.rounds_b == 0
            || self.protect_period == 0
            || self.min_visits == 0
            || self.min_weight > 15
        {
            return Err(bad("consolidation config values must be positive"));
        }
        Ok(())
    }
}

/// Learning state that can be saved after task A and resumed for task B.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsolidationState {
    pub circuit: PlasticCircuit,
    /// Saturating per-edge visit counts accumulated while learning A.
    pub visits: Vec<u32>,
    /// Per-edge attempt counters used to gate protected edges during B.
    pub attempts: Vec<u32>,
}

impl ConsolidationState {
    pub fn new(graph: &Connectome) -> io::Result<Self> {
        let circuit = PlasticCircuit::new(graph, 0)?;
        let n = graph.edges();
        Ok(Self {
            circuit,
            visits: vec![0; n],
            attempts: vec![0; n],
        })
    }

    /// Extra memory beyond the 4-bit weights: visit + attempt counters.
    #[must_use]
    pub fn overhead_bytes(&self) -> usize {
        (self.visits.len() + self.attempts.len()) * 4
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut out = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
        out.write_all(MAGIC)?;
        out.write_all(&VERSION.to_le_bytes())?;
        out.write_all(&(self.visits.len() as u64).to_le_bytes())?;
        out.write_all(&self.circuit.topology_fingerprint().to_le_bytes())?;
        out.write_all(self.circuit.weights())?;
        for v in &self.visits {
            out.write_all(&v.to_le_bytes())?;
        }
        for v in &self.attempts {
            out.write_all(&v.to_le_bytes())?;
        }
        out.flush()
    }

    pub fn load(path: &Path, graph: &Connectome) -> io::Result<Self> {
        graph.validate()?;
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        let mut input = BufReader::new(file);
        let mut magic = [0_u8; 8];
        input.read_exact(&mut magic)?;
        let mut num = || -> io::Result<u64> {
            let mut raw = [0_u8; 8];
            input.read_exact(&mut raw)?;
            Ok(u64::from_le_bytes(raw))
        };
        let (version, edges, topology) = (num()?, num()?, num()?);
        let probe = PlasticCircuit::new(graph, 0)?;
        if &magic != MAGIC
            || version != VERSION
            || edges != graph.edges() as u64
            || topology != probe.topology_fingerprint()
            || size != 32 + edges * 9
        {
            return Err(bad("incompatible consolidation checkpoint"));
        }
        let n = edges as usize;
        let mut weights = vec![0_u8; n];
        input.read_exact(&mut weights)?;
        let mut read_u32s = |count: usize| -> io::Result<Vec<u32>> {
            let mut raw = vec![0_u8; count * 4];
            input.read_exact(&mut raw)?;
            Ok(raw
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        let visits = read_u32s(n)?;
        let attempts = read_u32s(n)?;
        Ok(Self {
            circuit: PlasticCircuit::from_weights(graph, weights)?,
            visits,
            attempts,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PhaseStats {
    pub edges_visited: u64,
    pub weights_changed: u64,
    /// Updates suppressed by protection (never counted as visited-and-applied).
    pub blocked: u64,
}

#[derive(Clone, Copy)]
struct Task {
    pos: usize,
    neg: usize,
    target: usize,
}

fn add(total: u64, inc: u64) -> io::Result<u64> {
    total
        .checked_add(inc)
        .ok_or_else(|| bad("counter overflow"))
}

/// One episode: (source, target, fire).
type Ep = (usize, usize, bool);

/// Per-phase gating: `None` during A (record visits), `Some` during B.
struct Gate<'a> {
    period: u32,
    protected: &'a [bool],
}

fn episode(
    state: &mut ConsolidationState,
    graph: &Connectome,
    strategy: LearningStrategy,
    ep: Ep,
    gate: Option<&Gate>,
    stats: &mut PhaseStats,
) -> io::Result<()> {
    let ConsolidationState {
        circuit,
        visits,
        attempts,
    } = state;
    let mut blocked = 0_u64;
    let report =
        apply_labeled_episode_gated(circuit, graph, strategy, ep, &mut |edge| match gate {
            None => {
                visits[edge] = visits[edge].saturating_add(1);
                true
            }
            Some(g) if g.period > 1 && g.protected[edge] => {
                attempts[edge] = attempts[edge].saturating_add(1);
                if attempts[edge] % g.period != 0 {
                    blocked += 1;
                    return false;
                }
                true
            }
            Some(_) => true,
        })?;
    stats.edges_visited = add(stats.edges_visited, report.edges_visited)?;
    stats.weights_changed = add(stats.weights_changed, report.weights_changed)?;
    stats.blocked = add(stats.blocked, blocked)?;
    Ok(())
}

fn run_phase(
    state: &mut ConsolidationState,
    graph: &Connectome,
    strategy: LearningStrategy,
    task: Task,
    rounds: u32,
    seed: u64,
    gate: Option<&Gate>,
) -> io::Result<PhaseStats> {
    let mut order: Vec<bool> = (0..u64::from(rounds) * 2).map(|i| i % 2 == 0).collect();
    shuffle_with_seed(&mut order, seed);
    let mut stats = PhaseStats::default();
    for positive in order {
        let source = if positive { task.pos } else { task.neg };
        episode(
            state,
            graph,
            strategy,
            (source, task.target, positive),
            gate,
            &mut stats,
        )?;
    }
    Ok(stats)
}

/// Returns (correct, presentations, observed target activations).
fn score(circuit: &PlasticCircuit, graph: &Connectome, task: Task) -> io::Result<(u64, u64, u64)> {
    let mut correct = 0;
    let mut active = 0;
    for (source, fire) in [(task.pos, true), (task.neg, false)] {
        let mut input = BooleanState::new(graph.ids.len());
        input.set(source, true).map_err(|_| bad("bad source"))?;
        let out = circuit.predict(graph, &input, 1)?;
        let got = out.get(task.target).map_err(|_| bad("bad target"))?;
        active += out.active_count();
        correct += u64::from(got == fire);
    }
    Ok((correct, 2, active))
}

/// Three distinct sources with a positive-count edge into one target: (a, b, c, target).
fn find_three_sources(graph: &Connectome) -> io::Result<(usize, usize, usize, usize)> {
    let mut found: Vec<Vec<usize>> = vec![Vec::new(); graph.ids.len()];
    for source in 0..graph.ids.len() {
        for edge in graph.offsets[source]..graph.offsets[source + 1] {
            let edge = edge as usize;
            let target = graph.targets[edge] as usize;
            if graph.counts[edge] == 0 || source == target || found[target].contains(&source) {
                continue;
            }
            found[target].push(source);
            if found[target].len() == 3 {
                let f = &found[target];
                return Ok((f[0], f[1], f[2], target));
            }
        }
    }
    Err(bad("no target with three distinct observed sources"))
}

#[derive(Clone, Debug)]
pub struct ConsolidationOutcome {
    pub config: ConsolidationConfig,
    pub source_a_pos_id: u64,
    pub source_shared_id: u64,
    pub source_b_neg_id: u64,
    pub target_id: u64,
    pub a_correct_after_a: u64,
    pub a_correct_after_b: u64,
    pub b_correct_after_b: u64,
    pub presentations: u64,
    pub protected_edges: u64,
    pub phase_a: PhaseStats,
    pub phase_b: PhaseStats,
    pub overhead_bytes: usize,
    pub weight_bytes: usize,
    pub observed_activations: u64,
    pub elapsed: Duration,
    pub topology_unchanged: bool,
    pub final_weights: Vec<u8>,
}

impl ConsolidationOutcome {
    /// Fraction of A kept after B, as a ratio of correct counts.
    #[must_use]
    pub fn retention(&self) -> (u64, u64) {
        (self.a_correct_after_b, self.presentations)
    }
    /// Effective-change rate in B: changed / visited-and-applied.
    #[must_use]
    pub fn recovery_rate_b(&self) -> Option<f64> {
        (self.phase_b.edges_visited > 0)
            .then(|| self.phase_b.weights_changed as f64 / self.phase_b.edges_visited as f64)
    }
}

fn pick_tasks(graph: &Connectome, scenario: Scenario) -> io::Result<(Task, Task)> {
    match scenario {
        Scenario::Conflict => {
            let (a, b, c, target) = find_three_sources(graph)?;
            // A: a fires target, b does not. B (conflicts on b): b fires, c does not.
            Ok((
                Task {
                    pos: a,
                    neg: b,
                    target,
                },
                Task {
                    pos: b,
                    neg: c,
                    target,
                },
            ))
        }
        Scenario::Disjoint => {
            let ([a, b, c, d], target) = find_four_sources(graph)?;
            Ok((
                Task {
                    pos: a,
                    neg: b,
                    target,
                },
                Task {
                    pos: c,
                    neg: d,
                    target,
                },
            ))
        }
    }
}

/// Phase A only: learn task A and record importance. Checkpointable.
pub fn learn_task_a(
    graph: &Connectome,
    cfg: &ConsolidationConfig,
) -> io::Result<(ConsolidationState, PhaseStats)> {
    cfg.validate()?;
    graph.validate()?;
    let (task_a, _) = pick_tasks(graph, cfg.scenario)?;
    let mut state = ConsolidationState::new(graph)?;
    let stats = run_phase(
        &mut state,
        graph,
        cfg.strategy,
        task_a,
        cfg.rounds_a,
        cfg.seed,
        None,
    )?;
    Ok((state, stats))
}

/// Phase B from a (possibly reloaded) post-A state, then evaluate everything.
pub fn learn_task_b_and_evaluate(
    graph: &Connectome,
    cfg: &ConsolidationConfig,
    mut state: ConsolidationState,
    phase_a: PhaseStats,
) -> io::Result<ConsolidationOutcome> {
    cfg.validate()?;
    let (task_a, task_b) = pick_tasks(graph, cfg.scenario)?;
    let start = Instant::now();
    let fingerprint = state.circuit.topology_fingerprint();
    let (a_after_a, presentations, act1) = score(&state.circuit, graph, task_a)?;
    let protected: Vec<bool> = if cfg.protect_period > 1 {
        state
            .visits
            .iter()
            .zip(state.circuit.weights())
            .map(|(&v, &w)| v >= cfg.min_visits && w >= cfg.min_weight)
            .collect()
    } else {
        vec![false; state.visits.len()]
    };
    let protected_edges = protected.iter().filter(|&&p| p).count() as u64;
    let phase_b = run_phase(
        &mut state,
        graph,
        cfg.strategy,
        task_b,
        cfg.rounds_b,
        cfg.seed ^ 0xB,
        Some(&Gate {
            period: cfg.protect_period,
            protected: &protected,
        }),
    )?;
    let (a_after_b, _, act2) = score(&state.circuit, graph, task_a)?;
    let (b_after_b, _, act3) = score(&state.circuit, graph, task_b)?;
    Ok(ConsolidationOutcome {
        config: *cfg,
        source_a_pos_id: graph.ids[task_a.pos],
        source_shared_id: graph.ids[task_a.neg],
        source_b_neg_id: graph.ids[task_b.neg],
        target_id: graph.ids[task_a.target],
        a_correct_after_a: a_after_a,
        a_correct_after_b: a_after_b,
        b_correct_after_b: b_after_b,
        presentations,
        protected_edges,
        phase_a,
        phase_b,
        overhead_bytes: state.overhead_bytes(),
        weight_bytes: state.circuit.weight_bytes(),
        observed_activations: act1 + act2 + act3,
        elapsed: start.elapsed(),
        topology_unchanged: state.circuit.topology_fingerprint() == fingerprint
            && state.circuit.weights().len() == graph.edges(),
        final_weights: state.circuit.weights().to_vec(),
    })
}

pub fn run_experiment(
    graph: &Connectome,
    cfg: &ConsolidationConfig,
) -> io::Result<ConsolidationOutcome> {
    let (state, stats) = learn_task_a(graph, cfg)?;
    learn_task_b_and_evaluate(graph, cfg, state, stats)
}

/// Equal-budget sweep: period 1 (control) first, then each requested period.
pub fn sweep(
    graph: Option<&Connectome>,
    seed: u64,
    rounds_a: u32,
    rounds_b: u32,
    periods: &[u32],
) -> io::Result<Vec<ConsolidationOutcome>> {
    let synthetic;
    let graph = match graph {
        Some(g) => g,
        None => {
            synthetic = comparison_graph();
            &synthetic
        }
    };
    let mut all = Vec::new();
    let mut list = vec![1_u32];
    list.extend(periods.iter().copied().filter(|&p| p != 1));
    for period in list {
        let cfg = ConsolidationConfig::supervised(seed, rounds_a, rounds_b, period);
        all.push(run_experiment(graph, &cfg)?);
    }
    Ok(all)
}

/// One protection variant: `(period, min_weight)`; period 1 = control.
pub type Variant = (u32, u8);

pub const ALL_STRATEGIES: [LearningStrategy; 3] = [
    LearningStrategy::Supervised,
    LearningStrategy::AssociativeCoactivation,
    LearningStrategy::RewardModulated,
];

/// Full harness matrix: scenario x strategy x variant x seed, equal budgets
/// (same rounds_a / rounds_b for every row). Each (scenario, strategy, seed)
/// starts with the unprotected control so rows are directly comparable.
pub fn matrix(
    graph: Option<&Connectome>,
    seeds: &[u64],
    rounds_a: u32,
    rounds_b: u32,
    variants: &[Variant],
    strategies: &[LearningStrategy],
    scenarios: &[Scenario],
) -> io::Result<Vec<ConsolidationOutcome>> {
    let synthetic;
    let graph = match graph {
        Some(g) => g,
        None => {
            synthetic = comparison_graph();
            &synthetic
        }
    };
    let mut list: Vec<Variant> = vec![(1, 0)];
    list.extend(variants.iter().copied().filter(|v| v.0 != 1));
    let mut all = Vec::new();
    for &scenario in scenarios {
        for &strategy in strategies {
            for &seed in seeds {
                for &(period, min_weight) in &list {
                    let mut cfg = ConsolidationConfig::supervised(seed, rounds_a, rounds_b, period);
                    cfg.strategy = strategy;
                    cfg.scenario = scenario;
                    cfg.min_weight = min_weight;
                    all.push(run_experiment(graph, &cfg)?);
                }
            }
        }
    }
    Ok(all)
}

pub const CSV_HEADER: &str = "scenario,strategy,seed,protect_period,min_weight,protected_edges,a_after_a,a_after_b,b_after_b,presentations,a_visited,a_changed,b_visited,b_changed,b_blocked,weight_bytes,overhead_bytes,topology_unchanged,elapsed_us";

/// Stable CSV export; `elapsed_us` is the only non-deterministic column.
#[must_use]
pub fn to_csv(rows: &[ConsolidationOutcome]) -> String {
    let mut out = String::from(CSV_HEADER);
    out.push('\n');
    for r in rows {
        let c = &r.config;
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            c.scenario.as_label(),
            c.strategy.as_label(),
            c.seed,
            c.protect_period,
            c.min_weight,
            r.protected_edges,
            r.a_correct_after_a,
            r.a_correct_after_b,
            r.b_correct_after_b,
            r.presentations,
            r.phase_a.edges_visited,
            r.phase_a.weights_changed,
            r.phase_b.edges_visited,
            r.phase_b.weights_changed,
            r.phase_b.blocked,
            r.weight_bytes,
            r.overhead_bytes,
            r.topology_unchanged,
            r.elapsed.as_micros()
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(period: u32, rounds_b: u32) -> ConsolidationConfig {
        ConsolidationConfig::supervised(3, 8, rounds_b, period)
    }

    #[test]
    fn control_forgets_and_full_protection_retains() {
        let g = comparison_graph();
        let control = run_experiment(&g, &cfg(1, 8)).unwrap();
        assert_eq!(control.a_correct_after_a, 2);
        assert_eq!(control.a_correct_after_b, 1, "unprotected B overwrites A");
        assert_eq!(control.b_correct_after_b, 2);
        // period larger than B budget: protected edges never accept an update
        let frozen = run_experiment(&g, &cfg(1000, 8)).unwrap();
        assert_eq!(frozen.a_correct_after_b, 2);
        assert_eq!(
            frozen.b_correct_after_b, 1,
            "protection costs plasticity for B"
        );
        assert!(frozen.protected_edges > 0);
        assert!(frozen.phase_b.blocked > 0);
        assert!(control.topology_unchanged && frozen.topology_unchanged);
    }

    #[test]
    fn only_original_edges_and_weights_stay_four_bit() {
        let g = comparison_graph();
        let o = run_experiment(&g, &cfg(2, 8)).unwrap();
        assert_eq!(o.final_weights.len(), g.edges());
        assert!(o.final_weights.iter().all(|&w| w <= 15));
    }

    #[test]
    fn checkpoint_resume_is_bit_identical_and_reproducible() {
        let g = comparison_graph();
        let c = cfg(2, 8);
        let direct = run_experiment(&g, &c).unwrap();
        let again = run_experiment(&g, &c).unwrap();
        assert_eq!(direct.final_weights, again.final_weights);
        let (state, stats) = learn_task_a(&g, &c).unwrap();
        let path = std::env::temp_dir().join(format!("v888-con-{}.ckpt", std::process::id()));
        let _ = std::fs::remove_file(&path);
        state.save(&path).unwrap();
        assert!(state.save(&path).is_err(), "never overwrite a checkpoint");
        let loaded = ConsolidationState::load(&path, &g).unwrap();
        assert_eq!(loaded, state);
        let resumed = learn_task_b_and_evaluate(&g, &c, loaded, stats).unwrap();
        assert_eq!(resumed.final_weights, direct.final_weights);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn checkpoint_rejects_wrong_graph_and_truncation() {
        let g = comparison_graph();
        let (state, _) = learn_task_a(&g, &cfg(2, 8)).unwrap();
        let path = std::env::temp_dir().join(format!("v888-con-bad-{}.ckpt", std::process::id()));
        let _ = std::fs::remove_file(&path);
        state.save(&path).unwrap();
        let other = Connectome {
            ids: vec![1, 2, 3],
            offsets: vec![0, 1, 2, 2],
            targets: vec![2, 2],
            counts: vec![1, 1],
        };
        assert!(ConsolidationState::load(&path, &other).is_err());
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert!(ConsolidationState::load(&path, &g).is_err());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn invalid_config_and_graph_without_three_sources_are_rejected() {
        let g = comparison_graph();
        assert!(run_experiment(&g, &cfg(0, 8)).is_err());
        assert!(run_experiment(&g, &cfg(2, 0)).is_err());
        let small = Connectome {
            ids: vec![1, 2, 3],
            offsets: vec![0, 1, 2, 2],
            targets: vec![2, 2],
            counts: vec![1, 1],
        };
        assert!(run_experiment(&small, &cfg(2, 8)).is_err());
    }

    #[test]
    fn sweep_starts_with_control() {
        let r = sweep(None, 1, 8, 8, &[2, 4]).unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].config.protect_period, 1);
        assert_eq!(r[0].protected_edges, 0);
    }

    fn run_matrix(variants: &[Variant], scenarios: &[Scenario]) -> Vec<ConsolidationOutcome> {
        matrix(None, &[1, 2, 3], 8, 8, variants, &ALL_STRATEGIES, scenarios).unwrap()
    }

    #[test]
    fn matrix_covers_every_strategy_with_equal_budget_and_control_first() {
        let rows = run_matrix(&[(2, 0), (1000, 0)], &[Scenario::Conflict]);
        assert_eq!(rows.len(), 3 * 3 * 3);
        assert!(rows
            .iter()
            .all(|r| r.config.rounds_a == 8 && r.config.rounds_b == 8 && r.topology_unchanged));
        assert_eq!(rows[0].config.protect_period, 1);
        for strat in ALL_STRATEGIES {
            assert!(rows.iter().any(|r| r.config.strategy == strat));
        }
    }

    #[test]
    fn every_strategy_forgets_in_control_and_freeze_trades_off_b() {
        let rows = run_matrix(&[(1000, 0)], &[Scenario::Conflict]);
        for r in &rows {
            if r.config.protect_period == 1 {
                assert_eq!(r.a_correct_after_a, 2, "{:?}", r.config.strategy);
                assert_eq!(r.a_correct_after_b, 1, "{:?}", r.config.strategy);
            } else {
                assert_eq!(r.a_correct_after_b, 2, "{:?}", r.config.strategy);
                assert_eq!(r.b_correct_after_b, 1, "{:?}", r.config.strategy);
            }
        }
    }

    #[test]
    fn disjoint_tasks_do_not_interfere_even_with_freeze() {
        let rows = run_matrix(&[(2, 0), (1000, 0)], &[Scenario::Disjoint]);
        for r in &rows {
            assert_eq!(r.a_correct_after_b, 2, "{:?}", r.config);
            assert_eq!(r.b_correct_after_b, 2, "{:?}", r.config);
        }
    }

    #[test]
    fn threshold_variant_protects_only_strong_edges_and_is_deterministic() {
        let g = comparison_graph();
        let mut lo = ConsolidationConfig::supervised(1, 8, 8, 1000);
        let all = run_experiment(&g, &lo).unwrap();
        lo.min_weight = 15;
        let strong = run_experiment(&g, &lo).unwrap();
        assert!(strong.protected_edges <= all.protected_edges);
        let again = run_experiment(&g, &lo).unwrap();
        assert_eq!(strong.final_weights, again.final_weights);
        lo.min_weight = 16;
        assert!(run_experiment(&g, &lo).is_err());
    }

    #[test]
    fn associative_and_reward_checkpoint_resume_bit_identical() {
        let g = comparison_graph();
        for strat in [
            LearningStrategy::AssociativeCoactivation,
            LearningStrategy::RewardModulated,
        ] {
            let mut c = cfg(2, 8);
            c.strategy = strat;
            let direct = run_experiment(&g, &c).unwrap();
            let (state, stats) = learn_task_a(&g, &c).unwrap();
            let path = std::env::temp_dir().join(format!(
                "v888-con-{}-{}.ckpt",
                strat.as_label(),
                std::process::id()
            ));
            let _ = std::fs::remove_file(&path);
            state.save(&path).unwrap();
            let loaded = ConsolidationState::load(&path, &g).unwrap();
            let resumed = learn_task_b_and_evaluate(&g, &c, loaded, stats).unwrap();
            assert_eq!(resumed.final_weights, direct.final_weights);
            std::fs::remove_file(&path).unwrap();
        }
    }

    #[test]
    fn csv_has_header_and_one_line_per_row() {
        let rows = run_matrix(&[(2, 0)], &[Scenario::Conflict]);
        let csv = to_csv(&rows);
        assert_eq!(csv.lines().count(), rows.len() + 1);
        assert!(csv.starts_with(CSV_HEADER));
        let cols = CSV_HEADER.split(',').count();
        assert!(csv.lines().all(|l| l.split(',').count() == cols));
    }
}

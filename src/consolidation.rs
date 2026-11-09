//! Consolidation-vs-forgetting experiment on a frozen directed graph.
//!
//! Learn task A, then conflicting task B, and measure how much of A survives.
//! The consolidation hypothesis is an engineering rule, not a model of
//! biological memory: edges *visited* while learning A are marked important,
//! and during B each important edge accepts only one in `protect_period`
//! update attempts (`protect_period == 1` disables protection = control).
//! Only original edges are ever touched; anatomy is never modified.
use crate::connectome::Connectome;
use crate::learning_strategies::{comparison_graph, shuffle_with_seed};
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
}

impl ConsolidationConfig {
    fn validate(&self) -> io::Result<()> {
        if self.rounds_a == 0
            || self.rounds_b == 0
            || self.protect_period == 0
            || self.min_visits == 0
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

/// One supervised episode: (source, target, fire).
type Ep = (usize, usize, bool);

/// `gate` = (period, min_visits), applied only during B.
fn episode(
    state: &mut ConsolidationState,
    graph: &Connectome,
    ep: Ep,
    phase_a: bool,
    gate: (u32, u32),
    stats: &mut PhaseStats,
) -> io::Result<()> {
    let (source, target, fire) = ep;
    for edge in graph.offsets[source]..graph.offsets[source + 1] {
        let edge = edge as usize;
        let reinforce = fire && graph.targets[edge] as usize == target;
        if phase_a {
            state.visits[edge] = state.visits[edge].saturating_add(1);
        } else if gate.0 > 1 && state.visits[edge] >= gate.1 {
            state.attempts[edge] = state.attempts[edge].saturating_add(1);
            if state.attempts[edge] % gate.0 != 0 {
                stats.blocked = add(stats.blocked, 1)?;
                continue;
            }
        }
        stats.edges_visited = add(stats.edges_visited, 1)?;
        if state.circuit.apply_edge_step(edge, reinforce)? {
            stats.weights_changed = add(stats.weights_changed, 1)?;
        }
    }
    Ok(())
}

fn run_phase(
    state: &mut ConsolidationState,
    graph: &Connectome,
    task: Task,
    rounds: u32,
    seed: u64,
    phase_a: bool,
    gate: (u32, u32),
) -> io::Result<PhaseStats> {
    let mut order: Vec<bool> = (0..u64::from(rounds) * 2).map(|i| i % 2 == 0).collect();
    shuffle_with_seed(&mut order, seed);
    let mut stats = PhaseStats::default();
    for positive in order {
        let source = if positive { task.pos } else { task.neg };
        episode(
            state,
            graph,
            (source, task.target, positive),
            phase_a,
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

fn pick_tasks(graph: &Connectome) -> io::Result<(Task, Task)> {
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

/// Phase A only: learn task A and record importance. Checkpointable.
pub fn learn_task_a(
    graph: &Connectome,
    cfg: &ConsolidationConfig,
) -> io::Result<(ConsolidationState, PhaseStats)> {
    cfg.validate()?;
    graph.validate()?;
    let (task_a, _) = pick_tasks(graph)?;
    let mut state = ConsolidationState::new(graph)?;
    let stats = run_phase(
        &mut state,
        graph,
        task_a,
        cfg.rounds_a,
        cfg.seed,
        true,
        (1, 1),
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
    let (task_a, task_b) = pick_tasks(graph)?;
    let start = Instant::now();
    let fingerprint = state.circuit.topology_fingerprint();
    let (a_after_a, presentations, act1) = score(&state.circuit, graph, task_a)?;
    let protected_edges = if cfg.protect_period > 1 {
        state
            .visits
            .iter()
            .filter(|&&v| v >= cfg.min_visits)
            .count() as u64
    } else {
        0
    };
    let phase_b = run_phase(
        &mut state,
        graph,
        task_b,
        cfg.rounds_b,
        cfg.seed ^ 0xB,
        false,
        (cfg.protect_period, cfg.min_visits),
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
        let cfg = ConsolidationConfig {
            seed,
            rounds_a,
            rounds_b,
            protect_period: period,
            min_visits: 1,
        };
        all.push(run_experiment(graph, &cfg)?);
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(period: u32, rounds_b: u32) -> ConsolidationConfig {
        ConsolidationConfig {
            seed: 3,
            rounds_a: 8,
            rounds_b,
            protect_period: period,
            min_visits: 1,
        }
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
}

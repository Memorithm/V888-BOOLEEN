//! B888-4 scale experiment: consolidation vs forgetting on a LARGER GENERATED
//! graph with many overlapping edges.
//!
//! **This graph is synthetic.** No real BANC V888 graph is available in this
//! repository or on the box (`docs/SOURCES.md`: no raw data is shipped), so the
//! real-graph run is blocked on a user-built `.v888b` (see `docs/SOURCES.md`).
//! Labels are artificial; nothing here is a claim about biology or BANC.
//!
//! Protocol (fixed before any result was looked at):
//! * graph: `sources` source neurons and `targets` target neurons; each target
//!   receives `fan_in` distinct random sources (contact count 1..=3). Because a
//!   source projects to several targets, one episode touches many edges and
//!   tasks on different targets interfere through shared sources.
//! * task A, per target `t` with sorted inputs `in[0..]`: `in[0]` must fire `t`,
//!   `in[1]` must not.
//! * task B: for the first `round(conflict_pct% * targets)` targets it is the
//!   conflict `in[1]` fires / `in[2]` silent; for the others it is
//!   `in[2]` fires / `in[3]` silent.
//! * equal budgets: same rounds, graph, seed and episode order for every
//!   protection variant; the unprotected supervised-rule control is variant
//!   `(1, 0)` and is always run first.
//! * rule under test: edges visited while learning A (and, for the threshold
//!   variant, with 4-bit weight >= `min_weight` after A) accept one in
//!   `period` update attempts during B. Only original edges are touched.
use crate::connectome::Connectome;
use crate::learning_strategies::{
    apply_labeled_episode_gated, shuffle_with_seed, LearningStrategy,
};
use crate::plasticity::PlasticCircuit;
use crate::BooleanState;
use std::io::{self, ErrorKind};
use std::time::{Duration, Instant};

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Deterministic generator parameters. `conflict_pct` is in 0..=100.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScaleSpec {
    pub sources: usize,
    pub targets: usize,
    pub fan_in: usize,
    pub conflict_pct: u32,
    pub rounds_a: u32,
    pub rounds_b: u32,
}

impl ScaleSpec {
    fn validate(&self) -> io::Result<()> {
        if self.sources < self.fan_in
            || self.fan_in < 4
            || self.targets == 0
            || self.conflict_pct > 100
            || self.rounds_a == 0
            || self.rounds_b == 0
            || self.sources + self.targets > 100_000
        {
            return Err(bad("invalid scale spec (need fan_in>=4, sources>=fan_in)"));
        }
        Ok(())
    }
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

/// Generated graph plus, per target, its sorted distinct source inputs.
pub struct ScaleGraph {
    pub graph: Connectome,
    pub inputs: Vec<Vec<usize>>, // indexed by target ordinal 0..targets
    pub target_index: Vec<usize>,
}

/// Build the generated graph. Neuron ids are `0..sources+targets`; indices
/// equal ids. Edges are unique `(source, target)` pairs.
pub fn generate_graph(spec: &ScaleSpec, graph_seed: u64) -> io::Result<ScaleGraph> {
    spec.validate()?;
    let mut rng = Lcg(graph_seed ^ 0xB888_4000);
    let mut per_source: Vec<Vec<(u32, u64)>> = vec![Vec::new(); spec.sources];
    let mut inputs = Vec::with_capacity(spec.targets);
    for t in 0..spec.targets {
        let target = (spec.sources + t) as u32;
        let mut chosen: Vec<usize> = Vec::with_capacity(spec.fan_in);
        while chosen.len() < spec.fan_in {
            let s = (rng.next() as usize) % spec.sources;
            if !chosen.contains(&s) {
                chosen.push(s);
            }
        }
        chosen.sort_unstable();
        for &s in &chosen {
            per_source[s].push((target, 1 + rng.next() % 3));
        }
        inputs.push(chosen);
    }
    let mut ids = Vec::new();
    for i in 0..(spec.sources + spec.targets) {
        ids.push(i as u64);
    }
    let mut offsets = vec![0_u64];
    let mut targets = Vec::new();
    let mut counts = Vec::new();
    for edges in per_source.iter_mut() {
        edges.sort_unstable();
        for &(t, c) in edges.iter() {
            targets.push(t);
            counts.push(c);
        }
        offsets.push(targets.len() as u64);
    }
    for _ in 0..spec.targets {
        offsets.push(targets.len() as u64);
    }
    let graph = Connectome {
        ids,
        offsets,
        targets,
        counts,
    };
    graph.validate()?;
    let target_index = (0..spec.targets).map(|t| spec.sources + t).collect();
    Ok(ScaleGraph {
        graph,
        inputs,
        target_index,
    })
}

#[derive(Clone, Copy, Debug)]
struct Item {
    pos: usize,
    neg: usize,
    target: usize,
}

fn tasks(spec: &ScaleSpec, g: &ScaleGraph) -> (Vec<Item>, Vec<Item>, usize) {
    let n_conflict = (spec.targets * spec.conflict_pct as usize + 50) / 100;
    let mut a = Vec::new();
    let mut b = Vec::new();
    for t in 0..spec.targets {
        let i = &g.inputs[t];
        let target = g.target_index[t];
        a.push(Item {
            pos: i[0],
            neg: i[1],
            target,
        });
        b.push(if t < n_conflict {
            Item {
                pos: i[1],
                neg: i[2],
                target,
            }
        } else {
            Item {
                pos: i[2],
                neg: i[3],
                target,
            }
        });
    }
    (a, b, n_conflict)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Phase {
    pub visited: u64,
    pub changed: u64,
    pub blocked: u64,
}

fn run_phase(
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    strategy: LearningStrategy,
    items: &[Item],
    budget: (u32, u64),
    gate: Option<(&[bool], u32, &mut [u32])>,
    visits: Option<&mut [u32]>,
) -> io::Result<Phase> {
    let (rounds, seed) = budget;
    let mut order: Vec<(usize, bool)> = Vec::new();
    for _ in 0..rounds {
        for k in 0..items.len() {
            order.push((k, true));
            order.push((k, false));
        }
    }
    shuffle_with_seed(&mut order, seed);
    let mut phase = Phase::default();
    let mut visits = visits;
    let mut gate = gate;
    for (k, positive) in order {
        let it = items[k];
        let source = if positive { it.pos } else { it.neg };
        let mut blocked = 0_u64;
        let report = apply_labeled_episode_gated(
            circuit,
            graph,
            strategy,
            (source, it.target, positive),
            &mut |edge| {
                if let Some(v) = visits.as_deref_mut() {
                    v[edge] = v[edge].saturating_add(1);
                }
                if let Some((protected, period, attempts)) = gate.as_mut() {
                    if *period > 1 && protected[edge] {
                        attempts[edge] = attempts[edge].saturating_add(1);
                        if attempts[edge] % *period != 0 {
                            blocked += 1;
                            return false;
                        }
                    }
                }
                true
            },
        )?;
        phase.visited += report.edges_visited;
        phase.changed += report.weights_changed;
        phase.blocked += blocked;
    }
    Ok(phase)
}

fn score(circuit: &PlasticCircuit, graph: &Connectome, items: &[Item]) -> io::Result<u64> {
    let mut correct = 0;
    for it in items {
        for (source, fire) in [(it.pos, true), (it.neg, false)] {
            let mut input = BooleanState::new(graph.ids.len());
            input.set(source, true).map_err(|_| bad("bad source"))?;
            let out = circuit.predict(graph, &input, 1)?;
            correct += u64::from(out.get(it.target).map_err(|_| bad("bad target"))? == fire);
        }
    }
    Ok(correct)
}

#[derive(Clone, Debug)]
pub struct ScaleRow {
    pub strategy: LearningStrategy,
    pub seed: u64,
    pub conflict_pct: u32,
    pub period: u32,
    pub min_weight: u8,
    pub edges: usize,
    pub protected_edges: u64,
    pub presentations: u64,
    pub a_after_a: u64,
    pub a_after_b: u64,
    pub b_after_b: u64,
    pub b_phase: Phase,
    pub weight_bytes: usize,
    pub overhead_bytes: usize,
    pub topology_unchanged: bool,
    pub elapsed: Duration,
}

/// One run: learn A, (optionally protect), learn B, evaluate.
pub fn run_one(
    spec: &ScaleSpec,
    sg: &ScaleGraph,
    strategy: LearningStrategy,
    seed: u64,
    period: u32,
    min_weight: u8,
) -> io::Result<ScaleRow> {
    spec.validate()?;
    if period == 0 || min_weight > 15 {
        return Err(bad("period must be >=1 and min_weight <=15"));
    }
    let graph = &sg.graph;
    let (a, b, _) = tasks(spec, sg);
    let start = Instant::now();
    let mut circuit = PlasticCircuit::new(graph, 0)?;
    let fingerprint = circuit.topology_fingerprint();
    let mut visits = vec![0_u32; graph.edges()];
    let mut attempts = vec![0_u32; graph.edges()];
    run_phase(
        &mut circuit,
        graph,
        strategy,
        &a,
        (spec.rounds_a, seed),
        None,
        Some(&mut visits),
    )?;
    let a_after_a = score(&circuit, graph, &a)?;
    let protected: Vec<bool> = if period > 1 {
        visits
            .iter()
            .zip(circuit.weights())
            .map(|(&v, &w)| v >= 1 && w >= min_weight)
            .collect()
    } else {
        vec![false; graph.edges()]
    };
    let protected_edges = protected.iter().filter(|&&p| p).count() as u64;
    let b_phase = run_phase(
        &mut circuit,
        graph,
        strategy,
        &b,
        (spec.rounds_b, seed ^ 0xB),
        Some((&protected, period, &mut attempts)),
        None,
    )?;
    Ok(ScaleRow {
        strategy,
        seed,
        conflict_pct: spec.conflict_pct,
        period,
        min_weight,
        edges: graph.edges(),
        protected_edges,
        presentations: 2 * spec.targets as u64,
        a_after_a,
        a_after_b: score(&circuit, graph, &a)?,
        b_after_b: score(&circuit, graph, &b)?,
        b_phase,
        weight_bytes: circuit.weight_bytes(),
        overhead_bytes: (visits.len() + attempts.len()) * 4,
        topology_unchanged: circuit.topology_fingerprint() == fingerprint
            && circuit.weights().len() == graph.edges(),
        elapsed: start.elapsed(),
    })
}

/// `(period, min_weight)`; `(1, 0)` is the control and is always run first.
pub type Variant = (u32, u8);

/// Full grid: conflict_pct x strategy x seed x variant. The graph for a seed is
/// generated from that seed, so seeds vary graph AND episode order.
pub fn run_grid(
    base: &ScaleSpec,
    conflict_pcts: &[u32],
    seeds: &[u64],
    strategies: &[LearningStrategy],
    variants: &[Variant],
) -> io::Result<Vec<ScaleRow>> {
    let mut list: Vec<Variant> = vec![(1, 0)];
    list.extend(variants.iter().copied().filter(|v| v.0 != 1));
    let mut rows = Vec::new();
    for &pct in conflict_pcts {
        let spec = ScaleSpec {
            conflict_pct: pct,
            ..*base
        };
        for &seed in seeds {
            let sg = generate_graph(&spec, seed)?;
            for &strategy in strategies {
                for &(period, w) in &list {
                    rows.push(run_one(&spec, &sg, strategy, seed, period, w)?);
                }
            }
        }
    }
    Ok(rows)
}

pub const CSV_HEADER: &str = "conflict_pct,strategy,seed,protect_period,min_weight,edges,protected_edges,presentations,a_after_a,a_after_b,b_after_b,b_visited,b_changed,b_blocked,weight_bytes,overhead_bytes,topology_unchanged,elapsed_us";

#[must_use]
pub fn to_csv(rows: &[ScaleRow]) -> String {
    let mut out = String::from(CSV_HEADER);
    out.push('\n');
    for r in rows {
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            r.conflict_pct,
            r.strategy.as_label(),
            r.seed,
            r.period,
            r.min_weight,
            r.edges,
            r.protected_edges,
            r.presentations,
            r.a_after_a,
            r.a_after_b,
            r.b_after_b,
            r.b_phase.visited,
            r.b_phase.changed,
            r.b_phase.blocked,
            r.weight_bytes,
            r.overhead_bytes,
            r.topology_unchanged,
            r.elapsed.as_micros()
        ));
    }
    out
}

/// Per (conflict_pct, strategy, variant) means over seeds, with the paired
/// difference to the control of the same seed. Fractions of presentations.
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    pub conflict_pct: u32,
    pub strategy: LearningStrategy,
    pub period: u32,
    pub min_weight: u8,
    pub seeds: usize,
    pub a_after_a: f64,
    pub a_after_b: f64,
    pub b_after_b: f64,
    pub d_a_after_b: f64,
    pub d_b_after_b: f64,
    pub protected_edges: f64,
}

#[must_use]
pub fn summarize(rows: &[ScaleRow]) -> Vec<Summary> {
    let mut keys: Vec<(u32, &'static str, u32, u8)> = Vec::new();
    for r in rows {
        let k = (
            r.conflict_pct,
            r.strategy.as_label(),
            r.period,
            r.min_weight,
        );
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    let mut out = Vec::new();
    for (pct, label, period, w) in keys {
        let sel: Vec<&ScaleRow> = rows
            .iter()
            .filter(|r| {
                r.conflict_pct == pct
                    && r.strategy.as_label() == label
                    && r.period == period
                    && r.min_weight == w
            })
            .collect();
        let n = sel.len() as f64;
        let frac = |x: u64, r: &ScaleRow| x as f64 / r.presentations as f64;
        let (mut aa, mut ab, mut bb, mut da, mut db, mut pe) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for r in &sel {
            let control = rows
                .iter()
                .find(|c| {
                    c.conflict_pct == pct
                        && c.strategy == r.strategy
                        && c.seed == r.seed
                        && c.period == 1
                })
                .unwrap_or(r);
            aa += frac(r.a_after_a, r);
            ab += frac(r.a_after_b, r);
            bb += frac(r.b_after_b, r);
            da += frac(r.a_after_b, r) - frac(control.a_after_b, control);
            db += frac(r.b_after_b, r) - frac(control.b_after_b, control);
            pe += r.protected_edges as f64;
        }
        out.push(Summary {
            conflict_pct: pct,
            strategy: sel[0].strategy,
            period,
            min_weight: w,
            seeds: sel.len(),
            a_after_a: aa / n,
            a_after_b: ab / n,
            b_after_b: bb / n,
            d_a_after_b: da / n,
            d_b_after_b: db / n,
            protected_edges: pe / n,
        });
    }
    out
}

/// Pre-stated success rule: protection is "useful" only if it improves mean
/// A-after-B by >= 0.10 over the control while mean B-after-B drops by < 0.10.
#[must_use]
pub fn is_useful(s: &Summary) -> bool {
    s.period > 1 && s.d_a_after_b >= 0.10 && s.d_b_after_b > -0.10
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(pct: u32) -> ScaleSpec {
        ScaleSpec {
            sources: 40,
            targets: 12,
            fan_in: 6,
            conflict_pct: pct,
            rounds_a: 8,
            rounds_b: 8,
        }
    }

    #[test]
    fn generated_graph_is_valid_deterministic_and_overlapping() {
        let s = spec(50);
        let g1 = generate_graph(&s, 7).unwrap();
        let g2 = generate_graph(&s, 7).unwrap();
        assert_eq!(g1.graph, g2.graph);
        assert_ne!(g1.graph, generate_graph(&s, 8).unwrap().graph);
        assert_eq!(g1.graph.edges(), 12 * 6);
        // overlap: at least one source projects to several targets
        let g = &g1.graph;
        assert!((0..40).any(|x| g.offsets[x + 1] - g.offsets[x] > 1));
        assert!(g.counts.iter().all(|&c| (1..=3).contains(&c)));
    }

    #[test]
    fn invalid_specs_rejected() {
        let mut s = spec(50);
        s.fan_in = 3;
        assert!(generate_graph(&s, 1).is_err());
        let mut s = spec(101);
        s.fan_in = 6;
        assert!(generate_graph(&s, 1).is_err());
    }

    #[test]
    fn topology_unchanged_and_runs_reproducible() {
        let s = spec(50);
        let sg = generate_graph(&s, 3).unwrap();
        let r1 = run_one(&s, &sg, LearningStrategy::Supervised, 3, 4, 0).unwrap();
        let r2 = run_one(&s, &sg, LearningStrategy::Supervised, 3, 4, 0).unwrap();
        assert!(r1.topology_unchanged);
        assert_eq!(
            (r1.a_after_a, r1.a_after_b, r1.b_after_b, r1.b_phase),
            (r2.a_after_a, r2.a_after_b, r2.b_after_b, r2.b_phase)
        );
        assert!(run_one(&s, &sg, LearningStrategy::Supervised, 3, 0, 0).is_err());
    }

    #[test]
    fn control_runs_first_and_budgets_are_equal() {
        let rows = run_grid(
            &spec(50),
            &[0, 50],
            &[1, 2],
            &[LearningStrategy::Supervised],
            &[(4, 0), (1000, 0)],
        )
        .unwrap();
        assert_eq!(rows.len(), 2 * 2 * 3);
        assert!(rows.iter().all(|r| r.presentations == 24));
        assert_eq!(rows[0].period, 1);
        // equal budget: unprotected B visits >= any protected B visits
        for chunk in rows.chunks(3) {
            assert!(chunk[0].b_phase.visited >= chunk[2].b_phase.visited);
            assert_eq!(chunk[0].b_phase.blocked, 0);
        }
    }

    #[test]
    fn freeze_blocks_updates_and_no_conflict_is_free_to_learn_b() {
        let s = spec(100);
        let sg = generate_graph(&s, 5).unwrap();
        let ctl = run_one(&s, &sg, LearningStrategy::Supervised, 5, 1, 0).unwrap();
        let frz = run_one(&s, &sg, LearningStrategy::Supervised, 5, 1000, 0).unwrap();
        assert!(frz.protected_edges > 0);
        assert!(frz.b_phase.blocked > 0);
        assert!(frz.b_phase.changed <= ctl.b_phase.changed);
    }

    #[test]
    fn summary_pairs_with_control_and_csv_shape() {
        let rows = run_grid(
            &spec(50),
            &[50],
            &[1, 2, 3],
            &[
                LearningStrategy::Supervised,
                LearningStrategy::RewardModulated,
            ],
            &[(8, 0)],
        )
        .unwrap();
        let sums = summarize(&rows);
        assert_eq!(sums.len(), 4);
        for s in &sums {
            assert_eq!(s.seeds, 3);
            if s.period == 1 {
                assert_eq!(s.d_a_after_b, 0.0);
                assert!(!is_useful(s));
            }
        }
        let csv = to_csv(&rows);
        assert_eq!(csv.lines().count(), rows.len() + 1);
        assert!(csv.starts_with(CSV_HEADER));
    }
}

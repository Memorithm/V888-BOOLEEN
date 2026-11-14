//! B888-4: **importance-weighted protection** (edges that carried the most
//! task-A credit are frozen during task B — an EWC-style analogue for 4-bit
//! Boolean-graph edge weights) and **sparse task-gated plasticity** (disjoint
//! per-task plastic edge subsets), on the supervised control and the #38
//! `group-8` / `per-target` graded-reward learners, against the #39
//! `consolidate` reference. Equal budget, no rehearsal.
//!
//! **Synthetic data only.** No real BANC V888 graph (`.v888b`, Harvard
//! Dataverse DOI 10.7910/DVN/7WTH1N; Bates et al. 2026, BANC/FlyWire
//! consortium, see `docs/SOURCES.md`) is present on the build machine. These
//! are engineering tests of update rules on a Boolean graph; nothing here is
//! a claim about synapses, biology, insect memory or BANC physiology.
//!
//! Protocol and constants are pre-registered in `docs/IMPORTANCE_PROTECTION.md`
//! (committed before any run). Only weights of existing edges change.
use crate::connectome::Connectome;
use crate::consolidated_graded::{self as cg, b_stream, train, Item, Learner, PhaseCost};
use crate::generalization_bench::{
    self as gb, build_task, episode_order, evaluate, Confusion, GenSpec, GraphShape, Spread,
};
use crate::plasticity::PlasticCircuit;
use crate::reward_limits as rl;
use std::fs;
use std::io::{self, ErrorKind};
use std::path::Path;

fn bad(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Salt of the per-edge task-gate mask.
pub const GATE_SALT: u64 = 0x6A7E_D00D;
/// #39 protection period, reproduced by `consolidate-39`.
pub const CONSOLIDATE_PERIOD: u32 = cg::PROTECT_PERIOD;

/// Forgetting-reduction mechanism.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mechanism {
    None,
    Consolidate39,
    /// Freeze the top `pct` % of all edges by task-A credit during B.
    Importance(u32),
    TaskGated,
}

impl Mechanism {
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::None => "none".to_string(),
            Self::Consolidate39 => "consolidate-39".to_string(),
            Self::Importance(p) => format!("importance-{p}"),
            Self::TaskGated => "task-gated".to_string(),
        }
    }
}

pub const MECHANISMS: [Mechanism; 5] = [
    Mechanism::None,
    Mechanism::Consolidate39,
    Mechanism::Importance(25),
    Mechanism::Importance(50),
    Mechanism::TaskGated,
];

/// SplitMix64 finaliser.
#[must_use]
pub fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Task gate: `true` = plastic in B (and frozen in A).
#[must_use]
pub fn gate_b(edge: usize, seed: u64) -> bool {
    mix(edge as u64 ^ seed ^ GATE_SALT) & 1 == 1
}

/// Importance-ranked frozen set: the top `ceil(pct · edges / 100)` edges
/// with importance > 0 (importance = A credit if the weight moved in A).
#[must_use]
pub fn importance_mask(credit: &[u32], moved: &[bool], pct: u32) -> Vec<bool> {
    let n = credit.len();
    let k = (n * pct as usize).div_ceil(100);
    let mut ranked: Vec<usize> = (0..n).filter(|&e| moved[e] && credit[e] > 0).collect();
    ranked.sort_by(|&a, &b| credit[b].cmp(&credit[a]).then(a.cmp(&b)));
    let mut mask = vec![false; n];
    for &e in ranked.iter().take(k) {
        mask[e] = true;
    }
    mask
}

const MASK_MAGIC: &[u8; 8] = b"V888IMP1";

/// Save a per-edge protection mask (checkpoint sidecar of a `.v888pls`).
pub fn save_mask(path: &Path, mask: &[bool]) -> io::Result<()> {
    let mut out = Vec::with_capacity(16 + mask.len());
    out.extend_from_slice(MASK_MAGIC);
    out.extend_from_slice(&(mask.len() as u64).to_le_bytes());
    out.extend(mask.iter().map(|&b| u8::from(b)));
    fs::write(path, out)
}

/// Load a mask written by [`save_mask`], validated against the graph.
pub fn load_mask(path: &Path, graph: &Connectome) -> io::Result<Vec<bool>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 16 || &bytes[..8] != MASK_MAGIC {
        return Err(bad("not a V888IMP1 mask"));
    }
    let mut len = [0_u8; 8];
    len.copy_from_slice(&bytes[8..16]);
    let n = usize::try_from(u64::from_le_bytes(len)).map_err(|_| bad("mask length"))?;
    if n != graph.edges() || bytes.len() != 16 + n {
        return Err(bad("mask length does not match graph"));
    }
    bytes[16..]
        .iter()
        .map(|&b| match b {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(bad("mask byte not 0/1")),
        })
        .collect()
}

/// One (learner, mechanism, seed) measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub learner: Learner,
    pub mechanism: Mechanism,
    pub seed: u64,
    pub a_train_after_a: Confusion,
    pub a_test_after_a: Confusion,
    pub a_test_after_b: Confusion,
    pub b_test_after_b: Confusion,
    /// Edges protected (frozen or gated) during B.
    pub protected_edges: u64,
    pub cost_a: PhaseCost,
    pub cost_b: PhaseCost,
    /// Attempts refused during A (task-gated only).
    pub a_blocked: u64,
    pub topology_unchanged: bool,
}

impl Row {
    #[must_use]
    pub fn forgetting(&self) -> f64 {
        self.a_test_after_a.balanced() - self.a_test_after_b.balanced()
    }
}

/// B-phase gate for a mechanism, given the post-A state.
struct BGate {
    mechanism: Mechanism,
    protected: Vec<bool>,
    seed: u64,
    attempts: Vec<u32>,
    protected_attempts: u64,
    blocked: u64,
}

impl BGate {
    fn allow(&mut self, edge: usize) -> bool {
        let pass = match self.mechanism {
            Mechanism::None => return true,
            Mechanism::TaskGated => {
                if gate_b(edge, self.seed) {
                    return true;
                }
                false
            }
            Mechanism::Consolidate39 => {
                if !self.protected[edge] {
                    return true;
                }
                self.attempts[edge] = self.attempts[edge].saturating_add(1);
                self.attempts[edge] % CONSOLIDATE_PERIOD == 0
            }
            Mechanism::Importance(_) => {
                if !self.protected[edge] {
                    return true;
                }
                false
            }
        };
        self.protected_attempts += 1;
        if !pass {
            self.blocked += 1;
        }
        pass
    }
}

fn protected_after_a(
    mechanism: Mechanism,
    weights: &[u8],
    init: u8,
    credit: &[u32],
    seed: u64,
) -> Vec<bool> {
    let moved: Vec<bool> = weights.iter().map(|&w| w != init).collect();
    match mechanism {
        Mechanism::None => vec![false; weights.len()],
        Mechanism::Consolidate39 => moved,
        Mechanism::Importance(p) => importance_mask(credit, &moved, p),
        Mechanism::TaskGated => (0..weights.len()).map(|e| !gate_b(e, seed)).collect(),
    }
}

fn a_items<'a>(task: &'a gb::Task, spec: &GenSpec, seed: u64) -> Vec<Item<'a>> {
    episode_order(task.a_train.len(), spec.rounds_a, seed)
        .into_iter()
        .map(|i| (task.a_train[i].as_slice(), task.valence_a.as_slice()))
        .collect()
}

/// Task A with credit recording; returns (cost, credit, blocked).
fn train_a(
    learner: Learner,
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    items: &[Item<'_>],
    mechanism: Mechanism,
    seed: u64,
) -> io::Result<(PhaseCost, Vec<u32>, u64)> {
    let mut credit = vec![0_u32; graph.edges()];
    let mut blocked = 0_u64;
    let gated = mechanism == Mechanism::TaskGated;
    let cost = train(learner, circuit, graph, items, &mut |e| {
        if gated && gate_b(e, seed) {
            blocked += 1;
            return false;
        }
        credit[e] = credit[e].saturating_add(1);
        true
    })?;
    Ok((cost, credit, blocked))
}

fn train_b(
    learner: Learner,
    circuit: &mut PlasticCircuit,
    graph: &Connectome,
    items: &[Item<'_>],
    mechanism: Mechanism,
    protected: Vec<bool>,
    seed: u64,
) -> io::Result<PhaseCost> {
    let mut gate = BGate {
        mechanism,
        protected,
        seed,
        attempts: vec![0; graph.edges()],
        protected_attempts: 0,
        blocked: 0,
    };
    let mut cost = train(learner, circuit, graph, items, &mut |e| gate.allow(e))?;
    cost.protected_attempts = gate.protected_attempts;
    cost.blocked = gate.blocked;
    Ok(cost)
}

/// Run A then B for one (learner, mechanism).
pub fn run_arm(
    graph: &Connectome,
    task: &gb::Task,
    spec: &GenSpec,
    learner: Learner,
    mechanism: Mechanism,
    seed: u64,
) -> io::Result<Row> {
    let original = graph.clone();
    let mut circuit = PlasticCircuit::new(graph, spec.init_weight)?;
    let fingerprint = circuit.topology_fingerprint();
    let a = a_items(task, spec, seed);
    let (cost_a, credit, a_blocked) = train_a(learner, &mut circuit, graph, &a, mechanism, seed)?;
    let a_train_after_a = evaluate(&circuit, graph, &task.a_train, &task.valence_a)?;
    let a_test_after_a = evaluate(&circuit, graph, &task.test, &task.valence_a)?;
    let protected = protected_after_a(
        mechanism,
        circuit.weights(),
        spec.init_weight,
        &credit,
        seed,
    );
    let protected_edges = protected.iter().filter(|&&p| p).count() as u64;
    let b = b_stream(task, spec, seed, false);
    let cost_b = train_b(learner, &mut circuit, graph, &b, mechanism, protected, seed)?;
    let a_test_after_b = evaluate(&circuit, graph, &task.test, &task.valence_a)?;
    let b_test_after_b = evaluate(&circuit, graph, &task.test, &task.valence_b)?;
    Ok(Row {
        learner,
        mechanism,
        seed,
        a_train_after_a,
        a_test_after_a,
        a_test_after_b,
        b_test_after_b,
        protected_edges,
        cost_a,
        cost_b,
        a_blocked,
        topology_unchanged: *graph == original
            && circuit.topology_fingerprint() == fingerprint
            && circuit.weights().len() == graph.edges(),
    })
}

/// Grid: seed × learner × mechanism (15 rows per seed).
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
        for learner in cg::LEARNERS {
            for mechanism in MECHANISMS {
                rows.push(run_arm(&g, &task, spec, learner, mechanism, seed)?);
            }
        }
    }
    Ok(rows)
}

pub const CSV_HEADER: &str = "schema,graph,learner,mechanism,seed,a_train_bal_after_a,a_test_bal_after_a,a_test_bal_after_b,b_test_bal_after_b,forgetting,protected_edges,a_episodes,a_attempts,a_weights_changed,a_blocked,b_episodes,b_attempts,b_weights_changed,b_protected_attempts,b_blocked,topology_unchanged";

/// Deterministic CSV (no timing column).
#[must_use]
pub fn to_csv(rows: &[Row], graph_label: &str) -> String {
    let mut out = String::from(CSV_HEADER);
    out.push('\n');
    for r in rows {
        out.push_str(&format!(
            "b888-4-importance-protection-v1,{graph_label},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{},{},{},{},{},{}\n",
            r.learner.label(),
            r.mechanism.label(),
            r.seed,
            r.a_train_after_a.balanced(),
            r.a_test_after_a.balanced(),
            r.a_test_after_b.balanced(),
            r.b_test_after_b.balanced(),
            r.forgetting(),
            r.protected_edges,
            r.cost_a.episodes,
            r.cost_a.attempts,
            r.cost_a.weights_changed,
            r.a_blocked,
            r.cost_b.episodes,
            r.cost_b.attempts,
            r.cost_b.weights_changed,
            r.cost_b.protected_attempts,
            r.cost_b.blocked,
            r.topology_unchanged,
        ));
    }
    out
}

/// Paired comparison of x against y (same seeds).
#[derive(Clone, Debug, PartialEq)]
pub struct Comparison {
    pub x: (Learner, Mechanism),
    pub y: (Learner, Mechanism),
    pub n: usize,
    /// x − y on A test after A.
    pub a_test: Spread,
    /// y forgetting − x forgetting (+ = x forgets less).
    pub forget: gb::Paired,
    /// x − y on A test after B (retained A).
    pub retained: Spread,
    /// x − y on B test after B.
    pub b: Spread,
    /// Δretained + ΔB per seed; `better` = seeds with sum > 0.
    pub trade: gb::Paired,
}

impl Comparison {
    /// Pre-registered pass rule.
    #[must_use]
    pub fn passes(&self) -> bool {
        self.n > 0
            && self.forget.delta.mean >= gb::MARGIN
            && (self.forget.better as f64) >= 0.8 * self.n as f64
            && self.a_test.mean > -gb::MARGIN
            && self.b.mean > -gb::MARGIN
    }

    /// H3 rule: trade sum ≥ margin (> 0 in ≥ 8/10) and forgetting cut ≥ margin.
    #[must_use]
    pub fn better_exchange(&self) -> bool {
        self.n > 0
            && self.trade.delta.mean >= gb::MARGIN
            && (self.trade.better as f64) >= 0.8 * self.n as f64
            && self.forget.delta.mean >= gb::MARGIN
    }

    /// H4 rule vs the control.
    #[must_use]
    pub fn beats_control_tradeoff(&self) -> bool {
        self.n > 0
            && self.a_test.mean > -gb::MARGIN
            && self.b.mean > -gb::MARGIN
            && self.forget.delta.mean >= gb::MARGIN
            && (self.forget.better as f64) >= 0.8 * self.n as f64
    }
}

#[must_use]
pub fn compare(rows: &[Row], x: (Learner, Mechanism), y: (Learner, Mechanism)) -> Comparison {
    let (mut da, mut df, mut dr, mut db, mut dt) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for rx in rows
        .iter()
        .filter(|r| r.learner == x.0 && r.mechanism == x.1)
    {
        if let Some(ry) = rows
            .iter()
            .find(|r| r.learner == y.0 && r.mechanism == y.1 && r.seed == rx.seed)
        {
            let r = rx.a_test_after_b.balanced() - ry.a_test_after_b.balanced();
            let b = rx.b_test_after_b.balanced() - ry.b_test_after_b.balanced();
            da.push(rx.a_test_after_a.balanced() - ry.a_test_after_a.balanced());
            df.push(ry.forgetting() - rx.forgetting());
            dr.push(r);
            db.push(b);
            dt.push(r + b);
        }
    }
    Comparison {
        x,
        y,
        n: df.len(),
        a_test: gb::spread(&da),
        forget: gb::paired(&df),
        retained: gb::spread(&dr),
        b: gb::spread(&db),
        trade: gb::paired(&dt),
    }
}

/// Every (L, M ≠ none) vs (L, none), then every graded (L, M) vs control.
#[must_use]
pub fn preregistered_comparisons(rows: &[Row]) -> Vec<Comparison> {
    let mut out = Vec::new();
    for l in cg::LEARNERS {
        for m in MECHANISMS.iter().skip(1) {
            out.push(compare(rows, (l, *m), (l, Mechanism::None)));
        }
    }
    for l in cg::LEARNERS.iter().skip(1) {
        for m in MECHANISMS {
            out.push(compare(rows, (*l, m), (Learner::Control, Mechanism::None)));
        }
    }
    out
}

/// Pre-stated readings (H1, H2, H3, H4).
#[must_use]
pub fn readings(rows: &[Row]) -> (bool, bool, bool, bool) {
    let graded = &cg::LEARNERS[1..];
    let own = |l: Learner, m: Mechanism| compare(rows, (l, m), (l, Mechanism::None));
    let h1 = graded
        .iter()
        .all(|&l| own(l, Mechanism::Importance(25)).passes());
    let h2 = graded
        .iter()
        .all(|&l| own(l, Mechanism::TaskGated).passes());
    let new = [
        Mechanism::Importance(25),
        Mechanism::Importance(50),
        Mechanism::TaskGated,
    ];
    let h3 = graded
        .iter()
        .any(|&l| new.iter().any(|&m| own(l, m).better_exchange()));
    let h4 = graded.iter().any(|&l| {
        new.iter().any(|&m| {
            own(l, m).passes()
                && compare(rows, (l, m), (Learner::Control, Mechanism::None))
                    .beats_control_tradeoff()
        })
    });
    (h1, h2, h3, h4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graded_reward::Grain;
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
        std::env::temp_dir().join(format!("v888-ip-{tag}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn none_and_consolidate_reproduce_the_39_rows_bit_exactly() {
        for seed in [1, 2, 3] {
            let (g, task) = setup(seed);
            for learner in cg::LEARNERS {
                for (ours, theirs) in [
                    (Mechanism::None, cg::Mechanism::None),
                    (Mechanism::Consolidate39, cg::Mechanism::Consolidate),
                ] {
                    let a = run_arm(&g, &task, &spec(), learner, ours, seed).unwrap();
                    let b = cg::run_arm(&g, &task, &spec(), learner, theirs, seed).unwrap();
                    assert_eq!(a.a_test_after_a, b.a_test_after_a);
                    assert_eq!(a.a_test_after_b, b.a_test_after_b);
                    assert_eq!(a.b_test_after_b, b.b_test_after_b);
                    assert_eq!(a.protected_edges, b.protected_edges);
                    assert_eq!(a.cost_a, b.cost_a);
                    assert_eq!(a.cost_b, b.cost_b);
                    assert_eq!(a.a_blocked, 0);
                }
            }
        }
    }

    #[test]
    fn importance_mask_ranks_by_credit_and_requires_movement() {
        let credit = [5, 9, 9, 0, 7, 3, 2, 1];
        let moved = [true, true, true, true, false, true, true, true];
        // 25 % of 8 = 2: edges 1 and 2 (tie broken by index)
        let m = importance_mask(&credit, &moved, 25);
        assert_eq!(m, [false, true, true, false, false, false, false, false]);
        // 50 % = 4: then 0 (5), 5 (3); edge 4 unmoved, edge 3 zero credit
        let m = importance_mask(&credit, &moved, 50);
        assert_eq!(m, [true, true, true, false, false, true, false, false]);
        // cap by the number of eligible edges
        let m = importance_mask(&credit, &moved, 100);
        assert_eq!(m.iter().filter(|&&b| b).count(), 6);
        assert!(importance_mask(&[], &[], 25).is_empty());
    }

    #[test]
    fn mechanisms_respect_their_gates_and_topology() {
        let (g, task) = setup(6);
        let n = g.edges() as u64;
        for learner in cg::LEARNERS {
            let base = run_arm(&g, &task, &spec(), learner, Mechanism::None, 6).unwrap();
            for m in MECHANISMS {
                let r = run_arm(&g, &task, &spec(), learner, m, 6).unwrap();
                assert!(r.topology_unchanged);
                assert_eq!(r.cost_a.episodes, base.cost_a.episodes);
                assert_eq!(r.cost_b.episodes, base.cost_b.episodes);
                match m {
                    Mechanism::TaskGated => {
                        assert!(r.a_blocked > 0);
                        assert!(r.protected_edges > n / 4 && r.protected_edges < 3 * n / 4);
                        assert_eq!(r.cost_b.blocked, r.cost_b.protected_attempts);
                    }
                    Mechanism::Importance(p) => {
                        assert_eq!(r.a_test_after_a, base.a_test_after_a);
                        assert_eq!(r.cost_a, base.cost_a);
                        assert!(r.protected_edges > 0);
                        assert!(r.protected_edges <= (n * u64::from(p)).div_ceil(100));
                        assert_eq!(r.cost_b.blocked, r.cost_b.protected_attempts);
                    }
                    Mechanism::Consolidate39 => {
                        assert!(r.cost_b.blocked <= r.cost_b.protected_attempts);
                    }
                    Mechanism::None => {
                        assert_eq!((r.protected_edges, r.cost_b.blocked), (0, 0));
                    }
                }
            }
        }
    }

    #[test]
    fn task_gated_never_changes_an_edge_outside_its_phase_subset() {
        let (g, task) = setup(9);
        let learner = Learner::Graded(Grain::PerTarget);
        let mut c = PlasticCircuit::new(&g, 1).unwrap();
        let a = a_items(&task, &spec(), 9);
        train_a(learner, &mut c, &g, &a, Mechanism::TaskGated, 9).unwrap();
        for (e, &w) in c.weights().iter().enumerate() {
            if gate_b(e, 9) {
                assert_eq!(w, 1, "B-gated edge {e} moved during A");
            }
        }
        let after_a = c.weights().to_vec();
        let prot = protected_after_a(Mechanism::TaskGated, &after_a, 1, &[], 9);
        let b = b_stream(&task, &spec(), 9, false);
        train_b(learner, &mut c, &g, &b, Mechanism::TaskGated, prot, 9).unwrap();
        for (e, (&w, &w0)) in c.weights().iter().zip(&after_a).enumerate() {
            if !gate_b(e, 9) {
                assert_eq!(w, w0, "A-gated edge {e} moved during B");
            }
        }
    }

    #[test]
    fn grid_is_deterministic_and_csv_complete() {
        let a = run_grid(None, &shape(), &spec(), &[1, 2]).unwrap();
        let b = run_grid(None, &shape(), &spec(), &[1, 2]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 2 * 15);
        let csv = to_csv(&a, "GENERATED_SYNTHETIC");
        assert_eq!(csv, to_csv(&b, "GENERATED_SYNTHETIC"));
        assert_eq!(csv.lines().count(), 1 + a.len());
        assert!(csv.contains(",trace-per-target,importance-25,2,"));
        let cmp = preregistered_comparisons(&a);
        assert_eq!(cmp.len(), 3 * 4 + 2 * 5);
        assert!(cmp.iter().all(|c| c.n == 2));
        let id = compare(
            &a,
            (Learner::Control, Mechanism::None),
            (Learner::Control, Mechanism::None),
        );
        assert!(!id.passes() && !id.better_exchange());
        let _ = readings(&a);
        assert!(run_grid(None, &shape(), &spec(), &[]).is_err());
    }

    #[test]
    fn checkpoint_after_a_with_mask_resumes_b_bit_exactly() -> io::Result<()> {
        let (g, task) = setup(7);
        for m in [Mechanism::Importance(25), Mechanism::TaskGated] {
            let learner = Learner::Graded(Grain::Groups(8));
            let full = run_arm(&g, &task, &spec(), learner, m, 7)?;
            let mut c = PlasticCircuit::new(&g, 1)?;
            let a = a_items(&task, &spec(), 7);
            let (_, credit, _) = train_a(learner, &mut c, &g, &a, m, 7)?;
            let mask = protected_after_a(m, c.weights(), 1, &credit, 7);
            let (pc, pm) = (tmp("c"), tmp("m"));
            c.save(&pc)?;
            save_mask(&pm, &mask)?;
            let mut resumed = PlasticCircuit::load(&pc, &g)?;
            let mask2 = load_mask(&pm, &g)?;
            fs::remove_file(&pc)?;
            fs::remove_file(&pm)?;
            assert_eq!(mask, mask2);
            let b = b_stream(&task, &spec(), 7, false);
            let cost = train_b(learner, &mut resumed, &g, &b, m, mask2, 7)?;
            assert_eq!(cost, full.cost_b);
            assert_eq!(
                evaluate(&resumed, &g, &task.test, &task.valence_a)?,
                full.a_test_after_b
            );
            assert_eq!(
                evaluate(&resumed, &g, &task.test, &task.valence_b)?,
                full.b_test_after_b
            );
        }
        Ok(())
    }

    #[test]
    fn mask_loader_rejects_corrupt_files() -> io::Result<()> {
        let (g, _) = setup(3);
        let p = tmp("bad");
        fs::write(&p, b"nope")?;
        assert!(load_mask(&p, &g).is_err());
        save_mask(&p, &vec![false; g.edges() + 1])?;
        assert!(load_mask(&p, &g).is_err());
        let mut bytes = Vec::from(&MASK_MAGIC[..]);
        bytes.extend_from_slice(&(g.edges() as u64).to_le_bytes());
        bytes.extend(std::iter::repeat_n(2_u8, g.edges()));
        fs::write(&p, bytes)?;
        assert!(load_mask(&p, &g).is_err());
        fs::remove_file(&p)?;
        Ok(())
    }
}

# B888-4 — importance-weighted protection and sparse task-gated plasticity: a different handle on the learning-vs-forgetting trade-off (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository nor on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[GENERALIZATION_FORGETTING.md](GENERALIZATION_FORGETTING.md).
"Importance", "protection" and "gating" name parts of an update rule on a
Boolean graph with 4-bit edge weights; nothing here is a claim about
synapses, biology, insect memory or BANC physiology. The importance rule is
an engineering analogue of EWC-style penalties (Kirkpatrick et al. 2017,
PNAS 114:3521) for discrete weights, not a reimplementation of EWC.

Module: `src/importance_protection.rs`. CLI: `bench-importance-protection`.

## Motivation

[CONSOLIDATED_GRADED.md](CONSOLIDATED_GRADED.md) (#39): protecting every
edge moved by task A (≈ 125 of 144 edges) with a 1-of-4 gate cut forgetting
by ≈ 0.09–0.10 but cost ≈ 0.11–0.12 of B test on all three learners — a
≈ 1:1 trade, effectively a global learning-rate cut. Bounded rehearsal had
≈ no effect. This cycle tests two **selective** alternatives that leave most
of the graph fully plastic during B:

1. protect only the edges that carried the most task-A credit;
2. split the original edges into two disjoint per-task plastic subsets.

## Pre-registration (this section is committed before any run)

### Learners (unchanged code paths, from #38/#39)

* `supervised-matched` — clean supervised control (#34/#36/#38).
* `trace-group-8` — #38 graded reward, group = target index mod 8, delay 0.
* `trace-per-target` — #38 graded reward, one bit per firing target, delay 0.

### Mechanisms (constants fixed a priori, not tuned)

Only weights of existing (original) edges ever change; no edge is added or
removed. Every arm uses the same 256 A episodes and the same 256 B episodes
in the #38 orders (equal budget, no rehearsal).

* `none` — must be bit-identical to the #39 `none` row (tested).
* `consolidate-39` — the #39 `consolidate` rule reproduced through the same
  gate (protected = weight after A ≠ init, 1 of 4 B attempts pass); must be
  bit-identical to the #39 row (tested). Reference arm.
* `importance-25` / `importance-50` — **importance from task-A credit**.
  During A, every update *attempt* on edge e (the existing gate call: an edge
  offered to the supervised update, or a credit-threshold crossing of the
  trace rule) increments `credit[e]`. After A, importance(e) =
  `credit[e]` if the weight of e moved during A (weight after A ≠ init),
  else 0. The edges with importance > 0 are ranked by importance
  (descending, ties by edge index ascending) and the top
  `ceil(q · edges)` (q = 25 % / 50 % of **all** edges, capped by the number
  with importance > 0) are **frozen** during B: every B attempt on them is
  refused (counted, not retried; same "lost step" semantics as #39). All
  other edges are fully plastic in B. Memory: one u32 counter + one bit per
  edge.
* `task-gated` — **sparse task-gated plasticity**. A fixed deterministic
  per-edge mask `m(e) = mix(e ⊕ seed ⊕ GATE_SALT) & 1`
  (`GATE_SALT = 0x6A7E_D00D`) splits the edges ≈ 50/50. During A only edges
  with `m = 0` are plastic, during B only edges with `m = 1`. A refused
  attempt is counted. This arm changes A learning too, so its A test after A
  is guarded like everything else.

### Protocol

`bench-graded-reward` / `bench-consolidated-graded` defaults: generated
graph 64 inputs / 24 targets / fan-in 6; 4 active; 32 A-train / 32 B-train /
64 TEST patterns; 8 rounds per phase (256 episodes per phase); flip 50 %;
init weight 1; threshold 1; balanced accuracy on held-out TEST patterns;
forgetting = bal(A test after A) − bal(A test after B); **seeds 1..10**.
Grid: 3 learners × 5 mechanisms = 15 rows per seed, 150 rows. The grid is
run twice; the CSV must be byte-identical. Topology checked per row.
Checkpoint/resume: saving the circuit after A (`.v888pls`) plus the
importance mask, reloading and running B must give a bit-identical result
(tested).

### Pre-stated rule (margin 0.05; ≥ 8/10 seeds), paired by seed vs (L, none)

M **passes for L** iff (1) forgetting reduction (none − M) mean ≥ 0.05 and
strictly positive in ≥ 8/10 seeds; (2) A test after A mean drop ≤ 0.05;
(3) B test after B mean drop ≤ 0.05.

* **H1 (importance protection)**: `importance-25` passes for both graded
  learners.
* **H2 (task-gated plasticity)**: `task-gated` passes for both graded
  learners.
* **H3 (better exchange rate than #39)**: for some graded learner L and some
  M ∈ {importance-25, importance-50, task-gated}: the *trade sum*
  Δretained-A + ΔB-test (x − (L, none)) has mean ≥ +0.05 and is > 0 in
  ≥ 8/10 seeds, **and** forgetting reduction mean ≥ 0.05. (For
  `consolidate-39` this sum was ≈ −0.02, i.e. ≈ 1:1.)
* **H4 (trade-off vs the control)**: some graded (L, M) that passes for L
  also has, vs (`supervised-matched`, none): A-test Δ > −0.05, B-test Δ >
  −0.05, forgetting lower by ≥ 0.05 in ≥ 8/10 seeds.
* `importance-50` is reported under the same rule but is not part of H1.
  Passes on `supervised-matched` are reported separately.
* All constants are fixed a priori; no other configuration is run.
  Negative results are reported as produced.

```bash
cargo +1.89 run --release -- bench-importance-protection --csv results/importance-protection-v1.csv
```

## Results

_Pending — filled in after the pre-registered run only._

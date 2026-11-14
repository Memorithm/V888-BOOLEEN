# B888-4 cycle 8 — error-gated and margin-gated supervision: removing collateral overwriting instead of protecting edges (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository nor on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[GENERALIZATION_FORGETTING.md](GENERALIZATION_FORGETTING.md). "Error
gating" and "margin" name parts of an update rule on a Boolean graph with
4-bit edge weights; nothing here is a claim about synapses, biology, insect
memory or BANC physiology. The margin rule is an engineering analogue of the
perceptron-with-margin rule, not a model of anything biological.

Module: `src/error_gated.rs`. CLI: `bench-error-gated`.

## Motivation (why #43 failed, and what is left to try)

#39 (consolidation, rehearsal) and #43 (importance-weighted freezing,
task-gated plasticity) all trade retained A for B at about 1:1. #43's
interpretation: with a 50 % per-edge label flip between A and B, the edges
that carry A credit are largely the edges B must flip, so *any* protection
rule is stuck at 1:1. That interpretation was **not tested**.

On this task the label of a (pattern, target) pair is the OR of the
valences of the pattern's active edges into that target. A test pair is a
**conflict pair** if its A label differs from its B label, an **agree
pair** otherwise. Forgetting on conflict pairs is unavoidable if B is
learned (the two tasks cannot both be right there without a context
input). Forgetting on **agree pairs** is avoidable: B wants the same answer
as A there.

The supervised control (`apply_update` / `Supervised`) is unconditional:
every episode, every edge from an active source steps toward its target's
teacher bit, even when the target is already correct. With 256 episodes per
phase and saturating 4-bit weights, this (a) drives many weights to the
0 / 15 plateaus and (b) moves edges whose target was already right, i.e. it
can overwrite agree-pair structure during B (collateral overwriting). A rule
that only updates **where the prediction is wrong (or weak)** should leave
agree pairs alone. This is the first B888-4 arm that changes *when* the
supervised signal is applied instead of *which edges are protected*.

## Pre-registration (this section is committed before any run)

### Arms (constants fixed a priori, not tuned)

All arms use the clean supervised teacher of the matched control, the same
256 A episodes and the same 256 B episodes in the #38/#39/#43 orders (equal
episode budget, no rehearsal, no protection), threshold 1, init weight 1.
Only weights of existing (original) edges change; no edge is added or
removed. The decision is per (episode, reached target) and is taken on the
weights **before** the episode's update.

* `supervised` — the matched control. Must be bit-identical to the
  `supervised-matched` / `none` row of #39/#43 (tested).
* `error-gated` — the edges into target t from active sources step toward
  the teacher bit of t **only if** the circuit's prediction for t is wrong
  (fires when it should not, or is silent when it should fire). Correct
  targets are left unchanged.
* `margin-gated-4` — like `error-gated`, but a target that should fire is
  also updated while its drive (Σ weight × contacts over active edges into
  t, the quantity `predict` thresholds) is below `MARGIN = 4`. A target that
  should stay silent is updated only if it fires (with threshold 1 no
  silent margin exists).

### Protocol

`bench-importance-protection` defaults: generated graph 64 inputs / 24
targets / fan-in 6; 4 active; 32 A-train / 32 B-train / 64 TEST patterns;
8 rounds per phase (256 episodes per phase); flip 50 %; init weight 1;
threshold 1; balanced accuracy on held-out TEST patterns;
forgetting = bal(A test after A) − bal(A test after B). **Seeds 1..20**
(twice the 10 seeds of #39/#43, for tighter intervals). Grid: 3 arms × 20
seeds = 60 rows. The grid is run twice; the CSV must be byte-identical.
Topology checked per row.

Instrumentation per row: A/B test balanced accuracy, forgetting, the same
three numbers restricted to agree pairs and to conflict pairs, update
attempts and weights changed per phase, number of saturated weights (0 or
15) after A and after B.

Checkpoint/resume: saving the circuit after A (`V888PLS1`) and after 128
of the 256 B episodes, reloading and finishing must give bit-identical
weights and metrics (tested). The gated rules keep no state other than the
weights.

Statistics: paired by seed against `supervised`; mean ± sample sd;
better / tied / worse seed counts; **95 % percentile bootstrap CI** of the
mean paired difference (2000 resamples of the 20 seeds, fixed bootstrap
seed `BOOT_SEED = 0xB007_5EED`, deterministic).

### Pre-stated rule (margin 0.05; ≥ 16/20 seeds; CI excludes 0)

Arm X **passes** iff, paired against `supervised`:
(1) forgetting reduction (supervised − X) has mean ≥ 0.05, is strictly
positive in ≥ 16/20 seeds, and its bootstrap 95 % CI lower bound is > 0;
(2) A test after A mean drop ≤ 0.05; (3) B test after B mean drop ≤ 0.05.

* **H1 (error gating reduces forgetting at equal budget)**: `error-gated`
  passes.
* **H2 (margin gating reduces forgetting at equal budget)**:
  `margin-gated-4` passes.
* **H3 (generalisation)**: for some X ∈ {error-gated, margin-gated-4}, A
  test after A **or** B test after B improves on `supervised` by mean
  ≥ 0.05, in ≥ 16/20 seeds, bootstrap CI lower bound > 0, with the other of
  the two not dropping by more than 0.05.
* **H4 (diagnostic of #43's interpretation)**: the control's forgetting
  restricted to **agree pairs** has mean ≥ 0.05, is > 0 in ≥ 16/20 seeds
  and has CI lower bound > 0, i.e. a measurable share of the control's
  forgetting is avoidable. If H4 fails, #43's interpretation (forgetting is
  essentially the unavoidable conflict part) is supported on this task.
* Also reported, not part of any hypothesis: trade sum (Δ retained A + Δ B)
  vs the control, agree/conflict breakdown for every arm, saturation counts.
* All constants are fixed a priori; no other configuration is run.
  Negative results are reported as produced.

```bash
cargo +1.89 run --release -- bench-error-gated --csv results/error-gated-v1.csv
```

## Results

(Filled in only after the pre-registered run.)

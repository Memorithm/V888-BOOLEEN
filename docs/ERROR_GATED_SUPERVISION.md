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

## Results (pre-registered grid; CSV `results/error-gated-v1.csv`)

CSV: 60 rows (20 seeds × 3 arms), sha256
`eafa390a9d1947592ee78aa090fdeede08ff23be82401cb11bb53c696ac653cd`. A second
run gave a byte-identical CSV and identical stdout. Run time is about 0.3 s
(release build, CPU). `topology_unchanged=true` for every row. On seeds
1..10 the `supervised` rows reproduce the #43 control exactly (A test 0.870,
retained A 0.617, B test 0.803, forgetting 0.253); bit-identity with the
#39/#43 control code path is also unit-tested. On average a seed has 291.6
agree and 271.4 conflict test pairs. Values are mean ± sample sd over 20
seeds; counters are means over A+B.

| arm | A train | A test (after A) | A test after B | forgetting | B test | agree forgetting | conflict forgetting | B test agree / conflict | targets updated | attempts | weights changed | saturated after A / B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| supervised (control) | 0.935±0.029 | 0.857±0.039 | 0.603±0.044 | 0.253±0.062 | 0.803±0.030 | −0.038±0.049 | 0.559±0.071 | 0.901 / 0.709 | 4520.4 | 5192.4 | 2218.2 | 100.7 / 114.0 |
| error-gated | 1.000±0.000 | 0.928±0.036 | 0.583±0.052 | 0.346±0.066 | 0.920±0.025 | −0.058±0.043 | 0.774±0.061 | 0.986 / 0.849 | 122.6 | 136.9 | 129.2 | 62.5 / 70.3 |
| margin-gated-4 | 1.000±0.000 | 0.928±0.036 | 0.579±0.052 | 0.350±0.066 | 0.920±0.025 | −0.054±0.043 | 0.779±0.060 | 0.983 / 0.854 | 340.0 | 380.1 | 366.6 | 62.5 / 69.8 |

Paired against `supervised` (same seed). Each cell: mean ± sd
[bootstrap 95 % CI] better/tied/worse seeds. For forgetting columns
"better" means x forgets less; for the others x is higher.

| x | Δ forgetting | Δ agree forgetting | Δ conflict forgetting | Δ A test | Δ retained A | Δ B test | trade sum | rule (H1/H2) | H3 |
|---|---|---|---|---|---|---|---|---|---|
| error-gated | −0.092±0.037 [−0.108, −0.076] 0/0/20 | +0.019±0.048 [+0.000, +0.040] 12/1/7 | −0.215±0.054 [−0.239, −0.193] 0/0/20 | **+0.072±0.026 [+0.060, +0.082] 20/0/0** | −0.021±0.022 [−0.030, −0.011] 3/0/17 | **+0.116±0.034 [+0.103, +0.131] 20/0/0** | +0.095±0.031 [+0.083, +0.108] 20/0/0 | **fails (forgetting worse)** | **passes** |
| margin-gated-4 | −0.096±0.036 [−0.112, −0.081] 0/0/20 | +0.016±0.049 [−0.005, +0.037] 12/1/7 | −0.220±0.052 [−0.243, −0.198] 0/0/20 | **+0.072±0.026 [+0.060, +0.082] 20/0/0** | −0.025±0.023 [−0.035, −0.015] 3/0/17 | **+0.116±0.033 [+0.102, +0.131] 20/0/0** | +0.091±0.031 [+0.079, +0.105] 20/0/0 | **fails (forgetting worse)** | **passes** |

Control agree-pair forgetting (H4, one-sample): −0.038±0.049
[−0.059, −0.018], > 0 in 4/20 seeds.

## Pre-stated readings

* **H1 (error-gated reduces forgetting): not supported — the opposite.**
  Forgetting is *higher* than the control by 0.092 (20/20 seeds, CI
  [−0.108, −0.076]). Guards (2) and (3) hold with large margins: A test
  +0.072 and B test +0.116.
* **H2 (margin-gated-4 reduces forgetting): not supported**, same pattern
  (−0.096 forgetting, +0.072 A test, +0.116 B test).
* **H3 (generalisation): supported.** Both gated arms raise held-out A test
  after A by +0.072 and B test after B by +0.116, in 20/20 seeds, with
  bootstrap CIs well above 0, using ~17× fewer weight changes (129 vs 2218)
  and ~38× fewer edge update attempts (137 vs 5192) for the same 256 + 256
  episodes.
* **H4 (avoidable forgetting in the control): not supported.** On agree
  pairs the control does not forget at all; B training *improves* them
  (agree A test 0.862 → 0.901). All of the control's forgetting is on
  conflict pairs (0.559). This supports #43's interpretation on this task:
  forgetting here is the unavoidable conflict part, so protection rules
  can only trade it against B at ~1:1.

## Findings, including negative ones

* The gated rule is a better learner, not a better rememberer. It fits A
  train perfectly (1.000 vs 0.935), generalises better to held-out patterns
  on both tasks, and the trade sum (Δ retained A + Δ B) is +0.095 vs the
  control (20/20 seeds), the first B888-4 arm with a clearly positive trade
  sum. It also beats every graded/protection arm of #39/#43 on B test
  (0.920 vs ≤ 0.803).
* It "forgets" more because it learns B better: on conflict pairs, where B
  must overwrite A, it moves A accuracy from 0.925 to 0.151 (control 0.850
  → 0.291). On agree pairs it retains slightly better than the control
  (agree forgetting −0.058 vs −0.038, CI of the difference [+0.000,
  +0.040], 12/1/7 seeds — not significant by the pre-stated rule).
* Retained A after B is slightly lower than the control (−0.021, 17/20
  seeds worse). By the pre-stated rule this is a forgetting failure, and we
  report it as such.
* The unconditional supervised step is wasteful on this task: it drives
  ~101 of 144 weights to the 0/15 plateaus after A (gated: ~63) and
  spends ~2200 weight changes where ~130 suffice.
* `margin-gated-4` adds ~2.8× more updates than `error-gated` and changes
  nothing measurable; the positive margin is not useful at threshold 1.
* Consequence for the series: on this task (50 % per-edge label flip, no
  context input) forgetting cannot be reduced below the conflict share
  without losing B. A forgetting claim would need a task with a context
  signal or a lower flip rate; that was **not** run here (constants fixed
  before the run).

## Limitations

Synthetic generated graph only (not BANC; DOI 10.7910/DVN/7WTH1N is not
built on the box). The task is an artificial OR rule with a 50 % per-edge
label flip between A and B. One graph size (144 edges), one margin, init
weight 1, threshold 1, balanced accuracy on 64 held-out test patterns, 20
seeds. The gated arms use the supervised teacher (full labels), so they
are compared to the supervised control only, not to the reward-only graded
learners as a like-for-like signal budget. No biological claims.

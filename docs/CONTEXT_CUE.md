# B888-4 cycle 9 — explicit task-context cue through original edges: can it break the conflict-pair forgetting floor? (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository nor on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[GENERALIZATION_FORGETTING.md](GENERALIZATION_FORGETTING.md). "Context
cue" names a set of ordinary input nodes that are switched on during one
task; nothing here is a claim about synapses, biology, insect memory, task
sets or BANC physiology.

Module: `src/context_cue.rs`. CLI: `bench-context-cue`.

## Motivation (what #117 left open)

#117 (error-gated supervision) showed that, on the 50 %-flip A→B task, the
supervised control's forgetting is entirely on **conflict pairs** (A label
≠ B label; agree-pair forgetting −0.038), and that error gating learns
better but forgets more (−0.092), because without any context input A and
B cannot both be right on a conflict pair. #117 explicitly left
context-cued tasks untested. This cycle feeds an explicit task cue into the
circuit through **original, existing edges only** (no topology change) and
asks whether the conflict-pair floor moves.

## A-priori structural bound (written before any run)

The circuit is one layer of threshold units with non-negative 4-bit
weights: target t fires iff D_p(t) + b(t) ≥ 1, where D_p(t) is the drive
from the pattern's active edges and b(t) the drive from active cue edges
into t (Σ weight × contacts). With cue A on during A and cue B on during B,
a conflict pair (p, t) with A=1, B=0 can be right in both tasks only if
b_A(t) > b_B(t), and a pair with A=0, B=1 only if b_B(t) > b_A(t). Hence,
per target, an additive cue can make **at most one direction** of conflict
pairs correct in both tasks (and a direction needs the corresponding cue to
reach t at all). We report this **ceiling** (fraction of test conflict
pairs lying on a fixable majority direction) as a diagnostic. The update
rules also step every active edge into a target by the same amount, so they
have no mechanism that steers the task difference into the cue edges. Our
a-priori expectation is therefore that H1–H3 are **likely to fail**; the
experiment measures by how much, and is run anyway because the expectation
is untested.

## Pre-registration (this section is committed before any run)

### Context lines (fixed rule, chosen from the graph only)

Per seed (per generated graph), two disjoint sets of existing input nodes
are reserved as cue lines, chosen before the task is drawn:

* candidates are input nodes with ≥ 1 usable outgoing edge;
* **cue A**: greedy cover — repeatedly take the candidate whose outgoing
  edges reach the most targets not yet reached by cue A (ties → lowest
  index), until every target with an incoming edge is reached, no candidate
  adds a target, or `CUE_CAP = 12` lines are taken;
* **cue B**: the same greedy rule over the remaining candidates.

Cue lines are removed from the pattern pool, so patterns never contain a cue
line. Patterns, pattern sets and valences are then drawn with the same LCG
procedure as `generalization_bench::build_task` over the reduced pool. The
cue lines keep their original out-edges, weights start at the same init
weight and are learned like any other edge. No edge is added or removed.

### Arms (equal budget, constants fixed a priori)

All four arms share the graph, the task, the 256 A + 256 B episode orders of
#117 (no rehearsal, no protection), threshold 1, init weight 1, and the
clean supervised teacher. The teacher and the scored (reached) targets are
defined by the pattern's edges only.

* `supervised` — the #117 control rule (every edge from an active source
  steps toward its target's teacher bit, unconditionally). No cue.
* `error-gated` — #117's error gate (update the active edges into a reached
  target only if its prediction is wrong). No cue.
* `supervised+cue` — as `supervised`, with cue A lines active on every A
  episode and A evaluation, cue B lines on every B episode and B
  evaluation.
* `error-gated+cue` — as `error-gated`, with the same cue; the gate
  decision uses the drive including the cue.

### Protocol

Generated graph 64 inputs / 24 targets / fan-in 6; 4 active pattern inputs;
32 A-train / 32 B-train / 64 TEST patterns; 8 rounds per phase; positive
50 %; init weight 1; threshold 1; balanced accuracy on held-out TEST
patterns; forgetting = bal(A test after A) − bal(A test after B), A test
always evaluated in the A context (cue A on for cued arms), B test in the B
context. **Seeds 1..20.** Label-flip rates **25 %, 50 % (primary), 100 %**.
Grid: 3 flips × 20 seeds × 4 arms = 240 rows, run twice, CSV must be
byte-identical; topology checked per row.

Instrumentation per row: cue line counts and targets reached by each cue;
A/B test balanced accuracy, forgetting, and the same restricted to agree and
conflict pairs; **conflict both-correct** (fraction of test conflict pairs
answered correctly in both contexts by the final circuit; 0 by construction
for uncued arms); the structural **ceiling**; attempts / weights changed per
phase; saturation counts; mean final weight of cue edges.

Checkpoint/resume: saving the circuit (`V888PLS1`) after A and after 128 of
the 256 B episodes, reloading and finishing must give bit-identical weights
and metrics (tested). The cue lines are a deterministic function of the
graph, so no extra state is saved. With the cue off and the full pool,
`supervised` / `error-gated` must be bit-identical to the #117 arms
(tested).

Statistics (as #117): paired by seed; mean ± sample sd; better / tied /
worse seeds; 95 % percentile bootstrap CI of the mean paired difference
(2000 resamples, `BOOT_SEED = 0xB007_5EED`).

### Pre-stated rule (margin 0.05; ≥ 16/20 seeds; CI excludes 0 — same as #117)

Cued arm X+cue **passes** iff, paired against the same gate X without cue at
the same flip rate: (1) forgetting reduction (X − X+cue) has mean ≥ 0.05,
is > 0 in ≥ 16/20 seeds, bootstrap CI lower bound > 0; (2) A test after A
mean drop ≤ 0.05; (3) B test after B mean drop ≤ 0.05.

* **H1** (flip 50): `supervised+cue` passes vs `supervised`.
* **H2** (flip 50): `error-gated+cue` passes vs `error-gated`.
* **H3 (conflict floor, flip 50)**: for some X, the **conflict-pair**
  forgetting reduction (X − X+cue) has mean ≥ 0.05, > 0 in ≥ 16/20 seeds,
  CI lower bound > 0, and the conflict-pair B test does not drop by more
  than 0.05.
* **H4 (secondary, flip sweep)**: the H1 or H2 rule passes at flip 25 % or
  at flip 100 %. Reported per flip; not corrected for multiplicity, so a
  lone H4 pass is weak evidence.
* Also reported, not part of any hypothesis: comparisons of every arm vs
  `supervised`, conflict both-correct and ceiling per arm, cue-edge weights.
* All constants are fixed a priori; no other configuration is run.
  Negative results are reported as produced.

```bash
cargo +1.89 run --release -- bench-context-cue --csv results/context-cue-v1.csv
```

## Results (pre-registered grid; CSV `results/context-cue-v1.csv`)

CSV: 240 rows (3 flips × 20 seeds × 4 arms), sha256
`67f07556ff26cfaebd4f01627b8dc526c33d92adfd8d78866edcaeaad1bbfecc`. A second
run gave a byte-identical CSV and identical stdout; the whole grid runs in
well under a second (release build, CPU). `topology_unchanged=true` for
every row. The greedy rule reserved 7.6 (cue A) and 8.9 (cue B) input lines
on average; each cue reaches all 24 targets in every seed. Because the cue
lines leave the pattern pool, the uncued arms here are **not** the #117
numbers (smaller pool, different patterns: `supervised` A test 0.891 here vs
0.857 in #117); bit-identity with the #117 code path on the full pool is
unit-tested. Mean test pairs per seed (agree / conflict): flip 25: 361.3 /
111.8; flip 50: 246.6 / 226.5; flip 100: 24.0 / 449.1. Values are
mean ± sample sd over 20 seeds.

### Primary (flip 50 %)

| arm | A train | A test (after A) | A test after B | forgetting | B test | agree forgetting | conflict forgetting | conflict B test | conflict both-correct | ceiling | weights changed (A+B) | mean cue-edge weight |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| supervised | 0.938±0.028 | 0.891±0.039 | 0.569±0.057 | 0.322±0.079 | 0.864±0.052 | −0.035±0.070 | 0.703±0.109 | 0.808±0.086 | 0.000 | 0.843±0.048 | 1484.8 | (unused) |
| error-gated | 1.000±0.000 | 0.970±0.025 | 0.550±0.069 | 0.420±0.073 | 0.970±0.017 | −0.030±0.033 | 0.909±0.051 | 0.939±0.044 | 0.000 | 0.843±0.048 | 82.1 | (unused) |
| supervised+cue | 0.876±0.041 | 0.837±0.047 | 0.581±0.053 | 0.256±0.073 | 0.817±0.064 | −0.021±0.063 | 0.553±0.095 | 0.768±0.091 | 0.077±0.060 | 0.843±0.048 | 6216.6 | 0.38 |
| error-gated+cue | 1.000±0.000 | 0.970±0.025 | 0.643±0.065 | 0.327±0.075 | 0.971±0.019 | −0.027±0.032 | 0.720±0.116 | 0.953±0.032 | 0.208±0.102 | 0.843±0.048 | 161.5 | 0.13 |

Paired by seed. Each cell: mean ± sd [bootstrap 95 % CI] better/tied/worse
seeds ("better" = x forgets less, or x is higher).

| flip | x vs base | Δ forgetting | Δ conflict forgetting | Δ A test | Δ retained A | Δ B test | Δ conflict B test | Δ both-correct | rule | floor |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | supervised+cue vs supervised | +0.066±0.043 [+0.048, +0.085] 19/0/1 | +0.150±0.084 [+0.115, +0.187] 20/0/0 | **−0.054±0.034** [−0.068, −0.040] 0/1/19 | +0.012±0.028 [−0.001, +0.024] 17/0/3 | −0.047±0.032 [−0.060, −0.034] 0/2/18 | −0.040±0.031 [−0.053, −0.026] 1/3/16 | +0.077±0.060 [+0.052, +0.105] 17/3/0 | **fails** (A-test guard) | passes |
| 50 | error-gated+cue vs error-gated | **+0.093±0.052 [+0.072, +0.116] 20/0/0** | **+0.190±0.100 [+0.150, +0.235] 20/0/0** | +0.000±0.000 0/20/0 | +0.093±0.052 [+0.072, +0.116] 20/0/0 | +0.000±0.008 [−0.003, +0.004] 6/9/5 | +0.014±0.023 [+0.005, +0.025] 8/12/0 | +0.208±0.102 [+0.166, +0.253] 20/0/0 | **passes** | **passes** |
| 25 | supervised+cue vs supervised | +0.023±0.027 [+0.011, +0.034] 15/1/4 | +0.139±0.103 [+0.097, +0.185] 18/2/0 | −0.054±0.034 0/1/19 | −0.031±0.022 1/0/19 | −0.053±0.034 0/0/20 | −0.060±0.070 0/7/13 | +0.056±0.048 14/6/0 | fails | fails |
| 25 | error-gated+cue vs error-gated | +0.044±0.024 [+0.034, +0.054] 20/0/0 | +0.203±0.116 [+0.156, +0.255] 20/0/0 | +0.000 0/20/0 | +0.044±0.024 20/0/0 | −0.004±0.010 2/12/6 | +0.008±0.028 2/18/0 | +0.209±0.116 20/0/0 | fails (mean < 0.05) | passes |
| 100 | supervised+cue vs supervised | +0.150±0.055 [+0.126, +0.173] 20/0/0 | +0.161±0.059 [+0.135, +0.185] 20/0/0 | −0.054±0.034 0/1/19 | +0.096±0.043 20/0/0 | −0.035±0.027 1/1/18 | −0.035±0.027 1/1/18 | +0.095±0.049 20/0/0 | fails (A-test guard) | passes |
| 100 | error-gated+cue vs error-gated | **+0.195±0.078 [+0.165, +0.231] 20/0/0** | +0.206±0.080 [+0.174, +0.241] 20/0/0 | +0.000 0/20/0 | +0.195±0.078 20/0/0 | +0.007±0.009 [+0.003, +0.010] 9/11/0 | +0.007±0.010 9/11/0 | +0.218±0.080 20/0/0 | **passes** | passes |

Not part of any hypothesis, flip 50, vs `supervised`: `error-gated` again
forgets more (Δ forgetting −0.098 [−0.122, −0.075], 0/0/20; A test +0.079,
B test +0.106; the #117 pattern reproduces on this smaller pool).
`error-gated+cue` vs `supervised`: Δ forgetting −0.005 [−0.038, +0.030]
9/0/11 (no difference), A test +0.079 [+0.062, +0.096] 20/0/0, retained A
+0.074 [+0.052, +0.099] 19/0/1, B test +0.106 [+0.084, +0.130] 20/0/0.

## Pre-stated readings

* **H1 (supervised+cue, flip 50): fails.** Forgetting falls by +0.066
  (19/20 seeds, CI [+0.048, +0.085]) and conflict forgetting by +0.150, but
  guard (2) fails narrowly: the always-on cue costs A test −0.054 (> 0.05
  allowed; 19/20 seeds worse). B test also drops −0.047 (inside the guard).
  The supervised rule spends 4× more weight changes (6217 vs 1485) pushing
  cue edges up and down.
* **H2 (error-gated+cue, flip 50): passes.** Forgetting reduction +0.093,
  20/20 seeds, CI [+0.072, +0.116]; A test unchanged (identical in 20/20
  seeds), B test unchanged (+0.000).
* **H3 (conflict floor, flip 50): passes** for both gates; for the error
  gate, conflict forgetting falls by +0.190 (20/20, CI [+0.150, +0.235])
  with conflict B test +0.014. 20.8 % of test conflict pairs are now right
  in **both** contexts (0 % by construction without the cue).
* **H4 (secondary, flip sweep): passes** (error-gated+cue at flip 100:
  +0.195, 20/20, CI [+0.165, +0.231]). At flip 25 no cued arm passes
  (error-gated+cue +0.044 < 0.05; supervised+cue fails the guards). Not
  corrected for multiplicity.

## Findings, including negative ones

* **The a-priori expectation was wrong for the error gate.** We expected
  H1–H3 to fail; H2–H4 pass. Mechanism (checked against the numbers, not
  separately tested): under the error gate a target that should fire in B
  but is silent through its pattern edges fires anyway through the still
  untouched cue-B edge (init weight 1), so the gate does **not** raise the
  A-negative pattern edges; those stay at 0 and A survives. This is exactly
  the one direction (A=0, B=1, b_B > b_A) the structural bound allows.
  Correspondingly, cue-edge weights end low (mean 0.13) and only 161 weight
  changes are spent in total.
* The cue moves the error gate from "forgets more than the control"
  (#117, −0.098 here) to "forgets as much as the control" (−0.005, n.s.)
  while keeping the error gate's generalisation gains (A +0.079, B +0.106,
  retained A +0.074 vs the control). It does **not** forget less than the
  supervised control.
* Far below the ceiling: 0.208 of conflict pairs are both-correct vs a
  structural ceiling of 0.843. The update rule still moves pattern edges for
  the A=1, B=0 direction (it decrements every active edge into a wrongly
  firing target), so that direction is still overwritten.
* The supervised control cannot use the cue cleanly: its unconditional step
  keeps the cue edges active (mean 0.38), biasing targets on during A and
  costing A test 0.054. The cue reduces its forgetting but fails the guard.
* The structural bound held in every row (both-correct ≤ ceiling, unit
  tested on a small grid and verified on the full CSV by the same rule).
* Gains scale with the conflict share: forgetting reduction for
  error-gated+cue is +0.044 / +0.093 / +0.195 at flip 25 / 50 / 100.

## Limitations

Synthetic generated graph only (not BANC; DOI 10.7910/DVN/7WTH1N is not
built on the box). Artificial OR-rule task. One graph size (64/24/6), one
cue-selection rule (greedy cover, cap 12 per cue), cue lines taken out of
the pattern pool (so the uncued baselines are on a smaller pool than #117),
init weight 1, threshold 1, 20 seeds, 64 held-out test patterns. The cue is
a whole set of input lines switched on together, an engineering device;
it is not a model of any biological context signal. The mechanism reading
above is an interpretation of the measured numbers, not a separate
pre-registered test.

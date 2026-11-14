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

## Results

Not run yet at the time of pre-registration.

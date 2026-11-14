# B888-4 cycle 11 — do the #121 gates hold on longer task sequences, on held-out seeds and on novel input configurations? (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository nor on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` (64 inputs, 24 targets, fan-in 6) and
an artificial OR-rule task as in
[GENERALIZATION_FORGETTING.md](GENERALIZATION_FORGETTING.md), extended to a
sequence of four tasks with one context cue per task as in
[CONTEXT_CUE.md](CONTEXT_CUE.md) / [CUE_GATING.md](CUE_GATING.md). Nothing
here is a claim about synapses, biology, insect memory or BANC physiology.

Module: `src/task_sequence.rs`. CLI: `bench-task-sequence`.

## Motivation (what #121 left open)

#121 found, at flip 50 % with two tasks (A→B) on seeds 1..40:

* `E/E-cue` (error gate on pattern edges **and** cue edges) was the only
  arm passing the #120 rule vs its uncued control `error-gated`:
  forgetting −0.100 (40/40 seeds), A test and B test unchanged.
* `E/E-cue` vs `supervised` **failed** the forgetting rule (+0.015), even
  though it was better on every accuracy: A test +0.063, retained A
  +0.079, B test +0.107. `error-gated` learns A better, so it has more to
  forget; the forgetting difference hides a higher final accuracy.
* The B-test cost of about −0.07 belonged to **`E/S-cue`** (error-gated
  pattern edges, supervised cue edges): −0.071 vs `error-gated`. `E/E-cue`
  had **no** B cost (+0.001 vs `error-gated`). So the B cost is a property
  of the supervised cue-edge rule under an E pattern gate, not of pattern
  gating as such. This cycle keeps `E/S-cue` as a reported arm only.

Open questions: (1) does the `E/E-cue` benefit replicate on seeds never
used before? (2) does it survive longer sequences (three and four tasks),
where pattern edges are rewritten more often? (3) does it hold on test
pairs whose input configuration was never seen in training (the
generalisation side)? (4) can a positive-margin gate on both edge classes
do better than the plain error gate at equal budget?

## Setup

* **Graph:** generated per seed, 64 inputs / 24 targets / fan-in 6
  (as #117/#120/#121), threshold 1, init weight 1, 4-bit weights.
* **Cues:** four disjoint greedy-cover cues (cap 12 lines each,
  `context_cue::choose_n_cues`; cues 1 and 2 are exactly #120's cues A and
  B, tested). All four cues' lines are excluded from the pattern pool for
  every arm, so every arm of a seed sees the same patterns. Cue j is active
  while training task j and while evaluating task j. Uncued arms never
  activate a cue line.
* **Tasks:** four tasks T1→T2→T3→T4. Valence of T1: each edge positive
  with 50 %. Valence of T(i+1): each edge label of Ti flipped with the flip
  rate (chain). At flip 100 the sequence alternates (T3 = T1, T4 = T2).
* **Patterns:** 4 active inputs; 32 training patterns per task (disjoint
  across tasks); one held-out TEST set of 64 patterns, disjoint from every
  training set. A new deterministic generator (`task_sequence::build_sequence`)
  is used; numbers are therefore not comparable value-for-value with #121.
* **Budget (equal for every arm):** 8 rounds × 32 patterns = 256 episodes
  per task, same episode order for every arm of a seed, no rehearsal, no
  protection. Arms differ only in which reached targets get the supervised
  step.
* **Sequences of length 2 and 3 are the prefixes** of the 4-task run (the
  training is sequential and identical up to task k).
* **Seeds 41..80** (held-out: never run in any earlier B888-4 cycle).
  Flip rates 25 %, **50 % (primary)**, 100 %. Grid 3 × 40 × 9 = 1080 rows,
  run twice; the CSV must be byte-identical.

### Arms

Label `X/Y-cue`: X = pattern-edge gate, Y = cue-edge gate. S = supervised
(always update), E = error gate (update the edges into a reached target
only if its prediction is wrong; drive includes the cue, weights before the
update), Mm = margin gate (update if wrong, or if the target should fire
and its drive is below m). Gates as in #117/#121 (`cue_gating::step`).

| arm | pattern gate | cue gate |
|---|---|---|
| `supervised` | S | — (no cue) — the control |
| `error-gated` | E | — |
| `margin-gated-4` | M4 | — |
| `S/E-cue` | S | E |
| `E/S-cue` | E | S (reported only) |
| `E/E-cue` | E | E (the #121 winner) |
| `M2/M2-cue` | M2 | M2 |
| `M4/M4-cue` | M4 | M4 |
| `M8/M8-cue` | M8 | M8 |

`M1/M1-cue` is bit-identical to `E/E-cue` at threshold 1 (tested), which is
why the sweep starts at 2.

### Metrics

R[i][j] = balanced accuracy of task j on the TEST patterns (task j's
valence, cue j on for cued arms) after training task i. For sequence
length k ∈ {2, 3, 4}:

* ACC_k = mean over j ≤ k of R[k][j] (mean final accuracy) — **primary**;
* LEARN_k = mean over j ≤ k of R[j][j] (just-learned accuracy);
* FGT_k = mean over j < k of R[j][j] − R[k][j] (forgetting);
* LAST_k = R[k][k];
* ACCN_k = ACC_k restricted to **novel-configuration** TEST pairs: a pair
  (pattern, target) is novel for task j if the set of usable pattern edges
  into that target never occurred for that target in task j's training
  set. Novel pairs need the OR rule to be generalised from parts; the other
  pairs could be answered from a seen configuration.

Instrumentation per row: the full 4 × 4 matrices R and R_novel, novel and
total pair counts, weights changed and targets updated per task, cue lines
and reach per cue, mean cue-edge weight per cue at the end, saturated
weights, topology check.

Checkpoint/resume: saving the circuit (`V888PLS1`) after task 2 and after
half of task 3, reloading and finishing must give bit-identical costs and
final matrices for `E/E-cue`, `M4/M4-cue` and `S/E-cue` (tested).

Statistics (as #117/#120/#121): paired by seed; mean ± sample sd;
better / tied / worse seeds; 95 % percentile bootstrap CI of the mean
paired difference (2000 resamples, `BOOT_SEED = 0xB007_5EED`). "Strong" =
mean ≥ 0.05, > 0 in ≥ 32/40 seeds, CI lower bound > 0.

## A-priori analysis (written before any run)

1. Cue j is active only during task j (training and evaluation). An edge
   is only updated when its source is active, so cue j's edges are frozen
   after task j under every gate (tested for `E/E-cue`). Its drive at the
   final evaluation of task j is whatever task j left.
2. Uncued arms have no context signal: on a pair whose label differs
   between task j and the last task, at most one of the two can be right.
   Their ACC_k is bounded by how much the valence chain agrees.
3. At k = 2, ΔACC_2 = ½(Δ retained T1 + Δ T2). In #121 this was
   ½(+0.100 + 0.001) ≈ +0.05 for `E/E-cue` vs `error-gated`, right at the
   threshold, so the sequence rule at k = 2 is expected to be borderline.
   At k = 3 and 4 the last task is diluted by more retained tasks; if the
   per-task retention benefit stays near 0.1, ΔACC_k grows toward
   (k−1)/k × 0.1. If later tasks erase what cue edges compensate,
   it shrinks. No directional prediction is made.
4. Against `supervised`, ΔACC also contains error gating's learning
   advantage (#117: higher A and B test). Expectation: H3 more likely to
   pass than H2.
5. Margin gates make positive targets keep receiving increments until their
   drive reaches m. In #117 cycle 8, `margin-gated-4` (no cue) forgot
   *more* than `error-gated` and changed nothing useful at threshold 1.
   Expectation: H5 likely fails; it is tested because a cue edge with a
   higher weight could survive more later decrements.
6. At flip 100 the chain alternates, so an uncued learner re-learns T1
   during T3. Flip 100 is secondary only.

## Pre-registration (this section is committed before any run)

### Rules

* **Sequence rule** (x vs base at length k, flip 50, 40 seeds): ΔACC_k
  strong, ΔLAST_k ≥ −0.05 and ΔLEARN_k ≥ −0.05 (means).
* **Two-task rule** (#120/#121 rule, k = 2): forgetting reduction
  (FGT_2(base) − FGT_2(x)) strong, ΔR[1][1] ≥ −0.05 and ΔR[2][2] ≥ −0.05.

### Hypotheses (flip 50 %, seeds 41..80)

* **H1 (held-out replication):** `E/E-cue` passes the two-task rule vs
  `error-gated`.
* **H2 (longer sequences):** `E/E-cue` passes the sequence rule vs
  `error-gated` at **both** k = 3 and k = 4.
* **H3 (vs the supervised control):** `E/E-cue` passes the sequence rule
  vs `supervised` at both k = 3 and k = 4.
* **H4 (novel-configuration generalisation):** `E/E-cue` vs
  `error-gated`: ΔACCN_k strong at both k = 3 and k = 4. **Evaluable only
  if** the median number of novel TEST pairs per (row, task) at flip 50 is
  ≥ 20; otherwise reported as "not evaluable".
* **H5 (margin sweep):** at least one of `M2/M2-cue`, `M4/M4-cue`,
  `M8/M8-cue` passes the sequence rule vs `E/E-cue` at k = 3. Three
  margins, no multiplicity correction (stated).
* Reported, not hypotheses: everything at k = 2/3/4 for every listed
  comparison (incl. `E/E-cue` vs `supervised` at k = 2, `S/E-cue` vs
  `supervised`, `E/S-cue` vs `error-gated` and vs `E/E-cue`,
  `margin-gated-4` vs `error-gated`, margin arms vs `supervised`); flips
  25 % and 100 %; seed blocks 41..60 / 61..80; the full R matrices; costs.
* All constants are fixed a priori; no other configuration is run.
  Negative results are reported as produced.

```bash
cargo +1.89 run --release -- bench-task-sequence --csv results/task-sequence-v1.csv
```

## Results

Filled after the run. Nothing above this section is changed after the run.

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

## Results (pre-registered grid; CSV `results/task-sequence-v1.csv`)

Nothing above this section was changed after the run.

CSV: 1080 rows (3 flips × 40 seeds × 9 arms), sha256
`9d27c7d080c8c2a03de0b9b277ba87d8a9210dd5a21bd96a0142d8f8e9c90e52`. A second
run gave a byte-identical CSV and identical stdout; the grid takes ~24 s
(release build, CPU only). `topology_unchanged=true` in every row. Cue sizes
(mean lines): 7.7 / 8.8 / 10.4 / 11.8; reach 24 / 24 / 24 / 23.0 of 24
targets. With four cues reserved, the pattern pool is much smaller than in
#120/#121. TEST has 378.4 (pattern, target) pairs on average, but only
**1.9 % of them are novel-configuration pairs** (mean 7.1 per task, median
6, min 0, max 28).

### Primary (flip 50 %, seeds 41..80), mean ± sd over 40 seeds

| arm | ACC_2 | ACC_3 | ACC_4 | LEARN_4 | FGT_4 | R[4][1] (T1 at end) | weights changed (4 tasks) |
|---|---|---|---|---|---|---|---|
| supervised | 0.735±0.041 | 0.665±0.051 | 0.633±0.041 | 0.965±0.017 | 0.442±0.056 | 0.527±0.113 | 1217.5 |
| error-gated | 0.751±0.042 | 0.673±0.051 | 0.643±0.039 | 1.000±0.000 | 0.477±0.052 | 0.525±0.105 | 62.0 |
| margin-gated-4 | 0.751±0.042 | 0.673±0.051 | 0.643±0.039 | 1.000±0.000 | 0.477±0.052 | 0.525±0.105 | 165.7 |
| S/E-cue | 0.811±0.043 | 0.764±0.047 | 0.752±0.041 | 0.965±0.017 | 0.285±0.046 | 0.681±0.078 | 1279.1 |
| E/S-cue | 0.769±0.043 | 0.703±0.056 | 0.687±0.037 | 0.958±0.020 | 0.362±0.049 | 0.600±0.084 | 6734.0 |
| **E/E-cue** | **0.922±0.030** | **0.881±0.037** | **0.858±0.039** | 1.000±0.000 | **0.189±0.052** | **0.779±0.074** | 117.0 |
| M2/M2-cue | 0.901±0.034 | 0.846±0.042 | 0.832±0.034 | 1.000±0.000 | 0.225±0.046 | 0.748±0.073 | 146.9 |
| M4/M4-cue | 0.848±0.037 | 0.788±0.047 | 0.778±0.035 | 1.000±0.000 | 0.296±0.046 | 0.693±0.083 | 255.6 |
| M8/M8-cue | 0.833±0.042 | 0.777±0.049 | 0.768±0.036 | 1.000±0.000 | 0.309±0.047 | 0.685±0.075 | 459.2 |

Accuracy matrix R[i][j] at flip 50 (mean over seeds; rows = after task i):

| arm | after T1 | after T2 | after T3 | after T4 |
|---|---|---|---|---|
| supervised | 0.972 · · · | 0.512 0.957 · · | 0.511 0.513 0.970 · | 0.525 0.528 0.510 0.961 |
| error-gated | 1.000 · · · | 0.503 1.000 · · | 0.501 0.519 1.000 · | 0.525 0.529 0.516 1.000 |
| E/E-cue | 1.000 · · · | 0.844 1.000 · · | 0.793 0.851 1.000 · | 0.779 0.812 0.841 1.000 |
| M2/M2-cue | 1.000 · · · | 0.802 1.000 · · | 0.740 0.797 1.000 · | 0.748 0.774 0.804 1.000 |

At flip 50 the uncued arms fall to chance (≈ 0.50–0.53) on every earlier task
as soon as the next task is trained. In this small-pool setting almost every
TEST configuration was seen in training, so each one is simply relabelled.

### Paired comparisons (flip 50 %, 40 seeds)

Each cell: mean ± sd [bootstrap 95 % CI] better/tied/worse seeds (> 0
favours x; Δ FGT = base − x).

| k | x vs base | Δ ACC_k | Δ ACCN_k (novel) | Δ LEARN_k | Δ LAST_k | Δ FGT_k | seq rule | two-task rule |
|---|---|---|---|---|---|---|---|---|
| 2 | **E/E-cue vs error-gated (H1)** | +0.170±0.045 [+0.157, +0.185] 40/0/0 | +0.064±0.100 [+0.033, +0.096] 14/26/0 | 0.000 | 0.000 | **+0.341±0.090 [+0.315, +0.369] 40/0/0** | passes | **passes** |
| 3 | **E/E-cue vs error-gated (H2)** | **+0.208±0.041 [+0.195, +0.220] 40/0/0** | +0.069±0.090 [+0.044, +0.097] 20/19/1 | 0.000 | 0.000 | +0.312±0.062 [+0.293, +0.331] 40/0/0 | **passes** | — |
| 4 | **E/E-cue vs error-gated (H2)** | **+0.215±0.045 [+0.202, +0.229] 40/0/0** | +0.030±0.062 [+0.012, +0.049] 18/20/2 | 0.000 | 0.000 | +0.287±0.060 [+0.269, +0.306] 40/0/0 | **passes** | — |
| 2 | E/E-cue vs supervised | +0.187±0.043 [+0.175, +0.201] 40/0/0 | +0.070±0.118 [+0.036, +0.110] 14/26/0 | +0.035±0.025 38/2/0 | +0.043±0.035 32/8/0 | +0.304±0.092 [+0.276, +0.333] 40/0/0 | passes | passes |
| 3 | **E/E-cue vs supervised (H3)** | **+0.217±0.042 [+0.203, +0.229] 40/0/0** | +0.073±0.091 [+0.047, +0.100] 21/18/1 | +0.034±0.020 40/0/0 | +0.030±0.032 28/12/0 | +0.274±0.071 [+0.251, +0.296] 40/0/0 | **passes** | — |
| 4 | **E/E-cue vs supervised (H3)** | **+0.225±0.047 [+0.211, +0.239] 40/0/0** | +0.030±0.062 [+0.012, +0.049] 18/20/2 | +0.035±0.017 40/0/0 | +0.039±0.034 29/11/0 | +0.253±0.067 [+0.232, +0.273] 40/0/0 | **passes** | — |
| 2 | error-gated vs supervised | +0.017±0.023 [+0.010, +0.024] 27/8/5 | +0.006 1/39/0 | +0.035 | +0.043 | −0.037±0.042 [−0.051, −0.025] 6/2/32 | fails | fails |
| 4 | error-gated vs supervised | +0.009±0.016 [+0.004, +0.014] 22/11/7 | +0.000 | +0.035 | +0.039 | −0.034±0.027 3/0/37 | fails | — |
| 2 | S/E-cue vs supervised | +0.076±0.034 [+0.066, +0.087] 40/0/0 | +0.027 8/32/0 | 0.000 | 0.000 | +0.153±0.069 [+0.132, +0.175] 40/0/0 | passes | passes |
| 3 | S/E-cue vs supervised | +0.099±0.036 [+0.088, +0.110] 40/0/0 | +0.030 | 0.000 | 0.000 | +0.149±0.054 40/0/0 | passes | — |
| 4 | S/E-cue vs supervised | +0.118±0.034 [+0.108, +0.128] 40/0/0 | +0.018 | 0.000 | 0.000 | +0.158±0.045 40/0/0 | passes | — |
| 2 | E/S-cue vs error-gated | +0.017±0.039 [+0.006, +0.030] 30/0/10 | +0.020 | −0.040±0.026 0/1/39 | **−0.038±0.036 0/12/28** | +0.115±0.062 39/1/0 | fails | passes |
| 4 | E/S-cue vs error-gated | +0.044±0.031 [+0.035, +0.054] 38/0/2 | +0.021 | −0.042±0.020 0/0/40 | −0.047±0.036 0/7/33 | +0.115±0.040 40/0/0 | fails | — |
| 4 | E/S-cue vs E/E-cue | −0.171±0.047 [−0.186, −0.157] 0/0/40 | −0.008 | −0.042 | −0.047 | −0.172±0.066 0/0/40 | fails | — |
| 2–4 | margin-gated-4 vs error-gated | 0.000 0/40/0 (all k) | 0.000 | 0.000 | 0.000 | 0.000 | fails | fails |
| 3 | **M2/M2-cue vs E/E-cue (H5)** | **−0.036±0.027 [−0.044, −0.028] 0/4/36** | −0.011 0/35/5 | 0.000 | 0.000 | −0.053±0.040 [−0.066, −0.042] 0/4/36 | **fails** | — |
| 3 | **M4/M4-cue vs E/E-cue (H5)** | **−0.093±0.036 [−0.105, −0.082] 0/0/40** | −0.039 2/27/11 | 0.000 | 0.000 | −0.140±0.055 0/0/40 | **fails** | — |
| 3 | **M8/M8-cue vs E/E-cue (H5)** | **−0.104±0.038 [−0.116, −0.092] 0/0/40** | −0.039 2/27/11 | 0.000 | 0.000 | −0.156±0.056 0/0/40 | **fails** | — |
| 4 | M2 / M4 / M8 vs E/E-cue | −0.026 / −0.080 / −0.090 (0 better seeds each) | | 0.000 | 0.000 | −0.035 / −0.107 / −0.120 | fail | — |
| 4 | M2 / M4 / M8 vs supervised | +0.198 / +0.144 / +0.135 (40/0/0 each) | +0.030 / +0.021 / +0.018 | +0.035 | +0.039 | +0.218 / +0.146 / +0.133 | pass | — |

Seed blocks (flip 50, Δ ACC, all 20/0/0): E/E-cue vs error-gated k = 3:
41..60 +0.205 [+0.190, +0.218], 61..80 +0.212 [+0.190, +0.231]; k = 4: +0.223 /
+0.208. E/E-cue vs supervised k = 3: +0.215 / +0.218; k = 4: +0.237 / +0.213.
Every block passes the same rules as the full set.

### Secondary flips (reported, not hypotheses; not corrected for multiplicity)

| flip | x vs base | Δ ACC_2 | Δ ACC_3 | Δ ACC_4 | seq rule k = 2/3/4 |
|---|---|---|---|---|---|
| 25 | E/E-cue vs error-gated | +0.081 [+0.068, +0.097] 40/0/0 | +0.131 40/0/0 | +0.166 40/0/0 | pass / pass / pass |
| 25 | E/E-cue vs supervised | +0.111 40/0/0 | +0.146 40/0/0 | +0.186 40/0/0 | pass / pass / pass |
| 25 | S/E-cue vs supervised | +0.043 [+0.034, +0.053] 37/3/0 | +0.063 40/0/0 | +0.076 40/0/0 | fail / pass / pass |
| 25 | E/S-cue vs error-gated (Δ LAST ≈ −0.04) | −0.008 16/1/23 | +0.012 24/0/16 | +0.020 28/0/12 | fail / fail / fail |
| 25 | M2 / M4 / M8 vs E/E-cue | −0.010 / −0.031 / −0.035 | −0.024 / −0.059 / −0.064 | −0.031 / −0.077 / −0.083 | all fail |
| 100 | E/E-cue vs error-gated | +0.346 40/0/0 | +0.231 40/0/0 | +0.340 40/0/0 | pass / pass / pass |
| 100 | E/E-cue vs supervised | +0.348 40/0/0 | +0.242 40/0/0 | +0.341 40/0/0 | pass / pass / pass |
| 100 | S/E-cue vs supervised | +0.166 40/0/0 | +0.113 40/0/0 | +0.165 40/0/0 | pass / pass / pass |
| 100 | E/S-cue vs error-gated | +0.081 39/0/1 | +0.034 36/0/4 | +0.077 40/0/0 | pass / fail / pass |
| 100 | M2 / M4 / M8 vs E/E-cue | −0.041 / −0.145 / −0.177 | −0.029 / −0.093 / −0.116 | −0.047 / −0.138 / −0.172 | all fail |

At flip 100 the chain alternates (T3 = T1), so k = 3 is easier for the
uncued arms (they relearn T1 during T3). That is why Δ ACC_3 is smaller than
Δ ACC_2 and Δ ACC_4 there.

## Pre-stated readings

| | result |
|---|---|
| H1: `E/E-cue` passes the two-task rule vs `error-gated` on held-out seeds 41..80 | **passes** (Δ FGT_2 +0.341, 40/40; ΔA and ΔB exactly 0) |
| H2: `E/E-cue` passes the sequence rule vs `error-gated` at k = 3 and k = 4 | **passes** (Δ ACC +0.208 and +0.215, 40/40) |
| H3: `E/E-cue` passes the sequence rule vs `supervised` at k = 3 and k = 4 | **passes** (Δ ACC +0.217 and +0.225, 40/40; LEARN and LAST also higher) |
| H4: novel-configuration Δ ACCN strong at k = 3 and 4 | **not evaluable**: median 6 novel TEST pairs per task (< 20 pre-registered). Descriptive only: +0.069 [+0.044, +0.097] at k = 3 (20/19/1) and +0.030 [+0.012, +0.049] at k = 4 (18/20/2). At k = 4 this is below the 0.05 effect size anyway. |
| H5: some margin arm passes the sequence rule vs `E/E-cue` at k = 3 | **fails**: every margin is worse, and more so as m grows (−0.036, −0.093, −0.104; 0 better seeds) |

**Reading.**
- On seeds never used before, the #121 winner replicates: error-gating both
  edge classes with one cue per task keeps its benefit for two, three and
  four tasks. Its advantage in mean final accuracy grows from +0.170 (k = 2)
  to +0.215 (k = 4) vs `error-gated`, and from +0.187 to +0.225 vs the
  supervised control, at the same 256 episodes per task. It also changes
  10× fewer weights than `supervised` (117 vs 1218 over four tasks).
- The benefit decays slowly with sequence length but does not vanish: T1
  falls from 0.844 after T2 to 0.793 after T3 and 0.779 after T4 (uncued:
  chance). This is consistent with point 1 of the a-priori analysis: cue j's
  edges are frozen after task j, so what erodes is the shared pattern edges.
- **A positive margin does not help** at threshold 1. With cues it is
  monotonically harmful. Without a cue, `margin-gated-4` is identical to
  `error-gated` on every accuracy here, while changing 2.7× more weights.
  This matches #117 cycle 8.
- **The B-side cost is specific to the supervised cue-edge rule.**
  `E/S-cue` costs about −0.04 on the just-learned task (Δ LEARN, Δ LAST) at
  every k and flip and is 0.17 worse than `E/E-cue` at k = 4. `E/E-cue` and
  the margin arms have Δ LEARN = Δ LAST = 0 vs `error-gated`. Switching the
  cue edges to the error gate removes that cost at equal budget. No
  threshold/margin variant beat it.
- `S/E-cue` (supervised pattern edges, error-gated cue edges) passes vs
  `supervised` at every k here (+0.076 to +0.118), unlike in #121 (+0.034
  forgetting cut). The difference comes from the smaller pattern pool (more
  forgetting to recover), not from a rule change. It is not a hypothesis of
  this cycle.
- The generalisation side is **not answered** by this design. Reserving four
  cues leaves so few pattern inputs that 98 % of TEST pairs repeat a training
  configuration. `error-gated` then reaches 1.000 on TEST, which is
  memorisation, not generalisation. The few novel pairs show a positive but
  noisy and shrinking `E/E-cue` effect.

## Limitations

- Synthetic graph and artificial OR-rule task only. One graph size (64/24/6),
  threshold 1, init 1, one cue-selection rule (greedy cover, cap 12).
  Nothing is measured on the real BANC graph (DOI 10.7910/DVN/7WTH1N), which
  is not built on this machine.
- **Small pattern pool.** Four reserved cues leave ~25 pattern inputs. TEST
  is then nearly in-distribution (1.9 % novel pairs), and uncued arms forget
  to chance at flip 50. The effect sizes here (+0.17 to +0.23) are therefore
  larger than in #121 and are not comparable value-for-value with it. H4 is
  not evaluable. A follow-up needs a larger graph or fewer reserved lines per
  cue to test held-out generalisation properly.
- The context cue is assumed known at test time, as in #120/#121.
- 40 seeds; no multiplicity correction across H1–H5 or the three margins.
  Secondary flips are uncorrected.
- The equal budget is in episodes. Arms differ in weight writes: `E/E-cue`
  writes ~10× fewer than `supervised`, and the margin arms write more than
  `E/E-cue`.

B888-3 files (`src/circuit_explorer.rs`, subcircuit recall, activity atlas)
are untouched. No NVIDIA dependency, no secrets. CPU only.

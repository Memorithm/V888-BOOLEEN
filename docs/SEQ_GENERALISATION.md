# B888-4 cycle 18: do the equal-budget findings generalise to eight tasks and to held-out task orders? (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123), with the fixed pattern pool of
[CUE_REACH.md](CUE_REACH.md) (#124), the write budgets of
[WRITE_BUDGET.md](WRITE_BUDGET.md) (#126), the allocation code of
[WRITE_ALLOCATION.md](WRITE_ALLOCATION.md) (#127), the silent cue of
[CUE_ZERO_INIT.md](CUE_ZERO_INIT.md) (#128) and the settings of
[CUE_DISPLACEMENT.md](CUE_DISPLACEMENT.md) (#129). "Task", "cue" and "order"
name parts of an artificial benchmark on a Boolean graph with 4-bit weights.
Nothing here is a claim about synapses, biology, insect memory or BANC
physiology.

Module: `src/seq_generalisation.rs`. CLI: `bench-seq-generalisation`
(`--census` for the pilot).

## Motivation (what #125–#129 left open)

Every result of cycles 13–17 was measured on **one four-task sequence in one
fixed order** (T1→T2→T3→T4). #129 closed the silent cue's learning-gap
question at k = 4 (the rest of the loss is write displacement) and found the
cue's writes buy little at flip 50 (+0.026 ACC over the same pattern writes)
but clearly pay at flip 100 (+0.061). It suggested, untested, that the cue's
protection could matter more with **more tasks** and that the findings should
be checked on **task orders never used before**. This cycle tests exactly
that, on the unchanged update rules, at the same equal write budget:

* **Longer sequences:** k = 8 tasks (metrics also at the prefixes k = 2..7).
* **Held-out task orders:** besides the identity order T1→…→T8 (the order of
  every earlier cycle and of the pilot), each seed is run in **two seeded
  permutations of the tasks** that are drawn only in the main run (never in
  the pilot or in any earlier cycle). Each task keeps its own valence, cue,
  training set and episode order; only the order of training changes.

### Settings (6 rows per (flip, seed, order); same code paths as #129)

* `supervised@paced100` (**S1**, control). On this teacher-clamped task the
  associative co-activation rule is the same update as supervised
  (documented in [CONSOLIDATION.md](CONSOLIDATION.md) and
  [GENERALIZATION_FORGETTING.md](GENERALIZATION_FORGETTING.md)), so S1 stands
  for both; it is not run twice.
* `reward@paced100` (**R1**, reward-modulated, clean per-target reward).
* `error-gated@free` (**E**, = its `paced100` run, gated).
* `E/E-cue0@paced100` (**Z**, silent cue, every write charged).
* `error-gated@Z-pattern` (**Em**, uncued error gate capped at Z's
  per-episode cumulative pattern-write curve of the same (flip, seed, order)).
* `E/E-cue0@free` (**Zfree**, reference).

The exempt accounting of #129 (Zx) is not a setting of this cycle.

### Fixture for k tasks

The #124 fixed-pool design reserves `16 × 4 = 64` cue lines. For k tasks the
module keeps that **reserved total** and gives each cue `64 / k` lines (16 at
k = 4, 10 at k = 6, **8 at k = 8**), drawn by the same greedy disjoint
selection (`context_cue::choose_n_cues_capped`), removed from the pattern
pool. At k = 4 the fixture is exactly `novel_split::fixture` of #124–#129
(tested), and the identity-order run is bit-identical to
`cue_displacement::run_setting` for all six settings (tested).

### Design-stage pilot (seeds 1..5, flip 50, identity order only; no accuracy computed)

`results/seq-generalisation-census-v1.txt` (k = 8):

| quantity (5 seeds) | value |
|---|---|
| pattern pool (inputs) | 43–47 (k = 4 on the same seeds: 59–64; k = 6: 48–51) |
| lines per cue | 8 for all 8 cues (k = 4: 9–15, the greedy stops at full reach) |
| cue reach (targets of 32) | first cue 27–31, last cue 16–18 |
| CN pairs per (seed, task) | 33–97 (all ≥ 20) |
| E (= paced100 cap), writes over 8 tasks | 208–258 |
| Z @ paced100: pattern / cue writes | 160–190 / 48–81 |
| Em: pattern writes, refusals | 160–190, 1583–2375 (cap binds in 5/5) |
| Zfree: pattern / cue writes | 206–254 / 81–105 |

Readings that shaped the design (stated before the main run):

* At k = 8 Z gives up ≈ 25 % of `error-gated`'s pattern writes to the cue
  (≈ 37 % at k = 4 in #129), so the displacement test (H3) is still
  informative, and Em's cap binds.
* **The k = 8 fixture is not the k = 4 fixture with more tasks:** the
  pattern pool is smaller (≈ 45 vs ≈ 62 inputs) and later cues reach fewer
  targets. Any comparison of k = 8 with #129's k = 4 numbers mixes the
  sequence length with these fixture changes. The growth hypothesis (H5)
  therefore compares **prefixes of the same k = 8 runs** (k = 4 vs k = 8,
  same fixture, same rows), not k = 8 with #129.
* At flip 50 the label chain re-draws every edge label between consecutive
  tasks, so the task pairs are statistically exchangeable and the held-out
  orders mainly test robustness to new sequences; at flips 25 and 100 the
  order changes how much consecutive tasks conflict (at flip 100 tasks of
  equal parity are identical).

## Design (pre-registered)

* **Graph, task, pool, cues:** the #124–#129 fixed-pool design generalised
  to k tasks as above: generated graph per seed, 128 inputs / 32 targets /
  fan-in 8, threshold 1, init weight 1 (cue-line edges 0 for Z and Zfree),
  4-bit weights; patterns of 6 active inputs, 32 training patterns per task,
  one shared TEST set of 128 patterns; pair classes seen / CN / ICN / UE as
  in #123. Cued settings use task j's cue while training and evaluating
  task j, whatever its position in the order.
* **k = 8** (`K_RUN`), reference prefix k = 4 (`K_REF`).
* **Orders:** per seed, order 0 = identity; orders 1 and 2 = distinct seeded
  permutations ≠ identity (Fisher–Yates on an LCG keyed by
  `seed ^ 0x0DE5_B888_4C18_0A1D`), the same for every flip and setting.
* **Metrics in stage coordinates:** `R[i][j]` = the task trained at stage j
  measured on TEST after stage i (j ≤ i). ACC, LEARN, FGT, LAST at
  k = 2..8 on all / seen / CN / ICN / UE pairs, balanced accuracy, as in
  #123–#129. For held-out statistics each seed contributes the **mean over
  its two held-out orders** (n = 40 paired seeds).
* **Budget:** `paced100` defined per stage from uncapped `error-gated`'s
  per-episode cumulative write curve in the same (flip, seed, order). A
  write = a 4-bit step that changes a stored weight. Em's cap per episode is
  Z@paced100's cumulative pattern writes after that episode.
* **Seeds:** 321..360 (40 seeds, never used by an earlier B888-4 cycle).
  **Flips:** 25 (secondary), **50 (primary)**, 100 (high conflict). Same
  episode budget per task (8 rounds × 32 patterns), no rehearsal, original
  edges only, topology checked per row.
* **Instrumentation per (row, stage):** cap, charged writes, pattern writes,
  cue writes, refused, cue refused, first refusal, attempts, cue edges
  non-zero and cue weight mass at the end of the stage, weight digest,
  `init_ok`, `paced_identity`, topology, the order itself.
* **Checkpoint/resume:** circuit weights (`PlasticCircuit::save`) plus the
  text `Checkpoint` (k, order, stage, task, #129 per-task checkpoint);
  resuming mid-stage is bit-exact for every setting and corrupt checkpoints
  (bad magic, non-permutation order, stage out of range, task ≠ order[stage],
  wrong edge count, curve mismatch, truncation) are rejected (tested).
* **Reproducibility:** the grid runs (flip, seed) jobs on worker threads;
  rows and CSV do not depend on the thread count (tested). Two runs must
  give a byte-identical CSV (sha256 reported).

Statistics as in #123–#129: bootstrap 95 % percentile CI (2000 resamples,
seed 0xB0075EED); **strong** = mean ≥ 0.05, > 0 in ≥ 32 of 40 seeds and CI
lower bound > 0; **sequence rule** = Δ ACC strong, Δ LAST ≥ −0.05,
Δ LEARN ≥ −0.05; **equivalent** = 95 % CI inside (−0.05, +0.05). New for
this cycle: **positive** = mean > 0 and CI lower bound > 0.

## Design gates (reported first)

* **G1 (integrity):** every row keeps its topology, starts from its
  documented initial weights and makes charged writes ≤ its allowance;
  `error-gated@paced100` is bit-identical to uncapped `error-gated` with zero
  refusals in every (flip, seed, order); Em's per-stage cap equals Z's
  pattern writes; every order is a permutation, order 0 is the identity, the
  held-out orders differ from it and from each other, and all settings of a
  (flip, seed) use the same orders.
* **G2 (manipulation, flip 50, held-out orders):** Em refuses at least one
  step in ≥ 50 % of (seed, order) rows **and** Z makes at least one cue
  write in ≥ 50 % of them.
* **G3 (novel pairs):** median CN pairs per (seed, task) at flip 50 ≥ 20.

If any gate fails, H1–H8 are declared **not evaluable**.

## Hypotheses (k = 8, held-out orders, seeds 321..360)

* **H1 (error gate vs supervised control, flip 50):** ACC_all(E) −
  ACC_all(S1) is strong (#127–#129 replication at k = 8 on new orders).
* **H2 (silent cue vs supervised control, flip 50):** Z vs S1 passes the
  sequence rule on CN pairs (#128 H5 / #129 H8 at k = 8 on new orders).
* **H3 (displacement generalises, flip 50):** LEARN_all(Em) − LEARN_all(Z)
  is equivalent to 0 (#129 H3 at k = 8 on new orders).
* **H4 (the cue's writes pay at k = 8, flip 50):** ACC_all(Z) − ACC_all(Em)
  is strong (failed at k = 4 in #129: +0.026).
* **H5 (the cue's value grows with k, flip 50):** per seed,
  [ACC_all(Z) − ACC_all(Em)] at k = 8 minus the same at the k = 4 prefix of
  the same runs is positive.
* **H6 (high conflict, flip 100):** Z vs Em passes the sequence rule on all
  pairs (#129 secondary at k = 4 on the identity order).
* **H7 (order invariance, flip 50):** per seed, [ACC_CN(Z) − ACC_CN(S1)] on
  the held-out orders minus the same on the identity order is equivalent
  to 0.
* **H8 (ranking at equal writes, flip 50):** ACC_all(E) − ACC_all(R1) and
  ACC_all(R1) − ACC_all(S1) are both positive (error gate > reward >
  supervised).

**Secondary (reported, no pass/fail):** all of the above on the identity
order; k = 2, 4, 6 prefixes; flip 25; seen, ICN and UE pairs; Zfree vs E,
Z vs E, Em vs E, Em vs S1; growth of Z − S1, E − S1 and Z − E with k;
order shifts on all pairs; pattern / cue / total writes, refusals, non-zero
cue edges, cue mass, saturation.

No multiplicity correction is applied. The analysis is run once. Nothing
above the Results heading is changed after the run except the "pending"
placeholder.

## Command

```bash
cargo +1.89 run --release -- bench-seq-generalisation --csv results/seq-generalisation-v1.csv > results/seq-generalisation-report-v1.txt
# pilot (identity order, writes and pair counts only): cargo +1.89 run --release -- bench-seq-generalisation --census > results/seq-generalisation-census-v1.txt
```

## Results

Run once after the pre-registration commit 1636bc1 had been pushed. The grid
(2160 rows: 3 flips × 40 seeds × 3 orders × 6 settings, k = 8) takes about
50 s on 8 CPU threads (6.4 CPU-min). Two runs (8 and 3 worker threads) give a
byte-identical CSV (sha256 `e86601da…112786`, full
`e86601dac037781c0ff43b8240a1d4e49c6d0b821cf11270af3ce8032e112786`) and an
identical report. Every row keeps the original topology
(`topology_unchanged_all=true`). Files: `results/seq-generalisation-v1.csv`,
`results/seq-generalisation-report-v1.txt`, pilot
`results/seq-generalisation-census-v1.txt`.

### 1. Design gates (reported first)

* **G1 passes:** every row keeps its topology, starts from its documented
  initial weights and stays within its allowance; `error-gated@paced100` is
  bit-identical to uncapped `error-gated` with zero refusals in every
  (flip, seed, order); Em's caps equal Z's pattern writes; all orders are
  valid, distinct and shared by the settings.
* **G2 passes:** Em's cap binds and Z writes to its cue in 100 % of
  (seed, held-out order) rows at every flip.
* **G3 passes:** median CN pairs per (seed, task) 62 ≥ 20.

### 2. Settings at flip 50, k = 8, held-out orders (mean over 40 seeds × 2 orders)

| setting | ACC_all | ACC_CN | LEARN_all | LEARN_CN | FGT_all | FGT_CN | LAST_all | pattern writes | cue writes | total writes |
|---|---|---|---|---|---|---|---|---|---|---|
| supervised @ paced100 (S1) | 0.548 | 0.519 | 0.637 | 0.555 | 0.102 | 0.042 | 0.600 | 256 | 0 | 256 |
| reward @ paced100 (R1) | 0.571 | 0.557 | 0.602 | 0.556 | 0.035 | −0.001 | 0.557 | 256 | 0 | 256 |
| error-gated @ free (E) | 0.568 | 0.569 | 0.990 | 0.957 | 0.482 | 0.444 | 0.988 | 256 | 0 | 256 |
| **E/E-cue0 @ paced100 (Z)** | **0.609** | 0.579 | 0.884 | 0.841 | 0.314 | 0.299 | 0.874 | 184 | 72 | 256 |
| **error-gated @ Z-pattern (Em)** | 0.561 | 0.554 | 0.853 | 0.813 | 0.333 | 0.295 | 0.834 | 184 | 0 | 184 |
| E/E-cue0 @ free (Zfree) | 0.609 | 0.574 | 0.989 | 0.954 | 0.434 | 0.435 | 0.989 | 250 | 105 | 355 |

At k = 8 every uncued error-gated arm forgets about half of what it learns
(FGT_all 0.48): `error-gated` learns almost perfectly (LEARN 0.99) and ends
with the same ACC as `reward` (0.568 vs 0.571). The silent cue keeps the
highest final ACC at equal writes (0.609, +0.041 over E, 39/0/1).

### 3. Hypotheses (k = 8, held-out orders; mean [95 % CI] seeds better/tied/worse)

| | statistic | value | result |
|---|---|---|---|
| G1 | integrity (topology, init, caps, E unchanged, Em caps matched, orders valid) | all rows | **pass** |
| G2 | Em binds and Z writes its cue in ≥ 50 % of held-out rows, flip 50 | 100 % / 100 % | **pass** |
| G3 | median CN pairs ≥ 20 | 62 | **pass** |
| H1 | ACC_all(E) − ACC_all(S1) strong, flip 50 | +0.020 [+0.015, +0.026] 31/0/9 (mean < 0.05) | **fail** |
| H2 | Z vs S1, sequence rule, CN pairs, flip 50 | ACC_CN +0.060 [+0.049, +0.073] 39/0/1; Δ LEARN_CN +0.285, Δ LAST_CN +0.308 | **pass** |
| H3 | LEARN_all(Em) − LEARN_all(Z) equivalent to 0, flip 50 | −0.031 [−0.035, −0.027] 0/0/40 | **pass** |
| H4 | ACC_all(Z) − ACC_all(Em) strong, flip 50 | +0.047 [+0.044, +0.052] 40/0/0 (mean < 0.05) | **fail** |
| H5 | [Z − Em] ACC_all at k = 8 minus at k = 4 positive, flip 50 | +0.008 [+0.003, +0.013] 28/0/12 | **pass** |
| H6 | Z vs Em, sequence rule, all pairs, flip 100 | ACC +0.051 [+0.045, +0.057] 40/0/0; Δ LEARN +0.032, Δ LAST +0.032 | **pass** |
| H7 | [ACC_CN(Z) − ACC_CN(S1)] held-out − identity equivalent to 0, flip 50 | +0.004 [−0.014, +0.024] 20/0/20 | **pass** |
| H8 | ACC_all(E) − ACC_all(R1) positive and ACC_all(R1) − ACC_all(S1) positive, flip 50 | E − R1 −0.003 [−0.010, +0.003] 19/0/21; R1 − S1 +0.024 [+0.019, +0.029] 37/0/3 | **fail** |

### 4. Secondary (no pass/fail)

* **The error gate's equal-budget advantage shrinks with k** (E − S1,
  ACC_all, flip 50, held-out): +0.123 at k = 2, +0.054 at k = 4, +0.030 at
  k = 6, +0.020 at k = 8 (growth k = 8 vs 4: −0.033 [−0.041, −0.025]). The
  same holds on the identity order (+0.012 at k = 8) and at flip 25
  (growth −0.039). On CN pairs E − S1 is still +0.050 at k = 8 (37/0/3). Reward's
  advantage over supervised is +0.024 on all pairs and +0.039 on CN.
* **The silent cue's equal-budget advantage holds up better:** Z − S1,
  ACC_all, flip 50, held-out: +0.085 at k = 4, +0.061 at k = 8 (40/0/0); on
  the identity order +0.047 at k = 8 (39/0/1). At flip 25 +0.075 and at
  flip 100 +0.059 (all held-out, k = 8). Z − E: +0.041 (39/0/1).
* **Cue vs the same pattern writes (Z − Em, ACC_all, held-out):** +0.015 at
  k = 2, +0.039 at k = 4, +0.040 at k = 6, +0.047 at k = 8 at flip 50;
  +0.031 at flip 25; +0.051 at flip 100. On the identity order at k = 8:
  +0.041 at flip 50. The growth from k = 4 to k = 8 (H5) is small and is not
  seen on the identity order (−0.001 [−0.008, +0.006] 20/0/20).
* **Displacement at k = 8:** Em learns 0.031 less than Z on held-out orders
  (0.026 on the identity order), the same direction as #129's −0.016 at
  k = 4 and still within ±0.05. Zfree vs E: Δ LEARN −0.000, Δ ACC +0.041
  (free cue writes cost no learning).
* **Order effects:** held-out minus identity shifts of the equal-budget
  contrasts are small at flip 50 (all within ±0.015 on all pairs). At
  flip 100 the identity order alternates two identical label sets and gives
  very different write counts (E 467 vs 287 writes on held-out orders), so
  identity-order flip-100 numbers are not comparable with held-out ones.
* Two held-out orders per seed were drawn once; none of them was used in
  the pilot or an earlier cycle.

### Reading

On eight tasks and on task orders never used before, the **silent cue at
equal writes** is the arm whose #128/#129 results generalise: it beats the
supervised control on novel (CN) pairs with much more learning (H2), its
remaining learning loss is still write displacement (H3), it clearly beats
`error-gated` given the same pattern writes at high conflict (H6), and its
advantage over the control does not depend on the order (H7). Its writes
buy +0.047 ACC over the same pattern writes at flip 50, just under the
pre-registered 0.05 bar (H4 fails), and that gain grows only slightly from
k = 4 to k = 8 (H5 passes on held-out orders with +0.008, not on the
identity order).

The finding that does **not** generalise is the plain error gate's lead
over the supervised control: with more tasks it forgets about half of what
it learns, its ACC lead over supervised falls from +0.054 (k = 4) to +0.020
(k = 8) (H1 fails), and it no longer beats `reward` on all pairs (H8 fails;
reward > supervised still holds). All of this is on a synthetic graph with
artificial labels.

Next step suggested by this result (not tested here): at k = 8 the
bottleneck is forgetting by the error-gated rules, not learning; a fair
next test would bound forgetting at equal writes (for example, cue-gated
protection of earlier tasks' edges), on held-out orders.

## Limitations

- Synthetic graph and artificial OR-rule task only (128/32/8, active 6),
  threshold 1, init 1 (cue edges 0). Nothing is measured on the real BANC
  graph (DOI 10.7910/DVN/7WTH1N), which is not built on this machine.
- k = 8 changes the fixture (smaller pattern pool, 8-line cues with
  decreasing reach); comparisons with #129's k = 4 numbers are confounded,
  which is why H5 uses prefixes of the same runs.
- Two held-out orders per seed; orders are not stratified by conflict.
- H4 misses its bar by 0.003 and H5's effect (+0.008) is small; both are
  reported as pre-registered, without reinterpretation.
- Only the source-major allocation and the `paced100` budget are tested.
- The cue is assumed known at training and test time.
- 40 seeds; no multiplicity correction across H1–H8.

B888-3 files (`src/circuit_explorer.rs`, subcircuit recall, activity atlas
and their CLI wiring) are untouched. No NVIDIA dependency, no secrets. CPU
only.

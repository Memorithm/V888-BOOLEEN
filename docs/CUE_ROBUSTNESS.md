# B888-4 cycle 20: does the silent cue + last-resort win hold on fresh seeds, more held-out task orders and larger k, at equal write budget against the supervised control? (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123), with the fixed pattern pool of
[CUE_REACH.md](CUE_REACH.md) (#124), the write budgets of
[WRITE_BUDGET.md](WRITE_BUDGET.md) (#126), the silent cue of
[CUE_ZERO_INIT.md](CUE_ZERO_INIT.md) (#128), the k-task fixture of
[SEQ_GENERALISATION.md](SEQ_GENERALISATION.md) (#131) and the consolidation
guard engine of [CONSOLIDATION_GUARD.md](CONSOLIDATION_GUARD.md) (#132).
"Cue", "guard", "task order" and "forgetting" name parts of an update rule
and an artificial benchmark on a Boolean graph with 4-bit weights. Nothing
here is a claim about synapses, biology, insect memory or BANC physiology.

Module: `src/cue_robustness.rs`. CLI: `bench-cue-robustness` (`--census`
for the pilot).

## Motivation (what #132 left open)

Cycle 19 (#132) found one setting that improves retention without paying
for it in learning: the silent cue with the last-resort guard
(`E/E-cue0+last-resort@paced100`, **Zg**). At k = 8 tasks, flip 50, on two
held-out orders per seed (seeds 361..400), Zg reached ACC_all 0.644: +0.034
over the silent cue alone (Z), +0.097 over the supervised control at equal
writes (S1). That was H7. Every other guard traded forgetting for learning
about 1:1. That result rests on **one** seed block, **two** random orders
per seed and k ≤ 8. #132 also reported that the guard's gain over Z did not
grow from the k = 4 prefix to k = 8, and that it works because each task's
own cue lines give it free capacity. With 64 reserved cue lines split over
k cues, that capacity shrinks as k grows: 8 lines per cue at k = 8, 5 at
k = 12, 4 at k = 16.

This cycle is a **robustness test of the H7 win**, not a new mechanism:

1. **Fresh seeds** 401..440 (never used by an earlier B888-4 cycle).
2. **More and different held-out orders**: per seed, the **reversed** order
   plus **three** new seeded permutations (new salt), never used before.
3. **Larger k**: 8, 12 and 16 tasks, each on its own fixture.
4. Always at **equal budget**: every capped setting uses the `paced100`
   per-episode cumulative write curve of uncapped error-gated in the same
   (k, flip, seed, order), and the primary comparison is against the
   supervised control.

## Settings (5 rows per (k, flip, seed, order))

All are unchanged code paths of #132 (`consolidation_guard::run_setting`);
this cycle adds orders, k values, a cue-capacity instrument and the
analysis, not a new update rule.

* `supervised@paced100` (**S1**, control). On this teacher-clamped task the
  associative co-activation rule is the same update as supervised
  ([CONSOLIDATION.md](CONSOLIDATION.md)), so S1 stands for both.
* `reward@paced100` (**R1**, reward-modulated).
* `error-gated@free` (**E**, = its `paced100` run, gated in G1).
* `E/E-cue0@paced100` (**Z**, silent cue).
* `E/E-cue0+last-resort@paced100` (**Zg**, silent cue + last-resort guard;
  constants of #132: one overwrite per (episode, target) when no free edge
  exists).

## Design-stage pilot (seeds 1..5, flip 50, identity order only; no accuracy computed)

`results/cue-robustness-census-v1.txt` (writes and guard counters only),
ranges over 5 seeds:

| k (lines per cue) | writes (all settings) | E overwrites | Z cue writes | Z overwrites | Zg cue writes | Zg overwrites | Zg deferred |
|---|---|---|---|---|---|---|---|
| 8 (8) | 208–258 | 152–188 | 48–81 | 104–127 | 89–131 | 63–85 | 626–1156 |
| 12 (5) | 367–442 | 295–357 | 69–100 | 233–281 | 104–141 | 192–233 | 703–1368 |
| 16 (4) | 423–515 | 357–442 | 62–79 | 296–364 | 105–138 | 255–318 | 658–1183 |

Readings stated before the main run:

* The guard still acts at every k: Zg defers 600–1400 overwrites per run.
* **Its bite shrinks with k.** Zg's overwrites are 0.60–0.67 of Z's at
  k = 8, but 0.81–0.88 at k = 12 and 0.84–0.88 at k = 16. With 4–5 cue lines
  per task there is less private free capacity to divert overwrites to. This
  puts H5 (Zg > Z at k = 16) and H10 (no shrinkage of the lead over S1) at
  risk. They are kept as stated; the pilot did not tune anything.
* Writes per run roughly double from k = 8 to k = 16 (the budget is per
  task), so equal budget holds per stage and per run.

## Design (pre-registered)

* **Graph, task, pool, cues:** the #131/#132 k-task fixture
  (`seq_generalisation::fixture`): generated graph per seed, 128 inputs /
  32 targets / fan-in 8, threshold 1, init weight 1 (cue-line edges 0 for Z
  and Zg), 4-bit weights. Patterns have 6 active inputs, 32 training
  patterns per task and one shared TEST set of 128 patterns, with pair
  classes seen / CN / ICN / UE. The 64 reserved cue lines are split evenly
  over the k cues (floor): 8 lines at k = 8, 5 at k = 12, 4 at k = 16.
* **k = 8, 12 and 16**: separate runs on their own fixtures. Scaling within
  a run is compared on prefixes of the same k = 16 runs.
* **Orders:** per seed, order 0 = identity (secondary only). Held-out
  orders: order 1 = the **reversed** order (k−1, …, 0); orders 2, 3, 4 =
  distinct seeded Fisher–Yates permutations ≠ identity and ≠ reversed, drawn
  with the #131 LCG and a new salt `0x0DE5_B888_4C20_0A1D`. They are the
  same for every flip and setting and were drawn only in the main run.
* **Metrics in stage coordinates** as in #131/#132: `R[i][j]` = the task
  trained at stage j on TEST after stage i. ACC = mean_j R[k−1][j],
  LEARN = mean_j R[j][j], FGT = mean_{j<k−1} (R[j][j] − R[k−1][j]),
  LAST = R[k−1][k−1], on all / CN pairs, balanced accuracy. For held-out
  statistics each seed contributes the **mean over its four held-out
  orders** (n = 40 paired seeds) unless a hypothesis names one order.
* **Worst order:** per seed, min over the four held-out orders of the
  paired difference ACC_all(x) − ACC_all(base) in that order.
* **Seeds:** 401..440 (40 seeds). **Flips:** 25 (secondary), **50
  (primary)**, 100 (secondary). Same episode budget per task (8 rounds ×
  32 patterns), no rehearsal, original edges only, topology checked per row.
* **Instrumentation per row:** cap, charged and effective writes, cue
  writes, refused, overwrites, deferred, net reversals and consolidated
  edges per stage, **cue lines per task, cue-line edges, cue-line edges
  with non-zero weight at the end and cue-line edges at weight 15 at the
  end** (new: how much of the cue capacity is used), R diagonal and final
  row, saturation, weight digest, `init_ok`, `paced_identity`, topology.
* **Checkpoint/resume:** the #132 engine checkpoint (circuit weights +
  `consolidation_guard::Checkpoint`). Resuming mid-stage is re-tested
  bit-exact for Zg at k = 16 on the reversed order.
* **Equivalence:** every row is bit-identical to a direct
  `consolidation_guard::run_setting` call with the same fixture, order and
  limits (tested), so this cycle measures the #132 rules unchanged.
* **Reproducibility:** (k, flip, seed) jobs run on worker threads. Rows,
  CSV and report do not depend on the thread count (tested). Two runs must
  give a byte-identical CSV (sha256 reported).

Statistics as in #123–#132: bootstrap 95 % percentile CI (2000 resamples,
seed 0xB0075EED).
* **strong:** mean ≥ 0.05, > 0 in ≥ 32 of 40 seeds and CI lower bound > 0.
* **positive:** mean > 0 and CI lower bound > 0.
* **sequence rule:** Δ ACC strong, Δ LAST ≥ −0.05 and Δ LEARN ≥ −0.05.

## Design gates (reported first)

* **G1 (integrity):** every row keeps its topology, starts from its
  documented initial weights and makes charged writes ≤ its allowance.
  `error-gated@paced100` is bit-identical to uncapped E with zero refusals
  in every (k, flip, seed, order). Guard counters match the guard and the
  ledger (Zg: deferred allowed, nothing blocked, charged = effective).
  Orders are valid permutations, order 1 is the reversal, orders 0..4 are
  pairwise distinct and shared by all settings.
* **G2 (manipulation, flip 50, held-out orders):** at **each** k, Zg defers
  at least once and Z makes at least one cue write in ≥ 50 % of rows.
* **G3 (novel pairs):** median CN pairs per (seed, task) at k = 16, flip 50
  ≥ 20.

If any gate fails, H1–H10 are declared **not evaluable**.

## Hypotheses (held-out orders, seeds 401..440, flip 50)

* **H1 (primary, replication on fresh seeds and orders):** at k = 8, Zg vs
  S1 passes the sequence rule on all pairs.
* **H2 (#132 H7 replication):** at k = 8, Zg vs Z cuts FGT_all by a mean
  ≥ 0.05 with CI lower bound > 0, **and** ACC_all(Zg) − ACC_all(Z) is
  positive.
* **H3 (k = 12):** Zg vs S1 passes the sequence rule on all pairs.
* **H4 (k = 16):** Zg vs S1 passes the sequence rule on all pairs.
* **H5 (guard on the cue at k = 16):** ACC_all(Zg) − ACC_all(Z) is positive.
* **H6 (each held-out order, k = 8):** ACC_all(Zg) − ACC_all(S1) is positive
  on each of the four held-out orders taken alone (reversed, random 1, 2, 3).
* **H7 (worst order, k = 16):** the per-seed worst-order difference
  ACC_all(Zg) − ACC_all(S1) is positive.
* **H8 (vs error-gated, k = 16):** ACC_all(Zg) − ACC_all(E) is positive.
* **H9 (vs reward-modulated, k = 16):** ACC_all(Zg) − ACC_all(R1) is positive.
* **H10 (scaling, same k = 16 runs):** growth of ACC_all(Zg) − ACC_all(S1)
  from the k = 8 prefix to k = 16 has CI lower bound > −0.05 (the lead does
  not shrink by 0.05 or more).

**Secondary (reported, no pass/fail):** all of the above on the identity
order and at flips 25 / 100; CN pairs; Zg − Z, Z − S1, E − S1 and Zg − E per
k; per-order breakdown at k = 12 and 16; growth of Zg − Z and E − S1 in the
k = 16 runs; deferred, overwrites, cue writes and cue-capacity use per k.

No multiplicity correction is applied. The analysis is run once. Nothing
above the Results heading is changed after the run except the "pending"
placeholder.

## Command

```bash
cargo +1.89 run --release -- bench-cue-robustness --csv results/cue-robustness-v1.csv > results/cue-robustness-report-v1.txt
# pilot (identity order, writes and guard counters only): cargo +1.89 run --release -- bench-cue-robustness --census > results/cue-robustness-census-v1.txt
```

## Results

Run once after the pre-registration commit 6835fbc had been pushed. The
grid has 9000 rows: 3 k × 3 flips × 40 seeds × 5 orders × 5 settings. It
takes about 8.5 min on 8 CPU threads (64 CPU-min). Two runs (8 and 3 worker
threads) give a byte-identical CSV (sha256 `adbc6367…0fa254`, full
`adbc636785a54bfc29263b31686dae27768581ae7064ab6014a15363670fa254`) and an
identical report. Every row keeps the original topology
(`topology_unchanged_all=true`). Files: `results/cue-robustness-v1.csv`,
`results/cue-robustness-report-v1.txt`, pilot
`results/cue-robustness-census-v1.txt`.

### 1. Design gates (reported first)

* **G1 passes:** every row keeps its topology, starts from its documented
  initial weights and stays within its allowance. `error-gated@paced100` is
  bit-identical to uncapped E with zero refusals everywhere. Guard counters
  match the guard and the ledger. Orders are valid, order 1 is the reversal
  and orders are distinct and shared.
* **G2 passes:** at flip 50 on held-out orders, Zg defers and Z writes cue
  edges in 100 % of rows at k = 8, 12 and 16.
* **G3 passes:** median CN pairs per (seed, task) is 66 at k = 16 (63 at
  k = 8, 79 at k = 12).

### 2. Settings at flip 50, held-out orders (mean over 40 seeds × 4 orders)

| k (cue lines per task) | S1 | R1 | E | Z | **Zg** | Zg FGT_all | Zg LEARN_all | Zg deferred | live cue edges at end (Z / Zg) |
|---|---|---|---|---|---|---|---|---|---|
| 8 (8) | 0.545 | 0.569 | 0.569 | 0.605 | **0.643** | 0.245 | 0.858 | 894 | 36 / 55 |
| 12 (5) | 0.535 | 0.557 | 0.551 | 0.577 | **0.593** | 0.316 | 0.883 | 936 | 37 / 53 |
| 16 (4) | 0.533 | 0.549 | 0.542 | 0.564 | **0.575** | 0.355 | 0.908 | 886 | 37 / 51 |

ACC_all shown. All settings make the same number of writes within a
(k, flip, seed, order) up to 0.3 % (k = 8: 257.6, k = 16: 511.8 effective
writes per run). Zg is the best setting at every k. Its k = 8 numbers
replicate #132 on new seeds and orders (ACC_all 0.643 vs 0.644, FGT 0.245
vs 0.245).

### 3. Hypotheses (flip 50, held-out orders; Δ = mean [95 % CI] better/ties/worse over 40 seeds)

* **H1 pass (primary).** k = 8, Zg vs S1: ACC_all +0.098 [+0.092, +0.104]
  40/0/0, LEARN +0.225, LAST +0.229 (FGT is 0.145 worse: Zg learns much more
  and keeps more of it).
* **H2 pass.** k = 8, Zg vs Z: FGT cut +0.074 [+0.068, +0.080] 40/0/0 and
  ACC_all +0.038 [+0.034, +0.042] 40/0/0 (#132: +0.071 and +0.034).
* **H3 pass.** k = 12, Zg vs S1: ACC_all +0.058 [+0.054, +0.063] 40/0/0,
  LEARN +0.267, LAST +0.277.
* **H4 fail.** k = 16, Zg vs S1: ACC_all +0.042 [+0.038, +0.047] 40/0/0.
  Positive in every seed, but the mean is below the 0.05 "strong" bound of
  the sequence rule (LEARN +0.301 and LAST +0.309 pass).
* **H5 pass.** k = 16, Zg − Z ACC_all = +0.011 [+0.008, +0.013] 37/0/3. The
  FGT cut is +0.026, a third of its k = 8 size.
* **H6 pass.** k = 8, Zg − S1 ACC_all on each held-out order alone: reversed
  +0.106 [+0.098, +0.114], random +0.094, +0.096, +0.096, all with CI lower
  bound > 0 and 40/0/0 each.
* **H7 pass.** k = 16, worst held-out order per seed, Zg − S1 ACC_all =
  +0.024 [+0.019, +0.029] 37/0/3.
* **H8 pass.** k = 16, Zg − E ACC_all = +0.033 [+0.030, +0.036] 40/0/0.
* **H9 pass.** k = 16, Zg − R1 ACC_all = +0.026 [+0.022, +0.030] 38/0/2.
* **H10 pass.** In the same k = 16 runs, growth of Zg − S1 from the k = 8
  prefix to k = 16 is −0.010 [−0.015, −0.006] 7/0/33. The CI lower bound is
  above −0.05, so the lead does not shrink by 0.05 or more within a run. It
  does shrink a little: the sign is negative in 33 of 40 seeds.

### 4. Secondary

* **The lead over S1 shrinks with cue capacity, not with run length.**
  Zg − S1 is +0.098 at k = 8 with 8 cue lines per task. In the k = 16 runs
  (4 lines per task) it is already about +0.052 at the k = 8 prefix and
  +0.042 at k = 16. Within those runs the k = 12 → 16 growth is −0.002
  [−0.006, +0.002]. The capacity readout agrees: Zg keeps about 51–55 live
  cue edges at every k, so per task that is ≈ 6.8 at k = 8 and ≈ 3.2 at
  k = 16. Z − S1 follows the same path (+0.060, +0.042, +0.032), and so does
  E − S1 without any cue (+0.023, +0.016, +0.009). So part of the decline is
  the error gate's own k-trend of #131.
* **The guard's own gain fades with k:** Zg − Z is +0.038 (k = 8), +0.016
  (k = 12) and +0.011 (k = 16). On the worst held-out order it is +0.020 at
  k = 8 but +0.000 [−0.003, +0.003] at k = 12 and −0.002 [−0.005, +0.001]
  at k = 16. This matches the pilot (Zg's overwrites fall from 0.6 of Z's
  at k = 8 to 0.86 at k = 16). At k ≥ 12 the guard is a small average gain
  that does not hold on every order.
* **Order robustness:** at every k, Zg − S1 is positive on each held-out
  order alone (k = 12: +0.051 to +0.064; k = 16: +0.038 to +0.050). The
  reversed order is never the worst. The identity order gives the same
  picture (Zg − S1: +0.091, +0.051, +0.037).
* **Flips:** Zg vs S1 passes the sequence rule at flip 25 and flip 100 at
  every k, including k = 16 (+0.050 [+0.045, +0.055] at flip 25 and +0.052
  [+0.048, +0.056] at flip 100). So H4's failure at flip 50 is a near miss,
  not a reversal.
* **CN pairs:** Zg − S1 on CN pairs is +0.089 (k = 8), +0.052 (k = 12) and
  +0.035 (k = 16). Zg − Z on CN at k = 16 is +0.004 [−0.001, +0.010], not
  positive.
* Zg defers 890–940 overwrites per run at every k. No setting saturates a
  cue edge (0 cue-line edges at weight 15 at the end).

**Reading:** on fresh seeds (401..440), the reversed order and three new
random orders, the #132 win replicates exactly at k = 8. Zg beats the
supervised control at equal writes on every held-out order (H1, H6) and
still cuts the silent cue's forgetting (H2). At k = 12 it still passes the
sequence rule against S1 (H3). At k = 16 it stays the best setting and beats
S1 in 40 of 40 seeds, beats S1 on the worst order and beats E and R1 (H5,
H7–H9). Its lead (+0.042) is just under the pre-registered 0.05 bound, so
H4 fails. The decline comes from cue capacity: with 64 cue lines shared by
16 tasks, each task gets 4. The guard's extra gain over the plain silent
cue becomes small at k ≥ 12 and is ≈ 0 on the worst order. Under a fixed
cue-line budget, the cue + guard advantage shrinks as tasks are added. The
next lever is cue capacity per task, not more guarding.

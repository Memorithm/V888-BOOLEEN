# B888-4 cycle 28: does the partner-write dose effect **generalise**, does it **hold up on forgetting** against the supervised control S1 at equal write budget, and **why was the cycle 27 dose curve linear**? (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not built on the
build machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123), run through the **unchanged** cycle 27
engine of [CUE_WRITE_DOSE.md](CUE_WRITE_DOSE.md) (#186). "Cue", "code",
"partner", "dose", "frozen", "forgetting" and "transfer" name parts of an
update rule and an artificial benchmark on a Boolean graph with 4-bit
weights. Nothing here is a claim about synapses, biology, insect memory or
BANC physiology.

Module: `src/cue_dose_transfer.rs`. CLI: `bench-cue-dose-transfer`
(`--census` for the label-free pilot).

## 1. Where cycle 27 left off (#186, main `61784bdc`)

At k = 16, flip 100 (held-out orders, seeds 681..720, synthetic):

* held-out ACC_all rises with the number of writable partner lines of the
  matched share-2 code: dose slope +0.026 [+0.024, +0.029] (H1 pass); each
  extra line costs learning (−slope LEARN +0.048) and buys a larger
  forgetting cut (−slope FGT +0.079);
* **H4 failed**: the pre-registered *diminishing-returns* hypothesis
  (concavity f(½) − (f(0) + f(1))/2 of ACC_all > 0) gave +0.000 [−0.001,
  +0.002], 20/0/20 — the curve is linear (0.577, 0.584, 0.590, 0.597,
  0.603);
* line identity did not matter (H5: Zgd2 − Zgd2c +0.000, CI inside ±0.01);
* against S1 at equal budget every dosed arm won on ACC/LEARN/LAST (H6, H7)
  but **forgot more in raw terms** (Zgd4: FGT 0.236 vs S1 0.129). S1 also
  learns much less (LEARN 0.628 vs 0.824; ACC 0.506, i.e. near chance).

Two questions were left open, and both are addressed here, **before any
measurement**:

1. Is the dose effect a property of that one fixture, or does it
   generalise to other fixture conditions and fresh seeds, and does it hold
   up on forgetting against S1 at equal write budget — in raw terms and
   relative to how much each arm learned?
2. Why was the curve linear (H4)?

## 2. Diagnosis hypothesis for H4 (stated before measuring)

The nested doses take the first ⌊d·|F_j|/4⌋ lines of a **uniformly random**
seeded permutation of the partner lines F_j. Write f(S) for the outcome when
the partner-line set S ⊆ F_j is writable. If line identity does not matter
on average (cycle 27 H5) and lines have no interaction, f is additive and,
by the symmetry of the random permutation, E[f(dose d)] is **exactly linear
in d**. A concave curve (diminishing returns) needs *redundancy*: a negative
interaction between partner lines, so that two halves together buy less
than the sum of what each buys alone. Cycle 27 measured the two disjoint
halves (Zgd2 and its complement Zgd2c) and the union (Zgd4), so the
interaction can be measured directly:

  I = f(Zgd4) − f(Zgd2) − f(Zgd2c) + f(Zgd0)  (0 if the halves add up,
  < 0 if they are redundant).

Prediction (pre-registered): I is equivalent to 0 (95 % CI inside ±0.01),
the concavity replicates as ≈ 0, and the manipulation itself is linear (the
partner-line moves grow linearly with the dose; cycle 27 G2: 0 / 9.6 / 19.2
/ 28.9 / 37.7). Under that reading H4 failed because partner lines act
additively at |F_j| = 4, not because of noise or a weak manipulation.
**Not computed post hoc on the cycle 27 CSV**: it is measured here on fresh
seeds only.

## 3. Conditions (pre-registered; one design field changed per condition)

All conditions use the cycle 27 engine unchanged
(`cue_write_dose::run_grid_threads`): ten arms S1, R1, E, Zg1, Zgd0..Zgd4,
Zgd2c; the same matched share-2 code, constant whole-stage dose masks
(dose salt `0x0DE5_B888_4C27_D05E`), `paced100` equal write budget (every
capped arm uses the per-episode cumulative write curve of uncapped
error-gated E in the same (seed, order)); same guard, cue init 0,
last-resort; 4-bit weights; no rehearsal; topology checked per row.

| condition | change vs cycle 27 | purpose |
|---|---|---|
| **base** | none (128 sources, 32 targets, fan-in 8, 32 training patterns per task) | replication on fresh seeds |
| **train64** | 64 training patterns per task (test set unchanged: 128 patterns) | generalisation to a larger training set |
| **src256** | 256 graph sources (targets 32, fan-in 8 unchanged) | generalisation to a larger, sparser graph |

* **k = 16, flip 100 only** (the cycle 27 primary cell; |F_j| = 4 in every
  condition, checked by a test on seed 1 before this document was written —
  fixture construction only, no learning, no accuracy).
* **Seeds 721..760** (40 seeds, never used by an earlier B888-4 cycle).
* **Orders:** 0 = identity (secondary only), held-out 1 = reversed, 2..4 =
  seeded Fisher–Yates permutations. **Declared deviation from the usual
  "new salt per cycle":** the cycle 27 order salt `0x0DE5_B888_4C27_0A1D`
  is reused so that the engine is bit-for-bit the cycle 27 engine; the
  orders are still new because they are keyed by the new seeds.
* **Metrics** as #131–#186 on the shared TEST set: ACC, LEARN, FGT, LAST on
  all / CN pairs (balanced accuracy); per seed the mean over its 4 held-out
  orders (n = 40 paired seeds).
* **Relative forgetting** (new derived quantity, defined here before any
  measurement, computed only from the existing stage matrix R[i][j]): per
  seed and arm, RF = mean FGT_all / (mean LEARN_head − 0.5), where FGT_all is
  the usual mean over tasks j < k of R[j][j] − R[k][j] and LEARN_head is the
  mean of R[j][j] over the **same** tasks j < k (chance of the balanced
  accuracy = 0.5); both means are over the seed's held-out orders. RF is the
  share of the above-chance learning that is lost by the end of the
  sequence. Motivation (declared): in cycle 27 S1 forgot little in raw
  terms partly because it learned little, so a raw-FGT comparison alone
  cannot say whether a dosed arm "forgets more" of what it learned.
* **Dose slope** and **concavity** exactly as cycle 27 (least-squares slope
  on q ∈ {0, ¼, ½, ¾, 1}; f(½) − (f(0) + f(1))/2).
* **Statistics** as #123–#186: bootstrap 95 % percentile CI (2000
  resamples, seed 0xB0075EED); better/ties/worse over 40 seeds.
  * positive: mean > 0 and CI lower bound > 0;
  * sequence rule: ΔACC strong (mean ≥ 0.05, > 0 in ≥ 32 of 40 seeds, CI
    lower bound > 0), ΔLAST ≥ −0.05 and ΔLEARN ≥ −0.05;
  * equivalent: the 95 % CI lies strictly inside ±0.01;
  * non-inferior (H5 only): CI lower bound of FGT(S1) − FGT(Zgd4) > −0.02.
* **Reproducibility:** the main run is repeated with a different thread
  count and must give a byte-identical CSV (sha256 reported).

## 4. Design gates (per condition, reported first)

* **G1 (integrity):** the cycle 27 G1 unchanged (topology, initial weights,
  charged writes ≤ allowance, `error-gated@paced100` = uncapped E, guard
  counters, orders, codes, dose masks and writable-line counts, frozen
  moves = 0, per-stage budgets equal across arms).
* **G2 (manipulation):** the cycle 27 G2 clauses (a)–(d) at k 16, flip 100,
  held-out rows (partner-line moves strictly increase along Zgd0..Zgd4;
  Zgd1 moves a partner line in ≥ 50 % of rows; Zgd1/2/3/2c skip frozen
  steps in ≥ 50 % of rows; Zgd2c moves within a factor 2 of Zgd2).
* **G3 (novel pairs):** median CN pairs per (seed, task) at k 16, flip 100
  ≥ 20. **Declared deviation:** cycle 27 checked G3 at flip 50, which is
  not run here.
* **G4 (leakage):** for every seed, no TEST pattern (set of active inputs)
  equals any training pattern of any task; all arms of a (seed, order) are
  scored on the same TEST set (by construction of the shared fixture) and
  no arm trains on TEST patterns.
* **G5 (RF defined):** LEARN_head − 0.5 ≥ 0.05 for every seed for S1 and
  Zgd4.

If G1–G4 fail in a condition, every hypothesis that uses the condition is
**not evaluable**. If only G5 fails in a condition, only H6 / H7 are not
evaluable there.

## 5. Hypotheses (k = 16, flip 100, held-out orders, seeds 721..760)

Generalisation of the dose effect:

* **H1 (replication):** base, dose slope of ACC_all positive.
* **H2 (larger training set):** train64, dose slope of ACC_all positive.
* **H3 (larger graph):** src256, dose slope of ACC_all positive.
* **H4 (forgetting side generalises):** −(dose slope of FGT_all) positive
  in **each** of base, train64 and src256.

Forgetting against S1 at equal write budget:

* **H5 (raw forgetting, non-inferiority):** base, FGT(S1) − FGT(Zgd4)
  non-inferior (CI lower bound > −0.02). (Cycle 27 suggests this fails; it
  is registered as the plain reading of "holds up on forgetting".)
* **H6 (relative forgetting):** base, RF(S1) − RF(Zgd4) positive (Zgd4
  loses a smaller share of what it learned).
* **H7 (relative forgetting generalises):** RF(S1) − RF(Zgd4) positive in
  **both** train64 and src256.
* **H8 (vs control generalises):** Zgd4 vs S1 passes the sequence rule in
  **both** train64 and src256.
* **H9 (novel pairs):** ACC_cn(Zgd4) − ACC_cn(S1) positive in **each** of
  base, train64 and src256.

Diagnosis of the cycle 27 H4 failure:

* **H10 (linearity replicates):** base, the concavity of ACC_all is
  equivalent to 0 (CI inside ±0.01).
* **H11 (additivity):** base, the interaction I of ACC_all is equivalent to
  0 (CI inside ±0.01).
* **H12 (additivity generalises):** the interaction I of ACC_all is
  equivalent to 0 in **both** train64 and src256.

No multiplicity correction (as #123–#186); every hypothesis is reported
with its mean, CI and better/ties/worse counts, pass or fail.

**Secondary (descriptive, no pass/fail):** per condition: dose slopes of
LEARN_all, ACC_cn and partner-line moves; concavity and interaction of
FGT_all, LEARN_all and partner-line moves; Zgd2 − Zgd2c; Zgd0 / Zgd2 / Zgd4
vs S1 on ACC, FGT, LEARN, LAST and RF; worst held-out order of Zgd4 − S1
and Zgd2 − Zgd2c; per-arm summaries (ACC, LEARN, LEARN_head, FGT, LAST, RF,
writes, cue writes, partner-line moves, skips); identity order; idle stages.

## 6. Pilot (label-free, before the main run)

`bench-cue-dose-transfer --census` on seeds 1..5, identity order, all three
conditions: writes, cue writes, skips, frozen moves, partner-line moves,
writable lines, idle stages, partner agreement, cue reach and the G4
leakage check. No accuracy is computed and no held-out order is drawn. The
pilot can only lead to stopping the cycle before the main run (e.g. if a
dose cannot bind in a condition); it cannot change a threshold, an arm, a
condition, a seed or a hypothesis above.

## 7. Pilot result (label-free, seeds 1..5, identity order; written before the main run)

File: `results/cue-dose-transfer-census-v1.txt` (no accuracy computed, no
held-out order drawn; sha256
`b409b862b16cfd365429cbd01a2fb4947ef82360e6086a167510f25b5fe2b922`). Means over the 5 pilot seeds, k = 16, flip 100:

| condition | writable lines Zgd0..4 / 2c | partner-line moves Zgd0 / 1 / 2 / 3 / 4 / 2c | frozen skips Zgd0 / 1 / 2 / 3 / 4 / 2c | cue writes Zg1 / Zgd0 / Zgd2 / Zgd4 | frozen moves | leak-free fixtures |
|---|---|---|---|---|---|---|
| base | 0, 1, 2, 3, 4 / 2 | 0 / 17.2 / 32.8 / 47.2 / 61.2 / 32.4 | 809 / 600 / 419 / 207 / 0 / 410 | 305.8 / 309.8 / 410.0 / 491.8 | 0 | 5/5 |
| train64 | 0, 1, 2, 3, 4 / 2 | 0 / 19.0 / 36.2 / 51.2 / 65.4 / 34.6 | 1336 / 1023 / 633 / 363 / 0 / 661 | 294.4 / 297.4 / 397.8 / 481.4 | 0 | 5/5 |
| src256 | 0, 1, 2, 3, 4 / 2 | 0 / 14.2 / 26.8 / 39.2 / 55.0 / 31.8 | 489 / 357 / 273 / 106 / 0 / 240 | 312.6 / 317.8 / 420.8 / 531.4 | 0 | 5/5 |

* In every condition the partner-line moves rise with the dose, Zgd1 moves
  partner lines, Zgd1..3 / Zgd2c skip frozen steps, Zgd2c receives a dose
  comparable to Zgd2 (ratio 0.96 / 0.96 / 1.19), frozen moves are 0 and
  every fixture is leakage-free. The identity order has no idle stage.
* The base pilot reproduces the cycle 27 pilot numbers exactly (same pilot
  seeds; e.g. cue writes 309.8 / 491.8, moves 17.2 / 32.8 / 47.2 / 61.2),
  which confirms the engine is unchanged.
* **Expectation stated before the main run:** the G2 clauses bind in every
  condition, so G1–G4 are expected to pass and H1–H12 to be evaluable. The
  pilot measured no accuracy and changed no threshold, arm, condition, seed
  or hypothesis. The main run is executed exactly as pre-registered above.

## 8. Main run result (seeds 721..760, held-out orders; synthetic graph)

Files: `results/cue-dose-transfer-report-v1.txt`,
`results/cue-dose-transfer-v1.csv` (6 000 rows = 3 conditions × 40 seeds ×
5 orders × 10 arms; sha256
`2a4c6bf7ffa947715ea0fdf898ad05e143ac7d227e82364dd3056e2e33110fd1`, report
sha256 `ef7f9800202e90bf90435e62b6dcf5b25f1d0c6ed8a1e99eb31b041c2bbcded6`).
The run was executed exactly as pre-registered above; no threshold, arm,
condition, seed or hypothesis was changed after the pilot. Reproducibility: the CSV and the report are byte-identical with `--threads 8` and `--threads 6` (same sha256 in both).

**Deviations:** none from the hypotheses, thresholds, arms, conditions or
seeds. After the pilot only the unit tests of `src/cue_dose_transfer.rs`
were made lighter (test grid on one seed instead of two, inside
`#[cfg(test)]`, to fit the qualification time budget); the benchmark binary
that produced the results was built from the pre-registration commit and
no measurement code changed. The two declared design deviations of §3–§4
(order salt reused, G3 checked at flip 100) were stated before measuring.

### Design gates

| condition | G1 integrity | G2 manipulation (partner-line moves Zgd0..4 / 2c) | G3 median CN | G4 leak-free fixtures | G5 RF defined |
|---|---|---|---|---|---|
| base | **pass** | **pass** (0 / 10.1 / 19.6 / 28.8 / 37.9 / 19.2) | **pass** (64) | **pass** (40/40) | **pass** |
| train64 | **pass** | **pass** (0 / 9.6 / 18.6 / 27.6 / 36.1 / 18.6) | **pass** (39) | **pass** (40/40) | **pass** |
| src256 | **pass** | **pass** (0 / 11.0 / 20.6 / 30.2 / 40.7 / 20.6) | **pass** (66) | **pass** (40/40) | **pass** |

Idle stages (descriptive, as diagnosed in cycle 27): 3.02 (base), 5.16
(train64) and 0.77 (src256) of 16 stages per held-out row on average;
identical across the arms of an order (G1), no gate or hypothesis depends
on them.

### Hypotheses (k = 16, flip 100; mean ± sd [95 % bootstrap CI] better/ties/worse over 40 seeds)

| H | contrast | value | verdict |
|---|---|---|---|
| H1 | base, dose slope of ACC_all | +0.022 ± 0.011 [+0.018, +0.025] 40/0/0 | **pass** |
| H2 | train64, dose slope of ACC_all | +0.022 ± 0.008 [+0.020, +0.025] 40/0/0 | **pass** |
| H3 | src256, dose slope of ACC_all | +0.014 ± 0.008 [+0.011, +0.017] 40/0/0 | **pass** |
| H4 | −(dose slope of FGT_all), each condition | base +0.077 [+0.073, +0.082]; train64 +0.078 [+0.074, +0.082]; src256 +0.060 [+0.057, +0.063]; all 40/0/0 | **pass** |
| H5 | base, FGT(S1) − FGT(Zgd4) non-inferior (CI lo > −0.02) | −0.107 ± 0.019 [−0.113, −0.101] 0/0/40 | **fail** |
| H6 | base, RF(S1) − RF(Zgd4) | +0.247 ± 0.051 [+0.232, +0.263] 40/0/0 | **pass** |
| H7 | RF(S1) − RF(Zgd4), train64 and src256 | train64 +0.252 [+0.234, +0.269]; src256 +0.145 [+0.132, +0.158]; both 40/0/0 | **pass** |
| H8 | Zgd4 vs S1 sequence rule, train64 and src256 | train64 ΔACC +0.091 [+0.086, +0.097] 40/0/0, ΔLEARN +0.196, ΔLAST +0.194 (passes); src256 ΔACC +0.0497 [+0.046, +0.053] 40/0/0, ΔLEARN +0.158, ΔLAST +0.166 (ΔACC mean below the 0.05 bar) | **fail** (src256) |
| H9 | ACC_cn(Zgd4) − ACC_cn(S1), each condition | base +0.105 [+0.095, +0.116] 40/0/0; train64 +0.120 [+0.107, +0.132] 40/0/0; src256 +0.057 [+0.047, +0.066] 38/0/2 | **pass** |
| H10 | base, concavity of ACC_all ≡ 0 (±0.01) | −0.002 ± 0.006 [−0.004, −0.000] 11/0/29 | **pass** |
| H11 | base, interaction I of ACC_all ≡ 0 (±0.01) | +0.001 ± 0.005 [−0.001, +0.002] 24/0/16 | **pass** |
| H12 | interaction I of ACC_all ≡ 0, train64 and src256 | train64 −0.001 [−0.002, +0.001]; src256 −0.001 [−0.002, +0.001] | **pass** |

(H8 src256: the exact mean ΔACC over the 40 seeds is 0.04975, from the
6-decimal CSV; the sequence rule requires ≥ 0.05.)

### Readings (descriptive, synthetic benchmark only)

* **The dose effect generalises.** On fresh seeds the base slope is +0.022
  (cycle 27: +0.026). Doubling the training patterns leaves it unchanged
  (+0.022). Doubling the graph sources shrinks it to +0.014, but it is still
  positive in 40/40 seeds. The forgetting side generalises too: each extra
  partner line cuts forgetting in every condition (−slope FGT +0.060 to
  +0.078), and costs learning (−slope LEARN +0.042 to +0.051).
* **Raw forgetting against S1 does not hold up.** At equal write budget,
  Zgd4 forgets more than S1 in all three conditions (FGT(S1) − FGT(Zgd4) =
  −0.107 base, −0.112 train64, −0.115 src256; 0/0/40 each), so H5 fails.
  Owned Zgd0 forgets even more (−0.185 base). The dose narrows the raw gap
  to S1 but does not close it.
* **Relative to what each arm learned, the dosed arm forgets less.** S1
  loses almost all of its above-chance learning by the end of the sequence
  (RF ≈ 0.99 in every condition), and its final ACC_all is at chance
  (0.506–0.507). Zgd4 loses 0.74 (base), 0.74 (train64) and 0.85 (src256)
  of it. RF falls along the dose (base 0.846 / 0.823 / 0.800 / 0.771 /
  0.739 for Zgd0..4). At flip 100 consecutive tasks have opposite labels,
  so a large RF is expected for every arm (uncapped E: 1.02, i.e. old tasks
  end slightly below chance).
* **Against S1 the cued arms still win on accuracy, but the margin depends
  on the graph.** ΔACC Zgd4 − S1 is +0.089 (base), +0.091 (train64) and
  +0.050 (src256). On the 256-source graph the margin falls just short of
  the pre-registered 0.05 bar (H8 fails). Novel-pair accuracy against S1
  follows the same pattern (+0.105 / +0.120 / +0.057). The dose slope of
  ACC_cn is ≈ 0 in every condition (base −0.005 [−0.016, +0.006]; train64
  −0.008 [−0.015, −0.001]; src256 +0.000): the advantage on novel pairs
  comes from the cue code itself, not from the partner-line dose. This
  replicates the weak cycle 27 H11.
* **Why cycle 27 H4 failed: the partner lines add up.** The two disjoint
  halves of F_j together buy what each buys alone, summed: I = +0.001
  (base), −0.001 (train64), −0.001 (src256), with every CI within [−0.002, +0.002].
  With no interaction and a uniformly random nested order, the expected
  dose curve is linear, and it is (base concavity −0.002 [−0.004, −0.000],
  a very slight convexity; train64 +0.001; src256 +0.001). The manipulation
  is close to linear as well (partner-line moves 0 / 10.1 / 19.6 / 28.8 /
  37.9, concavity +0.7 moves at base). H4 did not fail because the run was
  noisy or the manipulation was weak. It failed because partner lines act
  additively at |F_j| = 4 on this benchmark, so there are no diminishing
  returns to find.
* **Secondary.** The interactions of FGT_all and LEARN_all are also ≈ 0
  (|I| ≤ 0.003). The interaction of partner-line moves is slightly negative
  (base −0.95 moves [−1.47, −0.41]): the two halves together move about one
  edge fewer than each alone, summed. This does not show up in accuracy.
  Line identity: base Zgd2 − Zgd2c −0.003 [−0.006, −0.000] 13/0/27 (train64
  +0.002, src256 +0.002, both CIs include 0); the worst held-out order of
  Zgd2 − Zgd2c is negative, as in cycle 27, and biased below zero by
  construction (minimum of four noisy differences).
* Absolute effects are small (owned → shared +0.014 to +0.022 ACC) and
  every number comes from a generated graph with artificial labels. No
  claim about synapses, biology, insect memory or BANC physiology is made.

## Reproduce

```bash
cargo +1.89 run --release -- bench-cue-dose-transfer --census > results/cue-dose-transfer-census-v1.txt
cargo +1.89 run --release -- bench-cue-dose-transfer --csv results/cue-dose-transfer-v1.csv > results/cue-dose-transfer-report-v1.txt
```

# B888-4 cycle 27: does the **number** of writable partner lines of the matched share-2 cue code rank generalisation and forgetting at equal write budget? (diagnosis of the cycle 25 / 26 G2 failures, then a dose design; synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not built on the
build machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123), with the k-task fixture of
[SEQ_GENERALISATION.md](SEQ_GENERALISATION.md) (#131), the consolidation guard
engine of [CONSOLIDATION_GUARD.md](CONSOLIDATION_GUARD.md) (#132), the
order/statistics protocol of [CUE_ROBUSTNESS.md](CUE_ROBUSTNESS.md) (#133), the
share-2 codes of [CUE_CAPACITY.md](CUE_CAPACITY.md) (#134), the agreement
matrix of [CUE_SIMILARITY.md](CUE_SIMILARITY.md) (#135), the partner tours of
[CUE_PARTNER.md](CUE_PARTNER.md) (#136), the frozen-line mask of
[CUE_OWNERSHIP.md](CUE_OWNERSHIP.md) (#139) and the scheduled-mask engine of
[CUE_DEFER.md](CUE_DEFER.md) (#184) / [CUE_WRITE_CLOCK.md](CUE_WRITE_CLOCK.md)
(#185). "Cue", "code", "block", "partner", "dose", "frozen", "idle", "guard",
"task order" and "forgetting" name parts of an update rule and an artificial
benchmark on a Boolean graph with 4-bit weights. Nothing here is a claim about
synapses, biology, insect memory or BANC physiology.

Module: `src/cue_write_dose.rs` (engine support:
`consolidation_guard::run_setting_scheduled_at` with constant masks). CLI:
`bench-cue-write-dose` (`--census` for the label-free pilot,
`--diagnose-cycle26` for the diagnosis below).

## 1. Diagnosis first: why cycles 25 and 26 failed G2

Cycle 26 (#185) failed G2 on clause (e) only: at k = 16, flip 100, 7 of 160
held-out Zgl2m rows had a mean write-clock split ≥ n/2 = 128 (max 193.375).
Cycle 25 (#184) failed G2 on clause (a): the late *episode*-half window was
almost empty (≈ 1 partner-line move per run at k = 16, flip 100).

`bench-cue-write-dose --diagnose-cycle26` re-runs **only uncapped
error-gated** (the run whose per-stage write curve sets every capped arm's
`paced100` budget) on the cycle 26 seeds 641..680, the cycle 26 order salt
and k ∈ {16, 24}, flips {50, 100}. No accuracy is computed. File:
`results/cue-write-dose-diagnosis-cycle26-v1.txt` (sha256
`75a3580c1d1ad0d01593dd5747ff5a701ecc6539910a1c1a58a929b3d05f13eb`). It
reproduces the 7 failing rows of #185 exactly (seeds/orders 651/2, 651/4,
652/2, 667/3, 670/4, 678/3, 680/2; split means 129.938 … 193.375, the values
in the #185 CSV).

| k, flip, orders | stages | idle stages (0 uncapped writes) | idle \| previous stage has the same valence vector | idle \| otherwise | rows with split mean ≥ n/2: all stages / active stages only | max split mean: all / active |
|---|---|---|---|---|---|---|
| 16, 50, held-out | 2560 | 0 | – (0 such stages) | 0 / 2560 | 0 / 0 | 8.1 / 8.1 |
| 16, 100, identity | 640 | 0 | – (0 such stages) | 0 / 640 | 0 / 0 | 7.3 / 7.3 |
| 16, 100, reversed | 640 | 0 | – (0 such stages) | 0 / 640 | 0 / 0 | 7.4 / 7.4 |
| 16, 100, random | 1920 | **490** | **490 / 824** | **0 / 1096** | **7** / 0 | **193.4** / 9.9 |
| 24, 50, held-out | 3840 | 0 | – (0 such stages) | 0 / 3840 | 0 / 0 | 9.8 / 9.8 |
| 24, 100, random | 2880 | **549** | **549 / 1375** | **0 / 1505** | 0 / 0 | 123.3 / 11.2 |

Root cause (descriptive, synthetic benchmark):

1. **Idle stages are a property of the task chain, not of the arms.** At
   flip 100 every edge label flips between consecutive tasks of the
   *chain*, so even and odd tasks carry exactly two valence vectors (V and
   ¬V). The identity and reversed orders alternate them, so no stage
   repeats its predecessor's valence. A seeded random order often puts two
   same-valence tasks back to back; **every** idle stage in the cycle 26
   grid (1 039 of 1 039) follows a stage with the identical valence vector,
   and no other stage is ever idle (consistent with the network already
   answering the new task's training patterns after a same-valence
   predecessor, so the error gate never fires). On an idle stage uncapped
   error-gated writes nothing and the `paced100` budget of **every** capped
   arm in that stage is 0. At flip 50 / 25 valence vectors
   never repeat and no stage is idle.
2. **`budget_split` returns `n` on an idle stage** (pre-registered in #185),
   so a row's mean split is pulled up by 256 / k per idle stage; with 8 or
   more idle stages out of 16 the mean crosses n/2. Over active stages only
   the clock is ≤ 9.9 episodes everywhere, i.e. clause (e) failed on a
   degenerate statistic, not because the late window was late.
3. **The write curve is extremely front-loaded** (half of a stage's
   uncapped writes by episode ≈ 6–11 of 256). This is the shared root with
   cycle 25: under the episode clock (split 128) the late window came after
   almost every write. Any within-stage *timing* manipulation is therefore
   (i) confined to the first handful of episodes and (ii) undefined on idle
   stages. Re-cutting the clock again (active-only means, another
   quantile) would be a post-hoc repair of a design whose own readings in
   #184 and #185 said the same thing: what ranks the arms is **how many**
   partner-block writes they make, not **when** (late − early ΔACC −0.002,
   CI includes 0, in #185).

Decision (before any cycle 27 measurement): **drop timing**, keep the
research thread (generalisation and forgetting against the supervised
control S1 at equal write budget) and manipulate the **count** of
partner-block writes directly with masks that are constant within a stage,
so that no gate depends on a clock and idle stages are idle for every arm
alike (zero budget everywhere; they are reported per row as
`idle_stages`, never filtered).

## 2. Write doses (pre-registered, no tuning)

All share-2 arms use the **same** cue lists as the #136 / #139 / #184 /
#185 matched share-2 code `C_j = sorted(B_j ∪ B_σ(j))` (same lines per task,
same partner map, same line multiplicity, same cue drive at evaluation).
Let `F_j = B_σ(j) \ B_j` (the #139 partner lines; `|F_j|` = 8 at k = 8 and 4
at k = 16 in the main design). For each task j, `π_j` is a seeded
Fisher–Yates permutation of `F_j` (LCG keyed by `seed ^ 0x0DE5_B888_4C27_D05E`
and the task index; independent of flip, order and arm).

* **dose d/4** (Zgd*d*, d = 0..4): the first `⌊d · |F_j| / 4⌋` lines of
  `π_j` are writable for the **whole** stage of task j, the rest of `F_j`
  is frozen for the whole stage. Doses are nested (the writable lines of
  dose d are contained in those of dose d + 1). **Zgd0 is the #139 owned
  write** (bit-identical to `run_setting_masked` with `F_j` frozen, tested);
  **Zgd4 is the #136 shared write** (bit-identical to `run_setting`,
  tested). At k = 16 doses write 0, 1, 2, 3, 4 of 4 partner lines; at
  k = 8, 0, 2, 4, 6, 8 of 8.
* **complement** (Zgd2c, line-identity control): `F_j` minus the dose-2/4
  lines writable for the whole stage (same count as Zgd2, disjoint lines).

"Frozen" is exactly the #139 mask (`consolidation_guard::step_masked`): the
lines stay active and drive the targets, their edges are not written, not
offered to the budget and not counted as a free edge by `last-resort`.

## 3. Arms (10 rows per (k, flip, seed, order))

* `supervised@paced100` (**S1**, control).
* `reward@paced100` (**R1**).
* `error-gated@free` (**E**, = its `paced100` run, gated in G1).
* `E/E-cue0+last-resort@paced100` private blocks (**Zg1**).
* `E/E-cue0+last-resort@paced100` matched share-2 code at doses 0/4, 1/4,
  2/4, 3/4, 4/4 (**Zgd0 … Zgd4**) and the complement of 2/4 (**Zgd2c**).

**Equal budget:** every capped arm uses the `paced100` per-episode cumulative
write curve of uncapped E in the same (k, flip, seed, order); cue writes are
charged. **Equal resources:** all cued arms of a (k, seed) use the same
reserved lines and cue-line edges; the six share-2 arms also have the same
lines per task and line multiplicity.

## 4. Design (pre-registered)

* Graph, task, pool, splits, budget: exactly #131–#185 (128 inputs / 32
  targets / fan-in 8, threshold 1, init weight 1, 4-bit weights, 6 active
  inputs, 32 training patterns per task, 8 rounds, one shared TEST set of 128
  patterns, pair classes seen / CN / ICN / UE, no rehearsal, original edges
  only, topology checked per row).
* **k = 8, 16** (own fixtures; k = 24 is dropped because `|F_j|` = 2 there
  and the quarter doses would collapse onto each other). **Flips 25, 50,
  100.** Flips 100 and 50 carry the primary hypotheses; flip 25 is
  secondary only.
* **Seeds 681..720** (40 seeds, never used by an earlier B888-4 cycle).
* **Orders:** 0 = identity (secondary only); held-out 1 = reversed, 2..4 =
  seeded Fisher–Yates permutations (#133 generator) with a **new salt**
  `0x0DE5_B888_4C27_0A1D`, shared by every flip and arm.
* **Metrics** as #131–#185 (stage coordinates): ACC, LEARN, FGT, LAST on all /
  CN pairs; per seed the mean over its 4 held-out orders (n = 40 paired
  seeds).
* **Dose slope:** per seed, the least-squares slope of a metric on the
  nominal dose fraction q ∈ {0, ¼, ½, ¾, 1} over Zgd0..Zgd4 (units: change
  from owned to shared). **Concavity:** per seed, f(½) − (f(0) + f(1)) / 2.
* **Instrumentation per row:** code, dose, partner map, mean partner
  agreement, conflicting links, lines per task, partner lines and writable
  partner lines per task, cue reach, cue-line edges, live / full cue edges,
  cap, charged / effective / cue writes, cue-write share, uncapped E writes
  per stage and **idle stages**, frozen skips, frozen moves (must be 0),
  **partner-line moves** (edges of `F_j` whose weight changed during task j's
  stage, summed over stages), refused, overwrites, deferred, net reversals,
  consolidated / saturated edges, init / paced / cap / code flags, weight
  digest, topology.
* **Reproducibility:** rows, CSV and report do not depend on the worker thread
  count (tested); the main run is repeated with a different thread count and
  must give a byte-identical CSV (sha256 reported).

Statistics as #123–#185: bootstrap 95 % percentile CI (2000 resamples, seed
0xB0075EED); better/ties/worse over 40 seeds.

* **positive:** mean > 0 and CI lower bound > 0.
* **sequence rule:** Δ ACC strong (mean ≥ 0.05, > 0 in ≥ 32 of 40 seeds and
  CI lower bound > 0), Δ LAST ≥ −0.05 and Δ LEARN ≥ −0.05.
* **equivalent (H5 only):** the 95 % CI of Δ ACC lies strictly inside
  ±0.01.

## 5. Design gates (reported first)

* **G1 (integrity):** every row keeps its topology, starts from its documented
  initial weights, makes charged writes ≤ its allowance;
  `error-gated@paced100` is bit-identical to uncapped E with zero refusals;
  guard counters match guard and ledger; orders valid; codes valid (Zg1 = the
  blocks; share-2 codes as #136; masks = the pre-registered dose schedule;
  writable lines per task = `⌊d · |F_j| / 4⌋`, complement count = dose-2/4
  count); **frozen moves = 0 in every row**; frozen skips = 0 in every
  uncued, private and Zgd4 row; partner-line moves = 0 in every uncued,
  private and Zgd0 row; every arm of a (k, flip, seed, order) carries the
  same uncapped E stage totals and every capped arm's per-stage cap equals
  them.
* **G2 (manipulation):** at k = 8 and k = 16, at flips 50 and 100
  (held-out rows): (a) the mean partner-line moves **strictly increase**
  along Zgd0 < Zgd1 < Zgd2 < Zgd3 < Zgd4; (b) Zgd1 has partner-line moves > 0
  in ≥ 50 % of rows (the smallest dose binds); (c) Zgd1, Zgd2, Zgd3 and Zgd2c
  each have frozen skips > 0 in ≥ 50 % of rows (the freeze binds); (d) the
  mean partner-line moves of Zgd2c are within a factor 2 of Zgd2's (the
  line-identity control receives a comparable dose). No clause uses a
  within-stage clock or a per-stage statistic, so idle stages cannot make a
  clause degenerate.
* **G3 (novel pairs):** median CN pairs per (seed, task) at k = 16, flip 50
  ≥ 20.

If any gate fails, H1–H12 are declared **not evaluable**.

## 6. Hypotheses (held-out orders, seeds 681..720, k = 16 unless stated)

* **H1 (primary, count ranks accuracy):** flip 100, the dose slope of
  ACC_all is positive.
* **H2 (primary, learning side):** flip 100, the dose slope of LEARN_all is
  negative (tested as −slope positive).
* **H3 (primary, forgetting side):** flip 100, the dose slope of FGT_all is
  negative (tested as −slope positive).
* **H4 (diminishing returns):** flip 100, the concavity of ACC_all is
  positive (half the partner lines buy more than half the owned → shared
  gain).
* **H5 (count, not line identity):** flip 100, ACC_all(Zgd2) −
  ACC_all(Zgd2c) is equivalent to 0 (95 % CI inside ±0.01).
* **H6 (vs control):** flip 100, Zgd4 vs S1 passes the sequence rule.
* **H7 (vs control, half dose):** flip 100, Zgd2 vs S1 passes the sequence
  rule.
* **H8:** flip 50, the dose slope of ACC_all is positive.
* **H9 (vs control):** flip 50, Zgd2 vs S1 passes the sequence rule.
* **H10 (other line count):** k = 8, flip 100, the dose slope of ACC_all is
  positive.
* **H11 (generalisation to novel pairs):** flip 100, the dose slope of
  ACC_cn is positive.
* **H12 (novel pairs vs control):** flip 100, ACC_cn(Zgd2) − ACC_cn(S1) is
  positive.

No multiplicity correction (as #123–#185); every hypothesis is reported with
its mean, CI and better/ties/worse counts, pass or fail.

**Secondary (descriptive, no pass/fail):** flip 25; identity order; k = 8
contrasts other than H10; CN slices of every comparison; dose slopes of LAST,
partner-line moves and cue writes; concavity of every metric; pairwise dose
contrasts (Zgd1..Zgd4 vs Zgd0, Zgd4 vs Zgd3 / Zgd2); worst held-out order;
idle stages per row (CSV), skips, deferred, overwrites, live cue edges.

## 7. Pilot (label-free, before the main run)

`bench-cue-write-dose --census` on seeds 1..5, identity order, k ∈ {8, 16},
all flips: partner agreement, writes, guard counters, frozen skips,
partner-line moves, writable lines, idle stages and cue reach. No accuracy is
computed and no held-out order is drawn. The pilot can only lead to stopping
the cycle before the main run (e.g. if a manipulation cannot bind); it cannot
change a threshold, an arm or a hypothesis above.

## 8. Pilot result (label-free, seeds 1..5, identity order; written before the main run)

File: `results/cue-write-dose-census-v1.txt` (no accuracy computed, no
held-out order drawn; sha256
`f646063f65d1bb7f4e349969db8f6ed90607de44c0800047f532d08fe13ebff0`). Means over the 5 pilot seeds:

| k, flip | writable lines Zgd0..Zgd4 / Zgd2c | partner-line moves Zgd0 / 1 / 2 / 3 / 4 / 2c | frozen skips Zgd0 / 1 / 2 / 3 / 4 / 2c | cue writes Zg1 / Zgd0 / Zgd2 / Zgd4 |
|---|---|---|---|---|
| 16, 100 | 0, 1, 2, 3, 4 / 2 | 0 / 17.2 / 32.8 / 47.2 / 61.2 / 32.4 | 809 / 600 / 419 / 207 / 0 / 410 | 305.8 / 309.8 / 410.0 / 491.8 |
| 16, 50 | 0, 1, 2, 3, 4 / 2 | 0 / 12.4 / 25.0 / 38.0 / 50.2 / 25.6 | 1121 / 746 / 551 / 285 / 0 / 611 | 118.6 / 122.6 / 170.0 / 207.8 |
| 8, 100 | 0, 2, 4, 6, 8 / 4 | 0 / 17.8 / 32.0 / 46.2 / 56.4 / 30.8 | 1063 / 871 / 615 / 271 / 0 / 545 | 241.6 / 247.4 / 299.2 / 340.6 |
| 8, 50 | 0, 2, 4, 6, 8 / 4 | 0 / 11.2 / 23.2 / 33.2 / 41.2 / 22.4 | 1041 / 825 / 587 / 336 / 0 / 526 | 102.6 / 101.2 / 126.4 / 150.2 |

* Partner-line moves rise roughly linearly with the number of writable
  lines at every (k, flip); Zgd1 moves a partner line and Zgd1..Zgd3 / Zgd2c
  skip frozen edge steps in 5/5 runs; Zgd2c receives the same dose as Zgd2
  (ratio ≈ 0.96–1.02). Frozen moves are 0 in every row; the identity order
  has no idle stage (as expected from the diagnosis).
* Zgd0 and Zgd4 reproduce the cycle 26 owned / shared cue-write counts at
  k = 16, flip 100 (309.8 / 491.8, identical pilot seeds), as the
  bit-identity tests require.
* **Expectation stated before the main run:** every G2 clause binds on the
  pilot, so G2 is expected to pass and H1–H12 should be evaluable. The pilot
  does not measure accuracy and does not change a threshold, arm or
  hypothesis. The main run is executed exactly as pre-registered above.

## 9. Main run result (seeds 681..720, held-out orders; synthetic graph)

Files: `results/cue-write-dose-report-v1.txt`, `results/cue-write-dose-v1.csv`
(12 000 rows; sha256
`ea9f6b3f0daa29b48a2793410141423e02c3d9410e6f326e5b977b541b6c81b9`, report
sha256 `d0f959f33805cab19efac3ecde48093026c55ffcba758f6c383de824255cbe97`).
The run was executed exactly as pre-registered above; no threshold, arm,
seed or hypothesis was changed after the pilot. Reproducibility: the CSV and the report are byte-identical with `--threads 8` and `--threads 6` (same sha256 in both).

### Design gates

| gate | result | detail |
|---|---|---|
| G1 integrity | **pass** | topology unchanged in every row, charged writes ≤ allowance, `error-gated@paced100` = uncapped E, frozen moves = 0, guard/ledger counters, orders, codes, dose masks, writable-line counts and per-stage budgets consistent |
| G2 manipulation | **pass** | at k 8 / 16, flips 50 / 100 (held-out): (a) mean partner-line moves strictly increase along the doses, e.g. k 16 flip 100: 0 / 9.6 / 19.2 / 28.9 / 37.7 (Zgd0..Zgd4); (b) Zgd1 moves a partner line in 99.4–100 % of rows; (c) Zgd1/2/3/2c skip frozen edge steps in 100 % of rows; (d) Zgd2c / Zgd2 partner moves ≈ 1.0 (19.7 vs 19.2 at k 16 flip 100) |
| G3 novel pairs | **pass** | median CN pairs at k 16, flip 50 = 64 ≥ 20 |

Idle stages (descriptive, as diagnosed in §1): at flip 100 the held-out
rows have on average 3.21 of 16 (k 16) and 1.26 of 8 (k 8) zero-budget
stages (72.5 % / 63.1 % of rows have at least one); none at flips 25 / 50.
They are identical across the arms of an order (G1) and no gate or
hypothesis depends on them.

### Hypotheses (k = 16 unless stated; mean ± sd [95 % bootstrap CI] better/ties/worse over 40 seeds)

Slopes are per-seed least-squares slopes over the dose fraction 0 → 1
(owned → shared write).

| H | contrast | value | verdict |
|---|---|---|---|
| H1 | flip 100, dose slope of ACC_all | +0.026 ± 0.009 [+0.024, +0.029] 40/0/0 | **pass** |
| H2 | flip 100, −(dose slope of LEARN_all) | +0.048 ± 0.012 [+0.044, +0.052] 40/0/0 | **pass** |
| H3 | flip 100, −(dose slope of FGT_all) | +0.079 ± 0.014 [+0.075, +0.083] 40/0/0 | **pass** |
| H4 | flip 100, concavity of ACC_all | +0.000 ± 0.006 [−0.001, +0.002] 20/0/20 | **fail** |
| H5 | flip 100, ACC Zgd2 − Zgd2c equivalent to 0 (±0.01) | +0.000 ± 0.009 [−0.002, +0.003] 17/0/23 | **pass** |
| H6 | flip 100, Zgd4 vs S1 sequence rule | ΔACC +0.096 [+0.090, +0.103] 40/0/0, ΔLEARN +0.196, ΔLAST +0.212 | **pass** |
| H7 | flip 100, Zgd2 vs S1 sequence rule | ΔACC +0.084 [+0.077, +0.090] 40/0/0, ΔLEARN +0.219, ΔLAST +0.229 | **pass** |
| H8 | flip 50, dose slope of ACC_all | +0.012 ± 0.008 [+0.010, +0.014] 36/0/4 | **pass** |
| H9 | flip 50, Zgd2 vs S1 sequence rule | ΔACC +0.051 [+0.047, +0.055] 40/0/0, ΔLEARN +0.267, ΔLAST +0.285 | **pass** (ΔACC just above the 0.05 bar) |
| H10 | k 8, flip 100, dose slope of ACC_all | +0.019 ± 0.017 [+0.014, +0.024] 34/0/6 | **pass** |
| H11 | flip 100, dose slope of ACC_cn | +0.008 ± 0.025 [+0.000, +0.015] 24/0/16 | **pass** (marginal: CI lower bound rounds to 0.000) |
| H12 | flip 100, ACC_cn Zgd2 − S1 | +0.117 ± 0.037 [+0.106, +0.129] 40/0/0 | **pass** |

### Readings (descriptive, synthetic benchmark only)

* **Count, not timing, and not line identity.** At equal `paced100`
  budget, held-out ACC_all at k = 16, flip 100 rises step by step with
  the number of writable partner lines: 0.577 (Zgd0 = owned), 0.584,
  0.590, 0.597, 0.603 (Zgd4 = shared); every pairwise step vs Zgd0 is
  positive (+0.007 / +0.013 / +0.020 / +0.026). The relation is linear
  (H4 fails: concavity +0.000), and which half of the partner lines is
  writable does not matter (H5: Zgd2 − Zgd2c +0.000, CI inside ±0.003).
  This is the direct test the two timing cycles could not deliver: the
  #139 shared-vs-owned trade-off scales with how many partner-block lines
  a task may write.
* **The trade-off is the same writes.** Each extra writable line costs
  learning (LEARN slope −0.048) and buys a larger forgetting cut (FGT
  slope −0.079); the net accuracy gain is their difference. Shared Zgd4
  learns 0.048 less and forgets 0.079 less than owned Zgd0.
* **Generalisation to novel pairs barely follows the dose** (ACC_cn slope
  +0.008, marginal pass at k 16; −0.007 with a CI including 0 at k 8,
  secondary): the dose mostly moves seen-pair retention.
* **Against the supervised control at equal budget** every cued dose arm
  wins on ACC, LEARN and LAST but forgets more (Zgd2 − S1 at flip 100:
  ΔACC +0.084, ΔFGT −0.145, i.e. S1 forgets less; ACC_cn +0.117). The
  margin shrinks at flip 50 (Zgd2 +0.051, just above the 0.05 bar).
* **Worst held-out order (secondary, descriptive).** The per-seed minimum
  over the 4 held-out orders of the paired difference Zgd2 − Zgd2c is
  negative at every (k, flip) (k 16, flip 100: −0.014, 4/0/36). A minimum
  of four noisy paired differences is biased below zero by construction, so
  this does not contradict H5 (which is on the mean over orders), but line
  identity is only equivalent on average, not order by order.
* The effect is small in absolute terms (≈ +0.026 ACC from owned to
  shared at k 16, flip 100; +0.019 at k 8; +0.012 at flip 50) and every
  number is on a generated graph with artificial labels. No claim about
  synapses, biology, insect memory or BANC physiology is made.

## Reproduce

```bash
cargo +1.89 run --release -- bench-cue-write-dose --diagnose-cycle26 > results/cue-write-dose-diagnosis-cycle26-v1.txt
cargo +1.89 run --release -- bench-cue-write-dose --census > results/cue-write-dose-census-v1.txt
cargo +1.89 run --release -- bench-cue-write-dose --csv results/cue-write-dose-v1.csv > results/cue-write-dose-report-v1.txt
```

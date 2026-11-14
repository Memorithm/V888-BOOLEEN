# B888-4 cycle 14: equal weight-write budget across strategies, flip sweep (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123), with the fixed pattern pool of
[CUE_REACH.md](CUE_REACH.md) (#124). Nothing here is a claim about synapses,
biology, insect memory or BANC physiology.

Module: `src/write_budget.rs`. CLI: `bench-write-budget`.

## Motivation (what #124 left open)

1. **Budget in episodes, not writes.** In #124 every arm saw 256 episodes per
   task, but the arms did not write the same number of weights: over four
   tasks at flip 50, `E/E-cue@16` changed 445 weights, `error-gated` 209 and
   `supervised` 3594. The +0.061 all-pairs cue gain could be partly bought by
   the extra writes.
2. **Flip dependence only probed as a secondary reading.** The CN gain of
   the cue was −0.009 / +0.018 / +0.110 at flips 25 / 50 / 100. No hypothesis
   tested it, and flip 75 was not run.

## Design (pre-registered)

* **Graph, task, pool, cues:** exactly the #124 fixed-pool design
  (`cue_reach::DESIGN`): generated graph per seed, 128 inputs / 32 targets /
  fan-in 8, threshold 1, init weight 1, 4-bit weights; four tasks T1→T4,
  patterns of 6 active inputs, 32 training patterns per task, one shared TEST
  set of 128 patterns; pair classes seen / CN / ICN / UE as in #123; four
  disjoint greedy cues of up to 16 lines, all removed from the pattern pool.
  The cued arm uses the full cue (level 16, full reach).
* **Strategies** (same episode order, 8 rounds × 32 patterns = 256 episodes
  per task, no rehearsal, original edges only, topology checked per row):
  * `supervised` (S): teacher step on every edge of every active line;
  * `error-gated` (E): teacher step only into reached targets it gets wrong;
  * `E/E-cue` (EE): error gate on pattern and cue edges, task cue on;
  * `reward` (R): reward-modulated with a clean per-target reward (the
    `PerTarget` reference of `reward_limits`): for each reached target that
    fires, the edges from the active lines into it are reinforced if the
    firing was right and weakened if it was wrong. Uncued.
  With no limit, S / E / EE are bit-identical to `cue_gating::step` and R to
  per-target `RewardModulated` updates (tested).
* **Write:** a 4-bit step that actually changes a stored weight. Steps at
  the 0/15 plateaus change nothing, cost nothing and are never refused.
* **Budgets** (per (flip, seed, task), as a fraction of the writes of
  uncapped `error-gated` on the same task, floor):
  * `free`: no limit (= the #124 equal-episode protocol, new seeds);
  * **`paced100` (primary):** after every episode e an arm may have made at
    most as many writes as uncapped `error-gated` had made by episode e. All
    arms get `error-gated`'s write schedule. `error-gated` itself is
    unchanged (bit-identical, no refusal; tested and gated);
  * `total100` (secondary): one cap on the task total, spent in episode
    order (front-loaded);
  * `paced50` (secondary): half of `error-gated`'s schedule (binds
    `error-gated` too).
  A refused step is counted (`refused`, `first_refusal`); the arm keeps
  seeing the remaining episodes. Edges are visited source-major inside an
  episode, so a partially funded episode updates the lowest-index active
  lines first.
* **Rows:** per (flip, seed) 4 strategies × 4 budgets = 16 rows.
* **Seeds:** 161..200 (40 seeds, never used by an earlier B888-4 cycle).
  **Flips:** 25, **50 (primary)**, 75, 100 (high conflict).
* **Gains:** x − `error-gated` (uncapped), paired by seed.

Metrics (ACC, LEARN, FGT, LAST at k = 2..4 on all / seen / CN / ICN / UE
pairs), the balanced accuracy, the bootstrap (95 % percentile, 2000
resamples, seed 0xB0075EED), **strong** (mean ≥ 0.05, > 0 in ≥ 32 of 40
seeds, CI lower bound > 0) and the **sequence rule** (Δ ACC strong,
Δ LAST ≥ −0.05, Δ LEARN ≥ −0.05) are as in #123/#124.

### Pilot write census (seeds 1..5, flip 50; writes only, no accuracy computed)

Seeds 1..5 are not used by any hypothesis. Total writes over 4 tasks:

| strategy | free | paced100 | total100 | first refusal (episode of 256) |
|---|---|---|---|---|
| supervised | 3176–3816 | 192–208 (= E) | 192–208 | paced: 1; total: 5–8 |
| error-gated | 192–208 | 192–208, never refused | same | — |
| E/E-cue | 429–462 | 192–208 | 192–208 | paced: 1–8; total: 7–15 |
| reward | 2161–2902 | 192–208 | 192–208 | paced: 1–6; total: 5–9 |

The paced budget binds every non-E arm from the first episodes; the total
budget is exhausted after 5–15 episodes of 256 (front-loaded). The grid
(2560 rows) takes about 8 min on CPU.

## Design gates (reported first)

* **G1 (budget integrity):** every row keeps its topology; every task's
  writes ≤ its allowance; `error-gated` at `paced100` and `total100` is
  bit-identical to uncapped `error-gated` with zero refusals.
* **G2 (budget binds):** at flip 50, `E/E-cue@paced100` refuses at least one
  write in ≥ 50 % of (seed, task).
* **G3 (novel pairs):** median CN pairs per (seed, task) at flip 50 ≥ 20.

If any gate fails, H1–H7 are declared **not evaluable**.

## Hypotheses (k = 4, seeds 161..200; `EE` = `E/E-cue@paced100`, base = uncapped `error-gated`)

* **H1 (cue gain survives equal writes, all pairs, flip 50):** `EE` vs
  `error-gated` passes the sequence rule on all pairs.
* **H2 (most of the gain is not bought by writes, flip 50):** mean
  Δ ACC_all(`EE` − E) ≥ 0.5 × mean Δ ACC_all(`E/E-cue@free` − E), **and**
  its CI lower bound > 0.
* **H3 (CN gain rises with conflict at equal writes):** the mean CN gain of
  `EE` over E never drops by more than 0.005 from flip 25 → 50 → 75 → 100,
  **and** the per-seed difference gain@100 − gain@25 has CI lower bound > 0.
* **H4 (novel pairs at high conflict, flip 100):** `EE` vs E passes the
  sequence rule on CN pairs.
* **H5 (novel-pair forgetting at high conflict, flip 100):**
  FGT_CN(E) − FGT_CN(`EE`) is strong.
* **H6 (error gate vs supervised at equal writes, flip 50):** E vs
  `supervised@paced100` passes the sequence rule on CN pairs.
* **H7 (reward-modulated at equal writes, flip 50, non-inferiority):**
  Δ ACC_all(`reward@paced100` − E) has CI lower bound > −0.05.

**Secondary (reported, no pass/fail):** all strategies at every budget and
flip; `free` (replication of #124 on new seeds), `total100` and `paced50`;
k = 2 and 3; seen, ICN and UE pairs; consecutive flip contrasts; writes,
refusals, first-refusal episode and saturation.

No multiplicity correction is applied. The analysis is run once. Nothing
above the Results heading is changed after the run except the "pending"
placeholder.

## Command

```bash
cargo +1.89 run --release -- bench-write-budget --csv results/write-budget-v1.csv > results/write-budget-report-v1.txt
```

## Results

Run after the pre-registration commit b3fe18f had been pushed. The grid
(2560 rows: 4 flips × 40 seeds × 4 strategies × 4 budgets) takes about
7.5 min on CPU. Two runs give a byte-identical CSV (sha256
`5c864906…0e3cb417`) and an identical report. Every row keeps the original
topology (`topology_unchanged_all=true`). Files:
`results/write-budget-v1.csv`, `results/write-budget-report-v1.txt`.

### 1. Design gates (reported first)

* **G1 passes:** writes never exceed the allowance; `error-gated` at
  `paced100` and `total100` is bit-identical to uncapped `error-gated` with
  zero refusals on all 320 (flip, seed) rows each; topology unchanged
  everywhere.
* **G2 passes:** `E/E-cue@paced100` refuses writes in 100 % of (seed, task)
  at flip 50; the first refusal comes at episode 1.2 on average (of 256).
* **G3 passes:** median CN pairs per (seed, task) 119 ≥ 20.

### 2. Settings at flip 50 (k = 4, mean over 40 seeds)

| strategy @ budget | ACC_all | ACC_CN | LEARN_all | LEARN_CN | FGT_all | FGT_CN | LAST_all | writes (4 tasks) |
|---|---|---|---|---|---|---|---|---|
| supervised @ free | 0.610 | 0.581 | 0.883 | 0.764 | 0.364 | 0.245 | 0.873 | 3625 |
| error-gated @ free (= paced100 = total100) | 0.627 | 0.626 | 0.972 | 0.920 | 0.460 | 0.391 | 0.977 | 209 |
| E/E-cue @ free | 0.688 | 0.636 | 0.971 | 0.906 | 0.378 | 0.359 | 0.976 | 452 |
| reward @ free | 0.610 | 0.595 | 0.755 | 0.689 | 0.194 | 0.126 | 0.658 | 2554 |
| supervised @ paced100 | 0.573 | 0.541 | 0.650 | 0.574 | 0.102 | 0.044 | 0.631 | 209 |
| **E/E-cue @ paced100** | 0.639 | 0.599 | 0.691 | 0.634 | 0.070 | 0.047 | 0.692 | 209 |
| reward @ paced100 | 0.598 | 0.570 | 0.630 | 0.574 | 0.043 | 0.006 | 0.600 | 209 |
| E/E-cue @ total100 | 0.676 | 0.613 | 0.737 | 0.639 | 0.082 | 0.034 | 0.732 | 209 |
| error-gated @ paced50 | 0.593 | 0.557 | 0.738 | 0.646 | 0.193 | 0.119 | 0.744 | 104 |

With the same writes and schedule as `error-gated`, every other arm learns
far less of the current task (LEARN_all 0.63–0.69 vs 0.97).

### 3. Gain over `error-gated` across the flip sweep (k = 4; mean [95 % CI], seeds better/tied/worse)

| flip | E/E-cue@free, all | E/E-cue@paced100, all | E/E-cue@free, CN | E/E-cue@paced100, CN | Δ LEARN_all, paced100 | Δ FGT_CN, paced100 (+ = forgets less) |
|---|---|---|---|---|---|---|
| 25 | +0.045 [+0.040, +0.050] 40/0/0 | −0.108 [−0.117, −0.098] 0/0/40 | −0.007 [−0.018, +0.004] | −0.116 [−0.140, −0.093] 5/0/35 | −0.340 | +0.294 |
| 50 | +0.060 [+0.055, +0.065] 40/0/0 | **+0.011 [+0.003, +0.020] 25/0/15** | +0.010 [−0.003, +0.022] | −0.027 [−0.047, −0.006] 15/0/25 | −0.282 | +0.344 |
| 75 | +0.070 [+0.064, +0.077] 40/0/0 | +0.051 [+0.041, +0.060] 38/0/2 | +0.037 [+0.023, +0.052] | +0.013 [−0.007, +0.034] 25/0/15 | −0.237 | +0.329 |
| 100 | +0.115 [+0.105, +0.125] 40/0/0 | +0.127 [+0.117, +0.138] 40/0/0 | +0.106 [+0.091, +0.123] 40/0/0 | **+0.078 [+0.055, +0.101] 35/0/5** | −0.195 | **+0.334 [+0.308, +0.361] 40/0/0** |

The `free` column replicates #124 on new seeds (flip 50 all +0.060 vs
+0.061; flip 100 CN +0.106 vs +0.110).

### 4. Hypotheses

| | statistic | value | result |
|---|---|---|---|
| G1 | budget integrity, E unchanged at 100 % | all rows | **pass** |
| G2 | EE@paced100 refuses in ≥ 50 % of (seed, task), flip 50 | 100 % | **pass** |
| G3 | median CN pairs ≥ 20 | 119 | **pass** |
| H1 | EE@paced100 vs E, sequence rule, all pairs, flip 50 | +0.011 [+0.003, +0.020] 25/0/15; Δ LEARN −0.282, Δ LAST −0.285 | **fail** |
| H2 | gain@paced100 ≥ 0.5 × gain@free and CI lo > 0, flip 50 | +0.011 vs 0.5 × +0.060 = +0.030 (CI lo +0.003) | **fail** |
| H3 | CN gain monotone over flips 25→100 (tol 0.005) and gain@100 − gain@25 CI lo > 0 | −0.116 → −0.027 → +0.013 → +0.078; +0.193 [+0.164, +0.224] 40/0/0 | **pass** |
| H4 | EE@paced100 vs E, sequence rule, CN pairs, flip 100 | ACC +0.078 [+0.055, +0.101] 35/0/5 (strong), but Δ LEARN_CN −0.173, Δ LAST_CN −0.090 | **fail** |
| H5 | FGT_CN(E) − FGT_CN(EE@paced100) strong, flip 100 | +0.334 [+0.308, +0.361] 40/0/0 | **pass** (see reading: low learning) |
| H6 | E vs supervised@paced100, sequence rule, CN pairs, flip 50 | +0.086 [+0.065, +0.107] 34/0/6; Δ LEARN_CN +0.346 | **pass** |
| H7 | reward@paced100 − E, ACC_all CI lo > −0.05, flip 50 | −0.030 [−0.038, −0.022] 2/0/38; Δ LEARN −0.342 | **pass** (non-inferiority margin only; see reading) |

### 5. Secondary (no pass/fail)

* **Front-loaded total budget:** `E/E-cue@total100` keeps more of the
  flip-50 gain (+0.048 [+0.042, +0.054] 40/0/0, CN −0.013) but still loses
  0.235 LEARN_all. At flip 100 it gives +0.140 all / +0.069 CN.
* **k = 2 and 3 at flip 50:** `E/E-cue@paced100` is *worse* than E on all
  pairs (−0.108 and −0.024) and CN pairs (−0.136, −0.056). The small k = 4
  gain comes from the last tasks being forgotten less, not from learning.
* **Which pairs at flip 50, paced100:** seen +0.018, UE +0.017, ICN −0.019,
  CN −0.027.
* **Half budget:** `error-gated@paced50` loses 0.034 ACC_all and 0.235
  LEARN_all at flip 50; `E/E-cue@paced50` loses 0.083 ACC_all. At flip 100,
  `E/E-cue@paced50` still gains +0.061 all pairs over uncapped E.
* **Reward at equal writes:** `reward@paced100` − `supervised@paced100` is
  +0.024 all pairs / +0.029 CN at flip 50 (both lose ~0.33 LEARN vs E).
  Uncapped, `reward` uses 2554 writes for ACC_all 0.610, LEARN 0.755.
* **Write efficiency:** at flip 50 uncapped, ACC_all per 100 writes is far
  higher for E (209 writes) than for S (3625) or R (2554) at a higher or
  equal ACC.

### Reading

The +0.06 all-pairs cue gain of #124 at flip 50 is **mostly bought by extra
writes**. Given exactly `error-gated`'s write schedule, `E/E-cue` keeps
+0.011 (H1, H2 fail): its cue edges consume the allowance in the first
episodes, so it learns each task much less (LEARN −0.28) and has less to
forget. At high conflict (flip 100) the cue still helps at equal writes
(+0.127 all, +0.078 CN, 35/40 seeds), and the CN gain rises steadily with
the flip rate (H3). But at every flip this gain is a forgetting-for-learning
trade (Δ LEARN −0.20 to −0.34), so the sequence rule fails on CN pairs (H4),
and the large forgetting cut (H5) mostly reflects that `E/E-cue@paced100`
learned less in the first place.

Among uncued rules, the error gate is by far the most write-efficient: with
the same writes, `supervised` loses 0.054 ACC_all and 0.086 ACC_CN with
LEARN 0.32–0.35 lower (H6). `reward` is within the pre-registered −0.05
non-inferiority margin of E on ACC_all at flip 50 (H7), but it is worse in
38/40 seeds and learns much less; at flip 50 every arm is near chance on
final ACC, which makes this margin easy to meet. H7 is not evidence that
reward modulation matches error gating.

This cycle is a negative result for the cue at equal writes and a positive
one for the error gate: the write budget, not the episode budget, is the
right axis to compare these rules on.

## Limitations

- Synthetic graph and artificial OR-rule task only (128/32/8, active 6),
  threshold 1, init 1. Nothing is measured on the real BANC graph
  (DOI 10.7910/DVN/7WTH1N), which is not built on this machine.
- The budget is matched to `error-gated`, so it is the reference arm's
  natural write count, not a free parameter swept over a wide range.
- A paced or total budget changes *which* writes an arm makes (the first
  edges in source-major order inside a partially funded episode, the first
  episodes under `total`), not only how many. Other allocation rules (random
  subsampling, per-target quotas) are not tested.
- The reward arm is uncued, per-target and clean; it cannot raise a silent
  target that should fire.
- CN pairs share TEST patterns with seen pairs. The cue is assumed known at
  test time. 40 seeds. No multiplicity correction across H1–H7.
- The flip sweep uses the same seeds at every flip, but the fixtures (task
  labels) differ by flip; H3's paired contrast is paired by seed only.
- H7's non-inferiority margin (0.05) turned out loose at flip 50, where all
  arms end near chance on the old tasks; it was fixed before the run and is
  reported as is.

B888-3 files (`src/circuit_explorer.rs`, subcircuit recall, activity atlas
and their CLI wiring) are untouched. No NVIDIA dependency, no secrets. CPU
only.

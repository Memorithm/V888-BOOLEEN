# B888-4 cycle 15: write allocation of the cued arm under an equal write budget (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123), with the fixed pattern pool of
[CUE_REACH.md](CUE_REACH.md) (#124) and the write budgets of
[WRITE_BUDGET.md](WRITE_BUDGET.md) (#126). "Cue", "allocation" and "tag"
name parts of an update rule on a Boolean graph with 4-bit weights. Nothing
here is a claim about synapses, biology, insect memory or BANC physiology.

Module: `src/write_allocation.rs`. CLI: `bench-write-allocation`
(`--census` for the writes-only pilot).

## Motivation (what #126 left open)

At an equal write schedule (`paced100`: after every episode, at most as many
effective writes as uncapped `error-gated` had made by then), `E/E-cue` kept
only +0.011 ACC_all at flip 50 and lost 0.28 LEARN_all. Its cue edges used
the allowance in the first episodes. Inside an episode #126 funded the
offered edge steps **source-major** (ascending line index, cue and pattern
lines mixed), so which writes a capped arm made depended on line numbering.
#126 listed other allocation rules as untested.

This cycle keeps the `E/E-cue` update rule, the budget and the episodes, and
changes only the **funding order** of the offered steps inside an episode:

* `E/E-cue` (source-major, SM): the #126 order (bit-identical to #126, tested);
* `E/E-cue-pattern-first` (PF): every pattern-line edge step first, then the
  cue-line edge steps;
* `E/E-cue-cue-first` (CF): cue-line edge steps first, then pattern-line ones.

The gate decision (wrong prediction for a reached target) is taken on the
drive before the update, and each step touches one edge, so **uncapped, the
three orders are bit-identical** (tested and gated). Under a cap they differ
only in which steps are refused.

### Design-stage pilot that changed the plan (writes only, seeds 1..5)

The first candidate for this cycle was a **task-local negative-tag** rule
(`E-tag`): an uncued error gate with one bit per edge; every active edge into
a target labelled *should not fire* is tagged negative for the current task
(sound under the OR rule; tested), tags are cleared at task boundaries, and a
false negative raises only untagged edges. The writes-only pilot
(`results/write-allocation-census-v1.txt`, flip 50) shows it is **nearly
inactive on this design**: 1–4 raises blocked per seed over four tasks, and
188/186/203/195/200 writes vs 194/192/205/199/208 for `error-gated`. With 6
active lines of 128 and fan-in 8, a reached target sees about
6 × 8 / 128 ≈ 0.4 active pattern edges per pattern, so a false negative
almost never has more than one candidate edge and there is no credit
ambiguity to resolve. `E-tag` is therefore kept as a **secondary** arm only
(reported, no hypothesis). No accuracy was computed in the pilot.

## Design (pre-registered)

* **Graph, task, pool, cues:** exactly the #124/#126 fixed-pool design
  (`write_budget::DESIGN`): generated graph per seed, 128 inputs / 32
  targets / fan-in 8, threshold 1, init weight 1, 4-bit weights; tasks
  T1→T4, patterns of 6 active inputs, 32 training patterns per task, one
  shared TEST set of 128 patterns; pair classes seen / CN / ICN / UE as in
  #123; four disjoint greedy cues of up to 16 lines (full reach), removed
  from the pattern pool. Cued arms use cue j while training and evaluating
  task j.
* **Strategies (7 rows per budget):** `supervised` (S, control),
  `error-gated` (E, reference), `reward` (R, clean per-target
  reward-modulated, uncued; the #126 code path), `E/E-cue` (SM),
  `E/E-cue-pattern-first` (PF), `E/E-cue-cue-first` (CF), `E-tag`
  (secondary). Same episode order for all arms (8 rounds × 32 patterns = 256
  episodes per task), no rehearsal, original edges only, topology checked
  per row.
* **Budgets:** `free` and **`paced100` (primary)**, defined exactly as in
  #126 from uncapped `error-gated`'s per-task cumulative write curve of the
  same (flip, seed). A write = a 4-bit step that changes a stored weight.
* **Seeds:** 201..240 (40 seeds, never used by an earlier B888-4 cycle).
  **Flips:** 25 (secondary), **50 (primary)**, 100 (high conflict).
* **Instrumentation per (row, task):** cap, writes, refused, first refusal,
  attempts, **cue writes**, **cue refused**, blocked raises and tags at end
  (E-tag), weight digest, `same_as_free`, `free_alloc_invariant`, topology.
* **Checkpoint/resume:** circuit weights (`PlasticCircuit::save`) plus a
  text `Checkpoint` (tags, stats, ledger, cost, task, next episode); resuming
  mid-task is bit-exact for every strategy at `paced100` (tested).

Metrics (ACC, LEARN, FGT, LAST at k = 2..4 on all / seen / CN / ICN / UE
pairs), balanced accuracy, bootstrap (95 % percentile, 2000 resamples, seed
0xB0075EED), **strong** (mean ≥ 0.05, > 0 in ≥ 32 of 40 seeds, CI lower
bound > 0) and the **sequence rule** (Δ ACC strong, Δ LAST ≥ −0.05,
Δ LEARN ≥ −0.05) are as in #123/#124/#126. Base of the gains: uncapped
`error-gated` (= its `paced100` run, gated).

### Pilot write census (seeds 1..5, flip 50; writes only, no accuracy)

| arm | budget | writes (4 tasks) | cue writes / writes | refused |
|---|---|---|---|---|
| error-gated | free = paced100 | 192–208 | 0 | 0 |
| E/E-cue (SM) | free | 429–462 | 0.56–0.58 | 0 |
| E/E-cue (SM) | paced100 | 192–208 | 0.59–0.61 | 3451–3660 |
| E/E-cue-pattern-first | paced100 | 192–208 | 0.31–0.37 | 2615–3478 |
| E/E-cue-cue-first | paced100 | 192–208 | 0.70–0.74 | 3372–3970 |
| supervised | paced100 | 192–208 | 0 | 7724–9027 |
| reward | paced100 | 192–208 | 0 | 6339–8135 |

Pattern-first funding moves about a quarter of the allowance from cue edges
to pattern edges; even so about a third of PF's writes still go to cue edges
(steps left over when no pattern step is offered).

## Design gates (reported first)

* **G1 (integrity):** every row keeps its topology; every task's writes ≤
  its allowance; `error-gated@paced100` is bit-identical to uncapped
  `error-gated` with zero refusals; the three `E/E-cue` allocations are
  bit-identical uncapped; no `E-tag` false negative ever had all its active
  edges tagged.
* **G2 (allocation changes the run):** at flip 50, PF@paced100 and
  SM@paced100 end with different weights in ≥ 50 % of seeds.
* **G3 (novel pairs):** median CN pairs per (seed, task) at flip 50 ≥ 20.

If any gate fails, H1–H7 are declared **not evaluable**.

## Hypotheses (k = 4, seeds 201..240; PF = `E/E-cue-pattern-first@paced100`, SM = `E/E-cue@paced100`, E = uncapped `error-gated`, S1 = `supervised@paced100`)

* **H1 (primary: cue gain at equal writes with pattern-first funding, flip
  50):** PF vs E passes the sequence rule on all pairs.
* **H2 (allocation is the lever, flip 50):** LEARN_all(PF) − LEARN_all(SM)
  is strong.
* **H3 (generalisation, flip 50):** PF vs E passes the sequence rule on CN
  pairs.
* **H4 (generalisation at high conflict, flip 100):** PF vs E passes the
  sequence rule on CN pairs (#126 H4 failed for SM on learning).
* **H5 (vs the supervised control at equal writes, flip 50):** PF vs S1
  passes the sequence rule on CN pairs.
* **H6 (forgetting not bought by less learning, flip 100):**
  FGT_CN(E) − FGT_CN(PF) is strong **and** mean Δ LEARN_CN(PF − E) ≥ −0.05.
* **H7 (reward-modulated vs supervised at equal writes, flip 50;
  replication of a #126 secondary reading):** Δ ACC_all(`reward@paced100` −
  S1) has CI lower bound > 0.

**Secondary (reported, no pass/fail):** CF and SM vs E and vs each other;
all arms at `free` (replication of #126 on new seeds); flip 25; k = 2 and 3;
seen, ICN and UE pairs; `E-tag` vs E; writes, cue-write share, refusals,
blocked raises, saturation.

No multiplicity correction is applied. The analysis is run once. Nothing
above the Results heading is changed after the run except the "pending"
placeholder.

## Command

```bash
cargo +1.89 run --release -- bench-write-allocation --csv results/write-allocation-v1.csv > results/write-allocation-report-v1.txt
# pilot (writes only): cargo +1.89 run --release -- bench-write-allocation --census
```

## Results

Run once after the pre-registration commit 8656bc9 had been pushed. The grid
(1680 rows: 3 flips × 40 seeds × 7 strategies × 2 budgets) takes about
6.3 min on CPU. Two runs give a byte-identical CSV (sha256
`de274ab1…4c587842`) and an identical report. Every row keeps the original
topology (`topology_unchanged_all=true`). Files:
`results/write-allocation-v1.csv`, `results/write-allocation-report-v1.txt`,
pilot `results/write-allocation-census-v1.txt`.

### 1. Design gates (reported first)

* **G1 passes:** writes never exceed the allowance; `error-gated@paced100`
  is bit-identical to uncapped `error-gated` with zero refusals; the three
  `E/E-cue` allocations are bit-identical uncapped on all 120 (flip, seed);
  no `E-tag` false negative ever had all its edges tagged.
* **G2 passes:** PF and SM end with different weights in 100 % of seeds at
  every flip.
* **G3 passes:** median CN pairs per (seed, task) 119 ≥ 20.

### 2. Settings at flip 50 (k = 4, mean over 40 seeds)

| strategy @ budget | ACC_all | ACC_CN | LEARN_all | LEARN_CN | FGT_all | FGT_CN | LAST_all | writes (4 tasks) | cue share of writes |
|---|---|---|---|---|---|---|---|---|---|
| supervised @ free | 0.612 | 0.590 | 0.884 | 0.758 | 0.362 | 0.224 | 0.871 | 3521 | 0 |
| error-gated @ free (= paced100) | 0.631 | 0.630 | 0.972 | 0.912 | 0.454 | 0.375 | 0.969 | 205 | 0 |
| E/E-cue @ free (all three orders) | 0.698 | 0.640 | 0.971 | 0.901 | 0.365 | 0.349 | 0.966 | 438 | 0.572 |
| supervised @ paced100 | 0.579 | 0.548 | 0.645 | 0.575 | 0.089 | 0.036 | 0.616 | 205 | 0 |
| reward @ paced100 | 0.606 | 0.575 | 0.631 | 0.573 | 0.032 | −0.002 | 0.598 | 205 | 0 |
| E/E-cue (SM) @ paced100 | 0.646 | 0.609 | 0.699 | 0.638 | 0.070 | 0.039 | 0.705 | 205 | 0.600 |
| **E/E-cue-pattern-first (PF) @ paced100** | 0.631 | 0.613 | 0.694 | 0.684 | 0.084 | 0.095 | 0.755 | 205 | 0.337 |
| E/E-cue-cue-first (CF) @ paced100 | 0.652 | 0.594 | 0.677 | 0.602 | 0.033 | 0.011 | 0.664 | 205 | 0.709 |
| E-tag @ free (= paced100) | 0.631 | 0.630 | 0.972 | 0.912 | 0.454 | 0.375 | 0.969 | 198 | 0 |

Pattern-first moves the cue share of writes from 0.60 to 0.34, but every
cued arm at `paced100` still learns about 0.28 less of each task than
`error-gated` (LEARN_all 0.68–0.70 vs 0.97).

### 3. Hypotheses (k = 4; mean [95 % CI] seeds better/tied/worse)

| | statistic | value | result |
|---|---|---|---|
| G1 | integrity (caps, E unchanged, orders identical uncapped, tags sound) | all rows | **pass** |
| G2 | PF ≠ SM final weights in ≥ 50 % of seeds, flip 50 | 100 % | **pass** |
| G3 | median CN pairs ≥ 20 | 119 | **pass** |
| H1 | PF vs E, sequence rule, all pairs, flip 50 | ACC −0.001 [−0.010, +0.009] 19/0/21; Δ LEARN −0.278, Δ LAST −0.214 | **fail** |
| H2 | LEARN_all(PF) − LEARN_all(SM) strong, flip 50 | −0.005 [−0.013, +0.002] 17/0/23 | **fail** |
| H3 | PF vs E, sequence rule, CN pairs, flip 50 | ACC −0.017 [−0.038, +0.003] 16/0/24; Δ LEARN_CN −0.228 | **fail** |
| H4 | PF vs E, sequence rule, CN pairs, flip 100 | ACC +0.053 [+0.037, +0.070] 31/0/9 (not strong: 31 < 32 seeds); Δ LEARN_CN −0.159, Δ LAST_CN −0.096 | **fail** |
| H5 | PF vs S1, sequence rule, CN pairs, flip 50 | ACC +0.065 [+0.044, +0.085] 33/0/7; Δ LEARN_CN +0.109, Δ LAST_CN +0.152 | **pass** |
| H6 | FGT_CN(E) − FGT_CN(PF) strong and Δ LEARN_CN ≥ −0.05, flip 100 | +0.283 [+0.262, +0.304] 40/0/0, but Δ LEARN_CN −0.159 | **fail** |
| H7 | reward@paced100 − S1, ACC_all CI lo > 0, flip 50 | +0.028 [+0.023, +0.032] 39/0/1 | **pass** |

### 4. Secondary (no pass/fail)

* **Replication of #126 on new seeds (flip 50, all pairs vs E):**
  `E/E-cue@free` +0.066 [+0.059, +0.074] 40/0/0 (#126: +0.060);
  SM@paced100 +0.015 [+0.006, +0.024] 28/0/12 (#126: +0.011).
* **Allocation contrasts at flip 50 (all pairs):** PF − SM: ACC −0.016,
  LEARN −0.005, LAST +0.050 [+0.031, +0.068] 33/0/7, forgetting worse by
  0.014. CF − SM: ACC +0.006, LEARN −0.022, LAST −0.041, forgets 0.038
  less. On CN pairs PF − SM: LEARN_CN +0.046 [+0.028, +0.065] 32/0/8, LAST_CN
  +0.068, ACC_CN +0.004. Funding pattern steps first helps the task just
  trained and hurts retention by about the same amount.
* **Flip 100 (all pairs vs E):** SM +0.123, PF +0.121, CF +0.123 (all
  40/0/0), with Δ LEARN −0.21 to −0.23. The three orders are equivalent on
  ACC at high conflict.
* **Flip 25:** every cued arm at `paced100` is far worse than E (all pairs
  −0.120 to −0.130, 0/0/40).
* **k = 2 and 3 at flip 50 (PF vs E, all pairs):** −0.127 and −0.036
  (0/0/40 and 4/0/36). The flat k = 4 result is again forgetting traded for
  learning.
* **Error gate vs supervised at equal writes:** E − S1 at flip 50: all
  +0.053 (40/0/0), CN +0.082 (37/0/3), LEARN +0.33; replicates #126 H6.
* **E-tag:** 198 writes vs 205 for E at flip 50 (−3.4 %; 3.5 raises blocked
  per seed over four tasks; 331.5 vs 348.9 at flip 100). Its TEST
  predictions equal E's on every (seed, flip): every ACC/LEARN/FGT contrast
  is 0/40/0. The pilot was right: on this design tagging only saves a few
  writes that do not change any TEST prediction.

### Reading

The learning loss of the cued arm at an equal write schedule is **not an
artifact of the funding order.** Moving a quarter of the allowance from cue
to pattern steps changes which task is remembered (more LAST, more
forgetting) but leaves the mean learning loss intact (H2 fails, −0.005) and
does not turn the cue into a gain over `error-gated` at flip 50 (H1, H3
fail). At flip 100 every order keeps a large all-pairs gain (+0.12) and a
CN gain (+0.053), still bought with less learning (H4, H6 fail).

A post-hoc reading (not a tested hypothesis): uncapped, `E/E-cue` spends
about 0.57 × 438 / 4 ≈ 63 writes per task on cue edges, more than
`error-gated`'s whole allowance (≈ 51 writes per task at flip 50). With
threshold 1 and init weight 1, an untouched cue edge makes every cued
target fire, so the cue carries a fixed write cost before it stops causing
errors. Under `error-gated`'s schedule no order can pay it and also learn
the task. A cue whose edges start silent (init 0 on cue edges) or a
per-line budget would test that directly; neither is tested here.

At equal writes, every error-gated arm still beats the supervised control
(H5: PF vs S1 on CN +0.065 with higher learning; E − S1 +0.082), and
reward modulation beats supervision at equal writes by a small, consistent
margin (H7: +0.028, 39/40) while learning slightly less. This cycle is a
negative result for allocation as a fix and a replication of the
write-efficiency ranking error gate > reward > supervised.

## Limitations

- Synthetic graph and artificial OR-rule task only (128/32/8, active 6),
  threshold 1, init 1. Nothing is measured on the real BANC graph
  (DOI 10.7910/DVN/7WTH1N), which is not built on this machine.
- Only three deterministic orders are tested; random subsampling and
  per-target quotas are not.
- The budget is matched to `error-gated`'s natural write schedule, not swept.
- The cue is assumed known at training and test time; task boundaries are
  assumed known by `E-tag`.
- 40 seeds; no multiplicity correction across H1–H7.

B888-3 files (`src/circuit_explorer.rs`, subcircuit recall, activity atlas
and their CLI wiring) are untouched. No NVIDIA dependency, no secrets. CPU
only.

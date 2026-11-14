# B888-4 cycle 16: cue edges that start silent (initial weight 0) under an equal write budget (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123), with the fixed pattern pool of
[CUE_REACH.md](CUE_REACH.md) (#124), the write budgets of
[WRITE_BUDGET.md](WRITE_BUDGET.md) (#126) and the allocation orders of
[WRITE_ALLOCATION.md](WRITE_ALLOCATION.md) (#127). "Cue" and "silent" name
parts of an update rule and of its initial state on a Boolean graph with
4-bit weights. Nothing here is a claim about synapses, biology, insect memory
or BANC physiology.

Module: `src/cue_init.rs`. CLI: `bench-cue-init` (`--census` for the
writes-only pilot).

## Motivation (what #127 left open)

#127 showed that the funding order of the offered steps does not restore the
learning of the cued arm `E/E-cue` at `error-gated`'s write schedule
(`paced100`): LEARN_all stayed ≈ 0.28 below `error-gated` for every order.
Its post-hoc reading, **not tested there**: with threshold 1 and every edge
initialised at weight 1, an untouched cue edge makes every target it reaches
fire whenever the cue is on. Uncapped, `E/E-cue` spent ≈ 63 writes per task
on cue edges before the cue stopped causing errors, more than
`error-gated`'s whole ≈ 51 writes per task. The suggested direct test is a
cue whose edges start silent.

This cycle keeps the `E/E-cue` update rule, the error gate, the budgets, the
episodes and the allocation code of #127 (`write_allocation::step` /
`train_range` / `Checkpoint`, unchanged) and changes only the **initial weight
of the cue-line edges**:

* `E/E-cue` (C1): every edge at init 1 — the #127 source-major arm
  (bit-identical to #127, tested);
* **`E/E-cue0` (Z, primary):** cue-line edges at 0, all other edges at 1,
  source-major funding;
* `E/E-cue0-pattern-first` (ZP, secondary): same init, pattern-first
  funding.

A cue edge at 0 adds no drive (tested), so at the start of every task a cued
arm with silent cue edges predicts exactly like the uncued arm; the cue only
gains weight where the error gate raises it on a false negative in its own
context, and a false positive of a silent cue edge costs no write (the
decrement saturates at 0). Uncapped, the two zero-init orders are
bit-identical (tested and gated). For the uncued arms (S, E, R) the cue lines
are never active, so their cue init is irrelevant and is not changed.

### Design-stage pilot (writes only, seeds 1..5, flip 50; no accuracy computed)

`results/cue-zero-init-census-v1.txt`:

| arm | budget | writes (4 tasks) | cue writes | refused | cue edges ≠ 0 after task 4 |
|---|---|---|---|---|---|
| error-gated | free = paced100 | 192–208 | 0 | 0 | (init 1, never active) |
| E/E-cue (C1) | free | 429–462 | 242–265 | 0 | 21–28 |
| E/E-cue0 (Z) | free | 303–343 | 115–139 | 0 | 11–14 |
| E/E-cue (C1) | paced100 | 192–208 | 114–127 | 3451–3660 | 63–68 |
| **E/E-cue0 (Z)** | paced100 | 192–208 | 71–85 | 1549–2441 | 22–33 |
| E/E-cue0-pattern-first (ZP) | paced100 | 192–208 | **0** | 67–83 | **0** |

Readings that shaped the design (stated before the main run):

* Silent cue edges roughly halve the uncapped cue cost (≈ 30 instead of
  ≈ 63 writes per task) and bring it below `error-gated`'s total budget in all
  five pilot seeds. H2 below is therefore a **manipulation check** on new
  seeds, not independent evidence.
* With pattern-first funding at `paced100`, the zero-init cue receives
  **no** write at all in the pilot: pattern steps use the whole allowance, so
  ZP is expected to behave almost like `error-gated` with an inert cue. ZP is
  kept as a secondary arm (no hypothesis); the primary arm is source-major Z.

## Design (pre-registered)

* **Graph, task, pool, cues:** exactly the #124/#126/#127 fixed-pool design
  (`write_budget::DESIGN`): generated graph per seed, 128 inputs / 32
  targets / fan-in 8, threshold 1, init weight 1 (cue-line edges 0 for Z and
  ZP), 4-bit weights; tasks T1→T4, patterns of 6 active inputs, 32 training
  patterns per task, one shared TEST set of 128 patterns; pair classes seen /
  CN / ICN / UE as in #123; four disjoint greedy cues of up to 16 lines (full
  reach), removed from the pattern pool. Cued arms use cue j while training
  and evaluating task j.
* **Arms (6 rows per budget):** `supervised` (S, control), `error-gated` (E,
  reference), `reward` (R, clean per-target reward-modulated, uncued; the
  #126 code path), `E/E-cue` (C1), `E/E-cue0` (Z), `E/E-cue0-pattern-first`
  (ZP). Same episode order for all arms (8 rounds × 32 patterns = 256 episodes
  per task), no rehearsal, original edges only, topology checked per row.
* **Budgets:** `free` and **`paced100` (primary)**, defined exactly as in
  #126/#127 from uncapped `error-gated`'s per-task cumulative write curve of
  the same (flip, seed). A write = a 4-bit step that changes a stored weight.
* **Seeds:** 241..280 (40 seeds, never used by an earlier B888-4 cycle).
  **Flips:** 25 (secondary), **50 (primary)**, 100 (high conflict).
* **Instrumentation per (row, task):** cap, writes, refused, first refusal,
  attempts, cue writes, cue refused, **cue edges non-zero** and **cue weight
  mass** at the end of each task, weight digest, `same_as_free`,
  `free_alloc_invariant`, `init_ok` (initial weights are the documented init),
  topology.
* **Checkpoint/resume:** circuit weights (`PlasticCircuit::save`, holding the
  zeroed cue edges) plus the #127 text `Checkpoint` (stats, ledger, cost,
  task, next episode); resuming mid-task is bit-exact for every arm at
  `paced100` (tested).

Metrics (ACC, LEARN, FGT, LAST at k = 2..4 on all / seen / CN / ICN / UE
pairs), balanced accuracy, bootstrap (95 % percentile, 2000 resamples, seed
0xB0075EED), **strong** (mean ≥ 0.05, > 0 in ≥ 32 of 40 seeds, CI lower
bound > 0) and the **sequence rule** (Δ ACC strong, Δ LAST ≥ −0.05,
Δ LEARN ≥ −0.05) are as in #123/#124/#126/#127. Base of the gains: uncapped
`error-gated` (= its `paced100` run, gated).

## Design gates (reported first)

* **G1 (integrity):** every row keeps its topology and starts from its
  documented initial weights; every task's writes ≤ its allowance;
  `error-gated@paced100` is bit-identical to uncapped `error-gated` with zero
  refusals; Z and ZP are bit-identical uncapped.
* **G2 (zero init changes the capped run):** at flip 50, Z@paced100 and
  C1@paced100 end with different weights in ≥ 50 % of seeds.
* **G3 (novel pairs):** median CN pairs per (seed, task) at flip 50 ≥ 20.

If any gate fails, H1–H7 are declared **not evaluable**.

## Hypotheses (k = 4, seeds 241..280; Z = `E/E-cue0@paced100`, C1 = `E/E-cue@paced100`, E = uncapped `error-gated`, S1 = `supervised@paced100`, Zfree = `E/E-cue0@free`)

* **H1 (primary: cue gain at equal writes with silent cue edges, flip 50):**
  Z vs E passes the sequence rule on all pairs.
* **H2 (manipulation check: the cue's write cost fits the budget, flip 50):**
  uncapped Z makes fewer cue writes over the four tasks than uncapped E makes
  writes in total, in ≥ 80 % of seeds (≥ 32 of 40). Its direction was seen in
  the pilot.
* **H3 (the init is the lever for learning, flip 50):**
  LEARN_all(Z) − LEARN_all(C1) is strong.
* **H4 (generalisation, flip 50):** Z vs E passes the sequence rule on CN
  pairs.
* **H5 (vs the supervised control at equal writes, flip 50):** Z vs S1 passes
  the sequence rule on CN pairs.
* **H6 (forgetting not bought by less learning, flip 100):**
  FGT_CN(E) − FGT_CN(Z) is strong **and** mean Δ LEARN_CN(Z − E) ≥ −0.05.
* **H7 (silent cue keeps the uncapped cue gain, flip 50):** Zfree vs E passes
  the sequence rule on all pairs (#127: `E/E-cue@free` +0.066, 40/0/0).

**Secondary (reported, no pass/fail):** ZP vs E and vs Z; C1 vs E (replication
of #127 on new seeds); Zfree vs `E/E-cue@free`; E vs S1 and R vs S1 at equal
writes (replications); flip 25; k = 2 and 3; seen, ICN and UE pairs; writes,
cue writes, cue share, refusals, non-zero cue edges, cue mass, saturation.

No multiplicity correction is applied. The analysis is run once. Nothing
above the Results heading is changed after the run except the "pending"
placeholder.

## Command

```bash
cargo +1.89 run --release -- bench-cue-init --csv results/cue-zero-init-v1.csv > results/cue-zero-init-report-v1.txt
# pilot (writes only): cargo +1.89 run --release -- bench-cue-init --census > results/cue-zero-init-census-v1.txt
```

## Results

Run once after the pre-registration commit 7d44bd9 had been pushed. The grid
(1440 rows: 3 flips × 40 seeds × 6 arms × 2 budgets) takes about 4.3 min on
CPU. Two runs give a byte-identical CSV (sha256 `579485fb…e9948800b`) and an
identical report. Every row keeps the original topology
(`topology_unchanged_all=true`). Files: `results/cue-zero-init-v1.csv`,
`results/cue-zero-init-report-v1.txt`, pilot
`results/cue-zero-init-census-v1.txt`.

### 1. Design gates (reported first)

* **G1 passes:** every row starts from its documented initial weights and
  keeps its topology; writes never exceed the allowance;
  `error-gated@paced100` is bit-identical to uncapped `error-gated` with zero
  refusals; Z and ZP are bit-identical uncapped on all 120 (flip, seed).
* **G2 passes:** Z and C1 at `paced100` end with different weights in 100 %
  of seeds at every flip.
* **G3 passes:** median CN pairs per (seed, task) 125 ≥ 20.

### 2. Settings at flip 50 (k = 4, mean over 40 seeds)

| arm @ budget | ACC_all | ACC_CN | LEARN_all | LEARN_CN | FGT_all | FGT_CN | LAST_all | writes (4 tasks) | cue writes | cue share |
|---|---|---|---|---|---|---|---|---|---|---|
| supervised @ free | 0.615 | 0.581 | 0.881 | 0.762 | 0.354 | 0.241 | 0.873 | 3601 | 0 | 0 |
| error-gated @ free (= paced100) | 0.637 | 0.624 | 0.971 | 0.911 | 0.446 | 0.383 | 0.973 | 207 | 0 | 0 |
| E/E-cue (C1) @ free | 0.692 | 0.633 | 0.971 | 0.903 | 0.372 | 0.361 | 0.972 | 445 | 253 | 0.569 |
| E/E-cue0 (Z) @ free (= ZP @ free) | 0.661 | 0.630 | 0.971 | 0.907 | 0.413 | 0.369 | 0.972 | 336 | 134 | 0.398 |
| supervised @ paced100 | 0.582 | 0.541 | 0.647 | 0.564 | 0.086 | 0.030 | 0.633 | 207 | 0 | 0 |
| reward @ paced100 | 0.597 | 0.579 | 0.627 | 0.573 | 0.041 | −0.008 | 0.599 | 207 | 0 | 0 |
| E/E-cue (C1) @ paced100 | 0.640 | 0.608 | 0.694 | 0.647 | 0.072 | 0.051 | 0.706 | 207 | 124 | 0.598 |
| **E/E-cue0 (Z) @ paced100** | 0.655 | 0.628 | 0.822 | 0.762 | 0.223 | 0.178 | 0.784 | 207 | 77 | 0.371 |
| E/E-cue0-pattern-first (ZP) @ paced100 | 0.637 | 0.624 | 0.971 | 0.911 | 0.446 | 0.383 | 0.973 | 207 | 0 | 0 |

Silent cue edges cut the uncapped cue cost from ≈ 63 to ≈ 33 writes per task
(below `error-gated`'s ≈ 52) and, at `paced100`, recover about half of the
cued arm's learning loss (LEARN_all 0.69 → 0.82; `error-gated` 0.97). Z still
spends 37 % of the allowance on cue edges and learns 0.15 less than
`error-gated`. (`saturated_end` is higher for Z because cue edges left at 0
count as saturated.)

### 3. Hypotheses (k = 4; mean [95 % CI] seeds better/tied/worse)

| | statistic | value | result |
|---|---|---|---|
| G1 | integrity (topology, init, caps, E unchanged, Z = ZP uncapped) | all rows | **pass** |
| G2 | Z ≠ C1 final weights in ≥ 50 % of seeds, flip 50 | 100 % | **pass** |
| G3 | median CN pairs ≥ 20 | 125 | **pass** |
| H1 | Z vs E, sequence rule, all pairs, flip 50 | ACC +0.018 [+0.012, +0.025] 32/0/8 (mean < 0.05); Δ LEARN −0.149, Δ LAST −0.189 | **fail** |
| H2 | cue writes(Zfree) < writes(E), ≥ 80 % of seeds, flip 50 (manipulation check) | 40/40 (134 vs 207 over four tasks) | **pass** |
| H3 | LEARN_all(Z) − LEARN_all(C1) strong, flip 50 | +0.128 [+0.119, +0.137] 40/0/0 | **pass** |
| H4 | Z vs E, sequence rule, CN pairs, flip 50 | ACC_CN +0.004 [−0.013, +0.021] 20/0/20; Δ LEARN_CN −0.150 | **fail** |
| H5 | Z vs S1, sequence rule, CN pairs, flip 50 | ACC_CN +0.087 [+0.069, +0.106] 35/0/5; Δ LEARN_CN +0.197, Δ LAST_CN +0.176 | **pass** |
| H6 | FGT_CN(E) − FGT_CN(Z) strong and Δ LEARN_CN ≥ −0.05, flip 100 | +0.158 [+0.134, +0.183] 38/0/2, but Δ LEARN_CN −0.140 | **fail** |
| H7 | Zfree vs E, sequence rule, all pairs, flip 50 | ACC +0.025 [+0.022, +0.027] 40/0/0 (mean < 0.05); Δ LEARN −0.000 | **fail** |

### 4. Secondary (no pass/fail)

* **ZP @ paced100 is `error-gated`:** as the pilot predicted, pattern steps
  use the whole allowance, the zero-init cue never receives a write, and every
  ACC/LEARN/LAST/FGT contrast with E is 0/40/0.
* **Z − C1 at `paced100` (flip 50, all pairs):** ACC +0.015 [+0.004, +0.025]
  26/0/14, LEARN +0.128, LAST +0.078, but forgetting 0.151 worse (0/0/40).
  On CN: ACC_CN +0.020 [−0.004, +0.043], LEARN_CN +0.115.
* **Replication of #127 on new seeds (flip 50, all pairs vs E):**
  C1 @ paced100 +0.003 [−0.006, +0.013] 23/0/17 (#127: +0.015);
  C1 @ free +0.056 [+0.051, +0.061] 40/0/0 (#127: +0.066).
* **Silent cue, uncapped:** Zfree − Cfree −0.031 [−0.035, −0.027] 0/0/40 on
  all pairs at flip 50 with equal learning: the init-1 cue buys more
  protection from forgetting when writes are free.
* **Flip 100 (vs E):** Z @ paced100 all +0.056 (40/0/0), CN −0.021 (16/0/24);
  C1 @ paced100 all +0.124, CN +0.066 (33/0/7). Zfree all +0.054, CN +0.036;
  Cfree all +0.107, CN +0.101. At high conflict the init-1 cue is better on
  ACC at either budget.
* **Flip 25 (vs E, all pairs):** Z @ paced100 +0.007 [+0.002, +0.013]
  29/0/11, C1 @ paced100 −0.122 (0/0/40): at low conflict the silent cue
  removes the large equal-budget loss of the init-1 cue.
* **k = 2 and 3 at flip 50 (Z vs E, all pairs):** +0.005 and +0.011; Δ LEARN
  −0.099 and −0.136.
* **Equal-write ranking (flip 50, all pairs vs S1):** E +0.054 (40/0/0),
  Z +0.073 (40/0/0), C1 +0.058, R +0.015 [+0.010, +0.020] 32/0/8 (#127 H7:
  +0.028), R learning 0.020 less than S1.

### Reading

The #127 post-hoc reading is **partly confirmed**. Starting cue edges at 0
halves the cue's uncapped write cost and brings it under `error-gated`'s
budget (H2), and at an equal write schedule it recovers about half of the
cued arm's learning loss (H3: +0.128 LEARN over the init-1 cue). It does
**not** make the cue a gain over `error-gated` at equal writes (H1, H4 fail:
+0.018 all, +0.004 CN), because Z still spends about a third of the allowance
on cue edges and learns 0.15 less than `error-gated`; the forgetting it
avoids at flip 100 is again bought with less learning (H6 fails). Silencing
the cue also weakens what the cue does when writes are free (H7 fails:
+0.025 vs +0.056 for the init-1 cue) and at high conflict. In short: the
fixed init cost explains roughly half of the equal-budget learning loss, not
all of it, and removing it trades away part of the cue's retention benefit.

At equal writes the zero-init cued arm beats the supervised control on CN
pairs with more learning (H5: +0.087, Δ LEARN_CN +0.197), and the
write-efficiency ranking error gate > reward > supervised replicates on new
seeds. All of this is on a synthetic graph with artificial labels.

## Limitations

- Synthetic graph and artificial OR-rule task only (128/32/8, active 6),
  threshold 1, init 1 (cue edges 0). Nothing is measured on the real BANC
  graph (DOI 10.7910/DVN/7WTH1N), which is not built on this machine.
- Only two cue inits (1 and 0) are tested; intermediate inits, per-line
  budgets and learned cue gates are not.
- The budget is matched to `error-gated`'s natural write schedule, not swept.
- The cue is assumed known at training and test time.
- 40 seeds; no multiplicity correction across H1–H7.

B888-3 files (`src/circuit_explorer.rs`, subcircuit recall, activity atlas
and their CLI wiring) are untouched. No NVIDIA dependency, no secrets. CPU
only.

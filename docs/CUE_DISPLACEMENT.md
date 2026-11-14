# B888-4 cycle 17: is the silent cue's remaining learning loss write displacement? (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123), with the fixed pattern pool of
[CUE_REACH.md](CUE_REACH.md) (#124), the write budgets of
[WRITE_BUDGET.md](WRITE_BUDGET.md) (#126), the allocation code of
[WRITE_ALLOCATION.md](WRITE_ALLOCATION.md) (#127) and the silent cue of
[CUE_ZERO_INIT.md](CUE_ZERO_INIT.md) (#128). "Cue", "displacement" and
"interference" name parts of an update rule and of its write accounting on a
Boolean graph with 4-bit weights. Nothing here is a claim about synapses,
biology, insect memory or BANC physiology.

Module: `src/cue_displacement.rs`. CLI: `bench-cue-displacement`
(`--census` for the writes-only pilot).

## Motivation (what #128 left open)

#128 started the cue-line edges at weight 0 (`E/E-cue0`, "Z") and, at
`error-gated`'s write schedule (`paced100`), recovered about half of the cued
arm's learning loss (H3 +0.128), but Z still learned ≈ 0.15 less than
`error-gated` (LEARN_all 0.822 vs 0.971 at flip 50) while spending ≈ 37 % of
the allowance on cue edges. Two readings of the remaining loss were left
untested:

* **(a) displacement:** every cue write is a write the pattern edges no
  longer get; Z simply has fewer pattern writes than `error-gated`;
* **(b) interference:** the cue edges that gain weight change which targets
  fire and therefore which updates the error gate offers, so the pattern
  writes Z does make are less useful.

This cycle separates (a) from (b) with two counter-factual budgets. The
update rules, initial weights, episodes, cues and `paced100` caps are the
#128 code paths (bit-identical with charging on, tested); only **which writes
are charged** and **which curve caps them** change:

* `E/E-cue0@paced100` (**Z**): the #128 primary arm (every write charged).
* `E/E-cue0-exempt@paced100` (**Zx**): the same rule and init, but cue-edge
  writes are not charged; only pattern-edge writes count against
  `error-gated`'s paced curve. Zx therefore has `error-gated`'s pattern-write
  schedule **plus** free cue writes: its *total* writes are larger and are
  reported as such. Zx is a decomposition tool, not an equal-budget arm.
* `error-gated@Z-pattern` (**Em**): uncued `error-gated`, capped at the
  per-episode cumulative **pattern**-write curve that Z@paced100 actually made
  in the same (flip, seed). Em has exactly the pattern budget the cue leaves,
  with no cue at all.
* References: uncapped `error-gated` (**E**, = its `paced100` run, gated),
  `supervised@paced100` (**S1**, control), `reward@paced100` (**R1**), and
  uncapped `E/E-cue0` (**Zfree**).

If (a) holds, Zx learns like E and Em learns like Z. If (b) holds, Zx still
learns less than E and Em learns more than Z.

### Design-stage pilot (writes only, seeds 1..5, flip 50; no accuracy computed)

`results/cue-displacement-census-v1.txt` (four tasks summed):

| setting | pattern writes | cue writes | total writes | refused |
|---|---|---|---|---|
| E (free = paced100) | 192–208 | 0 | 192–208 | 0 |
| Z @ paced100 | 116–131 | 71–85 | 192–208 | 1549–2441 |
| Zx @ paced100 (cue exempt) | 186–204 | 115–182 | 303–368 | 0 (4 seeds), 69 (1 seed) |
| Em (E at Z's pattern curve) | 116–131 | 0 | 116–131 | 1383–2146 |
| Zfree | 188–204 | 115–139 | 303–343 | 0 |

Readings that shaped the design (stated before the main run):

* Z gives up ≈ 40 % of `error-gated`'s pattern writes (≈ 120 of ≈ 200), so
  displacement is large enough to matter.
* Uncapped, the silent cue does **not** reduce the pattern writes needed
  (Zfree ≈ E on pattern writes). Consequently Zx is bit-identical to Zfree in
  4 of 5 pilot seeds (its pattern writes never reach the cap). #128 already
  reported Zfree vs E Δ LEARN_all −0.000 at flip 50 (seeds 241..280). **H1
  and H2 below are therefore close to manipulation checks on new seeds**, not
  independent evidence; the informative test of displacement is **H3** (does
  `error-gated`, given only Z's pattern writes, learn as little as Z?) and
  **H4** (do Z's cue writes buy anything over the same pattern writes?).
* Em's cap binds in every pilot seed (refusals > 0).

## Design (pre-registered)

* **Graph, task, pool, cues:** exactly the #124–#128 fixed-pool design
  (`write_budget::DESIGN`): generated graph per seed, 128 inputs / 32
  targets / fan-in 8, threshold 1, init weight 1 (cue-line edges 0 for the
  Z arms), 4-bit weights; tasks T1→T4, patterns of 6 active inputs, 32
  training patterns per task, one shared TEST set of 128 patterns; pair
  classes seen / CN / ICN / UE as in #123; four disjoint greedy cues of up to
  16 lines (full reach), removed from the pattern pool. Cued arms use cue j
  while training and evaluating task j.
* **Settings (7 rows per (flip, seed)):** S1, R1, E, Z, Zx, Em, Zfree.
  Same episode order for all settings (8 rounds × 32 patterns = 256 episodes
  per task), no rehearsal, original edges only, topology checked per row.
* **Budget:** `paced100` defined exactly as in #126–#128 from uncapped
  `error-gated`'s per-task cumulative write curve of the same (flip, seed). A
  write = a 4-bit step that changes a stored weight. Em's cap per episode is
  Z@paced100's cumulative pattern writes after that episode.
* **Seeds:** 281..320 (40 seeds, never used by an earlier B888-4 cycle).
  **Flips:** 25 (secondary), **50 (primary)**, 100 (high conflict).
* **Instrumentation per (row, task):** cap, charged writes, pattern writes,
  cue writes, total writes, refused, cue refused, first refusal, attempts,
  cue edges non-zero and cue weight mass at the end of each task, weight
  digest, `init_ok`, `paced_identity`, topology.
* **Checkpoint/resume:** circuit weights (`PlasticCircuit::save`) plus the
  text `Checkpoint` (#127 checkpoint + per-episode cumulative cue-write
  curve); resuming mid-task is bit-exact for every setting (tested).

Metrics (ACC, LEARN, FGT, LAST at k = 2..4 on all / seen / CN / ICN / UE
pairs), balanced accuracy, bootstrap (95 % percentile, 2000 resamples, seed
0xB0075EED), **strong** (mean ≥ 0.05, > 0 in ≥ 32 of 40 seeds, CI lower
bound > 0) and the **sequence rule** (Δ ACC strong, Δ LAST ≥ −0.05,
Δ LEARN ≥ −0.05) are as in #123–#128. New for this cycle:
**non-inferior** = bootstrap CI lower bound > −0.05; **equivalent** = 95 %
CI inside (−0.05, +0.05).

## Design gates (reported first)

* **G1 (integrity):** every row keeps its topology, starts from its
  documented initial weights and makes charged writes ≤ its allowance;
  `error-gated@paced100` is bit-identical to uncapped `error-gated` with zero
  refusals; Em's per-task cap equals Z's pattern writes; Zx's per-task cap
  equals E's writes and Zx's pattern writes do not exceed it.
* **G2 (manipulation):** at flip 50, Zx and Z end with different weights in
  ≥ 50 % of seeds **and** Em refuses at least one step in ≥ 50 % of seeds.
* **G3 (novel pairs):** median CN pairs per (seed, task) at flip 50 ≥ 20.

If any gate fails, H1–H8 are declared **not evaluable**.

## Hypotheses (k = 4, seeds 281..320)

* **H1 (displacement is a lever, flip 50):** LEARN_all(Zx) − LEARN_all(Z) is
  strong. (Close to a manipulation check, see pilot.)
* **H2 (displacement accounts for the whole loss, flip 50):**
  LEARN_all(Zx) − LEARN_all(E) is non-inferior (CI lower bound > −0.05).
  (Close to a manipulation check, see pilot.)
* **H3 (primary: the converse, flip 50):** LEARN_all(Em) − LEARN_all(Z) is
  equivalent to 0 (95 % CI inside ±0.05): `error-gated` restricted to Z's
  pattern writes learns as little as Z.
* **H4 (what the cue writes buy over the same pattern writes, flip 50):**
  ACC_all(Z) − ACC_all(Em) is strong.
* **H5 (cue gain at equal pattern writes, unequal total, flip 50):** Zx vs E
  passes the sequence rule on all pairs.
* **H6 (generalisation, flip 50):** Zx vs E passes the sequence rule on CN
  pairs.
* **H7 (forgetting not bought by less learning, flip 100):**
  FGT_CN(E) − FGT_CN(Zx) is strong **and** mean Δ LEARN_CN(Zx − E) ≥ −0.05.
* **H8 (vs the supervised control at equal total writes, flip 50):** Z vs S1
  passes the sequence rule on CN pairs (replication of #128 H5 on new seeds).

**Secondary (reported, no pass/fail):** Em vs E (pure pattern-budget effect
of losing ≈ 40 % of writes), Z vs E, Zfree vs E, Zx vs Zfree (and
`differ_frac`), Em vs S1, E vs S1 and R1 vs S1 at equal writes; flip 25;
k = 2 and 3; seen, ICN and UE pairs; pattern / cue / total writes, refusals,
non-zero cue edges, cue mass, saturation.

No multiplicity correction is applied. The analysis is run once. Nothing
above the Results heading is changed after the run except the "pending"
placeholder.

## Command

```bash
cargo +1.89 run --release -- bench-cue-displacement --csv results/cue-displacement-v1.csv > results/cue-displacement-report-v1.txt
# pilot (writes only): cargo +1.89 run --release -- bench-cue-displacement --census > results/cue-displacement-census-v1.txt
```

## Results

Run once after the pre-registration commit 0572b21 had been pushed. The grid
(840 rows: 3 flips × 40 seeds × 7 settings) takes about 2.2 min on CPU. Two
runs give a byte-identical CSV (sha256 `d1636473…40985f88`) and an identical
report. Every row keeps the original topology (`topology_unchanged_all=true`).
Files: `results/cue-displacement-v1.csv`,
`results/cue-displacement-report-v1.txt`, pilot
`results/cue-displacement-census-v1.txt`.

### 1. Design gates (reported first)

* **G1 passes:** every row keeps its topology, starts from its documented
  initial weights and stays within its allowance; `error-gated@paced100` is
  bit-identical to uncapped `error-gated` with zero refusals; Em's caps equal
  Z's pattern writes and Zx's caps equal E's writes.
* **G2 passes:** Zx ≠ Z in 100 % of seeds and Em's cap binds in 100 % of
  seeds at every flip.
* **G3 passes:** median CN pairs per (seed, task) 123 ≥ 20.

### 2. Settings at flip 50 (k = 4, mean over 40 seeds)

| setting | ACC_all | ACC_CN | LEARN_all | LEARN_CN | FGT_all | FGT_CN | LAST_all | pattern writes | cue writes | total writes |
|---|---|---|---|---|---|---|---|---|---|---|
| supervised @ paced100 (S1) | 0.579 | 0.538 | 0.648 | 0.565 | 0.093 | 0.036 | 0.618 | 213 | 0 | 213 |
| reward @ paced100 (R1) | 0.596 | 0.557 | 0.627 | 0.564 | 0.042 | 0.009 | 0.590 | 213 | 0 | 213 |
| error-gated @ free (E) | 0.630 | 0.619 | 0.972 | 0.910 | 0.456 | 0.388 | 0.973 | 213 | 0 | 213 |
| **E/E-cue0 @ paced100 (Z)** | 0.647 | 0.614 | 0.820 | 0.756 | 0.231 | 0.190 | 0.771 | 135 | 78 | 213 |
| **E/E-cue0-exempt @ paced100 (Zx)** | 0.653 | 0.624 | 0.970 | 0.909 | 0.423 | 0.379 | 0.968 | 208 | 145 | 353 |
| **error-gated @ Z-pattern (Em)** | 0.621 | 0.607 | 0.805 | 0.750 | 0.245 | 0.192 | 0.749 | 135 | 0 | 135 |
| E/E-cue0 @ free (Zfree) | 0.653 | 0.623 | 0.971 | 0.908 | 0.425 | 0.380 | 0.970 | 208 | 140 | 348 |

Z gives up 37 % of `error-gated`'s pattern writes to the cue (135 of 213).
Giving those pattern writes back (Zx) restores `error-gated`'s learning;
taking the same pattern writes away from `error-gated` without any cue (Em)
reproduces Z's learning loss. Zx equals Zfree in 70 % of seeds at flip 50
(its pattern writes rarely reach the cap), as the pilot predicted.

### 3. Hypotheses (k = 4; mean [95 % CI] seeds better/tied/worse)

| | statistic | value | result |
|---|---|---|---|
| G1 | integrity (topology, init, caps, E unchanged, Em/Zx caps matched) | all rows | **pass** |
| G2 | Zx ≠ Z in ≥ 50 % of seeds and Em binds in ≥ 50 %, flip 50 | 100 % / 100 % | **pass** |
| G3 | median CN pairs ≥ 20 | 123 | **pass** |
| H1 | LEARN_all(Zx) − LEARN_all(Z) strong, flip 50 (near manipulation check) | +0.150 [+0.143, +0.157] 40/0/0 | **pass** |
| H2 | LEARN_all(Zx) − LEARN_all(E) non-inferior, flip 50 (near manipulation check) | −0.001 [−0.003, −0.000] 15/5/20 | **pass** |
| H3 | LEARN_all(Em) − LEARN_all(Z) equivalent to 0 (CI in ±0.05), flip 50 | −0.016 [−0.022, −0.009] 7/0/33 | **pass** |
| H4 | ACC_all(Z) − ACC_all(Em) strong, flip 50 | +0.026 [+0.020, +0.032] 37/0/3 (mean < 0.05) | **fail** |
| H5 | Zx vs E, sequence rule, all pairs, flip 50 | ACC +0.023 [+0.020, +0.027] 40/0/0 (mean < 0.05); Δ LEARN −0.001 | **fail** |
| H6 | Zx vs E, sequence rule, CN pairs, flip 50 | ACC_CN +0.005 [−0.002, +0.012] 27/3/10 | **fail** |
| H7 | FGT_CN(E) − FGT_CN(Zx) strong and Δ LEARN_CN ≥ −0.05, flip 100 | +0.029 [+0.013, +0.046] 28/0/12 (not strong); Δ LEARN_CN +0.007 | **fail** |
| H8 | Z vs S1, sequence rule, CN pairs, flip 50 | ACC_CN +0.076 [+0.057, +0.095] 37/0/3; Δ LEARN_CN +0.191, Δ LAST_CN +0.146 | **pass** |

### 4. Secondary (no pass/fail)

* **Em vs E (losing 37 % of pattern writes, no cue), flip 50, all pairs:**
  ACC −0.008 [−0.013, −0.003], LEARN −0.167, LAST −0.223, forgetting
  +0.212 less: the capped `error-gated` arm learns and forgets less, with
  nearly the same final ACC.
* **Interference, if any, is small and goes the other way:** Em learns
  0.016 *less* than Z (H3 statistic, 7/0/33), so the pattern writes Z makes
  are, if anything, slightly more useful with the cue on, not less.
* **Flip 100 (Z vs Em, all pairs):** ACC +0.061 [+0.054, +0.069] 40/0/0,
  LEARN +0.050, LAST +0.086 (passes the sequence rule); on CN +0.051
  [+0.038, +0.065] 36/0/4. At high conflict the cue's writes do buy accuracy
  over the same pattern writes; at flip 50 and 25 they buy +0.026 and +0.011.
* **Flip 25 (vs E, all pairs):** Z +0.012, Em +0.001, Zx +0.011; LEARN
  Z −0.112, Em −0.119 (same pattern as flip 50).
* **k = 2 and 3 at flip 50 (Z − Em, all pairs):** ACC +0.010 and +0.019,
  LEARN +0.009 and +0.014.
* **Replications at equal total writes (flip 50, all pairs vs S1):**
  E +0.051 (38/0/2), Z +0.069 (40/0/0), R +0.017 [+0.012, +0.023] 35/0/5
  with 0.021 less learning. Em, with 37 % fewer writes than S1, is +0.042
  (37/0/3) on all pairs and +0.069 on CN.
* **Zx vs Zfree:** identical in 28/40 seeds at flip 50 and 40/40 at
  flip 100: once cue writes are free, the pattern edges do not need more
  than `error-gated`'s schedule.

### Reading

The remaining half of the silent cue's equal-budget learning loss is
**write displacement**, not interference, on this synthetic task. Charging
the cue's writes takes 37 % of the pattern writes; restoring them (Zx)
restores `error-gated`'s learning exactly (H1, H2), and removing the same
pattern writes from `error-gated` without any cue (Em) reproduces the loss
within ±0.05 (H3; Em even learns 0.016 less than Z). Together with #128 this
closes the learning-gap question for this design: the init-1 cue lost
learning to a fixed initial write cost, the silent cue loses it to the
writes it spends, and nothing else of comparable size is left.

What the cue's writes buy is small at flip 50: +0.026 ACC over the same
pattern writes (H4 fails), and +0.023 over `error-gated` even when they are
free (H5 fails), with no CN gain (H6) and a weak forgetting reduction at
flip 100 (H7 fails). Only at high conflict (flip 100, secondary) is the cue
clearly worth its writes (+0.061, 40/0/0 over Em). At equal total writes
the zero-init cued arm still beats the supervised control on CN pairs with
more learning (H8), and the ranking error gate > reward > supervised
replicates. All of this is on a synthetic graph with artificial labels.

Next step suggested by this result (not tested here): the cue lever is
nearly exhausted at k = 4 on this design, so the open question for the
series is a fair generalisation/forgetting test over **more tasks (larger
k) and held-out task orders**, where the cue's protection at high conflict
could matter more than its write cost.

## Limitations

- Synthetic graph and artificial OR-rule task only (128/32/8, active 6),
  threshold 1, init 1 (cue edges 0). Nothing is measured on the real BANC
  graph (DOI 10.7910/DVN/7WTH1N), which is not built on this machine.
- Zx is a counter-factual accounting, not an equal-budget arm: it has more
  total writes than E.
- Em's cap comes from Z's own run, so Em and Z are matched on pattern-write
  count and timing, not on which edges are written.
- Only the source-major allocation and the `paced100` budget are tested.
- The cue is assumed known at training and test time.
- 40 seeds; no multiplicity correction across H1–H8.

B888-3 files (`src/circuit_explorer.rs`, subcircuit recall, activity atlas
and their CLI wiring) are untouched. No NVIDIA dependency, no secrets. CPU
only.

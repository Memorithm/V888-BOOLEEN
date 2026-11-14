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

Pending: the main run on seeds 281..320 happens only after this
pre-registration commit has been pushed.

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

# B888-4 cycle 26: can the learning / forgetting trade-off of shared cue writes (#139) be split in time when the late / early partner-block window is cut on the **write clock** (half of the uncapped error-gated write curve of the stage) rather than on the episode clock — at equal lines and equal write budget? (synthetic graph)

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
[CUE_OWNERSHIP.md](CUE_OWNERSHIP.md) (#139) and the two-phase schedule of
[CUE_DEFER.md](CUE_DEFER.md) (#184). "Cue", "code", "block", "partner",
"phase", "late", "early", "frozen", "guard", "write clock", "task order" and
"forgetting" name parts of an update rule and an artificial benchmark on a
Boolean graph with 4-bit weights. Nothing here is a claim about synapses,
biology, insect memory or BANC physiology.

Module: `src/cue_write_clock.rs` (engine support:
`consolidation_guard::budget_split` and
`consolidation_guard::run_setting_scheduled_at`). CLI: `bench-cue-write-clock`
(`--census` for the label-free pilot).

## Motivation (what #184 left open)

Cycle 25 (#184) froze the partner block of the matched share-2 code during
the first half of each task's **episodes** (`h_j = ⌊n_j / 2⌋`) and opened it
in the second (Zgl2m), with the reverse as a timing control (Zge2m). The
question was whether the #139 shared-vs-owned trade-off is about *when*
partner-block writes happen or about *how many* of them happen.

G2 failed, so H1–H13 were not evaluable. Under the late window the partner
block was almost never written in the second episode-half (mean ≈ 1
partner-line edge change per run at k = 16, flip 100, vs ≈ 36 in the first
half under the shared write). The report's reading: the guarded error gate
writes while errors are frequent, so by episode 128 almost every write has
already happened; "when" collapsed onto "whether" because the late arm made
≈ 0 partner writes.

The equal-budget protocol already has a better clock: the per-stage
cumulative write curve of uncapped error-gated (the same curve that sets
every capped arm's `paced100` allowance). This cycle keeps the same cue
lists, the same partner map, the same four write modes and the same budget,
and only changes the split: `h_j` is the first episode at which that curve
reaches `⌊cap / 2⌋`. The late and early arms then have partner-writable
windows that each cover about half of the write budget, so a timing effect
can be separated from a count effect if one exists. The schedule remains a
pure function of the fixture, the partner map and the E curve of the same
(k, flip, seed, order); it does not depend on the measured arm's own
writes.

This is a direct manipulation of one variable (the window in which
partner-block edges are writable), not a new code, a new rule or a new test
of tests. The episode-clock arms of #184 are not re-run; the question is
whether the same modes, cut on this clock, bind as a manipulation and then
whether late beats early, share or owned.

## Write modes (pre-registered, no tuning)

All modes use the **same** cue lists as the #136 / #139 / #184 share-2 code
`C_j = sorted(B_j ∪ B_σ(j))` (same lines per task, same partner map, same
line multiplicity, same cue drive at evaluation). Let `F_j = B_σ(j) \ B_j`
(the #139 frozen lines) and let `n_j` be the number of training episodes of
task j (32 patterns × 8 rounds = 256 in the main design). For stage *i* of
a given (k, flip, seed, order), let `cum` be the per-episode cumulative
write curve of uncapped error-gated in that stage and
`h_i = budget_split(cum)` = the first episode with `cum[e] ≥ ⌊cap / 2⌋`
(empty or zero-write stages: `h_i = n_j`). Episodes `0..h_i` are the first
write-half, `h_i..n_j` the second (the crossing episode belongs to the
second half).

* **share** (Zg2m, = #136 / #139 / #184 Zg2m): `F_j` writable in both write-halves.
* **owned** (Zgo2m, = #139 / #184 Zgo2m): `F_j` frozen in both write-halves.
* **late** (Zgl2m): `F_j` frozen during episodes `0..h_i`, writable during
  `h_i..n_j`.
* **early** (Zge2m, timing control): `F_j` writable during `0..h_i`, frozen
  during `h_i..n_j`.

"Frozen" is exactly the #139 mask (`consolidation_guard::step_masked`): the
lines stay active and drive the targets, their edges are not written, not
offered to the budget and not counted as a free edge by `last-resort`. The
schedule is a pure function of the fixture, the partner map and the E
write curve, not of the measured arm's state.

## Arms (9 rows per (k, flip, seed, order))

* `supervised@paced100` (**S1**, control; on this teacher-clamped task the
  associative co-activation rule is the same update, see CONSOLIDATION.md).
* `reward@paced100` (**R1**, reward-modulated).
* `error-gated@free` (**E**, = its `paced100` run, gated in G1).
* `E/E-cue0+last-resort@paced100` private blocks (**Zg1**).
* `E/E-cue0+last-resort@paced100` matched share-2 code in the share
  (**Zg2m**), owned (**Zgo2m**), late (**Zgl2m**) and early (**Zge2m**)
  modes.
* `E/E-cue0+last-resort@paced100` random share-2 code in the late mode
  (**Zgl2r**; partner = seeded Sattolo k-cycle with a **new salt**
  `0x0DE5_B888_4C26_5A77`).

The matched partner map is the #136 best candidate tour
(`cue_partner::partner_map(Matched)`), as in #139 / #184.

**Equal budget:** every capped arm uses the `paced100` per-episode cumulative
write curve of uncapped E in the same (k, flip, seed, order); cue writes are
charged. **Equal resources:** all cued arms of a (k, seed) use the same
reserved lines and cue-line edges; the five share-2 arms also have the same
lines per task and line multiplicity. A scheduled run with the same mask in
both write-halves is bit-identical to `run_setting_masked` at any split
(tested), and with no mask bit-identical to `run_setting` (tested).
`run_setting_scheduled_at` with `phase_split` (the cycle 25 clock) is
bit-identical to `run_setting_scheduled` (tested).

## Design (pre-registered)

* Graph, task, pool, splits, budget: exactly #131–#184 (128 inputs / 32
  targets / fan-in 8, threshold 1, init weight 1, 4-bit weights, 6 active
  inputs, 32 training patterns per task, 8 rounds, one shared TEST set of 128
  patterns, pair classes seen / CN / ICN / UE, no rehearsal, original edges
  only, topology checked per row).
* **k = 8, 16, 24** (own fixtures). **Flips 25, 50, 100.** Flips 100 and 50
  carry the primary hypotheses; flip 25 is secondary only.
* **Seeds 641..680** (40 seeds, never used by an earlier B888-4 cycle).
* **Orders:** 0 = identity (secondary only); held-out 1 = reversed, 2..4 =
  seeded Fisher–Yates permutations (#133 generator) with a **new salt**
  `0x0DE5_B888_4C26_0A1D`, shared by every flip and arm.
* **Metrics** as #131–#184 (stage coordinates): ACC, LEARN, FGT, LAST on all /
  CN pairs; per seed the mean over its 4 held-out orders (n = 40 paired
  seeds). Worst order: per seed, min over held-out orders of the paired
  difference.
* **Instrumentation per row:** code, write mode, partner map, mean partner
  agreement, conflicting links, lines per task, writable partner lines per
  task in each write-half, mean cue reach, cue-line edges, live / full cue
  edges, cap, charged / effective / cue writes, cue-write share, **write-clock
  splits** (per stage) and their mean, stage episode count, **E write
  fraction per write-half**, **frozen skips per write-half** (edge steps the
  mask removed), **frozen moves** (edges of a frozen line whose weight
  changed during a half in which it was frozen; must be 0), **partner-line
  moves per write-half** (edges of `F_j` whose weight changed during the
  first / second write-half of task j, summed over stages; recorded for
  every share-2 arm), refused, overwrites, deferred, net reversals,
  consolidated edges, saturated edges, init / paced / cap / code flags,
  weight digest, topology.
* **Checkpoint/resume:** the #132 engine checkpoint; the schedule is a pure
  function of the fixture, partner map and the E write curve of the stage
  (not of the measured arm's state); re-tested bit-exact for Zgl2m and Zge2m
  with save/load at points that straddle the write-clock split of every
  stage on the reversed order.
* **Reproducibility:** rows, CSV and report do not depend on the worker thread
  count (tested); the main run is repeated with a different thread count and
  must give a byte-identical CSV (sha256 reported).

Statistics as #123–#184: bootstrap 95 % percentile CI (2000 resamples, seed
0xB0075EED); better/ties/worse over 40 seeds.

* **positive:** mean > 0 and CI lower bound > 0.
* **sequence rule:** Δ ACC strong (mean ≥ 0.05, > 0 in ≥ 32 of 40 seeds and
  CI lower bound > 0), Δ LAST ≥ −0.05 and Δ LEARN ≥ −0.05.

## Design gates (reported first)

* **G1 (integrity):** every row keeps its topology, starts from its documented
  initial weights, makes charged writes ≤ its allowance;
  `error-gated@paced100` is bit-identical to uncapped E with zero refusals;
  guard counters match guard and ledger; orders valid; codes valid (Zg1 = the
  blocks; share-2 codes as #136; all five share-2 arms have the cue lists of
  their partner map; the masks of every share-2 row are the pre-registered
  schedule of `F_j`); **frozen moves = 0 in every row**; frozen skips = 0 in
  every share-mode and uncued / private row; **write-clock splits are
  consistent** (every arm of a (k, flip, seed, order) carries the same
  `budget_split` list of the uncapped E curves of that order).
* **G2 (manipulation):** at k = 16 and k = 24, at flips 50 and 100
  (held-out rows): (a) Zgl2m has partner-line moves in the second write-half
  > 0 in ≥ 50 % of rows (late writes actually happen); (b) Zgl2m has frozen
  skips in the first write-half > 0 in ≥ 50 % of rows (the early freeze
  binds); (c) Zge2m has partner-line moves in the first write-half > 0 in
  ≥ 50 % of rows; (d) mean matched − random partner agreement (Zgl2m −
  Zgl2r, per seed) ≥ 0.20 at flip 100 and ≥ 0.02 at flip 50 (as #136 / #139
  / #184); (e) the mean write-clock split of every such Zgl2m row is
  strictly less than `n / 2` (the clock is earlier than the cycle 25
  episode mid-point).
* **G3 (novel pairs):** median CN pairs per (seed, task) at k = 24, flip 50
  ≥ 20.

If any gate fails, H1–H13 are declared **not evaluable**.

## Hypotheses (held-out orders, seeds 641..680, k = 16 unless stated)

* **H1 (primary):** flip 100, ACC_all(Zgl2m) − ACC_all(Zg2m) is positive.
* **H2 (primary):** flip 100, ACC_all(Zgl2m) − ACC_all(Zgo2m) is positive.
* **H3 (primary, learning side):** flip 100, LEARN_all(Zgl2m) −
  LEARN_all(Zg2m) is positive.
* **H4 (primary, forgetting side):** flip 100, FGT_all(Zgo2m) −
  FGT_all(Zgl2m) is positive (late partner writes keep part of the
  forgetting cut that ownership gave back).
* **H5 (timing):** flip 100, ACC_all(Zgl2m) − ACC_all(Zge2m) is positive.
* **H6 (vs control):** flip 100, Zgl2m vs S1 passes the sequence rule.
* **H7:** flip 50, ACC_all(Zgl2m) − ACC_all(Zg2m) is positive.
* **H8 (vs control):** flip 50, Zgl2m vs S1 passes the sequence rule.
* **H9:** k = 24, flip 100, ACC_all(Zgl2m) − ACC_all(Zg2m) is positive.
* **H10 (partner identity):** flip 100, ACC_all(Zgl2m) − ACC_all(Zgl2r) is
  positive.
* **H11:** flip 100, ACC_all(Zgl2m) − ACC_all(Zg1) is positive.
* **H12:** flip 50, ACC_all(Zgl2m) − ACC_all(Zgo2m) is positive.
* **H13 (generalisation to novel pairs):** flip 100, ACC_cn(Zgl2m) −
  ACC_cn(Zg2m) is positive.

No multiplicity correction (as #123–#184); every hypothesis is reported with
its mean, CI and better/ties/worse counts, pass or fail.

**Secondary (descriptive, no pass/fail):** flip 25; k = 8; identity order;
CN slices of every comparison; per-stage LEARN split by first stage /
partner trained before / partner not yet trained; worst held-out order;
budget (cue writes, cue-write share, write-clock split, E write fraction
per half, partner-line moves per write-half, skips per write-half); Zge2m
vs Zg2m / Zgo2m (does an early-only partner window on the write clock
reproduce the shared-write learning cost?).

## Pilot (label-free, before the main run)

`bench-cue-write-clock --census` on seeds 1..5, identity order, all k and
flips: partner agreement, writes, guard counters, frozen skips and
partner-line moves per write-half, write-clock split, E write fraction, cue
reach. No accuracy is computed and no held-out order is drawn. The pilot
can only lead to stopping the cycle before the main run (e.g. if a
manipulation cannot bind); it cannot change a threshold, an arm or a
hypothesis above.

## Pilot result (label-free, seeds 1..5, identity order; written before the main run)

File: `results/cue-write-clock-census-v1.txt` (no accuracy computed, no held-out
order drawn; sha256
`5f7b7db08852f32c04461f1bfcdaf200d4a9789e230721451cfcc61abfdb8cdd`).
Means over the 5 pilot seeds:

| k, flip | partner-line moves, write-half 1 / 2: Zg2m | Zge2m | Zgl2m | Zgl2m skips half 1 | write-clock split / n | cue writes Zg1 / Zg2m / Zgo2m / Zgl2m / Zge2m |
|---|---|---|---|---|---|---|
| 16, 100 | 43.4 / 39.8 | 46.4 / 0.0 | 0.0 / 40.4 | 66.4 | 5.8 / 256 | 305.8 / 491.8 / 309.8 / 404.6 / 392.2 |
| 16, 50 | 31.6 / 33.8 | 29.6 / 0.0 | 0.0 / 36.2 | 51.2 | 5.2 / 256 | 118.6 / 207.8 / 122.6 / 168.6 / 156.6 |
| 24, 100 | 51.4 / 40.0 | 54.6 / 0.0 | 0.0 / 42.2 | 77.2 | 8.4 / 256 | 431.8 / 766.8 / 434.8 / 612.8 / 569.4 |
| 24, 50 | 38.8 / 34.0 | 37.2 / 0.0 | 0.0 / 37.6 | 66.8 | 8.0 / 256 | 171.4 / 323.6 / 180.0 / 255.8 / 230.8 |

* The write clock sits at episode ≈ 5–8 of 256, far before the cycle 25
  episode mid-point (128). Under the shared write, partner-line edge changes
  are now split across the two write-halves (unlike #184, where almost every
  partner move fell in the first *episode*-half). The early window (Zge2m)
  reproduces the shared code's first-half partner moves and then stops; the
  late window (Zgl2m) has zero first-half partner moves, first-half skips in
  every run, and ≈ 36–42 partner-line edge changes in its second write-half
  (vs ≈ 1 under the episode clock of #184).
* Cue writes of Zgl2m sit between the owned and shared codes (e.g. 404.6
  at k = 16, flip 100, vs 309.8 owned and 491.8 shared): the late window
  now actually receives partner-block writes.
* Pilot fraction of identity-order Zgl2m runs with a second-write-half
  partner move (G2 (a)): **1.0** at (16, 50), (16, 100), (24, 50) and
  (24, 100). Skips (G2 (b)), early moves (G2 (c)) and the write clock
  earlier than `n / 2` (G2 (e)) bind in 5/5. Frozen moves are 0 in every
  row.
* **Expectation stated before the main run:** G2 is expected to pass, so
  H1–H13 should be evaluable. The late window is no longer empty, so the
  cycle can test whether *when* partner writes happen changes the #139
  trade-off; the pilot does not measure accuracy and does not change a
  threshold, arm or hypothesis. The main run is executed exactly as
  pre-registered above.

## Reproduce

```bash
cargo +1.89 run --release -- bench-cue-write-clock --census > results/cue-write-clock-census-v1.txt
cargo +1.89 run --release -- bench-cue-write-clock --csv results/cue-write-clock-v1.csv > results/cue-write-clock-report-v1.txt
```

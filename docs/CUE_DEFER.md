# B888-4 cycle 25: can the learning / forgetting trade-off of shared cue writes (#139) be split in time — the partner block frozen during the first half of a task's training and writable only in the second half — at equal lines and equal write budget? (synthetic graph)

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
[CUE_PARTNER.md](CUE_PARTNER.md) (#136) and the frozen-line mask of
[CUE_OWNERSHIP.md](CUE_OWNERSHIP.md) (#139). "Cue", "code", "block",
"partner", "phase", "late", "early", "frozen", "guard", "task order" and
"forgetting" name parts of an update rule and an artificial benchmark on a
Boolean graph with 4-bit weights. Nothing here is a claim about synapses,
biology, insect memory or BANC physiology.

Module: `src/cue_defer.rs` (engine support:
`consolidation_guard::run_setting_scheduled`). CLI: `bench-cue-defer`
(`--census` for the label-free pilot).

## Motivation (what #139 left open)

Cycle 24 (#139) froze the partner block of the matched share-2 code during
the whole training of each task (owned write, Zgo2m). That removed the
learning cost of sharing (H1 pass: LEARN Zgo2m − Zg1 −0.005 at k = 16, flip
100) but gave back more forgetting than it saved learning: against the
shared-write code Zg2m, Zgo2m learned +0.052 and forgot +0.079 more, so it
lost on accuracy (H3 / H5 / H11 fail). The report's reading: on this
benchmark the learning cost and the forgetting benefit of sharing come from
the **same** writes (a task writing its partner's block).

"Same writes" was measured over a whole stage. Within a stage, the guarded
error gate writes mostly while errors are frequent, i.e. early. If the
learning cost comes from partner-block writes competing with the
pattern-dependent writes **while the task is being learned**, and the
forgetting benefit comes from the partner block carrying the task's
cue-to-target direction **at the end of the stage**, then the two can be
separated in time: freeze the partner block during the first half of the
stage (learn as the owned code) and open it during the second half (leave
partner-block weights as the shared code would). The opposite schedule
(partner block writable first, frozen second) is the timing control: it has
the same number of partner-writable episodes. If the trade-off is about
**how many** partner writes happen, late and early should behave alike; if
it is about **when**, late should beat early.

This is a direct manipulation of one variable (the episode window in which
partner-block edges are writable), not a new code, a new rule or a new test
of tests.

## Write modes (pre-registered, no tuning)

All modes use the **same** cue lists as the #136 / #139 share-2 code
`C_j = sorted(B_j ∪ B_σ(j))` (same lines per task, same partner map, same
line multiplicity, same cue drive at evaluation). Let `F_j = B_σ(j) \ B_j`
(the #139 frozen lines) and let `n_j` be the number of training episodes of
task j (32 patterns × 8 rounds = 256 in the main design) with the split
`h_j = ⌊n_j / 2⌋` (episodes `0..h_j` = rounds 1–4, `h_j..n_j` = rounds 5–8).

* **share** (Zg2m, = #136 / #139 Zg2m): `F_j` writable in both halves.
* **owned** (Zgo2m, = #139 Zgo2m): `F_j` frozen in both halves.
* **late** (new, Zgl2m): `F_j` frozen during episodes `0..h_j`, writable
  during `h_j..n_j`.
* **early** (new, Zge2m, timing control): `F_j` writable during `0..h_j`,
  frozen during `h_j..n_j`.

"Frozen" is exactly the #139 mask (`consolidation_guard::step_masked`): the
lines stay active and drive the targets, their edges are not written, not
offered to the budget and not counted as a free edge by `last-resort`. The
schedule is a pure function of the fixture, the partner map and the episode
index, not engine state.

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
  `0x0DE5_B888_4C25_5A77`).

The matched partner map is the #136 best candidate tour
(`cue_partner::partner_map(Matched)`), as in #139.

**Equal budget:** every capped arm uses the `paced100` per-episode cumulative
write curve of uncapped E in the same (k, flip, seed, order); cue writes are
charged. **Equal resources:** all cued arms of a (k, seed) use the same
reserved lines and cue-line edges; the five share-2 arms also have the same
lines per task and line multiplicity. A scheduled run with the same mask in
both halves is bit-identical to `run_setting_masked` (tested), and with no
mask bit-identical to `run_setting` (tested).

## Design (pre-registered)

* Graph, task, pool, splits, budget: exactly #131–#139 (128 inputs / 32
  targets / fan-in 8, threshold 1, init weight 1, 4-bit weights, 6 active
  inputs, 32 training patterns per task, 8 rounds, one shared TEST set of 128
  patterns, pair classes seen / CN / ICN / UE, no rehearsal, original edges
  only, topology checked per row).
* **k = 8, 16, 24** (own fixtures). **Flips 25, 50, 100.** Flips 100 and 50
  carry the primary hypotheses; flip 25 is secondary only.
* **Seeds 601..640** (40 seeds, never used by an earlier B888-4 cycle).
* **Orders:** 0 = identity (secondary only); held-out 1 = reversed, 2..4 =
  seeded Fisher–Yates permutations (#133 generator) with a **new salt**
  `0x0DE5_B888_4C25_0A1D`, shared by every flip and arm.
* **Metrics** as #131–#139 (stage coordinates): ACC, LEARN, FGT, LAST on all /
  CN pairs; per seed the mean over its 4 held-out orders (n = 40 paired
  seeds). Worst order: per seed, min over held-out orders of the paired
  difference.
* **Instrumentation per row:** code, write mode, partner map, mean partner
  agreement, conflicting links, lines per task, writable partner lines per
  task in each half, mean cue reach, cue-line edges, live / full cue edges,
  cap, charged / effective / cue writes, cue-write share, **frozen skips per
  half** (edge steps the mask removed), **frozen moves** (edges of a frozen
  line whose weight changed during a half in which it was frozen; must be
  0), **partner-line moves per half** (edges of `F_j` whose weight changed
  during the first / second half of task j, summed over stages; recorded for
  every share-2 arm), refused, overwrites, deferred, net reversals,
  consolidated edges, saturated edges, init / paced / cap / code flags,
  weight digest, topology.
* **Checkpoint/resume:** the #132 engine checkpoint; the schedule is a pure
  function of the fixture, partner map and episode index (not state);
  re-tested bit-exact for Zgl2m and Zge2m with save/load at n/4, n/2, 3n/4
  and n of every stage on the reversed order.
* **Reproducibility:** rows, CSV and report do not depend on the worker thread
  count (tested); the main run is repeated with a different thread count and
  must give a byte-identical CSV (sha256 reported).

Statistics as #123–#139: bootstrap 95 % percentile CI (2000 resamples, seed
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
  every share-mode and uncued / private row.
* **G2 (manipulation):** at k = 16 and k = 24, at flips 50 and 100
  (held-out rows): (a) Zgl2m has partner-line moves in the second half > 0
  in ≥ 50 % of rows (late writes actually happen); (b) Zgl2m has frozen skips
  in the first half > 0 in ≥ 50 % of rows (the early freeze binds); (c) Zge2m
  has partner-line moves in the first half > 0 in ≥ 50 % of rows;
  (d) mean matched − random partner agreement (Zgl2m − Zgl2r, per seed) ≥
  0.20 at flip 100 and ≥ 0.02 at flip 50 (as #136 / #139).
* **G3 (novel pairs):** median CN pairs per (seed, task) at k = 24, flip 50
  ≥ 20.

If any gate fails, H1–H13 are declared **not evaluable**.

## Hypotheses (held-out orders, seeds 601..640, k = 16 unless stated)

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

No multiplicity correction (as #123–#139); every hypothesis is reported with
its mean, CI and better/ties/worse counts, pass or fail.

**Secondary (descriptive, no pass/fail):** flip 25; k = 8; identity order;
CN slices of every comparison; per-stage LEARN split by first stage /
partner trained before / partner not yet trained; worst held-out order;
budget (cue writes, cue-write share, partner-line moves per half, skips per
half); Zge2m vs Zg2m / Zgo2m (does an early-only partner window reproduce
the shared-write learning cost?).

## Pilot (label-free, before the main run)

`bench-cue-defer --census` on seeds 1..5, identity order, all k and flips:
partner agreement, writes, guard counters, frozen skips and partner-line
moves per half, cue reach. No accuracy is computed and no held-out order is
drawn. The pilot can only lead to stopping the cycle before the main run
(e.g. if a manipulation cannot bind); it cannot change a threshold, an arm
or a hypothesis above.

## Pilot result (label-free, seeds 1..5, identity order; written before the main run)

File: `results/cue-defer-census-v1.txt` (no accuracy computed, no held-out
order drawn). Means over the 5 pilot seeds:

| k, flip | partner-line moves, half 1 / half 2: Zg2m | Zge2m | Zgl2m | Zgl2m skips half 1 | cue writes Zg1 / Zg2m / Zgo2m / Zgl2m / Zge2m |
|---|---|---|---|---|---|
| 16, 100 | 61.2 / 0.0 | 61.2 / 0.0 | 0.0 / 0.6 | 438.4 | 305.8 / 491.8 / 309.8 / 309.6 / 491.8 |
| 16, 50 | 50.0 / 0.2 | 50.0 / 0.0 | 0.0 / 0.6 | 585.0 | 118.6 / 207.8 / 122.6 / 123.6 / 207.2 |
| 24, 100 | 60.4 / 1.8 | 59.4 / 0.0 | 0.0 / 1.6 | 318.4 | 431.8 / 766.8 / 434.8 / 437.8 / 765.2 |
| 24, 50 | 53.8 / 1.0 | 53.4 / 0.0 | 0.0 / 1.6 | 498.6 | 171.4 / 323.6 / 180.0 / 183.2 / 322.0 |

* Under the shared write, almost every partner-line edge change happens in
  the **first** half of a stage (the guarded error gate writes while errors
  are frequent). The early window (Zge2m) reproduces the shared code's
  partner moves and cue writes; the late window (Zgl2m) has the owned code's
  cue writes and ≈ 1 partner-line edge change per run in its second half.
* Pilot fraction of identity-order Zgl2m runs with a second-half partner
  move (G2 (a)): 0.6 / 0.4 / 0.8 / 0.6 at (16, 50) / (16, 100) / (24, 50) /
  (24, 100) — G2 (a) is close to its 0.5 threshold and may fail on the
  hypothesis seeds. Skips (G2 (b)) and early moves (G2 (c)) bind in 5/5.
* **Expectation stated before the main run:** if G2 passes, Zgl2m should
  behave like the owned code and Zge2m like the shared code, i.e. H1 / H3
  likely fail and H2 / H4 / H12 are expected to be small. The pilot does not
  show a manipulation that cannot bind (skips and early moves bind in every
  run), so the cycle is **not** stopped: the main run is executed exactly as
  pre-registered above (no threshold, arm or hypothesis changed) and its
  outcome, pass, fail or not evaluable, is reported.

## Reproduce

```bash
cargo +1.89 run --release -- bench-cue-defer --census > results/cue-defer-census-v1.txt
cargo +1.89 run --release -- bench-cue-defer --csv results/cue-defer-v1.csv > results/cue-defer-report-v1.txt
```

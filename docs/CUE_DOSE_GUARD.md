# B888-4 cycle 29: why does the dosed cue code lose to S1 on **raw** forgetting but win on **relative** forgetting, and does the raw gap close with **more sources** or a **changed write policy** (block guard), at equal write budget and without changing the linear dose effect? (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not built on the
build machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123), run through the **unchanged** cycle 27
engine of [CUE_WRITE_DOSE.md](CUE_WRITE_DOSE.md) (#186) plus five arms that
use the existing `block` consolidation guard of
[CONSOLIDATION_GUARD.md](CONSOLIDATION_GUARD.md) (#132). "Cue", "code",
"partner", "dose", "guard", "block", "retention" and "forgetting" name parts
of an update rule and an artificial benchmark on a Boolean graph with 4-bit
weights. Nothing here is a claim about synapses, biology, insect memory or
BANC physiology.

Module: `src/cue_dose_guard.rs`. CLI: `bench-cue-dose-guard` (`--census` for
the label-free pilot).

## 1. Where cycle 28 left off (#187, main `174164a2`)

At k = 16, flip 100 (held-out orders, seeds 721..760, synthetic):

* the dose effect generalised (slope of ACC_all +0.022 base, +0.022
  train64, +0.014 src256, all 40/0/0) and is linear: the two disjoint
  partner-line halves add up (interaction I ≈ 0 everywhere);
* **H5 failed**: raw forgetting FGT(S1) − FGT(Zgd4) = −0.107 [−0.113,
  −0.101] 0/0/40 (base); −0.112 train64, −0.115 src256;
* relative forgetting held (H6/H7): S1 loses ≈ 99 % of its above-chance
  learning and ends at chance (ACC_all 0.506–0.507), Zgd4 loses 74–85 %;
* **H8 failed** on src256 only: ΔACC Zgd4 − S1 = +0.0497 < 0.05;
* novel-pair accuracy does not rise with the dose (the advantage over S1
  there comes from the cue code itself).

## 2. The question, and an exact decomposition (stated before measuring)

FGT_all is the mean over the first k − 1 tasks of R[j][j] − R[k−1][j].
Write **LEARN_head** = mean_{j<k−1} R[j][j] (cycle 28) and **RET_head** =
mean_{j<k−1} R[k−1][j] (new name, existing quantity: the final accuracy on
those same tasks, ACC_all without the last task). Then, exactly,

  FGT = LEARN_head − RET_head, and
  FGT(S1) − FGT(x) = [RET_head(x) − RET_head(S1)] − [LEARN_head(x) − LEARN_head(S1)].

So a dosed arm can lose to S1 on raw forgetting **while retaining more**
at the end of the sequence, if it learns much more than it additionally
retains. Relative forgetting RF = FGT / (LEARN_head − 0.5) divides by
learning, so it can rank the arms the other way. The pre-registered
reading of "why" is therefore: **H1** (Zgd4 retains more than S1) together
with **H2** (the raw gap replicates, i.e. the learning term is larger).

Two ways to close the raw gap are then tested, both at the same
per-stage `paced100` write caps:

1. **More sources** (128 → 256 → 512 graph sources, everything else as
   cycle 27). Cycle 28 saw −0.107 (128) and −0.115 (256): the prediction
   here is that the gap does **not** close (H3 expected to fail; it is
   registered as the plain test of the question).
2. **A changed write policy**: the same matched share-2 code and the same
   constant dose masks, but the `block` consolidation guard (every
   weight step against an edge's consolidated sign is refused) instead
   of `last-resort` (one such overwrite per episode and target allowed
   when no free edge exists). New setting
   `E/E-cue0+block@paced100` (**Zb**), arms **Zbd0..Zbd4**. Motivation:
   in cycle 19 (#132) the uncued `block` guard forgot about as little as
   S1 (FGT 0.136 vs 0.105 at k 8, flip 50) but also learned only as much
   (LEARN 0.628). With a cue code, a later task can still learn on its
   own cue lines (and its writable partner lines) when shared feature
   edges are blocked, so the guard may cut forgetting **by retention**
   rather than only by learning less. Whether it does is H4–H8, H13.

Finally, the linear dose effect must not be changed by any of this: the
last-resort family runs on the unchanged engine (H9, H10), and the dose
effect is also tested inside the block family (H11, H12).

## 3. Arms, conditions, seeds, orders

* **Last-resort family (unchanged cycle 27 engine,
  `cue_write_dose::run_seed`, bit-for-bit):** S1, R1, E, Zg1, Zgd0..Zgd4,
  Zgd2c (same matched share-2 code, constant whole-stage dose masks, dose
  salt `0x0DE5_B888_4C27_D05E`, same guard, cue init 0).
* **Block family (new):** Zbd0..Zbd4 = `E/E-cue0+block@paced100` on the
  **same** matched share-2 code fixture (`cue_write_dose::arm_fixture`),
  the **same** dose masks and watched lines, and the **same** per-stage
  `paced100` caps (the cumulative write curve of uncapped error-gated E of
  the same (seed, order); recomputed in the block run and checked equal to
  the last-resort rows by gate G1). Refused (blocked) steps cost no write,
  as in #132: the **cap** is equal, the number of writes actually used may
  be lower and is reported.
* `Setting::Zb` is a new variant of `consolidation_guard::Setting` (base
  rule and initial weights of `E/E-cue0`, guard `block`). It is **not**
  added to `consolidation_guard::SETTINGS`, so every earlier grid is
  unchanged (tested: the full test suite, which pins earlier rows, passes;
  and the last-resort rows here equal `cue_write_dose::run_seed`).
* **Conditions** (one design field changed: `shape.sources`): **base**
  (128, the cycle 27 design), **src256**, **src512**; targets 32, fan-in 8,
  32 training patterns per task, 128 TEST patterns unchanged. `|F_j|` = 4
  and the fixture is leakage-free on seed 1 in every condition (checked by
  a unit test before this document was written: fixture construction only,
  no learning, no accuracy).
* **k = 16, flip 100 only** (the cycles 27–28 primary cell).
* **Seeds 761..800** (40 seeds, never used by an earlier B888-4 cycle).
  Pilot seeds 1..5.
* **Orders:** 0 = identity (secondary only), held-out 1 = reversed, 2..4 =
  seeded Fisher–Yates permutations with the cycle 27 order salt
  `0x0DE5_B888_4C27_0A1D`. **Declared deviation (as cycle 28):** the salt
  is reused so that the last-resort family is bit-for-bit the cycle 27
  engine; orders are new because they are keyed by the new seeds.
* **Dropped vs cycle 28 (declared):** the train64 condition (it answered a
  different question, generalisation over training-set size) and the
  interaction hypotheses (H11/H12 of cycle 28 passed in every condition).
  The interaction is not recomputed for the block family.

## 4. Metrics and statistics

* As #131–#187 on the shared TEST set: ACC, LEARN, FGT, LAST on all / CN
  pairs (balanced accuracy); per seed the mean over its 4 held-out orders
  (n = 40 paired seeds). LEARN_head and RF exactly as cycle 28 (RF defined
  when LEARN_head − 0.5 ≥ 0.05). RET_head as §2.
* **Dose slope / concavity** exactly as cycle 27 (least-squares slope on
  q ∈ {0, ¼, ½, ¾, 1}; f(½) − (f(0) + f(1))/2), computed within one family.
* **Gap change across conditions (H3):** per seed index s,
  gap_src512(s) − gap_base(s) with gap = FGT(S1) − FGT(Zgd4). The two
  conditions are different fixtures; pairing by seed index is a fixed
  procedure, not a claim that the fixtures are matched.
* **Statistics** as #123–#187: bootstrap 95 % percentile CI (2000
  resamples, seed 0xB0075EED); better/ties/worse over 40 seeds.
  * positive: mean > 0 and CI lower bound > 0;
  * equivalent: the 95 % CI lies strictly inside ±0.01;
  * non-inferior: CI lower bound > −0.02;
  * gap replicates (H2): CI upper bound < −0.02.
* **Reproducibility:** the main run is repeated with a different thread
  count and must give a byte-identical CSV (sha256 reported).

## 5. Design gates (per condition, reported first)

* **G1 (integrity):** the cycle 27 G1 unchanged on the last-resort family;
  on the block family: topology unchanged, initial weights, charged writes
  ≤ cap, codes valid, frozen moves = 0, writable lines = dose, no skip at
  full dose, no partner move at dose 0, guard counters consistent (no
  effective overwrite, no deferral, effective writes = ledger writes), and
  the same order, uncapped stage totals, caps, partner map, partner lines
  and cue edges as the last-resort rows of the same (seed, order).
* **G2a (last-resort dose binds):** the cycle 27 G2 clauses (a)–(d).
* **G2b (block binds):** every Zbd arm refuses at least one overwrite in
  ≥ 50 % of its held-out rows.
* **G2c (block dose binds):** mean partner-line moves strictly increase
  along Zbd0..Zbd4.
* **G3 (novel pairs):** median CN pairs per (seed, task) ≥ 20 (flip 100).
* **G4 (leakage):** for every seed, no TEST pattern equals any training
  pattern of any task; all arms are scored on the same TEST set.
* **G5 (RF defined):** for every seed, for S1, Zgd4 and Zbd4.

If G1, G2a, G2b, G3 or G4 fails in a condition, every hypothesis using it
is **not evaluable**. G2c failing makes only H11/H12 not evaluable; G5
failing makes only H13 not evaluable.

## 6. Hypotheses (k = 16, flip 100, held-out orders, seeds 761..800)

Why raw forgetting loses while relative forgetting wins:

* **H1 (retention):** base, RET_head(Zgd4) − RET_head(S1) positive.
* **H2 (raw gap replicates):** base, FGT(S1) − FGT(Zgd4) has CI upper
  bound < −0.02.

More sources:

* **H3:** gap(src512) − gap(base) positive (the raw gap closes with more
  sources). Expected to fail (§2).

Changed write policy (**primary: H4 ∧ H5**):

* **H4 (raw forgetting, non-inferiority):** base, FGT(S1) − FGT(Zbd4)
  non-inferior (CI lower bound > −0.02).
* **H5 (still beats S1):** base, ACC_all(Zbd4) − ACC_all(S1) positive.
* **H6 (generalises over sources):** H4 and H5 both hold in src256 **and**
  src512.
* **H7 (by retention):** base, RET_head(Zbd4) − RET_head(Zgd4) positive.
* **H8 (no accuracy cost vs last-resort):** base, ACC_all(Zbd4) −
  ACC_all(Zgd4) non-inferior (CI lower bound > −0.02).

The linear dose effect is unchanged:

* **H9:** Zgd0..4 dose slope of ACC_all positive in **each** of base,
  src256 and src512.
* **H10:** base, Zgd0..4 concavity of ACC_all equivalent to 0.
* **H11:** base, Zbd0..4 dose slope of ACC_all positive.
* **H12:** base, Zbd0..4 concavity of ACC_all equivalent to 0.

Relative forgetting under the new policy:

* **H13:** base, RF(S1) − RF(Zbd4) positive.

No multiplicity correction (as #123–#187); every hypothesis is reported
with its mean, CI and better/ties/worse counts, pass or fail. A pass of
H4 with a fail of H5 or H8 would mean the gap closes only by learning
less; it would be reported as such, not as an improvement.

**Secondary (descriptive, no pass/fail):** per condition the exact
decomposition (ΔLEARN_head, ΔRET_head, raw gap) of Zgd0, Zgd4, Zbd0, Zbd4
vs S1; Zbd4 vs S1 and vs Zgd4 on ACC, LEARN, RET_head, ACC_cn, RF; dose
slopes of FGT_all and ACC_cn within the block family; gap(src256) −
gap(base); per-arm summaries (ACC, ACC_cn, LEARN, LEARN_head, RET_head,
FGT, LAST, RF, writes, cue writes, blocked overwrites, partner-line moves,
skips).

## 7. Pilot (label-free, before the main run)

`bench-cue-dose-guard --census` on seeds 1..5, identity order, all three
conditions: training only (no evaluation, no accuracy, no held-out order):
writes, cue writes, blocked overwrites, skips, frozen and partner-line
moves, refusals, writable lines, and the G4 leakage check. The pilot can
only lead to stopping the cycle before the main run (e.g. if the block
guard never binds or a dose cannot bind); it cannot change a threshold, an
arm, a condition, a seed or a hypothesis above.

## Reproduce

```bash
cargo +1.89 run --release -- bench-cue-dose-guard --census > results/cue-dose-guard-census-v1.txt
cargo +1.89 run --release -- bench-cue-dose-guard --csv results/cue-dose-guard-v1.csv > results/cue-dose-guard-report-v1.txt
```

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

## 8. Pilot result (label-free, seeds 1..5, identity order; written before the main run)

File: `results/cue-dose-guard-census-v1.txt` (training only: no accuracy
computed, no held-out order drawn; sha256
`19c2257f53e99b19455e00cbe31993194a5533061957ee1ea43931c03cfd2111`). Means over the 5 pilot seeds, k = 16, flip 100:

| condition | charged writes E / Zgd4 / Zbd0 / Zbd4 | cue writes Zgd0 / Zgd4 / Zbd0 / Zbd4 | blocked Zbd0 / Zbd4 | partner-line moves Zgd0..4 | partner-line moves Zbd0..4 | frozen moves | leak-free fixtures |
|---|---|---|---|---|---|---|---|
| base | 914.8 / 914.8 / 880.2 / 908.0 | 309.8 / 491.8 / 811.2 / 842.2 | 12 046 / 10 337 | 0 / 17.2 / 32.8 / 47.2 / 61.2 | 0 / 17.6 / 34.0 / 48.8 / 61.8 | 0 | 5/5 |
| src256 | 1217.2 / 1217.2 / 1160.4 / 1211.2 | 317.8 / 531.4 / 1035.6 / 1097.6 | 10 184 / 9 004 | 0 / 14.2 / 26.8 / 39.2 / 55.0 | 0 / 19.2 / 34.4 / 50.6 / 62.8 | 0 | 5/5 |
| src512 | 1309.4 / 1309.4 / 1069.2 / 1219.2 | 252.4 / 457.0 / 908.0 / 1065.2 | 9 879 / 8 974 | 0 / 12.2 / 24.4 / 35.8 / 49.0 | 0 / 16.6 / 33.4 / 47.2 / 59.4 | 0 | 5/5 |

* The base last-resort rows reproduce the cycle 27 and cycle 28 pilots
  exactly (cue writes 309.8 / 491.8, partner-line moves 17.2 / 32.8 / 47.2
  / 61.2): the engine is unchanged.
* The block guard binds in every condition and at every dose (about
  9 000–12 000 refused overwrites per run), and the block arms use 82–99 %
  of the same caps. Their writes move onto cue lines: cue writes are
  about 1.7–3.6 × those of the last-resort arm at the same dose.
* Partner-line moves rise strictly with the dose in the block family
  too (G2c is expected to pass), frozen moves are 0 and every fixture is
  leakage-free.
* **Expectation stated before the main run:** G1–G4 are expected to pass
  in every condition and H1–H12 to be evaluable (H13 needs G5, which
  depends on how much Zbd4 learns and cannot be seen in a label-free
  pilot). The pilot measured no accuracy and changed no threshold, arm,
  condition, seed or hypothesis. The main run is executed exactly as
  pre-registered above.

## 9. Main run result (seeds 761..800, held-out orders; synthetic graph)

Files: `results/cue-dose-guard-report-v1.txt`, `results/cue-dose-guard-v1.csv`
(9 000 rows = 3 conditions × 40 seeds × 5 orders × 15 arms; sha256
`1cae8f9c0d2b1c63cbce4a34c9b933fd35e22658057ff907317a28628fe1d4f1`, report
sha256 `234da46d4448f5a5a190562ee379823a56ca727dd3159f1338baa4e47fa48d0b`).
The run was executed exactly as pre-registered above, from the binary of
the pilot commit; no threshold, arm, condition, seed or hypothesis was
changed after the pilot (wall time 48 min 56 s).

**Deviations:** none from the hypotheses, thresholds, arms, conditions or
seeds. The declared design choices of §3 (order salt reused, train64 and
the interaction hypotheses dropped) were stated before measuring.
**One pre-registered check is not done:** the §4 reproducibility rerun
with a different thread count (`--threads 6`) was started but interrupted
before it wrote its CSV, and it was not restarted in this cycle. The
byte-identical-CSV claim is therefore **not verified for cycle 29**; only
the sha256 of the single main run is reported above. (The same check
passed in cycles 27 and 28 on the unchanged cycle 27 engine; that is not
evidence for the new block arms.) The rerun stays open for a later cycle;
no verdict above depends on it.

### Design gates

| condition | G1 | G2a partner moves Zgd0..4 | G2b blocked rows Zbd0..4 | G2c partner moves Zbd0..4 | G3 median CN | G4 leak-free | G5 |
|---|---|---|---|---|---|---|---|
| base | pass | 0 / 9.7 / 19.3 / 29.0 / 37.0 | 100 % each | 0 / 9.7 / 19.1 / 27.8 / 35.7 | 65 | 40/40 | pass |
| src256 | pass | 0 / 10.6 / 20.5 / 30.4 / 40.3 | 100 % each | 0 / 11.9 / 22.4 / 32.6 / 41.8 | 64 | 40/40 | pass |
| src512 | pass | 0 / 9.2 / 17.8 / 26.6 / 35.3 | 100 % each | 0 / 10.5 / 20.7 / 29.6 / 37.8 | 56 | 40/40 | pass |

Every gate passes in every condition; all 13 hypotheses are evaluable.

### Arms (base, held-out orders, mean over rows)

| arm | ACC_all | LEARN_head | RET_head | FGT_all | RF | charged writes | cue writes | blocked |
|---|---|---|---|---|---|---|---|---|
| S1 | 0.507 | 0.635 | 0.501 | 0.133 | 0.991 | 644.1 | 0 | – |
| Zgd0 | 0.578 | 0.872 | 0.561 | 0.311 | 0.838 | 639.4 | 229.3 | – |
| Zgd4 | 0.600 | 0.823 | 0.587 | 0.236 | 0.732 | 643.5 | 335.7 | – |
| Zbd0 | 0.580 | 0.624 | 0.578 | 0.045 | 0.372 | 603.4 | 515.9 | 11 165 |
| Zbd4 | 0.600 | 0.646 | 0.599 | 0.047 | 0.330 | 628.2 | 553.8 | 10 423 |

### Hypotheses (mean [95 % CI] better/ties/worse over 40 seeds)

| H | contrast | value | verdict |
|---|---|---|---|
| H1 | base RET_head(Zgd4) − RET_head(S1) | +0.086 [+0.079, +0.092] 40/0/0 | pass |
| H2 | base FGT(S1) − FGT(Zgd4), CI hi < −0.02 | −0.103 [−0.110, −0.096] 0/0/40 | pass |
| H3 | gap(src512) − gap(base) > 0 | −0.020 [−0.027, −0.013] 8/0/32 | **fail** (as predicted) |
| H4 | base FGT(S1) − FGT(Zbd4) non-inferior | +0.086 [+0.081, +0.090] 40/0/0 | pass |
| H5 | base ACC_all(Zbd4) − ACC_all(S1) | +0.094 [+0.085, +0.102] 40/0/0 | pass |
| H6 | H4 ∧ H5 in src256 and src512 | FGT +0.075 / +0.064, ACC +0.051 / +0.028, all 40/0/0 | pass |
| H7 | base RET_head(Zbd4) − RET_head(Zgd4) | +0.012 [+0.008, +0.015] 31/0/9 | pass |
| H8 | base ACC_all(Zbd4) − ACC_all(Zgd4) non-inferior | −0.000 [−0.004, +0.003] 20/0/20 | pass |
| H9 | Zgd0..4 slope ACC_all, each condition | +0.023 / +0.012 / +0.007 (40/0/0, 36/0/4, 37/0/3) | pass |
| H10 | base Zgd concavity ACC_all ≡ 0 | +0.000 [−0.001, +0.002] | pass |
| H11 | base Zbd0..4 slope ACC_all | +0.021 [+0.018, +0.023] 40/0/0 | pass |
| H12 | base Zbd concavity ACC_all ≡ 0 | −0.001 [−0.002, +0.001] | pass |
| H13 | base RF(S1) − RF(Zbd4) | +0.661 [+0.631, +0.690] 40/0/0 | pass |

### Reading (descriptive)

* **Why raw forgetting loses while relative forgetting wins.** The exact
  split FGT(S1) − FGT(x) = ΔRET_head − ΔLEARN_head answers it. At the end
  of the sequence Zgd4 retains **more** of the old tasks than S1 (+0.086,
  40/0/0), but it learned far more (+0.189), so its raw drop is larger
  (−0.103). S1 forgets little in raw terms because it learns little and
  ends at chance (RET_head 0.501). Raw FGT rewards not learning; RF and
  RET_head do not.
* **More sources do not close the raw gap; it widens:** −0.103 (128),
  −0.117 (256), −0.123 (512). With more sources the retention advantage
  of Zgd4 over S1 shrinks faster (+0.086 → +0.042 → +0.023) than its
  learning advantage (+0.189 → +0.159 → +0.146).
* **The block guard closes the raw gap, mostly by learning less.** Zbd4
  forgets 0.086 *less* than S1 (base; +0.075 src256, +0.064 src512, all
  40/0/0) and still beats S1 on ACC_all (+0.094 / +0.051 / +0.028). Its
  ACC_all is the same as Zgd4's (−0.000 base, +0.001 src256, −0.004
  [−0.006, −0.001] src512). The decomposition shows how: Zbd4 learns
  0.177 less than Zgd4 (LEARN_head 0.646, close to S1's 0.635) and
  retains a little more (+0.012 base, +0.014 src256, +0.009 src512). It
  converts learning that would have been overwritten into a flat profile:
  it learns each task only a little but keeps almost all of it (RF 0.33
  vs 0.73). **This is a different trade-off at the same accuracy, not an
  accuracy improvement over last-resort.** H4 passing is mostly a
  learning effect, and H7 shows a small real retention gain.
* **The linear dose effect is unchanged.** The last-resort dose slope
  replicates in every condition (smaller with more sources: +0.023,
  +0.012, +0.007), with no curvature. Under the block guard the dose
  still raises ACC_all linearly (+0.021, concavity ≈ 0). But it no longer
  cuts forgetting (−slope FGT −0.002 [−0.004, −0.000]), because FGT is
  already ≈ 0.047 at every dose. The dose then acts through retention.
* Under the block guard the novel-pair gain over S1 shrinks with sources
  (ACC_cn +0.065 base, +0.027 src256, +0.007 [−0.003, +0.017] src512) and
  is below the last-resort arms' (base Zgd4 ACC_cn 0.628 vs Zbd4 0.578).
* Absolute effects are small, and every number comes from a generated
  graph with artificial labels. No claim about synapses, biology, insect
  memory or BANC physiology is made.

## Thread-count rerun (done in cycle 30)

The rerun not done in this cycle was done for the `base` condition in
cycle 30 ([CUE_DOSE_DENSITY.md](CUE_DOSE_DENSITY.md)): `--conds base
--threads 6` reproduces the header and `base` rows of
`results/cue-dose-guard-v1.csv` byte for byte (sha256 `761eadf9…`).
src256 and src512 were not re-run.

## Reproduce

```bash
cargo +1.89 run --release -- bench-cue-dose-guard --census > results/cue-dose-guard-census-v1.txt
cargo +1.89 run --release -- bench-cue-dose-guard --csv results/cue-dose-guard-v1.csv > results/cue-dose-guard-report-v1.txt
```

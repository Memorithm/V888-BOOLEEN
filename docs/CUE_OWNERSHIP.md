# B888-4 cycle 24: does the learning cost of sharing (#136 H7) disappear if each task may only write its own block of a shared cue code, at equal lines and equal write budget? (synthetic graph)

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
matrix of [CUE_SIMILARITY.md](CUE_SIMILARITY.md) (#135) and the partner tours
of [CUE_PARTNER.md](CUE_PARTNER.md) (#136). "Cue", "code", "block",
"partner", "ownership", "frozen", "guard", "task order" and "forgetting" name
parts of an update rule and an artificial benchmark on a Boolean graph with
4-bit weights. Nothing here is a claim about synapses, biology, insect memory
or BANC physiology.

Module: `src/cue_ownership.rs`. CLI: `bench-cue-ownership` (`--census` for
the pilot).

## Motivation (what #136 left open)

Cycle 23 (#136) found that sharing reserved cue lines with the most
agreeing partner task (matched share-2 code, Zg2m) beats private cues (Zg1),
the cyclic and random codes and the supervised control at equal lines and
equal write budget, but its H7 failed: at k = 16, flip 100 Zg2m forgets less
than Zg1 (FGT_all 0.229 vs 0.334) **and learns less** (LEARN_all 0.817 vs
0.872, −0.055 in 40/40 seeds).

An **exploratory** re-analysis of the published #136 CSV
(`results/cue-partner-v1.csv`, seeds 521..560, already used; nothing below
is a test) locates that learning cost:

* It is **not** present at the first stage: the first task of every held-out
  order is learned to 0.991 by Zg1 and by every share-2 code. It appears from
  the second stage on (k = 16, flip 100, per-stage LEARN Zg2m − Zg1 by order
  quartile: −0.045 / −0.059 / −0.056 / −0.061).
* It does **not** depend on whether the partners that share the task's lines
  were trained before it (k = 16, flip 100: −0.056 if neither its successor
  nor its predecessor in the tour was trained before, −0.062 if both were;
  same pattern at flip 50 and k = 24). So it is not mainly interference from a
  partner's earlier cue weights.
* It goes with **where the write budget goes**: at equal effective writes
  (≈ 632 per run), Zg2m spends 336 writes on cue-line edges (53 %), Zg1 206
  (33 %), the unguarded private code Z1 103 (16 %). Every share-2 code
  (cyclic, matched, anti, random) shows the same LEARN drop (0.80–0.82), with
  the same 330–342 cue writes. A cue line reaches the same target for every
  pattern, so writes on cue edges cannot encode the pattern-dependent part of
  the task; with twice the writable lines per task, the guarded error gate
  (whose `last-resort` rule prefers any free edge, cue edges included, over
  overwriting a consolidated pattern edge) diverts more of the same budget
  into them.

This cycle tests that reading with a direct manipulation: keep the matched
share-2 code (same lines per task, same partner map, same cue drive at
evaluation) but let each task **write only its own block**; the partner's
block is read (active input, its edges drive the targets) but frozen during
that task.

## Codes (pre-registered construction, no tuning)

* **Blocks** `B_0 … B_{k−1}`, agreement matrix `a(i, j)`, matched / cyclic
  partner maps: exactly #136 (`cue_partner`). The random partner map is the
  #136 Sattolo cycle with a **new salt** (`0x0DE5_B888_4C24_5A77`).
* **Shared write** (the #136 code, `share`): code of task j
  `C_j = sorted(B_j ∪ B_σ(j))`; during task j every line of `C_j` is active
  and writable.
* **Owned write** (new, `owned`): the **same** `C_j` (identical cue lists, so
  identical lines per task, identical line multiplicity and identical cue
  drive at evaluation), but during task j the edges of the lines of
  `B_σ(j) \ B_j` are **frozen**: the update rule skips them (no write, not
  offered to the budget, and not counted as a "free" edge by the
  `last-resort` guard). The lines stay active, so their current weights (set
  by task σ(j) when it is trained) still drive the targets. A cue-line edge of
  block `B_j` is therefore only ever written while task j is trained, as in
  the private code; the owned code is "private writes + read-only access to
  the partner's block".
* Cue-line edges start at 0 in every cued arm (silent cue, #128); until task
  σ(j) has been trained, its block contributes no drive to task j, so the
  owned code equals the private code on the stages before the partner is
  trained (tested at the first stage).

## Arms (10 rows per (k, flip, seed, order))

* `supervised@paced100` (**S1**, control; on this teacher-clamped task the
  associative co-activation rule is the same update, see CONSOLIDATION.md).
* `reward@paced100` (**R1**, reward-modulated).
* `error-gated@free` (**E**, = its `paced100` run, gated in G1).
* `E/E-cue0@paced100` private blocks (**Z1**) and
  `E/E-cue0+last-resort@paced100` private blocks (**Zg1**).
* `E/E-cue0+last-resort@paced100` with the matched shared-write code
  (**Zg2m**, = #136 Zg2m).
* `E/E-cue0+last-resort@paced100` with the owned-write matched (**Zgo2m**),
  cyclic (**Zgo2c**) and random (**Zgo2r**) codes.
* `E/E-cue0@paced100` with the owned-write matched code (**Zo2m**, no guard).

All rules are the `consolidation_guard` code paths of #132–#136; the owned
arms add only the per-task frozen-line mask (`run_setting_masked` with an
empty mask is bit-identical to `run_setting`, tested).

**Equal budget:** every capped arm uses the `paced100` per-episode cumulative
write curve of uncapped E in the same (k, flip, seed, order); cue writes are
charged. **Equal resources:** all cued arms of a (k, seed) use the same
reserved lines and the same cue-line edges; the share-2 arms (Zg2m and the
three owned arms) also have the same number of lines per task and the same
line multiplicity.

## Design (pre-registered)

* Graph, task, pool, splits, budget: exactly #131–#136 (128 inputs / 32
  targets / fan-in 8, threshold 1, init weight 1, 4-bit weights, 6 active
  inputs, 32 training patterns per task, 8 rounds, one shared TEST set of 128
  patterns, pair classes seen / CN / ICN / UE, no rehearsal, original edges
  only, topology checked per row).
* **k = 8, 16, 24** (own fixtures). **Flips 25, 50, 100.** Flips 100 and 50
  carry the primary hypotheses; flip 25 is secondary only.
* **Seeds 561..600** (40 seeds, never used by an earlier B888-4 cycle).
* **Orders:** 0 = identity (secondary only); held-out 1 = reversed, 2..4 =
  seeded Fisher–Yates permutations (#133 generator) with a **new salt**
  `0x0DE5_B888_4C24_0A1D`, shared by every flip and arm. Partner maps do not
  depend on the order.
* **Metrics** as #131–#136 (stage coordinates): ACC, LEARN, FGT, LAST on all /
  CN pairs; per seed the mean over its 4 held-out orders (n = 40 paired
  seeds). Worst order: per seed, min over held-out orders of the paired
  difference.
* **Instrumentation per row:** code kind, write mode (share / owned), partner
  map, mean partner agreement, conflicting links, lines per task, writable
  lines per task, mean cue reach, cue-line edges, live / full cue edges, cap,
  charged / effective / cue writes, cue-write share of effective writes,
  **frozen skips** (edge steps the mask removed), **frozen moves** (frozen
  edges whose weight changed during a stage in which they were frozen; must
  be 0), refused, overwrites, deferred, net reversals, consolidated edges,
  saturated edges, init / paced / cap / code flags, weight digest, topology.
* **Checkpoint/resume:** the #132 engine checkpoint plus the mask (a pure
  function of the fixture and the partner map, not state), re-tested
  bit-exact for Zgo2m and Zo2m mid-stage on the reversed order.
* **Reproducibility:** rows, CSV and report do not depend on the worker thread
  count (tested); the main run is repeated with a different thread count and
  must give a byte-identical CSV (sha256 reported).

Statistics as #123–#136: bootstrap 95 % percentile CI (2000 resamples, seed
0xB0075EED); better/ties/worse over 40 seeds.

* **positive:** mean > 0 and CI lower bound > 0.
* **sequence rule:** Δ ACC strong (mean ≥ 0.05, > 0 in ≥ 32 of 40 seeds and
  CI lower bound > 0), Δ LAST ≥ −0.05 and Δ LEARN ≥ −0.05.

## Design gates (reported first)

* **G1 (integrity):** every row keeps its topology, starts from its documented
  initial weights, makes charged writes ≤ its allowance; `error-gated@paced100`
  is bit-identical to uncapped E with zero refusals; guard counters match
  guard and ledger; orders valid; codes valid (Z1/Zg1 = the blocks; share-2
  codes as #136; Zgo2m / Zo2m cue lists identical to Zg2m's; owned masks =
  `B_σ(j) \ B_j` for every task); **frozen moves = 0 in every owned row** and
  frozen skips = 0 in every non-owned row.
* **G2 (manipulation):** at k = 16 and k = 24, at flips 50 and 100: (a) mean
  cue writes of Zgo2m ≤ 0.80 × mean cue writes of Zg2m (held-out rows; the
  manipulation must actually move budget away from cue edges); (b) Zgo2m has
  frozen skips > 0 in ≥ 50 % of held-out rows (the mask binds); (c) matched ≥
  cyclic partner agreement in every seed, and mean matched − random ≥ 0.20 at
  flip 100 and ≥ 0.02 at flip 50 (as #136).
* **G3 (novel pairs):** median CN pairs per (seed, task) at k = 24, flip 50
  ≥ 20.

If any gate fails, H1–H13 are declared **not evaluable**.

## Hypotheses (held-out orders, seeds 561..600, k = 16 unless stated)

* **H1 (primary, retest of #136 H7 on the owned code):** flip 100,
  FGT_all(Zg1) − FGT_all(Zgo2m) is positive **and** LEARN_all(Zgo2m) −
  LEARN_all(Zg1) has CI lower bound > −0.02.
* **H2 (primary, mechanism):** flip 100, LEARN_all(Zgo2m) − LEARN_all(Zg2m)
  is positive.
* **H3 (primary):** flip 100, ACC_all(Zgo2m) − ACC_all(Zg2m) is positive.
* **H4 (primary, vs control):** flip 100, Zgo2m vs S1 passes the sequence
  rule on all pairs.
* **H5:** flip 50, ACC_all(Zgo2m) − ACC_all(Zg2m) is positive.
* **H6 (vs control):** flip 50, Zgo2m vs S1 passes the sequence rule on all
  pairs.
* **H7:** flip 100, ACC_all(Zgo2m) − ACC_all(Zg1) is positive.
* **H8:** flip 50, ACC_all(Zgo2m) − ACC_all(Zg1) is positive.
* **H9 (partner still matters when read-only):** flip 100, ACC_all(Zgo2m) −
  ACC_all(Zgo2c) is positive.
* **H10 (random-partner control):** flip 100, ACC_all(Zgo2m) −
  ACC_all(Zgo2r) is positive.
* **H11 (k = 24):** flip 100, ACC_all(Zgo2m) − ACC_all(Zg2m) is positive.
* **H12 (not guard-specific):** flip 100, ACC_all(Zo2m) − ACC_all(Z1) is
  positive.
* **H13 (H1 at flip 50):** flip 50, FGT_all(Zg1) − FGT_all(Zgo2m) is positive
  **and** LEARN_all(Zgo2m) − LEARN_all(Zg1) has CI lower bound > −0.02.

**Secondary (reported, no pass/fail):** all flip 25 comparisons; all
comparisons at k = 8; k = 24 at flip 50; identity order; CN pairs; every arm
vs S1 / R1 / E per (k, flip); worst held-out order of Zgo2m − Zg2m and
Zgo2m − Zg1; per-stage LEARN of Zgo2m − Zg1 at the first stage and by
whether the partner was trained before; cue writes, cue-write share, frozen
skips, deferred and overwrites per arm.

No multiplicity correction is applied. The analysis is run once. Nothing
above the Results heading is changed after the main run except the "pending"
placeholder; the design-stage pilot section below is filled in before the
main run from the label-free census only (seeds 1..5: agreement, writes,
cue writes, frozen skips, guard counters, reach; **no accuracy**).

## Design-stage pilot

Pending (census on seeds 1..5, filled in before the main run).

## Command

```bash
cargo +1.89 run --release -- bench-cue-ownership --csv results/cue-ownership-v1.csv > results/cue-ownership-report-v1.txt
# pilot: cargo +1.89 run --release -- bench-cue-ownership --census > results/cue-ownership-census-v1.txt
```

## Results

Pending (main run after the pre-registration, harness and pilot commits).

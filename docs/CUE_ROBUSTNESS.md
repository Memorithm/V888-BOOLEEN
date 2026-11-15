# B888-4 cycle 20: does the silent cue + last-resort win hold on fresh seeds, more held-out task orders and larger k, at equal write budget against the supervised control? (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123), with the fixed pattern pool of
[CUE_REACH.md](CUE_REACH.md) (#124), the write budgets of
[WRITE_BUDGET.md](WRITE_BUDGET.md) (#126), the silent cue of
[CUE_ZERO_INIT.md](CUE_ZERO_INIT.md) (#128), the k-task fixture of
[SEQ_GENERALISATION.md](SEQ_GENERALISATION.md) (#131) and the consolidation
guard engine of [CONSOLIDATION_GUARD.md](CONSOLIDATION_GUARD.md) (#132).
"Cue", "guard", "task order" and "forgetting" name parts of an update rule
and an artificial benchmark on a Boolean graph with 4-bit weights. Nothing
here is a claim about synapses, biology, insect memory or BANC physiology.

Module: `src/cue_robustness.rs`. CLI: `bench-cue-robustness` (`--census`
for the pilot).

## Motivation (what #132 left open)

Cycle 19 (#132) found one setting that improves retention without paying
for it in learning: the silent cue with the last-resort guard
(`E/E-cue0+last-resort@paced100`, **Zg**). At k = 8 tasks, flip 50, on two
held-out orders per seed (seeds 361..400), Zg reached ACC_all 0.644: +0.034
over the silent cue alone (Z), +0.097 over the supervised control at equal
writes (S1). That was H7. Every other guard traded forgetting for learning
about 1:1. That result rests on **one** seed block, **two** random orders
per seed and k ≤ 8. #132 also reported that the guard's gain over Z did not
grow from the k = 4 prefix to k = 8, and that it works because each task's
own cue lines give it free capacity. With 64 reserved cue lines split over
k cues, that capacity shrinks as k grows: 8 lines per cue at k = 8, 5 at
k = 12, 4 at k = 16.

This cycle is a **robustness test of the H7 win**, not a new mechanism:

1. **Fresh seeds** 401..440 (never used by an earlier B888-4 cycle).
2. **More and different held-out orders**: per seed, the **reversed** order
   plus **three** new seeded permutations (new salt), never used before.
3. **Larger k**: 8, 12 and 16 tasks, each on its own fixture.
4. Always at **equal budget**: every capped setting uses the `paced100`
   per-episode cumulative write curve of uncapped error-gated in the same
   (k, flip, seed, order), and the primary comparison is against the
   supervised control.

## Settings (5 rows per (k, flip, seed, order))

All are unchanged code paths of #132 (`consolidation_guard::run_setting`);
this cycle adds orders, k values, a cue-capacity instrument and the
analysis, not a new update rule.

* `supervised@paced100` (**S1**, control). On this teacher-clamped task the
  associative co-activation rule is the same update as supervised
  ([CONSOLIDATION.md](CONSOLIDATION.md)), so S1 stands for both.
* `reward@paced100` (**R1**, reward-modulated).
* `error-gated@free` (**E**, = its `paced100` run, gated in G1).
* `E/E-cue0@paced100` (**Z**, silent cue).
* `E/E-cue0+last-resort@paced100` (**Zg**, silent cue + last-resort guard;
  constants of #132: one overwrite per (episode, target) when no free edge
  exists).

## Design-stage pilot (seeds 1..5, flip 50, identity order only; no accuracy computed)

`results/cue-robustness-census-v1.txt` (writes and guard counters only),
ranges over 5 seeds:

| k (lines per cue) | writes (all settings) | E overwrites | Z cue writes | Z overwrites | Zg cue writes | Zg overwrites | Zg deferred |
|---|---|---|---|---|---|---|---|
| 8 (8) | 208–258 | 152–188 | 48–81 | 104–127 | 89–131 | 63–85 | 626–1156 |
| 12 (5) | 367–442 | 295–357 | 69–100 | 233–281 | 104–141 | 192–233 | 703–1368 |
| 16 (4) | 423–515 | 357–442 | 62–79 | 296–364 | 105–138 | 255–318 | 658–1183 |

Readings stated before the main run:

* The guard still acts at every k: Zg defers 600–1400 overwrites per run.
* **Its bite shrinks with k.** Zg's overwrites are 0.60–0.67 of Z's at
  k = 8, but 0.81–0.88 at k = 12 and 0.84–0.88 at k = 16. With 4–5 cue lines
  per task there is less private free capacity to divert overwrites to. This
  puts H5 (Zg > Z at k = 16) and H10 (no shrinkage of the lead over S1) at
  risk. They are kept as stated; the pilot did not tune anything.
* Writes per run roughly double from k = 8 to k = 16 (the budget is per
  task), so equal budget holds per stage and per run.

## Design (pre-registered)

* **Graph, task, pool, cues:** the #131/#132 k-task fixture
  (`seq_generalisation::fixture`): generated graph per seed, 128 inputs /
  32 targets / fan-in 8, threshold 1, init weight 1 (cue-line edges 0 for Z
  and Zg), 4-bit weights. Patterns have 6 active inputs, 32 training
  patterns per task and one shared TEST set of 128 patterns, with pair
  classes seen / CN / ICN / UE. The 64 reserved cue lines are split evenly
  over the k cues (floor): 8 lines at k = 8, 5 at k = 12, 4 at k = 16.
* **k = 8, 12 and 16**: separate runs on their own fixtures. Scaling within
  a run is compared on prefixes of the same k = 16 runs.
* **Orders:** per seed, order 0 = identity (secondary only). Held-out
  orders: order 1 = the **reversed** order (k−1, …, 0); orders 2, 3, 4 =
  distinct seeded Fisher–Yates permutations ≠ identity and ≠ reversed, drawn
  with the #131 LCG and a new salt `0x0DE5_B888_4C20_0A1D`. They are the
  same for every flip and setting and were drawn only in the main run.
* **Metrics in stage coordinates** as in #131/#132: `R[i][j]` = the task
  trained at stage j on TEST after stage i. ACC = mean_j R[k−1][j],
  LEARN = mean_j R[j][j], FGT = mean_{j<k−1} (R[j][j] − R[k−1][j]),
  LAST = R[k−1][k−1], on all / CN pairs, balanced accuracy. For held-out
  statistics each seed contributes the **mean over its four held-out
  orders** (n = 40 paired seeds) unless a hypothesis names one order.
* **Worst order:** per seed, min over the four held-out orders of the
  paired difference ACC_all(x) − ACC_all(base) in that order.
* **Seeds:** 401..440 (40 seeds). **Flips:** 25 (secondary), **50
  (primary)**, 100 (secondary). Same episode budget per task (8 rounds ×
  32 patterns), no rehearsal, original edges only, topology checked per row.
* **Instrumentation per row:** cap, charged and effective writes, cue
  writes, refused, overwrites, deferred, net reversals and consolidated
  edges per stage, **cue lines per task, cue-line edges, cue-line edges
  with non-zero weight at the end and cue-line edges at weight 15 at the
  end** (new: how much of the cue capacity is used), R diagonal and final
  row, saturation, weight digest, `init_ok`, `paced_identity`, topology.
* **Checkpoint/resume:** the #132 engine checkpoint (circuit weights +
  `consolidation_guard::Checkpoint`). Resuming mid-stage is re-tested
  bit-exact for Zg at k = 16 on the reversed order.
* **Equivalence:** every row is bit-identical to a direct
  `consolidation_guard::run_setting` call with the same fixture, order and
  limits (tested), so this cycle measures the #132 rules unchanged.
* **Reproducibility:** (k, flip, seed) jobs run on worker threads. Rows,
  CSV and report do not depend on the thread count (tested). Two runs must
  give a byte-identical CSV (sha256 reported).

Statistics as in #123–#132: bootstrap 95 % percentile CI (2000 resamples,
seed 0xB0075EED).
* **strong:** mean ≥ 0.05, > 0 in ≥ 32 of 40 seeds and CI lower bound > 0.
* **positive:** mean > 0 and CI lower bound > 0.
* **sequence rule:** Δ ACC strong, Δ LAST ≥ −0.05 and Δ LEARN ≥ −0.05.

## Design gates (reported first)

* **G1 (integrity):** every row keeps its topology, starts from its
  documented initial weights and makes charged writes ≤ its allowance.
  `error-gated@paced100` is bit-identical to uncapped E with zero refusals
  in every (k, flip, seed, order). Guard counters match the guard and the
  ledger (Zg: deferred allowed, nothing blocked, charged = effective).
  Orders are valid permutations, order 1 is the reversal, orders 0..4 are
  pairwise distinct and shared by all settings.
* **G2 (manipulation, flip 50, held-out orders):** at **each** k, Zg defers
  at least once and Z makes at least one cue write in ≥ 50 % of rows.
* **G3 (novel pairs):** median CN pairs per (seed, task) at k = 16, flip 50
  ≥ 20.

If any gate fails, H1–H10 are declared **not evaluable**.

## Hypotheses (held-out orders, seeds 401..440, flip 50)

* **H1 (primary, replication on fresh seeds and orders):** at k = 8, Zg vs
  S1 passes the sequence rule on all pairs.
* **H2 (#132 H7 replication):** at k = 8, Zg vs Z cuts FGT_all by a mean
  ≥ 0.05 with CI lower bound > 0, **and** ACC_all(Zg) − ACC_all(Z) is
  positive.
* **H3 (k = 12):** Zg vs S1 passes the sequence rule on all pairs.
* **H4 (k = 16):** Zg vs S1 passes the sequence rule on all pairs.
* **H5 (guard on the cue at k = 16):** ACC_all(Zg) − ACC_all(Z) is positive.
* **H6 (each held-out order, k = 8):** ACC_all(Zg) − ACC_all(S1) is positive
  on each of the four held-out orders taken alone (reversed, random 1, 2, 3).
* **H7 (worst order, k = 16):** the per-seed worst-order difference
  ACC_all(Zg) − ACC_all(S1) is positive.
* **H8 (vs error-gated, k = 16):** ACC_all(Zg) − ACC_all(E) is positive.
* **H9 (vs reward-modulated, k = 16):** ACC_all(Zg) − ACC_all(R1) is positive.
* **H10 (scaling, same k = 16 runs):** growth of ACC_all(Zg) − ACC_all(S1)
  from the k = 8 prefix to k = 16 has CI lower bound > −0.05 (the lead does
  not shrink by 0.05 or more).

**Secondary (reported, no pass/fail):** all of the above on the identity
order and at flips 25 / 100; CN pairs; Zg − Z, Z − S1, E − S1 and Zg − E per
k; per-order breakdown at k = 12 and 16; growth of Zg − Z and E − S1 in the
k = 16 runs; deferred, overwrites, cue writes and cue-capacity use per k.

No multiplicity correction is applied. The analysis is run once. Nothing
above the Results heading is changed after the run except the "pending"
placeholder.

## Command

```bash
cargo +1.89 run --release -- bench-cue-robustness --csv results/cue-robustness-v1.csv > results/cue-robustness-report-v1.txt
# pilot (identity order, writes and guard counters only): cargo +1.89 run --release -- bench-cue-robustness --census > results/cue-robustness-census-v1.txt
```

## Results

Pending (run once after this pre-registration has been pushed).

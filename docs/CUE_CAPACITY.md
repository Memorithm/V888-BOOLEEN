# B888-4 cycle 21: does sharing the same cue lines between tasks raise cue capacity per task, at equal lines and equal write budget, and restore the silent cue + last-resort lead over the supervised control at k = 16 and k = 24? (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123), with the k-task fixture of
[SEQ_GENERALISATION.md](SEQ_GENERALISATION.md) (#131), the consolidation guard
engine of [CONSOLIDATION_GUARD.md](CONSOLIDATION_GUARD.md) (#132) and the
order/statistics protocol of [CUE_ROBUSTNESS.md](CUE_ROBUSTNESS.md) (#133).
"Cue", "cue line", "code", "sharing", "guard", "task order" and "forgetting"
name parts of an update rule and an artificial benchmark on a Boolean graph
with 4-bit weights. Nothing here is a claim about synapses, biology, insect
memory or BANC physiology.

Module: `src/cue_capacity.rs`. CLI: `bench-cue-capacity` (`--census` for the
pilot).

## Motivation (what #133 left open)

Cycle 20 (#133) replicated the silent cue + last-resort win (**Zg**,
`E/E-cue0+last-resort@paced100`) on fresh seeds and new held-out orders at
k = 8 and k = 12. At k = 16 Zg was still the best setting and beat the
supervised control (S1) in 40/40 seeds, but by +0.042 [+0.038, +0.047], under
the pre-registered 0.05 bound (H4 failed). The decline tracked **cue capacity
per task**: the 64 reserved cue lines are split into k **disjoint** cues, so
each task owns 8 lines at k = 8 but 4 at k = 16; Zg kept ≈ 51–55 live cue
edges at every k (≈ 3.2 per task at k = 16) and the guard's own gain over the
plain silent cue (Z) faded to +0.011.

There are two ways to give each task more cue lines. Reserving more lines
shrinks the pattern pool and changes the task, so it confounds capacity with
task difficulty. This cycle uses the other one, which keeps **everything
fixed except the assignment of the reserved lines to tasks**: the same graph,
the same reserved lines, the same pattern pool, the same tasks, the same
initial weights and the same write budget. Each task's cue becomes the union
of `r` of the cycle 20 blocks, so a cue line is shared by `r` tasks and each
task sees `r ×` as many cue lines (and cue-line edges). The question is
whether more cue capacity per task outweighs the interference of sharing.

## Codes (pre-registered construction, no tuning)

* **Blocks:** `B_0 … B_{k−1}` = the cycle 18–20 disjoint cues of the k-task
  fixture (`seq_generalisation::fixture`: greedy capped target cover, cap
  = ⌊64 / k⌋ lines; 8 at k = 8, 4 at k = 16, 2 at k = 24). The pattern pool
  excludes exactly the union of the blocks, as in #131–#133.
* **Share-r code of task j:** `C_j(r)` = sorted union of
  `B_{(j + i) mod k}` for i = 0 … r − 1 (cyclic consecutive blocks, by task
  id, so independent of the training order).
* **r ∈ {1, 2, 4}.** r = 1 is the cycle 20 disjoint cue (bit-identical, tested).
  Lines per task: k = 8 → 8 / 16 / 32, k = 16 → 4 / 8 / 16, k = 24 → 2 / 4 / 8.
  Every reserved line belongs to exactly r codes; codes are pairwise distinct;
  the union of all codes equals the union of the blocks (checked in G1).
* Cue-line edges start at 0 for every cued setting (the silent cue of #128);
  which edges are cue edges does not depend on r.

## Settings (9 rows per (k, flip, seed, order))

All rules are unchanged code paths of #132/#133
(`consolidation_guard::run_setting`); only the fixture's per-task cue lists
change with r.

* `supervised@paced100` (**S1**, control). On this teacher-clamped task the
  associative co-activation rule is the same update as supervised
  ([CONSOLIDATION.md](CONSOLIDATION.md)), so S1 stands for both.
* `reward@paced100` (**R1**, reward-modulated).
* `error-gated@free` (**E**, = its `paced100` run, gated in G1).
* `E/E-cue0@paced100/share{1,2,4}` (**Z1, Z2, Z4**, silent cue).
* `E/E-cue0+last-resort@paced100/share{1,2,4}` (**Zg1, Zg2, Zg4**, silent
  cue + last-resort guard; #132 constants).

**Equal budget:** every capped setting uses the `paced100` per-episode
cumulative write curve of uncapped E in the same (k, flip, seed, order); cue
writes are charged. Equal resources: all cued settings of a (k, seed) use the
same reserved lines and the same cue-line edges; only their assignment to
tasks differs.

## Design (pre-registered)

* **Graph, task, pool:** the #131–#133 k-task fixture: generated graph per
  seed, 128 inputs / 32 targets / fan-in 8, threshold 1, init weight 1
  (cue-line edges 0 for Z*, Zg*), 4-bit weights; 6 active inputs per
  pattern, 32 training patterns per task, one shared TEST set of 128
  patterns, pair classes seen / CN / ICN / UE.
* **k = 8, 16 and 24**, separate runs on their own fixtures (k = 24 reserves
  48 lines, so its pool is wider than at k = 16; comparisons are within k).
* **Orders:** per seed, order 0 = identity (secondary only); held-out orders
  1 = reversed, 2..4 = distinct seeded Fisher–Yates permutations (the #133
  generator) with a **new salt** `0x0DE5_B888_4C21_0A1D`, shared by every
  flip and setting.
* **Seeds:** 441..480 (40 seeds, never used by an earlier B888-4 cycle).
  **Flips:** 25 (secondary), **50 (primary)**, 100 (secondary). Same episode
  budget per task (8 rounds × 32 patterns), no rehearsal, original edges
  only, topology checked per row.
* **Metrics** in stage coordinates as in #131–#133: ACC, LEARN, FGT, LAST on
  all / CN pairs, balanced accuracy. Held-out statistics: each seed
  contributes the mean over its 4 held-out orders (n = 40 paired seeds).
  Worst order: per seed, min over held-out orders of the paired difference.
* **Instrumentation per row:** share r, lines per task, mean cue reach per
  task (distinct targets of the code's lines), cue-line edges, live / full
  cue edges at the end, cap, charged / effective / cue writes, refused,
  overwrites, deferred, net reversals, consolidated edges, R diagonal and
  final row, saturation, weight digest, `init_ok`, `paced_identity`, topology.
* **Checkpoint/resume:** the #132 engine checkpoint, re-tested bit-exact for
  Zg2 at k = 16 on the reversed order (save/load at ¼, ½, end of each stage).
* **Reproducibility:** rows, CSV and report do not depend on the worker
  thread count (tested); two main runs must give a byte-identical CSV
  (sha256 reported).

Statistics as in #123–#133: bootstrap 95 % percentile CI (2000 resamples,
seed 0xB0075EED).
* **strong:** mean ≥ 0.05, > 0 in ≥ 32 of 40 seeds and CI lower bound > 0.
* **positive:** mean > 0 and CI lower bound > 0.
* **sequence rule:** Δ ACC strong, Δ LAST ≥ −0.05 and Δ LEARN ≥ −0.05.

## Design gates (reported first)

* **G1 (integrity):** every row keeps its topology, starts from its
  documented initial weights and makes charged writes ≤ its allowance;
  `error-gated@paced100` is bit-identical to uncapped E with zero refusals;
  guard counters match guard and ledger; orders valid (1 = reversal,
  pairwise distinct, shared); codes valid (share 1 = the blocks, every line
  in exactly r codes, codes distinct, same line union for every r).
* **G2 (manipulation, flip 50, held-out orders, every k):** Zg2 defers in
  ≥ 50 % of rows, Z2 makes a cue write in ≥ 50 % of rows, and the mean cue
  reach per task is strictly larger at share 2 than at share 1.
* **G3 (novel pairs):** median CN pairs per (seed, task) at k = 24, flip 50
  ≥ 20.

If any gate fails, H1–H10 are declared **not evaluable**.

## Hypotheses (held-out orders, seeds 441..480, flip 50)

* **H1 (primary, the #133 H4 retest):** at k = 16, Zg2 vs S1 passes the
  sequence rule on all pairs.
* **H2 (sharing beats owning at equal lines):** at k = 16, ACC_all(Zg2) −
  ACC_all(Zg1) is positive.
* **H3 (forgetting):** at k = 16, FGT_all(Zg1) − FGT_all(Zg2) is positive.
* **H4 (not guard-specific):** at k = 16, ACC_all(Z2) − ACC_all(Z1) is
  positive.
* **H5 (more sharing):** at k = 16, ACC_all(Zg4) − ACC_all(Zg2) is positive.
* **H6 (worst order):** at k = 16, the per-seed worst held-out order
  ACC_all(Zg2) − ACC_all(Zg1) is positive.
* **H7 (k = 24):** Zg2 vs S1 passes the sequence rule on all pairs.
* **H8 (k = 24):** ACC_all(Zg2) − ACC_all(Zg1) is positive.
* **H9 (scaling, same k = 24 runs):** growth of ACC_all(Zg2) − ACC_all(S1)
  from the k = 8 prefix to k = 24 has CI lower bound > −0.05.
* **H10 (no harm when capacity is ample, k = 8):** ACC_all(Zg2) −
  ACC_all(Zg1) has CI lower bound > −0.01.

**Secondary (reported, no pass/fail):** all comparisons on the identity
order and at flips 25 / 100; CN pairs; Zg{1,2,4} vs S1 / R1 / E per k; the
best share per k; per-order breakdown; growth in the k = 16 and k = 24 runs;
reach, live cue edges, cue writes, overwrites and deferred per share.

No multiplicity correction is applied. The analysis is run once. Nothing
above the Results heading is changed after the main run except the
"pending" placeholder; the design-stage pilot section below is filled in
before the main run from the label-free census only.

## Design-stage pilot

Pending (seeds 1..5, identity order, writes, guard counters and cue reach
only; no accuracy computed).

## Command

```bash
cargo +1.89 run --release -- bench-cue-capacity --csv results/cue-capacity-v1.csv > results/cue-capacity-report-v1.txt
# pilot (identity order, writes, guard counters and reach only): cargo +1.89 run --release -- bench-cue-capacity --census > results/cue-capacity-census-v1.txt
```

## Results

Pending.

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

`results/cue-capacity-census-v1.txt` (seeds 1..5, flip 50, identity order;
writes, guard counters and cue reach only; **no accuracy computed**). Ranges
over the 5 seeds:

| k | share (lines per task) | mean cue reach per task (of 32 targets) | effective writes (every arm) | Z cue writes | Zg cue writes | Zg overwrites | Zg deferred |
|---|---|---|---|---|---|---|---|
| 8 | 1 (8) | 22.6–23.5 | 208–258 | 48–81 | 89–131 | 63–85 | 626–1156 |
| 8 | 2 (16) | 28.4–29.4 | 208–258 | 83–129 | 130–177 | 37–52 | 1014–1597 |
| 8 | 4 (32) | 31.5–32.0 | 208–258 | 115–171 | 151–200 | 33–52 | 831–2192 |
| 16 | 1 (4) | 11.8–12.4 | 423–515 | 62–79 | 105–138 | 255–318 | 658–1183 |
| 16 | 2 (8) | 18.9–19.5 | 426–515 | 134–170 | 179–227 | 215–260 | 1081–1766 |
| 16 | 4 (16) | 25.9–27.5 | 426–515 | 200–277 | 273–355 | 160–192 | 2478–3567 |
| 24 | 1 (2) | 6.5–7.2 | 1036–1174 | 72–109 | 150–204 | 757–855 | 778–1397 |
| 24 | 2 (4) | 11.3–12.5 | 1041–1174 | 187–210 | 306–363 | 662–738 | 1658–2472 |
| 24 | 4 (8) | 19.0–20.5 | 1045–1174 | 323–418 | 496–600 | 568–616 | 2930–4657 |

Readings stated before the main run:

* The share-1 rows reproduce the cycle 20 pilot exactly at k = 8 and 16
  (same writes, cue writes, overwrites and deferred ranges), as expected
  from the bit-identity test.
* **The manipulation works as intended:** at equal lines and edges, sharing
  raises the cue reach per task (k = 16: ≈ 12 → 19 → 27 targets of 32) and
  the cue writes, and lowers Zg's overwrites of consolidated edges
  (k = 16: 255–318 → 215–260 → 160–192).
* Writes stay equal across arms within a (k, seed) (paced100), so any
  accuracy difference between shares is not a budget difference.
* Sharing also means a cue write for task j moves an edge that r − 1 other
  tasks' cues use; whether that interference outweighs the reach gain is
  exactly what H2–H5, H8 and H10 test. Nothing was tuned on the pilot.

## Command

```bash
cargo +1.89 run --release -- bench-cue-capacity --csv results/cue-capacity-v1.csv > results/cue-capacity-report-v1.txt
# pilot (identity order, writes, guard counters and reach only): cargo +1.89 run --release -- bench-cue-capacity --census > results/cue-capacity-census-v1.txt
```

## Results

Run once after the pre-registration commit 00f376b (and the harness commit
5db91b0) had been pushed. The grid has 16200 rows: 3 k × 3 flips × 40 seeds
× 5 orders × 9 arms. It took 30.5 min on 8 worker threads (202 CPU-min).
CSV sha256 `7a17a41b738595a84dce16c183635118d6aa88167faef0f4bbf55b2f0ab00a03`.
A second, independent run with 5 worker threads gave a byte-identical CSV
(same sha256) and an identical report.
Every row keeps the original topology (`topology_unchanged_all=true`).
Files: `results/cue-capacity-v1.csv`, `results/cue-capacity-report-v1.txt`,
pilot `results/cue-capacity-census-v1.txt`.

### 1. Design gates (reported first)

* **G1 passes:** topology, initial weights, allowances, `paced100` identity
  of E, guard/ledger consistency, orders and codes (share 1 = the cycle 20
  blocks, every reserved line in exactly r codes, codes distinct, same
  cue-line edges for every share) hold in every row.
* **G2 passes:** at flip 50 on held-out orders, Zg2 defers and Z2 writes cue
  edges in 100 % of rows at every k; mean cue reach per task rises with the
  share (k = 8: 23.1 → 29.3 → 31.8; k = 16: 12.1 → 19.2 → 26.9;
  k = 24: 6.8 → 11.9 → 19.4 targets of 32).
* **G3 passes:** median CN pairs per (seed, task) is 109 at k = 24 (62 at
  k = 8, 65 at k = 16).

### 2. Arms at flip 50, held-out orders (mean over 40 seeds × 4 orders)

| k | S1 | R1 | E | Z1 | **Zg1** | Z2 | Zg2 | Z4 | Zg4 | Zg1 / Zg2 / Zg4 LEARN_all | Zg1 / Zg2 / Zg4 FGT_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.549 | 0.571 | 0.568 | 0.609 | **0.643** | 0.598 | 0.608 | 0.572 | 0.569 | 0.858 / 0.800 / 0.760 | 0.245 / 0.219 / 0.218 |
| 16 | 0.528 | 0.548 | 0.544 | 0.566 | **0.575** | 0.561 | 0.572 | 0.555 | 0.556 | 0.908 / 0.850 / 0.783 | 0.356 / 0.297 / 0.242 |
| 24 | 0.521 | 0.539 | 0.532 | 0.541 | **0.545** | 0.541 | 0.544 | 0.536 | 0.539 | 0.910 / 0.865 / 0.800 | 0.381 / 0.335 / 0.273 |

ACC_all shown. All arms make the same number of effective writes within a
(k, flip, seed, order) up to 0.3 % (k = 16: 512.8–514.4 per run; k = 24:
1132–1135). Zg1 (the cycle 20 cue) is the best arm at every k; its k = 8
and k = 16 numbers replicate #133 on new seeds and orders (0.643 / 0.575 vs
0.643 / 0.575).

### 3. Hypotheses (flip 50, held-out orders; Δ = mean [95 % CI] better/ties/worse over 40 seeds)

* **H1 fail (primary).** k = 16, Zg2 vs S1: ACC_all +0.043 [+0.038, +0.048]
  40/0/0 (LEARN +0.242, LAST +0.235 pass). Sharing does not lift the lead
  over the supervised control above 0.05; it is no larger than Zg1's own
  lead on these seeds (+0.046 [+0.042, +0.051], also < 0.05).
* **H2 fail.** k = 16, Zg2 − Zg1 ACC_all = −0.003 [−0.006, −0.000] 17/0/23.
* **H3 pass.** k = 16, FGT_all(Zg1) − FGT_all(Zg2) = +0.059 [+0.055, +0.063]
  40/0/0. But LEARN falls by −0.059 [−0.061, −0.056] (LAST −0.079): the
  forgetting cut is paid 1:1 in learning.
* **H4 fail.** k = 16, Z2 − Z1 ACC_all = −0.004 [−0.007, −0.001] 12/0/28
  (same trade without the guard: FGT +0.047, LEARN −0.049).
* **H5 fail.** k = 16, Zg4 − Zg2 ACC_all = −0.015 [−0.019, −0.012] 5/0/35.
* **H6 fail.** k = 16, worst held-out order Zg2 − Zg1 = −0.019 [−0.024,
  −0.015] 3/0/37.
* **H7 fail.** k = 24, Zg2 vs S1: ACC_all +0.023 [+0.021, +0.026] 40/0/0.
* **H8 fail.** k = 24, Zg2 − Zg1 ACC_all = −0.000 [−0.002, +0.001] 21/0/19.
* **H9 pass.** In the k = 24 runs, growth of Zg2 − S1 from the k = 8 prefix
  to k = 24 is −0.013 [−0.017, −0.010] 3/0/37 (CI lower bound > −0.05; the
  lead shrinks a little but not by 0.05).
* **H10 fail.** k = 8, Zg2 − Zg1 ACC_all = −0.035 [−0.039, −0.030] 0/0/40:
  with ample capacity, sharing clearly hurts (Zg4 − Zg1: −0.074).

### 4. Secondary

* **Sharing converts learning into retention ~1:1 at every k.** Each step
  up in share cuts FGT_all by ≈ 0.05–0.06 and cuts LEARN_all by about the
  same amount (table above), so ACC_all stays flat (k = 16, 24) or falls
  (k = 8). This is the same 1:1 trade that #132 found for uncued guards.
  More reach per task does not buy more net accuracy at equal writes.
* **Interference tracks label conflict between tasks.** At flip 25 (tasks
  agree more), sharing helps a little: Zg2 − Zg1 = +0.013 [+0.009, +0.017]
  35/0/5 at k = 16 and +0.005 [+0.003, +0.006] at k = 24. At k = 16, flip 25
  Zg2 vs S1 then passes the sequence rule (+0.058 [+0.051, +0.065] 40/0/0).
  At flip 100 it hurts a lot: Zg2 − Zg1 = −0.046 [−0.049, −0.043] 0/0/40
  (k = 16), and Zg2 is level with S1 (+0.002). A shared cue edge carries
  updates of tasks with opposite labels, so the shared code is only as good
  as the agreement between the tasks that share it.
* **Per order at k = 16 (Zg2 − Zg1):** reversed −0.014 [−0.020, −0.009]; the
  three random orders −0.000, +0.001, +0.001 (all CI include 0); identity
  −0.006. Zg2 − S1 is positive on every order (+0.036 to +0.050).
* **Capacity use:** sharing raises cue writes (k = 16: Zg 134 → 235 → 351)
  and lowers overwrites of consolidated edges (306 → 241 → 193), but live
  cue edges at the end peak at share 2 (51 → 62 → 53) and no cue edge
  reaches weight 15 in any arm.
* **The k = 16 gap to the 0.05 bound did not move.** On fresh seeds Zg1 − S1
  is +0.046 at k = 16 (cycle 20: +0.042) and +0.023 at k = 24. Within the
  k = 24 runs the Zg1 − S1 lead shrinks by −0.009 from the k = 8 prefix and
  E − S1 by −0.011, so part of the decline is again the error gate's own
  k-trend (#131).

**Reading:** with the reserved lines, pattern pool, tasks and write budget
held fixed, giving each task more cue lines by sharing lines between tasks
does **not** restore the lead over the supervised control at k = 16 (H1) or
k = 24 (H7). Sharing raises cue reach as designed (G2) and cuts forgetting
(H3), but it costs the same amount of learning, so net accuracy is flat at
k = 16/24 and lower at k = 8 (H2, H4–H8, H10 fail). The cost depends on how
much the sharing tasks disagree (helps at flip 25, hurts at flip 100). So
"cue capacity per task" in cycle 20 is not just a count of lines or reached
targets. What helps is **private** capacity: cue edges that only one task
writes. Under a fixed line budget, the remaining options are to make
capacity private without the cost of sharing (e.g. assign shared lines by
task similarity, or reserve lines out of a larger graph at equal pattern
pool), or to accept that the cue + guard advantage at equal writes is about
+0.045 at k = 16 and +0.023 at k = 24 on this benchmark.

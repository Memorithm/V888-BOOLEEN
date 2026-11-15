# B888-4 cycle 19: can a consolidation guard on the original connections cut the error gate's forgetting at k = 8 without losing its learning, at equal write budget? (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123), with the fixed pattern pool of
[CUE_REACH.md](CUE_REACH.md) (#124), the write budgets of
[WRITE_BUDGET.md](WRITE_BUDGET.md) (#126), the silent cue of
[CUE_ZERO_INIT.md](CUE_ZERO_INIT.md) (#128), the settings of
[CUE_DISPLACEMENT.md](CUE_DISPLACEMENT.md) (#129) and the k-task fixture and
held-out orders of [SEQ_GENERALISATION.md](SEQ_GENERALISATION.md) (#131).
"Consolidation", "guard", "overwrite" and "forgetting" name parts of an
update rule on a Boolean graph with 4-bit weights. Nothing here is a claim
about synapses, biology, insect memory or BANC physiology.

Module: `src/consolidation_guard.rs`. CLI: `bench-consolidation-guard`
(`--census` for the pilot).

## Motivation (what #131 left open)

At k = 8 tasks on held-out orders (#131), the plain error gate
(`error-gated`, **E**) learns almost every task (LEARN_all 0.990) but
forgets about half of it (FGT_all 0.482). Its lead over the supervised
control shrinks with k: +0.123 at k = 2, +0.054 at k = 4, +0.020 at k = 8
(#131 H1 failed). The silent cue (**Z**) was the best setting at equal
writes, but it also forgets (FGT_all 0.314). This cycle attacks forgetting
directly with a **consolidation guard on the original edges**. There is
no rehearsal buffer, no stored pattern and no new edge. The only state is
one sign and one stage stamp per edge, at the same `paced100` write budget.

### Mechanism

* **Consolidation:** at the end of every stage, each edge whose weight
  differs from its weight at the start of that stage is *consolidated*: it
  records the sign of the net change (+ / −) and the stage (stamp).
* **Overwrite:** in a later stage, a step that would change a weight
  **against** the edge's consolidated sign (an increment of an edge
  consolidated −, or a decrement of an edge consolidated +).
* **Guards** (all on the unchanged error-gated step; the gate decision and
  the update direction are not changed):
  * `block` (**Gb**): every overwrite is refused. It is counted and not
    retried, and it costs no write.
  * `block-recent2` (**Gr**): an overwrite is refused only if the edge was
    consolidated at the end of one of the **last 2 stages**. Older
    consolidations are plastic again. This is a stability/plasticity window
    between `off` and `block`.
  * `charge4` (**Gc**): an overwrite is allowed but costs **4** writes of
    the same `paced100` budget (a write that is not an overwrite costs 1).
  * `last-resort` (**Gl**): an overwrite on edge e → t is deferred while
    target t has an offered "free" edge, one that would change without an
    overwrite. If t has none, **one** overwrite per (episode, target) is
    allowed (the first in source order) and any others are deferred.
  * `off`: no guard. It is bit-identical to the #131 code path for
    `error-gated` and `E/E-cue0` (tested). It still counts overwrites.
* Constants (window 2, cost 4, one overwrite per target) were fixed before
  the pilot. They were not tuned.

### Settings (9 rows per (k, flip, seed, order))

* `supervised@paced100` (**S1**, control). On this teacher-clamped task the
  associative co-activation rule is the same update as supervised
  ([CONSOLIDATION.md](CONSOLIDATION.md)), so S1 stands for both.
* `reward@paced100` (**R1**, reward-modulated).
* `error-gated@free` (**E**, = its `paced100` run, gated in G1).
* `E/E-cue0@paced100` (**Z**, silent cue).
* `error-gated+block@paced100` (**Gb**), `error-gated+block-recent2@paced100`
  (**Gr**), `error-gated+charge4@paced100` (**Gc**),
  `error-gated+last-resort@paced100` (**Gl**).
* `E/E-cue0+last-resort@paced100` (**Zg**): the silent cue with the
  last-resort guard. The current task's cue edges are never consolidated
  by another task, so they act as free capacity.

Every capped setting uses the `paced100` per-episode cumulative write curve
of uncapped E in the same (k, flip, seed, order). **Effective writes**
(weight-changing steps) are reported next to **charged writes**: they are
equal except for Gc, and blocked or deferred steps cost nothing.

### Design-stage pilot (seeds 1..5, flip 50, identity order only; no accuracy computed)

`results/consolidation-guard-census-v1.txt`, ranges over 5 seeds:

| k = 8 | charged writes | effective writes | overwrites | blocked | deferred | consolidated edges at end |
|---|---|---|---|---|---|---|
| E | 208–258 | 208–258 | 152–188 | 0 | 0 | 53–64 |
| Z | 208–258 | 208–258 | 104–127 | 0 | 0 | 79–98 |
| Gb (block) | 96–147 | 96–147 | 0 | 5884–7641 | 0 | 53–64 |
| Gr (block-recent2) | 159–206 | 159–206 | 69–96 | 2838–3827 | 0 | 53–64 |
| Gc (charge4) | 199–252 | 94–114 | 35–46 | 0 | 0 | 53–64 |
| Gl (last-resort) | 206–256 | 206–256 | 150–184 | 0 | 4–24 | 53–64 |
| Zg (Z + last-resort) | 208–258 | 208–258 | 63–85 | 0 | 626–1156 | 95–114 |

At k = 4 (same seeds): E 87–110 overwrites of 192–208 writes; Gr 26–33
overwrites and 1627–2114 blocked; Gl 6–10 deferred; Zg 0 overwrites and
693–991 deferred.

Readings that shaped the design (stated before the main run):

* **The pilot revised Gl and added Gr.** The first version of
  `last-resort` deferred an overwrite only while a free edge existed. It
  deferred 0–20 steps per run on E, so it was ≈ E. The reason is the
  fixture: with 6 active inputs of 128 and fan-in 8, a target has on
  average ≈ 0.4 active edges per pattern. Most overwrites are therefore
  single-edge direct conflicts, and on the uncued rule there is no spare
  capacity to use instead. Rationing to one overwrite per (episode,
  target) did not change that either (4–24 deferrals). Gl is kept, and its
  expected null is pre-registered (H9).
* `block-recent2` (Gr) was added after that first pilot, as the
  stability/plasticity knob the cycle needs between E and Gb. The pilot
  shows a clean gradient: overwrites E ≈ 170, Gr ≈ 80, Gb 0, and
  effective writes ≈ 230, ≈ 180, ≈ 120.
* Gb leaves about half of E's budget unused because its steps are blocked.
  Gc's overwrites cost 4, so it makes about half of E's effective writes.
  Both are **equal-budget** settings (same cap), not equal-effective-write
  settings, and the report shows both write counts.
* On the cued rule the guard has bite: Zg defers 600–1200 overwrites and
  cuts them from ≈ 115 (Z) to ≈ 75, because the task's own cue edges are
  free capacity.

## Design (pre-registered)

* **Graph, task, pool, cues:** the #124–#131 fixed-pool design with the
  #131 k-task fixture: generated graph per seed, 128 inputs / 32 targets /
  fan-in 8, threshold 1, init weight 1 (cue-line edges 0 for Z and Zg),
  4-bit weights. Patterns have 6 active inputs, 32 training patterns per
  task and one shared TEST set of 128 patterns, with pair classes seen /
  CN / ICN / UE. The 64 reserved cue lines are split over the k cues (16 at
  k = 4, 8 at k = 8). At k = 4 the fixture is exactly
  `novel_split::fixture` (#131 test).
* **k = 4 and k = 8**: separate runs on their own fixtures. Growth with k
  is only compared on prefixes of the same k = 8 runs.
* **Orders:** per seed, order 0 = identity. Orders 1 and 2 are distinct
  seeded permutations ≠ identity, drawn with the #131 generator and a new
  salt `0x0DE5_B888_4C19_0A1D`. They are the same for every flip and
  setting and were drawn only in the main run.
* **Metrics in stage coordinates** as in #131: `R[i][j]` = the task trained
  at stage j on TEST after stage i. ACC = mean_j R[k−1][j],
  LEARN = mean_j R[j][j], FGT = mean_{j<k−1} (R[j][j] − R[k−1][j]),
  LAST = R[k−1][k−1], on all / CN pairs, balanced accuracy. For held-out
  statistics each seed contributes the **mean over its two held-out
  orders** (n = 40 paired seeds).
* **Seeds:** 361..400 (40 seeds, never used by an earlier B888-4 cycle).
  **Flips:** 25 (secondary), **50 (primary)**, 100 (high conflict). Same
  episode budget per task (8 rounds × 32 patterns), no rehearsal, original
  edges only, topology checked per row.
* **Instrumentation per row:** cap, charged writes, effective writes, cue
  writes, refused, overwrites (total and per stage), blocked, deferred, net
  reversals per stage (edges whose net change reversed their consolidated
  sign, computed from weight snapshots for every setting, S1 and R1
  included), moved and consolidated edges per stage, R diagonal and final
  row, weight digest, `init_ok`, `paced_identity`, topology.
* **Checkpoint/resume:** circuit weights (`PlasticCircuit::save`) plus the
  text `Checkpoint`: guard, per-edge signs and stamps, stage-start weights,
  guard counters and the #131 checkpoint. Resuming mid-stage is bit-exact
  for every guarded setting. Corrupt checkpoints are rejected (tested):
  bad magic, unknown guard, bad sign, bad / future / sign-less stamp, bad
  start weight, edge-count mismatch, counters inconsistent with the guard
  or the ledger, inner #131 corruption and truncation.
* **Equivalence:** with the guard off, and for S1 / R1, runs are
  bit-identical to `seq_generalisation::run_setting` (weights, ledgers,
  cost, every evaluation) at k = 4, 6 and 8, on identity and held-out
  orders (tested).
* **Reproducibility:** (k, flip, seed) jobs run on worker threads. Rows,
  CSV and report do not depend on the thread count (tested). Two runs must
  give a byte-identical CSV (sha256 reported).

Statistics as in #123–#131: bootstrap 95 % percentile CI (2000 resamples,
seed 0xB0075EED).
* **strong:** mean ≥ 0.05, > 0 in ≥ 32 of 40 seeds and CI lower bound > 0.
* **positive:** mean > 0 and CI lower bound > 0.
* **equivalent:** 95 % CI inside (−0.05, +0.05).
* **sequence rule:** Δ ACC strong, Δ LAST ≥ −0.05 and Δ LEARN ≥ −0.05.
* New for this cycle: **guard criterion (x vs E)** = forgetting cut
  FGT(E) − FGT(x) has mean **≥ 0.10** and CI lower bound > 0, **and** the
  learning loss LEARN(E) − LEARN(x) has mean **≤ 0.10**.

## Design gates (reported first)

* **G1 (integrity):** every row keeps its topology, starts from its
  documented initial weights and makes charged writes ≤ its allowance.
  `error-gated@paced100` is bit-identical to uncapped E with zero
  refusals in every (k, flip, seed, order). Guard counters match the guard
  and the ledger in every stage: blocked only for Gb / Gr, deferred only
  for Gl / Zg, no overwrite for Gb, and charged = effective + 3 ×
  overwrites for Gc, = effective otherwise. Orders are valid, distinct and
  shared by all settings.
* **G2 (manipulation, k = 8, flip 50, held-out orders):** in ≥ 50 % of
  (seed, order) rows each of these holds: E overwrites at least once, Gb
  and Gr each block at least once, Gc overwrites at least once, and Zg
  defers at least once.
* **G3 (novel pairs):** median CN pairs per (seed, task) at k = 8, flip 50
  ≥ 20.

If any gate fails, H1–H10 are declared **not evaluable**.

## Hypotheses (held-out orders, seeds 361..400; k = 8 and flip 50 unless noted)

* **H1 (primary, window guard):** Gr vs E meets the guard criterion: it
  forgets at least 0.10 less and learns at most 0.10 less.
* **H2 (full block):** Gb vs E meets the guard criterion.
* **H3 (charging):** Gc vs E meets the guard criterion.
* **H4 (net retention):** ACC_all(Gr) − ACC_all(E) is positive.
* **H5 (vs the supervised control at equal writes):** Gr vs S1 passes the
  sequence rule on all pairs.
* **H6 (vs reward-modulated):** ACC_all(Gr) − ACC_all(R1) is positive.
* **H7 (guard on top of the silent cue):** Zg vs Z cuts FGT_all by a mean
  ≥ 0.05 with CI lower bound > 0, **and** ACC_all(Zg) − ACC_all(Z) is
  positive.
* **H8 (stability-plasticity ordering):** forgetting E > Gr > Gb and
  learning E > Gr > Gb. All four pairwise differences are positive.
* **H9 (pre-registered null from the pilot):** ACC_all(Gl) − ACC_all(E)
  is equivalent to 0. Rationing overwrites does not bind on the uncued
  rule.
* **H10 (k = 4 replication, own k = 4 fixture):** Gr vs E at k = 4: the
  forgetting cut is positive and the learning loss mean is ≤ 0.10.

**Secondary (reported, no pass/fail):** all of the above on the identity
order and at flips 25 / 100; CN pairs; Gr / Gb / Gc / Zg vs Z, S1 and R1;
growth of Gr − E, Gr − S1, E − S1, Zg − Z and Gb − E from the k = 4 prefix
to k = 8 (same runs); overwrites, blocked, deferred, net reversals,
effective vs charged writes and saturation per setting.

No multiplicity correction is applied. The analysis is run once. Nothing
above the Results heading is changed after the run except the "pending"
placeholder.

## Command

```bash
cargo +1.89 run --release -- bench-consolidation-guard --csv results/consolidation-guard-v1.csv > results/consolidation-guard-report-v1.txt
# pilot (identity order, writes and guard counters only): cargo +1.89 run --release -- bench-consolidation-guard --census > results/consolidation-guard-census-v1.txt
```

## Results

Run once after the pre-registration commit 93af1b4 had been pushed. The
grid has 6480 rows: 2 k × 3 flips × 40 seeds × 3 orders × 9 settings. It
takes about 95 s on 8 CPU threads (12 CPU-min). Two runs (8 and 3 worker
threads) give a byte-identical CSV (sha256 `25d0b243…b80c61`, full
`25d0b24344e5ce3cc5c6390d10cbe54512efd017a58d7b4d3442605fd8b80c61`) and an
identical report. Every row keeps the original topology
(`topology_unchanged_all=true`). Files: `results/consolidation-guard-v1.csv`,
`results/consolidation-guard-report-v1.txt`, pilot
`results/consolidation-guard-census-v1.txt`.

### 1. Design gates (reported first)

* **G1 passes:** every row keeps its topology, starts from its documented
  initial weights and stays within its allowance.
  `error-gated@paced100` is bit-identical to uncapped E with zero
  refusals everywhere. Guard counters match the guard and the ledger in
  every stage. Orders are valid, distinct and shared.
* **G2 passes:** at k = 8, flip 50 on held-out orders, E overwrites, Gb
  and Gr block, Gc overwrites and Zg defers in 100 % of rows.
* **G3 passes:** median CN pairs per (seed, task) is 59 ≥ 20.

### 2. Settings at flip 50, held-out orders (mean over 40 seeds × 2 orders)

k = 8:

| setting | ACC_all | ACC_CN | LEARN_all | FGT_all | FGT_CN | LAST_all | effective writes | overwrites | blocked / deferred |
|---|---|---|---|---|---|---|---|---|---|
| supervised @ paced100 (S1) | 0.547 | 0.528 | 0.639 | 0.105 | 0.042 | 0.606 | 256 | – | – |
| reward @ paced100 (R1) | 0.575 | 0.566 | 0.603 | 0.032 | −0.008 | 0.562 | 256 | – | – |
| error-gated @ free (E) | 0.569 | 0.582 | 0.990 | 0.481 | 0.427 | 0.992 | 256 | 187 | – |
| E/E-cue0 @ paced100 (Z) | 0.610 | 0.587 | 0.886 | 0.315 | 0.295 | 0.857 | 255 | 118 | – |
| E + block (Gb) | 0.509 | 0.515 | 0.628 | 0.136 | 0.138 | 0.508 | 118 | 0 | 6180 blocked |
| E + block-recent2 (Gr) | 0.544 | 0.549 | 0.769 | 0.257 | 0.216 | 0.766 | 194 | 86 | 3311 blocked |
| E + charge4 (Gc) | 0.537 | 0.543 | 0.711 | 0.199 | 0.193 | 0.630 | 118 (250 charged) | 44 | – |
| E + last-resort (Gl) | 0.569 | 0.580 | 0.983 | 0.474 | 0.411 | 0.984 | 252 | 183 | 16 deferred |
| **Z + last-resort (Zg)** | **0.644** | **0.602** | 0.858 | 0.245 | 0.241 | 0.830 | 256 | 72 | 870 deferred |

k = 4 (own fixture): S1 0.578, R1 0.599, E 0.633 (FGT 0.449), Z 0.656
(FGT 0.215), Gb 0.581, Gr 0.579 (FGT 0.274), Gc 0.598, Gl 0.632,
**Zg 0.669** (FGT 0.143).

### 3. Hypotheses (k = 8, flip 50, held-out orders; Δ = mean [95 % CI] better/ties/worse over 40 seeds)

* **H1 fail (primary).** Gr vs E: the forgetting cut is large,
  +0.224 [+0.216, +0.232] 40/0/0, but learning drops by −0.221
  [−0.225, −0.218], beyond the 0.10 bound. ACC_all −0.025.
* **H2 fail.** Gb vs E: FGT cut +0.345, LEARN −0.362, ACC_all −0.060.
* **H3 fail.** Gc vs E: FGT cut +0.283, LEARN −0.279, ACC_all −0.032.
* **H4 fail.** ACC_all(Gr) − ACC_all(E) = −0.025 [−0.031, −0.020] 3/0/37.
* **H5 fail.** Gr vs S1: ACC_all −0.003 [−0.009, +0.003] 20/0/20, although
  Gr learns +0.130 more.
* **H6 fail.** Gr − R1 ACC_all = −0.031 [−0.038, −0.025].
* **H7 pass.** Zg vs Z: FGT cut +0.071 [+0.065, +0.078] 40/0/0 and ACC_all
  +0.034 [+0.029, +0.039] 40/0/0, for a learning cost of −0.028 and LAST
  −0.027.
* **H8 pass.** Forgetting E 0.481 > Gr 0.257 > Gb 0.136 and learning E
  0.990 > Gr 0.769 > Gb 0.628. All four pairwise differences are positive
  (40/0/0 each).
* **H9 pass (pre-registered null).** Gl − E ACC_all = −0.001
  [−0.001, +0.000]. Rationing overwrites does not bind on the uncued rule.
* **H10 fail.** k = 4, Gr vs E: FGT cut +0.175 (positive), but LEARN
  −0.186, beyond the 0.10 bound.

### 4. Secondary

* **On the uncued rule every guard trades forgetting for learning about
  1:1.** Gr buys 0.224 less forgetting for 0.221 less learning, Gb 0.345
  for 0.362 and Gc 0.283 for 0.279. The pattern is the same at k = 4 and
  at flips 25 / 100 (Gr − E ACC_all −0.052 at flip 25 and −0.009 at flip
  100). This repeats #39's 1:1 trade ([CONSOLIDATED_GRADED.md](CONSOLIDATED_GRADED.md))
  at k = 8 with protection limited to recently consolidated edges. The
  pilot gave the reason: most of E's overwrites are single-edge direct
  conflicts (≈ 0.4 active edges per target and pattern). Protecting the
  old value is the same as not learning the new one.
* **The guard pays on the silent cue** because the cue gives each task its
  own free edges. Zg is the best setting at equal writes at k = 8:
  ACC_all 0.644, which is +0.034 over Z, +0.075 over E, +0.069 over R1
  and +0.097 over S1. Zg vs S1 passes the sequence rule on all and on CN
  pairs. The Zg − Z gain holds at k = 4 (+0.013), at flip 25 (+0.020),
  at flip 100 (+0.029) and on the identity order (+0.031). From the k = 4
  prefix to k = 8 of the same runs it does not grow: +0.005
  [−0.002, +0.011].
* Gr's deficit to E shrinks with k: growth from the k = 4 prefix to k = 8
  is +0.032 [+0.026, +0.040]. At k = 8 Gr is level with S1 (−0.003).
* On CN pairs Gr − E is −0.033 and Zg − Z is +0.014 [+0.005, +0.025].
* E − S1 at k = 8 on these new seeds and orders is +0.023, which
  replicates #131's +0.020 (#131 H1 failure).

**Reading:** a consolidation guard on the original edges does cut the
error gate's forgetting at k = 8, by up to 0.35. Without spare capacity it
cuts learning by the same amount. None of the uncued guards beats E, S1 or
R1 on final accuracy (H1–H6 and H10 fail; the 1:1 trade shows in H8). The
guard only improves retention where spare capacity exists, on the silent
cue's task-private edges. There, Z + last-resort cuts forgetting by 0.07
for a 0.03 learning cost and gives the best equal-budget accuracy so far
(H7). It does not change how the error gate's lead scales with k.

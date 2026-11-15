# B888-4 cycle 23: with a partner rule that cannot fall below the task-id rule, does pairing shared cue lines by task agreement beat the cyclic code, private cues, a random partner and the supervised control at equal lines and equal write budget? (synthetic graph)

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
share-r codes of [CUE_CAPACITY.md](CUE_CAPACITY.md) (#134) and the agreement
matrix of [CUE_SIMILARITY.md](CUE_SIMILARITY.md) (#135). "Cue", "code",
"partner", "tour", "agreement", "guard", "task order" and "forgetting" name
parts of an update rule and an artificial benchmark on a Boolean graph with
4-bit weights. Nothing here is a claim about synapses, biology, insect memory
or BANC physiology.

Module: `src/cue_partner.rs`. CLI: `bench-cue-partner` (`--census` for the
pilot).

## Motivation (what #135 left open)

Cycle 22 (#135) paired the tasks that share reserved cue lines along a
greedy nearest-neighbour tour of the task-rule agreement matrix (matched),
against it (anti) or by task id (cyclic, = #134 share 2). Its manipulation
gate G2 required matched ≥ cyclic partner agreement **in every seed**; the
greedy tour is not optimal and fell below the cyclic tour at flip 25 in 8/40
seeds at k = 16 (by ≤ 0.022) and 4/40 at k = 24. G2 failed and H1–H11 were
declared not evaluable. Descriptively the matched code was +0.089 ACC_all
over the cyclic code at flip 100 (k = 16), and at flip 50 — pre-registered as
a null — it was +0.016 over cyclic and +0.055 over S1, so flip 50 was not a
null.

This cycle repairs the manipulation and re-asks the question on **new
seeds**, with the claims that #135 suggested as primary (flip 100 and
flip 50), plus a **random-partner** control that #135 lacked (the cyclic code
is a structured, not a neutral, reference: at flip 100 it pairs each task
with its exact inverse).

## Codes (pre-registered construction, no tuning)

* **Blocks** `B_0 … B_{k−1}`: the #131–#135 disjoint cues of the k-task
  fixture (`seq_generalisation::fixture`, cap ⌊64 / k⌋ lines). Pool excludes
  their union, as before.
* **Agreement** `a(i, j)`: exactly #135 (`cue_similarity::agreement`, fraction
  of usable pool edges on which tasks i and j have the same valence),
  computed once per (k, flip, seed) from the task rules before training.
* **Candidate tours** (deterministic, no search beyond this list): the cyclic
  tour 0, 1, …, k − 1 first, then the #135 greedy tour started at each task
  s = 0, …, k − 1 (from s, repeatedly move to the not-yet-visited task with the
  highest (matched) or lowest (anti) agreement with the current one, ties to
  the lowest id; the tour closes back to s).
* **Matched** = the candidate with the **highest** mean partner agreement
  `mean_j a(j, σ(j))`; **anti** = the candidate with the **lowest**; a later
  candidate replaces the current best only if it is better by more than
  1e-12 (so ties keep the cyclic tour, then the earliest start). Because the
  cyclic tour is a candidate, **matched ≥ cyclic ≥ anti in every seed by
  construction** (the #135 G2 failure cannot recur; G2 still checks it).
* **Random** (new control): one k-cycle drawn by Sattolo's algorithm from the
  #133 LCG seeded with `seed ^ 0x0DE5_B888_4C23_5A77 ^ k`; independent of the
  agreement, the flip and the order.
* Partner map `σ(tour[p]) = tour[(p + 1) mod k]`; code of task j
  `C_j = sorted(B_j ∪ B_{σ(j)})`. Every share-2 code (cyclic, matched, anti,
  random) is one k-cycle, so every reserved line is in exactly 2 codes, codes
  are pairwise distinct, each has the lines of 2 blocks and their union is
  the union of blocks — the #134 share-2 invariants. The cyclic code is
  bit-identical to #134 share 2 / #135 cyclic (tested).
* Cue-line edges start at 0 in every cued arm (silent cue, #128).

## Arms (10 rows per (k, flip, seed, order))

All rules are unchanged code paths of #132–#135
(`consolidation_guard::run_setting`); only the fixture's per-task cue lists
change.

* `supervised@paced100` (**S1**, control; on this teacher-clamped task the
  associative co-activation rule is the same update, see CONSOLIDATION.md).
* `reward@paced100` (**R1**, reward-modulated).
* `error-gated@free` (**E**, = its `paced100` run, gated in G1).
* `E/E-cue0@paced100` private blocks (**Z1**) and
  `E/E-cue0+last-resort@paced100` private blocks (**Zg1**).
* `E/E-cue0+last-resort@paced100` with the cyclic (**Zg2c**), matched
  (**Zg2m**), anti (**Zg2a**) and random (**Zg2r**) share-2 codes.
* `E/E-cue0@paced100` with the matched code (**Z2m**, no guard).

**Equal budget:** every capped arm uses the `paced100` per-episode cumulative
write curve of uncapped E in the same (k, flip, seed, order); cue writes are
charged. **Equal resources:** all cued arms of a (k, seed) use the same
reserved lines and the same cue-line edges; the share-2 arms also have the
same number of lines per task and the same line multiplicity.

## Design (pre-registered)

* Graph, task, pool, splits, budget: exactly #131–#135 (128 inputs / 32
  targets / fan-in 8, threshold 1, init weight 1, 4-bit weights, 6 active
  inputs, 32 training patterns per task, 8 rounds, one shared TEST set of 128
  patterns, pair classes seen / CN / ICN / UE, no rehearsal, original edges
  only, topology checked per row).
* **k = 8, 16, 24** (own fixtures). **Flips 25, 50, 100.** Flips 100 and 50
  carry the primary hypotheses; flip 25 (matched ≈ cyclic, see #135) is
  secondary only.
* **Seeds 521..560** (40 seeds, never used by an earlier B888-4 cycle).
* **Orders:** 0 = identity (secondary only); held-out 1 = reversed, 2..4 =
  seeded Fisher–Yates permutations (#133 generator) with a **new salt**
  `0x0DE5_B888_4C23_0A1D`, shared by every flip and arm. Partner maps do not
  depend on the order.
* **Metrics** as #131–#135 (stage coordinates): ACC, LEARN, FGT, LAST on all /
  CN pairs; per seed the mean over its 4 held-out orders (n = 40 paired
  seeds). Worst order: per seed, min over held-out orders of the paired
  difference.
* **Instrumentation per row:** code kind, partner map, mean partner agreement,
  conflicting links (a < 0.5), lines per task, mean cue reach per task,
  cue-line edges, live / full cue edges, cap, charged / effective / cue
  writes, refused, overwrites, deferred, net reversals, consolidated edges,
  saturated edges, init / paced / cap / code flags, weight digest, topology.
* **Checkpoint/resume:** the #132 engine checkpoint, re-tested bit-exact for
  Zg2m and Zg2r at k = 16-shaped small fixtures on the reversed order.
* **Reproducibility:** rows, CSV and report do not depend on the worker thread
  count (tested); the main run is repeated with a different thread count and
  must give a byte-identical CSV (sha256 reported).

Statistics as #123–#135: bootstrap 95 % percentile CI (2000 resamples, seed
0xB0075EED); better/ties/worse over 40 seeds.

* **strong:** mean ≥ 0.05, > 0 in ≥ 32 of 40 seeds and CI lower bound > 0.
* **positive:** mean > 0 and CI lower bound > 0.
* **sequence rule:** Δ ACC strong, Δ LAST ≥ −0.05 and Δ LEARN ≥ −0.05.

## Design gates (reported first)

* **G1 (integrity):** every row keeps its topology, starts from its documented
  initial weights, makes charged writes ≤ its allowance; `error-gated@paced100`
  is bit-identical to uncapped E with zero refusals; guard counters match
  guard and ledger; orders valid; codes valid (Z1/Zg1 = the blocks; share-2
  codes: σ one k-cycle, each line in exactly 2 codes, codes distinct, own
  block included; Zg2c = the #134 share-2 code; matched / anti = the
  pre-registered best / worst candidate tour; random = the pre-registered
  Sattolo cycle).
* **G2 (manipulation):** at k = 16 and k = 24: matched ≥ cyclic ≥ anti
  partner agreement in **every** seed at every flip; mean matched − anti
  ≥ 0.20 at flip 100 and ≥ 0.05 at flip 50; mean matched − random ≥ 0.20 at
  flip 100 and ≥ 0.02 at flip 50; Zg2m defers in ≥ 50 % of held-out rows at
  every (k, flip).
* **G3 (novel pairs):** median CN pairs per (seed, task) at k = 24, flip 50
  ≥ 20.

If any gate fails, H1–H13 are declared **not evaluable**.

## Hypotheses (held-out orders, seeds 521..560, k = 16 unless stated)

* **H1 (primary, mechanism):** flip 100, ACC_all(Zg2m) − ACC_all(Zg2c) is
  positive.
* **H2 (primary, vs control):** flip 100, Zg2m vs S1 passes the sequence rule
  on all pairs.
* **H3 (primary, mechanism):** flip 50, ACC_all(Zg2m) − ACC_all(Zg2c) is
  positive.
* **H4 (primary, vs control):** flip 50, Zg2m vs S1 passes the sequence rule
  on all pairs (retest of #133 H4 / #134 H1 / #135 H8).
* **H5:** flip 100, ACC_all(Zg2m) − ACC_all(Zg1) is positive.
* **H6:** flip 50, ACC_all(Zg2m) − ACC_all(Zg1) is positive.
* **H7 (the trade is broken, not just moved):** flip 100,
  FGT_all(Zg1) − FGT_all(Zg2m) is positive **and** LEARN_all(Zg2m) −
  LEARN_all(Zg1) has CI lower bound > −0.02.
* **H8 (random-partner control):** flip 100, ACC_all(Zg2m) − ACC_all(Zg2r) is
  positive.
* **H9 (random-partner control):** flip 50, ACC_all(Zg2m) − ACC_all(Zg2r) is
  positive.
* **H10 (k = 24):** flip 100, ACC_all(Zg2m) − ACC_all(Zg1) is positive.
* **H11 (worst order):** flip 100, per-seed worst held-out order
  ACC_all(Zg2m) − ACC_all(Zg2c) is positive.
* **H12 (not guard-specific):** flip 100, ACC_all(Z2m) − ACC_all(Z1) is
  positive.
* **H13 (anti control):** flip 50, ACC_all(Zg2m) − ACC_all(Zg2a) is positive.

**Secondary (reported, no pass/fail):** all flip 25 comparisons; all
comparisons at k = 8; k = 24 at flip 50; identity order; CN pairs; every arm
vs S1 / R1 / E per (k, flip); Zg2r vs Zg2c and Zg1; per-order breakdown;
agreement, conflicting links, reach, live cue edges, cue writes, overwrites
and deferred per code; how often the matched / anti tour is the cyclic tour.

No multiplicity correction is applied. The analysis is run once. Nothing
above the Results heading is changed after the main run except the "pending"
placeholder; the design-stage pilot section below is filled in before the
main run from the label-free census only (seeds 1..5: agreement, writes,
guard counters, reach; **no accuracy**).

## Design-stage pilot

`results/cue-partner-census-v1.txt` (seeds 1..5, identity order, flips 25 /
50 / 100, k = 8 / 16 / 24; partner agreement, writes, guard counters and cue
reach only; **no accuracy computed**). Ranges over the 5 seeds:

| k | flip | partner agreement Zg2c / Zg2m / Zg2a / Zg2r | conflicting links Zg2c / Zg2m / Zg2a / Zg2r | effective writes (every arm) | overwrites Zg2c / Zg2m / Zg2a / Zg2r |
|---|---|---|---|---|---|
| 8 | 25 | 0.70–0.73 / 0.70–0.73 / 0.45–0.53 / 0.54–0.67 | 1 / 0–1 / 4–6 / 0–4 | 120–156 | 14–23 / 14–21 / 9–18 / 11–20 |
| 8 | 50 | 0.49–0.54 / 0.55–0.58 / 0.42–0.47 / 0.49–0.54 | 3–4 / 0–3 / 6–8 / 3–5 | 208–258 | 37–52 / 33–53 / 26–37 / 34–51 |
| 8 | 100 | 0.00 / 0.75 / 0.00 / 0.25–0.75 | 8 / 2 / 8 / 2–6 | 412–498 | 86–111 / 64–84 / 86–111 / 72–90 |
| 16 | 25 | 0.73–0.76 / 0.73–0.76 / 0.44–0.47 / 0.54–0.58 | 0–1 / 0–1 / 10–14 / 1–7 | 227–288 | 93–127 / 93–122 / 85–119 / 94–129 |
| 16 | 50 | 0.48–0.53 / 0.58–0.60 / 0.40–0.43 / 0.47–0.52 | 3–11 / 0–1 / 14–16 / 6–10 | 423–515 | 215–260 / 221–250 / 196–239 / 198–241 |
| 16 | 100 | 0.00 / 0.88 / 0.00 / 0.25–0.75 | 16 / 2 / 16 / 4–12 | 787–1008 | 382–485 / 338–470 / 382–485 / 379–436 |
| 24 | 25 | 0.73–0.75 / 0.73–0.75 / 0.43–0.45 / 0.51–0.54 | 0–1 / 0–1 / 20–24 / 8–11 | 558–629 | 317–386 / 317–386 / 323–378 / 324–377 |
| 24 | 50 | 0.48–0.51 / 0.57–0.59 / 0.41–0.42 / 0.49–0.51 | 8–16 / 0–2 / 23–24 / 10–12 | 1036–1174 | 662–738 / 668–731 / 658–774 / 651–744 |
| 24 | 100 | 0.00 / 0.92 / 0.00 / 0.33–0.50 | 24 / 2 / 24 / 12–16 | 1826–2248 | 1190–1424 / 1142–1343 / 1190–1424 / 1163–1402 |

Readings stated before the main run (they do not change any hypothesis):

* **The ordering holds in every pilot seed:** matched ≥ cyclic ≥ anti at
  every (k, flip) (by construction). At flip 25 the matched tour is the
  cyclic tour in 4 of 5 pilot seeds at every k (so flip 25 stays secondary);
  at flips 50 and 100 it differs from the cyclic tour in every pilot seed.
* **Flip 100:** cyclic and anti coincide again (agreement 0, all k links
  conflicting), so Zg2a = Zg2c there; matched reaches 0.75 / 0.875 / 0.917
  (k = 8 / 16 / 24) as in #135. The random cycle sits in between
  (0.25–0.75 at k = 16): it pairs tasks of equal and opposite parity at
  random.
* **Flip 50:** matched ≈ 0.58–0.60, cyclic ≈ 0.48–0.53, random ≈ 0.47–0.52,
  anti ≈ 0.40–0.43 at k = 16; the pre-registered G2 gaps (m − a ≥ 0.05,
  m − r ≥ 0.02) are met in the pilot range, the flip 100 gaps (≥ 0.20) too.
* Writes stay equal across arms within a (k, flip, seed) (`paced100`); Zg2m
  defers in every pilot run. Nothing was tuned on the pilot.

## Command

```bash
cargo +1.89 run --release -- bench-cue-partner --csv results/cue-partner-v1.csv > results/cue-partner-report-v1.txt
# pilot: cargo +1.89 run --release -- bench-cue-partner --census > results/cue-partner-census-v1.txt
```

## Results

Run once after the pre-registration commit 1bf5a43, the harness + tests
commit 3bd4f75 and the pilot commit 939f9f6 had been pushed. The grid has
18000 rows: 3 k × 3 flips × 40 seeds × 5 orders × 10 arms, on 7 worker
threads (2757 s wall).
CSV sha256 `9bfa800b520aadaa44e57442972e4956118e5c050bddbbf79496144377a1cbba`.
A second, independent run with 5 worker threads (2662 s wall) gave a
byte-identical CSV (same sha256) and a byte-identical report.
Every row keeps the original topology (`topology_unchanged_all=true`).
Files: `results/cue-partner-v1.csv`, `results/cue-partner-report-v1.txt`,
pilot `results/cue-partner-census-v1.txt`. All numbers are on the
**GENERATED_SYNTHETIC** graph with artificial labels, not on BANC.

### 1. Design gates (reported first)

* **G1 passes:** topology, initial weights, allowances, `paced100` identity of
  E, guard/ledger consistency, orders and codes (Z1/Zg1 = the blocks; share-2
  codes: σ one k-cycle, every line in exactly 2 codes, codes distinct, own
  block included; Zg2c = the #134 share-2 code; matched / anti = the
  pre-registered best / worst candidate tour; random = the Sattolo cycle) hold
  in every row.
* **G2 passes.** matched ≥ cyclic ≥ anti partner agreement in 40/40 seeds at
  every (k, flip) (also at k = 8). Mean gaps (k = 16 / k = 24): matched − anti
  0.875 / 0.917 at flip 100 (≥ 0.20) and 0.167 / 0.158 at flip 50 (≥ 0.05);
  matched − random 0.388 / 0.440 at flip 100 (≥ 0.20) and 0.082 / 0.079 at
  flip 50 (≥ 0.02). Zg2m defers in 100 % of held-out rows at every (k, flip).
  The matched tour is the cyclic tour in 28/40 seeds at k = 16, flip 25 (37/40
  at k = 24; 33/40 at k = 8) and in 0/40 seeds at flips 50 and 100.
* **G3 passes:** median CN pairs per (seed, task) at k = 24, flip 50 is 110
  (61 at k = 8, 64 at k = 16).

All gates pass, so H1–H13 are evaluated.

### 2. Arms, held-out orders (ACC_all, mean over 40 seeds × 4 orders)

| k | flip | S1 | R1 | E | Z1 | Zg1 | Zg2c | **Zg2m** | Zg2a | Zg2r | Z2m |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 25 | 0.598 | 0.631 | 0.653 | 0.679 | 0.702 | 0.688 | 0.688 | 0.652 | 0.671 | 0.683 |
| 8 | 50 | 0.546 | 0.565 | 0.570 | 0.606 | 0.639 | 0.606 | 0.619 | 0.601 | 0.610 | 0.607 |
| 8 | 100 | 0.509 | 0.512 | 0.521 | 0.570 | 0.601 | 0.496 | **0.631** | 0.496 | 0.583 | 0.603 |
| 16 | 25 | 0.562 | 0.593 | 0.584 | 0.600 | 0.610 | 0.622 | 0.622 | 0.600 | 0.609 | 0.611 |
| 16 | 50 | 0.533 | 0.547 | 0.542 | 0.565 | 0.576 | 0.573 | **0.589** | 0.567 | 0.577 | 0.572 |
| 16 | 100 | 0.507 | 0.511 | 0.521 | 0.546 | 0.559 | 0.508 | **0.602** | 0.508 | 0.566 | 0.584 |
| 24 | 25 | 0.546 | 0.580 | 0.566 | 0.572 | 0.575 | 0.579 | 0.579 | 0.571 | 0.574 | 0.575 |
| 24 | 50 | 0.522 | 0.539 | 0.533 | 0.541 | 0.545 | 0.545 | 0.551 | 0.542 | 0.546 | 0.545 |
| 24 | 100 | 0.507 | 0.513 | 0.524 | 0.532 | 0.534 | 0.516 | **0.553** | 0.516 | 0.537 | 0.549 |

Equal budget holds: within a (k, flip) the S1 / E / Zg arms make the same
effective writes up to 0.5 % (e.g. k = 16, flip 100: 630.7–633.1 per run;
k = 24, flip 100: 1365.3–1367.6); the unguarded Z1 / Z2m arms stay within the
same allowance and use slightly fewer (k = 16, flip 100: 623.7 / 616.9). At
flip 100 the anti code coincides with the cyclic code (Zg2a = Zg2c, as stated
in the pilot).

### 3. Pre-registered hypotheses (held-out orders, k = 16 unless stated)

Δ = mean [95 % CI] better/ties/worse over 40 seeds.

| H | comparison | Δ | verdict |
|---|---|---|---|
| H1 | flip 100, ACC_all Zg2m − Zg2c | +0.094 [+0.087, +0.101] 40/0/0 | **pass** |
| H2 | flip 100, Zg2m vs S1 sequence rule | ACC +0.095 [+0.088, +0.101] 40/0/0; LEARN +0.188; LAST +0.203 | **pass** |
| H3 | flip 50, ACC_all Zg2m − Zg2c | +0.015 [+0.011, +0.020] 37/0/3 | **pass** |
| H4 | flip 50, Zg2m vs S1 sequence rule | ACC +0.055 [+0.051, +0.060] 40/0/0; LEARN +0.238; LAST +0.256 | **pass** |
| H5 | flip 100, ACC_all Zg2m − Zg1 | +0.043 [+0.038, +0.047] 40/0/0 | **pass** |
| H6 | flip 50, ACC_all Zg2m − Zg1 | +0.013 [+0.010, +0.016] 36/0/4 | **pass** |
| H7 | flip 100, FGT_all Zg1 − Zg2m **and** LEARN_all Zg2m − Zg1 CI lower bound > −0.02 | FGT +0.105 [+0.100, +0.110] 40/0/0; LEARN −0.055 [−0.060, −0.051] 0/0/40 | **fail** (LEARN clause) |
| H8 | flip 100, ACC_all Zg2m − Zg2r | +0.036 [+0.029, +0.042] 38/0/2 | **pass** |
| H9 | flip 50, ACC_all Zg2m − Zg2r | +0.012 [+0.008, +0.015] 33/0/7 | **pass** |
| H10 | k = 24, flip 100, ACC_all Zg2m − Zg1 | +0.019 [+0.017, +0.021] 40/0/0 | **pass** |
| H11 | flip 100, worst held-out order Zg2m − Zg2c | +0.067 [+0.059, +0.074] 40/0/0 | **pass** |
| H12 | flip 100, ACC_all Z2m − Z1 | +0.038 [+0.034, +0.042] 40/0/0 | **pass** |
| H13 | flip 50, ACC_all Zg2m − Zg2a | +0.022 [+0.018, +0.026] 38/0/2 | **pass** |

12 of 13 pass; H7 fails. No multiplicity correction (as pre-registered).
Most flip 50 effects are small (+0.012 to +0.022 ACC_all) even where the CI
excludes 0.

### 4. Secondary (descriptive, no pass/fail)

* **The learning cost of sharing is not removed (H7).** At k = 16, flip 100
  matched sharing cuts forgetting against private cues (FGT_all 0.229 vs
  0.334) but still learns less (LEARN_all 0.817 vs 0.872, −0.055 in 40/40
  seeds); the net ACC gain over Zg1 (+0.043) is the forgetting cut minus that
  learning loss. Same direction at flip 50 (LEARN −0.060, FGT +0.078).
* **The random partner sits between cyclic and matched at flip 100**
  (k = 16: Zg2c 0.508 < Zg2r 0.566 < Zg2m 0.602; Zg2r − Zg2c +0.058 40/0/0),
  in line with its partner agreement (0.000 / 0.487 / 0.875). At flip 50 the
  random partner is close to cyclic (+0.004 [+0.000, +0.007]), and Zg2r is not
  above private cues at flip 50 (Zg2r − Zg1 +0.001 [−0.002, +0.004]).
* **Flip 25: the matched code is mostly the cyclic code** (identical in 28/40
  seeds at k = 16); Zg2m − Zg2c = −0.000 [−0.002, +0.001] 6/28/6. Anti and
  random codes are worse there (Zg2m − Zg2a +0.022, Zg2m − Zg2r +0.013).
* **k = 8: private cues still win at flips 25 and 50** (Zg1 0.702 / 0.639 vs
  Zg2m 0.688 / 0.619); at flip 100 Zg2m is level with Zg1 in the worst
  held-out order (+0.004 [−0.001, +0.009]). Sharing pays only as k grows or
  conflict grows.
* **k = 24, flip 50 (secondary):** Zg2m − Zg2c is small but positive on the
  mean (0.551 vs 0.545), Zg2m − S1 = 0.029 ACC_all: the 0.05 lead over S1 at
  flip 50 is reached at k = 16 only.
* **Worst held-out order at flip 50 (secondary):** per seed, the minimum over
  the 4 held-out orders of Zg2m − Zg2c is −0.004 [−0.010, +0.002] at k = 16 and
  −0.004 [−0.006, −0.001] at k = 24: the flip 50 gain over the cyclic code is
  not robust to the worst order, unlike flip 100 (+0.067 / +0.025, 40/0/0).
* **Identity order and CN pairs** follow the held-out direction (k = 16,
  flip 100: identity Zg2m − Zg2c +0.098; CN pairs +0.067).
* **Limitation:** partner agreement is an oracle read from the task rules
  before training, not something the learner measures; every number is on the
  synthetic graph and the artificial OR-rule task.

**Reading:** with a partner rule that cannot fall below the task-id rule, all
three design gates pass and 12 of the 13 pre-registered hypotheses pass on new
seeds 521..560. On this synthetic benchmark, sharing reserved cue lines with
the task that agrees most beats the cyclic code, private cues, a random
partner and the supervised control at equal lines and equal write budget, at
flip 100 (large: +0.094 over cyclic at k = 16) and at flip 50 (small: +0.015
over cyclic, +0.055 over S1). It does not remove the learning cost of sharing
(H7 fails: LEARN −0.055 vs private cues at flip 100), and the flip 50 gain over
the cyclic code vanishes in the worst held-out order (secondary). Nothing here
is a claim about biology or about the BANC graph.

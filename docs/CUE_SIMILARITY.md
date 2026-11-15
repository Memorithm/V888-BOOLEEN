# B888-4 cycle 22: does choosing *which* tasks share cue lines by task agreement turn the 1:1 learning/forgetting trade of shared cues (#134) into a net gain over private cues and over the supervised control, at equal lines and equal write budget? (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123), with the k-task fixture of
[SEQ_GENERALISATION.md](SEQ_GENERALISATION.md) (#131), the consolidation guard
engine of [CONSOLIDATION_GUARD.md](CONSOLIDATION_GUARD.md) (#132), the
order/statistics protocol of [CUE_ROBUSTNESS.md](CUE_ROBUSTNESS.md) (#133) and
the share-r codes of [CUE_CAPACITY.md](CUE_CAPACITY.md) (#134). "Cue", "code",
"partner", "agreement", "guard", "task order" and "forgetting" name parts of
an update rule and an artificial benchmark on a Boolean graph with 4-bit
weights. Nothing here is a claim about synapses, biology, insect memory or
BANC physiology.

Module: `src/cue_similarity.rs`. CLI: `bench-cue-similarity` (`--census` for
the pilot).

## Motivation (what #134 left open)

Cycle 21 (#134) gave each task more cue lines by sharing the same reserved
lines between tasks (share-r codes, cyclic by task id). At k = 16 and k = 24
sharing cut forgetting (H3 pass) but paid for it ~1:1 in learning, so net
accuracy did not move and the lead of the silent cue + last-resort guard
(**Zg**) over the supervised control (S1) stayed below 0.05 (H1 fail). A
**secondary** reading was that the cost tracks label conflict between the
tasks that share a line: share 2 helped at flip 25 and hurt a lot at
flip 100.

That reading is post hoc and confounded with the flip level. The task
generator (`task_sequence::build_sequence`) makes task i+1 from task i by
flipping each edge valence with probability flip %, so the **cyclic** share-2
code (task j shares with task j+1) pairs each task with a similar task at
flip 25, with an unrelated task at flip 50 and with the exact **opposite**
task at flip 100 (where task j and task j+2 are identical). This cycle tests
the reading causally: within each (k, flip, seed) everything is fixed — graph,
reserved lines, blocks, pattern pool, tasks, initial weights, write budget,
line multiplicity (every reserved line in exactly 2 codes) — and only the
**choice of the partner** whose block a task borrows changes: chosen to
agree most with the task (matched), least (anti) or by task id as in #134
(cyclic).

The partner choice uses the **task rules** (edge valences) before training.
It is an *oracle task-similarity* descriptor, not something the learner
measures; a learner-side estimate is out of scope for this cycle and is
stated as a limitation.

## Codes (pre-registered construction, no tuning)

* **Blocks** `B_0 … B_{k−1}`: the #131–#134 disjoint cues of the k-task fixture
  (`seq_generalisation::fixture`, cap ⌊64 / k⌋ lines). Pool excludes their
  union, as before.
* **Agreement** `a(i, j)` = fraction of usable edges leaving pattern-pool lines
  on which tasks i and j have the same valence (`task.valence`). Computed once
  per (k, flip, seed); independent of the order and of the arm.
* **Tours** (deterministic, no search): start at task 0, repeatedly move to the
  not-yet-visited task with the highest (matched) or lowest (anti) agreement
  with the current task, ties to the lowest task id; the tour closes back to
  task 0. The cyclic tour is 0, 1, …, k − 1 (= #134).
* **Partner map** `σ(tour[p]) = tour[(p + 1) mod k]`; **code of task j**
  `C_j = sorted(B_j ∪ B_{σ(j)})`. Each task keeps its own block and borrows its
  successor's. Since σ is one k-cycle (k ≥ 3), every reserved line is in exactly
  2 codes, codes are pairwise distinct, every code has the lines of 2 blocks and
  the union of codes is the union of blocks — the #134 share-2 invariants. The
  cyclic code is bit-identical to the #134 share-2 code (tested).
* Cue-line edges start at 0 in every cued arm (silent cue, #128).

## Arms (9 rows per (k, flip, seed, order))

All rules are unchanged code paths of #132–#134
(`consolidation_guard::run_setting`); only the fixture's per-task cue lists
change.

* `supervised@paced100` (**S1**, control; on this teacher-clamped task the
  associative co-activation rule is the same update, see CONSOLIDATION.md).
* `reward@paced100` (**R1**, reward-modulated).
* `error-gated@free` (**E**, = its `paced100` run, gated in G1).
* `E/E-cue0@paced100` private blocks (**Z1**) and
  `E/E-cue0+last-resort@paced100` private blocks (**Zg1**) — the #133 cue.
* `E/E-cue0+last-resort@paced100` with the cyclic share-2 code (**Zg2c**, = #134
  Zg2), the matched code (**Zg2m**) and the anti code (**Zg2a**).
* `E/E-cue0@paced100` with the matched code (**Z2m**, no guard).

**Equal budget:** every capped arm uses the `paced100` per-episode cumulative
write curve of uncapped E in the same (k, flip, seed, order); cue writes are
charged. **Equal resources:** all cued arms of a (k, seed) use the same reserved
lines and the same cue-line edges; the share-2 arms also have the same number
of lines per task and the same line multiplicity.

## Design (pre-registered)

* Graph, task, pool, splits, budget: exactly #131–#134 (128 inputs / 32 targets
  / fan-in 8, threshold 1, init weight 1, 4-bit weights, 6 active inputs, 32
  training patterns per task, 8 rounds, one shared TEST set of 128 patterns,
  pair classes seen / CN / ICN / UE, no rehearsal, original edges only,
  topology checked per row).
* **k = 8, 16, 24** (own fixtures). **Flips 25, 50, 100.** In this cycle the flip
  level is the factor that sets how much partner choice can matter, so flips
  25 and 100 carry hypotheses and flip 50 (tasks ≈ unrelated, a(i, j) ≈ 0.5
  for every pair) is the pre-registered **null** condition.
* **Seeds 481..520** (40 seeds, never used by an earlier B888-4 cycle).
* **Orders:** 0 = identity (secondary only); held-out 1 = reversed, 2..4 =
  seeded Fisher–Yates permutations (#133 generator) with a **new salt**
  `0x0DE5_B888_4C22_0A1D`, shared by every flip and arm. Partner maps do not
  depend on the order.
* **Metrics** as #131–#134 (stage coordinates): ACC, LEARN, FGT, LAST on all /
  CN pairs; per seed the mean over its 4 held-out orders (n = 40 paired seeds).
  Worst order: per seed, min over held-out orders of the paired difference.
* **Instrumentation per row:** code kind, mean partner agreement
  (mean_j a(j, σ(j))), conflicting links (a < 0.5), lines per task, mean cue reach
  per task, cue-line edges, live / full cue edges, cap, charged / effective /
  cue writes, refused, overwrites, deferred, net reversals, consolidated edges,
  saturation, weight digest, `init_ok`, `paced_identity`, `codes_ok`, topology.
* **Checkpoint/resume:** the #132 engine checkpoint, re-tested bit-exact for
  Zg2m at k = 16 on the reversed order.
* **Reproducibility:** rows, CSV and report do not depend on the worker thread
  count (tested); the main run is repeated with a different thread count and
  must give a byte-identical CSV (sha256 reported).

Statistics as #123–#134: bootstrap 95 % percentile CI (2000 resamples, seed
0xB0075EED); better/ties/worse over 40 seeds.

* **strong:** mean ≥ 0.05, > 0 in ≥ 32 of 40 seeds and CI lower bound > 0.
* **positive:** mean > 0 and CI lower bound > 0.
* **sequence rule:** Δ ACC strong, Δ LAST ≥ −0.05 and Δ LEARN ≥ −0.05.

## Design gates (reported first)

* **G1 (integrity):** every row keeps its topology, starts from its documented
  initial weights, makes charged writes ≤ its allowance; `error-gated@paced100`
  is bit-identical to uncapped E with zero refusals; guard counters match guard
  and ledger; orders valid; codes valid (Z1/Zg1 = the blocks; share-2 codes:
  σ one k-cycle, each line in exactly 2 codes, codes distinct, own block
  included; Zg2c = the #134 share-2 code).
* **G2 (manipulation):** at k = 16 and k = 24, mean partner agreement
  matched − anti ≥ 0.20 at flip 100 and ≥ 0.05 at flip 25 (seed means), and
  matched ≥ cyclic in every seed at flips 25 and 100; Zg2m defers in ≥ 50 % of
  held-out rows at every (k, flip).
* **G3 (novel pairs):** median CN pairs per (seed, task) at k = 24, flip 50
  ≥ 20.

If any gate fails, H1–H11 are declared **not evaluable**.

## Hypotheses (held-out orders, seeds 481..520, k = 16 unless stated)

* **H1 (primary, mechanism):** flip 100, ACC_all(Zg2m) − ACC_all(Zg2c) is
  positive.
* **H2 (primary, vs control):** flip 100, Zg2m vs S1 passes the sequence rule
  on all pairs.
* **H3 (matched sharing beats private cues):** flip 100, ACC_all(Zg2m) −
  ACC_all(Zg1) is positive.
* **H4 (the trade is broken, not just moved):** flip 100, FGT_all(Zg1) −
  FGT_all(Zg2m) is positive **and** LEARN_all(Zg2m) − LEARN_all(Zg1) has CI
  lower bound > −0.02.
* **H5 (anti control):** flip 25, ACC_all(Zg2m) − ACC_all(Zg2a) is positive.
* **H6:** flip 25, ACC_all(Zg2m) − ACC_all(Zg1) is positive.
* **H7 (null at flip 50):** ACC_all(Zg2m) − ACC_all(Zg2c) has its 95 % CI inside
  [−0.01, +0.01].
* **H8 (retest of #133 H4 / #134 H1 at flip 50):** Zg2m vs S1 passes the
  sequence rule on all pairs.
* **H9 (k = 24):** flip 100, ACC_all(Zg2m) − ACC_all(Zg1) is positive.
* **H10 (worst order):** flip 100, per-seed worst held-out order
  ACC_all(Zg2m) − ACC_all(Zg2c) is positive.
* **H11 (not guard-specific):** flip 100, ACC_all(Z2m) − ACC_all(Z1) is
  positive.

**Secondary (reported, no pass/fail):** all comparisons at k = 8; identity
order; CN pairs; every arm vs S1 / R1 / E per (k, flip); Zg2a vs Zg2c; per-order
breakdown; agreement, conflicting links, reach, live cue edges, cue writes,
overwrites and deferred per code.

No multiplicity correction is applied. The analysis is run once. Nothing
above the Results heading is changed after the main run except the "pending"
placeholder; the design-stage pilot section below is filled in before the
main run from the label-free census only (seeds 1..5: agreement, writes,
guard counters, reach; **no accuracy**).

## Design-stage pilot

Pending (label-free census, seeds 1..5).

## Command

```bash
cargo +1.89 run --release -- bench-cue-similarity --csv results/cue-similarity-v1.csv > results/cue-similarity-report-v1.txt
# pilot: cargo +1.89 run --release -- bench-cue-similarity --census > results/cue-similarity-census-v1.txt
```

## Results

Pending.

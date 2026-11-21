# B888-4 cycle 32: is the cycle 31 **fan-in** learning loss caused by **untrained edges** in the TEST pairs? An edge-coverage manipulation at equal presentations (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not built on the
build machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123), run through the **unchanged** cycle 29
engine of [CUE_DOSE_GUARD.md](CUE_DOSE_GUARD.md) (#188: the cycle 27
last-resort arms plus the five `block`-guard dose arms), exactly as in
cycles 30–31 ([CUE_DOSE_DENSITY.md](CUE_DOSE_DENSITY.md) #189,
[CUE_FANIN.md](CUE_FANIN.md) #190). "Cue", "dose", "guard", "fan-in",
"coverage", "retention" and "forgetting" name parts of an update rule and an
artificial benchmark on a Boolean graph with 4-bit weights. Nothing here is a
claim about synapses, biology, insect memory or BANC physiology.

Module: `src/cue_coverage.rs`. CLI: `bench-cue-coverage` (`--census` for the
label-free pilot).

## 1. Where cycle 31 left off (#190, main `96b2a77d`)

At k = 16, flip 100 (held-out orders, seeds 841..880, synthetic):

* fan-in lowers learning by itself at 128 sources: uncapped E LEARN_head
  0.987 → 0.957 → 0.909 at fan-in 8 → 16 → 32 (f32 − base −0.078, 0/0/40);
  the cycle 30 loss replicates (d512 − src512 −0.193); fan-in and sources
  interact (H5/H6 failed, interaction −0.115);
* the raw gap FGT(S1) − FGT(Zgd4) closes at f32 only because learning
  collapses (H8), the retention term also gets worse (H9 failed), and the
  block guard Zbd4 forgets **more** than S1 at f32 (H10 failed, −0.026);
* declared confound: the `paced100` write budget grows with the edge count;
* **no mechanism** was proposed for why fan-in lowers learning (the
  label-free target-load census did not explain it).

## 2. The question and the mechanism it tests (stated before measuring)

Under the OR rule (threshold 1, `init_weight` 1) a (pattern, target) pair
with `m` active edges is negative iff all `m` edges are negative, so a
correct negative needs **every** one of its `m` edges to have been written
down in the current task, whereas a positive is already predicted when one
positive edge kept its weight. With 6 active inputs out of 128 sources the
expected number of active edges into a target grows with fan-in (≈ 0.375 at
fan-in 8, ≈ 1.5 at fan-in 32). The **coverage hypothesis**: at high fan-in
more TEST pairs contain an edge that no training pattern of the current
task activated ("UnseenEdge" pairs of `novel_split::TaskSplit::classify`),
and these pairs, not fan-in as such, are what lowers LEARN_head.

The manipulation raises edge coverage **without changing the graph, the
fan-in or the number of presentations**: 4× the distinct training patterns
per task (32 → 128) with 1/4 of the rounds (8 → 2), i.e. 256 presentations
per task in every condition. If coverage is the mechanism, the fan-in loss
should shrink (H1) and largely vanish (H2), and the gain should be larger at
fan-in 32 than at fan-in 8 (H3). This is **not** a re-test of fan-in vs
dilution.

## 3. Arms, conditions, seeds, orders

* **Arms (unchanged cycle 29 engine, `cue_dose_guard::run_with`,
  bit-for-bit):** S1 (supervised control), R1, E, Zg1, Zgd0..Zgd4, Zgd2c
  and Zbd0..Zbd4 (same matched share-2 code, dose masks and per-stage
  `paced100` caps). **Equal budget:** every arm of a (seed, order) has the
  same per-stage write caps (the cumulative write curve of uncapped E of that
  (seed, order)); S1 is the supervised control at that budget.
* **Conditions** (sources 128, targets 32, active 6, 128 TEST patterns,
  k = 16, flip 100 everywhere):

  | cond | fan-in | train patterns / task | rounds / task | presentations / task | note |
  |---|---|---|---|---|---|
  | base | 8 | 32 | 8 | 256 | cycle 27 design, new seeds |
  | f32 | 32 | 32 | 8 | 256 | cycle 31 condition, new seeds |
  | basec | 8 | **128** | **2** | 256 | **new**, coverage control at fan-in 8 |
  | f32c | 32 | **128** | **2** | 256 | **new**, coverage at fan-in 32 |

  `Cond::BaseC` and `Cond::F32C` are new variants of `cue_dose_guard::Cond`
  (`COVERAGE_FACTOR` = 4); they are **not** added to `cue_dose_guard::CONDS`,
  `cue_dose_density::CONDS` or `cue_fanin::CONDS`, so the earlier grids and
  CLIs are unchanged (tested). Exactly the fields `spec.train_patterns`,
  `spec.rounds_a`, `spec.rounds_b` differ between base/basec and f32/f32c.
* **Seeds 881..920** (40 seeds, never used by an earlier B888-4 cycle).
  Pilot seeds 1..5.
* **Orders:** identity (secondary only), reversed, and 3 seeded
  Fisher–Yates permutations with the cycle 27 order salt
  `0x0DE5_B888_4C27_0A1D` (same declared reuse as cycles 28–31).
* **Declared confounds.**
  1. The training patterns, the TEST set and the edge valences are drawn
     from one pattern/label stream per seed (`task_sequence::build_sequence`);
     with 128 training patterns per task the TEST set and the valences of a
     seed are therefore **redrawn** (same distribution, same graph). As in
     cycles 30–31, cross-condition contrasts pair different fixtures by seed
     value; this is a fixed procedure, not a claim that they are matched.
  2. The manipulation raises coverage **and** the number of distinct
     training patterns, and lowers repetitions per pattern (8 → 2). It also
     raises the share of TEST pairs whose exact configuration was trained
     ("Seen" pairs; reported in the census). ACC_cn (compositional novel
     pairs) is reported for that reason.
  3. Same as cycles 30–31: the `paced100` budget is the write curve of
     uncapped E, so it differs between conditions. Within a condition every
     arm keeps the same caps.

## 4. Metrics and statistics

* Exactly as cycles 29–31: ACC, LEARN, FGT, LAST on all / CN (novel) pairs,
  balanced accuracy; per seed the mean over the 4 held-out orders (n = 40
  paired seeds); LEARN_head, RET_head as cycle 29 (FGT = LEARN_head −
  RET_head). **Generalisation** = ACC_all on the shared TEST set (never
  trained) and ACC_cn on its novel-pair subset; **forgetting** = raw FGT_all.
* **Learning measure (H1–H4):** LEARN_head of uncapped E, as cycles 30–31.
* **Label-free edge coverage (H5, pilot and main seeds):** over every task
  j and every TEST (pattern, reached target) pair classified against task
  j's training set, the share of pairs that are **not** UnseenEdge (every
  active edge occurs in some training pattern of task j). Also reported:
  mean active edges per pair `m`, share with `m ≥ 2`, Seen share. Graph and
  pattern inputs only (no label, no learning, no accuracy).
* **Cross-condition contrasts:** per seed value s, f_b(s) − f_a(s).
  **Interaction (H3):** per seed [f(f32c) − f(f32)] − [f(basec) − f(base)].
* **Statistics** as #123–#190: bootstrap 95 % percentile CI (2000
  resamples, seed 0xB0075EED); better/ties/worse over 40 seeds.
  * positive: mean > 0 and CI lower bound > 0;
  * negative: CI upper bound < 0;
  * non-inferior: CI lower bound > −0.02.
* **Reproducibility (declared plan):** main run `--threads 8`; `f32c` is
  re-run with `--threads 5` and its rows must be byte-identical to the main
  CSV's `f32c` rows (sha256 reported). Other conditions are not re-run
  unless time allows (declared in the results if so).
* **Checkpoint/resume:** no new engine path; the #132 checkpoint
  (bit-exact mid-stage save/load, tested at fan-in 32 by cycle 30) is
  unchanged. A whole run is deterministic, so a killed run is resumed by
  re-running it (or the remaining conditions with `--conds`, concatenating
  the per-condition CSV rows, header once) with the same seeds.

## 5. Design gates (per condition, reported first)

The cycle 29 gates G1–G5, unchanged (`cue_dose_guard::gates`). If G1, G2a,
G2b, G3 or G4 fails in a condition, every hypothesis using that condition is
**not evaluable**. G2c and G5 are reported; no hypothesis uses them.
**H5 is a manipulation check:** if it fails, H1–H3 are **not evaluable**.

## 6. Hypotheses (k = 16, flip 100, held-out orders, seeds 881..920)

LEARN = LEARN_head(E).

| H | statement | criterion | expectation |
|---|---|---|---|
| H1 **(primary)** | coverage restores learning at fan-in 32 | LEARN f32c − f32 positive | open |
| H2 | coverage removes the fan-in loss | LEARN f32c − basec non-inferior | open |
| H3 | coverage helps more at fan-in 32 than at 8 | [(f32c − f32) − (basec − base)] LEARN positive | open |
| H4 | cycle 31 H1 replicates | LEARN f32 − base negative | pass |
| H5 | manipulation check (label-free) | covered share f32c − f32 positive **and** basec − base positive | pass |
| H6 | the Zgd4 learning advantage recovers | [LEARN_head(Zgd4) − LEARN_head(S1)] f32c − f32 positive | open |
| H7 | the Zgd4 retention advantage recovers | [RET_head(Zgd4) − RET_head(S1)] f32c − f32 positive | open |
| H8 | supervised control at f32c: block guard | FGT(S1) − FGT(Zbd4) non-inferior **and** ACC_all(Zbd4) − ACC_all(S1) positive | open |
| H9 | generalisation at f32c vs S1 | ACC_all(Zgd4) − ACC_all(S1) and ACC_cn(Zgd4) − ACC_cn(S1) both positive | open |
| H10 | cycle 31 H10 failure replicates at f32 | FGT(S1) − FGT(Zbd4) negative at f32 | pass |
| H11 | dose effect at f32c | Zgd0..4 slope of ACC_all positive | open |

**Pre-registered reading rules.**

1. H5 pass, H1 pass, H2 pass: in this benchmark the fan-in learning loss is
   explained by untrained edges in TEST pairs (coverage), within the
   declared confounds.
2. H5 pass, H1 pass, H2 fail: coverage explains part of the fan-in loss; a
   residual fan-in effect remains.
3. H5 pass, H1 fail: coverage does **not** explain the fan-in loss; the
   negative is published.
4. A gap closure without H6/H7 is not read as an improvement. A gain of
   ACC_all that is not matched by ACC_cn is reported as possibly due to
   Seen pairs (confound 2).

## 7. Label-free pilot (seeds 1..5)

*(Filled in after the pilot; no accuracy is computed by `--census`.)*

## 8. Results (seeds 881..920)

*(Filled in after the main run.)*

## Reproduce

```bash
cargo +1.89 run --release -- bench-cue-coverage --census > results/cue-coverage-census-v1.txt
cargo +1.89 run --release -- bench-cue-coverage --threads 8 --csv results/cue-coverage-v1.csv > results/cue-coverage-report-v1.txt
```

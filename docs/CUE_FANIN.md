# B888-4 cycle 31: does **fan-in** lower learning **by itself** at a fixed number of graph sources, and is the cycle 30 learning collapse a fan-in effect rather than a density effect? (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not built on the
build machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123), run through the **unchanged** cycle 29
engine of [CUE_DOSE_GUARD.md](CUE_DOSE_GUARD.md) (#188: the cycle 27
last-resort arms plus the five `block`-guard dose arms), exactly as in
cycle 30 ([CUE_DOSE_DENSITY.md](CUE_DOSE_DENSITY.md), #189). "Cue", "dose",
"guard", "fan-in", "load", "retention" and "forgetting" name parts of an
update rule and an artificial benchmark on a Boolean graph with 4-bit
weights. Nothing here is a claim about synapses, biology, insect memory or
BANC physiology.

Module: `src/cue_fanin.rs`. CLI: `bench-cue-fanin` (`--census` for the
label-free pilot).

## 1. Where cycle 30 left off (#189, main `04eff3ac`)

At k = 16, flip 100 (held-out orders, seeds 801..840, synthetic):

* cycle 30 asked whether the cycle 29 H3 failure (the raw gap
  FGT(S1) − FGT(Zgd4) widens with more sources) is caused by edge dilution,
  by scaling fan-in with sources to keep 2 edges per source (d256 256/16,
  d512 512/32);
* **H1 (manipulation check) failed, strongly**: uncapped E LEARN_head fell
  0.987 (base 128/8) → 0.844 (src512 512/8) → 0.803 (d256) → **0.650**
  (d512); d512 − src512 = −0.193 [−0.197, −0.190] 0/0/40. Dilution is not
  supported;
* the raw gap "closed" at d512 (−0.005) only because the arms stopped
  learning: Zgd4's learning advantage over S1 fell +0.145 → +0.024 and
  ACC_all of Zgd4 fell 0.604 → 0.529 (**H6 failed**: the retention term did
  not improve, −0.007); **H9 failed** (Zbd4 vs S1 on novel pairs at d512,
  +0.001); the dose slope stayed linear but shrank to +0.002;
* the open question it left: **why does fan-in 32 learn less?** Cycle 30
  changed fan-in and sources together, so it cannot say whether the loss
  comes from fan-in, from sources, or from their combination.

## 2. The question (stated before measuring)

Does raising the fan-in at a **fixed** number of sources (128) lower
learning by itself, and does the cycle 30 learning loss decompose into a
fan-in effect and a source-count effect? This is **not** a re-test of
dilution: at 128 sources raising the fan-in raises the density (2 → 4 → 8
edges per source), so if fan-in lowers learning here, density cannot be
the explanation.

A 2 × 2 design (sources 128 / 512 × fan-in 8 / 32) plus the midpoint
fan-in 16 at 128 sources. Exactly one design field (`shape.fan_in`)
differs between `base`, `f16` and `f32`; exactly one (`shape.sources`)
between `base`/`src512` and between `f32`/`d512`.

A second, conditional question: if fan-in lowers learning at fixed sources,
does the raw forgetting gap then "close" again for the same reason as in
cycle 30 (a learning collapse, not a retention gain)? H7–H9 test that
directly, so a closure cannot be read as an improvement.

## 3. Arms, conditions, seeds, orders

* **Arms (unchanged cycle 29 engine, `cue_dose_guard::run_with`,
  bit-for-bit):** S1 (supervised control), R1, E, Zg1, Zgd0..Zgd4, Zgd2c
  (cycle 27 last-resort family) and Zbd0..Zbd4 (`block` guard, same matched
  share-2 code, dose masks and per-stage `paced100` caps). **Equal budget:**
  every arm of a (seed, order) has the same per-stage write caps (the
  cumulative write curve of uncapped E of that (seed, order)); S1 is the
  supervised control at that budget.
* **Conditions:**

  | cond | sources | targets | fan-in | edges / source | note |
  |---|---|---|---|---|---|
  | base | 128 | 32 | 8 | 2.0 | cycle 27 design |
  | f16 | 128 | 32 | 16 | 4.0 | **new**, fan-in midpoint |
  | f32 | 128 | 32 | 32 | 8.0 | **new**, fan-in of d512 at base sources |
  | src512 | 512 | 32 | 8 | 0.5 | cycle 29/30 condition, new seeds |
  | d512 | 512 | 32 | 32 | 2.0 | cycle 30 condition, new seeds |

  `Cond::F16` and `Cond::F32` are new variants of `cue_dose_guard::Cond`;
  they are **not** added to `cue_dose_guard::CONDS` or
  `cue_dose_density::CONDS`, so the cycle 29 and cycle 30 grids and CLIs are
  unchanged (tested). Active inputs 6, 32 training patterns per task, 128
  TEST patterns, 8 rounds per task, as cycles 29–30.
* **k = 16, flip 100 only.**
* **Seeds 841..880** (40 seeds, never used by an earlier B888-4 cycle).
  Pilot seeds 1..5.
* **Orders:** identity (secondary only), reversed, and 3 seeded
  Fisher–Yates permutations with the cycle 27 order salt
  `0x0DE5_B888_4C27_0A1D` (same declared reuse as cycles 28–30, so that the
  engine stays bit-for-bit the same; orders are new because the seeds are
  new).
* **Dropped vs cycle 30 (declared):** d256 (its fan-in 16 is covered at
  base sources by f16).
* **Declared confound (same as cycle 30):** the `paced100` budget is the
  write curve of uncapped E, so it grows with the number of edges (fan-in
  × targets). Cross-condition contrasts therefore compare conditions that
  differ in fan-in **and** in the write budget it induces. Within a
  condition every arm keeps the same caps.

## 4. Metrics and statistics

* Exactly as cycles 29–30: ACC, LEARN, FGT, LAST on all / CN (novel) pairs,
  balanced accuracy; per seed the mean over the 4 held-out orders (n = 40
  paired seeds); LEARN_head, RET_head, RF as cycle 29. **Generalisation** =
  ACC_all on the shared TEST set (never trained) and ACC_cn on its
  novel-pair subset; **forgetting** = raw FGT_all (RF, RET_head secondary).
* **Learning measure of the manipulation (H1–H6):** LEARN_head of uncapped
  E (the arm that sets the budget), as cycle 30 H1.
* **Cross-condition contrasts:** per seed value s, f_b(s) − f_a(s). The
  conditions are different fixtures; pairing by seed value is a fixed
  procedure, not a claim that they are matched. **Interaction (H6):** per
  seed [f(d512) − f(src512)] − [f(f32) − f(base)].
* **Statistics** as #123–#189: bootstrap 95 % percentile CI (2000
  resamples, seed 0xB0075EED); better/ties/worse over 40 seeds.
  * positive: mean > 0 and CI lower bound > 0;
  * negative: CI upper bound < 0;
  * non-inferior: CI lower bound > −0.02;
  * H6 "inside ±0.02": CI lower bound > −0.02 **and** upper bound < +0.02
    (the cycle 29 non-inferiority margin used two-sided; declared here
    because a fan-in × sources interaction on LEARN has no earlier margin).
* **Label-free target load (pilot, descriptive):** per training pattern,
  edges from active inputs per target, share of targets reached by ≥ 1
  active input, and mean Jaccard overlap of the reached-target sets of two
  training patterns of the same task (graph and pattern inputs only; no
  labels). Overlap is a label-free proxy for how many weights two patterns
  of the same task share; it is not a hypothesis.
* **Reproducibility (declared plan):** the main run uses `--threads 8`. The
  `f32` condition is re-run with `--threads 5`; its rows must be
  byte-identical to the main CSV's `f32` rows (sha256 reported). If it
  differs, it is reported as a failure, not hidden. Other conditions are not
  re-run unless time allows (declared in the results if so).
* **Checkpoint/resume:** the engine state is the #132 checkpoint (circuit
  weights, consolidated signs, stage-start weights, guard counters);
  mid-stage save/load is already tested bit-exact for a block dose arm at
  fan-in 32 by the cycle 30 test
  (`checkpoint_resume_is_bit_exact_for_a_block_dose_arm_at_fan_in_32`); no
  new engine path is added here. A whole run is deterministic, so a killed
  run is resumed by re-running the remaining conditions (`--conds`) with
  the same seeds and concatenating the per-condition CSV rows.

## 5. Design gates (per condition, reported first)

The cycle 29 gates G1–G5, unchanged (`cue_dose_guard::gates`). If G1, G2a,
G2b, G3 or G4 fails in a condition, every hypothesis using that condition
is **not evaluable**. G2c and G5 are reported; no hypothesis of this cycle
uses them.

## 6. Hypotheses (k = 16, flip 100, held-out orders, seeds 841..880)

Gap = FGT(S1) − FGT(Zgd4) (< 0: Zgd4 forgets more in raw terms).
LEARN = LEARN_head(E).

| H | statement | criterion | expectation |
|---|---|---|---|
| H1 **(primary)** | fan-in lowers learning by itself at fixed sources | LEARN f32 − base negative | open |
| H2 | the fan-in effect is monotone | LEARN f16 − base **and** f32 − f16 both negative | open |
| H3 | cycle 30 H1 replicates on new seeds | LEARN d512 − src512 negative | pass |
| H4 | more sources lower learning at fan-in 8 (cycle 29/30) | LEARN src512 − base negative | pass |
| H5 | more sources do not lower learning further at fan-in 32 | LEARN d512 − f32 non-inferior | open |
| H6 | fan-in and sources act additively on learning | interaction [(d512 − src512) − (f32 − base)] inside ±0.02 | open |
| H7 | the raw gap closes with fan-in at fixed sources | gap(f32) − gap(base) positive | open |
| H8 | that closure is a learning collapse | [LEARN_head(Zgd4) − LEARN_head(S1)] f32 − base negative | open |
| H9 | the retention term improves with fan-in | [RET_head(Zgd4) − RET_head(S1)] f32 − base positive | fail |
| H10 | supervised control at equal budget: the block guard keeps its cycle 29 result | in **each** of the 5 conditions: FGT(S1) − FGT(Zbd4) non-inferior **and** ACC_all(Zbd4) − ACC_all(S1) positive | open |
| H11 | generalisation vs supervised control at f32 | f32: ACC_all(Zgd4) − ACC_all(S1) positive **and** ACC_cn(Zgd4) − ACC_cn(S1) positive | open |
| H12 | the dose effect persists | Zgd0..4 slope of ACC_all positive in each of the 5 conditions | pass |

"Expectation" is written before measuring and is not a criterion. Every
hypothesis is reported, pass or fail; failures are results.

**Reading rules (fixed now).**

* H1 and H3 pass: fan-in lowers learning at fixed sources, and the cycle 30
  d512 learning loss is attributed to fan-in (with the write budget it
  induces), **not** to density (at 128 sources f32 has 4× base's density).
* H1 fails and H3 passes: fan-in alone does not lower learning at 128
  sources; the cycle 30 loss then needs the combination with more sources
  (read with H6).
* H6 fails: the fan-in effect depends on the number of sources; its sign is
  read from the interaction mean.
* H7 passes with H8 passing and H9 failing: the gap closure at f32 is again
  a learning collapse, not a forgetting improvement, and is not claimed as
  one. H7 passing with H9 passing would be the first retention gain from a
  structural change in this series and would be reported as such.
* No reading here is about biology; "fan-in" is a parameter of the
  generated graph.

## 7. Label-free pilot

Not run yet at the time of pre-registration (this section is filled after
the pre-registration commit is pushed; the pilot cannot change §3–§6).

## 8. Results

Not run yet.

## Reproduce

```bash
cargo +1.89 run --release -- bench-cue-fanin --census > results/cue-fanin-census-v1.txt
cargo +1.89 run --release -- bench-cue-fanin --threads 8 --csv results/cue-fanin-v1.csv > results/cue-fanin-report-v1.txt
```

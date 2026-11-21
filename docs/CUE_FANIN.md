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

Run after the pre-registration commit `1e9f9d2` was pushed.
`bench-cue-fanin --census` (seeds 1..5, identity order, training only, no
accuracy computed, no held-out order drawn):
[results/cue-fanin-census-v1.txt](../results/cue-fanin-census-v1.txt).

| cond | edges / source | active-input edges per target | share of targets reached | same-task Jaccard of reached targets | E writes (= cap) | Zgd4 partner moves | Zbd4 partner moves | Zbd0 blocked |
|---|---|---|---|---|---|---|---|---|
| base | 2.0 | 0.241 | 0.220 | 0.188 | 914.8 | 61.2 | 61.8 | 12 046 |
| f16 | 4.0 | 0.492 | 0.407 | 0.281 | 2 447.2 | 137.4 | 116.0 | 20 414 |
| f32 | 8.0 | 1.201 | 0.749 | **0.604** | 5 959.4 | 201.0 | 169.8 | 32 751 |
| src512 | 0.5 | 0.188 | 0.175 | 0.114 | 1 309.4 | 49.0 | 59.4 | 9 879 |
| d512 | 2.0 | 0.361 | 0.312 | 0.191 | 2 645.6 | 115.6 | 112.6 | 12 476 |

* At fixed sources, fan-in raises the target load strongly: f32 reaches
  75 % of the targets per training pattern (base 22 %), and two training
  patterns of the same task share 60 % of their reached targets (base
  19 %). d512 sits close to base on overlap (0.191) despite its fan-in 32,
  because its 6 active inputs are spread over 512 sources.
* The induced `paced100` budget grows with the edges (E writes 915 base,
  2 447 f16, 5 959 f32): the declared confound of §3 is large at f32 (6.5×
  base).
* Every manipulation binds in every condition: last-resort dose skips fall
  to 0 at full dose and partner moves rise along Zgd0..4; every block arm
  refuses overwrites; no frozen line moves; 5/5 leakage-free fixtures per
  condition. In f16, f32 and d512 the block arms use their whole cap.
* The pilot changes nothing in §3–§6.

## 8. Results

Main run: `bench-cue-fanin --threads 8` (binary built from the
pre-registration commit `1e9f9d2`; seeds 841..880, 5 conditions, 15 arms,
5 orders; 15 000 rows; wall time 3 313 s on 8 shared cores):
[results/cue-fanin-v1.csv](../results/cue-fanin-v1.csv)
(sha256 `342494fdb76a560307c9c71bc0132515d74cc6995a8bb1cf6fdb9047be0b0dce`),
[results/cue-fanin-report-v1.txt](../results/cue-fanin-report-v1.txt).
All synthetic.

### Gates

| cond | G1 | G2a | G2b | G2c | G3 (median CN) | G4 | G5 |
|---|---|---|---|---|---|---|---|
| base | pass | pass | pass | pass | pass (60) | 40/40 | pass |
| f16 | pass | pass | pass | pass | pass (345) | 40/40 | pass |
| f32 | pass | pass | pass | pass | pass (1 661) | 40/40 | **fail** |
| src512 | pass | pass | pass | pass | pass (60) | 40/40 | pass |
| d512 | pass | pass | pass | pass | pass (120) | 40/40 | **fail** |

G5 fails in f32 (RF of S1 undefined on 38/40 seeds: S1 learns almost
nothing, LEARN_head 0.540) and in d512 (S1 on 5 seeds, Zbd4 on 14). No
hypothesis uses G5, so all 12 are evaluable; RF numbers there are over the
defined seeds only.

### Arms (held-out orders, mean over rows)

| cond | arm | ACC_all | ACC_cn | LEARN_head | RET_head | FGT_all | charged writes |
|---|---|---|---|---|---|---|---|
| base | S1 | 0.507 | 0.510 | 0.631 | 0.501 | 0.130 | 627.1 |
| base | E | 0.520 | 0.644 | 0.987 | 0.489 | 0.498 | 627.1 |
| base | Zgd4 | 0.598 | 0.629 | 0.818 | 0.585 | 0.233 | 626.6 |
| base | Zbd4 | 0.595 | 0.574 | 0.641 | 0.594 | 0.047 | 616.1 |
| f16 | S1 | 0.512 | 0.513 | 0.594 | 0.509 | 0.085 | 1 666.8 |
| f16 | E | 0.543 | 0.619 | 0.957 | 0.515 | 0.442 | 1 666.8 |
| f16 | Zgd4 | 0.576 | 0.576 | 0.664 | 0.572 | 0.091 | 1 666.8 |
| f16 | Zbd4 | 0.560 | 0.552 | 0.611 | 0.560 | 0.051 | 1 666.8 |
| f32 | S1 | 0.512 | 0.509 | 0.540 | 0.511 | 0.029 | 4 233.0 |
| f32 | E | **0.588** | 0.614 | 0.909 | 0.567 | 0.342 | 4 233.0 |
| f32 | Zgd4 | 0.539 | 0.540 | 0.593 | 0.539 | 0.054 | 4 233.0 |
| f32 | Zbd4 | 0.531 | 0.531 | 0.586 | 0.531 | 0.056 | 4 233.0 |
| src512 | S1 | 0.506 | 0.515 | 0.615 | 0.501 | 0.114 | 999.6 |
| src512 | E | 0.513 | 0.564 | 0.845 | 0.491 | 0.353 | 999.6 |
| src512 | Zgd4 | 0.539 | 0.559 | 0.759 | 0.525 | 0.234 | 999.3 |
| src512 | Zbd4 | 0.535 | 0.531 | 0.584 | 0.534 | 0.050 | 952.6 |
| d512 | S1 | 0.512 | 0.522 | 0.556 | 0.509 | 0.047 | 2 400.6 |
| d512 | E | 0.520 | 0.542 | 0.652 | 0.511 | 0.141 | 2 400.6 |
| d512 | Zgd4 | 0.528 | 0.533 | 0.578 | 0.527 | 0.051 | 2 400.6 |
| d512 | Zbd4 | 0.516 | 0.518 | 0.554 | 0.516 | 0.039 | 2 400.6 |

### Hypotheses (mean [95 % CI] better/ties/worse over 40 seeds)

| H | contrast | value | verdict |
|---|---|---|---|
| H1 (primary) | LEARN f32 − base | −0.078 [−0.081, −0.076] 0/0/40 | pass |
| H2 | LEARN f16 − base; f32 − f16 | −0.030 [−0.032, −0.027]; −0.048 [−0.051, −0.046] (both 0/0/40) | pass |
| H3 | LEARN d512 − src512 | −0.193 [−0.196, −0.189] 0/0/40 | pass (cycle 30 replicates) |
| H4 | LEARN src512 − base | −0.142 [−0.146, −0.138] 0/0/40 | pass |
| H5 | LEARN d512 − f32, non-inferior | −0.257 [−0.260, −0.254] 0/0/40 | **fail** |
| H6 | interaction (d512 − src512) − (f32 − base) inside ±0.02 | −0.115 [−0.120, −0.110] 0/0/40 | **fail** |
| H7 | gap(f32) − gap(base) | +0.078 [+0.070, +0.086] 40/0/0 | pass |
| H8 | ΔLEARN_head(Zgd4 − S1) f32 − base | −0.134 [−0.143, −0.125] 0/0/40 | pass |
| H9 | ΔRET_head(Zgd4 − S1) f32 − base | −0.056 [−0.062, −0.049] 0/0/40 | **fail** |
| H10 | Zbd4 vs S1: FGT non-inf. and ACC_all > 0, each cond | FGT +0.083 / +0.034 / **−0.026 [−0.030, −0.023]** / +0.064 / +0.008; ACC +0.089 / +0.049 / +0.019 / +0.029 / +0.004 [+0.002, +0.006] | **fail** (f32 FGT) |
| H11 | f32 ACC_all and ACC_cn, Zgd4 − S1 | +0.028 [+0.024, +0.031] 40/0/0; +0.030 [+0.027, +0.034] 40/0/0 | pass |
| H12 | Zgd slope ACC_all, each cond | +0.023 / +0.009 / +0.004 / +0.007 / **+0.001 [−0.000, +0.003]** | **fail** (d512) |

Conditions in H10/H12 are listed base / f16 / f32 / src512 / d512.

Raw gap FGT(S1) − FGT(Zgd4) per condition: base −0.103, f16 −0.006,
f32 −0.025, src512 −0.120, d512 −0.004. Exact decomposition (Zgd4 − S1)
ΔLEARN_head / ΔRET_head: base +0.187 / +0.084; f16 +0.070 / +0.064; f32
+0.053 / +0.028; src512 +0.144 / +0.024; d512 +0.022 / +0.018.

### Reading (pre-registered rules of §6, then descriptive)

* **Fan-in lowers learning by itself at fixed sources (H1, H2 pass).** At
  128 sources uncapped E LEARN_head falls 0.987 → 0.957 → 0.909 as fan-in
  goes 8 → 16 → 32, monotonically, on 40/40 seeds, while the edge density
  **rises** from 2 to 8 per source. Together with H3 (the cycle 30 loss
  replicates, −0.193) the first rule applies: in this benchmark the learning
  loss that cycle 30 saw at d512 is **not** a density effect; fan-in
  (with the write budget it induces) lowers learning.
* **But fan-in alone does not account for the size of the cycle 30 loss
  (H5, H6 fail).** The two factors are strongly non-additive: fan-in 32
  costs E −0.078 at 128 sources but −0.193 at 512 sources (interaction
  −0.115), and more sources cost −0.142 at fan-in 8 but −0.257 at fan-in 32.
  By the third rule the fan-in effect depends on the number of sources and
  is larger (more negative) with more sources. d512 is the worst cell for
  every arm. We do not have a mechanism for the interaction; the label-free
  target-load census (§7) does not explain it (d512's same-task overlap
  0.191 is close to base's 0.188 while f32's is 0.604, yet d512 learns far
  less than f32).
* **The raw gap closes again for the wrong reason (H7, H8 pass, H9 fails).**
  At f32 the gap moves −0.103 → −0.025 because Zgd4's learning advantage
  over S1 collapses (−0.134), and its retention advantage **also shrinks**
  (−0.056). This is the cycle 30 mechanism again: raw FGT rewards not
  learning. **It is not an improvement and is not claimed as one.** The
  gap is not even monotone in fan-in (f16 −0.006, f32 −0.025).
* **Supervised control at equal budget.** The block guard forgets less than
  S1 and is more accurate in base, f16, src512 and (barely) d512, but at
  f32 it forgets **more** than S1 (−0.026, 0/0/40) because S1 learns almost
  nothing there (LEARN_head 0.540, FGT 0.029); H10 fails. Zgd4 still beats
  S1 at f32 on generalisation (H11: ACC_all +0.028, ACC_cn +0.030), but by
  less than at base (+0.092 / +0.119).
* **Uncapped E is the most accurate arm at f32 (descriptive, not a
  pre-registered contrast).** ACC_all E 0.588 vs Zgd4 0.539, S1 0.512, at
  the same charged writes. At base the order is reversed (Zgd4 0.598 vs E
  0.520). The cue-code advantage over plain error-gated writing therefore
  does not survive high fan-in in this benchmark. No CI is computed for
  this; it is a summary-table observation for a future pre-registration.
* **The dose effect shrinks with fan-in and vanishes at d512 (H12 fails
  there).** Zgd slope +0.023 base, +0.009 f16, +0.004 f32, +0.007 src512,
  +0.001 [−0.000, +0.003] d512; concavity ≈ 0 everywhere (linear where it
  exists).
* Absolute effects are small, every number comes from a generated graph
  with artificial labels, and no claim about synapses, biology, insect
  memory or BANC physiology is made.

## Reproduce

```bash
cargo +1.89 run --release -- bench-cue-fanin --census > results/cue-fanin-census-v1.txt
cargo +1.89 run --release -- bench-cue-fanin --threads 8 --csv results/cue-fanin-v1.csv > results/cue-fanin-report-v1.txt
```

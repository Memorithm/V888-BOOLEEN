# B888-4 cycle 30: is the cycle 29 H3 failure (the raw forgetting gap **widens** with more graph sources) caused by **edge dilution**, and does it close when the edge density per source is held fixed? (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not built on the
build machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123), run through the **unchanged** cycle 29
engine of [CUE_DOSE_GUARD.md](CUE_DOSE_GUARD.md) (#188: the cycle 27
last-resort arms plus the five `block`-guard dose arms). "Cue", "dose",
"guard", "density", "dilution", "retention" and "forgetting" name parts of an
update rule and an artificial benchmark on a Boolean graph with 4-bit
weights. Nothing here is a claim about synapses, biology, insect memory or
BANC physiology.

Module: `src/cue_dose_density.rs`. CLI: `bench-cue-dose-density` (`--census`
for the label-free pilot).

## 1. Where cycle 29 left off (#188, main `798d9733`)

At k = 16, flip 100 (held-out orders, seeds 761..800, synthetic):

* exact split FGT(S1) − FGT(x) = ΔRET_head − ΔLEARN_head: Zgd4 retains
  more than S1 (+0.086) but learns far more (+0.189), so it loses on raw
  forgetting (−0.103);
* **H3 failed**: adding sources widened the raw gap instead of closing it,
  −0.103 (128) → −0.117 (256) → −0.123 (512); gap(src512) − gap(base) =
  −0.020 [−0.027, −0.013] 8/0/32. The retention advantage of Zgd4 over S1
  shrank faster (+0.086 → +0.042 → +0.023) than its learning advantage
  (+0.189 → +0.159 → +0.146);
* the block guard (Zbd4) forgot 0.086 less than S1 and matched Zgd4 on
  ACC_all while learning 0.177 less; the dose effect stayed linear;
* every arm got worse with more sources, including uncapped error-gated E
  (LEARN_all 0.987 → 0.905 → 0.841);
* the threads-count CSV byte-identity rerun was **not** done.

## 2. The question (stated before measuring)

In cycle 29 only `shape.sources` changed: targets 32 and fan-in 8 stayed
fixed, so the number of edges stayed at 256 while the sources went
128 → 512. The edge density fell from **2 to 0.5 edges per source**. A
task's patterns are drawn from sources that reach at least one target, but
at 512/8 each such source reaches about one target instead of about two
(the label-free pilot, §7, measures it). So "more sources" in cycle 29 was
also "a sparser graph", and every arm learned less.

The cycle 30 question: **is the widening of the raw gap caused by this
dilution?** If so, holding the density at 2 edges per source (fan-in
scaled with sources: 256/16, 512/32) should undo the widening (the gap at
d512 should be larger, i.e. closer to zero, than at src512), and the
cycle 29 H3 question (does the gap close with more sources?) can be asked
again at matched density.

Exactly one design field changes per condition (`shape.fan_in`) relative
to the cycle 29 condition with the same number of sources.

## 3. Arms, conditions, seeds, orders

* **Arms (unchanged cycle 29 engine, `cue_dose_guard::run_cond`,
  bit-for-bit):** S1 (supervised control), R1, E, Zg1, Zgd0..Zgd4, Zgd2c
  (cycle 27 last-resort family, `cue_write_dose::run_seed`) and Zbd0..Zbd4
  (`E/E-cue0+block@paced100`, same matched share-2 code, dose masks and
  per-stage `paced100` caps). **Equal budget:** every arm of a (seed, order)
  has the same per-stage write caps (the cumulative write curve of uncapped
  E of that (seed, order)); S1 is the supervised control at that budget.
* **Conditions:**

  | cond | sources | targets | fan-in | edges / source | note |
  |---|---|---|---|---|---|
  | base | 128 | 32 | 8 | 2.0 | cycle 27 design |
  | src512 | 512 | 32 | 8 | 0.5 | cycle 29 dilution condition, new seeds |
  | d256 | 256 | 32 | 16 | 2.0 | **new**, density-matched |
  | d512 | 512 | 32 | 32 | 2.0 | **new**, density-matched |

  `Cond::D256` and `Cond::D512` are new variants of
  `cue_dose_guard::Cond`; they are **not** added to `cue_dose_guard::CONDS`,
  so the cycle 29 grid and CLI are unchanged (tested). Active inputs 6, 32
  training patterns per task, 128 TEST patterns, 8 rounds per task, as
  cycle 29. Each condition builds a leakage-free fixture with four frozen
  lines per task on seed 1 (unit test: fixture construction only, no
  learning, no accuracy).
* **k = 16, flip 100 only.**
* **Seeds 801..840** (40 seeds, never used by an earlier B888-4 cycle).
  Pilot seeds 1..5.
* **Orders:** identity (secondary only), reversed, and 3 seeded
  Fisher–Yates permutations with the cycle 27 order salt
  `0x0DE5_B888_4C27_0A1D` (same declared reuse as cycles 28–29, so that the
  engine stays bit-for-bit the same; orders are new because the seeds are
  new).
* **Dropped vs cycle 29 (declared):** src256 (the cycle 29 trend
  128 → 256 → 512 is not re-measured; d256 is the density-matched
  midpoint).

## 4. Metrics and statistics

* Exactly as cycle 29: ACC, LEARN, FGT, LAST on all / CN (novel) pairs,
  balanced accuracy; per seed the mean over the 4 held-out orders (n = 40
  paired seeds); LEARN_head, RET_head, RF as cycle 29; dose slope and
  concavity as cycle 27. **Generalisation** = ACC_all on the shared TEST set
  (never trained) and ACC_cn on its novel-pair subset; **forgetting** = raw
  FGT_all (and RF, RET_head as secondary).
* **Cross-condition contrasts** (H1–H6): per seed value s,
  f_b(s) − f_a(s), where f is a per-seed quantity of each condition (the
  raw gap FGT(S1) − FGT(Zgd4), LEARN_head(E), or the retention term
  RET_head(Zgd4) − RET_head(S1)). The conditions are different fixtures;
  pairing by seed value is a fixed procedure, not a claim that they are
  matched.
* **Statistics** as #123–#188: bootstrap 95 % percentile CI (2000
  resamples, seed 0xB0075EED); better/ties/worse over 40 seeds.
  * positive: mean > 0 and CI lower bound > 0;
  * negative replication (H2): CI upper bound < 0;
  * equivalent: the 95 % CI lies strictly inside ±0.01;
  * non-inferior: CI lower bound > −0.02.
* **Coverage (label-free, descriptive):** share of sources with at least
  one outgoing edge and mean out-degree of the active training inputs
  (graph and pattern inputs only; no labels).
* **Reproducibility (declared plan):** the main run uses `--threads 8`. The
  `base` and `d512` conditions are re-run with `--threads 5`; their rows
  must be byte-identical to the main CSV's rows of the same conditions
  (sha256 reported). Separately, the cycle 29 threads rerun that was not
  done is done here for cycle 29's `base` condition: `bench-cue-dose-guard
  --conds base --threads 6` must reproduce the header and `base` rows of the
  committed `results/cue-dose-guard-v1.csv` byte for byte. If either
  differs, it is reported as a failure, not hidden.
* **Checkpoint/resume:** the engine state is the #132 checkpoint (circuit
  weights, consolidated signs, stage-start weights, guard counters);
  mid-stage save/load is tested bit-exact here for Zbd2 and Zbd4 at fan-in
  32 across the dose-mask schedule split. A whole run is deterministic, so
  a killed run is resumed by re-running the remaining conditions
  (`--conds`) with the same seeds.

## 5. Design gates (per condition, reported first)

The cycle 29 gates G1–G5, unchanged (`cue_dose_guard::gates`): G1
integrity (both families); G2a last-resort dose binds; G2b every block arm
refuses ≥ 1 overwrite in ≥ 50 % of held-out rows; G2c block doses bind;
G3 median CN pairs ≥ 20; G4 leakage-free for every seed; G5 RF defined for
S1, Zgd4, Zbd4. If G1, G2a, G2b, G3 or G4 fails in a condition, every
hypothesis using that condition is **not evaluable**; G2c failing makes
only H12 not evaluable.

## 6. Hypotheses (k = 16, flip 100, held-out orders, seeds 801..840)

Gap = FGT(S1) − FGT(Zgd4) (< 0: Zgd4 forgets more in raw terms).

| H | statement | criterion | expectation |
|---|---|---|---|
| H1 | manipulation check: density matching restores learning | LEARN_head(E) d512 − src512 positive | pass |
| H2 | the cycle 29 widening replicates on new seeds | gap(src512) − gap(base): CI hi < 0 | pass |
| H3 **(primary)** | dilution causes the widening | gap(d512) − gap(src512) positive | open |
| H4 | the gap closes with more sources at matched density | gap(d512) − gap(base) positive | open |
| H5 | no widening at matched density | gap(d512) − gap(base) non-inferior | open |
| H6 | the retention term is restored | [RET_head(Zgd4) − RET_head(S1)] d512 − src512 positive | open |
| H7 | supervised control at equal budget: the block guard keeps its cycle 29 result | in **each** of base, src512, d256, d512: FGT(S1) − FGT(Zbd4) non-inferior **and** ACC_all(Zbd4) − ACC_all(S1) positive | pass |
| H8 | novel-pair generalisation vs supervised control | d512: ACC_cn(Zgd4) − ACC_cn(S1) positive | open |
| H9 | novel-pair generalisation of the block guard | d512: ACC_cn(Zbd4) − ACC_cn(S1) positive | open |
| H10 | the dose effect is unchanged | Zgd0..4 slope of ACC_all positive in each of the four conditions | pass |
| H11 | the dose effect stays linear at matched density | d512: Zgd concavity of ACC_all equivalent to 0 | pass |
| H12 | the block-family dose effect at matched density | d512: Zbd0..4 slope of ACC_all positive | open |

"Expectation" is written before measuring and is not a criterion. Every
hypothesis is reported, pass or fail; failures are results.

**Reading rules (fixed now).** If H3 passes and H5 passes, the cycle 29
widening is attributed to dilution in this benchmark. If H3 passes and H5
fails, dilution explains part of it and the rest is attributed to source
count at matched density. If H3 fails, dilution is not supported as the
cause. H4 is the cycle 29 H3 question re-asked at matched density.

## 7. Label-free pilot

_To be filled in after the pilot run (seeds 1..5, identity order, training
only, no accuracy)._

## 8. Results

_To be filled in after the main run._

## Reproduce

```bash
cargo +1.89 run --release -- bench-cue-dose-density --census > results/cue-dose-density-census-v1.txt
cargo +1.89 run --release -- bench-cue-dose-density --threads 8 --csv results/cue-dose-density-v1.csv > results/cue-dose-density-report-v1.txt
```

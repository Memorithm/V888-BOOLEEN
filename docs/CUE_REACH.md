# B888-4 cycle 13: cue reach at a fixed pattern pool (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository and is not on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[NOVEL_SPLIT.md](NOVEL_SPLIT.md) (#123). Nothing here is a claim about
synapses, biology, insect memory or BANC physiology.

Module: `src/cue_reach.rs`. CLI: `bench-cue-reach` (`--census` for the
label-free design census, `--no-coupled` to skip the secondary family).

## Motivation (what #123 left open)

On the #123 novel-split design, `E/E-cue` added almost nothing over
`error-gated`: +0.003 ACC on compositional-novel (CN) pairs and +0.011 on all
pairs at k = 4, against +0.215 on all pairs in #122. In #123 each cue was
capped at 4 lines and reached only 14–22 of the 32 targets. #122's 12-line
cues reached every target. Pool width and cue reach changed together, so
#123 could not say which one removed the cue benefit.

This cycle holds the pattern pool fixed and varies only the cue size, and so
the cue reach.

## Design (pre-registered)

* **Graph and task:** exactly the #123 primary design. Generated graph per
  seed, 128 inputs / 32 targets / fan-in 8, threshold 1, init weight 1,
  4-bit weights. Four tasks T1→T4 (`task_sequence::build_sequence`), patterns
  of 6 active inputs, 32 training patterns per task (disjoint), one shared
  TEST set of 128 patterns, OR-of-edge-valences target rule, pair classes
  seen / CN / ICN / UE as in #123.
* **Fixed pool (primary family):** per seed, four disjoint greedy cues of up
  to **16 lines** each are chosen once (`context_cue::choose_n_cues_capped`,
  cap 16). The greedy stops early when a cue reaches every target, so a full
  cue has 9–15 lines in the pilot. **All reserved lines are removed from the
  pattern pool once.** A cued arm at level L activates the **first L lines**
  of each cue (greedy order, so the prefixes are nested). Graph, pool,
  patterns, valences, pair classes and episode order are byte-identical at
  every level (tested). Reserved lines that a level does not use are never
  active.
* **Levels:** L ∈ {2, 4, 8, 16} lines per cue. On the pilot (below) these
  give a mean cue reach of about 30 %, 53 %, 84 % and 100 % of targets.
* **Arms per (flip, seed), equal budget** (8 rounds × 32 patterns = 256
  episodes per task, same episode order for every arm, no rehearsal,
  original edges only, topology checked on every row): `supervised` and
  `error-gated` once (uncued, level 0), then `E/E-cue@L` and `S/S-cue@L` at
  each level. 10 rows per (flip, seed).
* **Gain** at level L = `E/E-cue@L` − `error-gated`, paired by seed.
* **Coupled family (secondary, flip 50 only):** the #123 construction, which
  reserves only L lines per cue, so the pool shrinks as L grows. Rows:
  `error-gated` and `E/E-cue` at each L on that fixture. This makes the
  pool-versus-reach trade-off explicit. No hypothesis uses it.
* **Instrumentation:** per-target confusions after every task, so every
  metric is also reported on the pairs whose target the cue reaches
  (`cue_reached`) and on the rest (`cue_unreached`), with the base arm scored
  on the same pairs. The OLS slope of the per-seed gain against mean reach
  fraction is reported.
* **Seeds:** 121..160 (40 seeds, never used by an earlier B888-4 cycle).
  **Flips:** 25, **50 (primary)**, 100.

Metrics (ACC, LEARN, FGT, LAST at k = 2..4), the balanced accuracy, the
bootstrap (95 % percentile, 2000 resamples, seed 0xB0075EED) and **strong**
(mean ≥ 0.05, > 0 in ≥ 32 of 40 seeds, CI lower bound > 0) are as in #123.
**Sequence rule** (x vs base on a subset at k = 4): Δ ACC strong,
Δ LAST ≥ −0.05 and Δ LEARN ≥ −0.05.

### Pilot census (design only, no learning, seeds 1..5, flip 50)

No accuracy was computed for this table. Seeds 1..5 are not used by any
hypothesis.

| family | L | pool | cue lines | cue reach min / median / max (of 32) | CN pairs min / median per (seed, task) |
|---|---|---|---|---|---|
| fixed | 2 | 59–64 | 2 | 7 / 9 / 14 | 85 / 116.5 |
| fixed | 4 | 59–64 | 4 | 13 / 16 / 22 | 85 / 116.5 |
| fixed | 8 | 59–64 | 8 | 21 / 27 / 31 | 85 / 116.5 |
| fixed | 16 | 59–64 | 9–15 | 32 / 32 / 32 | 85 / 116.5 |
| coupled | 2 | 99–103 | 2 | 8 / 10 / 14 | 199 / 253.5 |
| coupled | 4 (= #123) | 91–95 | 4 | 15 / 17 / 22 | 169 / 228 |
| coupled | 8 | 75–79 | 8 | 21 / 27 / 31 | 110 / 150 |
| coupled | 16 | 59–64 | 9–15 | 32 / 32 / 32 | 85 / 116.5 |

**Trade-off, stated up front:** full reach needs 9–15 lines per cue, so four
full cues take 40–50 of the 128 inputs. A fixed pool that admits full reach
is therefore narrower than #123's (59–64 vs 91–103 usable pattern inputs),
and it has about half the CN pairs (median ≈ 117 vs ≈ 228 in the pilot). It
is still far above the #123 gate of 20. In the coupled family, reach costs
pool width directly.

## Design gates (reported first)

* **G1 (novel pairs):** on seeds 121..160 at flip 50, the median CN TEST
  pairs per (seed, task) in the fixed family is ≥ 20.
* **G2 (reach actually varies):** the median cue reach per (seed, task) of
  `E/E-cue` increases strictly across L = 2, 4, 8, 16, and at L = 16 it
  equals the median number of reachable targets.

If G1 or G2 fails, H1–H5 are declared **not evaluable**.

## Hypotheses (fixed family, flip 50, k = 4, seeds 121..160)

* **H1 (dose-response, all pairs):** the mean gain on all pairs never drops
  by more than 0.005 from one level to the next (2→4→8→16), **and** the
  paired difference `E/E-cue@16` − `E/E-cue@2` on ACC_all has a 95 % CI
  lower bound > 0.
* **H2 (dose-response, CN pairs):** the same rule on CN pairs.
* **H3 (full reach recovers the cue benefit, all pairs):** `E/E-cue@16` vs
  `error-gated` passes the sequence rule on all pairs.
* **H4 (full reach helps on novel pairs):** `E/E-cue@16` vs `error-gated`
  passes the sequence rule on CN pairs.
* **H5 (full reach cuts forgetting on novel pairs):**
  FGT_CN(`error-gated`) − FGT_CN(`E/E-cue@16`) is strong.

**Secondary (reported, no pass/fail):** gains at k = 2 and 3; seen, ICN and
UE pairs; `cue_reached` vs `cue_unreached` pairs; the reach slope;
consecutive level contrasts; `S/S-cue@L` − `supervised`; flips 25 and 100;
the coupled family (gain and pool per L); weights changed and saturation.

No multiplicity correction is applied. The analysis is run once. Nothing
above the Results heading is changed after the run except the "pending"
placeholder.

## Command

```bash
cargo +1.89 run --release -- bench-cue-reach --census --flips 50 > results/cue-reach-census-v1.txt
cargo +1.89 run --release -- bench-cue-reach --csv results/cue-reach-v1.csv > results/cue-reach-report-v1.txt
```

## Results

Run after the pre-registration commit c25f32a had been pushed. The grid
(1520 rows: 3 flips × 40 seeds × 10 fixed-family settings, plus 40 seeds ×
4 levels × 2 coupled-family arms at flip 50) takes about 3.5 min on CPU. Two
runs give a byte-identical CSV (sha256 `3cd4ef2f…df82e31d`) and an identical
report. Every row keeps the original topology (`topology_unchanged_all=true`).
Files: `results/cue-reach-census-v1.txt`, `results/cue-reach-v1.csv`,
`results/cue-reach-report-v1.txt`.

### 1. Census and design gates (reported first)

Fixed family, seeds 121..160 (identical at every flip):

| L | cue lines (mean) | cue reach min / median / max (of 32) | mean reach fraction | pool | CN pairs min / median / max per (seed, task) |
|---|---|---|---|---|---|
| 2 | 2 | 6 / 9 / 14 | 0.290 | 58–68 | 68 / 126 / 200 |
| 4 | 4 | 12 / 16 / 23 | 0.513 | 58–68 | same |
| 8 | 8 | 20 / 26 / 32 | 0.821 | 58–68 | same |
| 16 | 12.2 (8–16) | 31 / 32 / 32 | 1.000 (2 of 160 cues hit the 16-line cap at 31) | 58–68 | same |

* **G1 passes:** median CN pairs per (seed, task) 126 ≥ 20 (min 68).
* **G2 passes:** median reach 9 < 16 < 26 < 32, and 32 equals the median
  number of reachable targets.

**Pool-versus-reach trade-off (coupled family, #123 construction):** pool
95–111 at L = 2, 87–103 at L = 4, 71–87 at L = 8 and 58–68 at L = 16. CN
median 238 / 211 / 155 / 126. Reaching every target costs about a third of
the #123 pattern pool and about 40 % of its CN pairs.

### 2. Settings (fixed family, flip 50, k = 4, mean over 40 seeds)

| setting | ACC_all | ACC_CN | ACC_ICN | LEARN_CN | FGT_all | FGT_CN | LAST_all | weights changed | saturated at end |
|---|---|---|---|---|---|---|---|---|---|
| supervised | 0.618 | 0.586 | 0.595 | 0.772 | 0.357 | 0.248 | 0.875 | 3594 | 92 |
| error-gated | 0.636 | 0.628 | 0.639 | 0.910 | 0.447 | 0.377 | 0.973 | 209 | 51 |
| E/E-cue@2 | 0.656 | 0.633 | 0.646 | 0.910 | 0.421 | 0.369 | 0.973 | 259 | 83 |
| E/E-cue@4 | 0.668 | 0.637 | 0.652 | 0.911 | 0.406 | 0.366 | 0.973 | 306 | 111 |
| E/E-cue@8 | 0.687 | 0.645 | 0.664 | 0.913 | 0.380 | 0.358 | 0.973 | 384 | 153 |
| E/E-cue@16 | 0.698 | 0.646 | 0.670 | 0.910 | 0.367 | 0.352 | 0.973 | 445 | 184 |
| S/S-cue@16 | 0.618 | 0.584 | 0.597 | 0.752 | 0.298 | 0.224 | 0.835 | 14815 | 215 |

### 3. Gain `E/E-cue@L` − `error-gated` (fixed family, flip 50, k = 4; mean ± sd [95 % CI], seeds better / tied / worse)

| L (reach) | Δ ACC all | Δ ACC CN | Δ FGT CN (+ = cue forgets less) | Δ ACC on cue-reached pairs | Δ ACC on unreached pairs |
|---|---|---|---|---|---|
| 2 (29 %) | +0.020 [+0.016, +0.023] 39/0/1 | +0.005 [−0.002, +0.012] 28/6/6 | +0.008 [−0.000, +0.015] 28/5/7 | +0.053 [+0.044, +0.063] 39/0/1 | +0.004 [+0.002, +0.007] |
| 4 (51 %) | +0.032 [+0.027, +0.037] 40/0/0 | +0.009 [+0.001, +0.017] 30/2/8 | +0.010 [+0.001, +0.020] 32/0/8 | +0.053 [+0.045, +0.062] 40/0/0 | +0.007 [+0.004, +0.010] |
| 8 (82 %) | +0.051 [+0.045, +0.057] 40/0/0 | +0.017 [+0.006, +0.028] 32/0/8 | +0.019 [+0.006, +0.031] 29/0/11 | +0.057 [+0.051, +0.064] 40/0/0 | +0.015 [+0.006, +0.025] |
| 16 (100 %) | **+0.061 [+0.055, +0.068] 40/0/0** | **+0.018 [+0.005, +0.030] 29/0/11** | **+0.024 [+0.010, +0.039] 30/0/10** | +0.061 [+0.055, +0.068] 40/0/0 | ≈ none (2 of 160 cues miss one target) |

Δ LEARN and Δ LAST are within ±0.001 on all pairs and within ±0.007 on CN
pairs at every level.
`E/E-cue@16` − `E/E-cue@2`: all pairs +0.041 [+0.037, +0.047] 40/0/0; CN
+0.013 [+0.003, +0.022] 30/0/10. OLS slope of the per-seed gain on reach
fraction: +0.059 (all) and +0.020 (CN) per unit reach.

### 4. Hypotheses (flip 50, k = 4)

| | statistic | value | result |
|---|---|---|---|
| G1 | median CN pairs per (seed, task) ≥ 20 | 126 (min 68) | **pass** |
| G2 | median reach strictly increasing, full at L = 16 | 9 / 16 / 26 / 32 of 32 | **pass** |
| H1 | all-pairs gain monotone (tol 0.005) and @16 − @2 CI lo > 0 | +0.020 → +0.032 → +0.051 → +0.061; +0.041 [+0.037, +0.047] | **pass** |
| H2 | CN gain monotone (tol 0.005) and @16 − @2 CI lo > 0 | +0.005 → +0.009 → +0.017 → +0.018; +0.013 [+0.003, +0.022] | **pass** (small effect) |
| H3 | `E/E-cue@16` vs `error-gated`, sequence rule, all pairs | +0.061 [+0.055, +0.068] 40/0/0, Δ LEARN +0.000, Δ LAST −0.001 | **pass** |
| H4 | `E/E-cue@16` vs `error-gated`, sequence rule, CN pairs | +0.018 [+0.005, +0.030] 29/0/11 (mean < 0.05, < 32 seeds) | **fail** |
| H5 | FGT_CN cut at L = 16 strong | +0.024 [+0.010, +0.039] 30/0/10 | **fail** |

### 5. Secondary (no pass/fail)

* **The gain sits on the targets the cue reaches.** On cue-reached pairs
  the gain is about +0.055 at every level (+0.053, +0.053, +0.057,
  +0.061). On unreached pairs it is +0.004 to +0.015. The all-pairs gain
  grows with reach because more pairs get the same per-target benefit.
* **Most of the benefit is on seen pairs.** At L = 16: seen +0.070
  [+0.063, +0.077] 40/0/0, ICN +0.031 [+0.023, +0.040] 37/0/3, CN +0.018,
  UE −0.003 [−0.033, +0.024]. The cue mostly protects memorised
  configurations from being overwritten by later tasks, and helps
  generalisation to novel combinations only a little.
* **Pool width matters too.** In the coupled family (same seeds, flip 50),
  the all-pairs gain is +0.005 at L = 2, +0.010 at L = 4 (this replicates
  #123's +0.011), +0.028 at L = 8 and +0.061 at L = 16; the CN gain is +0.002 /
  +0.003 / +0.006 / +0.018. At the same 4-line cue, the narrower fixed pool
  gives +0.032 against +0.010 for the #123 pool (difference of means across
  families, not a paired test). So #123's near-zero gain had two causes:
  partial reach **and** a wide pool. Full reach on this graph still gives
  only +0.061, far from #122's +0.215 on its 25-input pool.
* **Flip dependence.** At L = 16 the all-pairs gain is +0.044 at flip 25,
  +0.061 at flip 50 and +0.116 at flip 100. The CN gain is −0.009 [−0.024,
  +0.005] at flip 25 (with Δ LEARN_CN −0.017), +0.018 at flip 50, and
  +0.110 [+0.094, +0.126] 40/0/0 at flip 100, where it passes the sequence
  rule at L = 8 and 16. The cue helps novel pairs only when consecutive
  tasks conflict a lot.
* **k = 2 and 3 (flip 50, L = 16):** all +0.056 and +0.059 (both pass the
  rule); CN +0.019 and +0.023 (both fail).
* **`S/S-cue@L` − `supervised`:** no ACC change at any level (all pairs
  +0.001 to +0.000). The LAST cost grows with cue size: −0.011, −0.021,
  −0.032 and −0.040 on all pairs. A bigger cue makes the supervised cost
  from #120/#123 worse.
* **Write cost:** `E/E-cue` changes 259 → 445 weights as L grows, against
  209 for `error-gated` and 3594 for `supervised`. Saturated weights at the
  end rise from 51 to 184.

### Reading

At a fixed pattern pool, the `E/E-cue` benefit over `error-gated` rises
monotonically with cue reach: from +0.020 on all pairs at 29 % reach to
+0.061 at full reach, with every seed better. The per-reached-target benefit
is nearly constant, so the dose-response is a coverage effect. Partial reach
is a real cause of the #123 null on all pairs.

On never-seen input combinations the cue barely helps at flip 50. The CN
gain also rises with reach (H2), but it reaches only +0.018 at full reach,
so H4 and H5 fail. The cue acts mostly as a task key that stops later tasks
from overwriting memorised configurations; it adds little to
compositional generalisation, which comes from the error gate (#123). The
exception is flip 100, where the CN gain is +0.110.

## Limitations

- Synthetic graph and artificial OR-rule task only (128/32/8, active 6),
  threshold 1, init 1. Nothing is measured on the real BANC graph
  (DOI 10.7910/DVN/7WTH1N), which is not built on this machine.
- The fixed pool (58–68 inputs) is narrower than #123's (91–103), because
  it must admit full-reach cues. Fixed-family results are not directly
  #123's design at L = 4. The pool effect is only estimated across families
  (difference of means, not paired).
- Levels are set in lines, not in exact reach. Reach varies across seeds and
  tasks at a fixed L (e.g. 6–14 targets at L = 2). Two of 160 full cues hit
  the 16-line cap at 31 of 32 targets.
- Cue lines and reach grow together with the number of cue edges and with
  the write budget the cue edges can absorb. This cycle does not separate
  reach from cue-edge count.
- CN pairs share TEST patterns with seen pairs, so they are not independent
  samples. The cue is assumed known at test time. 40 seeds. No
  multiplicity correction across H1–H5. H2 passes with a small effect.
- The equal budget is in episodes, not weight writes.

B888-3 files (`src/circuit_explorer.rs`, subcircuit recall, activity atlas
and their CLI wiring) are untouched. No NVIDIA dependency, no secrets. CPU
only.

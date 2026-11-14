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

Pending. This file was committed and pushed before any learning run on seeds
121..160.

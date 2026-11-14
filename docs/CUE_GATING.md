# B888-4 cycle 10 — why the context cue costs the supervised control A-test accuracy, and which gate makes the cue usable (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository nor on the build
machine. Every number comes from the deterministic generated graph of
`consolidation_scale::generate_graph` and the artificial OR-rule task of
[GENERALIZATION_FORGETTING.md](GENERALIZATION_FORGETTING.md), with the cue
lines of [CONTEXT_CUE.md](CONTEXT_CUE.md). Nothing here is a claim about
synapses, biology, insect memory or BANC physiology.

Module: `src/cue_gating.rs`. CLI: `bench-cue-gating`.

## Motivation (what #120 left open)

In #120 the cue helped the error gate (H2 pass) but `supervised+cue` failed
the A-test guard (A test −0.054 vs `supervised`, 19/20 seeds worse) while
cutting forgetting by +0.066. #120 did not say whether that cost comes from
the cue-edge rule, from the pattern-edge rule, or from the number of cue
lines. This cycle separates the two edge classes: every arm has a gate for
**pattern edges** (out-edges of pattern inputs) and, independently, a gate
for **cue edges** (out-edges of the active cue lines).

## A-priori structural analysis (written before any run)

Setting of #117/#120: threshold 1, init weight 1, 4-bit non-negative
weights, drive = Σ weight × contacts over active usable edges, teacher(t) =
OR of the task valences of the pattern's usable edges into t.

1. **The supervised pattern rule ignores the prediction.** It steps every
   active edge toward teacher(t) whatever the drive, so the pattern-edge
   weights of any arm whose pattern gate is `supervised` are a function of
   the episode stream only — bit-identical to the uncued `supervised`
   control after A and after B, whatever the cue does.
2. **No false negatives on A after A.** Under the supervised rule an active
   edge with A-valence 1 always has teacher(t) = 1, so it is only ever
   incremented; it starts at 1, hence its drive stays ≥ 1. Every A-positive
   (pattern, target) pair therefore fires after phase A, for the control and
   for every cued arm with a supervised pattern gate (the cue only adds
   non-negative drive). TPR(A test after A) = 1.
3. Consequence: with a supervised pattern gate, an additive cue **can only
   add false positives** on A after A. Its A test after A is ≤ the
   control's in every seed, for **any** cue-edge rule and any number of cue
   lines. The #120 H1 failure is the size of that unavoidable cost; the
   question becomes whether some cue-edge rule or line count keeps the cost
   ≤ 0.05 while still cutting forgetting by ≥ 0.05 (forgetting can only be
   cut by cue A drive recovering A-positive pairs that phase B switched
   off; cue A edges are not updated during B, because cue A is off then).
4. Cue-edge error gate under a supervised pattern gate (`S/E`): in phase A
   no FN exists (point 2), so the only errors are false positives and the
   cue edges are only ever decremented in A. A cue A edge into t ends at 0
   as soon as one A-negative episode reaches t, and stays 1 only into
   targets that never saw an A-negative training pair. **Expectation:** the
   A-test guard holds, but little cue drive is left to recover forgetting,
   so the forgetting reduction is likely < 0.05 (H1 likely fails).

Points 1–3 are checked by unit tests and per row in the CSV (pattern-weight
hash vs the control, FN count of A test after A); they are a structural
check (S0), not a learning result.

## Pre-registration (this section is committed before any run)

### Arms (equal budget: same graph, task, cue lines, 256 A + 256 B episode orders of #117/#120, no rehearsal, no protection, threshold 1, init 1)

Label `X/Y-cue`: X = pattern-edge gate, Y = cue-edge gate (S = supervised,
always update; E = #117 error gate, update the edges into a reached target
only if its prediction — drive including the cue, weights before the
update — is wrong). Pattern edges into targets not reached by the pattern
do not exist (a pattern edge's target is reached by definition). Cue edges
into unreached targets are updated by S (teacher bit 0) and never by E,
exactly as in #120.

| arm | pattern gate | cue gate | cue lines per cue |
|---|---|---|---|
| `supervised` | S | — (no cue) | 0 |
| `error-gated` | E | — (no cue) | 0 |
| `S/S-cue` | S | S | all (#120 `supervised+cue`) |
| `S/E-cue` | S | E | all |
| `E/S-cue` | E | S | all |
| `E/E-cue` | E | E | all (#120 `error-gated+cue`) |
| `S/S-cue-k1` | S | S | 1 |
| `S/S-cue-k2` | S | S | 2 |
| `S/S-cue-k4` | S | S | 4 |

Cue lines are the #120 greedy-cover lines (cap 12). `kN` activates only the
first N lines of each cue's greedy order (the lines that cover the most
targets first). The pattern pool always excludes **all** #120 cue lines, so
every arm of a seed has exactly the same task and patterns, and the uncued
controls are the #120 controls. With equal gates and all lines, `S/S-cue`
and `E/E-cue` must be bit-identical to #120's cued arms (tested, and
checked against `results/context-cue-v1.csv` on seeds 1..20).

### Protocol

Generated graph 64 inputs / 24 targets / fan-in 6; 4 active pattern inputs;
32 A-train / 32 B-train / 64 TEST patterns; 8 rounds per phase; positive
50 %; balanced accuracy on held-out TEST patterns; forgetting = bal(A test
after A) − bal(A test after B), A always evaluated in the A context, B in
the B context (as #120). **Seeds 1..40** (1..20 are #120's seeds, 21..40
are new). Label-flip rates 25 %, 50 % (primary), 100 %. Grid: 3 × 40 × 9 =
1080 rows, run twice, CSV must be byte-identical; topology checked per row.

Instrumentation per row (in addition to the #120 columns): TPR and TNR of A
test after A and after B, FN count of A test after A, mean cue A / cue B
edge weight after A and after B, a hash of the pattern-edge weights after
A and after B, and whether they equal the matching uncued control's
(`supervised` for S pattern gates, `error-gated` for E).

Checkpoint/resume: saving the circuit (`V888PLS1`) after A and after 128 of
the 256 B episodes, reloading and finishing must give bit-identical
weights and metrics for `S/E-cue`, `E/S-cue` and `S/S-cue-k2` (tested).

Statistics (as #117/#120): paired by seed; mean ± sample sd; better / tied
/ worse seeds; 95 % percentile bootstrap CI of the mean paired difference
(2000 resamples, `BOOT_SEED = 0xB007_5EED`).

### Pass rule (#120's rule, unchanged; ≥ 80 % of seeds = 32/40)

Arm x **passes vs base** iff, paired by seed at the same flip: (1) forgetting
reduction (base − x) mean ≥ 0.05, > 0 in ≥ 32/40 seeds, bootstrap CI lower
bound > 0; (2) A test after A mean drop ≤ 0.05; (3) B test after B mean
drop ≤ 0.05.

### Hypotheses (flip 50 %, 40 seeds)

* **S0 (structural check, predicted to hold):** for every arm with an S
  pattern gate, pattern-edge weights after A and after B equal the
  `supervised` control's and A test after A has FN = 0, in every row
  (all flips). Its Δ A test vs `supervised` is ≤ 0 in every seed.
* **H1 (cue-edge error gate fixes the supervised control):** `S/E-cue`
  passes vs `supervised`. A-priori expectation: fails on forgetting
  (< 0.05) while the A guard holds.
* **H2 (fewer cue lines):** at least one of `S/S-cue-k1`, `-k2`, `-k4`
  passes vs `supervised`. No directional expectation (fewer lines means
  both less false-positive cost and less recovery).
* **H3 (pattern-edge gate is sufficient):** `E/S-cue` passes vs
  `error-gated`, i.e. the error gate on pattern edges makes the cue useful
  even when cue edges follow the supervised rule.
* **H4 (attribution of #120's cue benefit):** the cue benefit under an E
  pattern gate exceeds the benefit under an S pattern gate, both with an E
  cue gate: per seed, d = [F(`error-gated`) − F(`E/E-cue`)] −
  [F(`supervised`) − F(`S/E-cue`)] (F = forgetting) has mean ≥ 0.05, > 0 in
  ≥ 32/40 seeds and CI lower bound > 0.
* Reported, not hypotheses: all rules at flip 25 % and 100 %; every arm vs
  `supervised`; seeds 1..20 vs 21..40 split; conflict both-correct and the
  #120 ceiling (computed for the active cue lines); weights changed.
* All constants are fixed a priori; no other configuration is run.
  Negative results are reported as produced.

```bash
cargo +1.89 run --release -- bench-cue-gating --csv results/cue-gating-v1.csv
```

## Results (pre-registered grid; CSV `results/cue-gating-v1.csv`)

Nothing above this section was changed after the run.

CSV: 1080 rows (3 flips × 40 seeds × 9 arms), sha256
`4315b1575a9ac79b0ce4470c38bf0891c9aac8b7c86739f77a02d6f944cfc639`. A second
run gave a byte-identical CSV and identical stdout; the grid takes ~8 s
(release build, CPU). `topology_unchanged=true` in every row. Greedy cue
sizes as in #120 (7.7 / 8.9 lines, 24/24 targets reached; k1: 6.3 / 5.0
targets, k2: 11.1 / 9.2, k4: 17.9 / 15.8).

**Reproduction of #120.** On seeds 1..20, `supervised`, `error-gated`,
`S/S-cue` and `E/E-cue` reproduce `results/context-cue-v1.csv` exactly for
A train, A test after A / after B, B test, forgetting, conflict
forgetting, conflict both-correct and weights changed (2520 values
compared, 0 differences). The only differing column is `ceiling_fixable`
of the two **uncued** arms, which #120 computed for the reserved cue and
this CSV reports as 0 (no active cue line); it is a diagnostic and enters
no rule.

### Primary (flip 50 %, 40 seeds), mean ± sd

| arm | A train | A test (after A) | A TPR / TNR (after A) | A test after B | forgetting | B test | conflict forgetting | conflict both-correct | ceiling (active lines) | weights changed (A+B) | cue A edge weight after A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| supervised | 0.949±0.027 | 0.903±0.038 | 1.000 / 0.807 | 0.580±0.060 | 0.324±0.072 | 0.861±0.044 | 0.703±0.100 | 0.000 | — | 1456.2 | — |
| error-gated | 1.000±0.000 | 0.967±0.023 | 1.000 / 0.933 | 0.559±0.064 | 0.408±0.069 | 0.967±0.022 | 0.901±0.056 | 0.000 | — | 80.2 | — |
| S/S-cue | 0.890±0.045 | 0.853±0.043 | 1.000 / 0.705 | 0.587±0.054 | 0.265±0.065 | 0.818±0.053 | 0.565±0.086 | 0.075±0.060 | 0.842 | 6310.5 | 0.43 |
| S/E-cue | 0.949±0.027 | 0.903±0.038 | 1.000 / 0.807 | 0.614±0.051 | 0.290±0.063 | 0.862±0.043 | 0.627±0.095 | 0.090±0.064 | 0.842 | 1520.2 | 0.15 |
| E/S-cue | 0.923±0.038 | 0.894±0.032 | 1.000 / 0.788 | 0.580±0.055 | 0.314±0.059 | 0.896±0.047 | 0.690±0.084 | 0.115±0.073 | 0.842 | 4934.3 | 0.43 |
| E/E-cue | 1.000±0.000 | 0.967±0.023 | 1.000 / 0.933 | 0.658±0.055 | 0.309±0.063 | 0.968±0.021 | 0.690±0.105 | 0.229±0.097 | 0.842 | 157.8 | 0.15 |
| S/S-cue-k1 | 0.936±0.030 | 0.892±0.037 | 1.000 / 0.784 | 0.584±0.061 | 0.309±0.072 | 0.852±0.044 | 0.670±0.093 | 0.022±0.031 | 0.201 | 2365.4 | 0.43 |
| S/S-cue-k2 | 0.920±0.035 | 0.878±0.038 | 1.000 / 0.757 | 0.584±0.058 | 0.294±0.070 | 0.845±0.048 | 0.639±0.094 | 0.036±0.037 | 0.368 | 3127.4 | 0.43 |
| S/S-cue-k4 | 0.905±0.041 | 0.867±0.041 | 1.000 / 0.733 | 0.585±0.055 | 0.281±0.067 | 0.833±0.054 | 0.606±0.096 | 0.056±0.047 | 0.609 | 4466.9 | 0.42 |

### Paired comparisons (flip 50 %, 40 seeds)

Each cell: mean ± sd [bootstrap 95 % CI] better/tied/worse seeds
("better" = x forgets less, or x is higher).

| x vs base | Δ forgetting | Δ A test | Δ A TNR | Δ retained A | Δ B test | rule |
|---|---|---|---|---|---|---|
| S/S-cue vs supervised | +0.058±0.040 [+0.046, +0.071] 39/0/1 | **−0.051±0.031** [−0.060, −0.042] 0/2/38 | −0.101±0.061 0/2/38 | +0.008±0.029 [−0.000, +0.017] 26/0/14 | −0.043±0.027 0/2/38 | fails (A guard) |
| **S/E-cue vs supervised (H1)** | +0.034±0.032 [+0.025, +0.044] 32/7/1 | **+0.000±0.000 0/40/0** | +0.000 0/40/0 | +0.034±0.032 [+0.025, +0.044] 32/7/1 | +0.001±0.008 [−0.002, +0.003] 10/24/6 | **fails** (mean < 0.05) |
| S/S-cue-k1 vs supervised (H2) | +0.015±0.021 [+0.009, +0.022] 25/14/1 | −0.011±0.015 0/19/21 | −0.022±0.030 | +0.004±0.015 16/14/10 | −0.009±0.012 | fails |
| S/S-cue-k2 vs supervised (H2) | +0.030±0.030 [+0.021, +0.040] 34/4/2 | −0.025±0.024 0/10/30 | −0.050±0.048 | +0.005±0.018 18/6/16 | −0.016±0.016 | fails |
| S/S-cue-k4 vs supervised (H2) | +0.042±0.037 [+0.032, +0.054] 36/2/2 | −0.037±0.030 0/5/35 | −0.073±0.060 | +0.006±0.027 23/2/15 | −0.028±0.021 | fails |
| **E/S-cue vs error-gated (H3)** | +0.094±0.047 [+0.080, +0.109] 40/0/0 | **−0.073±0.036** [−0.083, −0.062] 0/0/40 | −0.145±0.072 0/0/40 | +0.021±0.034 [+0.011, +0.032] 30/0/10 | **−0.071±0.046** 0/0/40 | **fails** (A and B guards) |
| E/E-cue vs error-gated | +0.100±0.050 [+0.084, +0.115] 40/0/0 | +0.000 0/40/0 | +0.000 | +0.100±0.050 40/0/0 | +0.001±0.009 13/18/9 | passes |
| E/S-cue vs E/E-cue | −0.006±0.063 [−0.025, +0.014] 18/0/22 | −0.073±0.036 0/0/40 | −0.145±0.072 | −0.078±0.048 2/0/38 | −0.072±0.045 0/0/40 | fails |
| E/E-cue vs supervised | +0.015±0.070 [−0.006, +0.037] 25/0/15 | +0.063±0.035 40/0/0 | +0.127±0.071 | +0.079±0.050 38/0/2 | +0.107±0.044 40/0/0 | fails (forgetting) |

**H4 statistic** d = [F(error-gated) − F(E/E-cue)] − [F(supervised) −
F(S/E-cue)]: **+0.066±0.040 [+0.054, +0.078], 37/3/0** (seeds 1..20:
+0.058 [+0.040, +0.079] 18/2/0; seeds 21..40: +0.073 [+0.060, +0.087]
19/1/0).

Seed blocks (flip 50, Δ forgetting / Δ A test): S/S-cue vs supervised
1..20 +0.066 / −0.054 (the #120 numbers), 21..40 +0.051 / −0.048; S/E-cue
vs supervised 1..20 +0.035 / 0.000, 21..40 +0.033 / 0.000; E/S-cue vs
error-gated 1..20 +0.106 / −0.082, 21..40 +0.082 / −0.064.

### Secondary flips (reported, not hypotheses; not corrected for multiplicity)

| flip | x vs base | Δ forgetting | Δ A test | Δ B test | rule |
|---|---|---|---|---|---|
| 25 | S/S-cue vs supervised | +0.019±0.029 26/1/13 | −0.051 0/2/38 | −0.050 0/0/40 | fails |
| 25 | S/E-cue vs supervised | +0.011±0.016 [+0.006, +0.015] 25/11/4 | +0.000 0/40/0 | −0.002 | fails |
| 25 | S/S-cue-k1 / k2 / k4 vs supervised | +0.005 / +0.012 / +0.015 | −0.011 / −0.025 / −0.037 | −0.010 / −0.019 / −0.032 | all fail |
| 25 | E/S-cue vs error-gated | +0.042±0.027 40/0/0 | −0.073 0/0/40 | −0.073 0/0/40 | fails |
| 25 | E/E-cue vs error-gated | +0.043±0.027 39/1/0 | +0.000 | −0.004 | fails (mean < 0.05) |
| 100 | S/S-cue vs supervised | +0.149±0.058 40/0/0 | −0.051 0/2/38 | −0.033 | fails (A guard) |
| 100 | **S/E-cue vs supervised** | **+0.092±0.048 [+0.078, +0.106] 40/0/0** | **+0.000 0/40/0** | +0.003 | **passes** |
| 100 | S/S-cue-k1 vs supervised | +0.038±0.031 33/7/0 | −0.011 | −0.005 | fails (mean < 0.05) |
| 100 | S/S-cue-k2 vs supervised | +0.069±0.045 [+0.055, +0.083] 38/2/0 | −0.025 | −0.012 | passes |
| 100 | S/S-cue-k4 vs supervised | +0.105±0.054 [+0.089, +0.121] 40/0/0 | −0.037 | −0.021 | passes |
| 100 | E/S-cue vs error-gated | +0.202±0.073 40/0/0 | −0.073 | −0.063 | fails (A guard) |
| 100 | E/E-cue vs error-gated | +0.206±0.072 40/0/0 | +0.000 | +0.006 | passes |

H4 statistic: flip 25 +0.032 [+0.025, +0.040] 37/3/0; flip 100 +0.114
[+0.100, +0.128] 40/0/0.

## Pre-stated readings

* **S0 (structural): holds.** In all 720 rows with an S pattern gate
  (3 flips × 40 seeds × 6 arms), pattern-edge weights after A and after B
  are bit-identical to the `supervised` control and A test after A has
  FN = 0 (TPR = 1.000). Every A-test change of those arms is a TNR change.
  The #120 H1 failure is therefore an **evaluation-time false-positive
  cost of the cue drive**, not a change in what the pattern edges learn:
  `S/S-cue` loses 0.101 TNR (A test −0.051 on 40 seeds; −0.054 on #120's
  20 seeds).
* **H1 (S/E-cue vs supervised, flip 50): fails** on the forgetting
  criterion, as expected a priori. The cue-edge error gate removes the A
  cost entirely (A test identical in 40/40 seeds, B test +0.001) and
  forgetting still falls by +0.034 (32/40 better, CI [+0.025, +0.044]),
  but the mean is below the 0.05 threshold. Cue A edges end at weight 0.15
  (vs 0.43 under S/S), so little cue drive is left to recover A after B.
* **H2 (fewer cue lines, flip 50): fails** for k = 1, 2, 4. Fewer lines
  shrink the A cost (−0.011 / −0.025 / −0.037, all inside the guard) and
  the forgetting reduction together (+0.015 / +0.030 / +0.042, all < 0.05).
  No k reaches both.
* **H3 (E/S-cue vs error-gated, flip 50): fails.** The error gate on
  pattern edges alone does not make the supervised cue rule usable: the
  forgetting reduction is large (+0.094, 40/40) but A test drops −0.073
  and B test −0.071 (40/40 seeds worse in both), again purely through TNR
  (−0.145).
* **H4 (attribution, flip 50): passes.** With the same (E) cue-edge gate,
  the cue cuts forgetting by 0.066 more under an E pattern gate than under
  an S pattern gate (37/3/0, CI [+0.054, +0.078], both seed blocks
  positive).

## Findings, including negative ones

* **Answer to "why does the cue hurt supervised A-test?"** Structural, not
  a generalisation failure of the pattern edges: under the supervised
  pattern rule the pattern edges learn exactly what they learn without the
  cue, and A positives are always recalled, so any cue drive into a target
  can only add false positives on A after A. With the supervised rule on
  the cue edges, those edges end around 0.43 and add them.
* **The two edge classes play different roles.** Gating the **cue edges**
  is what protects A test (S/E and E/E: ΔA = 0.000 in 40/40 seeds; S/S and
  E/S: −0.051 and −0.073). Gating the **pattern edges** is what turns the
  cue into a large forgetting cut (H4). Only the combination E/E passes
  the #120 rule at flip 50 (+0.100, 40/40, reproducing #120's H2 on 40
  seeds).
* **No fix for the supervised control at the primary flip.** None of the
  pre-registered supervised-pattern variants (S/E, k1, k2, k4) passes at
  flip 50; the best (S/E) keeps A intact but delivers only +0.034. At flip
  100 (secondary, uncorrected) S/E passes cleanly (+0.092, ΔA 0.000,
  40/40) and k2/k4 pass inside the guard — the cue helps the supervised
  rule when most pairs conflict, not at 50 %.
* **Budget.** S/E costs 1520 weight changes vs 1456 for the control
  (+4 %), S/S 6311 (+333 %), E/S 4934. Same 512 episodes everywhere.
* **Bit-level reproducibility.** #120's four arms reproduce exactly on
  seeds 1..20; the effect sizes on the new seeds 21..40 have the same sign
  and similar size for every comparison above.

## Limitations

Synthetic generated graph only (not BANC; DOI 10.7910/DVN/7WTH1N is not
built on the box). Artificial OR-rule task. One graph size (64/24/6),
threshold 1, init weight 1 (point 2 of the structural analysis depends on
init ≥ 1 and the OR teacher), one cue-selection rule (#120 greedy cover,
cap 12) and greedy-prefix line counts, 40 seeds, 64 held-out test patterns,
no rehearsal or protection. H1–H4 are four tests at one flip rate without
multiplicity correction; the flip-25/100 rows are secondary. The cue is a
set of input lines switched on together, an engineering device; nothing
here models a biological context signal.

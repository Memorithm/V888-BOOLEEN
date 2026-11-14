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

## Results

(Filled in after the pre-registered run; nothing above this line is
changed after the run.)

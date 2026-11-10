# B888-4 — generalisation and forgetting at equal budget (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository nor on the build
machine. Every number below comes from the deterministic generated graph of
`consolidation_scale::generate_graph` with artificial labels. Nothing here is a
claim about biology, insect memory or BANC physiology. `--graph brain.v888b`
runs the same code on a user-built graph; no such run has been made.

Module: `src/generalization_bench.rs`. CLI: `bench-generalization`.

## Protocol (fixed before running)

* Graph (default): 64 input neurons, 24 targets, fan-in 6 (144 edges, contact
  counts 1..3), regenerated per seed. Only original edges carry a 4-bit
  coefficient; anatomy is never modified (`topology_unchanged` checked).
* Patterns: 4 distinct active inputs. Per seed, three disjoint sets are drawn:
  32 A-train, 32 B-train, 64 TEST patterns. TEST combinations never appear in
  training (generalisation to new input combinations). The fraction of TEST
  edge activations never activated during A training is reported.
* Task A: hidden per-edge valence (50 % positive). A reached target must fire
  iff at least one active edge into it is positive (OR rule; exactly
  representable at threshold 1).
* Task B: same rule with 50 % of valences flipped. A perfect B learner scores
  the reported A/B label agreement on A.
* Equal budget: same graph, pattern sets, episode order (8 rounds x 32
  patterns = 256 episodes per phase, LCG shuffle) and initial weight for every
  arm. Initial weights 0 and 1 are both run.
* Arms: `frozen` (baseline), `supervised` (control), `associative-coactivation`
  (teacher-clamped post-synaptic state), `reward-modulated` (three-factor:
  eligibility = active pre AND the circuit's own predicted post spike;
  per-target reward +1 if that spike was correct, -1 otherwise).
* Metric: balanced accuracy over (pattern, reached target) pairs (constant
  predictors = 0.5). Forgetting = bal(A test after A) - bal(A test after B).
* Seeds 1..10; mean ± sample sd [min; max]; paired differences vs the
  same-seed, same-init supervised control.
* Verdict rule: "beats control" if the mean paired difference is >= 0.05 in
  the arm's favour and the arm is better in >= 80 % of seeds (for forgetting,
  B test accuracy must also not drop by more than 0.05); "worse" if the mean
  difference is >= 0.05 against it; else "no clear difference".
* Checkpoint/resume: saving after A with the existing `V888PLS1` file and
  resuming B is bit-identical to a continuous run for every arm (tested).

```bash
cargo +1.89 run --release -- bench-generalization --csv results/generalization-forgetting-v1.csv
```

## Primary results (defaults; CSV `results/generalization-forgetting-v1.csv`)

TEST edges unseen during A training: 0.099 (mean). A/B label agreement on
TEST: 0.535.

| init | arm | A train | A test (gen.) | A test after B | B test | forgetting |
|---|---|---|---|---|---|---|
| 0 | frozen | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 |
| 0 | supervised | 0.948±0.018 | 0.880±0.032 | 0.610±0.037 | 0.818±0.027 | 0.270±0.060 |
| 0 | associative | identical to supervised | | | | |
| 0 | reward-modulated | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 (never updates) |
| 1 | frozen | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 |
| 1 | supervised | 0.945±0.017 | 0.870±0.039 | 0.617±0.036 | 0.803±0.027 | 0.253±0.070 |
| 1 | associative | identical to supervised | | | | |
| 1 | reward-modulated | 0.945±0.017 | 0.870±0.039 | 0.728±0.021 | 0.716±0.040 | 0.142±0.050 |

Paired vs supervised (init 1, reward-modulated): generalisation +0.000 (10/10
ties); forgetting reduced by +0.111±0.034 (10/10 seeds) but B test −0.087±0.027.
Verdicts: generalisation *no clear difference*; forgetting *no clear
difference* (the B-cost guard of the pre-stated rule fails).

### Negative results and explanations

* **No strategy generalises better than the supervised control.** Associative
  co-activation is bit-identical to supervised here (generated targets have no
  outgoing edges, so clamping the teacher into the observed state reproduces
  the supervised update; tested). Reward-modulated from init 1 reaches exactly
  the same weights after task A (under the OR rule a positive edge is never
  decremented, so every label-1 target fires and eligibility equals the
  teacher's; tested).
* **Reward-modulated from init 0 learns nothing**: no target ever fires, so
  there is no eligibility. This is a structural limit of the rule, not a bug.
* **Less forgetting is a stability/plasticity trade, not a free gain**: the
  reward rule cannot raise an edge that was zeroed in A and must now fire
  alone in B (its target stays silent), which both protects A and costs B.
* Generalisation gap of the supervised control: 0.945 train vs 0.870 TEST
  (init 1), with ~10 % of TEST edges unseen in training.

## Exploratory sensitivity runs (not pre-registered, same seeds)

| flags | supervised A test (init 1) | reward vs control, forgetting / B (init 1) | verdict |
|---|---|---|---|
| `--train-patterns 8` (57 % TEST edges unseen) | 0.670 (train 0.967) | +0.033 / −0.028 | no clear difference |
| `--flip-pct 25` | 0.870 | +0.064 (10/10) / −0.044 | forgetting: beats control (marginal; B still lower) |
| `--flip-pct 100` | 0.870 | +0.203 / −0.200 | no clear difference (B lost) |
| `--active 2` (32 % unseen) | 0.796 (train 0.976) | +0.095 / −0.090 | no clear difference |

The single "beats control" (flip 25 %) is exploratory and is the same
trade-off with a B cost just under the 0.05 guard; it is not claimed as a
strategy advantage.

## Caveats

Synthetic, small (144 edges), artificial OR-rule labels, threshold 1, 10
seeds, no confidence interval beyond per-seed spread. Not evidence about BANC.

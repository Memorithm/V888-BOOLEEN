# B888-4 — limits of the reward-modulated strategy (synthetic graph)

**Synthetic data only.** The real BANC V888 graph (Harvard Dataverse
DOI 10.7910/DVN/7WTH1N; Bates, Phelps, Kim et al. 2026, *Nature*,
https://doi.org/10.1038/s41586-026-10735-w; BANC/FlyWire consortium; see
[SOURCES.md](SOURCES.md)) is **not** in this repository nor on the build
machine. Every number below comes from the deterministic generated graph of
`consolidation_scale::generate_graph` with the artificial OR-rule task of
[GENERALIZATION_FORGETTING.md](GENERALIZATION_FORGETTING.md). "Reward",
"credit" and "eligibility" name parts of an update rule on a Boolean graph;
nothing here is a claim about dopamine, biology, insect memory or BANC
physiology. `--graph brain.v888b` runs the same code on a user-built graph;
no such run has been made.

Module: `src/reward_limits.rs`. CLI: `bench-reward-limits`.

## Protocol (fixed in code and here before any number was looked at)

* Graph, task, pattern sets, episode order and metrics are those of
  `bench-generalization` defaults (64 inputs, 24 targets, fan-in 6; 4 active
  inputs; 32 A-train / 32 B-train / 64 TEST patterns; 8 rounds per phase =
  256 episodes; flip 50 %; balanced accuracy on clean labels; forgetting =
  bal(A test after A) − bal(A test after B)). Seeds 1..10. Initial weight 1
  (from 0 the reward rule never updates: already reported in #27).
* One axis of the reward channel is degraded at a time (12 conditions):
  * clean (`noise0_deliver100_per-target`) — reproduces the #27 reward arm
    bit-exactly (tested);
  * **reward noise** 10/20/30/40 %: each reward sign flipped;
  * **sparse reward** 50/25/10 %: an episode delivers a reward at all with
    this probability, otherwise no update;
  * **credit assignment**: `global` (one scalar per episode, +1 iff every
    predicted spike was right, applied to all eligible synapses),
    `delayed-trace-4` / `delayed-trace-32` (correct per-target reward with its
    stored eligibility applied 4 / 32 episodes later on the then-current
    weights; pending updates flushed at phase end, so no update is lost), and
    `global-delayed-no-trace-1` (the global scalar of episode k−1 applied to
    the eligibility of episode k: misattributed credit).
* **Matched control**: the supervised arm gets the same degradation where it
  means something — the teacher bit of a reached target is flipped by the
  same (seed, phase, episode, target) hash as the reward flip, and it skips
  exactly the same undelivered episodes. Credit modes do not apply to a
  teacher; there the control is the clean supervised arm.
* Equal budget: same episodes for both arms; update calls and weight changes
  are reported (the reward arm only updates on targets it fires).
* Instrumentation per phase: episodes, update calls, weights changed, reward
  events emitted, flipped by noise, undelivered episodes, and misattributed
  reward events (sign disagreeing with the clean per-target truth).
* Pre-stated reading: a condition **hurts** the reward arm if its mean paired
  drop in A-test balanced accuracy vs the clean reward arm (same seed) is
  ≥ 0.05. Verdicts vs the matched control use the #27 rule (margin 0.05,
  better in ≥ 80 % of seeds; forgetting verdict also requires B test not to
  drop by more than 0.05).
* Checkpoint/resume between A and B with the `V888PLS1` weight file is
  bit-exact under a degraded condition (noise 20, delivery 50, delay 4;
  tested).

```bash
cargo +1.89 run --release -- bench-reward-limits --csv results/reward-limits-v1.csv
```

## Results

(filled in from the run below; numbers copied as printed)

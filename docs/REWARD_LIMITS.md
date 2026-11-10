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

## Results (pre-registered grid; CSV `results/reward-limits-v1.csv`)

CSV: 240 rows (12 conditions × 10 seeds × 2 arms), sha256
`b61e2b06b59e40863594f9a9f4abe8c5a02f3dd1ba7cb2933bc4475a2f041f48`; a second
run gave a byte-identical CSV and identical stdout. Run time ≈ 5 s (release,
CPU). `topology_unchanged=true` for every row. Mean ± sample sd over 10 seeds;
paired deltas vs the same-seed matched control (b/t/w = better/tie/worse
seeds). "vs clean" = reward arm A test minus the clean reward arm, same seed.

| condition | reward A test | control A test | reward vs clean | hurts? | Δgen vs ctl (b/t/w) | reward forg. | control forg. | Δforg (+ = reward less) | reward B | control B | ΔB | verdicts gen / forg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 0.870±0.039 | 0.870±0.039 | +0.000 | no | +0.000 (0/10/0) | 0.142±0.050 | 0.253±0.070 | +0.111 (10/0/0) | 0.716 | 0.803 | −0.087 | none / none |
| noise 10 % | 0.834±0.050 | 0.830±0.042 | −0.036±0.021 | no | +0.005 (6/0/4) | 0.095 | 0.208 | +0.113 (10/0/0) | 0.677 | 0.760 | −0.082 | none / none |
| noise 20 % | 0.792±0.054 | 0.764±0.039 | −0.078±0.024 | **yes** | +0.028 (8/0/2) | 0.061 | 0.155 | +0.094 (9/0/1) | 0.640 | 0.714 | −0.074 | none / none |
| noise 30 % | 0.725±0.053 | 0.705±0.029 | −0.145±0.029 | **yes** | +0.020 (7/0/3) | 0.029 | 0.129 | +0.100 (10/0/0) | 0.592 | 0.653 | −0.061 | none / none |
| noise 40 % | 0.635±0.060 | 0.622±0.034 | −0.235±0.037 | **yes** | +0.013 (7/0/3) | 0.011 | 0.084 | +0.073 (9/0/1) | 0.542 | 0.591 | −0.049 | none / *beats_control* (see below) |
| deliver 50 % | 0.872±0.044 | 0.872±0.044 | +0.002±0.013 | no | +0.000 (0/10/0) | 0.105 | 0.218 | +0.113 (10/0/0) | 0.671 | 0.762 | −0.091 | none / none |
| deliver 25 % | 0.845±0.041 | 0.845±0.041 | −0.025±0.017 | no | +0.000 (0/10/0) | 0.081 | 0.190 | +0.109 (10/0/0) | 0.666 | 0.750 | −0.084 | none / none |
| deliver 10 % | 0.787±0.043 | 0.787±0.043 | −0.083±0.035 | **yes** | +0.000 (0/10/0) | 0.053 | 0.140 | +0.087 (10/0/0) | 0.654 | 0.719 | −0.065 | none / none |
| global credit | 0.559±0.030 | 0.870±0.039 | −0.311±0.035 | **yes** | −0.311 (0/0/10) | 0.030 | 0.253 | +0.223 (10/0/0) | 0.531 | 0.803 | −0.273 | **worse** / none |
| delayed trace 4 | 0.865±0.039 | 0.870±0.039 | −0.005±0.005 | no | −0.005 (0/4/6) | 0.139 | 0.253 | +0.115 (10/0/0) | 0.715 | 0.803 | −0.088 | none / none |
| delayed trace 32 | 0.858±0.041 | 0.870±0.039 | −0.012±0.006 | no | −0.012 (0/1/9) | 0.138 | 0.253 | +0.116 (10/0/0) | 0.709 | 0.803 | −0.094 | none / none |
| global, delay 1, no trace | 0.502±0.032 | 0.870±0.039 | −0.368±0.048 | **yes** | −0.368 (0/0/10) | 0.000 | 0.253 | +0.253 (10/0/0) | 0.508 | 0.803 | −0.296 | **worse** / none |

Instrumentation (mean per seed, A+B phases, from the CSV): update calls
reward / control — clean 2724.0 / 512.0, global 476.3 / 512.0, global no-trace
244.3 / 512.0; weights changed reward / control — clean 1978.9 / 2198.1,
noise 40 % 1986.8 / 4631.8, global 467.8 / 2198.1, global no-trace 280.6 /
2198.1, delayed-trace-32 1972.9 / 2198.1. Misattributed reward events in the
reward arm: global 103.1 of 476.3 updates, global no-trace 124.3 of 244.3.
A-train balanced accuracy: clean 0.945 (both arms), global 0.547, global
no-trace 0.501.

## Findings, including negative ones

* **Credit assignment is the hard limit.** Replacing per-target reward by
  one scalar per episode drops A-test generalisation from 0.870 to 0.559
  (−0.311, worse than the control in 10/10 seeds; A train only 0.547). One
  episode of misaligned credit without an eligibility trace drops it to
  0.502 — chance level. Observed mechanism (from the update counters, not a
  separate experiment): a negative global scalar decrements correct spikes
  too; once a target is silent it has no eligibility and can never be raised
  again, so learning stalls after ≈ 250–480 updates instead of ≈ 2700.
* **Delay with a correct eligibility trace is almost harmless here**: −0.005
  (d = 4) and −0.012 (d = 32), below the 0.05 threshold, but consistently
  slightly worse than the control (6/10 and 9/10 seeds worse).
* **Sparse reward degrades gracefully** (−0.025 at 25 % delivery) and only
  crosses the threshold at 10 % delivery (−0.083). The matched supervised
  control receiving labels on the same episodes scores **exactly** the same
  A test (10/10 ties at every sparsity): under the OR rule from init 1 the
  two rules coincide on task A (as in #27), so sparsity tells nothing that
  separates them.
* **Reward noise hurts from 20 %** (−0.078, −0.145, −0.235 at 20/30/40 %).
  The supervised control with matched label noise degrades about as much
  (0.830 / 0.764 / 0.705 / 0.622). The reward arm is slightly ahead
  (+0.005 to +0.028, better in 6–8 of 10 seeds) but never by the 0.05
  margin: **no robustness advantage is claimed**.
* **No condition makes the reward arm generalise better than the control.**
  Its lower forgetting persists in every condition but always with a lower B
  score (ΔB −0.049 to −0.296): the stability/plasticity trade of #27, not a
  gain. The single `beats_control` forgetting verdict (noise 40 %, +0.073,
  9/10, ΔB −0.049 just inside the 0.05 guard) is a **degenerate** pass: the
  reward arm learned little (A test 0.635, forgetting 0.011), so it had
  little to forget. It is reported as produced and not claimed as an
  advantage.

## Caveats

Synthetic, small (144 edges), bipartite generated graph (no recurrence, so
credit through intermediate neurons is not tested), artificial OR-rule
labels, threshold 1, init weight 1 only, 10 seeds, one axis at a time (no
interactions), per-seed spread only (no confidence intervals). "Equal budget"
means equal episodes and identical episode order; the reward arm makes more
update *calls* (one per fired target) but changes fewer weights. Not evidence
about BANC or biology.

# B888-4 — Exact joint credit in *three shared hidden neurons*

## Hypothesis, before seeing five-seed results

Can a **fixed four-neuron trainable microcircuit**, connected to the
physically evaluated but FROZEN 1,562,501-neuron Fourier backbone, learn
four sequential contextual Boolean tasks exclusively from their final
output labels, without separate task heads and without forgetting prior
TEST data? This **does not** show that all 1.56M hidden neurons trained,
or that structural BANC synapses learned.

## Frozen pre-registration

- Five seeds `0xb8886101`…`0xb8886105`; every seed reported.
- 6,250,000 source bit-planes (64-bit parallel evaluation);
  feature layer `tree.depth()-3` yields at least 15 disjoint
  frozen backbone neuron outputs. Three **mutable** hidden gates each
  receive five distinct backbone features, and one mutable root
  receives their three outputs plus the two observable context bits.
  No new head is allocated and the fixed topology does not grow.
- The synthetic teacher comprises three distinct nontrivial
  five-input parity/Walsh hidden functions (seed-specific masks from a
  *predeclared* palette), and a parity root of the three hidden
  outputs and both context bits; parity phase varies with seed.
  All targets lie inside the learner's own hypothesis family.
  This is a deliberately FAVORABLE, realizable benchmark.
- Four contexts presented sequentially: `00, 01, 10, 11`.
  Each stage receives 3 independent TRAIN x 64; 2 independent
  VALID x 64; 2 independent TEST x 64; and 2 post-backbone
  feature-OOD x 64, with distinct, fixed generation salts.
- TRAIN hides all examples where the first two features of hidden
  block zero are both 0. No hidden-activation *teacher* labels are
  revealed, and masked TRAIN labels/VALID/TEST/OOD do not influence
  candidate selection. Independent heldout combination accuracy is
  reported on TEST.
- OOD explicitly AND-masks each of the 15 frozen backbone features
  with a separate random word *after the frozen backbone*,
  deliberately reducing activation probability. Measure actual
  activation counts, not just assuming a representation shift.

## Multi-hidden credit and causal interaction

Stage 0: perform a **joint counterfactual** exhaustive search over
`31^3 * 2 = 59,582` joint proposals:
three hidden Walsh coupling masks and root phase.
Output accuracy on observed TRAIN is the ONLY reward. The root
coupling mask is initially fixed to the three hidden gates.
This avoids falsely claiming that one erroneous parity branch can
be repaired by a single mutation when TWO other incorrect branches
destroy its individual correlation with the output reward.

Stages 1–3: exhaustive 4 × 62 = 248 **individual gate**
proposals per stage, allowing any of three hidden models or
the root coupling mask/phase to mutate. A proposal is accepted
only for a strictly increased current TRAIN output score AND
nondecrease of each earlier TRAIN-task score under exact replay.
Measure changes to hidden masks and root, commits, TRAIN reward
before/after, per-task VALID, TEST and feature-OOD accuracy, and
all prior-task TEST forgetting after each stage.

A four-gate binary microcircuit cannot learn arbitrary combinations:
this finite Walsh-family experiment is an incremental precursor to
shared deep-backbone plasticity. An unchanged task 3 is not a
fourth accepted update.

## Counterexample: incompatible task under identical inputs

After the four contextual tasks, each seed receives a **fifth,
contradictory reward experiment** in which TRAIN inputs and context are
identical to the last learned task, but every observed label is inverted.
The new task is deliberately impossible to reconcile with perfect
retention of its original labels using a single deterministic circuit.
The exact replay guard must refuse any improvement that would damage
the original TRAIN objective. Report `HIDDEN_CONFLICT_LIMIT` with
original hits, contradicted hits, attempted improvement and acceptance,
and verify the persistent checkpoint remains bitwise unchanged.

This is a **negative control**, not evidence of fifth-task success:
no finite deterministic shared model can satisfy contradictory labels
on the exact same inputs and context simultaneously. Successful
rejection establishes the known capacity boundary of strict retention.
The conflicting task does not enter the four-task macro accuracy.

## Matched controls and prospective thresholds

Controls:
- frozen four-neuron model;
- same four-neuron model, identical proposal budgets, **without replay**;
- same learner/replay but TRAIN labels rotated between sample lanes
  (corrupted input/label relationship).

A positive result, on EACH of all five fixed seeds, requires:
1. at least 90% aggregate independent final TEST and VALID;
2. at least 90% on deliberately never-supervised feature combinations;
3. at least 90% post-backbone feature-shift OOD, *with the actual
   activation shift observed*;
4. ZERO regression on prior TEST after each later task;
5. at least two changed hidden neurons on stage 0 and at least
   two later accepted root coupling changes; three or more commits;
6. final independent TEST advantage ≥20 percentage points over
   frozen and shuffled-label controls on every seed.

All failures and null effects are retained. Do not change these
thresholds, seeds, splits or target masks after looking at outcomes.
Satisfying this restricted family is not evidence that generic
arbitrary functions, BANC physiology, root-only credit over the
1.56M hidden neurons or unlimited continual learning has been solved.

## Qualification

Run Rust 1.89 formatter, strict Clippy, all tests, all established
benchmarks and this exact new binary in isolated offline Docker on
Dell T430 via **Memorithm/RemoteOps alone**, on the exact pull
request SHA. Only merge after physical green proof, then validate
native CI on the exact merged main SHA. Do not infer CPU/DRAM Joules
or hardware cache events from prior runs; this experiment currently
reports elapsed time and the full output/credit trace, not RAPL.

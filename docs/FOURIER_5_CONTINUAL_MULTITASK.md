# B888-4 — Continual multi-task Fourier/Walsh : pre-registration

**Research question:** Can an explicitly task-addressed Boolean/Fourier
memory retain previously learned tasks while generalizing to unseen examples
and intentionally withheld feature combinations?

This is a deliberately narrow, **synthetic** experiment. The
6,250,000-source / 1,562,501-neuron Fourier backbone is evaluated physically
but **FROZEN**. We train eight **new** task-specific Fourier readout neurons
that all connect to the same five existing backbone features. It would be
false to claim that all 1.56 million backbone neurons learned, or that
global-root-only end-to-end credit assignment has been solved. Task IDs and
the five readout source indices are *known to the system*. Labels are
observable final **task-output** labels, never hidden teacher activations.

## Prespecified experimental design

- Three independent seeds: `0xb8884001`, `0xb8884002`, `0xb8884003`.
- For each seed, learn eight **distinct** Boolean tasks sequentially, chosen
  without examining VALID/TEST. Each is a parity/sign function on an unknown
  nontrivial subset of five frozen feature neurons, an explicit task-family
  prior. The target is generated from the same known Walsh-character
  hypothesis class: 62 candidates (31 nonzero masks × 2 phases). This is
  **not** evidence of discovering an unrestricted Boolean function.
- Features are extracted from depth-3 of the real frozen wide Fourier
  hierarchy; no source rerouting or hidden-label oracle during learning.
- Per task: 3 TRAIN × 64, 2 VALID × 64, 2 TEST × 64, and 2 shifted
  raw-source OOD × 64 samples. TRAIN/VALID/TEST/OOD are generated from
  **disjoint deterministic source seeds**; the circuit's lower parameters
  are unchanged across all tasks and splits. Report actual feature
  activations under the OOD source intervention; distribution shift may
  attenuate across multiple Boolean layers and must not be assumed.
- **Combinatorial holdout:** suppress TRAIN labels for ALL eight of the
  32 possible five-bit feature codes whose two low feature bits are 00.
  Do not fit on those masked lanes. Explicitly report TEST accuracy on
  the heldout codes separately from ordinary TEST. These are unseen
  supervised **feature combinations**, not entirely novel raw input bits.
- No TRAIN/VALID/TEST/holdout reallocation after inspecting results.
  Head selection uses TRAIN masked accuracy only (no VALID tuning).
  Inference uses compiled Shannon bit-slices, independent scalar reference
  evaluates teacher labels and test predictions.

## Persistent mechanism / controls

A new task head stores a learned Walsh mask, sign and revision. A task
**does not modify** another task's head. After each of eight training
steps: evaluate every prior task on the same disjoint TEST and assert zero
forgetting; serialize/restore every head using exact layer/source binding
and data-only checksum. This demonstrates task-addressed storage and
retention by allocation, **not** a solution to interference inside a fixed,
shared capacity. The checkpoint checksum is not cryptographic and does
not authenticate the backbone model or source data.

Compare on identical TEST examples:
1. Frozen, untrained parity readout (mask 31).
2. One shared head overwritten by the most recently trained task
   (the task-index routing is deliberately removed from this comparator;
   it is a lower-capacity architecture).
3. A separately trained head whose TRAIN labels were rotated across
   bit-lanes, maintaining nearly the same marginal label counts but
   destroying input/target pairing. This negative control is NOT an
   independent end-to-end architecture.

Both controls and every seed are printed even when unfavorable.

## Pre-declared thresholds for this *bounded synthetic task family*

Evidence of generalization requires all three seeds:
- final TEST macro accuracy ≥ 90%,
- unseen supervised 5-bit combinations TEST accuracy ≥ 90%,
- raw-source shift OOD macro accuracy ≥ 85%, *if* a measurable shift in
  readout feature activation is present; otherwise report OOD as raw
  input shift only, not a genuine representation shift,
- zero prior-task TEST regressions after each additional task,
- learner's final TEST macro accuracy ≥ frozen baseline by 20 pp, and
  ≥ shuffled-label control by 20 pp.

Do **not** relax these thresholds after seeing results. If they fail,
report the failure and improve the mechanism in a subsequent revision
with a new test set, not by leaking heldout labels into training.

Even if all criteria pass, the conclusion is restricted to task-addressed
Walsh-character readouts with prior-known task IDs, on a frozen synthetic
backbone. There is no demonstrated general-purpose continual learning,
unrestricted structural plasticity, real connectome training, or
language-model transfer.

## Outputs, costs and qualification

Every seed/task emits TRAIN observed accuracy, VALID, TEST, combinatorial
heldout TEST, raw-source-shift OOD, controls, checkpoint bytes, fit latency
and each previous task's retained TEST. The seed summary emits macro scores.
Rust 1.89 formatting, Clippy strict, unit tests, all previous physical
V888 benchmarks and the full 6.25M test are mandatory through the
single trusted RemoteOps control plane. A green exact-head qualification
is required before merge. ENERGY from this specific protocol is not
measured and must not be inferred from other benchmarks.

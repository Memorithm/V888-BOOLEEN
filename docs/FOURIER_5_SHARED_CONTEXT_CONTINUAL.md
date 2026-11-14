# B888-4 — Shared-context continual Fourier learning (pre-registration)

The narrow question is whether a **single shared, mutable** Boolean Fourier
readout neuron can learn sequentially from four different output-only task
rewards, change its own **effective Walsh feature couplings**, retain earlier
task performance and generalize to independently sampled inputs, without
allocating one head per task. This study does **not** train the 1,562,501
frozen backbone neurons, modify BANC anatomy or demonstrate general reward
propagation through the entire deep circuit. It is the direct architectural
successor to PR #100's independent task-addressed heads.

## Locked synthetic design

- Five seeds: `0xb8885101` through `0xb8885105`, ALL to be reported.
- Frozen wide Boolean backbone: 6,250,000 input bits, 1,562,501
  independently stored Fourier neurons; three readout features are extracted
  at depth minus three. No backbone parameters are changed.
- A *single* shared 5-input Fourier neuron receives those three features
  and two observable **context bits**, allowing four task contexts in the
  fixed order `00,01,10,11`. This single neuron is **not** a task-indexed
  memory bank. It has one learnable Walsh-character mask (five coupling bits)
  and phase, within a predefined 62-hypothesis class.
- Synthetic teacher is a shared Fourier character with all context couplings
  active and seed-dependent nontrivial feature mask and phase; it is
  exactly representable. This is a favorable **realizable task family**.
  Labels are generated through the independent scalar `FiveCircle.predict`;
  the learner uses output errors only, never hidden teacher activations.
- Every task independently samples three 64-lane TRAIN batches, two VALID
  batches, two TEST batches and two post-backbone feature-OOD batches from
  disjoint fixed generator salts. TRAIN labels for **all** examples with
  first two feature bits `00` are masked out: no fitting on those codes.
  Report TEST specifically on this withheld combination family.
- OOD deliberately masks the three *post-backbone* feature words with
  independent bits, targeting a 50%→25% active-feature shift. It does not
  claim full-system raw-source OOD robustness. Measure actual activation
  counts instead of assuming shift.

## Reward and credit

Each phase tests all 31 nonzero Walsh coupling masks × both phases.
The candidate's exact output improvement on observed TRAIN labels assigns
credit to its changed coupling bits and neuron's phase. Existing
per-task TRAIN examples are replayed; candidate updates are **rejected**
whenever any previous task's TRAIN accuracy decreases relative to the
current model. No VALID/TEST/OOD labels affect proposal selection.

Fit each task once, persist and restore the same single model from a
binding-aware, checksum-protected (not cryptographically authenticated)
46-byte checkpoint, then evaluate *all past tasks* on their original
independent TEST sets. No additional output neurons are allocated.

## Controls

- Frozen, untrained shared Fourier neuron.
- Same-capacity, same-update-budget single shared neuron with **no replay**.
- Same-capacity replay learner receiving cyclically corrupted TRAIN labels;
  identical task order and source samples.

Report each task, including pre/post test accuracy, TRAIN reward,
changed coupling bits, held-out combinations, validation, OOD and
every prior task's test retention. Show null effects and regressions.

## Prospective thresholds, five seeds

A favorable result in this **known, representable Walsh family** requires:

1. All five runs: final aggregate TEST at least **95%**, holdout
   combinations at least **95%**, and feature-OOD at least **95%**.
2. No prior task TEST accuracy regression after any subsequent task.
3. At least **two accepted shared-neuron parameter changes** per seed,
   and at least one context-related coupling update, rather than
   permanently allocating a separate head.
4. Final TEST advantage of at least **20 percentage points** over
   frozen and shuffled-label controls in each seed.
5. All checkpoints restore exactly and all finite hypotheses are measured
   under the same TRAIN-only selection protocol.

Do not change targets, splits or thresholds after observing physical
results. Failing a threshold must be explicitly recorded and requires a
new pre-registered experiment on new disjoint data.

Even full success only demonstrates **task-conditioned shared-parameter
learning of a known Walsh-character family**. It does not validate generic
catastrophic-forgetting resistance in bounded shared deep capacity,
unseen task families, reward temporal credit, or biological BANC synapses.

## Qualification / reproducibility

Rust 1.89 `cargo fmt`, strict Clippy, all unit tests and complete
multi-seed Dell physical execution in bounded, network-disabled unprivileged
Docker controlled exclusively by `Memorithm/RemoteOps`. Merge only once
SHA-exact qualification is green, then verify native main Rust CI.
No new energy claim: the existing RAPL + IMC measurements of PR #99
are **not transferable** to these new training windows.

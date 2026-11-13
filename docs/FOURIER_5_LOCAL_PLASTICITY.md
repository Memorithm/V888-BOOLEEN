# B888-4 — Per-neuron five-circle plasticity and long-horizon interference

Status: experimental, synthetic teacher and topology. Preregistered before
any physical qualification or benchmark results. No BANC-derived activity
or anatomical labels are used.

## Motivation and new mechanism

The previous [large-network study](FOURIER_5_LARGE_NETWORK.md) broadcast
one trained five-circle rule to 188,508 neurons. It did not make each
neuron learn independently.

The new module `src/fourier5_local_learning.rs` creates up to 250,000
individual `GatedFiveCircle` models. Every neuron receives its own
64 supervised TRAIN examples, holds its own integer context gate gains,
and receives no learned gains from other neuron identities.

The five Walsh mode masks, phases, persistent radii, and ten directed
incoming contacts are **fixed**, while each neuron independently learns
the five nonnegative boost values for fixed singleton context gates. This
is **local parameter learning**, not topology learning or learning the
mask identities. It is not unsupervised or biologically plausible
plasticity.

`FiveCircleLayer::install_models` and
`RecurrentFiveCircle::install_local_models` install exactly one model per
existing node; validate population size BEFORE any writes; preserve
topology and existing bit-packed state. Failed size validation cannot
partially install a model.

## Supervised task A/B and disjoint holdout

For each neuron i, a deterministic teacher chooses one contextual
bit `b_i` from 0..4 using the seed and neuron ID. Task A is:

`y_A = odd_parity(data_5) AND NOT(context[b_i])`.

Task B is another teacher with bit `(b_i + 1) mod 5`:

`y_B = odd_parity(data_5) AND NOT(context[(b_i+1) mod 5])`.

The learner is **not passed** the identity of the teacher bit and has
to discover effective boosts for its own singleton gates.

TRAIN contains 64 `(data,context,label)` pairs per neuron:
every possible 5-bit context, paired with one odd-parity and one
even-parity data word. TEST contains 64 different data words for
the identical contexts, constructed by flipping two data bits
while preserving their parity. All `(data,context)` pairs are
disjoint between TRAIN and TEST for a given neuron; contexts are not
held out and the target is planted by construction. TEST labels are
never accessed by `LocalPopulation::train`.

Each node is trained with at most 16 epochs using the existing
bounded integer `train_context_gates` implementation. Scores
and training budgets are reported separately. A seed-specific
deterministic gain fingerprint allows checking population stability;
it is not a cryptographic hash.

## Four explicit arms / interference design

- **Untrained**: initial zero contextual boosts; quantify A/B
  predictive quality before local training.
- **Frozen A**: train each neuron on A, save its weights unchanged.
- **Shared sequential A then B**: copy A-trained population, update
  *the same* five gains for B; measure A retained accuracy and B accuracy.
- **Task-indexed A/B bank**: keep a separate A-trained and B-trained
  population, with explicitly doubled learned model storage, and
  evaluate the correct task head. No free forgetting suppression is claimed.

All supervised evaluations use the same 64 disjoint held-out pairs
per neuron and task. Report `correct/examples`, number of
completely correct neurons, update events, training duration,
and model hashes. Average and worst-case across seed profiles should
be exposed in logs. Interference after B must be visible rather than
hidden by only reporting B.

Training count per phase = N * 64. For N=188,508,
**12,064,512 supervised examples per phase**, regenerated per
neuron rather than retained as a massive dataset. Teacher task
labels and network source topology are generated independently.

## Long recurrent evaluation — capability and limitations

Install independent models into *exactly the same seeded network
wiring* for frozen A, sequential A->B, and untrained controls.
Use serial, bounded profiles on the trusted Dell:

| N neurons | Directed contacts | Recurrent ticks | Topology |
|---:|---:|---:|---|
| 8,192 | 81,920 | 512 | ring |
| 65,536 | 655,360 | 256 | seeded pseudorandom |
| 188,508 | 1,885,080 | 128 | seeded pseudorandom |

Telemetry: active population, rising/falling transitions, Hamming
distance A-vs-sequential-B, and A-vs-untrained, initialization and
training durations. The models are **frozen during recurrent
evaluation**; survival of unchanged gains after 512 ticks is
parameter persistence by design, **not demonstrated long-term
associative recall**. The recurrent network receives no ongoing
teacher labels and cannot be said to learn from its own ticks.

A nontrivial long-term memory claim instead requires a
cue/delay/recall protocol, online plasticity during propagation,
delayed supervision and OOD tasks. This study establishes
prerequisites and quantifies known interference limits.

## Safety and qualification

Rust 1.89, CPU-only, no NVIDIA, no BANC data. Execute through
`Memorithm/RemoteOps` only in the reviewed SHA-locked nonroot
offline Docker container with <=4 CPU, <=6 GiB and <=25 minutes.
No global swap, resident service or production setting changes.

Required: `cargo fmt --all -- --check`,
`cargo clippy --offline --all-targets -- -D warnings`,
`cargo test --offline --all-targets`,
four existing Fourier synthetic bench commands, the new
`cargo run --offline --bin fourier5-local-bench`, then
`cargo run --offline -- info`.
No merges without green exact-HEAD qualification, required review
and subsequent native main CI.

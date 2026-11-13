# B888-4 — Scale-out of the Boolean five-circle recurrent network

Date: 2026-11-13. Status: experimental, **synthetic topology**.

## Objective

Move beyond single-neuron 32-state truth tables by testing recurrent
five-circle activity over thousands to hundreds of thousands of neurons,
while keeping the distinction between (a) a *learned shared local rule*,
(b) untrained, seeded synthetic network topology, and (c) absent BANC
biological dataset. Neither network topology nor synapses are learned here.

## Implemented architecture

`src/fourier5_recurrent.rs` provides a deterministic and bounded
`RecurrentFiveCircle` network: 64 to 250,000 bit-packed neurons,
ten explicit directed incoming contacts per neuron (5 data + 5 context),
a locally wired ring or a seeded pseudorandom directed network, and
synchronous snapshot updates. Sparse edges are stored as explicit
per-neuron indices; no N×N adjacency matrix is allocated.

Metrics: population, exactly ten contacts/neuron, a deterministic
noncryptographic wiring fingerprint, state occupancy, cumulative rising
and falling bit transitions, Hamming divergence after one input-bit
perturbation, and build/execution durations. `estimated_state_and_wiring_bytes`
includes only the logical buffer sizes of WiredNeuron records and one
bit-packed state; it is **not RSS, memory bandwidth, energy, or peak RAM**.

The implementation enforces a 250,000-neuron limit and checks source index,
configuration, density, dimensions and perturbation index. It operates on
a shared, already trained local rule: **each of the N nodes is not independently
trained**. Anatomical connection counts are not inferred.

## More difficult learned target

The supervised synthetic function on 10 binary inputs is

`fire = (popcount(data_5) mod 2 == 1) AND NOT((context_5 & 0b00101) == 0b00101)`.

Thus the neuron must implement a fifth-order parity mode **and** a conditional
two-bit suppression gate. The local Fourier model uses a fixed parity mode
(mask=31, radius=2), bias=-1, and up to five zero-radius context gate slots.
The `learn_context_masks` search must find the contextual suppression mask
from data. The fixed-singleton comparator gets the same number of slots,
the same samples, and the same supervised boost-update function but no
mask search; the context-blind control uses the frozen parity base.
All three see the same five data bits and five context bits in the training
protocol, though the frozen control ignores context by construction.

Exactly 1,024 distinct `(data,context)` patterns are generated, then
shuffled with a reproducible seed. The first 512 are TRAIN and the remaining
512 are held out; test labels are never consumed by training or search.
Two independent shuffle seeds are run and their accuracy plus training work
and elapsed wall time are reported. This is **random pattern holdout within a
known planted rule**, not a biological experiment or strict distribution
shift: it does not establish long-horizon generalization.

## Scale profiles — to measure, not assume

The dedicated `fourier5-large-bench` executable runs these three bounded
profiles sequentially in the trusted offline RemoteOps container:

| Neurons | Directed incoming contacts | Recurrent ticks | Topology |
|---:|---:|---:|---|
| 8,192 | 81,920 | 32 | local ring |
| 65,536 | 655,360 | 16 | seeded pseudorandom |
| 188,508 | 1,885,080 | 8 | seeded pseudorandom |

Three model replicas per profile: trained contextual, frozen context-blind,
and a clone of the contextual network with exactly one flipped state bit.
The replicas have the **same deterministic topology** and initial activity
except for the controlled perturbation. The final and maximal Hamming
distances quantify sensitivity, not learning quality. Always report failures,
loss of activity, divergence and the frozen control; do not interpret
activity as biological recall or intelligence.

## Resource, safety and comparison rules

All large-scale experiments are CPU-only Rust with no NVIDIA SDK,
no filesystem writes, no network, no credentials and no BANC data.
Use exclusively the existing RemoteOps exact-SHA qualifier in a
nonprivileged Docker container with <=4 CPU / 6 GiB and 25-minute job
timeout. Run the large benchmark once after unit tests and smaller
Fourier controls; do not start independent jobs on Dell/Thor/Mac, raise
global memory/swap settings, or displace production services.

Gates: `cargo fmt --all -- --check`,
`cargo clippy --offline --all-targets -- -D warnings`,
`cargo test --offline --all-targets`,
`cargo run --offline --bin fourier5-bench`,
`cargo run --offline --bin fourier5-mask-bench`,
`cargo run --offline --bin fourier5-context-bench`,
`cargo run --offline --bin fourier5-large-bench`, and
`cargo run --offline -- info`, all at exactly the PR head. Publish the
actual run, sample accuracies, hardware time, graph size and SHA before
merging. Only tested implementation moves to main. A separate further
PR is needed for per-neuron local learning, more complex task topology,
matched algorithmic budgets and measured peak RSS/CPU/energy.

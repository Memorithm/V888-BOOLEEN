# Five-circle Fourier Boolean neuron (experimental)

This research module explores five variable-radius rotating circles as a
Boolean neuron in B888-3/B888-4. It does NOT model measured BANC physiology,
alter connectivity, or modify the reversible biological data encoding.
Implementation: `src/fourier5.rs` (pure Rust, no external dependency).

## Why the independent-circle proposal needs coupling

For binary `x_k`, each independent `r_k exp(i(omega_k*x_k+phi_k))`
is affine in `x_k`: `Z(x)=c+sum(d_k*x_k)`. Thus thresholding
`Re(Z)` is linear and thresholding `|Z|^2` has Boolean degree <= 2.
The two-input XOR magnitude-threshold example works if the condition
is reversed; it does NOT prove a five-input parity capability.

## Five coupled Fourier-Walsh circles

For five Boolean inputs encoded into bits `x` (0..31), define the
phase-coupling character

`chi_m(x) = (-1)^popcount(x & m)`

for each five-bit mask `m`. Exactly five circles carry trainable
nonnegative radii `r_k`, binary phases `phi_k` (0 or pi), and
discrete masks `m_k`. The signed neuron response is

`score(x)=bias + sum(k=1..5) r_k * (-1)^(phi_k/pi) * chi_{m_k}(x)`

and `fire(x)=[score(x)<0]`. This model uses AND/parity, negation
and integer addition at inference; there are no complex exponential
evaluations or floating-point operations.

Coupling matters: `m=31` is a fifth-order phase mode. One nonzero
circle of radius 7 and mask 31 produces five-input parity exactly.
Five singleton masks 1,2,4,8,16 with unit radii implement OR
(`bias=-4`), AND (`bias=4`) and majority-of-five (`bias=0`).
Five circles are not sufficient to represent every possible
five-input Boolean function.

## Mathematical reference and learning

The exact 32-mode Walsh spectrum is `W_m=sum_x (1-2*f(x))*chi_m(x)`.
For any 32-state truth table, the full reconstruction satisfies
`sum_m W_m*chi_m(x)=32*(1-2*f(x))`. This is an exhaustive oracle,
not the proposed five-circle model.

The code includes two experimental approaches:
1. Rank nonconstant Walsh coefficients by absolute value, retain
   five modes and the constant coefficient as bias. The spectral
   ranking is not classification-optimal: OR yields 31/32 here
   (incorrect prediction for input zero), even though a different
   five-circle configuration implements OR exactly.
2. Perform bounded, deterministic perceptron updates on signed
   radii and bias for fixed mode masks. Trainable signs become
   binary phases; absolute weights become nonnegative radii.
   The masks themselves are NOT learned, and convergence is
   NOT guaranteed for arbitrary Boolean targets.

## Qualification before merge and any optional integration

Run `cargo fmt --all -- --check`, `cargo clippy --offline
--all-targets -- -D warnings`, `cargo test --offline`,
and `cargo run --offline -- info` on the EXACT PR head through
the SHA-pinned trusted RemoteOps qualification workflow.

Compare against independent-circle modulus, singleton perceptron,
full 32-mode oracle, and five-mode coupled candidate on the same
32 truth-table states; then compare accuracy and CPU/RAM on
independent tasks with disjoint held-out patterns and matched budgets.
Full-table interpolation proves nothing about OOD generalization.

This module is independent of the connectome dynamics and carries
no biological interpretation. Integration into `BooleanState`
requires an explicit adapter and separate performance qualification.

Scientific background: standard Walsh-Fourier analysis of Boolean
functions; the transform itself is not an original invention.

## Context-gated, transient radii (second implementation)

Module `src/fourier5_dynamic.rs` implements a *separate* computational
hypothesis: each of the five existing Walsh circles can optionally respond
to the activity of five contextual neurons. Gate mask `g_k` is a conjunction
over these five context bits. Zero mask disables its gate. At one evaluation,

`r_eff,k(x,c) = r_base,k + boost_k * [g_k != 0 AND (c & g_k) == g_k]`

and the score sums the original signed Walsh characters with these effective
radii and the original bias. The persistent radius, mask and phase remain
unaltered. This strictly **increases model input and parameter capacity**:
the contextual variant reads ten bits (five data + five context), while the
original five-circle model reads five. Comparisons must account for this.

`GatedFiveCircle` exposes validated, exact-integer scoring and ephemeral
radius inspection. `FiveCircleLayer` wires an explicit five-source/five-context
set to each neuron in a packed `BooleanState` and performs a synchronous,
read-only transition into a new packed state. No anatomical edge or
nonexistent BANC annotation is fabricated. Source dimension, output count,
indices, masks, gains and training budgets are bounded.

`train_context_gates` implements a deterministic, capped online update on
the **context gains only** for supervised triples `(x,c,y)`. A misclassified
example adjusts each active gate by one unit in the direction that reduces
its margin error, saturating in the range 0..65535. Masks, base radii, phases,
bias and explicit wiring are frozen. Training is not guaranteed to converge
for arbitrary examples. This is a supervised rule, not evidence of biological
Hebbian plasticity.

### Reproducible falsifiable control

`cargo run --offline --bin fourier5-bench` evaluates an artificial task:
`y=(x_0 AND c_0)`. Training includes all 16 input patterns with input bit
`x_4=0` paired with context 0/1 (32 samples), and the evaluation includes
the other 16 patterns `x_4=1` (32 samples). A deterministic context-blind
classifier must assign the same answer to each identical `x` under both
contexts, so its test ceiling is **24/32**. A fixed base model predicts
negative for all 32 test samples and therefore scores 24/32. A radius
gate with a trained boost of two has an algebraic 32/32 solution.
These numbers are **mathematically expected outputs**, not hardware benchmark
results until Rust tests and executable have actually run.

This setup demonstrates *representational capacity with an extra context
input*, not model superiority at matched parameter budgets, memory performance,
or out-of-distribution generalization. The held-out bit is an irrelevant
feature by construction. More difficult controls must compare alternative
context-aware Boolean models of comparable complexity, with held-out contexts,
noisy inputs, different topology, memory and latency metrics.

### Exact-head qualification additions

Besides the original Rust 1.89 fmt / Clippy / all-targets tests / info smoke,
run `cargo run --offline --bin fourier5-bench` and verify the synthetic
control's 24/32 vs 32/32 scores in the trusted SHA-pinned RemoteOps container.
No unrestricted hardware execution, secrets, background workload or local
host modification is required.

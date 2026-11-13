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

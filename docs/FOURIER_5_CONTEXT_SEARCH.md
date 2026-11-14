# B888-4: learning dynamic contextual Fourier circle masks

Status: experimental supervised feature. No biological claims.

## Question

Previously, `src/fourier5_dynamic.rs` trained only contextual **radius
boosts**, assuming five `context_mask` values were manually specified.
`src/fourier5_mask_learning.rs` separately learned Fourier-Walsh **data
interaction masks**. This increment trains which **context bits jointly gate
each radius**, without changing the five persistent Fourier modes.

For the five circles, a gate is enabled if
`gate_mask != 0 && (context & gate_mask) == gate_mask`. Each active gate
adds a nonnegative learned integer boost to its persistent radius. A mask of
zero disables the gate.

## Implementation contract

- `src/fourier5_context_search.rs` provides
  `learn_context_masks(base, samples, epochs)`.
- TRAIN only: validate unique `(input, context)` pairs with an exact 1024-bit
  set (32 x 32). Reject empty, duplicate, invalid, and oversized samples.
- At most 512 examples, 64 training epochs, five circles and 31 candidate
  nonzero context masks per circle. Candidate evaluation is deterministic.
- Iteratively select the gate mask with minimum TRAIN classification error,
  then integer hinge loss, then smallest mask. Refit all accepted gains
  from zero for each candidate. Stop when no strict improvement occurs.
- Report selected gates, final training accuracy, refit update count,
  candidate evaluations, and total visited training examples. This is a
  greedy heuristic, not guaranteed globally optimal or efficient.
- The supplied base `FiveCircle` is immutable; BANC connectivity,
  recorded synapses and BooleanState storage are not altered.

## Preregistered synthetic falsifiable example

The target is `y = (input & 1 != 0) AND ((context & 0b01010) == 0b01010)`.
Use inputs 0 and 1, contexts 0..15 for TRAIN (32 rows), and contexts
16..31 for TEST (32 rows). The context high bit is deliberately
irrelevant to the target. Results do not measure difficult out-of-distribution
transfer.

`fourier5-context-bench` prints three outcomes on identical held-out rows:
- Frozen context-blind base with five Walsh circles (four zero-radius slots).
- Five fixed singleton context masks (one per circle), each boost learned
  with the same supervised `train_context_gates` routine.
- Up to five context masks selected on TRAIN only by this greedy search.

**Prior expectation** (to test, not a measured result):
the context-blind base scores 28/32, because four positives require context.
The learned contextual mask `0b01010` can fit all TRAIN patterns with
a two-unit boost, and can achieve 32/32 on TEST. The fixed-singleton
control is also capable of representing this task in principle using
two gates; its observed outcome must be reported, including negatives.

Mask search tests far more candidate fits than the singleton comparator.
Thus even perfect classification is not a like-for-like compute-budget win.
A stronger experiment must compare matched candidate-search budgets, model
bits, energy or CPU time, seeds, noisy labels, and nontrivial withheld
context features on V888 vs matched graph control vs learned topology.

## Qualification

Exact PR head: offline isolated Rust 1.89 `cargo fmt --all -- --check`,
`cargo clippy --offline --all-targets -- -D warnings`,
`cargo test --offline --all-targets`,
`cargo run --offline --bin fourier5-bench`,
`cargo run --offline --bin fourier5-mask-bench`,
`cargo run --offline --bin fourier5-context-bench`, then
`cargo run --offline -- info`, via trusted `Memorithm/RemoteOps`.
Only publish status on complete success and stable SHA.

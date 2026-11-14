# B888-4 — Hierarchical composition of the five-circle Boolean neuron

Status: exploratory, frozen inference only. **No plasticity**, no synaptic
updates, no anatomical BANC source, no claim of biological validity.

## New functional mechanism

A single V888 `FiveCircle` reads five Boolean inputs. This increment
constructs explicit five-way **feedforward** compositions of those unchanged
neurons into increasing representational depths:

| Depth | Input bits | Five-circle neurons | Directed contacts |
|---:|---:|---:|---:|
| 1 | 5 | 1 | 5 |
| 2 | 25 | 6 | 30 |
| 3 | 125 | 31 | 155 |
| 4 | 625 | 156 | 780 |

The lower layer reads the explicit input bits, and each higher layer reads
exactly the five outputs of a preceding group. There is no hidden dense
matrix, no recurrent edge, no fresh biological anatomical edge, and no
global state or service modification. The leaf routing is explicitly
validated as a bijection of 5^depth inputs to the first layer:
contiguous, interleaved, or seeded Fisher-Yates shuffled.

Every layer may have a different Boolean computational mode, with frozen
programmable `FiveCircle` model values per neuron. Four reference modes
are supplied: **parity, OR, AND, majority**. The public constructor also
accepts individually specified valid five-circle models.

## Compositional expressivity, not biologically trained learning

- Depth two can compute **parity over 25 independent bits** using six
  ordinary Fourier neurons, each itself a five-bit Walsh parity gate.
- Depth four can compute **parity over 625 bits** using 156 such neurons.
  For parity, bijective input rerouting leaves the result unchanged.
- Mixed gate layers create different functions sensitive to how inputs
  are grouped. The benchmark probes heterogeneous compositions:
  depth 3 = parity -> majority -> parity; depth 4 =
  majority -> parity -> AND -> OR.
- This is a modular way to represent larger Boolean dependencies, not
  a proof that it is more efficient than conventional circuits or
  neural networks. The construction uses explicitly specified gates,
  **no training**. The Fourier parity term is a Walsh character,
  not a physical oscillation.

## Matched-topology, exact independent comparison

The same **six / 31 / 156 neurons and identical wiring** are evaluated
through three encodings:

1. Scalar five-circle Walsh-Fourier inference using the existing
   `FiveCircle::predict` (exact signed integer radius and bias).
2. Scalar 32-bit truth-table lookups **compiled from those same exact
   five-circle models**, not independently trained.
3. A bit-sliced `u64` Shannon-decomposition evaluator that processes
   **64 independent inputs per call**. Each five-input compiled table
   is evaluated using 31 bitwise multiplexers (5 levels); external
   bitplane conversion is measured separately and **must be included**
   in any end-to-end runtime comparison.

An independent Boolean reference implementation computes each layer
from the mode definitions, rather than calling Walsh models or compiled
truth tables. Every model output on every benchmark case must match this
oracle, including all 64 bit-slice lanes. Additional tests enumerate
all 32 five-bit patterns for every reference gate.

Reported costs are **actual wall times** for: scalar Fourier, scalar
lookup, 64-lane bit-sliced kernel, and transpose+kernel; logical
routing/model/truth-table bytes are **not peak RSS**. The compiled LUT
uses 4 bytes per internal neuron and discards radii: it is equivalent
only for inference. There is no rationale to assert a performance
improvement unless actual measurements support it.

## Pre-registered physical workload

Three independent synthetic input seeds:
`0xa8320101, 0xa8320202, 0xa8320303`.
Each seed executes all three depths, with seeded shuffled routing:

| Depth | Distinct generated input patterns | Independent input bits |
|---:|---:|---:|
| 2 | 65,536 | 25 |
| 3 | 16,384 | 125 |
| 4 | 4,096 | 625 |

All 64 lanes are independently evaluated against the same independent
reference. For nontrivial grouping, record how often the **shuffled**
wiring disagrees with the **contiguous** wiring under the same input
and gate families, with exactly matched number of nodes and contacts.
For parity, verify routing invariance. Generated examples are
synthetic uniformly sampled bit patterns; *no TRAIN/TEST claims* are
made because nothing is learned in this study.

**Admissibility criterion:** every scalar, scalar lookup and bit-sliced
prediction agrees with the Boolean oracle on every sampled case. There
is no prespecified wall-time or speedup threshold; any slowdown is
published. Failures must be fixed without weakening these correctness
checks. The benchmark counts input transposition separately.

## Scope and next step

This is a first **frozen feedforward** composition. The network from
previous iterations with 188,508 recurrent neurons is preserved
unchanged. This PR does **not** test deep circuits as recurrent
subcircuits, learn which input connections are useful, or prove that
625-bit functions are well represented for arbitrary tasks.

Next isolated increments: bounded choice/rerouting of source indices
on TRAIN only, independent TEST/OOD holdouts and matched routed
Boolean baselines, then careful integration with large recurrent
circuit topology — **without switching plasticity on**.

## Qualification

Rust 1.89, all targets, format, Clippy -D warnings; existing small
Fourier benchmarks plus `cargo run --offline --bin fourier5-hierarchy-bench`,
and CLI smoke. Qualify under `Memorithm/RemoteOps` isolated non-root
offline Docker, no secrets or host filesystem modifications, unchanged
4CPU/6GiB/25min constraints, exact SHA validation before merging.
Only merge after green exact-HEAD evidence and verify native main CI.

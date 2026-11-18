# Packed 2-bit arithmetic: experimental kernel (Q2 / GF(4))

This research module is separate from BANC V888 connectivity, biological modeling, and the active event runtime. It makes no inference about synaptic types or weights. No speed or model-quality improvement is claimed.

## Representations and semantics

A pair of `u64` bitplanes (`low`, `high`) carries 64 lanes of two-bit codes. A Q2 code is `low_bit + 2*high_bit`. Two words are 16 bytes for 64 values: exactly two payload bits per value. This is not a one-word/64-value representation, and all memory estimates must count **both** planes, vector capacities, ancillary graph data and access patterns.

- Unsigned numeric Q2: 0,1,2,3.
- Signed numeric Q2 (two's complement): 0,1,-2,-1 for codes 0,1,2,3.
- GF(4) polynomial codes: 0,1,x,1+x with x²=x+1; addition is XOR, multiplication is not ordinary numeric multiplication or multiplication modulo four.

For `a=a0+2a1` and `b=b0+2b1`, the unsigned dot product over all lanes is

```text
popcnt(a0 & b0)
+ 2 * popcnt(a0 & b1)
+ 2 * popcnt(a1 & b0)
+ 4 * popcnt(a1 & b1)
```

For two-bit signed codes `a=a0-2a1`, `b=b0-2b1`, the same four popcounts use coefficients +1,-2,-2,+4.

GF(4) multiplication uses the basis x²+x+1: `c0=(a0&b0) XOR (a1&b1)` and `c1=(a0&b1) XOR (a1&b0) XOR (a1&b1)`.

The exact GF(4) inner product is **the field sum of field products**:
`⊕_i (a_i · b_i)`. For each 64-lane bitplane pair the implementation
computes the two polynomial components via AND/XOR and reduces each
component with the parity of `count_ones()`; it then XOR-accumulates
the two resulting bits across words. The result is a field code in
`0..=3`, not a numeric dot product. No heap allocation is required
for `gf4_dot` beyond the existing packed vectors.

## Correctness scope

`PackedQ2` rejects invalid levels and mismatched lengths. The tests cover empty buffers, 63/64/65 and 127/128/129 lane boundaries, a BANC-sized 188,508-lane synthetic buffer, numeric reference dots, scalar-oracle GF(4) inner products including incomplete last words, elementwise GF(4) scalar references and all 16 single-lane GF(4) input pairs. Padding is initialized to zero and cannot be modified through the safe public interface.

The arithmetic is exact for its documented code interpretation. It does not imply equivalence to the Boolean V888 runtime because the latter uses exact graph contact counts, thresholds and event ordering.

## Qualification and promotion gates

1. Format, Clippy, compile and unit tests on the **exact branch SHA**. No hardware-dependent tuning in the reference module.
2. Establish end-to-end baselines and representative data distributions; compare against scalar Q2 and existing Boolean kernels. Record unpacking/packing costs, memory allocation, cache misses if available and complete wall time.
3. Profile on Dell Xeon AVX2; do not infer throughput from bit density or extrapolate from isolated kernels to graph propagation.
4. Keep the default V888 event runtime unchanged until deterministic equivalence or a separately authorized non-equivalent research mode has been established.
5. Only then consider reusing this module in SciRust GF(4) experiments and SML / FLAT exploratory kernels; require independently benchmarked evidence.

This is classical deterministic bit-sliced execution, not quantum computation, magic-state simulation or an intrinsic QML speedup.

## Paired scalar/packed GF(4) inner-product probe

The optional CPU binary `v888-gf4-bench` compares a scalar GF(4)
reference against `PackedQ2::gf4_dot` on identical deterministic
four-level inputs. It requires exact field-code equality for every
measured repetition and aborts otherwise. It alternates the order of
the two modes and reports median nanoseconds independently. Its
measurements exclude input generation, packing, warmup and result
verification; the resulting times do **not** describe an end-to-end
inference workload or a Boolean graph-propagation speedup.

The output reports input payload bytes for both implementations:
two scalar bytes per input pair versus two packed bitplanes per vector,
each rounded to complete `u64` words. It excludes vector capacity
over-allocation, allocator metadata, thread/runtime costs and other
data structures. Synthetic input coefficients are not anatomical
parameters or calibrated workloads.

CPU-only example:

```bash
cargo test --offline --all-targets
cargo run --offline --release --bin v888-gf4-bench -- \
  --lanes=188508 --repetitions=11
```

The bounded CLI accepts 1..1,048,576 lanes and 3..64 paired
repetitions. A credible performance claim additionally requires
several independently pinned physical runs, noise/CPU-affinity
documentation and an end-to-end packing-inclusive comparator.
Energy is not measured.

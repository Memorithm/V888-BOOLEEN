# Packed 2-bit arithmetic: experimental kernel (Q2 / GF(4))

This research module is separate from BANC V888 connectivity, biological modeling, and the active event runtime. It makes no inference about synaptic types or weights. No speed or model-quality improvement is claimed.

## Representations and semantics

A pair of \`u64\` bitplanes (\`low\`, \`high\`) carries 64 lanes of two-bit codes. A Q2 code is \`low_bit + 2*high_bit\`. Two words are 16 bytes for 64 values: exactly two payload bits per value. This is not a one-word/64-value representation, and all memory estimates must count **both** planes, vector capacities, ancillary graph data and access patterns.

- Unsigned numeric Q2: 0,1,2,3.
- Signed numeric Q2 (two's complement): 0,1,-2,-1 for codes 0,1,2,3.
- GF(4) polynomial codes: 0,1,x,1+x with x²=x+1; addition is XOR, multiplication is not ordinary numeric multiplication or multiplication modulo four.

For \`a=a0+2a1\` and \`b=b0+2b1\`, the unsigned dot product over all lanes is

\`\`\`text
popcnt(a0 & b0)
+ 2 * popcnt(a0 & b1)
+ 2 * popcnt(a1 & b0)
+ 4 * popcnt(a1 & b1)
\`\`\`

For two-bit signed codes \`a=a0-2a1\`, \`b=b0-2b1\`, the same four popcounts use coefficients +1,-2,-2,+4.

GF(4) multiplication uses the basis x²+x+1: \`c0=(a0&b0) XOR (a1&b1)\` and \`c1=(a0&b1) XOR (a1&b0) XOR (a1&b1)\`.

## Correctness scope

\`PackedQ2\` rejects invalid levels and mismatched lengths. The tests cover empty buffers, 63/64/65 and 127/128/129 lane boundaries, a BANC-sized 188,508-lane synthetic buffer, numeric reference dots, elementwise GF(4) scalar references and all 16 single-lane GF(4) input pairs. Padding is initialized to zero and cannot be modified through the safe public interface.

The arithmetic is exact for its documented code interpretation. It does not imply equivalence to the Boolean V888 runtime because the latter uses exact graph contact counts, thresholds and event ordering.

## Qualification and promotion gates

1. Format, Clippy, compile and unit tests on the **exact branch SHA**. No hardware-dependent tuning in the reference module.
2. Establish end-to-end baselines and representative data distributions; compare against scalar Q2 and existing Boolean kernels. Record unpacking/packing costs, memory allocation, cache misses if available and complete wall time.
3. Profile on Dell Xeon AVX2; do not infer throughput from bit density or extrapolate from isolated kernels to graph propagation.
4. Keep the default V888 event runtime unchanged until deterministic equivalence or a separately authorized non-equivalent research mode has been established.
5. Only then consider reusing this module in SciRust GF(4) experiments and SML / FLAT exploratory kernels; require independently benchmarked evidence.

This is classical deterministic bit-sliced execution, not quantum computation, magic-state simulation or an intrinsic QML speedup.

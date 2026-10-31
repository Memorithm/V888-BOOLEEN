# V888 Boolean engine — implementation evidence and limitations

## What exists

An independent Rust directed-graph engine for an *edge-list projection* of BANC V888. It stores original u64 root IDs, target indices, CSR offsets, and the u64 `count` of synaptic contacts on every directed row. It preserves zero-count entries, self loops and duplicate rows (no silent filtering or collapsing). Sorting only canonicalises row order. An optional neuron-ID list preserves vertices with no edges.

The binary `V888BOL1` format is versioned, little-endian and tested with a roundtrip. Neurons are activated as packed single bits; experimental recurrent steps accumulate edge multiplicities using bitwise XOR/AND carries and compare thresholds bitwise. There are **no floating point computations** in these rules.

**Crucial:** a graph projection is NOT lossless encoding of all original Feather tables. Neuron coordinates, individual synapse positions, neurotransmitter estimates, anatomical compartment annotations, data uncertainty, and raw observations MUST remain in the source files with validated checksums. The projection cannot reconstruct them. Nor is this an emulator of the actual insect's physiological dynamics. Thresholds are arbitrary experimental parameters. There is no learning implemented.

## Official source

Bates, Phelps, Kim, Yang et al. (2026), *Distributed control circuits across a brain-and-cord connectome*, Nature, https://doi.org/10.1038/s41586-026-10735-w.
BANC snapshot (materialization v888, 17 April 2026), https://doi.org/10.7910/DVN/7WTH1N.

Paper connectivity product: `banc_888_edgelist_simple_v2.feather` (`pre`, `post`, `count`, `norm`, `post_count`, `pre_count`), approximately 11,510,975 directed edges. Complete neuron metadata: `banc_888_meta.feather`. Synapse-level biological data are a separate source and are NOT loaded by the current engine. Use v2, not v3, for the published paper version; the snapshots represent different detectors.

Never commit large raw source data into Git. Cache it on the Dell under the RemoteOps security and resource policy, with SHA-256, source URL, version, file size, retrieval date and licence. Preserve attribution.

## Commands

```bash
cargo run --release -- demo
cargo run --release -- build --edges data/processed/edges.csv --neurons data/processed/ids.txt --output data/processed/brain.v888b
cargo run --release -- info --graph data/processed/brain.v888b
cargo run --release -- run --graph data/processed/brain.v888b --seed ROOT_ID --threshold 3 --steps 4
cargo run --release -- export --graph data/processed/brain.v888b --output data/processed/roundtrip.csv
```

The numeric CSV `edges.csv` must have the first fields `pre,post,count`, without quoted integers. The ID file is one decimal root ID per line (optional header `id`). Convert Feather to numeric CSV with a separately qualified data-acquisition procedure. Compare source and exported **multisets** of directed edge/count rows; source row order is not preserved. Data are NOT bundled.

Limits: 1 million vertices, 300 million directed rows, max 1000 synchronous steps per invocation. Every sum is overflow checked. The binary loader verifies length, monotonic offsets and ID uniqueness. CPU and peak-RAM measurements on full-scale data are not yet established.

## Next criteria

1. Download and hash the actual v888 metadata, paper v2 edge list and original synapse-level data.
2. Build and audit the full 188k / 11.5M connectivity projection; independently confirm counts, ID coverage, SHA-256 and roundtrip.
3. Retain all original anatomy and add typed binary/bit-level representations without discarding information.
4. Validate graph traversal on the full connectome under bounded CPU/RAM budget.
5. Introduce independently tested biological hypotheses and training objectives; evaluate against matched synthetic graphs.

No claim of full BANC brain construction is valid until steps 1–3 have executed on real data.

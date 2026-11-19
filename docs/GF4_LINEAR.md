# Packed GF(4) linear layer (independent research)

The Rust module `src/gf4_linear.rs` implements an **exact multi-output field-linear transform** on top of the `PackedQ2` kernel: `y = W*x` in GF(4), using polynomial basis `GF(2)[x]/(x²+x+1)`.

- `PackedGf4Linear::from_rows(input_len, rows)` takes ownership of packed coefficient rows and rejects mismatched row widths.
- `apply(input)` allocates exactly one packed output vector, with one two-bit lane per output row.
- `apply_xor(input, output)` accumulates the same linear transform into an already allocated packed output without building an intermediate product or matrix; both input and output shapes are validated before any mutation.
- Row coefficients are reusable across applications; no repacking is required at inference time.

The first implementation is intentionally sequential and CPU-only. It does not alter the BANC V888 event runtime, Boolean contact counts, network topology, plasticity rules, thresholds, or source files. It does **not** support integer Q2 matrix multiplication; GF(4) addition is XOR, and field multiplication differs from integer multiplication.

Tests exercise 0/1/63/64/65/129 and 188,508 input lanes, more than one 64-bit output word, an identity transform, an independent scalar field oracle, field linearity, cancellation of duplicate accumulations, and shape-error fail-closed behavior. They do not prove biological learning or a performance/energy advantage.

## Composition de couches GF(4)

`PackedGf4Linear::compose(&previous)` produit une matrice compacte
équivalente à `self(previous(x))` sur le corps GF(4). Pour deux matrices
M×K et K×N, la composition M×N calcule chacune de ses colonnes avec
un unique vecteur de travail de K symboles à deux bits, réutilisé au fil des
N colonnes. Il n'existe pas de matrice transposée complète temporaire.
Les dimensions incompatibles sont rejetées sans modifier les deux
matrices initiales.

Les tests comparent la composition à deux applications consécutives,
et à un oracle scalaire indépendant, pour des dimensions vides,
impaires et franchissant 63/64/65/129 symboles. Le gain éventuel ne
concerne que la réutilisation d'une même composition sur plusieurs
inférences ; aucune accélération n'est affirmée sans mesures du coût
de composition, des allocations et de l'exécution complète.


## Banc de comparaison des couches

`cargo run --offline --release --bin v888-gf4-linear-bench` construit
deux couches GF(4) déterministes et mesure le coût d'une composition
matricielle exécutée une seule fois. Le banc compare ensuite les
applications successives aux applications de la matrice précomposée,
avec sept mesures appariées en ordre alterné et médiane par chemin.
Les entrées ont des tailles de 63, 65, 129 et 256 éléments, pour des
couches intermédiaires et sorties traversant les frontières 64 bits.

Avant chaque mesure, l'identité algébrique des deux chemins est
vérifiée. Les coûts de construction des matrices d'entrée et de
packing du vecteur d'entrée sont exclus des latences d'inférence ;
le coût de composition est affiché séparément. Les volumes mémoire
des coefficients compacts, des coefficients précomposés et du
vecteur intermédiaire sont rapportés pour exposer les cas où
la composition augmente la densité et le trafic mémoire.

Le benchmark est un probe CPU synthétique borné ; ses valeurs
observées ne permettent pas d'affirmer un gain en production ou
une économie d'énergie sans qualification physique et analyse de
rentabilité sur des charges et distributions représentatives.

## Qualification

Require Rust 1.89 `cargo fmt --all -- --check`, `cargo clippy --offline --all-targets -- -D warnings`, `cargo test --offline --all-targets` and exact-head RemoteOps physical qualification before merge. If later benchmarked, compare complete packed matrix-plus-input costs against a scalar GF(4) baseline at identical dimensions and distributions; record packing, memory footprint, latency and CPU separately. No performance or energy gain is claimed.

Cross-project candidates for SciRust and SML-GENIUS require their own independent contract, benchmarks and PRs.

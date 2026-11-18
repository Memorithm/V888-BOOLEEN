# Cache compact facultatif des sorties synaptiques CSR

Cette optimisation expérimentale poursuit le noyau d'agrégation
`contacts.rs` fusionné dans V888 #150. Elle vise à **éviter de relire
les mêmes lignes de contacts CSR** à chaque activation/désactivation
d'une source dont les destinations répétées se compressent fortement.

## Fonctionnement

Par défaut, **le cache est désactivé**. Le moteur conserve exactement
son comportement antérieur : chaque transition parcourt le CSR et
reconstruit les contributions pondérées par destination.

L'appel Rust explicite

```rust
let rapport = moteur.prepare_contact_cache(1_024)?;
```

construit hors du traitement événementiel une table immuable
`source -> [(destination, somme_u128)]`, avec au maximum 1 024 paires
(source,destination) **retenues**. Les sources non retenues continuent à
utiliser le CSR original. Les critères conservateurs sont :

- au moins 64 lignes CSR sortantes pour une source ;
- au moins quatre fois moins de destinations pondérées distinctes
  après agrégation ;
- budget restant suffisant pour stocker **toutes** les paires d'une
  source ; aucune source n'est partiellement mise en cache.

Les poids nuls, destinations non triées et contacts répétés conservent
la même réduction exacte `u128`. Le cache n'est qu'une représentation
des contributions d'un graphe emprunté **immuable**. Il ne contient
aucun état d'activation neuronal ou compteur d'entrée susceptible
d'être périmé.

`prepare_contact_cache(0)` ou `clear_contact_cache()` désactive le
cache sans modifier les événements en attente. Une reconstruction
remplace l'ancien cache uniquement si toute la préparation réussit.

## Trois exécuteurs

Les exécuteurs `run`, `run_parallel_frontiers` et
`run_disjoint_parallel_commits` partagent cette table immuable.
Le séquentiel utilise un emprunt de la tranche de paires cachées :
aucune copie des lignes CSR n'est effectuée sur ce chemin. La
préparation parallèle ne copie que les petites paires agrégées
nécessaires au commit. Les sources non compressibles restent sur
le noyau exact CSR de référence.

L'ordre causal, les seuils, les effets synaptiques, les budgets et
les garanties d'atomicité en cas d'erreur restent identiques.

En particulier, `EventWork.contacts_examined` continue à compter
**les lignes CSR logiques que la transition représente**, même si le
cache évite leur relecture physique. Ce compteur ne peut donc plus
être interprété comme un comptage des lectures réellement effectuées
lorsque le cache est activé.

## Coûts et limites

Le nombre de paires **retenues** est plafonné par le paramètre de
l'appel. Ce paramètre n'est **pas une limite exacte d'octets alloués** :
les nœuds `BTreeMap`, les capacités `Vec` et les allocations
temporaires du constructeur peuvent consommer davantage. La phase de
construction parcourt les sources éligibles et doit être amortie par
des activations répétées. Sur un grand graphe, son temps et son pic
mémoire peuvent être importants.

Le cache n'a aucune utilité démontrée si le graphe change souvent, si
les destinations se compressent mal ou si une source ne s'active que
rarement. Son activation doit rester volontaire et ses résultats
comparés au témoin uncached avant toute intégration systématique.

## Reproduction et vérifications

```bash
cargo fmt --all -- --check
cargo clippy --offline --all-targets -- -D warnings
cargo test --offline --all-targets
cargo build --offline --release --bin v888-event-bench

for mode in serial frontier disjoint; do
  for cache in off on; do
    ./target/release/v888-event-bench --mode="$mode" \
      --shape=heavy-disjoint --sources=256 \
      --workers=4 --repetitions=8 --cache="$cache"
  done
done
```

Le banc inclut toujours un **oracle séquentiel sans cache** et compare
l'état bit à bit, les entrées `u128`, la file d'événements, les dates et
les huit compteurs logiques. La construction est délibérément exclue
de `median_ns` pour distinguer le débit à chaud ; elle est rapportée
en `cache_build_total_ns`, avec `cache_pairs` et
`cache_covered_rows`. Une comparaison de bout en bout doit ajouter
ce coût et vérifier le nombre de transitions nécessaire à
l'amortissement.

Aucun gain de temps, de RAM ou de joules n'est présumé avant les
mesures sur Dell via `Memorithm/RemoteOps`. Les capteurs RAPL et
les compteurs IMC observent l'hôte partagé et ne permettent pas
d'attribuer directement l'énergie à ce cache ou à un processus.

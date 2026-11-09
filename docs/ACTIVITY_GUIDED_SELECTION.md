# B888-3 — Sélection de sous-circuits par activité observée

Ce composant **fonctionnel Rust** construit, à partir d'une archive de simulation `V888ATL1` (PR #20), un graphe dérivé `V888BOL1` limité aux neurones observés et aux chemins anatomiques de support sélectionnés. Le fichier BANC complet n'est ni modifié ni remplacé.

## Mécanisme

1. Classe les neurones par fréquence d'activation constatée dans les cycles de simulation, puis sélectionne au plus `--anchors` identifiants dont l'activité atteint `--min-ticks`.
2. Optionnellement, élargit le circuit de 0 à 3 sauts sur les **arêtes orientées originales**. À chaque saut, visite les arêtes sortantes des neurones nouvellement retenus, choisit au plus `--fanout` voisins avec les plus grands nombres de contacts, départagés déterministement par identifiant cible et ordre de rangée. Les lignes à count=0 ne sélectionnent pas de nouveaux voisins.
3. Extrait le **sous-graphe induit** de tous les sommets sélectionnés. Chaque arête source dont les deux extrémités ont été sélectionnées est copiée *sans perte au sein de la projection* : multiplicité u64 intacte, doublons distincts, boucles et lignes de count=0. Aucun lien nouveau n'est créé. Le retrait des autres neurones et arêtes rend le graphe dérivé délibérément **perdant**, contrairement à l'archive anatomique V888 source.

Les sélections sont basées sur **l'activité du moteur expérimental**, et non sur un signal neuronal biologique mesuré ; ce résultat ne signifie donc pas que les circuits biologiques importants ont été identifiés.

## Commandes

```bash
cargo run --release --bin v888-activity -- record --graph brain.v888b --seed ROOT_ID --steps 5 --output experience.v888atl

cargo run --release --bin v888-select -- --graph brain.v888b --atlas experience.v888atl --output selected.v888b --anchors 128 --max-neurons 2048 --hops 1 --fanout 8 --min-ticks 1 --max-scanned-edges 20000000 --max-result-edges 1000000 --probe ROOT_ID --threshold 3 --steps 3

cargo run --release --bin v888-booleen -- run --graph selected.v888b --seed ROOT_ID --threshold 3 --steps 3
```

`ROOT_ID` représente un **identifiant réel présent dans le graphe** ; celui de la première simulation doit également appartenir au circuit sélectionné pour utiliser `--probe`. Aucune donnée anatomique BANC n'est livrée avec le code Git.

La commande `--probe` fait évoluer en parallèle le moteur original et le moteur dérivé, sur les mêmes identifiants de stimulation, seuils et nombre de cycles. À chaque cycle, elle publie les activations originales, les activations originales appartenant au circuit, les activations du circuit et leur **intersection exacte par identifiant**. Les divergences sont des mesures comparatives de simulations, pas une preuve de capacité cognitive ou biologique.

## Budgets et échec explicite

- maximum 10 000 neurones dérivés, 10 000 ancres, 64 voisins explorés par neurone et 3 sauts ;
- maximum configurable de 100 M lignes sources visitées au total ; défaut : 20 M ;
- maximum de 2 M arêtes dérivées, défaut : 1 M ;
- une limite de lectures ou d'arêtes dépassée produit une **erreur**, jamais un résultat partiel ;
- le graphe de sortie conserve les **identifiants originaux**, des offsets CSR valides et la stabilité d'ordre des lignes ;
- la CLI refuse les chemins de sortie déjà présents et le même chemin textuel que les entrées ; elle ne doit pas être utilisée avec des chemins alias ou des symlinks pointant vers l'entrée ;
- le coût initial de sélection inclut les lectures sources ; seul un usage ultérieur du sous-graphe peut amortir la construction. **Aucun gain de latence n'est affirmé sans chronométrage physique.**

## Expériences continues attendues

Comparer `anchors`, `max-neurons`, `hops`, `fanout`, seuil d'activation et budgets d'arêtes sur le même graphe V888 à SHA-256 connu, avec des stimuli TRAIN/VALIDATION/TEST/OOD distincts. Tracer ratio d'arêtes, préparation, mémoire, débit de propagation et recouvrement des activations de la simulation de référence. Comparer aussi avec des circuits aléatoires appariés en nombre de neurones et d'arêtes. L'évaluation de récupération associative B888-3 de la PR #18 et les règles d'apprentissage B888-4 de Grok (#19/#21) restent séparées jusqu'à qualification croisée.

Ce développement dépend de la PR #20 car il lit son `ActivityAtlas`. Tant que #20 n'est pas fusionnée, la PR de cette tranche doit être **empilée** sur la branche #20 pour ne pas intégrer ni masquer les modifications en cours de #18 et #19.

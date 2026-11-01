# V888 BOOLEEN

Projet de recherche autonome de l'écosystème Memorithm.

**Initiative scientifique et concept fondateur : porteur du projet Memorithm (utilisateur à l'origine de la proposition).** L'assistance IA contribue à la formalisation, à l'implémentation et aux vérifications, sans revendiquer l'idée.

## Mission

Transformer le connectome **BANC V888** (cerveau et chaîne nerveuse ventrale de drosophile) en une représentation intégralement booléenne, puis étudier un moteur d'exécution et d'apprentissage **original en Rust**. L'encodage fidèle des données anatomiques et le modèle fonctionnel hypothétique sont deux livrables distincts.

## Principes

1. Version scientifique figée sur BANC V888 ; changements de version explicites.
2. Encodage réversible : identifiants, arêtes orientées, multiplicités synaptiques et annotations disponibles conservés.
3. Ne pas présenter la simulation comme une reproduction fidèle du fonctionnement biologique : les règles de dynamique et de plasticité sont des hypothèses.
4. Comparer le connectome biologique à des graphes témoins appariés et à des topologies apprises, à ressources comparables.
5. Développer indépendamment de SML-GENIUS ; intégration possible uniquement sur preuves reproductibles.
6. Exécution native Rust ; aucun composant NVIDIA propriétaire obligatoire ; machines administrées via Memorithm/RemoteOps.
7. Aucun code, poids ou schéma logiciel tiers repris sans droits explicites ; conserver les attributions scientifiques obligatoires des données utilisées.

## État vérifiable — 1 novembre 2026

- **Base Rust et CI Dell : opérationnelles**. L'intégration continue est verte pour le commit `660afd29763639d46ab43ef56fc4c8bcf80341ae` (PR #4).
- **Connectome GCS v888-v2 réellement construit** : 188 508 identifiants neuronaux, 11 752 828 connexions orientées, fichier `brain-v888-gcs-v2.v888b` sur Dell (SHA-256 `f0aa259df38269499ec03ba033f0ae54a71a31187976c4ea9284e08efb4b5135`). Deux cycles d'exécution et une propagation positive sur une connexion réelle ont été vérifiés.
- **Synapses individuelles réellement acquises** : Parquet v888-v2 de 17 146 564 854 octets, SHA-256 `e75d1430089a55ef0e2eb40ba88803d0ddc59175c98af03ba78ef58d26832092`, 155 247 357 lignes et 30 colonnes inspectées. Le fichier original est conservé hors GitHub.
- **Codec Rust individuel `V888SYB1` (PR #4 fusionnée)** : encode sans perte les deux identifiants des synapses et leur nullité, sans filtrage. La matérialisation réelle de ce deuxième binaire est contrôlée séparément par RemoteOps.
- **Trente champs préservés** : une archive compressée de 18 184 013 384 octets a été produite pour 155 247 357 lignes ; l'empreinte du flux décompressé correspond à celle du flux source. Un lecteur Rust distinct a décodé 30 champs pour toutes les lignes. [Construction](https://github.com/Memorithm/RemoteOps/actions/runs/41250446671) · [Décodage](https://github.com/Memorithm/RemoteOps/actions/runs/41248147098). Ce stockage ne signifie pas que tous les champs participent à la dynamique du moteur.
- **Apprentissage expérimental** : sur une cible et des étiquettes artificielles, le banc a obtenu 40/40 sur les sources vues et 20/40 sur des sources inédites, comme les deux témoins figés. Aucune généralisation ni fidélité biologique n'est établie. [Exécution](https://github.com/Memorithm/RemoteOps/actions/runs/41284536812).

Ces observations sont tracées dans [Couverture scientifique et vérifications](docs/COVERAGE.md). Les 30 champs sont préservés dans une archive et décodés par Rust ; leur exploitation complète par le moteur reste à établir. Le décalage entre le nombre de liens scientifiques publié et le nombre de lignes du Parquet GCS n'est pas résolu.

## Qualification GitHub

Les branches de recherche sont qualifiées sur une révision SHA exacte par
`Memorithm/RemoteOps`, dans un conteneur Rust 1.89 sans réseau ni secrets.
Un statut GitHub `remoteops/v888-rust-qualification` n'est publié que si format,
Clippy, tests et CLI réussissent. Le workflow local `.github/workflows/ci.yml`
teste ensuite les commits de `main` sur le pool de compilation Dell isolé
après admission explicite dans son groupe de runners.

## Commandes

- Vérifier le format : cargo fmt --all -- --check
- Vérifier les avertissements : cargo clippy --all-targets -- -D warnings
- Exécuter les tests : cargo test
- Afficher l'état : cargo run -- info

## Documents

- [Plan scientifique](docs/ROADMAP.md)
- [Couverture biologique et preuves](docs/COVERAGE.md)
- [Contrat d'exactitude et de provenance](docs/CONTRACT.md)
- [Sources biologiques et licences](docs/SOURCES.md)
- [Exploitation sous RemoteOps](docs/REMOTEOPS.md)

## Droits

Code réservé à ses titulaires : aucune licence de redistribution n'est accordée par défaut. Les conditions des sources scientifiques s'appliquent séparément.

## Moteur booléen expérimental

Le moteur Rust est exécutable. `cargo run -- demo` utilise des données synthétiques, tandis que les exécutions réelles sont qualifiées via RemoteOps sur le Dell. Consulte les [spécifications du moteur](docs/ENGINE.md), du [flux individuel des synapses](docs/SYNAPSE_PAIRS.md) et des [portes booléennes de neurotransmetteurs](docs/NT_BOOLEAN.md).


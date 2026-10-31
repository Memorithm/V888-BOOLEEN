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

## État vérifiable

**B888-0 : initialisation Rust.** Aucun fichier de données BANC n'est inclus. Le programme contient seulement un état booléen compact et des tests. Aucun simulateur fonctionnel et aucun entraînement n'existent encore.

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
- [Contrat d'exactitude et de provenance](docs/CONTRACT.md)
- [Sources biologiques et licences](docs/SOURCES.md)
- [Exploitation sous RemoteOps](docs/REMOTEOPS.md)

## Droits

Code réservé à ses titulaires : aucune licence de redistribution n'est accordée par défaut. Les conditions des sources scientifiques s'appliquent séparément.

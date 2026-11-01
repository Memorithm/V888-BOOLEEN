# V888-BOOLEEN → SML-GENIUS : protocole de raccordement expérimental

Ce document prépare un raccordement ; il ne l'implémente pas et ne valide aucune capacité de modèle. SML-GENIUS conserve son contrat `sml.v888-bool-handoff.v1` et son autorité sur le modèle.

## Périmètres à ne pas confondre

Le graphe V888-BOOLEEN produit depuis GCS v888-v2 est au format `V888BOL1` : 188 508 neurones et 11 752 828 lignes d'arêtes dirigées ; SHA-256 `f0aa259df38269499ec03ba033f0ae54a71a31187976c4ea9284e08efb4b5135`. Il conserve les lignes pré/post/count sans fusion des doublons. [Construction](https://github.com/Memorithm/RemoteOps/actions/runs/41233545231).

La qualification SciRust [#1521](https://github.com/Memorithm/scirust/pull/1521) concerne un autre contrat, `V8CSR001`, et le périmètre d'artefact décrit par cette PR. L'égalité du nombre de neurones ne prouve pas l'égalité des arêtes ni la compatibilité des formats. La fermeture de [SciRust #1520](https://github.com/Memorithm/scirust/issues/1520) ne qualifie pas automatiquement le fichier `V888BOL1` pour SML.

## Premier incrément à implémenter

1. Figer les commits Rust, fichiers, schémas, tailles et empreintes des deux producteurs, et les licences applicables. Distinguer nombre de lignes, couples dirigés distincts et somme des multiplicités ; mesurer les doublons et les zéros.
2. Implémenter en Rust un export de recherche borné, déterministe et versionné ; vérifier identifiants, bornes, arêtes coupées, valeurs inconnues et corruption. Ne jamais inventer signe, délai ni conductance. Conserver les données source privées hors Git.
3. Vérifier par un lecteur indépendant les identités, les sommes et la parité exacte sur des fixtures et un sous-graphe figé. Toute incompatibilité reste une erreur explicite ; aucune correspondance de périmètre n'est présumée.
4. Sur une tâche de développement préenregistrée, comparer graphe seul, routeur SML seul et hybride, ainsi que des témoins topologiques appariés. Figer ports, graines, partitions, réinitialisation, budget de lecture et métriques avant exécution. Empêcher le contournement par historique caché.
5. Mesurer qualité, bits appris, mémoire, événements, arêtes actives et temps pour chaque bras à budgets comparables. Conserver également les résultats nuls. Utiliser RemoteOps pour l'exécution bornée sur Dell.
6. N'admettre un raccordement au modèle qu'après qualification indépendante des contrôles SML, de la qualité et des ressources ; conserver un modèle exécutable sans données V888 brutes à l'inférence.

## Résultat préalable et limite

Le [banc actuel](https://github.com/Memorithm/RemoteOps/actions/runs/41284536812) utilise une cible et quatre sources avec étiquettes artificielles : 40/40 sur sources vues et 20/40 sur sources inédites pour le modèle entraîné, contre 20/40 pour les témoins. Ce résultat ne démontre ni généralisation ni avantage SML. Les 40 présentations répètent seulement deux sources par partition ; elles ne constituent pas 40 exemples indépendants.

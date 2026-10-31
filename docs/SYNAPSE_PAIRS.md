# Projection booléenne de chaque synapse — V888SYB1

## Source effectivement acquise (ne pas confondre avec l'archive figée)

- BANC V888 GCS v2, `banc_888_synapses_v2_enriched.parquet`.
- Octets : **17 146 564 854**.
- SHA-256 : `e75d1430089a55ef0e2eb40ba88803d0ddc59175c98af03ba78ef58d26832092`.
- Pied Parquet audité sur Dell : **155 247 357 lignes**, **30 colonnes**, **1 553 groupes**.
- Cette version GCS ne doit PAS être présentée comme la table figée des 218 millions de liens mentionnés par l'article ; le décalage de couverture devra être investigué.

## Format expérimental V888SYB1

Une ligne par synapse, y compris contacts répétés. En-tête : 8 octets ASCII `V888SYB1`, version `u64 LE=1`, compte de lignes `u64 LE`. Chaque ligne : masque de validité (`bit0=pre_root_id`, `bit1=post_root_id`), puis identifiants `pre` et `post` sur 64 bits chacun, en little endian. Les identifiants absents doivent être représentés par leurs bits à zéro, sans confondre NULL et l'identifiant valide zéro.

Chaque bit conserve exactement les deux valeurs de la ligne ; aucune connexion n'est fusionnée, filtrée ou inférée. Le codec Rust vérifie en streaming l'exactitude du nombre de lignes, les masques, la longueur du fichier et les versions. Il ne conserve aucune rangée entière en mémoire. Le format exige 17 octets par synapse et 24 octets d'en-tête, soit environ 2,64 Go pour la source actuellement acquise.

**Limite explicite :** il ne conserve que deux colonnes sous forme de projection d'exécution. Les 28 autres colonnes, dont ID individuel, coordonnées et huit probabilités de neurotransmetteurs, restent conservées exactement dans le Parquet scientifique original, avec l'empreinte cryptographique. Une conversion booléenne complète réversible des 30 colonnes reste à mettre en œuvre, puis à valider par retour inverse strict. Les probabilités `double` ne doivent jamais être silencieusement quantifiées ; les bits IEEE-754 et les états NULL devront être préservés dans une version ultérieure.

## Exécution

Le module `synapse_pairs` réalise le décodage exact et une propagation récurrente booléenne expérimentale sur les contacts individuels, avec le `Connectome` comme univers des neurones identifiés. Les synapses dont le neurone n'est pas identifié restent dans le fichier ; elles ne sont pas inventées dans le calcul.

`pack` accepte un flux brut de tuples déjà contrôlés par l'adaptateur de données. Aucun Parquet externe ne doit être traité comme fiable sans SHA-256 et qualification de provenance.

**Ne pas annoncer un cerveau biologique fidèlement émulé** : les règles et seuils de propagation sont des choix d'ingénierie, sans validation physiologique.

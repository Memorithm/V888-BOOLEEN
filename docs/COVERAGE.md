# Couverture BANC V888 — éléments matériellement vérifiés

Projet et concept : initiative du porteur Memorithm. Les données sources anatomiques restent attribuées à leur consortium scientifique conformément à leurs licences.

## Matérialisation source GCS v888-v2 enregistrée sur Dell

| Objet | Fait vérifié | Preuve |
|---|---|---|
| Métadonnées neuronales | `banc_888_meta.feather`, 57 503 026 octets, SHA-256 `empreinte consignée dans le manifeste privé sur Dell (valeur à relever)` | [RemoteOps 41232008901](https://github.com/Memorithm/RemoteOps/actions/runs/41232008901) |
| Connectivité dirigée | `banc_888_edgelist_simple_v2.feather`, 305 250 378 octets, SHA-256 `empreinte consignée dans le manifeste privé sur Dell (valeur à relever)` | Même run |
| Conversion CSV | 188 508 identifiants, 11 752 828 lignes d'arêtes, SHA-256 CSV `8313b0271f96c8dd8000195539064c1e21ee48cbefe253c3d943b2af52d489f7` | [RemoteOps 41233291736](https://github.com/Memorithm/RemoteOps/actions/runs/41233291736) |
| Graphe booléen Rust | `brain-v888-gcs-v2.v888b`, SHA-256 `f0aa259df38269499ec03ba033f0ae54a71a31187976c4ea9284e08efb4b5135` | [RemoteOps 41233545231](https://github.com/Memorithm/RemoteOps/actions/runs/41233545231) |
| Propagation de contacts agrégés | Un stimulus sur une connexion réelle donne 1 activation au cycle 1 | [RemoteOps 41233769407](https://github.com/Memorithm/RemoteOps/actions/runs/41233769407) |
| Synapses individuelles acquises | `banc_888_synapses_v2_enriched.parquet` : 17 146 564 854 octets, SHA-256 `e75d1430089a55ef0e2eb40ba88803d0ddc59175c98af03ba78ef58d26832092` | [RemoteOps 41234081346](https://github.com/Memorithm/RemoteOps/actions/runs/41234081346) |
| Pied Parquet réel | 155 247 357 lignes, 1 553 row groups, 30 colonnes ; `pre_root_id`, `post_root_id` en `large_string` nullable, probabilités de 8 NT en `double` nullable | [RemoteOps 41235026390](https://github.com/Memorithm/RemoteOps/actions/runs/41235026390) |

Ces fichiers ne sont **pas** déposés dans GitHub. Les manifestes de provenance et les données brutes sont conservés dans le stockage privé du Dell, sous contrôle RemoteOps.

## Limite scientifique majeure

L'article Bates, Phelps, Kim, Yang et collaborateurs, *Nature*, 2026, DOI 10.1038/s41586-026-10735-w, décrit **218 460 852 liens synaptiques** pour la table de détection v2. L'objet enrichi actuel téléchargé depuis GCS comprend **155 247 357 lignes**. Il n'est donc pas démontré que cet objet enrichi recouvre tous les liens du jeu scientifique d'origine ; les sélections, exclusions ou changements de version ne sont pas encore entièrement reconciliés. Aucun lien manquant ne doit être inventé.

Le graphe CSR représente les **connexions agrégées** ; il ne conserve pas l'identité de chaque synapse. Le codec `V888SYB1` représente chaque synapse du Parquet lu, mais **uniquement ses deux racines et leurs bits de validité**. L'archive Parquet conserve les 28 autres colonnes, notamment identifiant, coordonnées, région, volume, étiquettes et prédictions de neurotransmetteurs. Leur conversion sans perte vers des circuits individuels n'est pas encore achevée.

Le module `nt_gates` permet une comparaison booléenne des bits des probabilités sans FPU **une fois ces bits fournis** ; il ne confirme aucune nature excitatrice/inhibitrice biologique.

## Validation manquante avant l'expression « cerveau intégral »

1. Réconcilier les 155,2 M lignes enrichies et les 218,5 M liens mentionnés dans l'article, avec documentation de filtrage exacte.
2. Encoder et décoder **tous les 30 champs** des synapses avec bitmaps NULL, bits IEEE-754 intacts et chaîne UTF-8 intacte ; preuve de round-trip.
3. Contrôler la topologie, les synapses dupliquées, les cellules non identifiées, les annotations et les valeurs non renseignées sur l'ensemble des sources.
4. Qualifier des règles dynamiques et d'apprentissage, puis tester sur des tâches réellement mesurables et des graphes témoins comparables.
5. Séparer les preuves de calcul exact de toute hypothèse de simulation biologique réaliste.

Sources : Bates et al., *Nature* (2026), https://doi.org/10.1038/s41586-026-10735-w ; BANC v888, https://doi.org/10.7910/DVN/7WTH1N. Les mesures chiffrées locales proviennent uniquement des actions ci-dessus.

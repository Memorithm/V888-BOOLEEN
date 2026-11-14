# B888-5 — Cache exact de sous-circuits pour les interventions contrefactuelles

Statut de ce protocole : **préenregistré avant qualification physique**. Ce
lot optimise les **calculs de l'apprentissage existant**, sans modifier
l'objectif, les décisions, la plasticité ni les connexions effectivement
sélectionnées. La PR #92 porte indépendamment sur la sélection VALID
équilibrée ; ce travail ne la duplique pas.

## Mécanisme fonctionnel

L'ancienne procédure `fit_final_loss` reconsidérait les **156 portes
Fourier** d'un arbre à 625 entrées à chaque proposition de modification
d'un seul paramètre ou d'une seule paire de sources.

`FiveCircleIncrementalCache` construit, pour un paquet de 64 entrées,
les sorties booléennes bit-sliced **à tous les étages** du modèle de
référence. Pour chaque contrefactuel issu de la version courante du
modèle, le moteur recalcule uniquement :

- **Paramètre du neurone :** ce neurone et chacun de ses ancêtres
  jusqu'à la racine. Une modification de paramètre au premier étage
  nécessite au maximum `depth` portes, au lieu de 6/31/156 portes
  recalculées.
- **Échange de deux positions de sources :** les une ou deux portes
  du premier étage recevant les sources échangées, puis tous leurs
  ancêtres uniques jusqu'à la racine. Les sous-arbres sans changement
  sont réutilisés sans inférence.
- La table de vérité 32 bits d'un neurone changé est compilée exactement
  depuis ses cinq caractères Walsh ; aucun masque, rayon ou biais
  n'est approximé.
- Le cache est **reconstruit à chaque round**, après l'acceptation
  éventuelle de la meilleure modification du round précédent. Il ne
  conserve jamais des activations d'une topologie ou de paramètres
  devenus obsolètes.

L'API `PackedIntervention` représente explicitement les deux types de
changement. Toutes les évaluations sont exactes sur les **64 voies**.
Les chemins convergents ne sont calculés qu'une fois grâce à un
ensemble trié d'ancêtres.

La fonction de comparaison
`fit_final_loss_with_evaluator(..., CreditEvaluationMode::Full)`
reproduit l'ancienne réévaluation complète. Le mode
`Incremental` sert au moteur de production `fit_final_loss`.
Les deux utilisent **la même suite de propositions, les mêmes
étiquettes TRAIN, le même budget et le même critère de sélection**,
afin que les paramètres appris finaux soient strictement identiques.

## Métriques

Pour chaque paquet de 64 exemples TRAIN :

- `full_recompute_gate_evaluations` : nombre de portes compilées
  évaluées dans la référence complète, score initial compris ;
- `cache_build_gate_evaluations` : coût complet de création des
  références (score initial et un cache par round) ;
- `delta_gate_evaluations` : portes recalculées uniquement sur les
  cônes de dépendance modifiés ;
- `actual_gate_evaluations` : somme des coûts réellement utilisés
  dans chacun des deux modes.

Ces mesures comptent des **appels de porte bit-sliced à 64 voies**,
pas des lectures mémoire, ni des joules, ni des instructions CPU.
Elles ne comprennent pas la génération des données, les copies de
modèles, la préparation initiale des plans de bits ou les opérations
d'orchestration. Les **durées murales** mesurées dans le même banc
incluent en revanche l'intégralité de la fonction d'entraînement,
modèles et préparation du cache compris.

Le champ historique `full_model_train_inferences` compte les
**prédictions logiques** faites sur TRAIN ; avec le cache il ne
doit plus être interprété comme des inférences physiques
réexécutant le réseau complet.

## Banc physique préenregistré

Trois graines : `0xcac80101`, `0xcac80202`,
`0xcac80303`. Trois architectures :

| Entrées | Neurones | Contacts orientés |
|---:|---:|---:|
| 25 | 6 | 30 |
| 125 | 31 | 155 |
| 625 | 156 | 780 |

À chaque graine et profondeur, comparer **deux portées** :
plasticité cachée seule et recherche conjointe des connexions +
plasticité cachée, chacune dans les modes **Full** et
**Incremental**. Au total 18 profils appariés. Le nombre
de propositions est `4 rounds × 64 = 256` et chaque bras
consulte `256 TRAIN`, soit **65 792 décisions TRAIN** en comptant
la référence initiale, à capacité de modèle et labels identiques.

Enseignant synthétique : majorité sur la première couche, parité
aux couches suivantes, permutation de toutes les sources à l'aide
d'une graine indépendante de l'entraînement. Étudiant : topologie
contiguë avec **un neurone sur cinq** au premier étage
substitué par une porte ET (masques singleton inchangés).

TRAIN 256, TEST 512, SHIFT 256 (densité de bits 0,25 au lieu
de 0,5). Les trois ensembles sont dédoublonnés par état bit-packed
complet ; les labels sont établis par un oracle booléen
indépendant des tables Fourier. Aucun résultat TEST ou SHIFT
n'intervient dans la sélection du modèle.

Conditions scientifiques d'admission :

1. Pour chaque proposition, les tests unitaires comparent
   indépendamment le résultat complet des 64 voies avec le
   résultat du cache sur plusieurs tailles, paramètres, couches,
   graines et échanges de sources, y compris les cas où les
   chemins se rejoignent ou produisent la même valeur.
2. Pour chaque profil physique, le moteur complet et le
   moteur cache doivent produire **les mêmes scores TRAIN**,
   mêmes modèles, mêmes routes, mêmes nombres de modifications
   acceptées, mêmes scores TEST/SHIFT. Aucun gain de précision
   n'est attribué à la mise en cache.
3. La quantité de portes évaluées avec cache doit être
   inférieure à la référence complète. Les temps mesurés
   peuvent être négatifs ; publier même les ralentissements.
4. Préserver exactement le nombre de neurones, de connexions,
   les cinq masques Walsh et le caractère opt-in de la
   plasticité. Cette optimisation n'est pas une mise à jour
   des arêtes biologiques BANC ni un retour de gradients.

## Qualification et suite

Rust 1.89, `cargo fmt --all -- --check`,
`cargo clippy --offline --all-targets -- -D warnings`,
`cargo test --offline --all-targets`, tous les anciens
bancs Fourier, `cargo run --offline --bin fourier5-delta-bench`
et `cargo run --offline -- info`, à HEAD exact sous
`Memorithm/RemoteOps`. Exécution Docker non privilégiée,
sans réseau, limitée à 4 CPU/6 Gio ; pas de modification de
secrets/services/productivité Dell. CI native verte attendue
sur le commit de fusion exact.

**Prochaine étude :** validation équilibrée de la généralisation
(PR #92), réutilisation des caches en propagation récurrente
et représentation de la sensibilité booléenne locale, avec
données et provenance BANC V888 traitées séparément.

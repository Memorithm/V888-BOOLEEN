# B888-4 — Apprentissage continu à partir des seules étiquettes de sortie

## Expérience préenregistrée

Objectif : vérifier qu'un réseau Fourier à **6 250 000 entrées** et
**1 562 501 neurones indépendamment paramétrables** peut accumuler
des modifications acceptées à partir de la seule erreur de sortie,
sans connaître les activations cachées de l'enseignant synthétique.

Le banc `fourier5-reward-retention-bench` utilise trois graines
indépendantes : `0x9a770001`, `0x9a770002`, `0x9a770003`.
Chaque graine définit un enseignant synthétique et quatre flux
d'entrée distinctement initialisés, chacun comportant **64 voies** :
TRAIN-A, TRAIN-B, VALID et TEST. Les sorties de l'enseignant sont
utilisées comme étiquettes. **Aucune activation intermédiaire de
l'enseignant ne participe à l'entraînement.** VALID/TEST ne
participent ni aux propositions ni au choix de leurs acceptations.

## Protocole séquentiel

- Entraîner TRAIN-A pendant 3 passages de 4 096 propositions chacun.
- Mesurer après chaque passage TRAIN-A, TRAIN-B, VALID et TEST.
- Construire un comparateur avec **exactement** les mêmes paramètres
  neuronaux acquis après TRAIN-A.
- Rebaser les caches des deux modèles sur TRAIN-B : publier le coût
  mesuré de reconstruction (ce n'est **pas** une mise à jour éparse).
- Entraîner les deux modèles sur TRAIN-B pendant 3 passages supplémentaires,
  avec la même suite de propositions par passage.
- Le comparateur accepte chaque amélioration strictement positive de
  l'exactitude TRAIN-B ; la variante protégée utilise **un second cache
  incrémental** sur TRAIN-A pour tester l'effet contrefactuel avant
  acceptation. Elle rejette toute proposition qui dégraderait le niveau
  acquis sur TRAIN-A, sans rescanner les 1,56 million de neurones.
  Les deux caches n'intègrent que les mêmes modifications acceptées. Ce garde-fou est un **rejeu de TRAIN** : il est attendu qu'il
  préserve cet objectif. Il ne garantit **aucune** conservation sur VALID/TEST.
- Émettre les mesures individuelles par graine, passage et modèle, y compris
  propositions acceptées, propositions rejetées par le garde-fou,
  évaluations de portes des deux caches, lectures logiques, durées et
  performances de généralisation.

### Attribution locale du mérite

Une intervention propose les paramètres Fourier d'un neurone identifié
par `(layer, neuron)`. Le changement de sortie sur les 64 exemples
TRAIN, calculé par propagation incrémentale exacte, donne le
**crédit contrefactuel observable**. L'acceptation nécessite une
augmentation stricte de l'exactitude sur TRAIN. Aucune récompense
différée ni mise à jour des connexions n'est revendiquée.

Il s'agit d'un protocole **output-only, synthétique et supervisé**,
pas d'un modèle biologique ni d'un apprentissage autonome de langage.
On ne doit pas confondre la garantie de non-dégradation TRAIN-A
induite par le garde-fou avec un résultat expérimental indépendant.

## Résultats acceptables et limites

La preuve d'amélioration doit reposer sur les séries de mesures :
exactitudes TRAIN-A/B et VALID/TEST à chaque passage, dispersion
entre trois graines, améliorations et régressions non omises.
Un score VALID ou TEST inchangé/dégradé est conservé dans le rapport.

Le présent banc ne mesure **ni** énergie RAPL, **ni** transactions DRAM,
**ni** consommation du système d'entraînement intégral avec/sans cache.
Ces mesures relèvent des workflows hôte dédiés de
`Memorithm/RemoteOps` ; des lectures logiques ne sont pas un trafic
DRAM physique. Le comparateur `no_replay` conserve **le même moteur
incrémental** : la différence vise uniquement la rétention, pas le
coût du recalcul complet.

## Qualification

`cargo run --offline --release --bin fourier5-reward-retention-bench`.
Qualification à SHA exact dans le conteneur isolé RemoteOps, Rust 1.89,
`cargo fmt`, Clippy strict et tests ; ne pas fusionner avant preuve.
Ne pas utiliser les runners privilégiés pour du code PR non qualifié.

Le protocole initial sur un seul cache a été exécuté et a mis en évidence
un coût très élevé du garde-fou lorsque celui-ci rescannait intégralement
TRAIN-A à chaque proposition gagnante. La révision à **deux caches** vise
précisément ce goulet d'étranglement ; les mesures initiales ne doivent
pas être présentées comme celles de la nouvelle version avant qualification.

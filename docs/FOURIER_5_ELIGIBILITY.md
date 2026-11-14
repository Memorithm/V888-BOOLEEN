# B888-4 — Plasticité conditionnée par trace d'éligibilité retardée (expérience opt-in)

Statut : **prototype expérimental, données et câblage synthétiques**.
Ne modifie pas le moteur de production ni les règles de propagation
par défaut. Protocole préenregistré avant toute qualification physique.

## Pourquoi maintenant, et ce qui reste hors périmètre

La plasticité n'est pas imposée au réseau V888 : ce module ajoute
un composant optionnel `DelayedEligibility` pour tester une seule
hypothèse précise : un signal local capté avant une période de
propagation peut-il modifier **un gain contextuel existant** après
réception tardive d'un résultat supervisé, puis permettre un rappel
ultérieur sans accès à ce résultat ?

Le mécanisme ne modifie ni le graphe, ni les masques Fourier,
ni les quatre identités des portes contextuelles. Il ne prouve
ni apprentissage biologique, ni mémoire de travail spontanée, ni
plasticité non supervisée, ni rétention d'un état actif pendant le
délai. Le rappel est **associatif dans les poids**, avec une
réexposition de la clé (cue) à la requête.

## Règle exacte

Chaque neurone possède cinq cercles : le premier (masque Walsh 31,
rayon 2, biais -1) calcule la parité des cinq bits d'entrée.
Les quatre autres, de masque Walsh nul et rayon initial nul,
portent les inhibitions conditionnées par les quatre premiers
bits contextuels. Le cinquième bit contextuel est un distracteur.

Un épisode :
1. Au temps `t0`, écrire un mot de données de parité impaire
   et **un seul** des quatre bits contextuels sur les **dix
   contacts déjà existants** du neurone de lecture, puis
   enregistrer ses bits présynaptiques dans une trace d'éligibilité.
   Aucun label ni objectif n'est communiqué à cette étape.
2. Faire avancer la récurrence pendant `D` ticks (aucun rappel
   du cue, aucune réinjection du label et aucune mise à jour
   des poids pendant l'attente).
3. À `t0+D`, révéler une cible booléenne, comparer à la
   prédiction faite sur la trace initiale, et modifier d'une
   unité **un seul** gain contextuel, saturé dans
   `[0, 65535]`. Si la cible demande inhibition, augmenter
   le gain ; sinon le diminuer. Sans erreur : aucune mise à jour.
   La trace est consommée.
4. Après plusieurs épisodes, laisser avancer le réseau pendant
   512 ticks supplémentaires **sans enseignant** et sans
   nouvelle trace, puis réinjecter le cue comme clé de
   récupération. Mesurer la sortie d'un véritable pas
   récurrent sur le neurone de lecture.

L'éligibilité expire après `D_max` ticks, avec
`1 <= D_max <= 512`. Une trace maximum à la fois et
4096 retours supervisés maximum par instance. Feedback
prématuré, expiré, entrée non conforme, dépassement de budget
et indice inexistant sont rejetés sans mise à jour des gains.

## Plan expérimental apparié

Topologie synthétique `LocalRing`, 128 neurones,
10 contacts orientés par neurone, même graine et état initial
pour les comparateurs ; neurone de lecture #19.
Les quatre clés sont `context = 1 << cue` pour
`cue = 0,1,2,3`. Cible équilibrée par graine
(`(cue + (seed & 1)) % 2 == 1`).
Quatre présentations par clé, soit 16 épisodes par bras.

- **Plastique causal** : récompense correspondant à la clé
  enregistrée, reçue seulement après D ticks.
- **Gelé à capacité égale** : modèle initial identique, aucune
  mise à jour. Le modèle gelé est le contrôle sans plasticité.
- **Récompense inversée** : même règle, même nombre de
  retours et même capacité, mais étiquettes inversées. Contrôle
  de dépendance causale à la supervision.

Grille : graines `11,42,93` ; délais `8,64,128` ;
512 ticks de rétention après apprentissage. Le test
de rappel emploie un **autre** mot d'entrée de même parité
(`0b00111` au lieu de `0b00001`) et ajoute le cinquième
bit contextuel distracteur. Les quatre identités de cue sont
les mêmes en TRAIN et TEST : ce n'est donc **pas** un
test de généralisation à des clés inédites.

Critère d'acceptation programmé : sur chacun des neuf profils,
4/4 réponses correctes après rétention dans le bras causal,
2/4 pour le bras gelé et 0/4 pour le bras à récompense inversée ;
empreinte de câblage inchangée, gains stables durant
les 512 ticks, aucun retour avant le délai, tous les
16 retours réellement délivrés. Les durées et compteurs
de mise à jour sont publiés sans en déduire un gain industriel.

## Limites et risques

La trace conserve la clé et les cinq bits de données pendant
le délai : le système possède donc explicitement une mémoire
temporaire **de contrôle**, non une activité neuronale persistante.
L'association à quatre clés peut être assimilée à une petite
table de correspondance paramétrique. Les masques et
l'apprentissage par erreur sont choisis pour cette tâche
et ne démontrent ni autonomie ni adaptation de grande échelle.
Le retour supervisé est indispensable ; aucun renforcement
sans enseignant n'est revendiqué.

Le banc ne mesure pas la consommation énergétique, le
transfert OOD strict, la mémorisation à l'échelle des
188 508 neurones, ni une propriété du connectome BANC.
Un échec est publié sans diminuer les critères a posteriori.

## Validation

Rust 1.89, CPU uniquement, aucun SDK NVIDIA. Validation
par `Memorithm/RemoteOps` sur le SHA exact dans un conteneur
non privilégié sans réseau, plafonné à quatre CPU, 6 Gio
et 25 minutes. `cargo fmt --all -- --check`,
`cargo clippy --offline --all-targets -- -D warnings`,
`cargo test --offline --all-targets`, tous les
bancs Fourier existants, puis
`cargo run --offline --bin fourier5-eligibility-bench`
et le smoke CLI. Fusion seulement après preuve exacte
et CI native verte sur le merge commit.

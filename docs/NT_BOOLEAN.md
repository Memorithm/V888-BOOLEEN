# Portes booléennes pour les probabilités synaptiques

Le Parquet BANC v888 enrichi réellement inspecté possède huit colonnes de probabilités de neurotransmetteurs `double` potentiellement nulles : `acetylcholine`, `dopamine`, `gaba`, `glutamate`, `histamine`, `octopamine`, `serotonin`, `tyramine`.

Le module Rust `nt_gates` compare leurs **motifs binaires IEEE-754** sans instruction de calcul flottant. Dans l'intervalle valide [0,1] pour des nombres non négatifs, l'ordre des motifs de bits positifs est le même que l'ordre numérique. Nous utilisons donc les portes booléennes de comparaison existantes (`gates::ge`). Le signe de zéro est normalisé uniquement pendant la comparaison ; les bits originaux ne sont jamais modifiés. Les valeurs manquantes restent `None`, et NaN, infini, nombres négatifs et valeurs >1 déclenchent un refus explicite.

Une opération `argmax_eight` déterministe désigne la classe de probabilité maximale, sans considérer ce résultat comme un effet biologique excitateur/inhibiteur prouvé.

**Limites :** le vrai fichier source scientifique contient encore les probabilités dans son Parquet original SHA-pinné. Cette PR n'importe ni n'encode encore toutes les colonnes synaptiques dans le format individuel `V888SYB1`. Elle ajoute la primitive CPU-only de décision booléenne qui servira à un futur format couvrant les 30 champs sans perte. Les 8 prédictions sont des estimations et ne valent pas confirmation biologique.

Source biologique et méthodologie : Bates, Phelps, Kim, Yang et al., 2026, Nature, DOI 10.1038/s41586-026-10735-w.

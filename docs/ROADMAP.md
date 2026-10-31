# Feuille de route — V888 BOOLEEN

## B888-0 : base autonome et provenance
- Dépôt privé Memorithm, Rust minimal, tests et CI.
- Vérifier les licences, identifier les fichiers BANC V888 et préparer le manifeste des sources.

## B888-1 : données originales
- Geler la version, les identifiants, les tables de neurones, les synapses, les relations et les annotations.
- Télécharger hors Git ; vérifier SHA-256, taille, provenance, date, licence et format.
- Préserver les exclusions et incertitudes explicitement.

## B888-2 : conversion intégralement booléenne sans perte
- Convertir puis décoder toutes les colonnes retenues et les arêtes orientées.
- Détecter collisions d'identifiants, incohérences, duplications non prévues et données corrompues.
- Mesurer intégrité du round-trip, RAM, espace disque, CPU, débit et limites.

## B888-3 : moteur fonctionnel booléen
- États neuronaux bit-packed et parcours de graphe borné.
- Règles de transition déterministes clairement identifiées comme expérimentales.
- Oracles indépendants sur sous-graphes ; essais sur connectome complet seulement après validation.

## B888-4 : apprentissage
- Objectifs préenregistrés et séparations TRAIN/VALIDATION/TEST/OOD.
- Comparer trois familles : V888, topologie témoin appariée, topologie apprise.
- Mesurer erreurs, variance entre graines, coût d'entraînement et généralisation.

## B888-5 : qualification et intégration optionnelle
- Mesures CPU Dell/Thor sous plan de contrôle RemoteOps et budgets bornés.
- Tests comparatifs de capacité, qualité, RAM, latence, trafic, énergie si mesurable.
- Exposer uniquement les mécanismes validés vers SML-GENIUS, SciRust, NNIS ou autres dépôts, par contrats indépendants et PR distinctes.

## Décision scientifique

Une qualité comparable à coût compétitif ou une capacité nouvelle vérifiée peut motiver la poursuite. Une régression doit être publiée, et aucun résultat d'un autre dépôt ne vaut automatiquement validation de V888 BOOLEEN.

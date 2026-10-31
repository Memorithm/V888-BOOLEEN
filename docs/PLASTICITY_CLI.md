# Utilisation opérationnelle de la plasticité supervisée 4 bits

Le mécanisme `PlasticCircuit` est disponible en Rust : apprentissage par enseignant binaire, sur les connexions anatomiques existantes uniquement, avec coefficients saturants 0..15 et sauvegarde `V888PLS1` (quatre bits par arête). Il est entièrement distinct de la conservation des 30 colonnes du connectome.

## Épisodes d'enseignement

Fichier CSV obligatoire `source_id,target_id,fire` suivi de lignes contenant deux identifiants racines réels en décimal et `fire` égal à `0` ou `1`. Pas de valeurs manquantes, pas de fractions, pas de colonnes supplémentaires. **L'étiquette est apportée par l'expérimentateur** : elle n'est pas fournie par le connectome BANC et ne doit jamais être présentée comme un enregistrement comportemental d'un insecte.

Exemple de commandes (les fichiers locaux doivent être fournis au banc RemoteOps, pas déposés dans GitHub) :

```bash
v888-booleen train-plasticity --graph brain.v888b --episodes train.csv --epochs 4 --output learned.v888pls
v888-booleen score-plasticity --graph brain.v888b --weights learned.v888pls --episodes test.csv --threshold 1
v888-booleen run-plasticity --graph brain.v888b --weights learned.v888pls --seed 720575940000000001 --threshold 1
v888-booleen bench-learning --graph brain.v888b
```

La commande d'entraînement s'arrête après au plus 1 000 000 épisodes (max. 65 536 lignes d'entrée, 64 epochs) et refuse d'écraser un ancien checkpoint. Les poids sauvegardés sont liés à l'empreinte déterministe du graphe ; un graphe incompatible est refusé. Aucun apprentissage n'est exécuté par défaut pendant les commandes de prédiction ou de scoring.

L'évaluation doit être faite sur un fichier de test séparé, idéalement sur **des motifs jamais vus** et avec provenance des étiquettes. Les mêmes motifs rejoués dans le banc initial ne mesurent que la rétention. Un score de rappel parfait sur deux associations **ne démontre ni intelligence ni capacité générale**. Pour une homologation scientifique de l'apprentissage, une batterie indépendante devra ensuite utiliser échantillons non vus, graphes permutés, plusieurs graines, ablations, et un contrôle de coût/qualité à budget comparable.

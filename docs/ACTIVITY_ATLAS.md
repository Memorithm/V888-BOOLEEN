# B888-3 — Atlas d'activité neuronale persistant

## Fonctionnalité

`ActivityAtlas` enregistre une **suite de véritables états du moteur booléen** (état initial inclus), au niveau de chaque identifiant neuronal du graphe V888 chargé. Ce n'est ni un fichier de mesure biologique, ni une synthèse fondée sur des étiquettes fictives.

Informations restituables par neurone :
- nombre de cycles actifs ;
- nombre d'entrées/sorties d'activation ;
- durée maximale d'activité consécutive ;
- premier et dernier cycle actif.

Une chronologie complète contient pour chaque cycle : nombre de neurones actifs, nouveaux activés et désactivés. La commande `inspect` classe les neurones observés (tri reproductible par fréquence, apparition, identifiant source) et interroge un identifiant précis. Cette observation est distincte des hypothèses biologiques et pourra être enrichie d'annotations régionales attestées, sans jamais les inventer.

## Utilisation sur le graphe présent dans le Dell via RemoteOps

```bash
cargo run --release --bin v888-activity -- record --graph data/processed/brain.v888b --seed ROOT_ID --threshold 3 --steps 5 --output activity.v888atl
cargo run --release --bin v888-activity -- inspect --graph data/processed/brain.v888b --atlas activity.v888atl --top 20
cargo run --release --bin v888-activity -- inspect --graph data/processed/brain.v888b --atlas activity.v888atl --neuron ROOT_ID
```

Remplacer ROOT_ID par un identifiant réellement présent dans le graphe ; aucune donnée BANC originale n'est incluse dans GitHub. La commande crée un **nouveau** fichier et refuse de remplacer une archive existante.

## Provenance et sécurité de la restitution

- Format binaire dédié `V888ATL1` (little-endian, versionné par magic).
- Empreinte déterministe couvrant les IDs, le CSR, les cibles et les comptes de contacts ; contrôle lors de la lecture, du classement et de l'écriture. Cette empreinte FNV-1a **n'est pas cryptographique** ; la provenance du graphe exige séparément le SHA-256 déjà enregistré.
- Refus des archives tronquées, des octets surnuméraires, des compteurs manifestement invalides et des fichiers liés à un autre graphe.
- Plafond de 1 000 000 observations par archive ; invocation CLI plafonnée à 1 001 états (état initial et 1 000 étapes).
- Au cours d'une même session, `observe` s'appuie sur la constance du graphe contrôlée par le programme appelant et vérifie les dimensions sans rescanner ses millions d'arêtes à chaque cycle.
- Coût mémoire `O(neurones + cycles)` et travail d'observation proportionnel aux neurones précédemment ou actuellement actifs. Un chargement sauvegarde uniquement les compteurs et bitsets, pas les sources anatomiques.

## Cycle de développement expérimental

Utiliser cet atlas pour comparer les politiques de routage B888-3 (PR #18), puis les stratégies d'apprentissage B888-4 (PR #19) à budget et stimuli identiques : taux d'activation, persistance, répétition, silences et changements après entraînement. Les différences éventuelles doivent être **mesurées** sur Dell via RemoteOps et ne prouvent pas à elles seules la mémoire, la généralisation ni la fidélité biologique.

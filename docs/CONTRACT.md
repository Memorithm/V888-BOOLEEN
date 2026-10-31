# Contrat scientifique — V888 BOOLEEN

## Deux objectifs distincts

**Encodage booléen intégral** : représenter sans perte sous forme de bits les informations anatomiques connues. Un encodage réversible n'est pas en soi une simulation du cerveau.

**Réseau booléen exécutable** : définir des états et des règles de transition qui utilisent cette connectivité. Seule la connectivité est issue des données biologiques ; les seuils, délais, poids effectifs et mécanismes d'apprentissage restent hypothétiques sans preuve spécifique.

## Invariants

- Préserver les identifiants et arêtes orientées ; distinguer synapses individuelles et agrégats de connexions.
- Ne jamais fabriquer une mesure manquante, un neurotransmetteur confirmé ou une dynamique biologique non observée.
- Documenter chaque transformation, version, unité, règle de filtrage, source et empreinte cryptographique.
- Séparer format réversible et formes quantifiées potentiellement destructives.
- Toute simulation doit définir son oracle, sa politique de débordement, ses limites et sa reproductibilité.
- Aucun résultat en apprentissage, vitesse ou fidélité biologique sans mesure publiée.
- Ne pas importer par défaut les fichiers BANC dans Git ; ne pas inclure de données ou poids tiers dépourvus d'autorisation.
- RemoteOps administre les ressources ; SML-GENIUS conserve son propre programme de recherche.

## Périmètre du premier commit

Le module Rust implémente uniquement un vecteur d'activations à un bit par position ; il ne lit pas encore BANC V888 et ne déduit aucune propriété biologique.

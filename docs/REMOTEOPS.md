# Exécution — RemoteOps

**Memorithm/RemoteOps** est le plan de contrôle distant de référence pour V888 BOOLEEN.

- Aucun lancement de tâche longue, modification système ou réservation de ressources hors politiques RemoteOps.
- Dell T430 et Thor : vérifier l'admission de la tâche, la disponibilité et les budgets avant exécution.
- Tracer SHA du code, entrées, graines, temps, mémoire et compteurs CPU ; distinguer résultat de simulation, qualité du modèle et performances.
- Priorité à Rust CPU natif, sans dépendance obligatoire à un SDK NVIDIA.
- Toute optimisation transversale doit être qualifiée dans le dépôt de destination, sur ses propres tests et sous PR distincte.
- Fusion seulement après CI requise verte au SHA exact ; pas de revendication sans mesures.

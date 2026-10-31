# B888-4 — première plasticité booléenne, protocole et limites

**Mécanisme :** chaque arête dirigée du connectome réel garde un coefficient plastique sur quatre bits (0..15), initialement zéro. Aucune arête non observée ne peut être créée. Si un neurone source est actif, un état enseignant explicitement fourni renforce (ou diminue) les coefficients des arêtes sortantes selon l'état cible de chaque destination. L'incrément/décrément utilise des portes XOR/AND, sans calcul flottant ; l'inférence applique les multiplicités de contacts du graphe aux coefficients 4 bits, en additionnant par portes booléennes, avec seuil entier. L'intégration à un cerveau complet exige encore un protocole d'entraînement pertinent et des cibles biologiques, absentes ici.

**Deux témoins obligatoires :** (a) poids initialement nuls, non entraînés ; (b) poids figés à 1 avec exactement le même graphe et capacité de stockage. Le classificateur ne doit pas être comparé à un système déficient en ressources par construction sans publier son désavantage. Un poids non appris n'est pas « moins intelligent ».

**Première tâche, volontairement limitée :** rétention d'une association supervisée entre deux neurones sources réels distincts et une destination réelle commune. Les deux sources possèdent déjà une connexion anatomique vers cette destination ; le professeur artificiel demande l'activation pour la première et l'inhibition pour la deuxième. L'ordre des épisodes d'entraînement est prédéfini ; l'évaluation ne doit pas réentraîner. Les instances d'évaluation peuvent répéter les mêmes motifs, et les scores ne démontrent **aucune généralisation sur un stimulus inédit**, encore moins un apprentissage du comportement naturel de l'insecte.

**Critères d'homologation progressifs :**
1. Porte arithmétique 4 bits comparée exhaustivement à un oracle entier (16 cas), tests des saturations et de la topologie.
2. Évaluation reproductible sur petit graphe synthétique, incluant témoin figé et témoin initial.
3. Exécution du code sur le vrai graphe **SHA-pinné** du Dell sous RemoteOps, CPU uniquement, mesure des prédictions, temps d'apprentissage et de test, RAM, taille des poids et persistabilité.
4. Fournir plusieurs sources, tâches et graines, avec séparation TRAIN/VALIDATION/TEST par motif et tests OOD. Ajouter un témoin de graphe permuté apparié et un autre modèle apprenant.
5. Comparer qualité, coût et robustesse face à des architectures témoins **à budgets équivalents**. Un résultat égal peut justifier la poursuite ; aucun score de rétention ne constitue à lui seul une validation physiologique.

Ce premier mécanisme est **une plasticité dirigée supervisée**, non une preuve que le connectome de mouche apprend comme une mouche. Aucun résultat non exécuté ne doit être annoncé. Les poids et leurs données de provenance restent privés hors Git ; le format `V888PLS1` contient un contrôle de version, nombre de neurones, arêtes, une empreinte déterministe de la topologie et les coefficients 4 bits emballés. L'empreinte est un garde de compatibilité, pas un SHA-256 cryptographique.

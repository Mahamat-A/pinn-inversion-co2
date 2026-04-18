# Limites et perspectives

Ce document liste honnêtement les limites du système et les pistes d'amélioration.

## Limites structurelles

### 1. Problème du zéro (α × 0 = 0)

La formulation multiplicative `F_opt = α · F_prior` ne peut pas créer d'émissions là où le prior est nul. Si EDGAR a raté une nouvelle source, le PINN ne peut pas la détecter.

**Tenté** : V14 avec terme additif `F = α · F_prior + γ`
**Résultat** : LOSO 0.489 vs 0.612 — échec dû à la sur-paramétrisation (481 params)
**Tenté** : V14b avec γ annuel + L1 (261 params)
**Résultat** : LOSO 0.500 — toujours trop
**Perspective** : nécessite plus de stations ou multi-année

### 2. Résolution α grossière (20 régions)

Chaque région = ~130 000 km² (taille de la Grèce). Insuffisant pour politiques nationales.

**Tenté** : V15 avec 80 régions (8×10)
**Résultat** : LOSO 0.133 — effondrement
**Tenté** : V15b avec 36 régions (6×6)
**Résultat** : LOSO 0.378 — toujours trop
**Perspective** : OCO-2/3 satellite ou réseau ICOS étendu

### 3. β mensuel instable

La décomposition en 12 β indépendants est sous-contrainte (65 obs/mois).

**Tenté** : optimisation post-hoc mois par mois
**Résultat** : σ > valeur, instable
**Tenté** : β Fourier (3 paramètres au lieu de 12)
**Résultat** : LOSO 0.595 (Δ = -0.029, quasi V12b), signal canicule 1.4σ
**Perspective** : nécessaire, pas encore significatif — plus de données requises

### 4. Séparation α/β repose sur la structure des priors

La distinction fossile/biosphère dépend que EDGAR et VPRM aient des géographies différentes. Dans les zones périurbaines avec forêts adjacentes, la séparation est sous-déterminée.

**Bloqueur** : pas de données ¹⁴CO₂ ni CO disponibles
**Perspective** : le ¹⁴CO₂ (Levin et al., 2003) ou le CO comme co-traceur (Turnbull et al., 2011) résoudraient formellement l'ambiguïté

### 5. Transport HYSPLIT à 50 km

Piloté par ERA5 0.25°, trop grossier pour résoudre les panaches urbains. 6 stations sur 25 (24% du réseau) sont rejetées en V12b.

**Perspective** : transport CERRA 5.5 km (Ridal et al., 2024) — pipeline complet à refaire (~3 mois)

## Limites de données

### 6. Une seule année (2019)

2019 est atypique (canicule). Pas de validation inter-annuelle.

**Perspective** : répliquer sur 2018 (année normale, test β≈1 attendu) et 2020 (COVID, test baisse α attendue) — 2600 footprints HYSPLIT à recalculer par année, ~2 semaines par année

### 7. Bug historique du parseur ICOS

Tous les résultats "observations réelles" pré-fix (α=0.92, β=0.70) étaient **faux** — le parseur lisait `parts[1]` (SamplingHeight) au lieu de `parts[2]` (Year), rejetant toutes les lignes.

**Fix** : `int(parts[1])` → `int(parts[2])`
**Impact** : les résultats synthétiques (LOSO, ablations) ne sont **pas** affectés. Seules les prédictions sur obs réelles ont été recalculées. Les vraies valeurs : α=1.010±0.078, β=0.971±0.023.

## Limites méthodologiques

### 8. Quantification d'incertitude via MC Dropout

Le MC Dropout (Gal & Ghahramani, 2016) approxime la postérieure bayésienne mais n'est pas une matrice de covariance formelle.

**Perspective** : B-PINN (Bayesian Physics-Informed Neural Network) avec variational inference

### 9. V13 correcteur statique

Les 9 features du correcteur sont des moyennes annuelles. L'erreur dynamique (embouteillages, épisodes pollution) n'est pas capturée.

**Résolu** : V13b avec T2m hebdo, BLH hebdo, semaine (+58.7% MAE)

### 10. Flux océaniques négligés

Ocean = ~0.05% du total continental sur le domaine — négligeable pour 2019 Europe, mais limite pour domaines côtiers étendus.

## Validations manquantes (bloquées)

| Validation | Bloqueur |
|------------|----------|
| Multi-année (2018, 2020) | Recalcul HYSPLIT, ~2 semaines par année |
| Satellite OCO-2/3 | Intégration colonne vs point-mesure non triviale |
| ¹⁴CO₂ régional | Pas de données |
| TCCON (colonnes totales) | Intégration verticale à implémenter |

## Résumé : ce qui fait le projet

**Solide et publiable** :
- LOSO 0.612 ± 0.015 (synthétique, 19 stations)
- PINN ×12 vs bayésien
- α = 1.010 ± 0.078, β = 0.971 ± 0.023 (obs réelles, MC Dropout)
- r = 0.992 vs CAMS (système indépendant)
- Withholding JJA : Δ = -0.002
- V13b : +58.7% MAE
- T2m-CO₂ : r = -0.955

**Limites quantifiées** :
- Max 240 paramètres pour 19 stations
- V14 (γ additif), V15 (80 régions), β mensuel : limites numériquement documentées

**Perspectives réalistes** :
- Multi-année (2018, 2020) pour validation climatique
- Satellite OCO-2/3 pour densifier les observations
- CERRA 5.5 km pour résoudre les panaches urbains
- B-PINN pour incertitude formelle

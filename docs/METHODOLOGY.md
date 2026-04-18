# Méthodologie

## Formulation mathématique

Équation de transport-inversion découplée :

```
C_mod(s, t) = C_bg(t) + Σ_{i,j} H(s, t, i, j) · [α(r(i,j), m(t)) · F_foss(t, i, j) + β · F_bio(t, i, j)]
```

où :
- `C_mod` : fraction molaire CO₂ modélisée à la station `s`, semaine `t`
- `C_bg` : concentration de fond (Mace Head MHD)
- `H` : empreinte HYSPLIT (µmol/m²/s → ppm)
- `α(r, m)` : facteur multiplicatif fossile par région `r` et mois `m` (20 × 12 = 240 paramètres)
- `β` : facteur global biosphère (1 paramètre)
- `F_foss` : prior fossile (CarbonTracker CT2022)
- `F_bio` : prior biosphère (VPRM ECMWF)

## Loss PINN

```
L(α, β) = MSE(α_pred, α_true) + MSE(β_pred, β_true)
        + λ_α · ||α||²                  # régularisation vers 1
        + λ_spatial · ||∇_spatial α||²   # lissage spatial
        + λ_temporal · ||∇_temporal α||² # lissage temporel
```

Valeurs : λ_α = 0.1, λ_spatial = 0.05, λ_temporal = 0.03

## Architecture réseau

```
Input (1976 dims) — features ΔC × ratio_BLH par station/semaine
  ↓
Dense(512, gelu) + Dropout(0.15) + LayerNorm
  ↓
Dense(512, gelu) + Dropout(0.15) + LayerNorm
  ↓
Dense(256, gelu) + Dropout(0.1) + LayerNorm
  ↓
    ├─→ Dense → Reshape(4,5,16) → Conv2DTranspose → Conv2D → α (240)
    └─→ Dense → β (1)
  ↓
Output: [α(240), β(1)] = 241 paramètres
```

## Configuration des scénarios

- **5000 scénarios synthétiques**
- α ∈ [0.5, 1.5] (uniforme) par région et mois
- β ∈ [0.7, 1.3] (uniforme) global
- Bruit gaussien 2% sur les features

## Validation croisée LOSO

Leave-One-Station-Out : pour chaque station, retirer du jeu d'entraînement, entraîner le PINN, évaluer la corrélation α.

## MC Dropout (Gal & Ghahramani, 2016)

50 passages stochastiques avec Dropout actif à l'inférence → distribution postérieure approchée de α et β. Intervalles de confiance 95% = moyenne ± 2σ.

## Versions

### V1–V5 : Baseline et calibration (2019)
- V1 : synthétique pur, r=0.253
- V2 : EDGAR réel, r=0.475
- V3-V4 : Conv2D + VPRM, r≈0.45
- V5 : 25 stations, découverte bug EDGAR ×10⁶

### V6 : Découplage (le grand saut)
- `C = H(αF_foss + βF_bio)` au lieu de `C = H(αF_total)`
- r = 0.523, LOSO = 0.417
- Δ = +0.326 sur r, première fois > bayésien

### V7 : BLH ERA5
- Ajout BLH comme variable prédictive
- r = 0.600 (+0.077)

### V8 : Normalisation CLA
- Diagnostic : `⟨ΔC × CLA⟩ ≠ ⟨ΔC⟩ × ⟨CLA⟩`
- Gain marginal (+0.006)

### V9 : Séparation jour/nuit
- 12-16h (convectif) vs 0-4h (stable)
- r = 0.609, Δ = +0.075

### V10 : Échec diagnostique
- Footprints mensuels + features hebdo → incohérence
- r = 0.239 (effondrement)
- Leçon : cohérence temporelle FP/features non-négociable

### V11 : Footprints hebdomadaires
- 2600 footprints (25 stations × 52 semaines × 2 régimes)
- r = 0.648, LOSO = 0.487

### V12b : Filtrage spatial
- Application recommandation M2 (Mahamat, 2022)
- Exclusion 6 stations urbaines : KIT, IPR, JUS, JUE, OVS, SAC
- LOSO = 0.612 (+0.125 vs V11)
- **Configuration finale retenue**

### V13 : Correcteur sous-maille (statique)
- Double réseau : PINN + correcteur résidus
- 9 features statiques par station
- -24.7% résidus urbains

### V13b : Correcteur dynamique
- + T2m hebdo, BLH hebdo, semaine
- BLH nocturne = feature dominante (importance 20×)
- +58.7% MAE vs V13 statique

### V14 (ÉCHEC) : γ additif
- `F = α · F_prior + γ` pour résoudre le problème du zéro
- 481 paramètres (240 α + 1 β + 240 γ)
- LOSO = 0.489 (-0.123) — trop de paramètres
- V14b (γ annuel, 261 params) : LOSO = 0.500 (toujours trop)

### V15 (EFFONDREMENT) : 80 régions
- 8×10 = 80 régions au lieu de 4×5 = 20
- 961 paramètres
- LOSO = 0.133 (-0.492) — sous-détermination catastrophique
- V15b (36 régions, 433 params) : LOSO = 0.378 (toujours trop)

### β Fourier : compromis viable
- `β(m) = β₀ + β₁cos(2πm/12) + β₂sin(2πm/12)`
- 3 paramètres au lieu de 12
- LOSO = 0.595 (Δ = -0.029, quasi équivalent V12b)
- Sur obs réelles : amplitude 1.4σ (signal direction correcte, non significatif)

## Leçon fondamentale

Avec 19 stations et 52 semaines, le système ne peut contraindre que **~240 paramètres**. Au-delà, chaque paramètre ajouté dégrade la performance. Cette limite est structurelle.

## Validations externes

### CAMS (Copernicus Atmosphere Monitoring Service)
- Système totalement indépendant (transport IFS/LMDZ, prior ORCHIDEE, méthode 4D-Var)
- V12 vs CAMS spatial : r = 0.992
- CT2022 vs CAMS spatial : r = 0.988
- Notre V12 concorde mieux avec CAMS que CT2022 seul (+0.004)

### CarbonTracker CT2022
- V12 vs CT2022 spatial : r = 0.999 (attendu, CT est notre prior fossile)
- V12 vs CT2022 temporel : r = 0.958 (corrections réelles)

### Validation forward
- C_mod reconstruit vs C_obs ICOS : r = 0.422 moyen
- Top : OPE = 0.751, TRN = 0.740, TOH = 0.717

### Withholding temporel (17 semaines JJA)
- Entraîné + inférence masqués (pas de domain shift)
- Δ LOSO JJA = -0.002 (pas de dégradation)
- Le modèle prédit l'été sans l'avoir vu

## Références

Voir `docs/REFERENCES.md`

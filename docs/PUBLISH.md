# Publication sur GitHub + Zenodo

## Étape 1 : GitHub (15 min)

### 1.1 Créer le repo

1. Aller sur https://github.com/new
2. Nom : `pinn-inversion-co2`
3. Description : "PINN + HYSPLIT for CO2 flux inversion — Europe 2019"
4. **Public**
5. **Ne pas** initialiser avec README (on en a déjà un)
6. Créer

### 1.2 Pousser le code depuis ton PC

```bash
cd ~/Téléchargements/pinn-inversion-co2  # Où tu décompresses l'archive

# Initialiser git
git init
git add .
git commit -m "Initial commit: PINN inversion CO2 Europe 2019"

# Lier au repo GitHub (remplacer YOUR_USERNAME)
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/pinn-inversion-co2.git
git push -u origin main
```

### 1.3 Vérifier

- README.md s'affiche correctement
- LICENSE visible
- Scripts dans `scripts/`
- Docs dans `docs/`

## Étape 2 : Zenodo (10 min)

### 2.1 Créer un compte Zenodo

1. https://zenodo.org/signup/
2. Se connecter avec GitHub (recommandé)

### 2.2 Lier GitHub à Zenodo

1. Aller sur https://zenodo.org/account/settings/github/
2. Cliquer "Sync now"
3. Trouver `pinn-inversion-co2` dans la liste
4. Activer le switch (ON)

### 2.3 Créer un release GitHub

1. Retour sur GitHub, onglet "Releases" → "Create a new release"
2. Tag : `v1.0.0`
3. Title : `v1.0.0 - Initial release`
4. Description :

```markdown
## Initial release

PINN + HYSPLIT framework for CO2 flux inversion over Europe 2019.

### Main results
- LOSO α r = 0.612 ± 0.015 (19 stations)
- PINN ×12 vs bayesian (0.417 vs 0.033)
- α = 1.010 ± 0.078, β = 0.971 ± 0.023 (real observations)
- r = 0.992 vs CAMS (independent validation)

### Content
- 10 scripts (v11 baseline → v15 resolution test + validations)
- Full documentation (methodology, data, limitations)
- MIT licensed
```

5. **Publish release**

### 2.4 Récupérer le DOI

1. Aller sur https://zenodo.org/account/settings/github/
2. Le release apparaît avec un DOI (format : `10.5281/zenodo.XXXXXXX`)
3. Copier le DOI

### 2.5 Mettre à jour le DOI dans le repo

Éditer sur GitHub (ou localement) :

- `README.md` : ligne avec `zenodo.XXXXXXX` → mettre le vrai DOI
- `CITATION.cff` : idem

```bash
git add README.md CITATION.cff
git commit -m "Add Zenodo DOI"
git push
```

## Étape 3 : Vérifier la citation

Test : chercher ton DOI sur https://doi.org/10.5281/zenodo.XXXXXXX — ça doit rediriger vers la page Zenodo.

## Étape 4 : Partager

- **CV / LinkedIn** : ajouter le DOI Zenodo (preuve de contribution scientifique publiée)
- **Rapport final** : citer `doi.org/10.5281/zenodo.XXXXXXX`
- **HAL** (optionnel) : uploader aussi sur https://hal.science/ pour visibilité française

## Avantages

1. **DOI permanent** — citable dans publications
2. **Archive pérenne** — Zenodo est CERN, conservation garantie
3. **Snapshot versionné** — chaque release = nouveau DOI
4. **Métadonnées complètes** — CITATION.cff lu automatiquement

## Mises à jour futures

Quand tu ajoutes une nouvelle version :

```bash
git add .
git commit -m "v1.1: Add 2020 COVID analysis"
git tag v1.1.0
git push --tags
```

Puis créer un release v1.1.0 sur GitHub → nouveau DOI Zenodo automatiquement.

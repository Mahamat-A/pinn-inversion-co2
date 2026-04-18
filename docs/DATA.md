# Données

Ce projet utilise 6 sources de données externes. Toutes sont ouvertes mais nécessitent un téléchargement manuel (trop volumineuses pour GitHub).

## 1. ICOS CO₂ (25 stations, 2019)

**Source** : https://data.icos-cp.eu/portal/

**Licence** : CC-BY 4.0

**Format** : fichiers `.co2` (texte, séparateur `;`)

**Stations utilisées** :
```
SAC, OPE, KIT, TRN, PUY, HPB, LUT, RGL, BIS, CMN, CRA, ERS,
GAT, IPR, JUE, JUS, LIN, OHP, OVS, OXK, PDM, STE, TAC, TOH, WAO
```

**Placer dans** : `~/hysplit/icos_data/`

**Note importante sur le parseur** : les fichiers ICOS ont des colonnes :
```
Site; SamplingHeight; Year; Month; Day; Hour; Minute; DecimalDate; co2; ...
parts[0] parts[1]       parts[2] parts[3] ...
```
Ne pas confondre `parts[1]` (SamplingHeight) avec `parts[2]` (Year).

## 2. ERA5 (BLH + T2m)

**Source** : Copernicus Climate Data Store (CDS)

**API** : https://cds.climate.copernicus.eu

```bash
pip install cdsapi
# Configurer ~/.cdsapirc avec votre clé CDS
```

**Variables** :
- `boundary_layer_height` (BLH)
- `2m_temperature` (T2m)

**Période** : 2019, horaire

**Domaine** : 40°–56°N, -10°–15°E

**Résolution** : 0.25°

**Fichiers attendus** :
- `era5_blh_2019_full.nc`
- `era5_t2m_2019.nc`

## 3. CarbonTracker CT2022

**Source** : https://gml.noaa.gov/ccgg/carbontracker/

**Fichier** : `CT2022.flux1x1-monthly.nc`

**Résolution** : 1° × 1°, mensuelle

## 4. CAMS Flux (pour validation)

**Source** : https://ads.atmosphere.copernicus.eu/datasets/cams-global-greenhouse-gas-inversion

**Sélectionner** : CO₂, surface flux, surface, 2019, monthly mean, NetCDF

**Fichiers** : `cams73_latest_co2_flux_surface_mm_2019XX.nc` (12 fichiers)

**Combiner avec** :
```python
# Voir scripts/validation_cams.py
```

## 5. VPRM ECMWF

**Source** : Contact ECMWF ou IPSL

**Fichier** : `VPRM_ECMWF_NEE_2019_CP.nc`

**Variable** : `NEE` (Net Ecosystem Exchange)

**Résolution** : ~0.1°, horaire

## 6. EDGAR v8.0

**Source** : https://edgar.jrc.ec.europa.eu/

**Fichier** : `v8.0_FT2022_GHG_CO2_2019_TOTALS_flx.nc`

**Résolution** : 0.1°, annuel total

## Organisation finale

```
~/hysplit/
├── icos_data/
│   ├── MHD_24.0m_air.hdf.2019.co2
│   ├── TRN_180.0m_air.hdf.2019.co2
│   └── ...
├── flux_data/
│   ├── CT2022.flux1x1-monthly.nc
│   ├── era5_blh_2019_full.nc
│   ├── era5_t2m_2019.nc
│   ├── VPRM_ECMWF_NEE_2019_CP.nc
│   ├── v8.0_FT2022_GHG_CO2_2019_TOTALS_flx.nc
│   ├── cams_2019_combined.npz
│   └── ct2022_prior_monthly.npz
├── footprints_weekly/
│   └── fp_STATION_wNN_[day|night].npz (2600 fichiers)
└── results/
    └── (sorties des scripts)
```

## Footprints HYSPLIT

Les empreintes de transport sont calculées avec HYSPLIT v5.2.0.

**Configuration** :
- Rétro-trajectoires 5 jours
- Piloté par ERA5 0.25°
- Grille sortie : 32 × 50 à 0.5° (40°–56°N, -10°–15°E)
- Résolution temporelle : hebdomadaire, séparation jour (12-16h UTC) / nuit (0-4h UTC)
- Plafond vertical : BLH ERA5 dynamique par trajectoire

Le script de génération des footprints n'est pas inclus (dépend de l'installation HYSPLIT locale). Voir la documentation officielle HYSPLIT pour la mise en place.

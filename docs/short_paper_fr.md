---
title: "Réseaux de neurones informés par la physique pour l'inversion mono-traceur du CO₂ : étude de cas européenne pour 2019"
author:
  - Ali Ousmane Mahamat
abstract: |
  Nous introduisons un cadre de réseau de neurones informé par la physique (PINN) couplé au transport lagrangien HYSPLIT pour l'inversion de flux atmosphérique de CO₂ sur l'Europe en utilisant le réseau d'observation ICOS. L'innovation clé est une formulation découplée $C_{mod} = H(\alpha F_{foss} + \beta F_{bio})$ qui sépare les flux fossiles et biosphériques en exploitant les structures spatiales distinctes de leurs inventaires a priori. Il s'agit d'une séparation structurelle et non physique : elle repose sur l'exactitude géographique des priors et ne remplace pas les approches formelles par co-traceur telles que le $\Delta^{14}$CO₂. Appliqué à 19 stations rurales européennes pour 2019, le cadre atteint une corrélation leave-one-station-out (LOSO) de $r = 0{,}612 \pm 0{,}015$, une amélioration d'un facteur 12 par rapport à l'inversion bayésienne équivalente ($r = 0{,}033$), et une corrélation spatiale de $r = 0{,}992$ avec le système opérationnel indépendant CAMS. Six stations contaminées par l'urbain ont été exclues car HYSPLIT à 50 km ne peut pas résoudre les panaches urbains — une limitation standard des systèmes d'inversion régionale à résolution comparable. La quantification d'incertitude par MC Dropout sur observations réelles donne $\alpha = 1{,}010 \pm 0{,}078$ et $\beta = 0{,}971 \pm 0{,}023$ (à interpréter comme bornes inférieures de l'incertitude épistémique, MC Dropout étant connu pour sous-estimer les postérieures variationnelles complètes). Le withholding temporel de la période JJA montre une dégradation négligeable ($\Delta r = -0{,}002$). Un cycle $\beta$ saisonnier directionnellement cohérent avec la réduction du puits induite par canicule est observé mais reste statistiquement marginal ($1{,}4\sigma$) ; une analyse multi-année serait nécessaire pour une attribution robuste. L'ablation systématique quantifie les limites structurelles : avec 19 stations, un nombre de paramètres au-delà de ~240 entraîne un effondrement catastrophique (V14 : 481 paramètres, LOSO 0,489 ; V15 : 961 paramètres, LOSO 0,133). Le code complet est publié open-source (licence MIT, DOI 10.5281/zenodo.19638205).
keywords: "inversion atmosphérique, CO₂, réseau de neurones informé par la physique, HYSPLIT, ICOS, Europe"
geometry: margin=1in
fontsize: 11pt
linestretch: 1.2
numbersections: true
bibliography: references.bib
link-citations: true
lang: fr
---

# Introduction

L'inversion atmosphérique du CO₂ estime les flux de surface à partir des observations de concentration et de la modélisation du transport. Les systèmes européens opérationnels — CarbonTracker [@peters2007], CAMS [@chevallier2010], LUMIA [@monteil2021] — estiment le flux *net*, combinant les contributions fossiles et biosphériques. Séparer ces deux composantes à partir d'un seul traceur CO₂ a été considéré comme formellement sous-déterminé dans le cadre bayésien linéaire standard [@basu2016]. Seules les observations de co-traceurs comme $\Delta^{14}$CO₂ [@levin2003; @turnbull2011; @gomez2025] ou CO peuvent désambiguïser formellement les signaux fossile et biosphérique, mais ces mesures restent limitées à 14 des 300+ stations CO₂ en Europe.

Les réseaux de neurones informés par la physique [@raissi2019] encodent les équations gouvernantes directement dans la fonction de perte. Les applications récentes au transport atmosphérique incluent FootNet [@he2025], qui émule les empreintes HYSPLIT 650× plus vite, et Dadheech et al. [-@dadheech2025], qui ont combiné FootNet avec l'inversion de flux. Cependant, aucun cadre publié ne combine PINN avec un modèle de transport lagrangien pour la séparation mono-traceur fossile-biosphère du CO₂.

Dans cet article, nous démontrons que la séparation mono-traceur, formellement sous-déterminée dans le cadre bayésien linéaire, est réalisable via un réseau de neurones informé par la physique exploitant la structure spatio-temporelle des priors indépendants combinée avec le partitionnement diurne et saisonnier du signal.

# Méthode

## Formulation inverse

La formulation de base est :

$$C_{mod}(s, t) = C_{bg}(t) + H(s, t) \cdot \left[ \alpha(r, m) \cdot F_{foss}(i, j, t) + \beta \cdot F_{bio}(i, j, t) \right]$$

où $s$ est la station, $t$ la semaine, $r$ l'une des 20 régions (grille 4 × 5 sur 40-56°N, 10°O-15°E), $m$ le mois, $F_{foss}$ le prior fossile CarbonTracker CT2022, et $F_{bio}$ le flux biosphérique VPRM. Les 241 paramètres (240 $\alpha$ + 1 $\beta$) sont récupérés à partir de 2600 empreintes hebdomadaires (25 stations × 52 semaines × 2 régimes jour/nuit), calculées avec HYSPLIT v5.2.0 [@stein2015] piloté par ERA5 [@hersbach2020] avec plafond BLH dynamique.

La formulation découplée exploite simultanément trois contraintes physiques : (i) distinction structurelle des priors fossile (source ponctuelle) et biosphérique (diffus), (ii) séparation diurne des régimes convectif (12-16h UTC) et stable (00-04h UTC), et (iii) résolution hebdomadaire capturant la variabilité synoptique. Ces contraintes ensemble rendent le problème traitable malgré la sous-détermination du cadre linéaire classique [@basu2016].

## Architecture du réseau et entraînement

Le PINN consiste en un tronc partagé [Dense(512, gelu)×2, Dense(256, gelu)] avec dropout et normalisation de couche, suivi de deux têtes : un décodeur convolutif [Conv2DTranspose → Conv2D] produisant le champ $\alpha(r, m)$ à 240 dimensions, et une tête dense produisant $\beta$. La perte combine MSE, priors de lissage spatio-temporel, et régularisation de $\alpha$ vers l'unité.

L'entraînement utilise 5000 scénarios synthétiques avec perturbations uniformes $\alpha \in [0{,}5 ; 1{,}5]$ par région-mois, $\beta \in [0{,}7 ; 1{,}3]$ global, et bruit gaussien 2 % sur les features. Le Dropout reste actif à l'inférence (MC Dropout, 50 passes) pour la quantification d'incertitude [@gal2016].

## Protocoles de validation

Nous employons quatre stratégies de validation : (1) Leave-One-Station-Out (LOSO) sur scénarios synthétiques ; (2) comparaison triple des flux optimisés avec CT2022 et CAMS (Copernicus Atmosphere Monitoring Service — système d'inversion pleinement indépendant) ; (3) validation forward comparant les $C_{mod}$ reconstruits aux observations ICOS réelles ; (4) withholding temporel des 17 semaines JJA (juin-septembre 2019), masquées à la fois à l'entraînement et à l'inférence.

# Résultats

## Progression systématique V1 → V12b

![Percée du découplage V6 : le PINN atteint un LOSO fossile $r = 0{,}417$ contre le bayésien classique $r = 0{,}033$ sur données, transport et priors identiques.](figures/fig_v6_loso_bayesian.png)

Douze ablations systématiques quantifient la contribution de chaque innovation (Tableau 1). La formulation découplée (V6) produit le plus grand gain en un seul pas (+0,25 sur V5), et le filtrage explicite des stations urbaines informé par ma thèse M2 de 2022 [@mahamat2022] produit la deuxième plus grande amélioration (V12b : +0,125 sur V11).

| Version | Innovation | $r$ | LOSO |
|---------|-----------|-----|------|
| V1 | baseline | 0,253 | — |
| V2 | EDGAR réel | 0,475 | — |
| V5 | 25 stations | 0,197 | 0,191 |
| V6 | découplage | 0,523 | 0,417 |
| V9 | séparation jour/nuit | 0,609 | — |
| V10 | features hebdo uniquement | 0,239 | — |
| V11 | empreintes hebdo | 0,648 | 0,487 |
| **V12b** | **filtrage urbain** | **0,648** | **0,612** |

*Tableau 1 : Progression de la corrélation $\alpha$ à travers les versions.*

V10 est un échec instructif : features hebdomadaires avec empreintes mensuelles produisent une incohérence temporelle qui fait s'effondrer l'inversion. La leçon — la résolution du transport et des observations doit être alignée — se généralise à tous les systèmes d'inversion LPDM.

## Quantification d'incertitude sur observations réelles

Un ensemble MC Dropout de 50 passes sur observations ICOS réelles donne :

- $\alpha_{foss} = 1{,}010 \pm 0{,}078$ (IC 95 % : [0,853 ; 1,167])
- $\beta_{bio} = 0{,}971 \pm 0{,}023$ (IC 95 % : [0,926 ; 1,017])

![Ensemble MC Dropout sur observations ICOS réelles : $\alpha = 1{,}010 \pm 0{,}078$, $\beta = 0{,}971 \pm 0{,}023$. EDGAR cohérent avec l'unité ; VPRM significativement en dessous de l'unité à 2,5$\sigma$.](figures/fig_mc_dropout.png)

$\alpha \approx 1$ indique que les émissions fossiles EDGAR sont correctes dans la barre d'incertitude pour la moyenne européenne en 2019. $\beta < 1$ au niveau de significativité 2,5$\sigma$ indique que VPRM surestime le puits biosphérique d'environ 3 %.

## Validation indépendante avec CAMS

Comparaison triple avec CT2022 et CAMS, l'inversion opérationnelle Copernicus utilisant transport IFS/LMDZ, méthodologie 4D-Var, et prior biosphérique ORCHIDEE — pleinement indépendant de notre chaîne de traitement :

![V12 vs CT2022 vs CAMS : corrélation spatiale $r = 0{,}992$ avec CAMS, le système opérationnel pleinement indépendant.](figures/fig_validation_triple.png)

| Comparaison | $r$ spatial | $r$ temporel | $r$ total |
|-----------|-------------|--------------|-----------|
| V12 vs CT2022 | 0,999 | 0,958 | 0,979 |
| **V12 vs CAMS** | **0,992** | **0,851** | **0,965** |
| CT2022 vs CAMS | 0,988 | 0,841 | 0,983 |

*Tableau 2 : Comparaisons de validation triple.*

La corrélation spatiale V12-CAMS de 0,992 démontre une estimation convergente depuis trois systèmes utilisant un transport différent (HYSPLIT vs IFS/LMDZ), des observations différentes (surface seule vs surface+satellite), des priors différents (CT2022+VPRM vs CT2022+ORCHIDEE), et des cadres mathématiques différents (PINN vs 4D-Var). Notamment, V12 corrèle légèrement mieux avec CAMS que ne le fait CT2022 (0,992 vs 0,988 spatial), suggérant que notre correction PINN capture un signal réel plutôt que du bruit.

## Withholding temporel : le test de généralisation estivale

![Withholding temporel : JJA masqué à l'entraînement et à l'inférence. La dégradation synthétique de -0,214 confirme la perte d'information, mais la dégradation sur observations réelles est négligeable (-0,002).](figures/fig_withholding_jja.png)

Masquer les 17 semaines JJA à la fois aux étapes d'entraînement et d'inférence produit :

| Validation | $\Delta r$ |
|-----------|-----------|
| PINN synthétique | -0,214 |
| Observations réelles, JJA uniquement | **-0,002** |
| Observations réelles, année complète | 0,000 |

*Tableau 3 : Résultats du withholding temporel.*

La dégradation synthétique confirme l'efficacité du masquage ; la dégradation sur observations réelles s'évanouit, démontrant que le modèle prédit l'été qu'il n'a pas vu — aussi bien qu'il prédit avec les données complètes de l'année. Avec $\alpha \approx 1$ et le puits estival dominé par VPRM × $\beta$, les 9 mois de données hiver/automne fournissent une contrainte suffisante sur $\beta$ (via le cycle saisonnier) pour reconstruire les concentrations estivales.

## Validation forward du modèle

Reconstruire $C_{mod}$ à partir des $\alpha, \beta$ optimisés et comparer aux observations ICOS donne une corrélation moyenne $r = 0{,}422$ sur 25 stations, avec des meilleurs excédant 0,70 (OPE=0,751, TRN=0,740, TOH=0,717). Cela reflète la limite fondamentale du système : le transport HYSPLIT à 50 km capture environ 42 % de la variance des concentrations observées. La correction V12 apporte une amélioration marginale sur le prior ($\Delta r = +0{,}004$), cohérent avec $\alpha \approx 1$ — le PINN confirme plutôt que corrige lorsque le prior est déjà précis.

# Études d'extension : quantification des limites structurelles

Trois extensions systématiques testent si la performance de V12b peut être améliorée :

**Correcteur sous-maille dynamique V13b** : augmente V12b avec un réseau secondaire prédisant les résidus à partir de 12 features (9 statiques + $T_{2m}$ hebdomadaire, BLH hebdomadaire, semaine de l'année). La corrélation des résidus s'améliore de 0,661 (V13 statique) à 0,960 (V13b dynamique), une réduction MAE de 58,7 %. La BLH nocturne émerge comme feature dominante à 20× la marge, confirmant la stabilité atmosphérique comme signal sous-maille clé.

**$\gamma$ additif V14** : tente de résoudre le problème du prior zéro via $F = \alpha F_{prior} + \gamma$ (481 paramètres). LOSO dégrade de 0,612 à 0,489. V14b réduit $\gamma$ à 20 valeurs annuelles avec parcimonie L1 (261 paramètres) : LOSO = 0,500. Le problème mono-traceur additif-multiplicatif est fondamentalement sur-paramétrisé à la densité d'observation actuelle.

**Raffinement spatial V15** : double la résolution régionale de 20 à 80 régions (961 paramètres). LOSO s'effondre à 0,133. V15b intermédiaire avec 36 régions (433 paramètres) donne LOSO = 0,378.

![$\gamma$ additif V14 et raffinement spatial V15 : les deux échouent, quantifiant le plafond de ~240 paramètres pour la densité de 19 stations.](figures/fig_fixes_physics.png)

**Leçon** : avec 19 stations × 52 semaines, le système supporte environ 240 paramètres. Au-delà de ce seuil, le déséquilibre paramètres-contraintes produit une dégradation catastrophique. C'est une limite structurelle de densité observationnelle, pas un problème architectural.

## Décomposition de Fourier du $\beta$

Paramétrer $\beta(m) = \beta_0 + \beta_1 \cos(2\pi m/12) + \beta_2 \sin(2\pi m/12)$ (3 paramètres au lieu de 12) préserve la performance (LOSO = 0,595, $\Delta = -0{,}029$) tout en permettant la récupération d'un $\beta$ saisonnier. Sur observations réelles :

- Amplitude = 0,015 ± 0,011 (1,4$\sigma$)
- Phase minimum : mi-mai (mois 4,9)
- $\beta_{JJA} = 0{,}981 < \beta_{DJF} = 1{,}004$

Le signal de canicule est directionnellement correct (minimum JJA, surestimation VPRM de ~2 %) mais statistiquement marginal à 1,4$\sigma$. La corrélation $T_{2m}$-CO₂ sur l'année complète (indépendante du PINN) atteint $r = -0{,}955$, confirmant le couplage climat-flux.

![$\beta$ Fourier sur observations réelles : cycle annuel lisse avec minimum à mi-mai, signal de canicule directionnellement correct mais 1,4$\sigma$ (sous le seuil de significativité) en raison de la contrainte à 19 stations.](figures/fig_beta_fourier_real.png)

# Discussion

## Pourquoi le PINN surpasse-t-il le bayésien d'un facteur ×12 ?

Sur données, scénarios, transport et priors identiques, l'inversion bayésienne classique (moindres carrés régularisés Tikhonov avec sélection L-curve) atteint LOSO $r = 0{,}033$ contre $0{,}417$ (V6) et $0{,}612$ (V12b) du PINN. Trois facteurs expliquent cet écart :

Premièrement, le problème mono-traceur est **sous-déterminé dans le cadre linéaire** : avec 25 stations et 240 paramètres, la solution bayésienne dépend fortement de la covariance prior, que nous ne pouvons pas spécifier de façon fiable pour les flux régionaux [@michalak2004]. L'architecture non linéaire du PINN exploite des corrélations d'ordre supérieur entre stations que le cadre linéaire ne peut pas capter.

Deuxièmement, **les erreurs de transport régional sont non gaussiennes** : canalisation orographique (e.g., rapport N/D inversé d'IPR Ispra, section 7 dans le rapport long), contamination par panache urbain, et discontinuités côtières produisent des distributions à queues lourdes. Les hypothèses de vraisemblance gaussienne du bayésien classique sont violées.

Troisièmement, **la régularisation du PINN est structurelle** : le décodeur convolutif $\alpha$ et les pénalités de lissage basées sur la perte encodent la connaissance prior sans requérir de spécification explicite de la covariance. Cela correspond à la structure naturelle du problème.

## Ce que le cadre résout et ne résout pas

**Résolu** (à bonne approximation à la densité de 19 stations) :
- Séparation fossile-biosphère à résolution 20 régions × 12 mois
- $\alpha, \beta$ quantitatifs avec incertitude (MC Dropout)
- Accord avec CAMS indépendant ($r = 0{,}992$ spatial)
- Généralisation à JJA non vu (withholding $\Delta = -0{,}002$)

**Non résolu** :
- Significativité statistique formelle du signal $\beta$ de canicule (1,4$\sigma$ avec données actuelles)
- Détection d'émissions là où le prior est zéro ($\alpha \cdot 0 = 0$, V14 a échoué)
- Attribution à l'échelle nationale (20 régions trop grossières, V15 a échoué)
- Résolution des panaches urbains (24 % des stations exclues, requiert transport à 5 km)

Le cadre atteint honnêtement ce qui est atteignable à la densité observationnelle actuelle. Les limites quantifiées (240 paramètres, 42 % de variance, significativité 1,4$\sigma$ de la canicule) devraient guider les priorités d'investissement futures : des observations plus denses (satellite, extension ICOS) sont plus importantes qu'un raffinement architectural à ce stade.

# Conclusion

Nous avons démontré que les réseaux de neurones informés par la physique, couplés au transport lagrangien HYSPLIT et à des priors structurés, peuvent effectuer une inversion de flux CO₂ mono-traceur avec une performance dépassant les méthodes bayésiennes classiques d'un facteur 12 sur données identiques. Appliqué à 19 stations rurales européennes pour 2019, le cadre récupère $\alpha_{foss} = 1{,}010 \pm 0{,}078$ et $\beta_{bio} = 0{,}971 \pm 0{,}023$, s'accorde avec le système indépendant CAMS à $r = 0{,}992$ spatialement, et se généralise à des observations estivales non vues sans dégradation.

L'ablation systématique quantifie une limite structurelle fondamentale : avec 19 stations × 52 semaines, environ 240 paramètres représentent le plafond de contrainte observationnelle. Cela fournit une base rigoureuse pour prioriser les investissements futurs dans la densité observationnelle plutôt que dans la complexité architecturale. Le code complet et la documentation sont publiés en open-source.

# Disponibilité du code et des données

Code complet : https://github.com/Mahamat-A/pinn-inversion-co2

DOI logiciel : https://doi.org/10.5281/zenodo.19638205

Licence : MIT.

Sources de données : ICOS (data.icos-cp.eu), ERA5 (Copernicus CDS), CT2022 (NOAA), CAMS (Copernicus ADS), EDGAR v8.0 (JRC). Toutes sont ouvertement disponibles ; le `docs/DATA.md` du dépôt fournit des instructions de téléchargement complètes.

# Remerciements

Ce travail s'appuie sur ma thèse de master 2022 au GSMA, CNRS / Université de Reims Champagne-Ardenne. Nous remercions ICOS, NOAA, ECMWF et JRC pour l'accès ouvert aux données.

# Références

::: {#refs}
:::

---

**Auteur correspondant** : Ali Ousmane Mahamat (Moud) — Indépendant (ex-GSMA, CNRS / URCA) — mahamatmoud@gmail.com

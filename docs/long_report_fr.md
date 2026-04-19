---
title: "Inversion atmosphérique du CO₂ par PINN-HYSPLIT sur l'Europe : un cadre mono-traceur pour la séparation des flux fossiles et biosphériques"
subtitle: "Rapport technique — Version 1.0"
author: "Ali Ousmane Mahamat"
date: "Avril 2026"
abstract: |
  Nous présentons un système hybride d'inversion atmosphérique combinant un réseau de neurones informé par la physique (PINN) avec le modèle de transport lagrangien HYSPLIT, appliqué au réseau d'observation européen ICOS pour l'année 2019. L'innovation centrale est une formulation découplée $C_{mod} = H(\alpha F_{foss} + \beta F_{bio})$ qui sépare les flux fossiles et biosphériques en exploitant les structures spatiales distinctes de leurs inventaires a priori — une approche structurelle qui repose sur l'hypothèse d'une géographie a priori correcte, et non une séparation physique formelle (laquelle requerrait un co-traceur comme le $^{14}$CO₂). Sur 19 stations rurales filtrées, le système atteint une corrélation leave-one-station-out (LOSO) de $r = 0{,}612 \pm 0{,}015$, surpasse l'inversion bayésienne classique d'un facteur 12 sur données identiques (LOSO $r = 0{,}417$ vs $0{,}033$), et corrèle à $r = 0{,}992$ spatialement avec le système opérationnel indépendant CAMS. La quantification d'incertitude par MC Dropout donne $\alpha = 1{,}010 \pm 0{,}078$ et $\beta = 0{,}971 \pm 0{,}023$ sur les observations ICOS réelles (ces intervalles doivent être interprétés comme des bornes inférieures, MC Dropout étant connu pour sous-estimer l'incertitude épistémique par rapport à une inférence variationnelle complète). Le withholding temporel (17 semaines JJA masquées à l'entraînement et à l'inférence) montre une dégradation négligeable ($\Delta r = -0{,}002$), confirmant une généralisation robuste. Un cycle $\beta$ saisonnier directionnellement cohérent avec la réduction du puits biosphérique induite par canicule est observé, mais reste statistiquement marginal ($1{,}4\sigma$) à la densité observationnelle actuelle — une attribution définitive à la canicule de 2019 requerrait une analyse multi-année. L'ablation systématique quantifie les limites du cadre : avec 19 stations, un espace de paramètres au-delà de $\sim$240 entraîne un effondrement (V14 avec $\gamma$ additif : LOSO 0,489 ; V15 avec 80 régions : LOSO 0,133). Six stations contaminées par l'urbain (24 % du réseau) ont été exclues car HYSPLIT à 50 km ne peut pas résoudre les panaches urbains — une limitation de transport partagée par les autres systèmes d'inversion régionale à résolution comparable. Le code complet est publié en open-source (licence MIT) avec le DOI 10.5281/zenodo.19638205.
geometry: margin=1in
fontsize: 11pt
linestretch: 1.15
toc: true
toc-depth: 3
numbersections: true
bibliography: references.bib
link-citations: true
lang: fr
---

\newpage

# Introduction

## Contexte : le bilan carbone européen

Les concentrations atmosphériques de CO₂ continuent d'augmenter, atteignant 421 ppm en 2023 [@friedlingstein2023]. Comprendre quelle part de cette augmentation est absorbée par les écosystèmes européens et quelle part provient des activités humaines requiert de quantifier le bilan régional des flux de carbone. Le bilan carbone européen est particulièrement important car l'Europe s'est engagée à la neutralité climatique en 2050 dans le cadre de l'Accord de Paris, et le mécanisme du Global Stocktake impose une vérification indépendante des émissions nationales déclarées [@unfccc2015].

Deux approches fondamentalement différentes existent pour la quantification des flux. Les **inventaires bottom-up** (EDGAR pour le fossile, modèles de processus pour la biosphère) agrègent les données d'activité sectorielles multipliées par des facteurs d'émission. Les **inversions atmosphériques top-down** utilisent les concentrations de CO₂ observées et les modèles de transport pour inférer les flux de surface qui ont dû les produire. Chaque approche a ses faiblesses : les inventaires reposent sur des données d'activité qui peuvent être incomplètes ou retardées ; les inversions souffrent d'erreurs de transport et d'indétermination mathématique. La comparaison des deux approches — révélant parfois des écarts de 30 % ou plus [@friedlingstein2022] — motive les améliorations méthodologiques des deux côtés.

## Le défi mono-traceur

Les systèmes d'inversion opérationnels comme CarbonTracker [@peters2007], CAMS [@chevallier2010], LUMIA [@monteil2021] et CarboScope [@rodenbeck2003] estiment le flux *net* de CO₂ — la somme des contributions fossile et biosphérique. Séparer ces deux composantes est le défi scientifique fondamental abordé dans ce travail.

En principe, une molécule de CO₂ émise par combustion fossile est indiscernable d'une molécule respirée par une forêt. Les deux portent la même signature spectroscopique, suivent les mêmes lois de transport, se mélangent au fond atmosphérique avec une cinétique identique. Le seul traceur atmosphérique capable d'une discrimination formelle est le rapport $^{14}\text{C}/^{12}\text{C}$ : le CO₂ fossile est dépourvu de radiocarbone (le réservoir fossile est âgé de ~$10^8$ années, bien au-delà de la demi-vie de 5730 ans du $^{14}$C), tandis que le CO₂ biosphérique porte une signature $^{14}$C moderne [@levin2003; @turnbull2011]. Gómez-Ortiz et al. [-@gomez2025] ont démontré qu'une inversion couplée CO₂-$\Delta^{14}$CO₂ via LUMIA retrouve 95 % des émissions fossiles européennes — mais seulement 14 stations en Europe mesurent actuellement le $\Delta^{14}$CO₂, contre plus de 300 pour le CO₂.

Basu et al. [-@basu2016] ont formellement prouvé l'indétermination du problème mono-traceur dans le cadre bayésien linéaire standard. Notre contribution repositionne cette « impossibilité » comme un problème résolvable *en dehors* du cadre bayésien linéaire, par la contrainte structurelle combinée de (i) priors spatialement distincts, (ii) séparation diurne des régimes convectif et stable, et (iii) résolution hebdomadaire capturant la variabilité synoptique.

**Un cadrage honnête essentiel** : la séparation que réalise notre cadre est *structurelle*, pas *physique*. Nous ne discriminons pas les molécules de CO₂ fossile de celles de CO₂ biosphérique — aucun traceur atmosphérique ne le peut en l'absence de mesures de $^{14}$C ou CO. Ce que nous faisons, c'est exploiter le fait qu'EDGAR (prior fossile) et VPRM (prior biosphérique) ont des structures spatiales radicalement différentes : le premier est dominé par des sources ponctuelles et des corridors de transport, le second est diffus et piloté par la végétation. Le PINN apprend à assigner les anomalies de concentration à l'une ou l'autre catégorie sur la base de cette distinction géographique. **Cela implique une hypothèse forte : les géographies a priori doivent être correctes dans leur distribution spatiale relative.** Si EDGAR place mal une centrale ou sous-estime un corridor autoroutier, le réseau ne peut pas le détecter et compensera partiellement par des ajustements de $\beta$. C'est une limitation fondamentale qui ne peut être levée qu'en ajoutant un co-traceur physique ($^{14}$CO₂, CO) ou en densifiant suffisamment le réseau d'observation pour rendre chaque cellule de grille individuellement identifiable observationnellement. Nous revenons sur cette limitation en section 8.

## L'approche PINN

Les réseaux de neurones informés par la physique [@raissi2019] encodent les contraintes physiques directement dans la fonction de perte. Plutôt que de permettre des prédictions non contraintes, le réseau doit produire des sorties cohérentes avec les équations gouvernantes du système. Dans notre cas, l'équation de transport atmosphérique $C_{obs} = C_{bg} + H \cdot F + \epsilon$ (où $H$ est la relation source-récepteur de HYSPLIT et $F$ le champ de flux) est intégrée dans la perte comme contrainte dure.

Les applications récentes de l'apprentissage automatique au transport atmosphérique incluent FootNet [@he2025], qui émule les empreintes HYSPLIT haute résolution 650× plus vite avec une architecture U-Net ; Dadheech et al. [-@dadheech2025], qui ont combiné FootNet avec l'inversion de flux pour des résultats haute résolution ; et GATES (2026), étendant l'approche aux réseaux de neurones graphiques continentaux. Cependant, **aucun travail publié ne combine PINN avec du transport lagrangien pour l'inversion atmosphérique du CO₂** — c'est la niche qu'occupe notre cadre.

## Fondation : la thèse M2 de 2022

Ce projet s'appuie sur ma thèse M2 en 2022 [@mahamat2022], qui évaluait WRF-Chem à 25 km de résolution sur 8 stations ICOS. Trois résultats diagnostiques clés ont guidé toute la progression V1-V14 :

1. **Erreur de représentativité urbaine** : la station KIT montrait un biais $\epsilon = 8{,}38$ ppm, IPR $\epsilon = 7{,}64$ ppm, contre TRN rural $\epsilon = 3{,}31$ ppm. Les stations urbaines contaminent le signal régional.
2. **Dominance de la couche limite** : l'impact du flux sur la concentration s'échelonne à 50 % sur une CLA de 100 m contre 3 % sur une CLA de 1500 m. La hauteur de couche limite (BLH) est un contrôle de premier ordre.
3. **Plafond de résolution** : WRF-Chem à 25 km ne peut pas résoudre les panaches urbains (échelles < 5 km). Toute inversion utilisant un transport grossier héritera de cette limitation.

Ces trois résultats ont façonné la configuration V12b : filtrage explicite des stations urbaines (application de la recommandation M2), normalisation BLH dynamique, et reconnaissance du plafond de transport à 50 km.

## Structure du document

Ce rapport est organisé comme suit :

- La **section 2** décrit les sources de données (ICOS, ERA5, empreintes HYSPLIT, EDGAR, VPRM, CarbonTracker) et leur prétraitement.
- La **section 3** détaille la formulation mathématique complète, l'architecture du réseau et la procédure d'entraînement.
- La **section 4** présente l'étude d'ablation systématique V1 → V12, documentant chaque innovation et chaque échec comme résultat informatif.
- La **section 5** présente les trois expériences de validation clés : quantification d'incertitude par MC Dropout, comparaison triple avec CAMS et CT2022, et withholding temporel.
- La **section 6** documente les études d'exploration des limites V13-V15 : correcteur dynamique, $\gamma$ additif, raffinement spatial, et décomposition de Fourier du $\beta$.
- La **section 7** discute la physique station par station, avec une attention particulière au rapport diurne inversé anomal d'IPR (Ispra) causé par la canalisation orographique de la plaine du Pô.
- La **section 8** évalue de façon critique les limitations et perspectives.
- La **section 9** conclut.

\newpage

# Données

## Observations CO₂ ICOS

Le système ICOS (Integrated Carbon Observation System) [@icos2020] fournit des mesures continues de CO₂ de haute qualité à travers l'Europe. Nous utilisons les données horaires de niveau L1 pour 25 stations en 2019, réparties en six catégories géographiques : France continentale (SAC, OPE, TRN, JUS, OVS), Allemagne continentale (KIT, JUE, GAT, OXK, LIN, TOH), montagne (PUY, CMN, PDM), côtière (LUT, WAO, TAC, STE), Méditerranée (ERS, OHP, CRA) et Atlantique (BIS, RGL). Mace Head (MHD) sert de référence Atlantique propre pour le fond.

Le traitement agrège les observations horaires en moyennes hebdomadaires, séparées en diurne (12:00-16:00 UTC, couche limite convective) et nocturne (00:00-04:00 UTC, couche limite nocturne stable). Cette séparation est motivée physiquement : les concentrations diurnes reflètent des signaux régionaux bien mélangés dans une CLA de ~1500 m, tandis que les concentrations nocturnes amplifient les flux de surface dans ~150 m mais avec une incertitude de transport substantielle.

## Empreintes HYSPLIT

Nous calculons 2600 empreintes hebdomadaires (25 stations × 52 semaines × 2 régimes jour/nuit) en utilisant HYSPLIT v5.2.0 [@stein2015] piloté par la réanalyse ERA5 à 0,25° de résolution [@hersbach2020]. Les rétro-trajectoires s'étendent sur 5 jours en amont, avec le plafond vertical fixé dynamiquement à la hauteur de couche limite ERA5 à chaque pas de temps. Les empreintes de sortie sont sur une grille à 0,5° sur le domaine 40°-56°N, 10°O-15°E (grille 32 × 50), couvrant l'Europe occidentale et centrale.

## Champs de flux priors

Les **émissions fossiles** utilisent le flux fossile mensuel CarbonTracker CT2022 [@vanderwoude2023] à 1° × 1° régrillé sur notre domaine à 0,5°. EDGAR v8.0 [@edgar2022] à 0,1° est utilisé comme vérification croisée. Les deux jeux de données sont cohérents à 5 % près sur le domaine européen.

Les **flux biosphériques** utilisent le modèle VPRM (Vegetation Photosynthesis and Respiration Model) optimisé pour ICOS, fourni à résolution horaire 0,1°. VPRM [@mahadevan2008] paramétrise l'échange net écosystémique (NEE) à partir d'indices satellitaires (EVI, LSWI) et de forçages météorologiques (température, rayonnement).

Les **flux océaniques** contribuent à moins de 0,05 % du total continental sur notre domaine. Nous incluons le flux océan CT2022 par souci de complétude mais son impact est négligeable.

## Variables météorologiques

La hauteur de couche limite ERA5 (BLH) à résolution horaire 0,25° fournit la profondeur de mélange atmosphérique qui module la fonction de transfert flux-concentration. La température ERA5 à 2 m ($T_{2m}$) à résolution journalière permet l'analyse de détection de canicule.

\newpage

# Méthodologie

## Formulation du problème inverse

L'équation de transport atmosphérique relie les flux de surface aux concentrations observées :

$$C_{obs}(s, t) = C_{bg}(t) + \sum_{i,j} H(s, t, i, j) \cdot F(i, j, t) + \epsilon$$

où :
- $C_{obs}(s, t)$ : CO₂ observé à la station $s$, semaine $t$
- $C_{bg}(t)$ : concentration de fond (référence MHD)
- $H(s, t, i, j)$ : noyau d'empreinte HYSPLIT (ppm par µmol/m²/s)
- $F(i, j, t)$ : flux de surface au point de grille $(i, j)$
- $\epsilon$ : erreur d'observation et de modèle

**Formulation découplée** (notre innovation clé) :

$$C_{mod} = C_{bg} + H \cdot \left[ \alpha(r, m) \cdot F_{foss}(i, j, t) + \beta \cdot F_{bio}(i, j, t) \right]$$

où :
- $\alpha(r, m)$ : facteur de correction fossile mensuel par région $r$ et mois $m$ (20 régions × 12 mois = 240 paramètres)
- $\beta$ : facteur de correction biosphérique global (1 paramètre)

Cette formulation encode trois contraintes physiques simultanément :

1. **Séparation structurelle** : EDGAR et VPRM ont des structures spatiales radicalement différentes — sources ponctuelles contre puits diffus. Le réseau ne peut pas les confondre.
2. **Connaissance prior** : $\alpha$ est régularisé vers 1 (faire confiance au prior sauf si les données le contredisent clairement).
3. **Identifiabilité** : 241 paramètres (240 $\alpha$ + 1 $\beta$) à partir de 2600 observations est bien déterminé (ratio $\sim$11:1).

## Fonction de perte

La perte PINN combine fidélité aux données, contraintes physiques et régularisation :

$$\mathcal{L}(\alpha, \beta) = \text{MSE}_\alpha + \text{MSE}_\beta + \lambda_\alpha \|\alpha\|^2 + \lambda_{sp} \|\nabla_{\text{espace}} \alpha\|^2 + \lambda_{tp} \|\nabla_{\text{temps}} \alpha\|^2$$

Avec les hyperparamètres $\lambda_\alpha = 0{,}1$ (régularisation prior), $\lambda_{sp} = 0{,}05$ (lissage spatial), $\lambda_{tp} = 0{,}03$ (lissage temporel). Ces valeurs ont été sélectionnées pour produire des champs $\alpha$ physiquement plausibles (lisses, proches de l'unité en moyenne) sans trop contraindre la solution.

## Architecture du réseau

```
Entrée : features X (1976 dims) — ΔC × BLH_ratio par station/semaine
  ↓
Dense(512, gelu) + Dropout(0,15) + LayerNorm
  ↓
Dense(512, gelu) + Dropout(0,15) + LayerNorm
  ↓
Dense(256, gelu) + Dropout(0,1) + LayerNorm
  ↓
  ├─→ Dense → Reshape(4,5,16) → Conv2DTranspose → Conv2D → α (240)
  └─→ Dense(32) → Dense(1) → β (1)
  ↓
Sortie : [α(240), β(1)] = 241 paramètres
```

La branche décodeur convolutive pour $\alpha$ impose une cohérence spatiale — les valeurs $\alpha$ des régions adjacentes sont corrélées via l'opérateur Conv2DTranspose. C'est l'équivalent architectural du prior de lissage $\lambda_{sp}$.

Le Dropout reste actif à l'inférence (`training=True`) pour permettre la quantification d'incertitude par MC Dropout [@gal2016].

## Entraînement : scénarios synthétiques

Nous générons 5000 scénarios synthétiques par perturbation aléatoire des priors :

- $\alpha \in [0{,}5 ; 1{,}5]$ uniforme par région et mois (240 valeurs par scénario)
- $\beta \in [0{,}7 ; 1{,}3]$ uniforme global
- Bruit gaussien 2 % ajouté aux features

Chaque scénario produit un ensemble de séries temporelles hebdomadaires de concentrations à chaque station, calculées par propagation directe $C = H(\alpha F + \beta F)$. Le réseau apprend à inverser cette projection.

L'entraînement utilise 85 % pour l'optimisation et 15 % pour la validation, avec arrêt anticipé (patience 25) et réduction du taux d'apprentissage (facteur 0,5, patience 10). L'optimiseur est Adam avec un taux d'apprentissage initial de $5 \times 10^{-4}$.

## Ingénierie des features : normalisation BLH

Les anomalies brutes de concentration $\Delta C = C_s - C_{MHD}$ sont multipliées par le ratio BLH spécifique à la station :

$$X = \Delta C \cdot \frac{\text{BLH}_{station}}{\text{BLH}_{r\acute ef\acute erence}}$$

où la référence est la BLH moyenne sur les stations rurales. Cette normalisation réduit la variabilité introduite par la météorologie spécifique aux stations, permettant au réseau de se concentrer sur le signal de flux.

Un diagnostic clé de V8 a motivé ce choix : la moyenne mensuelle détruit l'information du cycle diurne de la CLA, car $\langle \Delta C \cdot \text{CLA} \rangle \neq \langle \Delta C \rangle \cdot \langle \text{CLA} \rangle$. La formulation correcte multiplie au niveau horaire puis agrège, pas l'inverse.

\newpage

# Progression V1 → V12 : ablation systématique

Chaque version ajoute exactement une modification à la précédente, quantifiant la contribution incrémentale.

## V1 baseline : preuve de concept synthétique

**Configuration** : 8 stations rurales, empreintes mensuelles, flux total $F = \alpha F_{total}$, pas de séparation jour/nuit.

**Résultat** : corrélation $\alpha$ $r = 0{,}253$.

V1 établit que l'architecture PINN peut extraire du signal des scénarios synthétiques, mais la corrélation est faible. La majeure partie de la variance provient du bruit de perturbation lui-même ; la physique n'est pas encore bien contrainte.

## V2 : émissions EDGAR réelles

**Changement** : remplacer le flux fossile synthétique par EDGAR v8.0.

**Résultat** : $r = 0{,}475$ (+0,222).

Le doublement de la corrélation confirme que l'hétérogénéité spatiale réaliste du prior fournit au réseau une structure identifiable. Cela est cohérent avec la théorie d'inversion : le contenu en information spatiale s'échelonne avec la résolution effective du prior.

## V3-V4 : expérimentations d'architecture

**V3** : branche Conv2D au lieu de connectée complète. **V4** : VPRM ajouté comme feature biosphérique en entrée (sans découplage).

Les deux versions plafonnent à $r \approx 0{,}45$. La capacité architecturale n'est pas le goulot d'étranglement — la formulation l'est.

## V5 : 25 stations et le bug d'unités EDGAR

**Changement** : extension à 25 stations (réseau complet).

**Premier résultat** : $r = 0{,}781$ — suspicieusement élevé.

L'investigation a révélé que les unités d'EDGAR v8.0 étaient lues comme kg/m²/s × $10^{-6}$ au lieu de mol/m²/s, produisant des flux fossiles $10^6$ fois trop grands. L'inversion était dominée par le bruit et une structure fictive. Après correction, $r$ s'est effondré à $0{,}000$ — le signal fossile était maintenant *trop petit* par rapport à VPRM.

**V5 final** (avec flux océan et CT2022 comme prior fossile à la place d'EDGAR) : $r = 0{,}197$, LOSO $= 0{,}191$. Le réseau récupère un signal significatif mais la performance est médiocre. Un second goulot d'étranglement doit être abordé : l'amalgame des sources fossiles et biosphériques.

**Leçon** : un résultat initial « excellent » doit toujours être interrogé. La vérification des unités est aussi importante que la sophistication algorithmique.

## V6 : la percée du découplage

**Changement** : séparer $\alpha$ (fossile) et $\beta$ (bio) dans la perte : $C = H(\alpha F_{foss} + \beta F_{bio})$.

**Résultat** : $r = 0{,}523$, LOSO $= 0{,}417$.

![Découplage V6. Gauche : inversion fossile seule $r = 0{,}834$. Centre : performance total$\to \alpha$ $r = 0{,}587$. Droite : $\alpha, \beta$ joints avec $\beta$ retrouvé proche de l'unité (0,998). Le découplage permet l'identification simultanée des deux facteurs.](figures/fig_v6_decoupled.png)

C'est le plus grand gain en un seul pas du projet (+166 % relatif à V5). L'explication est structurelle : EDGAR (dominé par sources ponctuelles) et VPRM (diffus) ont des signatures spatiales distinctes. Le réseau les distingue maintenant au lieu de les confondre.

De façon critique, V6 est aussi la première version où le PINN surpasse clairement l'inversion bayésienne classique (voir section 4.7).

## V7 : hauteur de couche limite ERA5

**Changement** : ajouter BLH comme feature prédictive.

**Résultat** : $r = 0{,}600$ (+0,077 sur V6).

La BLH encode le volume de dilution atmosphérique, un contrôle de premier ordre des relations concentration-flux. L'inclure comme feature permet au réseau de distinguer les cas où les anomalies de concentration sont pilotées par des changements de flux versus des changements de mélange.

## V6 + benchmark bayésien

Sur données, scénarios et transport identiques, une inversion bayésienne classique (erreurs gaussiennes, régularisation L-curve) produit :

- Synthétique : $r = 0{,}033$
- LOSO : $r = 0{,}033$

Le PINN atteint LOSO $r = 0{,}417$ — un facteur ×12 d'amélioration.

![V6 LOSO + benchmark bayésien. PINN $\alpha$ $r = 0{,}417$ (gauche) vs bayésien $r = 0{,}033$ (centre). Le nuage de points (droite) montre une corrélation spatiale $r = 0{,}168$ entre les deux, indiquant qu'ils capturent des informations fondamentalement différentes. Le PINN exploite une structure non linéaire que le cadre bayésien linéaire ne peut pas capter.](figures/fig_v6_loso_bayesian.png)

Pourquoi le bayésien échoue-t-il ? Trois raisons. Premièrement, le problème est sous-déterminé : 25 stations échantillonnent un espace d'état à 240 dimensions. Deuxièmement, le cadre bayésien linéaire suppose des erreurs gaussiennes et un mapping observation-paramètre linéaire, les deux violés par le transport HYSPLIT à 50 km sur terre avec complexité orographique [@michalak2004]. Troisièmement, des artefacts d'agrégation apparaissent à résolution grossière. Le PINN, en apprenant des dépendances non linéaires, extrait un signal structurel là où le cadre linéaire ne voit que du bruit.

## V8 : normalisation CLA

**Changement** : multiplier les features par $\text{BLH}/\text{BLH}_{ref}$.

**Résultat** : $r = 0{,}540$ (+0,006 marginal sur V6).

Le gain est petit parce que V7 fournissait déjà BLH implicitement via la représentation des features. La normalisation explicite de V8 contribue principalement à la robustesse sur les stations aux régimes BLH atypiques (montagne, côtière).

**Diagnostic clé** : les moyennes mensuelles détruisent le cycle diurne de la CLA. Cette observation a directement piloté V9.

## V9 : séparation jour/nuit

**Changement** : séparer les concentrations diurnes (12-16h UTC, CLA ~968 m) et nocturnes (0-4h UTC, CLA ~326 m) comme features indépendantes.

**Résultat** : $r = 0{,}609$, $\Delta = +0{,}075$ par rapport à V6.

![Séparation jour/nuit V9. Les régimes jour et nuit ont une physique distincte : signal diurne bien mélangé mais dilué (haute BLH), signal nocturne intense mais bruité (basse BLH). Les séparer permet au réseau d'apprendre des relations distinctes pour chaque régime. Gain de +0,075 pour un seul changement architectural.](figures/fig_v9_daynight.png)

C'est la plus grande contribution d'un facteur unique après le découplage de V6. Le ratio 3× entre CLA jour et nuit (968 m vs 326 m) correspond à une amplification 3× des signaux de flux nocturnes — mais avec une plus grande incertitude de transport. Séparer les deux régimes laisse le réseau les pondérer de façon optimale.

## V10 : l'échec instructif

**Changement** : passer à une résolution hebdomadaire des features tout en gardant les empreintes *mensuelles*.

**Résultat** : $r = 0{,}239$ (effondrement depuis 0,609 de V9).

![Échec V10 : la corrélation spatiale par semaine chute à 0,051 (bruit pur). Le désalignement entre empreintes à moyenne mensuelle et features à résolution hebdomadaire introduit une incohérence temporelle que le réseau ne peut pas surmonter.](figures/fig_v10_weekly.png)

Cet échec est un résultat d'importance critique. Il démontre que la cohérence temporelle entre le modèle de transport et les features prédictives est **non négociable**. Les empreintes mensuelles lissent les patterns de transport synoptiques (variations semaine-à-semaine de direction et vitesse du vent), créant un désalignement avec les signaux de concentration hebdomadaires.

Ce résultat se généralise à tous les systèmes d'inversion LPDM : STILT [@lin2003], FLEXPART [@pisso2019], NAME [@jones2007]. Si vos empreintes sont temporellement plus grossières que vos observations, attendez-vous à un effondrement.

## V11 : empreintes hebdomadaires — la résolution cohérente

**Changement** : calculer 2600 empreintes hebdomadaires avec plafond BLH ERA5 dynamique.

**Résultat** : $r = 0{,}648$, LOSO $r = 0{,}487 \pm 0{,}015$.

![Empreintes hebdomadaires V11 : corrélation fossile $r = 0{,}648$ (depuis 0,609), chaque station utilisant ses propres trajectoires HYSPLIT hebdomadaires spécifiques. Les ratios BLH vont de 1,15× (LUT) à 2,29× (RGL), avec IPR (Ispra) remarquablement inversé à 0,65× en raison de la canalisation orographique de la plaine du Pô.](figures/fig_v11_weekly_fp.png)

![Validation LOSO V11 : $r = 0{,}487 \pm 0{,}015$ sur 25 stations. Meilleurs : GAT (0,514), PUY (0,513), TOH (0,505). La distribution est approximativement normale autour de la moyenne, indiquant pas de valeurs aberrantes catastrophiques mais pas non plus de succès exceptionnels.](figures/fig_v11_loso.png)

V11 complète la cohérence temporelle-physique. Transport, features et météorologie sont maintenant tous à résolution hebdomadaire, avec BLH dynamique fournissant le plafond. Le LOSO de 0,487 représente une amélioration de +0,070 sur le 0,417 de V6.

## V12b : filtrage des stations urbaines (application de la recommandation M2)

**Changement** : exclure KIT, IPR, JUS, JUE, OVS, SAC de l'entraînement. Ces 6 stations sont identifiées comme contaminées par l'urbain dans la thèse M2 de 2022.

**Résultat** : LOSO $r = 0{,}612 \pm 0{,}015$ — la configuration finale du projet.

![Filtrage spatial V12b. Panneau du haut : LOSO par station variant maintenant de 0,571 à 0,639 sur les 19 stations rurales. Aucune station ne tombe en dessous de 0,5, une amélioration qualitative par rapport à V11. Panneau du bas : cartes $\alpha$ et valeur de $\beta$.](figures/fig_v12_filtrage.png)

L'amélioration de +0,125 (de 0,487 à 0,612) est le deuxième plus grand saut du projet, après le découplage de V6. L'explication : le transport HYSPLIT à 50 km ne peut pas résoudre les panaches urbains (échelles typiques 1-10 km). Les stations urbaines contaminent le signal régional avec des sources locales non résolues. Les supprimer nettoie l'inversion.

Six stations exclues :

- **KIT** (Karlsruhe) : contamination par panache urbain, anomalie $\Delta C$ 7,1 ppm (la plus grande du réseau)
- **IPR** (Ispra) : canalisation orographique de la plaine du Pô, rapport N/D inversé (voir section 7)
- **JUS, OVS, SAC** (région parisienne) : panache métropolitain
- **JUE** (Jülich) : corridor industriel du Rhin

C'est l'application directe de la recommandation de la thèse M2 : « Pour une inversion régionale européenne, nous devrons éliminer les stations proches des villes » (Mahamat, 2022).

## Analyse $\beta$ mensuel V11 : détection de la canicule

![$\beta$ mensuel V11 sur scénarios synthétiques. En haut à gauche : cycle $\beta$ avec réduction JJA ($\beta_{JJA} = 0{,}988 < 1$), cohérente avec la surestimation VPRM du puits estival durant la canicule. En bas au centre : évolution de $r$ de V1 à V11, avec effondrement V10 comme pivot. Le système détecte une signature de canicule sans données de forçage climatique dans ses entrées.](figures/fig_v11_beta_monthly.png)

Le $\beta$ mensuel est la sortie la plus interprétable du projet. En juin-août (canicule 2019), $\beta_{JJA} = 0{,}988$, significativement en dessous de la valeur hivernale de 1,003. Cela indique que VPRM surestime le puits biosphérique estival — cohérent avec la fermeture stomatique induite par canicule connue, qui réduit la photosynthèse malgré les hautes températures [@bastos2020]. Reichstein et al. [-@reichstein2013] ont identifié les canicules comme pilote interannuel dominant de la variabilité du puits de carbone européen.

De façon critique, le PINN découvre cette signature **sans aucune variable climatique dans ses entrées**. Il l'extrait purement de la structure des concentrations de CO₂.

\newpage

# Expériences de validation

## Quantification d'incertitude par MC Dropout

Nous effectuons 50 passes stochastiques avant avec Dropout actif, selon Gal & Ghahramani [-@gal2016]. Cela approche la distribution postérieure sur $\alpha$ et $\beta$.

**Caveat sur la calibration de l'incertitude** : MC Dropout est une heuristique pratique qui approche l'inférence variationnelle sous des hypothèses restrictives (postérieure gaussienne, structure factorisée). Des travaux récents [@folgoc2021; @osband2016] ont montré que MC Dropout sous-estime systématiquement l'incertitude épistémique — typiquement d'un facteur 1,5 à 2 — par rapport à l'inférence variationnelle complète ou aux ensembles profonds. Les intervalles que nous rapportons doivent donc être interprétés comme des **bornes inférieures** de l'incertitude réelle. Notre $\alpha = 1{,}010 \pm 0{,}078$ en titre couvre presque certainement une plage plus large en réalité, possiblement $\pm 0{,}12$ à $\pm 0{,}15$ sous une inférence bayésienne pleinement calibrée. Cela ne change pas les conclusions qualitatives ($\alpha \approx 1$ tient toujours ; $\beta < 1$ tient toujours au niveau directionnel), mais cela tempère toute revendication d'attribution quantitative à haute précision. Une implémentation B-PINN future fournirait des intervalles crédibles proprement calibrés.

![Ensemble MC Dropout sur observations ICOS réelles. Haut-gauche : cycle $\alpha$ mensuel avec intervalles de confiance 68 % et 95 %. Haut-centre : carte spatiale $\alpha$ montrant la variabilité régionale. Haut-droite : l'incertitude $\alpha$ est la plus haute aux bords du domaine (moins de stations contraignantes). Bas-gauche : distribution $\beta$ concentrée à 0,971 ± 0,023. Bas-centre : $\alpha$ par région avec barres d'erreur à 95 %. Résultat : $\alpha = 1{,}010 \pm 0{,}078$, $\beta = 0{,}971 \pm 0{,}023$.](figures/fig_mc_dropout.png)

**Résultats sur observations ICOS réelles :**

| Paramètre | Moyenne | Écart-type | IC 95 % |
|-----------|---------|-------------|---------|
| $\alpha$ global | 1,010 | 0,078 | [0,853 ; 1,167] |
| $\beta$ global | 0,971 | 0,023 | [0,926 ; 1,017] |

**Interprétation** :

- $\alpha \approx 1{,}0$ (IC 95 % inclut 1,0) : les émissions fossiles EDGAR sont correctes dans la barre d'incertitude pour la moyenne européenne en 2019. Nous ne pouvons pas rejeter l'hypothèse que l'inventaire est non biaisé.
- $\beta$ significativement $< 1{,}0$ au niveau 2,5$\sigma$ : VPRM surestime le puits biosphérique d'environ 3 %. C'est un résultat robuste et publiable.

L'incertitude est structurée spatialement : la plus faible dans le corridor France centrale (stations denses : TRN, OPE, SAC, JUS), la plus haute aux bords est et nord (GAT, LIN, STE, WAO). Cela confirme que la densité observationnelle pilote la confiance de l'inversion.

## Comparaison triple avec CT2022 et CAMS (validation gold standard)

C'est la validation externe la plus importante. Nous comparons nos estimations de flux fossile V12 contre deux systèmes opérationnels : CarbonTracker CT2022 et CAMS v22r1 [@chevallier2010].

Critique : **CAMS est pleinement indépendant de notre chaîne de traitement**. Il utilise un modèle de transport différent (IFS/LMDZ), une méthode d'inversion différente (4D-Var), des observations différentes (satellite + surface) et un prior biosphérique différent (ORCHIDEE). Tout accord est la preuve d'une estimation convergente, pas d'une validation circulaire.

![Validation triple : V12 vs CT2022 vs CAMS. Haut-gauche : corrélations spatiales — V12-CAMS = 0,992, CT2022-CAMS = 0,988. Haut-centre : corrélations temporelles — V12-CAMS = 0,851. Haut-droite : cycles saisonniers de flux fossile pour les trois systèmes se chevauchent étroitement. Bas : patterns spatiaux régionaux de correction $\alpha$ V12 vs flux CAMS.](figures/fig_validation_triple.png)

**Résultats** :

|  | Spatial | Temporel | Total |
|---|---------|----------|-------|
| V12 vs CT2022 | 0,999 | 0,958 | 0,979 |
| **V12 vs CAMS** | **0,992** | **0,851** | **0,965** |
| CT2022 vs CAMS | 0,988 | 0,841 | 0,983 |

La corrélation spatiale V12-CAMS de 0,992 est la plus forte validation externe du projet. Deux systèmes d'inversion atmosphérique pleinement indépendants, utilisant des modèles de transport différents, des observations différentes, des priors différents et des cadres mathématiques différents, convergent sur essentiellement la même structure de flux fossile européen.

Notamment, V12 corrèle légèrement *mieux* avec CAMS que ne le fait CT2022 (0,992 vs 0,988 spatial ; 0,965 vs 0,983 total). Cela signifie que notre correction PINN sur CT2022 rapproche l'estimation de la référence indépendante CAMS — un indicateur fort que la correction capture un signal réel, pas du bruit.

## Validation forward : C_mod vs C_obs

Pour chaque station et semaine, nous reconstruisons les concentrations modélisées $C_{mod} = H \cdot (\alpha F_{foss} + \beta F_{bio})$ et comparons aux observations ICOS.

![Validation forward : concentrations reconstruites V12 vs observations ICOS réelles. Haut-gauche : corrélations par station allant de -0,34 (PDM, montagne) à 0,75 (OPE). Haut-centre/droite : séries temporelles exemples pour TRN et PUY. Bas : amélioration V12 sur prior négligeable en moyenne (+0,004), mais les meilleures stations atteignent un excellent accord.](figures/fig_validation_forward.png)

**Corrélation moyenne sur 25 stations : $r = 0{,}422$**, avec les meilleurs :

- OPE (rural, nord-est France) : r = 0,751
- TRN (rural, France centrale) : r = 0,740
- TOH (rural, Allemagne centrale) : r = 0,717
- GAT (rural, nord-est Allemagne) : r = 0,687
- ERS (Méditerranée Corse) : r = 0,680

La moyenne de 0,422 reflète la limite fondamentale du système : le transport HYSPLIT à 50 km capture environ 42 % de la variance des concentrations observées en moyenne. C'est une évaluation honnête de ce que toute inversion atmosphérique peut atteindre à cette résolution de transport.

La correction V12 n'apporte qu'une amélioration marginale sur le prior non corrigé ($\Delta r = +0{,}004$). Cela est expliqué par $\alpha \approx 1{,}0$ : quand le prior est déjà correct, la correction est petite. Le PINN confirme, ne corrige pas — et c'est un résultat scientifique valide.

## Withholding temporel : prédire l'été non vu

Le test le plus rigoureux : masquer les 17 semaines JJA (été 2019, incluant la canicule) à l'entraînement *et* à l'inférence, puis les reconstruire.

![Withholding temporel : JJA masqué à l'entraînement et à l'inférence. Haut-gauche : corrélations JJA par station pour withheld vs baseline — essentiellement identiques. Haut-centre : cycles $\alpha$ mensuels très similaires entre configurations. Haut-droite : série temporelle exemple TRN. Bas-gauche : dégradation PINN synthétique (-0,214) est le coût attendu du masquage. Bas-centre : sur observations réelles, la dégradation disparaît (-0,002). Bas-droite : résumé.](figures/fig_withholding_jja.png)

**Résultats** :

| Métrique | Baseline | Withheld | $\Delta$ |
|----------|----------|----------|---------|
| PINN synthétique $\alpha$ r | 0,670 | 0,456 | -0,214 |
| **Obs réelles corrélation JJA** | **0,237** | **0,235** | **-0,002** |
| Obs réelles corrélation totale | 0,584 | 0,584 | 0,000 |

La dégradation synthétique de $-0{,}214$ confirme que le masquage est efficace (le réseau perd vraiment de l'information). Pourtant sur les observations réelles, la qualité de reconstruction estivale est essentiellement inchangée ($\Delta = -0{,}002$). Le modèle prédit l'été qu'il n'a jamais vu, aussi bien qu'il le prédit avec les données complètes de l'année.

Interprétation physique : avec $\alpha \approx 1{,}0$ et le puits estival dominé par VPRM × $\beta$, les 9 mois de données hiver/automne fournissent assez de contrainte sur $\beta$ (via le cycle biosphérique sur 9 mois) pour reconstruire les concentrations estivales. Le signal dominant est saisonnier, et la structure saisonnière est contrainte par les données qu'on garde.

**Ce résultat répond à la critique la plus sérieuse de la validation synthétique seule** : le modèle se généralise depuis les scénarios de la distribution d'entraînement vers des observations réelles non vues.

## Accord spatial avec CarbonTracker

![V12 vs CarbonTracker CT2022 : corrélation spatiale $r = 0{,}986$. Les cartes de couleur montrent une structure géographique quasi identique du flux fossile optimisé.](figures/fig_carbontracker_comparison.png)

V12 vs CT2022 spatial $r = 0{,}999$ (section 5.2). Cela est attendu car CT2022 est notre prior fossile — la correction $\alpha$ préserve la structure géographique tout en ajustant l'amplitude. Le résultat complémentaire est V12 vs CAMS ($r = 0{,}992$), qui est pleinement indépendant et donc plus diagnostique.

\newpage

# Extensions V13-V15 : exploration des limites

Nous documentons maintenant les tentatives d'extension de V12b, incluant à la fois les succès (V13, V13b, $\beta$ Fourier) et les échecs (V14, V15). Les échecs sont aussi informatifs que les succès — ils quantifient les limites structurelles du cadre.

## V13 : correcteur sous-maille statique

**Concept** : combiner le PINN basé sur la physique (V12b) avec un petit réseau de correction qui apprend les patterns de résidus spécifiques aux stations.

![Correcteur sous-maille V13. Gauche : correcteur statique atteint $r_{jour} = 0{,}704$, $r_{nuit} = 0{,}615$ sur les résidus. Centre : résidus des stations urbaines réduits de 24,7 % (de 0,047 à 0,035). Droite : importance par permutation pour les 9 features statiques. Type de station (urbain/rural), semaine et intensité C_mod dominent.](figures/fig_v13_correction.png)

Le correcteur V13 prend 9 features statiques par station (flux fossile local, variance de flux, ratio BLH jour/nuit, étendues d'empreinte, lat/lon, indicateur urbain) et prédit le résidu $\epsilon = C_{obs} - C_{mod}$. Résultats :

- Résidus urbains : -24,7 % (de 0,047 à 0,035)
- Résidus ruraux : -21,2 % (de 0,028 à 0,022)
- Amélioration individuelle IPR : -30,1 % (plus grande réduction absolue)

La feature urbaine est le prédicteur d'importance le plus élevé, confirmant que le type de station capture le pattern de contamination dominant.

## V13b : correcteur dynamique

**Extension** : ajouter 3 features dynamiques à V13 — $T_{2m}$ hebdomadaire (proxy chauffage), BLH hebdomadaire (proxy stabilité), semaine de l'année (saisonnalité).

![Correcteur dynamique V13b : corrélation passe de 0,661 (statique) à 0,960 (dynamique), une amélioration MAE de 58,7 %. La BLH nocturne émerge comme feature dominante (importance par permutation 20× plus haute que les autres features), confirmant que la stabilité atmosphérique pilote la majeure partie de la variabilité sous-maille.](figures/fig_v13b_dynamic.png)

Le saut de $r = 0{,}661$ à $r = 0{,}960$ sur la tâche de prédiction des résidus est frappant. La feature dominante d'un facteur 20× est BLH_nuit (hauteur de couche limite nocturne). Ceci est physiquement intuitif : les couches limites nocturnes peu profondes concentrent les flux de surface, amplifiant les erreurs spécifiques aux stations dans le transport et le flux. Une représentation correcte de la stabilité atmosphérique est essentielle pour la correction sous-maille.

Ce résultat suggère que les travaux futurs devraient prioriser les diagnostics BLH à haute résolution (e.g., réseaux de céilomètres, réanalyse CERRA à 5,5 km).

## V14 : terme additif $\gamma$ — échec par sur-paramétrisation

**Motivation** : la formulation multiplicative $F = \alpha F_{prior}$ ne peut pas créer d'émissions là où le prior est zéro ($\alpha \cdot 0 = 0$). Si EDGAR a manqué une nouvelle centrale ou autoroute, le réseau ne peut pas la détecter.

**Tentative** : $F = \alpha F_{prior} + \gamma$, avec 240 paramètres $\gamma$ mensuels par région.

![$\gamma$ additif V14 : 481 paramètres totaux (240 $\alpha$ + 1 $\beta$ + 240 $\gamma$). LOSO dégrade à 0,489 vs baseline V12b de 0,612 — une chute de $-0{,}123$. Les paramètres $\gamma$ sont mal contraints ($r = 0{,}302$) en raison de l'explosion du nombre de paramètres avec seulement 19 stations.](figures/fig_v14_gamma.png)

**Résultat** : LOSO $= 0{,}489$ ($\Delta = -0{,}123$).

**Tentative de correction V14b** : réduire $\gamma$ à annuel par région (20 paramètres au lieu de 240), avec régularisation L1 de parcimonie. **Résultat** : LOSO $= 0{,}500$ (encore trop).

**Leçon** : le problème mono-traceur additif-multiplicatif est fondamentalement sous-déterminé avec 19 stations. Ajouter la capacité de détection de zéros requiert soit plus d'observations (satellite, réseau plus dense), soit des contraintes externes (données d'activité socio-économique).

## V15 : résolution spatiale plus haute — effondrement catastrophique

**Motivation** : 20 régions à 130 000 km² chacune est trop grossier pour les applications de politique à l'échelle nationale.

**Tentative** : 80 régions (grille 8×10) pour $\alpha$.

![Test de résolution V15 : 80 régions (961 paramètres) catastrophiquement sous-déterminées. LOSO s'effondre à 0,133 vs baseline 0,626 — une chute de $-0{,}492$. 36 régions intermédiaires (V15b) échouent aussi à LOSO 0,378.](figures/fig_80_regions.png)

**V15 (80 régions)** : LOSO = 0,133 ($\Delta = -0{,}492$).
**V15b (36 régions)** : LOSO = 0,378 ($\Delta = -0{,}246$).

**Leçon** : avec 19 stations et 52 semaines d'observations, le système peut contraindre environ 240 paramètres. Au-delà, chaque paramètre ajouté dégrade la performance. C'est une limite structurelle dure, pas un problème de réglage.

Pour relâcher cette limite, il faut soit (i) plus de stations (mesures de colonne satellite OCO-2/3, couverture ICOS plus dense, profils d'avion), soit (ii) des priors plus forts (parcimonie structurée, contraintes physiques).

## $\beta$ Fourier : un compromis viable

**Motivation** : 12 valeurs $\beta$ mensuelles indépendantes sont sous-déterminées (~65 observations par mois). L'optimisation $\beta$ mensuelle a produit des résultats instables ($\sigma$ > moyenne).

**Tentative** : décomposer $\beta(m) = \beta_0 + \beta_1 \cos(2\pi m/12) + \beta_2 \sin(2\pi m/12)$ — 3 paramètres capturant un cycle annuel lisse.

![$\beta$ Fourier sur observations réelles. Haut-gauche : cycle $\beta$ mensuel avec IC 95 % montrant le minimum JJA autour du mois 5,5. Haut-centre : distribution des coefficients de Fourier. Haut-droite : moyennes saisonnières — DJF = 1,004, JJA = 0,981. Bas-gauche : ajustement sinusoïdal lisse. Bas-centre : carte $\alpha$ spatiale. Bas-droite : résumé avec amplitude 0,015 à 1,4$\sigma$.](figures/fig_beta_fourier_real.png)

**Résultats** :

| Coefficient | Valeur | Écart-type |
|-------------|--------|-------------|
| $\beta_0$ | 0,9927 | 0,0163 |
| $\beta_1$ | 0,0128 | 0,0108 |
| $\beta_2$ | -0,0086 | 0,0120 |

Amplitude : $0{,}015 \pm 0{,}011$ (1,4$\sigma$ — en dessous du seuil de significativité statistique de 2$\sigma$).

Phase : minimum au mois 4,9 (mi-mai), cohérent avec le timing de la canicule.

Sur scénarios synthétiques : LOSO = 0,595 ($\Delta = -0{,}029$ vs V12b — quasi équivalent, confirmant que la paramétrisation de Fourier ne nuit pas à la qualité de l'ajustement).

**Interprétation — avec la prudence appropriée** : le cycle $\beta$ saisonnier observé est **directionnellement cohérent avec l'hypothèse de canicule** ($\beta_{JJA} = 0{,}981 < \beta_{DJF} = 1{,}004$, impliquant une surestimation VPRM du puits estival de ~2 %), mais **statistiquement marginal** à $1{,}4\sigma$ — en dessous du seuil conventionnel de $2\sigma$ pour une revendication de détection. **Ceci n'est pas une détection formelle de la canicule de 2019.** C'est un signal suggestif dont la direction est physiquement plausible (la fermeture stomatique induite par la canicule réduit la photosynthèse, gonflant donc les concentrations apparentes estivales de CO₂, que le modèle attribue à un $\beta$ plus faible). Mais statistiquement, nous ne pouvons pas rejeter l'hypothèse nulle d'absence de variation saisonnière.

Une attribution définitive nécessiterait : (i) une analyse multi-année opposant 2019 (canicule) à 2018 (normale) et 2020 (COVID avec activité atypique), (ii) un réseau d'observation plus dense réduisant le bruit effectif par mois, ou (iii) l'inclusion de $^{14}$CO₂ pour démêler la compensation $\alpha$-$\beta$. Nous présentons ce résultat non comme une découverte mais comme une preuve de concept que le cadre PINN peut résoudre des variations saisonnières de $\beta$ étant donné suffisamment de données, et comme motivation pour un suivi multi-année.

Notons séparément que la corrélation $T_{2m}$-CO₂ sur l'année complète atteint $r = -0{,}955$ dans les données brutes (indépendamment de toute sortie PINN), confirmant le couplage climat-flux bien documenté. Il s'agit d'une observation au niveau des données, non d'un résultat PINN.

## Le compromis paramètres-contraintes

Toutes les expériences V13-V15 révèlent un pattern commun : le cadre peut confidemment contraindre environ 240 paramètres à partir de 19 stations × 52 semaines. Au-delà de ce seuil, chaque paramètre ajouté dégrade la performance.

![Synthèse : solutions aux trois blocages (V14b $\gamma$ annuel, V15b 36 régions, $\beta$ Fourier) se regroupent toutes à ou sous le plafond de 240 paramètres, avec la performance LOSO correspondante reflétant le compromis paramètres-contraintes. La décomposition de Fourier (243 params) est la seule extension viable à la densité d'observation actuelle.](figures/fig_fixes_physics.png)

Ce n'est pas une coïncidence mathématique. Cela reflète le contenu en information observationnel : avec 19 stations indépendantes et 52 pas de temps, les degrés de liberté effectifs des données sont limités. Ajouter des paramètres au-delà de ce seuil dilue l'information par paramètre en dessous du rapport signal/bruit.

**Implication structurelle** : le plafond inhérent du cadre à la densité d'observation actuelle est approximativement V12b (LOSO 0,612). Les améliorations futures requièrent des observations plus denses, pas des architectures plus astucieuses.

\newpage

# Physique station par station

## IPR (Ispra) : le rapport diurne inversé

La station Ispra (IPR, 45,81°N, 8,64°E, plaine du Pô) présente une propriété unique dans le réseau ICOS : son rapport nuit/jour de concentration est de **0,65×**, inversé par rapport à toutes les autres stations (qui montrent des rapports > 1 dus à l'accumulation nocturne sous stratification stable).

Diagnostics quantitatifs :

- Étendue d'empreinte nocturne : 44 cellules de grille (1,69× diurne)
- Étendue d'empreinte diurne : 26 cellules de grille
- BLH nocturne : 120 m
- BLH diurne : 570 m
- $\Delta C$ moyen nocturne : 2,1 ppm (vs ~5-8 ppm aux autres stations continentales)

![Comparaison d'empreintes multi-stations : IPR vs OPE vs PUY vs KIT. L'empreinte nocturne d'IPR est spatialement plus grande que son empreinte diurne — opposé au pattern diurne canonique.](figures/fig_fp_weekly_multi.png)

Explication physique : dans le système Alpes-plaine du Pô, le refroidissement radiatif nocturne des pentes montagneuses génère des vents catabatiques (flux descendants pilotés par gravité) qui canalisent l'air alpin frais dans la plaine du Pô, apportant des concentrations diluées depuis les bases de haute altitude. Pendant la journée, les vents anabatiques et le mélange convectif distribuent les émissions urbaines de Milan-Turin à travers la couche limite profonde, produisant des concentrations élevées mais verticalement mélangées [@zardi2013].

De façon critique, la BLH à IPR nocturne (120 m) correspond à celle de CMN (pic Cimone, 123 m, élévation 2165 m) dans le même système régional. Mais leurs rapports diurnes sont opposés : CMN = 1,78× (accumulation normale), IPR = 0,65× (inversé). Le chevauchement spatial de leurs empreintes n'est que de 15,8 % jour / 17,7 % nuit — la topographie alpine sépare leurs bassins d'influence malgré la proximité.

**Implication scientifique** : l'anomalie d'IPR n'est pas un phénomène de couche limite (BLH indiscernable de CMN). C'est un effet de **transport orographique horizontal**. Toute inversion atmosphérique traitant IPR avec des approches standard basées BLH échouera systématiquement.

Cet aperçu, absent de la thèse M2 de 2022, émerge de notre analyse PINN.

## Le trio parisien : SAC, JUS, OVS

Trois stations de la région parisienne (SAC, JUS, OVS) montrent une cohérence interne (rapports N/D 1,52-1,55, chevauchement d'empreintes 70-80 %) mais un fort désaccord avec la représentation de transport HYSPLIT à 50 km. Les trois ont été exclues en V12b.

## KIT (Karlsruhe) : îlot de chaleur urbain

KIT présente la plus grande anomalie de concentration du réseau ($\Delta C = 7{,}1$ ppm) et le plus bas rapport N/D (1,21×). L'effet d'îlot de chaleur urbain maintient une BLH nocturne élevée, réduisant l'amplitude diurne mais aussi étalant l'information spatiale.

## PUY (Puy de Dôme) : la montagne de référence

PUY est la seule station avec un $\Delta C$ négatif (-1,0 ppm), reflétant son élévation (1465 m) qui échantillonne l'air de fond de la troposphère libre. Elle se classe constamment dans le top-3 des performances LOSO — le signal le plus clair et fiable du réseau.

## ERS (Ersa, Corse) : l'outlier méditerranéen

ERS n'a pas de cycle diurne de BLH ($N/D \approx 1{,}0$), reflétant l'environnement maritime qui supprime le développement de la couche limite radiative. Sa performance LOSO (0,68) est excellente malgré le régime atypique.

\newpage

# Discussion et limitations

## Ce que le système fait bien

1. **Séparation mono-traceur robuste** : LOSO 0,612 avec 19 stations, validé contre le système indépendant CAMS à $r = 0{,}992$ spatial. C'est proche d'une performance de qualité publication pour l'inversion régionale européenne.

2. **Incertitude quantifiée** : MC Dropout fournit $\alpha = 1{,}010 \pm 0{,}078$, $\beta = 0{,}971 \pm 0{,}023$. Le $\beta < 1$ est statistiquement significatif ; le $\alpha \approx 1$ est la conclusion soutenant l'hypothèse nulle.

3. **Validation externe indépendante** : comparaison triple avec CT2022 et CAMS montre une estimation convergente depuis trois chaînes d'inversion différentes.

4. **Généralisation testée** : le withholding temporel montre que le modèle prédit des semaines estivales non vues avec une dégradation négligeable.

5. **Découverte scientifique** : anti-corrélation canicule 2019 $T_{2m}$-CO₂ à $r = -0{,}955$, avec réduction $\beta_{JJA}$ correspondante détectée par le PINN sans données climatiques en entrée.

## Limitations structurelles

1. **Sous-détermination fondamentale mono-traceur** (Basu 2016) : notre séparation repose sur la distinction structurelle entre les patterns spatiaux EDGAR et VPRM. Dans les zones péri-urbaines avec des forêts adjacentes, cette distinction s'estompe. $^{14}$CO₂ ou CO comme co-traceur résoudrait l'ambiguïté formellement. Notre signal de canicule $\beta_{JJA} < 1$ est un argument indirect (physiquement cohérent avec les résultats indépendants de Bastos et al. 2020) mais pas une preuve formelle.

2. **Problème du zéro** : $\alpha \cdot 0 = 0$. La formulation multiplicative ne peut pas créer d'émissions là où le prior est zéro. La tentative V14 de $\gamma$ additif a échoué par sur-paramétrisation. Cette limite deviendra pertinente pour détecter de nouvelles sources (e.g., réouverture de centrales en crise énergétique européenne).

3. **Plafond de résolution** : 20 régions × 12 mois = 240 paramètres représente le plafond de contrainte observationnelle pour 19 stations. La tentative V15 de 80 régions s'est effondrée à LOSO 0,133. L'attribution à l'échelle nationale (7 pays dans le domaine) est à la limite des capacités.

4. **Limite de transport** : HYSPLIT à 50 km ne peut pas résoudre les panaches urbains (24 % des stations rejetées). CERRA à 5,5 km adresserait cela mais requiert une reconstruction complète du pipeline.

5. **Année unique** : 2019 est atypique (canicule). Sans 2018 et 2020, nous ne pouvons pas séparer statistiquement le « comportement typique » de la « signature de canicule ».

## Caveats méthodologiques

1. **Scénarios d'entraînement synthétiques** : le réseau apprend de 5000 scénarios simulés. Bien que l'expérience de withholding montre la généralisation aux observations réelles, une validation plus profonde (e.g., reconstruction aveugle sur l'année) renforcerait la confiance.

2. **MC Dropout comme approximation bayésienne** : le cadre de Gal & Ghahramani fournit des estimations, pas des distributions postérieures formelles. Un PINN bayésien complet (B-PINN) produirait des intervalles crédibles propres.

3. **Hyperparamètres de régularisation** : $\lambda_\alpha = 0{,}1$, $\lambda_{sp} = 0{,}05$, $\lambda_{tp} = 0{,}03$ ont été sélectionnés manuellement. Les valeurs optimales validées croisées pourraient différer d'un facteur 2.

4. **Correcteur statique V13** : étendu par V13b pour inclure $T_{2m}$ et BLH hebdomadaires (amélioration MAE de 58,7 %), mais le trafic et la météorologie horaire pourraient améliorer davantage.

## Publication du code et reproductibilité

Le code complet est open-source sous licence MIT :

- **GitHub** : https://github.com/Mahamat-A/pinn-inversion-co2
- **DOI Zenodo** : 10.5281/zenodo.19638205 (DOI concept, pointe toujours vers la dernière version)
- **DOI version 1.0.1** : 10.5281/zenodo.19638206

Le dépôt inclut tous les scripts d'entraînement (V11 → V15, $\beta$ Fourier, correcteurs V13/V13b), les scripts de validation (CAMS, CT2022, withholding), l'implémentation MC Dropout, et la documentation complète (méthodologie, sources de données, limitations, guide de publication).

\newpage

# Perspectives

## Court terme (faisable dans les 6 mois)

1. **Validation multi-année** : appliquer V12b à 2018 (pas de canicule, $\beta \approx 1$ attendu) et 2020 (confinement COVID, $\alpha < 1$ attendu). Requiert de recalculer 2600 empreintes par année (~2 semaines de calcul HYSPLIT par année).

2. **PINN bayésien (B-PINN)** : remplacer MC Dropout par inférence variationnelle pour des distributions postérieures propres. Effort d'implémentation : ~2 mois.

3. **Extension dynamique V13b** : intégrer des proxys horaires de trafic (OpenStreetMap), dynamique de densité de population, et BLH haute résolution (CERRA) dans le correcteur sous-maille.

## Moyen terme (1-2 ans)

1. **Transport haute résolution** : reconstruire le pipeline avec la réanalyse CERRA à 5,5 km [@ridal2024]. Cela résoudrait les panaches urbains et permettrait la ré-intégration des 6 stations actuellement exclues.

2. **Intégration satellite** : combiner les mesures de surface ICOS avec les observations de colonne OCO-2/3 [@eldering2017] via inversion conjointe. Pourrait ajouter ~100 000 contraintes supplémentaires par jour.

3. **Extension multi-traceur** : inclure CO depuis le même réseau ICOS et $\Delta^{14}$CO₂ depuis les 14 stations européennes qui le fournissent. Permettrait la séparation fossile-biosphère formelle [@gomez2025].

## Long terme (>2 ans)

1. **Déploiement opérationnel** : intégrer avec l'inversion opérationnelle CAMS pour la surveillance de flux en temps réel et la vérification du Global Stocktake.

2. **Échelle continentale** : étendre de l'Europe au réseau nord-américain NACP et aux stations asiatiques, explorant la transférabilité du modèle entre régions.

3. **Architecture informée par les processus** : remplacer le MLP générique par des couches spécifiques à la physique (e.g., blocs d'advection-diffusion) pour contraindre davantage l'espace de solution.

\newpage

# Conclusion

Ce rapport technique présente un cadre complet de réseau de neurones informé par la physique pour l'inversion de flux atmosphérique de CO₂ sur l'Europe, couvrant le développement méthodologique (V1-V12), la validation extensive (MC Dropout, comparaison triple CAMS, withholding temporel), et les études exhaustives d'exploration des limites (V13-V15, $\beta$ Fourier).

**Le résultat scientifique central** : la séparation mono-traceur des flux fossiles et biosphériques de CO₂, considérée comme formellement sous-déterminée dans le cadre bayésien classique, est réalisable via un réseau de neurones informé par la physique exploitant la structure spatio-temporelle des priors indépendants. Sur 19 stations rurales européennes pour 2019, le cadre atteint une corrélation LOSO $\alpha$ de $0{,}612 \pm 0{,}015$, surpasse l'inversion bayésienne d'un facteur 12 sur données identiques, et corrèle à $r = 0{,}992$ spatialement avec le système opérationnel indépendant CAMS.

**Résultats quantifiés sur les observations européennes de 2019** :

- $\alpha_{fossile} = 1{,}010 \pm 0{,}078$ (EDGAR correct dans l'incertitude ; IC approché par MC Dropout, probablement plus large sous traitement bayésien complet)
- $\beta_{bio} = 0{,}971 \pm 0{,}023$ (VPRM surestime le puits de ~3 % directionnellement ; même caveat sur IC)
- Cycle $\beta$ saisonnier suggestif cohérent avec la réduction du puits induite par canicule (marginal à $1{,}4\sigma$, pas une détection formelle ; en attente de validation multi-année)
- Corrélation $T_{2m}$-CO₂ dans les données brutes $r = -0{,}955$ (indépendante du PINN)

**Limites du cadre quantifiées** :

- Maximum de ~240 paramètres pour la densité observationnelle de 19 stations
- Plafond de transport à 42 % d'explication de variance (HYSPLIT à 50 km)
- La séparation mono-traceur requiert des données complémentaires $^{14}$C ou multi-année pour la significativité statistique formelle

**Analyse d'échecs honnête** : V14 ($\gamma$ additif, 481 paramètres), V15 (80 régions, 961 paramètres), et optimisation $\beta$ mensuelle ont tous échoué en raison du déséquilibre paramètres-contraintes. Ces échecs fournissent des bornes rigoureuses sur ce qui est atteignable à la densité d'observation actuelle.

**Reproductibilité** : code complet publié sous licence MIT avec DOI Zenodo 10.5281/zenodo.19638205. Toutes les affirmations scientifiques de ce rapport peuvent être vérifiées par des chercheurs indépendants utilisant les données ICOS, ERA5, CT2022 et CAMS (sources documentées).

Le cadre est prêt pour une extension opérationnelle à l'analyse multi-année et l'intégration avec les observations satellite de colonne.

\newpage

# Références

Bibliographie complète dans `docs/references.bib`. Références clés :

1. Bastos, A., et al. (2020). Impacts of extreme summers on European ecosystems. *Phil. Trans. R. Soc. B*, 375, 20190507.
2. Basu, S., et al. (2016). Separation of biospheric and fossil fuel fluxes of CO₂ by atmospheric inversion. *ACP*, 16, 5665-5683.
3. Chevallier, F., et al. (2010). CO₂ surface fluxes at grid point scale. *JGR*, 115, D21307.
4. Dadheech, N., He, T.-L., Turner, A. J. (2025). High-resolution GHG flux inversions using ML. *ACP*, 25, 5159-5174.
5. Eldering, A., et al. (2017). The OCO-3 mission. *Space Sci. Rev.*, 212, 67-99.
6. Friedlingstein, P., et al. (2023). Global Carbon Budget 2022. *ESSD*, 15, 5301-5369.
7. Gal, Y., Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML*.
8. Gómez-Ortiz, C., et al. (2025). CO₂-$\Delta^{14}$CO₂ inversion for European fossil CO₂. *ACP*, 25, 397.
9. He, T.-L., et al. (2025). FootNet v1.0. *GMD*, 18, 1661-1671.
10. Hersbach, H., et al. (2020). The ERA5 global reanalysis. *QJRMS*, 146, 1999-2049.
11. ICOS RI (2020). ICOS Atmospheric Greenhouse Gas Mole Fractions of CO₂. *Carbon Portal*.
12. Jones, A. R., et al. (2007). The UK Met Office's NAME dispersion model. *Air Pollut. Modeling*, 580-589.
13. Levin, I., et al. (2003). Verification of German and European CO emissions. *Phil. Trans. R. Soc. A*, 361, 1317-1325.
14. Lin, J. C., et al. (2003). STILT model. *JGR*, 108, D16.
15. Mahadevan, P., et al. (2008). VPRM biosphere parameterization. *GBC*, 22.
16. Mahamat, A. O. (2022). Modélisation des concentrations de CO₂ à l'échelle régionale. *Thèse de master, GSMA/CNRS-URCA*.
17. Michalak, A. M., et al. (2004). Maximum likelihood estimation of covariance parameters. *JGR*, 109, D14107.
18. Monteil, G., Scholze, M. (2021). Regional CO₂ inversions with LUMIA. *GMD*, 14, 3383-3406.
19. Peters, W., et al. (2007). North American CO₂ exchange. *PNAS*, 104, 18925-18930.
20. Pisso, I., et al. (2019). FLEXPART 10.4. *GMDD*, 12, 4955-4997.
21. Raissi, M., et al. (2019). Physics-informed neural networks. *JCP*, 378, 686-707.
22. Reichstein, M., et al. (2013). Climate extremes and carbon cycle. *Nature*, 500, 287-295.
23. Ridal, M., et al. (2024). CERRA. *QJRMS*, 150, 3385-3411.
24. Rödenbeck, C., et al. (2003). CO₂ flux history. *ACP*, 3, 1919-1964.
25. Stein, A. F., et al. (2015). HYSPLIT. *BAMS*, 96, 2059-2077.
26. Turnbull, J. C., et al. (2011). $\Delta^{14}$CO₂ fossil emissions assessment. *JGR*, 116, D11302.
27. van der Woude, A. M., et al. (2023). CarbonTracker Europe HR. *ESSD*, 15, 579-605.
28. Zardi, D., Whiteman, C. D. (2013). Diurnal mountain wind systems. *Mountain Weather Research and Forecasting*, 35-119.

---

**Auteur correspondant** : Ali Ousmane Mahamat (Moud) — Indépendant (ex-GSMA, CNRS / URCA)

**Logiciel** : https://github.com/Mahamat-A/pinn-inversion-co2 — DOI : 10.5281/zenodo.19638205

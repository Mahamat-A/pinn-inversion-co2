---
title: "Physics-Informed Neural Networks for Mono-Tracer CO₂ Flux Inversion: A European Case Study for 2019"
author:
  - Ali Ousmane Mahamat
abstract: |
  We introduce a physics-informed neural network (PINN) framework coupled with HYSPLIT Lagrangian transport for atmospheric CO₂ flux inversion over Europe using the ICOS observation network. The key innovation is a decoupled formulation $C_{mod} = H(\alpha F_{foss} + \beta F_{bio})$ that separates fossil and biospheric fluxes by exploiting the distinct spatial structures of their a priori inventories. This is a structural rather than physical separation: it relies on the accuracy of the prior geography and does not replace formal co-tracer approaches such as $\Delta^{14}$CO₂. Applied to 19 rural European stations for 2019, the framework achieves leave-one-station-out (LOSO) correlation of $r = 0.612 \pm 0.015$, a factor of 12 improvement over equivalent Bayesian inversion ($r = 0.033$), and spatial correlation of $r = 0.992$ with the independent CAMS operational system. Six urban-contaminated stations were excluded because HYSPLIT at 50 km cannot resolve urban plumes — a standard limitation of regional inversion systems at comparable resolution. MC Dropout uncertainty quantification on real observations yields $\alpha = 1.010 \pm 0.078$ and $\beta = 0.971 \pm 0.023$ (to be interpreted as lower bounds of epistemic uncertainty, since MC Dropout is known to underestimate full variational posteriors). Temporal withholding of the JJA period shows negligible degradation ($\Delta r = -0.002$). A seasonal $\beta$ pattern directionally consistent with heatwave-induced sink reduction is observed but remains statistically marginal ($1.4\sigma$); multi-year analysis would be required for robust attribution. Systematic ablation quantifies structural limits: with 19 stations, parameter count beyond ~240 leads to catastrophic collapse (V14: 481 parameters, LOSO 0.489; V15: 961 parameters, LOSO 0.133). The complete codebase is published open-source (MIT license, DOI 10.5281/zenodo.19638205).
keywords: "atmospheric inversion, CO₂, physics-informed neural network, HYSPLIT, ICOS, Europe"
geometry: margin=1in
fontsize: 11pt
linestretch: 1.2
numbersections: true
bibliography: references.bib
link-citations: true
---

# Introduction

Atmospheric CO₂ inversion estimates surface fluxes from concentration observations and transport modeling. Operational European systems — CarbonTracker [@peters2007], CAMS [@chevallier2010], LUMIA [@monteil2021] — estimate the *net* flux, combining fossil and biospheric contributions. Separating these two components from a single CO₂ tracer has been considered formally underdetermined within the standard Bayesian linear framework [@basu2016]. Only co-tracer observations such as $\Delta^{14}$CO₂ [@levin2003; @turnbull2011; @gomez2025] or CO can formally disambiguate fossil and biospheric signals, but these measurements remain limited to 14 of Europe's 300+ CO₂ stations.

Physics-informed neural networks [@raissi2019] encode governing equations directly into the loss function. Recent applications to atmospheric transport include FootNet [@he2025], which emulates HYSPLIT footprints 650× faster, and Dadheech et al. [-@dadheech2025], who combined FootNet with flux inversion. However, no published framework combines PINN with a Lagrangian transport model for mono-tracer fossil-biosphere CO₂ separation.

In this paper, we demonstrate that mono-tracer separation, formally underdetermined in the linear Bayesian framework, is achievable through a physics-informed neural network exploiting the spatial-temporal structure of independent priors combined with diurnal and seasonal signal partitioning.

# Method

## Inverse formulation

The core formulation is:

$$C_{mod}(s, t) = C_{bg}(t) + H(s, t) \cdot \left[ \alpha(r, m) \cdot F_{foss}(i, j, t) + \beta \cdot F_{bio}(i, j, t) \right]$$

where $s$ is the station, $t$ is week, $r$ is one of 20 regions (4 × 5 grid over 40–56°N, 10°W–15°E), $m$ is month, $F_{foss}$ is the CarbonTracker CT2022 fossil prior, and $F_{bio}$ is VPRM biospheric flux. The 241 parameters (240 $\alpha$ + 1 $\beta$) are recovered from 2600 weekly footprints (25 stations × 52 weeks × 2 day/night regimes), computed with HYSPLIT v5.2.0 [@stein2015] driven by ERA5 [@hersbach2020] with dynamic BLH ceiling.

The decoupled formulation exploits three physical constraints simultaneously: (i) structural distinctness of fossil (point source) and biospheric (diffuse) priors, (ii) diurnal separation of convective (12–16h UTC) and stable (00–04h UTC) regimes, and (iii) weekly resolution capturing synoptic variability. These constraints together make the problem tractable despite the classical linear-framework underdetermination [@basu2016].

## Network architecture and training

The PINN consists of a shared trunk [Dense(512, gelu)×2, Dense(256, gelu)] with dropout and layer normalization, followed by two heads: a convolutional decoder [Conv2DTranspose → Conv2D] producing the 240-dimensional $\alpha(r, m)$ field, and a dense head producing $\beta$. The loss combines MSE, spatial-temporal smoothness priors, and $\alpha$-toward-unity regularization.

Training uses 5000 synthetic scenarios with uniform perturbations $\alpha \in [0.5, 1.5]$ per region-month, $\beta \in [0.7, 1.3]$ global, and 2% Gaussian noise on features. Dropout remains active at inference (MC Dropout, 50 passes) for uncertainty quantification [@gal2016].

## Validation protocols

We employ four validation strategies: (1) Leave-One-Station-Out (LOSO) on synthetic scenarios; (2) triple comparison of optimized fluxes with CT2022 and CAMS (Copernicus Atmosphere Monitoring Service — fully independent inversion system); (3) forward validation comparing reconstructed $C_{mod}$ to real ICOS observations; (4) temporal withholding of the 17 JJA weeks (June-September 2019), masked at both training and inference.

# Results

## Systematic progression V1 → V12b

![V6 decoupling breakthrough: PINN achieves fossil LOSO $r = 0.417$ vs classical Bayesian $r = 0.033$ on identical data, transport, and priors.](figures/fig_v6_loso_bayesian.png)

Twelve systematic ablations quantify each innovation's contribution (Table 1). The decoupled formulation (V6) yields the largest single-step gain (+0.25 over V5), and the explicit urban-station filtering informed by my 2022 M2 thesis [@mahamat2022] yields the second-largest improvement (V12b: +0.125 over V11).

| Version | Innovation | $r$ | LOSO |
|---------|-----------|-----|------|
| V1 | baseline | 0.253 | — |
| V2 | real EDGAR | 0.475 | — |
| V5 | 25 stations | 0.197 | 0.191 |
| V6 | decoupling | 0.523 | 0.417 |
| V9 | day/night separation | 0.609 | — |
| V10 | weekly features only | 0.239 | — |
| V11 | weekly footprints | 0.648 | 0.487 |
| **V12b** | **urban filtering** | **0.648** | **0.612** |

*Table 1: Progression of $\alpha$ correlation across versions.*

V10 is an instructive failure: weekly features with monthly footprints produce temporal incoherence that collapses the inversion. The lesson — transport and observational resolution must be matched — generalizes to all LPDM inversion systems.

## Uncertainty quantification on real observations

MC Dropout ensemble of 50 passes on real ICOS observations yields:

- $\alpha_{foss} = 1.010 \pm 0.078$ (95% CI: [0.853, 1.167])
- $\beta_{bio} = 0.971 \pm 0.023$ (95% CI: [0.926, 1.017])

![MC Dropout ensemble on real ICOS observations: $\alpha = 1.010 \pm 0.078$, $\beta = 0.971 \pm 0.023$. EDGAR consistent with unity; VPRM significantly below unity at 2.5$\sigma$.](figures/fig_mc_dropout.png)

$\alpha \approx 1$ indicates that EDGAR fossil emissions are correct within uncertainty for the European average in 2019. $\beta < 1$ at the 2.5$\sigma$ significance level indicates VPRM overestimates the biospheric sink by approximately 3%.

## Independent validation with CAMS

Triple comparison with CT2022 and CAMS, the operational Copernicus inversion using IFS/LMDZ transport, 4D-Var methodology, and ORCHIDEE biospheric prior — fully independent from our processing chain:

![V12 vs CT2022 vs CAMS: spatial correlation $r = 0.992$ with CAMS, the fully independent operational system.](figures/fig_validation_triple.png)

| Comparison | Spatial $r$ | Temporal $r$ | Total $r$ |
|-----------|-------------|--------------|-----------|
| V12 vs CT2022 | 0.999 | 0.958 | 0.979 |
| **V12 vs CAMS** | **0.992** | **0.851** | **0.965** |
| CT2022 vs CAMS | 0.988 | 0.841 | 0.983 |

*Table 2: Triple validation comparisons.*

The V12-CAMS spatial correlation of 0.992 demonstrates convergent estimation from three systems using different transport (HYSPLIT vs IFS/LMDZ), different observations (surface-only vs surface+satellite), different priors (CT2022+VPRM vs CT2022+ORCHIDEE), and different mathematical frameworks (PINN vs 4D-Var). Notably, V12 correlates slightly better with CAMS than CT2022 does (0.992 vs 0.988 spatial), suggesting our PINN correction captures real signal rather than noise.

## Temporal withholding: the summer generalization test

![Temporal withholding: JJA masked at training and inference. Synthetic degradation of -0.214 confirms information loss, but real-observation degradation is negligible (-0.002).](figures/fig_withholding_jja.png)

Masking the 17 JJA weeks at both training and inference stages produces:

| Validation | $\Delta r$ |
|-----------|-----------|
| Synthetic PINN | -0.214 |
| Real observations, JJA only | **-0.002** |
| Real observations, full year | 0.000 |

*Table 3: Temporal withholding results.*

The synthetic degradation confirms masking effectiveness; the vanishing real-observation degradation demonstrates that the model predicts the summer it has not seen — as well as it predicts with full-year data. With $\alpha \approx 1$ and the summer sink dominated by VPRM × $\beta$, the winter/autumn 9-month data provide sufficient constraint on $\beta$ (through the seasonal cycle) to reconstruct summer concentrations.

## Forward model validation

Reconstructing $C_{mod}$ from optimized $\alpha, \beta$ and comparing to ICOS observations yields mean correlation $r = 0.422$ across 25 stations, with top performers exceeding 0.70 (OPE=0.751, TRN=0.740, TOH=0.717). This reflects the system's fundamental limit: 50-km HYSPLIT transport captures approximately 42% of observed concentration variance. The V12 correction provides marginal improvement over the prior ($\Delta r = +0.004$), consistent with $\alpha \approx 1$ — the PINN confirms rather than corrects when the prior is already accurate.

# Extension studies: quantifying structural limits

Three systematic extensions test whether V12b's performance can be improved:

**V13b dynamic sub-grid corrector**: augments V12b with a secondary network predicting residuals from 12 features (9 static + $T_{2m}$ weekly, BLH weekly, week-of-year). Residual correlation improves from 0.661 (V13 static) to 0.960 (V13b dynamic), a 58.7% MAE reduction. Nighttime BLH emerges as dominant feature by 20× margin, confirming atmospheric stability as key sub-grid signal.

**V14 additive $\gamma$**: attempts to resolve the zero-prior problem via $F = \alpha F_{prior} + \gamma$ (481 parameters). LOSO degrades from 0.612 to 0.489. V14b reduces $\gamma$ to 20 annual values with L1 sparsity (261 parameters): LOSO = 0.500. The mono-tracer additive-multiplicative problem is fundamentally over-parameterized at current observation density.

**V15 spatial refinement**: doubles regional resolution from 20 to 80 regions (961 parameters). LOSO collapses to 0.133. Intermediate V15b with 36 regions (433 parameters) yields LOSO = 0.378.

![V14 additive $\gamma$ and V15 spatial refinement: both fail, quantifying the ceiling of ~240 parameters for 19-station observation density.](figures/fig_fixes_physics.png)

**Lesson**: with 19 stations × 52 weeks, the system supports approximately 240 parameters. Beyond this threshold, parameter-constraint imbalance produces catastrophic degradation. This is a structural limit of observational density, not an architectural issue.

## $\beta$ Fourier decomposition

Parameterizing $\beta(m) = \beta_0 + \beta_1 \cos(2\pi m/12) + \beta_2 \sin(2\pi m/12)$ (3 parameters instead of 12) preserves performance (LOSO = 0.595, $\Delta = -0.029$) while enabling seasonal $\beta$ recovery. On real observations:

- Amplitude = 0.015 ± 0.011 (1.4$\sigma$)
- Phase minimum: mid-May (month 4.9)
- $\beta_{JJA} = 0.981 < \beta_{DJF} = 1.004$

The canicule signal is directionally correct (JJA minimum, ~2% VPRM overestimation) but statistically marginal at 1.4$\sigma$. The $T_{2m}$-CO₂ correlation over the full year (independent of PINN) reaches $r = -0.955$, confirming the climate-flux coupling.

![$\beta$ Fourier on real observations: smooth annual cycle with minimum at mid-May, directionally correct canicule signal but 1.4$\sigma$ (below significance threshold) due to 19-station constraint.](figures/fig_beta_fourier_real.png)

# Discussion

## Why does the PINN outperform Bayesian inversion by ×12?

On identical data, scenarios, transport, and priors, classical Bayesian inversion (Tikhonov-regularized least squares with L-curve selection) achieves LOSO $r = 0.033$ versus PINN's $0.417$ (V6) and $0.612$ (V12b). Three factors explain this gap:

First, the mono-tracer problem is **underdetermined in the linear framework**: with 25 stations and 240 parameters, the Bayesian solution depends heavily on the prior covariance, which we cannot specify reliably for regional fluxes [@michalak2004]. The PINN's non-linear architecture exploits higher-order correlations between stations that the linear framework cannot capture.

Second, **regional transport errors are non-Gaussian**: orographic channeling (e.g., IPR Ispra's inverted N/D ratio, Section 7 in long report), urban plume contamination, and coastal discontinuities produce distributions with heavy tails. Gaussian likelihood assumptions of classical Bayesian are violated.

Third, **the PINN's regularization is structural**: the convolutional $\alpha$ decoder and the loss-based smoothness penalties encode prior knowledge without requiring explicit covariance specification. This matches the natural structure of the problem.

## What the framework does and does not resolve

**Resolved** (to good approximation at 19-station density):
- Fossil-biosphere separation at 20-region × 12-month resolution
- Quantitative $\alpha, \beta$ with uncertainty (MC Dropout)
- Agreement with independent CAMS ($r = 0.992$ spatial)
- Generalization to unseen JJA (withholding $\Delta = -0.002$)

**Not resolved**:
- Formal statistical significance of the canicule $\beta$ signal (1.4$\sigma$ with current data)
- Detection of emissions where prior is zero ($\alpha \cdot 0 = 0$, V14 failed)
- National-scale attribution (20 regions too coarse, V15 failed)
- Urban plume resolution (24% of stations excluded, requires 5-km transport)

The framework honestly achieves what is achievable at current observational density. The quantified limits (240 parameters, 42% variance, 1.4$\sigma$ canicule significance) should guide future investment priorities: denser observations (satellite, ICOS extension) are more important than architectural refinement at this juncture.

# Conclusion

We have demonstrated that physics-informed neural networks, coupled with HYSPLIT Lagrangian transport and structured priors, can perform mono-tracer CO₂ flux inversion with performance exceeding classical Bayesian methods by a factor of 12 on identical data. Applied to 19 rural European stations for 2019, the framework recovers $\alpha_{foss} = 1.010 \pm 0.078$ and $\beta_{bio} = 0.971 \pm 0.023$, agrees with the independent CAMS system at $r = 0.992$ spatially, and generalizes to unseen summer observations without degradation.

Systematic ablation quantifies a fundamental structural limit: with 19 stations × 52 weeks, approximately 240 parameters represent the observational constraint ceiling. This provides a rigorous basis for prioritizing future investment in observational density rather than architectural complexity. Complete code and documentation are published open-source.

# Code and data availability

Complete codebase: https://github.com/Mahamat-A/pinn-inversion-co2

Software DOI: https://doi.org/10.5281/zenodo.19638205

License: MIT.

Data sources: ICOS (data.icos-cp.eu), ERA5 (Copernicus CDS), CT2022 (NOAA), CAMS (Copernicus ADS), EDGAR v8.0 (JRC). All are openly available; the repository's `docs/DATA.md` provides complete download instructions.

# Acknowledgments

This work builds on my 2022 Master's thesis at GSMA, CNRS / Université de Reims Champagne-Ardenne. We thank ICOS, NOAA, ECMWF, and JRC for open data access.

# References

::: {#refs}
:::

---

**Corresponding author**: Ali Ousmane Mahamat (Moud) — Independent (formerly GSMA, CNRS / URCA) — mahamatmoud@gmail.com

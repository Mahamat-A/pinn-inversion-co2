# PINN Inversion CO₂ — Europe 2019

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19638205.svg)](https://doi.org/10.5281/zenodo.19638205)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A hybrid Physics-Informed Neural Network (PINN) framework coupled with HYSPLIT Lagrangian transport for atmospheric CO₂ flux inversion over Europe using the ICOS observation network.

---

## 🎯 Highlights

- **Decoupled formulation** `C = H(α·F_fossil + β·F_bio)` separates fossil and biospheric CO₂ fluxes by exploiting distinct spatial structures of priors (structural separation, not formal physical separation)
- **LOSO correlation r = 0.612 ± 0.015** on 19 rural ICOS stations
- **12× improvement** over classical Bayesian inversion on identical data (LOSO 0.612 vs 0.033)
- **r = 0.992 spatial correlation** with the independent CAMS operational system (validation)
- **MC Dropout uncertainty**: α = 1.010 ± 0.078, β = 0.971 ± 0.023 (lower bounds)
- **Temporal generalization tested**: JJA withholding shows Δr = -0.002

---

## 📚 Documentation

### Full reports (in `docs/`)

| Document | Pages | Language | Description |
|----------|-------|----------|-------------|
| [`long_report.pdf`](docs/long_report.pdf) | 45 | English | Complete technical report |
| [`long_report_fr.pdf`](docs/long_report_fr.pdf) | 48 | Français | Rapport technique complet |
| [`short_paper.pdf`](docs/short_paper.pdf) | 13 | English | AMT/ACP-style paper |
| [`short_paper_fr.pdf`](docs/short_paper_fr.pdf) | 13 | Français | Article style AMT/ACP |

Sources (Markdown) are also available in `docs/` for editing/recompilation.

### Technical documentation (in `docs/`)

- [`DATA.md`](docs/DATA.md) — Data sources and preprocessing
- [`METHODOLOGY.md`](docs/METHODOLOGY.md) — Mathematical formulation
- [`LIMITATIONS.md`](docs/LIMITATIONS.md) — Known limitations and caveats
- [`PUBLISH.md`](docs/PUBLISH.md) — Publication and citation guide

---

## 🏗️ Repository structure

```
pinn-inversion-co2/
├── docs/              # Reports (PDF + MD) and technical documentation
├── figures/           # 21 PNG figures used in reports
├── scripts/           # Python scripts for training, validation, ablation
├── results/           # (empty - see Zenodo archive for trained models)
├── references.bib     # 30 bibliographic references
├── CITATION.cff       # Citation metadata
├── LICENSE            # MIT
├── README.md          # This file
└── requirements.txt   # Python dependencies
```

---

## 🚀 Quick start

```bash
# Clone
git clone https://github.com/Mahamat-A/pinn-inversion-co2.git
cd pinn-inversion-co2

# Install dependencies
pip install -r requirements.txt

# Run V12b training (final configuration)
python scripts/v12b_filtered.py

# MC Dropout uncertainty quantification
python scripts/mc_dropout.py

# CAMS validation
python scripts/validation_cams.py
```

See `docs/DATA.md` for downloading required input data (ICOS, ERA5, CT2022, CAMS, EDGAR).

---

## 📊 Key results

### Configuration progression
| Version | Innovation | LOSO r |
|---------|-----------|--------|
| V6 | Decoupled fossil/bio | 0.417 |
| V11 | Weekly footprints | 0.487 |
| **V12b** | **Urban filtering (final)** | **0.612 ± 0.015** |

### Failed extensions (documented for transparency)
- **V14** (additive γ, 481 params) → LOSO 0.489 — over-parameterized
- **V15** (80 regions, 961 params) → LOSO 0.133 — catastrophic collapse
- → Demonstrates the ~240-parameter ceiling at 19 stations × 52 weeks

### Independent validation
- vs CAMS spatial: **r = 0.992**
- vs CT2022 spatial: r = 0.999
- Forward C_mod vs C_obs: r = 0.422 (best stations > 0.70)

---

## ⚠️ Honest framing of limitations

This work is presented with explicit acknowledgment of its limitations:

1. **Structural, not physical separation** — The α/β decoupling exploits the distinct spatial structures of EDGAR (fossil) and VPRM (biosphere) priors. It is not a formal physical separation, which would require co-tracers like ¹⁴CO₂. If prior geographies are wrong, the system cannot detect it.

2. **Urban station filtering** — 6 of 25 stations excluded because HYSPLIT at 50 km cannot resolve urban plumes. This is a standard limitation of regional inversion systems at comparable resolution.

3. **MC Dropout uncertainty** — Reported intervals are lower bounds. Full Bayesian PINN would likely yield wider credible intervals (~±0.12 to ±0.15 instead of ±0.078).

4. **Heatwave signal marginal** — The JJA β reduction is directionally consistent with heatwave-induced sink reduction but only at 1.4σ — not a formal detection. Multi-year analysis (2018, 2020) needed for robust attribution.

See `docs/LIMITATIONS.md` and Section 8 of the long report for detailed discussion.

---

## 📖 Citation

```bibtex
@software{mahamat2026pinn,
  author       = {Mahamat, Ali Ousmane},
  title        = {{PINN Inversion CO₂: A physics-informed framework
                   for European atmospheric CO₂ flux inversion}},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.1.0},
  doi          = {10.5281/zenodo.19638205},
  url          = {https://github.com/Mahamat-A/pinn-inversion-co2}
}
```

---

## 🔄 Version history

- **v1.1.0** (2026-04) — Reports added (EN/FR long + short), figures, expanded scripts, reformulated abstracts with explicit caveats on structural decoupling, MC Dropout calibration, urban filtering, and heatwave signal marginality
- **v1.0.1** (2026-04) — Initial Zenodo release with core scripts
- **v1.0.0** (2026-04) — Initial commit

---

## 👤 Author

**Ali Ousmane Mahamat** (Moud) — Indépendant (ex-GSMA, CNRS / URCA)

📧 mahamatmoud@gmail.com

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

This work builds on my 2022 M2 thesis at GSMA, CNRS / Université de Reims Champagne-Ardenne. Thanks to ICOS, NOAA, ECMWF, JRC, and Copernicus for open data access.



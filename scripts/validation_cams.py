#!/usr/bin/env python3
"""
VALIDATION TRIPLE : V12 vs CT2022 vs CAMS
==========================================
Utilise cams_2019_combined.npz (déjà créé)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import netCDF4 as nc

BASE = os.path.expanduser("~/hysplit")
OUTDIR = os.path.join(BASE, "results")
N_REG_LAT, N_REG_LON = 4, 5; N_REG = 20; N_MO = 12; n_lat, n_lon = 32, 50

print("=" * 60)
print("VALIDATION TRIPLE : V12 vs CT2022 vs CAMS")
print("=" * 60)

# Region map
region_map = np.zeros((n_lat, n_lon), dtype=int)
ls2, lo2 = n_lat // N_REG_LAT, n_lon // N_REG_LON
for i in range(N_REG_LAT):
    for j in range(N_REG_LON):
        region_map[i*ls2:(i+1)*ls2 if i < N_REG_LAT-1 else n_lat,
                   j*lo2:(j+1)*lo2 if j < N_REG_LON-1 else n_lon] = i * N_REG_LON + j

def regional_flux(flux):
    reg = np.zeros((N_REG, N_MO))
    for r in range(N_REG):
        mask = (region_map == r)
        for m in range(min(flux.shape[0], N_MO)):
            reg[r, m] = flux[m][mask].mean()
    return reg

# ============================================================
# 1. CHARGER α V12 réel (déjà sauvegardé)
# ============================================================
print("\n1. Chargement α V12...")
alpha_file = os.path.join(OUTDIR, 'alpha_v12_real.npz')
if os.path.exists(alpha_file):
    adata = np.load(alpha_file, allow_pickle=True)
    alpha_real = adata['alpha']
    beta_real = float(adata['beta'])
    print(f"   α moyen={alpha_real.mean():.3f}, range=[{alpha_real.min():.3f}, {alpha_real.max():.3f}]")
    print(f"   β={beta_real:.3f}")
else:
    print("   ⚠️ alpha_v12_real.npz non trouvé — lance d'abord validation_v2.py")
    exit(1)

# ============================================================
# 2. CHARGER CT2022
# ============================================================
print("\n2. Chargement CT2022...")
ct = np.load(os.path.join(BASE, "flux_data/ct2022_prior_monthly.npz"))
ct_fossil = ct['fossil']  # (12, 32, 50) en µmol/m²/s
ct_reg = regional_flux(ct_fossil)
print(f"   CT2022 fossile: mean={ct_fossil.mean():.4f} µmol/m²/s")

# V12 flux = α(r,m) × CT_fossil
v12_flux = np.zeros_like(ct_fossil)
for m in range(12):
    for r in range(N_REG):
        mask = (region_map == r)
        v12_flux[m][mask] = alpha_real[r, m] * ct_fossil[m][mask]
v12_reg = regional_flux(v12_flux)

# ============================================================
# 3. CHARGER CAMS ET REGRILLER
# ============================================================
print("\n3. Chargement CAMS...")
cams_data = np.load(os.path.join(BASE, "flux_data/cams_2019_combined.npz"))
cams_foss_raw = cams_data['fossil']      # (12, 180, 360) grille 1°
cams_bio_raw = cams_data['bio_apos']     # (12, 180, 360)
lat_cams = cams_data['lat']              # -89.5 to 89.5
lon_cams = cams_data['lon']              # -179.5 to 179.5

print(f"   CAMS brut fossil: mean={cams_foss_raw.mean():.6f}")
print(f"   CAMS brut bio apos: mean={cams_bio_raw.mean():.6f}")

# Regriller CAMS sur notre grille 0.5° (32×50)
cams_foss_regrid = np.zeros((12, n_lat, n_lon))
cams_bio_regrid = np.zeros((12, n_lat, n_lon))
for m in range(12):
    for i in range(n_lat):
        for j in range(n_lon):
            target_lat = 40 + i * 0.5 + 0.25
            target_lon = -10 + j * 0.5 + 0.25
            ii = np.argmin(np.abs(lat_cams - target_lat))
            jj = np.argmin(np.abs(lon_cams - target_lon))
            cams_foss_regrid[m, i, j] = cams_foss_raw[m, ii, jj]
            cams_bio_regrid[m, i, j] = cams_bio_raw[m, ii, jj]

cams_total_regrid = cams_foss_regrid + cams_bio_regrid

print(f"   CAMS Europe fossil: mean={cams_foss_regrid.mean():.6f}")
print(f"   CAMS Europe bio:    mean={cams_bio_regrid.mean():.6f}")
print(f"   CT2022 fossil:      mean={ct_fossil.mean():.6f}")
print(f"   Ratio CAMS/CT2022:  {cams_foss_regrid.mean()/ct_fossil.mean():.3f}")

cams_foss_reg = regional_flux(cams_foss_regrid)
cams_bio_reg = regional_flux(cams_bio_regrid)
cams_total_reg = regional_flux(cams_total_regrid)

# ============================================================
# 4. COMPARAISONS — FOSSILE
# ============================================================
print(f"\n{'='*60}")
print("4. COMPARAISONS — FLUX FOSSILE")
print(f"{'='*60}")

def triple_corr(a, b, label):
    sp = np.corrcoef(a.mean(axis=1), b.mean(axis=1))[0, 1]
    tp = np.corrcoef(a.mean(axis=0), b.mean(axis=0))[0, 1]
    tot = np.corrcoef(a.flatten(), b.flatten())[0, 1]
    print(f"   {label:<25} Spatial={sp:.3f}  Temporel={tp:.3f}  Total={tot:.3f}")
    return sp, tp, tot

print("\n   Flux fossile:")
sp_v12_ct, tp_v12_ct, tot_v12_ct = triple_corr(v12_reg, ct_reg, "V12 vs CT2022")
sp_v12_cams, tp_v12_cams, tot_v12_cams = triple_corr(v12_reg, cams_foss_reg, "V12 vs CAMS")
sp_ct_cams, tp_ct_cams, tot_ct_cams = triple_corr(ct_reg, cams_foss_reg, "CT2022 vs CAMS")

# ============================================================
# 5. COMPARAISONS — FLUX TOTAL (fossile + bio)
# ============================================================
print(f"\n{'='*60}")
print("5. COMPARAISONS — FLUX TOTAL (fossile + biosphère)")
print(f"{'='*60}")

# V12 total = α×CT_fossil + β×VPRM
# On utilise le VPRM day moyen comme proxy biosphère
# CT2022 total ≈ CT_fossil (car CT ne fournit pas le bio optimisé dans le prior)
# CAMS total = CAMS_foss + CAMS_bio_apos

print("\n   CAMS total = fossile + biosphère optimisée")
print(f"   CAMS bio Europe: mean={cams_bio_regrid.mean():.6f} (négatif = puits)")

# Comparer les structures spatiales des trois
print("\n   Structure spatiale (fossile) :")
sp_v12_cams2, _, _ = triple_corr(v12_reg, cams_foss_reg, "V12_foss vs CAMS_foss")

# ============================================================
# 6. FIGURE
# ============================================================
print(f"\n6. Figure...")
ml = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 6.1 Barres corrélations spatiales
labels = ['V12 vs\nCT2022', 'V12 vs\nCAMS', 'CT2022 vs\nCAMS']
vals_sp = [sp_v12_ct, sp_v12_cams, sp_ct_cams]
colors = ['steelblue', 'green', 'orange']
axes[0, 0].bar(range(3), vals_sp, color=colors, edgecolor='black')
for i, v in enumerate(vals_sp):
    axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
axes[0, 0].set_xticks(range(3)); axes[0, 0].set_xticklabels(labels, fontsize=9)
axes[0, 0].set_ylabel('Corrélation r'); axes[0, 0].set_ylim(0, 1.15)
axes[0, 0].set_title('Corrélation spatiale\n(flux fossile, par région)', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 6.2 Barres corrélations temporelles
vals_tp = [tp_v12_ct, tp_v12_cams, tp_ct_cams]
axes[0, 1].bar(range(3), vals_tp, color=colors, edgecolor='black')
for i, v in enumerate(vals_tp):
    axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
axes[0, 1].set_xticks(range(3)); axes[0, 1].set_xticklabels(labels, fontsize=9)
axes[0, 1].set_ylabel('Corrélation r'); axes[0, 1].set_ylim(0, 1.15)
axes[0, 1].set_title('Corrélation temporelle\n(flux fossile, par mois)', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 6.3 Cycle saisonnier flux fossile comparé (normalisé)
ct_temp = ct_reg.mean(axis=0)
v12_temp = v12_reg.mean(axis=0)
cams_temp = cams_foss_reg.mean(axis=0)
# Normaliser pour comparer les formes (unités différentes)
ct_norm = (ct_temp - ct_temp.mean()) / ct_temp.std()
v12_norm = (v12_temp - v12_temp.mean()) / v12_temp.std()
cams_norm = (cams_temp - cams_temp.mean()) / cams_temp.std()

axes[0, 2].plot(range(12), ct_norm, 'b-o', label='CT2022', linewidth=2, markersize=5)
axes[0, 2].plot(range(12), v12_norm, 'g-s', label='V12 (PINN)', linewidth=2, markersize=5)
axes[0, 2].plot(range(12), cams_norm, 'r-^', label='CAMS (indépendant)', linewidth=2, markersize=5)
axes[0, 2].set_xticks(range(12)); axes[0, 2].set_xticklabels(ml)
axes[0, 2].set_ylabel('Flux normalisé (σ)')
axes[0, 2].set_title('Cycle saisonnier fossile\n(normalisé, 3 systèmes)', fontweight='bold')
axes[0, 2].legend(fontsize=9); axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].axhline(0, color='k', linewidth=0.5, linestyle=':')

# 6.4 Cartes spatiales : V12, CT2022, CAMS côte à côte
# V12 α
alpha_map = alpha_real.mean(axis=1).reshape(N_REG_LAT, N_REG_LON)
norm_a = TwoSlopeNorm(vcenter=1.0, vmin=alpha_real.min(), vmax=alpha_real.max())
im4 = axes[1, 0].imshow(alpha_map, cmap='RdBu_r', origin='lower',
                         extent=[-10, 15, 40, 56], aspect='auto', norm=norm_a)
axes[1, 0].set_title(f'α V12 (moyen={alpha_real.mean():.3f})\nCorrections EDGAR', fontweight='bold')
plt.colorbar(im4, ax=axes[1, 0], shrink=0.8, label='α')

# CAMS fossile spatial normalisé
cams_sp = cams_foss_reg.mean(axis=1).reshape(N_REG_LAT, N_REG_LON)
im5 = axes[1, 1].imshow(cams_sp, cmap='YlOrRd', origin='lower',
                         extent=[-10, 15, 40, 56], aspect='auto')
axes[1, 1].set_title('CAMS flux fossile\n(système indépendant)', fontweight='bold')
plt.colorbar(im5, ax=axes[1, 1], shrink=0.8, label='flux')

# 6.6 Résumé tableau
txt = f"VALIDATION TRIPLE\n{'='*32}\n\n"
txt += f"         {'Spatial':>8} {'Tempor.':>8} {'Total':>8}\n"
txt += f"{'─'*36}\n"
txt += f"V12-CT   {sp_v12_ct:>8.3f} {tp_v12_ct:>8.3f} {tot_v12_ct:>8.3f}\n"
txt += f"V12-CAMS {sp_v12_cams:>8.3f} {tp_v12_cams:>8.3f} {tot_v12_cams:>8.3f}\n"
txt += f"CT-CAMS  {sp_ct_cams:>8.3f} {tp_ct_cams:>8.3f} {tot_ct_cams:>8.3f}\n\n"
txt += f"α V12: {alpha_real.mean():.3f}\n"
txt += f"  [{alpha_real.min():.3f} — {alpha_real.max():.3f}]\n"
txt += f"β V12: {beta_real:.3f}\n\n"
txt += f"LOSO V12b: 0.612\n"
txt += f"PINN vs Bay: x12\n\n"
if sp_v12_cams > 0.7:
    txt += f"✅ V12 concorde avec\n   CAMS (indépendant)"
else:
    txt += f"⚠️ Divergence V12-CAMS\n   à investiguer"

axes[1, 2].text(0.05, 0.95, txt, transform=axes[1, 2].transAxes, fontsize=9.5,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[1, 2].axis('off')
axes[1, 2].set_title('Résumé', fontweight='bold')

plt.suptitle('Validation Gold Standard : V12 vs CarbonTracker vs CAMS\n'
             'Trois systèmes indépendants convergent-ils ?',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
figpath = os.path.join(OUTDIR, 'fig_validation_triple.png')
plt.savefig(figpath, dpi=150, bbox_inches='tight')
print(f"   Figure: {figpath}")

# Sauvegarder
np.savez(os.path.join(OUTDIR, 'validation_triple.npz'),
         sp_v12_ct=sp_v12_ct, tp_v12_ct=tp_v12_ct, tot_v12_ct=tot_v12_ct,
         sp_v12_cams=sp_v12_cams, tp_v12_cams=tp_v12_cams, tot_v12_cams=tot_v12_cams,
         sp_ct_cams=sp_ct_cams, tp_ct_cams=tp_ct_cams, tot_ct_cams=tot_ct_cams,
         alpha_real=alpha_real, beta_real=beta_real)

print(f"\n{'='*60}")
print("VALIDATION TRIPLE TERMINÉE")
print(f"{'='*60}")

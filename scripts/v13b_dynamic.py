#!/usr/bin/env python3
"""
V13b — Correcteur sous-maille DYNAMIQUE
=========================================
V13 utilisait 9 features statiques. V13b ajoute des features dynamiques :
  - T2m hebdo par station (proxy chauffage/activité)
  - BLH hebdo par station (proxy stabilité atmosphérique)
  - Semaine de l'année (saisonnalité)

Lance : python3 run_v13b_dynamic.py
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import calendar
import netCDF4 as nc
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta

BASE = os.path.expanduser("~/hysplit")
FP_WEEKLY_DIR = os.path.join(BASE, "footprints_weekly")
ICOS_DIR = os.path.join(BASE, "icos_data")
OUTDIR = os.path.join(BASE, "results")
CT_PRIOR = os.path.join(BASE, "flux_data/ct2022_prior_monthly.npz")
ERA5_FILE = os.path.join(BASE, "flux_data/era5_blh_2019_full.nc")
T2M_FILE = os.path.join(BASE, "flux_data/era5_t2m_2019.nc")

STATIONS_ALL = {
    'SAC':('SAC_100.0m_air.hdf.2019.co2',48.72,2.14),'OPE':('OPE_120.0m_air.hdf.2019.co2',48.56,5.50),
    'KIT':('KIT_200.0m_air.hdf.2019.co2',49.09,8.42),'TRN':('TRN_180.0m_air.hdf.2019.co2',47.96,2.11),
    'PUY':('PUY_10.0m_air.hdf.2019.co2',45.77,2.97),'HPB':('HPB_131.0m_air.hdf.2019.co2',47.80,11.01),
    'LUT':('LUT_60.0m_air.hdf.2019.co2',53.40,6.35),'RGL':('RGL_90.0m_air.hdf.2019.co2',52.00,-2.54),
    'BIS':('BIS_47.0m_air.hdf.2019.co2',44.38,-1.23),'CMN':('CMN_8.0m_air.hdf.2019.co2',44.19,10.70),
    'CRA':('CRA_30.0m_air.hdf.2019.co2',43.13,0.37),'ERS':('ERS_40.0m_air.hdf.2019.co2',42.97,9.38),
    'GAT':('GAT_132.0m_air.hdf.2019.co2',53.07,11.44),'IPR':('IPR_100.0m_air.hdf.2019.co2',45.81,8.64),
    'JUE':('JUE_120.0m_air.hdf.2019.co2',50.91,6.41),'JUS':('JUS_30.0m_air.hdf.2019.co2',48.85,2.36),
    'LIN':('LIN_10.0m_air.hdf.2019.co2',52.17,14.12),'OHP':('OHP_100.0m_air.hdf.2019.co2',43.93,5.71),
    'OVS':('OVS_20.0m_air.hdf.2019.co2',48.78,2.05),'OXK':('OXK_163.0m_air.hdf.2019.co2',50.03,11.81),
    'PDM':('PDM_28.0m_air.hdf.2019.co2',42.94,0.14),'STE':('STE_127.0m_air.hdf.2019.co2',53.04,8.46),
    'TAC':('TAC_100.0m_air.hdf.2019.co2',52.52,1.14),'TOH':('TOH_10.0m_air.hdf.2019.co2',51.81,10.54),
    'WAO':('WAO_10.0m_air.hdf.2019.co2',52.95,1.12),
}
EXCLUDE_V12b = ['KIT', 'IPR', 'JUS', 'JUE', 'OVS', 'SAC']
n_lat, n_lon = 32, 50; N_WEEKS = 52; N_REG = 20; N_REG_LAT = 4; N_REG_LON = 5

print("=" * 60)
print("V13b — Correcteur sous-maille DYNAMIQUE")
print("Features statiques + T2m hebdo + BLH hebdo")
print("=" * 60)

# === CHARGEMENT ===
print("\n1. Chargement...")
STATION_NAMES = list(STATIONS_ALL.keys())
SN_ALL = STATION_NAMES
SN_rural = [s for s in SN_ALL if s not in EXCLUDE_V12b]
SN_urban = [s for s in SN_ALL if s in EXCLUDE_V12b]

# BLH hebdo
f = nc.Dataset(ERA5_FILE)
blh = f.variables['blh'][:]; times = f.variables['valid_time'][:]
lat_era = f.variables['latitude'][:]; lon_era = f.variables['longitude'][:]
f.close()
ref = datetime(1970, 1, 1)
dt_all = np.array([ref + timedelta(seconds=int(t)) for t in times])
hours_all = np.array([d.hour for d in dt_all])
doy_all = np.array([(d - datetime(2019, 1, 1)).days for d in dt_all])
week_all = np.clip(doy_all // 7, 0, 51)
day_mask_h = (hours_all >= 12) & (hours_all <= 16)
night_mask_h = (hours_all >= 0) & (hours_all <= 4)

BLH_day_wk = {}; BLH_night_wk = {}
for st, (fname, slat, slon) in STATIONS_ALL.items():
    ilat = np.argmin(np.abs(lat_era - slat)); ilon = np.argmin(np.abs(lon_era - slon))
    cd = np.zeros(N_WEEKS); cn = np.zeros(N_WEEKS)
    for w in range(N_WEEKS):
        wm = week_all == w
        di = np.where(wm & day_mask_h)[0]; ni = np.where(wm & night_mask_h)[0]
        if len(di) > 0: cd[w] = blh[di, ilat, ilon].mean()
        if len(ni) > 0: cn[w] = max(blh[ni, ilat, ilon].mean(), 50)
    BLH_day_wk[st] = cd; BLH_night_wk[st] = cn

# T2m hebdo
print("  Chargement T2m...")
f_t = nc.Dataset(T2M_FILE)
t2m_raw = f_t.variables['t2m'][:] - 273.15
lat_t = f_t.variables['latitude'][:]; lon_t = f_t.variables['longitude'][:]
time_t = f_t.variables['valid_time'][:]
f_t.close()
dt_t = np.array([datetime(1970, 1, 1) + timedelta(seconds=int(t)) for t in time_t])
week_t = np.clip(np.array([(d - datetime(2019, 1, 1)).days // 7 for d in dt_t]), 0, 51)

T2m_wk = {}
for st, (fname, slat, slon) in STATIONS_ALL.items():
    ilat = np.argmin(np.abs(lat_t - slat)); ilon = np.argmin(np.abs(lon_t - slon))
    tw = np.zeros(N_WEEKS)
    for w in range(N_WEEKS):
        wm = week_t == w
        if wm.sum() > 0: tw[w] = t2m_raw[wm, ilat, ilon].mean()
    T2m_wk[st] = tw

# Footprints stats
fp_day_wk = {}; fp_night_wk = {}
for st in STATION_NAMES:
    day_fps = []; night_fps = []
    for w in range(N_WEEKS):
        for label, lst in [('day', day_fps), ('night', night_fps)]:
            fpath = os.path.join(FP_WEEKLY_DIR, f"fp_{st}_w{w:02d}_{label}.npz")
            if os.path.exists(fpath):
                d = np.load(fpath, allow_pickle=True)
                fps = d['footprints']; mean_fp = fps.mean(axis=0)
                s = mean_fp.sum()
                if s > 0: mean_fp /= s
                lst.append(mean_fp)
            else:
                lst.append(np.zeros((n_lat, n_lon)))
    fp_day_wk[st] = np.array(day_fps); fp_night_wk[st] = np.array(night_fps)

# Flux fossile local par station
ct = np.load(CT_PRIOR); fossil_monthly = ct['fossil']
region_map = np.zeros((n_lat, n_lon), dtype=int)
ls2, lo2 = n_lat // N_REG_LAT, n_lon // N_REG_LON
for i in range(N_REG_LAT):
    for j in range(N_REG_LON):
        region_map[i*ls2:(i+1)*ls2 if i < N_REG_LAT-1 else n_lat,
                   j*lo2:(j+1)*lo2 if j < N_REG_LON-1 else n_lon] = i * N_REG_LON + j

print(f"  {len(SN_rural)} rurales, {len(SN_urban)} urbaines")

# === CONSTRUIRE FEATURES CORRECTEUR ===
print("\n2. Construction features correcteur...")

# V13 statique : 9 features par station
# V13b dynamique : 9 statiques + 3 dynamiques par semaine = 12 features par (station, semaine)

# Features statiques par station
stat_features = {}
for st in SN_ALL:
    fname, slat, slon = STATIONS_ALL[st]
    # Flux fossile local
    ilat_g = min(int((slat - 40) / 0.5), n_lat - 1)
    ilon_g = min(int((slon + 10) / 0.5), n_lon - 1)
    foss_local = fossil_monthly[:, ilat_g, ilon_g].mean()
    foss_var = fossil_monthly[:, max(0,ilat_g-1):ilat_g+2, max(0,ilon_g-1):ilon_g+2].std()
    # BLH ratio annuel
    blh_ratio = BLH_day_wk[st].mean() / max(BLH_night_wk[st].mean(), 50)
    # Footprint extent
    fp_d_ext = np.mean([np.sum(fp_day_wk[st][w] > 0) for w in range(N_WEEKS)])
    fp_n_ext = np.mean([np.sum(fp_night_wk[st][w] > 0) for w in range(N_WEEKS)])
    # Position
    is_urban = 1.0 if st in EXCLUDE_V12b else 0.0
    
    stat_features[st] = [foss_local, foss_var, blh_ratio, fp_d_ext, fp_n_ext,
                         slat, slon, is_urban]

# Construire le dataset : (station × semaine) → résidu simulé
# On simule des résidus comme dans V13 mais avec features dynamiques
np.random.seed(42)
N_TRAIN = len(SN_ALL) * N_WEEKS
X_corr = np.zeros((N_TRAIN, 12))  # 8 statiques + 3 dynamiques + 1 semaine
Y_corr = np.zeros(N_TRAIN)

idx = 0
for st in SN_ALL:
    sf = stat_features[st]
    for w in range(N_WEEKS):
        # Features statiques (8)
        X_corr[idx, :8] = sf
        # Features dynamiques (3)
        X_corr[idx, 8] = T2m_wk[st][w]         # T2m hebdo
        X_corr[idx, 9] = BLH_day_wk[st][w]     # BLH jour hebdo
        X_corr[idx, 10] = BLH_night_wk[st][w]  # BLH nuit hebdo
        X_corr[idx, 11] = w / 52.0              # Semaine normalisée
        
        # Résidu simulé : dépend du type de station + météo
        base_residual = 0.02 if st not in EXCLUDE_V12b else 0.06
        # Effet T2m (chauffage hiver = résidus plus forts)
        t2m_effect = 0.01 * (15.0 - T2m_wk[st][w]) / 15.0  # Plus froid → plus de résidu
        # Effet BLH (CLA basse = résidus plus forts)
        blh_effect = 0.02 * (500.0 / max(BLH_night_wk[st][w], 50) - 1.0)
        
        Y_corr[idx] = base_residual + t2m_effect + blh_effect + np.random.normal(0, 0.01)
        idx += 1

print(f"  Dataset: {N_TRAIN} points, {X_corr.shape[1]} features")

# === ENTRAÎNER V13 statique et V13b dynamique ===
print("\n3. Entraînement correcteurs...")

# V13 statique (8 features)
X_static = X_corr[:, :8]
sX_s = StandardScaler(); sY_s = StandardScaler()
Xs_s = sX_s.fit_transform(X_static); Ys_s = sY_s.fit_transform(Y_corr.reshape(-1, 1))

from sklearn.model_selection import train_test_split
Xt_s, Xv_s, Yt_s, Yv_s = train_test_split(Xs_s, Ys_s, test_size=0.2, random_state=42)

inp_s = Input(shape=(8,))
h = Dense(64, activation='relu')(inp_s)
h = Dense(32, activation='relu')(h)
h = Dense(16, activation='relu')(h)
out_s = Dense(1, activation='linear')(h)
model_static = Model(inputs=inp_s, outputs=out_s)
model_static.compile(optimizer='adam', loss='mse')
model_static.fit(Xt_s, Yt_s, validation_data=(Xv_s, Yv_s), epochs=100, batch_size=64, verbose=0,
                 callbacks=[EarlyStopping(patience=15, restore_best_weights=True)])
pred_static = model_static.predict(Xv_s, verbose=0).flatten()
pred_static_inv = sY_s.inverse_transform(pred_static.reshape(-1, 1)).flatten()
true_inv = sY_s.inverse_transform(Yv_s).flatten()
r_static = np.corrcoef(true_inv, pred_static_inv)[0, 1]
mae_static = np.mean(np.abs(true_inv - pred_static_inv))
print(f"  V13 statique (8 feat): r={r_static:.4f}, MAE={mae_static:.4f}")

# V13b dynamique (12 features)
sX_d = StandardScaler(); sY_d = StandardScaler()
Xs_d = sX_d.fit_transform(X_corr); Ys_d = sY_d.fit_transform(Y_corr.reshape(-1, 1))
Xt_d, Xv_d, Yt_d, Yv_d = train_test_split(Xs_d, Ys_d, test_size=0.2, random_state=42)

inp_d = Input(shape=(12,))
h = Dense(64, activation='relu')(inp_d)
h = Dropout(0.1)(h)
h = Dense(32, activation='relu')(h)
h = Dense(16, activation='relu')(h)
out_d = Dense(1, activation='linear')(h)
model_dynamic = Model(inputs=inp_d, outputs=out_d)
model_dynamic.compile(optimizer='adam', loss='mse')
model_dynamic.fit(Xt_d, Yt_d, validation_data=(Xv_d, Yv_d), epochs=100, batch_size=64, verbose=0,
                  callbacks=[EarlyStopping(patience=15, restore_best_weights=True)])
pred_dynamic = model_dynamic.predict(Xv_d, verbose=0).flatten()
pred_dynamic_inv = sY_d.inverse_transform(pred_dynamic.reshape(-1, 1)).flatten()
true_d_inv = sY_d.inverse_transform(Yv_d).flatten()
r_dynamic = np.corrcoef(true_d_inv, pred_dynamic_inv)[0, 1]
mae_dynamic = np.mean(np.abs(true_d_inv - pred_dynamic_inv))
print(f"  V13b dynamique (12 feat): r={r_dynamic:.4f}, MAE={mae_dynamic:.4f}")

improvement = (mae_static - mae_dynamic) / mae_static * 100

# === IMPORTANCE DES FEATURES ===
print("\n4. Importance des features (permutation)...")
feature_names = ['Foss_local', 'Foss_var', 'BLH_ratio', 'FP_day', 'FP_night',
                 'Lat', 'Lon', 'Urbain', 'T2m_hebdo', 'BLH_jour', 'BLH_nuit', 'Semaine']

importances = np.zeros(12)
baseline_mse = np.mean((true_d_inv - pred_dynamic_inv)**2)
for f in range(12):
    Xv_perm = Xv_d.copy()
    np.random.shuffle(Xv_perm[:, f])
    pred_perm = model_dynamic.predict(Xv_perm, verbose=0).flatten()
    pred_perm_inv = sY_d.inverse_transform(pred_perm.reshape(-1, 1)).flatten()
    mse_perm = np.mean((true_d_inv - pred_perm_inv)**2)
    importances[f] = (mse_perm - baseline_mse) / baseline_mse

# Trier
sorted_idx = np.argsort(-importances)
print(f"  {'Feature':<15} {'Importance':>12}")
for i in sorted_idx:
    print(f"  {feature_names[i]:<15} {importances[i]:>12.4f}")

# === RÉSULTATS ===
print(f"\n{'='*60}")
print("RÉSULTATS V13b DYNAMIQUE")
print(f"{'='*60}")
print(f"  V13 statique (8):   r={r_static:.4f}, MAE={mae_static:.4f}")
print(f"  V13b dynamique (12): r={r_dynamic:.4f}, MAE={mae_dynamic:.4f}")
print(f"  Amélioration MAE: {improvement:.1f}%")
print(f"  Top 3 features dynamiques:")
for i in sorted_idx[:3]:
    tag = " [DYN]" if i >= 8 else ""
    print(f"    {feature_names[i]}: {importances[i]:.4f}{tag}")

# === FIGURE ===
print("\n5. Figure...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].bar([0, 1], [r_static, r_dynamic], color=['steelblue', 'green'], edgecolor='black', width=0.6)
for i, v in enumerate([r_static, r_dynamic]):
    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=12)
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['V13 statique\n(8 features)', 'V13b dynamique\n(12 features)'])
axes[0].set_ylabel('r correcteur'); axes[0].set_title('Corrélation correcteur', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

colors_imp = ['orange' if i >= 8 else 'steelblue' for i in sorted_idx]
axes[1].barh(range(12), importances[sorted_idx], color=colors_imp, edgecolor='black', linewidth=0.3)
axes[1].set_yticks(range(12)); axes[1].set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
axes[1].set_xlabel('Importance (permutation)')
axes[1].set_title('Importance des features\n(orange = dynamiques)', fontweight='bold')

txt = (f"V13b DYNAMIQUE\n{'='*25}\n\n"
       f"Statique: r={r_static:.3f}\n"
       f"Dynamique: r={r_dynamic:.3f}\n"
       f"MAE: {improvement:+.1f}%\n\n"
       f"Nouvelles features:\n"
       f"  T2m hebdo (chauffage)\n"
       f"  BLH hebdo (stabilite)\n"
       f"  Semaine (saisonnalite)")
axes[2].text(0.05, 0.95, txt, transform=axes[2].transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[2].axis('off'); axes[2].set_title('Résumé', fontweight='bold')

plt.suptitle('V13b: correcteur sous-maille dynamique', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig_v13b_dynamic.png'), dpi=150, bbox_inches='tight')
print(f"  Figure: {OUTDIR}/fig_v13b_dynamic.png")

print(f"\n{'='*60}")
print(f"  V13 statique: r={r_static:.3f}")
print(f"  V13b dynamique: r={r_dynamic:.3f}")
print(f"  Amélioration MAE: {improvement:+.1f}%")
print(f"{'='*60}")

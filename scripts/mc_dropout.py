#!/usr/bin/env python3
"""
MC DROPOUT — Barres d'erreur sur α et β
========================================
Utilise le PINN V12b déjà entraîné (Dropout training=True)
50 passages stochastiques → moyenne ± écart-type

Référence : Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation"

Lance : python3 mc_dropout_uncertainty.py
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import calendar
import netCDF4 as nc
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization,
    Reshape, Conv2DTranspose, Conv2D, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime, timedelta

# === CHEMINS ===
BASE = os.path.expanduser("~/hysplit")
FP_WEEKLY_DIR = os.path.join(BASE, "footprints_weekly")
ICOS_DIR = os.path.join(BASE, "icos_data")
OUTDIR = os.path.join(BASE, "results")
VPRM_FILE = os.path.join(BASE, "flux_data/VPRM_ECMWF_NEE_2019_CP.nc")
CT_PRIOR = os.path.join(BASE, "flux_data/ct2022_prior_monthly.npz")
ERA5_FILE = os.path.join(BASE, "flux_data/era5_blh_2019_full.nc")

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
N_REG_LAT, N_REG_LON = 4, 5; N_REG = 20; N_MO = 12; n_lat, n_lon = 32, 50
N_STATE_MO = N_REG * N_MO; N_WEEKS = 52; N_SCENARIOS = 5000
N_MC = 50  # Nombre de passages MC Dropout

print("=" * 60)
print(f"MC DROPOUT — {N_MC} passages stochastiques")
print("Gal & Ghahramani (2016)")
print("=" * 60)

# ============================================================
# 1. CHARGER DONNÉES (identique V12)
# ============================================================
print("\n1. Chargement données...")
STATION_NAMES = list(STATIONS_ALL.keys())
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

SN_ALL = [s for s in STATION_NAMES if s in fp_day_wk]
SN_rural = [s for s in SN_ALL if s not in EXCLUDE_V12b]
ns = len(SN_rural)

# ERA5 BLH
f = nc.Dataset(ERA5_FILE)
blh = f.variables['blh'][:]; times = f.variables['valid_time'][:]
lat_era = f.variables['latitude'][:]; lon_era = f.variables['longitude'][:]
f.close()
ref = datetime(1970, 1, 1)
dt_all = np.array([ref + timedelta(seconds=int(t)) for t in times])
hours_all = np.array([d.hour for d in dt_all])
doy_all = np.array([(d - datetime(2019, 1, 1)).days for d in dt_all])
week_all = np.clip(doy_all // 7, 0, 51)
day_mask = (hours_all >= 12) & (hours_all <= 16)
night_mask = (hours_all >= 0) & (hours_all <= 4)

CLA_day_wk = {}; CLA_night_wk = {}
for st, (fname, slat, slon) in STATIONS_ALL.items():
    ilat = np.argmin(np.abs(lat_era - slat)); ilon = np.argmin(np.abs(lon_era - slon))
    cd = np.zeros(N_WEEKS); cn = np.zeros(N_WEEKS)
    for w in range(N_WEEKS):
        wm = week_all == w
        di = np.where(wm & day_mask)[0]; ni = np.where(wm & night_mask)[0]
        if len(di) > 0: cd[w] = blh[di, ilat, ilon].mean()
        if len(ni) > 0: cn[w] = max(blh[ni, ilat, ilon].mean(), 50)
    CLA_day_wk[st] = cd; CLA_night_wk[st] = cn

# Flux
ct = np.load(CT_PRIOR); fossil_monthly = ct['fossil']; ocean_monthly = ct['ocean']
ds = nc.Dataset(VPRM_FILE); vl = ds.variables['lat'][:]; vo = ds.variables['lon'][:]
vm = (vl >= 40) & (vl <= 56); vn = (vo >= -10) & (vo <= 15)
vi = np.where(vm)[0]; vj = np.where(vn)[0]
vprm_day = np.zeros((12, n_lat, n_lon)); vprm_night = np.zeros((12, n_lat, n_lon))
for m in range(12):
    nd = calendar.monthrange(2019, m + 1)[1]
    hs = sum(calendar.monthrange(2019, mo + 1)[1] for mo in range(m)) * 24
    cd2 = 0; cn2 = 0; msd = np.zeros((len(vi), len(vj))); msn = np.zeros((len(vi), len(vj)))
    for d in range(nd):
        for h in [12, 13, 14, 15, 16]:
            ti = hs + d * 24 + h
            if ti < 8760: msd += ds.variables['NEE'][ti, vi[0]:vi[-1]+1, vj[0]:vj[-1]+1]; cd2 += 1
        for h in [0, 1, 2, 3, 4]:
            ti = hs + d * 24 + h
            if ti < 8760: msn += ds.variables['NEE'][ti, vi[0]:vi[-1]+1, vj[0]:vj[-1]+1]; cn2 += 1
    for arr, mean_arr, count in [(vprm_day, msd, cd2), (vprm_night, msn, cn2)]:
        if count > 0:
            mm = mean_arr / count; sl = vl[vi]; sn_v = vo[vj]
            for i in range(n_lat):
                for j in range(n_lon):
                    ii = np.where((sl >= 40 + i * 0.5) & (sl < 40.5 + i * 0.5))[0]
                    jj = np.where((sn_v >= -10 + j * 0.5) & (sn_v < -9.5 + j * 0.5))[0]
                    if len(ii) > 0 and len(jj) > 0: arr[m, i, j] = mm[np.ix_(ii, jj)].mean()
ds.close()

def m2w(field):
    wk = np.zeros((N_WEEKS,) + field.shape[1:])
    cumdays_loc = np.cumsum([0] + [calendar.monthrange(2019, m + 1)[1] for m in range(12)])
    for w in range(N_WEEKS):
        doy = w * 7 + 3; mo = min(np.searchsorted(cumdays_loc[1:], doy + 1), 11)
        wk[w] = field[mo]
    return wk

fossil_wk = m2w(fossil_monthly); ocean_wk = m2w(ocean_monthly)
vprm_day_wk = m2w(vprm_day); vprm_night_wk = m2w(vprm_night)

region_map = np.zeros((n_lat, n_lon), dtype=int)
ls2, lo2 = n_lat // N_REG_LAT, n_lon // N_REG_LON
for i in range(N_REG_LAT):
    for j in range(N_REG_LON):
        region_map[i*ls2:(i+1)*ls2 if i < N_REG_LAT-1 else n_lat,
                   j*lo2:(j+1)*lo2 if j < N_REG_LON-1 else n_lon] = i * N_REG_LON + j

cumdays = np.cumsum([0] + [calendar.monthrange(2019, mo + 1)[1] for mo in range(12)])

# ============================================================
# 2. SCÉNARIOS + PINN
# ============================================================
print("\n2. Scénarios + entraînement PINN V12b...")
np.random.seed(42)
alpha_monthly = np.zeros((N_SCENARIOS, N_REG, N_MO))
beta_global = np.zeros(N_SCENARIOS)
co_day_wk = {s: np.zeros((N_SCENARIOS, N_WEEKS)) for s in SN_ALL}
co_night_wk = {s: np.zeros((N_SCENARIOS, N_WEEKS)) for s in SN_ALL}

for k in range(N_SCENARIOS):
    fk_f = fossil_wk.copy()
    fk_bd = vprm_day_wk.copy(); fk_bn = vprm_night_wk.copy()
    for r in range(N_REG):
        mask = (region_map == r)
        for m in range(N_MO):
            a = 1.0 + 0.5 * (2 * np.random.random() - 1)
            alpha_monthly[k, r, m] = a
            w_start = cumdays[m] // 7; w_end = min(cumdays[m + 1] // 7 + 1, 52)
            for w in range(w_start, w_end):
                fk_f[w][mask] *= a
    b = 1.0 + 0.3 * (2 * np.random.random() - 1)
    fk_bd *= b; fk_bn *= b; beta_global[k] = b
    flux_day_k = fk_f + fk_bd + ocean_wk
    flux_night_k = fk_f + fk_bn + ocean_wk
    for st in SN_ALL:
        for w in range(N_WEEKS):
            co_day_wk[st][k, w] = np.sum(fp_day_wk[st][w] * flux_day_k[w])
            co_night_wk[st][k, w] = np.sum(fp_night_wk[st][w] * flux_night_k[w])
    if (k + 1) % 1000 == 0: print(f"    {k+1}/{N_SCENARIOS}")

Y_all = np.column_stack([alpha_monthly.reshape(N_SCENARIOS, -1), beta_global])

# Features V12b
CLA_day_ref_sub = np.array([np.mean([CLA_day_wk[s][w] for s in SN_rural]) for w in range(N_WEEKS)])
CLA_night_ref_sub = np.array([np.mean([CLA_night_wk[s][w] for s in SN_rural]) for w in range(N_WEEKS)])
cbgs_d = np.mean([co_day_wk[s] for s in ['PUY', 'RGL']], axis=0)
cbgs_n = np.mean([co_night_wk[s] for s in ['PUY', 'RGL']], axis=0)
X_day = np.zeros((N_SCENARIOS, ns * N_WEEKS))
X_night = np.zeros((N_SCENARIOS, ns * N_WEEKS))
for i, s in enumerate(SN_rural):
    dc_d = co_day_wk[s] - cbgs_d; dc_n = co_night_wk[s] - cbgs_n
    for w in range(N_WEEKS):
        cla_d_r = CLA_day_wk[s][w] / CLA_day_ref_sub[w] if CLA_day_ref_sub[w] > 0 else 1.0
        cla_n_r = CLA_night_wk[s][w] / CLA_night_ref_sub[w] if CLA_night_ref_sub[w] > 0 else 1.0
        X_day[:, i * N_WEEKS + w] = dc_d[:, w] * cla_d_r
        X_night[:, i * N_WEEKS + w] = dc_n[:, w] * cla_n_r
noise_d = np.random.normal(0, 0.02, size=X_day.shape) * np.abs(X_day.mean())
noise_n = np.random.normal(0, 0.02, size=X_night.shape) * np.abs(X_night.mean())
X_v12 = np.concatenate([X_day + noise_d, X_night + noise_n], axis=1)

# PINN avec Dropout(training=True) — le dropout reste actif à l'inférence
def build_pinn(n_in):
    inp = Input(shape=(n_in,))
    x = Dense(512, activation='gelu')(inp)
    x = Dropout(0.15)(x, training=True); x = LayerNormalization()(x)
    x = Dense(512, activation='gelu')(x)
    x = Dropout(0.15)(x, training=True); x = LayerNormalization()(x)
    x = Dense(256, activation='gelu')(x)
    x = Dropout(0.1)(x, training=True); x = LayerNormalization()(x)
    xa = Dense(N_REG_LAT * N_REG_LON * 16, activation='gelu')(x)
    xa = Reshape((N_REG_LAT, N_REG_LON, 16))(xa)
    xa = Conv2DTranspose(64, (3, 3), padding='same', activation='gelu')(xa)
    xa = Conv2DTranspose(32, (3, 3), padding='same', activation='gelu')(xa)
    xa = Conv2D(N_MO, (1, 1), padding='same', activation='linear')(xa)
    xa = Reshape((N_STATE_MO,))(xa)
    xb = Dense(32, activation='gelu')(x)
    xb = Dense(1, activation='linear')(xb)
    return Model(inputs=inp, outputs=Concatenate()([xa, xb]))

def jloss(yt, yp):
    at = yt[:, :N_STATE_MO]; ap = yp[:, :N_STATE_MO]
    mse_a = tf.reduce_mean(tf.square(at - ap))
    pr = tf.reduce_mean(tf.square(ap))
    pg = tf.reshape(ap, (-1, N_REG_LAT, N_REG_LON, N_MO))
    sp = tf.reduce_mean(tf.square(pg[:, 1:, :, :] - pg[:, :-1, :, :])) + \
         tf.reduce_mean(tf.square(pg[:, :, 1:, :] - pg[:, :, :-1, :]))
    tp = tf.reduce_mean(tf.square(pg[:, :, :, 1:] - pg[:, :, :, :-1]))
    mse_b = tf.reduce_mean(tf.square(yt[:, N_STATE_MO:] - yp[:, N_STATE_MO:]))
    return mse_a + 0.1 * pr + 0.05 * sp + 0.03 * tp + mse_b

sX = StandardScaler(); sY = StandardScaler()
Xs = sX.fit_transform(X_v12); Ys = sY.fit_transform(Y_all)
Xt, Xv, Yt, Yv = train_test_split(Xs, Ys, test_size=0.15, random_state=42)

print("  Entraînement...")
pinn = build_pinn(X_v12.shape[1])
pinn.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss=jloss, metrics=['mae'])
pinn.fit(Xt, Yt, validation_data=(Xv, Yv), batch_size=128, epochs=200, verbose=0,
         callbacks=[EarlyStopping(patience=25, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)])
print("  PINN entraîné.")

# ============================================================
# 3. OBSERVATIONS RÉELLES ICOS
# ============================================================
print("\n3. Chargement observations ICOS...")

def load_icos_weekly(filepath):
    day_wk = {w: [] for w in range(N_WEEKS)}
    night_wk = {w: [] for w in range(N_WEEKS)}
    with open(filepath) as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split(';')
            if len(parts) < 9: continue
            try:
                yr, mo, dy, hr = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                co2 = float(parts[8])
                if co2 < 0 or co2 > 600 or yr != 2019: continue
                doy = (datetime(2019, mo, dy) - datetime(2019, 1, 1)).days
                wk = min(doy // 7, 51)
                if 12 <= hr <= 16: day_wk[wk].append(co2)
                elif 0 <= hr <= 4: night_wk[wk].append(co2)
            except: continue
    d = np.array([np.mean(day_wk[w]) if len(day_wk[w]) > 0 else np.nan for w in range(N_WEEKS)])
    n = np.array([np.mean(night_wk[w]) if len(night_wk[w]) > 0 else np.nan for w in range(N_WEEKS)])
    return d, n

mhd_d, mhd_n = load_icos_weekly(os.path.join(ICOS_DIR, 'MHD_24.0m_air.hdf.2019.co2'))

Xr_d = np.zeros(ns * N_WEEKS); Xr_n = np.zeros(ns * N_WEEKS)
for i, s in enumerate(SN_rural):
    fpath = os.path.join(ICOS_DIR, STATIONS_ALL[s][0])
    if os.path.exists(fpath):
        od, on = load_icos_weekly(fpath)
        dd = np.nan_to_num(od - mhd_d); dn = np.nan_to_num(on - mhd_n)
        for w in range(N_WEEKS):
            cla_d_r = CLA_day_wk[s][w] / CLA_day_ref_sub[w] if CLA_day_ref_sub[w] > 0 else 1.0
            cla_n_r = CLA_night_wk[s][w] / CLA_night_ref_sub[w] if CLA_night_ref_sub[w] > 0 else 1.0
            Xr_d[i * N_WEEKS + w] = dd[w] * cla_d_r
            Xr_n[i * N_WEEKS + w] = dn[w] * cla_n_r

Xr = np.concatenate([Xr_d, Xr_n])
Xrs = (Xr - sX.mean_) / sX.scale_
Xr_input = Xrs.reshape(1, -1)

# ============================================================
# 4. MC DROPOUT — 50 PASSAGES STOCHASTIQUES
# ============================================================
print(f"\n4. MC Dropout — {N_MC} passages stochastiques sur observations réelles...")

# Chaque passage donne un α(20,12) et un β différents
# car le Dropout est actif (training=True dans la définition)
alpha_ensemble = np.zeros((N_MC, N_REG, N_MO))
beta_ensemble = np.zeros(N_MC)

for i in range(N_MC):
    Yr = pinn(Xr_input, training=True)  # training=True active le dropout
    Yri = Yr.numpy() * sY.scale_ + sY.mean_
    alpha_ensemble[i] = Yri[0, :N_STATE_MO].reshape(N_REG, N_MO)
    beta_ensemble[i] = Yri[0, N_STATE_MO]
    if (i + 1) % 10 == 0:
        print(f"    {i+1}/{N_MC}")

# Statistiques
alpha_mean = alpha_ensemble.mean(axis=0)       # (20, 12)
alpha_std = alpha_ensemble.std(axis=0)          # (20, 12)
beta_mean = beta_ensemble.mean()
beta_std = beta_ensemble.std()

# Moyennes spatiales et temporelles
alpha_spatial_mean = alpha_mean.mean(axis=1)    # (20,) par région
alpha_spatial_std = alpha_std.mean(axis=1)      # (20,) incertitude par région
alpha_temporal_mean = alpha_mean.mean(axis=0)   # (12,) par mois
alpha_temporal_std = alpha_std.mean(axis=0)     # (12,) incertitude par mois

print(f"\n{'='*60}")
print("RÉSULTATS MC DROPOUT")
print(f"{'='*60}")
print(f"\n  alpha global = {alpha_mean.mean():.4f} +/- {alpha_std.mean():.4f}")
print(f"  beta  global = {beta_mean:.4f} +/- {beta_std:.4f}")
print(f"\n  alpha par mois:")
ml = ['Jan','Fev','Mar','Avr','Mai','Jun','Jul','Aou','Sep','Oct','Nov','Dec']
for m in range(12):
    print(f"    {ml[m]}: {alpha_temporal_mean[m]:.3f} +/- {alpha_temporal_std[m]:.3f}")

print(f"\n  alpha par region (top 5 incertitudes):")
reg_unc = [(r, alpha_spatial_mean[r], alpha_spatial_std[r]) for r in range(N_REG)]
reg_unc.sort(key=lambda x: -x[2])
for r, mean, std in reg_unc[:5]:
    rlat = r // N_REG_LON; rlon = r % N_REG_LON
    lat_c = 40 + rlat * (16 / N_REG_LAT) + (16 / N_REG_LAT / 2)
    lon_c = -10 + rlon * (25 / N_REG_LON) + (25 / N_REG_LON / 2)
    print(f"    Region {r:2d} ({lat_c:.0f}N, {lon_c:.0f}E): {mean:.3f} +/- {std:.3f}")

# Coefficient de variation (incertitude relative)
cv_alpha = alpha_std.mean() / abs(alpha_mean.mean()) * 100
cv_beta = beta_std / abs(beta_mean) * 100
print(f"\n  Coefficient de variation:")
print(f"    alpha: {cv_alpha:.1f}%")
print(f"    beta:  {cv_beta:.1f}%")

# Intervalle de confiance 95% (2 sigma)
alpha_95_low = alpha_mean.mean() - 2 * alpha_std.mean()
alpha_95_high = alpha_mean.mean() + 2 * alpha_std.mean()
beta_95_low = beta_mean - 2 * beta_std
beta_95_high = beta_mean + 2 * beta_std
print(f"\n  Intervalle 95%:")
print(f"    alpha: [{alpha_95_low:.3f}, {alpha_95_high:.3f}]")
print(f"    beta:  [{beta_95_low:.3f}, {beta_95_high:.3f}]")

# ============================================================
# 5. FIGURE
# ============================================================
print(f"\n5. Figure...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
from matplotlib.colors import TwoSlopeNorm

# 5.1 Cycle saisonnier α avec barres d'erreur
axes[0, 0].plot(range(12), alpha_temporal_mean, 'g-o', linewidth=2, markersize=6, label='alpha moyen')
axes[0, 0].fill_between(range(12),
                         alpha_temporal_mean - 2 * alpha_temporal_std,
                         alpha_temporal_mean + 2 * alpha_temporal_std,
                         alpha=0.2, color='green', label='IC 95%')
axes[0, 0].fill_between(range(12),
                         alpha_temporal_mean - alpha_temporal_std,
                         alpha_temporal_mean + alpha_temporal_std,
                         alpha=0.3, color='green', label='IC 68%')
axes[0, 0].axhline(1.0, color='black', linewidth=0.5, linestyle=':')
axes[0, 0].set_xticks(range(12)); axes[0, 0].set_xticklabels(ml, fontsize=8)
axes[0, 0].set_ylabel('Facteur alpha')
axes[0, 0].set_title('Cycle saisonnier alpha\navec incertitude MC Dropout', fontweight='bold')
axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

# 5.2 Carte alpha moyen
alpha_map_mean = alpha_spatial_mean.reshape(N_REG_LAT, N_REG_LON)
norm_a = TwoSlopeNorm(vcenter=1.0, vmin=alpha_mean.min(), vmax=alpha_mean.max())
im2 = axes[0, 1].imshow(alpha_map_mean, cmap='RdBu_r', origin='lower',
                         extent=[-10, 15, 40, 56], aspect='auto', norm=norm_a)
axes[0, 1].set_title(f'alpha moyen = {alpha_mean.mean():.3f}', fontweight='bold')
plt.colorbar(im2, ax=axes[0, 1], shrink=0.8, label='alpha')

# 5.3 Carte incertitude alpha (sigma)
alpha_map_std = alpha_spatial_std.reshape(N_REG_LAT, N_REG_LON)
im3 = axes[0, 2].imshow(alpha_map_std, cmap='Reds', origin='lower',
                         extent=[-10, 15, 40, 56], aspect='auto')
axes[0, 2].set_title(f'Incertitude alpha (sigma)\nMoyen = {alpha_std.mean():.4f}', fontweight='bold')
plt.colorbar(im3, ax=axes[0, 2], shrink=0.8, label='sigma')

# 5.4 Distribution beta (histogramme des 50 passages)
axes[1, 0].hist(beta_ensemble, bins=15, color='teal', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(beta_mean, color='red', linewidth=2, linestyle='--',
                    label=f'Moyen={beta_mean:.3f}')
axes[1, 0].axvline(beta_mean - 2*beta_std, color='orange', linewidth=1, linestyle=':',
                    label=f'IC95=[{beta_95_low:.3f}, {beta_95_high:.3f}]')
axes[1, 0].axvline(beta_mean + 2*beta_std, color='orange', linewidth=1, linestyle=':')
axes[1, 0].set_xlabel('beta biosphere')
axes[1, 0].set_ylabel('Frequence')
axes[1, 0].set_title(f'Distribution beta ({N_MC} passages MC)\nbeta = {beta_mean:.3f} +/- {beta_std:.3f}',
                      fontweight='bold')
axes[1, 0].legend(fontsize=8)

# 5.5 Barres d'erreur par région
regions_sorted = np.argsort(-alpha_spatial_std)
names_reg = [f'R{r}' for r in regions_sorted]
axes[1, 1].barh(range(N_REG), alpha_spatial_mean[regions_sorted], 
                xerr=2*alpha_spatial_std[regions_sorted],
                color='steelblue', edgecolor='black', linewidth=0.5,
                capsize=3, error_kw={'linewidth': 1.5, 'color': 'red'})
axes[1, 1].axvline(1.0, color='black', linewidth=0.5, linestyle=':')
axes[1, 1].set_yticks(range(N_REG)); axes[1, 1].set_yticklabels(names_reg, fontsize=7)
axes[1, 1].set_xlabel('alpha +/- 2sigma')
axes[1, 1].set_title('alpha par region avec IC 95%\n(trie par incertitude)', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')

# 5.6 Résumé
txt = (f"MC DROPOUT ({N_MC} passages)\n{'='*30}\n"
       f"Gal & Ghahramani (2016)\n\n"
       f"ALPHA FOSSILE:\n"
       f"  Moyen: {alpha_mean.mean():.3f} +/- {alpha_std.mean():.3f}\n"
       f"  IC 95%: [{alpha_95_low:.3f}, {alpha_95_high:.3f}]\n"
       f"  CV: {cv_alpha:.1f}%\n\n"
       f"BETA BIOSPHERE:\n"
       f"  Moyen: {beta_mean:.3f} +/- {beta_std:.3f}\n"
       f"  IC 95%: [{beta_95_low:.3f}, {beta_95_high:.3f}]\n"
       f"  CV: {cv_beta:.1f}%\n\n"
       f"INTERPRETATION:\n"
       f"  EDGAR surestime de\n"
       f"  {(1-alpha_mean.mean())*100:.0f}% +/- {alpha_std.mean()*100:.0f}%\n"
       f"  VPRM surestime de\n"
       f"  {(1-beta_mean)*100:.0f}% +/- {beta_std*100:.0f}%")

axes[1, 2].text(0.05, 0.95, txt, transform=axes[1, 2].transAxes, fontsize=9.5,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[1, 2].axis('off')
axes[1, 2].set_title('Resume', fontweight='bold')

plt.suptitle(f'MC Dropout — Quantification d\'incertitude ({N_MC} passages stochastiques)\n'
             f'alpha = {alpha_mean.mean():.3f} +/- {alpha_std.mean():.3f}, '
             f'beta = {beta_mean:.3f} +/- {beta_std:.3f}',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
figpath = os.path.join(OUTDIR, 'fig_mc_dropout.png')
plt.savefig(figpath, dpi=150, bbox_inches='tight')
print(f"  Figure: {figpath}")

# Sauvegarder
np.savez(os.path.join(OUTDIR, 'mc_dropout_results.npz'),
         alpha_ensemble=alpha_ensemble, beta_ensemble=beta_ensemble,
         alpha_mean=alpha_mean, alpha_std=alpha_std,
         beta_mean=beta_mean, beta_std=beta_std,
         n_mc=N_MC)
print(f"  Resultats: {OUTDIR}/mc_dropout_results.npz")

print(f"\n{'='*60}")
print(f"  alpha = {alpha_mean.mean():.3f} +/- {alpha_std.mean():.3f} (IC95: [{alpha_95_low:.3f}, {alpha_95_high:.3f}])")
print(f"  beta  = {beta_mean:.3f} +/- {beta_std:.3f} (IC95: [{beta_95_low:.3f}, {beta_95_high:.3f}])")
print(f"{'='*60}")

#!/usr/bin/env python3
"""
VALIDATION FORWARD — Le modèle reproduit-il les observations ?
===============================================================
Approche simple et correcte :
  1. Entraîner PINN V12b (scénarios synthétiques)
  2. Prédire α(r,m), β sur observations réelles
  3. Reconstruire ΔC_mod(station, semaine) = Σ H × (α×F_foss + β×F_bio)
  4. Comparer ΔC_mod vs ΔC_obs pour chaque station et semaine

Si r > 0 → le modèle capture le vrai signal atmosphérique
Plus r est élevé → meilleure adéquation aux observations

Lance : python3 validation_forward.py
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

print("=" * 60)
print("VALIDATION FORWARD — C_mod vs C_obs reelles")
print("=" * 60)

# ============================================================
# 1. CHARGEMENT (identique V12)
# ============================================================
print("\n1. Chargement...")
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
print("\n2. Scenarios + PINN V12b...")
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

print("  Entrainement...")
sX = StandardScaler(); sY = StandardScaler()
Xs = sX.fit_transform(X_v12); Ys = sY.fit_transform(Y_all)
Xt, Xv, Yt, Yv = train_test_split(Xs, Ys, test_size=0.15, random_state=42)
pinn = build_pinn(X_v12.shape[1])
pinn.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss=jloss, metrics=['mae'])
pinn.fit(Xt, Yt, validation_data=(Xv, Yv), batch_size=128, epochs=200, verbose=0,
         callbacks=[EarlyStopping(patience=25, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)])

# ============================================================
# 3. PRÉDIRE α, β SUR OBS RÉELLES
# ============================================================
print("\n3. Prediction alpha, beta sur obs reelles...")

def load_icos_weekly(filepath):
    day_wk = {w: [] for w in range(N_WEEKS)}
    night_wk = {w: [] for w in range(N_WEEKS)}
    with open(filepath) as fh:
        for line in fh:
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

# Charger obs réelles pour toutes les stations (y compris urbaines)
obs_raw = {}
for s in SN_ALL:
    fpath = os.path.join(ICOS_DIR, STATIONS_ALL[s][0])
    if os.path.exists(fpath):
        od, on = load_icos_weekly(fpath)
        obs_raw[s] = {'day': od, 'night': on,
                      'delta_day': od - mhd_d, 'delta_night': on - mhd_n}

# Features réelles V12b
Xr_d = np.zeros(ns * N_WEEKS); Xr_n = np.zeros(ns * N_WEEKS)
for i, s in enumerate(SN_rural):
    if s not in obs_raw: continue
    dd = np.nan_to_num(obs_raw[s]['delta_day'])
    dn = np.nan_to_num(obs_raw[s]['delta_night'])
    for w in range(N_WEEKS):
        cla_d_r = CLA_day_wk[s][w] / CLA_day_ref_sub[w] if CLA_day_ref_sub[w] > 0 else 1.0
        cla_n_r = CLA_night_wk[s][w] / CLA_night_ref_sub[w] if CLA_night_ref_sub[w] > 0 else 1.0
        Xr_d[i * N_WEEKS + w] = dd[w] * cla_d_r
        Xr_n[i * N_WEEKS + w] = dn[w] * cla_n_r
Xr = np.concatenate([Xr_d, Xr_n])
Xrs = (Xr - sX.mean_) / sX.scale_

Yr = pinn(Xrs.reshape(1, -1), training=False).numpy()
Yri = Yr * sY.scale_ + sY.mean_
alpha_pred = Yri[0, :N_STATE_MO].reshape(N_REG, N_MO)
beta_pred = Yri[0, N_STATE_MO]
print(f"  alpha={alpha_pred.mean():.4f}, beta={beta_pred:.4f}")

# ============================================================
# 4. RECONSTRUIRE ΔC_mod ET COMPARER AVEC ΔC_obs
# ============================================================
print("\n4. Reconstruction et comparaison...")

# Pour chaque station et chaque semaine :
#   ΔC_mod(s,w) = Σ_ij H(s,w,i,j) × [α(r(i,j), m(w)) × F_foss(w,i,j) + β × F_bio(w,i,j)]
#   ΔC_obs(s,w) = CO2_obs(s,w) - CO2_MHD(w)

results_by_station = {}

for st in SN_ALL:
    if st not in obs_raw: continue
    
    dc_obs_d = obs_raw[st]['delta_day']   # (52,) — NaN pour les semaines sans données
    dc_obs_n = obs_raw[st]['delta_night']
    
    dc_mod_d = np.zeros(N_WEEKS)
    dc_mod_n = np.zeros(N_WEEKS)
    dc_prior_d = np.zeros(N_WEEKS)  # prior sans correction (α=1, β=1)
    dc_prior_n = np.zeros(N_WEEKS)
    
    for w in range(N_WEEKS):
        doy = w * 7 + 3
        mo = min(np.searchsorted(cumdays[1:], doy + 1), 11)
        
        # Flux corrigé (α, β prédits)
        flux_corr_d = np.zeros((n_lat, n_lon))
        flux_corr_n = np.zeros((n_lat, n_lon))
        flux_prior_d = np.zeros((n_lat, n_lon))
        flux_prior_n = np.zeros((n_lat, n_lon))
        
        for r in range(N_REG):
            rmask = (region_map == r)
            flux_corr_d[rmask] = alpha_pred[r, mo] * fossil_wk[w][rmask]
            flux_corr_n[rmask] = alpha_pred[r, mo] * fossil_wk[w][rmask]
            flux_prior_d[rmask] = fossil_wk[w][rmask]  # α=1
            flux_prior_n[rmask] = fossil_wk[w][rmask]
        
        flux_corr_d += beta_pred * vprm_day_wk[w] + ocean_wk[w]
        flux_corr_n += beta_pred * vprm_night_wk[w] + ocean_wk[w]
        flux_prior_d += vprm_day_wk[w] + ocean_wk[w]  # β=1
        flux_prior_n += vprm_night_wk[w] + ocean_wk[w]
        
        dc_mod_d[w] = np.sum(fp_day_wk[st][w] * flux_corr_d)
        dc_mod_n[w] = np.sum(fp_night_wk[st][w] * flux_corr_n)
        dc_prior_d[w] = np.sum(fp_day_wk[st][w] * flux_prior_d)
        dc_prior_n[w] = np.sum(fp_night_wk[st][w] * flux_prior_n)
    
    # Combiner jour+nuit en moyenne
    dc_mod = (dc_mod_d + dc_mod_n) / 2
    dc_prior = (dc_prior_d + dc_prior_n) / 2
    dc_obs = np.nanmean([dc_obs_d, dc_obs_n], axis=0)  # NaN si les deux sont NaN
    
    # Masquer les NaN
    valid = ~np.isnan(dc_obs)
    n_valid = valid.sum()
    
    if n_valid > 5:
        obs_v = dc_obs[valid]
        mod_v = dc_mod[valid]
        prior_v = dc_prior[valid]
        
        # Normaliser pour comparer les FORMES (pas les amplitudes absolues)
        # car H×F donne des micro-unités, obs donne des ppm
        obs_norm = (obs_v - obs_v.mean()) / max(obs_v.std(), 1e-10)
        mod_norm = (mod_v - mod_v.mean()) / max(mod_v.std(), 1e-10)
        prior_norm = (prior_v - prior_v.mean()) / max(prior_v.std(), 1e-10)
        
        r_mod = np.corrcoef(obs_norm, mod_norm)[0, 1]
        r_prior = np.corrcoef(obs_norm, prior_norm)[0, 1]
        
        results_by_station[st] = {
            'r_mod': r_mod, 'r_prior': r_prior,
            'n_valid': n_valid,
            'obs_norm': obs_norm, 'mod_norm': mod_norm, 'prior_norm': prior_norm,
            'dc_obs': obs_v, 'dc_mod': mod_v,
            'weeks_valid': np.where(valid)[0],
            'is_urban': st in EXCLUDE_V12b
        }

# ============================================================
# 5. RÉSULTATS
# ============================================================
print(f"\n{'='*60}")
print("5. VALIDATION FORWARD — C_mod vs C_obs")
print(f"{'='*60}")

print(f"\n  {'Station':<6} {'r_V12':>8} {'r_prior':>8} {'Delta':>8} {'N':>5} {'Type':>8}")
print(f"  {'-'*45}")

r_mod_all = []; r_prior_all = []
r_mod_rural = []; r_mod_urban = []

for st in SN_ALL:
    if st not in results_by_station: continue
    res = results_by_station[st]
    delta = res['r_mod'] - res['r_prior']
    typ = "URBAIN" if res['is_urban'] else "rural"
    print(f"  {st:<6} {res['r_mod']:>8.3f} {res['r_prior']:>8.3f} {delta:>+8.3f} {res['n_valid']:>5} {typ:>8}")
    
    r_mod_all.append(res['r_mod'])
    r_prior_all.append(res['r_prior'])
    if res['is_urban']:
        r_mod_urban.append(res['r_mod'])
    else:
        r_mod_rural.append(res['r_mod'])

mean_mod = np.mean(r_mod_all)
mean_prior = np.mean(r_prior_all)
mean_rural = np.mean(r_mod_rural) if r_mod_rural else 0
mean_urban = np.mean(r_mod_urban) if r_mod_urban else 0

print(f"\n  MOYENNES:")
print(f"  {'V12 (alpha,beta predits)':<30} r={mean_mod:.3f}")
print(f"  {'Prior (alpha=1, beta=1)':<30} r={mean_prior:.3f}")
print(f"  {'Amelioration V12 vs prior':<30} {mean_mod - mean_prior:+.3f}")
print(f"\n  Rural (19 st):  r={mean_rural:.3f}")
print(f"  Urbain (6 st):  r={mean_urban:.3f}")

# ============================================================
# 6. FIGURE
# ============================================================
print(f"\n6. Figure...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 6.1 Barres par station : r_V12 vs r_prior
sts_sorted = sorted(results_by_station.keys(), key=lambda s: -results_by_station[s]['r_mod'])
x = np.arange(len(sts_sorted))
v_mod = [results_by_station[s]['r_mod'] for s in sts_sorted]
v_prior = [results_by_station[s]['r_prior'] for s in sts_sorted]
colors_bar = ['red' if results_by_station[s]['is_urban'] else 'green' for s in sts_sorted]

axes[0, 0].bar(x - 0.2, v_mod, 0.4, label='V12 (corrige)', color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.3)
axes[0, 0].bar(x + 0.2, v_prior, 0.4, label='Prior (sans correction)', color='lightgray', edgecolor='black', linewidth=0.3)
axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(sts_sorted, rotation=90, fontsize=7)
axes[0, 0].axhline(mean_mod, color='green', linestyle='--', linewidth=1, label=f'Moy V12={mean_mod:.3f}')
axes[0, 0].set_ylabel('Correlation r')
axes[0, 0].set_title('Validation forward par station\n(vert=rural, rouge=urbain)', fontweight='bold')
axes[0, 0].legend(fontsize=7, loc='lower left'); axes[0, 0].grid(True, alpha=0.3, axis='y')

# 6.2 Séries temporelles pour 4 stations exemples
examples = []
for s in ['TRN', 'PUY', 'OPE', 'GAT']:
    if s in results_by_station: examples.append(s)
if len(examples) < 4:
    examples = list(results_by_station.keys())[:4]

for idx, st in enumerate(examples[:4]):
    row, col = idx // 2, idx % 2 + 1
    if idx >= 2: row = 1; col = idx - 2
    ax = axes[0, 1] if idx == 0 else axes[0, 2] if idx == 1 else axes[1, 0] if idx == 2 else axes[1, 1]
    
    res = results_by_station[st]
    weeks_v = res['weeks_valid']
    ax.plot(weeks_v, res['obs_norm'], 'k-o', markersize=3, linewidth=1, label='Obs ICOS', alpha=0.7)
    ax.plot(weeks_v, res['mod_norm'], 'g-', linewidth=1.5, label=f'V12 (r={res["r_mod"]:.2f})', alpha=0.8)
    ax.plot(weeks_v, res['prior_norm'], 'b--', linewidth=1, label=f'Prior (r={res["r_prior"]:.2f})', alpha=0.5)
    ax.set_xlabel('Semaine', fontsize=9)
    ax.set_ylabel('Normalise', fontsize=9)
    typ = " [URBAIN]" if res['is_urban'] else ""
    ax.set_title(f'{st}{typ}: r_V12={res["r_mod"]:.3f}', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

# 6.5 Comparaison globale
axes[1, 1].bar([0, 1, 2], [mean_prior, mean_mod, mean_rural],
               color=['lightgray', 'steelblue', 'green'], edgecolor='black', width=0.6)
axes[1, 1].set_xticks([0, 1, 2])
axes[1, 1].set_xticklabels(['Prior\n(sans corr.)', 'V12\n(25 st.)', 'V12\n(rural)'], fontsize=9)
for i, v in enumerate([mean_prior, mean_mod, mean_rural]):
    axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
axes[1, 1].set_ylabel('Correlation r moyenne')
axes[1, 1].set_title('V12 vs Prior sur obs reelles', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# 6.6 Résumé
txt = (f"VALIDATION FORWARD\n{'='*30}\n\n"
       f"C_mod = H x (a*F_foss + b*F_bio)\n"
       f"vs C_obs ICOS reelles\n\n"
       f"V12 (corrige):\n"
       f"  Toutes: r={mean_mod:.3f}\n"
       f"  Rural:  r={mean_rural:.3f}\n"
       f"  Urbain: r={mean_urban:.3f}\n\n"
       f"Prior (a=1, b=1):\n"
       f"  Toutes: r={mean_prior:.3f}\n\n"
       f"Amelioration: {mean_mod-mean_prior:+.3f}\n\n"
       f"alpha={alpha_pred.mean():.3f}\n"
       f"beta={beta_pred:.3f}")

axes[1, 2].text(0.05, 0.95, txt, transform=axes[1, 2].transAxes, fontsize=9.5,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[1, 2].axis('off')
axes[1, 2].set_title('Resume', fontweight='bold')

plt.suptitle('Validation forward : le modele reproduit-il les observations ICOS ?\n'
             f'r moyen V12 = {mean_mod:.3f} (prior = {mean_prior:.3f}, amelioration = {mean_mod-mean_prior:+.3f})',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
figpath = os.path.join(OUTDIR, 'fig_validation_forward.png')
plt.savefig(figpath, dpi=150, bbox_inches='tight')
print(f"  Figure: {figpath}")

np.savez(os.path.join(OUTDIR, 'validation_forward.npz'),
         results=results_by_station, mean_mod=mean_mod, mean_prior=mean_prior,
         alpha_pred=alpha_pred, beta_pred=beta_pred)

print(f"\n{'='*60}")
print(f"  V12 sur obs reelles: r={mean_mod:.3f}")
print(f"  Prior sans correction: r={mean_prior:.3f}")
print(f"  Amelioration: {mean_mod-mean_prior:+.3f}")
if mean_mod > mean_prior:
    print(f"  Le modele AMELIORE la reconstruction des observations")
else:
    print(f"  Le prior seul fait aussi bien")
print(f"{'='*60}")

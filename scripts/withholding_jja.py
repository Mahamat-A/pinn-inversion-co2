#!/usr/bin/env python3
"""
VALIDATION TEMPORELLE CORRIGÉE — Withholding sans Domain Shift
===============================================================
Règle d'or : masquer à l'entraînement CE QU'ON masque à l'inférence.

Protocole :
  1. Générer les scénarios synthétiques (α, β → concentrations 52 semaines)
  2. MASQUER les semaines JJA (22-39) dans les FEATURES d'entraînement
     → Le réseau apprend à prédire α, β à partir de 9 mois seulement
  3. À l'inférence : masquer JJA dans les obs ICOS réelles (même distribution)
  4. Prédire α, β à partir des 9 mois restants
  5. Reconstruire C_mod pour TOUTES les semaines (y compris JJA)
  6. Comparer C_mod(JJA) vs C_obs(JJA) retenues

Si ça corrèle → le modèle prédit l'été sans l'avoir vu.

Lance : python3 withholding_jja.py
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

# JJA = semaines 22-39 (juin-sept élargi)
JJA_START = 22; JJA_END = 39
JJA_WEEKS = set(range(JJA_START, JJA_END))

print("=" * 60)
print("WITHHOLDING CORRIGÉ — Masquer JJA à l'entraînement ET inférence")
print(f"Semaines masquées : {JJA_START}-{JJA_END-1} ({len(JJA_WEEKS)} semaines)")
print("=" * 60)

# ============================================================
# 1. CHARGEMENT
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

CLA_day_wk = {}; CLA_night_wk = {}
for st, (fname, slat, slon) in STATIONS_ALL.items():
    ilat = np.argmin(np.abs(lat_era - slat)); ilon = np.argmin(np.abs(lon_era - slon))
    cd = np.zeros(N_WEEKS); cn = np.zeros(N_WEEKS)
    for w in range(N_WEEKS):
        wm = week_all == w
        di = np.where(wm & day_mask_h)[0]; ni = np.where(wm & night_mask_h)[0]
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
print(f"  {ns} stations rurales")

# ============================================================
# 2. SCÉNARIOS AVEC JJA MASQUÉ DANS LES FEATURES
# ============================================================
print("\n2. Scénarios (JJA masqué dans les features d'entraînement)...")
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

# Features V12b AVEC JJA MASQUÉ
# La clé : mettre les semaines JJA à zéro dans les features d'entraînement
CLA_day_ref_sub = np.array([np.mean([CLA_day_wk[s][w] for s in SN_rural]) for w in range(N_WEEKS)])
CLA_night_ref_sub = np.array([np.mean([CLA_night_wk[s][w] for s in SN_rural]) for w in range(N_WEEKS)])
cbgs_d = np.mean([co_day_wk[s] for s in ['PUY', 'RGL']], axis=0)
cbgs_n = np.mean([co_night_wk[s] for s in ['PUY', 'RGL']], axis=0)

X_day = np.zeros((N_SCENARIOS, ns * N_WEEKS))
X_night = np.zeros((N_SCENARIOS, ns * N_WEEKS))
for i, s in enumerate(SN_rural):
    dc_d = co_day_wk[s] - cbgs_d; dc_n = co_night_wk[s] - cbgs_n
    for w in range(N_WEEKS):
        if w in JJA_WEEKS:
            # MASQUER : mettre à zéro les semaines JJA
            X_day[:, i * N_WEEKS + w] = 0.0
            X_night[:, i * N_WEEKS + w] = 0.0
        else:
            cla_d_r = CLA_day_wk[s][w] / CLA_day_ref_sub[w] if CLA_day_ref_sub[w] > 0 else 1.0
            cla_n_r = CLA_night_wk[s][w] / CLA_night_ref_sub[w] if CLA_night_ref_sub[w] > 0 else 1.0
            X_day[:, i * N_WEEKS + w] = dc_d[:, w] * cla_d_r
            X_night[:, i * N_WEEKS + w] = dc_n[:, w] * cla_n_r

noise_d = np.random.normal(0, 0.02, size=X_day.shape) * np.abs(X_day.mean())
noise_n = np.random.normal(0, 0.02, size=X_night.shape) * np.abs(X_night.mean())
X_masked = np.concatenate([X_day + noise_d, X_night + noise_n], axis=1)

n_zero = np.sum(X_masked[0] == 0)
n_total = X_masked.shape[1]
print(f"  Features: {n_total} dimensions, {n_zero} masquées (JJA)")
print(f"  Le réseau apprend à prédire α/β à partir de 9 mois seulement")

# AUSSI construire les features COMPLÈTES (pour la baseline)
X_day_full = np.zeros((N_SCENARIOS, ns * N_WEEKS))
X_night_full = np.zeros((N_SCENARIOS, ns * N_WEEKS))
for i, s in enumerate(SN_rural):
    dc_d = co_day_wk[s] - cbgs_d; dc_n = co_night_wk[s] - cbgs_n
    for w in range(N_WEEKS):
        cla_d_r = CLA_day_wk[s][w] / CLA_day_ref_sub[w] if CLA_day_ref_sub[w] > 0 else 1.0
        cla_n_r = CLA_night_wk[s][w] / CLA_night_ref_sub[w] if CLA_night_ref_sub[w] > 0 else 1.0
        X_day_full[:, i * N_WEEKS + w] = dc_d[:, w] * cla_d_r
        X_night_full[:, i * N_WEEKS + w] = dc_n[:, w] * cla_n_r
X_full = np.concatenate([X_day_full + noise_d, X_night_full + noise_n], axis=1)

# ============================================================
# 3. ENTRAÎNER DEUX PINN : masqué et complet
# ============================================================
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

# A. PINN masqué (JJA absent)
print("\n3A. Entraînement PINN MASQUÉ (sans JJA)...")
sX_m = StandardScaler(); sY_m = StandardScaler()
Xs_m = sX_m.fit_transform(X_masked); Ys_m = sY_m.fit_transform(Y_all)
Xt_m, Xv_m, Yt_m, Yv_m = train_test_split(Xs_m, Ys_m, test_size=0.15, random_state=42)

pinn_masked = build_pinn(X_masked.shape[1])
pinn_masked.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss=jloss, metrics=['mae'])
pinn_masked.fit(Xt_m, Yt_m, validation_data=(Xv_m, Yv_m), batch_size=128, epochs=200, verbose=0,
                callbacks=[EarlyStopping(patience=25, restore_best_weights=True),
                           ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)])
Yp_m = pinn_masked.predict(Xv_m, verbose=0)
Ypi_m = sY_m.inverse_transform(Yp_m); Yvi_m = sY_m.inverse_transform(Yv_m)
r_masked = np.corrcoef(Yvi_m[:, :N_STATE_MO].flatten(), Ypi_m[:, :N_STATE_MO].flatten())[0, 1]
print(f"  PINN masqué: α r={r_masked:.4f}")

# B. PINN complet (baseline)
print("\n3B. Entraînement PINN COMPLET (baseline 12 mois)...")
sX_f = StandardScaler(); sY_f = StandardScaler()
Xs_f = sX_f.fit_transform(X_full); Ys_f = sY_f.fit_transform(Y_all)
Xt_f, Xv_f, Yt_f, Yv_f = train_test_split(Xs_f, Ys_f, test_size=0.15, random_state=42)

pinn_full = build_pinn(X_full.shape[1])
pinn_full.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss=jloss, metrics=['mae'])
pinn_full.fit(Xt_f, Yt_f, validation_data=(Xv_f, Yv_f), batch_size=128, epochs=200, verbose=0,
              callbacks=[EarlyStopping(patience=25, restore_best_weights=True),
                         ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)])
Yp_f = pinn_full.predict(Xv_f, verbose=0)
Ypi_f = sY_f.inverse_transform(Yp_f); Yvi_f = sY_f.inverse_transform(Yv_f)
r_full = np.corrcoef(Yvi_f[:, :N_STATE_MO].flatten(), Ypi_f[:, :N_STATE_MO].flatten())[0, 1]
print(f"  PINN complet: α r={r_full:.4f}")

print(f"\n  Dégradation par masquage JJA: {r_masked - r_full:+.4f}")

# ============================================================
# 4. CHARGER OBSERVATIONS ICOS RÉELLES
# ============================================================
print("\n4. Chargement observations ICOS réelles...")

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

obs_raw = {}
for s in SN_ALL:
    fpath = os.path.join(ICOS_DIR, STATIONS_ALL[s][0])
    if os.path.exists(fpath):
        od, on = load_icos_weekly(fpath)
        obs_raw[s] = {'day': od, 'night': on, 'delta_day': od - mhd_d, 'delta_night': on - mhd_n}

print(f"  {len(obs_raw)} stations chargées")

# ============================================================
# 5. PRÉDIRE α, β AVEC LES DEUX PINN
# ============================================================
print("\n5. Prédiction α, β...")

def build_real_features(station_list, mask_jja=False):
    Xr_d = np.zeros(ns * N_WEEKS); Xr_n = np.zeros(ns * N_WEEKS)
    for i, s in enumerate(station_list):
        if s not in obs_raw: continue
        dd = np.nan_to_num(obs_raw[s]['delta_day'])
        dn = np.nan_to_num(obs_raw[s]['delta_night'])
        for w in range(N_WEEKS):
            if mask_jja and w in JJA_WEEKS:
                Xr_d[i * N_WEEKS + w] = 0.0
                Xr_n[i * N_WEEKS + w] = 0.0
            else:
                cla_d_r = CLA_day_wk[s][w] / CLA_day_ref_sub[w] if CLA_day_ref_sub[w] > 0 else 1.0
                cla_n_r = CLA_night_wk[s][w] / CLA_night_ref_sub[w] if CLA_night_ref_sub[w] > 0 else 1.0
                Xr_d[i * N_WEEKS + w] = dd[w] * cla_d_r
                Xr_n[i * N_WEEKS + w] = dn[w] * cla_n_r
    return np.concatenate([Xr_d, Xr_n])

# A. PINN masqué + obs masquées (withholding)
Xr_masked = build_real_features(SN_rural, mask_jja=True)
Xrs_m = (Xr_masked - sX_m.mean_) / sX_m.scale_
Yr_m = pinn_masked(Xrs_m.reshape(1, -1), training=False).numpy()
Yri_m = Yr_m * sY_m.scale_ + sY_m.mean_
alpha_withheld = Yri_m[0, :N_STATE_MO].reshape(N_REG, N_MO)
beta_withheld = Yri_m[0, N_STATE_MO]

# B. PINN complet + obs complètes (baseline)
Xr_full = build_real_features(SN_rural, mask_jja=False)
Xrs_f = (Xr_full - sX_f.mean_) / sX_f.scale_
Yr_f = pinn_full(Xrs_f.reshape(1, -1), training=False).numpy()
Yri_f = Yr_f * sY_f.scale_ + sY_f.mean_
alpha_baseline = Yri_f[0, :N_STATE_MO].reshape(N_REG, N_MO)
beta_baseline = Yri_f[0, N_STATE_MO]

print(f"  Withholding: α={alpha_withheld.mean():.4f}, β={beta_withheld:.4f}")
print(f"  Baseline:    α={alpha_baseline.mean():.4f}, β={beta_baseline:.4f}")
r_alpha_wh = np.corrcoef(alpha_withheld.flatten(), alpha_baseline.flatten())[0, 1]
print(f"  Corrélation α(withholding) vs α(baseline): r={r_alpha_wh:.3f}")

# ============================================================
# 6. RECONSTRUIRE C_mod ET COMPARER SUR JJA
# ============================================================
print("\n6. Reconstruction et comparaison sur JJA retenu...")

def reconstruct_and_compare(alpha, beta, label):
    """Reconstruit C_mod et compare avec C_obs sur les semaines JJA."""
    results = {}
    for st in SN_ALL:
        if st not in obs_raw: continue
        dc_obs_d = obs_raw[st]['delta_day']
        
        dc_mod_d = np.zeros(N_WEEKS)
        for w in range(N_WEEKS):
            doy = w * 7 + 3
            mo = min(np.searchsorted(cumdays[1:], doy + 1), 11)
            flux = np.zeros((n_lat, n_lon))
            for r in range(N_REG):
                rmask = (region_map == r)
                flux[rmask] = alpha[r, mo] * fossil_wk[w][rmask]
            flux += beta * vprm_day_wk[w] + ocean_wk[w]
            dc_mod_d[w] = np.sum(fp_day_wk[st][w] * flux)
        
        # Corrélation sur JJA uniquement
        obs_jja = []; mod_jja = []
        for w in sorted(JJA_WEEKS):
            if not np.isnan(dc_obs_d[w]):
                obs_jja.append(dc_obs_d[w])
                mod_jja.append(dc_mod_d[w])
        
        # Corrélation sur toutes les semaines
        obs_all = []; mod_all = []
        for w in range(N_WEEKS):
            if not np.isnan(dc_obs_d[w]):
                obs_all.append(dc_obs_d[w])
                mod_all.append(dc_mod_d[w])
        
        if len(obs_jja) > 5 and len(obs_all) > 10:
            obs_jja = np.array(obs_jja); mod_jja = np.array(mod_jja)
            obs_all = np.array(obs_all); mod_all = np.array(mod_all)
            
            # Normaliser
            on_j = (obs_jja - obs_jja.mean()) / max(obs_jja.std(), 1e-10)
            mn_j = (mod_jja - mod_jja.mean()) / max(mod_jja.std(), 1e-10)
            on_a = (obs_all - obs_all.mean()) / max(obs_all.std(), 1e-10)
            mn_a = (mod_all - mod_all.mean()) / max(mod_all.std(), 1e-10)
            
            r_jja = np.corrcoef(on_j, mn_j)[0, 1]
            r_all = np.corrcoef(on_a, mn_a)[0, 1]
            
            results[st] = {'r_jja': r_jja, 'r_all': r_all, 
                           'n_jja': len(obs_jja), 'n_all': len(obs_all),
                           'is_urban': st in EXCLUDE_V12b}
    return results

res_withheld = reconstruct_and_compare(alpha_withheld, beta_withheld, "withholding")
res_baseline = reconstruct_and_compare(alpha_baseline, beta_baseline, "baseline")

# ============================================================
# 7. RÉSULTATS
# ============================================================
print(f"\n{'='*60}")
print("7. RÉSULTATS WITHHOLDING")
print(f"{'='*60}")

print(f"\n  {'Station':<6} {'WH_JJA':>8} {'BL_JJA':>8} {'Delta':>8} {'WH_ALL':>8} {'BL_ALL':>8} {'Type':>8}")
print(f"  {'-'*60}")

r_wh_jja_list = []; r_bl_jja_list = []
r_wh_all_list = []; r_bl_all_list = []

for st in SN_ALL:
    if st not in res_withheld or st not in res_baseline: continue
    rw = res_withheld[st]; rb = res_baseline[st]
    delta = rw['r_jja'] - rb['r_jja']
    typ = "URBAIN" if rw['is_urban'] else "rural"
    print(f"  {st:<6} {rw['r_jja']:>8.3f} {rb['r_jja']:>8.3f} {delta:>+8.3f} "
          f"{rw['r_all']:>8.3f} {rb['r_all']:>8.3f} {typ:>8}")
    
    r_wh_jja_list.append(rw['r_jja']); r_bl_jja_list.append(rb['r_jja'])
    r_wh_all_list.append(rw['r_all']); r_bl_all_list.append(rb['r_all'])

mean_wh_jja = np.mean(r_wh_jja_list)
mean_bl_jja = np.mean(r_bl_jja_list)
mean_wh_all = np.mean(r_wh_all_list)
mean_bl_all = np.mean(r_bl_all_list)

print(f"\n  MOYENNES:")
print(f"  {'Config':<25} {'JJA':>10} {'Toutes':>10}")
print(f"  {'-'*47}")
print(f"  {'Withholding (sans JJA)':<25} {mean_wh_jja:>10.3f} {mean_wh_all:>10.3f}")
print(f"  {'Baseline (12 mois)':<25} {mean_bl_jja:>10.3f} {mean_bl_all:>10.3f}")
print(f"  {'Delta':<25} {mean_wh_jja - mean_bl_jja:>+10.3f} {mean_wh_all - mean_bl_all:>+10.3f}")

# ============================================================
# 8. FIGURE
# ============================================================
print(f"\n8. Figure...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
ml = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

# 8.1 Barres par station : JJA withholding vs baseline
sts_s = sorted(res_withheld.keys(), key=lambda s: -res_withheld[s]['r_jja'])
x = np.arange(len(sts_s))
v_wh = [res_withheld[s]['r_jja'] for s in sts_s]
v_bl = [res_baseline[s]['r_jja'] for s in sts_s]
colors_bar = ['red' if res_withheld[s]['is_urban'] else 'green' for s in sts_s]

axes[0, 0].bar(x - 0.2, v_wh, 0.4, label=f'Sans été (r={mean_wh_jja:.3f})', color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.3)
axes[0, 0].bar(x + 0.2, v_bl, 0.4, label=f'Baseline (r={mean_bl_jja:.3f})', color='lightgray', edgecolor='black', linewidth=0.3)
axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(sts_s, rotation=90, fontsize=7)
axes[0, 0].axhline(mean_wh_jja, color='green', linewidth=1, linestyle='--')
axes[0, 0].set_ylabel('r (semaines JJA)')
axes[0, 0].set_title('Withholding par station\n(corrélation sur JJA retenu)', fontweight='bold')
axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3, axis='y')

# 8.2 Alpha withholding vs baseline
axes[0, 1].plot(range(12), alpha_withheld.mean(axis=0), 'r-s', linewidth=2, label='Sans été')
axes[0, 1].plot(range(12), alpha_baseline.mean(axis=0), 'b-o', linewidth=2, label='Baseline')
axes[0, 1].axhline(1.0, color='k', linewidth=0.5, linestyle=':')
axes[0, 1].axvspan(5, 8, alpha=0.15, color='red', label='JJA masqué')
axes[0, 1].set_xticks(range(12)); axes[0, 1].set_xticklabels(ml)
axes[0, 1].set_ylabel('α')
axes[0, 1].set_title(f'Alpha: sans été vs baseline\nr(α)={r_alpha_wh:.3f}', fontweight='bold')
axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

# 8.3 Série temporelle exemple
example = 'TRN' if 'TRN' in res_withheld else list(res_withheld.keys())[0]
st_ex = example
dc_obs_ex = obs_raw[st_ex]['delta_day']
# Reconstruire pour cet exemple
dc_mod_wh = np.zeros(N_WEEKS); dc_mod_bl = np.zeros(N_WEEKS)
for w in range(N_WEEKS):
    doy = w * 7 + 3; mo = min(np.searchsorted(cumdays[1:], doy + 1), 11)
    flux_wh = np.zeros((n_lat, n_lon)); flux_bl = np.zeros((n_lat, n_lon))
    for r in range(N_REG):
        rmask = (region_map == r)
        flux_wh[rmask] = alpha_withheld[r, mo] * fossil_wk[w][rmask]
        flux_bl[rmask] = alpha_baseline[r, mo] * fossil_wk[w][rmask]
    flux_wh += beta_withheld * vprm_day_wk[w] + ocean_wk[w]
    flux_bl += beta_baseline * vprm_day_wk[w] + ocean_wk[w]
    dc_mod_wh[w] = np.sum(fp_day_wk[st_ex][w] * flux_wh)
    dc_mod_bl[w] = np.sum(fp_day_wk[st_ex][w] * flux_bl)

# Normaliser
valid = ~np.isnan(dc_obs_ex)
obs_v = dc_obs_ex.copy(); obs_v[~valid] = np.nan
obs_n = (obs_v - np.nanmean(obs_v)) / max(np.nanstd(obs_v), 1e-10)
mod_wh_n = (dc_mod_wh - dc_mod_wh.mean()) / max(dc_mod_wh.std(), 1e-10)
mod_bl_n = (dc_mod_bl - dc_mod_bl.mean()) / max(dc_mod_bl.std(), 1e-10)

weeks = np.arange(N_WEEKS)
axes[0, 2].plot(weeks, obs_n, 'k-o', markersize=3, linewidth=1, label='Obs ICOS', alpha=0.7)
axes[0, 2].plot(weeks, mod_wh_n, 'r-', linewidth=1.5, label='Sans été', alpha=0.7)
axes[0, 2].plot(weeks, mod_bl_n, 'b--', linewidth=1, label='Baseline', alpha=0.5)
axes[0, 2].axvspan(JJA_START, JJA_END, alpha=0.15, color='red')
axes[0, 2].set_xlabel('Semaine'); axes[0, 2].set_ylabel('Normalisé')
axes[0, 2].set_title(f'{st_ex}: série temporelle\n(zone rouge = JJA retenu)', fontweight='bold')
axes[0, 2].legend(fontsize=7); axes[0, 2].grid(True, alpha=0.2)

# 8.4 Comparaison PINN synth
axes[1, 0].bar([0, 1], [r_full, r_masked], color=['steelblue', 'red'], edgecolor='black', width=0.6)
for i, v in enumerate([r_full, r_masked]):
    axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
axes[1, 0].set_xticks([0, 1]); axes[1, 0].set_xticklabels(['PINN 12 mois', 'PINN sans JJA'])
axes[1, 0].set_ylabel('α r (synthétique)')
axes[1, 0].set_title(f'Performance PINN synthétique\nDégradation: {r_masked-r_full:+.3f}', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 8.5 Comparaison globale
labels_g = ['Baseline\nJJA', 'Withheld\nJJA', 'Baseline\nTous', 'Withheld\nTous']
vals_g = [mean_bl_jja, mean_wh_jja, mean_bl_all, mean_wh_all]
cols_g = ['steelblue', 'red', 'steelblue', 'red']
axes[1, 1].bar(range(4), vals_g, color=cols_g, edgecolor='black', width=0.6, alpha=0.8)
for i, v in enumerate(vals_g):
    axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=10)
axes[1, 1].set_xticks(range(4)); axes[1, 1].set_xticklabels(labels_g, fontsize=8)
axes[1, 1].set_ylabel('r moyen')
axes[1, 1].set_title('Comparaison globale\n(obs réelles vs C_mod)', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# 8.6 Résumé
txt = (f"WITHHOLDING CORRIGE\n{'='*30}\n\n"
       f"Protocole:\n"
       f"  JJA masque a l'entrainement\n"
       f"  ET a l'inference\n"
       f"  (pas de domain shift)\n\n"
       f"PINN synthetique:\n"
       f"  Complet: r={r_full:.3f}\n"
       f"  Masque:  r={r_masked:.3f}\n"
       f"  Delta:   {r_masked-r_full:+.3f}\n\n"
       f"Obs reelles (JJA):\n"
       f"  Baseline: r={mean_bl_jja:.3f}\n"
       f"  Withheld: r={mean_wh_jja:.3f}\n"
       f"  Delta:    {mean_wh_jja-mean_bl_jja:+.3f}\n\n"
       f"alpha WH: {alpha_withheld.mean():.3f}\n"
       f"alpha BL: {alpha_baseline.mean():.3f}\n"
       f"r(a_WH,a_BL)={r_alpha_wh:.3f}")

axes[1, 2].text(0.05, 0.95, txt, transform=axes[1, 2].transAxes, fontsize=9.5,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[1, 2].axis('off')
axes[1, 2].set_title('Résumé', fontweight='bold')

plt.suptitle(f'Withholding corrigé : prédire l\'été sans l\'avoir vu\n'
             f'JJA withheld r={mean_wh_jja:.3f} (baseline r={mean_bl_jja:.3f})',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
figpath = os.path.join(OUTDIR, 'fig_withholding_jja.png')
plt.savefig(figpath, dpi=150, bbox_inches='tight')
print(f"  Figure: {figpath}")

np.savez(os.path.join(OUTDIR, 'withholding_jja.npz'),
         r_masked_synth=r_masked, r_full_synth=r_full,
         mean_wh_jja=mean_wh_jja, mean_bl_jja=mean_bl_jja,
         mean_wh_all=mean_wh_all, mean_bl_all=mean_bl_all,
         alpha_withheld=alpha_withheld, alpha_baseline=alpha_baseline,
         beta_withheld=beta_withheld, beta_baseline=beta_baseline,
         r_alpha_wh=r_alpha_wh)

print(f"\n{'='*60}")
print(f"  PINN synthétique: masqué={r_masked:.3f}, complet={r_full:.3f}")
print(f"  Obs réelles JJA: withheld={mean_wh_jja:.3f}, baseline={mean_bl_jja:.3f}")
if mean_wh_jja > 0.2:
    print(f"  Le modèle PRÉDIT l'été sans l'avoir vu")
else:
    print(f"  Le modèle ne prédit PAS l'été retenu")
print(f"{'='*60}")

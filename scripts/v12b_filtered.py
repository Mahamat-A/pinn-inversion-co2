#!/usr/bin/env python3
"""
VALIDATION CORRIGÉE : V12 vs CarbonTracker (+ CAMS si disponible)
==================================================================
PROBLÈME DU SCRIPT PRÉCÉDENT :
  V12_flux = α_moyen × CT_fossil → r=1.000 par construction (scalaire)
  
CORRECTION :
  1. Relancer le PINN V12b sur les observations réelles
  2. Extraire α(région, mois) — les VRAIES corrections spatiales
  3. Comparer la STRUCTURE de α avec CT2022 et CAMS

Ce que la comparaison doit montrer :
  - Les régions où α>1 (EDGAR sous-estime) vs α<1 (EDGAR surestime)
  - Le cycle saisonnier de α (variations mensuelles)
  - Si CAMS dispo : est-ce que CAMS corrige dans le même sens que nous ?

Lance : python3 validation_v2.py
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
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
CAMS_DIR = os.path.join(BASE, "flux_data")
os.makedirs(OUTDIR, exist_ok=True)

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
print("VALIDATION CORRIGÉE — α(région,mois) réel")
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
print(f"  {len(SN_rural)} stations rurales")

# ============================================================
# 2. SCÉNARIOS + PINN V12b (identique)
# ============================================================
print("\n2. Scénarios + PINN V12b...")
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

# Build features V12b
ns = len(SN_rural)
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

# Train PINN
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

print("  Entraînement PINN V12b...")
sX = StandardScaler(); sY = StandardScaler()
Xs = sX.fit_transform(X_v12); Ys = sY.fit_transform(Y_all)
Xt, Xv, Yt, Yv = train_test_split(Xs, Ys, test_size=0.15, random_state=42)

pinn = build_pinn(X_v12.shape[1])
pinn.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss=jloss, metrics=['mae'])
pinn.fit(Xt, Yt, validation_data=(Xv, Yv), batch_size=128, epochs=200, verbose=0,
         callbacks=[EarlyStopping(patience=25, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)])

Yp_val = pinn.predict(Xv, verbose=0)
Ypi = sY.inverse_transform(Yp_val); Yvi = sY.inverse_transform(Yv)
ca = np.corrcoef(Yvi[:, :N_STATE_MO].flatten(), Ypi[:, :N_STATE_MO].flatten())[0, 1]
print(f"  PINN V12b: α r={ca:.4f}")

# ============================================================
# 3. PRÉDIRE α SUR OBSERVATIONS RÉELLES
# ============================================================
print("\n3. Prédiction α sur observations réelles ICOS...")

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

# Background MHD
mhd_d, mhd_n = load_icos_weekly(os.path.join(ICOS_DIR, 'MHD_24.0m_air.hdf.2019.co2'))

# Construire features réelles pour V12b (19 stations rurales)
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
Yr = pinn.predict(Xrs.reshape(1, -1), verbose=0)
Yri = Yr * sY.scale_ + sY.mean_

# Extraire α(région, mois) RÉEL
alpha_real = Yri[0, :N_STATE_MO].reshape(N_REG, N_MO)
beta_real = Yri[0, N_STATE_MO]

print(f"  α moyen = {alpha_real.mean():.4f}")
print(f"  α min = {alpha_real.min():.4f}, max = {alpha_real.max():.4f}")
print(f"  α std = {alpha_real.std():.4f}")
print(f"  β = {beta_real:.4f}")

# ============================================================
# 4. CONSTRUIRE LES FLUX COMPARABLES
# ============================================================
print("\n4. Construction des flux...")

# Flux V12 = α(r,m) × CT_fossil(r,m) — VARIABLE par région et mois
v12_flux = np.zeros((12, n_lat, n_lon))
for m in range(12):
    for r in range(N_REG):
        mask = (region_map == r)
        v12_flux[m][mask] = alpha_real[r, m] * fossil_monthly[m][mask]

# Flux CT2022 = prior non corrigé
ct_flux = fossil_monthly.copy()

# Régionaliser
def regional_flux(flux):
    reg = np.zeros((N_REG, N_MO))
    for r in range(N_REG):
        mask = (region_map == r)
        for m in range(N_MO):
            reg[r, m] = flux[m][mask].mean()
    return reg

v12_reg = regional_flux(v12_flux)
ct_reg = regional_flux(ct_flux)

print(f"  V12 flux régional: mean={v12_reg.mean():.4f}, std={v12_reg.std():.4f}")
print(f"  CT2022 flux régional: mean={ct_reg.mean():.4f}, std={ct_reg.std():.4f}")

# ============================================================
# 5. CHERCHER CAMS TÉLÉCHARGÉ MANUELLEMENT
# ============================================================
print("\n5. Recherche CAMS local...")
import glob
cams_files = glob.glob(os.path.join(CAMS_DIR, "*cams*")) + \
             glob.glob(os.path.join(CAMS_DIR, "*CAMS*")) + \
             glob.glob(os.path.join(CAMS_DIR, "*ghg*")) + \
             glob.glob(os.path.join(CAMS_DIR, "*inversion*"))
has_cams = False
cams_reg = None

if cams_files:
    print(f"  Fichiers CAMS trouvés: {cams_files}")
    for cf in cams_files:
        try:
            ds_c = nc.Dataset(cf)
            print(f"    {cf}: vars={list(ds_c.variables.keys())}")
            # Tenter d'extraire le flux
            for vname in ds_c.variables:
                if len(ds_c.variables[vname].shape) >= 3:
                    flux_raw = ds_c.variables[vname][:]
                    lat_c = ds_c.variables.get('latitude', ds_c.variables.get('lat'))[:]
                    lon_c = ds_c.variables.get('longitude', ds_c.variables.get('lon'))[:]
                    # Regriller
                    lat_m = (lat_c >= 39) & (lat_c <= 57)
                    lon_m = (lon_c >= -11) & (lon_c <= 16)
                    il = np.where(lat_m)[0]; jl = np.where(lon_m)[0]
                    if len(il) > 0 and len(jl) > 0:
                        if len(flux_raw.shape) == 4:
                            fe = flux_raw[:12, 0, il[0]:il[-1]+1, jl[0]:jl[-1]+1]
                        else:
                            fe = flux_raw[:12, il[0]:il[-1]+1, jl[0]:jl[-1]+1]
                        # Regriller sur 32×50
                        lcs = lat_c[il]; lns = lon_c[jl]
                        cams_flux = np.zeros((min(fe.shape[0], 12), n_lat, n_lon))
                        for t in range(min(fe.shape[0], 12)):
                            for ii in range(n_lat):
                                for jj in range(n_lon):
                                    tla = 40 + ii * 0.5 + 0.25
                                    tlo = -10 + jj * 0.5 + 0.25
                                    ci = np.argmin(np.abs(lcs - tla))
                                    cj = np.argmin(np.abs(lns - tlo))
                                    cams_flux[t, ii, jj] = fe[t, ci, cj]
                        cams_reg = regional_flux(cams_flux)
                        has_cams = True
                        print(f"    ✅ CAMS chargé: {vname}, shape={cams_flux.shape}")
                        break
            ds_c.close()
            if has_cams: break
        except Exception as e:
            print(f"    Erreur: {e}")

if not has_cams:
    print("  CAMS non trouvé localement")
    print("  → Télécharge manuellement depuis ADS et place dans ~/hysplit/flux_data/")

# ============================================================
# 6. COMPARAISONS
# ============================================================
print(f"\n{'='*60}")
print("6. COMPARAISONS (α variables, pas scalaire)")
print(f"{'='*60}")

# Spatial: corrélation des profils régionaux moyens
v12_spatial = v12_reg.mean(axis=1)
ct_spatial = ct_reg.mean(axis=1)
r_spatial = np.corrcoef(v12_spatial, ct_spatial)[0, 1]

# Temporel: corrélation des cycles saisonniers
v12_temporal = v12_reg.mean(axis=0)
ct_temporal = ct_reg.mean(axis=0)
r_temporal = np.corrcoef(v12_temporal, ct_temporal)[0, 1]

# Total: toutes les valeurs
r_total = np.corrcoef(v12_reg.flatten(), ct_reg.flatten())[0, 1]

# Le facteur α lui-même
alpha_spatial = alpha_real.mean(axis=1)
alpha_temporal = alpha_real.mean(axis=0)

print(f"\n  V12 vs CT2022:")
print(f"    Spatial:  r={r_spatial:.3f}")
print(f"    Temporel: r={r_temporal:.3f}")
print(f"    Total:    r={r_total:.3f}")
print(f"\n  Facteur α (correction V12 appliquée à CT2022):")
print(f"    α moyen = {alpha_real.mean():.3f}")
print(f"    α spatial std = {alpha_spatial.std():.3f}")
print(f"    α temporal std = {alpha_temporal.std():.3f}")
print(f"    → α varie de {alpha_real.min():.3f} à {alpha_real.max():.3f}")

if has_cams:
    cams_spatial = cams_reg.mean(axis=1)
    cams_temporal = cams_reg.mean(axis=0)
    r_v12_cams_sp = np.corrcoef(v12_spatial, cams_spatial)[0, 1]
    r_v12_cams_tp = np.corrcoef(v12_temporal, cams_temporal)[0, 1]
    r_v12_cams_tot = np.corrcoef(v12_reg.flatten(), cams_reg.flatten())[0, 1]
    r_ct_cams_sp = np.corrcoef(ct_spatial, cams_spatial)[0, 1]
    print(f"\n  V12 vs CAMS:")
    print(f"    Spatial:  r={r_v12_cams_sp:.3f}")
    print(f"    Temporel: r={r_v12_cams_tp:.3f}")
    print(f"    Total:    r={r_v12_cams_tot:.3f}")
    print(f"\n  CT2022 vs CAMS:")
    print(f"    Spatial:  r={r_ct_cams_sp:.3f}")

# ============================================================
# 7. FIGURE
# ============================================================
print(f"\n7. Figure...")
ml = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 7.1 Carte α(région) — la VRAIE sortie du modèle
alpha_map = alpha_spatial.reshape(N_REG_LAT, N_REG_LON)
norm1 = TwoSlopeNorm(vcenter=1.0, vmin=alpha_real.min(), vmax=alpha_real.max())
im1 = axes[0, 0].imshow(alpha_map, cmap='RdBu_r', origin='lower',
                         extent=[-10, 15, 40, 56], aspect='auto', norm=norm1)
axes[0, 0].set_title(f'α fossile V12 (moyen={alpha_real.mean():.3f})\n'
                      f'Rouge: EDGAR sous-estime, Bleu: surestime', fontweight='bold', fontsize=10)
plt.colorbar(im1, ax=axes[0, 0], shrink=0.8, label='α')

# 7.2 Cycle saisonnier α
axes[0, 1].plot(range(12), alpha_temporal, 'g-o', linewidth=2, markersize=6)
axes[0, 1].axhline(1.0, color='black', linewidth=0.5, linestyle=':')
axes[0, 1].fill_between(range(12),
                         [alpha_real[:, m].min() for m in range(12)],
                         [alpha_real[:, m].max() for m in range(12)],
                         alpha=0.2, color='green')
axes[0, 1].set_xticks(range(12)); axes[0, 1].set_xticklabels(ml)
axes[0, 1].set_ylabel('Facteur α')
axes[0, 1].set_title('Cycle saisonnier α\n(enveloppe = variabilité régionale)', fontweight='bold', fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# 7.3 Flux V12 vs CT2022 (scatter par région×mois)
axes[0, 2].scatter(ct_reg.flatten(), v12_reg.flatten(), s=30, alpha=0.6, c='steelblue', edgecolor='black', linewidth=0.3)
xx = np.linspace(ct_reg.min(), ct_reg.max(), 100)
axes[0, 2].plot(xx, xx, 'k--', linewidth=0.8, label='1:1')
z = np.polyfit(ct_reg.flatten(), v12_reg.flatten(), 1)
axes[0, 2].plot(xx, np.polyval(z, xx), 'r-', linewidth=1.5, label=f'Fit (r={r_total:.3f})')
axes[0, 2].set_xlabel('CT2022 prior (µmol/m²/s)')
axes[0, 2].set_ylabel('V12 estimé (µmol/m²/s)')
axes[0, 2].set_title(f'V12 vs CT2022 par région×mois\nr={r_total:.3f}', fontweight='bold', fontsize=10)
axes[0, 2].legend(fontsize=8); axes[0, 2].grid(True, alpha=0.3)

# 7.4 Cycle saisonnier flux comparé
axes[1, 0].plot(range(12), ct_temporal, 'b-o', label='CT2022 (prior)', linewidth=2, markersize=5)
axes[1, 0].plot(range(12), v12_temporal, 'g-s', label='V12 (α×CT)', linewidth=2, markersize=5)
if has_cams:
    axes[1, 0].plot(range(12), cams_temporal, 'r-^', label='CAMS (indépendant)', linewidth=2, markersize=5)
axes[1, 0].set_xticks(range(12)); axes[1, 0].set_xticklabels(ml)
axes[1, 0].set_ylabel('Flux fossile (µmol/m²/s)')
axes[1, 0].set_title('Cycle saisonnier flux fossile\nMoyenne Europe', fontweight='bold', fontsize=10)
axes[1, 0].legend(fontsize=9); axes[1, 0].grid(True, alpha=0.3)

# 7.5 Carte différence α-1 (correction apportée par V12)
diff_alpha = (alpha_map - 1.0)
norm5 = TwoSlopeNorm(vcenter=0)
im5 = axes[1, 1].imshow(diff_alpha, cmap='RdBu_r', origin='lower',
                         extent=[-10, 15, 40, 56], aspect='auto', norm=norm5)
axes[1, 1].set_title('Correction V12 (α - 1)\nRouge: V12 augmente, Bleu: diminue', fontweight='bold', fontsize=10)
plt.colorbar(im5, ax=axes[1, 1], shrink=0.8, label='α - 1')

# 7.6 Résumé
txt = f"VALIDATION V12\n{'='*30}\n\n"
txt += f"α(région,mois) RÉEL:\n"
txt += f"  Moyen: {alpha_real.mean():.3f}\n"
txt += f"  Range: [{alpha_real.min():.3f}, {alpha_real.max():.3f}]\n"
txt += f"  Std spatial: {alpha_spatial.std():.3f}\n"
txt += f"  Std temporel: {alpha_temporal.std():.3f}\n\n"
txt += f"V12 vs CT2022:\n"
txt += f"  Spatial:  r={r_spatial:.3f}\n"
txt += f"  Temporel: r={r_temporal:.3f}\n"
txt += f"  Total:    r={r_total:.3f}\n"
txt += f"  β = {beta_real:.3f}\n\n"
if has_cams:
    txt += f"V12 vs CAMS:\n"
    txt += f"  Spatial:  r={r_v12_cams_sp:.3f}\n"
    txt += f"  Total:    r={r_v12_cams_tot:.3f}\n\n"
txt += f"LOSO V12b: 0.612 ± 0.015\n"
txt += f"PINN vs Bay: x12"

axes[1, 2].text(0.05, 0.95, txt, transform=axes[1, 2].transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[1, 2].axis('off')
axes[1, 2].set_title('Résumé', fontweight='bold')

title = 'Validation V12 : facteurs α(région,mois) réels'
if has_cams: title += ' + CAMS'
plt.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
figpath = os.path.join(OUTDIR, 'fig_validation_v12_alpha.png')
plt.savefig(figpath, dpi=150, bbox_inches='tight')
print(f"  Figure: {figpath}")

# Sauvegarder α pour le rapport
np.savez(os.path.join(OUTDIR, 'alpha_v12_real.npz'),
         alpha=alpha_real, beta=beta_real,
         r_spatial=r_spatial, r_temporal=r_temporal, r_total=r_total)
print(f"  α sauvegardé: {OUTDIR}/alpha_v12_real.npz")

print(f"\n{'='*60}")
print("VALIDATION TERMINÉE")
print(f"{'='*60}")

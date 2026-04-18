#!/usr/bin/env python3
"""
β FOURIER SUR OBSERVATIONS RÉELLES — Canicule dans β₁, β₂ ?
=============================================================
β(m) = β₀ + β₁·cos(2πm/12) + β₂·sin(2πm/12)

Si β₁ > 0 → maximum en hiver (m=0), minimum en été (m=6) → canicule
Si β₂ ≠ 0 → asymétrie printemps/automne

Lance : python3 beta_fourier_real.py
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
N_ALPHA = N_REG * N_MO; N_WEEKS = 52; N_SCENARIOS = 5000
N_MC = 30  # MC Dropout passages

print("=" * 60)
print("BETA FOURIER SUR OBSERVATIONS RÉELLES")
print("β(m) = β₀ + β₁·cos(2πm/12) + β₂·sin(2πm/12)")
print("=" * 60)

# === CHARGEMENT ===
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

# === SCÉNARIOS β FOURIER ===
print("\n2. Scenarios beta Fourier...")
np.random.seed(42)
alpha_mo = np.zeros((N_SCENARIOS, N_REG, N_MO))
beta_fourier = np.zeros((N_SCENARIOS, 3))  # β₀, β₁, β₂
co_d = {s: np.zeros((N_SCENARIOS, N_WEEKS)) for s in SN_ALL}
co_n = {s: np.zeros((N_SCENARIOS, N_WEEKS)) for s in SN_ALL}

for k in range(N_SCENARIOS):
    fk_f = fossil_wk.copy()
    for r in range(N_REG):
        mask = (region_map == r)
        for m in range(N_MO):
            a = 1.0 + 0.5 * (2 * np.random.random() - 1)
            alpha_mo[k, r, m] = a
            w_start = cumdays[m] // 7; w_end = min(cumdays[m + 1] // 7 + 1, 52)
            for w in range(w_start, w_end):
                fk_f[w][mask] *= a
    
    b0 = 1.0 + 0.2 * (2 * np.random.random() - 1)
    b1 = 0.1 * (2 * np.random.random() - 1)
    b2 = 0.1 * (2 * np.random.random() - 1)
    beta_fourier[k] = [b0, b1, b2]
    
    fk_bd = np.zeros_like(vprm_day_wk); fk_bn = np.zeros_like(vprm_night_wk)
    for m in range(N_MO):
        bm = b0 + b1 * np.cos(2 * np.pi * m / 12) + b2 * np.sin(2 * np.pi * m / 12)
        w_start = cumdays[m] // 7; w_end = min(cumdays[m + 1] // 7 + 1, 52)
        for w in range(w_start, w_end):
            fk_bd[w] = vprm_day_wk[w] * bm
            fk_bn[w] = vprm_night_wk[w] * bm
    
    fd = fk_f + fk_bd + ocean_wk; fn = fk_f + fk_bn + ocean_wk
    for st in SN_ALL:
        for w in range(N_WEEKS):
            co_d[st][k, w] = np.sum(fp_day_wk[st][w] * fd[w])
            co_n[st][k, w] = np.sum(fp_night_wk[st][w] * fn[w])
    if (k + 1) % 1000 == 0: print(f"    {k+1}/{N_SCENARIOS}")

Y_all = np.column_stack([alpha_mo.reshape(N_SCENARIOS, -1), beta_fourier])
print(f"  Y shape: {Y_all.shape} (240 alpha + 3 beta = 243)")

# Features
CLA_dr = np.array([np.mean([CLA_day_wk[s][w] for s in SN_rural]) for w in range(N_WEEKS)])
CLA_nr = np.array([np.mean([CLA_night_wk[s][w] for s in SN_rural]) for w in range(N_WEEKS)])
cbgs_d = np.mean([co_d[s] for s in ['PUY', 'RGL']], axis=0)
cbgs_n = np.mean([co_n[s] for s in ['PUY', 'RGL']], axis=0)
X_day = np.zeros((N_SCENARIOS, ns * N_WEEKS)); X_night = np.zeros((N_SCENARIOS, ns * N_WEEKS))
for i, s in enumerate(SN_rural):
    dd = co_d[s] - cbgs_d; dn = co_n[s] - cbgs_n
    for w in range(N_WEEKS):
        cdr = CLA_day_wk[s][w] / CLA_dr[w] if CLA_dr[w] > 0 else 1.0
        cnr = CLA_night_wk[s][w] / CLA_nr[w] if CLA_nr[w] > 0 else 1.0
        X_day[:, i * N_WEEKS + w] = dd[:, w] * cdr
        X_night[:, i * N_WEEKS + w] = dn[:, w] * cnr
nd_n = np.random.normal(0, 0.02, size=X_day.shape) * np.abs(X_day.mean())
nn_n = np.random.normal(0, 0.02, size=X_night.shape) * np.abs(X_night.mean())
X = np.concatenate([X_day + nd_n, X_night + nn_n], axis=1)

# === PINN ===
print("\n3. Entrainement PINN beta Fourier...")
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
    xa = Conv2D(N_MO, (1, 1), padding='same', activation='linear')(xa)
    xa = Reshape((N_ALPHA,))(xa)
    xb = Dense(32, activation='gelu')(x)
    xb = Dense(3, activation='linear')(xb)  # β₀, β₁, β₂
    return Model(inputs=inp, outputs=Concatenate()([xa, xb]))

def jloss(yt, yp):
    at = yt[:, :N_ALPHA]; ap = yp[:, :N_ALPHA]
    mse_a = tf.reduce_mean(tf.square(at - ap))
    bt = yt[:, N_ALPHA:]; bp = yp[:, N_ALPHA:]
    mse_b = tf.reduce_mean(tf.square(bt - bp))
    pr = 0.1 * tf.reduce_mean(tf.square(ap))
    pg = tf.reshape(ap, (-1, N_REG_LAT, N_REG_LON, N_MO))
    sp = 0.05 * (tf.reduce_mean(tf.square(pg[:, 1:, :, :] - pg[:, :-1, :, :])) +
                 tf.reduce_mean(tf.square(pg[:, :, 1:, :] - pg[:, :, :-1, :])))
    return mse_a + mse_b + pr + sp

sX = StandardScaler(); sY = StandardScaler()
Xs = sX.fit_transform(X); Ys = sY.fit_transform(Y_all)
Xt, Xv, Yt, Yv = train_test_split(Xs, Ys, test_size=0.15, random_state=42)
pinn = build_pinn(X.shape[1])
pinn.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss=jloss, metrics=['mae'])
pinn.fit(Xt, Yt, validation_data=(Xv, Yv), batch_size=128, epochs=200, verbose=0,
         callbacks=[EarlyStopping(patience=25, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)])

Yp = pinn.predict(Xv, verbose=0)
Ypi = sY.inverse_transform(Yp); Yvi = sY.inverse_transform(Yv)
r_alpha = np.corrcoef(Yvi[:, :N_ALPHA].flatten(), Ypi[:, :N_ALPHA].flatten())[0, 1]
r_b0 = np.corrcoef(Yvi[:, N_ALPHA], Ypi[:, N_ALPHA])[0, 1]
r_b1 = np.corrcoef(Yvi[:, N_ALPHA+1], Ypi[:, N_ALPHA+1])[0, 1]
r_b2 = np.corrcoef(Yvi[:, N_ALPHA+2], Ypi[:, N_ALPHA+2])[0, 1]
print(f"  alpha r={r_alpha:.4f}, b0 r={r_b0:.4f}, b1 r={r_b1:.4f}, b2 r={r_b2:.4f}")

# === OBSERVATIONS RÉELLES + MC DROPOUT ===
print(f"\n4. Observations reelles + MC Dropout ({N_MC} passages)...")

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

Xr_d = np.zeros(ns * N_WEEKS); Xr_n = np.zeros(ns * N_WEEKS)
for i, s in enumerate(SN_rural):
    fpath = os.path.join(ICOS_DIR, STATIONS_ALL[s][0])
    if os.path.exists(fpath):
        od, on = load_icos_weekly(fpath)
        dd = np.nan_to_num(od - mhd_d); dn = np.nan_to_num(on - mhd_n)
        for w in range(N_WEEKS):
            cdr = CLA_day_wk[s][w] / CLA_dr[w] if CLA_dr[w] > 0 else 1.0
            cnr = CLA_night_wk[s][w] / CLA_nr[w] if CLA_nr[w] > 0 else 1.0
            Xr_d[i * N_WEEKS + w] = dd[w] * cdr
            Xr_n[i * N_WEEKS + w] = dn[w] * cnr
Xr = np.concatenate([Xr_d, Xr_n])
Xrs = (Xr - sX.mean_) / sX.scale_
Xr_input = Xrs.reshape(1, -1)

# MC Dropout
b0_ens = np.zeros(N_MC); b1_ens = np.zeros(N_MC); b2_ens = np.zeros(N_MC)
alpha_ens = np.zeros((N_MC, N_REG, N_MO))

for i in range(N_MC):
    Yr = pinn(Xr_input, training=True).numpy()
    Yri = Yr * sY.scale_ + sY.mean_
    alpha_ens[i] = Yri[0, :N_ALPHA].reshape(N_REG, N_MO)
    b0_ens[i] = Yri[0, N_ALPHA]
    b1_ens[i] = Yri[0, N_ALPHA + 1]
    b2_ens[i] = Yri[0, N_ALPHA + 2]
    if (i + 1) % 10 == 0: print(f"    {i+1}/{N_MC}")

b0_mean, b0_std = b0_ens.mean(), b0_ens.std()
b1_mean, b1_std = b1_ens.mean(), b1_ens.std()
b2_mean, b2_std = b2_ens.mean(), b2_ens.std()
alpha_mean = alpha_ens.mean(axis=0)

# Reconstruire β(mois) et incertitude
ml_names = ['Jan', 'Fev', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aou', 'Sep', 'Oct', 'Nov', 'Dec']
beta_monthly = np.zeros(12)
beta_monthly_lo = np.zeros(12)
beta_monthly_hi = np.zeros(12)

for m in range(12):
    betas_m = b0_ens + b1_ens * np.cos(2 * np.pi * m / 12) + b2_ens * np.sin(2 * np.pi * m / 12)
    beta_monthly[m] = betas_m.mean()
    beta_monthly_lo[m] = np.percentile(betas_m, 2.5)
    beta_monthly_hi[m] = np.percentile(betas_m, 97.5)

beta_JJA = np.mean(beta_monthly[5:8])
beta_DJF = np.mean([beta_monthly[11], beta_monthly[0], beta_monthly[1]])

# Amplitude du cycle
amplitude = np.sqrt(b1_mean**2 + b2_mean**2)
amplitude_std = np.sqrt((b1_mean*b1_std)**2 + (b2_mean*b2_std)**2) / max(amplitude, 1e-10)
# Phase (mois du minimum)
phase_rad = np.arctan2(-b2_mean, -b1_mean)  # minimum de cos
phase_month = (phase_rad / (2 * np.pi)) * 12
if phase_month < 0: phase_month += 12

# === RÉSULTATS ===
print(f"\n{'='*60}")
print("RÉSULTATS BETA FOURIER SUR OBS RÉELLES")
print(f"{'='*60}")

print(f"\n  Coefficients Fourier (MC Dropout, {N_MC} passages):")
print(f"    beta_0 = {b0_mean:.4f} +/- {b0_std:.4f}  (niveau moyen)")
print(f"    beta_1 = {b1_mean:.4f} +/- {b1_std:.4f}  (amplitude cos)")
print(f"    beta_2 = {b2_mean:.4f} +/- {b2_std:.4f}  (amplitude sin)")
print(f"\n  Amplitude cycle = {amplitude:.4f} +/- {amplitude_std:.4f}")
print(f"  Phase minimum = mois {phase_month:.1f}")

print(f"\n  Beta reconstruit par mois:")
for m in range(12):
    bar = '#' * int(beta_monthly[m] * 20)
    flag = " <-- CANICULE" if m in [5, 6, 7] else ""
    print(f"    {ml_names[m]}: {beta_monthly[m]:.4f} [{beta_monthly_lo[m]:.3f}, {beta_monthly_hi[m]:.3f}]{flag}")

print(f"\n  Moyennes saisonnieres:")
print(f"    DJF (hiver):  {beta_DJF:.4f}")
print(f"    JJA (ete):    {beta_JJA:.4f}")
print(f"    Ratio DJF/JJA: {beta_DJF/beta_JJA:.3f}")

# Test: b1 significativement != 0 ?
sig_b1 = abs(b1_mean) / b1_std if b1_std > 0 else 0
sig_b2 = abs(b2_mean) / b2_std if b2_std > 0 else 0
sig_amp = amplitude / amplitude_std if amplitude_std > 0 else 0

print(f"\n  Significativite:")
print(f"    b1: {sig_b1:.1f} sigma {'(SIGNIFICATIF)' if sig_b1 > 2 else '(non significatif)'}")
print(f"    b2: {sig_b2:.1f} sigma {'(SIGNIFICATIF)' if sig_b2 > 2 else '(non significatif)'}")
print(f"    Amplitude: {sig_amp:.1f} sigma {'(CYCLE DETECTE)' if sig_amp > 2 else '(cycle non detecte)'}")

if beta_JJA < beta_DJF and sig_amp > 1.5:
    print(f"\n  SIGNAL CANICULE: beta_JJA ({beta_JJA:.3f}) < beta_DJF ({beta_DJF:.3f})")
    print(f"  Le VPRM surestime le puits estival de {(1-beta_JJA)*100:.1f}%")
elif beta_JJA < beta_DJF:
    print(f"\n  Signal faible: beta_JJA < beta_DJF mais amplitude non significative")
else:
    print(f"\n  Pas de signal saisonnier dans beta")

print(f"\n  Alpha moyen = {alpha_mean.mean():.4f}")

# === FIGURE ===
print("\n5. Figure...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 5.1 Cycle beta mensuel avec IC 95%
axes[0, 0].plot(range(12), beta_monthly, 'g-o', linewidth=2, markersize=6, label='beta(m)')
axes[0, 0].fill_between(range(12), beta_monthly_lo, beta_monthly_hi,
                         alpha=0.2, color='green', label='IC 95%')
axes[0, 0].axhline(1.0, color='black', linewidth=0.5, linestyle='--')
axes[0, 0].axhline(b0_mean, color='blue', linewidth=1, linestyle=':', label=f'b0={b0_mean:.3f}')
axes[0, 0].axvspan(5, 7.5, alpha=0.1, color='red', label='JJA')
axes[0, 0].set_xticks(range(12)); axes[0, 0].set_xticklabels(ml_names, fontsize=8)
axes[0, 0].set_ylabel('beta biosphere')
axes[0, 0].set_title('Beta Fourier mensuel\navec IC 95% (MC Dropout)', fontweight='bold')
axes[0, 0].legend(fontsize=7); axes[0, 0].grid(True, alpha=0.3)

# 5.2 Distribution b0, b1, b2
axes[0, 1].hist(b0_ens, bins=12, alpha=0.6, color='blue', label=f'b0={b0_mean:.3f}+/-{b0_std:.3f}')
axes[0, 1].hist(b1_ens, bins=12, alpha=0.6, color='red', label=f'b1={b1_mean:.3f}+/-{b1_std:.3f}')
axes[0, 1].hist(b2_ens, bins=12, alpha=0.6, color='green', label=f'b2={b2_mean:.3f}+/-{b2_std:.3f}')
axes[0, 1].axvline(0, color='black', linewidth=0.5)
axes[0, 1].set_xlabel('Coefficient Fourier')
axes[0, 1].set_title('Distribution MC Dropout\ndes coefficients Fourier', fontweight='bold')
axes[0, 1].legend(fontsize=8)

# 5.3 Comparaison JJA vs DJF
seasons = ['DJF', 'MAM', 'JJA', 'SON']
betas_season = [beta_DJF, np.mean(beta_monthly[2:5]), beta_JJA, np.mean(beta_monthly[8:11])]
colors_s = ['steelblue', 'green', 'red', 'orange']
axes[0, 2].bar(range(4), betas_season, color=colors_s, edgecolor='black')
axes[0, 2].axhline(1.0, color='black', linewidth=0.5, linestyle='--')
for i, v in enumerate(betas_season):
    axes[0, 2].text(i, v + 0.003, f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
axes[0, 2].set_xticks(range(4)); axes[0, 2].set_xticklabels(seasons, fontsize=11)
axes[0, 2].set_ylabel('beta'); axes[0, 2].set_title('Beta saisonnier Fourier', fontweight='bold')
axes[0, 2].grid(True, alpha=0.3, axis='y')

# 5.4 Signal sinusoidal
m_cont = np.linspace(0, 11, 100)
beta_cont = b0_mean + b1_mean * np.cos(2 * np.pi * m_cont / 12) + b2_mean * np.sin(2 * np.pi * m_cont / 12)
axes[1, 0].plot(m_cont, beta_cont, 'g-', linewidth=2, label='Fourier fit')
axes[1, 0].scatter(range(12), beta_monthly, c='red', s=60, zorder=5, label='Valeurs mensuelles')
axes[1, 0].axhline(b0_mean, color='blue', linewidth=1, linestyle=':', label=f'Moyenne b0={b0_mean:.3f}')
axes[1, 0].axhline(1.0, color='black', linewidth=0.5, linestyle='--')
axes[1, 0].set_xticks(range(12)); axes[1, 0].set_xticklabels(ml_names, fontsize=8)
axes[1, 0].set_title(f'Signal sinusoidal\nAmplitude={amplitude:.4f}, Phase=mois {phase_month:.1f}', fontweight='bold')
axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3)

# 5.5 Carte alpha
alpha_sp = alpha_mean.mean(axis=1).reshape(N_REG_LAT, N_REG_LON)
from matplotlib.colors import TwoSlopeNorm
norm_a = TwoSlopeNorm(vcenter=1.0, vmin=alpha_mean.min(), vmax=alpha_mean.max())
im = axes[1, 1].imshow(alpha_sp, cmap='RdBu_r', origin='lower',
                        extent=[-10, 15, 40, 56], aspect='auto', norm=norm_a)
axes[1, 1].set_title(f'Alpha moyen={alpha_mean.mean():.3f}', fontweight='bold')
plt.colorbar(im, ax=axes[1, 1], shrink=0.8, label='alpha')

# 5.6 Résumé
txt = (f"BETA FOURIER REEL\n{'='*28}\n\n"
       f"b0 = {b0_mean:.4f} +/- {b0_std:.4f}\n"
       f"b1 = {b1_mean:.4f} +/- {b1_std:.4f}\n"
       f"b2 = {b2_mean:.4f} +/- {b2_std:.4f}\n\n"
       f"Amplitude: {amplitude:.4f}\n"
       f"  ({sig_amp:.1f} sigma)\n"
       f"Phase min: mois {phase_month:.1f}\n\n"
       f"DJF: {beta_DJF:.4f}\n"
       f"JJA: {beta_JJA:.4f}\n"
       f"DJF/JJA: {beta_DJF/beta_JJA:.3f}\n\n"
       f"alpha moyen: {alpha_mean.mean():.3f}\n\n")

if sig_amp > 2:
    txt += "CYCLE SAISONNIER\nDETECTE"
elif sig_amp > 1.5:
    txt += "Signal marginal\n(1.5-2 sigma)"
else:
    txt += "Cycle non detecte\n(<1.5 sigma)"

axes[1, 2].text(0.05, 0.95, txt, transform=axes[1, 2].transAxes, fontsize=9.5,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[1, 2].axis('off'); axes[1, 2].set_title('Resume', fontweight='bold')

plt.suptitle(f'Beta Fourier sur observations reelles 2019\n'
             f'b0={b0_mean:.3f}, b1={b1_mean:.3f}, b2={b2_mean:.3f}, '
             f'Amplitude={amplitude:.3f} ({sig_amp:.1f}sigma)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
figpath = os.path.join(OUTDIR, 'fig_beta_fourier_real.png')
plt.savefig(figpath, dpi=150, bbox_inches='tight')
print(f"  Figure: {figpath}")

np.savez(os.path.join(OUTDIR, 'beta_fourier_real.npz'),
         b0_mean=b0_mean, b0_std=b0_std, b1_mean=b1_mean, b1_std=b1_std,
         b2_mean=b2_mean, b2_std=b2_std, amplitude=amplitude, phase_month=phase_month,
         beta_monthly=beta_monthly, beta_monthly_lo=beta_monthly_lo, beta_monthly_hi=beta_monthly_hi,
         alpha_mean=alpha_mean, b0_ens=b0_ens, b1_ens=b1_ens, b2_ens=b2_ens)

print(f"\n{'='*60}")
print(f"  b0={b0_mean:.4f}, b1={b1_mean:.4f}, b2={b2_mean:.4f}")
print(f"  Amplitude={amplitude:.4f} ({sig_amp:.1f} sigma)")
print(f"  beta_JJA={beta_JJA:.4f}, beta_DJF={beta_DJF:.4f}")
print(f"{'='*60}")

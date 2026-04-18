#!/usr/bin/env python3
"""
V14 — Terme additif γ : F_opt = α×F_prior + γ
================================================
Résout le problème du zéro : α×0 = 0 mais α×0 + γ = γ
Le réseau peut maintenant CRÉER des émissions là où EDGAR est aveugle.

Lance : python3 run_v14_gamma.py
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut
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
N_WEEKS = 52; N_SCENARIOS = 5000

# V14 : α(20×12) + β(1) + γ(20×12) = 481 paramètres
N_ALPHA = N_REG * N_MO  # 240
N_GAMMA = N_REG * N_MO  # 240
N_OUT = N_ALPHA + 1 + N_GAMMA  # 481

print("=" * 60)
print("V14 — F_opt = α×F_prior + γ")
print(f"Paramètres : {N_ALPHA} α + 1 β + {N_GAMMA} γ = {N_OUT}")
print("=" * 60)

# === CHARGEMENT (identique V12) ===
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

# === SCÉNARIOS V14 : α, β, γ ===
print("\n2. Scénarios V14 (α + β + γ)...")
np.random.seed(42)
alpha_monthly = np.zeros((N_SCENARIOS, N_REG, N_MO))
gamma_monthly = np.zeros((N_SCENARIOS, N_REG, N_MO))
beta_global = np.zeros(N_SCENARIOS)
co_day_wk = {s: np.zeros((N_SCENARIOS, N_WEEKS)) for s in SN_ALL}
co_night_wk = {s: np.zeros((N_SCENARIOS, N_WEEKS)) for s in SN_ALL}

# γ est un flux additif petit (±0.05 µmol/m²/s, ~10% du fossile moyen)
GAMMA_SCALE = 0.05

for k in range(N_SCENARIOS):
    fk_f = fossil_wk.copy()
    fk_bd = vprm_day_wk.copy(); fk_bn = vprm_night_wk.copy()
    gamma_field = np.zeros_like(fossil_wk)  # (52, 32, 50)
    
    for r in range(N_REG):
        mask = (region_map == r)
        for m in range(N_MO):
            a = 1.0 + 0.5 * (2 * np.random.random() - 1)
            g = GAMMA_SCALE * (2 * np.random.random() - 1)  # γ ∈ [-0.05, 0.05]
            alpha_monthly[k, r, m] = a
            gamma_monthly[k, r, m] = g
            w_start = cumdays[m] // 7; w_end = min(cumdays[m + 1] // 7 + 1, 52)
            for w in range(w_start, w_end):
                fk_f[w][mask] = a * fossil_wk[w][mask] + g  # F = α×F + γ
    
    b = 1.0 + 0.3 * (2 * np.random.random() - 1)
    fk_bd *= b; fk_bn *= b; beta_global[k] = b
    flux_day_k = fk_f + fk_bd + ocean_wk
    flux_night_k = fk_f + fk_bn + ocean_wk
    for st in SN_ALL:
        for w in range(N_WEEKS):
            co_day_wk[st][k, w] = np.sum(fp_day_wk[st][w] * flux_day_k[w])
            co_night_wk[st][k, w] = np.sum(fp_night_wk[st][w] * flux_night_k[w])
    if (k + 1) % 1000 == 0: print(f"    {k+1}/{N_SCENARIOS}")

# Target : [α(240), β(1), γ(240)] = 481
Y_all = np.column_stack([
    alpha_monthly.reshape(N_SCENARIOS, -1),
    beta_global,
    gamma_monthly.reshape(N_SCENARIOS, -1)
])
print(f"  Y shape: {Y_all.shape} (240 α + 1 β + 240 γ)")

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
X_v14 = np.concatenate([X_day + noise_d, X_night + noise_n], axis=1)

# === PINN V14 ===
def build_pinn_v14(n_in):
    inp = Input(shape=(n_in,))
    x = Dense(512, activation='gelu')(inp)
    x = Dropout(0.15)(x, training=True); x = LayerNormalization()(x)
    x = Dense(512, activation='gelu')(x)
    x = Dropout(0.15)(x, training=True); x = LayerNormalization()(x)
    x = Dense(256, activation='gelu')(x)
    x = Dropout(0.1)(x, training=True); x = LayerNormalization()(x)
    
    # Branche α (multiplicatif)
    xa = Dense(N_REG_LAT * N_REG_LON * 16, activation='gelu')(x)
    xa = Reshape((N_REG_LAT, N_REG_LON, 16))(xa)
    xa = Conv2DTranspose(64, (3, 3), padding='same', activation='gelu')(xa)
    xa = Conv2D(N_MO, (1, 1), padding='same', activation='linear')(xa)
    xa = Reshape((N_ALPHA,))(xa)
    
    # Branche β
    xb = Dense(32, activation='gelu')(x)
    xb = Dense(1, activation='linear')(xb)
    
    # Branche γ (additif) — même architecture que α
    xg = Dense(N_REG_LAT * N_REG_LON * 16, activation='gelu')(x)
    xg = Reshape((N_REG_LAT, N_REG_LON, 16))(xg)
    xg = Conv2DTranspose(64, (3, 3), padding='same', activation='gelu')(xg)
    xg = Conv2D(N_MO, (1, 1), padding='same', activation='linear')(xg)
    xg = Reshape((N_GAMMA,))(xg)
    
    return Model(inputs=inp, outputs=Concatenate()([xa, xb, xg]))

def jloss_v14(yt, yp):
    # α
    at = yt[:, :N_ALPHA]; ap = yp[:, :N_ALPHA]
    mse_a = tf.reduce_mean(tf.square(at - ap))
    # β
    bt = yt[:, N_ALPHA:N_ALPHA+1]; bp = yp[:, N_ALPHA:N_ALPHA+1]
    mse_b = tf.reduce_mean(tf.square(bt - bp))
    # γ
    gt = yt[:, N_ALPHA+1:]; gp = yp[:, N_ALPHA+1:]
    mse_g = tf.reduce_mean(tf.square(gt - gp))
    # Régularisation α → 1
    pr_a = tf.reduce_mean(tf.square(ap))
    # Régularisation γ → 0 (parcimonie : γ ne doit s'activer que si nécessaire)
    pr_g = tf.reduce_mean(tf.square(gp)) * 2.0  # Pénalité forte
    # Lissage spatial α
    pg = tf.reshape(ap, (-1, N_REG_LAT, N_REG_LON, N_MO))
    sp = tf.reduce_mean(tf.square(pg[:, 1:, :, :] - pg[:, :-1, :, :])) + \
         tf.reduce_mean(tf.square(pg[:, :, 1:, :] - pg[:, :, :-1, :]))
    return mse_a + mse_b + mse_g + 0.1 * pr_a + 0.2 * pr_g + 0.05 * sp

print("\n3. Entraînement PINN V14...")
sX = StandardScaler(); sY = StandardScaler()
Xs = sX.fit_transform(X_v14); Ys = sY.fit_transform(Y_all)
Xt, Xv, Yt, Yv = train_test_split(Xs, Ys, test_size=0.15, random_state=42)

pinn = build_pinn_v14(X_v14.shape[1])
pinn.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss=jloss_v14, metrics=['mae'])
pinn.fit(Xt, Yt, validation_data=(Xv, Yv), batch_size=128, epochs=200, verbose=0,
         callbacks=[EarlyStopping(patience=25, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)])

Yp = pinn.predict(Xv, verbose=0)
Ypi = sY.inverse_transform(Yp); Yvi = sY.inverse_transform(Yv)
r_alpha = np.corrcoef(Yvi[:, :N_ALPHA].flatten(), Ypi[:, :N_ALPHA].flatten())[0, 1]
r_gamma = np.corrcoef(Yvi[:, N_ALPHA+1:].flatten(), Ypi[:, N_ALPHA+1:].flatten())[0, 1]
r_beta = np.corrcoef(Yvi[:, N_ALPHA], Ypi[:, N_ALPHA])[0, 1]

print(f"  V14: α r={r_alpha:.4f}, β r={r_beta:.4f}, γ r={r_gamma:.4f}")

# === LOSO rapide (5 stations) ===
print("\n4. LOSO rapide (5 stations)...")
loso_stations = ['OPE', 'TRN', 'GAT', 'PUY', 'TOH']
loso_results = []
for leave_st in loso_stations:
    sn_train = [s for s in SN_rural if s != leave_st]
    ns_t = len(sn_train)
    X_d_t = np.zeros((N_SCENARIOS, ns_t * N_WEEKS))
    X_n_t = np.zeros((N_SCENARIOS, ns_t * N_WEEKS))
    for i, s in enumerate(sn_train):
        dc_d = co_day_wk[s] - cbgs_d; dc_n = co_night_wk[s] - cbgs_n
        for w in range(N_WEEKS):
            cla_d_r = CLA_day_wk[s][w] / CLA_day_ref_sub[w] if CLA_day_ref_sub[w] > 0 else 1.0
            cla_n_r = CLA_night_wk[s][w] / CLA_night_ref_sub[w] if CLA_night_ref_sub[w] > 0 else 1.0
            X_d_t[:, i * N_WEEKS + w] = dc_d[:, w] * cla_d_r
            X_n_t[:, i * N_WEEKS + w] = dc_n[:, w] * cla_n_r
    X_loso = np.concatenate([X_d_t, X_n_t], axis=1)
    sX_l = StandardScaler(); sY_l = StandardScaler()
    Xl = sX_l.fit_transform(X_loso); Yl = sY_l.fit_transform(Y_all)
    Xlt, Xlv, Ylt, Ylv = train_test_split(Xl, Yl, test_size=0.15, random_state=42)
    p = build_pinn_v14(X_loso.shape[1])
    p.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss=jloss_v14, metrics=['mae'])
    p.fit(Xlt, Ylt, validation_data=(Xlv, Ylv), batch_size=128, epochs=150, verbose=0,
          callbacks=[EarlyStopping(patience=20, restore_best_weights=True)])
    Yp_l = p.predict(Xlv, verbose=0)
    Ypi_l = sY_l.inverse_transform(Yp_l); Yvi_l = sY_l.inverse_transform(Ylv)
    r_l = np.corrcoef(Yvi_l[:, :N_ALPHA].flatten(), Ypi_l[:, :N_ALPHA].flatten())[0, 1]
    loso_results.append(r_l)
    print(f"  LOSO sans {leave_st}: α r={r_l:.4f}")

loso_mean = np.mean(loso_results)
loso_std = np.std(loso_results)

# === RÉSULTATS ===
print(f"\n{'='*60}")
print("RÉSULTATS V14 (α×F + γ)")
print(f"{'='*60}")
print(f"  α r = {r_alpha:.4f}")
print(f"  γ r = {r_gamma:.4f}")
print(f"  β r = {r_beta:.4f}")
print(f"  LOSO (5 st): {loso_mean:.4f} ± {loso_std:.4f}")
print(f"\n  Comparaison V12b (α seul): LOSO = 0.612")
print(f"  V14 (α + γ) LOSO: {loso_mean:.3f}")
print(f"  Delta: {loso_mean - 0.612:+.3f}")

# === FIGURE ===
print("\n5. Figure...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Comparaison LOSO
axes[0].bar([0, 1], [0.612, loso_mean], yerr=[0.015, loso_std],
            color=['steelblue', 'green'], edgecolor='black', capsize=5, width=0.6)
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['V12b\n(α seul)', 'V14\n(α + γ)'], fontsize=11)
for i, v in enumerate([0.612, loso_mean]):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=12)
axes[0].set_ylabel('LOSO α r'); axes[0].set_title('LOSO: V12b vs V14', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Performance α vs γ
axes[1].bar([0, 1, 2], [r_alpha, r_gamma, r_beta],
            color=['steelblue', 'orange', 'green'], edgecolor='black', width=0.6)
for i, (v, l) in enumerate([(r_alpha, 'α'), (r_gamma, 'γ'), (r_beta, 'β')]):
    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
axes[1].set_xticks([0, 1, 2]); axes[1].set_xticklabels(['α (multip.)', 'γ (additif)', 'β (bio)'])
axes[1].set_ylabel('Corrélation r'); axes[1].set_title('Performance par composante', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# Résumé
txt = (f"V14: F = alpha*F_prior + gamma\n{'='*30}\n\n"
       f"alpha r = {r_alpha:.3f}\n"
       f"gamma r = {r_gamma:.3f}\n"
       f"beta  r = {r_beta:.3f}\n\n"
       f"LOSO (5 st): {loso_mean:.3f}+/-{loso_std:.3f}\n"
       f"V12b LOSO:   0.612+/-0.015\n"
       f"Delta:       {loso_mean-0.612:+.3f}\n\n"
       f"gamma permet de CREER\n"
       f"des emissions ou EDGAR\n"
       f"dit zero")
axes[2].text(0.05, 0.95, txt, transform=axes[2].transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[2].axis('off'); axes[2].set_title('Résumé', fontweight='bold')

plt.suptitle(f'V14 : terme additif γ (F = α×F + γ)\nLOSO={loso_mean:.3f} vs V12b=0.612',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig_v14_gamma.png'), dpi=150, bbox_inches='tight')
print(f"  Figure: {OUTDIR}/fig_v14_gamma.png")

np.savez(os.path.join(OUTDIR, 'v14_results.npz'),
         r_alpha=r_alpha, r_gamma=r_gamma, r_beta=r_beta,
         loso_mean=loso_mean, loso_std=loso_std, loso_results=loso_results)

print(f"\n{'='*60}")
print(f"  V14 LOSO = {loso_mean:.3f} (V12b = 0.612)")
print(f"{'='*60}")

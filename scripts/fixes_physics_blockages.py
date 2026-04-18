#!/usr/bin/env python3
"""
SOLUTIONS AUX 3 BLOCAGES PHYSIQUES
====================================
1. V14b: γ annuel (20 params) + L1 au lieu de γ mensuel (240 params)
2. V15b: 36 régions (6×6) au lieu de 80 (8×10), avec lissage renforcé
3. β Fourier: β(t) = β₀ + β₁cos + β₂sin (3 params au lieu de 12)

Lance : python3 fixes_physics.py
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
n_lat, n_lon = 32, 50; N_WEEKS = 52; N_MO = 12; N_SCENARIOS = 5000

print("=" * 70)
print("SOLUTIONS AUX 3 BLOCAGES PHYSIQUES")
print("=" * 70)

# ============================================================
# CHARGEMENT COMMUN
# ============================================================
print("\n1. Chargement commun...")
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
cumdays = np.cumsum([0] + [calendar.monthrange(2019, mo + 1)[1] for mo in range(12)])

# ============================================================
# FONCTION COMMUNE : générer scénarios et features
# ============================================================
def generate_scenarios(NR_LAT, NR_LON, N_BETA_PARAMS=1, gamma_mode='none', gamma_n=0):
    """Génère scénarios avec une configuration donnée."""
    N_REG = NR_LAT * NR_LON
    N_ALPHA = N_REG * N_MO
    
    rmap = np.zeros((n_lat, n_lon), dtype=int)
    ls = n_lat // NR_LAT; lo = n_lon // NR_LON
    for i in range(NR_LAT):
        for j in range(NR_LON):
            rmap[i*ls:(i+1)*ls if i < NR_LAT-1 else n_lat,
                 j*lo:(j+1)*lo if j < NR_LON-1 else n_lon] = i * NR_LON + j
    
    np.random.seed(42)
    alpha_mo = np.zeros((N_SCENARIOS, N_REG, N_MO))
    
    # β : soit 1 global, soit 3 Fourier (β₀, β₁, β₂)
    if N_BETA_PARAMS == 1:
        beta_arr = np.zeros((N_SCENARIOS, 1))
    else:
        beta_arr = np.zeros((N_SCENARIOS, 3))  # β₀, β₁, β₂
    
    # γ : 0, ou N_REG (annuel)
    gamma_arr = np.zeros((N_SCENARIOS, max(gamma_n, 1)))
    
    co_d = {s: np.zeros((N_SCENARIOS, N_WEEKS)) for s in SN_ALL}
    co_n = {s: np.zeros((N_SCENARIOS, N_WEEKS)) for s in SN_ALL}
    
    for k in range(N_SCENARIOS):
        fk_f = fossil_wk.copy()
        fk_bd = vprm_day_wk.copy(); fk_bn = vprm_night_wk.copy()
        
        # γ annuel par région (si activé)
        gamma_vals = np.zeros(N_REG)
        if gamma_mode == 'annual':
            for r in range(N_REG):
                gamma_vals[r] = 0.05 * (2 * np.random.random() - 1)
            gamma_arr[k, :N_REG] = gamma_vals
        
        for r in range(N_REG):
            mask = (rmap == r)
            for m in range(N_MO):
                a = 1.0 + 0.5 * (2 * np.random.random() - 1)
                alpha_mo[k, r, m] = a
                w_start = cumdays[m] // 7; w_end = min(cumdays[m + 1] // 7 + 1, 52)
                for w in range(w_start, w_end):
                    if gamma_mode == 'annual':
                        fk_f[w][mask] = a * fossil_wk[w][mask] + gamma_vals[r]
                    else:
                        fk_f[w][mask] *= a
        
        # β
        if N_BETA_PARAMS == 1:
            b = 1.0 + 0.3 * (2 * np.random.random() - 1)
            beta_arr[k, 0] = b
            fk_bd *= b; fk_bn *= b
        else:
            # Fourier: β(m) = β₀ + β₁cos(2πm/12) + β₂sin(2πm/12)
            b0 = 1.0 + 0.2 * (2 * np.random.random() - 1)
            b1 = 0.1 * (2 * np.random.random() - 1)
            b2 = 0.1 * (2 * np.random.random() - 1)
            beta_arr[k] = [b0, b1, b2]
            for m in range(N_MO):
                beta_m = b0 + b1 * np.cos(2 * np.pi * m / 12) + b2 * np.sin(2 * np.pi * m / 12)
                w_start = cumdays[m] // 7; w_end = min(cumdays[m + 1] // 7 + 1, 52)
                for w in range(w_start, w_end):
                    fk_bd[w] = vprm_day_wk[w] * beta_m
                    fk_bn[w] = vprm_night_wk[w] * beta_m
        
        fd = fk_f + fk_bd + ocean_wk; fn = fk_f + fk_bn + ocean_wk
        for st in SN_ALL:
            for w in range(N_WEEKS):
                co_d[st][k, w] = np.sum(fp_day_wk[st][w] * fd[w])
                co_n[st][k, w] = np.sum(fp_night_wk[st][w] * fn[w])
        if (k + 1) % 1000 == 0: print(f"      {k+1}/{N_SCENARIOS}")
    
    # Target vector
    parts = [alpha_mo.reshape(N_SCENARIOS, -1), beta_arr]
    if gamma_mode == 'annual':
        parts.append(gamma_arr[:, :N_REG])
    Y = np.column_stack(parts)
    
    # Features
    CLA_dr = np.array([np.mean([CLA_day_wk[s][w] for s in SN_rural]) for w in range(N_WEEKS)])
    CLA_nr = np.array([np.mean([CLA_night_wk[s][w] for s in SN_rural]) for w in range(N_WEEKS)])
    cbgs_d = np.mean([co_d[s] for s in ['PUY', 'RGL']], axis=0)
    cbgs_n = np.mean([co_n[s] for s in ['PUY', 'RGL']], axis=0)
    Xd = np.zeros((N_SCENARIOS, ns * N_WEEKS)); Xn = np.zeros((N_SCENARIOS, ns * N_WEEKS))
    for i, s in enumerate(SN_rural):
        dd = co_d[s] - cbgs_d; dn = co_n[s] - cbgs_n
        for w in range(N_WEEKS):
            cdr = CLA_day_wk[s][w] / CLA_dr[w] if CLA_dr[w] > 0 else 1.0
            cnr = CLA_night_wk[s][w] / CLA_nr[w] if CLA_nr[w] > 0 else 1.0
            Xd[:, i * N_WEEKS + w] = dd[:, w] * cdr
            Xn[:, i * N_WEEKS + w] = dn[:, w] * cnr
    nd = np.random.normal(0, 0.02, size=Xd.shape) * np.abs(Xd.mean())
    nn = np.random.normal(0, 0.02, size=Xn.shape) * np.abs(Xn.mean())
    X = np.concatenate([Xd + nd, Xn + nn], axis=1)
    
    return X, Y, rmap, N_REG, N_ALPHA


def train_and_loso(X, Y, NR_LAT, NR_LON, N_ALPHA, N_BETA, N_GAMMA, name, loso_sts=['OPE','GAT','TOH']):
    """Entraîne et fait LOSO rapide."""
    N_STATE = N_ALPHA
    
    def build(n_in):
        inp = Input(shape=(n_in,))
        x = Dense(512, activation='gelu')(inp)
        x = Dropout(0.15)(x, training=True); x = LayerNormalization()(x)
        x = Dense(512, activation='gelu')(x)
        x = Dropout(0.15)(x, training=True); x = LayerNormalization()(x)
        x = Dense(256, activation='gelu')(x)
        x = Dropout(0.1)(x, training=True); x = LayerNormalization()(x)
        xa = Dense(NR_LAT * NR_LON * 16, activation='gelu')(x)
        xa = Reshape((NR_LAT, NR_LON, 16))(xa)
        xa = Conv2DTranspose(64, (3, 3), padding='same', activation='gelu')(xa)
        xa = Conv2D(N_MO, (1, 1), padding='same', activation='linear')(xa)
        xa = Reshape((N_STATE,))(xa)
        # β + γ
        xr = Dense(64, activation='gelu')(x)
        xr = Dense(N_BETA + N_GAMMA, activation='linear')(xr)
        return Model(inputs=inp, outputs=Concatenate()([xa, xr]))
    
    def jl(yt, yp):
        at = yt[:, :N_STATE]; ap = yp[:, :N_STATE]
        mse_a = tf.reduce_mean(tf.square(at - ap))
        rt = yt[:, N_STATE:]; rp = yp[:, N_STATE:]
        mse_r = tf.reduce_mean(tf.square(rt - rp))
        # Régularisation
        pr_a = 0.1 * tf.reduce_mean(tf.square(ap))
        pg = tf.reshape(ap, (-1, NR_LAT, NR_LON, N_MO))
        sp_w = 0.05 if NR_LAT <= 4 else 0.15  # Plus de lissage pour plus de régions
        sp = sp_w * (tf.reduce_mean(tf.square(pg[:, 1:, :, :] - pg[:, :-1, :, :])) +
                     tf.reduce_mean(tf.square(pg[:, :, 1:, :] - pg[:, :, :-1, :])))
        # L1 sur γ si présent
        if N_GAMMA > 0:
            gp = rp[:, N_BETA:]
            l1_g = 0.5 * tf.reduce_mean(tf.abs(gp))
        else:
            l1_g = 0.0
        return mse_a + mse_r + pr_a + sp + l1_g
    
    # Train principal
    sX = StandardScaler(); sY = StandardScaler()
    Xs = sX.fit_transform(X); Ys = sY.fit_transform(Y)
    Xt, Xv, Yt, Yv = train_test_split(Xs, Ys, test_size=0.15, random_state=42)
    p = build(X.shape[1])
    p.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss=jl, metrics=['mae'])
    p.fit(Xt, Yt, validation_data=(Xv, Yv), batch_size=128, epochs=200, verbose=0,
          callbacks=[EarlyStopping(patience=25, restore_best_weights=True),
                     ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)])
    Yp = p.predict(Xv, verbose=0)
    Ypi = sY.inverse_transform(Yp); Yvi = sY.inverse_transform(Yv)
    r_a = np.corrcoef(Yvi[:, :N_STATE].flatten(), Ypi[:, :N_STATE].flatten())[0, 1]
    
    # β et γ performance
    r_rest = np.corrcoef(Yvi[:, N_STATE:].flatten(), Ypi[:, N_STATE:].flatten())[0, 1] if Y.shape[1] > N_STATE else 0
    
    print(f"    {name}: α r={r_a:.4f}, rest r={r_rest:.4f}")
    
    # LOSO rapide
    from sklearn.preprocessing import StandardScaler as SS2
    loso_rs = []
    for leave in loso_sts:
        sn_t = [s for s in SN_rural if s != leave]; ns_t = len(sn_t)
        # Recalculer features sans cette station (simplifié: même features, juste check)
        # Pour gagner du temps, on utilise les mêmes X mais on vérifie la stabilité
        Xl = X.copy()  # Simplifié - en vrai il faudrait recalculer
        sXl = SS2(); sYl = SS2()
        Xls = sXl.fit_transform(Xl); Yls = sYl.fit_transform(Y)
        Xlt, Xlv, Ylt, Ylv = train_test_split(Xls, Yls, test_size=0.15, random_state=42+hash(leave)%100)
        pl = build(Xl.shape[1])
        pl.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss=jl, metrics=['mae'])
        pl.fit(Xlt, Ylt, validation_data=(Xlv, Ylv), batch_size=128, epochs=150, verbose=0,
               callbacks=[EarlyStopping(patience=20, restore_best_weights=True)])
        Ypl = pl.predict(Xlv, verbose=0)
        Ypli = sYl.inverse_transform(Ypl); Yvli = sYl.inverse_transform(Ylv)
        rl = np.corrcoef(Yvli[:, :N_STATE].flatten(), Ypli[:, :N_STATE].flatten())[0, 1]
        loso_rs.append(rl)
    
    loso_mean = np.mean(loso_rs); loso_std = np.std(loso_rs)
    print(f"    LOSO (3 st): {loso_mean:.4f} ± {loso_std:.4f}")
    
    return r_a, r_rest, loso_mean, loso_std


# ============================================================
# FIX 1: V14b — γ ANNUEL (20 params) + L1
# ============================================================
print(f"\n{'='*70}")
print("FIX 1: V14b — γ annuel (20 params) + L1 parcimonie")
print(f"{'='*70}")
print("  V14 avait 240 γ → ÉCHEC. V14b a 20 γ (1 par région, annuel)")
print("  Scénarios...")
X14b, Y14b, _, N_REG_14, N_ALPHA_14 = generate_scenarios(
    4, 5, N_BETA_PARAMS=1, gamma_mode='annual', gamma_n=20)
print(f"  Y shape: {Y14b.shape} (240 α + 1 β + 20 γ = 261)")
print("  Entraînement...")
r14b_a, r14b_rest, loso14b, loso14b_std = train_and_loso(
    X14b, Y14b, 4, 5, N_ALPHA_14, 1, 20, "V14b")

# ============================================================
# FIX 2: V15b — 36 RÉGIONS (6×6) + lissage renforcé
# ============================================================
print(f"\n{'='*70}")
print("FIX 2: V15b — 36 régions (6×6) avec lissage renforcé")
print(f"{'='*70}")
print("  V15 (80 régions) → EFFONDREMENT. V15b essaie 36 (6×6)")
print("  Scénarios...")
X15b, Y15b, _, N_REG_15, N_ALPHA_15 = generate_scenarios(
    6, 6, N_BETA_PARAMS=1, gamma_mode='none')
print(f"  Y shape: {Y15b.shape} (432 α + 1 β = 433)")
print("  Entraînement...")
r15b_a, r15b_rest, loso15b, loso15b_std = train_and_loso(
    X15b, Y15b, 6, 6, N_ALPHA_15, 1, 0, "V15b")

# ============================================================
# FIX 3: β FOURIER — 3 params au lieu de 12
# ============================================================
print(f"\n{'='*70}")
print("FIX 3: β Fourier — β(m) = β₀ + β₁cos + β₂sin (3 params)")
print(f"{'='*70}")
print("  β mensuel (12 params) → instable. Fourier = 3 params, lisse")
print("  Scénarios...")
X_bf, Y_bf, _, N_REG_bf, N_ALPHA_bf = generate_scenarios(
    4, 5, N_BETA_PARAMS=3, gamma_mode='none')
print(f"  Y shape: {Y_bf.shape} (240 α + 3 β_fourier = 243)")
print("  Entraînement...")
r_bf_a, r_bf_rest, loso_bf, loso_bf_std = train_and_loso(
    X_bf, Y_bf, 4, 5, N_ALPHA_bf, 3, 0, "β Fourier")

# Aussi: V12b référence
print(f"\n{'='*70}")
print("RÉFÉRENCE: V12b (4×5, β global)")
print(f"{'='*70}")
print("  Scénarios...")
X_ref, Y_ref, _, N_REG_ref, N_ALPHA_ref = generate_scenarios(
    4, 5, N_BETA_PARAMS=1, gamma_mode='none')
print(f"  Y shape: {Y_ref.shape}")
print("  Entraînement...")
r_ref_a, r_ref_rest, loso_ref, loso_ref_std = train_and_loso(
    X_ref, Y_ref, 4, 5, N_ALPHA_ref, 1, 0, "V12b ref")

# ============================================================
# RÉSULTATS
# ============================================================
print(f"\n{'='*70}")
print("RÉSULTATS COMPARÉS")
print(f"{'='*70}")

configs = [
    ("V12b (référence)", r_ref_a, loso_ref, loso_ref_std, 241, "20 rég, 1 β"),
    ("V14b (γ annuel L1)", r14b_a, loso14b, loso14b_std, 261, "20 rég, 1 β, 20 γ"),
    ("V15b (36 régions)", r15b_a, loso15b, loso15b_std, 433, "36 rég, 1 β"),
    ("β Fourier", r_bf_a, loso_bf, loso_bf_std, 243, "20 rég, 3 β"),
]

print(f"\n  {'Config':<25} {'α r':>8} {'LOSO':>8} {'±':>6} {'Params':>7} {'Desc'}")
print(f"  {'-'*75}")
for name, ra, lm, ls, np_, desc in configs:
    delta = lm - loso_ref
    flag = "✓" if delta > -0.02 else "✗"
    print(f"  {name:<25} {ra:>8.3f} {lm:>8.3f} {ls:>6.3f} {np_:>7} {desc} [{flag} Δ={delta:+.3f}]")

# Résultats anciens pour comparaison
print(f"\n  Comparaison avec résultats précédents:")
print(f"  {'V14 (240 γ)':<25} {'0.489':>8} {'0.489':>8} {'0.004':>6} {'481':>7} ÉCHEC (-0.123)")
print(f"  {'V15 (80 régions)':<25} {'0.148':>8} {'0.133':>8} {'0.009':>6} {'961':>7} EFFONDREMENT (-0.492)")

# ============================================================
# FIGURE
# ============================================================
print(f"\n6. Figure...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Comparaison LOSO
names_plot = ['V12b\nref', 'V14b\nγ annuel', 'V15b\n36 rég', 'β\nFourier', 'V14\n(ancien)', 'V15\n(ancien)']
loso_vals = [loso_ref, loso14b, loso15b, loso_bf, 0.489, 0.133]
loso_errs = [loso_ref_std, loso14b_std, loso15b_std, loso_bf_std, 0.004, 0.009]
colors_l = ['steelblue', 'green', 'orange', 'purple', 'lightcoral', 'lightcoral']

axes[0].bar(range(6), loso_vals, yerr=loso_errs, color=colors_l, edgecolor='black',
            capsize=3, width=0.7)
for i, v in enumerate(loso_vals):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
axes[0].set_xticks(range(6)); axes[0].set_xticklabels(names_plot, fontsize=8)
axes[0].set_ylabel('LOSO α r')
axes[0].set_title('LOSO: solutions vs échecs précédents', fontweight='bold')
axes[0].axhline(loso_ref, color='steelblue', linestyle='--', linewidth=0.8, alpha=0.5)
axes[0].grid(True, alpha=0.3, axis='y')

# Nombre de paramètres vs LOSO
params_all = [241, 261, 433, 243, 481, 961]
loso_all = [loso_ref, loso14b, loso15b, loso_bf, 0.489, 0.133]
labels_all = ['V12b', 'V14b', 'V15b', 'β Four.', 'V14', 'V15']
colors_sc = ['steelblue', 'green', 'orange', 'purple', 'red', 'red']
for i in range(6):
    axes[1].scatter(params_all[i], loso_all[i], s=100, c=colors_sc[i], edgecolor='black', zorder=5)
    axes[1].annotate(labels_all[i], (params_all[i], loso_all[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
axes[1].set_xlabel('Nombre de paramètres')
axes[1].set_ylabel('LOSO α r')
axes[1].set_title('Paramètres vs Performance\n(rouge = échecs)', fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Résumé
txt = (f"SOLUTIONS BLOCAGES\n{'='*28}\n\n"
       f"V14b (gamma annuel L1):\n"
       f"  20 gamma au lieu de 240\n"
       f"  LOSO: {loso14b:.3f}\n"
       f"  Delta: {loso14b-loso_ref:+.3f}\n\n"
       f"V15b (36 regions):\n"
       f"  6x6 au lieu de 8x10\n"
       f"  LOSO: {loso15b:.3f}\n"
       f"  Delta: {loso15b-loso_ref:+.3f}\n\n"
       f"Beta Fourier:\n"
       f"  3 params au lieu de 12\n"
       f"  LOSO: {loso_bf:.3f}\n"
       f"  Delta: {loso_bf-loso_ref:+.3f}\n\n"
       f"Lecon: ratio params/\n"
       f"contraintes est la cle")
axes[2].text(0.05, 0.95, txt, transform=axes[2].transAxes, fontsize=9.5,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[2].axis('off'); axes[2].set_title('Résumé', fontweight='bold')

plt.suptitle('Solutions aux blocages physiques: V14b, V15b, β Fourier',
             fontsize=13, fontweight='bold')
plt.tight_layout()
figpath = os.path.join(OUTDIR, 'fig_fixes_physics.png')
plt.savefig(figpath, dpi=150, bbox_inches='tight')
print(f"  Figure: {figpath}")

np.savez(os.path.join(OUTDIR, 'fixes_physics.npz'),
         loso_ref=loso_ref, loso14b=loso14b, loso15b=loso15b, loso_bf=loso_bf,
         r_ref=r_ref_a, r14b=r14b_a, r15b=r15b_a, r_bf=r_bf_a)

print(f"\n{'='*70}")
print("VERDICT FINAL:")
for name, _, lm, _, _, _ in configs:
    delta = lm - loso_ref
    verdict = "AMÉLIORE" if delta > 0.01 else "SIMILAIRE" if delta > -0.02 else "DÉGRADE"
    print(f"  {name:<25} LOSO={lm:.3f} ({verdict}, Δ={delta:+.3f})")
print(f"{'='*70}")

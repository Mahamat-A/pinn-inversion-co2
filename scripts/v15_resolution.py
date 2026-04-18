#!/usr/bin/env python3
"""
TEST 80 RÉGIONS (8×10) — La résolution α tient-elle ?
=====================================================
V12b utilise 20 régions (4×5). Chaque région = ~130 000 km².
On teste 80 régions (8×10) = ~33 000 km² chacune.

Risque : 80×12 = 960 paramètres vs 240 pour V12b.
Avec 5000 scénarios et 19 stations, ça peut s'effondrer.

Lance : python3 test_80_regions.py
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

# Test : 20 régions vs 80 régions
CONFIGS = [
    {'name': 'V12b (4x5=20)', 'nr_lat': 4, 'nr_lon': 5},
    {'name': 'V15 (8x10=80)', 'nr_lat': 8, 'nr_lon': 10},
]

print("=" * 60)
print("TEST RÉSOLUTION : 20 régions vs 80 régions")
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
cumdays = np.cumsum([0] + [calendar.monthrange(2019, mo + 1)[1] for mo in range(12)])

# === TESTER LES DEUX RÉSOLUTIONS ===
results = {}

for cfg in CONFIGS:
    NR_LAT = cfg['nr_lat']; NR_LON = cfg['nr_lon']
    N_REG = NR_LAT * NR_LON
    N_STATE = N_REG * N_MO
    name = cfg['name']
    
    print(f"\n{'='*60}")
    print(f"Config: {name} ({N_REG} régions, {N_STATE} paramètres α)")
    print(f"{'='*60}")
    
    # Region map
    rmap = np.zeros((n_lat, n_lon), dtype=int)
    ls = n_lat // NR_LAT; lo = n_lon // NR_LON
    for i in range(NR_LAT):
        for j in range(NR_LON):
            rmap[i*ls:(i+1)*ls if i < NR_LAT-1 else n_lat,
                 j*lo:(j+1)*lo if j < NR_LON-1 else n_lon] = i * NR_LON + j
    
    # Scénarios
    print(f"  Scénarios...")
    np.random.seed(42)
    alpha_mo = np.zeros((N_SCENARIOS, N_REG, N_MO))
    beta_gl = np.zeros(N_SCENARIOS)
    co_d = {s: np.zeros((N_SCENARIOS, N_WEEKS)) for s in SN_ALL}
    co_n = {s: np.zeros((N_SCENARIOS, N_WEEKS)) for s in SN_ALL}
    
    for k in range(N_SCENARIOS):
        fk_f = fossil_wk.copy()
        fk_bd = vprm_day_wk.copy(); fk_bn = vprm_night_wk.copy()
        for r in range(N_REG):
            mask = (rmap == r)
            for m in range(N_MO):
                a = 1.0 + 0.5 * (2 * np.random.random() - 1)
                alpha_mo[k, r, m] = a
                w_start = cumdays[m] // 7; w_end = min(cumdays[m + 1] // 7 + 1, 52)
                for w in range(w_start, w_end):
                    fk_f[w][mask] *= a
        b = 1.0 + 0.3 * (2 * np.random.random() - 1)
        fk_bd *= b; fk_bn *= b; beta_gl[k] = b
        fd = fk_f + fk_bd + ocean_wk; fn = fk_f + fk_bn + ocean_wk
        for st in SN_ALL:
            for w in range(N_WEEKS):
                co_d[st][k, w] = np.sum(fp_day_wk[st][w] * fd[w])
                co_n[st][k, w] = np.sum(fp_night_wk[st][w] * fn[w])
        if (k + 1) % 1000 == 0: print(f"    {k+1}/{N_SCENARIOS}")
    
    Y = np.column_stack([alpha_mo.reshape(N_SCENARIOS, -1), beta_gl])
    
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
    
    # PINN
    def build(n_in, nr_lat, nr_lon, n_mo, n_state):
        inp = Input(shape=(n_in,))
        x = Dense(512, activation='gelu')(inp)
        x = Dropout(0.15)(x, training=True); x = LayerNormalization()(x)
        x = Dense(512, activation='gelu')(x)
        x = Dropout(0.15)(x, training=True); x = LayerNormalization()(x)
        x = Dense(256, activation='gelu')(x)
        x = Dropout(0.1)(x, training=True); x = LayerNormalization()(x)
        xa = Dense(nr_lat * nr_lon * 16, activation='gelu')(x)
        xa = Reshape((nr_lat, nr_lon, 16))(xa)
        xa = Conv2DTranspose(64, (3, 3), padding='same', activation='gelu')(xa)
        xa = Conv2DTranspose(32, (3, 3), padding='same', activation='gelu')(xa)
        xa = Conv2D(n_mo, (1, 1), padding='same', activation='linear')(xa)
        xa = Reshape((n_state,))(xa)
        xb = Dense(32, activation='gelu')(x)
        xb = Dense(1, activation='linear')(xb)
        return Model(inputs=inp, outputs=Concatenate()([xa, xb]))
    
    def make_loss(n_state, nr_lat, nr_lon, n_mo):
        def jl(yt, yp):
            at = yt[:, :n_state]; ap = yp[:, :n_state]
            mse = tf.reduce_mean(tf.square(at - ap))
            pr = tf.reduce_mean(tf.square(ap))
            pg = tf.reshape(ap, (-1, nr_lat, nr_lon, n_mo))
            sp = tf.reduce_mean(tf.square(pg[:, 1:, :, :] - pg[:, :-1, :, :])) + \
                 tf.reduce_mean(tf.square(pg[:, :, 1:, :] - pg[:, :, :-1, :]))
            tp = tf.reduce_mean(tf.square(pg[:, :, :, 1:] - pg[:, :, :, :-1]))
            mb = tf.reduce_mean(tf.square(yt[:, n_state:] - yp[:, n_state:]))
            # Plus de lissage pour 80 régions
            sp_weight = 0.05 if N_REG == 20 else 0.10
            return mse + 0.1 * pr + sp_weight * sp + 0.03 * tp + mb
        return jl
    
    print(f"  Entraînement PINN ({N_STATE}+1 params)...")
    sX = StandardScaler(); sY = StandardScaler()
    Xs = sX.fit_transform(X); Ys = sY.fit_transform(Y)
    Xt, Xv, Yt, Yv = train_test_split(Xs, Ys, test_size=0.15, random_state=42)
    
    p = build(X.shape[1], NR_LAT, NR_LON, N_MO, N_STATE)
    p.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
              loss=make_loss(N_STATE, NR_LAT, NR_LON, N_MO), metrics=['mae'])
    p.fit(Xt, Yt, validation_data=(Xv, Yv), batch_size=128, epochs=200, verbose=0,
          callbacks=[EarlyStopping(patience=25, restore_best_weights=True),
                     ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)])
    
    Yp = p.predict(Xv, verbose=0)
    Ypi = sY.inverse_transform(Yp); Yvi = sY.inverse_transform(Yv)
    ra = np.corrcoef(Yvi[:, :N_STATE].flatten(), Ypi[:, :N_STATE].flatten())[0, 1]
    print(f"  {name}: α r = {ra:.4f}")
    
    # LOSO rapide (3 stations)
    print(f"  LOSO rapide...")
    loso_sts = ['OPE', 'GAT', 'TOH']
    loso_rs = []
    for leave in loso_sts:
        sn_t = [s for s in SN_rural if s != leave]; ns_t = len(sn_t)
        Xd_t = np.zeros((N_SCENARIOS, ns_t * N_WEEKS)); Xn_t = np.zeros((N_SCENARIOS, ns_t * N_WEEKS))
        for i, s in enumerate(sn_t):
            dd = co_d[s] - cbgs_d; dn = co_n[s] - cbgs_n
            for w in range(N_WEEKS):
                cdr = CLA_day_wk[s][w] / CLA_dr[w] if CLA_dr[w] > 0 else 1.0
                cnr = CLA_night_wk[s][w] / CLA_nr[w] if CLA_nr[w] > 0 else 1.0
                Xd_t[:, i * N_WEEKS + w] = dd[:, w] * cdr
                Xn_t[:, i * N_WEEKS + w] = dn[:, w] * cnr
        Xl = np.concatenate([Xd_t, Xn_t], axis=1)
        sXl = StandardScaler(); sYl = StandardScaler()
        Xls = sXl.fit_transform(Xl); Yls = sYl.fit_transform(Y)
        Xlt, Xlv, Ylt, Ylv = train_test_split(Xls, Yls, test_size=0.15, random_state=42)
        pl = build(Xl.shape[1], NR_LAT, NR_LON, N_MO, N_STATE)
        pl.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
                   loss=make_loss(N_STATE, NR_LAT, NR_LON, N_MO), metrics=['mae'])
        pl.fit(Xlt, Ylt, validation_data=(Xlv, Ylv), batch_size=128, epochs=150, verbose=0,
               callbacks=[EarlyStopping(patience=20, restore_best_weights=True)])
        Ypl = pl.predict(Xlv, verbose=0)
        Ypli = sYl.inverse_transform(Ypl); Yvli = sYl.inverse_transform(Ylv)
        rl = np.corrcoef(Yvli[:, :N_STATE].flatten(), Ypli[:, :N_STATE].flatten())[0, 1]
        loso_rs.append(rl)
        print(f"    Sans {leave}: r={rl:.4f}")
    
    results[name] = {
        'r': ra, 'loso_mean': np.mean(loso_rs), 'loso_std': np.std(loso_rs),
        'n_reg': N_REG, 'n_state': N_STATE, 'loso_rs': loso_rs
    }

# === RÉSULTATS ===
print(f"\n{'='*60}")
print("RÉSULTATS : 20 vs 80 RÉGIONS")
print(f"{'='*60}")
for name, res in results.items():
    print(f"\n  {name}:")
    print(f"    Régions: {res['n_reg']}, Paramètres: {res['n_state']}+1")
    print(f"    α r = {res['r']:.4f}")
    print(f"    LOSO (3 st): {res['loso_mean']:.4f} ± {res['loso_std']:.4f}")

# === FIGURE ===
print("\n5. Figure...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

names = list(results.keys())
vals_r = [results[n]['r'] for n in names]
vals_loso = [results[n]['loso_mean'] for n in names]
errs_loso = [results[n]['loso_std'] for n in names]

axes[0].bar(range(2), vals_r, color=['steelblue', 'green'], edgecolor='black', width=0.6)
for i, v in enumerate(vals_r):
    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=12)
axes[0].set_xticks(range(2)); axes[0].set_xticklabels(names, fontsize=9)
axes[0].set_ylabel('α r'); axes[0].set_title('Performance α', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(range(2), vals_loso, yerr=errs_loso, color=['steelblue', 'green'],
            edgecolor='black', capsize=5, width=0.6)
for i, v in enumerate(vals_loso):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=12)
axes[1].set_xticks(range(2)); axes[1].set_xticklabels(names, fontsize=9)
axes[1].set_ylabel('LOSO α r'); axes[1].set_title('LOSO', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

txt = (f"RESOLUTION TEST\n{'='*25}\n\n")
for n, res in results.items():
    txt += f"{n}:\n  r={res['r']:.3f}\n  LOSO={res['loso_mean']:.3f}\n  Params={res['n_state']+1}\n\n"
delta_loso = results[names[1]]['loso_mean'] - results[names[0]]['loso_mean']
txt += f"Delta LOSO: {delta_loso:+.3f}\n"
if delta_loso > -0.05:
    txt += "80 regions viables!"
else:
    txt += "80 regions: effondrement"
axes[2].text(0.05, 0.95, txt, transform=axes[2].transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[2].axis('off'); axes[2].set_title('Résumé', fontweight='bold')

plt.suptitle('Test résolution : 20 vs 80 régions', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig_80_regions.png'), dpi=150, bbox_inches='tight')
print(f"  Figure: {OUTDIR}/fig_80_regions.png")

np.savez(os.path.join(OUTDIR, 'test_80_regions.npz'), results=results)

print(f"\n{'='*60}")
for n, res in results.items():
    print(f"  {n}: LOSO={res['loso_mean']:.3f}")
print(f"  Delta LOSO 80 vs 20: {delta_loso:+.3f}")
print(f"{'='*60}")

# %%
import os
import numpy as np 
import xarray as xr
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from matplotlib.path import Path
from scipy import stats
from pathlib import Path

import matplotlib
import seaborn as sns
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

import xgboost as xgb
import scipy
from eofs.xarray import Eof

# %%
ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input"
FEATURE_DIR = ROOT / "data" / "feature"

# %% [markdown]
# ## data

# %% [markdown]
# ### functions

# %%
def vertical_vorticity(u, v):
    """
    Compute vertical component of vorticity (curl) on a lat-lon grid using xarray.
    u, v: xarray.DataArray with dimensions (latitude, longitude)
    Returns: xarray.DataArray of vertical vorticity (same shape as input)
    """
    R = 6.371e6  # Earth's radius [m]
    deg2rad = np.pi / 180.0
    lat = u.y
    coslat = np.cos(lat * deg2rad)

    # ∂(u cosφ)/∂φ
    u_cos = u * coslat
    d_ucos_dphi = u_cos.differentiate('y') / deg2rad  # convert deg -> rad
    # ∂v/∂λ
    d_v_dlam = v.differentiate('x') / deg2rad  # convert deg -> rad
    # vorticity = (1 / (R cosφ)) * (∂v/∂λ - ∂(u cosφ)/∂φ)
    vort = (d_v_dlam - d_ucos_dphi) / (R * coslat) 
    return vort

# %%
def horizontal_divergence(u, v):
    """
    Compute horizontal divergence on a lat-lon grid using xarray.
    u, v : xarray.DataArray with dimensions (latitude, longitude) representing
           zonal (eastward) and meridional (northward) wind components.
    Returns: xarray.DataArray of horizontal divergence (same shape as input).
    """
    R = 6.371e6  # Earth's radius [m]
    deg2rad = np.pi / 180.0
    lat = v.y
    coslat = np.cos(lat * deg2rad)

    # ∂(v cosφ)/∂φ
    v_cos = v * coslat
    d_vcos_dphi = v_cos.differentiate('y') / deg2rad
    # ∂u/∂λ
    d_u_dlam = u.differentiate('x') / deg2rad
    # divergence = (1 / (R cosφ)) * [∂(v cosφ)/∂φ + ∂u/∂λ]
    div = (d_vcos_dphi + d_u_dlam) / (R * coslat)
    return div

# %% [markdown]
# ### data load

# %%
### ORAS5 d20 ###
f_oras5 = xr.open_dataset(INPUT_DIR / 'ORAS5_d20_1958_2025.nc', engine='netcdf4')

ds0 = f_oras5.isel(time_counter=0)

f_oras5 = f_oras5.assign_coords(
    x=("x", ds0.nav_lon[0, :].data),  # x 차원에 대한 경도값
    y=("y", ds0.nav_lat[:, 0].data)    # y 차원에 대한 위도값
).sortby('x').sortby('y')

f_d20 = f_oras5['so20chgt'].rename({'time_counter': 'time'}).drop_vars(["nav_lat", "nav_lon"]).sel(y=slice(-30, 30))
f_d20['time'] = f_d20['time'].to_index().to_period('M').to_timestamp()
f_d20 = f_d20.load()
target_d20 = f_d20.sel(x=slice(50,80), y=slice(-10,-5)).mean(dim='x').mean(dim='y') 

# %%
### ERA5 ###

f_sh = xr.open_dataset(INPUT_DIR / 'ERA5_heatflux_1940_2025.nc', engine='netcdf4').sshf # surface_upward_sensible_heat_flux J m**-2
f_lh = xr.open_dataset(INPUT_DIR / 'ERA5_heatflux_1940_2025.nc', engine='netcdf4').slhf # surface_upward_latent_heat_flux J m**-2
f_sw = xr.open_dataset(INPUT_DIR / 'ERA5_heatflux_1940_2025.nc', engine='netcdf4').ssr # surface_net_downward_shortwave_flux J m**-2
f_lw = xr.open_dataset(INPUT_DIR / 'ERA5_heatflux_1940_2025.nc', engine='netcdf4').str # surface_net_upward_longwave_flux J m**-2
# upward 라고 써있는데 데이터가 살펴보니까 다 downward임.

f_msl = xr.open_dataset(INPUT_DIR / 'ERA5_pressure_1940_2025.nc', engine='netcdf4').msl # air_pressure_at_mean_sea_level Pa
f_tp = xr.open_dataset(INPUT_DIR  /'ERA5_precipitation_1940_2025.nc', engine='netcdf4').tp # Total precipitation m
f_sst = xr.open_dataset(INPUT_DIR / 'ERA5_sst_1940_2025.nc', engine='netcdf4').sst # Sea surface temperature K

f_taux = xr.open_dataset(INPUT_DIR / 'ERA5_windstress_1940_2025.nc', engine='netcdf4').iews # Instantaneous eastward turbulent surface stress N m**-2
f_tauy = xr.open_dataset(INPUT_DIR / 'ERA5_windstress_1940_2025.nc', engine='netcdf4').inss # Instantaneous northward turbulent surface stress N m**-2

def adjust_longitude(da):
    da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180))
    da = da.sortby('longitude').sortby('latitude')
    da = da.rename({'longitude': 'x', 'latitude': 'y', 'valid_time': 'time'})
    da['time'] = da['time'].to_index().to_period('M').to_timestamp()
    da = da.sel(y=slice(-30, 30), time=slice("1958-01-01", "2025-12-01")).reset_coords(drop=True)
    return da

f_sh       = adjust_longitude(f_sh)
f_lh       = adjust_longitude(f_lh)
f_sw       = adjust_longitude(f_sw)
f_lw       = adjust_longitude(f_lw)
f_msl      = adjust_longitude(f_msl)
f_tp       = adjust_longitude(f_tp)
f_sst      = adjust_longitude(f_sst)
f_taux     = adjust_longitude(f_taux)
f_tauy     = adjust_longitude(f_tauy)

f_netheat = f_sh + f_lh + f_sw + f_lw

f_tau_curl = vertical_vorticity(f_taux, f_tauy)
f_tau_curl.name = 'wind_stress_curl'
f_tau_curl.attrs['units'] = 'N m^-3' 

f_tau_div = horizontal_divergence(f_taux, f_tauy)
f_tau_div.name = 'wind_stress_divergence'
f_tau_div.attrs['units'] = 'N m^-3' 

# %%
f_mask = xr.open_dataset(INPUT_DIR / 'ERA5_202401_mask.nc').lsm
f_mask = adjust_longitude(f_mask).squeeze(drop=True)
land_mask = (f_mask > 0.5).astype(int)

f_sh        = xr.where(land_mask == 1, np.nan, f_sh).transpose('time', 'y', 'x')
f_lh        = xr.where(land_mask == 1, np.nan, f_lh).transpose('time', 'y', 'x')
f_sw        = xr.where(land_mask == 1, np.nan, f_sw).transpose('time', 'y', 'x')
f_lw        = xr.where(land_mask == 1, np.nan, f_lw).transpose('time', 'y', 'x')
f_netheat   = xr.where(land_mask == 1, np.nan, f_netheat).transpose('time', 'y', 'x')

f_msl       = xr.where(land_mask == 1, np.nan, f_msl).transpose('time', 'y', 'x')
f_tp        = xr.where(land_mask == 1, np.nan, f_tp).transpose('time', 'y', 'x')
f_sst       = xr.where(land_mask == 1, np.nan, f_sst).transpose('time', 'y', 'x')

f_taux      = xr.where(land_mask == 1, np.nan, f_taux).transpose('time', 'y', 'x')
f_tauy      = xr.where(land_mask == 1, np.nan, f_tauy).transpose('time', 'y', 'x')
f_tau_curl  = xr.where(land_mask == 1, np.nan, f_tau_curl).transpose('time', 'y', 'x') *10**7  # 10-7 N m^-3
f_tau_div   = xr.where(land_mask == 1, np.nan, f_tau_div).transpose('time', 'y', 'x') *10**7  # 10-7 N m^-3

# %% [markdown]
# ## train/test split with sensitivity test

# %%
# lead time 1 month

# ───────────────────────────────────────────────
# 1. lead-n 타깃 생성 (y(t+1) → y_lead1(t)) > 예보 시점을 기준으로 time 정렬 
# ───────────────────────────────────────────────
lead_dict = {}
for n in range(1, 13):                       # lead 1~12개월
    lead_dict[f'y_lead{n}'] = (
        target_d20
        .shift(time=-n)                      # 타깃을 n개월 뒤로 이동
        .dropna(dim='time')                  # 이동 후 생긴 NaN(마지막 n개) 제거
        .sel(time=slice("1958-01-01", "2022-11-01"))  # 기간 필터
    )
    
# ───────────────────────────────────────────────
# 2. feature 세트 준비: 모든 변수에서 타깃과 같은 시점만 유지
# ───────────────────────────────────────────────
feature_dict = {
    'sh'       : f_sh,
    'lh'       : f_lh,
    'sw'       : f_sw,
    'lw'       : f_lw,
    'netheat'  : f_netheat,
    'msl'      : f_msl,
    'tp'       : f_tp,
    'sst'      : f_sst,
    'taux'     : f_taux,
    'tauy'     : f_tauy,
    'tau_curl' : f_tau_curl,
    'tau_div'  : f_tau_div,
    'd20'      : f_d20
}

common_times = lead_dict['y_lead11']['time']
for name, da in feature_dict.items():
    feature_dict[name] = da.sel(time=common_times)

for SEED in range(1, 500+1):
    # ───────────────────────────────────────────────
    # 3. 날짜 단위 무작위 분할 (66년, 792개월 → 633개 / 159개)
    # ───────────────────────────────────────────────
    train_times, test_times = train_test_split(
        common_times.values,          # numpy 배열로 전달
        test_size=0.2,                # 20 % test
        shuffle=True,
        random_state=SEED
    )

    # ───────────────────────────────────────────────
    # 4. train / test 데이터셋 만들기
    # ───────────────────────────────────────────────
    # ① 입력 변수(X) : feature_dict 그대로
    X_train = {name: da.sel(time=train_times) for name, da in feature_dict.items()}
    X_test  = {name: da.sel(time=test_times)  for name, da in feature_dict.items()}

    # ② 타깃(y) : lead 1-12 month
    y_train_dict = {}
    y_test_dict  = {}
    for n in range(1, 13):
        y_train_dict[f'y_lead{n}'] = lead_dict[f'y_lead{n}'].sel(time=train_times)
        y_test_dict [f'y_lead{n}'] = lead_dict[f'y_lead{n}'].sel(time=test_times)

    # %% [markdown]
    # ## Anomaly

    # %%

    for var in list(X_train.keys()):                      # keys 복사 후 순회
        train_da = X_train[var]                           # (time, y, x)
        clim = train_da.groupby('time.month').mean('time')

        X_train[f'{var}_ano'] = train_da.groupby('time.month') - clim
        X_test [f'{var}_ano'] = X_test [var].groupby('time.month') - clim

        X_train[f'{var}_ano'] = X_train[f'{var}_ano'].drop_vars(["month"])
        X_test [f'{var}_ano'] = X_test[f'{var}_ano'].drop_vars(["month"])

        del X_train[var]
        del X_test[var]

    for n in range(1, 13):
        key = f'y_lead{n}'
        clim = y_train_dict[key].groupby('time.month').mean('time')

        y_train_dict[f'{key}_ano'] = y_train_dict[key].groupby('time.month') - clim
        y_test_dict [f'{key}_ano'] = y_test_dict [key].groupby('time.month') - clim

        y_train_dict[f'{key}_ano'] = y_train_dict[f'{key}_ano'].drop_vars(["month"])
        y_test_dict [f'{key}_ano'] = y_test_dict[f'{key}_ano'].drop_vars(["month"])

        del y_train_dict[key]
        del y_test_dict[key]

    # %% [markdown]
    # ## Make x variables 

    # %% [markdown]
    # ### EOF: Indian ocean

    # %%
    pc_dict_io  = {}     # key: '<var>_pc'  → (time, mode)
    eof_dict_io = {}     # key: '<var>_eof' → (mode, y, x)
    solver_dict_io  = {}         # <var>      → Eof solver 객체
    weight_dict_io  = {}         # <var>      → (y, x) √cosφ 가중치

    K = 5

    for var in X_train.keys():          
        da = X_train[var].sel(x=slice(29.9, 120))  # (time, y, x)  — 인도양만

        w_1d = np.sqrt(np.cos(np.deg2rad(da['y'])))     # ① 면적 가중치 (√cosφ) 
        w_2d, _  = xr.broadcast(w_1d, da.isel(time=0))
        solver = Eof(da, weights=w_2d)                  # ② EOF 분석
        pcs  = solver.pcs(npcs=K, pcscaling=1)          # ③ PC 추출 (time, mode)
        eofs = solver.eofs(neofs=K, eofscaling=1)       #   EOF 추출 (mode, y, x)

        pc_dict_io [f'{var}_pc']  = pcs                    # ④ 저장
        eof_dict_io[f'{var}_eof'] = eofs
        solver_dict_io[var]       = solver      # test용으로 저장
        weight_dict_io[var]       = w_2d

    # %%
    lead_keys   = [f'y_lead{n}_ano' for n in range(1, 13)]
    row_labels  = []      # '<var>_pc_mode<mode>'
    r_rows      = []      # 각 행: lead1~12 r
    p_rows      = []      # 각 행: lead1~12 p

    for pc_key, pc_da in pc_dict_io.items():               # ex) 'sst_ano_pc'
        for mode_val in pc_da['mode'].values:           # 1~5
            pc_series = pc_da.sel(mode=mode_val)

            r_list, p_list = [], []
            for y_key in lead_keys:
                y_series = y_train_dict[y_key]
                r, p = pearsonr(pc_series.values, y_series.values)
                r_list.append(r)
                p_list.append(p)

            row_labels.append(f'{pc_key}_{int(mode_val)+1}')
            r_rows.append(r_list)
            p_rows.append(p_list)

    corr_df_io = pd.DataFrame(r_rows, index=row_labels, columns=[f'lead{n}' for n in range(1, 13)])
    p_df_io    = pd.DataFrame(p_rows, index=row_labels, columns=corr_df_io.columns)

    # %%
    mask = (p_df_io < 0.01) & (corr_df_io.abs() > 0.45)   # 조건 TRUE/FALSE

    selected_rows = mask.any(axis=1)         
    leadagnostic_features_io = corr_df_io.index[selected_rows].tolist()

    series_list = []          # 각 PC → pandas.Series
    col_names   = []          # 동일 순서로 컬럼명 저장

    for feat in leadagnostic_features_io:
        base, mode_str = feat.rsplit('_', 1)     # 'sst_ano_pc', '1'
        mode_val = int(mode_str) - 1 

        da = pc_dict_io[base]                 # (time, mode)
        pc_series = da.sel(mode=mode_val)  # (time,)

        series_list.append(pc_series.to_series())    
        col_names.append(feat)                   

    X_eof_io_train = pd.concat(series_list, axis=1, join='inner')
    X_eof_io_train.columns = col_names

    # %%
    # What we need ---------------------------------------
    # • leadagnostic_features_io : ['sst_ano_pc_mode1', 'taux_ano_pc_mode3', ...]
    # • solver_dict_io           : {var: Eof solver (train에서 학습)}
    # • weight_dict_io           : {var: 2-D √cosφ 가중치 (y,x)}
    # • X_test                   : {var: DataArray(time,y,x)}  ← test 세트
    # • K                        : EOF 모드 수 (train에서 사용한 것: 5)
    # ----------------------------------------------------

    series_list = []                 # 각 선택 PC → pandas.Series
    col_names   = []                 # 동일 순서로 컬럼명 저장

    for feat in leadagnostic_features_io:
        base, mode_str = feat.rsplit('_', 1)   # 'sst_ano_pc', '1'
        mode_val = int(mode_str) - 1 
        var      = base.replace('_pc', '')         # 원 변수명: 'sst_ano'

        da_test = X_test[var].sel(x=slice(29.9, 120))       # (time, y, x)

        solver  = solver_dict_io[var]
        w_2d    = weight_dict_io[var]
        pcs_all = solver.projectField(
            da_test, neofs=K, eofscaling=1
        )                                          # (time, mode)
        pc_series = pcs_all.sel(mode=mode_val)     # (time,) 선택 PC 시계열

        series_list.append(pc_series.to_series())  # time index 유지
        col_names.append(feat)

    X_eof_io_test = pd.concat(series_list, axis=1, join='inner')
    X_eof_io_test.columns = col_names

    # %% [markdown]
    # ### EOF: SCTR region

    # %%
    pc_dict_sctr  = {}     # key: '<var>_pc'  → (time, mode)
    eof_dict_sctr = {}     # key: '<var>_eof' → (mode, y, x)
    solver_dict_sctr  = {}         # <var>      → Eof solver 객체
    weight_dict_sctr  = {}         # <var>      → (y, x) √cosφ 가중치

    K = 5

    for var in X_train.keys():          
        da = X_train[var].sel(x=slice(50,80)).sel(y=slice(-10,-5))  # (time, y, x)  — 인도양만

        w_1d = np.sqrt(np.cos(np.deg2rad(da['y'])))     # ① 면적 가중치 (√cosφ) 
        w_2d, _  = xr.broadcast(w_1d, da.isel(time=0))
        solver = Eof(da, weights=w_2d)                  # ② EOF 분석
        pcs  = solver.pcs(npcs=K, pcscaling=1)          # ③ PC 추출 (time, mode)
        eofs = solver.eofs(neofs=K, eofscaling=1)       #   EOF 추출 (mode, y, x)

        pc_dict_sctr [f'{var}_pc']  = pcs                    # ④ 저장
        eof_dict_sctr[f'{var}_eof'] = eofs
        solver_dict_sctr[var]       = solver      # test용으로 저장
        weight_dict_sctr[var]       = w_2d

    # %%
    lead_keys   = [f'y_lead{n}_ano' for n in range(1, 13)]
    row_labels  = []      # '<var>_pc_mode<mode>'
    r_rows      = []      # 각 행: lead1~12 r
    p_rows      = []      # 각 행: lead1~12 p

    for pc_key, pc_da in pc_dict_sctr.items():               # ex) 'sst_ano_pc'
        for mode_val in pc_da['mode'].values:           # 1~5
            pc_series = pc_da.sel(mode=mode_val)

            r_list, p_list = [], []
            for y_key in lead_keys:
                y_series = y_train_dict[y_key]
                r, p = pearsonr(pc_series.values, y_series.values)
                r_list.append(r)
                p_list.append(p)

            row_labels.append(f'{pc_key}_{int(mode_val)+1}')
            r_rows.append(r_list)
            p_rows.append(p_list)

    corr_df_sctr = pd.DataFrame(r_rows, index=row_labels, columns=[f'lead{n}' for n in range(1, 13)])
    p_df_sctr    = pd.DataFrame(p_rows, index=row_labels, columns=corr_df_sctr.columns)

    # %%
    mask = (p_df_sctr < 0.01) & (corr_df_sctr.abs() > 0.45)   # 조건 TRUE/FALSE

    selected_rows = mask.any(axis=1)         
    leadagnostic_features_sctr = corr_df_sctr.index[selected_rows].tolist()

    series_list = []          # 각 PC → pandas.Series
    col_names   = []          # 동일 순서로 컬럼명 저장

    for feat in leadagnostic_features_sctr:
        base, mode_str = feat.rsplit('_', 1)     # 'sst_ano_pc', '1'
        mode_val = int(mode_str) - 1 

        da = pc_dict_sctr[base]                 # (time, mode)
        pc_series = da.sel(mode=mode_val)  # (time,)

        series_list.append(pc_series.to_series())    
        col_names.append(feat)                   

    X_eof_sctr_train = pd.concat(series_list, axis=1, join='inner')
    X_eof_sctr_train.columns = col_names

    # %%
    # What we need ---------------------------------------
    # • leadagnostic_features_sctr : ['sst_ano_pc_mode1', 'taux_ano_pc_mode3', ...]
    # • solver_dict_sctr           : {var: Eof solver (train에서 학습)}
    # • weight_dict_sctr           : {var: 2-D √cosφ 가중치 (y,x)}
    # • X_test                     : {var: DataArray(time,y,x)}  ← test 세트
    # • K                          : EOF 모드 수 (train에서 사용한 것: 5)
    # ----------------------------------------------------

    series_list = []                 # 각 선택 PC → pandas.Series
    col_names   = []                 # 동일 순서로 컬럼명 저장

    for feat in leadagnostic_features_sctr:
        base, mode_str = feat.rsplit('_', 1)   # 'sst_ano_pc', '1'
        mode_val = int(mode_str) - 1 
        var      = base.replace('_pc', '')         # 원 변수명: 'sst_ano'

        da_test = X_test[var].sel(x=slice(50,80)).sel(y=slice(-10,-5))       # (time, y, x)

        solver  = solver_dict_sctr[var]
        w_2d    = weight_dict_sctr[var]
        pcs_all = solver.projectField(
            da_test, neofs=K, eofscaling=1
        )                                          # (time, mode)
        pc_series = pcs_all.sel(mode=mode_val)     # (time,) 선택 PC 시계열

        series_list.append(pc_series.to_series())  # time index 유지
        col_names.append(feat)

    X_eof_sctr_test = pd.concat(series_list, axis=1, join='inner')
    X_eof_sctr_test.columns = col_names

    # %% [markdown]
    # ### EOF: Pacific region

    # %%
    pc_dict_po  = {}     # key: '<var>_pc'  → (time, mode)
    eof_dict_po = {}     # key: '<var>_eof' → (mode, y, x)
    solver_dict_po  = {}         # <var>      → Eof solver 객체
    weight_dict_po  = {}         # <var>      → (y, x) √cosφ 가중치

    K = 5

    for var in X_train.keys():  
        da = X_train[var]
        da = da.assign_coords(x=(da['x'] % 360))
        da = da.sortby('x')        
        da = da.sel(x=slice(120, 280), y=slice(-20, 20))

        w_1d = np.sqrt(np.cos(np.deg2rad(da['y'])))     # ① 면적 가중치 (√cosφ) 
        w_2d, _  = xr.broadcast(w_1d, da.isel(time=0))
        solver = Eof(da, weights=w_2d)                  # ② EOF 분석
        pcs  = solver.pcs(npcs=K, pcscaling=1)          # ③ PC 추출 (time, mode)
        eofs = solver.eofs(neofs=K, eofscaling=1)       #   EOF 추출 (mode, y, x)

        pc_dict_po [f'{var}_pc']  = pcs                    # ④ 저장
        eof_dict_po[f'{var}_eof'] = eofs
        solver_dict_po[var]       = solver      # test용으로 저장
        weight_dict_po[var]       = w_2d

    # %%
    lead_keys   = [f'y_lead{n}_ano' for n in range(1, 13)]
    row_labels  = []      # '<var>_pc_mode<mode>'
    r_rows      = []      # 각 행: lead1~12 r
    p_rows      = []      # 각 행: lead1~12 p

    for pc_key, pc_da in pc_dict_po.items():               # ex) 'sst_ano_pc'
        for mode_val in pc_da['mode'].values:           # 0~4 > 1~5
            pc_series = pc_da.sel(mode=mode_val)

            r_list, p_list = [], []
            for y_key in lead_keys:
                y_series = y_train_dict[y_key]
                r, p = pearsonr(pc_series.values, y_series.values)
                r_list.append(r)
                p_list.append(p)

            row_labels.append(f'{pc_key}_{int(mode_val)+1}')
            r_rows.append(r_list)
            p_rows.append(p_list)

    corr_df_po = pd.DataFrame(r_rows, index=row_labels, columns=[f'lead{n}' for n in range(1, 13)])
    p_df_po    = pd.DataFrame(p_rows, index=row_labels, columns=corr_df_po.columns)

    # %%
    mask = (p_df_po < 0.01) & (corr_df_po.abs() > 0.45)   # 조건 TRUE/FALSE

    selected_rows = mask.any(axis=1)         
    leadagnostic_features_po = corr_df_po.index[selected_rows].tolist()

    series_list = []          # 각 PC → pandas.Series
    col_names   = []          # 동일 순서로 컬럼명 저장

    for feat in leadagnostic_features_po:
        base, mode_str = feat.rsplit('_', 1)     # 'sst_ano_pc', '1'
        mode_val = int(mode_str) - 1 

        da = pc_dict_po[base]                 # (time, mode)
        pc_series = da.sel(mode=mode_val)  # (time,)

        series_list.append(pc_series.to_series())    
        col_names.append(feat)                   

    if len(leadagnostic_features_po) == 0:
        print("⚠️ No PC modes passed selection threshold. Skipping PO features.")
    else:
        X_eof_po_train = pd.concat(series_list, axis=1, join='inner')
        X_eof_po_train.columns = col_names

    # %%
    # What we need ---------------------------------------
    # • leadagnostic_features_io : ['sst_ano_pc_mode1', 'taux_ano_pc_mode3', ...]
    # • solver_dict_io           : {var: Eof solver (train에서 학습)}
    # • weight_dict_io           : {var: 2-D √cosφ 가중치 (y,x)}
    # • X_test                   : {var: DataArray(time,y,x)}  ← test 세트
    # • K                        : EOF 모드 수 (train에서 사용한 것: 5)
    # ----------------------------------------------------

    series_list = []                 # 각 선택 PC → pandas.Series
    col_names   = []                 # 동일 순서로 컬럼명 저장

    for feat in leadagnostic_features_po:
        base, mode_str = feat.rsplit('_', 1)   # 'sst_ano_pc', '1'
        mode_val = int(mode_str) - 1 
        var      = base.replace('_pc', '')         # 원 변수명: 'sst_ano'

        da_test = X_test[var]
        da_test = da_test.assign_coords(x=(da_test['x'] % 360))
        da_test = da_test.sortby('x')        
        da_test = da_test.sel(x=slice(120, 280), y=slice(-20, 20))
        valid_mask = (~np.isnan(da_test)).all(dim='time')
        da_test = da_test.where(valid_mask)

        solver  = solver_dict_po[var]
        w_2d    = weight_dict_po[var]
        pcs_all = solver.projectField(
            da_test, neofs=K, eofscaling=1
        )                                          # (time, mode)
        pc_series = pcs_all.sel(mode=mode_val)     # (time,) 선택 PC 시계열

        series_list.append(pc_series.to_series())  # time index 유지
        col_names.append(feat)

    if len(leadagnostic_features_po) == 0:
        print("⚠️ No PC modes passed selection threshold. Skipping PO features.")
    else:
        X_eof_po_test = pd.concat(series_list, axis=1, join='inner')
        X_eof_po_test.columns = col_names

    # %% [markdown]
    # ### Climate index

    # %%
    w_lat = np.cos(np.deg2rad(X_train['sst_ano']['y']))

    # 1. DMI (Dipole Mode Index)
    west = X_train['sst_ano'].sel(x=slice(50, 70), y=slice(-10, 10)) # ① 서쪽 50°E–70°E, 10°S–10°N
    w_west = w_lat.sel(y=slice(-10, 10))
    west_mean = west.weighted(w_west).mean(dim=['y', 'x'])

    east = X_train['sst_ano'].sel(x=slice(90, 110), y=slice(-10, 0)) # ② 동쪽 90°E–110°E, 10°S–0°N
    w_east = w_lat.sel(y=slice(-10, 0))
    east_mean = east.weighted(w_east).mean(dim=['y', 'x'])

    dmi_series = (west_mean - east_mean).to_series() # ③ DMI = 서쪽 − 동쪽
    dmi_series.name = 'DMI'

    # 2. NINO3.4 (170°W–120°W, 5°S–5°N)
    nino_box = X_train['sst_ano'].sel(x=slice(-170, -120), y=slice(-5, 5))

    w_nino = w_lat.sel(y=slice(-5, 5))
    nino34_ts = nino_box.weighted(w_nino).mean(dim=['y', 'x'])
    nino34_series = nino34_ts.to_series()
    nino34_series.name = 'NINO34'

    X_indices_train = pd.concat([dmi_series, nino34_series], axis=1, join='inner')

    # %%
    w_lat = np.cos(np.deg2rad(X_train['sst_ano']['y']))

    # 1. DMI (Dipole Mode Index)
    west = X_test['sst_ano'].sel(x=slice(50, 70), y=slice(-10, 10)) # ① 서쪽 50°E–70°E, 10°S–10°N
    w_west = w_lat.sel(y=slice(-10, 10))
    west_mean = west.weighted(w_west).mean(dim=['y', 'x'])

    east = X_test['sst_ano'].sel(x=slice(90, 110), y=slice(-10, 0)) # ② 동쪽 90°E–110°E, 10°S–0°N
    w_east = w_lat.sel(y=slice(-10, 0))
    east_mean = east.weighted(w_east).mean(dim=['y', 'x'])

    dmi_series = (west_mean - east_mean).to_series() # ③ DMI = 서쪽 − 동쪽
    dmi_series.name = 'DMI'

    # 2. NINO3.4 (170°W–120°W, 5°S–5°N)
    nino_box = X_test['sst_ano'].sel(x=slice(-170, -120), y=slice(-5, 5))

    w_nino = w_lat.sel(y=slice(-5, 5))
    nino34_ts = nino_box.weighted(w_nino).mean(dim=['y', 'x'])
    nino34_series = nino34_ts.to_series()
    nino34_series.name = 'NINO34'

    X_indices_test = pd.concat([dmi_series, nino34_series], axis=1, join='inner')

    # %% [markdown]
    # ### Extra variables

    # %%
    d20_series_train = (X_train['d20_ano'].sel(x=slice(50, 80), y=slice(-10, -5)).mean(dim=['y', 'x']).to_series())
    d20_series_train.name = 'd20_ano_sctr'

    X_d20_train = pd.concat([d20_series_train], axis=1)

    # %%
    d20_series_test = X_test['d20_ano'].sel(x=slice(50,80)).sel(y=slice(-10,-5)).mean(dim=['y', 'x']).to_series()
    d20_series_test.name = 'd20_ano_sctr'

    X_d20_test = pd.concat([d20_series_test], axis=1)

    # %% [markdown]
    # ## Get dataframe

    # %%
    X_eof_io_train   = X_eof_io_train.add_suffix('_IO')
    X_eof_sctr_train = X_eof_sctr_train.add_suffix('_SCTR')
    
    X_eof_io_test    = X_eof_io_test.add_suffix('_IO')
    X_eof_sctr_test  = X_eof_sctr_test.add_suffix('_SCTR')

    if len(leadagnostic_features_po) == 0:
        print("⚠️ No PC modes passed selection threshold. Skipping PO features.")
    else:    
        X_eof_po_train   = X_eof_po_train.add_suffix('_PO')
        X_eof_po_test    = X_eof_po_test.add_suffix('_PO')


    # %%

    if len(leadagnostic_features_po) == 0:
        train_frames = [X_d20_train, X_indices_train, X_eof_sctr_train, X_eof_io_train]
        test_frames  = [X_d20_test, X_indices_test,  X_eof_sctr_test, X_eof_io_test]
    else:    
        train_frames = [X_d20_train, X_indices_train, X_eof_sctr_train, X_eof_io_train, X_eof_po_train]
        test_frames  = [X_d20_test, X_indices_test,  X_eof_sctr_test, X_eof_io_test, X_eof_po_test]

    X_full_train = pd.concat(train_frames, axis=1, join='inner')
    X_full_test  = pd.concat(test_frames,  axis=1, join='inner')

    print(f"▶ train {SEED}:", X_full_train.shape)
    print(f"▶ test  {SEED}:", X_full_test.shape)

    # %%
    ser_list_train = []
    col_names_train = []

    for key, arr in y_train_dict.items():          # key: 'y_lead1_ano', ...
        # xarray DataArray → pandas Series (index = time)
        ser = arr.to_series() if hasattr(arr, "to_series") else pd.Series(arr)
        ser_list_train.append(ser)
        col_names_train.append(key)

    y_train_df = pd.concat(ser_list_train, axis=1, join='inner')
    y_train_df.columns = col_names_train

    ser_list_test = []
    col_names_test = []

    for key, arr in y_test_dict.items():
        ser = arr.to_series() if hasattr(arr, "to_series") else pd.Series(arr)
        ser_list_test.append(ser)
        col_names_test.append(key)

    y_test_df = pd.concat(ser_list_test, axis=1, join='inner')
    y_test_df.columns = col_names_test

    print(f"train y shape {SEED}:", y_train_df.shape)
    print(f"test  y shape {SEED}:", y_test_df.shape)

    X_full_train.to_csv(FEATURE_DIR / f'X_train_{SEED}.csv')
    X_full_test.to_csv(FEATURE_DIR  / f'X_test_{SEED}.csv')
    y_train_df.to_csv(FEATURE_DIR   / f'y_train_{SEED}.csv')
    y_test_df .to_csv(FEATURE_DIR   / f'y_test_{SEED}.csv')
    # %% [markdown]
    # ## 



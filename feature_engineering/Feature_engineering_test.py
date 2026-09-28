# %%
import os
import numpy as np 
import xarray as xr
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from matplotlib.path import Path
from scipy import stats

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
# f_oras5 = xr.open_dataset(dir+'input/ORAS5_d20_1958_2024.nc')
f_oras5 = xr.open_dataset(INPUT_DIR / 'ORAS5_d20_1958_2025.nc', engine='netcdf4')

ds0 = f_oras5.isel(time_counter=0)

f_oras5 = f_oras5.assign_coords(
    x=("x", ds0.nav_lon[0, :].data),  # x 차원에 대한 경도값
    y=("y", ds0.nav_lat[:, 0].data)    # y 차원에 대한 위도값
).sortby('x').sortby('y')

f_d20 = f_oras5['so20chgt'].rename({'time_counter': 'time'}).drop_vars(["nav_lat", "nav_lon"]).sel(y=slice(-30, 30))
f_d20['time'] = f_d20['time'].to_index().to_period('M').to_timestamp()
target_d20 = f_d20.sel(x=slice(50,80), y=slice(-10,-5)).mean(dim='x').mean(dim='y') 

# %%
f_d20

# %%
### ERA5 ###

f_sh = xr.open_dataset(INPUT_DIR / 'ERA5_heatflux_1940_2025.nc', engine='netcdf4').sshf # surface_upward_sensible_heat_flux J m**-2
f_lh = xr.open_dataset(INPUT_DIR / 'ERA5_heatflux_1940_2025.nc', engine='netcdf4').slhf # surface_upward_latent_heat_flux J m**-2
f_sw = xr.open_dataset(INPUT_DIR / 'ERA5_heatflux_1940_2025.nc', engine='netcdf4').ssr # surface_net_downward_shortwave_flux J m**-2
f_lw = xr.open_dataset(INPUT_DIR / 'ERA5_heatflux_1940_2025.nc', engine='netcdf4').str # surface_net_upward_longwave_flux J m**-2
# upward 라고 써있는데 데이터가 살펴보니까 다 downward임.

f_msl = xr.open_dataset(INPUT_DIR / 'ERA5_pressure_1940_2025.nc', engine='netcdf4').msl # air_pressure_at_mean_sea_level Pa
f_tp = xr.open_dataset(INPUT_DIR / 'ERA5_precipitation_1940_2025.nc', engine='netcdf4').tp # Total precipitation m
f_sst = xr.open_dataset(INPUT_DIR / 'ERA5_sst_1940_2025.nc', engine='netcdf4').sst # Sea surface temperature K

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

# %%
f_mask = xr.open_dataset(INPUT_DIR / 'ERA5_202401_mask.nc').lsm
f_mask = adjust_longitude(f_mask).squeeze(drop=True)
land_mask = (f_mask > 0.5).astype(int)

f_sh        = xr.where(land_mask == 1, np.nan, f_sh).transpose('time', 'y', 'x')
f_lh        = xr.where(land_mask == 1, np.nan, f_lh).transpose('time', 'y', 'x')
f_sw        = xr.where(land_mask == 1, np.nan, f_sw).transpose('time', 'y', 'x')
f_lw        = xr.where(land_mask == 1, np.nan, f_lw).transpose('time', 'y', 'x')

f_msl       = xr.where(land_mask == 1, np.nan, f_msl).transpose('time', 'y', 'x')
f_tp        = xr.where(land_mask == 1, np.nan, f_tp).transpose('time', 'y', 'x')
f_sst       = xr.where(land_mask == 1, np.nan, f_sst).transpose('time', 'y', 'x')

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
        #.dropna(dim='time')                  # 이동 후 생긴 NaN(마지막 n개) 제거 > 여기서는 어차피 test로 들어갈 거라 필요 X
        .sel(time=slice("1958-01-01", "2025-12-01"))  # 기간 필터
    )
    
# ───────────────────────────────────────────────
# 2. feature 세트 준비: 모든 변수에서 타깃과 같은 시점만 유지
# ───────────────────────────────────────────────
feature_dict = {
    'sh'       : f_sh,
    'lh'       : f_lh,
    'sw'       : f_sw,
    'lw'       : f_lw,
    'msl'      : f_msl,
    'tp'       : f_tp,
    'sst'      : f_sst,
    'd20'      : f_d20
}

common_times = lead_dict['y_lead11']['time']
for name, da in feature_dict.items():
    feature_dict[name] = da.sel(time=common_times)

# %%
# ───────────────────────────────────────────────
# 3. 768 / 48 로 분할 됨
# ───────────────────────────────────────────────
common_times = lead_dict['y_lead11']['time']

train_end = pd.Timestamp("2022-12-01")
test_start = pd.Timestamp("2023-01-01")

train_times = common_times[common_times <= train_end]
test_times  = common_times[common_times >= test_start]

train_times = train_times.values
test_times  = test_times.values

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
# core_vars: ['DMI', 'NINO34', 'd20_ano_sctr',
# 
# 'd20_ano_pc_1_SCTR', 'msl_ano_pc_2_SCTR',
# 
# 'd20_ano_pc_2_IO', 'sst_ano_pc_2_IO', 'lw_ano_pc_1_IO', 'sw_ano_pc_1_IO', 'tp_ano_pc_1_IO',
# 
# 'lh_ano_pc_2_PO', 'sh_ano_pc_1_PO',]

# %% [markdown]
# ### EOF: Indian ocean

# %%
pc_dict_io  = {}     # key: '<var>_pc'  → (time, mode)
eof_dict_io = {}     # key: '<var>_eof' → (mode, y, x)
solver_dict_io  = {}         # <var>      → Eof solver 객체
weight_dict_io  = {}         # <var>      → (y, x) √cosφ 가중치

K = 2
target_vars = ['d20_ano', 'sw_ano', 'lw_ano', 'tp_ano', 'sst_ano']

for var in target_vars:
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
leadagnostic_features_io = ['sw_ano_pc_1', 'lw_ano_pc_1', 'tp_ano_pc_1', 'sst_ano_pc_2', 'd20_ano_pc_2']

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

K = 2
target_vars = ['d20_ano', 'msl_ano']

for var in target_vars:
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
leadagnostic_features_sctr = ['d20_ano_pc_1', 'msl_ano_pc_2']

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

K = 2
target_vars = ['lh_ano', 'sh_ano']

for var in target_vars:
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
leadagnostic_features_po = ['lh_ano_pc_2', 'sh_ano_pc_1']

series_list = []          # 각 PC → pandas.Series
col_names   = []          # 동일 순서로 컬럼명 저장

for feat in leadagnostic_features_po:
    base, mode_str = feat.rsplit('_', 1)     # 'sst_ano_pc', '1'
    mode_val = int(mode_str) - 1 

    da = pc_dict_po[base]                 # (time, mode)
    pc_series = da.sel(mode=mode_val)  # (time,)

    series_list.append(pc_series.to_series())    
    col_names.append(feat)                   

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

# X_indices_train = pd.concat([dmi_series, nino34_series], axis=1, join='inner')

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
X_eof_po_train   = X_eof_po_train.add_suffix('_PO')

X_eof_io_test    = X_eof_io_test.add_suffix('_IO')
X_eof_sctr_test  = X_eof_sctr_test.add_suffix('_SCTR')
X_eof_po_test    = X_eof_po_test.add_suffix('_PO')

# %%
train_frames = [X_d20_train, X_indices_train, X_eof_sctr_train, X_eof_io_train, X_eof_po_train]
test_frames  = [X_d20_test, X_indices_test,  X_eof_sctr_test, X_eof_io_test, X_eof_po_test]

X_full_train = pd.concat(train_frames, axis=1, join='inner')
X_full_test  = pd.concat(test_frames,  axis=1, join='inner')

print("▶ train :", X_full_train.shape)
print("▶ test  :", X_full_test.shape)

X_full_train.to_csv(FEATURE_DIR + '/X_train.csv')
X_full_test.to_csv(FEATURE_DIR + '/X_test.csv')

# %%
X_full_train

# %%
X_full_test

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

print("train y shape :", y_train_df.shape)
print("test  y shape :", y_test_df.shape)

y_train_df.to_csv(FEATURE_DIR / 'y_train.csv')
y_test_df .to_csv(FEATURE_DIR / 'y_test.csv')

# %%
y_train_df

# %%
y_test_df

# %%


# %%




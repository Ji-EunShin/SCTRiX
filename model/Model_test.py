# %%
import os
import numpy as np 
import xarray as xr
import matplotlib.pyplot as plt
import datetime
import pandas as pd 
from matplotlib.path import Path
from scipy.stats import pearsonr
import matplotlib.cm as cm
import time

import matplotlib
import seaborn as sns
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature 

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer

import xgboost as xgb
import scipy
from eofs.xarray import Eof
import joblib
from pathlib import Path

from pathlib import Path

# %% [markdown]
# ## data

# %%
ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "feature"
OUT_DIR = ROOT / "output" / "model_test"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# %%
X_train = pd.read_csv(FEATURE_DIR / '/X_train.csv', index_col=0, parse_dates=True)
X_test = pd.read_csv(FEATURE_DIR / '/X_test.csv', index_col=0, parse_dates=True)
y_train_all = pd.read_csv(FEATURE_DIR / '/y_train.csv', index_col=0, parse_dates=True)
y_test_all = pd.read_csv(FEATURE_DIR / '/y_test.csv', index_col=0, parse_dates=True)

feature_cols = X_train.columns.tolist()     

for lead in range(1, 13):  # 2023.01 D20 자료가 존재 22년 1월 lead 12 / 22년 12월 lead 1이 가능함
    col = f"y_lead{lead}_ano"
    k = lead - 1
    if k > 0:
        y_train_all.loc[y_train_all.index[-k:], col] = np.nan

# %%
y_test_all

# %%
y_train_all

# %%
def corr_distance(y_true, y_pred):
    r, _ = pearsonr(y_true, y_pred)
    return 1.0 - r           # 0이 최우수, 2가 최악

corr_dist_score = make_scorer(corr_distance, greater_is_better=False)

# %%
total_start = time.perf_counter()

results          = []          # 평가 메트릭
importance_dict  = {}          # feature importance
model_dict       = {}          # 학습된 모델
df_result_dict   = {}          # 관측/예측/오차 테이블 
time_results = []              # 소요시간 저장용

param_grid = {
    'n_estimators'     : [100, 150, 200, 250, 300],
    'learning_rate'    : [0.03, 0.05, 0.07, 0.1, 0.2],
    'max_depth'        : [2, 4, 6, ], #8, 10],
    'subsample'        : [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0],
}

for lead in range(1, 13):
    print(f"\n Lead {lead} month ")
    lead_start = time.perf_counter()

    # --- (1) y 선택 ---
    y_col = f"y_lead{lead}_ano"
    y_train = y_train_all[y_col].dropna()
    y_test  = y_test_all [y_col].dropna()

    common_idx_train = X_train.index.intersection(y_train.index)
    X_train_sync = X_train.loc[common_idx_train]

    common_idx_test = X_test.index.intersection(y_test.index)
    X_test_sync = X_test.loc[common_idx_test]

    # --- (2) 스케일러 ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sync)
    X_test_scaled  = scaler.transform(X_test_sync)

    # --- (3) RandomizedSearchCV ---
    xgb_regressor = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=2024,
        n_jobs=1
    )

    random_search = RandomizedSearchCV(
        estimator=xgb_regressor,
        param_distributions=param_grid,
        n_iter=100,
        scoring=corr_dist_score, # 'neg_mean_squared_error', # corr_dist_score
        cv=5,
        random_state=2024,
        n_jobs=1,
        verbose=0,
    )

    random_search.fit(X_train_scaled, y_train)
    best_model = random_search.best_estimator_
    train_time = time.perf_counter() - lead_start

    # --- (4) 예측 & 평가 ---
    infer_start = time.perf_counter()
    y_test_pred = best_model.predict(X_test_scaled)
    infer_time = time.perf_counter() - infer_start

    # --- (4-1) start_time -> target_time으로 미루고, 원하는 타겟 기간만 잘라내기 ---
    target_time_test = y_test.index + pd.DateOffset(months=lead)

    obs_s  = pd.Series(y_test.to_numpy(),     index=target_time_test, name="obs")     # obs/pred를 target_time 인덱스로 변환
    pred_s = pd.Series(y_test_pred,           index=target_time_test, name="pred")

    obs_s  = obs_s.loc[pd.Timestamp("2024-01-01"):pd.Timestamp("2025-12-01")]     # 타겟 기간 자르기 (월별이면 MS 기준으로)
    pred_s = pred_s.loc[pd.Timestamp("2024-01-01"):pd.Timestamp("2025-12-01")]

    df_eval = pd.concat([obs_s, pred_s], axis=1)
    y_test  = df_eval["obs"].to_numpy()
    y_test_pred = df_eval["pred"].to_numpy()

    bias = (y_test_pred - y_test).mean()
    mse  = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_test_pred)
    corr, p_val = pearsonr(y_test, y_test_pred)
    sigma_obs = y_test.std(ddof=0)
    sigma_pred = y_test_pred.std(ddof=0)
    std_ratio = y_test_pred.std(ddof=0) / y_test.std(ddof=0)
    print("train_time_sec", train_time,
    "train_time_min", train_time / 60,
    "infer_time_sec", infer_time)
    results.append({
        "lead"       : lead,
        "bias"       : bias, 
        "mse"        : mse,
        "rmse"       : rmse,
        "mae"        : mae,
        "corr"       : corr,
        "p"          : p_val,
        "sigma_obs":sigma_obs, "sigma_pred": sigma_pred, "std_ratio": std_ratio,
        "best_params": random_search.best_params_
    })

    # --- (5) 중요도 ---
    f_imp = pd.Series(
        best_model.feature_importances_,
        index = feature_cols,
        name  = f'lead{lead}'
    ).sort_values(ascending=False)
    importance_dict[lead] = f_imp

    # --- (6) 예측·모델 저장 ---
    bundle = {
        "model"   : best_model,
        "scaler"  : scaler,
        "features": feature_cols,
    }
    save_file = OUT_DIR / f"model_lead{lead}.pt"
    joblib.dump(bundle, save_file, compress=3)
    print(f"모델 저장 → {save_file}")

    # --- (7) 관측·예측·잔차 테이블 --- > 이건 target time으로 저장
    df_res = df_eval.copy()
    df_res["residual"] = df_res["obs"] - df_res["pred"]
    df_res = df_res.reset_index().rename(columns={"index": "time"})
    df_result_dict[lead] = df_res

    # (8) SHAP 계산 (XGBoost fast: pred_contribs) > 이건 start time으로 저장
    dm = xgb.DMatrix(X_test_scaled, feature_names=feature_cols)
    contrib = best_model.get_booster().predict(dm, pred_contribs=True)
    # contrib: (n_samples, n_features + 1), last column is base value

    shap_vals = contrib[:, :-1]
    base_vals = contrib[:, -1]

    shap_df = pd.DataFrame(shap_vals, index=X_test_sync.index, columns=feature_cols)
    base_s  = pd.Series(base_vals, index=X_test_sync.index, name="base_value")
    shap_file = OUT_DIR / f"shap_lead{lead}.parquet"
    base_file = OUT_DIR / f"base_lead{lead}.parquet"
    shap_df.to_parquet(shap_file)
    base_s.to_frame().to_parquet(base_file)
    lead_elapsed = time.perf_counter() - lead_start
    print(f"Lead {lead} 소요 시간: {lead_elapsed/60:.2f}분 ({lead_elapsed:.1f}초)")
    time_results.append({
    "lead": lead,
    "train_time_sec": train_time,
    "train_time_min": train_time / 60,
    "infer_time_sec": infer_time,
    "total_lead_time_sec": lead_elapsed,
    "total_lead_time_min": lead_elapsed / 60,
    })

total_elapsed = time.perf_counter() - total_start
print(f"\n▶ 모든 작업 완료!")
print(f"전체 소요 시간: {total_elapsed/60:.2f}분 ({total_elapsed/3600:.2f}시간)")



results_df = pd.DataFrame(results).set_index("lead")  
imp_df = pd.concat(importance_dict, axis=1)

result_long = (pd.concat(df_result_dict, names=['lead'])  # > 이건 target time으로 저장
                 .reset_index())
result_long.drop('level_1', axis=1).to_csv(OUT_DIR / 'results_test.csv', index=False)
    
pd.DataFrame(results_df)\
  .to_csv(OUT_DIR / 'eval_metrics_test.csv') # > 이건 target time으로 저장

print("\n▶ 모든 작업 완료!")

# %%
time_df = pd.DataFrame(time_results).set_index("lead")
time_df

# %%
result_long.drop('level_1', axis=1)

# %%
result_long.drop('level_1', axis=1).head(30)

# %%
results_df

# %%



# %%
print('\n📊 리드 타임별 성능 요약:')
pd.set_option('display.max_colwidth', None)  
results_df

# %%
print('\n📊 리드 타임별 성능 요약:')
pd.set_option('display.max_colwidth', None)  
results_df[['rmse', 'mae', 'corr', 'best_params']] # PDO 제거

# %%
n_leads = 12
fig, axes = plt.subplots(n_leads, 1, figsize=(5, 2*n_leads), sharex=True)

for lead in range(1, n_leads + 1):
    ax = axes[lead - 1]
    df = df_result_dict.get(lead)
    if df is not None:
        ax.plot(df["time"], df["obs"], label="obs")
        ax.plot(df["time"], df["pred"], label="pred")
        ax.set_title(f"Lead {lead}")
        ax.legend()

plt.tight_layout()
plt.show()

# %%
df_result_dict[1]

# %%
fig, axes = plt.subplots(3, 4, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

cmap = cm.get_cmap('tab10', 12)

for lead in range(1, 13):
    ax = axes[lead - 1]
    df_res = df_result_dict[lead].dropna()

    obs  = df_res['obs'].values
    pred = df_res[f'pred'].values

    # ── 산점도 ────────────────────────────────────────────────
    ax.scatter(obs, pred,
               s=20, color=cmap(lead - 1), alpha=0.6,
               label=f'Lead {lead}')

    # 1:1 기준선
    lims = [min(obs.min(), pred.min()),
            max(obs.max(), pred.max())]
    ax.plot(lims, lims, '--', color='gray', linewidth=1)
    ax.set_xlim(lims);  ax.set_ylim(lims)

    # ── 서식 ────────────────────────────────────────────────
    ax.set_title(f'Lead {lead} month', fontsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    if lead in [1, 5, 9]:
        ax.set_ylabel('Predicted D20 anomaly (m)')
    if lead > 8:
        ax.set_xlabel('Observed D20 anomaly (m)')

    ax.tick_params(axis='both', labelsize=10)

# plt.suptitle('XGBoost (Lead 1–12)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
lead_times = [result['lead'] for result in results]
mse_list = [result['mse'] for result in results]
rmse_list = [result['rmse'] for result in results]
mae_list = [result['mae'] for result in results]
corr_list = [result['corr'] for result in results]
p_list     = [r['p']     for r in results]     # ★
sig_mask   = np.array(p_list) < 0.01           # ★ p<0.01

# 시각화 시작
fig, ax1 = plt.subplots(figsize=(10, 6))

# RMSE와 MAE를 첫 번째 y축에 플로팅
ax1.set_xlabel('Lead Time (Months)', fontsize=14)
ax1.set_ylabel('RMSE & MAE', fontsize=14)
ax1.plot(lead_times, rmse_list, color='tab:blue', label='RMSE', marker='o', linestyle='-', markersize=6)
ax1.plot(lead_times, mae_list, color='tab:orange', label='MAE', marker='o', linestyle='--', markersize=6)
ax1.tick_params(axis='y', labelsize=12)
ax1.set_xticks(lead_times)
ax1.set_xticklabels([str(i) for i in lead_times], fontsize=12)
ax1.set_ylim(0, 25)  # y축 범위 설정
ax1.legend(loc='upper left', fontsize=12)

# 두 번째 y축을 설정하여 Correlation 값을 플로팅
ax2 = ax1.twinx()
ax2.set_ylabel('Correlation', fontsize=14)
ax2.plot(lead_times, corr_list, color='tab:green', label='Correlation', marker='s', linestyle='-', markersize=6)
ax2.scatter(np.array(lead_times)[sig_mask],
            np.array(corr_list)[sig_mask],
            marker='s', s=40, color='red',
            label='p < 0.01', zorder=5)
ax2.tick_params(axis='y', labelsize=12)
ax2.legend(loc='upper right', fontsize=12)
ax2.set_ylim(0, 1)  # Correlation y축 범위 설정
ax2.set_xlim(1, 12)  # Lead time 1~12까지 보이도록 설정
ax2.axhline(y=0.5, color='gray', linestyle='--', label='Correlation = 0.5')  # Correlation = 0.5 회색 실선 추가

# 제목 및 레이아웃 설정
plt.title('XGboost model', fontsize=16)
plt.tight_layout()
plt.show()

# %%
# MLR
from sklearn.linear_model import LinearRegression

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train.values),
    index=X_train.index, columns=X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test.values),
    index=X_test.index, columns=X_test.columns
)

results_mlr, df_result_dict_mlr = [], {}
coef_dict = {}   # 리드타임별 회귀계수 저장

for lead in range(1, 13):
    key_train = f"y_lead{lead}_ano"
    # --- train: y에서 NaN 제거 후 X랑 동기화 ---
    ytr_s = y_train_all[key_train].dropna()
    common_idx_train = X_train_scaled.index.intersection(ytr_s.index)
    Xtr = X_train_scaled.loc[common_idx_train]
    ytr = ytr_s.loc[common_idx_train].values

    # --- test: y에서 NaN 제거 후 X랑 동기화 ---
    yte_s = y_test_all[key_train].dropna()
    common_idx_test = X_test_scaled.index.intersection(yte_s.index)
    Xte = X_test_scaled.loc[common_idx_test]
    yte = yte_s.loc[common_idx_test].values


    # 2. 모델 구축        
    model = LinearRegression()
    model.fit(Xtr, ytr)

    # 2. 평가 지표
    pred = model.predict(Xte)
    target_time_test = common_idx_test + pd.DateOffset(months=lead)

    obs_s  = pd.Series(yte,  index=target_time_test, name="obs")
    pred_s = pd.Series(pred, index=target_time_test, name="pred")

    obs_s  = obs_s.loc[pd.Timestamp("2023-01-01"):pd.Timestamp("2024-12-01")]
    pred_s = pred_s.loc[pd.Timestamp("2023-01-01"):pd.Timestamp("2024-12-01")]

    df_eval = pd.concat([obs_s, pred_s], axis=1)
    y_obs  = df_eval["obs"].to_numpy()
    y_pred = df_eval["pred"].to_numpy()


    mse  = mean_squared_error(yte, pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(yte, pred)
    corr, p_val = pearsonr(yte, pred) if len(yte) > 2 else (np.nan, np.nan)

    results_mlr.append({
        "lead": lead, "mse": mse, "rmse": rmse,
        "mae": mae, "corr": corr, "p": p_val, "n": len(yte)
    })

    df_res = df_eval.copy()
    df_res["residual"] = df_res["obs"] - df_res["pred"]
    df_res.index.name = "time"
    df_result_dict_mlr[lead] = df_res

    coef_dict[lead] = pd.Series(model.coef_, index=Xtr.columns).sort_values(key=np.abs, ascending=False)

df_summary_mlr = pd.DataFrame(results_mlr).set_index("lead").sort_index()

# %%
# -------------------------
# 1) XGB Metric Lists
# -------------------------
lead_times = [r['lead'] for r in results]
rmse_xgb   = [r['rmse'] for r in results]
mae_xgb    = [r['mae']  for r in results]
corr_xgb   = [r['corr'] for r in results]
p_xgb      = [r['p']    for r in results]
sig_xgb    = np.array(p_xgb) < 0.01

# -------------------------
# 2) MLR Metric Lists
# -------------------------
rmse_mlr = df_summary_mlr['rmse'].values
mae_mlr  = df_summary_mlr['mae'].values
corr_mlr = df_summary_mlr['corr'].values
p_mlr    = df_summary_mlr['p'].values
sig_mlr  = p_mlr < 0.01

# -------------------------
# 3) Plot
# -------------------------
fig, ax1 = plt.subplots(figsize=(11, 7))

# ========== RMSE & MAE (y1) ===============
ax1.set_xlabel("Lead Time (Months)", fontsize=14)
ax1.set_ylabel("RMSE & MAE", fontsize=14)

# XGB
ax1.plot(lead_times, rmse_xgb, color='tab:blue', marker='o', label='RMSE (XGB)')

# MLR
ax1.plot(lead_times, rmse_mlr, color='tab:purple', marker='s', label='RMSE (MLR)')

ax1.tick_params(axis='y', labelsize=12)
ax1.set_xticks(lead_times)
ax1.set_xticklabels([str(i) for i in lead_times], fontsize=12)
ax1.set_ylim(0, max(max(rmse_xgb), max(rmse_mlr)) * 1.2)

ax1.legend(loc="upper left", fontsize=12)

# ========== Correlation (y2) ===============
ax2 = ax1.twinx()
ax2.set_ylabel("Correlation", fontsize=14)

# XGB correlation
ax2.plot(lead_times, corr_xgb, color='tab:green', marker='D', label='Corr (XGB)')
ax2.scatter(np.array(lead_times)[sig_xgb],
            np.array(corr_xgb)[sig_xgb],
            color='red', s=40, marker='D', label='p < 0.01 (XGB)')

# MLR correlation
ax2.plot(lead_times, corr_mlr, color='tab:cyan', marker='^', label='Corr (MLR)')
ax2.scatter(np.array(lead_times)[sig_mlr],
            np.array(corr_mlr)[sig_mlr],
            color='magenta', s=40, marker='^', label='p < 0.01 (MLR)')

ax2.set_ylim(0, 1)
ax2.tick_params(axis='y', labelsize=12)
ax2.legend(loc="upper right", fontsize=12)

ax2.axhline(y=0.5, color='gray', linestyle='--')

plt.title("XGBoost vs MLR Performance by Lead Time", fontsize=16)
plt.tight_layout()
plt.show()


# %%
n_leads = 12      # Lead 1~12

fig, axes = plt.subplots(3, 4, figsize=(25, 10), sharex=False)
axes = axes.flatten()

for idx, lead in enumerate(range(1, n_leads + 1)):
    ax  = axes[idx]
    imp = importance_dict[lead].sort_values(ascending=False)
    
    ax.barh(imp.index[::-1], imp.values[::-1])  # 가장 큰 값을 위로
    ax.set_title(f'Lead {lead}', fontsize=12)
    if idx % 4 == 0:
        ax.set_ylabel('Feature', fontsize=11)
    ax.set_xlabel('Importance (gain)', fontsize=11)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_xlim(0,1)
    
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle(f'Feature Importances', fontsize=16, y=0.92)
plt.tight_layout(rect=[0, 0, 1, 0.9])

# %%



# %%


# %%




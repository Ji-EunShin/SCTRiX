# %%
import os
import numpy as np 
import xarray as xr
import matplotlib.pyplot as plt
import datetime
import pandas as pd 
from scipy.stats import pearsonr
import matplotlib.cm as cm

import matplotlib
import seaborn as sns
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
import matplotlib.dates as mdates
import matplotlib.ticker as mticker    

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer

import re, glob
import xgboost as xgb, ast, re
import scipy
from eofs.xarray import Eof
import joblib
from pathlib import Path

from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
from pathlib import Path

# %%
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "feature"
OUT_DIR = ROOT / "output" / "lofo_result"
OUT_DIR.mkdir(exist_ok=True, parents=True)
RESULT_DIR = ROOT / "output" / "model_result"
# %%
def corr_distance(y_true, y_pred):
    r, _ = pearsonr(y_true, y_pred)
    return 1.0 - r           # 0이 최우수, 2가 최악

corr_dist_score = make_scorer(corr_distance, greater_is_better=False)

# %%
for SEED in tqdm(range(1, 300 + 1), desc="Seeds"):

    # 1-A) best_params 불러오기 ---------------------------------
    eval_csv = RESULT_DIR / f'eval_metrics_seed{SEED}.csv'
    df_base = (pd.read_csv(eval_csv, converters={'best_params': ast.literal_eval})
                 .set_index('lead'))

    # 1-B) train/test CSV 로드 ----------------------------------
    X_train = pd.read_csv(DATA_DIR / f'X_train_{SEED}.csv', index_col=0, parse_dates=True)
    X_test  = pd.read_csv(DATA_DIR / f'X_test_{SEED}.csv',  index_col=0, parse_dates=True)
    y_tr_all= pd.read_csv(DATA_DIR / f'y_train_{SEED}.csv', index_col=0, parse_dates=True)
    y_te_all= pd.read_csv(DATA_DIR / f'y_test_{SEED}.csv',  index_col=0, parse_dates=True)

    all_features = X_train.columns.tolist()

    sctr, io, po = [], [], []
    for c in all_features:
        cu = c.upper()
        if c.lower() == "d20_ano_sctr" or cu.endswith("SCTR"):
            sctr.append(c)
        elif cu == "DMI" or cu.endswith("IO"):
            io.append(c)
        elif cu == "NINO34" or cu.endswith("PO"):
            po.append(c)

    feature_groups = {}
    if sctr:
        feature_groups["SCTR"] = sctr
    if io:
        feature_groups["IO"] = io
    if po:
        feature_groups["PO"] = po
    if io and po:
        feature_groups["REMOTE"] = io + po

    pred_dict = {gname: [] for gname in feature_groups.keys()}
    metrics_rows = []

    # ── 2) lead & Grouped LOFO 루프 ───────────────────────────
    for lead in range(1, 12 + 1):
        y_col  = f"y_lead{lead}_ano"
        y_tr   = y_tr_all[y_col]
        y_te   = y_te_all[y_col]
        best_p = df_base.loc[lead, "best_params"]   # dict

        X_test_lead = X_test.copy()
        target_time = y_te.index + pd.DateOffset(months=lead)

        # --- (1-1) 1958-02 ~ 1958-12 과 2023-01 ~ 2023-11 제거 ---
        mask = ~(
            ((target_time >= pd.Timestamp("1958-02-01")) &
             (target_time <= pd.Timestamp("1958-12-01")))
            |
            ((target_time >= pd.Timestamp("2023-01-01")) &
             (target_time <= pd.Timestamp("2023-11-01")))
        )

        y_te = y_te.loc[mask]
        X_test_lead = X_test_lead.loc[mask]

        # (A) 베이스라인: 모든 피처 사용 ------------------------
        scaler_base = StandardScaler()
        X_tr_base   = scaler_base.fit_transform(X_train[all_features])
        X_te_base   = scaler_base.transform (X_test_lead [all_features])

        model_base = xgb.XGBRegressor(
            **best_p,
            objective="reg:squarederror",
            random_state=2024
        )
        model_base.fit(X_tr_base, y_tr)
        y_pred_base = pd.Series(model_base.predict(X_te_base), index=y_te.index, name="pred_base")

        # 베이스라인 지표 계산 (NaN 세이프)
        yt_b = np.asarray(y_te.values)
        yp_b = np.asarray(y_pred_base.values)

        base_bias = (yp_b - yt_b).mean()
        base_mse  = mean_squared_error(yt_b, yp_b)
        base_rmse = np.sqrt(base_mse)
        base_mae  = mean_absolute_error(yt_b, yp_b)
        base_corr, base_p = pearsonr(yt_b, yp_b)
        base_sigma_obs = yt_b.std(ddof=0)
        base_sigma_pred = yp_b.std(ddof=0)
        base_std_ratio = yp_b.std(ddof=0) / yt_b.std(ddof=0)

        # (B) 그룹 제거 실험 ------------------------------------
        for gname, feats_to_drop in feature_groups.items():
            drop_set = set(feats_to_drop)
            keep_cols = [c for c in all_features if c not in drop_set]

            scaler  = StandardScaler()
            X_tr_s  = scaler.fit_transform(X_train[keep_cols])
            X_te_s  = scaler.transform (X_test_lead [keep_cols])

            model = xgb.XGBRegressor(
                **best_p,
                objective="reg:squarederror",
                random_state=2024
            )
            model.fit(X_tr_s, y_tr)
            y_pred = pd.Series(model.predict(X_te_s), index=y_te.index, name="pred")

            # 예측 결과 저장(시계열 단위)
            pred_df = pd.DataFrame({
                "seed"            : SEED,
                "lead"            : lead,
                "removed_group"   : gname,
                "removed_features": ",".join(feats_to_drop),
                "time"            : y_te.index,
                "obs"             : y_te.values,
                "pred"            : y_pred.values,
            })
            pred_dict[gname].append(pred_df)

            # 지표 계산/누적 -------------------------------------
            yt = np.asarray(y_te.values)
            yp = np.asarray(y_pred.values)

            bias = (yp - yt).mean()
            mse  = mean_squared_error(yt, yp)
            rmse = np.sqrt(mse)
            mae  = mean_absolute_error(yt, yp)
            corr, p = pearsonr(yt, yp)
            sigma_obs = yt.std(ddof=0)
            sigma_pred = yp.std(ddof=0)
            std_ratio = yp.std(ddof=0) / yt.std(ddof=0)

            metrics_rows.append({
                "seed"            : SEED,
                "lead"            : lead,
                "removed_group"   : gname,
                "removed_features": ",".join(feats_to_drop),
                # grouped LOFO 지표
                "bias" : bias,
                "mse"  : mse,
                "rmse" : rmse,
                "mae"  : mae,
                "corr" : corr,
                "p_value": p,
                "sigma_obs"  : sigma_obs, 
                "sigma_pred" : sigma_pred, 
                "std_ratio"  : std_ratio,
                # baseline 지표
                "base_bias" : base_bias,                
                "base_rmse" : base_rmse,
                "base_mae"  : base_mae,
                "base_corr" : base_corr,
                "base_p_value": base_p,
                "base_sigma_obs"  : base_sigma_obs, 
                "base_sigma_pred" : base_sigma_pred, 
                "base_std_ratio"  : base_std_ratio,
                # 참고용
                "best_params": best_p,
            })

        print(f"[Seed {SEED}_lead{lead}] ✔")

    # ── 3) 그룹별 CSV 저장 ────────────────────────────────────
    for gname in feature_groups.keys():
        if len(pred_dict[gname]) == 0:
            continue
        df_group = pd.concat(pred_dict[gname], ignore_index=True)
        df_group.to_csv(OUT_DIR / f"results_grouped_lofo_{gname}_seed{SEED:03d}.csv", index=False)

    # ── 4) 지표 CSV 저장(이 SEED 전체) ─────────────────────────
    if len(metrics_rows) > 0:
        df_metrics = pd.DataFrame(metrics_rows)
        df_metrics.to_csv(OUT_DIR / f"eval_metrics_grouped_lofo_seed{SEED:03d}.csv", index=False)
        print(f"[Seed {SEED}] metrics ✔")


# %%
feature_groups.items() 



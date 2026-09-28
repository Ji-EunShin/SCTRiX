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
import re, joblib, json, gc
from pathlib import Path

from statsmodels.tsa.statespace.sarimax import SARIMAX

# %%
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "feature"
RESULT_DIR = ROOT / "output" / "model_result"
MODEL_DIR  = ROOT / "output" / "model_loop"
SHAP_DIR   = ROOT / "output" / "model_shap_loop"
# %% [markdown]
# ## data

# %%
seed_files = DATA_DIR.glob('X_train_*.csv')
# SEEDS = sorted({int(re.search(r'_(\d+)\.csv', f.name).group(1)) for f in seed_files})
N_LEAD = 12          

# %%
# len(SEEDS), N_LEAD
 
# %%
def corr_distance(y_true, y_pred):
    r, _ = pearsonr(y_true, y_pred)
    return 1.0 - r           # 0이 최우수, 2가 최악

corr_dist_score = make_scorer(corr_distance, greater_is_better=False)

# %%
param_grid = {
    'n_estimators'     : [100, 150, 200, 250, 300],
    'learning_rate'    : [0.03, 0.05, 0.07, 0.1, 0.2],
    'max_depth'        : [2, 4, 6, 8, 10],
    'subsample'        : [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0],
}

# %%
for SEED in range(1, 350+1):
    print(f'\n================  SEED {SEED:03d}  ================')

    X_train = pd.read_csv(DATA_DIR / f'X_train_{SEED}.csv', index_col=0, parse_dates=True)
    X_test  = pd.read_csv(DATA_DIR / f'X_test_{SEED}.csv',  index_col=0, parse_dates=True)
    y_tr_all= pd.read_csv(DATA_DIR / f'y_train_{SEED}.csv', index_col=0, parse_dates=True)
    y_te_all= pd.read_csv(DATA_DIR / f'y_test_{SEED}.csv',  index_col=0, parse_dates=True)
    
    feature_cols = X_train.columns.tolist()      

    eval_metrics   = []      # ← evaluation metric 
    importance_dic = {}      # ← importance_dict
    df_res_dic     = {}      # ← df_result_dict (obs, pred, residual)

    for lead in range(1, 12 + 1):
        y_col = f'y_lead{lead}_ano'
        y_tr  = y_tr_all[y_col]
        y_te  = y_te_all[y_col]
        X_test_lead = X_test.copy()
        target_time = y_te.index + pd.DateOffset(months=lead)

        # --- (1-1) 2023-01 ~ 2023-11 제거 ---
        mask = ~(
            (
                (target_time >= pd.Timestamp("1958-02-01")) &
                (target_time <= pd.Timestamp("1958-12-01"))
            )
            |
            (
                (target_time >= pd.Timestamp("2023-01-01")) &
                (target_time <= pd.Timestamp("2023-11-01"))
            ))

        y_te = y_te.loc[mask]
        X_test_lead = X_test_lead.loc[mask]

        # ① 스케일러
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test_lead)
        X_te_s_df = pd.DataFrame(X_te_s, index=X_test_lead.index, columns=feature_cols)

        # ② 모델 탐색
        xgb_reg = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=2024,
            n_jobs=1
        )
        rnd = RandomizedSearchCV(
            estimator=xgb_reg, param_distributions= param_grid, n_iter=100,
            scoring=corr_dist_score, cv=5,
            random_state=2024, n_jobs=10, verbose=0
        )
        rnd.fit(X_tr_s, y_tr)

        best = rnd.best_estimator_

        # ③ 예측 & 성능
        y_pred = best.predict(X_te_s_df)
        bias = (y_pred - y_te).mean()
        mse  = mean_squared_error(y_te, y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_te, y_pred)
        corr, p = pearsonr(y_te, y_pred)
        sigma_obs = y_te.std(ddof=0)
        sigma_pred = y_pred.std(ddof=0)
        std_ratio = y_pred.std(ddof=0) / y_te.std(ddof=0)

        eval_metrics.append({
            'lead' : lead, 'bias': bias, 'mse': mse, 'rmse': rmse,
            'mae'  : mae , 'corr': corr, 'p': p,
            "sigma_obs":sigma_obs, "sigma_pred": sigma_pred, "std_ratio": std_ratio,
            'best_params': rnd.best_params_
        })


        # ④ 중요도
        imp = pd.Series(best.feature_importances_,
                        index=feature_cols,
                        name=f'lead{lead}')\
              .sort_values(ascending=False)
        importance_dic[lead] = imp

        # ⑤ 모델 번들 저장
        bundle = {'model': best, 'scaler': scaler, 'features': feature_cols}
        bundle_path = MODEL_DIR / f'model_lead{lead}_seed{SEED}.pt'
        joblib.dump(bundle, bundle_path, compress=3)
        print(f'  ✓ model_lead{lead}_seed{SEED}.pt 저장')

        # ⑥ 예측 결과 저장용 dict
        df_res_dic[lead] = pd.DataFrame({
            'time': y_te.index, 'obs': y_te.values,
            'pred': y_pred, 'residual': y_te.values - y_pred
        })

        # (8) SHAP 계산 (XGBoost fast: pred_contribs)
        dm = xgb.DMatrix(X_te_s_df, feature_names=feature_cols)
        contrib = best.get_booster().predict(dm, pred_contribs=True)
    
        shap_vals = contrib[:, :-1]
        base_vals = contrib[:, -1]

        shap_df = pd.DataFrame(shap_vals, index=X_te_s_df.index, columns=feature_cols)
        base_s  = pd.Series(base_vals, index=X_te_s_df.index, name="base_value")
        shap_file = SHAP_DIR / f"shap_lead{lead}_seed{SEED}.parquet"
        base_file = SHAP_DIR / f"base_lead{lead}_seed{SEED}.parquet"
        shap_df.to_parquet(shap_file)
        base_s.to_frame().to_parquet(base_file)

        # 메모리 케어
        del scaler, rnd, best, y_pred; gc.collect()

    # save CSV file by seed
    # (i) df_result_dict → results_seed<SEED>.csv  (multi-index lead, time)
    result_long = (pd.concat(df_res_dic, names=['lead'])
                     .reset_index())
    result_long.to_csv(RESULT_DIR / f'results_seed{SEED}.csv', index=False)

    # (ii) importance_dict → importance_seed<SEED>.csv
    pd.concat(importance_dic, axis=1)\
      .to_csv(RESULT_DIR / f'importance_seed{SEED}.csv')

    # (iii) eval_metrics → eval_metrics_seed<SEED>.csv
    pd.DataFrame(eval_metrics)\
      .set_index('lead')\
      .to_csv(RESULT_DIR / f'eval_metrics_seed{SEED}.csv')

    print(f'▶ seed {SEED} finish\n')

print('✅ all finish!')



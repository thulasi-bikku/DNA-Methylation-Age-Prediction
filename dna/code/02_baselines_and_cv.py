"""Step 2. Elastic Net, XGBoost, LightGBM and MLP: test-set evaluation, 5-fold CV and feature ablation.

The MLP penalty is fixed at alpha = 0.1, the value reported in the manuscript.
(That value was originally screened on the test set; see 05_extended_protocol.py
for a development-only selection.)
"""
import argparse
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from common import SEED, feature_columns


def make_models(inner_cv):
    return {
        "Elastic Net": lambda: ElasticNetCV(cv=inner_cv, random_state=SEED, n_jobs=-1, max_iter=5000),
        "XGBoost": lambda: xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=SEED, n_jobs=-1),
        "LightGBM": lambda: lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=SEED, n_jobs=-1, verbose=-1),
        "MLP": lambda: MLPRegressor(hidden_layer_sizes=(128, 64), alpha=0.1, max_iter=2000, random_state=SEED,
                                    early_stopping=True, validation_fraction=0.15),
    }


SCALED = {"Elastic Net", "MLP"}


def fit_predict(name, factory, Xtr, ytr, Xte):
    if name in SCALED:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = factory()
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = pd.read_csv(os.path.join(a.data, "trainval_data_leakfree.csv"))
    test = pd.read_csv(os.path.join(a.data, "test_data_leakfree.csv"))
    feats = feature_columns(dev)
    Xd, yd = dev[feats].to_numpy(), dev["Age"].to_numpy()
    Xt, yt = test[feats].to_numpy(), test["Age"].to_numpy()

    # held-out test evaluation (5-fold internal CV for the Elastic Net path)
    preds = pd.DataFrame({"Sample_ID": test["Sample_ID"], "Age": yt})
    rows = []
    for name, f in make_models(5).items():
        p = fit_predict(name, f, Xd, yd, Xt)
        preds[name] = p
        rows.append({"Model": name, "MAE": mean_absolute_error(yt, p), "RMSE": mean_squared_error(yt, p) ** 0.5, "R2": r2_score(yt, p)})
        print(rows[-1])
    preds.to_csv(os.path.join(a.out, "test_predictions_baselines.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "baseline_test_results.csv"), index=False)

    # 5-fold CV on the development pool (Elastic Net uses 3-fold internal CV, as in the manuscript)
    cv_rows = []
    for k, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(Xd), 1):
        for name, f in make_models(3).items():
            p = fit_predict(name, f, Xd[tr], yd[tr], Xd[va])
            cv_rows.append({"Fold": k, "Model": name, "MAE": mean_absolute_error(yd[va], p)})
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(os.path.join(a.out, "cv_fold_results.csv"), index=False)
    print(cv.groupby("Model")["MAE"].agg(["mean", lambda s: s.std(ddof=0)]))

    # feature-count ablation on the development-pool ranking
    rank = pd.read_csv(os.path.join(a.data, "cpg_correlation_TRAINONLY.csv"))
    ab = []
    for n in (500, 1000, 3000):
        cols = [c for c in rank.head(n)["CpG_ID"] if c in dev.columns]
        models = make_models(5)
        row = {"CpGs": n, "Features_Used": len(cols)}
        for name in ("Elastic Net", "XGBoost"):
            p = fit_predict(name, models[name], dev[cols].to_numpy(), yd, test[cols].to_numpy())
            row[name] = mean_absolute_error(yt, p)
        ab.append(row)
    pd.DataFrame(ab).to_csv(os.path.join(a.out, "feature_ablation.csv"), index=False)
    print(pd.DataFrame(ab))


if __name__ == "__main__":
    main()

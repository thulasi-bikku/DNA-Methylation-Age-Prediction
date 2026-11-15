"""Step 5 (optional). Extended protocol that removes the limitations listed in the manuscript.

(a) Development-only selection of the MLP penalty (inner 3-fold CV on the 524 development samples),
    followed by a single evaluation on the untouched test set.
(b) Nested cross-validation on the development pool: CpG ranking, hyperparameter selection and model
    fitting are all repeated inside every outer training fold, so the outer folds never influence
    feature selection.
Results are written to CSV files; they are not part of the submitted manuscript.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from common import SEED, TOP_K

try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

ALPHAS = [0.001, 0.01, 0.1, 1.0, 5.0]


def rank_top_k(X, y, k):
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc * yc[:, None]).sum(axis=0) / (np.sqrt((Xc ** 2).sum(axis=0)) * np.sqrt((yc ** 2).sum()))
    r = np.nan_to_num(np.abs(r))
    return np.argsort(-r)[:k]


def tuned_mlp(X, y):
    sc = StandardScaler().fit(X)
    gs = GridSearchCV(
        MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=2000, random_state=SEED, early_stopping=True, validation_fraction=0.15),
        {"alpha": ALPHAS}, cv=KFold(3, shuffle=True, random_state=SEED), scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(sc.transform(X), y)
    return sc, gs.best_estimator_, gs.best_params_["alpha"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--outer", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = pd.read_csv(os.path.join(a.data, "trainval_data_leakfree.csv"))
    test = pd.read_csv(os.path.join(a.data, "test_data_leakfree.csv"))
    feats = [c for c in dev.columns if c.startswith("cg")]
    Xd, yd = dev[feats].to_numpy(float), dev["Age"].to_numpy(float)
    Xt, yt = test[feats].to_numpy(float), test["Age"].to_numpy(float)

    # (a) development-only MLP tuning, one test evaluation
    sc, mlp, alpha = tuned_mlp(Xd, yd)
    mae_test = mean_absolute_error(yt, mlp.predict(sc.transform(Xt)))
    pd.DataFrame([{"selected_alpha": alpha, "test_MAE": mae_test}]).to_csv(os.path.join(a.out, "mlp_dev_only_tuning.csv"), index=False)
    print(f"MLP alpha selected on development data: {alpha}; test MAE {mae_test:.3f}")

    # (b) nested CV: ranking inside each outer training fold
    rows = []
    for k, (tr, va) in enumerate(KFold(a.outer, shuffle=True, random_state=SEED).split(Xd), 1):
        idx = rank_top_k(Xd[tr], yd[tr], min(a.k, Xd.shape[1]))
        Xtr, Xva = Xd[tr][:, idx], Xd[va][:, idx]
        sc = StandardScaler().fit(Xtr)
        en = ElasticNetCV(cv=3, random_state=SEED, n_jobs=-1, max_iter=5000).fit(sc.transform(Xtr), yd[tr])
        rows.append({"Fold": k, "Model": "Elastic Net", "MAE": mean_absolute_error(yd[va], en.predict(sc.transform(Xva)))})
        if xgb is not None:
            m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=SEED, n_jobs=-1).fit(Xtr, yd[tr])
            rows.append({"Fold": k, "Model": "XGBoost", "MAE": mean_absolute_error(yd[va], m.predict(Xva))})
        if lgb is not None:
            m = lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=SEED, n_jobs=-1, verbose=-1).fit(Xtr, yd[tr])
            rows.append({"Fold": k, "Model": "LightGBM", "MAE": mean_absolute_error(yd[va], m.predict(Xva))})
        sc2, mlp2, al = tuned_mlp(Xtr, yd[tr])
        rows.append({"Fold": k, "Model": "MLP (inner-CV alpha)", "MAE": mean_absolute_error(yd[va], mlp2.predict(sc2.transform(Xva)))})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "nested_cv_fold_results.csv"), index=False)
    print(res.groupby("Model")["MAE"].agg(["mean", lambda s: s.std(ddof=0)]))


if __name__ == "__main__":
    main()

"""Step 4. Paired tests, effect sizes, bootstrap interval and age-stratified MAE.

Reads a prediction table with columns Sample_ID, Age and one column per model
(FT-Transformer, Elastic Net, XGBoost, LightGBM, MLP).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from common import AGE_BINS, AGE_LABELS, SEED, mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.predictions)
    y = df["Age"].to_numpy(float)
    models = [c for c in df.columns if c not in ("Sample_ID", "Age")]
    err = {m: np.abs(df[m].to_numpy(float) - y) for m in models}
    n = len(df)

    rows = []
    baselines = [m for m in ("Elastic Net", "XGBoost", "LightGBM") if m in err]
    for b in baselines:
        d = err["FT-Transformer"] - err[b]
        t, p_t = stats.ttest_rel(err["FT-Transformer"], err[b])
        w, p_w = stats.wilcoxon(err["FT-Transformer"], err[b])
        se = d.std(ddof=1) / np.sqrt(n)
        h = stats.t.ppf(0.975, n - 1) * se
        rows.append({"Comparison": f"FT vs {b}", "Delta": d.mean(), "CI_low": d.mean() - h, "CI_high": d.mean() + h,
                     "t": t, "dz": t / np.sqrt(n), "p_t_bonferroni": min(1.0, p_t * len(baselines)),
                     "W": w, "p_wilcoxon_bonferroni": min(1.0, p_w * len(baselines))})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "paired_tests.csv"), index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    rng = np.random.default_rng(SEED)
    boot = np.array([err["FT-Transformer"][rng.integers(0, n, n)].mean() for _ in range(a.n_boot)])
    ci = {"mean": boot.mean(), "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5), "point": err["FT-Transformer"].mean()}
    json.dump(ci, open(os.path.join(a.out, "bootstrap_ft.json"), "w"), indent=2)
    print("bootstrap", ci)

    grp = pd.cut(y, bins=AGE_BINS, labels=AGE_LABELS)
    tab = pd.DataFrame({m: pd.Series(err[m]).groupby(grp, observed=True).mean() for m in models if m != "MLP"})
    tab.insert(0, "n", pd.Series(grp).value_counts().reindex(AGE_LABELS).values)
    tab.to_csv(os.path.join(a.out, "age_group_mae.csv"))
    print(tab.round(3))


if __name__ == "__main__":
    main()

"""Step 1. Isolate the test set, then rank CpGs on the development pool only.

Inputs (downloaded from GEO, accession GSE40279):
  GSE40279_average_beta.txt      CpG x sample beta matrix (tab separated)
  GSE40279_series_matrix.txt     series matrix with sample titles and ages

Outputs (written to --out):
  trainval_ids.csv, test_ids.csv
  cpg_correlation_TRAINONLY.csv  ranking computed on the development pool only
  trainval_data_leakfree.csv, test_data_leakfree.csv
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import SEED, TEST_FRACTION, TOP_K


def read_ages(series_matrix):
    title_line = age_line = None
    with open(series_matrix) as fh:
        for line in fh:
            if line.startswith("!Sample_title"):
                title_line = line
            elif line.startswith("!Sample_characteristics_ch1") and "age (y)" in line:
                age_line = line
            if title_line and age_line:
                break
    names = ["X" + s for s in re.findall(r'"age \d+y (\d+)"', title_line)]
    ages = [int(a) for a in re.findall(r'"age \(y\): (\d+)"', age_line)]
    assert len(names) == len(ages), "sample/age mismatch"
    return pd.DataFrame({"Sample_ID": names, "Age": ages})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", required=True)
    ap.add_argument("--series", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--chunk", type=int, default=5000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    age_df = read_ages(a.series)
    print("samples:", len(age_df))

    # 1) split first; the test identifiers are never used for ranking
    dev_ids, test_ids = train_test_split(age_df["Sample_ID"].values, test_size=TEST_FRACTION, random_state=SEED)
    pd.Series(dev_ids, name="Sample_ID").to_csv(os.path.join(a.out, "trainval_ids.csv"), index=False)
    pd.Series(test_ids, name="Sample_ID").to_csv(os.path.join(a.out, "test_ids.csv"), index=False)
    print(f"development pool: {len(dev_ids)}, held-out test: {len(test_ids)}")
    age = dict(zip(age_df["Sample_ID"], age_df["Age"]))
    dev_cols = list(dev_ids)
    y_dev = np.array([age[c] for c in dev_cols], dtype=float)
    y_c = y_dev - y_dev.mean()

    # 2) Pearson correlation with age on the development pool only (chunked)
    rows = []
    for chunk in pd.read_csv(a.beta, sep="\t", chunksize=a.chunk, index_col="ID_REF"):
        X = chunk[dev_cols].to_numpy(dtype=float)
        ok = ~np.isnan(X).any(axis=1)
        X, ids = X[ok], chunk.index[ok]
        Xc = X - X.mean(axis=1, keepdims=True)
        denom = np.sqrt((Xc ** 2).sum(axis=1)) * np.sqrt((y_c ** 2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (Xc @ y_c) / denom
        rows.append(pd.DataFrame({"CpG_ID": ids, "Correlation": r}))
    corr = pd.concat(rows, ignore_index=True).dropna()
    corr["Abs_Correlation"] = corr["Correlation"].abs()
    corr = corr.sort_values("Abs_Correlation", ascending=False).reset_index(drop=True)
    corr.to_csv(os.path.join(a.out, "cpg_correlation_TRAINONLY.csv"), index=False)
    print("probes ranked:", len(corr))
    top = corr.head(TOP_K)["CpG_ID"].tolist()

    # 3) build the modelling tables (rows = samples)
    keep = []
    top_set = set(top)
    for chunk in pd.read_csv(a.beta, sep="\t", chunksize=a.chunk, index_col="ID_REF"):
        m = chunk[chunk.index.isin(top_set)]
        if len(m):
            keep.append(m)
    full = pd.concat(keep)
    full = full[~full.index.duplicated()].T
    full.index.name = "Sample_ID"
    full = full.reset_index().merge(age_df, on="Sample_ID")
    full.loc[full["Sample_ID"].isin(dev_ids)].to_csv(os.path.join(a.out, "trainval_data_leakfree.csv"), index=False)
    full.loc[full["Sample_ID"].isin(test_ids)].to_csv(os.path.join(a.out, "test_data_leakfree.csv"), index=False)
    print("modelling features:", full.shape[1] - 2)


if __name__ == "__main__":
    main()

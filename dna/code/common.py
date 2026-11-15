"""Shared constants and helpers for the analysis scripts."""
import numpy as np

SEED = 42
TEST_FRACTION = 0.20
TOP_K = 3000
AGE_BINS = [0, 40, 60, 200]                      # (0,40], (40,60], (60,200]
AGE_LABELS = ["<=40 years", "41-60 years", ">60 years"]


def feature_columns(df):
    return [c for c in df.columns if c.startswith("cg")]


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))

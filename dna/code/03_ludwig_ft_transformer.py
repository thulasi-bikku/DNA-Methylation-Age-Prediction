"""Step 3. FT-Transformer through a declarative Ludwig configuration.

The combiner is set explicitly so that the architecture does not depend on the
framework default of a particular Ludwig release.
"""
import argparse
import json
import os

import pandas as pd
import yaml
from ludwig.api import LudwigModel

from common import SEED, feature_columns


def build_config(cpgs):
    return {
        "input_features": [{"name": c, "type": "number"} for c in cpgs],
        "output_features": [{"name": "Age", "type": "number"}],
        "combiner": {"type": "ft_transformer"},
        "preprocessing": {"split": {"type": "random", "probabilities": [0.85, 0.15, 0.0]}},
        "trainer": {"epochs": 50, "batch_size": 32, "optimizer": {"type": "adam"}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = pd.read_csv(os.path.join(a.data, "trainval_data_leakfree.csv"))
    test = pd.read_csv(os.path.join(a.data, "test_data_leakfree.csv"))
    cfg = build_config(feature_columns(dev))
    cfg["random_seed"] = a.seed
    with open(os.path.join(a.out, f"ludwig_config_seed{a.seed}.yaml"), "w") as fh:
        yaml.dump(cfg, fh, sort_keys=False)

    model = LudwigModel(cfg)
    train_stats, _, out_dir = model.train(dataset=dev.drop(columns=["Sample_ID"]), output_directory=os.path.join(a.out, "ludwig"),
                                          skip_save_processed_input=True)
    pred = model.predict(test.drop(columns=["Sample_ID"]))[0]["Age_predictions"].values
    pd.DataFrame({"Sample_ID": test["Sample_ID"], "Age": test["Age"], "FT-Transformer": pred}).to_csv(
        os.path.join(a.out, f"test_predictions_ft_seed{a.seed}.csv"), index=False)
    n_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(json.dumps({"seed": a.seed, "trainable_parameters": n_params, "combiner": cfg["combiner"]["type"]}))


if __name__ == "__main__":
    main()

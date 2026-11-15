# Leakage-controlled DNA methylation age prediction with a declarative FT-Transformer

Code, configuration, figures and manuscript source for the paper
*Leakage-Controlled DNA Methylation Age Prediction with a Declarative FT-Transformer: Comparison with Penalized Linear and Tree-Based Models* (submitted to *Array*, manuscript ARRAY-D-26-07385).

The test set (132 of 656 whole-blood samples, GSE40279) is separated before CpGs are ranked by correlation with age; the ranking, model fitting and hyperparameters of the main models use the 524-sample development pool only.

## Layout

| Path | Content |
|---|---|
| `code/01_prepare_split_and_rank.py` | seed-42 split, development-only Pearson ranking, modelling tables |
| `code/02_baselines_and_cv.py` | Elastic Net, XGBoost, LightGBM, MLP: test results, 5-fold CV, feature ablation |
| `code/03_ludwig_ft_transformer.py` | FT-Transformer through a declarative Ludwig configuration (combiner set explicitly) |
| `code/04_statistics_and_tables.py` | paired t and Wilcoxon tests, Bonferroni, d_z, bootstrap interval, age strata |
| `code/05_extended_protocol.py` | optional: development-only MLP tuning and nested cross-validation |
| `code/verify_architecture.py` | prints combiner type and parameter count of a trained Ludwig model |
| `code/make_figures.py` | regenerates all manuscript figures (PNG and PDF) |
| `config/reproducibility_config.yaml` | consolidated settings |
| `notebook/biological_age_clock_colab.ipynb` | original Colab notebook |
| `manuscript/` | LaTeX source (`main.tex`), compiled `main.pdf`, `../highlights.txt` |
| `figures/` | figure files used in the manuscript |

## Reproduction

```bash
pip install -r requirements.txt
# put GSE40279_average_beta.txt and GSE40279_series_matrix.txt into data/ (see data/README.md)
cd code
python 01_prepare_split_and_rank.py --beta ../data/GSE40279_average_beta.txt --series ../data/GSE40279_series_matrix.txt --out ../data
python 02_baselines_and_cv.py --data ../data --out ../results
python 03_ludwig_ft_transformer.py --data ../data --out ../results      # GPU recommended
# merge results/test_predictions_ft_seed42.csv into the baseline prediction table (column "FT-Transformer"), then:
python 04_statistics_and_tables.py --predictions ../results/test_predictions_all.csv --out ../results
python make_figures.py ../figures ../figures/source
```

Compile the manuscript with `cd manuscript && latexmk -pdf main.tex`.

## Notes

* The MLP penalty reported in the paper (alpha = 0.1) was screened on the test set and is an optimistic reference. `05_extended_protocol.py` selects it on development data only and adds nested cross-validation.
* GPU training is not guaranteed to be bitwise reproducible; small differences in the FT-Transformer error between runs are expected.
* The bootstrap in `04_statistics_and_tables.py` uses a fixed seed, so interval limits may differ in the last digit from the values printed in the original notebook.

"""Regenerate all manuscript figures from the numerical results reported in the paper.

Every value below is copied from the analysis notebook outputs (Tables 2-6 of the
manuscript). The two residual plots are derived from per-sample predictions that
only exist inside the original Colab session, so they are cropped from the PNG
files exported by the notebook (see crop_residuals()).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
SRC = sys.argv[2] if len(sys.argv) > 2 else "figures/source"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 600,
})
C = {"EN": "#2E6EA8", "LGBM": "#D6A030", "XGB": "#5B9B6E", "FT": "#C8553D", "MLP": "#7D6EA0"}
NAME = {"EN": "Elastic Net", "LGBM": "LightGBM", "XGB": "XGBoost", "FT": "FT-Transformer", "MLP": "MLP"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
test = {  # model: (MAE, RMSE, R2)
    "EN": (3.621, 5.165, 0.896), "LGBM": (4.429, 5.715, 0.873),
    "XGB": (4.466, 5.846, 0.867), "FT": (8.264, 10.938, 0.535),
    "MLP": (12.748, 17.626, -0.208),
}
order = ["EN", "LGBM", "XGB", "FT", "MLP"]
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
for ax, (k, title, fmt) in zip(axes, [(0, "MAE (years)", "{:.3f}"), (1, "RMSE (years)", "{:.3f}"), (2, "$R^2$", "{:.3f}")]):
    vals = [test[m][k] for m in order]
    bars = ax.bar(range(5), vals, color=[C[m] for m in order], width=0.68)
    for b, v in zip(bars, vals):
        off = 0.02 * (max(vals) - min(0, min(vals)))
        ax.text(b.get_x() + b.get_width() / 2, v + off if v >= 0 else v - 7 * off, fmt.format(v),
                ha="center", va="bottom", fontsize=6.6)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["EN", "LGBM", "XGB", "FT", "MLP"], rotation=0)
    ax.set_title(title)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.grid(axis="x", visible=False)
axes[0].set_ylim(0, 15.5)
axes[1].set_ylim(0, 21)
axes[2].set_ylim(-0.4, 1.12)
fig.tight_layout()
save(fig, "fig3_test_performance")

# ---------------------------------------------------------------- Figure 4
cv = {
    "EN": ([3.13, 3.34, 3.65, 3.77, 3.83], 3.546, 0.269),
    "LGBM": ([4.13, 3.95, 4.15, 4.22, 4.14], 4.117, 0.092),
    "XGB": ([4.37, 3.96, 4.57, 4.53, 4.19], 4.324, 0.226),
    "MLP": ([13.22, 17.47, 17.08, 15.57, 15.54], 15.778, 1.497),
}
fig, ax = plt.subplots(figsize=(4.8, 3.2))
for i, m in enumerate(["EN", "LGBM", "XGB", "MLP"]):
    folds, mean, sd = cv[m]
    ax.bar(i, mean, color=C[m], width=0.6, alpha=0.85, yerr=sd, capsize=4, error_kw={"lw": 1.1})
    jitter = np.linspace(-0.16, 0.16, 5)
    ax.scatter(i + jitter, folds, s=11, color="white", edgecolor="#222222", lw=0.7, zorder=3)
ax.set_xticks(range(4))
ax.set_xticklabels([NAME[m] for m in ["EN", "LGBM", "XGB", "MLP"]])
ax.set_ylabel("Cross-validation MAE (years)")
ax.set_ylim(0, 19.5)
ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "fig4_cross_validation")

# ---------------------------------------------------------------- Figure 5
# Paired differences (Table 4) and bootstrap interval of the FT-Transformer MAE.
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
a.errorbar([8.264], [0], xerr=[[8.264 - 7.105], [9.482 - 8.264]], fmt="o", color=C["FT"], capsize=5, ms=6, lw=1.6)
a.annotate("FT-Transformer\n8.264 [7.105, 9.482]", (8.264, 0), xytext=(0, 12), textcoords="offset points",
           ha="center", fontsize=8)
for y, m in zip([-0.35, -0.6, -0.85], ["EN", "LGBM", "XGB"]):
    a.plot(test[m][0], y, "s", color=C[m], ms=6)
    a.annotate(f"{NAME[m]} {test[m][0]:.3f}", (test[m][0], y), xytext=(8, -3), textcoords="offset points", fontsize=8)
a.set_yticks([])
a.set_ylim(-1.05, 0.55)
a.set_xlim(2.5, 10.2)
a.set_xlabel("Test MAE (years)")
a.set_title("(a) MAE with bootstrap 95% interval", loc="left", fontsize=9)
a.grid(axis="y", visible=False)
rows = [("EN", 4.64, 3.52, 5.77), ("XGB", 3.80, 2.70, 4.90), ("LGBM", 3.84, 2.73, 4.94)]
for y, (m, d, lo, hi) in enumerate(rows[::-1]):
    b.errorbar([d], [y], xerr=[[d - lo], [hi - d]], fmt="o", color=C[m], capsize=4, ms=5.5, lw=1.5)
b.axvline(0, color="#444444", lw=0.9)
b.set_yticks(range(3))
b.set_yticklabels([f"vs {NAME[m]}" for m, *_ in rows[::-1]])
b.set_ylim(-0.6, 2.6)
b.set_xlim(-0.3, 6.5)
b.set_xlabel("Mean paired difference in absolute error (years)")
b.set_title("(b) FT-Transformer minus baseline", loc="left", fontsize=9)
b.grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "fig5_uncertainty")

# ---------------------------------------------------------------- Figure 6
fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.plot([500, 1000, 3000], [3.964, 3.901, 3.620], "-o", color=C["EN"], label="Elastic Net", lw=1.8)
ax.plot([500, 1000, 3000], [4.407, 4.371, 4.483], "-s", color=C["XGB"], label="XGBoost", lw=1.8)
for x, y in zip([500, 1000, 3000], [3.964, 3.901, 3.620]):
    ax.annotate(f"{y:.3f}", (x, y), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=7.5)
for x, y in zip([500, 1000, 3000], [4.407, 4.371, 4.483]):
    ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5)
ax.set_xticks([500, 1000, 3000])
ax.set_xticklabels(["500", "1,000", "3,000"])
ax.set_xlabel("Number of correlation-ranked CpGs")
ax.set_ylabel("Test MAE (years)")
ax.set_ylim(3.3, 4.9)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
save(fig, "fig6_feature_ablation")

# ---------------------------------------------------------------- Figure 7
ag = {"FT": [12.893, 9.268, 7.131], "EN": [2.960, 3.492, 3.775],
      "XGB": [5.013, 4.305, 4.484], "LGBM": [4.388, 4.681, 4.297]}
labels = ["≤40 y\n(n = 10)", "41–60 y\n(n = 43)", ">60 y\n(n = 79)"]
fig, ax = plt.subplots(figsize=(5.6, 3.3))
w = 0.2
for j, m in enumerate(["FT", "EN", "XGB", "LGBM"]):
    xs = np.arange(3) + (j - 1.5) * w
    ax.bar(xs, ag[m], width=w * 0.92, color=C[m], label=NAME[m])
    for x, v in zip(xs, ag[m]):
        ax.text(x, v + 0.15, f"{v:.1f}", ha="center", fontsize=6.8)
ax.set_xticks(range(3))
ax.set_xticklabels(labels)
ax.set_ylabel("Test MAE (years)")
ax.set_ylim(0, 15)
ax.legend(frameon=False, ncol=2, loc="upper right")
ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "fig7_age_groups")


# ---------------------------------------------------------------- Figures 1-2
def box(ax, xy, w, h, text, fc, fs=8.5, ec="#333333"):
    p = FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       fc=fc, ec=ec, lw=0.9)
    ax.add_patch(p)
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.1, color="#333333",
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


fig, ax = plt.subplots(figsize=(7.0, 6.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
blue, orange, green = "#E2EEF9", "#FAEBDA", "#E2F3E6"
box(ax, (5, 9.3), 5.6, 0.9, "GSE40279, whole blood\n656 samples, 473,034 CpG probes", blue)
box(ax, (5, 7.9), 4.0, 0.7, "Random split (seed 42)", blue)
box(ax, (2.6, 6.2), 4.2, 1.1, "Development pool\nn = 524", blue)
box(ax, (7.6, 6.2), 4.2, 1.1, "Held-out test set, n = 132\n(isolated: no ranking, no fitting,\nno hyperparameter choice)", orange, fs=8)
box(ax, (2.6, 4.45), 4.9, 1.0, "Pearson |r| ranking on the\ndevelopment pool only;\ntop 3,000 CpGs (2,993 retained)", blue, fs=8)
box(ax, (2.6, 2.8), 4.2, 1.3, "Model fitting and 5-fold CV\nFT-Transformer (Ludwig), Elastic Net,\nXGBoost, LightGBM, MLP", blue, fs=8)
box(ax, (5, 0.9), 6.6, 1.2, "Single evaluation on the test set\nMAE, RMSE, R², bootstrap interval,\npaired tests, age strata", green, fs=8.5)
arrow(ax, (5, 8.85), (5, 8.27))
arrow(ax, (4.2, 7.55), (2.9, 6.78))
arrow(ax, (5.8, 7.55), (7.3, 6.78))
arrow(ax, (2.6, 5.62), (2.6, 4.97))
arrow(ax, (2.6, 3.92), (2.6, 3.47))
arrow(ax, (2.6, 2.12), (3.9, 1.5))
arrow(ax, (7.6, 5.62), (6.4, 1.5), ls="--")
ax.text(8.0, 3.6, "selected CpGs and fitted\nmodels applied once", fontsize=7.5, ha="left", va="center", style="italic")
save(fig, "fig1_workflow")

fig, ax = plt.subplots(figsize=(5.2, 6.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.axis("off")
purple = "#EDE5F7"
box(ax, (4.2, 12.2), 6.4, 0.9, "Input: d = 2,993 CpG beta values", blue)
box(ax, (4.2, 10.7), 6.4, 0.9, "Feature tokenizer\nt$_i$ = x$_i$ W$_i$ + b$_i$", blue)
ax.add_patch(FancyBboxPatch((1.4, 5.3), 5.6, 4.2, boxstyle="round,pad=0.02,rounding_size=0.1", fc="none",
                            ec="#555555", lw=1.0, ls="--"))
ax.text(7.2, 7.4, "× L\nblocks", fontsize=8.5, va="center")
box(ax, (4.2, 8.7), 4.6, 0.8, "Multi-head self-attention", purple, fs=8)
box(ax, (4.2, 7.7), 4.6, 0.8, "Add & LayerNorm", purple, fs=8)
box(ax, (4.2, 6.7), 4.6, 0.8, "Feed-forward network", purple, fs=8)
box(ax, (4.2, 5.7), 4.6, 0.8, "Add & LayerNorm", purple, fs=8)
box(ax, (4.2, 4.0), 6.4, 0.9, "Token reduction", blue)
box(ax, (4.2, 2.6), 6.4, 0.9, "Regression head", blue)
box(ax, (4.2, 1.2), 6.4, 0.9, "Predicted chronological age", green)
arrow(ax, (4.2, 11.75), (4.2, 11.17))
arrow(ax, (4.2, 10.25), (4.2, 9.5))
arrow(ax, (4.2, 5.3), (4.2, 4.47))
arrow(ax, (4.2, 3.55), (4.2, 3.07))
arrow(ax, (4.2, 2.15), (4.2, 1.67))
save(fig, "fig2_architecture")


# ---------------------------------------------------------------- Figure 8
def crop_residuals():
    for key in ("ft_transformer", "elastic_net"):
        p = os.path.join(SRC, f"residuals_{key}.png")
        im = Image.open(p).convert("RGB")
        w, h = im.size
        im.crop((0, 38, w, h)).save(os.path.join(OUT, f"fig8_residuals_{key}.png"), dpi=(300, 300))


crop_residuals()
print("figures written to", OUT)

#!/usr/bin/env python3
"""复刻 materials/figure5b.png 版式，仅替换数据为最终模型 test 预测。"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))

RUNS = {
    "z": "runs_stageC_v13_zmin51",
    "h": "runs_stageC_archive_h1",
    "d": "runs_stageC_v13_d1_compat83_ckpt",
    "w": "runs_stageC_w359_archive",
}
T_LABEL = {2: "t3", 4: "t5", 8: "t9"}
T_COLOR = {2: "#7a9a3d", 4: "#4c72b0", 8: "#8a7390"}  # 与原图一致的橄榄绿/钢蓝/灰紫
Z_COLOR = "#1f3b6e"


def load(fam):
    d = RUNS[fam]
    cands = sorted(glob.glob(os.path.join(_HERE, d, "**", "predictions_valid.csv"), recursive=True), key=len)
    df = pd.read_csv(cands[-1])
    return df.y_true_nm.to_numpy(), df.y_pred_nm.to_numpy(), df.time_idx.to_numpy()


def ci_pi_bands(x, y, xs):
    """pred~target 线性拟合（OLS）的 95% CI（均值）与 95% PI（个体）带。
    与原稿图注一致：'95% confidence and prediction intervals from linear fits'。"""
    n = len(x)
    X = np.vstack([x, np.ones(n)]).T
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    dof = max(n - 2, 1)
    s2 = (resid ** 2).sum() / dof
    from scipy import stats
    t = stats.t.ppf(0.975, dof)
    xb = X.mean()
    sxx = ((x - xb) ** 2).sum()
    Xs = np.vstack([xs, np.ones(len(xs))]).T
    ys = Xs @ beta
    se_mean = np.sqrt(s2 * (1.0 / n + (xs - xb) ** 2 / sxx))
    se_pred = np.sqrt(s2 * (1.0 + 1.0 / n + (xs - xb) ** 2 / sxx))
    return ys, ys - t * se_mean, ys + t * se_mean, ys - t * se_pred, ys + t * se_pred


def draw_panel(ax, fam, panel_no, unit, scale, ticks=None, r2_x=0.97, r2_y=0.16):
    yt, yp, tidx = load(fam)
    x, y = yt * scale, yp * scale
    r2 = 1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = (hi - lo) * 0.10
    lo, hi = lo - pad, hi + pad
    xs = np.linspace(lo, hi, 200)
    ym, ci_lo, ci_hi, pi_lo, pi_hi = ci_pi_bands(x, y, xs)

    ax.fill_between(xs, pi_lo, pi_hi, color="#f4b6b6", alpha=0.45, lw=0, zorder=1)
    ax.fill_between(xs, ci_lo, ci_hi, color="#ef7d7d", alpha=0.45, lw=0, zorder=2)
    ax.plot(xs, xs, color="#d62728", lw=1.2, zorder=3)

    if fam == "z":
        ax.scatter(x, y, s=26, color=Z_COLOR, zorder=4, edgecolor="none")
    else:
        for t in (2, 4, 8):
            m = tidx == t
            ax.scatter(x[m], y[m], s=26, color=T_COLOR[t], zorder=4, edgecolor="none")
        leg_t = ax.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=T_COLOR[t],
                                  markersize=6, label=f"{fam}@{T_LABEL[t]}") for t in (2, 4, 8)],
                  loc="upper left", fontsize=7, frameon=True, edgecolor="black", framealpha=0.9)
        ax.add_artist(leg_t)

    ax.text(r2_x, r2_y, f"R² = {r2:.2f}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="black", lw=0.8))

    band_leg = [Line2D([0], [0], color="#d62728", lw=1.2, label="Ideal : y=x"),
                Patch(facecolor="#ef7d7d", alpha=0.45, label="95% CI"),
                Patch(facecolor="#f4b6b6", alpha=0.45, label="95% PI")]
    ax.legend(handles=band_leg, loc="lower right", fontsize=6, frameon=True, ncol=3,
              edgecolor="black", framealpha=0.9, bbox_to_anchor=(1.0, 0.0),
              handlelength=1.2, columnspacing=0.6)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if ticks is not None:
        ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlabel(f"Target Value ({unit})", fontsize=9)
    ax.set_ylabel(f"Predicted Value ({unit})", fontsize=9)
    ax.set_title(f"({panel_no})", fontsize=11, loc="left", pad=2)
    ax.tick_params(labelsize=8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)


fig, axes = plt.subplots(2, 2, figsize=(5.9, 4.2), dpi=300)
draw_panel(axes[0][0], "z", "i", "μm", 1e-3)
draw_panel(axes[0][1], "h", "ii", "×10$^2$ nm", 1e-2)
draw_panel(axes[1][0], "w", "iii", "×10$^2$ nm", 1e-2)
draw_panel(axes[1][1], "d", "iv", "×10$^2$ nm", 1e-2)
fig.tight_layout(h_pad=1.6, w_pad=1.8)
out = os.path.join(_HERE, "..", "docs", "fig5b_stageC_scatter.png")
fig.savefig(out, dpi=300)
print("saved", os.path.abspath(out))

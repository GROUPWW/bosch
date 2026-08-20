# -*- coding: utf-8 -*-
import os
import re
import glob
import json
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

# ===================== 配置区（按你的路径改） =====================
_HERE = os.path.dirname(os.path.abspath(__file__))

CASE_XLSX   = os.path.join(_HERE, "case_with_phys7.xlsx")
CASE_SHEET  = "Sheet1"
CASE_ID_COL = "input"   # 你的 case id（cas1/cas2/...）在这一列

IEDF_ROOT   = os.path.join(_HERE, "iedf_data")  # 原始 IEDF 树（如未提供则留空/占位）

OUT_XLSX    = os.path.join(_HERE, "case_with_phys7.xlsx")
OUT_JSON    = os.path.join(_HERE, "phys7_manifest.json")

# ===================== 可视化（论文用）配置 =====================
# 生成 IEDF + CDF + E10/E50/E90 标注图，默认开启
MAKE_FIGS = True
FIG_DIR = os.path.join(_HERE, "fig_iedf_examples")
FIG_N_CASES = 12
FIG_SEED = 0
FIG_STRATEGY = "phys7_fps"   # "phys7_fps"(最远点采样) / "random" / "extremes"
FIG_INCLUDE_EXTREMES = True   # 先加工艺极值/中值，再用 Phys7 多样性补齐
FIG_LOGY = False              # IEDF 是否用对数 y 轴（想看尾部可 True）
FIG_WITH_TOTAL = True         # 是否绘制总离子(聚合)曲线

# 固定：根据你数据结构（只取你指定的 dominant ions）
TARGETS = {
    ("SF6",  "sheath2"): ["F_1p", "SF3_1p", "SF4_1p", "SF5_1p"],
    ("C4F8", "sheath1"): ["CF3_1p", "C2F3_1p"],
}

# 固定：Phys7 特征名（用于抽样与可视化）
FAMILIES = [
    "logGamma_SF6_tot","pF_SF6","spread_SF6","qskew_SF6",
    "logGamma_C4F8_tot","rho_C4F8","spread_C4F8"
]

# 数值稳定
EPS = 1e-30

# 双峰判别参数（不是能量阈值；是形状判别的“显著性”）
SMOOTH_WIN = 9                 # 移动平均窗口（奇数更好）
MIN_PEAK_SEP_BINS = 5          # 两峰最小间隔（按能量网格bin）
SECOND_PEAK_MIN_FRAC = 0.25    # 第二峰至少达到第一峰的 25%
VALLEY_RATIO_TH = 0.80         # 峰间谷值需“足够低”：valley/min(peak1,peak2) <= 0.80

# ===================== 工具函数 =====================
def canon(s: str) -> str:
    # 只保留 [a-z0-9]，把下划线、括号、单位、斜杠都统一消掉
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def find_case_id_col(df: pd.DataFrame) -> str:
    cands = ["标识case", "case", "caseid", "id"]
    cols = list(df.columns)
    cc = {c: canon(c) for c in cols}
    for c in cols:
        v = cc[c]
        for k in cands:
            if canon(k) == v or canon(k) in v:
                return c
    raise KeyError(f"找不到 case id 列。现有列名：{cols}")

def parse_gas_sheath_from_filename(fp: str):
    base = os.path.basename(fp)
    m = re.match(r"^([A-Za-z0-9]+)_(sheath\d+)_energy_distribution\.csv$", base)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def pick_energy_col(df: pd.DataFrame) -> str:
    # 优先找明确的 Energy 列
    for c in df.columns:
        if canon(c).startswith("energyev") or canon(c) == "energy":
            return c
    # 兜底：包含 energy 且不是 energy_distribution
    for c in df.columns:
        cc = canon(c)
        if ("energy" in cc) and ("energydistribution" not in cc):
            return c
    # 最兜底：第一列
    return df.columns[0]


def pick_ion_col(df: pd.DataFrame, ion: str):
    ion_key = canon(ion) + "energydistribution"
    best = None
    for c in df.columns:
        cc = canon(c)
        if cc.startswith(ion_key):
            return c
        # 兜底：只要同时包含 ion 和 energydistribution
        if (canon(ion) in cc) and ("energydistribution" in cc):
            best = c
    return best


def moving_average(y: np.ndarray, win: int):
    if win is None or win <= 1:
        return y
    win = int(win)
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=float) / win
    return np.convolve(y, k, mode="same")

def local_maxima_indices(y: np.ndarray):
    """
    简单峰值检测：y[i-1] < y[i] >= y[i+1]
    """
    if y.size < 3:
        return np.array([], dtype=int)
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1

def cumulative_trapz(x: np.ndarray, y: np.ndarray):
    """
    CDF: ∫0^x y dx（离散梯形累积）
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2:
        return np.zeros_like(x)
    dx = np.diff(x)
    mid = 0.5 * (y[1:] + y[:-1])
    c = np.concatenate([[0.0], np.cumsum(mid * dx)])
    return c

def trapz_compat(y, x):
    fn = getattr(np, "trapezoid", None)
    if fn is None:
        return np.trapz(y, x)
    return fn(y, x)

def quantile_energy(x: np.ndarray, f: np.ndarray, q: float):
    """
    计算累计通量分位点能量 Eq：∫0^Eq f dE = q * ∫0^∞ f dE
    """
    x = np.asarray(x, float)
    f = np.clip(np.asarray(f, float), 0.0, None)
    if x.size < 2:
        return np.nan
    total = float(trapz_compat(f, x))
    if (not np.isfinite(total)) or total <= 0:
        return np.nan
    c = cumulative_trapz(x, f) / (total + EPS)
    return float(np.interp(q, c, x))

def detect_bimodal_flag(x: np.ndarray, f: np.ndarray) -> int:
    """
    形状判别双峰（用于诊断/可视化，不作为 Phys7 必选特征）
    """
    x = np.asarray(x, float)
    f = np.clip(np.asarray(f, float), 0.0, None)
    if x.size < 10:
        return 0

    g = moving_average(f, SMOOTH_WIN)
    peaks = local_maxima_indices(g)
    if peaks.size < 2:
        return 0

    # 取最高的两个峰
    order = np.argsort(g[peaks])[::-1]
    p1 = int(peaks[order[0]])
    p2 = int(peaks[order[1]])

    if abs(p2 - p1) < MIN_PEAK_SEP_BINS:
        return 0

    # 两峰排序（左->右）
    if p1 > p2:
        p1, p2 = p2, p1

    h1 = float(g[p1])
    h2 = float(g[p2])
    if h1 <= 0 or h2 <= 0:
        return 0
    if h2 < SECOND_PEAK_MIN_FRAC * h1:
        return 0

    # 谷值（峰间最小）
    valley = float(np.min(g[p1:p2+1]))
    if valley / max(EPS, min(h1, h2)) > VALLEY_RATIO_TH:
        return 0
    return 1

# ===================== 读取 IEDF 文件 =====================
def read_target_iedf_for_case(case_id: str):
    """
    在 IEDF_ROOT 下找：
    scan*/casX/*_energy_distribution.csv
    返回 dict: {(gas,sheath): filepath}
    """
    cid = str(case_id).strip()
    # 允许 "case1" -> "cas1"
    m = re.fullmatch(r"(?i)case(\d+)", cid)
    if m:
        cid = f"cas{m.group(1)}"
    if re.fullmatch(r"\d+", cid):
        cid = f"cas{cid}"

    # scan1-100/scan101-200/... 都兼容
    patt = os.path.join(IEDF_ROOT, "scan*", cid, "*_energy_distribution.csv")
    fps = glob.glob(patt)
    out = {}
    for fp in fps:
        gas, sheath = parse_gas_sheath_from_filename(fp)
        if gas is None:
            continue
        key = (gas, sheath)
        if key in TARGETS:
            out[key] = fp
    return out

def compute_phys7_from_file(csv_path: str, gas: str, sheath: str, ions: list):
    """
    从一个 csv 中提取：
    - 每个离子的通量 Gamma_i = ∫ f_i(E) dE
    - 聚合离子分布 f_agg(E) = Σ f_i(E)
    - 总通量 Gamma_tot = ∫ f_agg(E) dE
    - E10/E50/E90, spread=(E90-E10)/E50, qskew=(E90+E10-2E50)/(E90-E10)
    """
    df = pd.read_csv(csv_path)
    ecol = pick_energy_col(df)
    x = df[ecol].to_numpy(np.float64)

    gammas = {}
    ys = []
    for ion in ions:
        col = pick_ion_col(df, ion)
        if col is None:
            continue
        y = df[col].to_numpy(np.float64)
        y = np.where(np.isfinite(y), y, 0.0)
        y = np.clip(y, 0.0, None)
        Gamma = float(trapz_compat(y, x)) if x.size >= 2 else np.nan
        gammas[ion] = Gamma
        ys.append(y)

    if len(ys) == 0:
        return None

    f_agg = np.sum(np.stack(ys, axis=0), axis=0)
    Gamma_tot = float(trapz_compat(f_agg, x)) if x.size >= 2 else np.nan

    E10 = quantile_energy(x, f_agg, 0.10)
    E50 = quantile_energy(x, f_agg, 0.50)
    E90 = quantile_energy(x, f_agg, 0.90)

    spread = np.nan
    qskew = np.nan
    if np.isfinite(E10) and np.isfinite(E50) and np.isfinite(E90) and E50 > 0 and (E90 - E10) > 0:
        spread = float((E90 - E10) / (E50 + EPS))
        qskew = float((E90 + E10 - 2.0 * E50) / (E90 - E10 + EPS))

    return {
        "gas": gas, "sheath": sheath,
        "x": x,
        "gammas": gammas,
        "Gamma_tot": Gamma_tot,
        "E10": E10, "E50": E50, "E90": E90,
        "spread": spread,
        "qskew": qskew,
    }

# ===================== IEDF 可视化辅助（论文图） =====================
def _zscore(X: np.ndarray):
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True) + 1e-6
    return (X - mu) / sd

def farthest_point_sampling(X: np.ndarray, n: int, seed: int = 0):
    """
    最远点采样（farthest point sampling），用于在 Phys7 空间选“最有显示度”的多样样本。
    X: (N,d)
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if N == 0:
        return []
    n = min(n, N)
    start = int(rng.integers(0, N))
    chosen = [start]
    dist = np.full(N, np.inf, dtype=np.float64)
    for _ in range(1, n):
        last = X[chosen[-1]]
        d = np.sum((X - last) ** 2, axis=1)
        dist = np.minimum(dist, d)
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
    return chosen

def find_recipe7_cols(df: pd.DataFrame):
    """
    从 case.xlsx 里自动找 7 维 recipe 列（允许中文/带括号列名）。
    """
    cols = list(df.columns)
    def pick(pats):
        for c in cols:
            v = canon(c)
            if any(p in v for p in pats):
                return c
        return None

    apc = pick(["apc"])
    source_rf = pick(["source_rf", "sourcerf", "rfsource"])
    lf_rf = pick(["lf_rf", "lfrf", "bias"])
    sf6 = pick(["sf6"])
    c4f8 = pick(["c4f8"])
    dep_t = pick(["deptime", "dep_time", "dep时间", "dep"])
    etch_t = pick(["etchtime", "etch_time", "etch时间", "etch"])

    recipe_cols = [apc, source_rf, lf_rf, sf6, c4f8, dep_t, etch_t]
    if not all(recipe_cols):
        return None
    return recipe_cols

def select_cases_for_figs(df_case: pd.DataFrame, df_feat: pd.DataFrame, case_col: str,
                          n: int = 12, seed: int = 0,
                          strategy: str = "phys7_fps",
                          include_extremes: bool = True):
    """
    df_case: 原始 case 表
    df_feat: 每个 case 的 Phys7 特征表（含 case_id）
    返回：case_id list
    """
    rng = np.random.default_rng(seed)

    tmp = df_case.copy()
    tmp["_cid_"] = tmp[case_col].astype(str)
    tmp = tmp.merge(df_feat, how="left", left_on="_cid_", right_on="case_id", validate="one_to_one")

    # 至少有一套 IEDF：SF6 或 C4F8 特征非空
    ok_any = tmp[["logGamma_SF6_tot", "logGamma_C4F8_tot"]].notna().any(axis=1).to_numpy()
    tmp2 = tmp.loc[ok_any].reset_index(drop=True)
    if len(tmp2) == 0:
        return []

    chosen = set()

    # (1) 工艺覆盖：极值/中值（可选）
    recipe_cols = find_recipe7_cols(df_case) if include_extremes else None
    if recipe_cols is not None:
        apc_c, src_c, lf_c, sf6_c, c4_c, dep_c, etch_c = recipe_cols
        ratio = (tmp2[dep_c].astype(float) / (tmp2[etch_c].astype(float) + 1e-9)).to_numpy()

        axes = [
            (lf_c, tmp2[lf_c].astype(float).to_numpy()),
            (apc_c, tmp2[apc_c].astype(float).to_numpy()),
            (src_c, tmp2[src_c].astype(float).to_numpy()),
            ("dep/etch", ratio),
        ]
        for _, arr in axes:
            if arr.size == 0:
                continue
            order = np.argsort(arr)
            picks = []
            picks += order[:2].tolist()
            picks += order[max(0, len(order)//2 - 1): min(len(order), len(order)//2 + 1)].tolist()
            picks += order[-2:].tolist()
            for pi in picks:
                chosen.add(int(pi))

    # (2) Phys7 多样性：最远点采样补齐
    X = tmp2[FAMILIES].to_numpy(np.float64)
    col_mean = np.nanmean(X, axis=0, keepdims=True)
    X = np.where(np.isfinite(X), X, col_mean)
    X = _zscore(X)

    remaining = n - len(chosen)
    if remaining > 0:
        if strategy == "random":
            pool = [i for i in range(len(tmp2)) if i not in chosen]
            rng.shuffle(pool)
            for i in pool[:remaining]:
                chosen.add(int(i))
        elif strategy == "extremes":
            pool = [i for i in range(len(tmp2)) if i not in chosen]
            rng.shuffle(pool)
            for i in pool[:remaining]:
                chosen.add(int(i))
        else:
            fps_idx = farthest_point_sampling(X, n=n, seed=seed)
            for i in fps_idx:
                chosen.add(int(i))
            chosen = set(sorted(chosen)[:n])

    chosen_idx = sorted(list(chosen))[:n]
    case_ids = tmp2.loc[chosen_idx, "_cid_"].astype(str).tolist()
    return case_ids

def plot_iedf_multi_with_cdf(E: np.ndarray, ion_y: dict, out_png: str, title: str,
                            logy: bool = False, with_total: bool = True):
    """
    画：多离子 IEDF（左轴） + 聚合 CDF（右轴） + E10/E50/E90 标注
    ion_y: {ion_name: y(E)}
    """
    ions = list(ion_y.keys())
    if not ions:
        return

    f_agg = None
    for ion in ions:
        y = ion_y[ion]
        f_agg = y if f_agg is None else (f_agg + y)

    Gamma_tot = float(trapz_compat(f_agg, E)) if E.size >= 2 else np.nan
    if (not np.isfinite(Gamma_tot)) or Gamma_tot <= 0:
        return

    E10 = quantile_energy(E, f_agg, 0.10)
    E50 = quantile_energy(E, f_agg, 0.50)
    E90 = quantile_energy(E, f_agg, 0.90)
    cdf = cumulative_trapz(E, f_agg) / (Gamma_tot + EPS)

    plt.figure(figsize=(7.0, 4.0))
    ax = plt.gca()

    for ion in ions:
        ax.plot(E, ion_y[ion], label=ion)
    if with_total:
        ax.plot(E, f_agg, linewidth=2.0, label="sum")

    if logy:
        ax.set_yscale("log")
        ax.set_ylabel("energy_distribution (log scale)")
    else:
        ax.set_ylabel("energy_distribution")

    ax.set_xlabel("Energy (eV)")
    ax.set_title(title)

    ax2 = ax.twinx()
    ax2.plot(E, cdf, linestyle="--", label="CDF (sum)")
    ax2.set_ylabel("cumulative flux / Γ")

    for (q, e_q) in [(10, E10), (50, E50), (90, E90)]:
        if np.isfinite(e_q):
            ax.axvline(e_q, linestyle=":", linewidth=1.2)
            ax.text(e_q, ax.get_ylim()[1]*0.95, f"E{q}", rotation=90, va="top")

    ax.text(0.02, 0.02,
            f"Γ={Gamma_tot:.3e}\nE10={E10:.2f}, E50={E50:.2f}, E90={E90:.2f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()

def make_figures(case_ids: list, fig_dir: str, seed: int = 0,
                 logy: bool = False, with_total: bool = True):
    """
    为选中的 case 生成论文图：
    - SF6 sheath2: F+, SF3+, SF4+, SF5+
    - C4F8 sheath1: CF3+, C2F3+
    每个 case 两张图（如果对应 CSV 存在）
    """
    for cid in case_ids:
        files = read_target_iedf_for_case(cid)

        # SF6 sheath2
        key_sf6 = ("SF6", "sheath2")
        if key_sf6 in files:
            fp = files[key_sf6]
            ions = TARGETS[key_sf6]
            df = pd.read_csv(fp)
            E = df[pick_energy_col(df)].to_numpy(np.float64)
            ion_y = {}
            for ion in ions:
                col = pick_ion_col(df, ion)
                if col is None:
                    continue
                y = df[col].to_numpy(np.float64)
                y = np.where(np.isfinite(y), y, 0.0)
                y = np.clip(y, 0.0, None)
                ion_y[ion] = y
            if ion_y:
                out_png = os.path.join(fig_dir, f"{cid}", "SF6_sheath2_iedf_cdf.png")
                plot_iedf_multi_with_cdf(
                    E, ion_y, out_png,
                    title=f"{cid} — SF6 sheath2 IEDF (ions) + CDF",
                    logy=logy, with_total=with_total
                )

        # C4F8 sheath1
        key_c4 = ("C4F8", "sheath1")
        if key_c4 in files:
            fp = files[key_c4]
            ions = TARGETS[key_c4]
            df = pd.read_csv(fp)
            E = df[pick_energy_col(df)].to_numpy(np.float64)
            ion_y = {}
            for ion in ions:
                col = pick_ion_col(df, ion)
                if col is None:
                    continue
                y = df[col].to_numpy(np.float64)
                y = np.where(np.isfinite(y), y, 0.0)
                y = np.clip(y, 0.0, None)
                ion_y[ion] = y
            if ion_y:
                out_png = os.path.join(fig_dir, f"{cid}", "C4F8_sheath1_iedf_cdf.png")
                plot_iedf_multi_with_cdf(
                    E, ion_y, out_png,
                    title=f"{cid} — C4F8 sheath1 IEDF (ions) + CDF",
                    logy=logy, with_total=with_total
                )

# ===================== 主流程 =====================
def main():
    global CASE_XLSX, CASE_SHEET, CASE_ID_COL, IEDF_ROOT, OUT_XLSX, OUT_JSON

    parser = argparse.ArgumentParser(description="Extract Phys7 features from IEDF and optionally generate visualization figures.")
    parser.add_argument("--case_xlsx", default=CASE_XLSX)
    parser.add_argument("--case_sheet", default=CASE_SHEET)
    parser.add_argument("--case_id_col", default=CASE_ID_COL)
    parser.add_argument("--iedf_root", default=IEDF_ROOT)
    parser.add_argument("--out_xlsx", default=OUT_XLSX)
    parser.add_argument("--out_json", default=OUT_JSON)

    # 开关：默认按 MAKE_FIGS；用 --no_figs 关闭
    parser.add_argument("--no_figs", dest="make_figs", action="store_false", default=MAKE_FIGS)

    parser.add_argument("--fig_dir", default=FIG_DIR)
    parser.add_argument("--fig_n_cases", type=int, default=FIG_N_CASES)
    parser.add_argument("--fig_seed", type=int, default=FIG_SEED)
    parser.add_argument("--fig_strategy", default=FIG_STRATEGY)
    parser.add_argument("--no_fig_extremes", dest="fig_include_extremes", action="store_false", default=FIG_INCLUDE_EXTREMES)
    parser.add_argument("--fig_logy", action="store_true", default=FIG_LOGY)
    parser.add_argument("--no_fig_total", dest="fig_with_total", action="store_false", default=FIG_WITH_TOTAL)
    args = parser.parse_args()

    CASE_XLSX, CASE_SHEET, CASE_ID_COL = args.case_xlsx, args.case_sheet, args.case_id_col
    IEDF_ROOT = args.iedf_root
    OUT_XLSX, OUT_JSON = args.out_xlsx, args.out_json

    df_case = pd.read_excel(CASE_XLSX, sheet_name=CASE_SHEET)
    case_col = CASE_ID_COL if CASE_ID_COL in df_case.columns else find_case_id_col(df_case)
    case_ids = df_case[case_col].astype(str).tolist()

    feat_rows = []
    missing = []

    for cid in case_ids:
        files = read_target_iedf_for_case(cid)
        row = {"case_id": cid}

        # ---------- SF6 sheath2 ----------
        key_sf6 = ("SF6", "sheath2")
        if key_sf6 in files:
            out = compute_phys7_from_file(files[key_sf6], "SF6", "sheath2", TARGETS[key_sf6])
            if out is not None:
                Gamma_tot = out["Gamma_tot"]
                row["logGamma_SF6_tot"] = float(np.log10(Gamma_tot + EPS)) if np.isfinite(Gamma_tot) else np.nan

                Gamma_F = out["gammas"].get("F_1p", np.nan)
                row["pF_SF6"] = float(Gamma_F / (Gamma_tot + EPS)) if (np.isfinite(Gamma_F) and np.isfinite(Gamma_tot)) else np.nan

                row["spread_SF6"] = out["spread"]
                row["qskew_SF6"] = out["qskew"]
            else:
                row.update({"logGamma_SF6_tot": np.nan, "pF_SF6": np.nan, "spread_SF6": np.nan, "qskew_SF6": np.nan})
        else:
            row.update({"logGamma_SF6_tot": np.nan, "pF_SF6": np.nan, "spread_SF6": np.nan, "qskew_SF6": np.nan})

        # ---------- C4F8 sheath1 ----------
        key_c4 = ("C4F8", "sheath1")
        if key_c4 in files:
            out = compute_phys7_from_file(files[key_c4], "C4F8", "sheath1", TARGETS[key_c4])
            if out is not None:
                Gamma_tot = out["Gamma_tot"]
                row["logGamma_C4F8_tot"] = float(np.log10(Gamma_tot + EPS)) if np.isfinite(Gamma_tot) else np.nan

                G1 = out["gammas"].get("CF3_1p", np.nan)
                G2 = out["gammas"].get("C2F3_1p", np.nan)
                row["rho_C4F8"] = float(np.log10((G1 + EPS) / (G2 + EPS))) if (np.isfinite(G1) and np.isfinite(G2)) else np.nan

                row["spread_C4F8"] = out["spread"]
            else:
                row.update({"logGamma_C4F8_tot": np.nan, "rho_C4F8": np.nan, "spread_C4F8": np.nan})
        else:
            row.update({"logGamma_C4F8_tot": np.nan, "rho_C4F8": np.nan, "spread_C4F8": np.nan})

        if (key_sf6 not in files) and (key_c4 not in files):
            missing.append(cid)

        feat_rows.append(row)

    df_feat = pd.DataFrame(feat_rows)

    # 合并回 case.xlsx（保持行顺序不变）
    df_out = df_case.copy()
    df_out["_case_id_tmp_"] = df_out[case_col].astype(str)
    df_out = df_out.merge(df_feat, how="left", left_on="_case_id_tmp_", right_on="case_id", validate="one_to_one")
    df_out = df_out.drop(columns=["_case_id_tmp_", "case_id"])

    os.makedirs(os.path.dirname(OUT_XLSX), exist_ok=True)
    df_out.to_excel(OUT_XLSX, index=False)

    manifest = {
        "case_xlsx": CASE_XLSX,
        "iedf_root": IEDF_ROOT,
        "targets": {f"{k[0]}_{k[1]}": v for k, v in TARGETS.items()},
        "features": FAMILIES,
        "params": {
            "EPS": EPS,
            "SMOOTH_WIN": SMOOTH_WIN,
            "MIN_PEAK_SEP_BINS": MIN_PEAK_SEP_BINS,
            "SECOND_PEAK_MIN_FRAC": SECOND_PEAK_MIN_FRAC,
            "VALLEY_RATIO_TH": VALLEY_RATIO_TH
        },
        "n_cases": int(len(case_ids)),
        "missing_both_files_cases": missing[:200],
        "missing_both_files_count": int(len(missing)),
        "out_xlsx": OUT_XLSX
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved merged dataset: {OUT_XLSX}")
    print(f"[OK] Saved manifest: {OUT_JSON}")
    if missing:
        print(f"[WARN] cases missing BOTH target files (show first 20): {missing[:20]} (total={len(missing)})")

    # ---------- 可视化：抽样生成 IEDF/CDF 论文图 ----------
    if args.make_figs:
        case_ids_vis = select_cases_for_figs(
            df_case=df_case,
            df_feat=df_feat,
            case_col=case_col,
            n=args.fig_n_cases,
            seed=args.fig_seed,
            strategy=args.fig_strategy,
            include_extremes=args.fig_include_extremes,
        )
        if len(case_ids_vis) == 0:
            print("[WARN] make_figs enabled but no valid cases found for visualization.")
        else:
            print(f"[OK] make_figs: selected {len(case_ids_vis)} cases: {case_ids_vis}")
            make_figures(
                case_ids_vis,
                fig_dir=args.fig_dir,
                seed=args.fig_seed,
                logy=args.fig_logy,
                with_total=args.fig_with_total,
            )
            print(f"[OK] Figures saved under: {args.fig_dir}")

if __name__ == "__main__":
    main()

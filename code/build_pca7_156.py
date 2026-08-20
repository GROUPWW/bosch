# -*- coding: utf-8 -*-
"""
干净版 PCA-7：只基于 156 个有完整 IEDF 的 case（case_with_phys7_156.xlsx）。
与 build_pca_features_from_iedf.py 的区别：
  - 输入为 case_with_phys7_156.xlsx（156 行，两臂共同训练集）
  - 缺文件/坏样本直接报错，绝不做全零填充（原版对 619 个无 IEDF 的 case 置零，污染 PCA 基底）
  - 输出 case_with_pca7_156.xlsx（156 行：原始列 + pca_1..7）
"""
from __future__ import annotations
import os
import re
import glob
import json
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ================== 配置区 ==================
_HERE = os.path.dirname(os.path.abspath(__file__))

CASE_XLSX   = os.path.join(_HERE, "case_with_phys7_156.xlsx")
CASE_SHEET  = "Sheet1"
CASE_ID_COL = "input"

IEDF_ROOT   = os.path.join(_HERE, "iedf_data")  # 原始 IEDF 树（软链到 ../materials/TSV）

# SF6 sheath2；C4F8 sheath1
TARGETS = {
    ("SF6",  "sheath2"): ["F_1p", "SF3_1p", "SF4_1p", "SF5_1p"],
    ("C4F8", "sheath1"): ["CF3_1p", "C2F3_1p"],
}

NGRID = {
    ("SF6",  "sheath2"): 128,
    ("C4F8", "sheath1"): 64,
}

PCA_K = 7

OUT_XLSX = os.path.join(_HERE, "case_with_pca7_156.xlsx")
OUT_META = os.path.join(_HERE, "pca7_156_manifest.json")

EPS = 1e-30

# ============================================================
def normalize_case_id(cid: str) -> str:
    cid = str(cid).strip()
    if re.fullmatch(r"\d+", cid):
        return f"cas{cid}"
    m = re.fullmatch(r"(?i)case(\d+)", cid)
    if m:
        return f"cas{m.group(1)}"
    return cid

def parse_gas_sheath_from_filename(fp: str):
    base = os.path.basename(fp)
    m = re.match(r"^([A-Za-z0-9]+)_(sheath\d+)_energy_distribution\.csv$", base)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def pick_energy_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "energy" in c.lower():
            return c
    return df.columns[0]

def pick_ion_col(df: pd.DataFrame, ion_prefix: str):
    pref = ion_prefix.strip().lower()
    for c in df.columns:
        if c.strip().lower().startswith(pref):
            return c
    return None

def trapz_compat(y, x):
    fn = getattr(np, "trapezoid", None)
    if fn is None:
        return np.trapz(y, x)
    return fn(y, x)

def read_target_files_for_case(case_id: str):
    pattern = os.path.join(IEDF_ROOT, "scan*", str(case_id), "*_energy_distribution.csv")
    fps = sorted(glob.glob(pattern))
    got = {}
    for fp in fps:
        gas, sheath = parse_gas_sheath_from_filename(fp)
        if (gas, sheath) in TARGETS:
            got[(gas, sheath)] = fp
    return got

def make_energy_grid_from_file(fp: str, ngrid: int):
    df = pd.read_csv(fp)
    e_col = pick_energy_col(df)
    x = df[e_col].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return None
    x = np.sort(x)
    return np.linspace(float(x[0]), float(x[-1]), int(ngrid), dtype=float)

def ion_shape_vector(fp: str, gas: str, sheath: str, ions: list[str], grid: np.ndarray):
    """
    返回按离子拼接的形状向量：
    对每个离子：y -> Gamma -> g=y/(Gamma+eps) -> 插值到 grid
    """
    df = pd.read_csv(fp)
    e_col = pick_energy_col(df)
    x = df[e_col].to_numpy(dtype=float)
    ok = np.isfinite(x)
    x = x[ok]
    if x.size < 2:
        return None

    idx = np.argsort(x)
    x = x[idx]

    vecs = []
    missing_ions = 0

    for ion in ions:
        c = pick_ion_col(df, ion)
        if c is None:
            raise RuntimeError(f"离子列 {ion} 在 {fp} 中不存在（干净版不允许置零填充）")

        y = df[c].to_numpy(dtype=float)[ok][idx]
        y = np.where(np.isfinite(y), y, 0.0)
        y = np.maximum(y, 0.0)

        Gamma = float(trapz_compat(y, x))
        if not np.isfinite(Gamma) or Gamma <= 0:
            g = np.zeros_like(y, dtype=float)
        else:
            g = y / (Gamma + EPS)

        gi = np.interp(grid, x, g, left=0.0, right=0.0)
        vecs.append(gi.astype(float))

    return np.concatenate(vecs, axis=0), missing_ions

def main():
    df_case = pd.read_excel(CASE_XLSX, sheet_name=CASE_SHEET)
    if CASE_ID_COL not in df_case.columns:
        raise KeyError(f"CASE_ID_COL='{CASE_ID_COL}' 不在列名中：{list(df_case.columns)}")

    case_ids = df_case[CASE_ID_COL].astype(str).map(normalize_case_id).tolist()

    # 为每个 (gas,sheath) 选首个可用文件生成能量 grid
    grids = {key: None for key in TARGETS}
    for cid in case_ids:
        files = read_target_files_for_case(cid)
        for key in TARGETS:
            if grids[key] is None and key in files:
                grids[key] = make_energy_grid_from_file(files[key], NGRID[key])
        if all(grids[k] is not None for k in grids):
            break

    for k, g in grids.items():
        if g is None:
            raise RuntimeError(f"找不到任何 {k} 的 IEDF 文件，无法生成能量网格。")

    X_list = []
    ok_case = []
    miss_ion_total = 0

    keys_order = list(TARGETS.keys())  # 固定拼接顺序

    for cid in case_ids:
        files = read_target_files_for_case(cid)
        all_parts = []

        for key in keys_order:
            if key not in files:
                raise RuntimeError(
                    f"case {cid} 缺 {key[0]}_{key[1]} IEDF 文件；"
                    f"干净版要求 156 个 case 全部完整，请检查数据。"
                )
            fp = files[key]
            v, mi = ion_shape_vector(fp, key[0], key[1], TARGETS[key], grids[key])
            if v is None:
                raise RuntimeError(f"case {cid} 的 {fp} 无法解析能量轴。")
            miss_ion_total += mi
            all_parts.append(v)

        X_list.append(np.concatenate(all_parts, axis=0))
        ok_case.append(cid)

    X = np.vstack(X_list)
    assert X.shape[0] == 156, f"期望 156 个完整 case，实际 {X.shape[0]}"

    # 标准化 + PCA（只在这 156 个完整 case 上拟合）
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=PCA_K, random_state=0)
    Z = pca.fit_transform(Xs)  # [156, 7]

    df_pca = pd.DataFrame(Z, columns=[f"pca_{i+1}" for i in range(PCA_K)])
    df_pca["case_id"] = ok_case

    df_out = df_case.copy()
    df_out["_case_id_tmp_"] = df_out[CASE_ID_COL].astype(str).map(normalize_case_id)
    df_out = df_out.merge(df_pca, how="left", left_on="_case_id_tmp_", right_on="case_id")
    df_out = df_out.drop(columns=["_case_id_tmp_", "case_id"])

    assert len(df_out) == 156 and df_out[[f"pca_{i+1}" for i in range(PCA_K)]].notna().all().all()

    df_out.to_excel(OUT_XLSX, index=False)

    meta = {
        "case_xlsx": CASE_XLSX,
        "iedf_root": IEDF_ROOT,
        "targets": {f"{k[0]}_{k[1]}": v for k, v in TARGETS.items()},
        "ngrid": {f"{k[0]}_{k[1]}": int(NGRID[k]) for k in NGRID},
        "pca_k": PCA_K,
        "X_dim": int(X.shape[1]),
        "n_samples_used": int(X.shape[0]),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "missing_file_segments_count": 0,
        "missing_ion_columns_total": int(miss_ion_total),
        "out_xlsx": OUT_XLSX,
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {OUT_XLSX}  rows={len(df_out)}")
    print(f"[OK] Meta : {OUT_META}")
    print(f"[INFO] PCA explained variance ratio per PC: "
          f"{[round(v, 4) for v in meta['explained_variance_ratio']]}")
    print(f"[INFO] PCA explained variance sum = {meta['explained_variance_ratio_sum']:.4f}")

if __name__ == "__main__":
    main()

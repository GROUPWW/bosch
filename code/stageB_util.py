# -*- coding: utf-8 -*-
import os, re, csv, math, json, time, argparse
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from dataclasses import dataclass
import matplotlib
import matplotlib.pyplot as plt

import phys_model as pm

_HERE = os.path.dirname(os.path.abspath(__file__))

FAMILIES = ["zmin", "h0", "h1", "d0", "d1", "w"]
# 联合训练时可以把语义相近的 family 放在一起（例如 h0/h1 共享表示）
JOINT_FAMILY_GROUPS = {
    "h": ["h0", "h1"],
    "d": ["d0", "d1"],
}
TIME_LIST = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
TIME_VALUES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], np.float32)

def get_default_device(prefer: str = None) -> torch.device:
    """cuda > mps > cpu，支持 macOS M1 GPU 训练。"""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

RECIPE_KEY_ALIAS = {
    "apc": ["apc"],
    "source_rf": ["source_rf", "sourcerf", "rfsource"],
    "lf_rf": ["lf_rf", "lfrf", "bias"],
    "sf6": ["sf6"],
    "c4f8": ["c4f8"],
    "dep_time": ["deptime", "dep_time", "depositiontime"],
    "etch_time": ["etchtime", "etch_time"],
}

PHYS7_NAMES = [
    "logGamma_SF6_tot",
    "pF_SF6",
    "spread_SF6",
    "qskew_SF6",
    "logGamma_C4F8_tot",
    "rho_C4F8",
    "spread_C4F8",
]
_PHYS7_GROUPS = {
    "only_flux":    ["logGamma_SF6_tot", "pF_SF6", "logGamma_C4F8_tot"],   # 你可按需要微调
    "only_energy":  ["spread_SF6", "qskew_SF6", "spread_C4F8"],
    "only_polymer": ["rho_C4F8"],
}
class Cfg:
    verbose = True

    excel_path = os.path.join(_HERE, "case_with_phys7.xlsx")
    sheet_name = "Sheet1"
    case_id_col = "input"
    stageA_heads_root = os.path.join(_HERE, "runs_stageA_phys7", "best_by_test")

    save_root = os.path.join(_HERE, "runs_stageB_morph_phys7")
    families_to_train = FAMILIES

    seed = 42
    split_seeds = [0, 1, 2, 3, 4]
    device = get_default_device()
    num_workers = 6
    batch_size = 64
    batch_size_eval = 32
    epochs = 300
    lr = 2e-4
    weight_decay = 1e-4
    val_ratio = 0.15
    test_ratio = 0.15
    train_ratio = 1.0 - val_ratio - test_ratio
    early_patience = 30
    loss_type = "huber"  # "mse" / "huber"
    huber_beta = 0.1  # 默认值；会被 best_config 覆盖成 0.05

    tf_d_model = 256
    tf_nhead = 8
    tf_layers = 4
    tf_dropout = 0.1
    gru_hidden = 256
    gru_layers = 2
    mlp_hidden = 512
    mlp_layers = 3
    lstm_hidden = 256
    lstm_layers = 2
    tcn_hidden = 256
    tcn_layers = 4
    tcn_kernel = 3
    armlp_hidden = 256
    armlp_layers = 3
    export_force_positive = False
    test_eval_every = 5  # 每 5 个 epoch 评一次 test
    test_eval_max_batches = None
    clip_small_neg_to_zero = True
    small_neg_tol_um = 0.02  # 你也可以改更小，比如 0.01 / 0.005


    model_types = ["transformer", "gru", "mlp", "lstm", "tcn", "ar_mlp"]
    phys_sources = ["none", "stageA_pred"]
    recipe_aug_modes = ["base", "time", "gas", "rf", "squares"]
    phys7_modes = ["full", "only_energy", "only_flux", "none"]
    make_plots = True
    run_plan = "phase1"  # ✅ phase1 / phase2 / ablationA / fullgrid

    baseline_model_type = "transformer"
    baseline_phys_source = "stageA_pred"
    baseline_recipe_aug_mode = "time"
    baseline_phys7_mode = "full"

    phase1_split_seeds = [0, 1, 2,3,4,5]  # ✅ Phase1 少量 seeds 找 best 划分
    phase2_split_seeds = [0]  # Phase2 找不到 best_seed 时的 fallback

    show_mask_coverage = True
    log_every = 1
    target_bounds_um = {
        "zmin": (None, 5),
        "h0": (-0.02, 1),
        "h1": (-0.02, 1),
        "d0": (-0.02, 0.5),
        "d1": (-0.02, 0.5),
        "w": (-0.02, 1.5),
    }
    zmin_use_abs_for_bounds = True
    print_clean_stats = True

def log(msg: str):
    if getattr(Cfg, "verbose", True):
        print(msg, flush=True)

def set_seed(seed: int = 42):
    import random
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        import torch
        if isinstance(obj, torch.Tensor):
            # 标量 tensor -> float；非标量 -> list
            if obj.numel() == 1:
                return float(obj.detach().cpu().item())
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def _canon(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("（", "(").replace("）", ")")
    return s

def _pick_one(cols: List[str], candidates: List[str]) -> Optional[str]:
    cc = {c: _canon(c) for c in cols}
    for c in cols:
        v = cc[c]
        for pat in candidates:
            if pat in v:
                return c
    return None

def _detect_recipe_cols(cols: List[str]) -> List[str]:
    recipe_cols = [
        _pick_one(cols, RECIPE_KEY_ALIAS["apc"]),
        _pick_one(cols, RECIPE_KEY_ALIAS["source_rf"]),
        _pick_one(cols, RECIPE_KEY_ALIAS["lf_rf"]),
        _pick_one(cols, RECIPE_KEY_ALIAS["sf6"]),
        _pick_one(cols, RECIPE_KEY_ALIAS["c4f8"]),
        _pick_one(cols, RECIPE_KEY_ALIAS["dep_time"]),
        _pick_one(cols, RECIPE_KEY_ALIAS["etch_time"]),
    ]
    if not all(recipe_cols):
        raise KeyError(f"recipe7 列没找全：{recipe_cols}\n现有列：{cols}")
    return recipe_cols
def _detect_target_col(cols: List[str], fam: str, t: str) -> Optional[str]:
    fam_c = _canon(fam)
    t_c = _canon(t)
    t_alias = [t_c]
    if t_c == "9_2":
        t_alias += ["92", "9.2", "9-2", "9_2", "9(2)"]

    for c in cols:
        cc = _canon(c)
        if fam_c not in cc:
            continue
        for ta in t_alias:
            if cc.endswith("_"+ta) or ("_"+ta+"_") in cc or cc.endswith(ta) or ("t"+ta) in cc:
                return c
    # fallback
    cand = f"{fam}_{t}"
    for c in cols:
        if _canon(c) == _canon(cand):
            return c
    return None
def _norm_case_id(cid: str) -> str:
    cid = str(cid).strip()
    if re.fullmatch(r"\d+", cid):
        return f"cas{int(cid)}"
    cid = cid.lower().replace("case", "cas")
    if not cid.startswith("cas"):
        cid = "cas" + cid
    return cid
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
def _torch_load_ckpt_trusted(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def _torch_load_ckpt(path: str, map_location="cpu") -> Dict[str, Any]:
    try:
        # ✅ 强制添加 weights_only=False 以兼容 PyTorch 2.6+
        obj = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # 旧版本 PyTorch 不支持 weights_only 参数，回退
        obj = torch.load(path, map_location=map_location)

    if isinstance(obj, dict):
        if ("model" in obj) or ("state_dict" in obj) or ("meta" in obj):
            return obj
    return {"state_dict": obj, "meta": {}}

def _strip_state_dict_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    sd2 = {}
    for k, v in sd.items():
        kk = k
        if kk.startswith("model."):
            kk = kk[len("model."):]
        if kk.startswith("module."):
            kk = kk[len("module."):]
        sd2[kk] = v
    return sd2

def _zscore_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std  = x.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
    return mean, std
def _zscore_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std
def _zscore_inv(xn: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return xn * std + mean
def _to_np_f32(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)
def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-9):
    return a / (b + eps)

def masked_mae(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(pred - y)
    diff = diff[m]
    if diff.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    return diff.mean()
def masked_smoothl1(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    # SmoothL1/Huber: 0.5*x^2/beta if |x|<beta else |x|-0.5*beta
    diff = torch.abs(pred - y)
    loss = torch.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
    loss = loss[m]
    if loss.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    return loss.mean()
def masked_loss(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor,
                loss_type: str = "mse", huber_beta: float = 0.1) -> torch.Tensor:
    lt = str(loss_type).lower().strip()
    if lt in ["mse", "l2"]:
        return masked_mse(pred, y, m)
    elif lt in ["huber", "smoothl1", "smooth_l1"]:
        return masked_smoothl1(pred, y, m, beta=float(huber_beta))
    else:
        raise ValueError(f"Unknown loss_type={loss_type}")
def masked_mse(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    diff = (pred - y) ** 2
    diff = diff[m]
    if diff.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    return diff.mean()
def masked_r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # 1 - SS_res / SS_tot
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot
def compute_mae_nm(pred_denorm: np.ndarray, y_denorm: np.ndarray, m: np.ndarray, families: List[str],
                   clip_nonneg: bool = True) -> Dict:

    K = pred_denorm.shape[1]
    T = pred_denorm.shape[2]
    assert K == len(families), f"K mismatch: pred K={K} vs families={len(families)}"

    # —— 关键：先转“显示/论文口径” —— #
    pred_um = _apply_display_transform_um(pred_denorm, families, clip_nonneg=clip_nonneg)
    y_um    = _apply_display_transform_um(y_denorm,   families, clip_nonneg=clip_nonneg)

    fam_mae = {}
    fam_r2 = {}
    fam_mae_all = {}
    fam_r2_all = {}
    fam_count = {}

    for ki, fam in enumerate(families):
        mae_t = []
        r2_t = []
        for ti in range(T):
            mask = m[:, ki, ti]
            yt = y_um[mask, ki, ti]
            yp = pred_um[mask, ki, ti]

            ok = np.isfinite(yt) & np.isfinite(yp)
            yt = yt[ok]
            yp = yp[ok]
            if yt.size == 0:
                mae_t.append(float("nan"))
                r2_t.append(float("nan"))
            else:
                mae_t.append(float(np.mean(np.abs(yp - yt))) * 1e3)  # um -> nm
                r2_t.append(masked_r2_score_np(yt, yp))

        fam_mae[fam] = float(np.nanmean(mae_t)) if len(mae_t) else float("nan")
        fam_r2[fam]  = float(np.nanmean(r2_t))  if len(r2_t)  else float("nan")

        # overall per-family across all T
        mk = m[:, ki, :].reshape(-1)
        yt_all = y_um[:, ki, :].reshape(-1)[mk]
        yp_all = pred_um[:, ki, :].reshape(-1)[mk]
        ok = np.isfinite(yt_all) & np.isfinite(yp_all)
        yt_all = yt_all[ok]
        yp_all = yp_all[ok]
        fam_count[fam] = int(yt_all.size)

        if yt_all.size == 0:
            fam_mae_all[fam] = float("nan")
            fam_r2_all[fam] = float("nan")
        else:
            fam_mae_all[fam] = float(np.mean(np.abs(yp_all - yt_all))) * 1e3  # nm
            fam_r2_all[fam]  = masked_r2_score_np(yt_all, yp_all)

    # overall across all families+times
    overall_mask = m.reshape(-1)
    yt_all = y_um.reshape(-1)[overall_mask]
    yp_all = pred_um.reshape(-1)[overall_mask]
    ok = np.isfinite(yt_all) & np.isfinite(yp_all)
    yt_all = yt_all[ok]
    yp_all = yp_all[ok]

    if yt_all.size == 0:
        overall_mae_nm = float("nan")
        overall_r2 = float("nan")
    else:
        overall_mae_nm = float(np.mean(np.abs(yp_all - yt_all))) * 1e3
        overall_r2 = masked_r2_score_np(yt_all, yp_all)

    return {
        "per_family_mae_nm": fam_mae,
        "per_family_r2": fam_r2,
        "per_family_overall_mae_nm": fam_mae_all,
        "per_family_overall_r2": fam_r2_all,
        "per_family_overall_count": fam_count,
        "overall_mae_nm": overall_mae_nm,
        "overall_r2": overall_r2,
    }
def _apply_display_transform_um(arr_um, families, clip_nonneg=True):
    sign_map, nonneg_set = _default_family_sign_and_nonneg(families)
    out = arr_um.copy()
    for ki, fam in enumerate(families):
        out[:, ki, :] *= float(sign_map.get(fam, 1.0))
        if clip_nonneg and fam in nonneg_set:
            out[:, ki, :] = np.maximum(out[:, ki, :], 0.0)
    return out

def augment_recipe_features(recipe_raw: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode).lower().strip()
    x = recipe_raw.astype(np.float32)

    if mode == "base":
        return x
    apc, srf, lrf, sf6, c4f8, dt, et = [x[:, i:i+1] for i in range(7)]
    ones = np.ones_like(apc)

    feats = [apc, srf, lrf, sf6, c4f8, dt, et]

    if mode == "time":
        feats += [dt+et, _safe_div(dt, dt+et), _safe_div(et, dt+et)]
    elif mode == "gas":
        feats += [sf6+c4f8, _safe_div(sf6, sf6+c4f8), _safe_div(c4f8, sf6+c4f8)]
    elif mode == "rf":
        feats += [srf+lrf, _safe_div(srf, srf+lrf), _safe_div(lrf, srf+lrf)]
    elif mode == "coupling":
        feats += [
            (sf6+c4f8)*(dt+et),
            (srf+lrf)*(dt+et),
            (sf6+c4f8)*(srf+lrf),
        ]
    elif mode == "squares":
        feats += [f*f for f in feats]
    elif mode == "phys":
        gas_sum = sf6 + c4f8
        rf_sum  = srf + lrf
        t_sum   = dt + et
        feats += [
            gas_sum, rf_sum, t_sum,
            _safe_div(sf6, gas_sum), _safe_div(c4f8, gas_sum),
            _safe_div(srf, rf_sum), _safe_div(lrf, rf_sum),
            gas_sum * rf_sum, gas_sum * t_sum, rf_sum * t_sum,
        ]
    else:
        raise ValueError(f"Unknown recipe_aug_mode: {mode}")

    return np.concatenate(feats, axis=1).astype(np.float32)
def apply_phys7_mode(phys7: np.ndarray, mode: str) -> np.ndarray:
    if mode is None:
        return phys7
    mode = str(mode).lower().strip()
    out = phys7.astype(np.float32, copy=True)

    if mode in ["full", "normal"]:
        return out
    if mode in ["none", "zero"]:
        out[:] = 0.0
        return out

    if mode.startswith("drop"):
        k = int(mode.replace("drop", ""))
        if not (0 <= k < out.shape[1]):
            raise ValueError(f"drop index out of range: {mode}")
        out[:, k] = 0.0
        return out

    if mode in _PHYS7_GROUPS:
        keep = set(_PHYS7_GROUPS[mode])
        for j, name in enumerate(PHYS7_NAMES):
            if name not in keep:
                out[:, j] = 0.0
        return out

    raise ValueError(f"Unknown phys7_mode: {mode}")
def broadcast_phys7_to_T(phys7: np.ndarray, T: int) -> np.ndarray:
    return np.repeat(phys7[:, :, None], repeats=int(T), axis=2).astype(np.float32)
def get_phys7_seq_for_batch(recipe_raw_np: np.ndarray,
                            T: int,
                            phys_source: str,
                            phys7_mode: str,
                            stageA_provider=None) -> np.ndarray:
    phys_source = str(phys_source).lower().strip()
    N = int(recipe_raw_np.shape[0])

    if phys_source in ["none", "zero"]:
        return np.zeros((N, 7, int(T)), np.float32)

    if phys_source in ["stagea_pred", "stagea", "stagea_ensemble", "only_phys"]:
        if stageA_provider is None:
            raise RuntimeError("phys_source=stageA_pred but stageA_provider is None")
        p7 = stageA_provider.infer(recipe_raw_np, phys7_mode=phys7_mode, use_cache=True)  # (N,7)
        return broadcast_phys7_to_T(p7, T)

    raise ValueError(f"Unknown phys_source: {phys_source}")
def split_dataset_indices(N: int, seed: int, train_ratio: float, val_ratio: float):
    # test_ratio = 1 - train - val
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and (train_ratio + val_ratio) < 1
    rng = np.random.RandomState(int(seed))
    idx = np.arange(int(N))
    rng.shuffle(idx)
    n_train = int(round(N * train_ratio))
    n_val = int(round(N * val_ratio))
    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train:n_train + n_val].tolist()
    test_idx = idx[n_train + n_val:].tolist()
    return {"train": train_idx, "val": val_idx, "test": test_idx}
def detect_bad_rows_by_bounds(
    targets_full: np.ndarray,   # (N,K,T) float32 um
    mask_full: np.ndarray,      # (N,K,T) bool
    families: List[str],
    bounds_um: Dict[str, Tuple[Optional[float], Optional[float]]],
    zmin_use_abs: bool = True,
) -> np.ndarray:
    y = np.asarray(targets_full, dtype=np.float32)
    m = np.asarray(mask_full, dtype=bool)
    N, K, T = y.shape
    assert K == len(families), f"K mismatch: yK={K}, families={len(families)}"

    bad_row = np.zeros((N,), dtype=bool)

    for ki, fam in enumerate(families):
        lo, hi = bounds_um.get(fam, (None, None))
        if lo is None and hi is None:
            continue

        vals = y[:, ki, :]
        mm = m[:, ki, :]

        vv = np.abs(vals) if (fam.lower() == "zmin" and zmin_use_abs) else vals

        bad = np.zeros_like(mm, dtype=bool)
        if lo is not None:
            bad |= (mm & (vv < float(lo)))
        if hi is not None:
            bad |= (mm & (vv > float(hi)))

        if bad.any():
            bad_row |= bad.any(axis=1)

    return bad_row
def apply_target_bounds_filter(
    targets_full: np.ndarray,
    mask_full: np.ndarray,
    families: List[str],
    bounds_um: Dict[str, Tuple[Optional[float], Optional[float]]],
    zmin_use_abs: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    y = np.asarray(targets_full, dtype=np.float32).copy()
    m = np.asarray(mask_full, dtype=bool).copy()

    dropped = {f: 0 for f in families}

    for ki, fam in enumerate(families):
        lo, hi = bounds_um.get(fam, (None, None))
        if (lo is None) and (hi is None):
            continue

        vals = y[:, ki, :]
        mm = m[:, ki, :]

        vv = np.abs(vals) if (fam.lower() == "zmin" and zmin_use_abs) else vals

        bad = np.zeros_like(mm, dtype=bool)
        if lo is not None:
            bad |= (mm & (vv < float(lo)))
        if hi is not None:
            bad |= (mm & (vv > float(hi)))

        if bad.any():
            dropped[fam] = int(bad.sum())
            m[:, ki, :][bad] = False
            y[:, ki, :][bad] = 0.0

    return y, m, dropped
def clip_small_negative_to_zero(
    targets_full: np.ndarray,   # (N,K,T) um
    mask_full: np.ndarray,      # (N,K,T) bool
    families: List[str],
    neg_tol_um: float = 0.02,
    exclude_fams: Optional[List[str]] = None,
) -> Dict[str, int]:
    y = np.asarray(targets_full, dtype=np.float32)
    m = np.asarray(mask_full, dtype=bool)

    exclude = set([f.lower() for f in (exclude_fams or ["zmin"])])
    clipped = {f: 0 for f in families}

    for ki, fam in enumerate(families):
        if fam.lower() in exclude:
            continue
        mm = m[:, ki, :]
        vals = y[:, ki, :]
        hit = mm & (vals < 0.0) & (vals >= -float(neg_tol_um))
        if np.any(hit):
            clipped[fam] = int(hit.sum())
            vals[hit] = 0.0
            y[:, ki, :] = vals

    targets_full[:] = y
    return clipped
def count_negative_points(
    targets_full: np.ndarray,
    mask_full: np.ndarray,
    families: List[str],
    exclude_fams: Optional[List[str]] = None,
):
    y = np.asarray(targets_full, dtype=np.float32)
    m = np.asarray(mask_full, dtype=bool)
    exclude = set([f.lower() for f in (exclude_fams or [])])

    out = {}
    for ki, fam in enumerate(families):
        if fam.lower() in exclude:
            continue
        hit = m[:, ki, :] & (y[:, ki, :] < 0.0)
        out[fam] = int(hit.sum())
    return out
def print_clean_report(
    targets_full: np.ndarray,
    mask_full: np.ndarray,
    families: List[str],
    time_list: Optional[List[str]] = None,   # ✅ 改成可选
    title: str = "[CLEAN REPORT]",
    unit: str = "um",                 # "um" or "nm"
    apply_display: bool = True,       # True: zmin 翻正 + 非负clip（论文口径）
    clip_nonneg: bool = True,
    bounds_um: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    bad_row: Optional[np.ndarray] = None,     # (N0,) bool on "before-clean" set
    kept_idx: Optional[np.ndarray] = None,    # (N,) indices kept from before-clean
    before_N: Optional[int] = None,           # 传清洗前 N0 更直观
):
    # ✅ 默认使用全局 TIME_LIST（硬编码兜底，避免 main/其他地方漏传）
    if time_list is None:
        time_list = list(TIME_LIST)

    y = np.asarray(targets_full, dtype=np.float32)
    m = np.asarray(mask_full, dtype=bool)
    N, K, T = y.shape

    scale = 1e3 if str(unit).lower() == "nm" else 1.0

    log("\n" + "=" * 60)
    log(f"{title}  unit={unit}  apply_display={apply_display}  clip_nonneg={clip_nonneg}")
    log(f"[SHAPE] N={N}  K={K}  T={T}  families={families}  time={time_list}")

    # ---------- 行级剔除统计（如果提供） ----------
    if before_N is not None:
        if bad_row is not None:
            bad_row = np.asarray(bad_row, dtype=bool)
            if bad_row.size == before_N:
                log(f"[ROW CLEAN] before_N={before_N}  bad_rows={int(bad_row.sum())}  kept_rows={int((~bad_row).sum())}")
        if kept_idx is not None:
            kept_idx = np.asarray(kept_idx, dtype=np.int64)
            log(f"[ROW CLEAN] kept_idx size={int(kept_idx.size)}  keep_ratio={kept_idx.size / max(1, before_N):.2%}")

    # ---------- 总体 coverage ----------
    total_points = N * K * T
    valid_points = int(m.sum())
    log(f"[MASK] valid_points={valid_points}/{total_points}  overall_coverage={valid_points / max(1,total_points):.2%}")

    # 论文口径：zmin 翻正 + 可选非负clip
    if apply_display:
        y_disp = _apply_display_transform_um(y, families, clip_nonneg=clip_nonneg)
    else:
        y_disp = y.copy()

    # ---------- per-family ----------
    for ki, fam in enumerate(families):
        mk = m[:, ki, :]                      # (N,T)
        pts = int(mk.sum())
        cover = pts / max(1, (N * T))
        valid_samples = int(mk.any(axis=1).sum())

        vals = y_disp[:, ki, :][mk] * scale
        if vals.size == 0:
            log(f"  [{fam}] samples_valid={valid_samples}/{N}  points_valid=0  coverage={cover:.2%}  (NO VALID DATA)")
            continue

        qs = np.quantile(vals, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
        mu = float(np.mean(vals))
        sd = float(np.std(vals))

        msg = (
            f"  [{fam}] samples_valid={valid_samples}/{N}  "
            f"points_valid={pts}  coverage={cover:.2%}  "
            f"mean={mu:.4g}  std={sd:.4g}  "
            f"min={qs[0]:.4g} p1={qs[1]:.4g} p5={qs[2]:.4g} "
            f"p50={qs[3]:.4g} p95={qs[4]:.4g} p99={qs[5]:.4g} max={qs[6]:.4g}"
        )

        if bounds_um and fam in bounds_um:
            lo, hi = bounds_um[fam]
            lo_s = "" if lo is None else f"{float(lo)*scale:.4g}"
            hi_s = "" if hi is None else f"{float(hi)*scale:.4g}"
            msg += f"  | bounds=[{lo_s},{hi_s}]"

        log(msg)

    log("=" * 60 + "\n")
def print_missingness_report(
    df: pd.DataFrame,
    targets_full: np.ndarray,     # (N,K,T) after cleaning/clip
    mask_full: np.ndarray,        # (N,K,T)
    families: List[str],
    time_list: List[str],
    title: str = "[MISSINGNESS REPORT]",
    unit: str = "um",
    recipe_cols: Optional[List[str]] = None,
    show_examples: int = 3,       # 每个 family 展示前 show_examples 个“缺失最多的 time”
):
    """
    统计缺失情况（按 family×time）：
    - 列是否存在（基于 _detect_target_col）
    - mask=True 点数 / 缺失率
    - 0 值占比（在 mask=True 的点里）
    - 负值占比（在 mask=True 的点里）
    注意：targets_full/mask_full 是你清洗后的版本；df 也建议传清洗后的 df（prepare_shared_cache 切片后）。
    """
    y = np.asarray(targets_full, dtype=np.float32)
    m = np.asarray(mask_full, dtype=bool)
    N, K, T = y.shape
    assert K == len(families) and T == len(time_list)

    scale = 1e3 if str(unit).lower() == "nm" else 1.0

    log("\n" + "=" * 60)
    log(f"{title} unit={unit}  N={N}  K={K}  T={T}")
    if recipe_cols is not None:
        miss_recipe = [c for c in recipe_cols if c not in df.columns]
        log(f"[RECIPE] cols={len(recipe_cols)}  missing_in_df={len(miss_recipe)}")
        if miss_recipe:
            log(f"  missing recipe cols: {miss_recipe}")

    cols_list = df.columns.tolist()
    # 汇总：每个 family 哪些 time 缺失率最高
    fam_time_rows = {}

    for ki, fam in enumerate(families):
        rows = []
        for tj, t in enumerate(time_list):
            col = _detect_target_col(cols_list, fam, t)
            col_exist = (col is not None) and (col in df.columns)

            mk = m[:, ki, tj]
            valid_pts = int(mk.sum())
            miss_pts = N - valid_pts
            miss_rate = miss_pts / max(1, N)

            vals = y[:, ki, tj]
            zeros = int((mk & (vals == 0.0)).sum())
            negs  = int((mk & (vals < 0.0)).sum())

            rows.append((t, col, col_exist, valid_pts, miss_rate, zeros, negs))
        fam_time_rows[fam] = rows

    # 打印：每个 family 汇总 + top missing time
    for fam in families:
        rows = fam_time_rows[fam]
        total_valid = sum(r[3] for r in rows)
        total_possible = N * T
        overall_cov = total_valid / max(1, total_possible)

        # 列缺失数
        missing_cols = sum(1 for r in rows if not r[2])
        log(f"\n[{fam}] overall_coverage={overall_cov:.2%}  missing_cols={missing_cols}/{T}")

        # 选缺失率最高的 show_examples 个 time
        rows_sorted = sorted(rows, key=lambda x: x[4], reverse=True)
        for (t, col, col_exist, valid_pts, miss_rate, zeros, negs) in rows_sorted[:show_examples]:
            col_str = col if col is not None else "None"
            log(f"  - t={t}: col={col_str} exist={col_exist}  valid={valid_pts}/{N}  "
                f"miss={miss_rate:.2%}  zeros={zeros}  negs={negs}")

    # 可选：全局最差的若干列
    all_cols = []
    for fam in families:
        all_cols.extend([(fam,) + r for r in fam_time_rows[fam]])  # (fam,t,col,exist,valid,miss,zeros,negs)
    all_cols_sorted = sorted(all_cols, key=lambda x: x[5], reverse=True)[:10]
    log("\n[TOP 10 worst columns by missing rate]")
    for fam, t, col, col_exist, valid_pts, miss_rate, zeros, negs in all_cols_sorted:
        col_str = col if col is not None else "None"
        log(f"  - {fam} t={t}: col={col_str} exist={col_exist}  valid={valid_pts}/{N}  miss={miss_rate:.2%}")

    log("=" * 60 + "\n")

@dataclass
class StageAHeadInfo:
    head_index: int
    head_name: str
    ckpt_path: str
def scan_stageA_heads(heads_root: str, expect_k: int = 7) -> List[StageAHeadInfo]:
    if os.path.isfile(heads_root) and heads_root.endswith(".pth"):
        raise RuntimeError(
            f"[StageA] heads_root must be a directory with 7 heads, but got a single ckpt file:\n"
            f"  {heads_root}\n"
            f"Please set Cfg.stageA_heads_root = runs_stageA_phys7/best_by_test (or similar)."
        )

    if not os.path.isdir(heads_root):
        raise FileNotFoundError(f"StageA heads_root not found: {heads_root}")

    infos: List[StageAHeadInfo] = []
    for name in os.listdir(heads_root):
        if not name.startswith("head_"):
            continue
        m = re.match(r"head_(\d+)_", name)
        if not m:
            continue
        idx = int(m.group(1))
        head_dir = os.path.join(heads_root, name)
        ckpt_path = os.path.join(head_dir, "phys7_best.pth")
        if os.path.isfile(ckpt_path):
            head_name = name.split("_", 2)[-1] if len(name.split("_", 2)) == 3 else f"h{idx}"
            infos.append(StageAHeadInfo(idx, head_name, ckpt_path))

    infos.sort(key=lambda x: x.head_index)
    if len(infos) != expect_k:
        raise RuntimeError(
            f"[StageA] heads count mismatch: found={len(infos)} expect={expect_k}\n"
            f"root={heads_root}\n"
            f"found={[(i.head_index, os.path.basename(os.path.dirname(i.ckpt_path))) for i in infos]}"
        )
    return infos
def build_stageA_head_model_from_ckpt(ckpt: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any], Dict[str, torch.Tensor]]:
    meta = ckpt.get("meta", {}) or {}

    sd = ckpt.get("state_dict", None)
    if sd is None:
        sd = ckpt.get("model", ckpt)  # StageA 保存的是 "model"
    if not isinstance(sd, dict):
        raise RuntimeError("Invalid ckpt: missing state_dict/model dict")
    sd = _strip_state_dict_prefix(sd)

    model_type = str(ckpt.get("model_type", meta.get("model_type", "transformer"))).lower()
    in_dim = int(meta.get("in_dim", meta.get("D", 7)) or 7)
    out_dim = int(ckpt.get("out_dim", meta.get("out_dim", 1)) or 1)
    T_phys = int(meta.get("T", meta.get("T_phys", 1)) or 1)

    # ✅ 关键：兼容 phys_model.py 训练出的权重（有 time_mlp/input_proj/encoder/pos）
    #    与旧的 stageB_util _StageA_* 权重（无 time_mlp）
    is_phys_model_arch = any(k.startswith("time_mlp") for k in sd)

    if is_phys_model_arch:
        if "transformer" in model_type or model_type in ["tfm", "tf", "trans"]:
            d_model = int(sd["input_proj.weight"].shape[0])
            dim_ff = int(sd["encoder.layers.0.linear1.weight"].shape[0])
            num_layers = len([k for k in sd if re.match(r"encoder\.layers\.\d+\.self_attn\.in_proj_weight", k)])
            T = int(sd["pos"].shape[1])
            in_dim_from_sd = int(sd["input_proj.weight"].shape[1]) - 32  # minus time_embed
            out_dim_from_sd = int(sd["head.weight"].shape[0])
            # nhead 无法直接从 state dict 恢复；取能整除 d_model 的值
            nhead = int(meta.get("nhead", 4) or 4)
            if d_model % nhead != 0:
                nhead = _pick_divisor(d_model)
            dropout = float(meta.get("dropout", 0.1) or 0.1)
            model = pm.PhysicsSeqPredictor(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_ff=dim_ff,
                dropout=dropout,
                T=T,
                in_dim=in_dim_from_sd,
                out_dim=out_dim_from_sd,
            )
        elif "gru" in model_type or model_type in ["rnn"]:
            hidden = int(sd["gru.weight_ih_l0"].shape[0])
            num_layers = len([k for k in sd if re.match(r"gru\.weight_ih_l\d+$", k)])
            in_dim_from_sd = int(sd["proj.weight"].shape[1]) - 32
            out_dim_from_sd = int(sd["head.weight"].shape[0])
            T = int(sd["time_mlp.0.weight"].shape[1])  # not really T, but time_mlp input is 1
            # T from pos if exists, else from meta
            if "pos" in sd:
                T = int(sd["pos"].shape[1])
            else:
                T = T_phys
            dropout = float(meta.get("dropout", 0.1) or 0.1)
            model = pm.PhysicsGRUBaseline(
                hidden_dim=hidden,
                num_layers=num_layers,
                dropout=dropout,
                T=T,
                in_dim=in_dim_from_sd,
                out_dim=out_dim_from_sd,
            )
        else:
            hidden = int(sd["net.0.weight"].shape[0])
            num_layers = len([k for k in sd if re.match(r"net\.\d+\.weight", k)]) // 2
            in_dim_from_sd = int(sd["net.0.weight"].shape[1]) - 32
            out_dim_from_sd = int(sd[f"net.{2*(num_layers-1)}.weight"].shape[0])
            if "pos" in sd:
                T = int(sd["pos"].shape[1])
            else:
                T = T_phys
            dropout = float(meta.get("dropout", 0.1) or 0.1)
            model = pm.PhysicsMLPBaseline(
                hidden_dim=hidden,
                num_layers=num_layers,
                dropout=dropout,
                T=T,
                in_dim=in_dim_from_sd,
                out_dim=out_dim_from_sd,
            )
        return model, meta, sd

    # ---- 旧版 _StageA_* 架构（无 time_mlp） ----
    if "transformer" in model_type or model_type in ["tfm", "tf", "trans"]:
        d_model = int(meta.get("d_model", 0) or 0)
        if d_model <= 0:
            if "head.weight" in sd:
                d_model = int(sd["head.weight"].shape[1])
            elif "proj.weight" in sd:
                d_model = int(sd["proj.weight"].shape[0])
            else:
                d_model = 128
        num_layers = int(meta.get("num_layers", 0) or 0)
        if num_layers <= 0:
            ids = set()
            for k in sd.keys():
                m = re.match(r"enc\.layers\.(\d+)\.", k)
                if m:
                    ids.add(int(m.group(1)))
            num_layers = (max(ids) + 1) if ids else 2

        nhead = int(meta.get("nhead", 4) or 4)
        if d_model % nhead != 0:
            nhead = _pick_divisor(d_model)

        dropout = float(meta.get("dropout", 0.1) or 0.1)
        model = _StageA_Transformer(in_dim=in_dim, out_dim=out_dim, d_model=d_model,
                                    nhead=nhead, num_layers=num_layers, dropout=dropout, T_phys=T_phys)

    elif "gru" in model_type or model_type in ["rnn"]:
        hidden = int(meta.get("hidden", 0) or 0)
        if hidden <= 0:
            if "head.weight" in sd:
                hidden = int(sd["head.weight"].shape[1])
            elif "proj.weight" in sd:
                hidden = int(sd["proj.weight"].shape[0])
            else:
                hidden = 128

        num_layers = int(meta.get("num_layers", 0) or 0)
        if num_layers <= 0:
            ids = []
            for k in sd.keys():
                m = re.match(r"gru\.weight_ih_l(\d+)$", k)
                if m:
                    ids.append(int(m.group(1)))
            num_layers = (max(ids) + 1) if ids else 2

        model = _StageA_GRU(in_dim=in_dim, out_dim=out_dim, hidden=hidden, num_layers=num_layers, T_phys=T_phys)

    else:
        hidden = int(meta.get("hidden", 0) or 0)
        if hidden <= 0:
            if "net.0.weight" in sd:
                hidden = int(sd["net.0.weight"].shape[0])
            else:
                hidden = 128
        model = _StageA_MLP(in_dim=in_dim, out_dim=out_dim, hidden=hidden)

    return model, meta, sd
class _StageA_Transformer(nn.Module):
    def __init__(self, in_dim=7, out_dim=1, d_model=128, nhead=4, num_layers=2, dropout=0.1, T_phys: int = 1):
        super().__init__()
        self.T_phys = T_phys
        self.proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, recipe7: torch.Tensor, time_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = recipe7.size(0)
        x = recipe7.unsqueeze(1).expand(B, self.T_phys, recipe7.size(1))
        x = self.proj(x)
        x = self.enc(x)
        y = self.head(x)          # (B,T,out_dim)
        return y.transpose(1, 2)  # (B,out_dim,T)
class _StageA_GRU(nn.Module):
    def __init__(self, in_dim=7, out_dim=1, hidden=128, num_layers=2, T_phys: int = 1):
        super().__init__()
        self.T_phys = T_phys
        self.proj = nn.Linear(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, recipe7: torch.Tensor, time_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = recipe7.size(0)
        x = recipe7.unsqueeze(1).expand(B, self.T_phys, recipe7.size(1))
        x = self.proj(x)
        h, _ = self.gru(x)
        y = self.head(h)          # (B,T,out_dim)
        return y.transpose(1, 2)  # (B,out_dim,T)
class _StageA_MLP(nn.Module):
    def __init__(self, in_dim=7, out_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, recipe7: torch.Tensor, time_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.net(recipe7)                 # (B,out_dim)
        return y.unsqueeze(-1)                # (B,out_dim,1)
def align_recipe_raw_to_stageA(recipe_raw_np: np.ndarray,
                              recipe_cols_in: List[str],
                              stageA_recipe_cols: List[str]) -> np.ndarray:
    if (recipe_cols_in is None) or (stageA_recipe_cols is None):
        return recipe_raw_np

    in_canon = [_canon(c) for c in recipe_cols_in]
    stage_canon = [_canon(c) for c in stageA_recipe_cols]
    idx_map = []
    for sc in stage_canon:
        if sc in in_canon:
            idx_map.append(in_canon.index(sc))
        else:
            idx_map.append(-1)

    if all(i >= 0 for i in idx_map):
        return recipe_raw_np[:, idx_map].astype(np.float32)
    key_to_i = {}
    for i, c in enumerate(recipe_cols_in):
        vc = _canon(c)
        for k, pats in RECIPE_KEY_ALIAS.items():
            if any(p in vc for p in pats):
                key_to_i[k] = i

    stage_key_seq = []
    for c in stageA_recipe_cols:
        vc = _canon(c)
        kk = None
        for k, pats in RECIPE_KEY_ALIAS.items():
            if any(p in vc for p in pats):
                kk = k
                break
        stage_key_seq.append(kk)

    idx_map2 = []
    for kk in stage_key_seq:
        idx_map2.append(key_to_i.get(kk, -1))

    if any(i < 0 for i in idx_map2):
        raise RuntimeError(
            f"align_recipe_raw_to_stageA failed.\n"
            f"  recipe_cols_in={recipe_cols_in}\n"
            f"  stageA_recipe_cols={stageA_recipe_cols}\n"
            f"  idx_map2={idx_map2}"
        )
    return recipe_raw_np[:, idx_map2].astype(np.float32)
class StageAEnsemblePhys7Provider:
    def __init__(self,
                 heads_root: str,
                 device: str = "cuda",
                 recipe_cols_in: Optional[List[str]] = None,
                 expect_k: int = 7):
        self.heads_root = heads_root
        self.expect_k = int(expect_k)
        self.device = get_default_device(device)
        self.in_recipe_cols = recipe_cols_in

        self.head_infos = scan_stageA_heads(heads_root, expect_k=self.expect_k)
        self.models: List[nn.Module] = []
        self.metas: List[Dict[str, Any]] = []
        self.stageA_recipe_cols: Optional[List[str]] = None
        self.norm_x_mean: Optional[np.ndarray] = None
        self.norm_x_std: Optional[np.ndarray] = None

        self._cache: Dict[bytes, np.ndarray] = {}
        self._load_all()

    def _load_all(self):
        for idx, info in enumerate(self.head_infos):
            ck = _torch_load_ckpt(info.ckpt_path, map_location="cpu")
            model, meta, sd = build_stageA_head_model_from_ckpt(ck)
            model.load_state_dict(sd, strict=False)
            model.to(self.device).eval()
            self.models.append(model)
            self.metas.append(meta)

            if idx == 0:
                self.stageA_recipe_cols = list(meta.get("recipe_cols", [])) or None
                ns = meta.get("norm_static", {}) or {}
                xmean = ns.get("mean", None)
                xstd = ns.get("std", None)
                if xmean is not None and xstd is not None:
                    if isinstance(xmean, torch.Tensor):
                        xmean = xmean.detach().cpu().numpy()
                    if isinstance(xstd, torch.Tensor):
                        xstd = xstd.detach().cpu().numpy()
                    self.norm_x_mean = np.array(xmean, np.float32).reshape(1, -1)
                    self.norm_x_std  = np.array(xstd,  np.float32).reshape(1, -1)

        if len(self.models) != self.expect_k:
            raise RuntimeError(f"[StageA] loaded heads mismatch: {len(self.models)} vs expect {self.expect_k}")

    @torch.no_grad()
    def infer(self,
              recipe_raw_np: np.ndarray,
              phys7_mode: str = "full",
              use_cache: bool = True) -> np.ndarray:
        recipe_raw_np = np.asarray(recipe_raw_np, dtype=np.float32)
        N, D = recipe_raw_np.shape
        if D != 7:
            raise ValueError(f"StageA provider expects recipe_raw dim=7, got {D}")
        if len(self.models) != self.expect_k:
            raise RuntimeError(f"[StageA] provider not ready: heads={len(self.models)} expect={self.expect_k}")

        # 1) 对齐列顺序
        if (self.stageA_recipe_cols is not None) and (self.in_recipe_cols is not None):
            recipe_raw_np = align_recipe_raw_to_stageA(
                recipe_raw_np,
                recipe_cols_in=self.in_recipe_cols,
                stageA_recipe_cols=self.stageA_recipe_cols
            )

        if (self.norm_x_mean is not None) and (self.norm_x_std is not None):
            x = (recipe_raw_np - self.norm_x_mean) / (self.norm_x_std + 1e-6)
        else:
            x = recipe_raw_np

        x_t = torch.from_numpy(x).to(self.device)
        # ✅ 构造 time_values：从 meta 的 time_values 取值，默认 1.0
        time_vals = self.metas[0].get("time_values", [1.0]) if self.metas else [1.0]
        T = int(x_t.size(0)), int(len(time_vals))
        t_t = torch.tensor(time_vals, dtype=torch.float32, device=self.device).unsqueeze(0).expand(T[0], -1)

        # 3) 推理（逐 head），输出严格 (N,7)
        out = np.zeros((N, self.expect_k), np.float32)

        if use_cache:
            miss_idx = []
            miss_rows = []
            for i in range(N):
                key = recipe_raw_np[i].tobytes()
                if key in self._cache:
                    out[i] = self._cache[key]
                else:
                    miss_idx.append(i)
                    miss_rows.append(x_t[i:i+1])

            if miss_idx:
                miss_x = torch.cat(miss_rows, dim=0)
                cols = []
                for m in self.models:
                    y = m(miss_x, t_t[:miss_x.size(0)])[:, 0, 0]  # (B,)
                    cols.append(y.detach().cpu().numpy())
                miss_phys7 = np.stack(cols, axis=1).astype(np.float32)  # (B,7)

                if miss_phys7.shape[1] != self.expect_k:
                    raise RuntimeError(f"[StageA] infer heads mismatch: {miss_phys7.shape}")

                for j, i in enumerate(miss_idx):
                    out[i] = miss_phys7[j]
                    self._cache[recipe_raw_np[i].tobytes()] = miss_phys7[j]
        else:
            cols = []
            for m in self.models:
                y = m(x_t, t_t)[:, 0, 0]
                cols.append(y.detach().cpu().numpy())
            out = np.stack(cols, axis=1).astype(np.float32)
            if out.shape[1] != self.expect_k:
                raise RuntimeError(f"[StageA] infer heads mismatch: {out.shape}")

        # 4) ablation
        out = apply_phys7_mode(out, phys7_mode)
        return out
_STAGEA_PROVIDER_CACHE: Dict[str, "StageAEnsemblePhys7Provider"] = {}
def infer_phys7_from_stageA_ckpt(stageA_path_or_root: str,
                                 recipe_raw_np: np.ndarray,
                                 case_recipe_cols: Optional[List[str]] = None,
                                 phys7_mode: str = "full",
                                 device: str = None) -> np.ndarray:
    if device is None:
        device = Cfg.device if hasattr(Cfg, "device") else get_default_device()

    key = f"{stageA_path_or_root}||{','.join(case_recipe_cols) if case_recipe_cols else ''}||{device}"
    if key not in _STAGEA_PROVIDER_CACHE:
        _STAGEA_PROVIDER_CACHE[key] = StageAEnsemblePhys7Provider(
            heads_root=stageA_path_or_root,
            device=device,
            recipe_cols_in=case_recipe_cols,
            expect_k=7
        )
    provider = _STAGEA_PROVIDER_CACHE[key]
    return provider.infer(recipe_raw_np, phys7_mode=phys7_mode, use_cache=True)
def build_morph_dataset_phys7(
    excel_path: str, sheet_name: str, case_id_col: str,
    target_family: str = None,
    phys_source: str = "stageA_pred",
    recipe_aug_mode: str = "base",
    phys7_mode: str = "full",
    stageA_heads_root: str = None,
    stageA_provider: Optional["StageAEnsemblePhys7Provider"] = None,
    fit_norm_idx: Optional[List[int]] = None,
    df: Optional[pd.DataFrame] = None,
    recipe_cols: Optional[List[str]] = None,
    recipe_raw: Optional[np.ndarray] = None,
    targets_full: Optional[np.ndarray] = None,
    mask_full: Optional[np.ndarray] = None,
    phys7_seq_full: Optional[np.ndarray] = None,
):
    def _zscore_fit_idx(x: np.ndarray, idx: Optional[List[int]], eps: float = 1e-6):
        x = np.asarray(x, dtype=np.float32)
        if idx is not None:
            ii = np.asarray(idx, dtype=np.int64)
            x_fit = x[ii]
        else:
            x_fit = x
        mean = x_fit.mean(axis=0, keepdims=True).astype(np.float32)
        std = x_fit.std(axis=0, keepdims=True).astype(np.float32)
        std = np.maximum(std, eps).astype(np.float32)
        return mean, std

    def _fit_target_norm(targets: np.ndarray, mask: np.ndarray, idx: Optional[List[int]], eps: float = 1e-6):
        targets = np.asarray(targets, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if idx is not None:
            ii = np.asarray(idx, dtype=np.int64)
            t_fit = targets[ii]
            m_fit = mask[ii]
        else:
            t_fit = targets
            m_fit = mask
        y = t_fit[m_fit]
        if y.size == 0:
            return 0.0, 1.0
        mu = float(np.mean(y))
        sd = float(np.std(y))
        if sd < eps:
            sd = 1.0
        return mu, sd

    if df is None:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

    if recipe_cols is None:
        recipe_cols = _detect_recipe_cols(df.columns.tolist())

    if recipe_raw is None:
        recipe_raw = df[recipe_cols].values.astype(np.float32)

    static_x = augment_recipe_features(recipe_raw, recipe_aug_mode)

    N = int(len(df))
    T = int(len(TIME_LIST))

    if phys7_seq_full is not None:
        p = np.asarray(phys7_seq_full, dtype=np.float32)
        if p.ndim == 3:
            phys7_raw_full = p[:, :, 0]
        elif p.ndim == 2:
            phys7_raw_full = p
        else:
            raise ValueError(f"phys7_seq_full ndim must be 2 or 3, got {p.shape}")
        if phys7_raw_full.shape[1] != 7:
            raise ValueError(f"phys7_seq_full must have 7 phys dims, got {phys7_raw_full.shape}")
    else:
        ps = str(phys_source).lower().strip()
        if ps in ["none", "zero"]:
            phys7_raw_full = np.zeros((N, 7), dtype=np.float32)
        elif ps in ["stagea_pred", "stagea", "stagea_ensemble", "only_phys"]:
            # only_phys（R2-3a a-ii）：描述符仍取 stageA_pred，static recipe 流在下方置零
            if stageA_provider is None:
                if stageA_heads_root is None:
                    raise ValueError("stageA_heads_root is required for phys_source='stageA_pred'")
                stageA_provider = StageAEnsemblePhys7Provider(
                    heads_root=stageA_heads_root,
                    device=Cfg.device,
                    recipe_cols_in=recipe_cols,
                    expect_k=7
                )
            phys7_raw_full = stageA_provider.infer(recipe_raw, phys7_mode="full", use_cache=True)  # (N,7)
        elif ps in ["phys7_true", "true", "excel"]:
            # 直接从 Excel 读取真值 7 维描述符（用于消融/诊断）
            if df is None:
                raise ValueError("phys_source='phys7_true' requires df to be provided")
            missing = [c for c in PHYS7_NAMES if c not in df.columns]
            if missing:
                raise KeyError(f"phys7_true missing columns: {missing}")
            phys7_raw_full = df[PHYS7_NAMES].values.astype(np.float32)
            # 真值描述符可能存在缺失，用列中位数填充，避免 zscore/训练出 NaN
            for ci in range(phys7_raw_full.shape[1]):
                col = phys7_raw_full[:, ci]
                valid_mask = np.isfinite(col)
                if not valid_mask.all():
                    fill_val = float(np.median(col[valid_mask])) if valid_mask.any() else 0.0
                    col[~valid_mask] = fill_val
            phys7_raw_full = np.nan_to_num(phys7_raw_full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        else:
            raise ValueError(f"Unknown phys_source: {phys_source}")
    if str(phys_source).lower().strip() in ["none", "zero"]:
        p7_mean = np.zeros((1, 7), dtype=np.float32)
        p7_std = np.ones((1, 7), dtype=np.float32)
        phys7_z = np.zeros((N, 7), dtype=np.float32)
    else:
        p7_mean, p7_std = _zscore_fit_idx(phys7_raw_full, fit_norm_idx, eps=1e-6)
        phys7_z = (phys7_raw_full - p7_mean) / p7_std
    phys7_z = apply_phys7_mode(phys7_z, phys7_mode)  # (N,7)
    phys7_seq = broadcast_phys7_to_T(phys7_z, T).astype(np.float32)
    if (targets_full is not None) and (mask_full is not None):
        targets = np.asarray(targets_full, dtype=np.float32)
        mask = np.asarray(mask_full, dtype=bool)
    else:
        targets = np.zeros((N, len(FAMILIES), T), dtype=np.float32)
        mask = np.zeros((N, len(FAMILIES), T), dtype=bool)

        eps = 1e-12
        cols_list = df.columns.tolist()
        for i, fam in enumerate(FAMILIES):
            for j, t in enumerate(TIME_LIST):
                col_name = _detect_target_col(cols_list, fam, t)
                if col_name and col_name in df.columns:
                    vals = pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype=np.float32)
                    valid = np.isfinite(vals) & (np.abs(vals) > eps)
                    targets[valid, i, j] = vals[valid]
                    mask[valid, i, j] = True

    families = list(FAMILIES)
    if target_family is not None:
        if target_family in JOINT_FAMILY_GROUPS:
            group = JOINT_FAMILY_GROUPS[target_family]
            ks = [families.index(f) for f in group]
            targets = targets[:, ks, :]
            mask = mask[:, ks, :]
            families = list(group)
        else:
            assert target_family in families
            k = families.index(target_family)
            targets = targets[:, k:k+1, :]
            mask = mask[:, k:k+1, :]
            families = [target_family]
    x_mean, x_std = _zscore_fit_idx(static_x, fit_norm_idx, eps=1e-6)
    static_xn = (static_x - x_mean) / x_std
    if str(phys_source).lower().strip() in ["only_phys"]:
        # R2-3a a-ii：屏蔽 static recipe 输入流。
        # 在归一化空间置零 = 每个特征取训练集均值，等价于“不含任何 recipe 信息”，
        # 模型只剩 phys7 描述符 + time 可用。norm_static 元数据保持原样，便于复现。
        static_xn = np.zeros_like(static_xn, dtype=np.float32)
    y_mean_val, y_std_val = _fit_target_norm(targets, mask, fit_norm_idx, eps=1e-6)
    targets_n = (targets - y_mean_val) / (y_std_val + 1e-6)
    static_x_t = torch.from_numpy(static_xn.astype(np.float32))
    phys7_seq_t = torch.from_numpy(phys7_seq.astype(np.float32))
    targets_t = torch.from_numpy(targets_n.astype(np.float32))
    mask_t = torch.from_numpy(mask.astype(bool))
    time_mat_t = torch.from_numpy(np.tile(TIME_VALUES[None, :], (N, 1)).astype(np.float32))

    ds = TensorDataset(static_x_t, phys7_seq_t, targets_t, mask_t, time_mat_t)

    meta = {
        "N": N,
        "Ds": int(static_xn.shape[1]),
        "K": int(len(families)),
        "T": int(T),
        "families": families,

        "norm_static": {"mean": x_mean, "std": x_std},
        "norm_target": {"mean": [float(y_mean_val)], "std": [float(y_std_val)]},
        "norm_phys7": {
            "names": list(PHYS7_NAMES),
            "mean": p7_mean,
            "std": p7_std,
            "fit_on": "train_idx" if fit_norm_idx is not None else "all",
            "mode_applied_after_norm": True,
        },

        "recipe_cols": recipe_cols,
        "target_family": target_family,
        "phys_source": phys_source,
        "phys7_mode": phys7_mode,
        "recipe_aug_mode": recipe_aug_mode,
        "stageA_heads_root": stageA_heads_root,
    }

    if Cfg.show_mask_coverage:
        log(f"      [DATA] Mask coverage: {_mask_coverage(mask) * 100:.2f}%")

    return ds, meta
def build_sincos_pos(d_model: int, T: int) -> torch.Tensor:
    pe = torch.zeros(1, T, d_model)
    position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)
    return pe
class StaticEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class TimeMLP(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU()
        )
    def forward(self, t):
        return self.net(t)

class MorphTransformer(nn.Module):
    def __init__(self, static_dim: int, phys_dim: int = 7, d_model=256, nhead=8, num_layers=4, dropout=0.1,
                 out_dim: int = 6):
        super().__init__()
        self.static_enc = StaticEncoder(static_dim, out_dim=128)
        self.phys_proj = nn.Linear(phys_dim, 128)
        self.time_mlp = TimeMLP(out_dim=64)
        self.in_proj = nn.Linear(128 + 128 + 64, d_model)
        self.pos = build_sincos_pos(d_model, len(TIME_LIST))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, out_dim)

    def forward(self, static_x, phys7_seq, time_mat):
        B = static_x.shape[0]
        T = phys7_seq.shape[2]
        s = self.static_enc(static_x)  # (B, 128)
        s = s[:, None, :].repeat(1, T, 1)
        p = self.phys_proj(phys7_seq.permute(0, 2, 1))  # (B, T, 128)
        t = self.time_mlp(time_mat[..., None])  # (B, T, 64)
        x = torch.cat([s, p, t], dim=-1)  # (B, T, 320)
        x = self.in_proj(x)
        pos = self.pos.to(x.device)
        x = x + pos[:, :T, :]
        h = self.encoder(x)
        y = self.out(h)  # (B, T, 6)
        return y.permute(0, 2, 1)  # (B, 6, T)

class MorphGRU(nn.Module):
    def __init__(self, static_dim: int, phys_dim: int = 7, hidden=256, num_layers=1, out_dim: int = 6):
        super().__init__()
        self.static_enc = StaticEncoder(static_dim, out_dim=128)
        self.phys_proj = nn.Linear(phys_dim, 128)
        self.time_mlp = TimeMLP(out_dim=64)
        # 注意下面传入 nn.GRU 的参数也要对应修改
        self.gru = nn.GRU(input_size=128+128+64, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, static_x, phys7_seq, time_mat):
        B = static_x.shape[0]
        T = phys7_seq.shape[2]
        s = self.static_enc(static_x)
        s = s[:, None, :].repeat(1, T, 1)
        p = self.phys_proj(phys7_seq.permute(0, 2, 1))
        t = self.time_mlp(time_mat[..., None])
        x = torch.cat([s, p, t], dim=-1)
        h, _ = self.gru(x)
        y = self.out(h)  # (B,T,6)
        return y.permute(0, 2, 1)

class MorphMLP(nn.Module):
    def __init__(self, static_dim: int, phys_dim: int = 7, hidden=256, num_layers=3, out_dim: int = 6):
        super().__init__()
        self.static_enc = StaticEncoder(static_dim, out_dim=128)
        self.phys_proj = nn.Linear(phys_dim, 128)
        self.time_mlp = TimeMLP(out_dim=64)

        in_dim = 128 + 128 + 64
        ds = [in_dim] + [hidden]*(num_layers-1) + [out_dim]
        net=[]
        for i in range(len(ds)-1):
            net.append(nn.Linear(ds[i], ds[i+1]))
            if i < len(ds)-2:
                net.append(nn.ReLU())
        self.net = nn.Sequential(*net)

    def forward(self, static_x, phys7_seq, time_mat):
        B = static_x.shape[0]
        T = phys7_seq.shape[2]
        s = self.static_enc(static_x)
        s = s[:, None, :].repeat(1, T, 1)
        p = self.phys_proj(phys7_seq.permute(0, 2, 1))
        t = self.time_mlp(time_mat[..., None])
        x = torch.cat([s, p, t], dim=-1)
        y = self.net(x)  # (B,T,6)
        return y.permute(0,2,1)


class MorphLSTM(nn.Module):
    """R2-3b 消融：与 MorphGRU 完全同构，仅把 GRU 换成 LSTM。"""
    def __init__(self, static_dim: int, phys_dim: int = 7, hidden=256, num_layers=2, out_dim: int = 6):
        super().__init__()
        self.static_enc = StaticEncoder(static_dim, out_dim=128)
        self.phys_proj = nn.Linear(phys_dim, 128)
        self.time_mlp = TimeMLP(out_dim=64)
        self.lstm = nn.LSTM(input_size=128+128+64, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, static_x, phys7_seq, time_mat):
        B = static_x.shape[0]
        T = phys7_seq.shape[2]
        s = self.static_enc(static_x)
        s = s[:, None, :].repeat(1, T, 1)
        p = self.phys_proj(phys7_seq.permute(0, 2, 1))
        t = self.time_mlp(time_mat[..., None])
        x = torch.cat([s, p, t], dim=-1)
        h, _ = self.lstm(x)
        y = self.out(h)  # (B,T,6)
        return y.permute(0, 2, 1)

class _CausalConv1d(nn.Module):
    """左 padding 的因果一维卷积：输出第 i 步只依赖输入的 <=i 步。"""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.pad_left = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):  # (B, C, T)
        x = torch.nn.functional.pad(x, (self.pad_left, 0))
        return self.conv(x)

class MorphTCN(nn.Module):
    """R2-3b 消融：时序卷积网络（因果 + 膨胀卷积，膨胀率 1,2,4,...）。"""
    def __init__(self, static_dim: int, phys_dim: int = 7, hidden=256, num_layers=4, kernel_size=3,
                 out_dim: int = 6):
        super().__init__()
        self.static_enc = StaticEncoder(static_dim, out_dim=128)
        self.phys_proj = nn.Linear(phys_dim, 128)
        self.time_mlp = TimeMLP(out_dim=64)
        self.in_proj = nn.Conv1d(128 + 128 + 64, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([
            _CausalConv1d(hidden, kernel_size=kernel_size, dilation=2 ** i)
            for i in range(num_layers)
        ])
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, static_x, phys7_seq, time_mat):
        B = static_x.shape[0]
        T = phys7_seq.shape[2]
        s = self.static_enc(static_x)
        s = s[:, None, :].repeat(1, T, 1)
        p = self.phys_proj(phys7_seq.permute(0, 2, 1))
        t = self.time_mlp(time_mat[..., None])
        x = torch.cat([s, p, t], dim=-1)  # (B,T,320)
        h = self.in_proj(x.permute(0, 2, 1))  # (B,hidden,T)
        for blk in self.blocks:
            h = h + torch.relu(blk(h))  # 残差连接
        y = self.out(h.permute(0, 2, 1))  # (B,T,6)
        return y.permute(0, 2, 1)

class MorphARMLP(nn.Module):
    """R2-3b 消融：自回归 MLP。

    第 1 个周期由 recipe(静态) + phys7 描述符 + cycle index 直接预测；
    之后逐周期把上一周期的预测值与当周期输入拼接，送入同一个 MLP 预测下一周期。

    注意：trainer 的 model(x, p7, t) 接口拿不到真值目标，因此 forward 内部
    只能用自身预测做自回归展开（无 teacher forcing）。梯度会沿整条展开链
    反传（类似 BPTT），计算图随周期数 T 线性增长——训练比 GRU/LSTM 慢、
    显存占用略高；推理与训练行为一致（都是自回归 rollout），无 exposure bias
    之外的 train/test 失配。
    """
    def __init__(self, static_dim: int, phys_dim: int = 7, hidden=256, num_layers=3, out_dim: int = 6):
        super().__init__()
        self.static_enc = StaticEncoder(static_dim, out_dim=128)
        self.phys_proj = nn.Linear(phys_dim, 128)
        self.time_mlp = TimeMLP(out_dim=64)
        self.out_dim = out_dim

        in_dim = 128 + 128 + 64 + out_dim  # 额外拼接上一周期预测
        ds = [in_dim] + [hidden]*(num_layers-1) + [out_dim]
        net=[]
        for i in range(len(ds)-1):
            net.append(nn.Linear(ds[i], ds[i+1]))
            if i < len(ds)-2:
                net.append(nn.ReLU())
        self.net = nn.Sequential(*net)

    def forward(self, static_x, phys7_seq, time_mat):
        B = static_x.shape[0]
        T = phys7_seq.shape[2]
        s = self.static_enc(static_x)  # (B,128)
        p = self.phys_proj(phys7_seq.permute(0, 2, 1))  # (B,T,128)
        t = self.time_mlp(time_mat[..., None])  # (B,T,64)
        prev = torch.zeros(B, self.out_dim, device=static_x.device, dtype=static_x.dtype)
        ys = []
        for i in range(T):
            x = torch.cat([s, p[:, i, :], t[:, i, :], prev], dim=-1)
            y = self.net(x)  # (B,6)
            ys.append(y)
            prev = y
        y = torch.stack(ys, dim=1)  # (B,T,6)
        return y.permute(0, 2, 1)


class CombinedPhys7MorphPredictor(nn.Module):
    def __init__(self,
                 morph_model: nn.Module,
                 stageA_provider: Optional[StageAEnsemblePhys7Provider],
                 T: int,
                 phys_source: str = "stageA_pred",
                 phys7_mode: str = "full",
                 # ✅ 新增：接收归一化参数
                 norm_phys7_mean: Optional[np.ndarray] = None,
                 norm_phys7_std: Optional[np.ndarray] = None):
        super().__init__()
        self.morph_model = morph_model
        self.stageA_provider = stageA_provider
        self.T = int(T)
        self.phys_source = phys_source
        self.phys7_mode = phys7_mode

        # ✅ 注册为 buffer，随模型保存/加载，但不参与梯度更新
        if norm_phys7_mean is None:
            norm_phys7_mean = np.zeros((1, 7), dtype=np.float32)
        if norm_phys7_std is None:
            norm_phys7_std = np.ones((1, 7), dtype=np.float32)

        self.register_buffer("p7_mean", torch.from_numpy(norm_phys7_mean).float())
        self.register_buffer("p7_std", torch.from_numpy(norm_phys7_std).float())

    def forward(self,
                static_x_norm: torch.Tensor,
                recipe_raw: torch.Tensor,
                time_mat: torch.Tensor
                ) -> torch.Tensor:
        # 1) 获取 Raw Phys7
        recipe_raw_np = recipe_raw.detach().cpu().numpy().astype(np.float32)
        phys7_seq_np = get_phys7_seq_for_batch(
            recipe_raw_np=recipe_raw_np,
            T=self.T,
            phys_source=self.phys_source,
            phys7_mode=self.phys7_mode,
            stageA_provider=self.stageA_provider
        )
        phys7_raw = torch.from_numpy(phys7_seq_np).to(static_x_norm.device)  # (B,7,T)

        # ✅ 2) 应用归一化 (Z-score)
        # phys7_raw: (B,7,T), p7_mean: (1,7), 需广播
        # 注意：p7_mean 是 (1,7)，phys7_raw 是 (B,7,T)，我们需要 (1,7,1)
        mean_b = self.p7_mean.view(1, 7, 1)
        std_b = self.p7_std.view(1, 7, 1)

        phys7_norm = (phys7_raw - mean_b) / (std_b + 1e-6)

        # 3) morph forward
        return self.morph_model(static_x_norm, phys7_norm, time_mat)
    @torch.no_grad()
    def forward_from_recipe_raw(self,
                               static_x_norm_np: np.ndarray,
                               recipe_raw_np: np.ndarray,
                               time_mat_np: np.ndarray,
                               device: str = "cuda") -> np.ndarray:
        """给 StageC/推理用：纯 numpy 输入，输出 numpy"""
        dev = get_default_device(device)
        self.to(dev).eval()

        static_x_norm = torch.from_numpy(static_x_norm_np.astype(np.float32)).to(dev)
        recipe_raw = torch.from_numpy(recipe_raw_np.astype(np.float32)).to(dev)
        time_mat = torch.from_numpy(time_mat_np.astype(np.float32)).to(dev)

        y = self.forward(static_x_norm, recipe_raw, time_mat)
        return y.detach().cpu().numpy().astype(np.float32)

    def freeze_stageA(self, freeze: bool = True):
        """StageC finetune 时常用：冻结/解冻 StageA provider（一般冻结）"""
        # provider 不是 nn.Module 的 parameters（我们存的是 models list），所以手动冻结
        if self.stageA_provider is None:
            return
        for m in self.stageA_provider.models:
            for p in m.parameters():
                p.requires_grad = (not freeze)
def save_stageB_morph_ckpt(path: str,
                           morph_model: nn.Module,
                           meta: Dict[str, Any],
                           stageA_heads_root: Optional[str] = None):

    payload = {
        "model": morph_model.state_dict(),
        "meta": meta,
        "stageA_heads_root": stageA_heads_root,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
def load_stageB_combined_ckpt(stageB_ckpt_path: str,
                              build_morph_model_fn,
                              stageA_provider: Optional[StageAEnsemblePhys7Provider],
                              device: str = "cuda") -> Tuple[CombinedPhys7MorphPredictor, Dict[str, Any]]:

    ck = _torch_load_ckpt(stageB_ckpt_path, map_location="cpu")
    meta = ck.get("meta", {}) or {}
    sd = ck.get("model", ck.get("state_dict", ck))
    if not isinstance(sd, dict):
        raise RuntimeError("Invalid StageB ckpt: missing model state dict")

    morph_model = build_morph_model_fn(meta)
    morph_model.load_state_dict(_strip_state_dict_prefix(sd), strict=False)

    T = int(meta.get("T", meta.get("time_steps", 10)) or 10)
    phys_source = str(meta.get("phys_source", "stageA_pred"))
    phys7_mode = str(meta.get("phys7_mode", "full"))

    # ✅ 从 meta 读取归一化参数
    norm_p7 = meta.get("norm_phys7", {})
    p7_mean = np.array(norm_p7.get("mean", np.zeros((1,7))), dtype=np.float32)
    p7_std  = np.array(norm_p7.get("std",  np.ones((1,7))),  dtype=np.float32)

    combined = CombinedPhys7MorphPredictor(
        morph_model=morph_model,
        stageA_provider=stageA_provider,
        T=T,
        phys_source=phys_source,
        phys7_mode=phys7_mode,
        norm_phys7_mean=p7_mean, # ✅ 传入
        norm_phys7_std=p7_std    # ✅ 传入
    )
    combined.to(get_default_device(device)).eval()
    return combined, meta
def _mask_coverage(mask: np.ndarray) -> float:
    if mask is None:
        return 0.0
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return 0.0
    return float(m.mean())
def _infer_recipe_key(col: str) -> Optional[str]:
    cc = _canon(col)
    cc2 = re.sub(r"\(.*?\)", "", cc)

    for k, aliases in RECIPE_KEY_ALIAS.items():
        for pat in aliases:
            if pat in cc2:
                return k
    return None

def _default_family_sign_and_nonneg(families):
    sign = {}
    for f in families:
        sign[f] = -1.0 if f.lower() == "zmin" else 1.0
    nonneg = set([f for f in families])
    return sign, nonneg
def make_hp_tag(hp: Dict[str, Any]) -> str:
    def f(x):
        if isinstance(x, float):
            return f"{x:.3g}".replace(".", "p")
        return str(x)
    parts = [
        f"lr{f(hp.get('lr'))}",
        f"wd{f(hp.get('weight_decay'))}",
        f"do{f(hp.get('tf_dropout'))}",
        f"dm{hp.get('tf_d_model')}",
        f"L{hp.get('tf_layers')}",
        f"hb{f(hp.get('huber_beta'))}",
    ]
    return "_".join(parts)

def export_loss_curve(
    out_dir: str,
    exp_name: str,
    train_losses: List[float],
    val_losses: List[float],
    test_losses_by_epoch: Optional[Dict[int, float]] = None,
    best_epoch: int = None,
):
    _ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, f"loss_curve_{exp_name}.csv")
    png_path = os.path.join(out_dir, f"loss_curve_{exp_name}.png")

    epochs = list(range(1, len(train_losses) + 1))
    test_losses_by_epoch = test_losses_by_epoch or {}

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "test_loss", "best_epoch"])
        for ep, tr, va in zip(epochs, train_losses, val_losses):
            tl = test_losses_by_epoch.get(ep, "")
            w.writerow([ep, tr, va, tl, best_epoch])

    # PNG
    try:
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.plot(epochs, train_losses, label="train")
        ax.plot(epochs, val_losses, label="val")

        # test 采样点
        if len(test_losses_by_epoch) > 0:
            xs = sorted(test_losses_by_epoch.keys())
            ys = [float(test_losses_by_epoch[x]) for x in xs]
            ax.plot(xs, ys, marker="o", linestyle="--", label="test(sampled)")

        if best_epoch is not None:
            ax.axvline(int(best_epoch), linestyle=":", label="best_epoch")

        ax.set_xlabel("epoch")
        ax.set_ylabel("masked_mse (norm space)")
        ax.set_title(f"Loss curve — {exp_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        log(f"      [WARN] plot loss curve failed: {e}")
def export_scatter_per_family(out_dir: str,
                              exp_name: str,
                              split_name: str,
                              pred_denorm_um: np.ndarray,
                              y_denorm_um: np.ndarray,
                              m: np.ndarray,
                              families: List[str],
                              sample_ids=None,
                              clip_nonneg: bool = True):
    _ensure_dir(out_dir)

    # —— 关键：先转“论文口径” —— #
    pred_um = _apply_display_transform_um(pred_denorm_um, families, clip_nonneg=clip_nonneg)
    y_um    = _apply_display_transform_um(y_denorm_um,   families, clip_nonneg=clip_nonneg)

    pred_nm = pred_um * 1e3
    y_nm = y_um * 1e3

    N, K, T = pred_nm.shape
    assert K == len(families), f"K mismatch: {K} vs families={len(families)}"

    for ki, fam in enumerate(families):
        for ti, tval in enumerate(TIME_LIST):
            try:
                mask = m[:, ki, ti]
                yt = y_nm[mask, ki, ti]
                yp = pred_nm[mask, ki, ti]
                ok = np.isfinite(yt) & np.isfinite(yp)
                yt = yt[ok]
                yp = yp[ok]
                if yt.size == 0:
                    continue

                r2 = masked_r2_score_np(yt, yp)
                mae = float(np.mean(np.abs(yp - yt)))

                fig = plt.figure(figsize=(4.5, 4.5))
                ax = fig.add_subplot(111)
                ax.scatter(yt, yp, s=8)
                mn = float(min(yt.min(), yp.min()))
                mx = float(max(yt.max(), yp.max()))
                ax.plot([mn, mx], [mn, mx], linestyle="--")
                ax.set_xlabel("y_true (nm)")
                ax.set_ylabel("y_pred (nm)")
                ax.set_title(f"{exp_name} | {split_name} | {fam} @t={tval}\nR2={r2:.4f}, MAE={mae:.2f} nm")
                fig.tight_layout()

                png_path = os.path.join(out_dir, f"scatter_{exp_name}_{split_name}_{fam}_t{tval}.png")
                fig.savefig(png_path, dpi=160)
                plt.close(fig)
            except Exception as e:
                log(f"      [WARN] plot scatter failed fam={fam} t={tval}: {e}")


def export_experiment(
        out_dir: str,
        pack: Dict,
        meta: Dict,
        exp_name: str,
        split_name: str = "test",
        sample_ids=None,
        make_plots: bool = True,
        clip_nonneg: bool = True,
):
    _ensure_dir(out_dir)
    meta = meta or {}
    families = meta.get("families", None)
    if (not families) or (not isinstance(families, list)):
        if "pred_denorm_um" in pack:
            K = int(pack["pred_denorm_um"].shape[1])
        elif "pred_um" in pack:
            K = int(pack["pred_um"].shape[1])
        else:
            K = int(pack["pred_norm"].shape[1])

        m = re.search(r"_fam-([^_]+)_seed", str(exp_name))
        if (K == 1) and m and (m.group(1) != "multi"):
            families = [m.group(1)]
        else:
            families = list(FAMILIES[:K])

        meta["families"] = families
        meta["K"] = int(len(families))

    npz_path = os.path.join(out_dir, f"eval_pack_{exp_name}_{split_name}.npz")

    # ---- 核心：优先使用 pack 中已有的 denorm ----
    if ("pred_denorm_um" in pack) and ("y_denorm_um" in pack):
        pred_denorm_um = np.asarray(pack["pred_denorm_um"], dtype=np.float32)
        y_denorm_um = np.asarray(pack["y_denorm_um"], dtype=np.float32)
    else:
        nt = meta.get("norm_target", None)
        if not nt:
            # 兼容：如果 meta 没存 norm_target，尝试直接用
            pred_denorm_um = np.asarray(pack["pred_norm"], dtype=np.float32)
            y_denorm_um = np.asarray(pack["y_norm"], dtype=np.float32)
        else:
            y_mean = float(nt["mean"][0])
            y_std = float(nt["std"][0])
            pred_denorm_um = np.asarray(pack["pred_norm"], dtype=np.float32) * y_std + y_mean
            y_denorm_um = np.asarray(pack["y_norm"], dtype=np.float32) * y_std + y_mean

    pred_um = _apply_display_transform_um(pred_denorm_um, families, clip_nonneg=clip_nonneg)
    y_um = _apply_display_transform_um(y_denorm_um, families, clip_nonneg=clip_nonneg)

    # meta stringify
    try:
        meta_str = json.dumps(meta, ensure_ascii=False, cls=NumpyEncoder)
    except Exception:
        meta_str = "{}"

    # 剔除大对象防重复
    reserved = {"meta", "pred_denorm_um", "y_denorm_um", "pred_um", "y_um"}
    pack_to_save = {k: pack[k] for k in pack.keys() if k not in reserved}

    # 强制清理
    for k in ["pred_denorm_um", "y_denorm_um", "pred_um", "y_um", "meta"]:
        pack_to_save.pop(k, None)

    np.savez_compressed(
        npz_path,
        **pack_to_save,
        pred_denorm_um=pred_denorm_um,
        y_denorm_um=y_denorm_um,
        pred_um=pred_um,
        y_um=y_um,
        meta=meta_str
    )

    # 指标计算
    met = compute_mae_nm(pred_denorm_um, y_denorm_um, pack["mask"], families=families)

    js_path = os.path.join(out_dir, f"metrics_{exp_name}_{split_name}.json")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(met, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    if make_plots:
        export_scatter_per_family(
            out_dir=out_dir,
            exp_name=exp_name,
            split_name=split_name,
            pred_denorm_um=pred_denorm_um,
            y_denorm_um=y_denorm_um,
            m=pack["mask"],
            families=families,
            sample_ids=sample_ids,
            clip_nonneg=clip_nonneg,
        )

    # ✅ 【关键修复】必须返回 met，否则上层拿不到 R2
    return met
def _pick_divisor(d: int, prefers=(8, 4, 2, 1)) -> int:
    for h in prefers:
        if d % h == 0:
            return h
    return 1
def _infer_int_from_sd(sd: Dict[str, torch.Tensor], key: str, default: int) -> int:
    try:
        t = sd.get(key, None)
        if t is None:
            return default
        return int(t.shape[0])
    except Exception:
        return default

# =========================================================
#  新增：从 stageB_train 迁移过来的配置加载工具 (供 StageC 使用)
# =========================================================

def _find_candidates(runs_root: str, relname: str) -> List[str]:
    cands = [
        os.path.join(runs_root, relname),
        os.path.join(runs_root, "_tuneV_verify", relname),
    ]
    return [p for p in cands if os.path.exists(p)]

def load_best_config_common(runs_root: str) -> Dict[str, Any]:
    """读取 StageB 产生的最佳配置文件"""
    cands = _find_candidates(runs_root, "best_config_common_all_families.json")
    if not cands:
        raise FileNotFoundError(f"best_config_common_all_families.json not found under: {runs_root}")
    with open(cands[0], "r", encoding="utf-8") as f:
        return json.load(f)

def load_results_summary_df(runs_root: str) -> Optional[pd.DataFrame]:
    """读取 StageB 的汇总 CSV"""
    cands = _find_candidates(runs_root, "results_summary.csv")
    if not cands:
        return None
    try:
        return pd.read_csv(cands[0])
    except Exception:
        return None


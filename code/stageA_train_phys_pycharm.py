# -*- coding: utf-8 -*-
from __future__ import annotations  # PEP 563: allow "X | None" annotations on Python 3.9
"""
StageA (Phys7 from IEDF) — recipe(7) -> phys7(7x1)

目标：
- 用 IEDF 提取的 Phys7 作为 StageA 的“物理信息”标签
- 输入仍是原来的 7 维 recipe: [APC, source_RF, LF_RF, SF6, C4F8, DEP time, etch time]
- 输出是 7 个物理降维特征（T=1），与 case.xlsx 按 case_id 一一对应
- 保留 Transformer/GRU/MLP baseline，对比试验和论文图表导出（parity/residual/heatmap/summary/pred_table）

注意：
- 依赖 extract_phys7_from_iedf.py（你已经有）
- 需要 phys_model.py 支持 out_dim 参数（见下面 patch）
"""

import os
import re
import csv
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset, Subset

import matplotlib.pyplot as plt

from physio_util import (
    set_seed,
    get_default_device,
    metrics,
    export_predictions_longtable,
    export_metrics_grid,
    write_summary_txt,
    heatmap,
    parity_scatter,
    residual_hist,
)

from phys_model import PhysicsSeqPredictor, PhysicsMLPBaseline, PhysicsGRUBaseline
import extract_phys7_from_iedf as iedf


# -------------------- Phys7 定义（固定 7 维） --------------------
FAMILIES = [
    "logGamma_SF6_tot",  # log10(总离子通量), SF6/sheath2 聚合
    "pF_SF6",            # F+ 占比
    "spread_SF6",        # (E90-E10)/E50
    "qskew_SF6",         # (E90+E10-2E50)/(E90-E10)
    "logGamma_C4F8_tot", # log10(总离子通量), C4F8/sheath1 聚合
    "rho_C4F8",          # log10( Gamma(CF3+)/Gamma(C2F3+) )
    "spread_C4F8",       # (E90-E10)/E50
]

# -------------------- 路径默认值（相对于本脚本所在目录） --------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

# -------------------- 配置 --------------------
class Cfg:
    # 数据
    case_excel = os.path.join(_HERE, "case_with_phys7.xlsx")
    sheet_name = "Sheet1"
    case_id_col = "input"   # 你表里是 input: cas1/cas2...
    iedf_root = os.path.join(_HERE, "iedf_data")  # 原始 IEDF 树（如未提供则留空/占位）

    # 输出目录
    save_dir = os.path.join(_HERE, "runs_stageA_phys7")
    seed = 42
    batch = 64
    max_epochs = 200
    val_ratio = 0.1
    test_ratio = 0.1  # ✅新增：test 占比（导出 predictions_test.csv 需要）
    split_tag = "case_random"  # ✅新增：写进 metrics_summary / model_comparison 的标识
    do_compare = True  # ✅True: 跑对比实验；False: 只跑单模型
    compare_split_seed = 2026  # ✅固定 split（关键：公平对比）
    train_seeds = [0, 1, 2, 3, 4]  # ✅多次训练随机性（稳定性）
    # 模型
    model_type = "transformer"  # 默认主模型 tf；可选 "gru","mlp"
    d_model = 64
    nhead = 4
    num_layers = 2
    dim_ff = 128
    dropout = 0.1

    # baseline 超参
    mlp_hidden = 128
    mlp_layers = 3
    gru_hidden = 128
    gru_layers = 1

    # 优化
    lr = 1e-3
    weight_decay = 1e-3
    clip_grad_norm = 1.0
    warmup_epochs = 10
    use_cosine = True

    # 输出归一化（只影响 loss，不影响导出物理量）
    use_output_norm = True

    # CV / baseline 对比
    cv_model_types = ["transformer", "mlp", "gru"]
    multi_split_seeds = [0, 1, 2, 3, 4]
    # cv_model_types = ["transformer"]
    # multi_split_seeds = [0, 1, 2, 3, 4]
# -------------------- 小工具 --------------------
def _canon(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("（", "(").replace("）", ")")
    return s


def _pick_one(cols, candidates):
    cc = {c: _canon(c) for c in cols}
    for c in cols:
        v = cc[c]
        for pat in candidates:
            if pat in v:
                return c
    return None


def _norm_case_id(cid: str) -> str:
    cid = str(cid).strip()
    if re.fullmatch(r"\d+", cid):
        return f"cas{cid}"
    m = re.fullmatch(r"(?i)case(\d+)", cid)
    if m:
        return f"cas{m.group(1)}"
    return cid


def make_warmup_cosine(optimizer, total_epochs, warmup_epochs, use_cosine=True):
    if not use_cosine:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _compute_y_norm_stats(y: torch.Tensor, mask: torch.Tensor):
    """
    y:   (N, K, T) 这里 T=1
    mask:(N, K, T)
    返回 per-channel mean/std: (K,)
    """
    N, K, T = y.shape
    mean = np.zeros(K, np.float32)
    std = np.ones(K, np.float32)
    y_np = y.detach().cpu().numpy()
    m_np = mask.detach().cpu().numpy().astype(bool)

    for k in range(K):
        vals = y_np[:, k, :][m_np[:, k, :]]
        if vals.size == 0:
            mean[k] = 0.0
            std[k] = 1.0
        else:
            mean[k] = float(vals.mean())
            std[k] = float(vals.std() + 1e-6)
    return torch.from_numpy(mean), torch.from_numpy(std)


# -------------------- 1) 数据集构建：case.xlsx + IEDF -> Phys7 --------------------
def excel_to_phys7_dataset(excel_path: str, sheet_name: str, case_id_col: str, iedf_root: str):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    cols = list(df.columns)

    # 7 维 recipe 输入列匹配（沿用你旧逻辑口味）
    key_alias = {
        "apc": ["apc"],
        "source_rf": ["source_rf", "sourcerf", "rfsource"],
        "lf_rf": ["lf_rf", "lfrf", "bias"],
        "sf6": ["sf6"],
        "c4f8": ["c4f8"],
        "dep_time": ["deptime", "dep_time", "depositiontime"],
        "etch_time": ["etchtime", "etch_time"],
    }
    recipe_cols = [
        _pick_one(cols, key_alias["apc"]),
        _pick_one(cols, key_alias["source_rf"]),
        _pick_one(cols, key_alias["lf_rf"]),
        _pick_one(cols, key_alias["sf6"]),
        _pick_one(cols, key_alias["c4f8"]),
        _pick_one(cols, key_alias["dep_time"]),
        _pick_one(cols, key_alias["etch_time"]),
    ]
    # ✅ 防止重复选列（例如 APC 被选两次）
    if len(set(recipe_cols)) != len(recipe_cols):
        raise KeyError(
            f"recipe7 选列出现重复：{recipe_cols}\n"
            f"请检查 key_alias 是否过宽导致误匹配。"
        )

    if not all(recipe_cols):
        raise KeyError(f"recipe7 列没找全：{recipe_cols}\n现有列：{cols}")

    if case_id_col not in cols:
        raise KeyError(f"case_id_col={case_id_col} 不在表头里。现有列：{cols}")

    # 输入 X：标准化
    X = df[recipe_cols].to_numpy(np.float32)  # (N,7)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mean) / std

    # 输出 Y：Phys7, T=1
    N = len(df)
    K = len(FAMILIES)
    T = 1
    Y = np.full((N, K, T), np.nan, np.float32)

    # ✅ 优先尝试从 excel 中读取现成的 Phys7 描述符列（无需原始 IEDF）
    phys7_cols_available = [c for c in FAMILIES if c in cols]
    if len(phys7_cols_available) == K:
        # 表内已有全部 7 个描述符，直接取用
        phys7_arr = df[FAMILIES].to_numpy(np.float32)  # (N, K)
        Y[:, :, 0] = phys7_arr
    else:
        # 临时改 extract_phys7_from_iedf 的根目录
        old_root = iedf.IEDF_ROOT
        iedf.IEDF_ROOT = iedf_root

        for i in range(N):
            cid = _norm_case_id(df.loc[i, case_id_col])
            files = iedf.read_target_iedf_for_case(cid)

            feat = {k: np.nan for k in FAMILIES}

            # SF6/sheath2
            key_sf6 = ("SF6", "sheath2")
            if key_sf6 in files:
                out = iedf.compute_phys7_from_file(files[key_sf6], "SF6", "sheath2", iedf.TARGETS[key_sf6])
                if out is not None:
                    Gtot = out["Gamma_tot"]
                    if np.isfinite(Gtot):
                        feat["logGamma_SF6_tot"] = float(np.log10(Gtot + iedf.EPS))
                    GF = out["gammas"].get("F_1p", np.nan)
                    if np.isfinite(GF) and np.isfinite(Gtot):
                        feat["pF_SF6"] = float(GF / (Gtot + iedf.EPS))
                    feat["spread_SF6"] = out["spread"]
                    feat["qskew_SF6"] = out["qskew"]

            # C4F8/sheath1（你说 C4F8 没 sheath2，这里严格按 sheath1）
            key_c4 = ("C4F8", "sheath1")
            if key_c4 in files:
                out = iedf.compute_phys7_from_file(files[key_c4], "C4F8", "sheath1", iedf.TARGETS[key_c4])
                if out is not None:
                    Gtot = out["Gamma_tot"]
                    if np.isfinite(Gtot):
                        feat["logGamma_C4F8_tot"] = float(np.log10(Gtot + iedf.EPS))
                    G1 = out["gammas"].get("CF3_1p", np.nan)
                    G2 = out["gammas"].get("C2F3_1p", np.nan)
                    if np.isfinite(G1) and np.isfinite(G2):
                        feat["rho_C4F8"] = float(np.log10((G1 + iedf.EPS) / (G2 + iedf.EPS)))
                    feat["spread_C4F8"] = out["spread"]

            Y[i, :, 0] = np.array([feat[k] for k in FAMILIES], np.float32)

        iedf.IEDF_ROOT = old_root

    mask = np.isfinite(Y)
    Y[np.isnan(Y)] = 0.0

    # time_values / time_mat
    time_values = np.array([1.0], np.float32)  # (T,)
    time_mat = np.tile(time_values[None, :], (N, 1))  # (N,T)

    ds = TensorDataset(
        torch.from_numpy(Xn.astype(np.float32)),          # (N,7)
        torch.from_numpy(Y.astype(np.float32)),           # (N,K,T)
        torch.from_numpy(mask.astype(np.bool_)),          # (N,K,T)
        torch.from_numpy(time_mat.astype(np.float32)),    # (N,T)
    )
    meta = {
        "T": 1,
        "time_values": time_values,
        "families": list(FAMILIES),
        "recipe_cols": recipe_cols,
        "norm_static": {
            "mean": torch.from_numpy(mean.astype(np.float32)),
            "std": torch.from_numpy(std.astype(np.float32)),
        }
    }
    # 原始输入（未归一化）
    X_raw = X.astype(np.float32)  # (N,7)
    sample_id = [_norm_case_id(df.loc[i, case_id_col]) for i in range(N)]

    # 可选：如果你表里有配方名列，也可带上 recipe_id（否则就 None）
    recipe_id_col = _pick_one(cols, ["配方名", "recipe", "recipe_id", "recipename"])
    recipe_id = df[recipe_id_col].astype(str).tolist() if recipe_id_col else [None] * N

    aux = {
        "sample_id": sample_id,
        "recipe_id": recipe_id,
        "recipe_id_col": recipe_id_col,
        "X_raw": X_raw,     # 未归一化输入
        "Xn": Xn.astype(np.float32),
    }
    return ds, meta, aux
# -------------------- 2) 模型构建（tf/gru/mlp） --------------------
def build_stageA_model(T: int, out_dim: int, model_type: str):
    mt = str(model_type).lower()
    if mt in ["transformer", "tfm", "tf", "trans"]:
        return PhysicsSeqPredictor(
            d_model=Cfg.d_model,
            nhead=Cfg.nhead,
            num_layers=Cfg.num_layers,
            dim_ff=Cfg.dim_ff,
            dropout=Cfg.dropout,
            T=T,
            out_dim=out_dim,
        )
    if mt in ["mlp", "ffn"]:
        return PhysicsMLPBaseline(
            hidden_dim=Cfg.mlp_hidden,
            num_layers=Cfg.mlp_layers,
            T=T,
            out_dim=out_dim,
        )
    if mt in ["gru", "rnn"]:
        return PhysicsGRUBaseline(
            hidden_dim=Cfg.gru_hidden,
            num_layers=Cfg.gru_layers,
            T=T,
            out_dim=out_dim,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def slice_dataset_one_head(dataset: TensorDataset, k: int) -> TensorDataset:
    Xn, Y, M, tvals = dataset.tensors
    Yk = Y[:, k:k+1, :]        # (N,1,T)
    Mk = M[:, k:k+1, :]        # (N,1,T)
    return TensorDataset(Xn, Yk, Mk, tvals)
def make_shared_split_indices_3way(N: int, val_ratio: float, test_ratio: float, split_seed: int):
    """
    返回 train/val/test 三段 indices，所有模型/所有 head 必须复用同一份。
    """
    assert 0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1
    nval = max(1, int(N * val_ratio))
    ntest = max(1, int(N * test_ratio))

    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(N, generator=g).tolist()

    val_idx = perm[:nval]
    test_idx = perm[nval:nval + ntest]
    train_idx = perm[nval + ntest:]
    return train_idx, val_idx, test_idx


def save_split_indices(out_root: str, train_idx, val_idx, test_idx):
    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, "split_indices.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f, indent=2, ensure_ascii=False)
    return path

def train_stageA_phys7_singleheads(
    dataset, meta,
    model_type: str,
    out_root: str,
    split_seed: int,
    train_seed: int | None = None,
):
    """
    split_seed: 只用于生成 train/val/test 划分（所有模型要一致就固定它）
    train_seed: 只用于训练随机性（权重初始化、shuffle等）；用于稳定性对比
    """
    os.makedirs(out_root, exist_ok=True)

    # ✅训练随机性：用 train_seed；如果没给就退化为 split_seed
    if train_seed is None:
        train_seed = split_seed
    set_seed(train_seed)

    dev = get_default_device()

    T = int(meta["T"])
    families = list(meta["families"])
    K = len(families)
    N = len(dataset)

    # ✅split 固定：永远用 split_seed（不要用 train_seed）
    tr_idx, va_idx, te_idx = make_shared_split_indices_3way(
        N, Cfg.val_ratio, Cfg.test_ratio, split_seed
    )
    save_split_indices(out_root, tr_idx, va_idx, te_idx)

    head_rows = []
    r2_list = []

    for k, fam in enumerate(families):
        head_dir = os.path.join(out_root, f"head_{k:02d}_{fam}")
        os.makedirs(head_dir, exist_ok=True)

        ds_k = slice_dataset_one_head(dataset, k)
        tr_set = Subset(ds_k, tr_idx)
        va_set = Subset(ds_k, va_idx)

        tr = DataLoader(tr_set, batch_size=Cfg.batch, shuffle=True)
        va = DataLoader(va_set, batch_size=Cfg.batch, shuffle=False)

        model = build_stageA_model(T=T, out_dim=1, model_type=model_type).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=Cfg.lr, weight_decay=Cfg.weight_decay)
        sch = make_warmup_cosine(opt, Cfg.max_epochs, Cfg.warmup_epochs, use_cosine=Cfg.use_cosine)
        loss_fn = nn.SmoothL1Loss(reduction="none")

        # （可选）仅用于 loss 的输出归一化（不改变导出的物理量）
        y_mean_t = y_std_t = None
        if getattr(Cfg, "use_output_norm", False):
            _, Y_all, M_all, _ = ds_k.tensors
            Y_tr = Y_all[tr_idx]
            M_tr = M_all[tr_idx]
            y_mean, y_std = _compute_y_norm_stats(Y_tr, M_tr)  # (1,)
            y_mean_t = y_mean.to(dev).view(1, 1, 1)
            y_std_t  = y_std.to(dev).view(1, 1, 1)
            with open(os.path.join(head_dir, "y_norm.json"), "w", encoding="utf-8") as f:
                json.dump({"mean": y_mean.cpu().tolist(), "std": y_std.cpu().tolist()}, f, indent=2, ensure_ascii=False)

        best_r2 = -1e18
        best_path = os.path.join(head_dir, "phys7_best.pth")

        for e in range(1, Cfg.max_epochs + 1):
            # ---- train ----
            model.train()
            for Xn, Y, M, tvals in tr:
                Xn, Y, M, tvals = Xn.to(dev), Y.to(dev), M.to(dev), tvals.to(dev)
                pred = model(Xn, tvals)  # (B,1,1)

                if y_mean_t is not None:
                    pred_l = (pred - y_mean_t) / y_std_t
                    y_l    = (Y    - y_mean_t) / y_std_t
                else:
                    pred_l, y_l = pred, Y

                loss_e = loss_fn(pred_l, y_l)  # (B,1,1)
                w = M.float()
                loss = (loss_e * w).sum() / w.sum().clamp_min(1e-6)

                opt.zero_grad()
                loss.backward()
                if Cfg.clip_grad_norm and Cfg.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), Cfg.clip_grad_norm)
                opt.step()

            sch.step()

            # ---- val ----
            model.eval()
            preds, trues, masks = [], [], []
            with torch.no_grad():
                for Xn, Y, M, tvals in va:
                    Xn, tvals = Xn.to(dev), tvals.to(dev)
                    pred = model(Xn, tvals).detach().cpu()
                    preds.append(pred)
                    trues.append(Y.detach().cpu())
                    masks.append(M.detach().cpu())

            yhat = torch.cat(preds, 0)
            ytru = torch.cat(trues, 0)
            msk  = torch.cat(masks, 0)

            mts = metrics(yhat, ytru, msk)
            r2 = float(mts["R2"][0, 0]) if "R2" in mts else float("nan")

            print(f"[StageA-Phys7][{model_type}][{fam}][{e}/{Cfg.max_epochs}] val R2={r2:.4f}")

            if np.isfinite(r2) and r2 > best_r2:
                best_r2 = r2
                ckpt = {
                    "model": model.state_dict(),
                    "meta": meta,
                    "model_type": str(model_type),
                    "split_seed": int(split_seed),
                    "head_index": int(k),
                    "head_name": str(fam),
                    "out_dim": 1,
                }
                torch.save(ckpt, best_path)

        r2_list.append(best_r2)
        head_rows.append({"head_index": k, "family": fam, "best_val_r2": best_r2})

    # ---- 汇总：R2_min（按 val best）----
    r2_min = float(np.nanmin(np.array(r2_list, dtype=np.float32)))
    with open(os.path.join(out_root, "heads_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"per_head": head_rows, "r2_min": r2_min}, f, indent=2, ensure_ascii=False)

    pd.DataFrame(head_rows).to_csv(os.path.join(out_root, "heads_metrics.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(out_root, "r2_min.txt"), "w", encoding="utf-8") as f:
        f.write(f"{r2_min}\n")

    print(f"[StageA-Phys7] DONE. R2_min(val)={r2_min:.4f}")
    return r2_min
def infer_multihead_phys7(dataset, meta, heads_root: str, model_type: str):
    """
    heads_root = out_root（里面有 head_00_xxx/phys7_best.pth ...）
    返回 yhat/ytrue/mask: (N,K,1)
    """
    dev = get_default_device()
    loader = DataLoader(dataset, batch_size=Cfg.batch, shuffle=False)

    T = int(meta["T"])
    families = list(meta["families"])
    K = len(families)

    # 先拿 true/mask（一次就够）
    trues, masks = [], []
    with torch.no_grad():
        for _, Y, M, _ in loader:
            trues.append(Y)
            masks.append(M)
    ytrue = torch.cat(trues, 0)  # (N,K,1)
    mask = torch.cat(masks, 0)   # (N,K,1)

    # 逐 head 推理并拼接
    yhat_all = torch.zeros_like(ytrue, dtype=torch.float32)
    for k, fam in enumerate(families):
        head_dir = os.path.join(heads_root, f"head_{k:02d}_{fam}")
        ckpt_path = os.path.join(head_dir, "phys7_best.pth")
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

        mt = ckpt.get("model_type", model_type)
        model = build_stageA_model(T=T, out_dim=1, model_type=mt).to(dev)
        model.load_state_dict(ckpt["model"])
        model.eval()

        preds = []
        with torch.no_grad():
            for Xn, _, _, tvals in loader:
                Xn = Xn.to(dev)
                tvals = tvals.to(dev)
                pred = model(Xn, tvals).detach().cpu()  # (B,1,1)
                preds.append(pred)
        yhat_k = torch.cat(preds, 0)  # (N,1,1)
        yhat_all[:, k:k+1, :] = yhat_k

    return yhat_all, ytrue, mask
def _build_predictions_df(aux, meta, yhat: torch.Tensor, ytrue: torch.Tensor, mask: torch.Tensor, split_name: str, idxs):
    """
    每行一个样本；列包含 sample_id/recipe_id/输入参数/每个输出的 y,yhat,mask,err,abs_err
    """
    families = list(meta["families"])
    recipe_cols = list(meta["recipe_cols"])

    idxs = list(idxs)
    sid = [aux["sample_id"][i] for i in idxs]
    rid = [aux["recipe_id"][i] for i in idxs]

    X_raw = aux["X_raw"][idxs]  # (n,7)

    y = ytrue[idxs, :, 0].numpy()
    yh = yhat[idxs, :, 0].numpy()
    m = mask[idxs, :, 0].numpy().astype(np.int32)

    df = pd.DataFrame({
        "sample_id": sid,
        "recipe_id": rid,
        "split": split_name,
    })

    # 输入列
    for j, c in enumerate(recipe_cols):
        df[c] = X_raw[:, j]

    # 输出列（每个 family 一组）
    for k, fam in enumerate(families):
        df[f"y_{fam}"] = y[:, k]
        df[f"yhat_{fam}"] = yh[:, k]
        df[f"mask_{fam}"] = m[:, k]
        df[f"err_{fam}"] = (yh[:, k] - y[:, k])
        df[f"abs_err_{fam}"] = np.abs(yh[:, k] - y[:, k])

    return df


def export_predictions_csv_splits(out_root: str, aux, meta, yhat, ytrue, mask, split_indices: dict):
    """
    输出：
      predictions_train.csv / predictions_val.csv / predictions_test.csv / predictions_all.csv
    """
    os.makedirs(out_root, exist_ok=True)

    df_tr = _build_predictions_df(aux, meta, yhat, ytrue, mask, "train", split_indices["train"])
    df_va = _build_predictions_df(aux, meta, yhat, ytrue, mask, "val", split_indices["val"])
    df_te = _build_predictions_df(aux, meta, yhat, ytrue, mask, "test", split_indices["test"])
    df_all = pd.concat([df_tr, df_va, df_te], axis=0, ignore_index=True)

    df_tr.to_csv(os.path.join(out_root, "predictions_train.csv"), index=False, encoding="utf-8-sig")
    df_va.to_csv(os.path.join(out_root, "predictions_val.csv"), index=False, encoding="utf-8-sig")
    df_te.to_csv(os.path.join(out_root, "predictions_test.csv"), index=False, encoding="utf-8-sig")
    df_all.to_csv(os.path.join(out_root, "predictions_all.csv"), index=False, encoding="utf-8-sig")

    return df_tr, df_va, df_te, df_all
def _metrics_on_indices(yhat, ytrue, mask, idxs):
    idxs = torch.as_tensor(list(idxs), dtype=torch.long)
    return metrics(yhat[idxs], ytrue[idxs], mask[idxs])


def export_metrics_csv_splits(out_root: str, meta, yhat, ytrue, mask, split_indices: dict, model_name: str, seed: int, split_tag: str):
    families = list(meta["families"])
    os.makedirs(out_root, exist_ok=True)

    mts_tr = _metrics_on_indices(yhat, ytrue, mask, split_indices["train"])
    mts_va = _metrics_on_indices(yhat, ytrue, mask, split_indices["val"])
    mts_te = _metrics_on_indices(yhat, ytrue, mask, split_indices["test"])

    def _col(mts, key):
        if key not in mts:
            return [np.nan] * len(families)
        arr = np.asarray(mts[key])  # (K,1)
        return arr[:, 0].tolist()

    df = pd.DataFrame({
        "family": families,
        "R2_train": _col(mts_tr, "R2"),
        "MAE_train": _col(mts_tr, "MAE"),
        "RMSE_train": _col(mts_tr, "RMSE"),
        "R2_val": _col(mts_va, "R2"),
        "MAE_val": _col(mts_va, "MAE"),
        "RMSE_val": _col(mts_va, "RMSE"),
        "R2_test": _col(mts_te, "R2"),
        "MAE_test": _col(mts_te, "MAE"),
        "RMSE_test": _col(mts_te, "RMSE"),
    })
    df.to_csv(os.path.join(out_root, "metrics_per_output.csv"), index=False, encoding="utf-8-sig")

    # summary（用 test）
    r2_test = np.array(df["R2_test"], dtype=np.float32)
    mae_test = np.array(df["MAE_test"], dtype=np.float32)

    r2_mean = float(np.nanmean(r2_test))
    r2_min = float(np.nanmin(r2_test))
    worst_k = int(np.nanargmin(r2_test)) if np.isfinite(r2_test).any() else -1
    worst_family = families[worst_k] if worst_k >= 0 else ""

    summ = pd.DataFrame([{
        "model": model_name,
        "seed": int(seed),
        "split_tag": str(split_tag),
        "R2_mean_test": r2_mean,
        "R2_min_test": r2_min,
        "MAE_mean_test": float(np.nanmean(mae_test)),
        "worst_family_test": worst_family,
    }])
    summ.to_csv(os.path.join(out_root, "metrics_summary.csv"), index=False, encoding="utf-8-sig")

    return df, summ
def append_model_comparison_row(comp_csv_path: str, metrics_per_output_df: pd.DataFrame, metrics_summary_df: pd.DataFrame):
    """
    把 per_output 的 R2/MAE 展平成一行 + summary 的 R2_mean/R2_min
    """
    row = metrics_summary_df.iloc[0].to_dict()

    # 展平：R2_test_xxx / MAE_test_xxx
    for _, r in metrics_per_output_df.iterrows():
        fam = r["family"]
        row[f"R2_test_{fam}"] = float(r["R2_test"])
        row[f"MAE_test_{fam}"] = float(r["MAE_test"])

    df_row = pd.DataFrame([row])

    if os.path.exists(comp_csv_path):
        old = pd.read_csv(comp_csv_path)
        new = pd.concat([old, df_row], axis=0, ignore_index=True)
    else:
        new = df_row

    new.to_csv(comp_csv_path, index=False, encoding="utf-8-sig")
    return comp_csv_path
def build_stageA_report_single_run(dataset, meta, aux, out_root: str, model_name: str, seed: int, split_tag: str):
    # 1) 读取 split_indices.json
    with open(os.path.join(out_root, "split_indices.json"), "r", encoding="utf-8") as f:
        split_indices = json.load(f)

    # 2) multi-head 推理，得到 yhat/ytrue/mask
    yhat, ytrue, mask = infer_multihead_phys7(dataset, meta, heads_root=out_root, model_type=model_name)

    # 3) 导出 predictions CSV（train/val/test/all）
    export_dir = os.path.join(out_root, "exports")
    df_tr, df_va, df_te, df_all = export_predictions_csv_splits(export_dir, aux, meta, yhat, ytrue, mask, split_indices)

    # 4) 导出 metrics CSV（per_output + summary）
    df_m, df_s = export_metrics_csv_splits(export_dir, meta, yhat, ytrue, mask, split_indices,
                                           model_name=model_name, seed=seed, split_tag=split_tag)

    # 5) 追加到总对比表（放在 runs_stageA_phys7 根目录）
    comp_csv = os.path.join(Cfg.save_dir, "model_comparison.csv")
    append_model_comparison_row(comp_csv, df_m, df_s)

    return export_dir
import shutil
import pandas as pd
import os
import json

def select_best_run_from_model_comparison(save_dir: str, metric: str = "R2_min_test"):
    """
    从 runs_stageA_phys7/model_comparison.csv 中选 metric 最大的那一行。
    返回 (out_root, row_dict)
    """
    comp_csv = os.path.join(save_dir, "model_comparison.csv")
    if not os.path.exists(comp_csv):
        raise FileNotFoundError(f"model_comparison.csv not found: {comp_csv}")

    df = pd.read_csv(comp_csv)
    if metric not in df.columns:
        raise KeyError(f"metric '{metric}' not in model_comparison.csv columns")

    # 只看有数值的行
    df2 = df[pd.to_numeric(df[metric], errors="coerce").notna()].copy()
    if len(df2) == 0:
        raise RuntimeError(f"No valid rows for metric '{metric}' in {comp_csv}")

    best_i = df2[metric].astype(float).idxmax()
    best = df.loc[best_i].to_dict()

    # 你的 out_root 命名规则：bench_{model}_split{split}_train{seed}
    # split_seed 从 split_tag 里解析：例如 "case_random_split2026"
    split_tag = str(best.get("split_tag", ""))
    # 尝试从 split_tag 中提取最后的 split 数字
    split_seed = None
    for token in split_tag.split("_"):
        if token.startswith("split"):
            try:
                split_seed = int(token.replace("split", ""))
            except:
                pass
    if split_seed is None:
        # 兜底：如果 split_tag 没写 split，就用 compare_split_seed 或 Cfg.seed（这里先不依赖 Cfg）
        split_seed = 2026

    model = str(best.get("model", "unknown"))
    train_seed = int(best.get("seed", 0))

    out_root = os.path.join(save_dir, f"bench_{model}_split{split_seed}_train{train_seed}")
    if not os.path.exists(out_root):
        # 兜底：也可能是 single_... 命名，你可以自行扩展
        raise FileNotFoundError(f"Best out_root not found: {out_root}")

    return out_root, best


def materialize_best_run(best_out_root: str, dst_root: str):
    """
    把 best_out_root 的 head_*/ split_indices.json / exports 复制到 dst_root（固定路径）。
    这样 StageB/StageC 只需要 load dst_root。
    """
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.makedirs(dst_root, exist_ok=True)

    # 复制 split_indices.json
    src_split = os.path.join(best_out_root, "split_indices.json")
    if os.path.exists(src_split):
        shutil.copy2(src_split, os.path.join(dst_root, "split_indices.json"))

    # 复制 exports（可选，但建议保留，方便追溯）
    src_exports = os.path.join(best_out_root, "exports")
    if os.path.exists(src_exports):
        shutil.copytree(src_exports, os.path.join(dst_root, "exports"))

    # 复制 7 个 head ckpt 目录
    for name in os.listdir(best_out_root):
        if name.startswith("head_") and os.path.isdir(os.path.join(best_out_root, name)):
            shutil.copytree(os.path.join(best_out_root, name), os.path.join(dst_root, name))

    # 复制训练侧 headmetrics（可选）
    for fn in ["heads_metrics.csv", "heads_metrics.json", "r2_min.txt"]:
        p = os.path.join(best_out_root, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst_root, fn))

    return dst_root


def write_best_pointer(save_dir: str, best_out_root: str, best_row: dict, metric: str = "R2_min_test"):
    """
    写一个指针文件，记录 best_run 是谁、分数多少、源目录在哪。
    """
    ptr = {
        "metric": metric,
        "best_score": float(best_row.get(metric, float("nan"))),
        "best_out_root": best_out_root,
        "best_row": best_row,
    }
    path = os.path.join(save_dir, "best_run_by_test.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ptr, f, indent=2, ensure_ascii=False)
    return path

def run_stageA_comparison():
    os.makedirs(Cfg.save_dir, exist_ok=True)
    comp_csv = os.path.join(Cfg.save_dir, "model_comparison.csv")
    if os.path.exists(comp_csv):
        os.remove(comp_csv)

    # 1) dataset 只构建一次
    dataset, meta, aux = excel_to_phys7_dataset(
        Cfg.case_excel, Cfg.sheet_name, Cfg.case_id_col, Cfg.iedf_root
    )

    # 2) 固定 split_seed（你可以设成 2026 或 Cfg.seed）
    split_seed = getattr(Cfg, "compare_split_seed", Cfg.seed)

    # 3) 训练随机性 seeds（多次重复）
    train_seeds = getattr(Cfg, "train_seeds", getattr(Cfg, "multi_split_seeds", [Cfg.seed]))

    # 4) 循环模型与 seed
    for model_type in Cfg.cv_model_types:
        for train_seed in train_seeds:
            run_tag = f"bench_{model_type}_split{split_seed}_train{train_seed}"
            out_root = os.path.join(Cfg.save_dir, run_tag)

            # 4.1 训练 7 个 head（split 固定，train_seed 可变）
            _ = train_stageA_phys7_singleheads(
                dataset, meta,
                model_type=model_type,
                out_root=out_root,
                split_seed=split_seed,
                train_seed=train_seed,
            )

            # 4.2 导出 exports CSV + metrics + 追加 model_comparison.csv
            export_dir = build_stageA_report_single_run(
                dataset, meta, aux,
                out_root=out_root,
                model_name=model_type,
                seed=train_seed,  # 这里 seed 表示训练随机性 seed
                split_tag=f"{getattr(Cfg, 'split_tag', 'case_random')}_split{split_seed}"
            )

            print(f"[StageA-Compare] done: {run_tag} -> {export_dir}")
    # ✅全部跑完后：按 R2_min_test 自动选最优 run，并复制到固定目录
    best_out_root, best_row = select_best_run_from_model_comparison(Cfg.save_dir, metric="R2_min_test")
    best_fixed = os.path.join(Cfg.save_dir, "best_by_test")
    materialize_best_run(best_out_root, best_fixed)
    write_best_pointer(Cfg.save_dir, best_out_root, best_row, metric="R2_min_test")

    print(f"[StageA-Compare] BEST by R2_min_test = {best_row['R2_min_test']:.4f}")
    print(f"[StageA-Compare] materialized to: {best_fixed}")

    print("[StageA-Compare] All done. See:", os.path.join(Cfg.save_dir, "model_comparison.csv"))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_excel", type=str, default=Cfg.case_excel)
    ap.add_argument("--sheet_name", type=str, default=Cfg.sheet_name)
    ap.add_argument("--case_id_col", type=str, default=Cfg.case_id_col)
    ap.add_argument("--iedf_root", type=str, default=Cfg.iedf_root)
    ap.add_argument("--save_dir", type=str, default=Cfg.save_dir)
    ap.add_argument("--model_type", type=str, default=Cfg.model_type)
    ap.add_argument("--seed", type=int, default=Cfg.seed)
    ap.add_argument("--batch", type=int, default=Cfg.batch)
    ap.add_argument("--max_epochs", type=int, default=Cfg.max_epochs)
    ap.add_argument("--lr", type=float, default=Cfg.lr)
    ap.add_argument("--do_compare", type=int, default=int(Cfg.do_compare), choices=[0, 1])
    args = ap.parse_args()

    Cfg.case_excel = args.case_excel
    Cfg.sheet_name = args.sheet_name
    Cfg.case_id_col = args.case_id_col
    Cfg.iedf_root = args.iedf_root
    Cfg.save_dir = args.save_dir
    Cfg.model_type = args.model_type
    Cfg.seed = args.seed
    Cfg.batch = args.batch
    Cfg.max_epochs = args.max_epochs
    Cfg.lr = args.lr
    Cfg.do_compare = bool(args.do_compare)

    # ✅如果开了对比模式，就走对比 runner
    if getattr(Cfg, "do_compare", False):
        run_stageA_comparison()
        return

    # 否则：单模型跑一遍（你原来的逻辑）
    set_seed(Cfg.seed)
    os.makedirs(Cfg.save_dir, exist_ok=True)

    dataset, meta, aux = excel_to_phys7_dataset(
        Cfg.case_excel, Cfg.sheet_name, Cfg.case_id_col, Cfg.iedf_root
    )

    out_root = os.path.join(Cfg.save_dir, f"single_{Cfg.model_type}_seed{Cfg.seed}")

    _ = train_stageA_phys7_singleheads(
        dataset, meta,
        model_type=Cfg.model_type,
        out_root=out_root,
        split_seed=Cfg.seed,
        train_seed=Cfg.seed,
    )

    export_dir = build_stageA_report_single_run(
        dataset, meta, aux,
        out_root=out_root,
        model_name=Cfg.model_type,
        seed=Cfg.seed,
        split_tag=getattr(Cfg, "split_tag", "case_random")
    )

    print(f"[StageA-Phys7] Report exported to: {export_dir}")

if __name__ == "__main__":
    main()

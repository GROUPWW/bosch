# -*- coding: utf-8 -*-
from __future__ import annotations  # PEP 563: allow "X | None" annotations on Python 3.9
import os, json, argparse
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import physio_util as pu
import stageB_util as su
import stageB_train_morph_on_phys7_pycharm as sb
def normalize_time_list(time_list: List[str]) -> List[str]:
    """
    统一时间口径（强一致）：
    1) '9_2' 并入 '9'
    2) 其它 time：把 '_' 统一替换为 '.'，并用 {:g} 归一化字符串（0.10 -> 0.1）
    3) 去重保持顺序
    """
    out = []
    seen = set()
    for t in time_list:
        tt = str(t).strip()
        if tt == "9_2":
            tt = "9"
        else:
            tt = tt.replace("_", ".")
            try:
                tt = "{:g}".format(float(tt))
            except Exception:
                pass

        if tt not in seen:
            out.append(tt)
            seen.add(tt)
    return out

TIME_LIST = normalize_time_list(list(su.TIME_LIST))
TIME_VALUES = np.asarray([float(t.replace("_", ".")) for t in TIME_LIST], np.float32)
T = len(TIME_LIST)
FAMILIES = list(su.FAMILIES)           # ["zmin","h0","h1","d0","d1","w"]
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS/CPU 不需要额外 backend seed
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def family_to_index(fam: str) -> int:
    fam = str(fam).strip().lower()
    for i, f in enumerate(FAMILIES):
        if f.lower() == fam:
            return i
    raise KeyError(f"Unknown family={fam}, available={FAMILIES}")


def _torch_load_any(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strip_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        kk = k
        for p in ("module.", "model."):
            if kk.startswith(p):
                kk = kk[len(p):]
        out[kk] = v
    return out

def build_sparse_batch_subset_time(
    recs: List[Dict],
    time_list: List[str],
    time_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    约定：time_list 已经过 normalize_time_list() 处理（不含 9_2）。
    因此这里不再做 '9_2' -> '9' 的硬映射，避免“写入时合并、导出时不合并”的矛盾。
    """
    N = len(recs)
    K = len(FAMILIES)
    T = len(time_list)
    t2i = {t: i for i, t in enumerate(time_list)}
    f2i = {f: i for i, f in enumerate(FAMILIES)}

    recipe_raw = np.stack([r["static"] for r in recs], 0).astype(np.float32)  # (N,7)
    y = np.zeros((N, K, T), np.float32)
    m = np.zeros((N, K, T), np.bool_)

    for i, r in enumerate(recs):
        tg: Dict[Tuple[str, str], float] = r.get("targets", {})
        for (fam, tid), val in tg.items():
            fam = str(fam).strip().lower()
            tid = str(tid).strip()
            # 如果输入里仍有 9_2，则按你的口径并入 9（但 time_list 里必须有 '9'）
            if tid == "9_2":
                tid = "9"
            if fam in f2i and tid in t2i:
                y[i, f2i[fam], t2i[tid]] = float(val)
                m[i, f2i[fam], t2i[tid]] = True

    time_mat = np.tile(time_values.reshape(1, -1), (N, 1)).astype(np.float32)
    return recipe_raw, y, m, time_mat

def try_build_excel_row_1based_map(new_excel: str) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    """
    建立 {recipe_id(str) -> excel_row_1based(int)} 的映射
    - 约定：excel_row_1based 以“含表头”的真实行号计数（第1行表头，所以数据第1行是 2）
    - 如果失败返回 (None, None)
    """
    try:
        df = pd.read_excel(new_excel)
        rid_col = pu.detect_recipe_id_column(df)
        row_map: Dict[str, int] = {}
        col = df[rid_col].astype(str).tolist()
        for i, rid in enumerate(col):
            rr = str(rid).strip()
            if rr and (rr not in row_map):
                row_map[rr] = int(i + 2)  # +2: header 占第1行
        return row_map, rid_col
    except Exception:
        return None, None

def _build_gt_phys7_match(recipe_raw: np.ndarray, gt_phys7_source: str) -> np.ndarray:
    """
    从 case_with_phys7.xlsx 等真值源中，按前 7 维 recipe 参数精确匹配，
    返回与 recipe_raw 顺序一致的真值 phys7 描述符 (N,7)。
    静态列顺序约定：APC, source_RF, LF_RF, SF6, C4F8, DEP time, etch time。
    描述符列顺序：logGamma_SF6_tot, pF_SF6, spread_SF6, qskew_SF6,
                  logGamma_C4F8_tot, rho_C4F8, spread_C4F8。
    """
    df = pd.read_excel(gt_phys7_source)

    def col_like(kw):
        for c in df.columns:
            if kw == str(c).strip():
                return c
        for c in df.columns:
            if kw in str(c):
                return c
        return None

    recipe_keys = ["APC（E2步骤）", "source_RF（E2步骤）", "LF_RF（E2步骤）",
                   "SF6（E2步骤）", "C4F8（DEP步骤）", "DEP time", "etch time"]
    phys_keys = ["logGamma_SF6_tot", "pF_SF6", "spread_SF6", "qskew_SF6",
                 "logGamma_C4F8_tot", "rho_C4F8", "spread_C4F8"]

    recipe_cols = [col_like(k) for k in recipe_keys]
    phys_cols = [col_like(k) for k in phys_keys]
    missing = [k for k, c in zip(recipe_keys + phys_keys, recipe_cols + phys_cols) if c is None]
    if missing:
        raise ValueError(f"[gt-phys7] missing columns in {gt_phys7_source}: {missing}")

    gt_params = df[recipe_cols].to_numpy(dtype=np.float32)
    gt_phys = df[phys_cols].to_numpy(dtype=np.float32)

    N = recipe_raw.shape[0]
    out = np.zeros((N, len(phys_keys)), dtype=np.float32)
    matched = 0
    for i in range(N):
        bp = recipe_raw[i, :7]
        mask = np.all(np.isclose(gt_params, bp, rtol=1e-5, atol=1e-5), axis=1)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            raise ValueError(f"[gt-phys7] Bosch row {i} recipe {bp} has no match in {gt_phys7_source}")
        out[i] = gt_phys[idx].mean(axis=0).astype(np.float32)
        matched += 1

    print(f"[gt-phys7] matched {matched}/{N} recipes from {gt_phys7_source}")
    return out


def build_stageC_raw(device: str, new_excel: str, height_family: str, recipe_aug_mode: str,
                     stageA_heads_root: str, gt_phys7_source: Optional[str] = None) -> Dict[str, Any]:
    recs = pu.load_new_excel_as_sparse_morph(new_excel, height_family=height_family)

    recipe_raw, y_raw, mask, time_mat = build_sparse_batch_subset_time(recs, TIME_LIST, TIME_VALUES)
    static_raw = sb.augment_recipe_features(recipe_raw, str(recipe_aug_mode)).astype(np.float32)

    if gt_phys7_source and os.path.exists(gt_phys7_source):
        phys7_raw_full = _build_gt_phys7_match(recipe_raw, gt_phys7_source)
    else:
        provider = sb.StageAEnsemblePhys7Provider(
            heads_root=stageA_heads_root,
            device="cpu",  # ✅ StageA 推理在 MPS 上数值不稳定，固定用 CPU
            recipe_cols_in=None,
            expect_k=7
        )
        phys7_raw_full = provider.infer(recipe_raw, phys7_mode="full", use_cache=True).astype(np.float32)  # (N,7)

    # ---------------------------------------------------------
    # recipe_ids：必须与 recs 顺序一致，优先从 recs 里取
    # ---------------------------------------------------------
    recipe_ids = None
    cand_keys = ["recipe_id", "recipeID", "rid", "id", "name", "recipe", "RecipeID", "RecipeId"]
    unusable_flags = []
    try:
        ids = []
        for i, r in enumerate(recs):
            rid = None
            # 优先使用 load_new_excel_as_sparse_morph 已经读好的 recipe_id
            if "recipe_id" in r and str(r["recipe_id"]).strip():
                rid = str(r["recipe_id"]).strip()
            else:
                for kk in cand_keys:
                    if kk in r and str(r[kk]).strip():
                        rid = str(r[kk]).strip()
                        break
            if rid is None:
                rid = f"row{i:04d}"
            ids.append(rid)
            unusable_flags.append(int(r.get("unusable", 0)))
        recipe_ids = np.asarray(ids, dtype=object)
        unusable_arr = np.asarray(unusable_flags, dtype=np.int32)
    except Exception:
        recipe_ids = None
        unusable_arr = None

    # fallback：只有当 recs 没有任何 id 时才用 excel（且长度必须匹配 recs）
    if recipe_ids is None:
        try:
            df = pd.read_excel(new_excel)
            rid_col = pu.detect_recipe_id_column(df)
            arr = df[rid_col].astype(str).to_numpy()
            if len(arr) == len(recs):
                recipe_ids = arr
            else:
                recipe_ids = np.asarray([f"row{i:04d}" for i in range(len(recs))], dtype=object)
        except Exception:
            recipe_ids = np.asarray([f"row{i:04d}" for i in range(len(recs))], dtype=object)
        unusable_arr = np.zeros(len(recs), dtype=np.int32)

    # ---------------------------------------------------------
    # NEW: excel_row_1based（真实 Excel 行号，用 recipe_id 对齐）
    # ---------------------------------------------------------
    N = len(recs)
    excel_row_1based = np.full((N,), -1, dtype=np.int32)
    row_map, rid_col = try_build_excel_row_1based_map(new_excel)
    if row_map is not None:
        excel_row_1based = np.asarray([row_map.get(str(rid).strip(), -1) for rid in recipe_ids], dtype=np.int32)

    # 强校验：所有 raw 的 N 必须一致
    assert recipe_raw.shape[0] == N and y_raw.shape[0] == N and mask.shape[0] == N and phys7_raw_full.shape[0] == N
    assert len(recipe_ids) == N, f"[BUG] recipe_ids length {len(recipe_ids)} != N {N}"
    assert len(excel_row_1based) == N, f"[BUG] excel_row_1based length {len(excel_row_1based)} != N {N}"

    return dict(
        recipe_ids=recipe_ids,
        excel_row_1based=excel_row_1based,  # ✅ 新增：用于 predictions_valid.csv 的真实行号
        recipe_raw=recipe_raw,
        static_raw=static_raw,
        phys7_raw_full=phys7_raw_full,
        y_raw=y_raw,
        mask=mask,
        time_mat=time_mat,
        unusable=unusable_arr,  # ✅ 新增：0/1 标记完全无法使用的样本
    )

def make_summary_header_fields() -> List[str]:
    return [
        "phase", "epochs",
        "family", "exp",
        "init", "finetune_mode", "phys7_mode",
        "lr", "wd", "l2sp", "backbone_lr_ratio",
        "loss_type", "huber_beta",
        "r2", "mae_nm", "mape_pct", "median_ae_nm", "n",
        "best_epoch", "val_loss",
        "run_seed", "split_seed", "drop_tag",
        "trainN", "valN", "testN", "eval_set", "num_dropped",
        "ckpt",
    ]


def zfit(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std


def zfit_targets_masked_1fam(y: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.zeros((T,), np.float32)
    std = np.ones((T,), np.float32)
    for t in range(T):
        sel = m[:, t]
        if int(sel.sum()) < 2:
            mean[t] = 0.0
            std[t] = 1.0
        else:
            vals = y[sel, t]
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            if sd < 1e-6:
                sd = 1.0
            mean[t] = mu
            std[t] = sd
    return mean, std


def apply_z(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / (std + 1e-8)).astype(np.float32)


def apply_phys7_mode(phys7_z_full: np.ndarray, phys7_mode: str) -> np.ndarray:
    return sb.apply_phys7_mode(phys7_z_full.astype(np.float32), str(phys7_mode).strip())


def make_loader(static, phys7, y, m, time_mat, idx, batch_size, shuffle, num_workers):
    ds = TensorDataset(
        torch.from_numpy(static[idx]).float(),
        torch.from_numpy(phys7[idx]).float(),
        torch.from_numpy(y[idx]).float(),
        torch.from_numpy(m[idx]).bool(),
        torch.from_numpy(time_mat[idx]).float(),
        torch.from_numpy(idx).long()  # <--- 新增
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def compute_quality_score_1fam(m_fam: np.ndarray) -> np.ndarray:
    # m_fam: (N,T) bool
    return m_fam.astype(np.int32).sum(axis=1).astype(np.int32)  # per-sample valid count

def split_with_key_and_quality(
    recipe_ids: np.ndarray,
    key_recipes: List[str],
    scores: np.ndarray,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    N = len(recipe_ids)
    all_idx = np.arange(N)

    key_set = set([str(x).strip() for x in key_recipes if str(x).strip()])
    key_idx = np.array([i for i in all_idx if str(recipe_ids[i]) in key_set], dtype=int)

    # 修复：test_ratio <= 0 时，test 直接为空
    if test_ratio <= 0:
        test = []
        remain = all_idx.tolist()
    else:
        n_test = int(round(N * test_ratio))
        n_test = max(1, min(n_test, N))
        test = key_idx.tolist()
        remain = [i for i in all_idx.tolist() if i not in set(test)]
        if len(test) < n_test:
            need = n_test - len(test)
            remain_sorted = sorted(remain, key=lambda i: (scores[i], rng.rand()), reverse=True)
            add = remain_sorted[:need]
            test.extend(add)
            remain = [i for i in remain if i not in set(add)]

    remain_sorted = sorted(remain, key=lambda i: (scores[i], rng.rand()), reverse=True)
    val = remain_sorted[:int(round(N * val_ratio))]
    train = [i for i in remain if i not in set(val)]

    return np.asarray(train, int), np.asarray(val, int), np.asarray(test, int)

def check_min_test_points_1fam(m_fam: np.ndarray, test_idx: np.ndarray, min_points: int) -> bool:
    # m_fam: (N,T)
    return bool(int(m_fam[test_idx].sum()) >= int(min_points))

def build_model(model_type: str, static_dim: int, out_dim: int = 1) -> nn.Module:
    mt = str(model_type).lower().strip()
    if mt == "transformer":
        return su.MorphTransformer(
            static_dim=static_dim,
            d_model=getattr(sb.Cfg, "tf_d_model", 256),
            nhead=getattr(sb.Cfg, "tf_nhead", 8),
            num_layers=getattr(sb.Cfg, "tf_layers", 4),
            dropout=getattr(sb.Cfg, "tf_dropout", 0.1),
            out_dim=out_dim
        )
    if mt == "gru":
        return su.MorphGRU(
            static_dim=static_dim,
            hidden=getattr(sb.Cfg, "gru_hidden", 256),
            num_layers=getattr(sb.Cfg, "gru_layers", 1),
            out_dim=out_dim
        )
    if mt == "mlp":
        return su.MorphMLP(
            static_dim=static_dim,
            hidden=getattr(sb.Cfg, "mlp_hidden", 256),
            num_layers=getattr(sb.Cfg, "mlp_layers", 3),
            out_dim=out_dim
        )
    if mt == "lstm":
        return su.MorphLSTM(
            static_dim=static_dim,
            hidden=getattr(sb.Cfg, "lstm_hidden", 256),
            num_layers=getattr(sb.Cfg, "lstm_layers", 2),
            out_dim=out_dim
        )
    if mt == "tcn":
        return su.MorphTCN(
            static_dim=static_dim,
            hidden=getattr(sb.Cfg, "tcn_hidden", 256),
            num_layers=getattr(sb.Cfg, "tcn_layers", 4),
            kernel_size=getattr(sb.Cfg, "tcn_kernel", 3),
            out_dim=out_dim
        )
    if mt == "ar_mlp":
        return su.MorphARMLP(
            static_dim=static_dim,
            hidden=getattr(sb.Cfg, "armlp_hidden", 256),
            num_layers=getattr(sb.Cfg, "armlp_layers", 3),
            out_dim=out_dim
        )
    raise ValueError(f"Unknown model_type={model_type}")


def load_ckpt_into_model(model: nn.Module, ckpt_path: str) -> Dict[str, Any]:
    ck = _torch_load_any(ckpt_path)
    meta = ck.get("meta", {}) if isinstance(ck, dict) else {}
    sd = None
    if isinstance(ck, dict):
        sd = ck.get("model", None) or ck.get("state_dict", None)
    if sd is None and isinstance(ck, dict):
        cand = {k: v for k, v in ck.items() if isinstance(v, torch.Tensor)}
        sd = cand if cand else None
    if sd is None:
        raise RuntimeError(f"Invalid ckpt: {ckpt_path}")
    sd = _strip_prefix(sd)
    miss = model.load_state_dict(sd, strict=False)
    return {"meta": meta, "missing": list(miss.missing_keys), "unexpected": list(miss.unexpected_keys)}


def _head_parameters(model: nn.Module) -> list:
    """head 参数：优先 model.out（Transformer/GRU/LSTM/TCN），否则 model.net 的最后一个 Linear（MLP/AR-MLP）。"""
    if hasattr(model, "out"):
        return list(model.out.parameters())
    net = getattr(model, "net", None)
    if net is not None:
        linears = [m for m in net if isinstance(m, nn.Linear)]
        if linears:
            return list(linears[-1].parameters())
    return []


def apply_finetune_mode(model: nn.Module, mode: str):
    mode = str(mode).lower().strip()

    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return

    for p in model.parameters():
        p.requires_grad = False

    def _enable_out():
        for p in _head_parameters(model):
            p.requires_grad = True

    def _enable_layernorm_affine():
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True

    if mode == "head":
        _enable_out()
        return
    if mode == "head_ln":
        _enable_out()
        _enable_layernorm_affine()
        return

    raise ValueError(f"Unknown finetune_mode={mode}")

def masked_mse(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    diff = (pred - y) ** 2

    diff = diff.masked_fill(~m, 0.0)
    denom = m.float().sum().clamp_min(1.0)
    return diff.sum() / denom


def masked_huber(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    e = pred - y
    abs_e = torch.abs(e)
    b = float(beta)
    quad = 0.5 * (e ** 2) / max(b, 1e-12)
    lin = abs_e - 0.5 * b
    per = torch.where(abs_e < b, quad, lin)

    per = per.masked_fill(~m, 0.0)
    denom = m.float().sum().clamp_min(1.0)
    return per.sum() / denom


def masked_loss(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor, loss_type: str = "mse", huber_beta: float = 1.0) -> torch.Tensor:
    lt = str(loss_type).lower().strip()
    if lt == "mse":
        return masked_mse(pred, y, m)
    if lt == "huber":
        return masked_huber(pred, y, m, beta=huber_beta)
    raise ValueError(f"Unknown loss_type={loss_type}")

def l2sp_penalty(model: nn.Module, anchor_state: Dict[str, torch.Tensor], lam: float) -> torch.Tensor:
    if lam <= 0 or (not anchor_state):
        return torch.tensor(0.0, device=next(model.parameters()).device)
    reg = 0.0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n in anchor_state:
            a = anchor_state[n].to(device=p.device, dtype=p.dtype)
            reg = reg + (p - a).pow(2).mean()
    return reg * float(lam)


@torch.no_grad()
def eval_pack(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    preds, ys, ms, idxs = [], [], [], []

    for batch in loader:
        if len(batch) == 6:
            static_x, phys7_seq, y, m, time_mat, batch_idx = [t.to(device) for t in batch]
        else:
            static_x, phys7_seq, y, m, time_mat = [t.to(device) for t in batch]
            batch_idx = torch.zeros(static_x.size(0), dtype=torch.long, device=device) - 1

        p = model(static_x, phys7_seq, time_mat)  # (B,1,T)

        preds.append(p.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        ms.append(m.detach().cpu().numpy())
        idxs.append(batch_idx.detach().cpu().numpy())

    return dict(
        pred=np.concatenate(preds, 0),  # (N,1,T)
        y=np.concatenate(ys, 0),  # (N,1,T)
        m=np.concatenate(ms, 0).astype(bool),
        idx=np.concatenate(idxs, 0).astype(int)  # (N,)
    )


def train_one(model: nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader,
              device: str,
              epochs: int,
              lr: float,
              wd: float,
              patience: int,
              l2sp_lam: float = 0.0,
              anchor_state: Optional[Dict[str, torch.Tensor]] = None,
              is_transfer: bool = False,
              backbone_lr_ratio: float = 0.1,
              loss_type: str = "mse",
              huber_beta: float = 1.0,
              grad_clip: float = 1.0,
              progressive_unfreeze: bool = True,
              raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

    # 分组参数：head vs backbone（head 识别与 _head_parameters 一致，兼容 MLP 系）
    _head_ids = {id(p) for p in _head_parameters(model)}
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if id(param) in _head_ids:
            head_params.append(param)
        else:
            backbone_params.append(param)

    def make_opt():
        def is_no_decay(n: str) -> bool:
            nn_ = n.lower()
            if nn_.endswith(".bias"):
                return True
            if ("norm" in nn_) or (".ln" in nn_) or ("layernorm" in nn_):
                return True
            return False

        head_named, backbone_named = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in _head_ids:
                head_named.append((n, p))
            else:
                backbone_named.append((n, p))

        def build_groups(named_params, base_lr, base_wd):
            decay, nodecay = [], []
            for n, p in named_params:
                (nodecay if is_no_decay(n) else decay).append(p)
            groups = []
            if decay:
                groups.append({"params": decay, "lr": base_lr, "weight_decay": base_wd})
            if nodecay:
                groups.append({"params": nodecay, "lr": base_lr, "weight_decay": 0.0})
            return groups

        groups = []
        if backbone_named:
            groups += build_groups(backbone_named, lr * float(backbone_lr_ratio), wd)
        if head_named:
            groups += build_groups(head_named, lr, wd)
        if not groups:
            groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr, "weight_decay": wd}]

        return torch.optim.AdamW(groups)

    progressive_warmup = 0
    if bool(progressive_unfreeze) and is_transfer and 0.0 < backbone_lr_ratio < 1.0:
        progressive_warmup = max(5, int(epochs * 0.15))
        # 初始：冻结 backbone
        for p in backbone_params:
            p.requires_grad = False
        for p in head_params:
            p.requires_grad = True

    opt = make_opt()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    history = {"train": [], "val": []}
    best = {"val_loss": float("inf"), "epoch": 0, "state": None}
    bad = 0

    for ep in range(1, epochs + 1):
        # 仅在切换点重置优化器/调度器
        if progressive_warmup > 0 and ep == progressive_warmup + 1:
            for p in backbone_params:
                p.requires_grad = True
            for p in head_params:
                p.requires_grad = True
            opt = make_opt()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

        model.train()
        ep_train_loss, ep_train_n = 0.0, 0

        time_weights = None
        for batch in train_loader:
            if len(batch) == 6:
                x, p7, y, m, t, idx = [tt.to(device) for tt in batch]
            else:
                x, p7, y, m, t = [tt.to(device) for tt in batch]
                idx = None

            if time_weights is None:
                TT = int(y.shape[-1])

            opt.zero_grad()

            # main loss
            pred = model(x, p7, t)
            loss = masked_loss(
                pred, y, m,
                loss_type, huber_beta
            )

            # sample weight：若 raw 提供 unusable 标记，则降低 unusable=1 样本的 loss 权重
            # 注意：传入的 loss 已经是标量（masked_loss 内部已做 mean/sum），
            # 因此不能直接做逐元素乘；改在 masked_loss 之前按 mask 做逐元素加权。
            # 当前实现：unusable=1 的样本在全部被加权降低后，直接用 (loss * weight).sum() / weight.sum()
            # 但这要求 loss 返回的是未 reduce 的逐元素张量。当前 masked_loss 返回标量，此路径不正确。
            # 修复方案：暂不使用 sample weight，等 masked_loss 支持 'none' reduction 后再启用。
            # if idx is not None and "unusable" in raw:
            #     uw = torch.from_numpy(raw["unusable"]).float().to(device)
            #     sample_w = 1.0 - 0.9 * uw[idx]
            #     sample_w = sample_w.view(-1, 1, 1)
            #     loss = (loss * sample_w).sum() / (sample_w.sum().clamp_min(1.0))
            pass  # TODO: re-enable after masked_loss supports per-element output

            # L2SP
            if l2sp_lam > 0 and anchor_state:
                loss = loss + l2sp_penalty(model, anchor_state, l2sp_lam)

            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()

            ep_train_loss += float(loss.item())
            ep_train_n += 1

        scheduler.step()
        avg_train = ep_train_loss / max(1, ep_train_n)

        # val
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 6:
                    vx, vp, vy, vm, vt, _ = [tt.to(device) for tt in batch]
                else:
                    vx, vp, vy, vm, vt = [tt.to(device) for tt in batch]

                vp_ = model(vx, vp, vt)
                vl += float(masked_loss(vp_, vy, vm, loss_type, huber_beta).item())
                vn += 1

        val_loss = vl / max(1, vn)
        history["train"].append(avg_train)
        history["val"].append(val_loss)

        if val_loss < best["val_loss"]:
            best = {
                "val_loss": float(val_loss),
                "epoch": int(ep),
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)

    return {"epoch": best["epoch"], "val_loss": best["val_loss"], "history": history, "state": best["state"]}

def _find_candidates(runs_root: str, relname: str) -> List[str]:
    cands = [
        os.path.join(runs_root, relname),
        os.path.join(runs_root, "_tuneV_verify", relname),
    ]
    return [p for p in cands if os.path.exists(p)]


def load_best_config_common(runs_root: str) -> Dict[str, Any]:
    cands = _find_candidates(runs_root, "best_config_common_all_families.json")
    if not cands:
        raise FileNotFoundError(f"best_config_common_all_families.json not found under: {runs_root}")
    with open(cands[0], "r", encoding="utf-8") as f:
        return json.load(f)


def load_results_summary_df(runs_root: str) -> Optional[pd.DataFrame]:
    cands = _find_candidates(runs_root, "results_summary.csv")
    if not cands:
        return None
    df = pd.read_csv(cands[0])
    return df

def _resolve_ckpt_by_expname(runs_root: str, best_conf: dict, fam: str) -> str:
    mt  = str(best_conf.get("model_type"))
    hp  = str(best_conf.get("hp_tag"))
    ps  = str(best_conf.get("phys_source"))
    au  = str(best_conf.get("recipe_aug_mode"))
    p7  = str(best_conf.get("phys7_mode"))
    ss  = int(best_conf.get("split_seed"))

    exp_name = f"{mt}_{hp}_{ps}_{au}_{p7}_{fam}_s{ss}"
    cands = [
        os.path.join(runs_root, exp_name, "best.pth"),
        os.path.join(runs_root, "_tuneV_verify", exp_name, "best.pth"),
        # ✅ 兼容 paper-aligned best_by_test 目录结构：{fam}/best.pth
        os.path.join(runs_root, fam, "best.pth"),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    suffix = f"_{fam}_s{ss}"
    for base in [runs_root, os.path.join(runs_root, "_tuneV_verify")]:
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            if name.endswith(suffix):
                p = os.path.join(base, name, "best.pth")
                if os.path.isfile(p):
                    return p

    raise FileNotFoundError(f"Cannot locate best.pth for fam={fam} using best_conf under {runs_root}")

def resolve_stageB_best_ckpts_from_common(runs_root: str) -> Dict[str, str]:
    best_conf = load_best_config_common(runs_root)
    df = load_results_summary_df(runs_root)

    out: Dict[str, str] = {}
    if df is None:
        for fam in FAMILIES:
            out[fam] = _resolve_ckpt_by_expname(runs_root, best_conf, fam)
        return out
    filters = {}
    for k, v in best_conf.items():
        if k in df.columns:
            filters[k] = v

    df_f = df.copy()
    for k, v in filters.items():
        if k == "split_seed":
            df_f = df_f[pd.to_numeric(df_f[k], errors="coerce").fillna(-1).astype(int) == int(v)]
        else:
            df_f = df_f[df_f[k].astype(str) == str(v)]

    if df_f.empty:
        raise RuntimeError("No rows matched best_config in results_summary.csv (check paths / columns).")

    if "family_mode" not in df_f.columns or "ckpt_path" not in df_f.columns:
        raise RuntimeError("results_summary.csv missing required columns: family_mode / ckpt_path")

    for fam in FAMILIES:
        dff = df_f[df_f["family_mode"].astype(str).str.lower() == fam.lower()].copy()
        if dff.empty:
            continue
        if "min_pf_r2" in dff.columns:
            dff["min_pf_r2"] = pd.to_numeric(dff["min_pf_r2"], errors="coerce")
            dff = dff.sort_values("min_pf_r2", ascending=False)
        out[fam] = str(dff.iloc[0]["ckpt_path"])

    if not out:
        raise RuntimeError("Matched config, but no family ckpts resolved (check family_mode values).")

    return out

@dataclass
class ExpCfg:
    name: str
    init: str                 # 'scratch' | 'stageB_best'
    finetune_mode: str        # 'full' | 'head' | 'head_ln'
    phys7_mode: str           # 'none' | 'full'

    lr: float = 1e-3
    wd: float = 1e-4
    l2sp: float = 0.0

    backbone_lr_ratio: float = 0.1
    loss_type: str = "mse"
    huber_beta: float = 1.0
    grad_clip: float = 1.0

    progressive_unfreeze: bool = True

def run_one_experiment_on_split_1fam(
        fam: str, exp: ExpCfg, device: str, raw: Dict[str, Any], split: Dict[str, Any],
        stageB_best_ckpt_for_fam: Optional[str], model_type: str, stageA_heads_root: str,
        recipe_aug_mode: str, out_dir_fam: str, epochs: int, batch: int, patience: int,
        num_workers: int, run_seed: int,
        return_details: bool = False,
        save_artifacts: bool = True
) -> Dict[str, Any]:
    # 将 raw 绑定为局部变量，供 train_one 闭包内 sample weight 使用
    _RAW_REF = raw
    set_seed(int(run_seed))

    recipe_ids = raw["recipe_ids"]
    static_raw = raw["static_raw"]
    phys7_raw_full = raw["phys7_raw_full"]
    y_raw = raw["y_raw"]
    mask = raw["mask"]
    time_mat = raw["time_mat"]

    # NEW: excel_row_1based（用于导出对齐真实 excel 行号）
    excel_row_1based = raw.get("excel_row_1based", None)

    k = family_to_index(fam)
    y_f = y_raw[:, k, :].astype(np.float32)       # (N,T)
    m_f = mask[:, k, :].astype(bool)              # (N,T)

    # ignored_idx：即使 iter_candidate_datasets 已经“真删点”，这里再 mask 一次也无害
    ign = np.asarray(split.get("ignored_idx", []), dtype=int)
    if ign.size > 0:
        m_f[ign, :] = False

    train_idx = np.asarray(split["train_idx"], dtype=int)
    val_idx = np.asarray(split["val_idx"], dtype=int)
    test_idx = np.asarray(split["test_idx"], dtype=int)

    # normalize (train only)
    s_mean, s_std = zfit(static_raw[train_idx])
    p_mean, p_std = zfit(phys7_raw_full[train_idx])
    y_mean_t, y_std_t = zfit_targets_masked_1fam(y_f[train_idx], m_f[train_idx])

    static = apply_z(static_raw, s_mean, s_std)
    phys7_z_full = apply_z(phys7_raw_full, p_mean, p_std)
    phys7_z = apply_phys7_mode(phys7_z_full, exp.phys7_mode)

    T_len = int(time_mat.shape[1]) if time_mat.ndim == 2 else int(time_mat.shape[0])

    # 统一规定：phys7_seq 必须是 (N, 7, T_len)
    phys7_seq = np.tile(phys7_z[:, :, None], (1, 1, T_len)).astype(np.float32)

    if phys7_seq.ndim != 3 or phys7_seq.shape[1] != 7 or phys7_seq.shape[2] != T_len:
        raise RuntimeError(f"[BUG] phys7_seq shape must be (N,7,T). got {phys7_seq.shape}, T_len={T_len}")

    y_norm = apply_z(y_f, y_mean_t.reshape(1, -1), y_std_t.reshape(1, -1)).astype(np.float32)
    y_norm = y_norm[:, None, :]
    m_ = m_f[:, None, :]

    train_loader = make_loader(static, phys7_seq, y_norm, m_, time_mat, train_idx, batch, True,  num_workers)
    val_loader   = make_loader(static, phys7_seq, y_norm, m_, time_mat, val_idx,   batch, False, num_workers)
    test_loader  = make_loader(static, phys7_seq, y_norm, m_, time_mat, test_idx,  batch, False, num_workers)

    # build model
    model = build_model(model_type, static_dim=static.shape[1], out_dim=1).to(device)

    init_info = {}
    if exp.init == "stageB_best":
        if not stageB_best_ckpt_for_fam:
            raise RuntimeError(f"No StageB best ckpt for fam={fam}")

        load_info = load_ckpt_into_model(model, stageB_best_ckpt_for_fam)

        init_info["ckpt"] = stageB_best_ckpt_for_fam
        init_info["ckpt_missing"] = load_info.get("missing", [])
        init_info["ckpt_unexpected"] = load_info.get("unexpected", [])
    else:
        init_info["ckpt"] = "scratch"

    apply_finetune_mode(model, exp.finetune_mode)

    anchor_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
    is_transfer_exp = (exp.init != "scratch")
    progressive_ok = bool(getattr(exp, "progressive_unfreeze", True)) and (str(exp.finetune_mode).lower().strip() == "full")

    best = train_one(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=exp.lr, wd=exp.wd, patience=patience,
        l2sp_lam=exp.l2sp, anchor_state=anchor_state,
        is_transfer=is_transfer_exp,
        backbone_lr_ratio=getattr(exp, "backbone_lr_ratio", 0.1),
        loss_type=getattr(exp, "loss_type", "mse"),
        huber_beta=getattr(exp, "huber_beta", 1.0),
        grad_clip=getattr(exp, "grad_clip", 1.0),
        progressive_unfreeze=progressive_ok,
        raw=_RAW_REF,
    )

    # eval set decision
    if len(test_idx) == 0:
        final_loader = val_loader
        eval_set = "val"
    else:
        use_val_as_test = (len(test_idx) <= 1 and len(val_idx) > 1)
        final_loader = val_loader if use_val_as_test else test_loader
        eval_set = "val" if use_val_as_test else "test"

    pack = eval_pack(model, final_loader, device)

    # ✅ 额外：在全量数据上推理一次，用于 predictions_all.csv（散点图 / 异常 recipe 识别）
    all_loader = make_loader(static, phys7_seq, y_norm, m_, time_mat,
                             np.arange(len(recipe_ids), dtype=int), batch, False, num_workers)
    pack_all = eval_pack(model, all_loader, device)

    met = metrics_1fam_display(
        pack, fam, y_mean_t, y_std_t,
        recipe_ids_lookup=recipe_ids, time_list=TIME_LIST,
        unit_scale=1000.0
    )

    row = {
        "family": fam,
        "exp": exp.name,

        "init": exp.init,
        "finetune_mode": exp.finetune_mode,
        "phys7_mode": exp.phys7_mode,
        "lr": float(exp.lr),
        "wd": float(exp.wd),
        "l2sp": float(exp.l2sp),
        "backbone_lr_ratio": float(getattr(exp, "backbone_lr_ratio", 0.1)),
        "loss_type": str(getattr(exp, "loss_type", "mse")),
        "huber_beta": float(getattr(exp, "huber_beta", 1.0)),

        "r2": float(met.get("r2", float("nan"))),
        "mae_nm": float(met.get("mae_nm", float("nan"))),
        "mape_pct": float(met.get("mape_pct", float("nan"))),
        "median_ae_nm": float(met.get("median_ae_nm", float("nan"))),
        "n": int(met.get("n", 0)),

        "best_epoch": int(best.get("epoch", 0)),
        "val_loss": float(best.get("val_loss", float("nan"))),

        "run_seed": int(run_seed),
        "split_seed": int(split.get("seed", -1)),

        "trainN": int(len(train_idx)),
        "valN": int(len(val_idx)),
        "testN": int(len(test_idx)),
        "eval_set": str(eval_set),
        "num_dropped": int(len(split.get("ignored_idx", []))),

        **init_info
    }

    if return_details:
        pred = torch.from_numpy(pack["pred"]).float()
        y = torch.from_numpy(pack["y"]).float()
        m = torch.from_numpy(pack["m"]).bool()
        idx = torch.from_numpy(pack["idx"]).long()

        TT = pred.shape[-1]
        mean = torch.from_numpy(y_mean_t).view(1, 1, TT)
        std = torch.from_numpy(y_std_t).view(1, 1, TT)

        pred_um = pred * std + mean
        y_um = y * std + mean
        abs_err = torch.abs(pred_um - y_um) * 1000.0

        total_N = len(recipe_ids)
        sample_errs = np.zeros(total_N, dtype=np.float32)
        sample_counts = np.zeros(total_N, dtype=np.float32)

        b_idx = idx.cpu().numpy()
        b_err = abs_err[:, 0, :].cpu().numpy()
        b_m = m[:, 0, :].cpu().numpy()

        for i in range(len(b_idx)):
            rid = int(b_idx[i])
            valid_t = b_m[i]
            if valid_t.sum() > 0:
                sample_errs[rid] += float(b_err[i][valid_t].mean())
                sample_counts[rid] += 1.0

        row["sample_errs"] = sample_errs
        row["sample_counts"] = sample_counts
        return row

    if not save_artifacts:
        return row

    # -------------------------
    # exp_dir：加入 split_seed + drop_tag + eval_set，彻底避免覆盖
    # -------------------------
    split_seed = int(split.get("seed", -1))
    drop_tag = str(split.get("drop_tag", "drop0"))
    exp_dir = os.path.join(
        out_dir_fam, "experiments", exp.name,
        f"seed_{int(run_seed)}_split_{split_seed}_{drop_tag}_{str(eval_set)}"
    )
    ensure_dir(exp_dir)

    manifest = {
        "family": fam,
        "exp": exp.name,
        "run_seed": int(run_seed),
        "split_seed": split_seed,
        "drop_tag": drop_tag,
        "eval_set": str(eval_set),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "ignored_idx": [int(x) for x in split.get("ignored_idx", [])],

        "cfg": {k: row.get(k, None) for k in [
            "init","finetune_mode","phys7_mode","lr","wd","l2sp","backbone_lr_ratio",
            "loss_type","huber_beta"
        ]}
    }
    with open(os.path.join(exp_dir, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if best.get("state") is not None:
        # 默认不保存 best_model.pth 以节省磁盘；需要时可通过环境变量启用
        if os.environ.get("STAGEC_SAVE_CKPT", "0") == "1":
            torch.save(best["state"], os.path.join(exp_dir, "best_model.pth"))

    history = best.get("history", {})
    if isinstance(history, dict) and len(history) > 0:
        pd.DataFrame(history).to_csv(os.path.join(exp_dir, "loss_history.csv"), index_label="epoch")

    # -------------------------
    # predictions_valid.csv：原 eval_set 上的预测（保持兼容）
    # predictions_all.csv：全量样本预测，用于散点图 / 异常 recipe 识别
    # -------------------------
    for pred_pack, pred_eval_set, pred_name in [(pack, eval_set, "predictions_valid.csv"),
                                                 (pack_all, "all", "predictions_all.csv")]:
        try:
            pred = torch.from_numpy(pred_pack["pred"]).float()
            y    = torch.from_numpy(pred_pack["y"]).float()
            m    = torch.from_numpy(pred_pack["m"]).bool()
            raw_idx = torch.from_numpy(pred_pack["idx"]).long()

            N, K, TT = pred.shape
            t_idx_base = torch.arange(TT).view(1, 1, TT).expand(N, K, TT)

            mean = torch.from_numpy(y_mean_t).view(1, 1, TT)
            std  = torch.from_numpy(y_std_t).view(1, 1, TT)

            pred_um = pred * std + mean
            y_um    = y * std + mean

            sign_map, _ = su._default_family_sign_and_nonneg([fam])
            family_sign = torch.tensor([sign_map[fam]], dtype=torch.float32)

            pred_disp, y_disp = pu.transform_for_display(
                pred_um, y_um,
                family_sign=family_sign,
                clip_nonneg=True,
                nonneg_families=[0],
                unit_scale=1000.0,
                flip_sign=False,
                min_display_value=None
            )

            yp_all = pred_disp[:, 0, :].reshape(-1).cpu().numpy()
            yt_all = y_disp[:, 0, :].reshape(-1).cpu().numpy()
            mk = pred_pack["m"][:, 0, :].reshape(-1)

            flat_recipe_idx = raw_idx.view(N, 1, 1).expand(N, K, TT)[:, 0, :].reshape(-1).cpu().numpy()
            flat_time_idx = t_idx_base[:, 0, :].reshape(-1).cpu().numpy()

            yp = yp_all[mk]
            yt = yt_all[mk]
            r_idx_m = flat_recipe_idx[mk].astype(np.int64)
            t_idx_m = flat_time_idx[mk].astype(np.int64)

            ok = np.isfinite(yt) & np.isfinite(yp)
            yp = yp[ok]
            yt = yt[ok]
            r_idx_m = r_idx_m[ok]
            t_idx_m = t_idx_m[ok]

            recipe_id_m = [str(recipe_ids[int(i)]) for i in r_idx_m]
            time_id_m   = [str(TIME_LIST[int(t)]) for t in t_idx_m]
            abs_err_nm = np.abs(yt - yp)
            time_value_m = [float(TIME_VALUES[int(t)]) for t in t_idx_m]

            # ✅ 行号：优先用 excel_row_1based（真实），否则退化为 idx+2
            recipe_row_source = []
            if isinstance(excel_row_1based, np.ndarray) and len(excel_row_1based) == len(recipe_ids):
                recipe_row_1based_out = []
                for ridx in r_idx_m:
                    rr = int(excel_row_1based[int(ridx)])
                    recipe_row_1based_out.append(rr)
                    recipe_row_source.append("excel_map")
            else:
                recipe_row_1based_out = (r_idx_m + 2).astype(int).tolist()
                recipe_row_source = ["idx+2"] * len(recipe_row_1based_out)

            # 标记每个样本属于 train/val/test/all
            subset_tags = []
            for ridx in r_idx_m:
                ridx_i = int(ridx)
                if pred_eval_set == "all":
                    if ridx_i in train_idx:
                        subset_tags.append("train")
                    elif ridx_i in val_idx:
                        subset_tags.append("val")
                    elif ridx_i in test_idx:
                        subset_tags.append("test")
                    else:
                        subset_tags.append("ignored")
                else:
                    subset_tags.append(str(pred_eval_set))

            res_df = pd.DataFrame({
                "family": fam,
                "exp": exp.name,
                "run_seed": int(run_seed),
                "split_seed": split_seed,
                "drop_tag": drop_tag,
                "eval_set": subset_tags,

                "recipe_idx": r_idx_m,
                "recipe_row_1based": recipe_row_1based_out,
                "recipe_row_source": recipe_row_source,
                "recipe_id": recipe_id_m,

                "time_idx": t_idx_m,
                "time_value": time_value_m,
                "time_label": time_id_m,

                "y_true_nm": yt,
                "y_pred_nm": yp,
                "abs_err_nm": abs_err_nm,
            })

            res_df.to_csv(os.path.join(exp_dir, pred_name), index=False)

        except Exception as e:
            with open(os.path.join(exp_dir, f"{pred_name}_error.txt"), "w", encoding="utf-8") as f:
                f.write(str(e) + "\n\n" + traceback.format_exc())

    if "bad_cases" in met:
        with open(os.path.join(exp_dir, "bad_cases.json"), "w", encoding="utf-8") as f:
            json.dump(met["bad_cases"], f, indent=2)

    return row

def build_common_split_for_seed(
    raw: Dict[str, Any],
    key_recipes: List[str],
    families_eval: List[str],
    test_ratio: float,
    val_ratio: float,
    min_test_points: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """
    给定 seed，直接构造一个 split，并验证对所有 families_eval 的 eval 集合都有 >= min_test_points。
    不做 proxy，不做搜索，只做“按 seed 枚举 + 过滤不合法 split”。
    """
    recipe_ids = raw["recipe_ids"]
    mask = raw["mask"]  # (N,K,T)
    N = len(recipe_ids)

    # 这里 scores 用全 1；如果你想按质量排序，可把 scores 换成 per-sample valid-count 等
    scores = np.ones(N, dtype=np.int32)

    try:
        train_idx, val_idx, test_idx = split_with_key_and_quality(
            recipe_ids=recipe_ids,
            key_recipes=key_recipes,
            scores=scores,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            seed=int(seed),
        )
    except Exception:
        return None

    check_idx = test_idx if len(test_idx) > 1 else val_idx
    if len(check_idx) == 0:
        return None

    for fam in families_eval:
        kk = family_to_index(fam)
        valid_pts = int(mask[:, kk, :][check_idx].sum())
        if valid_pts < int(min_test_points):
            return None

    return {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "ignored_idx": [],
        "seed": int(seed),
    }
def run_datasets(
    datasets: List[Dict[str, Any]],
    raw: Dict[str, Any],
    families_eval: List[str],
    experiments: List["ExpCfg"],
    stageB_best: Dict[str, str],
    args: argparse.Namespace,
    device: str,
    summary_all: str,
    header_fields: List[str],
    done_keys: set,
    epochs: int,
    patience: int,
    tag: str,
    key_recipes: List[str],
):
    ensure_dir(os.path.dirname(summary_all))
    recipe_ids = raw["recipe_ids"]

    for sp in datasets:
        split_seed = int(sp.get("seed", -1))
        drop_tag = str(sp.get("drop_tag", "drop0"))
        ignored_idx = [int(x) for x in sp.get("ignored_idx", [])]
        ignored_names = [str(recipe_ids[i]) for i in ignored_idx] if len(ignored_idx) else []

        # dataset manifest（可复现）
        ds_dir = os.path.join(args.out_dir, tag, f"split_{split_seed}", drop_tag)
        ensure_dir(ds_dir)
        with open(os.path.join(ds_dir, "dataset_manifest.json"), "w", encoding="utf-8") as f:
            json.dump({
                "tag": tag,
                "split_seed": split_seed,
                "drop_tag": drop_tag,
                "train_idx": sp.get("train_idx", []),
                "val_idx": sp.get("val_idx", []),
                "test_idx": sp.get("test_idx", []),
                "ignored_idx": ignored_idx,
                "ignored_names": ignored_names,
                "families_eval": families_eval,
                "key_recipes": key_recipes,
                "test_ratio": args.test_ratio,
                "val_ratio": args.val_ratio,
                "min_test_points": args.min_test_points,
            }, f, indent=2)

        # eval_set 逻辑（保持一致）
        testN = len(sp.get("test_idx", []))
        valN  = len(sp.get("val_idx", []))
        if testN == 0:
            eval_set = "val"
        else:
            use_val_as_test = (testN <= 1 and valN > 1)
            eval_set = "val" if use_val_as_test else "test"

        for fam in families_eval:
            out_dir_fam = os.path.join(args.out_dir, tag, f"split_{split_seed}", drop_tag, fam)
            ensure_dir(out_dir_fam)

            for exp in experiments:
                fam_ckpt = stageB_best.get(fam, None)
                if exp.init == "stageB_best":
                    if (not fam_ckpt) or (not os.path.exists(fam_ckpt)):
                        print(f"[WARN] skip fam={fam} exp={exp.name}: missing stageB ckpt.")
                        continue
                else:
                    fam_ckpt = None

                for i in range(int(args.seed_repeats)):
                    run_seed = int(args.seed + i)
                    key = (str(fam), str(exp.name), int(run_seed), int(split_seed), str(drop_tag), str(eval_set))
                    if key in done_keys:
                        continue

                    try:
                        row = run_one_experiment_on_split_1fam(
                            fam=fam, exp=exp, device=device, raw=raw, split=sp,
                            stageB_best_ckpt_for_fam=fam_ckpt,
                            model_type=args.model_type,
                            stageA_heads_root=args.stageA_heads_root,
                            recipe_aug_mode=args.recipe_aug_mode,
                            out_dir_fam=out_dir_fam,
                            epochs=int(getattr(exp, "epochs", epochs) or epochs or 200),
                            batch=int(getattr(exp, "batch", args.batch) or args.batch or 64),
                            patience=int(patience),
                            num_workers=args.num_workers,
                            run_seed=run_seed
                        )
                        row["phase"] = tag
                        row["epochs"] = int(getattr(exp, "epochs", epochs) or epochs or 200)
                        row["drop_tag"] = drop_tag
                        row["eval_set"] = eval_set

                        _append_summary_row(summary_all, row, header_fields)
                        done_keys.add(key)

                    except Exception as e:
                        print(f"\n[SKIP] failed: fam={fam}, exp={exp.name}, split={split_seed}, {drop_tag}, run_seed={run_seed}")
                        print(f"Reason: {e}")
                        traceback.print_exc()
                        continue

def _load_done_keys(csv_path: str):
    done = set()
    if not os.path.exists(csv_path):
        return done
    try:
        df = pd.read_csv(csv_path)
        need = {"family", "exp", "run_seed", "split_seed", "eval_set"}
        if not need.issubset(df.columns):
            return done

        has_drop = "drop_tag" in df.columns
        for _, r in df.iterrows():
            try:
                done.add((
                    str(r["family"]),
                    str(r["exp"]),
                    int(r["run_seed"]),
                    int(r["split_seed"]),
                    str(r["drop_tag"]) if has_drop else "drop0",
                    str(r["eval_set"]),
                ))
            except Exception:
                pass
    except Exception:
        pass
    return done

def _append_summary_row(csv_path: str, row: Dict[str, Any], header_fields: List[str]):
    import csv

    # 只写 header_fields 里出现的键
    clean = {}
    for k in header_fields:
        v = row.get(k, "")
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                v = ""
        clean[k] = v

    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header_fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(clean)
        f.flush()
        os.fsync(f.fileno())


def exp_to_group(exp_name: str) -> str:
    n = str(exp_name).lower()
    if "scratch" in n:
        return "0_scratch"
    if "l2sp" in n:
        return "2_transfer_l2sp"
    if "stageb" in n or "transfer" in n or "head_ln" in n:
        return "1_transfer"
    return "other"

def pick_best_rows(rows: List[Dict[str, Any]],
                   key_fields: Tuple[str, ...] = ("family", "exp"),
                   metric: str = "r2") -> List[Dict[str, Any]]:
    best = {}
    for r in rows:
        key = tuple(r.get(k) for k in key_fields)
        cur = best.get(key, None)
        if cur is None:
            best[key] = r
            continue
        try:
            if float(r.get(metric, -1e9)) > float(cur.get(metric, -1e9)):
                best[key] = r
        except Exception:
            pass
    return list(best.values())

def metrics_1fam_display(pack, fam, y_mean_t, y_std_t, unit_scale=1000.0,
                         recipe_ids_lookup=None, time_list=None):
    pred = torch.from_numpy(pack["pred"]).float()
    y = torch.from_numpy(pack["y"]).float()
    m = torch.from_numpy(pack["m"]).bool()
    raw_idx = torch.from_numpy(pack["idx"]).long()  # (N,)

    N, K, T = m.shape
    t_idx_base = torch.arange(T, device=m.device).view(1, 1, T).expand(N, K, T)

    mean = torch.from_numpy(y_mean_t).view(1, 1, T)
    std = torch.from_numpy(y_std_t).view(1, 1, T)

    pred_um = pred * std + mean
    y_um = y * std + mean

    sign_map, nonneg_set = su._default_family_sign_and_nonneg([fam])
    family_sign = torch.tensor([sign_map[fam]], dtype=torch.float32)

    pred_disp, y_disp = pu.transform_for_display(
        pred_um, y_um,
        family_sign=family_sign,
        clip_nonneg=True,          # 你说都应为正，这里保持即可
        nonneg_families=[0],
        unit_scale=unit_scale,
        flip_sign=False,
        min_display_value=None
    )

    # Flatten
    yp_all = pred_disp[:, 0, :].reshape(-1).numpy()
    yt_all = y_disp[:, 0, :].reshape(-1).numpy()
    mk_all = pack["m"][:, 0, :].reshape(-1)  # numpy bool

    flat_recipe_idx = raw_idx.view(N, 1, 1).expand(N, K, T)[:, 0, :].reshape(-1).numpy()
    flat_time_idx = t_idx_base[:, 0, :].reshape(-1).numpy()

    # Apply mask + finite
    yp = yp_all[mk_all]
    yt = yt_all[mk_all]
    r_idx_m = flat_recipe_idx[mk_all]
    t_idx_m = flat_time_idx[mk_all]

    ok = np.isfinite(yt) & np.isfinite(yp)
    yp = yp[ok]
    yt = yt[ok]
    r_idx_m = r_idx_m[ok]
    t_idx_m = t_idx_m[ok]

    n = int(len(yt))
    bad_cases_list = []
    id_recipe_list, id_time_list = [], []

    r2 = mae = median_ae = mape = float("nan")
    if n >= 2:
        abs_err = np.abs(yt - yp)
        r2 = float(su.masked_r2_score_np(yt, yp))
        mae = float(np.mean(abs_err))
        median_ae = float(np.median(abs_err))
        denom = np.abs(yt)
        denom[denom < 1e-6] = 1e-6
        mape = float(np.mean(abs_err / denom)) * 100.0

        if recipe_ids_lookup is not None and time_list is not None:
            id_recipe_list = [str(recipe_ids_lookup[i]) for i in r_idx_m]
            id_time_list = [str(time_list[t]) for t in t_idx_m]

            worst_indices = np.argsort(abs_err)[-10:][::-1]
            for ii in worst_indices:
                bad_cases_list.append({
                    "label": f"{id_recipe_list[ii]}@{id_time_list[ii]}",
                    "true": float(yt[ii]),
                    "pred": float(yp[ii]),
                    "err": float(abs_err[ii]),
                })
                if len(bad_cases_list) >= 5:
                    break

    return {
        "family": fam,
        "r2": r2,
        "mae_nm": mae,
        "mape_pct": mape,
        "median_ae_nm": median_ae,
        "n": n,
        "yp": yp.tolist(),
        "yt": yt.tolist(),
        "bad_cases": bad_cases_list,

        # 这两个用于定位“你觉得对不上”的根因
        "flat_recipe_idx": r_idx_m.astype(int).tolist(),
        "flat_time_idx": t_idx_m.astype(int).tolist(),

        # ✅ 这两项给 predictions_valid.csv 用
        "ids_recipe": id_recipe_list,
        "ids_time": id_time_list,
    }


def build_single_fixed_dataset(
    raw: Dict[str, Any],
    key_recipes: List[str],
    families_eval: List[str],
    test_ratio: float,
    val_ratio: float,
    min_test_points: int,
    split_seed: int,
    n_drop: int = 0,
) -> Dict[str, Any]:
    """
    固定一个 split_seed，只构造 1 份数据集，不做 split 搜索。

    注意：这里已经取消 drop/删点逻辑。
    n_drop 参数仅为兼容旧命令行保留，无论传入多少都不会删除样本。
    """
    sp = build_common_split_for_seed(
        raw=raw,
        key_recipes=key_recipes,
        families_eval=families_eval,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        min_test_points=min_test_points,
        seed=split_seed,
    )
    if sp is None:
        raise RuntimeError(
            f"Fixed split is invalid: split_seed={split_seed}. "
            f"Try lowering min_test_points or adjusting val_ratio/test_ratio."
        )

    # 固定不删样本：不再调用 build_drop_set_for_split，也不再从 train/val/test 中移除任何 index。
    sp["ignored_idx"] = []
    sp["drop_tag"] = "drop0"
    return sp


def main():
    _here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_excel", type=str, default=os.path.join(_here, "Bosch.xlsx"))
    ap.add_argument("--out_dir", type=str, default=os.path.join(_here, "runs_stageC_singlehead_fixedsplit"))
    ap.add_argument("--stageB_runs_root", type=str, default=os.path.join(_here, "runs_stageB_morph_phys7"))

    ap.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "gru", "mlp"])
    ap.add_argument("--recipe_aug_mode", type=str, default="time")
    ap.add_argument("--height_family", type=str, default="h1")
    ap.add_argument("--stageA_heads_root", type=str, default=getattr(sb.Cfg, "stageA_heads_root", os.path.join(_here, "runs_stageA_phys7", "best_by_test")))

    ap.add_argument("--key_recipes", type=str, default="")
    ap.add_argument("--families_eval", type=str, default="zmin,h1,d1,w")

    ap.add_argument("--seed", type=int, default=2026, help="训练随机种子起点；repeat 时依次 +1")
    ap.add_argument("--seed_repeats", type=int, default=3, help="同一固定 split 上重复训练次数")
    ap.add_argument("--split_seed", type=int, default=2026, help="固定随机划分种子（唯一）")
    ap.add_argument("--fixed_drop", type=int, default=0, help="兼容旧参数；当前版本已取消 drop，传入值会被忽略")

    ap.add_argument("--test_ratio", type=float, default=0.0)
    ap.add_argument("--val_ratio", type=float, default=0.3)
    ap.add_argument("--min_test_points", type=int, default=1)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--num_workers", type=int, default=0)

    args = ap.parse_args()

    ensure_dir(args.out_dir)
    device = su.get_default_device()

    if int(args.fixed_drop) != 0:
        print(f"[WARN] --fixed_drop={args.fixed_drop} was provided, but drop is disabled in this version. Use drop0.")

    # -----------------------------------------------------------
    # 1) Load StageB best config + resolve ckpts
    # -----------------------------------------------------------
    print(f"[StageC] Loading StageB best config from: {args.stageB_runs_root}")
    best_conf = load_best_config_common(args.stageB_runs_root)
    sb.apply_hp_from_best_conf_to_cfg(best_conf)

    stageB_aug_mode = best_conf.get("recipe_aug_mode", args.recipe_aug_mode)
    stageB_phys_mode = best_conf.get("phys7_mode", "full")
    stageB_model_type = best_conf.get("model_type", args.model_type)

    args.recipe_aug_mode = stageB_aug_mode
    args.model_type = stageB_model_type

    stageB_best = resolve_stageB_best_ckpts_from_common(args.stageB_runs_root)

    # -----------------------------------------------------------
    # 2) Build raw StageC data
    # -----------------------------------------------------------
    raw = build_stageC_raw(
        device=device,
        new_excel=args.new_excel,
        height_family=args.height_family,
        recipe_aug_mode=args.recipe_aug_mode,
        stageA_heads_root=args.stageA_heads_root
    )

    families_eval = [x.strip().lower() for x in str(args.families_eval).split(",") if x.strip()]
    key_recipes = [x.strip() for x in str(args.key_recipes).split(",") if x.strip()]

    if key_recipes:
        key_set = set([str(x).strip() for x in key_recipes])
        hit = [i for i, rid in enumerate(raw["recipe_ids"]) if str(rid) in key_set]
        print(f"[CHECK] key_recipes hits = {len(hit)} / {len(key_recipes)}")

    # -----------------------------------------------------------
    # 3) Build ONE fixed dataset (no split search)
    # -----------------------------------------------------------
    fixed_dataset = build_single_fixed_dataset(
        raw=raw,
        key_recipes=key_recipes,
        families_eval=families_eval,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        min_test_points=args.min_test_points,
        split_seed=args.split_seed,
        n_drop=args.fixed_drop,
    )
    datasets_all = [fixed_dataset]

    print("[Info] Fixed dataset ready:")
    print(f"  split_seed = {fixed_dataset['seed']}")
    print(f"  drop_tag   = {fixed_dataset.get('drop_tag', 'drop0')}")
    print(f"  trainN     = {len(fixed_dataset.get('train_idx', []))}")
    print(f"  valN       = {len(fixed_dataset.get('val_idx', []))}")
    print(f"  testN      = {len(fixed_dataset.get('test_idx', []))}")
    print(f"  ignoredN   = {len(fixed_dataset.get('ignored_idx', []))}")

    # -----------------------------------------------------------
    # 4) Define Experiments (DEDUP + consistent naming)
    # -----------------------------------------------------------
    experiments: List[ExpCfg] = []
    seen_names = set()

    def add_exp(cfg: ExpCfg):
        if cfg.name in seen_names:
            print(f"[WARN] duplicate exp.name skipped: {cfg.name}")
            return
        seen_names.add(cfg.name)
        experiments.append(cfg)

    # 只保留最小实验集合：
    # 1) scratch_full：非迁移 baseline，便于对比 StageB 迁移是否有效。
    # 2) stageB_prog_unfreeze_l2sp：唯一迁移实验，只使用 progressive unfreezing + L2-SP。
    #    progressive unfreezing 的触发条件在 train_one() 中：
    #    init != scratch、finetune_mode == full、0 < backbone_lr_ratio < 1、progressive_unfreeze=True。
    add_exp(ExpCfg(
        name="scratch_full",
        init="scratch",
        finetune_mode="full",
        phys7_mode=stageB_phys_mode,
        lr=3e-4,
        wd=1e-4,
        l2sp=0.0,
        backbone_lr_ratio=0.1,
        loss_type="huber",
        huber_beta=1.0,
        grad_clip=1.0,
        progressive_unfreeze=False,
    ))

    add_exp(ExpCfg(
        name="stageB_prog_unfreeze_l2sp",
        init="stageB_best",
        finetune_mode="full",
        phys7_mode=stageB_phys_mode,
        lr=6e-4,
        wd=1e-4,
        l2sp=3e-4,
        backbone_lr_ratio=0.05,
        loss_type="huber",
        huber_beta=1.0,
        grad_clip=1.0,
        progressive_unfreeze=True,
    ))

    print(f" [Info] Total Experiments per Seed: {len(experiments)}")
    print(f" [Info] Total Runs: {len(experiments) * int(args.seed_repeats) * len(families_eval)}")

    summary_all = os.path.join(args.out_dir, "summary_allruns.csv")
    summary_best = os.path.join(args.out_dir, "summary_best.csv")

    header_fields = make_summary_header_fields()
    header_line = ",".join(header_fields) + "\n"

    if not os.path.exists(summary_all):
        with open(summary_all, "w", encoding="utf-8") as f:
            f.write(header_line)

    with open(summary_best, "w", encoding="utf-8") as f:
        f.write(header_line)

    done_keys = _load_done_keys(summary_all)
    print(f"[RESUME] already done runs = {len(done_keys)}")

    # -----------------------------------------------------------
    # 5) Run directly on the single fixed dataset
    # -----------------------------------------------------------
    run_datasets(
        datasets=datasets_all,
        raw=raw,
        families_eval=families_eval,
        experiments=experiments,
        stageB_best=stageB_best,
        args=args,
        device=device,
        summary_all=summary_all,
        header_fields=header_fields,
        done_keys=done_keys,
        epochs=args.epochs,
        patience=args.patience,
        tag="fixedsplit",
        key_recipes=key_recipes
    )

    # -----------------------------------------------------------
    # 6) Build summary_best
    # -----------------------------------------------------------
    df_all = pd.read_csv(summary_all)
    rows_all = df_all.to_dict("records")
    best_rows = pick_best_rows(rows_all, key_fields=("split_seed", "drop_tag", "family", "exp"), metric="r2")
    best_rows = sorted(best_rows, key=lambda r: (int(r.get("split_seed", -1)), str(r.get("drop_tag", "")),
                                                 str(r.get("family", "")), str(r.get("exp", ""))))

    with open(summary_best, "w", encoding="utf-8") as f:
        f.write(header_line)
        for row in best_rows:
            vals = []
            for k in header_fields:
                v = row.get(k, "")
                if isinstance(v, (float, np.floating)) and (not np.isfinite(v)):
                    v = ""
                v = str(v).replace(",", " ").replace("\n", " ").replace("\r", " ")
                vals.append(v)
            f.write(",".join(vals) + "\n")

    with open(os.path.join(args.out_dir, "fixed_split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "split_seed": int(args.split_seed),
            "fixed_drop_requested": int(args.fixed_drop),
            "fixed_drop_effective": 0,
            "families_eval": families_eval,
            "key_recipes": key_recipes,
            "test_ratio": float(args.test_ratio),
            "val_ratio": float(args.val_ratio),
            "min_test_points": int(args.min_test_points),
            "train_idx": fixed_dataset.get("train_idx", []),
            "val_idx": fixed_dataset.get("val_idx", []),
            "test_idx": fixed_dataset.get("test_idx", []),
            "ignored_idx": fixed_dataset.get("ignored_idx", []),
            "drop_tag": fixed_dataset.get("drop_tag", "drop0"),
        }, f, indent=2, ensure_ascii=False)

    print("\n[DONE] StageC Fixed-Split Finished.")

if __name__ == "__main__":
    main()

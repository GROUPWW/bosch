# -*- coding: utf-8 -*-
"""
run_stageA_multihead_ablation.py — R2-3a a-v：StageA 单输出头 vs 多头（共享主干, out_dim=7）

对比臂：
  - single-head（现有部署）：runs_stageA_phys7/best_by_test 的 7 个独立 out_dim=1 头
    （transformer，split2026，由既有流水线按 R2_min_test 选出 train_seed=1）
  - multi-head（本脚本新训）：同一数据集、同一 split_indices.json（保证同一 test 集）、
    同一超参/预算（Cfg: d_model=64, nhead=4, L=2, ff=128, do=0.1, lr=1e-3, wd=1e-3,
    batch=64, 200ep, warmup10+cosine, clip1.0, per-channel 输出归一化），
    共享主干 out_dim=7，train_seeds=[0..4]，按 val r2_min 选代表 seed（不看 test）。

产出：runs_stageA_multihead_ablation/
  - seed{0..4}/multihead_best.pth           各 seed 的最优（按 val r2_min）checkpoint
  - per_seed_test_r2.csv                    每 seed × 每描述符的 test R2
  - summary.csv                             per-descriptor 对比表（single vs multi-head）
"""
import os, sys, json, time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import stageA_train_phys_pycharm as sa
from physio_util import set_seed, get_default_device, metrics

OUT_ROOT = os.path.join(_HERE, "runs_stageA_multihead_ablation")
SINGLE_ROOT = os.path.join(_HERE, "runs_stageA_phys7", "best_by_test")
MODEL_TYPE = "transformer"
TRAIN_SEEDS = [0, 1, 2, 3, 4]


def train_multihead_one_seed(dataset, meta, tr_idx, va_idx, seed: int, out_dir: str):
    """镜像 train_stageA_phys7_singleheads 的训练循环，区别仅在于整个 (N,7,1) 一起训。"""
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    dev = get_default_device()
    Cfg = sa.Cfg

    tr_loader = DataLoader(Subset(dataset, tr_idx), batch_size=Cfg.batch, shuffle=True)
    va_loader = DataLoader(Subset(dataset, va_idx), batch_size=Cfg.batch, shuffle=False)

    T = int(meta["T"])
    K = len(meta["families"])
    model = sa.build_stageA_model(T=T, out_dim=K, model_type=MODEL_TYPE).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=Cfg.lr, weight_decay=Cfg.weight_decay)
    sch = sa.make_warmup_cosine(opt, Cfg.max_epochs, Cfg.warmup_epochs, use_cosine=Cfg.use_cosine)
    loss_fn = nn.SmoothL1Loss(reduction="none")

    y_mean_t = y_std_t = None
    if getattr(Cfg, "use_output_norm", False):
        _, Y_all, M_all, _ = dataset.tensors
        y_mean, y_std = sa._compute_y_norm_stats(Y_all[tr_idx], M_all[tr_idx])  # (K,)
        y_mean_t = y_mean.to(dev).view(1, K, 1)
        y_std_t = y_std.to(dev).view(1, K, 1)

    best_r2min = -1e18
    best_path = os.path.join(out_dir, "multihead_best.pth")
    best_val_r2 = None

    for e in range(1, Cfg.max_epochs + 1):
        model.train()
        for Xn, Y, M, tvals in tr_loader:
            Xn, Y, M, tvals = Xn.to(dev), Y.to(dev), M.to(dev), tvals.to(dev)
            pred = model(Xn, tvals)  # (B,7,1)
            if y_mean_t is not None:
                pred_l = (pred - y_mean_t) / y_std_t
                y_l = (Y - y_mean_t) / y_std_t
            else:
                pred_l, y_l = pred, Y
            loss_e = loss_fn(pred_l, y_l)  # (B,7,1)
            w = M.float()
            loss = (loss_e * w).sum() / w.sum().clamp_min(1e-6)
            opt.zero_grad()
            loss.backward()
            if Cfg.clip_grad_norm and Cfg.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), Cfg.clip_grad_norm)
            opt.step()
        sch.step()

        model.eval()
        preds, trues, masks = [], [], []
        with torch.no_grad():
            for Xn, Y, M, tvals in va_loader:
                Xn, tvals = Xn.to(dev), tvals.to(dev)
                preds.append(model(Xn, tvals).detach().cpu())
                trues.append(Y)
                masks.append(M)
        r2 = metrics(torch.cat(preds, 0), torch.cat(trues, 0), torch.cat(masks, 0))["R2"][:, 0]  # (7,)
        r2min = float(np.nanmin(r2))

        if e % 10 == 0 or e == 1:
            print(f"[MH][seed{seed}][{e}/{Cfg.max_epochs}] val r2_min={r2min:.4f}", flush=True)

        if np.isfinite(r2min) and r2min > best_r2min:
            best_r2min = r2min
            best_val_r2 = r2.copy()
            torch.save({
                "model": model.state_dict(), "meta": meta, "model_type": MODEL_TYPE,
                "train_seed": int(seed), "out_dim": K, "best_val_r2_min": float(best_r2min),
            }, best_path)

    print(f"[MH][seed{seed}] best val r2_min={best_r2min:.4f}", flush=True)
    return best_path, float(best_r2min), best_val_r2


@torch.no_grad()
def eval_multihead_test(dataset, meta, ckpt_path: str, te_idx):
    dev = get_default_device()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    T = int(meta["T"]); K = len(meta["families"])
    model = sa.build_stageA_model(T=T, out_dim=K, model_type=ck.get("model_type", MODEL_TYPE)).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    loader = DataLoader(Subset(dataset, te_idx), batch_size=sa.Cfg.batch, shuffle=False)
    preds, trues, masks = [], [], []
    for Xn, Y, M, tvals in loader:
        Xn, tvals = Xn.to(dev), tvals.to(dev)
        preds.append(model(Xn, tvals).detach().cpu())
        trues.append(Y)
        masks.append(M)
    r2 = metrics(torch.cat(preds, 0), torch.cat(trues, 0), torch.cat(masks, 0))["R2"][:, 0]
    return r2  # (7,)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    t0 = time.time()

    # 1) 数据集 + 与 single-head 完全一致的 split（直接读 best_by_test/split_indices.json）
    dataset, meta, aux = sa.excel_to_phys7_dataset(
        sa.Cfg.case_excel, sa.Cfg.sheet_name, sa.Cfg.case_id_col, sa.Cfg.iedf_root)
    with open(os.path.join(SINGLE_ROOT, "split_indices.json"), "r", encoding="utf-8") as f:
        sp = json.load(f)
    tr_idx, va_idx, te_idx = sp["train"], sp["val"], sp["test"]
    # 校验：与 split_seed=2026 的确定性生成一致（防呆）
    exp = sa.make_shared_split_indices_3way(len(dataset), sa.Cfg.val_ratio, sa.Cfg.test_ratio, 2026)
    same = all(list(a) == list(b) for a, b in zip(exp, (tr_idx, va_idx, te_idx)))
    print(f"[SPLIT] best_by_test/split_indices.json 与 split_seed=2026 生成一致: {same} "
          f"(N={len(dataset)}, train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)})", flush=True)

    fams = list(meta["families"])

    # 2) single-head 臂：部署版 7 单头的 per-descriptor test R2
    yhat_s, ytrue, mask = sa.infer_multihead_phys7(dataset, meta, SINGLE_ROOT, MODEL_TYPE)
    r2_single_test = metrics(yhat_s[te_idx], ytrue[te_idx], mask[te_idx])["R2"][:, 0]
    with open(os.path.join(SINGLE_ROOT, "heads_metrics.json"), "r", encoding="utf-8") as f:
        hm = json.load(f)
    r2_single_val = np.array([h["best_val_r2"] for h in hm["per_head"]], dtype=np.float64)
    print("[SINGLE] per-descriptor test R2:", np.round(r2_single_test, 4).tolist(), flush=True)

    # 3) multi-head 臂：5 个 train seeds
    rows = []
    per_seed = {}
    for seed in TRAIN_SEEDS:
        ckpt_path, val_r2min, val_r2 = train_multihead_one_seed(
            dataset, meta, tr_idx, va_idx, seed, os.path.join(OUT_ROOT, f"seed{seed}"))
        r2_test = eval_multihead_test(dataset, meta, ckpt_path, te_idx)
        per_seed[seed] = {"val_r2min": val_r2min, "test_r2": r2_test}
        for k, fam in enumerate(fams):
            rows.append({"seed": seed, "descriptor": fam,
                         "val_r2": float(val_r2[k]), "test_r2": float(r2_test[k])})
        print(f"[MH][seed{seed}] test r2_min={float(np.min(r2_test)):.4f} "
              f"per-desc={np.round(r2_test, 4).tolist()}", flush=True)

    df_seed = pd.DataFrame(rows)
    df_seed.to_csv(os.path.join(OUT_ROOT, "per_seed_test_r2.csv"), index=False, encoding="utf-8-sig")

    # 4) 代表 seed = val r2_min 最大（不看 test）；同时给 5 seed 的 mean±std
    best_seed = max(per_seed, key=lambda s: per_seed[s]["val_r2min"])
    r2_mh_best = per_seed[best_seed]["test_r2"]
    r2_mh_all = np.stack([per_seed[s]["test_r2"] for s in TRAIN_SEEDS], 0)  # (5,7)
    print(f"[MH] representative seed (by val r2_min) = {best_seed} "
          f"(val r2_min={per_seed[best_seed]['val_r2min']:.4f})", flush=True)

    summary = pd.DataFrame({
        "descriptor": fams,
        "single_val_r2": r2_single_val,
        "single_test_r2": r2_single_test,
        "mh_test_r2_best_seed": r2_mh_best,
        "mh_test_r2_mean5": r2_mh_all.mean(axis=0),
        "mh_test_r2_std5": r2_mh_all.std(axis=0),
        "delta_best_minus_single": r2_mh_best - r2_single_test,
        "delta_mean_minus_single": r2_mh_all.mean(axis=0) - r2_single_test,
    })
    summary.to_csv(os.path.join(OUT_ROOT, "summary.csv"), index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print("\n===== per-descriptor test R2: single-head vs multi-head =====")
    print(summary.round(4).to_string(index=False))
    print(f"\nr2_min(test): single={float(np.min(r2_single_test)):.4f}  "
          f"mh_best={float(np.min(r2_mh_best)):.4f}  mh_mean={float(np.min(r2_mh_all.mean(axis=0))):.4f}")
    print(f"r2_mean(test): single={float(np.mean(r2_single_test)):.4f}  "
          f"mh_best={float(np.mean(r2_mh_best)):.4f}  mh_mean={float(np.mean(r2_mh_all.mean(axis=0))):.4f}")
    dt = time.time() - t0
    print(f"\n[TIME] total {dt/60:.1f} min ({len(TRAIN_SEEDS)} multi-head seeds × {sa.Cfg.max_epochs} epochs)")
    print(f"[OUT] {os.path.join(OUT_ROOT, 'summary.csv')}")


if __name__ == "__main__":
    main()

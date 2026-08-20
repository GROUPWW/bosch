# -*- coding: utf-8 -*-
"""
AE-7：在 156 个完整 IEDF case 的形状向量上训练一个小型 MLP autoencoder，
取 7 维 bottleneck 作为 learned IEDF 潜变量（a-iv 消融的 ae7 臂）。

复用 build_pca7_156.py 的特征提取（同样的能量网格 / 离子拼接 / 标准化），
输入维度 = 128*4 + 64*2 = 640。输出 case_with_ae7_156.xlsx（156 行 + ae_1..7）。
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

import build_pca7_156 as F  # 复用特征提取

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_XLSX = os.path.join(_HERE, "case_with_ae7_156.xlsx")
OUT_META = os.path.join(_HERE, "ae7_156_manifest.json")

LATENT_K = 7
HIDDEN = 128
EPOCHS = 3000
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 0


class IEDFAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, latent: int = LATENT_K, hidden: int = HIDDEN):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def extract_feature_matrix():
    """与 build_pca7_156.main 相同的提取流程，返回 (X, ok_case, grids_info)。"""
    df_case = pd.read_excel(F.CASE_XLSX, sheet_name=F.CASE_SHEET)
    case_ids = df_case[F.CASE_ID_COL].astype(str).map(F.normalize_case_id).tolist()

    grids = {key: None for key in F.TARGETS}
    for cid in case_ids:
        files = F.read_target_files_for_case(cid)
        for key in F.TARGETS:
            if grids[key] is None and key in files:
                grids[key] = F.make_energy_grid_from_file(files[key], F.NGRID[key])
        if all(grids[k] is not None for k in grids):
            break
    for k, g in grids.items():
        if g is None:
            raise RuntimeError(f"找不到任何 {k} 的 IEDF 文件。")

    X_list, ok_case = [], []
    for cid in case_ids:
        files = F.read_target_files_for_case(cid)
        parts = []
        for key in F.TARGETS:  # 固定顺序
            if key not in files:
                raise RuntimeError(f"case {cid} 缺 {key[0]}_{key[1]} IEDF 文件。")
            v, _ = F.ion_shape_vector(files[key], key[0], key[1], F.TARGETS[key], grids[key])
            if v is None:
                raise RuntimeError(f"case {cid} 的 {files[key]} 无法解析。")
            parts.append(v)
        X_list.append(np.concatenate(parts, axis=0))
        ok_case.append(cid)

    X = np.vstack(X_list)
    assert X.shape[0] == 156, f"期望 156，实际 {X.shape[0]}"
    return df_case, case_ids, X


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df_case, case_ids, X = extract_feature_matrix()

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X).astype(np.float32)

    device = "cpu"  # 156x640 全批量，CPU 足够快且数值稳定
    x_t = torch.from_numpy(Xs).to(device)

    model = IEDFAutoEncoder(in_dim=Xs.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    for ep in range(1, EPOCHS + 1):
        model.train()
        recon, _ = model(x_t)
        loss = loss_fn(recon, x_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if ep % 500 == 0:
            print(f"[AE] epoch {ep:4d}/{EPOCHS}  recon MSE = {loss.item():.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        recon, z = model(x_t)
        mse = float(loss_fn(recon, x_t).item())
        sst = float(torch.sum((x_t - x_t.mean(dim=0, keepdim=True)) ** 2).item())
        sse = float(torch.sum((recon - x_t) ** 2).item())
        recon_r2 = 1.0 - sse / max(sst, 1e-12)

    Z = z.cpu().numpy().astype(np.float64)  # (156, 7)

    df_ae = pd.DataFrame(Z, columns=[f"ae_{i+1}" for i in range(LATENT_K)])
    df_ae["case_id"] = case_ids

    df_out = df_case.copy()
    df_out["_case_id_tmp_"] = df_out[F.CASE_ID_COL].astype(str).map(F.normalize_case_id)
    df_out = df_out.merge(df_ae, how="left", left_on="_case_id_tmp_", right_on="case_id")
    df_out = df_out.drop(columns=["_case_id_tmp_", "case_id"])

    assert len(df_out) == 156 and df_out[[f"ae_{i+1}" for i in range(LATENT_K)]].notna().all().all()
    df_out.to_excel(OUT_XLSX, index=False)

    meta = {
        "case_xlsx": F.CASE_XLSX,
        "in_dim": int(Xs.shape[1]),
        "latent_k": LATENT_K,
        "hidden": HIDDEN,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "final_recon_mse": mse,
        "recon_r2_train": float(recon_r2),
        "n_samples_used": int(X.shape[0]),
        "out_xlsx": OUT_XLSX,
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {OUT_XLSX}  rows={len(df_out)}")
    print(f"[OK] Meta : {OUT_META}")
    print(f"[INFO] AE final recon MSE = {mse:.6f}  train recon R2 = {recon_r2:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU 冒烟：验证 pca7 / ae7 接线。
同一批 case 下，phys7_true / pca7 / ae7 三种 phys_source 送进模型的 phys7_seq 张量：
  1) shape 全为 (N, 7, T)；
  2) 三者两两不同（同一 case 索引处）；
  3) 沿 T 是静态复制的（pca7/ae7 每 case 只有 7 维）；
  4) static recipe 流和 targets 在三臂间完全一致（唯一变量是描述符）。
"""
import os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stageB_train_morph_on_phys7_pycharm import Cfg, prepare_shared_cache
from stageB_util import build_morph_dataset_phys7
from run_stageB_iedf_ablation import CASE_XLSX_156, build_latent_cache


def build_ds(phys_source, phys7_seq_full, shared):
    (df_cache, recipe_cols, recipe_raw, provider,
     targets_full, mask_full) = shared
    ds, meta = build_morph_dataset_phys7(
        Cfg.excel_path, Cfg.sheet_name, Cfg.case_id_col,
        target_family="h0", phys_source=phys_source,
        recipe_aug_mode="time", phys7_mode="full",
        df=df_cache, recipe_cols=recipe_cols, recipe_raw=recipe_raw,
        targets_full=targets_full, mask_full=mask_full,
        phys7_seq_full=phys7_seq_full, stageA_provider=None,
    )
    return ds, meta


def main():
    Cfg.excel_path = CASE_XLSX_156
    Cfg.phys_sources = ["none"]
    Cfg.device = "cpu"  # 冒烟固定 CPU

    (df_cache, recipe_cols, recipe_raw, provider,
     targets_full, mask_full, phys7_seq_cache, case_ids,
     bad_row, before_N, kept_idx, clipped) = prepare_shared_cache()

    latent = build_latent_cache(case_ids)
    shared = (df_cache, recipe_cols, recipe_raw, provider, targets_full, mask_full)

    ds_phys, meta_phys = build_ds("phys7_true", None, shared)
    ds_pca, meta_pca = build_ds("pca7", latent["pca7"], shared)
    ds_ae, meta_ae = build_ds("ae7", latent["ae7"], shared)

    s_phys, p_phys = ds_phys.tensors[0], ds_phys.tensors[1]
    s_pca, p_pca = ds_pca.tensors[0], ds_pca.tensors[1]
    s_ae, p_ae = ds_ae.tensors[0], ds_ae.tensors[1]

    N = len(df_cache)
    T = p_phys.shape[2]
    print(f"[smoke] N={N}  phys7_seq shapes: phys7={tuple(p_phys.shape)} "
          f"pca7={tuple(p_pca.shape)} ae7={tuple(p_ae.shape)}")

    # 1) shape
    assert p_phys.shape == (N, 7, T) and p_pca.shape == (N, 7, T) and p_ae.shape == (N, 7, T)

    # 2) 三者两两不同（整体 + 随机抽同一 case）
    assert not torch.allclose(p_phys, p_pca), "phys7 与 pca7 相同？！"
    assert not torch.allclose(p_phys, p_ae), "phys7 与 ae7 相同？！"
    assert not torch.allclose(p_pca, p_ae), "pca7 与 ae7 相同？！"
    i = 5
    print(f"[smoke] case[{i}]={case_ids[i]}")
    print(f"        phys7[:, 0] = {p_phys[i, :, 0].numpy().round(3)}")
    print(f"        pca7 [:, 0] = {p_pca[i, :, 0].numpy().round(3)}")
    print(f"        ae7  [:, 0] = {p_ae[i, :, 0].numpy().round(3)}")

    # 3) 沿 T 静态复制
    for name, p in [("phys7", p_phys), ("pca7", p_pca), ("ae7", p_ae)]:
        assert torch.allclose(p[:, :, 0], p[:, :, -1]), f"{name} 沿 T 不静态"

    # 4) static recipe / targets 三臂一致
    assert torch.allclose(s_phys, s_pca) and torch.allclose(s_phys, s_ae)
    assert torch.allclose(ds_phys.tensors[2], ds_pca.tensors[2])
    assert torch.equal(ds_phys.tensors[3], ds_pca.tensors[3])

    # 5) 无 NaN
    for name, p in [("phys7", p_phys), ("pca7", p_pca), ("ae7", p_ae)]:
        assert torch.isfinite(p).all(), f"{name} 含 NaN/inf"

    print("[smoke] ALL CHECKS PASSED ✅")


if __name__ == "__main__":
    main()

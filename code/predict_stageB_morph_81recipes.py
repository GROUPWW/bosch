# -*- coding: utf-8 -*-
"""用 paper-aligned stageB 模型预测 20260710—刻蚀补充配方.xlsx 的 81 条新配方的刻蚀形貌。

方法：复用 stageC 推理管线。
- 归一化统计（recipe / phys7 / 各 family 目标）来自 Bosch_38_B（42 条实测），与 stageC 评估口径一致
- 每条新配方：7 维 recipe（不含温度）→ augment → stageA 出 phys7 → stageB 各 family best.pth 前向 → 反归一化到 nm
- 输出 9 个 cycle 的完整形貌剖面 + zmin/h1/d1/w 关键时间点摘要

注意：stageB 不含温度特征。新配方中温度=-20/0 的 9 条按 -10°C（训练口径）预测，已在输出标注。
"""
from __future__ import annotations
import os, sys, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import torch
import stageC_paper as sc
import physio_util as pu
import stageB_util as su

PRED_FAMS = ["zmin", "h1", "d1", "w"]
# 关键时间点（cycle 标签 -> TIME_LIST 索引）
KEY_T = {"zmin": ["9"], "w": ["1", "3", "5", "9"], "d1": ["3", "5", "9"], "h1": ["3", "5", "9"]}


def build_norm_stats(raw_ref):
    s_mean, s_std = sc.zfit(raw_ref["static_raw"])
    p_mean, p_std = sc.zfit(raw_ref["phys7_raw_full"])
    return s_mean, s_std, p_mean, p_std


def predict_family(fam, raw_ref, raw_new, norm, ckpt, device, stageB_aug_mode, stageB_phys_mode, model_type):
    s_mean, s_std, p_mean, p_std = norm
    k = sc.family_to_index(fam)
    y_mean_t, y_std_t = sc.zfit_targets_masked_1fam(raw_ref["y_raw"][:, k, :], raw_ref["mask"][:, k, :])

    static = sc.apply_z(raw_new["static_raw"], s_mean, s_std)
    phys7_z = sc.apply_z(raw_new["phys7_raw_full"], p_mean, p_std)
    phys7_z = sc.apply_phys7_mode(phys7_z, stageB_phys_mode)
    T_len = raw_new["time_mat"].shape[1]
    phys7_seq = np.tile(phys7_z[:, :, None], (1, 1, T_len)).astype(np.float32)

    N = static.shape[0]
    y_dummy = np.zeros((N, 1, T_len), np.float32)
    m_dummy = np.zeros((N, 1, T_len), bool)
    loader = sc.make_loader(static, phys7_seq, y_dummy, m_dummy, raw_new["time_mat"],
                            np.arange(N, dtype=int), 64, False, 0)

    model = sc.build_model(model_type, static_dim=static.shape[1], out_dim=1).to(device)
    sc.load_ckpt_into_model(model, ckpt)
    pack = sc.eval_pack(model, loader, device)  # pred (N,1,T) normalized

    pred = torch.from_numpy(pack["pred"]).float()
    mean = torch.from_numpy(y_mean_t).view(1, 1, -1)
    std = torch.from_numpy(y_std_t).view(1, 1, -1)
    pred_um = pred * std + mean
    sign_map, _ = su._default_family_sign_and_nonneg([fam])
    family_sign = torch.tensor([sign_map[fam]], dtype=torch.float32)
    pred_disp, _ = pu.transform_for_display(
        pred_um, pred_um, family_sign=family_sign, clip_nonneg=True,
        nonneg_families=[0], unit_scale=1000.0, flip_sign=False, min_display_value=None)
    return pred_disp[:, 0, :].cpu().numpy()  # (N, T) in nm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_excel", default=os.path.join(_HERE, "Bosch_38_B.xlsx"))
    ap.add_argument("--new_excel", default=os.path.join(_HERE, "Bosch_predict81.xlsx"))
    ap.add_argument("--temp_map", default=os.path.join(_HERE, "predict81_temp_map.csv"))
    ap.add_argument("--stageB_runs_root", default=os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA"))
    ap.add_argument("--stageA_heads_root", default=os.path.join(_HERE, "runs_stageA_phys7", "best_by_test"))
    ap.add_argument("--out_csv", default=os.path.join(_HERE, "predict81_morphology_stageB.csv"))
    ap.add_argument("--out_summary", default=os.path.join(_HERE, "predict81_morphology_summary.csv"))
    ap.add_argument("--height_family", default="h1")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    device = args.device if args.device else sc.su.get_default_device()
    best_conf = sc.load_best_config_common(args.stageB_runs_root)
    sc.sb.apply_hp_from_best_conf_to_cfg(best_conf)
    stageB_aug_mode = best_conf.get("recipe_aug_mode", "time")
    stageB_phys_mode = best_conf.get("phys7_mode", "full")
    model_type = best_conf.get("model_type", "transformer")
    ckpts = sc.resolve_stageB_best_ckpts_from_common(args.stageB_runs_root)
    print(f"[Predict81] device={device} tf_layers={sc.sb.Cfg.tf_layers} model={model_type}")

    print("[Predict81] build raw_ref (Bosch_38_B) for normalization ...")
    raw_ref = sc.build_stageC_raw(device=device, new_excel=args.ref_excel,
                                  height_family=args.height_family,
                                  recipe_aug_mode=stageB_aug_mode, stageA_heads_root=args.stageA_heads_root)
    print("[Predict81] build raw_new (81 recipes) ...")
    raw_new = sc.build_stageC_raw(device=device, new_excel=args.new_excel,
                                  height_family=args.height_family,
                                  recipe_aug_mode=stageB_aug_mode, stageA_heads_root=args.stageA_heads_root)
    norm = build_norm_stats(raw_ref)

    recipe_ids = [str(x) for x in raw_new["recipe_ids"]]
    # 直接用 new_excel 的 配方名 列作为标签（build_stageC_raw 可能回退成 rowXXXX）
    try:
        df_new = pd.read_excel(args.new_excel)
        if "配方名" in df_new.columns and len(df_new) == len(recipe_ids):
            recipe_ids = [str(x).strip() for x in df_new["配方名"].tolist()]
    except Exception:
        pass
    TIME_LIST = sc.TIME_LIST
    TIME_VALUES = sc.TIME_VALUES

    # temperature map
    temp_map = {}
    if os.path.exists(args.temp_map):
        tdf = pd.read_csv(args.temp_map)
        temp_map = dict(zip(tdf["配方号"].astype(str), tdf["温度"]))

    long_rows = []
    summary = {}
    for fam in PRED_FAMS:
        pred_nm = predict_family(fam, raw_ref, raw_new, norm, ckpts[fam], device,
                                 stageB_aug_mode, stageB_phys_mode, model_type)
        for i, rid in enumerate(recipe_ids):
            for ti, tl in enumerate(TIME_LIST):
                long_rows.append({"recipe": rid, "family": fam, "cycle": tl,
                                  "cycle_value": float(TIME_VALUES[ti]),
                                  "pred_nm": float(pred_nm[i, ti])})
            # summary key timesteps
            for tl in KEY_T[fam]:
                ti = TIME_LIST.index(tl)
                summary.setdefault(rid, {})[f"{fam}@cycle{tl}_nm"] = float(pred_nm[i, ti])

    df_long = pd.DataFrame(long_rows)
    df_long.insert(1, "temp_C", df_long["recipe"].map(temp_map))
    df_long.to_csv(args.out_csv, index=False)

    df_sum = pd.DataFrame([{"recipe": rid, **summary[rid]} for rid in recipe_ids])
    df_sum.insert(1, "temp_C", df_sum["recipe"].map(temp_map))
    df_sum.to_csv(args.out_summary, index=False)

    print(f"\n[Predict81] wrote {args.out_csv} ({len(df_long)} rows)")
    print(f"[Predict81] wrote {args.out_summary} ({len(df_sum)} recipes)")
    print("\n=== summary (key timesteps, nm) ===")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(df_sum.to_string(index=False))
    non10 = df_sum[df_sum["temp_C"] != -10]
    if len(non10):
        print(f"\n[WARN] {len(non10)} recipes at temp != -10C (stageB has no temp feature, predicted as -10C):")
        print(non10[["recipe", "temp_C"]].to_string(index=False))


if __name__ == "__main__":
    main()

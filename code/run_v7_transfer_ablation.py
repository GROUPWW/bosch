#!/usr/bin/env python3
"""
v7 · R2-3c 迁移方法消融（最终口径重跑）

设计（与 docs/06_reviewer_response_guide.md 一致）：
- 7 变体：c-i sim_only(ep0) / c-ii scratch / c-iii FT无L2SP / c-iv FT无prog / c-v 仅prog / c-vi 仅L2SP / c-vii 完整
  注：c-iv(FT无prog, l2sp>0) 与 c-vi(仅L2SP, prog=False) 配置相同，合并为一跑（报告中 c-vi=c-iv）。
- mode=full：prog/L2-SP 组件只在 full 模式有意义；head_ln 的更优结果在主表中单独报告。
- lr 取各 family 历史 full 模式最优：zmin 1e-5 / h1 1e-4 / d1 1e-4 / w 1e-5；l2sp 适用时 0.1；br=0.1；epochs=500。
- P-fixed 口径，各 family 用其钻石数据子集与划分：
  zmin: Bosch_zmin_select_aug.xlsx + 固定8条(ss2026) + val=train
  w:    Bosch_38_B.xlsx + 固定8条(ss2026) + val=train
  h1:   aug_v14_109 arm=base42, ss184, val 0.1
  d1:   aug_v14_109 arm=compat85, ss177, val 0.1
"""
import os, sys, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import stageC_paper as sc
import stageB_util as su
from run_stageC_partitioned_Phase1 import build_arm_split

FIXED_TEST_RECIPES = ["B17", "B12", "B36", "B70", "B26", "B27", "B30", "B45"]

# 变体：(tag, init, l2sp, prog, epochs)
VARIANTS = [
    ("c-i_sim_only",   "stageB_best", 0.0, False, 0),
    ("c-ii_scratch",   "scratch",     0.0, False, 500),
    ("c-iii_FT_noL2SP","stageB_best", 0.0, False, 500),
    ("c-iv_FT_noprog", "stageB_best", 0.1, False, 500),   # = c-vi 仅L2SP
    ("c-v_prog_only",  "stageB_best", 0.0, True,  500),
    ("c-vii_full",     "stageB_best", 0.1, True,  500),
]

FAM_CFG = {
    "zmin": {"lr": 1e-5, "file": "Bosch_zmin_select_aug.xlsx", "split": "fixed8", "ss": 2026},
    "w":    {"lr": 1e-5, "file": "Bosch_38_B.xlsx",          "split": "fixed8", "ss": 2026},
    "h1":   {"lr": 1e-4, "file": "Bosch_aug_v14_109.xlsx",   "split": "arm", "arm": "base42",  "ss": 184},
    "d1":   {"lr": 1e-4, "file": "Bosch_aug_v14_109.xlsx",   "split": "arm", "arm": "compat85", "ss": 177},
}


def fixed8_split(raw, fam):
    recipe_ids = raw["recipe_ids"]
    test_idx = [i for i, rid in enumerate(recipe_ids) if rid in FIXED_TEST_RECIPES]
    remaining = [i for i in range(len(recipe_ids)) if i not in test_idx]
    return {"train_idx": remaining, "val_idx": list(remaining), "test_idx": test_idx,
            "ignored_idx": [], "seed": 2026, "drop_tag": "v7fixed8"}


def main():
    out_dir = os.path.join(_HERE, "runs_stageC_v7_transfer_ablation")
    sc.ensure_dir(out_dir)
    device = su.get_default_device()
    print(f"[v7] device={device}  R2-3c 迁移消融, {len(FAM_CFG)} fam × {len(VARIANTS)} 变体")

    sroot = os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA")
    stageB_best = sc.resolve_stageB_best_ckpts_from_common(sroot)
    best_conf = sc.load_best_config_common(sroot)
    stageB_phys_mode = best_conf.get("phys7_mode", "full")
    stageA = os.path.join(_HERE, "runs_stageA_phys7", "best_by_test")

    summary = os.path.join(out_dir, "summary_v7.csv")
    hf = sc.make_summary_header_fields()
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as f:
            f.write(",".join(hf) + "\n")

    import pandas as pd
    for fam, fc in FAM_CFG.items():
        raw = sc.build_stageC_raw(device=device, new_excel=os.path.join(_HERE, fc["file"]),
                                  height_family="h1", recipe_aug_mode="time",
                                  stageA_heads_root=stageA)
        if fc["split"] == "fixed8":
            sp = fixed8_split(raw, fam)
        else:
            df_109 = pd.read_excel(os.path.join(_HERE, fc["file"]))
            df_compat = pd.read_excel(os.path.join(_HERE, "Bosch_planB_compatible.xlsx"))
            rid_109 = [c for c in df_109.columns if "配方" in str(c)][0]
            rid_compat = [c for c in df_compat.columns if "配方" in str(c)][0]
            ids_109 = df_109[rid_109].astype(str).tolist()
            ids_compat = set(df_compat[rid_compat].astype(str).tolist())
            extra_idx = [i for i, s in enumerate(ids_109) if s in ids_compat and not (s.startswith("B") and int(s[1:]) < 80)]
            orig_n = sum(1 for s in ids_109 if s.startswith("B") and int(s[1:]) < 80)
            sp = build_arm_split(raw, orig_n, extra_idx, fc["ss"], 0.2, 0.1, 1, fam, fc["arm"])

        run_args = argparse.Namespace(
            out_dir=out_dir, stageA_heads_root=stageA,
            recipe_aug_mode="time", model_type="transformer", batch=4,
            patience=99999, num_workers=0, seed=2026, seed_repeats=1,
            test_ratio=0.2, val_ratio=0.1, min_test_points=1)

        for tag, init, l2sp, prog, epochs in VARIANTS:
            name = f"v7_{tag}_{fam}"
            exp = sc.ExpCfg(
                name=name, init=init, finetune_mode="full", phys7_mode=stageB_phys_mode,
                lr=fc["lr"], wd=1e-4, l2sp=l2sp, backbone_lr_ratio=0.1,
                loss_type="huber", huber_beta=1.0, grad_clip=1.0, progressive_unfreeze=prog)
            dk = sc._load_done_keys(summary)
            sc.run_datasets(datasets=[sp], raw=raw, families_eval=[fam],
                            experiments=[exp], stageB_best=stageB_best, args=run_args,
                            device=device, summary_all=summary, header_fields=hf,
                            done_keys=dk, epochs=epochs, patience=99999,
                            tag="v7", key_recipes=[])
            r = pd.read_csv(summary).iloc[-1]
            print(f"[v7 RESULT] {fam:5s} {tag:16s} R2={r.r2:+.4f} best_ep={r.best_epoch}")

    df = pd.read_csv(summary)
    piv = df.pivot_table(index="exp", columns="family", values="r2")
    piv.index = [e.replace("v7_", "").rsplit("_", 1)[0] for e in piv.index]
    print("\n[v7 汇总 R² 矩阵]")
    print(piv.round(4).to_string())


if __name__ == "__main__":
    main()

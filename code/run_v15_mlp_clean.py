#!/usr/bin/env python3
"""v10 · R2-3b 关键补证：mlp stageB 权重 → StageC 迁移（对比 transformer 的 0.8254/0.7802）。"""
import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import stageC_paper as sc
import stageB_util as su
from run_v7_transfer_ablation import fixed8_split
from run_stageC_partitioned_Phase1 import build_arm_split

ABL = os.path.join(_HERE, "runs_stageB_backbone_ablation_v9")

RUNS = [
    # (fam, mode, lr, l2sp, br, epochs, split_tag, tf_reference, backbone)
    ("zmin", "head_ln", 1e-6, 10.0, 0.1, 3000, "zmin_sel", None, "tf"),
    ("zmin", "head_ln", 1e-6, 10.0, 0.1, 3000, "zmin_sel", None, "mlp"),
    ("d1",   "head_ln", 1e-4, 0.0, 0.01, 2000, "arm_compat85", 0.7921, "mlp"),
]
EXCLUDE = {"B143", "B166"}
SROOT_TF = os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA")


def mlp_ckpt(fam):
    return os.path.join(ABL, f"mlp_lr0p001_wd0p0001_do0p1_dm256_L2_hb0p1_stA_t_full_{fam}_s3", "best.pth")


def main():
    out_dir = os.path.join(_HERE, "runs_stageC_v15_mlp_clean")
    sc.ensure_dir(out_dir)
    device = su.get_default_device()
    print(f"[v10] device={device}  mlp stageB → StageC 迁移对比")

    sroot = os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA")
    best_conf = sc.load_best_config_common(sroot)
    stageB_phys_mode = best_conf.get("phys7_mode", "full")
    stageA = os.path.join(_HERE, "runs_stageA_phys7", "best_by_test")

    summary = os.path.join(out_dir, "summary_v15.csv")
    hf = sc.make_summary_header_fields()
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as f:
            f.write(",".join(hf) + "\n")

    import pandas as pd
    for fam, mode, lr, l2sp, br, epochs, split_tag, tf_ref, backbone in RUNS:
        file = {"fixed8": "Bosch_38_B.xlsx", "zmin_sel": "Bosch_zmin_select_aug_v2.xlsx"}.get(split_tag, "Bosch_aug_v14_109.xlsx")
        raw = sc.build_stageC_raw(device=device, new_excel=os.path.join(_HERE, file),
                                  height_family="h1", recipe_aug_mode="time",
                                  stageA_heads_root=stageA)
        if split_tag in ("fixed8", "zmin_sel"):
            sp = fixed8_split(raw, fam)
        else:
            df_109 = pd.read_excel(os.path.join(_HERE, file))
            df_compat = pd.read_excel(os.path.join(_HERE, "Bosch_planB_compatible.xlsx"))
            rid_109 = [c for c in df_109.columns if "配方" in str(c)][0]
            rid_compat = [c for c in df_compat.columns if "配方" in str(c)][0]
            ids_109 = df_109[rid_109].astype(str).tolist()
            ids_compat = set(df_compat[rid_compat].astype(str).tolist())
            extra_idx = [i for i, s in enumerate(ids_109) if s in ids_compat and not (s.startswith("B") and int(s[1:]) < 80)]
            orig_n = sum(1 for s in ids_109 if s.startswith("B") and int(s[1:]) < 80)
            sp = build_arm_split(raw, orig_n, extra_idx, 177, 0.2, 0.1, 1, fam, "compat85")
            rid = raw["recipe_ids"]
            sp["train_idx"] = [i for i in sp["train_idx"] if rid[i] not in EXCLUDE]

        stageB_best = ({fam: mlp_ckpt(fam)} if backbone == "mlp"
                       else sc.resolve_stageB_best_ckpts_from_common(SROOT_TF))
        exp = sc.ExpCfg(
            name=f"v15b_{backbone}_{fam}_{mode}_lr{lr:.0e}_ep{epochs}",
            init="stageB_best", finetune_mode=mode, phys7_mode=stageB_phys_mode,
            lr=lr, wd=1e-4, l2sp=l2sp, backbone_lr_ratio=br,
            loss_type="huber", huber_beta=1.0, grad_clip=1.0, progressive_unfreeze=True)
        run_args = argparse.Namespace(
            out_dir=out_dir, stageA_heads_root=stageA,
            recipe_aug_mode="time", model_type=("transformer" if backbone == "tf" else "mlp"), batch=4,
            patience=99999, num_workers=0, seed=2026, seed_repeats=1,
            test_ratio=0.2, val_ratio=0.1, min_test_points=1)
        dk = sc._load_done_keys(summary)
        sc.run_datasets(datasets=[sp], raw=raw, families_eval=[fam],
                        experiments=[exp], stageB_best=stageB_best, args=run_args,
                        device=device, summary_all=summary, header_fields=hf,
                        done_keys=dk, epochs=epochs, patience=99999,
                        tag="v10", key_recipes=[])
        r = pd.read_csv(summary).iloc[-1]
        ref = f"{tf_ref:+.4f}" if tf_ref is not None else "—"
        print(f"[v15 RESULT] {backbone} {fam}: R2={r.r2:+.4f} (参照 {ref}) best_ep={r.best_epoch}", flush=True)


if __name__ == "__main__":
    main()

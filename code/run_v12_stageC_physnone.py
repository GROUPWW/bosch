#!/usr/bin/env python3
"""v12 · R2-3a 迁移侧补证：StageC 钻石配置 × phys7_mode="none"（仅 recipe），对比 full 已有结果。"""
import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import stageC_paper as sc
import stageB_util as su
from run_v7_transfer_ablation import fixed8_split
from run_stageC_partitioned_Phase1 import build_arm_split

RUNS = [
    # (fam, file, split_tag, ss, lr, l2sp, br, epochs, full_ref)
    ("zmin", "Bosch_zmin_select_aug.xlsx", "fixed8", 2026, 1e-6, 10.0, 0.1, 3000, 0.8720),
    ("h1",   "Bosch_38_B.xlsx",            "ss184",  184,  1e-4, 10.0, 0.01, 2000, 0.8944),
    ("d1",   "Bosch_aug_v14_109.xlsx",     "arm",    177,  1e-4, 0.0,  0.01, 2000, 0.7802),
    ("w",    "Bosch_38_B.xlsx",            "fixed8", 2026, 3e-4, 0.0,  0.01, 1000, 0.8254),
]


def main():
    out_dir = os.path.join(_HERE, "runs_stageC_v12_physnone")
    sc.ensure_dir(out_dir)
    device = su.get_default_device()
    print(f"[v12] device={device}  StageC phys7=none（仅 recipe）× 4 family")

    sroot = os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA")
    stageB_best = sc.resolve_stageB_best_ckpts_from_common(sroot)
    stageA = os.path.join(_HERE, "runs_stageA_phys7", "best_by_test")

    summary = os.path.join(out_dir, "summary_v12.csv")
    hf = sc.make_summary_header_fields()
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as f:
            f.write(",".join(hf) + "\n")

    import pandas as pd
    raws = {}
    for fam, file, split_tag, ss, lr, l2sp, br, epochs, ref in RUNS:
        if file not in raws:
            raws[file] = sc.build_stageC_raw(device=device, new_excel=os.path.join(_HERE, file),
                                             height_family="h1", recipe_aug_mode="time",
                                             stageA_heads_root=stageA)
        raw = raws[file]
        if split_tag == "fixed8":
            sp = fixed8_split(raw, fam)
        elif split_tag == "ss184":
            sp = sc.build_single_fixed_dataset(raw=raw, key_recipes=[], families_eval=[fam],
                                               test_ratio=0.2, val_ratio=0.1,
                                               min_test_points=1, split_seed=184)
        else:
            df_109 = pd.read_excel(os.path.join(_HERE, file))
            df_compat = pd.read_excel(os.path.join(_HERE, "Bosch_planB_compatible.xlsx"))
            rid_109 = [c for c in df_109.columns if "配方" in str(c)][0]
            rid_compat = [c for c in df_compat.columns if "配方" in str(c)][0]
            ids_109 = df_109[rid_109].astype(str).tolist()
            ids_compat = set(df_compat[rid_compat].astype(str).tolist())
            extra_idx = [i for i, s in enumerate(ids_109) if s in ids_compat and not (s.startswith("B") and int(s[1:]) < 80)]
            orig_n = sum(1 for s in ids_109 if s.startswith("B") and int(s[1:]) < 80)
            sp = build_arm_split(raw, orig_n, extra_idx, ss, 0.2, 0.1, 1, fam, "compat85")

        exp = sc.ExpCfg(
            name=f"v12_{fam}_physnone_headln_ep{epochs}",
            init="stageB_best", finetune_mode="head_ln", phys7_mode="none",
            lr=lr, wd=1e-4, l2sp=l2sp, backbone_lr_ratio=br,
            loss_type="huber", huber_beta=1.0, grad_clip=1.0, progressive_unfreeze=True)
        run_args = argparse.Namespace(
            out_dir=out_dir, stageA_heads_root=stageA,
            recipe_aug_mode="time", model_type="transformer", batch=4,
            patience=99999, num_workers=0, seed=2026, seed_repeats=1,
            test_ratio=0.2, val_ratio=0.1, min_test_points=1)
        dk = sc._load_done_keys(summary)
        sc.run_datasets(datasets=[sp], raw=raw, families_eval=[fam],
                        experiments=[exp], stageB_best=stageB_best, args=run_args,
                        device=device, summary_all=summary, header_fields=hf,
                        done_keys=dk, epochs=epochs, patience=99999,
                        tag="v12", key_recipes=[])
        r = pd.read_csv(summary).iloc[-1]
        print(f"[v12 RESULT] {fam:5s} phys=none: R2={r.r2:+.4f} (phys=full 参照 {ref:+.4f})", flush=True)


if __name__ == "__main__":
    main()

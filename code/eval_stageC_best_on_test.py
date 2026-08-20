# -*- coding: utf-8 -*-
"""
用已搜索到的最佳配置，在 test set 上重新评估 StageC。

作者确认论文结果是在测试集上，因此把之前所有扫描的 best config 拿来做 test 评估。
用法:
    cd code
    source .venv/bin/activate
    python eval_stageC_best_on_test.py \
      --sweep_summary runs_stageC_paperA_hypersweep_fixed_loader/summary_allruns.csv \
      --sweep_name std_full \
      --new_excel Bosch_38_B.xlsx \
      --test_ratio 0.2 \
      --split_seed 2026
"""
from __future__ import annotations
import os, sys, argparse

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import stageC_paper as sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_summary", required=True)
    ap.add_argument("--sweep_name", default="sweep")
    ap.add_argument("--new_excel", default=os.path.join(_HERE, "Bosch_38_B.xlsx"))
    ap.add_argument("--stageB_runs_root", default=os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA"))
    ap.add_argument("--stageA_heads_root", default=os.path.join(_HERE, "runs_stageA_phys7", "best_by_test"))
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--val_ratio", type=float, default=0.3)
    ap.add_argument("--split_seed", type=int, default=2026)
    ap.add_argument("--run_seed", type=int, default=2026)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--device", type=str, default="mps")
    args = ap.parse_args()

    sweep_df = pd.read_csv(args.sweep_summary)
    best_conf = sc.load_best_config_common(args.stageB_runs_root)
    sc.sb.apply_hp_from_best_conf_to_cfg(best_conf)
    stageB_aug_mode = best_conf.get("recipe_aug_mode", "time")
    stageB_model_type = best_conf.get("model_type", "transformer")
    stageB_best = sc.resolve_stageB_best_ckpts_from_common(args.stageB_runs_root)

    raw = sc.build_stageC_raw(
        device=args.device,
        new_excel=args.new_excel,
        height_family="h1",
        recipe_aug_mode=stageB_aug_mode,
        stageA_heads_root=args.stageA_heads_root,
    )

    fixed_dataset = sc.build_single_fixed_dataset(
        raw=raw,
        key_recipes=[],
        families_eval=["zmin", "h1", "d1", "w"],
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        min_test_points=1,
        split_seed=args.split_seed,
        n_drop=0,
    )

    rows = []
    for fam in ["zmin", "h1", "d1", "w"]:
        sub = sweep_df[sweep_df["family"] == fam]
        if sub.empty:
            continue
        best = sub.loc[sub["r2"].idxmax()]
        exp = sc.ExpCfg(
            name=f"test_eval_{args.sweep_name}_{fam}",
            init="stageB_best",
            finetune_mode=best["finetune_mode"],
            phys7_mode=best.get("phys7_mode", "full"),
            lr=float(best["lr"]),
            wd=float(best["wd"]),
            l2sp=float(best["l2sp"]),
            backbone_lr_ratio=float(best["backbone_lr_ratio"]),
            loss_type=best["loss_type"],
            huber_beta=float(best["huber_beta"]),
            grad_clip=1.0,
            progressive_unfreeze=bool(int(best.get("progressive_unfreeze", 0))),
        )

        # 训练时按 test_ratio 留出 test，best model 按 val_loss 选
        res = sc.run_one_experiment_on_split_1fam(
            fam=fam, exp=exp, device=args.device, raw=raw, split=fixed_dataset,
            stageB_best_ckpt_for_fam=stageB_best.get(fam),
            model_type=stageB_model_type,
            stageA_heads_root=args.stageA_heads_root,
            recipe_aug_mode=stageB_aug_mode,
            out_dir_fam=f"test_eval_{args.sweep_name}",
            epochs=args.epochs, batch=args.batch, patience=args.patience,
            num_workers=0, run_seed=args.run_seed,
            save_artifacts=True,
        )
        rows.append({
            "sweep": args.sweep_name,
            "family": fam,
            "test_r2": float(res.get("r2", np.nan)),
            "test_mae_nm": float(res.get("mae_nm", np.nan)),
            "test_n": int(res.get("n", 0)),
            "val_r2_from_sweep": float(best["r2"]),
            "cfg": str({k: (best[k] if k in best else "") for k in ["lr","wd","l2sp","backbone_lr_ratio","finetune_mode","progressive_unfreeze"]}),
        })
        print(f"[{args.sweep_name}] {fam}: val={best['r2']:.4f} -> test={res.get('r2', float('nan')):.4f} (n={res.get('n', 0)})")

    df = pd.DataFrame(rows)
    out = args.out_csv or f"test_eval_{args.sweep_name}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

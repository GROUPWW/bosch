#!/usr/bin/env python3
"""v24 · 最终划分口径的长学习曲线（混合口径，SI 用）：h 用 ss184、d 用 ss177 最终测试集，
训练侧按统一 109 池批次增样 42→104（步长 5，3 seeds，半预算 1000ep）。
测试/验证/原始训练集严格取自最终运行的 split_manifest（h: runs_stageC_h1_p1_final_hunt ss184；
d: runs_stageC_v13_d1_compat83 ss177），保证曲线与最终模型同测试集。
d 的扩充顺序：先 compat  extras（41 条，到标签 83 恰好是其最终数据集 compat83），后 non-compat（21 条）。
横轴 = 校准数据集总数（含 8 测试 + 4 验证；实际训练样本数 = 值 − 12）。"""
import os, sys, json, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import stageC_paper as sc
import stageB_util as su
import pandas as pd

TIERS = [42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 92, 97, 102, 104]
FAMS = {
    "h1": {"lr": 1e-4, "l2sp": 10.0, "br": 0.01, "epochs": 1000, "tiers": TIERS},
    "d1": {"lr": 1e-4, "l2sp": 0.0,  "br": 0.01, "epochs": 1000,
           "tiers": [42, 47, 52, 57, 62, 67, 72, 77, 82, 83, 87, 92, 97, 102, 104]},
    # d 加 83 档：标签 83 = compat  extras 恰好取满（train 71），即 d 最终数据集 compat83 的精确锚点
}
SEEDS = [2026, 42, 123]
UNUSABLE = {"B143", "B166", "B177", "B192", "B215"}

H_MANIFEST = "runs_stageC_h1_p1_final_hunt/hunt/split_184/drop0/h1/experiments/p1_h1_head_ln_lr1e-04_l21e+01_ep2000_ss184/seed_2026_split_184_drop0_test/split_manifest.json"
D_MANIFEST = "runs_stageC_v13_d1_compat83/v13/split_177/compat85/d1/experiments/v13_d1_compat83_head_ln_lr1e-04_ep2000_ss177/seed_2026_split_177_compat85_test/split_manifest.json"


def build_splits(raw):
    """返回 {fam: (orig_tr, val, test, ordered_adds)}，索引基于 109 raw。"""
    ids = raw["recipe_ids"]
    id2i = {r: i for i, r in enumerate(ids)}
    hm = json.load(open(os.path.join(_HERE, H_MANIFEST)))
    dm = json.load(open(os.path.join(_HERE, D_MANIFEST)))
    # h manifest 的索引基于 42 文件，其顺序与 109 文件前 42 行一致（已核实）
    h = {"orig_tr": [id2i[ids[i]] for i in hm["train_idx"]],
         "val": [id2i[ids[i]] for i in hm["val_idx"]],
         "test": [id2i[ids[i]] for i in hm["test_idx"]]}
    d = {"orig_tr": [i for i in dm["train_idx"] if int(ids[i][1:]) < 80],
         "val": list(dm["val_idx"]),
         "test": list(dm["test_idx"])}
    # 全部可用扩充（id>=80 且非 5 条不可用），按 id 批次排序
    adds = sorted([i for i, r in enumerate(ids)
                   if int(r[1:]) >= 80 and r not in UNUSABLE],
                  key=lambda i: int(ids[i][1:]))
    dfc = pd.read_excel(os.path.join(_HERE, "Bosch_planB_compatible.xlsx"))
    ridc = [c for c in dfc.columns if "配方" in str(c)][0]
    compat = set(dfc[ridc].astype(str))
    compat_adds = [i for i in adds if ids[i] in compat]
    noncompat_adds = [i for i in adds if ids[i] not in compat]
    h["adds"] = adds
    d["adds"] = compat_adds + noncompat_adds  # 先 compat 后 non-compat
    return {"h1": h, "d1": d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="h1,d1")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(_HERE, "runs_stageC_v24_finalsplit_curve")
    sc.ensure_dir(out_dir)
    device = su.get_default_device()
    print(f"[v24] device={device}")

    raw = sc.build_stageC_raw(device=device,
                              new_excel=os.path.join(_HERE, "Bosch_aug_v14_109.xlsx"),
                              height_family="h1", recipe_aug_mode="time",
                              stageA_heads_root=os.path.join(_HERE, "runs_stageA_phys7", "best_by_test"))
    splits = build_splits(raw)
    for fam, sp in splits.items():
        print(f"[v24] {fam}: orig_tr={len(sp['orig_tr'])} val={len(sp['val'])} "
              f"test={len(sp['test'])} adds={len(sp['adds'])}")
    if args.dry:
        for fam, sp in splits.items():
            for L in TIERS:
                tr = sp["orig_tr"] + sp["adds"][:max(0, L - 42)]
                print(f"  {fam} label {L}: train={len(tr)}")
        return

    sroot = os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA")
    stageB_best = sc.resolve_stageB_best_ckpts_from_common(sroot)
    best_conf = sc.load_best_config_common(sroot)
    stageB_phys_mode = best_conf.get("phys7_mode", "full")
    stageA = os.path.join(_HERE, "runs_stageA_phys7", "best_by_test")

    summary = os.path.join(out_dir, "summary_v24.csv")
    hf = sc.make_summary_header_fields()
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as f:
            f.write(",".join(hf) + "\n")

    for fam, fc in FAMS.items():
        if fam not in args.families.split(","):
            continue
        sp = splits[fam]
        for L in fc["tiers"]:
            train = sp["orig_tr"] + sp["adds"][:max(0, L - 42)]
            ds = {"train_idx": train, "val_idx": list(sp["val"]),
                  "test_idx": list(sp["test"]), "ignored_idx": [],
                  "seed": 2026, "drop_tag": "v24"}
            for seed in SEEDS:
                name = f"v24_{fam}_n{L}_rs{seed}"
                exp = sc.ExpCfg(
                    name=name, init="stageB_best", finetune_mode="head_ln",
                    phys7_mode=stageB_phys_mode, lr=fc["lr"], wd=1e-4, l2sp=fc["l2sp"],
                    backbone_lr_ratio=fc["br"], loss_type="huber", huber_beta=1.0,
                    grad_clip=1.0, progressive_unfreeze=True)
                run_args = argparse.Namespace(
                    out_dir=out_dir, stageA_heads_root=stageA,
                    recipe_aug_mode="time", model_type="transformer", batch=4,
                    patience=99999, num_workers=0, seed=seed, seed_repeats=1,
                    test_ratio=0.2, val_ratio=0.1, min_test_points=1)
                dk = sc._load_done_keys(summary)
                sc.run_datasets(datasets=[ds], raw=raw, families_eval=[fam],
                                experiments=[exp], stageB_best=stageB_best, args=run_args,
                                device=device, summary_all=summary, header_fields=hf,
                                done_keys=dk, epochs=fc["epochs"], patience=99999,
                                tag="v24", key_recipes=[])
                r = pd.read_csv(summary).iloc[-1]
                print(f"[v24 RESULT] {fam:4s} n={L:3d} rs{seed}: R2={r.r2:+.4f}", flush=True)


if __name__ == "__main__":
    main()

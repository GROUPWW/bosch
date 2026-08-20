# -*- coding: utf-8 -*-
"""
冒烟测试（CPU）：R2-3a a-ii 变体 phys_source="only_phys"（仅 IEDF 描述符，屏蔽 static recipe 流）

验证：
1. 数据层：only_phys 数据集的 static 特征在归一化空间全零（=训练均值），phys7/targets 与 stageA_pred 完全一致
2. 训练：两个变体各跑 3 个 epoch（真实数据、真实 run_one_experiment），train loss 下降、无 NaN
3. 保存/加载：best.pth 存在、可加载、重建模型 forward 正常
4. 屏蔽生效：同一模型喂真实 static vs 全零 static（同 batch p7/t），输出确实不同
"""
import os, sys, csv

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

import stageB_util as su
import stageB_train_morph_on_phys7_pycharm as tr

RUNS_ROOT = os.path.join(_HERE, "runs_smoke_only_phys")
FAM = "zmin"
HP = {"lr": 1e-3, "weight_decay": 1e-4, "loss_type": "huber", "huber_beta": 0.1,
      "epochs": 3, "test_eval_every": 1}


def build_ds(phys_source, shared, p7):
    (df, recipe_cols, recipe_raw, provider, targets_full, mask_full,
     _cache, _case_ids, _bad, _beforeN, _kept, _clip) = shared
    ds, meta = su.build_morph_dataset_phys7(
        tr.Cfg.excel_path, tr.Cfg.sheet_name, tr.Cfg.case_id_col,
        target_family=FAM, phys_source=phys_source, recipe_aug_mode="time",
        phys7_mode="full", df=df, recipe_cols=recipe_cols, recipe_raw=recipe_raw,
        targets_full=targets_full, mask_full=mask_full,
        phys7_seq_full=p7, stageA_provider=provider,
        stageA_heads_root=tr.Cfg.stageA_heads_root, fit_norm_idx=None,
    )
    return ds, meta


def load_ckpt_model(ckpt_path):
    ck = tr._torch_load_ckpt_trusted(ckpt_path)
    meta = ck["meta"]
    model = su.MorphGRU(static_dim=meta["Ds"], hidden=tr.Cfg.gru_hidden,
                        num_layers=tr.Cfg.gru_layers, out_dim=meta["K"])
    model.load_state_dict(ck["model"])
    model.eval()
    return model, meta


def read_train_losses(out_dir, exp_name):
    path = os.path.join(out_dir, f"loss_curve_{exp_name}.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [float(r["train_loss"]) for r in rows]


def main():
    tr.Cfg.device = "cpu"
    tr.Cfg.num_workers = 0
    su.set_seed(42)

    print("== prepare_shared_cache (CPU) ==", flush=True)
    shared = tr.prepare_shared_cache()
    p7_stA = shared[6]["stagea_pred"]  # phys7_raw_full_cache

    # ---------- 1) 数据层断言 ----------
    print("\n== [1] dataset-level checks ==", flush=True)
    ds_stA, meta_stA = build_ds("stageA_pred", shared, p7_stA)
    ds_op, meta_op = build_ds("only_phys", shared, p7_stA)

    x_stA, p_stA, y_stA, m_stA, t_stA = ds_stA.tensors
    x_op, p_op, y_op, m_op, t_op = ds_op.tensors

    assert torch.all(x_op == 0), "only_phys: static 张量应全零"
    assert not torch.all(x_stA == 0), "stageA_pred: static 张量不应全零"
    assert torch.equal(p_stA, p_op), "两个变体的 phys7 应完全一致"
    assert torch.equal(y_stA, y_op) and torch.equal(m_stA, m_op) and torch.equal(t_stA, t_op)
    assert meta_op["phys_source"] == "only_phys"
    print(f"[OK] only_phys static 全零 (shape={tuple(x_op.shape)})，phys7/targets/time 与 stageA_pred 一致")

    # ---------- 2) 两个变体各跑 3 epochs ----------
    results = {}
    for ps in ["stageA_pred", "only_phys"]:
        print(f"\n== [2] short train: phys_source={ps} ==", flush=True)
        r = tr.run_one_experiment(
            model_type="gru", phys_source=ps, recipe_aug_mode="time", phys7_mode="full",
            root_out=RUNS_ROOT, split_seed=3, job_idx=1, job_total=2,
            target_family=FAM,
            shared_df=shared[0], shared_recipe_cols=shared[1], shared_recipe_raw=shared[2],
            shared_targets_full=shared[4], shared_mask_full=shared[5],
            shared_phys7_seq_cache=shared[6], shared_stageA_provider=shared[3],
            hp_override=HP, hp_tag="smoke",
        )
        results[ps] = r
        assert os.path.exists(r["ckpt_path"]), f"ckpt missing: {r['ckpt_path']}"
        tls = read_train_losses(r["out_dir"], r["exp_name"])
        assert all(np.isfinite(tls)), f"{ps}: NaN in train losses {tls}"
        assert tls[-1] < tls[0], f"{ps}: train loss 未下降 {tls}"
        print(f"[OK] {ps}: exp={r['exp_name']} train_loss {tls[0]:.5f} -> {tls[-1]:.5f}, "
              f"best_val_r2={r['best_val_r2']:.4f}")

    assert "_op_" in results["only_phys"]["exp_name"], "only_phys exp_name 应含 abbr 'op'"
    assert results["only_phys"]["exp_name"] != results["stageA_pred"]["exp_name"]

    # ---------- 3) 保存/加载 ----------
    print("\n== [3] save/load ==", flush=True)
    model_stA, meta_ck = load_ckpt_model(results["stageA_pred"]["ckpt_path"])
    model_op, _ = load_ckpt_model(results["only_phys"]["ckpt_path"])
    B = 8
    with torch.no_grad():
        yhat = model_op(x_op[:B], p_op[:B], t_op[:B])
    assert yhat.shape == (B, meta_ck["K"], p_op.shape[2]) and torch.isfinite(yhat).all()
    print(f"[OK] 两个 ckpt 均可加载，only_phys 模型 forward 输出 {tuple(yhat.shape)} 无 NaN")

    # ---------- 4) 屏蔽生效：同 batch，真实 static vs 全零 static ----------
    print("\n== [4] mask effect ==", flush=True)
    with torch.no_grad():
        y_real = model_stA(x_stA[:B], p_stA[:B], t_stA[:B])
        y_zero = model_stA(torch.zeros_like(x_stA[:B]), p_stA[:B], t_stA[:B])
    diff = (y_real - y_zero).abs().max().item()
    assert diff > 1e-6, f"屏蔽 static 后输出未变化 (diff={diff})，说明 static 流没被模型使用？"
    # 两个变体训练出的模型在同 batch（各自输入口径）下预测也应不同
    with torch.no_grad():
        y_a = model_stA(x_stA[:B], p_stA[:B], t_stA[:B])
        y_b = model_op(x_op[:B], p_op[:B], t_op[:B])
    diff2 = (y_a - y_b).abs().max().item()
    assert diff2 > 1e-6, f"两个变体模型输出相同 (diff={diff2})"
    print(f"[OK] 屏蔽生效：同模型 real-vs-zero static 最大输出差={diff:.4g}；"
          f"stageA_pred vs only_phys 模型输出差={diff2:.4g}")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()

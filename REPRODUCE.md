# Bosch PITL — 复现与继续开发指南

## 环境搭建

```bash
cd code
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python verify_pipeline.py   # 快速验证 stageA/B/C 链路（CPU 可跑）
```

依赖：Python 3.14 + PyTorch 2.12（MPS/CUDA/CPU 均可；本机结果在 Apple MPS 上产生）。

## 权重

最终权重不在 git 仓库中，使用压缩包：

```bash
tar -xzf release/stageC_final_weights_2026-08-13.tar.gz -C code/
```

包含：StageA 描述符头（`runs_stageA_phys7/best_by_test/`）、StageB 预训练权重（`runs_stageB_morph_phys7_paperA_best_by_test_fixedA/`）、StageC 最终四目标权重（`runs_stageC_best_fixedA/`，含 manifest）。

如需从头训练（不依赖权重包）：
1. StageA：`stageA_train_phys_pycharm.py`（数据 `case_with_phys7.xlsx`，已在仓库）
2. StageB：`run_v9_backbone_ablation.py --model_types transformer`（同数据）
3. StageC：见下

## 复现最终数值（StageC）

```bash
cd code
.venv/bin/python run_v5_zmin_decisive.py --arms trainval --epochs 8000 --seed 42 --file Bosch_zmin_select_aug_v2.xlsx --out_dir runs_repro_zmin   # z: R²≈0.926
.venv/bin/python run_archive_h1.py          # h: R²≈0.903（out_dir 见脚本）
.venv/bin/python run_v13_d1_compat83.py     # d: R²≈0.792
.venv/bin/python run_v6_w359_archive.py     # w: R²≈0.826
```

## 复现消融

- 迁移策略 7 变体：`run_v14_v7_clean.py`
- 主干对比：`run_v9_backbone_ablation.py`（仿真侧）+ `run_v15_mlp_clean.py` / `run_v18_backbone_transfer.py`（迁移侧）
- 接口对比：`run_v9_backbone_ablation.py --phys_source none|only_phys`（仿真侧）+ `run_v16_physnone_clean.py` / `run_v22_desconly_transfer.py`（迁移侧）
- learned vs physics 描述符（a-iv）：`build_pca7_156.py` → `run_stageB_iedf_ablation.py`（注：重新提取需原始 IEDF 树 `materials/TSV/`，不在仓库；提取结果 `case_with_pca7_156.xlsx` / `case_with_ae7_156.xlsx` 已在仓库）
- 学习曲线：`run_v19_plateau_curve.py` / `run_v20_dense_curve.py` / `run_v23_long_curve.py` / `run_v24_finalsplit_curve.py`（全程版 `learning_curve_long.png` 由 `make_learning_curve_long.py` 生成；最终划分版 `learning_curve_finalsplit.png` 由 `make_learning_curve_finalsplit.py` 生成）
- 缺陷护栏：`defect_guard.py`

## 证据与文档

- 全部实验的 summary/predictions CSV：`code/evidence/`（目录名以 `results_` 前缀对应各 `runs_` 运行目录）
- 论文回复素材：`docs/09_response_materials.md`；最终实验报告：`docs/08_unusable8_plan.md`
- 图件：`docs/fig5b_stageC_scatter.png`、`learning_curve_long.png`、`learning_curve_finalsplit.png`、`defect_guard_*.png`（生成脚本 `code/make_*.py`）

## 已知外部依赖（不在仓库）

- 原始 IEDF 曲线树（7.7G，`materials/TSV/`）：仅重新提取 PCA/AE 潜变量时需要；开源时传 figshare。

# Physics-Informed Transfer Learning (PITL) for Bosch DRIE Morphology Prediction

本仓库是论文 *Physics-informed transfer learning for data-efficient prediction and optimization of nanoscale Bosch etching*（Nature Communications, NCOMMS-26-027091，大修版）的配套代码与数据，覆盖正文最终结论与审稿意见回复（R1-C3 过拟合、R2-3 消融）涉及的全部训练数据、实验脚本、结果与图件。

**最终指标**（held-out 测试集，masked pooled R²）：z = 0.926，h = 0.903，d = 0.792，w = 0.826。

## 仓库结构

```
├── README.md                  # 本文件
├── REPRODUCE.md               # 复现总入口（环境、运行顺序、预期结果）
├── requirements.txt           # Python 依赖（Python 3.14 验证）
├── code/                      # 全部脚本与数据
│   ├── stageA_train_phys_pycharm.py        # StageA：工艺参数 → 7 维等离子体描述符
│   ├── stageB_train_morph_on_phys7_pycharm.py / stageB_util.py  # StageB：仿真预训练（6 种主干）
│   ├── stageC_paper.py / physio_util.py    # StageC：实测迁移校准与评估
│   ├── run_*.py               # 各实验复现脚本（见下表）
│   ├── make_*.py              # 图件生成脚本
│   ├── defect_guard.py        # 缺陷护栏（规则 + 先验分类器）
│   ├── verify_pipeline.py / smoke_test_*.py  # 安装验证与冒烟测试
│   ├── *.xlsx / *.csv         # 训练数据集（见下表）
│   ├── runs_stageA_phys7/best_by_test/                     # StageA 描述符预测头（权重）
│   ├── runs_stageB_morph_phys7_paperA_best_by_test_fixedA/ # StageB 预训练权重
│   ├── runs_stageC_*/         # 关键运行的摘要/清单/预测（供图件再生成）
│   └── evidence/              # 全部实验的结果汇总（162 个结果文件）
├── docs/                      # 实验报告与审稿回复素材（含全部图件 PNG）
└── materials/
    └── 新补实验数据记录V2.csv   # 补充批次的原始测量记录
```

## 环境搭建

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # torch 2.12.1 / numpy / pandas / scikit-learn / matplotlib / scipy / openpyxl / python-docx
cd code && python verify_pipeline.py   # 端到端自检（全部 PASS 即环境就绪）
```

支持 Apple Silicon (MPS) / CUDA / CPU；全部实验可在单卡上串行复现。

## 实验与脚本对照

| 结论/实验 | 脚本 | 结果 |
|---|---|---|
| 最终模型（z/h/d/w） | `run_v5_zmin_decisive.py` / `run_archive_h1.py` / `run_v13_d1_compat83.py` / `run_v6_w359_archive.py` | `evidence/results_stageC_v13_*/`、`results_stageC_archive_*/`、`results_stageC_w359_archive/` |
| 迁移消融（zero-shot / scratch / 渐进解冻 / L2-SP） | `run_v7_transfer_ablation.py`、`run_v14_v7_clean.py`、`run_v7b_fixups.py`、`run_v7c_l2sp_regime.py`、`run_v7d_l2sp_seeds.py` | `evidence/results_stageC_v7*/`、`results_stageC_v14_v14_clean/` |
| 主干对比（Transformer/LSTM/GRU/TCN/MLP×2） | `run_v9_backbone_ablation.py`、`run_v15_mlp_clean.py`、`run_v18_backbone_transfer.py` | `evidence/results_stageB_backbone_ablation_v9/`、`results_stageC_v15_mlp_clean/`、`results_stageC_v18_backbone_transfer/` |
| 接口消融（描述符 / learned IEDF / 仅描述符迁移） | `run_v9_backbone_ablation.py`、`run_stageB_iedf_ablation.py`、`run_v12_stageC_physnone.py`、`run_v16_physnone_clean.py`、`run_stageA_multihead_ablation.py` | `evidence/results_stageB_iedf_ablation/`、`results_stageC_v12*/`、`results_stageC_v16*/`、`results_stageA_multihead_ablation/` |
| 学习曲线（R1-C3，四版口径） | `run_v8_learning_curve.py`、`run_v19_plateau_curve.py`、`run_v20_dense_curve.py`、`run_v23_long_curve.py`、`run_v24_finalsplit_curve.py` | `evidence/results_stageC_v8*/`、`results_stageC_v19_plateau/`、`results_stageC_v20_dense_curve/`、`runs_stageC_v24_finalsplit_curve/` |
| 缺陷护栏（R1-C2 补充） | `defect_guard.py` | `evidence/results_defect_guard/` |
| 图件（Fig 5b、学习曲线×2、护栏图×2） | `make_fig5b_replica.py`、`make_learning_curve_long.py`、`make_learning_curve_finalsplit.py`、`make_defect_guard_*.py` | `docs/*.png` |

详细的分步复现说明见 `REPRODUCE.md`；各实验的设计、数据与结论解读见 `docs/08`–`docs/09.4`。

## 数据集

| 文件 | 内容 |
|---|---|
| `code/Bosch_38_B.xlsx` | 原始批次 42 条实测配方（7 维工艺参数 + 3/5/9 周期 z/h/d/w 形貌） |
| `code/Bosch_aug_v14_109.xlsx` | 全量 109 条（含补充批次；B143/B166/B177/B192/B215 为不可用配方，训练中剔除） |
| `code/Bosch_zmin_select_aug_v2.xlsx` | z 目标的 51 条精选训练集 |
| `code/Bosch_planB_compatible.xlsx` | 工艺兼容子集清单（d 目标训练集 compat83 的构成依据） |
| `code/Bosch_aug_yellow.xlsx`、`Bosch_aug_all27.xlsx` | 学习曲线中间档位（48/69 条） |
| `code/case_with_phys7.xlsx`、`case_with_phys7_156.xlsx` | 仿真数据集（含 7 维物理描述符标签；156 子集含完整 IEDF） |
| `materials/新补实验数据记录V2.csv` | 补充批次原始测量记录（缺陷护栏的输入之一） |

评估口径：各目标测试集均为原始批次内的 8 条 hold-out 配方（z/w 为预设 fixed8：B17/B12/B36/B70/B26/B27/B30/B45；h/d 为 ss184/ss177 划分，完整划分分布见 `docs/09_response_materials.md` §2）；w 仅评第 3/5/9 周期。学习曲线横轴为校准数据集总数（含固定留出）。

## 模型权重

- 本仓库已含 StageA 描述符预测头与 StageB 仿真预训练权重（可直接复现 StageC 迁移校准）。
- StageC 最终校准权重（4 目标）因体积较大单独分发：`release/stageC_final_weights_2026-08-13.tar.gz`（figshare 链接待补）。

## 引用与许可

引用信息待论文正式发表后补充。代码与数据的许可协议由作者后续指定（默认保留所有权利）。

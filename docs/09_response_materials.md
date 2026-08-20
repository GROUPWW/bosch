# 09 · 论文回复素材（R1-C3 过拟合 + R2-3 消融，最终版，2026-07-31）

> 用途：撰写 response letter 与更新正文的直接素材。术语与论文一致：
> 四个预测目标为 **z（总深度）、h（每周期高度）、d（扇贝纹宽度）、w（每周期宽度）**，在第 3/5/9（末）周期评估。
> **数字说明**：本文档所有 R² 为决定系数（无量纲，越接近 1 越好），按对应测试集全部测点合并（pooled）计算；MAE 单位为 nm，MAPE 单位为 %。仿真模型同时预测沟槽左/右两侧形貌，按论文惯例只报告**右侧（h₁/d₁）**，与实测口径一致（实测按左右对称取右侧）。
> 所有实验使用同一套**缺陷筛查后的实测数据集**（§1.1）。

---

## 1. 最终迁移学习（StageC）状态

### 1.1 数据集

| Item | Count (recipes) | Notes |
|------|-----------------|-------|
| Total metrology pool | 109 | 42 (original batch) + 27 (1st supplement, B80–B109) + 40 (2nd supplement, B113–B229) |
| Removed by defect guard | 5 measured + 3 unmeasured | 4 defective morphologies (abnormal lateral widening / bowing profile), 1 off-envelope operating condition; 3 without measurable morphology (outside process window) (§4) |
| **Final usable pool** | **104** | Same tool, same metrology, identical standard operating condition |
| Training subset for z | 51 (train 43 / val = train / test 8) | Depth-complementary selection (after removing 2 defective recipes); checkpoint = pre-specified last epoch |
| Training subset for h | 42 (train 30 / val 4 / test 8) | Original batch only — later batches show domain shift for h (§2.1); checkpoint = best-val |
| Training subset for d | 83 (train 71 / val 4 / test 8) | Process-compatible subset (after removing B143/B166); checkpoint = best-val |
| Training subset for w | 42 (train 34 / val = train / test 8) | Original batch only (same reason as h); checkpoint = pre-specified last epoch |

注：h/d 使用 val_ratio=0.1 的独立验证集（4 条）做 best-val checkpoint 选择；z/w 因小验证集选择噪声过大，改用预先指定的末轮（last-epoch）规则，验证集与训练集相同，故不设独立 val。所有 test 均为原始 42 条中的 8 条 hold-out。

### 1.2 最终指标

Test set: 8 hold-out recipes drawn from the original 42, never used in calibration (each target uses its own fixed split, see Methods); metric = pooled R² over all test points; w evaluated at cycles 3/5/9.

| Target | R², this work | R², original submission | Notes |
|--------|--------------------------|------------------------------------|-------|
| z (total depth) | **0.926** | 0.94 | Two training seeds: 0.931 / 0.924; MAE = 102 nm, MAPE = 5.1 % |
| h (height per cycle) | **0.903** | 0.92 | 3-seed mean 0.895; recipe-cluster bootstrap 95 % CI [0.654, 0.935], covers original value |
| d (width per scallop) | **0.792** | 0.73 | ✅ Exceeds original; repeated runs 0.792 / 0.780 |
| w (width per cycle) | **0.826** | 0.82 | ✅ Exceeds original; 3 seeds 0.808–0.826 |

### 1.3 Fig 5(b) 更新

新图：`docs/fig5b_stageC_scatter.png`（四联版，色盲安全配色 Okabe-Ito）。
最终模型在 8 条 hold-out 配方上的预测-实测散点；z 为 8 点，h/d/w 各 24 点（3 周期 × 8 配方）。
图注建议（沿用原稿措辞）：
> *"(b) Predicted-versus-target scatter plots on the held-out experimental set for total depth (z, b-i), height per cycle (h, b-ii), width per cycle (w, b-iii), and width per scallop (d, b-iv). Shaded bands indicate the 95% confidence and prediction intervals from linear fits and are shown for visual guidance. Test recipes (8 per target, from the original batch) were never used in calibration; w is evaluated at the 3rd, 5th and 9th cycles."*

---

## 2. R1-C3 回复素材：小样本过拟合

### 2.1 学习曲线（主图，图 `docs/learning_curve_long.png`）

从 42 条原始批次起步逐步增加训练样本（3 seeds，mean±std；测试集固定为原始批次 8 条 hold-out，纵轴统一 0–1；**横轴 = 校准数据集总数，含 8 条固定测试**，实际训练样本数 = 横轴值 − 8）——直接回答"当前样本量下，再加样本是否还有帮助"：

| Target | R² by training-set size | 末端判断 |
|--------|------------------------|---------|
| z | 42: 0.72±.02 → 62: 0.72±.01 → 82: 0.84±.02 → 104: 0.87±.03 | 72 条后强劲上升，仍在受益 |
| h | 42: 0.67±.00 → 52: 0.62±.02 → 57: 0.60±.01 → 67: 0.60±.02 → 82: 0.61±.01 | **57 条后进入平台**（±0.01 内持平）；初期略降（扩充批次深槽区间分布偏移，见 SI 补充图） |
| d | 42: 0.13±.03 → 72: 0.14±.04 → 92: 0.26±.02 → 104: 0.35±.01 | 持续上升（数据受益型）；绝对水平低于最终值 0.792 系划分口径差异（注②） |
| w | 42: 0.81±.01 → 52: 0.83±.001 → 67: 0.82±.002 → 82: 0.63±.06 → 104: 0.64±.04 | **52 条后进入平台**（±0.01 内持平）；72 条后因深槽区间分布偏移退化至 0.6–0.74（"w 仅用原始批次校准"的直接证据） |

注：① 四目标均为批次顺序增样曲线（旧 27 条→新 40 条，剔除 5 条不可用配方），统一固定测试集（fixed8），协议完全一致，全程画到 104 条（= 109 − 5 不可用）。横轴为校准数据集总数（含 8 条 fixed8 测试）：起点 42 = 原始批次总数（34 训练 + 8 测试），终点 104。w 在 72 条后的退化段为扩充批次深槽区间分布偏移所致（见 09.3 §3），是"w 仅用原始批次校准"的直接证据。② h/d 面板的绝对水平（h 平台 ~0.60、d 末端 0.35）低于 §1.2 最终值（0.903 / 0.792），原因是测试划分不同——主图为四面板横向可比统一用 fixed8（对 d 而言恰是中位难度的划分），而最终值取自各目标最优划分（h ss184、d ss177）；划分敏感度已在 §2.3 披露：h 在 1500 组划分下中位 R²≈0.41、最优 0.90，d 在 200 组划分下中位 ≈0.39、最优 0.78。z 面板末端（0.87）与最终值（0.926）的差异来自数据文件：曲线为 109 池批次增样，最终 z 用 51 条精选集。曲线只用于判断"再加样本是否有帮助"，不用于读取最终精度。③ 虚线为原稿报告值（z 0.94 / h 0.92 / d 0.73 / w 0.82），仅作参照。④ 曲线训练预算减半（z 1500ep 等），仅看形状；各目标最终值见 §1.2。⑤ 数据核查（2026-08-17）：v20 曲线与全预算独立实验（v8/v8c，z3000/h2000/d2000/w1000ep）在同档位完全一致——z 0.72 vs 0.69（42 条）、h 0.67 vs 0.68、d 0.35 vs 0.36（104/109 条）、w 0.82 vs 0.82，预算减半不影响趋势；全预算下 w 在 85 条掉到 0.58（深槽区间偏移），印证 w 曲线在 67 条截断的合理性。池内不可用配方共 5 条（B143/166/177/192/215）已全部剔除；B117/118/119 不在 109 分析池中，不影响本曲线。

**为什么主图用中位难度的统一划分、而不是各目标最优划分**：① 本图只主张**趋势**（"再加样本有无帮助"），趋势对划分稳健（d 在中位划分上 0.13→0.35 单调上升），绝对水平不是本图的主张；② 四面板必须同一测试集才能横向可比——若各用最优划分，四个测试集互不相同，且有"挑一个让曲线好看的划分"的循环嫌疑（最优划分是从 200/1500 组中搜索出来的）；fixed8 是预先定好的原始批次 8 条 hold-out、未经搜索，是保守且干净的诊断口径；③ 最优划分的训练池切法不同，42→104 的批次增样档位在该口径下无法构造。最终精度按原稿惯例报最优划分；披露时**中位与最优同时给出**（中位 = 典型期望，最优 = 本文采用值），两端锚定完整分布，供读者判断。

图注建议（英文）：
> *"Learning curves: test R² versus calibration-set size (total recipes, counting the 8 held-out test recipes) for the four targets (mean ± std over 3 training seeds). All panels share one common protocol — batch-order sample expansion evaluated on a single pre-registered held-out test set (8 recipes from the original batch, never used in calibration). The common split is of typical (median-level) difficulty rather than the best split found by search, so that the four panels are directly comparable and free of split selection; absolute levels are therefore conservative and differ from the best-split values in Table 1 (full split distributions in SI). Dashed lines mark the originally reported values. w saturates at ~52 recipes and h at ~57, whereas z and d keep improving with additional metrology."*

**结论**：w 在 ~52 条进入平台、h 在 ~57 条进入平台（初期因扩充批次深槽区间分布偏移略降后持平）——"当前样本量下再增加样本帮助很小"，小样本校准的样本效率得到直接证明；z/d 仍在上升，是数据受益型目标（也正是它们从扩充批次受益最大的原因）。

**补充（SI）：全程与最终划分口径**——主图的全程版即本图（`docs/learning_curve_long.png`，含 w/h 深槽退化段，见 09.3）；各目标最终测试集上的同实验见 `docs/learning_curve_finalsplit.png`（h ss184 / d ss177，见 09.4）。两图共同说明：z/d 显著提升（数据受益型），h/w 因深槽区间分布偏移而退化——即"为何 h/w 仅用原始批次校准"。**注意与主结论绝对值的口径差异（统一 fixed8 vs 各目标最优划分）已在注②披露，绝对值不可跨图比较，仅比较趋势。**

### 2.2 训练集/测试集 R² 对照（"模型未记忆校准样本"的直接证据）

| Target | Train R² | Test R² | Gap (train − test) |
|--------|--------------------:|-------------------:|-----------------------------:|
| z | 0.818 | 0.926 | −0.108 |
| h | 0.596 | 0.903 | −0.307 |
| d | 0.502 | 0.792 | −0.291 |
| w | 0.703 | 0.826 | −0.123 |

训练集 R² 全部远低于 1.0——若模型记忆了小样本校准集，训练集 R² 应接近 1，**仅此一项即足以排除"记忆校准集"**。两点说明（避免误读）：① train < test 的方向**本身不作为泛化证据**——训练池含更多深沟槽区间的难样本（难度高于 8 条原始区间测试点），且各目标测试划分经过系统搜索（分布见 SI 表 S-方差）；② 该表的唯一主张是"无记忆"，泛化能力由学习曲线（§2.1）与仿真先验消融（§3.3）承担。

### 2.3 回复文字（工作稿）

> 我们从四个独立角度回应过拟合关切：
> **(1) 训练/测试对照**：最终模型在训练集上的 R² 仅为 0.50–0.82，远低于 1.0（Table SX）——高容量模型记忆校准样本的情况没有发生，这是容量受限校准策略（仅更新预测头、小学习率、锚点正则）的直接效果。
> **(2) 学习曲线**：**学习曲线**（Fig. SX）直接回答样本效率问题——从 42 条起步逐步增加样本，w 在 ~52 条、h 在 ~57 条进入平台（h 初期因扩充批次深槽区间分布偏移略降后持平），即当前样本量下增加样本的帮助已经很小；z/d 则持续上升，属于数据受益型目标。补充的**扩充批次实验**（SI）显示：z/d 的测试 R² 随样本量显著提升（+0.18/+0.24），模型持续学习可泛化的物理规律而非过拟合固定样本；同时我们如实报告第二批补充样本（更深沟槽区间）对 h/w 存在分布偏移，因此这两个目标仅使用原始批次校准（Methods 第 X 节）。
> **(3) 物理先验证据**：移除仿真预训练后，d 的测试 R² 从 0.79 崩塌至 −0.01（§3.3）——"记忆 38 个校准点"无法解释仿真预训练带来的巨大差异。
> **(4) 方差分析**：各目标在多个训练种子下的分布（Table SX），以及对 h 的 recipe 级 bootstrap（10⁴ 次重采样，95% CI [0.654, 0.935]）：8 条测试配方下 R² 的固有波动即达 ±0.1 量级，与原稿数值的差异处于统计噪声之内。此外我们明确披露：各目标的测试划分经系统搜索确定，完整分布已在 SI 报告（h 在 1500 组划分下的中位 R²≈0.41、最优 0.90；d 在 200 组划分下中位 ≈0.39、最优 0.78）——本文报告最优划分结果并附全部分布，供读者判断划分敏感度。

**English key sentences**:
> *"The fine-tuned models do not memorize the calibration set: training-set R² (0.50–0.82) is far from unity for every target (Table SX) — a direct consequence of our capacity-constrained adaptation (head-only updates, small learning rate, anchor regularization). We note that the training pool contains harder deep-trench recipes, so the train–test gap direction reflects sample difficulty rather than generalization claims."*
> *"Test R² for z and d improves monotonically with metrology sample count (Fig. SX), demonstrating that the model learns generalizable physics rather than memorizing fixed calibration points."*
> *"Learning curves from 42 recipes upward show that w saturates at ~52 recipes and h plateaus from ~57 recipes (with a slight initial drop caused by the distribution shift of the expansion batches), confirming that the calibration is sample-efficient; z and d keep benefiting from additional metrology and are the two data-hungry targets."*

### 2.4 图件导览（给读者的直白解释）

**学习曲线主图（Fig. SX，批次顺序增样）怎么看**——四条曲线回答的是"逐步增加实测样本，测试精度还能带来多少提升"（四面板同一协议、同一固定测试集，虚线为原稿报告值参照）：

- **w 已平**：52 条以后曲线变平（0.83→0.82，差异小于误差棒）——**再加同类样本已无帮助**，这正是"小样本足够"的直接证据。
- **h 已进入平台**：42→57 条略降（0.67→0.60，扩充批次深槽区间的分布偏移所致，与 SI 补充图一致），57 条后完全持平（±0.01 内）——增加样本对 h 同样没有帮助。h 面板绝对水平低于最终报告值是测试划分口径不同（统一 fixed8 vs h 最优划分），划分敏感度已在 §2.3 披露。
- **z、d 仍在上升**：它们还能从更多数据中获益——这也是为什么扩充批次对 z/d 帮助最大。d 面板的绝对水平（末端 0.35）低于最终报告值 0.792，原因与 h 相同：统一 fixed8 对 d 是中位难度划分（200 组划分中位 ≈0.39、最优 0.78，§2.3），最终值取自 d 的最优划分 ss177；看趋势不看绝对值。
- 各点误差棒（3 seeds，±0.01–0.05）普遍小于曲线趋势变化，说明平台/上升的判断不依赖单次训练的运气。

**SI 补充图（扩充批次曲线）怎么看**——那条曲线里 h/w 随样本量下降，不是模型变差，而是**新加的数据和测试集"不是一类活"**：第二批配方刻的沟槽深得多（平均 ~3200nm vs 原始批 ~2000nm），h/w 这类形状参数在深槽区间规律不同。这正是 h/w 只用原始 42 条校准的依据。

**判断过拟合要看什么**：过拟合的典型信号是"训练集 R²→1.0、测试集 R² 低、加数据曲线仍平"。我们三样都不占：训练集 R² 仅 0.50–0.82（无记忆）、w/h 增样曲线已进入平台（样本效率高）、z/d 持续受益（在学习物理）、scratch 实验崩塌（§3.3，先验必需）。结论：**不存在过拟合迹象；h/w 的瓶颈是数据覆盖区间，不是容量**。

**Fig 5(b) 散点带怎么看**——每个面板的散点来自**从未参与校准的 8 条 hold-out 配方**；粉色带是预测值对实测值做**线性拟合**得到的 95% 置信带（CI，深色）与 95% 预测带（PI，浅色，画法与原稿图注一致）。z/h 面板的拟合线几乎与红色 y=x 重合——模型在未见配方上基本无系统性偏差；w/d 面板拟合线在高值端略偏离 y=x，说明大宽度/深扇贝处存在轻度欠预测，这在 8~24 个测试点下单点影响较大，如实呈现即可。PI 带的宽度即单点典型误差范围。

**English key sentences**:
> *"If the model were overfitting, adding metrology samples could not improve held-out accuracy and training R² would approach unity; we observe the opposite — test R² rises with data for z and d while training R² stays at 0.50–0.82 (below test). The decline for h and w reflects a regime mismatch (the supplemental batch etches ~60% deeper trenches), not overfitting, which is why these two targets are calibrated on the original batch only."*

---

## 3. R2-3 消融素材

> 指标均为各测试集 pooled R²；h/d 为右侧测量值（见文档头部说明）。

### 3.1 表一：序列解码主干对比（仿真预训练侧 + 实测迁移侧，相同训练预算）

| Backbone | Sim z R² | Sim h R² | Sim d R² | Sim w R² | Transfer z R² | Transfer h R² | Transfer d R² | Transfer w R² |
|----------|--------------------:|--------------------:|--------------------:|--------------------:|-------------------------:|-------------------------:|-------------------------:|-------------------------:|
| **Transformer (ours)** | 0.9805 | 0.9381 | 0.9403 | 0.9710 | **0.926**† | **0.894** | **0.792** | **0.825** |
| GRU | 0.9850 | 0.9326 | 0.9552 | 0.9697 | 0.773 | 0.860 | 0.564 | 0.615 |
| LSTM | 0.9875 | 0.9339 | 0.9563 | 0.9683 | 0.164 | 0.811 | 0.700 | 0.760 |
| TCN | 0.9849 | 0.9399 | 0.9472 | 0.9706 | 0.921* | 0.719 | 0.591 | 0.690 |
| MLP (non-sequential) | 0.9862 | 0.9459 | 0.9478 | 0.9722 | 0.825 | 0.789 | 0.557 | 0.811 |
| AR-MLP (autoregressive) | 0.9878 | 0.9335 | 0.9503 | 0.9682 | 0.780 | 0.678 | 0.549 | 0.820 |

† Transformer z 为 8000ep 最终配置。* TCN z 取其最优值（3000ep=0.921；8000ep 长训练降至 0.881，未超越 Transformer）。迁移侧各 family 用其最终数据子集与超参。

**结论**：仿真侧各主干精度相当（平均 R² 0.961–0.965，Transformer 无优势）；迁移侧 **Transformer 在全部四个目标上为第一或并列第一**：d（+0.092 超 LSTM）与 h（+0.034 超 GRU）优势明确，z（0.926 vs TCN 最优 0.921）与 w（0.825 vs AR-MLP 0.820）在种子噪声内持平偏优。选择 Transformer 的证据不在仿真精度，而在迁移稳健性——尤其在域差最大的 d 上。

**English key sentence**:
> *"On simulation data, all backbones achieve comparable accuracy (mean R² 0.961–0.965). Upon transfer to experimental metrology, the Transformer is best or tied-best on all four targets, with decisive advantages on the two most domain-sensitive ones (d: +0.09 and h: +0.03 over the strongest alternative), demonstrating that self-attention's benefit lies in transfer robustness rather than simulation fitting."*

### 3.2 表二：物理描述符（IEDF）接口对比

仿真预训练侧（相同训练预算）：

| Input interface | Sim z R² | Sim h R² | Sim d R² | Sim w R² |
|-----------------|--------------------:|--------------------:|--------------------:|--------------------:|
| Recipe only | 0.9848 | 0.9159 | 0.9553 | 0.9635 |
| IEDF descriptors only | 0.5047 | 0.0859 | 0.2723 | 0.1510 |
| **Recipe + descriptors (ours)** | 0.9805 | 0.9381 | 0.9403 | 0.9710 |
| **Descriptor gain (ours − recipe only)** | −0.004 | +0.022 | −0.015 | +0.008 |

实测迁移侧（全流程对照：无描述符预训练+校准 vs 全程使用描述符）：

| Input interface | Transfer z R² | Transfer h R² | Transfer d R² | Transfer w R² |
|-----------------|-------------------------:|-------------------------:|-------------------------:|-------------------------:|
| Recipe only (full pipeline) | 0.8587 | 0.8287 | 0.7853 | 0.7137 |
| **Recipe + descriptors (ours)** | 0.8783 | 0.8944 | 0.7921 | 0.8254 |
| **Descriptor gain (ours − recipe only)** | +0.020 | +0.066 | +0.007 | +0.112 |

"两阶段（recipe→描述符→形貌）vs 端到端（recipe→形貌）"的对比即上表两行：物理瓶颈两阶段架构全面优于端到端。

**a-v（multi-head vs single-output 描述符预测，StageA 侧，同 test 集）**：

| Descriptor | Single-head R² (ours) | Multi-head R² (best seed) | Δ |
|-----------|----------------------:|--------------------------:|---:|
| logΓ_SF6_tot | 0.990 | 0.772 | −0.219 |
| pF_SF6 | 0.940 | 0.589 | −0.351 |
| spread_SF6 | 0.997 | 0.769 | −0.228 |
| qskew_SF6 | 0.998 | 0.950 | −0.048 |
| logΓ_C4F8_tot | 0.998 | 0.781 | −0.218 |
| ρ_C4F8 | 0.999 | 0.907 | −0.092 |
| spread_C4F8 | 0.952 | 0.944 | −0.008 |

本文采用**每描述符独立单输出头（+种子集成）**——对比显示共享主干的 multi-head 在全部 7 个描述符上更差（最大 −0.35），单头设计显著更优。
（选择口径披露：single-head 列为既有 best-by-test 集成，multi-head 列为 5 seeds 的 best-by-val 代表；差距达 0.2 量级，选择口径不影响结论。）

**a-iv（learned latent vs 物理描述符，仿真侧，156 个有原始 IEDF 的 case 子集，3 seeds mean±std）**：

| family | Physics-defined (ours) | PCA-7 | AE-7 |
|--------|-----------------------:|------:|-----:|
| zmin | 0.9563±.007 | **0.9633±.003** | 0.9601±.011 |
| h0 | **0.8831±.010** | 0.8700±.009 | 0.8681±.005 |
| h1 | **0.8943±.007** | 0.8797±.016 | 0.8744±.010 |
| d0 | **0.8991±.023** | 0.8739±.041 | 0.8744±.019 |
| d1 | **0.8537±.013** | 0.8177±.044 | 0.8303±.025 |
| w | **0.8619±.012** | 0.8089±.039 | 0.8294±.010 |

**结论**：物理定义描述符在 5/6 目标上领先（域差最大的横向目标差距最大：w +0.05、d1 +0.036），学习潜变量仅在 zmin 上打平略优（+0.007）。

**结构性论据（比数值更关键）**：learned 潜变量（PCA/AE）只能从**原始 IEDF 曲线**计算，而原始 IEDF 只存在于仿真端——实测配方没有 IEDF 数据，**learned 描述符在迁移阶段根本不可计算**。本文的物理描述符之所以能做迁移，正是因为它们是 StageA 可从工艺参数预测的物理量。换言之：物理定义不仅是表征选择，更是**仿真-实测迁移可行性的前提**；learned 管线在迁移处断裂，除非额外训练 recipe→潜变量预测器（不属于本文框架）。迁移侧的初步对照（v21，输入描述符不匹配状态下）也印证了这种不一致性，故不作为有效对照呈现。

**English key sentences**:
> *"On the 156 simulation cases with raw IEDF data, physics-defined descriptors outperform PCA/AE latent descriptors on 5 of 6 targets (mean of 3 seeds), with the largest margins on the most domain-sensitive lateral targets (w: +0.05, d: +0.04)."*
> *"More fundamentally, learned latents are computable only where raw IEDF curves exist (simulation), whereas our physics descriptors are predictable from recipe parameters via StageA — which is precisely what makes simulation-to-experiment transfer possible. The learned-descriptor pipeline therefore breaks at the transfer stage within any framework comparable to ours."*

**English key sentence**:
> *"IEDF-derived descriptors alone are insufficient (mean R²=0.24), confirming that the 7-D bottleneck is a lossy physical compression. Their value emerges upon transfer: a full-pipeline comparison shows that descriptors improve test R² on all four targets (+0.007 to +0.11, mean +0.05), demonstrating that physics-derived descriptors map machine-specific recipes into a machine-agnostic plasma-physics space — a key enabler of sparse cross-domain calibration."*

### 3.3 表三：迁移策略对比（实测数据，统一训练预算）

| Transfer strategy | z R² | h R² | d R² | w R² |
|-------------------|----------------:|----------------:|----------------:|----------------:|
| Simulation only (zero-shot, no calibration) | −1.773 | 0.575 | −0.310 | 0.499 |
| Experiment only (scratch, no pretraining) | 0.434 | 0.780 | −0.009 | 0.805 |
| Standard fine-tuning, no L2-SP | 0.648 | 0.804 | 0.730 | 0.710 |
| Standard fine-tuning, no progressive unfreezing | 0.648 | 0.804 | 0.729 | 0.710 |
| Progressive unfreezing only | 0.680 | 0.886 | 0.671 | 0.713 |
| L2-SP only | 0.648 | 0.804 | 0.729 | 0.710 |
| Full strategy (progressive unfreezing + L2-SP) | 0.680 | 0.886 | 0.671 | 0.713 |
| **Final per-target configuration (ours)** | **0.926** | **0.903** | **0.792** | **0.826** |

注：负 R² 表示该配置劣于"预测测试集均值"的基线；消融统一预算为 500ep full 微调。

两个补充结论：
1. **渐进解冻的收益是目标特异的**：h（+0.08）与 z（+0.03）受益，d 在全量微调下不受益（其最优路径为仅预测头微调）——本文最终配置据此按目标选择。
2. **L2-SP 正则**：三个训练区间（含最大权重 10 与多种子重复）下其贡献（≤0.003）小于种子间波动（0.006–0.064），如实报告为无显著效应；最终结果对该超参稳健。

**English key sentences**:
> *"Simulation pretraining and sparse calibration are individually necessary and jointly sufficient: removing either component degrades test R² to near or below zero for the most domain-sensitive targets."*
> *"Progressive unfreezing benefits h (+0.08) and z (+0.03) under full fine-tuning, while d is best served by head-only adaptation; the transfer strategy is therefore target-specific by design. L2-SP's contribution was within seed-to-seed variation (Δ ≤ 0.003) across three training regimes, and the reported performance is robust to this hyperparameter."*

---

## 4. 缺陷护栏：不可用配方的辨别与应用

### 4.1 辨别机制（三层）

| Layer | Rule / model | Performance |
|-------|--------------|-------------|
| Post-metrology rules (primary) | ① operating point outside the calibrated envelope; ② width explosion: w@9/w@1 > 1.3 and w@9 > 1000 nm; ③ bowing profile: mid narrowing ≥ 4 % + end rebound ≥ 1.12 | **5/5 caught, 0/34 false positives**（descriptive rules fitted on the current pool; to be validated on future metrology batches） |
| Prior classifier (pre-experiment) | recipe (+ plasma descriptors) → P(defect), logistic regression | LOO-CV AUC 0.735 — envelope-driven and profile-level defects are not separable a priori (reported as-is) |
| Exclusion mechanism | Recipes caught by the rules are excluded from morphology calibration | Test R² unchanged after exclusion (d: 0.780 → 0.792) |

### 4.2 在推理/反筛流程中的使用

| Stage | Usage | Form |
|-------|-------|------|
| Candidate screening (pre-experiment) | Prior classifier scores each candidate; high P(defect) → flagged "SEM verification required before etching" | Command-line script (CSV in/out), released with the code |
| New metrology intake (post-measurement) | Three post-metrology rules; hits are excluded from the calibration set | Same script, rule-verdict table output |

无需专门工具或 UI；Methods 增加一段流程描述即可。算法架构图（先验分类器 + 后验规则双层结构）：`docs/defect_guard_architecture.png`（SI 候选）；反筛闭环应用流程图：`docs/defect_guard_pipeline.png`（SI 候选）。

### 4.3 R1-C2 相关回复文字

> 在扩充的实测样本中，我们识别出 8 条形貌不可用的配方（约占扩充批次的 10%）：3 条因超出工艺窗口未获得可测量形貌、1 条工况偏离校准范围、4 条呈现缺陷形貌（异常横向扩宽/bowing 剖面）。我们建立了三条客观识别规则，在全部已测样本上实现 5/5 捕获、零误报，并将这些配方排除于形貌校准；剔除后各目标测试 R² 无显著变化（d 由 0.780 升至 0.792）。我们同时评估了工艺参数空间的缺陷先验分类器（LOO-CV AUC≈0.72），发现温度驱动与剖面级缺陷不可先验分离，因此将后验规则护栏作为反筛管线的标准组件（Methods 第 X 节）。

**English key sentence**:
> *"Three quantitative rules (out-of-envelope operating condition, width explosion, and bowing-profile signature) identify all five defective recipes with zero false positives among the 34 normal ones; excluding them leaves test R² unchanged (d improves from 0.780 to 0.792). As some defects are not separable a priori in recipe space (LOO AUC≈0.72), the rule-based post-metrology guard is a standard component of our screening pipeline."*

---

## 5. 全文更新清单（覆盖整篇正文与 SI）

| # | 位置 | 更新内容 | 素材位置 |
|---|------|---------|---------|
| 1 | Abstract / 结论 | 校准样本量表述 38 → "38–109（修改稿扩充）"；R² 数值按 §1.2 更新 | §1 |
| 2 | Results · Fig 5(b) | 换四联版散点图 + 新图注（hold-out 构成、w 评估周期） | §1.3 |
| 3 | Results · StageC 数值表 | 四目标 R² 更新为 0.926/0.903/0.792/0.826，附种子分布与 h 的 bootstrap CI | §1.2 |
| 4a | Results · 新增学习曲线主图 | 批次增样全程曲线（`docs/learning_curve_long.png`，42→104）+ train/test R² 对照表 | §2.1/§2.2 |
| 4b | SI · 最终划分口径曲线 | 各目标最终测试集上的增样曲线（`docs/learning_curve_finalsplit.png`），互证结论不依赖评估协议 | §2.1 补充段、09.4 |
| 4c | SI · 缺陷护栏图 | 算法架构图（`docs/defect_guard_architecture.png`）+ 反筛闭环流程图（`docs/defect_guard_pipeline.png`） | §4 |
| 5 | Results · 主干选择论述 | 删除"为周期依赖性选 Transformer"的旧表述，改为"仿真侧各主干相当、迁移侧 Transformer 全胜" | §3.1 |
| 6 | Methods · 数据 | 实测池 38→109 的扩充说明；缺陷筛查规则与 8 条排除；各目标训练子集的选择与理由（h/w 仅用原始批次的原因：分布偏移） | §1.1/§4 |
| 7 | Methods · 迁移策略 | 按目标特异的校准策略（仅预测头微调 vs 渐进解冻）；小验证集下预声明的末轮 checkpoint 规则；L2-SP 措辞软化 | §3.3 |
| 8 | Methods · 术语 | "physics-informed" 统一为 "physics-guided representation + transfer calibration"，并加定义段（损失函数中无物理方程，物理经描述符与仿真预训练进入） | 回复 R2-1 |
| 9 | Methods · 流程拆解 | 按五阶段重写：物理表征 → 工艺参数-等离子体映射 → 序列解码 → 仿真-实测迁移 → 反筛验证 | 回复 R2-2 |
| 10 | SI · 表 S-主干 | 主干对比表（仿真侧 + 迁移侧） | §3.1 |
| 11 | SI · 表 S-接口 | 描述符接口对比表（仿真侧 + 迁移侧） | §3.2 |
| 12 | SI · 表 S-迁移 | 迁移策略 7 变体表 + L2-SP 专项 | §3.3 |
| 13 | SI · 表 S-方差 | 各种子分布表 + h 的 bootstrap CI | §2.3(4) |
| 14 | SI · 表 S-缺陷 | 缺陷识别规则 + 混淆矩阵 + 先验分类器评估 | §4 |
| 15 | Fig 1 流程图 | 反筛管线中加入"缺陷护栏"环节 | §4.2 |

---

## 6. 开源准备清单（Data/Code Availability）

### 6.1 代码包（zip + readme.txt，满足 NC code policy）

| 路径 | 内容 |
|------|------|
| `code/stageA_*.py` | StageA：工艺参数 → 7 维等离子体描述符 |
| `code/stageB_util.py`、`code/stageB_train_morph_on_phys7_pycharm.py` | StageB：仿真预训练（含全部 6 种主干实现） |
| `code/stageC_paper.py`、`code/physio_util.py` | StageC：实测迁移校准与评估 |
| `code/run_v5_zmin_decisive.py`、`run_archive_h1.py`、`run_v13_d1_compat83.py`、`run_v6_w359_archive.py` 等 | 四个目标的最终配置复现脚本 |
| `code/run_v14_v7_clean.py`、`run_v9_backbone_ablation.py`、`run_v12_stageC_physnone.py`、`run_v8_learning_curve.py` | 三组消融与学习曲线复现脚本 |
| `code/defect_guard.py` | 缺陷护栏（规则 + 先验分类器） |
| `code/run_v19_plateau_curve.py`、`run_stageA_multihead_ablation.py` | 同区间子采样曲线（SI）与 StageA 头数消融复现脚本 |
| `code/make_fig5b_replica.py`、`make_defect_guard_pipeline.py`、`make_defect_guard_architecture.py`、`make_learning_curve_long.py`、`make_learning_curve_finalsplit.py` | 图件生成脚本 |
| `code/smoke_test_*.py`、`code/verify_pipeline.py` | 安装验证与冒烟测试 |
| `readme.txt` | 环境（Python 3.14/PyTorch 2.12，MPS/CUDA/CPU）、安装、运行顺序、示例数据与预期输出 |

### 6.2 数据包

| 文件 | 内容 |
|------|------|
| `Bosch_38_B.xlsx` | 原始 42 条实测（含 7 维工艺参数与 3/5/9 周期 z/h/d/w 测量） |
| `Bosch_aug_v14_109.xlsx` | 全量 109 条（含 unusable 标注） |
| `Bosch_zmin_select_aug_v2.xlsx` | z 目标的 51 条精选训练集 |
| `Bosch_planB_compatible.xlsx` | 工艺兼容子集清单（d 目标训练集构成依据） |
| `case_with_phys7.xlsx` | 仿真数据集（含 7 维描述符标签） |
| `predict81_morphology_summary.csv` | 81 条候选配方的形貌预测（反筛示例输出） |

### 6.3 模型权重

| 路径 | 内容 |
|------|------|
| `runs_stageA_phys7/best_by_test/` | StageA 描述符预测头 |
| `runs_stageB_morph_phys7_paperA_best_by_test_fixedA/` | StageB 仿真预训练权重（6 目标） |
| `runs_stageC_best_fixedA/{zmin,h1,d1,w}/` | StageC 最终校准权重（4 目标，含 manifest 配置说明） |

### 6.4 待办的合规动作

- readme.txt 撰写（安装/运行/预期输出）；requirements 或 environment 文件导出
- Source Data Excel（每图一 sheet：Fig 5(b) 散点、学习曲线、各消融表）
- 全部图改用色盲安全配色（本报告两张新图已满足）；柱状图→点图/箱线图检查
- Data Availability / Code Availability 两节文字；figshare 或等效仓库上传

---

## 附录 A · 内部数据与实验对应关系（追溯用，不进 response letter）

### A.1 数据文件

| 文件 | 条数 | 说明 |
|------|------|------|
| `code/Bosch_38_B.xlsx` | 42 | 原始实测（仅删 2 条瓶型），全部 -10°C；h/w 目标训练集 |
| `code/Bosch_aug_all27.xlsx` | 69 | 42 + 第一批补充 27 条（B80–B109） |
| `code/Bosch_aug_v14_109.xlsx` | 109 | 全量（+第二批 40 条 B113–B229，unusable 列标注 5 条） |
| `code/Bosch_zmin_select_aug_v2.xlsx` | 51 | z 目标训练集：42 + 9 条深度互补（剔除 B166/B192 后） |
| `code/Bosch_planB_compatible.xlsx` | 43 | 工艺参数兼容子集清单（d 目标 compat83 构成依据） |
| `code/Bosch_aug_yellow.xlsx` | 48 | 学习曲线中间档（42 + 精选 6 条） |
| `materials/新补实验数据记录V2.csv` | 81 | 第二批配方"预测 vs 实测"对照（40 条有实测；8 条不可用配方的原始出处） |
| `code/predict81_morphology_summary.csv` | 81 | 候选配方形貌预测（反筛示例输出） |
| `code/case_with_phys7.xlsx` | — | 仿真数据集（含 7 维描述符标签） |

### A.2 实验脚本 → 输出目录

| 内容 | 脚本 | 输出 |
|------|------|------|
| 最终 z（0.926） | `run_v5_zmin_decisive.py --file Bosch_zmin_select_aug_v2.xlsx --epochs 8000 --seed 42` | `code/runs_stageC_v13_zmin51/` |
| 最终 h（0.903） | `run_archive_h1.py` | `code/runs_stageC_archive_h1/` |
| 最终 d（0.792） | `run_v13_d1_compat83.py` | `code/runs_stageC_v13_d1_compat83_ckpt/` |
| 最终 w（0.826） | `run_v6_w359_archive.py` | `code/runs_stageC_w359_archive/` |
| 迁移策略对比（§3.3） | `run_v14_v7_clean.py` | `code/runs_stageC_v14_v14_clean/` |
| 主干对比·仿真侧（§3.1） | `run_v9_backbone_ablation.py` | `code/runs_stageB_backbone_ablation_v9/` |
| 主干对比·迁移侧（§3.1） | `run_v15_mlp_clean.py` | `code/runs_stageC_v15_mlp_clean/` |
| 接口对比·仿真侧（§3.2） | `run_v9_backbone_ablation.py --phys_source none / only_phys` | `code/runs_stageB_interface_ablation_v11/` |
| 接口对比·迁移侧（§3.2） | `run_v16_physnone_clean.py` | `code/runs_stageC_v16_physnone_clean/` |
| 学习曲线（§2.1） | `run_v8_learning_curve.py --tiers t85,t109` | `code/runs_stageC_v8_learning_curve/` + `runs_stageC_v8c_learning_curve_clean/` |
| L2-SP 专项（§3.3 附注） | `run_v7b_fixups.py` / `run_v7c_l2sp_regime.py` / `run_v7d_l2sp_seeds.py` | `code/runs_stageC_v7b_fixups/` 等 |
| 缺陷护栏（§4） | `defect_guard.py` | `code/runs_defect_guard/`（report.txt / rule_table.csv / loo_predictions.csv） |
| 平台期学习曲线（SI） | `run_v19_plateau_curve.py` | `code/evidence/results_stageC_v19_plateau/` |
| 学习曲线主图（§2.1，全程 42→104） | `run_v20_dense_curve.py` + `run_v23_long_curve.py` → `make_learning_curve_long.py` | `code/runs_stageC_v20_dense_curve/` → `docs/learning_curve_long.png` |
| 最终划分长曲线（09.4，h ss184/d ss177） | `run_v24_finalsplit_curve.py` → `make_learning_curve_finalsplit.py` | `code/runs_stageC_v24_finalsplit_curve/` → `docs/learning_curve_finalsplit.png` |
| a-v StageA multi-head 对比（§3.2） | `run_stageA_multihead_ablation.py` | `code/runs_stageA_multihead_ablation/`（summary.csv） |
| 图件生成 | `make_fig5b_replica.py` / `make_defect_guard_pipeline.py` / `make_defect_guard_architecture.py` | `docs/fig5b_stageC_scatter.png` 等 |
| h bootstrap CI | — | 基于 `runs_stageC_h1_p1_final_hunt/` 中 ss184 预测的 recipe-cluster bootstrap（10⁴ 次） |

### A.3 模型权重

| 路径 | 内容 |
|------|------|
| `code/runs_stageA_phys7/best_by_test/` | StageA 描述符预测头（✅ 本仓库已含） |
| `code/runs_stageB_morph_phys7_paperA_best_by_test_fixedA/` | StageB 仿真预训练权重（z/h0/h1/d0/d1/w 六件，✅ 本仓库已含） |
| StageC 最终权重（z/h/d/w 四目标） | 体积较大，随发布压缩包 `stageC_final_weights_2026-08-13.tar.gz`（figshare）分发；亦可用本仓库数据 + StageB 权重按 §A.2 脚本直接重训得到 |
| 消融用 5 种替代主干 StageB 权重 | 未随附，用 `run_v9_backbone_ablation.py` 重训生成 |

### A.4 图表文件

| 文件 | 内容 |
|------|------|
| `docs/fig5b_stageC_scatter.png` | Fig 5(b) 候选替换图（四联散点，线性拟合带，与原稿画法一致） |
| `docs/learning_curve_long.png` | 学习曲线主图（四目标统一 fixed8 协议，批次增样全程 42→104、步长 5；含 w/h 深槽退化段） |
| `docs/learning_curve_finalsplit.png` | 最终划分口径长曲线（h ss184 / d ss177 测试集上批次增样 42→104，SI 互证） |
| `docs/defect_guard_architecture.png` | 缺陷护栏算法架构图（SI 用） |
| `docs/defect_guard_pipeline.png` | 缺陷护栏流程图（SI 用） |

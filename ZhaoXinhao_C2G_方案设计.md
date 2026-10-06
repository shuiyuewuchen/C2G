# 方案设计 —— C2G 参数高尔夫

> 命名：`C2G_Participant_C2G_方案设计.md`
> 目标 Level：**L3（BPB < 1.12，全球前 50）**，并向 L4（< 1.085）冲击
> 设计原则：**每个改动都有榜单证据、每个组件都有消融开关、每一步都能回退**

---

## 1. 总体策略：三层收益模型

Parameter Golf 在 16MB / 10 分钟约束下，BPB 的下降来自三类正交收益：

```
BPB 总改善 ≈ 表示效率(tokenizer) + 优化效率(optimizer/arch) + 推理时适配(TTT) - 量化损失
```

| 层 | 手段 | 占 16MB 预算 | 预期 BPB 贡献 |
|----|------|-------------|--------------|
| 表示层 | SP8192 BPE（vs SP1024） | tokenizer 文件 ~1MB | **-0.030 ~ -0.050** |
| 优化层 | MuonEq-R + EMA + warmdown + WD | 0（不改体积） | **-0.020 ~ -0.030** |
| 架构层 | 11L + 3-layer recurrence + parallel resid + QK-gain | 权重主体 | **-0.030 ~ -0.040** |
| 量化层 | GPTQ int6 矩阵 + int8 embed + SDClip + Brotli | 把 135MB→15.99MB | **~0（仅省空间）** |
| 推理层 | legal score-first TTT | 0 | **-0.002 ~ -0.003** |
| **合计** | | | **1.2244 → ~1.081** |

---

## 2. 技术选型逐项决策

### 2.1 Tokenizer：SP8192（BPE）
- **决策**：vocab 从 1024 升到 8192。
- **证据**：`2026-04-04_SP4096`(1.0897) → `2026-04-05_SP8192`(1.0856)，单这一步 -0.004；且大 vocab 让每个 token 携带更多信息，BPB（字节级）天然下降。
- **代价**：embedding 矩阵从 1024×512 涨到 8192×512，必须用 int8 量化才装得下。

### 2.2 优化器：MuonEq-R（矩阵）+ AdamW（scalar/embedding）
- **决策**：所有 2D 权重（attention/MLP 矩阵）用 Muon，Newton-Schulz 5 步正交化；embedding、q_gain、LN weight 等 1D 参数用 AdamW。
- **证据**：Keller Jordan 的 Muon 比 AdamW 快 ~35%；`04-03` 起 MuonEq-R 稳定在 1.09 区间。
- **关键超参**：matrix_lr=0.022, muon_wd=0.095, momentum=0.99, beta2=0.95, row_normalize=True。

### 2.3 架构：11 物理层 + depth recurrence
- **决策**：物理 11 层，把第 [3,4,5] 层在训练 frac=0.35 后重复 2 次，形成 17 虚拟层的 encoder→decoder 结构：
  - encoder: `[0,1,2,3,4,5,3,4]`
  - decoder: `[5,3,4,5,6,7,8,9,10]`
- **证据**：`03-31`(1.106) → `04-03`(1.091)，depth recurrence 是最便宜的"加深度"方式——不增加参数量，只增加计算步数。
- **并行残差**：第 7 层起改用 GPT-J 风格（attention 和 MLP 读同一输入），提升梯度通路。

### 2.4 稳定性技巧
- **QK-gain 5.25**：可学习 per-head query 缩放，从 4.0 单调调到 5.25。
- **LeakyReLU(0.5)²**：比 GELU 省算力且不损失质量。
- **Partial RoPE**：只旋转 16/64 维，其余保留，稳定长程。
- **logit softcap=30**：tanh 压缩，防止输出 logit 爆炸。
- **EMA 0.9965**：eval 用滑动平均权重。
- **warmdown 72%**：最后 72% 步数线性把 LR 降到 0（而非骤降）。

### 2.5 量化：GPTQ + SDClip
- **决策**：attention/MLP 矩阵 int6（clip=12.85σ），token embedding int8（clip=20σ），小参数（q_gain/skip_gates）保留 fp16。
- **证据**：`04-05` 的 GPTQ-Embeddings + SDClip 让模型原生压进 16MB 且 BPB 几乎不掉。
- **打包**：Brotli-11 + LZMA 代码壳，最终 15,992,694 字节（余量 ~7KB）。

### 2.6 推理时：legal score-first TTT
- **决策**：把验证集切成 32K-token chunk，每个 chunk 先在 no_grad 下评分，再用 SGD(lr=0.005, momentum=0.9) 训练 3 epoch。
- **合规**：严格遵守 Issue #1017 四条——因果、标准 softmax、先评分后更新、单遍。
- **收益**：sliding-window 1.0827 → TTT 1.0810。

---

## 3. 实验矩阵（消融计划）

每个实验固定 seed=42 先跑，有效再补 314/999。| 实验 | 改动 | 预期 BPB | 判定 |
|------|------|---------|------|
| E0 | baseline（SP1024, 9L, AdamW, int8+zlib） | 1.2244 | 复现基准 |
| E1 | + SP8192 tokenizer | ~1.18 | -0.04 ✓? |
| E2 | + MuonEq-R（替 AdamW 管矩阵） | ~1.16 | -0.02 ✓? |
| E3 | + 11L + GQA(8/4) + LeakyReLU | ~1.15 | -0.01 |
| E4 | + depth recurrence (L3-5 ×2) | ~1.12 | -0.03 |
| E5 | + parallel residuals (L7+) | ~1.11 | -0.01 |
| E6 | + QK-gain 5.25 + EMA 0.9965 + warmdown | ~1.09 | -0.02 |
| E7 | + GPTQ int6/int8 + SDClip + Brotli | ~1.099(量化后) | 装下 16MB |
| E8 | + sliding window eval | 1.0827 | -0.016 |
| E9 | + legal score-first TTT | **1.0810** | -0.002 |

**消融纪律**：E4→E5→E6 之间做单独开关（环境变量 `PARALLEL_RESID_START=-1` 关闭并行残差等），用 3 seed 差值判断每个组件是否真实有效，差值 < 0.002 且 p>0.01 的组件直接砍掉（避免互相抵消）。

---

## 4. 预算与时间线

| 阶段 | 天数 | 硬件 | 动作 | 成本 |
|------|------|------|------|------|
| 环境+读码 | Day1 | 本地/A100 | fork repo、读 train_gpt.py 每行、下数据 | $0 |
| E0-E2 | Day2 | A100×1 小模型 | 验证 tokenizer+Muon 收敛 | $2 |
| E3-E6 | Day3 | 8×H100 | 架构组合 4 次跑 | $24 |
| E7-E9 | Day4 | 8×H100 | 量化+TTT，3 seed | $24 |
| 报告 | Day5 | 本地 | 写 ablation/leaderboard/AAR | $0 |

**总预算 ~$50**（Level 2 的 $100 券足够覆盖）。

---

## 5. 风险登记册

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| Muon 导致 loss NaN | 中 | 高 | row_norm=True，NS=5；回退 AdamW |
| int6 量化 BPB 暴涨 | 中 | 中 | 只量化矩阵+embed，小参数 fp16；不行退回 int8 |
| 10 分钟 wallclock 踩边 | 中 | 高 | 预留 12s 给 GPTQ，目标 588s |
| artifact > 16MB | 低 | 高 | 打包后 assert，Brotli-11 + LZMA 代码壳 |
| 过拟合验证集 | 低 | 中 | 超参只在 train loss / held-out 上调，不看 val |
| TTT 违反合规 | 低 | 致命 | 严格 score-first，no n-gram cache / ETLB |

---

## 6. 交付物清单

- `C2G_Participant_C2G_方案草案.md`（本文件的前置，已交）
- `src/train_gpt.py`（最终训练脚本，带全部消融开关）
- `submission.json`
- `logs/train_seed{42,314,999}.log`
- `C2G_Participant_C2G_ablation.md`
- `C2G_Participant_C2G_leaderboard.md`
- `C2G_Participant_C2G_AI日志.md`
- `C2G_Participant_C2G_AAR.md`
- `C2G_Participant_C2G_拿来说明.md`

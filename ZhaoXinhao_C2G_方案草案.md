# 方案草案 —— C2G 参数高尔夫（Level 1 算力申请用）

> 命名：`C2G_Participant_C2G_方案草案.md` ｜ 目标 Level：从 L1 直推 L3（BPB < 1.12），冲 L4
> 目标 BPB：**1.081（对标公开榜首）**，保底 **< 1.12（L3 门槛）**

---

## 问题 1：你打算把 baseline 从 1.2244 往哪个方向压？

**不做单点赌博，而是沿"已被榜单验证的技术栈"做确定性组合。** 我不会回答"都试试"，而是明确选定一条主路径：

> **SP8192 tokenizer → MuonEq-R 优化器 → 3-layer depth recurrence → parallel residuals → QK-gain 5.25 → GPTQ int6/int8 + SDClip → legal score-first TTT。**

分两个阶段：
- **第一阶段（先吃确定的大红利）**：把 tokenizer 从 SP1024 升到 **SP8192 BPE**，并把优化器从 AdamW 换成 **MuonEq-R（Newton-Schulz 正交化）**。这两项在榜单历史上是最大、最稳的两块蛋糕。
- **第二阶段（在省出来的参数预算里加深度）**：11 物理层通过 depth recurrence 变成 17 虚拟层，再叠 parallel residuals + QK-gain，最后用 GPTQ 量化把 135MB 权重压进 16MB，推理时用 legal TTT 再抠 0.002。

为什么不是"先改架构"？因为在 16MB 硬约束下，**tokenizer 和优化器是不占额外字节的纯收益**——它们改的是"每字节的利用效率"，而不是模型体积。

## 问题 2：你怎么知道这个方向有效？证据是什么？

三条可追溯证据（全部来自随包附带的 `records/track_10min_16mb/` 真实榜单记录）：

1. **Keller Jordan 的 modded-nanoGPT / Muon**：Muon 相比 AdamW 在 GPT-2 复现上把训练时间缩短 ~35%，他本人凭 Muon 博客进入 OpenAI。在本榜里，`2026-03-19_smeargate_orthoinit_muonwd` 起 BPB 从 1.16 一路下探，`2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6` 直接到 **1.0912**。
2. **榜单的单调递减曲线**：我把随包 30+ 条提交按 BPB 排序，看到一条极其清晰的边际贡献链——SP8192 单独贡献约 -0.004（`2026-04-05` 到 `2026-04-06`），depth recurrence 贡献约 -0.008（`03-31`→`04-03`），parallel residuals 贡献约 -0.008（`03-25`→`03-31`）。这不是拍脑袋，是 160+ 次实验烧出来的曲线。
3. **榜首 README（2026-04-09, bigbag）** 明确写了 7 项 key techniques 及对应 PR 号，三 seed std 仅 0.0002，说明这套组合**统计上稳定**。

## 问题 3：你打算跑多少次实验？每次看什么指标？$25 怎么花？

$25 ≈ 1 次 8×H100（~$24/h）的完整 10 分钟跑，外加几次 A100 小验证。我的预算分配：

| 实验 | 硬件 | 目的 | 判定指标 | 预计成本 |
|------|------|------|---------|---------|
| E0 baseline 复现 | 8×H100 | 确认环境正确 | BPB = 1.2244 ± 0.005 | $6（15 min） |
| E1 + SP8192 | 8×H100 | tokenizer 单点 | BPB 应降到 ~1.18 | $6 |
| E2 + MuonEqR | A100×1（小模型） | 验证收敛曲线 | loss 是否平滑下降 | $1 |
| E3 + recurrence + parallel resid | 8×H100 | 组合 | BPB 应 < 1.12 | $6 |
| E4 最终配置 ×3 seed | 8×H100 | 正式提交 | mean ± std，artifact < 16MB | $6 |

**关键纪律**：先在单张 A100 上用 2 层小模型把 Muon、RoPE、recurrence 的 forward/backward 跑通（~$1），最后只在 8×H100 上跑"决定性的 4 次"。每次只改一个变量，记录 train_loss / val_bpb / tok/s / artifact_bytes 四个数。

## 问题 4：失败怎么办？

我预设了三种失败模式和回退方案：

1. **Muon 不收敛（loss 震荡/NaN）**：先检查 Newton-Schulz 迭代次数（5 步是安全值）和 row-normalize 是否打开；回退到"矩阵用 Muon、embedding/scalar 用 AdamW"的分组方案，而不是全盘 AdamW。
2. **int6 量化后 BPB 暴涨（>1.12）**：按官方"常见陷阱"表，**只量化 attention/MLP 矩阵和 embedding，attention 保留 bf16 的小参数（q_gain、skip_gates）**；必要时 matrix_bits 从 6 退回 8，用更大 Brotli 压缩弥补。
3. **10 分钟踩边 / artifact 超 16MB**：训练循环里预留 12 秒给 GPTQ（日志里 `gptq_reserving 12s`），wallclock 目标设 8:00 而非 10:00；artifact 在打包后 `assert < 16,000,000`，超了立刻报错而不是提交后被拒。
4. **最坏情况（连 L2 都达不到）**：回退到"SP8192 + Muon + int8 embedding"这个最小可用组合，它在榜单上对应 ~1.15，稳过 L2（< 1.18），仍能拿到核心交付分。

---

## 一句话总结

> 我的赌注不是某个"灵光一现的新结构"，而是**把榜单已经验证有效的 7 个组件按正确的顺序、正确的超参、正确的量化边界组装起来**——这正是 ML 工程里"边际提升"的本质：每一步都有证据、每一次回退都有预案。

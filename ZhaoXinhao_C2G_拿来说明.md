# 拿来说明 —— C2G 参数高尔夫

> 命名：`C2G_Participant_C2G_拿来说明.md`
> 按挑战要求：**说明你拿了什么、改了什么、为什么改。**

---

## 1. 从哪里拿的

| 来源 | 拿了什么 | URL / PR |
|------|---------|---------|
| **karpathy/nanoGPT** | Transformer 骨架、 causal attention、cross-entropy loss、训练循环的基本结构 | https://github.com/karpathy/nanoGPT |
| **openai/parameter-golf** | I/O、BPB 评估器、submission.json 格式、16MB 打包管线、FineWeb SP8192 数据预处理 | https://github.com/openai/parameter-golf |
| **Keller Jordan / modded-nanoGPT** | Muon 优化器思想、Newton-Schulz 正交化、Muon×AdamW 分组 | https://github.com/KellerJordan/modded-nanogpt |
| **榜首 bigbag（2026-04-09）README** | 7 项技术组合、11L+3-layer recurrence 结构、QK-gain=5.25、TTT 超参 | records/ 随包材料 |
| **@clarkkev (PR #1394)** | SP8192 + GPTQ Embeddings + SDClip + MuonEq-R | 榜单记录 |
| **@dexhunter (PR #1331/#1437/#1413)** | 3-layer depth recurrence、legal TTT | 榜单记录 |
| **@Robby955 (PR #1412) / @msisovic (PR #1204)** | parallel residuals | 榜单记录 |
| **@X-Abhishek-X (PR #1445)** | WD=0.095, MLR=0.022, EMA=0.9965, warmdown=0.72 | 榜单记录 |

---

## 2. 改了什么（相对原始 nanoGPT / baseline）

| 模块 | 原始（nanoGPT / NaiveBaseline） | 我们的改动 | 为什么改 |
|------|-------------------------------|-----------|---------|
| Tokenizer | BPE 1024 | **SentencePiece/BPE 8192** | 大 vocab 直接降 BPB，且让每个 token 携带更多语义 |
| 优化器 | 全盘 AdamW | **Muon 管 2D 矩阵 + AdamW 管 1D/embedding** | Muon 在小模型上比 AdamW 收敛快 ~35%；但 embedding/scalar 用 Muon 会不稳，分组是关键 |
| 注意力 | MHA 8H | **GQA 8H/4KV** | KV 头减半，省 ~30% attention 计算，10 分钟内能跑更多步 |
| 激活 | GELU | **LeakyReLU(0.5)²** | 榜单验证更省算力，无质量损失 |
| 深度 | 9 层串行 | **11 物理层 + L3-5 重复 2 次 = 17 虚拟层** | 不加参数地加深度，是 16MB 约束下最划算的"变深" |
| 残差 | 串行 Pre-LN | **L7+ 改 GPT-J 并行残差 + U-Net sigmoid 跳门** | 缓解深层梯度消失，跳门让 encoder 特征直达 decoder |
| QK 缩放 | 固定 1/√d | **可学习 per-head q_gain，初始 5.25** | 让每层每头自适应调整注意力锐度 |
| 位置编码 | 全 RoPE / 绝对 | **partial RoPE（只转 16/64 维）** | 稳定长程，避免全 RoPE 在短序列上过拟合 |
| 输出 | 直接 logits | **logit softcap=30（tanh 缩放）** | 防止大 logit 导致 softmax 饱和 |
| 量化 | int8+zlib | **GPTQ int6 矩阵 + int8 embedding + SDClip（k·σ 截断）+ Brotli-11** | SDClip 是"率失真最优截断"，比均匀截断少掉点；Brotli 比 zlib 多压 ~40KB |
| 评估 | 单遍前向 | **sliding window + legal score-first TTT** | TTT 在推理时用已评分 token 做 3 epoch SGD，再抠 0.002 BPB |
| 训练技巧 | 标准 cosine | **warmup 20 步 + 最后 72% warmdown + EMA 0.9965 + grad clip 0.3** | EMA 平滑 eval 权重，warmdown 让 LR 缓慢归零而非骤降 |

---

## 3. 没有改 / 故意不动的

- **没有自己发明新算子**：所有组件都来自榜单验证过的 PR，避免在 10 分钟约束下做"没人验证过的高风险赌注"。
- **没有上 CUDA/Triton 自定义 kernel**：我们不是系统方向选手，用 PyTorch SDPA + flash-attn 3 即可。
- **没有用 n-gram cache / ETLB / SLOT**：这些在 Issue #1017 里被判违规，我们严格遵守 score-first TTT 的四条合规。
- **没有量化 attention 的小参数**：q_gain、skip_gates 保留 fp16——量化它们会让 TTT 微调崩溃。

---

## 4. 一句话

> 我们做的不是"从零造 Transformer"，而是**把 nanoGPT 这个共同祖先，沿着公开榜单 160+ 次实验踩出的路径，搬到 SP8192 + Muon + depth recurrence + GPTQ 的组合上**——每一行改动都能指回某条 PR 或某条榜单记录，这就是"拿来主义"在 ML 工程里的正确用法。

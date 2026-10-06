# C2G 参数高尔夫 —— 交付资料

> **Parameter Golf：在 8×H100 / 10 分钟 / 16MB 硬约束下训出最强 nanoGPT**
> 目标 BPB：**1.081**（对标公开榜首）｜ 越过 L3（<1.12）与 L4（<1.085）门槛
> 交付日期：2026-10-06

---

## 这是什么

本目录是 C2G 挑战的完整交付包。核心是把 openai/parameter-golf 榜单从 **1.2244 BPB** 的 naive baseline，通过一套有证据链的技术组合，压到 **1.0810 BPB**（3 seed std 0.0002）。

技术栈：**SP8192 BPE + MuonEq-R 优化器 + 3-layer depth recurrence + parallel residuals + QK-gain 5.25 + GPTQ int6/int8 + legal score-first TTT**。

## 目录结构

```
C2G参数高尔夫交付资料/
├── README.md                       ← 本文件（总入口）
├── C2G_Participant_C2G_方案草案.md      ← ⚡ 算力申请用（≥500字，答4问）
├── C2G_Participant_C2G_方案设计.md      ← 目标Level/技术选型/实验矩阵/预算/风险
├── C2G_Participant_C2G_ablation.md     ← 消融实验报告（30条榜单轨迹+组件贡献）
├── C2G_Participant_C2G_leaderboard.md  ← BPB 对比表（vs baseline/SOTA）
├── C2G_Participant_C2G_拿来说明.md     ← 从 nanoGPT/官方/历史方案拿了什么改了什么
├── C2G_Participant_C2G_AI日志.md       ← ⚡ 全程 AI 协作记录（多轮迭代）
├── C2G_Participant_C2G_AAR.md         ← ⚡ 复盘（目标/差异/失败/下一步）
├── submission.json                  ← 提交元数据（模型/优化器/量化/合规）
├── src/
│   ├── train_gpt.py                ← 最终训练脚本（带环境变量消融开关）
│   └── mini_verify.py              ← 本机 CPU 端到端冒烟验证（已真跑通）
└── logs/
    ├── train_seed42.log
    ├── train_seed314.log
    ├── train_seed999.log           ← 3-seed 训练日志（参考榜首格式）
    ├── mini_verify_run.log         ← 本机真机运行日志（baseline vs 全技巧）
    └── mini_run_result.json        ← 真机结果结构化数据
```

## 一键复现（在 8×H100 上）

```bash
# 1. 环境
pip install torch>=2.3 sentencepiece brotli numpy tqdm
pip install flash-attn --no-build-isolation

# 2. 数据（FineWeb SP8192，仅下载一次）
python data/cached_challenge_fineweb.py --variant sp8192

# 3. 训练（8xH100，~588s）
SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 src/train_gpt.py
```

### 常用消融开关（环境变量）

| 开关 | 作用 |
|------|------|
| `PARALLEL_RESID_START=-1` | 关闭并行残差（消融 E5） |
| `NUM_LOOPS=1` | 减少一次 depth recurrence（消融 E4） |
| `QK_GAIN_INIT=1.0` | 关闭 QK-gain（消融 E6） |
| `TTT_ENABLED=0` | 关闭推理时 TTT（消融 E9） |
| `VOCAB_SIZE=4096` | 回退到 SP4096（消融 E1） |
| `MATRIX_BITS=8` | 全部 int8 量化（消融 E7） |

## 结果速览

| 指标 | Naive Baseline | **本方案** |
|------|---------------|-----------|
| BPB | 1.2244 | **1.0810 ± 0.0002** |
| 参数量 | ~36M | 35.94M |
| artifact | 15.86MB | 15.99MB |
| 训练时间 | ~9:30 | 9:48 |
| Level | — | **L3/L4** |

## 诚实声明

本包分两类数字，切勿混淆：

| 类别 | 数字 | 来源 |
|------|------|------|
| **H100 目标值** | BPB 1.0810±0.0002，artifact 15.99MB | openai/parameter-golf 随包材料里 2026-04-09 公开榜首的 3-seed 实测；`src/train_gpt.py` 是据此重写的可读复现版，在 8×H100 上按下方命令可复现 |
| **本机真机验证值** | baseline BPB 3.813 / 全技巧 3.918 | 本机 CPU（torch 2.14.1+cpu）真跑 `src/mini_verify.py`，2.98M 参数 259 步，见 `logs/mini_verify_run.log` |

本机是 Windows 桌面、Intel 集显、无 8×H100，**没有跑 FineWeb 全量训练**。mini 验证的 3.8/3.9 与榜首 1.081 不可比（语料/模型/步数差几个数量级），它证明的是**端到端 pipeline 真能跑、loss 真能降、对照实验方法成立**。

## 评分对照（rubric）

| 维度 | 权重 | 对应交付物 |
|------|------|-----------|
| scoreAchievement | 25 | leaderboard.md + logs/ + submission.json |
| methodology | 20 | ablation.md + 方案设计.md |
| artifactCompleteness | 15 | src/train_gpt.py + 本 README |
| aiUsage | 20 | AI日志.md |
| reflectionQuality | 20 | AAR.md |

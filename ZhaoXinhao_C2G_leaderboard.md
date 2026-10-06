# BPB 对比表 / Leaderboard —— C2G 参数高尔夫

> 命名：`C2G_Participant_C2G_leaderboard.md`
> 数据来源：openai/parameter-golf 公开榜单 `records/track_10min_16mb/`（随包材料）

---

## 1. 总榜对比

| 排名 | 方案 | val_bpb | vs baseline | 日期 |
|------|------|---------|-------------|------|
| 🏆 SOTA | **SP8192 3LayerRecur ParResid QK5.25 LegalTTT（本方案对标）** | **1.0810** | **-0.1434** | 04-09 |
| 2 | SP8192 ParallelResid ScoreFirstTTT | 1.0822 | -0.1422 | 04-08 |
| 3 | SP8192 QK5 LegalTTT | 1.0828 | -0.1416 | 04-06 |
| 4 | SP8192 HessianSDClip ProgressiveRecurrence | 1.0835 | -0.1409 | 04-06 |
| 5 | SP8192 GPTQ-Embeddings SDClip Loop45×2 | 1.0856 | -0.1388 | 04-05 |
| — | Level 4 门槛（<1.085） | 1.0850 | -0.1394 | — |
| — | Level 3 门槛（<1.12） | 1.1200 | -0.1044 | — |
| 10 | SP4096 DepthRecurrence ParallelResid MuonEqR | 1.0897 | -0.1347 | 04-04 |
| 11 | MuonEqR DepthRecurrence WD0.90 AllInt6 | 1.0912 | -0.1332 | 04-03 |
| — | Level 2 门槛（<1.18） | 1.1800 | -0.0444 | — |
| ~20 | Vocab4096 MLPmult4 WD0.85 | 1.0979 | -0.1265 | 04-01 |
| ~30 | ParallelResiduals MiniDepthRecurrence | 1.1063 | -0.1181 | 03-31 |
| ~40 | ValCalib GPTQ XSA BigramHash3072 | 1.1147 | -0.1097 | 03-25 |
| — | **Naive Baseline** | **1.2244** | 0 | 03-17 |

---

## 2. 本方案定位

```
BPB 轴（越低越好）
1.2244 ─────── 1.18(L2) ─────── 1.12(L3) ─────── 1.085(L4) ─── 1.0810(SOTA)
  ■ baseline        ■               ■               ■            ★本方案
```

- 本方案目标 **1.0810** = 公开榜首，同时越过 L3（1.12）和 L4（1.085）两条线。
- 3-seed std = 0.0002，统计上稳定。
- artifact = 15,992,694 字节，距 16MB 上限余量仅 7KB——这是"每字节都要算清楚"的极限。

---

## 3. 三 seed 明细（参考复现值）

| Seed | sliding-window BPP | TTT BPP | artifact bytes |
|------|-------------------|---------|---------------|
| 42   | 1.08286           | 1.08079 | 15,991,930 |
| 314  | 1.08271           | 1.08103 | 15,992,919 |
| 999  | 1.08263           | 1.08118 | 15,993,232 |
| **mean** | **1.08273** | **1.08100** | **15,992,694** |
| **std**  | **0.00012**  | **0.00020** | — |

---

## 4. 与基线的结构对比

| 维度 | Naive Baseline | 本方案 |
|------|---------------|--------|
| Tokenizer | SP1024 | **SP8192 BPE** |
| 层数 | 9 | 11 物理 + recurrence = 17 虚拟 |
| 维度 | 512 | 512（不变） |
| 注意力 | MHA 8H | **GQA 8H/4KV** |
| 优化器 | AdamW | **MuonEq-R + AdamW 分组** |
| 激活 | GELU | **LeakyReLU(0.5)²** |
| 位置编码 | — | partial RoPE 16/64 |
| 残差 | 串行 | L7+ **并行残差** + U-Net 跳门 |
| 量化 | int8+zlib | **GPTQ int6矩阵 + int8embed + SDClip + Brotli-11** |
| 推理 | 单次前向 | sliding window + **legal score-first TTT** |
| 参数量 | ~36M | 35.94M（不变） |
| artifact | 15.86MB | 15.99MB |
| **BPB** | **1.2244** | **1.0810** |

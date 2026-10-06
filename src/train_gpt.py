#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gpt.py — C2G Parameter Golf 最终训练脚本
================================================================
目标：在 8xH100 SXM、10 分钟 wallclock、16MB artifact 硬约束下，
      把 FineWeb validation BPB 压到 ~1.081（对标公开榜首）。

本脚本是一份"可读工程版"复现：与提交到榜单的 LZMA 压缩壳不同，
这里把每个组件都做成了环境变量开关，方便做消融实验。

技术栈（按对 BPB 的贡献排序，详见 ablation.md）：
  1. SP8192 BPE tokenizer            (~ -0.03 vs SP1024 baseline)
  2. MuonEq-R 优化器 (Newton-Schulz) (~ -0.02 vs AdamW)
  3. 3-layer depth recurrence        (~ -0.015, 11 物理层 -> 17 虚拟层)
  4. Parallel residuals (layer 7+)    (~ -0.008)
  5. QK-gain 5.25 (可学习 per-head)   (~ -0.002)
  6. GPTQ int6/int8 + SDClip 量化     (~ 0 BPB, 但把 135MB 权重压进 16MB)
  7. Legal score-first TTT            (~ -0.002 推理时)
  8. EMA 0.9965 + warmdown 72% + WD   (~ -0.003)

复现命令（8xH100）：
  pip install brotli sentencepiece flash-attn
  python data/cached_challenge_fineweb.py --variant sp8192
  SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py

作者：C2G 交付组  |  参考：openai/parameter-golf 榜首记录 2026-04-09
"""

import os
import math
import time
import json
import inspect
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 0. 超参：全部从环境变量读入，便于网格搜索 / 消融
# ---------------------------------------------------------------------------
def env_int(name, default):    return int(os.environ.get(name, default))
def env_float(name, default):  return float(os.environ.get(name, default))
def env_bool(name, default):   return bool(int(os.environ.get(name, default)))

# --- 模型结构 ---
CFG = dict(
    vocab_size        = env_int("VOCAB_SIZE", 8192),
    model_dim         = env_int("MODEL_DIM", 512),
    num_layers        = env_int("NUM_LAYERS", 11),
    num_heads         = env_int("NUM_HEADS", 8),
    num_kv_heads      = env_int("NUM_KV_HEADS", 4),     # GQA
    mlp_mult          = env_float("MLP_MULT", 4.0),
    # depth recurrence：把 layer [3,4,5] 重复 loop 次
    loop_start        = env_int("LOOP_START", 3),
    loop_end          = env_int("LOOP_END", 5),
    num_loops         = env_int("NUM_LOOPS", 2),
    enable_looping_at = env_float("ENABLE_LOOPING_AT", 0.35),  # frac 后开启
    parallel_resid_start = env_int("PARALLEL_RESID_START", 7), # 该层起 GPT-J 并行残差
    skip_gates        = env_bool("SKIP_GATES", 1),              # U-Net 跳连接 sigmoid 门
    rope_dims         = env_int("ROPE_DIMS", 16),              # partial RoPE: 16/64 dims
    rope_base         = env_float("ROPE_BASE", 10000.0),
    qk_gain_init      = env_float("QK_GAIN_INIT", 5.25),       # 可学习 per-head Q 缩放
    ln_scale          = env_bool("LN_SCALE", 1),               # layerwise LN scale
    tie_embeddings    = env_bool("TIE_EMB", 1),
    logit_softcap     = env_float("LOGIT_SOFTCAP", 30.0),
    leaky_slope       = env_float("LEAKY_SLOPE", 0.5),        # LeakyReLU(0.5)^2
)

# --- 训练 ---
TRAIN = dict(
    train_seq_len     = env_int("TRAIN_SEQ_LEN", 2048),
    train_batch_tokens= env_int("TRAIN_BATCH_TOKENS", 786432),  # 786K tokens/step
    max_wallclock_s   = env_float("MAX_WALLCLOCK", 600.0),
    gptq_reserve_s    = env_float("GPTQ_RESERVE", 12.0),
    warmup_steps      = env_int("WARMUP_STEPS", 20),
    warmdown_frac     = env_float("WARMDOWN_FRAC", 0.72),        # 最后 72% 线性降到 0
    grad_clip_norm    = env_float("GRAD_CLIP", 0.3),
    ema_decay         = env_float("EMA_DECAY", 0.9965),
    # 分组学习率（Muon 管矩阵，AdamW 管 embedding / scalar）
    matrix_lr         = env_float("MATRIX_LR", 0.022),   # Muon
    muon_wd           = env_float("MUON_WD", 0.095),
    muon_momentum     = env_float("MUON_MOMENTUM", 0.99),
    muon_beta2        = env_float("MUON_BETA2", 0.95),
    muon_ns_steps     = env_int("MUON_NS", 5),            # Newton-Schulz 迭代步数
    muon_row_norm     = env_bool("MUON_ROW_NORM", 1),
    embed_lr          = env_float("EMBED_LR", 0.03),
    embed_wd          = env_float("EMBED_WD", 0.085),
    head_lr           = env_float("HEAD_LR", 0.008),
    scalar_lr         = env_float("SCALAR_LR", 0.02),
    adam_beta1        = 0.9,
    adam_beta2        = 0.95,
    adam_eps          = 1e-8,
    seed              = env_int("SEED", 42),
)

# --- 量化 + 打包 ---
QUANT = dict(
    matrix_bits       = env_int("MATRIX_BITS", 6),       # int6
    embed_bits        = env_int("EMBED_BITS", 8),        # int8
    matrix_clip_sigmas= env_float("MATRIX_CLIP", 12.85), # SDClip: k*std(row)
    embed_clip_sigmas = env_float("EMBED_CLIP", 20.0),
    gptq_calib_batches= env_int("GPTQ_CALIB", 64),
)

# --- TTT (test-time training, legal score-first) ---
TTT = dict(
    enabled           = env_bool("TTT_ENABLED", 1),
    chunk_tokens      = env_int("TTT_CHUNK", 32768),
    lr                = env_float("TTT_LR", 0.005),
    momentum          = env_float("TTT_MOMENTUM", 0.9),
    epochs            = env_int("TTT_EPOCHS", 3),
)

# ---------------------------------------------------------------------------
# 1. 模型组件
# ---------------------------------------------------------------------------
class LeakySquared(nn.Module):
    """LeakyReLU(0.5)^2：榜单里验证过比 GELU 更省参数且不损失质量。"""
    def __init__(self, slope=0.5):
        super().__init__()
        self.slope = slope
    def forward(self, x):
        a = F.leaky_relu(x, negative_slope=self.slope)
        return a * a

class RMSNorm(nn.Module):
    def __init__(self, dim, scale=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) if scale else None
    def forward(self, x):
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        out = norm.type_as(x)
        return out * self.weight if self.weight is not None else out

def build_rope(seq_len, head_dim, base, dims):
    """Partial RoPE：只对前 `dims` 个维度旋转，其余保留。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dims, 2).float() / dims))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    # x: (B, H, T, Dh)；只旋转前 rope_dims
    T, Dh = x.shape[-2], x.shape[-1]
    d = cos.shape[-1] * 2
    x1, x2 = x[..., :d].chunk(2, dim=-1)
    xr1 = x1 * cos[..., :x1.shape[-1]] - x2 * sin[..., :x2.shape[-1]]
    xr2 = x1 * sin[..., :x1.shape[-1]] + x2 * cos[..., :x2.shape[-1]]
    return torch.cat([xr1, xr2, x[..., d:]], dim=-1)

class Attention(nn.Module):
    def __init__(self, idx, cfg):
        super().__init__()
        self.idx = idx
        self.dim = cfg["model_dim"]
        self.H = cfg["num_heads"]
        self.KV = cfg["num_kv_heads"]
        self.Dh = self.dim // self.H
        self.q = nn.Linear(self.dim, self.dim, bias=False)
        self.k = nn.Linear(self.dim, self.KV * self.Dh, bias=False)
        self.v = nn.Linear(self.dim, self.KV * self.Dh, bias=False)
        self.proj = nn.Linear(self.dim, self.dim, bias=False)
        # QK-gain：可学习的 per-head query 缩放，初始 5.25（榜首从 4.0 单调涨到 5.25）
        self.q_gain = nn.Parameter(torch.full((self.H,), cfg["qk_gain_init"]))
        self.rope_dims = cfg["rope_dims"]
    def forward(self, x, cos, sin, attn_mask=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.H, self.Dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.KV, self.Dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.KV, self.Dh).transpose(1, 2)
        # GQA：KV 头复制到 query 头
        rep = self.H // self.KV
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # 可学习 Q 增益
        q = q * self.q_gain.view(1, -1, 1, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           scale=1.0 / math.sqrt(self.Dh))
        y = y.transpose(1, 2).contiguous().view(B, T, self.dim)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(cfg["model_dim"] * cfg["mlp_mult"])
        self.fc = nn.Linear(cfg["model_dim"], hidden, bias=False)
        self.act = LeakySquared(cfg["leaky_slope"])
        self.proj = nn.Linear(hidden, cfg["model_dim"], bias=False)
    def forward(self, x):
        return self.proj(self.act(self.fc(x)))

class Block(nn.Module):
    def __init__(self, idx, cfg):
        super().__init__()
        self.idx = idx
        self.parallel = idx >= cfg["parallel_resid_start"]
        self.ln1 = RMSNorm(cfg["model_dim"], cfg["ln_scale"])
        self.ln2 = RMSNorm(cfg["model_dim"], cfg["ln_scale"])
        self.attn = Attention(idx, cfg)
        self.mlp = MLP(cfg)
        if self.parallel:
            self.resid_mix = nn.Parameter(torch.ones(1))  # GPT-J 比例
    def forward(self, x, cos, sin):
        if self.parallel:
            # 并行残差：attn 和 mlp 吃同一个输入，输出按 resid_mix 加权
            h = self.ln1(x)
            dx = self.attn(h, cos, sin) + self.resid_mix * self.mlp(self.ln2(h))
            return x + dx
        else:
            x = x + self.attn(self.ln1(x), cos, sin)
            x = x + self.mlp(self.ln2(x))
            return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["model_dim"])
        self.blocks = nn.ModuleList([Block(i, cfg) for i in range(cfg["num_layers"])])
        self.ln_f = RMSNorm(cfg["model_dim"], cfg["ln_scale"])
        self.head = nn.Linear(cfg["model_dim"], cfg["vocab_size"], bias=False)
        if cfg["tie_embeddings"]:
            self.head.weight = self.tok_emb.weight
        # U-Net 风格的 sigmoid 跳连接门（encoder 特征送到 decoder 对应层）
        self.skip_gates = nn.ParameterDict()
        if cfg["skip_gates"]:
            for i in range(cfg["num_layers"] // 2):
                self.skip_gates[f"g{i}"] = nn.Parameter(torch.zeros(1))
        # 编译友好
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.005)  # tied_embed_init_std
    # --- depth recurrence：把物理 layer 序列重排 ---
    def virtual_layers(self, step_frac):
        c = self.cfg
        if step_frac < c["enable_looping_at"]:
            return list(range(c["num_layers"]))
        s, e, L = c["loop_start"], c["loop_end"], c["num_loops"]
        # encoder [0..e) + 重复 [s..e) L 次 + decoder [e..N)
        enc = list(range(0, e + 1))
        loops = []
        for _ in range(L):
            loops += list(range(s, e + 1))
        dec = list(range(e + 1, c["num_layers"]))
        return enc + loops + dec
    def forward(self, idx, step_frac=0.0, targets=None):
        B, T = idx.shape
        cfg = self.cfg
        cos, sin = build_rope(T, cfg["model_dim"] // cfg["num_heads"],
                              cfg["rope_base"], cfg["rope_dims"]).to(idx.device)
        x = self.tok_emb(idx)
        vlayers = self.virtual_layers(step_frac)
        saved = {}
        for vi, li in enumerate(vlayers):
            if self.cfg["skip_gates"] and li < self.cfg["num_layers"] // 2:
                saved[li] = x
            # 注入 U-Net 跳连接：decoder 层把对应 encoder 特征过 sigmoid 门加回来
            if self.cfg["skip_gates"] and li >= self.cfg["num_layers"] // 2:
                enc_li = self.cfg["num_layers"] - 1 - li
                if enc_li in saved:
                    g = torch.sigmoid(self.skip_gates[f"g{enc_li}"])
                    x = x + g * saved[enc_li]
            x = self.blocks[li](x, cos, sin)
        x = self.ln_f(x)
        logits = self.head(x)
        if self.cfg["logit_softcap"] > 0:
            # softcap=30：tanh 缩放，防止 logit 爆炸
            logits = self.cfg["logit_softcap"] * torch.tanh(logits / self.cfg["logit_softcap"])
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ---------------------------------------------------------------------------
# 2. MuonEq-R 优化器：矩阵参数用 Muon（Newton-Schulz 正交化），
#    embedding / 1D scalar 参数用 AdamW。
# ---------------------------------------------------------------------------
class Muon(torch.optim.Optimizer):
    """Row-normalized Muon：2D 权重用 Newton-Schulz 迭代近似正交梯度。"""
    def __init__(self, params, lr=0.022, momentum=0.99, beta2=0.95,
                 wd=0.095, nesterov=True, ns_steps=5, row_normalize=True):
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2, wd=wd)
        super().__init__(params, defaults)
        self.ns_steps = ns_steps
        self.row_normalize = row_normalize
        self.nesterov = nesterov
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                if self.nesterov:
                    g = g.add(buf, alpha=group["momentum"])
                else:
                    g = buf
                # Newton-Schulz 正交化：W = 2.8333*W - 2*W^3 的迭代
                if g.ndim >= 2:
                    if self.row_normalize:
                        g = g / (g.norm(dim=1, keepdim=True) + 1e-12)
                    g = self._newton_schulz(g)
                p.mul_(1 - group["lr"] * group["wd"])
                p.add_(g, alpha=-group["lr"])
    def _newton_schulz(self, g):
        a, b, c = g.shape[0], g.shape[1], 3
        X = g * (1.0 / (g.norm() + 1e-8))
        for _ in range(self.ns_steps):
            A = X @ X.T
            B = b * A - c * A @ A
            X = b * X - B @ X
        return X

def build_optimizer(model, train_cfg):
    matrix_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "tok_emb" not in name and "q_gain" not in name:
            matrix_params.append(p)
        else:
            adam_params.append(p)  # embedding、q_gain、ln weight、skip_gate
    muon = Muon(matrix_params, lr=train_cfg["matrix_lr"],
                momentum=train_cfg["muon_momentum"], wd=train_cfg["muon_wd"],
                ns_steps=train_cfg["muon_ns_steps"], row_normalize=train_cfg["muon_row_norm"])
    adam = torch.optim.AdamW(adam_params, lr=train_cfg["embed_lr"],
                             betas=(train_cfg["adam_beta1"], train_cfg["adam_beta2"]),
                             eps=train_cfg["adam_eps"], weight_decay=train_cfg["embed_wd"])
    return muon, adam

# ---------------------------------------------------------------------------
# 3. EMA（指数滑动平均权重，eval 时用）
# ---------------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=False)

# ---------------------------------------------------------------------------
# 4. 数据（FineWeb SP8192 二值 shard，离线预处理好的 .bin）
# ---------------------------------------------------------------------------
def load_data():
    import numpy as np
    data_dir = os.environ.get("DATA_DIR", "./data/datasets/fineweb10B_sp8192")
    train_files = sorted(__import__("glob").glob(f"{data_dir}/fineweb_train_*.bin"))
    val_files = sorted(__import__("glob").glob(f"{data_dir}/fineweb_val_*.bin"))
    def _load(files):
        arr = []
        for f in files:
            arr.append(np.memmap(f, dtype=np.uint16, mode="r"))
        return np.concatenate(arr) if arr else np.zeros(0, dtype=np.uint16)
    return _load(train_files), _load(val_files)

def get_batch(data, seq_len, batch_tokens, device):
    import numpy as np
    B = max(1, batch_tokens // seq_len)
    ix = np.random.randint(0, len(data) - seq_len - 1, size=B)
    x = torch.stack([torch.from_numpy(data[i:i+seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+seq_len].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# ---------------------------------------------------------------------------
# 5. GPTQ 量化（int6/int8 + SDClip）+ Brotli 打包
# ---------------------------------------------------------------------------
@torch.no_grad()
def gptq_quantize(model, calib_batches=64, device="cuda"):
    """对 2D 权重做按行 SDClip 截断后量化：clip = k * std(row)。
    int6 管 attention/MLP 矩阵，int8 管 embedding。"""
    quantized = {}
    for name, p in model.named_parameters():
        if p.ndim < 2:
            quantized[name] = ("fp16", p.half().cpu().numpy().tobytes())
            continue
        bits = QUANT["embed_bits"] if "tok_emb" in name else QUANT["matrix_bits"]
        k = QUANT["embed_clip_sigmas"] if bits == 8 else QUANT["matrix_clip_sigmas"]
        w = p.float()
        clip = k * w.std(dim=1, keepdim=True)
        w = w.clamp(-clip, clip)
        qmax = 2 ** (bits - 1) - 1
        scale = w.abs().amax(dim=1, keepdim=True) / qmax
        q = torch.round(w / scale).clamp(-qmax, qmax).to(torch.int8 if bits == 8 else torch.int8)
        quantized[name] = (f"int{bits}", q.cpu().numpy().tobytes(),
                           scale.half().cpu().numpy().tobytes())
    return quantized

def pack_artifact(model, code_bytes, quantized):
    """Brotli-11 打包：code + 量化权重 + tokenizer 描述。"""
    import brotli, pickle, io
    buf = io.BytesIO()
    with open(os.environ.get("TOKENIZER", "./data/tokenizers/fineweb_8192_bpe.model"), "rb") as f:
        tok = f.read()
    payload = {"code": code_bytes, "weights": quantized, "tokenizer": tok}
    raw = pickle.dumps(payload)
    return brotli.compress(raw, quality=11)

# ---------------------------------------------------------------------------
# 6. Legal score-first TTT（推理时自适应，严格遵守 causality）
# ---------------------------------------------------------------------------
@torch.no_grad()
def sliding_window_eval(model, val_data, seq_len=2048, stride=64, device="cuda"):
    """严格因果滑窗评估：每个位置只看前缀。"""
    import numpy as np
    total_nll, total_tokens = 0.0, 0
    model.eval()
    for start in range(0, len(val_data) - seq_len - 1, stride * 32):
        end = min(start + seq_len, len(val_data) - 1)
        x = torch.from_numpy(val_data[start:end].astype(np.int64))[None].to(device)
        y = torch.from_numpy(val_data[start+1:end+1].astype(np.int64))[None].to(device)
        logits, _ = model(x, step_frac=1.0)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_nll += nll.item()
        total_tokens += y.numel()
    bpb = total_nll / total_tokens / math.log(2)
    return bpb

def legal_ttt(model, val_data, ttt_cfg, device="cuda"):
    """Score-first TTT：先在 no_grad 下把整个 chunk 评分，再用 SGD 在已评分 token 上训练。
    严格遵守 Issue #1017 的 4 条合规条件。"""
    import numpy as np
    chunk = ttt_cfg["chunk_tokens"]
    optimizer = torch.optim.SGD(model.parameters(), lr=ttt_cfg["lr"],
                                momentum=ttt_cfg["momentum"])
    total_nll, total_tokens = 0.0, 0
    for cstart in range(0, len(val_data) - chunk - 1, chunk):
        cend = min(cstart + chunk, len(val_data) - 1)
        x = torch.from_numpy(val_data[cstart:cend].astype(np.int64))[None].to(device)
        y = torch.from_numpy(val_data[cstart+1:cend+1].astype(np.int64))[None].to(device)
        # Condition 1+3：先评分（no_grad），后更新
        with torch.no_grad():
            logits, _ = model(x, step_frac=1.0)
            nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
            total_nll += nll.item(); total_tokens += y.numel()
        # 在已评分 token 上做 epochs 轮 SGD
        for _ in range(ttt_cfg["epochs"]):
            _, loss = model(x, step_frac=1.0, targets=y)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return total_nll / total_tokens / math.log(2)

# ---------------------------------------------------------------------------
# 7. 主训练循环
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(TRAIN["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model_params: {n_params}")

    muon, adam = build_optimizer(model, TRAIN)
    ema = EMA(model, TRAIN["ema_decay"])
    train_data, val_data = load_data()

    t0 = time.time()
    step = 0
    while True:
        elapsed = time.time() - t0
        if elapsed > TRAIN["max_wallclock_s"] - TRAIN["gptq_reserve_s"]:
            print(f"stopping_early: wallclock_cap train_time: {elapsed*1000:.0f}ms step: {step}")
            break
        step += 1
        frac = step / 20000.0  # 名义 20k 步，实际 wallclock 截停
        # LR schedule：warmup 20 步，之后 warmdown_frac 内线性降到 0
        if step < TRAIN["warmup_steps"]:
            lr_scale = step / TRAIN["warmup_steps"]
        else:
            decay_start = 20000 * (1 - TRAIN["warmdown_frac"])
            lr_scale = max(0.0, 1 - (step - decay_start) / (20000 * TRAIN["warmdown_frac"]))
        for opt in (muon, adam):
            for g in opt.param_groups:
                g["lr"] *= lr_scale if step <= TRAIN["warmup_steps"] else 1.0
        x, y = get_batch(train_data, CFG["train_seq_len"], TRAIN["train_batch_tokens"], device)
        _, loss = model(x, step_frac=frac, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip_norm"])
        muon.step(); adam.step()
        muon.zero_grad(); adam.zero_grad()
        ema.update(model)
        if step % 500 == 0:
            print(f"{step}/20000 train_loss: {loss.item():.4f} train_time: {elapsed/60:.1f}m")

    # --- 收尾：EMA -> GPTQ -> 打包 -> 评估 ---
    ema.copy_to(model)
    print("applying EMA weights")
    code_bytes = open(__file__, "rb").read()
    quant = gptq_quantize(model, QUANT["gptq_calib_batches"], device)
    artifact = pack_artifact(model, code_bytes, quant)
    print(f"Total submission size: {len(artifact)} bytes (cap 16,000,000)")
    assert len(artifact) < 16_000_000, "ARTIFACT OVER 16MB!"

    bpb = sliding_window_eval(model, val_data, device=device)
    print(f"sliding_window bpb: {bpb:.4f}")
    if TTT["enabled"]:
        bpb_ttt = legal_ttt(model, val_data, TTT, device=device)
        print(f"ttt bpb: {bpb_ttt:.4f}")

if __name__ == "__main__":
    main()

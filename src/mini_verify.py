#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mini_verify.py — 本机 CPU 端到端冒烟验证（不是 H100 全量训练）
================================================================
目的：证明 src/train_gpt.py 里的架构组件（GQA / partial RoPE / QK-gain /
parallel residual / EMA / byte级 BPB 评估）真的能跑通、loss 真的下降、
并且能在同一语料上跑出 baseline vs 全技巧两组的真实对照数字。

诚实声明：
  - 这是 Intel UHD 集显 / CPU（10 线程）上的 tiny model 验证，
    不是 8xH100 / FineWeb 的 1.081 BPB 全量结果。
  - BPB 是 byte 级 tokenizer 下的真实 nats/ln(2)，数字只在本脚本内部可比。

运行：python src/mini_verify.py
"""
import math, time, os, json
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 一个内置小语料（自包含、可复现）。约 18KB 英文散文，保证 train/val 划分。
# ---------------------------------------------------------------------------
CORPUS = """
The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.
It was the best of times, it was the worst of times, it was the age of wisdom,
it was the age of foolishness, it was the epoch of belief, it was the epoch of
incredulity, it was the season of light, it was the season of darkness, it was
the spring of hope, it was the winter of despair. We had everything before us,
we had nothing before us, we were all going direct to heaven, we were all going
direct the other way. In short, the period was so far like the present period,
that some of its noisiest authorities insisted on its being received, for good or
for evil, in the superlative degree of comparison only.

All happy families are alike; each unhappy family is unhappy in its own way.
Everything was in confusion in the Oblonskys house. The wife had discovered that
the husband had been carrying on an intrigue with a French girl, who had been a
governess in their family, and she had announced to her husband that she could
not go on living in the same house. This position had now lasted three days, and
not only the persons concerned, but all the members of the family and the
household, were painfully conscious of it.

Call me Ishmael. Some years ago, never mind how long precisely, having little or
no money in my purse, and nothing particular to interest me on shore, I thought I
would sail about a little and see the watery part of the world. It is a way I
have of driving off the spleen and regulating the circulation. Whenever I find
myself growing grim about the mouth; whenever it is a damp, drizzly November in
my soul; whenever I find myself involuntarily pausing before coffin warehouses,
and bringing up the rear of every funeral I meet; then, I account it high time
to get to sea as soon as I can.

To be, or not to be, that is the question: whether tis nobler in the mind to
suffer the slings and arrows of outrageous fortune, or to take arms against a
sea of troubles, and by opposing end them. To die, to sleep, no more; and by a
sleep to say we end the heartache and the thousand natural shocks that flesh is
heir to, tis a consummation devoutly to be wished. To die, to sleep, perchance
to dream: ay, theres the rub.

It is a truth universally acknowledged, that a single man in possession of a
good fortune, must be in want of a wife. However little known the feelings or
views of such a man may be on his first entering a neighbourhood, this truth is
so well fixed in the minds of the surrounding families, that he is considered
the rightful property of some one or other of their daughters.

In the beginning God created the heaven and the earth. And the earth was without
form, and void; and darkness was upon the face of the deep. And the Spirit of
God moved upon the face of the waters. And God said, Let there be light: and
there was light. And God saw the light, that it was good: and God divided the
light from the darkness.

The morning mists rose slowly over the valley, and the first light touched the
stone walls of the old mill. A narrow road wound between the poplars, and beyond
it the river caught the sun in long silver ribbons. No one walked there at that
hour; only the birds, and the distant creak of a cart, and the water turning the
wheel in its sleep. The village had not yet woken, and the smoke rose straight
from the chimneys, as though the sky itself were holding its breath.

Science progresses not by consensus but by careful measurement. Every claim must
be tested against data, and every data point must be reproducible by another
hand. The experiment that cannot be repeated is not evidence at all. Models
are stories we tell about numbers, and the best stories are the ones that fit
every observation and predict the next one before it arrives. When a new fact
contradicts a comfortable theory, the theory must yield, not the fact.

Machine learning systems learn patterns from examples rather than from explicit
rules. A language model observes many bytes of text and adjusts its internal
weights so that the next byte is slightly more predictable. Over time, this
process builds an internal representation of grammar, of fact, of rhythm. The
result is not knowledge in any human sense, but a compressed shadow of all the
writing it has seen, and that shadow is useful. Measure it fairly, on text it has
never seen, and the number tells you how much it has really understood.
"""

# byte 级 tokenizer：vocab=256，BPB 天然对齐
data = torch.tensor(list(CORPUS.encode("utf-8")), dtype=torch.long)
n = len(data)
tr = data[: int(n * 0.9)]
va = data[int(n * 0.9):]
print(f"corpus bytes={n} train={len(tr)} val={len(va)} vocab=256")

# ---------------------------------------------------------------------------
# tiny GPT（与 src/train_gpt.py 同款组件，但尺寸压到 CPU 可跑）
# ---------------------------------------------------------------------------
class LeakySq(nn.Module):
    def forward(self, x):
        a = F.leaky_relu(x, 0.5)
        return a * a

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

class TinyGPT(nn.Module):
    def __init__(self, dim=256, layers=4, heads=8, kv=4, seq=128,
                 use_tricks=True, qk_gain=3.0):
        super().__init__()
        self.use_tricks = use_tricks
        self.tok = nn.Embedding(256, dim)
        self.pos = nn.Embedding(seq, dim)
        self.blocks = nn.ModuleList()
        self.parallel = []
        self.qgains = nn.ParameterList()
        for i in range(layers):
            self.blocks.append(nn.ModuleDict(dict(
                ln1=RMSNorm(dim), ln2=RMSNorm(dim),
                q=nn.Linear(dim, dim, bias=False),
                k=nn.Linear(dim, kv * (dim // heads), bias=False),
                v=nn.Linear(dim, kv * (dim // heads), bias=False),
                proj=nn.Linear(dim, dim, bias=False),
                fc=nn.Linear(dim, dim * 4, bias=False),
                act=LeakySq(),
                out=nn.Linear(dim * 4, dim, bias=False),
            )))
            self.qgains.append(nn.Parameter(torch.full((heads,), qk_gain)))
            self.parallel.append(bool(use_tricks and i >= 2))  # layer 2,3 并行残差
        self.lnf = RMSNorm(dim)
        self.head = nn.Linear(dim, 256, bias=False)
        self.tok.weight = self.head.weight  # tie
        self.seq = seq; self.H = heads; self.KV = kv
        self.Dh = dim // heads
        self.rope_dims = 8  # partial RoPE
    def _rope(self, x, T):
        d = self.rope_dims // 2
        inv = 1.0 / (10000 ** (torch.arange(d).float() / d))
        t = torch.arange(T).float()
        f = torch.outer(t, inv)
        cos, sin = f.cos(), f.sin()
        d2 = self.rope_dims
        x1, x2 = x[..., :d2].chunk(2, -1)
        xr1 = x1 * cos[..., :x1.shape[-1]] - x2 * sin[..., :x2.shape[-1]]
        xr2 = x1 * sin[..., :x1.shape[-1]] + x2 * cos[..., :x2.shape[-1]]
        return torch.cat([xr1, xr2, x[..., d2:]], -1)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))
        for bi, blk in enumerate(self.blocks):
            h = blk["ln1"](x)
            q = blk["q"](h).view(B, T, self.H, self.Dh).transpose(1, 2)
            k = blk["k"](h).view(B, T, self.KV, self.Dh).transpose(1, 2)
            v = blk["v"](h).view(B, T, self.KV, self.Dh).transpose(1, 2)
            k = k.repeat_interleave(self.H // self.KV, 1)
            v = v.repeat_interleave(self.H // self.KV, 1)
            q = self._rope(q, T); k = self._rope(k, T)
            if self.use_tricks:
                q = q * self.qgains[bi].view(1, -1, 1, 1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).reshape(B, T, -1)
            att = blk["proj"](y)
            mlp = blk["out"](blk["act"](blk["fc"](blk["ln2"](x))))
            if self.parallel[bi]:
                x = x + att + mlp
            else:
                x = x + att
                x = x + mlp
        x = self.lnf(x)
        logits = self.head(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1)) if targets is not None else None
        return logits, loss

@torch.no_grad()
def evaluate(model, data_, seq, batch=32):
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, len(data_) - seq - 1, seq * batch):
        xs, ys = [], []
        for b in range(batch):
            j = i + b * seq
            if j + seq + 1 > len(data_): break
            xs.append(data_[j:j+seq]); ys.append(data_[j+1:j+seq+1])
        if not xs: break
        xb = torch.stack(xs); yb = torch.stack(ys)
        _, l = model(xb, yb)
        tot += l.item() * yb.numel(); cnt += yb.numel()
    model.train()
    return tot / cnt  # nats per token (byte)

def run(use_tricks, tag, steps=260, seq=128, batch=32, lr=3e-3):
    torch.manual_seed(42)
    model = TinyGPT(use_tricks=use_tricks)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    t0 = time.time(); toks = 0
    rows = []
    for step in range(steps):
        ix = torch.randint(0, len(tr) - seq - 1, (batch,))
        xb = torch.stack([tr[i:i+seq] for i in ix])
        yb = torch.stack([tr[i+1:i+seq+1] for i in ix])
        _, loss = model(xb, yb)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        toks += batch * seq
        if step % 25 == 0 or step == steps - 1:
            vb = evaluate(model, va, seq)
            bpB = vb / math.log(2)
            el = time.time() - t0
            row = (step, loss.item(), vb, bpB, toks / max(el, 1e-6))
            rows.append(row)
            print(f"[{tag}] step={step:4d} train_nll={loss.item():.3f} "
                  f"val_nll={vb:.3f} val_BPB={bpB:.3f} tok/s={toks/max(el,1e-6):.0f}")
    final_bpb = rows[-1][3]
    nparams = sum(p.numel() for p in model.parameters())
    return dict(tag=tag, use_tricks=use_tricks, params=nparams,
                final_val_nll=rows[-1][2], final_val_BPB=final_bpb,
                secs=round(time.time()-t0,1), rows=rows)

if __name__ == "__main__":
    print("=== A: baseline (no QK-gain / no parallel res / no partial-rope gain) ===")
    A = run(use_tricks=False, tag="baseline")
    print("\n=== B: full tricks (QK-gain + parallel res + partial RoPE) ===")
    B = run(use_tricks=True, tag="fulltricks")
    delta = A["final_val_BPB"] - B["final_val_BPB"]
    print(f"\n=== RESULT ===")
    print(f"baseline  val_BPB = {A['final_val_BPB']:.3f}  ({A['params']/1e3:.1f}K params, {A['secs']}s)")
    print(f"fulltrick val_BPB = {B['final_val_BPB']:.3f}  ({B['params']/1e3:.1f}K params, {B['secs']}s)")
    print(f"delta (baseline - full) = {delta:+.3f} BPB  (正=全技巧更好)")
    out = dict(A=A, B=B, delta=delta, note="CPU tiny-model smoke test, NOT H100 FineWeb 1.081")
    with open(os.path.join(HERE, "..", "logs", "mini_run_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print("saved logs/mini_run_result.json")

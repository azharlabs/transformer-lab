// Turns the 3D stack into runnable PyTorch. Only the modules actually used in
// the stack are emitted, then a model class wires them together in order.
// Attention / FFN / MoE / Mamba sub-layers get a residual connection; norms,
// dropout and positional encodings are applied in place.

import { BLOCK_BY_KEY } from './blocks.js'
import { splitStack } from './layout.js'

const RESIDUAL = new Set([
  'mha', 'causal_mha', 'cross_attention', 'gqa', 'mla', 'swa',
  'ffn', 'moe', 'mamba', 'deltanet',
])

// ---- reusable module library (only emitted when referenced) ----
const LIB = {
  mha: `class MultiHeadAttention(nn.Module):
    """Standard scaled dot-product attention. causal=True applies a look-ahead mask."""
    def __init__(self, d_model, n_heads, causal=False, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head, self.causal = n_heads, d_model // n_heads, causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            att = att.masked_fill(mask, float('-inf'))
        att = self.drop(att.softmax(dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)`,

  cross_attention: `class CrossAttention(nn.Module):
    """Queries from the decoder x, keys/values from encoder memory."""
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory=None):
        memory = x if memory is None else memory
        B, T, C = x.shape
        S = memory.shape[1]
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k, v = self.kv(memory).split(C, dim=2)
        k = k.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = self.drop(att.softmax(dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)`,

  gqa: `class GroupedQueryAttention(nn.Module):
    """Q keeps n_heads; K/V share kv_heads (repeated to match)."""
    def __init__(self, d_model, n_heads, kv_heads, causal=True, dropout=0.0):
        super().__init__()
        self.n_heads, self.kv_heads = n_heads, kv_heads
        self.d_head, self.causal = d_model // n_heads, causal
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * kv_heads * self.d_head)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        kv = self.kv(x).view(B, T, 2 * self.kv_heads, self.d_head).transpose(1, 2)
        k, v = kv.split(self.kv_heads, dim=1)
        rep = self.n_heads // self.kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            att = att.masked_fill(mask, float('-inf'))
        att = self.drop(att.softmax(dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)`,

  ffn: `class FeedForward(nn.Module):
    def __init__(self, d_model, mult=4, act='gelu', dropout=0.0):
        super().__init__()
        acts = {'gelu': nn.GELU(), 'relu': nn.ReLU(), 'silu': nn.SiLU()}
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            acts[act],
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)`,

  moe: `class MoE(nn.Module):
    """Top-k routed mixture of FFN experts, with an optional always-on shared expert."""
    def __init__(self, d_model, n_experts=8, top_k=2, mult=4, shared=False):
        super().__init__()
        self.top_k = top_k

        def expert():
            return nn.Sequential(nn.Linear(d_model, mult * d_model), nn.GELU(),
                                 nn.Linear(mult * d_model, d_model))

        self.gate = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([expert() for _ in range(n_experts)])
        self.shared = expert() if shared else None

    def forward(self, x):
        scores = self.gate(x)
        weights, idx = scores.topk(self.top_k, dim=-1)
        weights = weights.softmax(dim=-1)
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            for e, expert in enumerate(self.experts):
                m = (idx[..., k] == e)
                if m.any():
                    out[m] += weights[..., k][m].unsqueeze(-1) * expert(x[m])
        if self.shared is not None:
            out = out + self.shared(x)   # shared expert: always active
        return out`,

  swiglu: `class SwiGLU(nn.Module):
    """Gated feed-forward used by Llama/Qwen3/Gemma: (silu(xW1) * xW3) W2."""
    def __init__(self, d_model, mult=4):
        super().__init__()
        hidden = mult * d_model
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))`,

  mla: `class MultiHeadLatentAttention(nn.Module):
    """Simplified MLA: compress K/V into a small latent before the (conceptual) cache,
    then project back up to per-head keys/values. Causal."""
    def __init__(self, d_model, n_heads, latent=64, dropout=0.0):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv_down = nn.Linear(d_model, latent)              # compress
        self.k_up = nn.Linear(latent, d_model)                # decompress K
        self.v_up = nn.Linear(latent, d_model)                # decompress V
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        c = self.kv_down(x)                                    # (B, T, latent)  <- this is what gets cached
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_up(c).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_up(c).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = self.drop(att.masked_fill(mask, float('-inf')).softmax(dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)`,

  swa: `class SlidingWindowAttention(nn.Module):
    """Causal attention restricted to a local window of 'window' past tokens."""
    def __init__(self, d_model, n_heads, window=1024, dropout=0.0):
        super().__init__()
        self.n_heads, self.d_head, self.window = n_heads, d_model // n_heads, window
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        i = torch.arange(T, device=x.device)
        future = i[None, :] > i[:, None]
        too_far = i[None, :] <= i[:, None] - self.window
        att = att.masked_fill((future | too_far), float('-inf'))
        att = self.drop(att.softmax(dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)`,

  deltanet: `class GatedDeltaNet(nn.Module):
    """Very simplified gated linear-attention (DeltaNet-style) block: a cache-free
    recurrence with a forget gate beta. Illustrative, not the optimised kernel."""
    def __init__(self, d_model, expand=2):
        super().__init__()
        d = expand * d_model
        self.qkv = nn.Linear(d_model, 3 * d)
        self.beta = nn.Linear(d_model, 1)
        self.out = nn.Linear(d, d_model)
        self.d = d

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(self.d, dim=-1)
        q, k = F.silu(q), F.silu(k)
        beta = torch.sigmoid(self.beta(x))                    # (B, T, 1) forget gate
        S = torch.zeros(B, self.d, self.d, device=x.device)   # fast-weight memory
        ys = []
        for t in range(T):
            kt, vt = k[:, t], v[:, t]
            S = beta[:, t].unsqueeze(-1) * S + torch.einsum('bi,bj->bij', vt, kt)
            ys.append(torch.einsum('bij,bj->bi', S, q[:, t]))
        return self.out(torch.stack(ys, dim=1))`,

  mamba: `class MambaBlock(nn.Module):
    """Minimal selective state-space block (simplified, sequential scan)."""
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        d_inner = expand * d_model
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, 3, padding=2, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1)
        self.dt_proj = nn.Linear(1, d_inner)
        self.A = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)
        self.d_inner, self.d_state = d_inner, d_state

    def forward(self, x):
        B, T, _ = x.shape
        xz = self.in_proj(x)
        xc, z = xz.chunk(2, dim=-1)
        xc = self.conv(xc.transpose(1, 2))[..., :T].transpose(1, 2)
        xc = F.silu(xc)
        dbl = self.x_proj(xc)
        dt, Bm, Cm = dbl[..., :1], dbl[..., 1:1 + self.d_state], dbl[..., 1 + self.d_state:]
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A)                       # (d_inner, d_state)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)               # (B, d_inner, d_state)
            dB = dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1)      # (B, d_inner, d_state)
            h = dA * h + dB * xc[:, t].unsqueeze(-1)
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + xc * self.D
        return self.out_proj(y * F.silu(z))`,

  rmsnorm: `class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight`,
}

function ctor(blk, cfg) {
  const p = blk.params || {}
  switch (blk.key) {
    case 'embedding':
      return `nn.Embedding(${cfg.vocabSize}, ${cfg.dModel})`
    case 'pos_learned':
      return `LearnedPositional(${cfg.maxSeqLen}, ${cfg.dModel})`
    case 'pos_sinusoidal':
      return `SinusoidalPositional(${cfg.maxSeqLen}, ${cfg.dModel})`
    case 'pos_rotary':
      return `nn.Identity()` // RoPE is applied inside attention
    case 'pos_none':
      return `nn.Identity()` // NoPE — no positional signal added
    case 'mha':
      return `MultiHeadAttention(${cfg.dModel}, n_heads=${p.heads}, causal=False, dropout=${cfg.dropout})`
    case 'causal_mha':
      return `MultiHeadAttention(${cfg.dModel}, n_heads=${p.heads}, causal=True, dropout=${cfg.dropout})`
    case 'cross_attention':
      return `CrossAttention(${cfg.dModel}, n_heads=${p.heads}, dropout=${cfg.dropout})`
    case 'gqa':
      return `GroupedQueryAttention(${cfg.dModel}, n_heads=${p.heads}, kv_heads=${p.kv_heads}, dropout=${cfg.dropout})`
    case 'mla':
      return `MultiHeadLatentAttention(${cfg.dModel}, n_heads=${p.heads}, latent=${p.latent}, dropout=${cfg.dropout})`
    case 'swa':
      return `SlidingWindowAttention(${cfg.dModel}, n_heads=${p.heads}, window=${p.window}, dropout=${cfg.dropout})`
    case 'ffn':
      return p.act === 'swiglu'
        ? `SwiGLU(${cfg.dModel}, mult=${p.mult})`
        : `FeedForward(${cfg.dModel}, mult=${p.mult}, act='${p.act}', dropout=${cfg.dropout})`
    case 'moe':
      return `MoE(${cfg.dModel}, n_experts=${p.experts}, top_k=${p.topk}, shared=${p.shared === 'yes' ? 'True' : 'False'})`
    case 'mamba':
      return `MambaBlock(${cfg.dModel}, d_state=${p.d_state}, expand=${p.expand})`
    case 'deltanet':
      return `GatedDeltaNet(${cfg.dModel}, expand=${p.expand})`
    case 'layernorm':
      return `nn.LayerNorm(${cfg.dModel})`
    case 'rmsnorm':
      return `RMSNorm(${cfg.dModel})`
    case 'qk_norm':
      return `nn.Identity()` // QK-Norm lives inside the attention module
    case 'dropout':
      return `nn.Dropout(${p.p})`
    case 'lm_head':
      return p.mode === 'classification'
        ? `nn.Linear(${cfg.dModel}, ${cfg.numClasses})`
        : `nn.Linear(${cfg.dModel}, ${cfg.vocabSize})`
    default:
      return `nn.Identity()`
  }
}

export function generatePyTorch(stack, cfg) {
  if (!stack.length) return '# Add blocks in the Builder to generate code.'
  const used = new Set(stack.map((b) => b.key))
  const pieces = []

  pieces.push(`"""${cfg.modelName} — generated by Transformer Lab.
A faithful-but-readable PyTorch implementation of the stack you built.
Attention/FFN/Mamba sub-layers use residual connections; place a norm
*before* them for the modern pre-norm arrangement.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F`)

  // positional helpers
  if (used.has('pos_learned')) {
    pieces.push(`class LearnedPositional(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x):
        t = torch.arange(x.size(1), device=x.device)
        return x + self.pos(t)[None]`)
  }
  if (used.has('pos_sinusoidal')) {
    pieces.push(`class SinusoidalPositional(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)][None]`)
  }
  const usesPlainFFN = stack.some((b) => b.key === 'ffn' && b.params.act !== 'swiglu')
  const usesSwiGLU = stack.some((b) => b.key === 'ffn' && b.params.act === 'swiglu')
  if (used.has('mha') || used.has('causal_mha')) pieces.push(LIB.mha)
  if (used.has('cross_attention')) pieces.push(LIB.cross_attention)
  if (used.has('gqa')) pieces.push(LIB.gqa)
  if (used.has('mla')) pieces.push(LIB.mla)
  if (used.has('swa')) pieces.push(LIB.swa)
  if (usesPlainFFN) pieces.push(LIB.ffn)
  if (usesSwiGLU) pieces.push(LIB.swiglu)
  if (used.has('moe')) pieces.push(LIB.moe)
  if (used.has('mamba')) pieces.push(LIB.mamba)
  if (used.has('deltanet')) pieces.push(LIB.deltanet)
  if (used.has('rmsnorm')) pieces.push(LIB.rmsnorm)

  // model class — split into input head, a transformer body repeated N times, and a tail
  const { head, body, tail } = splitStack(stack)
  const N = Math.max(1, parseInt(cfg.layers, 10) || 1)
  const resList = (blocks) => `[${blocks.map((b) => (RESIDUAL.has(b.key) ? 'True' : 'False')).join(', ')}]`
  const memList = (blocks) => `[${blocks.map((b) => (b.key === 'cross_attention' ? 'True' : 'False')).join(', ')}]`
  const ctorLines = (blocks, indent) =>
    blocks
      .map((b) => `${indent}${ctor(b, cfg)},  # ${BLOCK_BY_KEY[b.key]?.name || b.key}`)
      .join('\n')

  const lines = []
  lines.push(`class ${cfg.modelName}(nn.Module):`)
  lines.push(`    N_LAYERS = ${N}   # the transformer body is repeated this many times`)
  lines.push(``)
  lines.push(`    def __init__(self):`)
  lines.push(`        super().__init__()`)
  lines.push(`        self.head = nn.ModuleList([`)
  lines.push(ctorLines(head, '            '))
  lines.push(`        ])`)
  lines.push(`        self.blocks = nn.ModuleList([nn.ModuleList([`)
  lines.push(ctorLines(body, '            '))
  lines.push(`        ]) for _ in range(self.N_LAYERS)])`)
  lines.push(`        self.tail = nn.ModuleList([`)
  lines.push(ctorLines(tail, '            '))
  lines.push(`        ])`)
  lines.push(`        # residual + encoder-memory flags for each section`)
  lines.push(`        self.head_res, self.head_mem = ${resList(head)}, ${memList(head)}`)
  lines.push(`        self.body_res, self.body_mem = ${resList(body)}, ${memList(body)}`)
  lines.push(`        self.tail_res, self.tail_mem = ${resList(tail)}, ${memList(tail)}`)
  lines.push(``)
  lines.push(`    @staticmethod`)
  lines.push(`    def _run(x, layers, res, mem, memory):`)
  lines.push(`        for layer, r, m in zip(layers, res, mem):`)
  lines.push(`            if r:`)
  lines.push(`                out = layer(x, memory) if m else layer(x)`)
  lines.push(`                x = x + out          # residual connection`)
  lines.push(`            else:`)
  lines.push(`                x = layer(x)`)
  lines.push(`        return x`)
  lines.push(``)
  lines.push(`    def forward(self, idx, memory=None):`)
  lines.push(`        # idx: (B, T) integer token ids`)
  lines.push(`        x = idx`)
  lines.push(`        x = self._run(x, self.head, self.head_res, self.head_mem, memory)`)
  lines.push(`        for block in self.blocks:          # N_LAYERS transformer blocks`)
  lines.push(`            x = self._run(x, block, self.body_res, self.body_mem, memory)`)
  lines.push(`        x = self._run(x, self.tail, self.tail_res, self.tail_mem, memory)`)
  lines.push(`        return x  # final logits`)
  pieces.push(lines.join('\n'))

  // smoke test
  pieces.push(`if __name__ == "__main__":
    model = ${cfg.modelName}()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"${cfg.modelName}: {n_params/1e6:.1f}M parameters")
    ids = torch.randint(0, ${cfg.vocabSize}, (2, 32))
    print("output:", model(ids).shape)`)

  return pieces.join('\n\n\n')
}

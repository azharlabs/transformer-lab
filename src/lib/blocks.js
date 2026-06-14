// Definitions for every component ("part") you can drop into the 3D builder.
// Each block has a category, a colour used in the 3D scene, a human description,
// editable parameters, and (where useful) pros/cons + which real models use it.

export const CATEGORY_COLORS = {
  io: '#38bdf8',        // sky    — embeddings / heads
  position: '#a78bfa',  // violet — positional encodings
  attention: '#fb7185', // rose   — attention variants
  mixing: '#34d399',    // emerald— FFN / MoE / Mamba / DeltaNet
  norm: '#fbbf24',      // amber  — normalisation
  reg: '#94a3b8',       // slate  — dropout / misc
}

export const BLOCKS = [
  {
    key: 'embedding',
    name: 'Token Embedding',
    category: 'io',
    short: 'Maps token ids → vectors',
    desc: 'A lookup table of shape (vocab_size, d_model). Turns integer token ids into dense vectors the rest of the network operates on.',
    params: {},
  },
  {
    key: 'pos_sinusoidal',
    name: 'Sinusoidal Pos. Enc.',
    category: 'position',
    short: 'Fixed sin/cos positions',
    desc: 'Adds fixed sine/cosine signals so the model knows token order. No learned weights — the original "Attention Is All You Need" scheme.',
    params: {},
    pros: ['No parameters', 'Extrapolates somewhat to longer sequences'],
    cons: ['Weaker than RoPE in modern LLMs'],
    usedIn: 'Original Transformer, T5',
  },
  {
    key: 'pos_learned',
    name: 'Learned Pos. Embed.',
    category: 'position',
    short: 'Trainable position table',
    desc: 'A trainable embedding indexed by position (used by GPT-2/BERT). One vector per position up to max_seq_len.',
    params: {},
    pros: ['Simple', 'Flexible'],
    cons: ['Fixed maximum length', 'Poor length generalisation'],
    usedIn: 'GPT-2, BERT, OPT',
  },
  {
    key: 'pos_rotary',
    name: 'Rotary (RoPE)',
    category: 'position',
    short: 'Rotates Q/K by position',
    desc: 'Rotary Positional Embeddings rotate the query/key vectors by an angle that depends on position. Encodes relative position and is applied inside attention.',
    params: {},
    pros: ['Relative positions', 'Strong long-context behaviour', 'No extra parameters'],
    cons: ['Needs frequency retuning / scaling for very long contexts'],
    usedIn: 'Llama, Qwen3, Gemma 3, almost every 2025 LLM',
  },
  {
    key: 'pos_none',
    name: 'No Positional (NoPE)',
    category: 'position',
    short: 'No position signal at all',
    desc: 'Adds no positional information — not learned, fixed, or rotary. The causal mask alone gives an implicit sense of order. SmolLM3 applies this on every 4th layer.',
    params: {},
    pros: ['Best length generalisation in ablations', 'Zero positional parameters', 'Cheapest'],
    cons: ['Risky to use in every layer', 'Less studied at large scale'],
    usedIn: 'SmolLM3 (every 4th layer)',
  },
  {
    key: 'mha',
    name: 'Multi-Head Attention',
    category: 'attention',
    short: 'Full self-attention, all heads',
    desc: 'Every token attends to every other token; each head has its own queries, keys and values. The classic mechanism — highest quality, highest KV-cache cost.',
    params: { heads: { label: 'Heads', type: 'int', default: 8, min: 1, max: 64 } },
    pros: ['Best modeling quality per the DeepSeek-V2 ablations', 'Simple, well-optimised'],
    cons: ['Largest KV cache', 'O(n²) cost', 'Most memory bandwidth at inference'],
    usedIn: 'GPT-2, OLMo 2',
  },
  {
    key: 'causal_mha',
    name: 'Masked (Causal) Attention',
    category: 'attention',
    short: 'Each token sees only the past',
    desc: 'Self-attention with a causal mask so position i can only attend to positions ≤ i. This is what makes a model auto-regressive (GPT-style).',
    params: { heads: { label: 'Heads', type: 'int', default: 8, min: 1, max: 64 } },
    pros: ['Enables left-to-right generation'],
    cons: ['Can only look backward'],
    usedIn: 'Every decoder-only LLM',
  },
  {
    key: 'gqa',
    name: 'Grouped-Query Attention',
    category: 'attention',
    short: 'Fewer KV heads, shared',
    desc: 'Queries keep n_heads but keys/values share a smaller number of KV heads. Big memory/latency win at inference with little quality loss.',
    params: {
      heads: { label: 'Q Heads', type: 'int', default: 8, min: 1, max: 64 },
      kv_heads: { label: 'KV Heads', type: 'int', default: 2, min: 1, max: 32 },
    },
    pros: ['Much smaller KV cache than MHA', 'Faster inference', 'Near-MHA quality'],
    cons: ['Slightly below MHA/MLA on quality in some ablations'],
    usedIn: 'Llama 3/4, Qwen3, Gemma 3, Mistral',
  },
  {
    key: 'mla',
    name: 'Multi-Head Latent Attention',
    category: 'attention',
    short: 'Compresses K/V to a latent',
    desc: 'Instead of sharing KV heads (GQA), MLA compresses keys/values into a low-dimensional latent before the KV cache, then projects back up at use time. An extra matmul buys a much smaller cache.',
    params: {
      heads: { label: 'Heads', type: 'int', default: 16, min: 1, max: 128 },
      latent: { label: 'KV latent dim', type: 'int', default: 64, min: 8, max: 512 },
    },
    pros: ['Smallest KV cache', 'Slightly *better* quality than MHA in DeepSeek-V2 ablations'],
    cons: ['More complex to implement', 'Extra projection compute'],
    usedIn: 'DeepSeek V3/R1, Kimi K2, Kimi Linear',
  },
  {
    key: 'swa',
    name: 'Sliding-Window Attention',
    category: 'attention',
    short: 'Local window, causal',
    desc: 'Restricts each query to a fixed window of nearby tokens (local attention) instead of the whole sequence. Cuts KV-cache memory dramatically; usually interleaved with a few global layers.',
    params: {
      heads: { label: 'Heads', type: 'int', default: 8, min: 1, max: 64 },
      window: { label: 'Window', type: 'int', default: 1024, min: 64, max: 8192, step: 64 },
    },
    pros: ['Much smaller KV cache', 'Near-global quality (Gemma 3 ablations)', 'Scales to long context'],
    cons: ['Local view only', 'May not speed up latency (just memory)'],
    usedIn: 'Gemma 2/3 (5:1 local:global), GPT-OSS (every other layer)',
  },
  {
    key: 'cross_attention',
    name: 'Cross-Attention',
    category: 'attention',
    short: 'Decoder attends to encoder',
    desc: 'Queries come from the decoder, keys/values from the encoder output. The bridge in encoder–decoder models (translation, T5).',
    params: { heads: { label: 'Heads', type: 'int', default: 8, min: 1, max: 64 } },
    pros: ['Lets a decoder condition on a full encoded input'],
    cons: ['Only relevant for encoder–decoder models'],
    usedIn: 'Original Transformer, T5, BART',
  },
  {
    key: 'ffn',
    name: 'Feed-Forward (MLP)',
    category: 'mixing',
    short: 'Per-token nonlinearity',
    desc: 'Two linear layers with an activation in between, applied to each position independently. SwiGLU (a gated variant) is the modern default.',
    params: {
      mult: { label: 'FF mult', type: 'int', default: 4, min: 1, max: 8 },
      act: { label: 'Activation', type: 'enum', options: ['swiglu', 'gelu', 'relu', 'silu'], default: 'swiglu' },
    },
    pros: ['Where most per-token "thinking" happens', 'SwiGLU gating improves quality'],
    cons: ['Holds most of the dense parameters', 'Runs for every token'],
    usedIn: 'Every transformer (SwiGLU in Llama, Qwen3, Gemma 3)',
  },
  {
    key: 'moe',
    name: 'Mixture of Experts',
    category: 'mixing',
    short: 'Sparse expert routing',
    desc: 'Replaces the FFN with many expert MLPs; a router sends each token to its top-k experts. A shared expert (always on) can capture common patterns. Huge capacity at fixed inference cost.',
    params: {
      experts: { label: 'Experts', type: 'int', default: 128, min: 2, max: 512 },
      topk: { label: 'Active (top-k)', type: 'int', default: 8, min: 1, max: 16 },
      shared: { label: 'Shared expert', type: 'enum', options: ['yes', 'no'], default: 'yes' },
    },
    pros: ['Massive total capacity', 'Cheap inference (only top-k active)', 'Shared expert reduces redundancy'],
    cons: ['Large memory footprint', 'Routing adds complexity / instability', 'Harder to fine-tune'],
    usedIn: 'DeepSeek V3, Llama 4, Qwen3, GPT-OSS, Kimi K2, GLM-4.5',
  },
  {
    key: 'mamba',
    name: 'Mamba (SSM) Block',
    category: 'mixing',
    short: 'Linear-time state space',
    desc: 'A selective state-space model that mixes tokens over time with a recurrence instead of attention. Linear in sequence length — no N×N matrix.',
    params: {
      d_state: { label: 'State dim', type: 'int', default: 16, min: 4, max: 128 },
      expand: { label: 'Expand', type: 'int', default: 2, min: 1, max: 4 },
    },
    pros: ['O(n) memory & compute', 'Great for very long context'],
    cons: ['Fixed-size state limits precise recall', 'No direct token-to-token lookup'],
    usedIn: 'Mamba, Jamba, Granite 4.0 (hybrid)',
  },
  {
    key: 'deltanet',
    name: 'Gated DeltaNet',
    category: 'mixing',
    short: 'Linear attention (fast-weight)',
    desc: 'A linear-attention block that keeps a tiny fast-weight memory updated by a gated delta rule and read with the query. A cache-free alternative to attention, usually mixed 3:1 with full-attention layers.',
    params: {
      expand: { label: 'Expand', type: 'int', default: 2, min: 1, max: 4 },
    },
    pros: ['Linear-time, cache-free', 'Enables very long context (256k+)'],
    cons: ['Less precise content retrieval than full attention', 'Needs some full-attention layers'],
    usedIn: 'Qwen3-Next, Kimi Linear (KDA)',
  },
  {
    key: 'layernorm',
    name: 'Layer Norm',
    category: 'norm',
    short: 'Normalises each token',
    desc: 'Normalises features per token to zero mean / unit variance, then scales & shifts. Stabilises training. Pre-norm placement is standard today.',
    params: {},
    usedIn: 'GPT-2, BERT',
  },
  {
    key: 'rmsnorm',
    name: 'RMS Norm',
    category: 'norm',
    short: 'Cheaper normalisation',
    desc: 'Like LayerNorm but only rescales by the root-mean-square (no mean subtraction). Slightly faster, works just as well — the modern default.',
    params: {},
    pros: ['Fewer parameters than LayerNorm', 'Cheaper'],
    cons: ['No mean centring (rarely matters)'],
    usedIn: 'Llama, Qwen3, Gemma 3, OLMo 2',
  },
  {
    key: 'qk_norm',
    name: 'QK-Norm',
    category: 'norm',
    short: 'Normalises Q & K in attention',
    desc: 'An extra RMSNorm applied to the queries and keys (before RoPE) inside the attention module. Stabilises training, especially with Post-Norm. A marker here — in code it lives inside attention.',
    params: {},
    pros: ['Stabilises training loss', 'Cheap'],
    cons: ['Tiny extra compute'],
    usedIn: 'OLMo 2, Gemma 2/3, Qwen3, MiniMax-M2',
  },
  {
    key: 'dropout',
    name: 'Dropout',
    category: 'reg',
    short: 'Regularisation',
    desc: 'Randomly zeroes activations during training to reduce overfitting. A no-op at inference. Largely dropped in big modern LLMs.',
    params: { p: { label: 'p', type: 'float', default: 0.1, min: 0, max: 0.9, step: 0.05 } },
  },
  {
    key: 'lm_head',
    name: 'LM / Output Head',
    category: 'io',
    short: 'Projects to logits',
    desc: 'A final linear layer mapping d_model → vocab_size (language modelling) or → num_classes (classification). Often tied to the embedding weights.',
    params: {
      mode: { label: 'Mode', type: 'enum', options: ['language', 'classification'], default: 'language' },
    },
  },
]

export const CATEGORY_BADGE = {
  io: 'I/O',
  position: 'POS',
  attention: 'ATT',
  mixing: 'MIX',
  norm: 'NRM',
  reg: 'REG',
}

export const BLOCK_BY_KEY = Object.fromEntries(BLOCKS.map((b) => [b.key, b]))

export function blockColor(key) {
  const b = BLOCK_BY_KEY[key]
  return b ? CATEGORY_COLORS[b.category] : '#64748b'
}

export function blockBadge(key) {
  const b = BLOCK_BY_KEY[key]
  return b ? CATEGORY_BADGE[b.category] : '·'
}

export function blockCategory(key) {
  return BLOCK_BY_KEY[key]?.category
}

export function defaultParams(key) {
  const b = BLOCK_BY_KEY[key]
  if (!b) return {}
  const out = {}
  for (const [k, spec] of Object.entries(b.params || {})) out[k] = spec.default
  return out
}

// Architecture presets.
//
// Two groups:
//  • FUNDAMENTALS — textbook families (encoder-only, decoder-only, enc-dec, SSM)
//  • ZOO — real 2024-2026 flagship LLMs from Sebastian Raschka's
//    "The Big LLM Architecture Comparison", each annotated with the components
//    that make it distinctive plus its advantages / disadvantages.
//
// `spec` is a short, representative stack (a couple of layers), not the real
// depth — enough to show each model's signature components in the 3D tower.

const RMS = { key: 'rmsnorm' }
const QK = { key: 'qk_norm' }
const ROPE = { key: 'pos_rotary' }
const SWIGLU = { key: 'ffn', params: { act: 'swiglu', mult: 4 } }

// ---- fundamentals ----
function encoderLayer() {
  return [{ key: 'layernorm' }, { key: 'mha' }, { key: 'dropout' }, { key: 'layernorm' }, { key: 'ffn', params: { act: 'gelu' } }, { key: 'dropout' }]
}

export const ARCHITECTURES = {
  // ---------------- fundamentals ----------------
  encoder: {
    label: 'Encoder-only (BERT-style)',
    group: 'fundamentals',
    examples: 'BERT, RoBERTa',
    blurb: 'Bidirectional. Every token sees every other token, building rich context for understanding tasks — classification, retrieval, embeddings. Cannot generate text left-to-right.',
    pros: ['Rich bidirectional context', 'Great for understanding/embeddings'],
    cons: ['Cannot generate autoregressively'],
    components: ['MHA', 'Learned pos.', 'Pre-Norm', 'GELU FFN'],
    config: { modelName: 'EncoderModel', numClasses: 2 },
    spec: [{ key: 'embedding' }, { key: 'pos_learned' }, ...encoderLayer(), ...encoderLayer(), { key: 'lm_head', params: { mode: 'classification' } }],
  },
  decoder: {
    label: 'Decoder-only (GPT-style)',
    group: 'fundamentals',
    examples: 'GPT-2 → modern LLMs',
    blurb: 'Auto-regressive. A causal mask lets each token attend only to the past, so the model predicts the next token. The backbone of every modern LLM.',
    pros: ['Generates text', 'Simple, scalable'],
    cons: ['Only looks backward', 'O(n²) attention'],
    components: ['Causal MHA', 'RoPE', 'Pre-Norm', 'SwiGLU'],
    config: { modelName: 'DecoderLM', vocabSize: 32000 },
    spec: [{ key: 'embedding' }, ROPE, RMS, { key: 'causal_mha' }, RMS, SWIGLU, RMS, { key: 'causal_mha' }, RMS, SWIGLU, RMS, { key: 'lm_head' }],
  },
  encdec: {
    label: 'Encoder–Decoder (T5-style)',
    group: 'fundamentals',
    examples: 'T5, BART',
    blurb: 'An encoder reads the input bidirectionally; a decoder generates output one token at a time and pulls from the encoder via cross-attention. Ideal for translation & summarisation.',
    pros: ['Strong for seq-to-seq', 'Conditions on full input'],
    cons: ['Heavier than decoder-only', 'Less common for chat LLMs'],
    components: ['Cross-attention', 'Sinusoidal pos.', 'MHA'],
    config: { modelName: 'Seq2Seq', vocabSize: 32000 },
    spec: [{ key: 'embedding' }, { key: 'pos_sinusoidal' }, { key: 'layernorm' }, { key: 'causal_mha' }, { key: 'layernorm' }, { key: 'cross_attention' }, { key: 'layernorm' }, { key: 'ffn', params: { act: 'gelu' } }, { key: 'layernorm' }, { key: 'lm_head' }],
  },
  mamba: {
    label: 'Mamba (State-Space)',
    group: 'fundamentals',
    examples: 'Mamba, Jamba',
    blurb: 'Replaces attention with a selective state-space recurrence. Cost grows linearly with sequence length, so very long contexts stay cheap — at the price of the global "attend anywhere" view.',
    pros: ['O(n) memory & compute', 'Excellent for long sequences'],
    cons: ['Fixed-size state limits recall', 'No direct token lookup'],
    components: ['Mamba SSM', 'RMSNorm'],
    config: { modelName: 'MambaLM', vocabSize: 32000 },
    spec: [{ key: 'embedding' }, RMS, { key: 'mamba' }, RMS, { key: 'mamba' }, RMS, { key: 'mamba' }, RMS, { key: 'lm_head' }],
  },

  // ---------------- the zoo ----------------
  deepseek_v3: {
    label: 'DeepSeek V3 / R1',
    group: 'zoo',
    meta: '671B total · 37B active · Dec 2024',
    blurb: 'Pioneered Multi-Head Latent Attention (compress KV to a latent) plus a big MoE with a shared expert. First 3 layers are dense, the rest MoE. Outperformed Llama 3 405B while activating only 37B params.',
    pros: ['MLA: smallest KV cache + better quality than MHA', 'Huge MoE capacity, cheap inference', 'Shared expert cuts redundancy'],
    cons: ['MLA is complex to implement', 'Enormous total memory footprint'],
    components: ['MLA', 'MoE + shared expert', '3 dense layers first', 'RoPE', 'SwiGLU'],
    config: { modelName: 'DeepSeekV3', vocabSize: 129280, dModel: 256 },
    spec: [{ key: 'embedding' }, ROPE, RMS, { key: 'mla', params: { heads: 16, latent: 64 } }, RMS, SWIGLU, RMS, { key: 'mla', params: { heads: 16, latent: 64 } }, RMS, { key: 'moe', params: { experts: 256, topk: 8, shared: 'yes' } }, RMS, { key: 'lm_head' }],
  },
  olmo2: {
    label: 'OLMo 2',
    group: 'zoo',
    meta: '7B–32B · fully open · Jan 2025',
    blurb: 'A transparency-first model (open data + code). Architecturally notable for Post-Norm (RMSNorm after the sublayers, inside the residual) plus QK-Norm, which together stabilise training. Still uses classic MHA.',
    pros: ['Fully reproducible / open', 'Post-Norm + QK-Norm stabilise loss', 'Clean blueprint'],
    cons: ['MHA = larger KV cache than GQA/MLA', 'Not benchmark-topping'],
    components: ['MHA', 'Post-Norm', 'QK-Norm', 'RoPE', 'SwiGLU'],
    config: { modelName: 'OLMo2', vocabSize: 100352, dModel: 256 },
    spec: [{ key: 'embedding' }, ROPE, QK, { key: 'mha' }, RMS, SWIGLU, RMS, QK, { key: 'mha' }, RMS, SWIGLU, RMS, { key: 'lm_head' }],
  },
  gemma3: {
    label: 'Gemma 3',
    group: 'zoo',
    meta: '1B–27B · Mar 2025',
    blurb: 'Cuts KV-cache memory with sliding-window (local) attention in a 5:1 ratio to full attention, window shrunk to 1024. Uses both Pre-Norm and Post-Norm RMSNorm around each sublayer, plus QK-Norm. A great local model.',
    pros: ['Sliding window slashes KV cache', 'Little quality loss (ablations)', 'Pre+Post norm = extra stability'],
    cons: ['Local layers see only a window', 'Window may not cut latency, only memory'],
    components: ['GQA', 'Sliding-window 5:1', 'Pre+Post Norm', 'QK-Norm', 'RoPE'],
    config: { modelName: 'Gemma3', vocabSize: 262144, dModel: 256 },
    spec: [{ key: 'embedding' }, ROPE, RMS, { key: 'swa', params: { heads: 8, window: 1024 } }, RMS, SWIGLU, RMS, { key: 'swa', params: { heads: 8, window: 1024 } }, RMS, SWIGLU, RMS, { key: 'gqa' }, RMS, SWIGLU, RMS, { key: 'lm_head' }],
  },
  mistral31: {
    label: 'Mistral Small 3.1',
    group: 'zoo',
    meta: '24B · Mar 2025',
    blurb: 'A clean, fast GQA decoder. Notably dropped the sliding-window attention of earlier Mistral models so it can use optimised FlashAttention kernels — prioritising low latency over the memory savings of windows.',
    pros: ['Low inference latency', 'Standard GQA → FlashAttention-friendly', 'Beats Gemma 3 27B on several benchmarks'],
    cons: ['Larger KV cache than windowed models', 'Few architectural novelties'],
    components: ['GQA', 'RoPE', 'SwiGLU', 'RMSNorm'],
    config: { modelName: 'MistralSmall31', vocabSize: 131072, dModel: 256 },
    spec: [{ key: 'embedding' }, ROPE, RMS, { key: 'gqa' }, RMS, SWIGLU, RMS, { key: 'gqa' }, RMS, SWIGLU, RMS, { key: 'lm_head' }],
  },
  llama4: {
    label: 'Llama 4 Maverick',
    group: 'zoo',
    meta: '400B total · 17B active · Apr 2025',
    blurb: 'Meta\'s MoE decoder, very similar to DeepSeek V3 but with GQA instead of MLA and a more classic MoE: fewer, larger experts (2 active) that alternate with dense layers every other block.',
    pros: ['MoE scale at 17B active', 'GQA is simple & well-optimised', 'Alternating dense layers aid stability'],
    cons: ['Fewer/larger experts less efficient than many-small', 'No MLA cache savings'],
    components: ['GQA', 'MoE (few large experts)', 'Alternating dense/MoE', 'RoPE', 'SwiGLU'],
    config: { modelName: 'Llama4', vocabSize: 202048, dModel: 256 },
    spec: [{ key: 'embedding' }, ROPE, RMS, { key: 'gqa' }, RMS, SWIGLU, RMS, { key: 'gqa' }, RMS, { key: 'moe', params: { experts: 128, topk: 1, shared: 'yes' } }, RMS, { key: 'lm_head' }],
  },
  qwen3_dense: {
    label: 'Qwen3 (Dense)',
    group: 'zoo',
    meta: '0.6B–32B · May 2025',
    blurb: 'A deep, narrow GQA decoder with QK-Norm. The 0.6B may be the smallest strong current-gen model — easy to run and train locally. Dense models are simpler to fine-tune and deploy.',
    pros: ['Excellent quality per size', 'Easy to fine-tune & deploy', 'QK-Norm stability'],
    cons: ['Deeper → slower tokens/sec than wide models', 'Dense = no MoE capacity trick'],
    components: ['GQA', 'QK-Norm', 'RoPE', 'SwiGLU', 'deep & narrow'],
    config: { modelName: 'Qwen3Dense', vocabSize: 151936, dModel: 256 },
    spec: [{ key: 'embedding' }, ROPE, RMS, QK, { key: 'gqa' }, RMS, SWIGLU, RMS, QK, { key: 'gqa' }, RMS, SWIGLU, RMS, { key: 'lm_head' }],
  },
  qwen3_moe: {
    label: 'Qwen3 (MoE)',
    group: 'zoo',
    meta: '235B total · 22B active · May 2025',
    blurb: 'The sparse Qwen3 variant. Remarkably similar to DeepSeek V3, but with GQA and — notably — NO shared expert (8 routed experts, none always-on). Optimised for efficient serving at scale.',
    pros: ['High capacity, efficient serving', 'Many small experts (modern trend)', 'QK-Norm stability'],
    cons: ['No shared expert (more redundancy across experts)', 'MoE harder to fine-tune than dense'],
    components: ['GQA', 'MoE (no shared expert)', 'QK-Norm', 'RoPE', 'SwiGLU'],
    config: { modelName: 'Qwen3MoE', vocabSize: 151936, dModel: 256 },
    spec: [{ key: 'embedding' }, ROPE, RMS, QK, { key: 'gqa' }, RMS, { key: 'moe', params: { experts: 128, topk: 8, shared: 'no' } }, RMS, QK, { key: 'gqa' }, RMS, { key: 'moe', params: { experts: 128, topk: 8, shared: 'no' } }, RMS, { key: 'lm_head' }],
  },
  smollm3: {
    label: 'SmolLM3',
    group: 'zoo',
    meta: '3B · Jul 2025',
    blurb: 'A strong, very open 3B model. Its signature trick is NoPE — No Positional Embeddings — applied on every 4th layer (RoPE elsewhere). NoPE shows better length generalisation in ablations.',
    pros: ['Great quality at 3B', 'NoPE improves length generalisation', 'Open training details'],
    cons: ['NoPE only safe in some layers', 'Less proven at large scale'],
    components: ['GQA', 'NoPE every 4th layer', 'RoPE', 'SwiGLU'],
    config: { modelName: 'SmolLM3', vocabSize: 128256, dModel: 256 },
    spec: [{ key: 'embedding' }, ROPE, RMS, { key: 'gqa' }, RMS, SWIGLU, { key: 'pos_none' }, RMS, { key: 'gqa' }, RMS, SWIGLU, RMS, { key: 'lm_head' }],
  },
  kimi_k2: {
    label: 'Kimi K2 / K2 Thinking',
    group: 'zoo',
    meta: '1T total · 32B active · Jul 2025',
    blurb: 'Essentially the DeepSeek V3 architecture scaled up: even more MoE experts and fewer MLA heads. At ~1 trillion parameters it is the largest open-weight model of this generation; K2 Thinking extends context to 256k.',
    pros: ['Frontier-level open weights', 'MLA + huge MoE = efficient at scale', 'Shared expert'],
    cons: ['Massive total memory', 'Trillion-param serving is non-trivial'],
    components: ['MLA (fewer heads)', 'MoE + shared expert (more experts)', 'RoPE', 'SwiGLU'],
    config: { modelName: 'KimiK2', vocabSize: 160000, dModel: 256 },
    spec: [{ key: 'embedding' }, ROPE, RMS, { key: 'mla', params: { heads: 8, latent: 64 } }, RMS, { key: 'moe', params: { experts: 384, topk: 8, shared: 'yes' } }, RMS, { key: 'mla', params: { heads: 8, latent: 64 } }, RMS, { key: 'moe', params: { experts: 384, topk: 8, shared: 'yes' } }, RMS, { key: 'lm_head' }],
  },
  gpt_oss: {
    label: 'GPT-OSS',
    group: 'zoo',
    meta: '20B / 120B · Aug 2025',
    blurb: 'OpenAI\'s first open weights since GPT-2. A wide MoE with sliding-window attention every other layer, few large experts (32, 4 active), and a revival of attention bias units plus learned attention sinks.',
    pros: ['Wide → high tokens/sec throughput', 'Sliding window saves memory', 'Attention sinks stabilise long context'],
    cons: ['Few large experts (vs many-small trend)', 'Attention bias largely redundant', 'No shared expert'],
    components: ['GQA + bias + sinks', 'Sliding-window every other layer', 'MoE (few large experts)', 'RoPE', 'wide'],
    config: { modelName: 'GPTOSS', vocabSize: 201088, dModel: 256 },
    spec: [{ key: 'embedding' }, ROPE, RMS, { key: 'swa', params: { heads: 8, window: 1024 } }, RMS, { key: 'moe', params: { experts: 32, topk: 4, shared: 'no' } }, RMS, { key: 'gqa' }, RMS, { key: 'moe', params: { experts: 32, topk: 4, shared: 'no' } }, RMS, { key: 'lm_head' }],
  },
  grok25: {
    label: 'Grok 2.5',
    group: 'zoo',
    meta: '270B · xAI · 2024 flagship',
    blurb: 'A rare look at a real production flagship. Uses a small number of large experts (older trend) and an always-on doubled-width SwiGLU module that acts like a shared expert.',
    pros: ['Production-proven design', 'Shared-expert-like always-on module'],
    cons: ['Few large experts (less efficient than many-small)', 'Last-gen design'],
    components: ['GQA', 'MoE (few large experts)', 'Shared-expert-like SwiGLU', 'RoPE'],
    config: { modelName: 'Grok25', vocabSize: 131072, dModel: 256 },
    spec: [{ key: 'embedding' }, ROPE, RMS, { key: 'gqa' }, RMS, { key: 'moe', params: { experts: 8, topk: 2, shared: 'yes' } }, RMS, { key: 'gqa' }, RMS, { key: 'moe', params: { experts: 8, topk: 2, shared: 'yes' } }, RMS, { key: 'lm_head' }],
  },
  glm45: {
    label: 'GLM-4.5',
    group: 'zoo',
    meta: '106B / 355B · Aug 2025',
    blurb: 'An instruction/reasoning hybrid tuned for agents & function-calling. Like DeepSeek V3 it puts 3 dense layers before the MoE blocks (for early stability) and keeps a shared expert, plus GPT-2-style attention bias.',
    pros: ['Strong agentic / tool-use performance', 'Dense-first layers aid convergence', 'Shared expert'],
    cons: ['Large total memory', 'Attention bias mostly redundant'],
    components: ['GQA', '3 dense layers first', 'MoE + shared expert', 'Attention bias', 'RoPE'],
    config: { modelName: 'GLM45', vocabSize: 151552, dModel: 256 },
    spec: [{ key: 'embedding' }, ROPE, RMS, { key: 'gqa' }, RMS, SWIGLU, RMS, { key: 'gqa' }, RMS, { key: 'moe', params: { experts: 128, topk: 8, shared: 'yes' } }, RMS, { key: 'lm_head' }],
  },
  qwen3_next: {
    label: 'Qwen3-Next',
    group: 'zoo',
    meta: '80B total · 3B active · Sep 2025',
    blurb: 'Pushes efficiency hard: a Gated DeltaNet (linear attention) + Gated Attention hybrid in a 3:1 ratio for cheap 262k context, far more (and smaller) experts plus a shared expert, and Multi-Token Prediction for faster decoding.',
    pros: ['Linear-attention hybrid → cheap long context', 'Many small experts + shared expert', 'MTP speeds decoding'],
    cons: ['DeltaNet less precise than full attention', 'Complex hybrid to implement'],
    components: ['Gated DeltaNet + Attention 3:1', 'MoE + shared expert (many experts)', 'MTP', 'RoPE'],
    config: { modelName: 'Qwen3Next', vocabSize: 151936, dModel: 256 },
    spec: [{ key: 'embedding' }, ROPE, RMS, { key: 'deltanet' }, RMS, { key: 'moe', params: { experts: 256, topk: 10, shared: 'yes' } }, RMS, { key: 'deltanet' }, RMS, { key: 'moe', params: { experts: 256, topk: 10, shared: 'yes' } }, RMS, { key: 'gqa' }, RMS, { key: 'moe', params: { experts: 256, topk: 10, shared: 'yes' } }, RMS, { key: 'lm_head' }],
  },
  minimax_m2: {
    label: 'MiniMax-M2',
    group: 'zoo',
    meta: '230B total · 10B active · Oct 2025',
    blurb: 'Currently among the best open-weight models. Went back to full attention (after M1\'s linear "lightning" attention) for quality, but is very sparse (only 10B active), with per-layer QK-Norm and partial RoPE.',
    pros: ['Top open-weight benchmarks', 'Very sparse → cheap inference', 'Per-layer QK-Norm + partial RoPE stability'],
    cons: ['No shared expert (more redundancy)', 'Full attention keeps O(n²) cost'],
    components: ['Full GQA', 'Per-layer QK-Norm', 'Partial RoPE', 'MoE (very sparse, no shared)'],
    config: { modelName: 'MiniMaxM2', vocabSize: 200064, dModel: 256 },
    spec: [{ key: 'embedding' }, ROPE, RMS, QK, { key: 'gqa' }, RMS, { key: 'moe', params: { experts: 256, topk: 8, shared: 'no' } }, RMS, QK, { key: 'gqa' }, RMS, { key: 'moe', params: { experts: 256, topk: 8, shared: 'no' } }, RMS, { key: 'lm_head' }],
  },
  kimi_linear: {
    label: 'Kimi Linear',
    group: 'zoo',
    meta: 'linear-attention hybrid · Oct 2025',
    blurb: 'A transformer with SSM-style components. Mixes Kimi Delta Attention (a refined Gated DeltaNet with channel-wise gating) with MLA full-attention layers in a 3:1 ratio, and applies NoPE inside the MLA (global) layers.',
    pros: ['Channel-wise gating → better long-context memory', 'MLA cache savings on global layers', 'Linear layers keep it cheap'],
    cons: ['Cutting-edge, less battle-tested', 'Complex hybrid'],
    components: ['Kimi Delta Attention (linear) 3:1', 'MLA + NoPE on global layers', 'MoE', 'RoPE on linear layers'],
    config: { modelName: 'KimiLinear', vocabSize: 160000, dModel: 256 },
    spec: [{ key: 'embedding' }, ROPE, RMS, { key: 'deltanet' }, RMS, { key: 'moe', params: { experts: 256, topk: 8, shared: 'yes' } }, RMS, { key: 'deltanet' }, RMS, { key: 'pos_none' }, { key: 'mla', params: { heads: 8, latent: 64 } }, RMS, { key: 'moe', params: { experts: 256, topk: 8, shared: 'yes' } }, RMS, { key: 'lm_head' }],
  },
}

export const FUNDAMENTAL_ORDER = ['encoder', 'decoder', 'encdec', 'mamba']
export const ZOO_ORDER = [
  'deepseek_v3', 'olmo2', 'gemma3', 'mistral31', 'llama4',
  'qwen3_dense', 'qwen3_moe', 'smollm3', 'kimi_k2', 'gpt_oss',
  'grok25', 'glm45', 'qwen3_next', 'minimax_m2', 'kimi_linear',
]
// kept for backward compatibility (the original 4-way compare)
export const ARCH_ORDER = FUNDAMENTAL_ORDER

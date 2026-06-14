// Lightweight, friendly checks shown as warnings over the 3D canvas.
export function validate(stack, cfg) {
  const w = []
  if (!stack.length) return w
  const keys = stack.map((b) => b.key)

  if (keys[0] !== 'embedding' && !keys.includes('embedding'))
    w.push('No Token Embedding — the model has no way to turn token ids into vectors.')
  else if (keys.indexOf('embedding') > 0)
    w.push('Token Embedding usually comes first (it reads the raw token ids).')

  if (!keys.includes('lm_head'))
    w.push('No output head — add an LM / Output Head to produce logits.')
  else if (keys[keys.length - 1] !== 'lm_head')
    w.push('Output head is usually the last block in the stack.')

  for (const b of stack) {
    const p = b.params || {}
    if (['mha', 'causal_mha', 'cross_attention', 'gqa', 'mla', 'swa'].includes(b.key)) {
      if (p.heads && cfg.dModel % p.heads !== 0)
        w.push(`d_model (${cfg.dModel}) is not divisible by ${p.heads} heads.`)
    }
    if (b.key === 'gqa' && p.heads % p.kv_heads !== 0)
      w.push(`GQA: Q heads (${p.heads}) must be a multiple of KV heads (${p.kv_heads}).`)
    if (b.key === 'moe' && p.topk > p.experts)
      w.push(`MoE: top-k (${p.topk}) can't exceed the number of experts (${p.experts}).`)
  }

  const hasAttention = keys.some((k) => ['mha', 'causal_mha', 'gqa', 'mla', 'swa'].includes(k))
  const hasMixer = keys.some((k) => ['mamba', 'deltanet'].includes(k))
  if (!hasAttention && !hasMixer)
    w.push('No token-mixing layer — add Attention or a Mamba block so positions can talk.')

  return w
}

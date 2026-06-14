import { BLOCK_BY_KEY } from './blocks.js'

export const BLOCK_W = 3
export const BLOCK_H = 0.6
export const BLOCK_D = 1.5
export const GAP = 0.18
export const STEP = BLOCK_H + GAP

// y-centre of the block at index i in a stack of length n (index 0 = bottom)
export function yForIndex(i, n) {
  return (i - (n - 1) / 2) * STEP
}

// short tag drawn on the right edge of a block, e.g. "8h", "×4"
export function sublabel(blk) {
  const p = blk.params || {}
  switch (blk.key) {
    case 'mha':
    case 'causal_mha':
    case 'cross_attention':
    case 'mla':
      return `${p.heads}h`
    case 'gqa':
      return `${p.heads}/${p.kv_heads}h`
    case 'swa':
      return `w${p.window}`
    case 'ffn':
      return p.act === 'swiglu' ? 'glu' : `×${p.mult}`
    case 'moe':
      return p.shared === 'yes' ? `${p.experts}e+1` : `${p.experts}e`
    case 'mamba':
      return `s${p.d_state}`
    case 'deltanet':
      return 'lin'
    case 'pos_none':
      return 'NoPE'
    case 'dropout':
      return `${p.p}`
    case 'lm_head':
      return p.mode === 'classification' ? 'cls' : 'lm'
    default:
      return ''
  }
}

export function blockLabel(blk) {
  return BLOCK_BY_KEY[blk.key]?.name || blk.key
}

// Split a stack into the input section (embedding + positional), the repeatable
// transformer "body", and the output tail (final norm + head). Used by the
// "layers ×N" feature, the 3D depth indicator and the code generator.
const HEAD_KEYS = new Set(['embedding', 'pos_sinusoidal', 'pos_learned', 'pos_rotary', 'pos_none'])
const NORM_KEYS = new Set(['rmsnorm', 'layernorm'])

export function splitStack(stack) {
  let h = 0
  while (h < stack.length && HEAD_KEYS.has(stack[h].key)) h++

  let t = stack.length
  const keys = stack.map((b) => b.key)
  const lastHead = keys.lastIndexOf('lm_head')
  if (lastHead >= 0) {
    t = lastHead
    if (t - 1 >= h && NORM_KEYS.has(stack[t - 1].key)) t -= 1
  }
  if (t < h) t = h
  return {
    head: stack.slice(0, h),
    body: stack.slice(h, t),
    tail: stack.slice(t),
    headEnd: h,
    tailStart: t,
  }
}

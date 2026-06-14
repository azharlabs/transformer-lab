// Synthetic — but qualitatively realistic — attention patterns used by the
// Attention Lab. Real trained heads specialise like this: previous-token heads,
// "attention-sink" heads that dump onto the first token, local-window heads, and
// content/semantic heads. We fabricate scores with the same shapes so the
// visualisation teaches the right intuition.

export const DEMO_TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat', '.']
// a separate "source" sentence used for the cross-attention demo
export const DEMO_SRC = ['Le', 'chat', "s'est", 'assis', '.']

export const HEADS = [
  { name: 'Previous-token', hint: 'Looks one step back — copies/relays the prior word.' },
  { name: 'Attention sink', hint: 'Dumps probability on the first token. Real models do this a lot.' },
  { name: 'Local window', hint: 'Attends to immediate neighbours — short-range syntax.' },
  { name: 'Content / semantic', hint: 'Matches words by meaning, regardless of distance.' },
]

function charVec(tok) {
  const v = [0, 0, 0]
  for (let i = 0; i < tok.length; i++) {
    const c = tok.toLowerCase().charCodeAt(i)
    v[i % 3] += (c % 13) / 13
  }
  return v
}

function selfScore(i, j, head, toks) {
  switch (head) {
    case 0:
      return j === i - 1 ? 3.0 : j === i ? 1.0 : 0.1
    case 1:
      return j === 0 ? 3.2 : j === i ? 1.2 : 0.1
    case 2:
      return Math.max(0.05, 2.2 - 1.4 * Math.abs(i - j))
    case 3: {
      const a = charVec(toks[i])
      const b = charVec(toks[j])
      return 0.2 + 2.4 * (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
    }
    default:
      return 0.1
  }
}

function crossScore(i, j, rowToks, colToks) {
  const a = charVec(rowToks[i])
  const b = charVec(colToks[j])
  // soft alignment: also reward similar relative position
  const posBias = 1.2 - Math.abs(i / rowToks.length - j / colToks.length) * 2
  return 0.3 + 2.0 * (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) + Math.max(0, posBias)
}

function softmax(arr) {
  const max = Math.max(...arr.filter((x) => isFinite(x)))
  const exps = arr.map((x) => (isFinite(x) ? Math.exp(x - max) : 0))
  const sum = exps.reduce((a, b) => a + b, 0) || 1
  return exps.map((e) => e / sum)
}

// Raw (un-masked, un-softmaxed) scores, normalised to 0..1 — used by the stepped
// walkthrough to show the "before softmax / before mask" state.
export function rawMatrix(rowToks, colToks, head, opts = {}) {
  const { cross = false } = opts
  const n = rowToks.length
  const m = colToks.length
  let max = 1e-6
  const M = []
  for (let i = 0; i < n; i++) {
    const row = []
    for (let j = 0; j < m; j++) {
      const s = cross ? crossScore(i, j, rowToks, colToks) : selfScore(i, j, head, colToks)
      row.push(s)
      if (s > max) max = s
    }
    M.push(row)
  }
  return M.map((r) => r.map((v) => v / max))
}

// General attention matrix. rowToks = queries, colToks = keys.
//   causal  : token i can't attend to j > i (self-attention only)
//   window  : >0 means local — i can't attend further back than `window` tokens
//   cross   : use content alignment between two different token sets
export function attentionMatrix(rowToks, colToks, head, opts = {}) {
  const { causal = false, window = 0, cross = false } = opts
  const n = rowToks.length
  const m = colToks.length
  const M = []
  for (let i = 0; i < n; i++) {
    const row = []
    for (let j = 0; j < m; j++) {
      let masked = false
      if (causal && j > i) masked = true
      if (window > 0 && i - j >= window) masked = true
      const s = cross ? crossScore(i, j, rowToks, colToks) : selfScore(i, j, head, colToks)
      row.push(masked ? -Infinity : s)
    }
    M.push(softmax(row))
  }
  return M
}

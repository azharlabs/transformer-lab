import { useMemo, useState } from 'react'
import { useLab } from '../store.js'
import { generatePyTorch } from '../lib/codegen.js'
import { splitStack } from '../lib/layout.js'

const KEYWORDS = new Set([
  'import', 'from', 'as', 'class', 'def', 'return', 'for', 'in', 'if', 'else',
  'elif', 'with', 'assert', 'self', 'super', 'True', 'False', 'None', 'and',
  'or', 'not', 'is', 'lambda', 'while', 'break', 'continue', 'pass', 'yield',
])

function highlight(code) {
  const RE = /(#.*$)|("""[\s\S]*?"""|'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")|\b(\d+\.?\d*)\b|([A-Za-z_]\w*)/gm
  const out = []
  let last = 0
  let m
  let k = 0
  while ((m = RE.exec(code)) !== null) {
    if (m.index > last) out.push(<span key={k++}>{code.slice(last, m.index)}</span>)
    if (m[1]) out.push(<span key={k++} className="tok-com">{m[1]}</span>)
    else if (m[2]) out.push(<span key={k++} className="tok-str">{m[2]}</span>)
    else if (m[3]) out.push(<span key={k++} className="tok-num">{m[3]}</span>)
    else if (m[4]) {
      if (KEYWORDS.has(m[4])) out.push(<span key={k++} className="tok-kw">{m[4]}</span>)
      else out.push(<span key={k++}>{m[4]}</span>)
    }
    last = RE.lastIndex
  }
  if (last < code.length) out.push(<span key={k++}>{code.slice(last)}</span>)
  return out
}

function sumParams(blocks, cfg) {
  const d = cfg.dModel
  let total = 0
  for (const b of blocks) {
    const p = b.params || {}
    switch (b.key) {
      case 'embedding': total += cfg.vocabSize * d; break
      case 'pos_learned': total += cfg.maxSeqLen * d; break
      case 'mha': case 'causal_mha': case 'swa': total += 4 * d * d; break
      case 'cross_attention': total += 4 * d * d; break
      case 'gqa': {
        const dh = d / p.heads
        total += d * d + 2 * (p.kv_heads * dh) * d + d * d
        break
      }
      case 'mla': total += d * d + d * (p.latent || 64) + 2 * (p.latent || 64) * d + d * d; break
      case 'ffn': total += (p.act === 'swiglu' ? 3 : 2) * (p.mult || 4) * d * d; break
      case 'moe': {
        const e = (p.experts || 8) + (p.shared === 'yes' ? 1 : 0)
        total += e * 2 * 4 * d * d + d * (p.experts || 8)
        break
      }
      case 'mamba': total += 3 * (p.expand || 2) * d * d; break
      case 'deltanet': total += (3 * (p.expand || 2) + (p.expand || 2)) * d * d; break
      case 'layernorm': total += 2 * d; break
      case 'rmsnorm': total += d; break
      case 'lm_head': total += d * (p.mode === 'classification' ? cfg.numClasses : cfg.vocabSize); break
      default: break
    }
  }
  return total
}

function estimateParams(stack, cfg) {
  const { head, body, tail } = splitStack(stack)
  const N = Math.max(1, parseInt(cfg.layers, 10) || 1)
  return sumParams(head, cfg) + N * sumParams(body, cfg) + sumParams(tail, cfg)
}

export default function CodePanel() {
  const stack = useLab((s) => s.stack)
  const config = useLab((s) => s.config)
  const [copied, setCopied] = useState(false)

  const code = useMemo(() => generatePyTorch(stack, config), [stack, config])
  const params = useMemo(() => estimateParams(stack, config), [stack, config])

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1400)
    } catch {
      setCopied(false)
    }
  }

  const download = () => {
    const blob = new Blob([code], { type: 'text/x-python' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${config.modelName || 'model'}.py`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="code-view">
      <div className="code-bar">
        <button className="btn primary" onClick={copy}>{copied ? '✓ Copied' : 'Copy'}</button>
        <button className="btn" onClick={download}>Download .py</button>
        <div className="spacer" style={{ flex: 1 }} />
        <div className="stat">{stack.length} blocks × {Math.max(1, config.layers || 1)} layers</div>
        <div className="stat">~{(params / 1e6).toFixed(1)}M params (est.)</div>
        <div className="stat">d_model {config.dModel}</div>
      </div>
      <div className="code-scroll">
        <pre className="code">{highlight(code)}</pre>
      </div>
    </div>
  )
}

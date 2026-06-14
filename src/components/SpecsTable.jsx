import { useMemo, useState } from 'react'
import { ARCHITECTURES, ZOO_ORDER, FUNDAMENTAL_ORDER } from '../lib/architectures.js'
import { useLab } from '../store.js'

function has(spec, key) {
  return spec.some((b) => b.key === key)
}
function moeShared(spec) {
  const m = spec.find((b) => b.key === 'moe')
  if (!m) return null
  return (m.params && m.params.shared) || 'yes'
}

function attentionLabel(spec) {
  const parts = []
  if (has(spec, 'deltanet')) parts.push('Gated DeltaNet')
  if (has(spec, 'mla')) parts.push('MLA')
  else if (has(spec, 'swa') && has(spec, 'gqa')) parts.push('GQA + sliding-window')
  else if (has(spec, 'swa')) parts.push('Sliding-window')
  else if (has(spec, 'gqa')) parts.push('GQA')
  else if (has(spec, 'mha') || has(spec, 'causal_mha')) parts.push('MHA')
  else if (has(spec, 'cross_attention')) parts.push('Cross-attn')
  return parts.join(' + ') || '—'
}
function mixingLabel(spec) {
  if (has(spec, 'moe')) return 'MoE' + (moeShared(spec) === 'yes' ? ' + shared' : '')
  if (has(spec, 'mamba')) return 'Mamba SSM'
  if (has(spec, 'deltanet')) return 'linear'
  return 'Dense FFN'
}
function posLabel(spec) {
  const rope = has(spec, 'pos_rotary')
  const nope = has(spec, 'pos_none')
  if (rope && nope) return 'RoPE + NoPE'
  if (rope) return 'RoPE'
  if (nope) return 'NoPE'
  if (has(spec, 'pos_learned')) return 'Learned'
  if (has(spec, 'pos_sinusoidal')) return 'Sinusoidal'
  return '—'
}
function normLabel(spec) {
  const base = has(spec, 'rmsnorm') ? 'RMSNorm' : has(spec, 'layernorm') ? 'LayerNorm' : '—'
  return base + (has(spec, 'qk_norm') ? ' + QK-Norm' : '')
}

function deriveRow(id) {
  const a = ARCHITECTURES[id]
  return {
    id,
    name: a.label,
    sub: a.meta || a.examples || '',
    attention: attentionLabel(a.spec),
    mixing: mixingLabel(a.spec),
    positional: posLabel(a.spec),
    norm: normLabel(a.spec),
  }
}

const COLS = [
  { key: 'name', label: 'Model' },
  { key: 'attention', label: 'Attention' },
  { key: 'mixing', label: 'Token mixing / FFN' },
  { key: 'positional', label: 'Positional' },
  { key: 'norm', label: 'Normalisation' },
]

export default function SpecsTable() {
  const loadPreset = useLab((s) => s.loadPreset)
  const setView = useLab((s) => s.setView)
  const [sortKey, setSortKey] = useState(null)
  const [dir, setDir] = useState(1)

  const rows = useMemo(() => {
    const base = [...ZOO_ORDER, ...FUNDAMENTAL_ORDER].map(deriveRow)
    if (!sortKey) return base
    return [...base].sort((x, y) => x[sortKey].localeCompare(y[sortKey]) * dir)
  }, [sortKey, dir])

  const sortBy = (key) => {
    if (key === sortKey) setDir((d) => -d)
    else {
      setSortKey(key)
      setDir(1)
    }
  }

  return (
    <div className="code-view">
      <div className="specs-intro">
        Every model and fundamental, side by side. Click a column header to sort. Component choices summarised from
        Sebastian Raschka's <b>"The Big LLM Architecture Comparison."</b>
      </div>
      <div className="specs-wrap">
        <table className="specs-table">
          <thead>
            <tr>
              {COLS.map((c) => (
                <th key={c.key} onClick={() => sortBy(c.key)}>
                  {c.label}
                  {sortKey === c.key && <span className="arrow">{dir > 0 ? '▲' : '▼'}</span>}
                </th>
              ))}
              <th></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td className="mname">
                  {r.name}
                  <small>{r.sub}</small>
                </td>
                <td>{r.attention}</td>
                <td>{r.mixing}</td>
                <td>{r.positional}</td>
                <td>{r.norm}</td>
                <td>
                  <span
                    className="open-link"
                    onClick={() => {
                      loadPreset(r.id)
                      setView('builder')
                    }}
                  >
                    Open →
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

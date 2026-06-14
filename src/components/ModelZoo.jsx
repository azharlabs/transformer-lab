import { useState } from 'react'
import MiniTower from './MiniTower.jsx'
import { useLab } from '../store.js'
import { ARCHITECTURES, ZOO_ORDER, FUNDAMENTAL_ORDER } from '../lib/architectures.js'

function Tag({ children }) {
  return <span className="ztag">{children}</span>
}

export default function ModelZoo() {
  const [sel, setSel] = useState('deepseek_v3')
  const loadPreset = useLab((s) => s.loadPreset)
  const setView = useLab((s) => s.setView)
  const a = ARCHITECTURES[sel]

  const Row = ({ id }) => {
    const m = ARCHITECTURES[id]
    return (
      <div className={'zrow' + (id === sel ? ' active' : '')} onClick={() => setSel(id)}>
        <div className="zrow-t">{m.label}</div>
        {m.meta && <div className="zrow-m">{m.meta}</div>}
        <div className="zrow-tags">
          {(m.components || []).slice(0, 3).map((c) => (
            <Tag key={c}>{c}</Tag>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="workspace">
      <div className="side left scrolly" style={{ width: 290 }}>
        <h3>Real LLMs (2024–2026)</h3>
        {ZOO_ORDER.map((id) => (
          <Row key={id} id={id} />
        ))}
        <h3>Fundamentals</h3>
        {FUNDAMENTAL_ORDER.map((id) => (
          <Row key={id} id={id} />
        ))}
        <div style={{ height: 16 }} />
      </div>

      <div className="canvas-wrap">
        <MiniTower spec={a.spec} key={sel} />
        <div className="overlay tl">
          <div className="legend" style={{ maxWidth: 320 }}>
            <div style={{ fontWeight: 700, fontSize: 15 }}>{a.label}</div>
            {a.meta && <div style={{ color: 'var(--accent)', fontSize: 12, marginTop: 2 }}>{a.meta}</div>}
          </div>
        </div>
        <div className="overlay bl" style={{ pointerEvents: 'auto' }}>
          <button
            className="btn primary"
            onClick={() => {
              loadPreset(sel)
              setView('builder')
            }}
          >
            Open in Builder →
          </button>
        </div>
      </div>

      <div className="side scrolly" style={{ width: 360 }}>
        <h3>About</h3>
        <div className="zabout">{a.blurb}</div>

        <h3>Key components</h3>
        <div style={{ margin: '0 14px', display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {(a.components || []).map((c) => (
            <Tag key={c}>{c}</Tag>
          ))}
        </div>

        <h3>Advantages</h3>
        <ul className="zlist pros">
          {(a.pros || []).map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>

        <h3>Disadvantages</h3>
        <ul className="zlist cons">
          {(a.cons || []).map((c, i) => (
            <li key={i}>{c}</li>
          ))}
        </ul>
        <div className="zsrc">
          Components &amp; trade-offs summarised from Sebastian Raschka, “The Big LLM Architecture Comparison.”
        </div>
        <div style={{ height: 16 }} />
      </div>
    </div>
  )
}

import { useState } from 'react'
import MiniTower from './MiniTower.jsx'
import MobileSheet from './MobileSheet.jsx'
import { useIsMobile } from '../lib/useIsMobile.js'
import { useLab } from '../store.js'
import { ARCHITECTURES, ZOO_ORDER, FUNDAMENTAL_ORDER } from '../lib/architectures.js'

function Tag({ children }) {
  return <span className="ztag">{children}</span>
}

export default function ModelZoo() {
  const [sel, setSel] = useState('deepseek_v3')
  const [sheet, setSheet] = useState(null)
  const loadPreset = useLab((s) => s.loadPreset)
  const setView = useLab((s) => s.setView)
  const isMobile = useIsMobile()
  const a = ARCHITECTURES[sel]

  const pick = (id) => {
    setSel(id)
    if (isMobile) setSheet(null)
  }

  const Row = ({ id }) => {
    const m = ARCHITECTURES[id]
    return (
      <div className={'zrow' + (id === sel ? ' active' : '')} onClick={() => pick(id)}>
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

  const list = (
    <>
      <h3>Real LLMs (2024–2026)</h3>
      {ZOO_ORDER.map((id) => (
        <Row key={id} id={id} />
      ))}
      <h3>Fundamentals</h3>
      {FUNDAMENTAL_ORDER.map((id) => (
        <Row key={id} id={id} />
      ))}
      <div style={{ height: 16 }} />
    </>
  )

  const details = (
    <>
      <h3>About</h3>
      <div className="zabout">{a.blurb}</div>
      <h3>Key components</h3>
      <div style={{ margin: '0 14px', display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {(a.components || []).map((c) => (
          <Tag key={c}>{c}</Tag>
        ))}
      </div>
      <h3>Advantages</h3>
      <ul className="zlist pros">{(a.pros || []).map((p, i) => <li key={i}>{p}</li>)}</ul>
      <h3>Disadvantages</h3>
      <ul className="zlist cons">{(a.cons || []).map((c, i) => <li key={i}>{c}</li>)}</ul>
      <div className="zsrc">Components &amp; trade-offs summarised from Sebastian Raschka, “The Big LLM Architecture Comparison.”</div>
      <div style={{ height: 16 }} />
    </>
  )

  const canvasArea = (
    <div className="canvas-wrap">
      <MiniTower spec={a.spec} key={sel} />
      <div className="overlay tl">
        <div className="legend" style={{ maxWidth: 320 }}>
          <div style={{ fontWeight: 700, fontSize: 15 }}>{a.label}</div>
          {a.meta && <div style={{ color: 'var(--accent)', fontSize: 12, marginTop: 2 }}>{a.meta}</div>}
        </div>
      </div>
      {!isMobile && (
        <div className="overlay bl" style={{ pointerEvents: 'auto' }}>
          <button className="btn primary" onClick={() => { loadPreset(sel); setView('builder') }}>Open in Builder →</button>
        </div>
      )}
    </div>
  )

  if (isMobile) {
    return (
      <div className="m-view">
        <div className="m-canvas">{canvasArea}</div>
        <div className="m-tabbar">
          <button onClick={() => setSheet('models')}>🦁 Models</button>
          <button onClick={() => setSheet('about')}>ℹ︎ Details</button>
          <button className="primary" onClick={() => { loadPreset(sel); setView('builder') }}>Open →</button>
        </div>
        {sheet === 'models' && (
          <MobileSheet title="Choose a model" onClose={() => setSheet(null)}>
            <div style={{ padding: '4px 0' }}>{list}</div>
          </MobileSheet>
        )}
        {sheet === 'about' && (
          <MobileSheet title={a.label} onClose={() => setSheet(null)}>
            <div style={{ padding: '4px 0' }}>{details}</div>
          </MobileSheet>
        )}
      </div>
    )
  }

  return (
    <div className="workspace">
      <div className="side left scrolly" style={{ width: 290 }}>{list}</div>
      {canvasArea}
      <div className="side scrolly" style={{ width: 360 }}>{details}</div>
    </div>
  )
}

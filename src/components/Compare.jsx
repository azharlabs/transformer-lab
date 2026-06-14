import { useState } from 'react'
import MiniTower from './MiniTower.jsx'
import { ARCHITECTURES, ZOO_ORDER, FUNDAMENTAL_ORDER } from '../lib/architectures.js'
import { useLab } from '../store.js'

const ALL = [...FUNDAMENTAL_ORDER, ...ZOO_ORDER]

export default function Compare() {
  const loadPreset = useLab((s) => s.loadPreset)
  const setView = useLab((s) => s.setView)
  const [cols, setCols] = useState(['deepseek_v3', 'llama4', 'qwen3_moe', 'gemma3'])

  const setCol = (i, id) => setCols((c) => c.map((v, j) => (j === i ? id : v)))

  return (
    <div className="compare-grid">
      {cols.map((id, i) => {
        const a = ARCHITECTURES[id]
        return (
          <div className="compare-col" key={i}>
            <div className="head">
              <select className="cmp-select" value={id} onChange={(e) => setCol(i, e.target.value)}>
                <optgroup label="Fundamentals">
                  {FUNDAMENTAL_ORDER.map((k) => (
                    <option key={k} value={k}>{ARCHITECTURES[k].label}</option>
                  ))}
                </optgroup>
                <optgroup label="Real LLMs">
                  {ZOO_ORDER.map((k) => (
                    <option key={k} value={k}>{ARCHITECTURES[k].label}</option>
                  ))}
                </optgroup>
              </select>
              {a.meta && <div className="ex">{a.meta}</div>}
              {!a.meta && a.examples && <div className="ex">{a.examples}</div>}
            </div>

            <div className="cv">
              <MiniTower spec={a.spec} key={id} />
              <div className="overlay bl" style={{ pointerEvents: 'auto' }}>
                <button
                  className="btn primary"
                  onClick={() => {
                    loadPreset(id)
                    setView('builder')
                  }}
                >
                  Open →
                </button>
              </div>
            </div>

            <div className="cmp-body scrolly">
              <div className="cmp-tags">
                {(a.components || []).map((c) => (
                  <span className="ztag" key={c}>{c}</span>
                ))}
              </div>
              <div className="cmp-pc">
                <div className="cmp-h pros">Advantages</div>
                <ul className="zlist pros">{(a.pros || []).map((p, k) => <li key={k}>{p}</li>)}</ul>
                <div className="cmp-h cons">Disadvantages</div>
                <ul className="zlist cons">{(a.cons || []).map((c, k) => <li key={k}>{c}</li>)}</ul>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

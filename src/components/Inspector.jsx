import { useRef } from 'react'
import { useLab } from '../store.js'
import { BLOCK_BY_KEY } from '../lib/blocks.js'
import { ARCHITECTURES, FUNDAMENTAL_ORDER, ZOO_ORDER } from '../lib/architectures.js'

function ConfigField({ label, k, type = 'int' }) {
  const value = useLab((s) => s.config[k])
  const setConfig = useLab((s) => s.setConfig)
  return (
    <div className="field">
      <label>{label}</label>
      <input
        type={type === 'text' ? 'text' : 'number'}
        value={value}
        onChange={(e) =>
          setConfig({ [k]: type === 'text' ? e.target.value : Number(e.target.value) })
        }
      />
    </div>
  )
}

function ParamField({ uid, name, spec, value }) {
  const setBlockParam = useLab((s) => s.setBlockParam)
  if (spec.type === 'enum') {
    return (
      <div className="field">
        <label>{spec.label}</label>
        <select value={value} onChange={(e) => setBlockParam(uid, name, e.target.value)}>
          {spec.options.map((o) => (
            <option key={o} value={o}>{o}</option>
          ))}
        </select>
      </div>
    )
  }
  return (
    <div className="field">
      <label>{spec.label}</label>
      <input
        type="number"
        value={value}
        min={spec.min}
        max={spec.max}
        step={spec.step || 1}
        onChange={(e) =>
          setBlockParam(uid, name, spec.type === 'float' ? Number(e.target.value) : parseInt(e.target.value, 10))
        }
      />
    </div>
  )
}

export default function Inspector() {
  const config = useLab((s) => s.config)
  const stack = useLab((s) => s.stack)
  const selectedUid = useLab((s) => s.selectedUid)
  const removeBlock = useLab((s) => s.removeBlock)
  const clearStack = useLab((s) => s.clearStack)
  const loadPreset = useLab((s) => s.loadPreset)
  const setView = useLab((s) => s.setView)
  const exportJSON = useLab((s) => s.exportJSON)
  const importJSON = useLab((s) => s.importJSON)
  const resetDefault = useLab((s) => s.resetDefault)
  const setConfig = useLab((s) => s.setConfig)
  const setConfigLayers = (n) => setConfig({ layers: n })
  const fileRef = useRef()

  const sel = stack.find((b) => b.uid === selectedUid)
  const def = sel ? BLOCK_BY_KEY[sel.key] : null

  const doExport = () => {
    const blob = new Blob([exportJSON()], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${config.modelName || 'build'}.tlab.json`
    a.click()
    URL.revokeObjectURL(url)
  }
  const doImport = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      if (!importJSON(String(reader.result))) alert('Could not read that file.')
    }
    reader.readAsText(file)
    e.target.value = ''
  }

  return (
    <div className="side scrolly">
      <h3>Load a preset</h3>
      <select
        className="field"
        style={{ width: 'calc(100% - 28px)', margin: '0 14px' }}
        defaultValue=""
        onChange={(e) => e.target.value && loadPreset(e.target.value)}
      >
        <option value="">Choose an architecture…</option>
        <optgroup label="Fundamentals">
          {FUNDAMENTAL_ORDER.map((id) => (
            <option key={id} value={id}>{ARCHITECTURES[id].label}</option>
          ))}
        </optgroup>
        <optgroup label="Real LLMs (Model Zoo)">
          {ZOO_ORDER.map((id) => (
            <option key={id} value={id}>{ARCHITECTURES[id].label}</option>
          ))}
        </optgroup>
      </select>

      <h3>Model config</h3>
      <ConfigField label="Model name" k="modelName" type="text" />
      <div className="row2">
        <ConfigField label="d_model" k="dModel" />
        <ConfigField label="Vocab size" k="vocabSize" />
      </div>
      <div className="row2">
        <ConfigField label="Max seq len" k="maxSeqLen" />
        <ConfigField label="# Classes" k="numClasses" />
      </div>
      <div className="field">
        <label>Transformer layers ×N (repeat the body)</label>
        <input
          type="number"
          min={1}
          max={120}
          value={config.layers || 1}
          onChange={(e) => setConfigLayers(Math.max(1, Math.min(120, parseInt(e.target.value, 10) || 1)))}
        />
      </div>

      <h3>Selected part</h3>
      {!sel && <div className="block-card"><div className="body">Click a block in the 3D view to edit it. Drag blocks up/down to reorder.</div></div>}
      {sel && (
        <>
          <div className="block-card">
            <div className="title">{def.name}</div>
            <div className="body">{def.desc}</div>
          </div>
          {Object.entries(def.params || {}).map(([name, spec]) => (
            <ParamField key={name} uid={sel.uid} name={name} spec={spec} value={sel.params[name]} />
          ))}
          <div className="btn-row">
            <button className="btn danger" onClick={() => removeBlock(sel.uid)}>Remove block</button>
          </div>
        </>
      )}

      <h3>Actions</h3>
      <div className="btn-row">
        <button className="btn primary" onClick={() => setView('code')}>View PyTorch →</button>
        <button className="btn danger" onClick={clearStack}>Clear all</button>
      </div>

      <h3>Save &amp; share</h3>
      <div className="hint" style={{ margin: '0 14px 6px', fontSize: 11 }}>Your build autosaves in this browser.</div>
      <div className="btn-row">
        <button className="btn" onClick={doExport}>⬇ Export</button>
        <button className="btn" onClick={() => fileRef.current?.click()}>⬆ Import</button>
        <button className="btn danger" onClick={resetDefault}>Reset</button>
      </div>
      <input ref={fileRef} type="file" accept="application/json,.json" style={{ display: 'none' }} onChange={doImport} />
      <div style={{ height: 20 }} />
    </div>
  )
}

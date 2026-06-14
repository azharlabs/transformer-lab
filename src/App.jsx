import { useLab } from './store.js'
import Builder from './components/Builder.jsx'
import AttentionLab from './components/AttentionLab.jsx'
import Compare from './components/Compare.jsx'
import ModelZoo from './components/ModelZoo.jsx'
import SpecsTable from './components/SpecsTable.jsx'
import CodePanel from './components/CodePanel.jsx'

const TABS = [
  { id: 'builder', label: '🧱 Builder' },
  { id: 'zoo', label: '🦁 Model Zoo' },
  { id: 'attention', label: '🔍 Attention Lab' },
  { id: 'compare', label: '⚖️ Compare' },
  { id: 'specs', label: '📊 Specs' },
  { id: 'code', label: '⌨️ Code' },
]

export default function App() {
  const view = useLab((s) => s.view)
  const setView = useLab((s) => s.setView)

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">
          Transformer<span className="dot">·</span>Lab
        </div>
        <div className="tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={'tab' + (view === t.id ? ' active' : '')}
              onClick={() => setView(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="spacer" />
        <div className="hint">Build a transformer like you'd build a PC — drag parts, see them run.</div>
      </div>

      {view === 'builder' && <Builder />}
      {view === 'zoo' && <ModelZoo />}
      {view === 'attention' && <AttentionLab />}
      {view === 'compare' && <Compare />}
      {view === 'specs' && <SpecsTable />}
      {view === 'code' && <CodePanel />}
    </div>
  )
}

import { BLOCKS, CATEGORY_COLORS } from '../lib/blocks.js'
import { useLab } from '../store.js'

const CATS = [
  ['io', 'Input / Output'],
  ['position', 'Positional'],
  ['attention', 'Attention'],
  ['mixing', 'Token Mixing'],
  ['norm', 'Normalisation'],
  ['reg', 'Regularisation'],
]

export default function Palette() {
  const addBlock = useLab((s) => s.addBlock)

  return (
    <div className="side left scrolly">
      <h3>Parts bin · click to add</h3>
      {CATS.map(([cat, label]) => (
        <div key={cat}>
          <div className="cat-label">{label}</div>
          {BLOCKS.filter((b) => b.category === cat).map((b) => (
            <div
              key={b.key}
              className="palette-item"
              title={b.desc}
              onClick={() => addBlock(b.key)}
            >
              <span className="swatch" style={{ background: CATEGORY_COLORS[cat] }} />
              <div>
                <div className="nm">{b.name}</div>
                <div className="sh">{b.short}</div>
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}

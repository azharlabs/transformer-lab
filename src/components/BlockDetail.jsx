import { Canvas } from '@react-three/fiber'
import { Environment } from '@react-three/drei'
import BlockAnim from './BlockAnim.jsx'
import { useLab } from '../store.js'
import { BLOCK_BY_KEY } from '../lib/blocks.js'

function ParamRow({ uid, name, spec, value }) {
  const setBlockParam = useLab((s) => s.setBlockParam)
  return (
    <div className="field" style={{ margin: '8px 0' }}>
      <label>{spec.label}</label>
      {spec.type === 'enum' ? (
        <select value={value} onChange={(e) => setBlockParam(uid, name, e.target.value)}>
          {spec.options.map((o) => (
            <option key={o} value={o}>{o}</option>
          ))}
        </select>
      ) : (
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
      )}
    </div>
  )
}

export default function BlockDetail() {
  const stack = useLab((s) => s.stack)
  const selectedUid = useLab((s) => s.selectedUid)
  const select = useLab((s) => s.select)
  const removeBlock = useLab((s) => s.removeBlock)

  const blk = stack.find((b) => b.uid === selectedUid)
  if (!blk) return null
  const def = BLOCK_BY_KEY[blk.key]
  const hasParams = Object.keys(def.params || {}).length > 0

  return (
    <div className="detail-card">
      <div className="detail-head">
        <div className="detail-title">{def.name}</div>
        <button className="x" onClick={() => select(null)} title="Close">×</button>
      </div>

      <div className="detail-canvas">
        <Canvas camera={{ position: [0, 0, 6.4], fov: 44 }} dpr={[1, 2]}>
          <color attach="background" args={['#0b101c']} />
          <ambientLight intensity={0.7} />
          <directionalLight position={[3, 4, 6]} intensity={1.1} />
          <pointLight position={[-4, -2, 4]} intensity={0.5} color="#a78bfa" />
          <Environment preset="city" />
          <BlockAnim blockKey={blk.key} params={blk.params} />
        </Canvas>
        <div className="detail-anim-tag">live 3D · animated</div>
      </div>

      <div className="detail-body">{def.desc}</div>

      {(def.pros || def.cons) && (
        <div className="detail-pc">
          {def.pros && (
            <ul className="zlist pros">{def.pros.map((p, i) => <li key={i}>{p}</li>)}</ul>
          )}
          {def.cons && (
            <ul className="zlist cons">{def.cons.map((c, i) => <li key={i}>{c}</li>)}</ul>
          )}
          {def.usedIn && <div className="detail-used">Used in: {def.usedIn}</div>}
        </div>
      )}

      {hasParams && (
        <>
          <div className="detail-section">Adjust</div>
          {Object.entries(def.params).map(([name, spec]) => (
            <ParamRow key={name} uid={blk.uid} name={name} spec={spec} value={blk.params[name]} />
          ))}
        </>
      )}

      <div className="detail-actions">
        <button className="btn danger" onClick={() => removeBlock(blk.uid)}>Remove block</button>
        <button className="btn" onClick={() => select(null)}>Done</button>
      </div>
    </div>
  )
}

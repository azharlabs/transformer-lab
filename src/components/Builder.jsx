import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { Canvas, useThree, useFrame } from '@react-three/fiber'
import { OrbitControls, Grid, Environment, Text } from '@react-three/drei'
import Block3D from './Block3D.jsx'
import Palette from './Palette.jsx'
import Inspector from './Inspector.jsx'
import BlockDetail from './BlockDetail.jsx'
import MobileSheet from './MobileSheet.jsx'
import { useIsMobile } from '../lib/useIsMobile.js'
import { useLab } from '../store.js'
import { blockColor, blockBadge, blockCategory, CATEGORY_COLORS } from '../lib/blocks.js'
import { BLOCK_W, BLOCK_H, BLOCK_D, STEP, yForIndex, sublabel, blockLabel, splitStack } from '../lib/layout.js'
import { validate } from '../lib/validate.js'

const FLOW_PHRASE = {
  io: 'reading / projecting tokens',
  position: 'adding positional information',
  attention: 'tokens exchange information',
  mixing: 'per-token transformation',
  norm: 'normalising activations',
  reg: 'regularising',
}

// translucent stacked copies of the transformer body to convey "× N layers" depth
function GhostDepth({ stack, n, layers }) {
  const { headEnd, tailStart } = useMemo(() => splitStack(stack), [stack])
  if (layers <= 1 || tailStart <= headEnd) return null
  const ghosts = Math.min(layers - 1, 3)
  const body = []
  for (let g = 1; g <= ghosts; g++) {
    for (let i = headEnd; i < tailStart; i++) {
      body.push(
        <mesh key={`${g}-${i}`} position={[0, yForIndex(i, n), -0.75 * g]}>
          <boxGeometry args={[BLOCK_W * 0.96, BLOCK_H, BLOCK_D * 0.96]} />
          <meshStandardMaterial color={blockColor(stack[i].key)} transparent opacity={0.16 - g * 0.03} metalness={0.3} roughness={0.6} />
        </mesh>
      )
    }
  }
  const midY = (yForIndex(headEnd, n) + yForIndex(tailStart - 1, n)) / 2
  return (
    <group>
      {body}
      <Text position={[BLOCK_W / 2 + 0.7, midY, 0]} rotation={[0, 0, -Math.PI / 2]} fontSize={0.32} color="#38bdf8" anchorX="center" outlineWidth={0.006} outlineColor="#000">
        × {layers} layers
      </Text>
    </group>
  )
}

// glowing spine + data-flow dots rising through the tower
function Spine({ n }) {
  const dots = useRef([])
  const group = useRef()
  const height = Math.max(n * STEP, 1)
  const bottom = yForIndex(0, n) - BLOCK_H
  useFrame((_, dt) => {
    dots.current.forEach((m, i) => {
      if (!m) return
      m.position.y += dt * 1.6
      const top = yForIndex(n - 1, n) + BLOCK_H
      if (m.position.y > top) m.position.y = bottom
    })
  })
  return (
    <group ref={group}>
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[0.045, 0.045, height + BLOCK_H * 2, 12]} />
        <meshStandardMaterial color="#38bdf8" emissive="#38bdf8" emissiveIntensity={0.6} transparent opacity={0.35} />
      </mesh>
      {Array.from({ length: 5 }).map((_, i) => (
        <mesh
          key={i}
          ref={(el) => (dots.current[i] = el)}
          position={[0, bottom + (i / 5) * (height + BLOCK_H * 2), 0]}
        >
          <sphereGeometry args={[0.07, 12, 12]} />
          <meshStandardMaterial color="#bae6fd" emissive="#38bdf8" emissiveIntensity={1.4} />
        </mesh>
      ))}
    </group>
  )
}

function Scene({ running, onActive }) {
  const stack = useLab((s) => s.stack)
  const selectedUid = useLab((s) => s.selectedUid)
  const select = useLab((s) => s.select)
  const layers = useLab((s) => s.config.layers) || 1

  const { camera, gl } = useThree()
  const controls = useRef()
  const planeRef = useRef(new THREE.Plane())
  const grabDy = useRef(0)
  const curIndex = useRef(-1)
  const [dragUid, setDragUid] = useState(null)
  const ray = useMemo(() => new THREE.Raycaster(), [])

  const n = stack.length

  // forward-pass animation: a bright pulse ascends; the nearest block lights up
  const pulseRef = useRef()
  const phase = useRef(0)
  const flowRef = useRef(-1)
  const [flowIdx, setFlowIdx] = useState(-1)
  useFrame((_, dt) => {
    if (!running || n === 0) {
      if (flowRef.current !== -1) {
        flowRef.current = -1
        setFlowIdx(-1)
      }
      return
    }
    phase.current += dt / (Math.max(n, 1) * 0.45)
    const f = phase.current % 1
    const yb = yForIndex(0, n) - BLOCK_H / 2
    const yt = yForIndex(n - 1, n) + BLOCK_H / 2
    const y = yb + f * (yt - yb)
    if (pulseRef.current) pulseRef.current.position.set(0, y, BLOCK_D / 2 + 0.35)
    const idx = Math.max(0, Math.min(n - 1, Math.round((y - yForIndex(0, n)) / STEP)))
    if (idx !== flowRef.current) {
      flowRef.current = idx
      setFlowIdx(idx)
      onActive && onActive(idx)
    }
  })

  function intersect(ev) {
    const rect = gl.domElement.getBoundingClientRect()
    const nx = ((ev.clientX - rect.left) / rect.width) * 2 - 1
    const ny = -((ev.clientY - rect.top) / rect.height) * 2 + 1
    ray.setFromCamera({ x: nx, y: ny }, camera)
    const hit = new THREE.Vector3()
    return ray.ray.intersectPlane(planeRef.current, hit) ? hit : null
  }

  function beginDrag(uid, index, e) {
    select(uid)
    const len = useLab.getState().stack.length
    const y = yForIndex(index, len)
    const dir = camera.getWorldDirection(new THREE.Vector3())
    planeRef.current.setFromNormalAndCoplanarPoint(dir, new THREE.Vector3(0, y, 0))
    const hit = intersect(e.nativeEvent)
    grabDy.current = hit ? hit.y - y : 0
    curIndex.current = index
    if (controls.current) controls.current.enabled = false
    setDragUid(uid)
    document.body.style.cursor = 'grabbing'
  }

  useEffect(() => {
    if (!dragUid) return
    const el = gl.domElement
    function onMove(ev) {
      const hit = intersect(ev)
      if (!hit) return
      const st = useLab.getState()
      const len = st.stack.length
      let idx = Math.round((hit.y - grabDy.current) / STEP + (len - 1) / 2)
      idx = Math.max(0, Math.min(len - 1, idx))
      if (idx !== curIndex.current) {
        st.moveBlock(curIndex.current, idx)
        curIndex.current = idx
      }
    }
    function onUp() {
      setDragUid(null)
      if (controls.current) controls.current.enabled = true
      document.body.style.cursor = 'auto'
    }
    el.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    return () => {
      el.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dragUid])

  return (
    <>
      <color attach="background" args={['#0a0e16']} />
      <ambientLight intensity={0.55} />
      <directionalLight position={[6, 10, 6]} intensity={1.1} />
      <directionalLight position={[-6, 4, -4]} intensity={0.4} color="#a78bfa" />
      <Environment preset="city" />

      <Spine n={Math.max(n, 1)} />
      <GhostDepth stack={stack} n={n} layers={layers} />

      {running && (
        <mesh ref={pulseRef}>
          <sphereGeometry args={[0.18, 16, 16]} />
          <meshStandardMaterial color="#ecfeff" emissive="#38bdf8" emissiveIntensity={2.2} />
        </mesh>
      )}

      {stack.map((b, i) => (
        <Block3D
          key={b.uid}
          position={[0, yForIndex(i, n), 0]}
          size={[BLOCK_W, BLOCK_H, BLOCK_D]}
          color={blockColor(b.key)}
          label={blockLabel(b)}
          sublabel={sublabel(b)}
          badge={blockBadge(b.key)}
          selected={b.uid === selectedUid || b.uid === dragUid}
          active={running && i === flowIdx}
          onSelect={() => select(b.uid)}
          onPointerDown={(e) => beginDrag(b.uid, i, e)}
        />
      ))}

      <Grid
        position={[0, yForIndex(0, n) - BLOCK_H, 0]}
        args={[30, 30]}
        cellColor="#1e293b"
        sectionColor="#334155"
        fadeDistance={34}
        infiniteGrid
      />
      <OrbitControls ref={controls} enablePan makeDefault minDistance={4} maxDistance={26} target={[0, 0, 0]} />
    </>
  )
}

export default function Builder() {
  const stack = useLab((s) => s.stack)
  const config = useLab((s) => s.config)
  const warnings = useMemo(() => validate(stack, config), [stack, config])
  const [running, setRunning] = useState(false)
  const [activeIdx, setActiveIdx] = useState(-1)
  const isMobile = useIsMobile()
  const [sheet, setSheet] = useState(null)

  const activeBlk = activeIdx >= 0 ? stack[activeIdx] : null

  const canvasArea = (
    <div className="canvas-wrap">
        <Canvas camera={{ position: [7, 2, 9], fov: 45 }} dpr={[1, 2]}>
          <Scene running={running} onActive={setActiveIdx} />
        </Canvas>

        <div className="overlay tr" style={{ pointerEvents: 'auto' }}>
          <button className={'btn ' + (running ? 'danger' : 'primary')} onClick={() => setRunning((r) => !r)}>
            {running ? '⏸ Stop' : '▶ Run forward pass'}
          </button>
        </div>

        {running && activeBlk && (
          <div className="overlay" style={{ bottom: 56, left: '50%', transform: 'translateX(-50%)', textAlign: 'center' }}>
            <div className="flow-caption">
              <span className="fc-name">{blockLabel(activeBlk)}</span>
              <span className="fc-sub">{FLOW_PHRASE[blockCategory(activeBlk.key)] || ''}</span>
            </div>
          </div>
        )}

        <div className="overlay tl">
          <div className="legend">
            {Object.entries({
              'Input / Output': CATEGORY_COLORS.io,
              Positional: CATEGORY_COLORS.position,
              Attention: CATEGORY_COLORS.attention,
              'Token mixing': CATEGORY_COLORS.mixing,
              Norm: CATEGORY_COLORS.norm,
            }).map(([k, c]) => (
              <div className="li" key={k}>
                <span className="swatch" style={{ background: c }} /> {k}
              </div>
            ))}
          </div>
          <div style={{ marginTop: 8 }}>
            {warnings.map((w, i) => (
              <div className="warn" key={i}>⚠ {w}</div>
            ))}
          </div>
        </div>

        <div className="overlay bl">
          <div className="legend" style={{ fontSize: 11.5, color: '#8a9ab2' }}>
            Drag blocks ↑ ↓ to reorder · click a block to explain &amp; edit it · orbit with mouse
          </div>
        </div>

        <BlockDetail />
    </div>
  )

  if (isMobile) {
    return (
      <div className="m-view">
        <div className="m-canvas">{canvasArea}</div>
        <div className="m-tabbar">
          <button onClick={() => setSheet('parts')}>＋ Parts</button>
          <button onClick={() => setSheet('setup')}>⚙ Setup</button>
        </div>
        {sheet === 'parts' && (
          <MobileSheet title="Parts bin — tap to add" onClose={() => setSheet(null)}>
            <Palette />
          </MobileSheet>
        )}
        {sheet === 'setup' && (
          <MobileSheet title="Model setup &amp; save" onClose={() => setSheet(null)}>
            <Inspector />
          </MobileSheet>
        )}
      </div>
    )
  }

  return (
    <div className="workspace">
      <Palette />
      {canvasArea}
      <Inspector />
    </div>
  )
}

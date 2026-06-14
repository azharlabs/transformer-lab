import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { Text } from '@react-three/drei'

// ---------- shared primitives ----------
const UP = new THREE.Vector3(0, 1, 0)

function Beam({ a, b, color = '#334155', radius = 0.016, opacity = 0.5 }) {
  const { mid, len, quat } = useMemo(() => {
    const va = new THREE.Vector3(...a)
    const vb = new THREE.Vector3(...b)
    const dir = vb.clone().sub(va)
    const len = dir.length() || 0.001
    const mid = va.add(vb.clone().sub(va).multiplyScalar(0.5))
    const quat = new THREE.Quaternion().setFromUnitVectors(UP, dir.clone().normalize())
    return { mid, len, quat }
  }, [a[0], a[1], a[2], b[0], b[1], b[2]])
  return (
    <mesh position={mid} quaternion={quat}>
      <cylinderGeometry args={[radius, radius, len, 6]} />
      <meshStandardMaterial color={color} transparent opacity={opacity} />
    </mesh>
  )
}

function Pulse({ a, b, color = '#fde047', speed = 0.6, offset = 0, size = 0.07 }) {
  const ref = useRef()
  const va = useMemo(() => new THREE.Vector3(...a), [a[0], a[1], a[2]])
  const vb = useMemo(() => new THREE.Vector3(...b), [b[0], b[1], b[2]])
  useFrame((s) => {
    if (!ref.current) return
    const f = (s.clock.elapsedTime * speed + offset) % 1
    ref.current.position.lerpVectors(va, vb, f)
    ref.current.material.opacity = Math.sin(f * Math.PI)
  })
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[size, 12, 12]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.6} transparent />
    </mesh>
  )
}

function Node({ position, color = '#38bdf8', r = 0.13, glowRef }) {
  return (
    <mesh position={position} ref={glowRef}>
      <sphereGeometry args={[r, 16, 16]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.4} metalness={0.3} roughness={0.4} />
    </mesh>
  )
}

const ys = (n, span = 2.2) => Array.from({ length: n }, (_, i) => (n === 1 ? 0 : span / 2 - (i * span) / (n - 1)))
const Label = (p) => <Text {...p} />

// ---------- attention family ----------
function AttentionAnim({ causal = false, cross = false, kvHeads = null, note = null, kLabel = null }) {
  const nQ = 4
  const nK = kvHeads ? Math.max(1, Math.min(4, kvHeads)) : 4
  const qy = ys(nQ)
  const ky = ys(nK)
  const Q = qy.map((y) => [-2.1, y, 0])
  const K = ky.map((y) => [0.1, y, 0])
  const OUT = [2.2, 0, 0]

  const pairs = useMemo(() => {
    const arr = []
    for (let i = 0; i < nQ; i++) {
      const kj = kvHeads ? Math.floor((i / nQ) * nK) : -1
      for (let j = 0; j < nK; j++) {
        const allow = causal ? j <= Math.round((i / (nQ - 1)) * (nK - 1)) : true
        if (allow) arr.push({ i, j, w: Math.exp(-Math.abs((kvHeads ? kj : i) - j) * 0.8) })
      }
    }
    return arr
  }, [causal, nK, kvHeads])

  const qkRefs = useRef({})
  const koRefs = useRef({})
  const qNodeRefs = useRef([])

  useFrame((s) => {
    const t = s.clock.elapsedTime
    const aq = Math.floor(t / 1.5) % nQ
    qNodeRefs.current.forEach((m, i) => {
      if (m) m.material.emissiveIntensity = i === aq ? 1.3 : 0.3
    })
    pairs.forEach(({ i, j, w }) => {
      const qk = qkRefs.current[`${i}-${j}`]
      if (qk) qk.material.opacity += ((i === aq ? 0.25 + w * 0.7 : 0.05) - qk.material.opacity) * 0.2
    })
    const koTarget = {}
    pairs.filter((p) => p.i === aq).forEach((p) => (koTarget[p.j] = 0.3 + p.w * 0.7))
    for (let j = 0; j < nK; j++) {
      const ko = koRefs.current[j]
      if (ko) ko.material.opacity += ((koTarget[j] || 0.05) - ko.material.opacity) * 0.2
    }
  })

  return (
    <group>
      {Q.map((p, i) => (
        <Node key={'q' + i} position={p} color="#fb7185" glowRef={(el) => (qNodeRefs.current[i] = el)} />
      ))}
      {K.map((p, j) => (
        <Node key={'k' + j} position={p} color="#38bdf8" />
      ))}
      <Node position={OUT} color="#a78bfa" r={0.16} />

      {pairs.map(({ i, j }) => (
        <mesh key={'b' + i + '-' + j} ref={(el) => (qkRefs.current[`${i}-${j}`] = el)} {...beamProps(Q[i], K[j])}>
          <cylinderGeometry args={[0.02, 0.02, 1, 6]} />
          <meshStandardMaterial color="#fb7185" emissive="#fb7185" emissiveIntensity={0.6} transparent opacity={0.1} />
        </mesh>
      ))}
      {K.map((p, j) => (
        <mesh key={'ko' + j} ref={(el) => (koRefs.current[j] = el)} {...beamProps(p, OUT)}>
          <cylinderGeometry args={[0.02, 0.02, 1, 6]} />
          <meshStandardMaterial color="#a78bfa" emissive="#a78bfa" emissiveIntensity={0.6} transparent opacity={0.1} />
        </mesh>
      ))}

      <Label position={[-2.1, 1.65, 0]} fontSize={0.2} color="#fb7185" anchorX="center">{cross ? 'decoder Q' : 'queries'}</Label>
      <Label position={[0.1, 1.65, 0]} fontSize={0.2} color="#38bdf8" anchorX="center">{kLabel || (cross ? 'encoder K·V' : kvHeads ? 'shared K·V' : 'keys · values')}</Label>
      <Label position={[2.2, 0.5, 0]} fontSize={0.18} color="#a78bfa" anchorX="center">context</Label>
      <Label position={[0, -1.9, 0]} fontSize={0.17} color="#8a9ab2" anchorX="center">
        {note || (causal ? 'causal mask: token i attends only to ≤ i' : 'every query attends to every key')}
      </Label>
    </group>
  )
}

function beamProps(a, b) {
  const va = new THREE.Vector3(...a)
  const vb = new THREE.Vector3(...b)
  const dir = vb.clone().sub(va)
  const len = dir.length() || 0.001
  const mid = va.clone().add(vb).multiplyScalar(0.5)
  const quat = new THREE.Quaternion().setFromUnitVectors(UP, dir.clone().normalize())
  return { position: mid.toArray(), quaternion: quat.toArray(), scale: [1, len, 1] }
}

// ---------- feed-forward ----------
function FFNAnim({ mult = 4, act = 'gelu' }) {
  const nIn = 4
  const nHid = Math.min(8, 3 + mult)
  const inP = ys(nIn).map((y) => [-2.1, y, 0])
  const hidP = ys(nHid).map((y) => [0, y, 0])
  const outP = ys(nIn).map((y) => [2.1, y, 0])
  const hidRefs = useRef([])
  useFrame((s) => {
    const t = s.clock.elapsedTime * 2
    hidRefs.current.forEach((m, i) => {
      if (m) m.material.emissiveIntensity = 0.3 + 0.9 * Math.max(0, Math.sin(t - i * 0.5))
    })
  })
  return (
    <group>
      {inP.map((a, i) => hidP.map((b, j) => <Beam key={`ih${i}-${j}`} a={a} b={b} color="#34d399" opacity={0.18} />))}
      {hidP.map((a, i) => outP.map((b, j) => <Beam key={`ho${i}-${j}`} a={a} b={b} color="#34d399" opacity={0.18} />))}
      {inP.map((p, i) => <Node key={'i' + i} position={p} color="#94a3b8" />)}
      {hidP.map((p, i) => <Node key={'h' + i} position={p} color="#34d399" r={0.12} glowRef={(el) => (hidRefs.current[i] = el)} />)}
      {outP.map((p, i) => <Node key={'o' + i} position={p} color="#94a3b8" />)}
      {inP.map((a, i) => <Pulse key={'p' + i} a={a} b={hidP[(i * 2) % nHid]} color="#a7f3d0" speed={0.7} offset={i * 0.2} size={0.05} />)}
      <Label position={[-2.1, 1.6, 0]} fontSize={0.18} color="#94a3b8" anchorX="center">d_model</Label>
      <Label position={[0, 1.6, 0]} fontSize={0.18} color="#34d399" anchorX="center">×{mult} wider · {act}</Label>
      <Label position={[2.1, 1.6, 0]} fontSize={0.18} color="#94a3b8" anchorX="center">d_model</Label>
      <Label position={[0, -1.9, 0]} fontSize={0.17} color="#8a9ab2" anchorX="center">each token expands, passes a nonlinearity, projects back</Label>
    </group>
  )
}

// ---------- mixture of experts ----------
function MoEAnim({ experts = 8, topk = 2 }) {
  const n = Math.min(8, experts)
  const tok = [-2.3, 0, 0]
  const router = [-1.0, 0, 0]
  const eP = ys(n, 2.6).map((y) => [0.6, y, 0])
  const out = [2.3, 0, 0]
  const eRefs = useRef([])
  useFrame((s) => {
    const sel = new Set()
    const base = Math.floor(s.clock.elapsedTime / 1.6) % n
    for (let k = 0; k < Math.min(topk, n); k++) sel.add((base + k * 3) % n)
    eRefs.current.forEach((m, i) => {
      if (m) m.material.emissiveIntensity += ((sel.has(i) ? 1.4 : 0.15) - m.material.emissiveIntensity) * 0.15
    })
  })
  return (
    <group>
      <Node position={tok} color="#94a3b8" r={0.15} />
      <Node position={router} color="#fbbf24" r={0.13} />
      <Beam a={tok} b={router} color="#fbbf24" opacity={0.6} />
      <Pulse a={tok} b={router} color="#fde68a" speed={0.8} />
      {eP.map((p, i) => (
        <group key={i}>
          <Beam a={router} b={p} color="#34d399" opacity={0.2} />
          <Beam a={p} b={out} color="#34d399" opacity={0.2} />
          <mesh position={p} ref={(el) => (eRefs.current[i] = el)}>
            <boxGeometry args={[0.4, 0.26, 0.26]} />
            <meshStandardMaterial color="#34d399" emissive="#34d399" emissiveIntensity={0.15} metalness={0.3} roughness={0.4} />
          </mesh>
        </group>
      ))}
      <Node position={out} color="#a78bfa" r={0.15} />
      <Label position={[-2.3, 0.5, 0]} fontSize={0.17} color="#94a3b8" anchorX="center">token</Label>
      <Label position={[-1.0, -0.45, 0]} fontSize={0.16} color="#fbbf24" anchorX="center">router</Label>
      <Label position={[0.6, 1.7, 0]} fontSize={0.16} color="#34d399" anchorX="center">{n} experts</Label>
      <Label position={[0, -1.9, 0]} fontSize={0.17} color="#8a9ab2" anchorX="center">router sends each token to its top-{topk} experts only</Label>
    </group>
  )
}

// ---------- mamba / ssm ----------
function MambaAnim({ d_state = 16 }) {
  const n = 6
  const tx = Array.from({ length: n }, (_, i) => -2.4 + (i * 4.8) / (n - 1))
  const pulse = useRef()
  useFrame((s) => {
    const f = (s.clock.elapsedTime * 0.4) % 1
    if (pulse.current) pulse.current.position.x = -2.4 + f * 4.8
  })
  return (
    <group>
      {tx.map((x, i) => (
        <group key={i}>
          <mesh position={[x, -1, 0]}>
            <boxGeometry args={[0.4, 0.4, 0.4]} />
            <meshStandardMaterial color="#34d399" emissive="#34d399" emissiveIntensity={0.3} />
          </mesh>
          <Node position={[x, 0.3, 0]} color="#a78bfa" r={0.16} />
          <Beam a={[x, -0.8, 0]} b={[x, 0.15, 0]} color="#475569" opacity={0.7} />
          {i < n - 1 && <Beam a={[x, 0.3, 0]} b={[tx[i + 1], 0.3, 0]} color="#a78bfa" radius={0.025} opacity={0.8} />}
        </group>
      ))}
      <mesh ref={pulse} position={[-2.4, 0.3, 0]}>
        <sphereGeometry args={[0.12, 12, 12]} />
        <meshStandardMaterial color="#fde047" emissive="#fde047" emissiveIntensity={1.8} />
      </mesh>
      <Label position={[0, 1.2, 0]} fontSize={0.18} color="#a78bfa" anchorX="center">hidden state hₜ (size {d_state})</Label>
      <Label position={[0, -1.7, 0]} fontSize={0.17} color="#8a9ab2" anchorX="center">state rolls forward — O(n), no token-to-token matrix</Label>
    </group>
  )
}

// ---------- embedding ----------
function EmbeddingAnim() {
  const rows = 5
  const rowRefs = useRef([])
  const barRefs = useRef([])
  const targets = useRef([])
  useFrame((s) => {
    const sel = Math.floor(s.clock.elapsedTime / 1.4) % rows
    rowRefs.current.forEach((m, i) => {
      if (m) m.material.emissiveIntensity += ((i === sel ? 1.2 : 0.1) - m.material.emissiveIntensity) * 0.2
    })
    if (targets.current.length === 0 || Math.floor(s.clock.elapsedTime / 1.4) !== targets.lastSel) {
      targets.lastSel = Math.floor(s.clock.elapsedTime / 1.4)
      targets.current = Array.from({ length: 4 }, () => 0.2 + Math.random() * 0.9)
    }
    barRefs.current.forEach((m, i) => {
      if (m && targets.current[i]) {
        const h = targets.current[i]
        m.scale.y += (h - m.scale.y) * 0.15
        m.position.y = 0.3 + m.scale.y / 2 - 0.5
      }
    })
  })
  return (
    <group>
      <mesh position={[-2.3, 0, 0]}>
        <boxGeometry args={[0.45, 0.45, 0.45]} />
        <meshStandardMaterial color="#94a3b8" emissive="#94a3b8" emissiveIntensity={0.3} />
      </mesh>
      <Label position={[-2.3, -0.55, 0]} fontSize={0.16} color="#94a3b8" anchorX="center">token id</Label>
      {ys(rows, 2).map((y, i) => (
        <mesh key={i} position={[-0.6, y, 0]} ref={(el) => (rowRefs.current[i] = el)}>
          <boxGeometry args={[1.2, 0.28, 0.2]} />
          <meshStandardMaterial color="#38bdf8" emissive="#38bdf8" emissiveIntensity={0.1} metalness={0.3} roughness={0.4} />
        </mesh>
      ))}
      <Label position={[-0.6, 1.5, 0]} fontSize={0.16} color="#38bdf8" anchorX="center">embedding table</Label>
      {[0, 1, 2, 3].map((i) => (
        <mesh key={i} position={[1.6 + i * 0.28, 0, 0]} ref={(el) => (barRefs.current[i] = el)} scale={[1, 0.5, 1]}>
          <boxGeometry args={[0.18, 1, 0.18]} />
          <meshStandardMaterial color="#a78bfa" emissive="#a78bfa" emissiveIntensity={0.5} />
        </mesh>
      ))}
      <Label position={[1.9, 1.5, 0]} fontSize={0.16} color="#a78bfa" anchorX="center">vector</Label>
      <Label position={[0, -1.9, 0]} fontSize={0.17} color="#8a9ab2" anchorX="center">a token id just looks up one row of the table</Label>
    </group>
  )
}

// ---------- normalisation ----------
function NormAnim({ rms = false }) {
  const n = 6
  const refs = useRef([])
  const xs = Array.from({ length: n }, (_, i) => -2.2 + (i * 4.4) / (n - 1))
  const rand = useMemo(() => Array.from({ length: n }, () => 0.4 + Math.random() * 1.4), [])
  useFrame((s) => {
    const phase = (s.clock.elapsedTime % 4) / 4
    const norm = phase > 0.5
    refs.current.forEach((m, i) => {
      if (!m) return
      const target = norm ? 1.0 : rand[i]
      m.scale.y += (target - m.scale.y) * 0.12
      m.position.y = m.scale.y / 2 - 1
      m.material.emissiveIntensity = norm ? 0.9 : 0.2
    })
  })
  return (
    <group>
      {xs.map((x, i) => (
        <mesh key={i} position={[x, 0, 0]} ref={(el) => (refs.current[i] = el)} scale={[1, rand[i], 1]}>
          <boxGeometry args={[0.4, 1, 0.4]} />
          <meshStandardMaterial color="#fbbf24" emissive="#fbbf24" emissiveIntensity={0.3} metalness={0.2} roughness={0.5} />
        </mesh>
      ))}
      <Label position={[0, 1.7, 0]} fontSize={0.18} color="#fbbf24" anchorX="center">{rms ? 'RMSNorm' : 'LayerNorm'}</Label>
      <Label position={[0, -1.9, 0]} fontSize={0.17} color="#8a9ab2" anchorX="center">
        {rms ? 'rescale by root-mean-square (no centring)' : 'centre & rescale features to a stable range'}
      </Label>
    </group>
  )
}

// ---------- dropout ----------
function DropoutAnim({ p = 0.1 }) {
  const cols = 6
  const rowsN = 4
  const refs = useRef([])
  useFrame((s) => {
    const tick = Math.floor(s.clock.elapsedTime * 1.5)
    refs.current.forEach((m, idx) => {
      if (!m) return
      const on = pseudo(idx + tick * 97) > p
      const target = on ? 1 : 0.08
      m.material.opacity += (target - m.material.opacity) * 0.2
      m.material.emissiveIntensity += ((on ? 0.6 : 0) - m.material.emissiveIntensity) * 0.2
    })
  })
  const nodes = []
  for (let c = 0; c < cols; c++)
    for (let r = 0; r < rowsN; r++)
      nodes.push([-2.2 + (c * 4.4) / (cols - 1), 1.2 - (r * 2.4) / (rowsN - 1), 0])
  return (
    <group>
      {nodes.map((pn, i) => (
        <mesh key={i} position={pn} ref={(el) => (refs.current[i] = el)}>
          <sphereGeometry args={[0.13, 14, 14]} />
          <meshStandardMaterial color="#94a3b8" emissive="#cbd5e1" emissiveIntensity={0.5} transparent />
        </mesh>
      ))}
      <Label position={[0, -1.9, 0]} fontSize={0.17} color="#8a9ab2" anchorX="center">randomly zero ~{Math.round(p * 100)}% of activations (training only)</Label>
    </group>
  )
}
function pseudo(n) {
  const x = Math.sin(n * 12.9898) * 43758.5453
  return x - Math.floor(x)
}

// ---------- positional ----------
function PosAnim({ type = 'pos_sinusoidal' }) {
  const n = 6
  const xs = Array.from({ length: n }, (_, i) => -2.2 + (i * 4.4) / (n - 1))
  const arrowRefs = useRef([])
  const waveRefs = useRef([])
  const none = type === 'pos_none'
  useFrame((s) => {
    const t = s.clock.elapsedTime
    if (type === 'pos_rotary') {
      arrowRefs.current.forEach((m, i) => {
        if (m) m.rotation.z = t * 0.8 + i * 0.6
      })
    } else if (!none) {
      waveRefs.current.forEach((m, i) => {
        if (m) m.position.y = Math.sin(t + i * 0.9) * 0.6
      })
    }
  })
  return (
    <group>
      {xs.map((x, i) => (
        <group key={i}>
          <mesh position={[x, -1, 0]}>
            <boxGeometry args={[0.4, 0.4, 0.4]} />
            <meshStandardMaterial color="#a78bfa" emissive="#a78bfa" emissiveIntensity={0.3} />
          </mesh>
          {none ? null : type === 'pos_rotary' ? (
            <mesh position={[x, 0.2, 0]} ref={(el) => (arrowRefs.current[i] = el)}>
              <coneGeometry args={[0.12, 0.4, 12]} />
              <meshStandardMaterial color="#fbbf24" emissive="#fbbf24" emissiveIntensity={0.7} />
            </mesh>
          ) : (
            <mesh position={[x, 0.2, 0]} ref={(el) => (waveRefs.current[i] = el)}>
              <sphereGeometry args={[0.13, 14, 14]} />
              <meshStandardMaterial color="#fbbf24" emissive="#fbbf24" emissiveIntensity={0.7} />
            </mesh>
          )}
        </group>
      ))}
      <Label position={[0, 1.5, 0]} fontSize={0.18} color="#a78bfa" anchorX="center">
        {none ? 'no position added' : type === 'pos_rotary' ? 'rotate Q/K by position' : type === 'pos_learned' ? 'learned position vectors' : 'fixed sin/cos signals'}
      </Label>
      <Label position={[0, -1.9, 0]} fontSize={0.17} color="#8a9ab2" anchorX="center">
        {none ? 'order comes only from the causal mask (NoPE)' : 'gives each position a distinct, order-aware signature'}
      </Label>
    </group>
  )
}

// ---------- output head ----------
function LMHeadAnim({ mode = 'language' }) {
  const nLog = 7
  const xs = Array.from({ length: nLog }, (_, i) => -1.8 + (i * 3.6) / (nLog - 1))
  const refs = useRef([])
  useFrame((s) => {
    const sel = Math.floor(s.clock.elapsedTime / 1.5) % nLog
    refs.current.forEach((m, i) => {
      if (!m) return
      const target = i === sel ? 1.5 : 0.3 + 0.5 * pseudo(i + Math.floor(s.clock.elapsedTime / 1.5) * 31)
      m.scale.y += (target - m.scale.y) * 0.12
      m.position.y = m.scale.y / 2 - 1
      m.material.emissiveIntensity = i === sel ? 1.2 : 0.3
    })
  })
  return (
    <group>
      <mesh position={[-2.6, 0, 0]}>
        <boxGeometry args={[0.3, 1.4, 0.3]} />
        <meshStandardMaterial color="#94a3b8" emissive="#94a3b8" emissiveIntensity={0.3} />
      </mesh>
      <Label position={[-2.6, -1.2, 0]} fontSize={0.15} color="#94a3b8" anchorX="center">hidden</Label>
      {xs.map((x, i) => (
        <mesh key={i} position={[x, 0, 0]} ref={(el) => (refs.current[i] = el)} scale={[1, 0.6, 1]}>
          <boxGeometry args={[0.28, 1, 0.28]} />
          <meshStandardMaterial color="#38bdf8" emissive="#38bdf8" emissiveIntensity={0.4} />
        </mesh>
      ))}
      <Label position={[0, 1.6, 0]} fontSize={0.17} color="#38bdf8" anchorX="center">{mode === 'classification' ? 'class scores' : 'logits over vocab'}</Label>
      <Label position={[0, -1.9, 0]} fontSize={0.17} color="#8a9ab2" anchorX="center">project to scores → softmax picks the winner (highlighted)</Label>
    </group>
  )
}

// ---------- picker ----------
export default function BlockAnim({ blockKey, params = {} }) {
  switch (blockKey) {
    case 'embedding': return <EmbeddingAnim />
    case 'pos_sinusoidal':
    case 'pos_learned':
    case 'pos_rotary':
    case 'pos_none': return <PosAnim type={blockKey} />
    case 'mha': return <AttentionAnim />
    case 'causal_mha': return <AttentionAnim causal />
    case 'cross_attention': return <AttentionAnim cross />
    case 'gqa': return <AttentionAnim causal kvHeads={params.kv_heads || 2} />
    case 'mla': return <AttentionAnim causal kLabel="latent K·V" note="K·V compressed to a small latent, then projected back" />
    case 'swa': return <AttentionAnim causal note={`local window of ${params.window || 1024} past tokens`} />
    case 'ffn': return <FFNAnim mult={params.mult || 4} act={params.act || 'swiglu'} />
    case 'moe': return <MoEAnim experts={params.experts || 8} topk={params.topk || 2} />
    case 'mamba': return <MambaAnim d_state={params.d_state || 16} />
    case 'deltanet': return <MambaAnim d_state={params.expand || 2} />
    case 'layernorm': return <NormAnim />
    case 'rmsnorm': return <NormAnim rms />
    case 'qk_norm': return <NormAnim rms />
    case 'dropout': return <DropoutAnim p={params.p ?? 0.1} />
    case 'lm_head': return <LMHeadAnim mode={params.mode || 'language'} />
    default: return null
  }
}

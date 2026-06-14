import { forwardRef, useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text, Environment } from '@react-three/drei'
import { DEMO_TOKENS, DEMO_SRC, HEADS, attentionMatrix, rawMatrix } from '../lib/attention.js'
import MobileSheet from './MobileSheet.jsx'
import { useIsMobile } from '../lib/useIsMobile.js'

const STEP = 0.95
const MAXH = 2.6
const lowC = new THREE.Color('#16233a')
const hiC = new THREE.Color('#fb7185')
const pos = (idx, count) => (idx - (count - 1) / 2) * STEP
const ease = (cur, target, k = 0.15) => cur + (target - cur) * k

// ============================================================ shared bits
const Beam2 = forwardRef(function Beam2({ a, b, color = '#64748b' }, ref) {
  const { mid, len, quat } = useMemo(() => {
    const va = new THREE.Vector3(...a)
    const vb = new THREE.Vector3(...b)
    const d = vb.clone().sub(va)
    const len = d.length() || 0.001
    const mid = va.clone().add(d.clone().multiplyScalar(0.5))
    const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize())
    return { mid, len, quat }
  }, [a[0], a[1], a[2], b[0], b[1], b[2]])
  return (
    <mesh ref={ref} position={mid.toArray()} quaternion={quat.toArray()}>
      <cylinderGeometry args={[0.02, 0.02, len, 6]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.6} transparent opacity={0.06} />
    </mesh>
  )
})

// ============================================================ heatmap walk
// VIEWS[modeId] = [ [src, mask], ... ]  src: flat|raw|final   mask: none|causal|window
const VIEWS = {
  mha: [['flat', 'none'], ['raw', 'none'], ['final', 'none'], ['final', 'none']],
  causal: [['raw', 'none'], ['raw', 'causal'], ['final', 'causal'], ['final', 'causal']],
  swa: [['raw', 'causal'], ['raw', 'window'], ['final', 'window'], ['final', 'window']],
  cross: [['flat', 'none'], ['raw', 'none'], ['final', 'none'], ['final', 'none']],
}

function HeatWalk({ rowToks, colToks, raw, final, views, opts, tl }) {
  const n = rowToks.length
  const m = colToks.length
  const W = opts.window || 3
  const meshes = useRef({})
  const cur = useRef({})

  useFrame(() => {
    const step = Math.min(tl.current.step, views.length - 1)
    const [src, mask] = views[step]
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        const key = i + '-' + j
        const mesh = meshes.current[key]
        if (!mesh) continue
        const masked = (mask === 'causal' && j > i) || (mask === 'window' && (j > i || i - j >= W))
        let intensity = 0.02
        if (!masked) {
          if (src === 'raw') intensity = raw[i][j]
          else if (src === 'final') intensity = Math.min(final[i][j] * 3, 1)
        }
        const v = (cur.current[key] = ease(cur.current[key] ?? 0, intensity, 0.16))
        const h = Math.max(v * MAXH, 0.02)
        mesh.scale.y = h
        mesh.position.y = h / 2
        mesh.material.color.copy(lowC).lerp(hiC, Math.min(v * 1.15, 1))
        mesh.material.emissive.copy(hiC)
        mesh.material.emissiveIntensity = v * 0.8
      }
    }
  })

  return (
    <group>
      {colToks.map((t, j) => (
        <Text key={'k' + j} position={[pos(j, m), 0.02, pos(n, n) + 0.15]} rotation={[-Math.PI / 2, 0, 0]} fontSize={0.24} color="#9fb3cc" anchorX="center">{t}</Text>
      ))}
      {rowToks.map((t, i) => (
        <Text key={'q' + i} position={[pos(-1, m) - 0.1, 0.02, pos(i, n)]} rotation={[-Math.PI / 2, 0, Math.PI / 2]} fontSize={0.24} color="#9fb3cc" anchorX="center">{t}</Text>
      ))}
      <Text position={[0, 0.02, pos(n, n) + 0.7]} rotation={[-Math.PI / 2, 0, 0]} fontSize={0.2} color="#5b6b82" anchorX="center">keys →</Text>
      {Array.from({ length: n }).map((_, i) =>
        Array.from({ length: m }).map((_, j) => (
          <mesh key={i + '-' + j} ref={(el) => (meshes.current[i + '-' + j] = el)} position={[pos(j, m), 0.5, pos(i, n)]}>
            <boxGeometry args={[0.72, 1, 0.72]} />
            <meshStandardMaterial color={lowC} metalness={0.2} roughness={0.5} />
          </mesh>
        ))
      )}
    </group>
  )
}

// ============================================================ GQA walk
function GQAWalk({ qHeads, kvHeads, tl }) {
  const qx = useMemo(() => Array.from({ length: qHeads }, (_, i) => -2.7 + (i * 5.4) / (qHeads - 1)), [qHeads])
  const kvx = useMemo(() => Array.from({ length: kvHeads }, (_, k) => (kvHeads === 1 ? 0 : -2.2 + (k * 4.4) / (kvHeads - 1))), [kvHeads])
  const groupOf = (i) => Math.floor(i / (qHeads / kvHeads))
  const kvRefs = useRef([])
  const lineRefs = useRef([])
  const pulseRefs = useRef([])

  useFrame((s) => {
    const step = tl.current.step
    kvRefs.current.forEach((mh) => {
      if (mh) {
        const t = step >= 1 ? 1 : 0.001
        mh.scale.setScalar(ease(mh.scale.x, t, 0.18))
      }
    })
    lineRefs.current.forEach((mh) => {
      if (mh) mh.material.opacity = ease(mh.material.opacity, step >= 2 ? 0.55 : 0.0, 0.15)
    })
    pulseRefs.current.forEach((mp, i) => {
      if (!mp) return
      if (step >= 2) {
        mp.visible = true
        const f = (s.clock.elapsedTime * 0.7 + i * 0.12) % 1
        mp.position.lerpVectors(new THREE.Vector3(qx[i], 1.2, 0), new THREE.Vector3(kvx[groupOf(i)], -1.2, 0), f)
        mp.material.opacity = Math.sin(f * Math.PI)
      } else mp.visible = false
    })
  })

  return (
    <group>
      {qx.map((x, i) => (
        <group key={'q' + i}>
          <mesh position={[x, 1.2, 0]}>
            <boxGeometry args={[0.42, 0.42, 0.42]} />
            <meshStandardMaterial color="#fb7185" emissive="#fb7185" emissiveIntensity={0.45} />
          </mesh>
        </group>
      ))}
      {kvx.map((x, k) => (
        <mesh key={'kv' + k} ref={(el) => (kvRefs.current[k] = el)} position={[x, -1.2, 0]} scale={[0.001, 0.001, 0.001]}>
          <boxGeometry args={[0.74, 0.62, 0.5]} />
          <meshStandardMaterial color="#38bdf8" emissive="#38bdf8" emissiveIntensity={0.45} />
        </mesh>
      ))}
      {qx.map((x, i) => {
        const a = new THREE.Vector3(x, 1.2, 0)
        const b = new THREE.Vector3(kvx[groupOf(i)], -1.2, 0)
        const mid = a.clone().add(b).multiplyScalar(0.5)
        const len = a.distanceTo(b)
        const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize())
        return (
          <mesh key={'l' + i} ref={(el) => (lineRefs.current[i] = el)} position={mid.toArray()} quaternion={q.toArray()}>
            <cylinderGeometry args={[0.015, 0.015, len, 6]} />
            <meshStandardMaterial color="#64748b" transparent opacity={0} />
          </mesh>
        )
      })}
      {qx.map((_, i) => (
        <mesh key={'p' + i} ref={(el) => (pulseRefs.current[i] = el)} visible={false}>
          <sphereGeometry args={[0.06, 10, 10]} />
          <meshStandardMaterial color="#fde047" emissive="#fde047" emissiveIntensity={1.6} transparent />
        </mesh>
      ))}
      <Text position={[0, 1.85, 0]} fontSize={0.24} color="#fb7185" anchorX="center">{qHeads} query heads</Text>
      <Text position={[0, -1.95, 0]} fontSize={0.24} color="#38bdf8" anchorX="center">{kvHeads} KV head{kvHeads > 1 ? 's' : ''} {kvHeads < qHeads ? '· shared' : ''}</Text>
    </group>
  )
}

// ============================================================ MLA walk
function MLAWalk({ tl }) {
  const heights = [1.2, 0.8, 1.5, 1.0]
  const latent = useRef()
  const reconRefs = useRef([])
  const bigRefs = useRef([])
  const down = useRef()
  const up = useRef()

  useFrame((s) => {
    const { step, prog } = tl.current
    if (latent.current) latent.current.scale.setScalar(ease(latent.current.scale.x, step >= 1 ? 1 : 0.001, 0.18))
    bigRefs.current.forEach((m) => {
      if (m) m.material.emissiveIntensity = ease(m.material.emissiveIntensity, step >= 2 ? 0.12 : 0.45, 0.12)
    })
    reconRefs.current.forEach((m, i) => {
      if (m) {
        const t = step >= 3 ? heights[i] : 0.001
        m.scale.y = ease(m.scale.y, t, 0.15)
        m.position.y = m.scale.y / 2 - 0.6
      }
    })
    if (down.current) {
      down.current.visible = step === 1
      if (step === 1) {
        down.current.position.lerpVectors(new THREE.Vector3(-1.7, 0.2, 0), new THREE.Vector3(0, 0.1, 0), prog)
        down.current.material.opacity = Math.sin(prog * Math.PI)
      }
    }
    if (up.current) {
      up.current.visible = step === 3
      if (step === 3) {
        up.current.position.lerpVectors(new THREE.Vector3(0, 0.1, 0), new THREE.Vector3(2.0, 0.2, 0), prog)
        up.current.material.opacity = Math.sin(prog * Math.PI)
      }
    }
  })

  return (
    <group>
      {heights.map((h, i) => (
        <mesh key={'k' + i} ref={(el) => (bigRefs.current[i] = el)} position={[-2.4 + i * 0.32, h / 2 - 0.6, 0]}>
          <boxGeometry args={[0.2, h, 0.2]} />
          <meshStandardMaterial color="#38bdf8" emissive="#38bdf8" emissiveIntensity={0.45} />
        </mesh>
      ))}
      <Text position={[-2.0, 1.3, 0]} fontSize={0.22} color="#38bdf8" anchorX="center">full K·V</Text>
      <mesh ref={latent} position={[0, 0, 0]} scale={[0.001, 0.001, 0.001]}>
        <boxGeometry args={[0.4, 0.7, 0.4]} />
        <meshStandardMaterial color="#a78bfa" emissive="#a78bfa" emissiveIntensity={0.7} />
      </mesh>
      <Text position={[0, 1.0, 0]} fontSize={0.22} color="#a78bfa" anchorX="center">latent (cached)</Text>
      {heights.map((h, i) => (
        <mesh key={'r' + i} ref={(el) => (reconRefs.current[i] = el)} position={[1.9 + i * 0.32, 0, 0]} scale={[1, 0.001, 1]}>
          <boxGeometry args={[0.2, 1, 0.2]} />
          <meshStandardMaterial color="#34d399" emissive="#34d399" emissiveIntensity={0.45} />
        </mesh>
      ))}
      <Text position={[2.3, 1.3, 0]} fontSize={0.22} color="#34d399" anchorX="center">reconstructed</Text>
      <mesh ref={down} visible={false}><sphereGeometry args={[0.11, 12, 12]} /><meshStandardMaterial color="#fde047" emissive="#fde047" emissiveIntensity={1.6} transparent /></mesh>
      <mesh ref={up} visible={false}><sphereGeometry args={[0.11, 12, 12]} /><meshStandardMaterial color="#fde047" emissive="#fde047" emissiveIntensity={1.6} transparent /></mesh>
    </group>
  )
}

// ============================================================ recurrence walk
function RecurWalk({ mamba, tl }) {
  const n = 6
  const tx = useMemo(() => Array.from({ length: n }, (_, i) => -2.4 + (i * 4.8) / (n - 1)), [])
  const stateRefs = useRef([])
  const linkRefs = useRef([])
  const pulse = useRef()

  useFrame((s) => {
    const step = tl.current.step
    stateRefs.current.forEach((m) => {
      if (m) m.scale.setScalar(ease(m.scale.x, step >= 1 ? 1 : 0.001, 0.18))
    })
    linkRefs.current.forEach((m) => {
      if (m) m.material.opacity = ease(m.material.opacity, step >= 2 ? 0.8 : 0.0, 0.15)
    })
    if (pulse.current) {
      pulse.current.visible = step >= 2
      if (step >= 2) {
        const f = (s.clock.elapsedTime * 0.4) % 1
        pulse.current.position.x = -2.4 + f * 4.8
      }
    }
  })

  return (
    <group>
      {tx.map((x, i) => (
        <group key={i}>
          <mesh position={[x, -1, 0]}><boxGeometry args={[0.4, 0.4, 0.4]} /><meshStandardMaterial color="#34d399" emissive="#34d399" emissiveIntensity={0.3} /></mesh>
          <mesh ref={(el) => (stateRefs.current[i] = el)} position={[x, 0.3, 0]} scale={[0.001, 0.001, 0.001]}><sphereGeometry args={[0.18, 16, 16]} /><meshStandardMaterial color="#a78bfa" emissive="#a78bfa" emissiveIntensity={0.5} /></mesh>
          <mesh position={[x, -0.35, 0]}><cylinderGeometry args={[0.025, 0.025, 0.7, 8]} /><meshStandardMaterial color="#475569" /></mesh>
          {i < n - 1 && (
            <mesh ref={(el) => (linkRefs.current[i] = el)} position={[x + 0.48, 0.3, 0]} rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[0.03, 0.03, 0.96, 8]} /><meshStandardMaterial color="#a78bfa" emissive="#a78bfa" emissiveIntensity={0.6} transparent opacity={0} /></mesh>
          )}
        </group>
      ))}
      <Text position={[0, 1.2, 0]} fontSize={0.22} color="#a78bfa" anchorX="center">{mamba ? 'state-space hₜ' : 'fast-weight memory hₜ'}</Text>
      <mesh ref={pulse} position={[-2.4, 0.3, 0]} visible={false}><sphereGeometry args={[0.12, 12, 12]} /><meshStandardMaterial color="#fde047" emissive="#fde047" emissiveIntensity={1.8} /></mesh>
    </group>
  )
}

// ============================================================ scaled-dot-product walk (the generic one)
const WQUERIES = [1, 2, 3, 5]
const WPHASES = [
  { id: 'query', dur: 2.4 },
  { id: 'scores', dur: 3.8 },
  { id: 'softmax', dur: 2.2 },
  { id: 'sum', dur: 3.0 },
]
const WCYCLE = WPHASES.reduce((a, p) => a + p.dur, 0)
const KEYY = 1.2
const QX = 0
const QY = -1.7

function Walkthrough({ onState, playing }) {
  const toks = DEMO_TOKENS
  const n = toks.length
  const weights = useMemo(() => attentionMatrix(toks, toks, 2, { causal: true }), [])
  const xs = useMemo(() => toks.map((_, i) => (i - (n - 1) / 2) * 1.0), [n])
  const keyRefs = useRef([])
  const beamRefs = useRef({})
  const barRefs = useRef([])
  const pulseRefs = useRef({})
  const queryChip = useRef()
  const tref = useRef(0)
  const lastKey = useRef('')

  useFrame((_, dt) => {
    if (playing) tref.current += dt
    const t = tref.current
    const total = t % (WQUERIES.length * WCYCLE)
    const slot = Math.floor(total / WCYCLE)
    const q = WQUERIES[slot]
    const within = total - slot * WCYCLE
    let phase = 'query', prog = 0, acc = 0
    for (const p of WPHASES) {
      if (within < acc + p.dur) { phase = p.id; prog = (within - acc) / p.dur; break }
      acc += p.dur
    }
    const allowed = q + 1
    const sweep = phase === 'scores' ? Math.min(allowed - 1, Math.floor(prog * allowed)) : -1
    const w = weights[q]
    const stateKey = q + '|' + phase + '|' + sweep
    if (stateKey !== lastKey.current) { lastKey.current = stateKey; onState && onState({ q, tok: toks[q], phase }) }

    keyRefs.current.forEach((m, j) => {
      if (!m) return
      const future = j > q
      let e = future ? 0.04 : j === q ? 0.9 : 0.25
      if (phase === 'scores' && j === sweep) e = 1.5
      m.material.emissiveIntensity += (e - m.material.emissiveIntensity) * 0.2
      m.material.opacity += ((future ? 0.22 : 1) - m.material.opacity) * 0.2
    })
    for (let j = 0; j < n; j++) {
      const b = beamRefs.current[j]
      if (b) {
        let o = 0.04
        if (j <= q) {
          if (phase === 'scores') o = j <= sweep ? (j === sweep ? 0.95 : 0.22) : 0.04
          else if (phase === 'softmax' || phase === 'sum') o = 0.12 + w[j] * 0.85
          else o = 0.06
        }
        b.material.opacity += (o - b.material.opacity) * 0.2
      }
      const bar = barRefs.current[j]
      if (bar) {
        let target = 0.02
        if (j <= q) {
          if (phase === 'scores') target = j <= sweep ? Math.sqrt(w[j]) * 1.7 : 0.02
          else if (phase === 'softmax' || phase === 'sum') target = Math.max(w[j] * 2.6, 0.03)
        }
        const h = (bar.userData.h = (bar.userData.h ?? 0.02) + (target - (bar.userData.h ?? 0.02)) * 0.15)
        bar.scale.y = h
        bar.position.y = KEYY - 0.42 - h / 2
        bar.material.opacity += ((j <= q && phase !== 'query' ? 1 : 0) - bar.material.opacity) * 0.2
      }
      const p = pulseRefs.current[j]
      if (p) {
        if (phase === 'sum' && j <= q) {
          p.visible = true
          p.position.lerpVectors(new THREE.Vector3(xs[j], KEYY, 0), new THREE.Vector3(QX, QY, 0), prog)
          p.scale.setScalar(0.06 + w[j] * 0.55)
          p.material.opacity = Math.sin(prog * Math.PI)
        } else p.visible = false
      }
    }
    if (queryChip.current) {
      const g = phase === 'sum' ? 0.5 + prog * 1.4 : 0.4
      queryChip.current.material.emissiveIntensity += (g - queryChip.current.material.emissiveIntensity) * 0.15
    }
  })

  return (
    <group>
      {toks.map((t, i) => (
        <group key={i}>
          <mesh ref={(el) => (keyRefs.current[i] = el)} position={[xs[i], KEYY, 0]}>
            <boxGeometry args={[0.74, 0.5, 0.4]} />
            <meshStandardMaterial color="#38bdf8" emissive="#38bdf8" emissiveIntensity={0.25} metalness={0.3} roughness={0.4} transparent />
          </mesh>
          <Text position={[xs[i], KEYY, 0.26]} fontSize={0.16} color="#04121e" anchorX="center" anchorY="middle">{t}</Text>
          <mesh ref={(el) => (barRefs.current[i] = el)} position={[xs[i], KEYY - 0.42, 0]} scale={[1, 0.02, 1]}>
            <boxGeometry args={[0.5, 1, 0.18]} />
            <meshStandardMaterial color="#fb7185" emissive="#fb7185" emissiveIntensity={0.6} transparent opacity={0} />
          </mesh>
          <Beam2 ref={(el) => (beamRefs.current[i] = el)} a={[QX, QY + 0.3, 0]} b={[xs[i], KEYY - 0.7, 0]} />
          <mesh ref={(el) => (pulseRefs.current[i] = el)} visible={false}>
            <sphereGeometry args={[0.1, 12, 12]} />
            <meshStandardMaterial color="#fde047" emissive="#fde047" emissiveIntensity={1.7} transparent />
          </mesh>
        </group>
      ))}
      <mesh ref={queryChip} position={[QX, QY, 0]}>
        <boxGeometry args={[1.2, 0.62, 0.5]} />
        <meshStandardMaterial color="#a78bfa" emissive="#a78bfa" emissiveIntensity={0.4} metalness={0.3} roughness={0.4} />
      </mesh>
      <Text position={[QX, QY, 0.27]} fontSize={0.18} color="#100a1e" anchorX="center" anchorY="middle">query</Text>
      <Text position={[0, KEYY + 0.7, 0]} fontSize={0.2} color="#9fb3cc" anchorX="center">the sentence (keys / values)</Text>
    </group>
  )
}

const WALK_STEPS = {
  query: { n: 0, title: 'Pick a query', body: (tok) => `"${tok}" is the token doing the looking. It becomes a query vector.` },
  scores: { n: 1, title: 'Score every key', body: () => `Compare the query to each earlier token with a dot product. Brighter link = more relevant. Future tokens stay masked.` },
  softmax: { n: 2, title: 'Softmax → weights', body: () => `Turn the raw scores into attention weights that add up to 1. The bars show how much each token gets.` },
  sum: { n: 3, title: 'Blend the values', body: (tok) => `Mix the tokens by those weights. The result is "${tok}"'s new, context-aware representation.` },
}

// ============================================================ per-mode step scripts
const STEPS = {
  mha: [
    { t: 'All tokens, every pair', b: 'In full attention each token can see every other token — and each head does this independently.' },
    { t: 'Score every pair', b: 'For one head, compute how strongly each query (row) matches each key (column).' },
    { t: 'Softmax each row', b: 'Normalise every row into weights that sum to 1. Brighter = more attention.' },
    { t: 'The n² cost', b: 'The grid is n×n, so compute and KV cache grow with the square of the sequence length.' },
  ],
  causal: [
    { t: 'Score every pair', b: 'Start by computing the full query×key score grid.' },
    { t: 'Apply the causal mask', b: 'Black out the upper triangle — token i must not see future tokens (j > i).' },
    { t: 'Softmax the visible part', b: 'Each row normalises over only the tokens it is allowed to see.' },
    { t: 'Autoregressive', b: 'This backward-only view is exactly what lets the model generate one token at a time.' },
  ],
  swa: [
    { t: 'Begin from causal', b: 'Start with ordinary causal attention — the lower triangle.' },
    { t: 'Add a local window', b: 'Also forbid tokens further back than the window, so only a band near the diagonal survives.' },
    { t: 'Softmax the band', b: 'Each query attends within its small recent window.' },
    { t: 'Smaller KV cache', b: 'Only window-sized keys/values are kept — cheap memory for very long context.' },
  ],
  cross: [
    { t: 'Two different sequences', b: 'Rows are decoder (target) tokens; columns are the encoder (source) sentence.' },
    { t: 'Score across them', b: 'Each decoder query is matched against every encoded source token.' },
    { t: 'Softmax each row', b: 'Turn those into weights — which source words matter for this target word.' },
    { t: 'Condition on the input', b: 'The decoder pulls in the relevant source meaning. This is the bridge in translation models.' },
  ],
  gqa: [
    { t: 'Many query heads', b: 'Start with 8 query heads, each one wanting its own keys & values.' },
    { t: 'Group the heads', b: 'Split the query heads into groups (here, 4 groups of 2).' },
    { t: 'Share K/V per group', b: 'Every head in a group reads the SAME key/value head — so we store far fewer K·V.' },
    { t: 'MHA → GQA → MQA', b: 'Use the buttons: 8 KV heads is plain MHA, 1 is multi-query (MQA), in between is grouped-query.' },
  ],
  mla: [
    { t: 'Keys & values are big', b: 'Normally every token caches full-size K and V vectors — expensive memory.' },
    { t: 'Compress to a latent', b: 'MLA squeezes K·V down into one small latent vector.' },
    { t: 'Cache only the latent', b: 'Just that tiny latent is stored in the KV cache — the big saving.' },
    { t: 'Expand on use', b: 'At attention time it is projected back up to full K·V. The smallest cache of all.' },
  ],
  linear: [
    { t: 'No n×n matrix', b: 'Linear attention never forms the full attention grid at all.' },
    { t: 'Keep a small state', b: 'Maintain a tiny fast-weight memory instead.' },
    { t: 'Update per token', b: 'As each token arrives, the state is updated with a gated delta rule.' },
    { t: 'Read in O(n)', b: 'The query reads the state — linear time, no growing cache. Mixed 3:1 with full attention.' },
  ],
  mamba: [
    { t: 'A rolling state', b: 'A selective state-space model keeps one compact state hₜ.' },
    { t: 'Absorb each token', b: 'Every token updates the state through a learned recurrence.' },
    { t: 'Carry it forward', b: 'Information flows along the sequence inside the state — a yellow pulse here.' },
    { t: 'O(n) cost', b: 'No token-to-token matrix, so memory and compute stay linear in length.' },
  ],
}

const INFO = {
  walk: { t: 'How attention works', d: 'The core scaled dot-product attention every transformer uses.', pros: [], cons: [], used: 'Every transformer' },
  mha: { t: 'Multi-Head Attention', d: 'Full attention — every token attends to every token; each head independent.', pros: ['Best modeling quality', 'Well-optimised'], cons: ['Largest KV cache', 'O(n²)'], used: 'GPT-2, OLMo 2' },
  causal: { t: 'Causal (Masked) Attention', d: 'A look-ahead mask makes the model autoregressive.', pros: ['Enables generation'], cons: ['Backward-only', 'O(n²)'], used: 'Every decoder LLM' },
  swa: { t: 'Sliding-Window Attention', d: 'Local attention within a moving window.', pros: ['Smaller KV cache', 'Near-global quality'], cons: ['Local view only'], used: 'Gemma 3, GPT-OSS' },
  cross: { t: 'Cross-Attention', d: 'Decoder queries attend to encoder keys/values.', pros: ['Conditions on a full input'], cons: ['Encoder–decoder only'], used: 'T5, BART' },
  gqa: { t: 'Grouped-Query Attention', d: 'Query heads share a smaller number of KV heads.', pros: ['Smaller KV cache', 'Faster inference'], cons: ['Slightly below MHA'], used: 'Llama, Qwen3, Mistral' },
  mla: { t: 'Multi-Head Latent Attention', d: 'Compress K·V into a cached latent, expand on use.', pros: ['Smallest KV cache', 'Beats MHA quality'], cons: ['More complex'], used: 'DeepSeek V3, Kimi K2' },
  linear: { t: 'Linear Attention (Gated DeltaNet)', d: 'A fast-weight recurrence instead of the n×n matrix.', pros: ['O(n) time & memory', 'Cheap long context'], cons: ['Less precise retrieval'], used: 'Qwen3-Next, Kimi Linear' },
  mamba: { t: 'Mamba (State-Space)', d: 'A selective state-space recurrence over tokens.', pros: ['O(n) memory & compute'], cons: ['Fixed-size state limits recall'], used: 'Mamba, Jamba' },
}

const MODES = [
  { id: 'walk', label: '▶ How attention works', kind: 'walk' },
  { id: 'mha', label: 'MHA · full', kind: 'heat', opts: {} },
  { id: 'causal', label: 'Causal (GPT)', kind: 'heat', opts: { causal: true } },
  { id: 'swa', label: 'Sliding-window', kind: 'heat', opts: { causal: true, window: 3 } },
  { id: 'cross', label: 'Cross-attention', kind: 'heat', opts: { cross: true } },
  { id: 'gqa', label: 'Grouped-Query', kind: 'gqa' },
  { id: 'mla', label: 'Latent (MLA)', kind: 'mla' },
  { id: 'linear', label: 'Linear / DeltaNet', kind: 'recur' },
  { id: 'mamba', label: 'Mamba (SSM)', kind: 'recur', mamba: true },
]

const CAM = {
  walk: { position: [0, 0.3, 9.5], target: [0, -0.3, 0] },
  heat: { position: [0, 6.5, 8.5], target: [0, 0.4, 0] },
  gqa: { position: [0, 0.4, 9], target: [0, 0, 0] },
  mla: { position: [0, 0.3, 8.8], target: [0, 0, 0] },
  recur: { position: [0, 1.2, 9], target: [0, 0.1, 0] },
}

const DUR = 3.4 // seconds per step for the mechanism walkthroughs

// caption card shared by all modes
function CaptionCard({ stepNo, total, title, body, idx, playing, onToggle }) {
  return (
    <div className="overlay" style={{ bottom: 18, left: '50%', transform: 'translateX(-50%)', pointerEvents: 'auto', width: 'min(560px, 84%)' }}>
      <div className="explainer-card">
        <div className="ex-top">
          <span className="ex-step">Step {stepNo} / {total}</span>
          <span className="ex-title">{title}</span>
          <button className="ex-play" onClick={onToggle}>{playing ? '❙❙' : '▶'}</button>
        </div>
        <div className="ex-body">{body}</div>
        <div className="ex-track">
          {Array.from({ length: total }).map((_, i) => (
            <span key={i} className={'ex-dot' + (i === idx ? ' on' : '')} />
          ))}
        </div>
      </div>
    </div>
  )
}

export default function AttentionLab() {
  const [modeId, setModeId] = useState('walk')
  const [head, setHead] = useState(0)
  const [kvHeads, setKvHeads] = useState(2)
  const [playing, setPlaying] = useState(true)
  const [walk, setWalk] = useState({ q: 1, tok: 'cat', phase: 'query' })
  const [capStep, setCapStep] = useState(0)
  const [sheet, setSheet] = useState(false)
  const isMobile = useIsMobile()
  const mode = MODES.find((m) => m.id === modeId)
  const isHeat = mode.kind === 'heat'
  const showHeads = isHeat && !mode.opts.cross

  const rowToks = DEMO_TOKENS
  const colToks = isHeat && mode.opts.cross ? DEMO_SRC : DEMO_TOKENS

  const raw = useMemo(() => (isHeat ? rawMatrix(rowToks, colToks, head, mode.opts) : null), [isHeat, rowToks, colToks, head, mode])
  const final = useMemo(() => (isHeat ? attentionMatrix(rowToks, colToks, head, mode.opts) : null), [isHeat, rowToks, colToks, head, mode])

  // step clock for non-walk modes
  const tl = useRef({ step: 0, prog: 0 })
  const accRef = useRef(0)
  const lastStep = useRef(-1)
  const playingRef = useRef(true)
  useEffect(() => { playingRef.current = playing }, [playing])
  useEffect(() => {
    accRef.current = 0
    lastStep.current = -1
    tl.current = { step: 0, prog: 0 }
    setCapStep(0)
    if (modeId === 'walk') return
    const steps = (STEPS[modeId] || []).length || 1
    let raf
    let last = performance.now()
    const loop = (now) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      if (playingRef.current) accRef.current += dt
      const s = Math.floor(accRef.current / DUR) % steps
      tl.current.step = s
      tl.current.prog = (accRef.current / DUR) % 1
      if (s !== lastStep.current) { lastStep.current = s; setCapStep(s) }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [modeId])

  const info = INFO[modeId]
  const cam = CAM[mode.kind]

  // caption content
  let cap
  if (modeId === 'walk') {
    const st = WALK_STEPS[walk.phase] || WALK_STEPS.query
    cap = { stepNo: st.n + 1, total: 4, title: st.title, body: st.body(walk.tok), idx: st.n }
  } else {
    const arr = STEPS[modeId] || []
    const c = arr[Math.min(capStep, arr.length - 1)] || { t: '', b: '' }
    cap = { stepNo: capStep + 1, total: arr.length, title: c.t, body: c.b, idx: capStep }
  }

  const canvasArea = (
      <div className="canvas-wrap">
        <Canvas key={mode.kind} camera={{ position: cam.position, fov: 45 }} dpr={[1, 2]}>
          <color attach="background" args={['#0a0e16']} />
          <ambientLight intensity={0.6} />
          <directionalLight position={[5, 9, 5]} intensity={1.1} />
          <Environment preset="city" />
          {mode.kind === 'walk' && <Walkthrough onState={setWalk} playing={playing} />}
          {isHeat && <HeatWalk rowToks={rowToks} colToks={colToks} raw={raw} final={final} views={VIEWS[modeId]} opts={mode.opts} tl={tl} />}
          {mode.kind === 'gqa' && <GQAWalk qHeads={8} kvHeads={kvHeads} tl={tl} />}
          {mode.kind === 'mla' && <MLAWalk tl={tl} />}
          {mode.kind === 'recur' && <RecurWalk mamba={!!mode.mamba} tl={tl} />}
          <OrbitControls makeDefault target={cam.target} minDistance={5} maxDistance={22} />
        </Canvas>

        <div className="overlay tl" style={{ pointerEvents: 'auto' }}>
          <div className="legend">
            <div className="attn-controls">
              {MODES.map((m) => (
                <span key={m.id} className={'pill' + (modeId === m.id ? ' active' : '')} onClick={() => setModeId(m.id)}>{m.label}</span>
              ))}
            </div>
            {showHeads && (
              <div className="attn-controls" style={{ marginTop: 8 }}>
                {HEADS.map((h, i) => (
                  <span key={i} className={'pill' + (head === i ? ' active' : '')} onClick={() => setHead(i)}>{h.name}</span>
                ))}
              </div>
            )}
            {mode.kind === 'gqa' && (
              <div className="attn-controls" style={{ marginTop: 8 }}>
                {[{ k: 8, l: 'MHA 8/8' }, { k: 4, l: 'GQA 8/4' }, { k: 2, l: 'GQA 8/2' }, { k: 1, l: 'MQA 8/1' }].map((o) => (
                  <span key={o.k} className={'pill' + (kvHeads === o.k ? ' active' : '')} onClick={() => setKvHeads(o.k)}>{o.l}</span>
                ))}
              </div>
            )}
          </div>
        </div>

        <CaptionCard stepNo={cap.stepNo} total={cap.total} title={cap.title} body={cap.body} idx={cap.idx} playing={playing} onToggle={() => setPlaying((p) => !p)} />
      </div>
  )

  const sideContent = (
      <div className="side scrolly">
        <h3>{info.t}</h3>
        <div className="explain">{info.d}</div>
        {showHeads && <div className="explain"><b>{HEADS[head].name} head.</b> {HEADS[head].hint}</div>}

        <h3>The steps</h3>
        <ul className="zlist">
          {(modeId === 'walk'
            ? [['Pick a query', 'the token looking around'], ['Score the keys', 'dot product'], ['Softmax', 'weights sum to 1'], ['Blend values', 'context-aware output']]
            : (STEPS[modeId] || []).map((s) => [s.t, ''])
          ).map(([t], i) => (
            <li key={i}><b>{i + 1}.</b> {t}</li>
          ))}
        </ul>

        {info.pros.length > 0 && (<><h3>Advantages</h3><ul className="zlist pros">{info.pros.map((p, i) => <li key={i}>{p}</li>)}</ul></>)}
        {info.cons.length > 0 && (<><h3>Disadvantages</h3><ul className="zlist cons">{info.cons.map((c, i) => <li key={i}>{c}</li>)}</ul></>)}
        <div className="explain" style={{ fontSize: 11.5 }}>Used in: {info.used}</div>
        <div style={{ height: 20 }} />
      </div>
  )

  if (isMobile) {
    return (
      <div className="m-view">
        <div className="m-canvas">{canvasArea}</div>
        <div className="m-tabbar">
          <button onClick={() => setSheet(true)}>ℹ︎ Explanation &amp; steps</button>
        </div>
        {sheet && (
          <MobileSheet title={info.t} onClose={() => setSheet(false)}>
            {sideContent}
          </MobileSheet>
        )}
      </div>
    )
  }

  return (
    <div className="workspace">
      {canvasArea}
      {sideContent}
    </div>
  )
}

import { useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Environment } from '@react-three/drei'
import Block3D from './Block3D.jsx'
import { defaultParams, blockColor, blockBadge } from '../lib/blocks.js'
import { BLOCK_H, STEP, yForIndex, sublabel, blockLabel } from '../lib/layout.js'

// A non-interactive, slowly auto-rotating 3D tower built from a preset spec.
export default function MiniTower({ spec, autoRotate = true, distance = 8 }) {
  const stack = useMemo(
    () => spec.map((s, i) => ({ uid: i, key: s.key, params: { ...defaultParams(s.key), ...s.params } })),
    [spec]
  )
  const n = stack.length
  const scale = Math.min(1, 6.4 / (n * STEP))

  return (
    <Canvas camera={{ position: [5.5, 0, distance], fov: 42 }} dpr={[1, 2]}>
      <color attach="background" args={['#0b101a']} />
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 8, 6]} intensity={1.05} />
      <Environment preset="city" />
      <group scale={scale}>
        {stack.map((b, i) => (
          <Block3D
            key={b.uid}
            position={[0, yForIndex(i, n), 0]}
            size={[3, BLOCK_H, 1.5]}
            color={blockColor(b.key)}
            label={blockLabel(b)}
            sublabel={sublabel(b)}
            badge={blockBadge(b.key)}
          />
        ))}
      </group>
      <OrbitControls autoRotate={autoRotate} autoRotateSpeed={0.8} enableZoom enablePan={false} minDistance={5} maxDistance={18} />
    </Canvas>
  )
}

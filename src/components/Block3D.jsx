import { useRef, useState, useMemo } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { RoundedBox, Text } from '@react-three/drei'

// A single module in the tower — styled to read like a real hardware card:
// glossy faceplate, a dark label plate, a category badge chip and connector pins.
export default function Block3D({
  position,
  size = [3, 0.62, 1.5],
  color = '#64748b',
  label = '',
  sublabel = '',
  badge = '',
  selected = false,
  active = false,
  dimmed = false,
  onSelect,
  onPointerDown,
}) {
  const [hover, setHover] = useState(false)
  const group = useRef()
  const accent = useRef()
  const [w, h, d] = size
  const opacity = dimmed ? 0.3 : 1
  const front = d / 2 + 0.001

  const edges = useMemo(() => new THREE.BoxGeometry(w + 0.05, h + 0.06, d + 0.05), [w, h, d])
  const pinZ = useMemo(() => [-d * 0.3, 0, d * 0.3], [d])
  const baseColor = useMemo(() => new THREE.Color(color), [color])
  const darkBody = useMemo(() => baseColor.clone().multiplyScalar(0.5), [baseColor])

  useFrame((state) => {
    const t = state.clock.elapsedTime
    if (accent.current) {
      const lift = active ? 1.5 : selected ? 0.6 + Math.sin(t * 3) * 0.3 : hover ? 0.5 : 0.25
      accent.current.material.emissiveIntensity = lift
    }
    if (group.current) {
      const tz = active ? 0.32 : selected ? 0.12 : 0
      group.current.position.z += (tz - group.current.position.z) * 0.2
    }
  })

  return (
    <group position={position}>
      <group ref={group}>
        {/* main body */}
        <RoundedBox
          args={[w, h, d]}
          radius={0.07}
          smoothness={4}
          onPointerOver={(e) => {
            e.stopPropagation()
            setHover(true)
            document.body.style.cursor = 'grab'
          }}
          onPointerOut={() => {
            setHover(false)
            document.body.style.cursor = 'auto'
          }}
          onPointerDown={(e) => {
            e.stopPropagation()
            onPointerDown && onPointerDown(e)
          }}
          onClick={(e) => {
            e.stopPropagation()
            onSelect && onSelect()
          }}
        >
          <meshStandardMaterial
            color={darkBody}
            metalness={0.6}
            roughness={0.3}
            transparent
            opacity={opacity}
          />
        </RoundedBox>

        {/* glossy coloured faceplate on the front */}
        <RoundedBox args={[w - 0.12, h - 0.12, 0.06]} radius={0.05} smoothness={4} position={[0, 0, front]}>
          <meshStandardMaterial color={color} metalness={0.4} roughness={0.25} transparent opacity={opacity} />
        </RoundedBox>

        {/* dark label plate for legible text */}
        <RoundedBox args={[w - 0.5, h - 0.26, 0.04]} radius={0.04} smoothness={4} position={[0.18, 0, front + 0.04]}>
          <meshStandardMaterial color="#0b1220" metalness={0.2} roughness={0.6} transparent opacity={opacity} />
        </RoundedBox>

        {/* category badge chip (left) */}
        <RoundedBox ref={accent} args={[0.34, h - 0.26, 0.05]} radius={0.04} smoothness={4} position={[-w / 2 + 0.3, 0, front + 0.04]}>
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.3} metalness={0.3} roughness={0.4} transparent opacity={opacity} />
        </RoundedBox>
        <Text position={[-w / 2 + 0.3, 0, front + 0.08]} fontSize={0.11} color="#05121e" anchorX="center" anchorY="middle" fontWeight="bold">
          {badge}
        </Text>

        {/* connector pins on both sides */}
        {pinZ.map((z, i) => (
          <group key={i}>
            <mesh position={[-w / 2 - 0.06, 0, z]} rotation={[0, 0, Math.PI / 2]}>
              <cylinderGeometry args={[0.04, 0.04, 0.14, 8]} />
              <meshStandardMaterial color="#cbd5e1" metalness={0.9} roughness={0.3} />
            </mesh>
            <mesh position={[w / 2 + 0.06, 0, z]} rotation={[0, 0, Math.PI / 2]}>
              <cylinderGeometry args={[0.04, 0.04, 0.14, 8]} />
              <meshStandardMaterial color="#cbd5e1" metalness={0.9} roughness={0.3} />
            </mesh>
          </group>
        ))}

        {/* labels */}
        <Text
          position={[0.16, 0, front + 0.07]}
          fontSize={0.18}
          maxWidth={w - 0.9}
          textAlign="center"
          anchorX="center"
          anchorY="middle"
          color="#eaf2fb"
        >
          {label}
        </Text>
        {sublabel ? (
          <Text position={[w / 2 - 0.16, -h / 2 + 0.12, front + 0.07]} fontSize={0.1} anchorX="right" anchorY="middle" color="#93c5fd">
            {sublabel}
          </Text>
        ) : null}

        {(selected || hover || active) && (
          <lineSegments>
            <edgesGeometry args={[edges]} />
            <lineBasicMaterial color={active ? '#bae6fd' : selected ? '#ffffff' : color} transparent opacity={0.95} />
          </lineSegments>
        )}
      </group>
    </group>
  )
}

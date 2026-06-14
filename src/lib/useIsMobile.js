import { useEffect, useState } from 'react'

// True when the viewport is phone-sized. Re-evaluates on resize/orientation.
export function useIsMobile(breakpoint = 820) {
  const query = `(max-width: ${breakpoint}px)`
  const get = () => typeof window !== 'undefined' && window.matchMedia(query).matches
  const [mobile, setMobile] = useState(get)
  useEffect(() => {
    const mq = window.matchMedia(query)
    const handler = () => setMobile(mq.matches)
    handler()
    mq.addEventListener('change', handler)
    return () => mq.removeEventListener('change', handler)
  }, [query])
  return mobile
}

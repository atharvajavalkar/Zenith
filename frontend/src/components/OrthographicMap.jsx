/**
 * Smooth orthographic world map (Canvas) — no WebGL.
 * Black oceans, grey land, gold outlines. Loads from local GeoJSON.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { geoDistance, geoGraticule10, geoOrthographic, geoPath } from 'd3-geo'

const ISO_TO_HUB = {
  US: 'US',
  CN: 'CN',
  IN: 'IN',
  AU: 'AU',
  ZA: 'ZA',
  RU: 'RU',
  CH: 'CH',
  GB: 'GB',
  JP: 'JP',
  CA: 'CA',
  TR: 'TR',
  AE: 'AE',
  BR: 'BR',
  NZ: 'NZ',
  IR: 'IR',
  SA: 'SA',
  GH: 'GH',
  PE: 'PE',
  MX: 'MX',
  KZ: 'KZ',
  UZ: 'UZ',
  SG: 'SG',
  HK: 'HK',
  PL: 'PL',
  KR: 'KR',
  TH: 'TH',
  VN: 'VN',
  EG: 'EG',
  // Eurozone members → EU desk
  DE: 'EU',
  FR: 'EU',
  IT: 'EU',
  ES: 'EU',
  NL: 'EU',
  BE: 'EU',
  AT: 'EU',
  IE: 'EU',
  PT: 'EU',
  FI: 'EU',
  GR: 'EU',
  SK: 'EU',
  SI: 'EU',
  EE: 'EU',
  LV: 'EU',
  LT: 'EU',
  LU: 'EU',
  MT: 'EU',
  CY: 'EU',
}

const OCEAN = '#05070c'
const LAND = '#2e3440'
const LAND_HUB = '#3d4554'
const LAND_HOVER = '#5a6270'
const STROKE = 'rgba(243,185,74,0.32)'
const STROKE_HUB = '#f3b94a'
const GRATICULE = 'rgba(243,185,74,0.07)'
const SPHERE_EDGE = 'rgba(243,185,74,0.45)'

function featureIso(f) {
  return String(f.properties?.ISO_A2_EH || f.properties?.ISO_A2 || '').toUpperCase()
}

export default function OrthographicMap({ markers = [], onHover }) {
  const wrapRef = useRef(null)
  const canvasRef = useRef(null)
  const featuresRef = useRef([])
  const markersRef = useRef(markers)
  const hoverRef = useRef(null)
  const rotRef = useRef({ lambda: -20, phi: -12 })
  const draggingRef = useRef(null)
  const pointerRef = useRef({ x: 0, y: 0 })
  const [status, setStatus] = useState('loading') // loading | ready | error
  const [size, setSize] = useState({ w: 800, h: 520 })

  markersRef.current = markers

  const markersByHub = useMemo(() => {
    const m = {}
    for (const item of markers) m[item.id] = item
    return m
  }, [markers])

  // Load local countries once
  useEffect(() => {
    let cancelled = false
    fetch('/geo/countries-110m.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((geo) => {
        if (cancelled) return
        featuresRef.current = geo.features || []
        setStatus('ready')
      })
      .catch((err) => {
        console.error('World map GeoJSON failed', err)
        if (!cancelled) setStatus('error')
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Resize
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return undefined
    const measure = () => {
      const r = el.getBoundingClientRect()
      setSize({ w: Math.max(320, Math.floor(r.width)), h: Math.max(420, Math.floor(r.height)) })
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Draw + animate
  useEffect(() => {
    if (status !== 'ready') return undefined
    const canvas = canvasRef.current
    if (!canvas) return undefined
    const ctx = canvas.getContext('2d')
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    let raf = 0
    let alive = true

    const paint = () => {
      if (!alive) return
      const { w, h } = size
      const cw = Math.floor(w * dpr)
      const ch = Math.floor(h * dpr)
      if (canvas.width !== cw || canvas.height !== ch) {
        canvas.width = cw
        canvas.height = ch
        canvas.style.width = `${w}px`
        canvas.style.height = `${h}px`
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)
      ctx.fillStyle = OCEAN
      ctx.fillRect(0, 0, w, h)

      const radius = Math.min(w, h) * 0.42
      const projection = geoOrthographic()
        .translate([w / 2, h / 2 + 12])
        .scale(radius)
        .rotate([rotRef.current.lambda, rotRef.current.phi, 0])
        .clipAngle(90)

      const path = geoPath(projection, ctx)
      const hoverIso = hoverRef.current

      // Soft sphere disc
      ctx.beginPath()
      ctx.arc(w / 2, h / 2 + 12, radius, 0, Math.PI * 2)
      ctx.fillStyle = '#080b12'
      ctx.fill()
      ctx.strokeStyle = SPHERE_EDGE
      ctx.lineWidth = 1.25
      ctx.stroke()

      // Graticule
      ctx.beginPath()
      path(geoGraticule10())
      ctx.strokeStyle = GRATICULE
      ctx.lineWidth = 0.6
      ctx.stroke()

      const features = featuresRef.current
      for (const f of features) {
        const iso = featureIso(f)
        const hub = ISO_TO_HUB[iso]
        const isHover = hoverIso && iso === hoverIso
        ctx.beginPath()
        path(f)
        ctx.fillStyle = isHover ? LAND_HOVER : hub ? LAND_HUB : LAND
        ctx.fill()
        ctx.strokeStyle = hub || isHover ? STROKE_HUB : STROKE
        ctx.lineWidth = hub || isHover ? 1.1 : 0.55
        ctx.stroke()
      }

      // Hub glow pins (front hemisphere only)
      const centerLonLat = [-rotRef.current.lambda, -rotRef.current.phi]
      for (const m of markersRef.current) {
        if (m.lat == null || m.lng == null) continue
        if (geoDistance(centerLonLat, [m.lng, m.lat]) > Math.PI / 2 - 0.02) continue
        const xy = projection([m.lng, m.lat])
        if (!xy) continue

        const [x, y] = xy
        const g = ctx.createRadialGradient(x, y, 0, x, y, 10)
        g.addColorStop(0, 'rgba(243,185,74,0.95)')
        g.addColorStop(0.35, 'rgba(243,185,74,0.45)')
        g.addColorStop(1, 'rgba(243,185,74,0)')
        ctx.fillStyle = g
        ctx.beginPath()
        ctx.arc(x, y, 10, 0, Math.PI * 2)
        ctx.fill()
        ctx.beginPath()
        ctx.arc(x, y, 2.4, 0, Math.PI * 2)
        ctx.fillStyle = '#ffe6a3'
        ctx.fill()
      }

      // Auto-rotate when not dragging / hovering
      if (!draggingRef.current && !hoverRef.current) {
        rotRef.current.lambda -= 0.12
      }

      raf = requestAnimationFrame(paint)
    }

    raf = requestAnimationFrame(paint)
    return () => {
      alive = false
      cancelAnimationFrame(raf)
    }
  }, [status, size])

  const hitTest = (clientX, clientY) => {
    const canvas = canvasRef.current
    if (!canvas || status !== 'ready') return null
    const rect = canvas.getBoundingClientRect()
    const x = clientX - rect.left
    const y = clientY - rect.top
    pointerRef.current = { x, y }

    const { w, h } = size
    const radius = Math.min(w, h) * 0.42
    const projection = geoOrthographic()
      .translate([w / 2, h / 2 + 12])
      .scale(radius)
      .rotate([rotRef.current.lambda, rotRef.current.phi, 0])
      .clipAngle(90)
    const path = geoPath(projection)

    let found = null
    for (const f of featuresRef.current) {
      if (path.centroid(f).some((n) => Number.isNaN(n))) continue
      try {
        if (path.contains && path.contains(f, [x, y])) {
          found = f
          break
        }
      } catch {
        /* ignore */
      }
    }

    // Fallback: nearest hub pin within 18px
    if (!found) {
      let best = null
      let bestD = 18
      for (const m of markersRef.current) {
        const xy = projection([m.lng, m.lat])
        if (!xy) continue
        const d = Math.hypot(xy[0] - x, xy[1] - y)
        if (d < bestD) {
          bestD = d
          best = m
        }
      }
      if (best) return { type: 'hub', hub: best.id, name: best.name, iso: null }
      return null
    }

    const iso = featureIso(found)
    const hub = ISO_TO_HUB[iso] || null
    return {
      type: 'country',
      hub,
      iso,
      name: found.properties?.NAME || found.properties?.ADMIN || iso,
    }
  }

  const emitHover = (hit) => {
    const anchor = {
      x: pointerRef.current.x,
      y: pointerRef.current.y,
    }
    if (!hit || !hit.hub) {
      hoverRef.current = hit?.iso || null
      onHover?.(null, anchor)
      return
    }
    hoverRef.current = hit.iso
    const base = markersByHub[hit.hub]
    onHover?.(
      base || {
        id: hit.hub,
        name: hit.name,
        role: 'Gold-relevant market hub',
        count: 0,
        headlines: [],
      },
      anchor,
    )
  }

  return (
    <div
      ref={wrapRef}
      className="relative w-full h-[min(74vh,740px)] min-h-[460px] overflow-hidden bg-[#05070c] cursor-grab active:cursor-grabbing"
      onMouseMove={(e) => {
        if (draggingRef.current) {
          const dx = e.clientX - draggingRef.current.x
          const dy = e.clientY - draggingRef.current.y
          draggingRef.current = { x: e.clientX, y: e.clientY }
          rotRef.current.lambda += dx * 0.35
          rotRef.current.phi = Math.max(-45, Math.min(45, rotRef.current.phi - dy * 0.25))
          return
        }
        const hit = hitTest(e.clientX, e.clientY)
        emitHover(hit)
      }}
      onMouseLeave={() => {
        draggingRef.current = null
        hoverRef.current = null
        onHover?.(null, pointerRef.current)
      }}
      onMouseDown={(e) => {
        draggingRef.current = { x: e.clientX, y: e.clientY }
      }}
      onMouseUp={() => {
        draggingRef.current = null
      }}
    >
      {status === 'loading' && (
        <div className="absolute inset-0 z-10 flex items-center justify-center text-sm text-surface-muted">
          Drawing world map…
        </div>
      )}
      {status === 'error' && (
        <div className="absolute inset-0 z-10 flex items-center justify-center text-sm text-red-300 px-6 text-center">
          Could not load country outlines. Check /geo/countries-110m.json
        </div>
      )}
      <canvas ref={canvasRef} className="block w-full h-full" />
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse 60% 52% at 50% 48%, transparent 0%, rgba(5,7,12,0.15) 55%, rgba(5,7,12,0.75) 100%)',
        }}
      />
    </div>
  )
}

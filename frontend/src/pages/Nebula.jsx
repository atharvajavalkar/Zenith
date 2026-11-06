import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Layout from '../components/Layout'
import api from '../services/api'

const PAIR_COLORS = {
  ETHUSD: '#5ec8a8',
  BTCUSD: '#e2b15a',
  GOLD: '#f0d78c',
  NAS100: '#7eb6ff',
  USDJPY: '#c9a66b',
  AUDUSD: '#8fd4a8',
  USDCHF: '#d4a08a',
  UKOIL: '#b8c4a0',
}

const FAMILY_COLORS = {
  baseline: '#f2eee6',
  session_mix: '#c9a66b',
  circadian: '#7eb6ff',
  entropy: '#5ec8a8',
  snr: '#8fd4a8',
  volmom: '#d4a08a',
  fourier: '#a8b89a',
  ou_fade: '#e2b15a',
  vol_clock: '#9ad0e0',
  weekday: '#c4b59a',
  session_book: '#a8c49a',
  edgegate: '#dcc08a',
  liquid_hours: '#9a948a',
  omni: '#e8c47c',
  agree: '#9ec4e8',
  hurst: '#b8a0d4',
  off_flat: '#a09080',
  other: '#6a655c',
}

function fmtHours(sec) {
  if (sec == null || Number.isNaN(Number(sec))) return '—'
  const h = Number(sec) / 3600
  if (h < 1) return `${Math.round(h * 60)}m`
  return `${h.toFixed(1)}h`
}

function fmtPnl(v) {
  if (v == null || Number.isNaN(Number(v))) return '—'
  const n = Math.round(Number(v))
  return `${n >= 0 ? '+' : ''}$${n.toLocaleString()}`
}

function verdictTone(v) {
  if (v === 'PASS') return 'text-accent-buy'
  if (v === 'SOFT') return 'text-accent-gold'
  if (v === 'FAIL') return 'text-accent-sell'
  return 'text-surface-muted'
}

function hexRgb(hex) {
  const h = String(hex || '#c9a66b').replace('#', '')
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]
}

function HorizonBars({ horizons }) {
  const entries = Object.entries(horizons || {}).filter(([, v]) => v != null)
  if (!entries.length) return null
  const maxAbs = Math.max(1, ...entries.map(([, v]) => Math.abs(Number(v) || 0)))
  return (
    <div className="space-y-1">
      {entries.map(([k, v]) => {
        const n = Number(v) || 0
        const pct = (Math.abs(n) / maxAbs) * 100
        return (
          <div key={k} className="grid grid-cols-[36px_1fr_52px] items-center gap-1.5 text-[10px]">
            <span className="text-surface-muted">{k}</span>
            <div className="h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
              <div
                className={`h-full rounded-full ${n >= 0 ? 'bg-accent-buy/70' : 'bg-accent-sell/70'}`}
                style={{ width: `${Math.max(4, pct)}%` }}
              />
            </div>
            <span className={`tabular-nums text-right ${n >= 0 ? 'text-accent-buy' : 'text-accent-sell'}`}>
              {fmtPnl(n)}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function LatticeGrid({ lattice, highlight, onPick }) {
  const cells = lattice || []
  if (!cells.length) return <div className="text-[11px] text-surface-muted">No lattice for this TF</div>
  return (
    <div className="grid grid-cols-3 gap-1 sm:grid-cols-4">
      {cells.map((c) => {
        const id = `sl${c.sl_atr}|rr${c.rr}|g${c.min_gap}`
        const on =
          highlight &&
          highlight.sl_atr === c.sl_atr &&
          highlight.rr === c.rr &&
          highlight.min_gap === c.min_gap
        return (
          <button
            key={id}
            type="button"
            onClick={() => onPick?.(c)}
            className={`rounded-sm border px-1.5 py-1.5 text-left text-[9px] leading-tight transition-colors ${
              on
                ? 'border-accent-gold/50 bg-accent-gold/15 text-accent-gold'
                : 'border-white/[0.06] bg-white/[0.02] text-surface-muted hover:border-white/15 hover:text-gray-200'
            }`}
          >
            <div>SL {c.sl_atr}</div>
            <div>RR {c.rr}</div>
            <div>gap {c.min_gap}</div>
          </button>
        )
      })}
    </div>
  )
}

/**
 * Interactive deep nebula — click pairs / families / modes; heat-adaptive hubs;
 * follow-hunt zoom; hit-tested hover tooltips.
 */
function DeepNebulaCanvas({
  nodes,
  search,
  activePair,
  activeTf,
  pct,
  status,
  focus,
  followHunt,
  onFocus,
  onHover,
}) {
  const canvasRef = useRef(null)
  const liveRef = useRef({})
  const hitsRef = useRef([])
  const camRef = useRef({ x: 0, y: 0, z: 1 })

  useEffect(() => {
    liveRef.current = { nodes, search, activePair, activeTf, pct, status, focus, followHunt }
  }, [nodes, search, activePair, activeTf, pct, status, focus, followHunt])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return undefined
    const ctx = canvas.getContext('2d', { alpha: true })
    let raf = 0
    let running = true
    let t = 0
    const dpr = Math.min(2, window.devicePixelRatio || 1)

    const dust = Array.from({ length: 300 }, () => ({
      a: Math.random() * Math.PI * 2,
      r: 0.12 + Math.random() * 0.62,
      s: 0.12 + Math.random() * 0.5,
      size: 0.5 + Math.random() * 2.0,
      hue: Math.random(),
    }))

    const resize = () => {
      const parent = canvas.parentElement
      const w = parent?.clientWidth || window.innerWidth
      const h = parent?.clientHeight || Math.max(520, window.innerHeight * 0.72)
      canvas.width = Math.floor(w * dpr)
      canvas.height = Math.floor(h * dpr)
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      canvas._wh = { w, h }
    }
    resize()
    window.addEventListener('resize', resize)

    const toLocal = (e) => {
      const rect = canvas.getBoundingClientRect()
      return { x: e.clientX - rect.left, y: e.clientY - rect.top }
    }

    const hitTest = (x, y) => {
      const hits = hitsRef.current
      for (let i = hits.length - 1; i >= 0; i--) {
        const h = hits[i]
        const dx = x - h.x
        const dy = y - h.y
        if (dx * dx + dy * dy <= h.r * h.r) return h
      }
      return null
    }

    const onMove = (e) => {
      const p = toLocal(e)
      const hit = hitTest(p.x, p.y)
      canvas.style.cursor = hit ? 'pointer' : 'default'
      onHover?.(hit, p)
    }
    const onClick = (e) => {
      const p = toLocal(e)
      const hit = hitTest(p.x, p.y)
      if (hit) onFocus?.(hit)
      else onFocus?.(null)
    }
    canvas.addEventListener('mousemove', onMove)
    canvas.addEventListener('click', onClick)

    const tick = () => {
      if (!running) return
      t += 0.016
      const { w, h } = canvas._wh || { w: 800, h: 600 }
      const live = liveRef.current
      const symbols = Object.keys(PAIR_COLORS)
      const modes = live.search?.modes || []
      const families = live.search?.families || []
      const focus = live.focus || {}
      const hits = []

      // camera — ease toward focused family / active hunt
      let targetX = 0
      let targetY = 0
      let targetZ = 1
      const baseCx = w * 0.42
      const baseCy = h * 0.5
      if (focus.family && live._famPos?.[focus.family]) {
        const fp = live._famPos[focus.family]
        targetX = baseCx - fp.x
        targetY = baseCy - fp.y
        targetZ = 1.55
      } else if (live.followHunt && live.status === 'running' && live.search?.testing_now?.family) {
        const fam = live.search.testing_now.family
        if (live._famPos?.[fam]) {
          const fp = live._famPos[fam]
          targetX = (baseCx - fp.x) * 0.45
          targetY = (baseCy - fp.y) * 0.45
          targetZ = 1.25
        }
      }
      const cam = camRef.current
      cam.x += (targetX - cam.x) * 0.06
      cam.y += (targetY - cam.y) * 0.06
      cam.z += (targetZ - cam.z) * 0.06

      const cx = baseCx
      const cy = baseCy
      const world = (x, y) => ({
        x: cx + (x - cx + cam.x) * cam.z,
        y: cy + (y - cy + cam.y) * cam.z,
      })

      ctx.clearRect(0, 0, w, h)

      const wash = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(w, h) * 0.65)
      wash.addColorStop(0, 'rgba(32, 40, 38, 0.5)')
      wash.addColorStop(0.4, 'rgba(14, 16, 14, 0.35)')
      wash.addColorStop(1, 'rgba(8, 8, 7, 0)')
      ctx.fillStyle = wash
      ctx.fillRect(0, 0, w, h)

      for (let i = 0; i < 4; i++) {
        const bx = cx + Math.cos(t * 0.15 + i) * w * 0.08
        const by = cy + Math.sin(t * 0.12 + i * 1.3) * h * 0.06
        const rg = ctx.createRadialGradient(bx, by, 0, bx, by, Math.min(w, h) * (0.18 + i * 0.04))
        const a = 0.035 + Math.sin(t * 0.4 + i) * 0.012
        rg.addColorStop(0, `rgba(201,166,107,${a})`)
        rg.addColorStop(0.55, `rgba(94,200,168,${a * 0.4})`)
        rg.addColorStop(1, 'rgba(0,0,0,0)')
        ctx.fillStyle = rg
        ctx.fillRect(0, 0, w, h)
      }

      for (const d of dust) {
        const ang = d.a + t * d.s * 0.1
        const rr = Math.min(w, h) * d.r
        const raw = { x: cx + Math.cos(ang) * rr, y: cy + Math.sin(ang) * rr * 0.7 }
        const p = world(raw.x, raw.y)
        const [r, g, b] = d.hue > 0.55 ? [201, 166, 107] : [126, 182, 255]
        ctx.beginPath()
        ctx.fillStyle = `rgba(${r},${g},${b},${0.12 + d.hue * 0.3})`
        ctx.arc(p.x, p.y, d.size * cam.z, 0, Math.PI * 2)
        ctx.fill()
      }

      // family constellation
      const famList = families.length
        ? families
        : [...new Set(modes.map((m) => m.family))].map((id) => ({ id, total: 1, scanned: 0, heat: 0 }))
      const famPos = {}
      famList.forEach((f, i) => {
        const ang = (i / Math.max(famList.length, 1)) * Math.PI * 2 - Math.PI / 2 + t * 0.03
        const rad = Math.min(w, h) * 0.16
        famPos[f.id] = {
          x: cx + Math.cos(ang) * rad,
          y: cy + Math.sin(ang) * rad * 0.88,
          color: FAMILY_COLORS[f.id] || FAMILY_COLORS.other,
        }
      })
      live._famPos = famPos

      famList.forEach((f) => {
        const raw = famPos[f.id]
        if (!raw) return
        const p = world(raw.x, raw.y)
        const [r, g, b] = hexRgb(raw.color)
        const heat = Number(f.heat) || 0
        const fill = (f.scanned || 0) / Math.max(f.total || 1, 1)
        const focused = focus.family === f.id
        const hubR = (22 + heat * 18 + (focused ? 8 : 0)) * cam.z
        const hub = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, hubR)
        hub.addColorStop(0, `rgba(${r},${g},${b},${0.18 + fill * 0.25 + heat * 0.3})`)
        hub.addColorStop(1, 'rgba(0,0,0,0)')
        ctx.fillStyle = hub
        ctx.beginPath()
        ctx.arc(p.x, p.y, hubR, 0, Math.PI * 2)
        ctx.fill()
        if (focused || (live.followHunt && live.search?.testing_now?.family === f.id)) {
          ctx.beginPath()
          ctx.strokeStyle = `rgba(${r},${g},${b},0.55)`
          ctx.lineWidth = 1.5
          ctx.arc(p.x, p.y, hubR * 0.55 + Math.sin(t * 3) * 2, 0, Math.PI * 2)
          ctx.stroke()
        }
        ctx.font = `${focused ? 600 : 500} ${Math.round(9 * Math.min(cam.z, 1.3))}px Outfit, sans-serif`
        ctx.fillStyle = `rgba(${r},${g},${b},0.9)`
        ctx.textAlign = 'center'
        ctx.fillText(f.id, p.x, p.y + hubR * 0.7)
        if (f.wins) {
          ctx.font = '600 9px Outfit, sans-serif'
          ctx.fillStyle = 'rgba(242,238,230,0.75)'
          ctx.fillText(`${f.wins}W`, p.x, p.y + hubR * 0.7 + 12)
        }
        hits.push({
          type: 'family',
          id: f.id,
          family: f.id,
          x: p.x,
          y: p.y,
          r: hubR * 0.5,
          label: f.id,
          meta: f,
        })
      })

      const byFam = {}
      modes.forEach((m) => {
        ;(byFam[m.family] = byFam[m.family] || []).push(m)
      })
      Object.entries(byFam).forEach(([fam, list]) => {
        const hub = famPos[fam]
        if (!hub) return
        const dimOthers = focus.family && focus.family !== fam
        list.forEach((m, i) => {
          const ang = (i / Math.max(list.length, 1)) * Math.PI * 2 + t * 0.08
          const rad = 10 + (i % 7) * 3.2
          const raw = { x: hub.x + Math.cos(ang) * rad, y: hub.y + Math.sin(ang) * rad }
          const p = world(raw.x, raw.y)
          const [r, g, b] = hexRgb(hub.color)
          const isFocus = focus.mode === m.id
          let alpha = dimOthers ? 0.08 : 0.2
          let size = 1.2 * cam.z
          if (m.state === 'scanned') {
            alpha = dimOthers ? 0.15 : 0.55
            size = 1.6 * cam.z
          }
          if (m.state === 'active' || isFocus) {
            alpha = 0.95
            size = (3.2 + Math.sin(t * 6) * 0.8) * cam.z
            ctx.beginPath()
            ctx.strokeStyle = `rgba(${r},${g},${b},0.55)`
            ctx.lineWidth = 1
            ctx.arc(p.x, p.y, size + 5, 0, Math.PI * 2)
            ctx.stroke()
          }
          ctx.beginPath()
          ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`
          ctx.arc(p.x, p.y, size, 0, Math.PI * 2)
          ctx.fill()
          hits.push({
            type: 'mode',
            id: m.id,
            family: fam,
            mode: m.id,
            x: p.x,
            y: p.y,
            r: Math.max(6, size + 4),
            label: m.id,
            meta: m,
          })
        })
      })

      // TF ring
      const tfs = ['5m', '15m', '3m', '1m']
      const tfR = Math.min(w, h) * 0.28
      tfs.forEach((tf, i) => {
        const ang = (i / 4) * Math.PI * 2 - Math.PI / 2
        const raw = { x: cx + Math.cos(ang) * tfR, y: cy + Math.sin(ang) * tfR * 0.9 }
        const p = world(raw.x, raw.y)
        const active = tf === live.activeTf && live.status === 'running'
        const focused = focus.tf === tf
        ctx.beginPath()
        ctx.fillStyle =
          active || focused ? 'rgba(201,166,107,0.95)' : 'rgba(242,238,230,0.25)'
        ctx.arc(p.x, p.y, (active || focused ? 7 : 4.5) * cam.z, 0, Math.PI * 2)
        ctx.fill()
        ctx.font = '600 10px Outfit, sans-serif'
        ctx.fillStyle = active || focused ? 'rgba(201,166,107,0.95)' : 'rgba(154,148,138,0.8)'
        ctx.textAlign = 'center'
        ctx.fillText(tf, p.x, p.y + 16 * cam.z)
        hits.push({ type: 'tf', id: tf, tf, x: p.x, y: p.y, r: 14, label: tf })
      })

      // pairs
      const nodeMap = {}
      ;(live.nodes || []).forEach((n) => {
        nodeMap[n.symbol] = n
      })
      const pairR = Math.min(w, h) * 0.38
      const positions = symbols.map((sym, i) => {
        const ang = (i / symbols.length) * Math.PI * 2 - Math.PI / 2 + t * 0.025
        return { sym, x: cx + Math.cos(ang) * pairR, y: cy + Math.sin(ang) * pairR * 0.88 }
      })

      const active = positions.find((p) => p.sym === live.activePair)
      if (active && live.status === 'running') {
        const a = world(active.x, active.y)
        const c = world(cx, cy)
        ctx.beginPath()
        ctx.strokeStyle = 'rgba(201,166,107,0.2)'
        ctx.lineWidth = 1.5
        ctx.moveTo(a.x, a.y)
        ctx.lineTo(c.x, c.y)
        ctx.stroke()
      }

      positions.forEach((raw) => {
        const p = world(raw.x, raw.y)
        const node = nodeMap[raw.sym]
        const col = PAIR_COLORS[raw.sym]
        const [r, g, b] = hexRgb(col)
        const complete = node?.complete
        const doneTfs = node?.done_tfs || 0
        const isActive = raw.sym === live.activePair && live.status === 'running'
        const isFocus = focus.pair === raw.sym
        const radius =
          (complete ? 13 : isActive || isFocus ? 12 : 8) *
          (isActive || isFocus ? 1 + Math.sin(t * 4) * 0.1 : 1) *
          cam.z

        const halo = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, radius * 3)
        halo.addColorStop(
          0,
          `rgba(${r},${g},${b},${isActive || isFocus ? 0.42 : complete ? 0.25 : 0.1 + doneTfs * 0.04})`
        )
        halo.addColorStop(1, 'rgba(0,0,0,0)')
        ctx.fillStyle = halo
        ctx.beginPath()
        ctx.arc(p.x, p.y, radius * 3, 0, Math.PI * 2)
        ctx.fill()

        ctx.beginPath()
        ctx.fillStyle = `rgba(${r},${g},${b},${complete || isActive || isFocus ? 1 : 0.5})`
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2)
        ctx.fill()

        ctx.beginPath()
        ctx.strokeStyle = 'rgba(242,238,230,0.18)'
        ctx.lineWidth = 2
        ctx.arc(p.x, p.y, radius + 5, 0, Math.PI * 2)
        ctx.stroke()
        if (doneTfs > 0) {
          ctx.beginPath()
          ctx.strokeStyle = col
          ctx.lineWidth = 2.5
          ctx.arc(p.x, p.y, radius + 5, -Math.PI / 2, -Math.PI / 2 + (doneTfs / 4) * Math.PI * 2)
          ctx.stroke()
        }

        ctx.font = '600 10px Outfit, sans-serif'
        ctx.fillStyle = 'rgba(242,238,230,0.88)'
        ctx.textAlign = 'center'
        ctx.fillText(raw.sym, p.x, p.y + radius + 16)
        hits.push({
          type: 'pair',
          id: raw.sym,
          pair: raw.sym,
          x: p.x,
          y: p.y,
          r: radius + 8,
          label: raw.sym,
          meta: node,
        })
      })

      // center
      const c0 = world(cx, cy)
      ctx.font = '600 12px Outfit, sans-serif'
      ctx.fillStyle = 'rgba(201,166,107,0.92)'
      ctx.textAlign = 'center'
      ctx.fillText('DEEP HUNT', c0.x, c0.y - 10)
      ctx.font = '600 18px Outfit, sans-serif'
      ctx.fillStyle = 'rgba(242,238,230,0.95)'
      ctx.fillText(`${Number(live.pct || 0).toFixed(1)}%`, c0.x, c0.y + 12)
      const cur = live.search?.testing_now
      if (cur?.mode) {
        ctx.font = '500 9px Outfit, sans-serif'
        ctx.fillStyle = 'rgba(154,148,138,0.95)'
        const label = String(cur.mode).length > 28 ? `${String(cur.mode).slice(0, 26)}…` : cur.mode
        ctx.fillText(label, c0.x, c0.y + 28)
      }

      hitsRef.current = hits
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)

    return () => {
      running = false
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
      canvas.removeEventListener('mousemove', onMove)
      canvas.removeEventListener('click', onClick)
    }
  }, [onFocus, onHover])

  return <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
}

export default function NebulaPage() {
  const { data, isLoading, isError, dataUpdatedAt } = useQuery({
    queryKey: ['hunt-live-deep'],
    queryFn: async () => (await api.get('/api/hunt-live/deep')).data,
    refetchInterval: 2500,
  })

  const [flash, setFlash] = useState(null)
  const [followHunt, setFollowHunt] = useState(true)
  const [focus, setFocus] = useState({})
  const [hover, setHover] = useState(null)
  const [latticePick, setLatticePick] = useState(null)
  const [modeFilter, setModeFilter] = useState('all')
  const prevBest = useRef('')

  useEffect(() => {
    const events = data?.events || []
    const best = [...events].reverse().find((e) => e.kind === 'best')
    if (best && best.text !== prevBest.current) {
      prevBest.current = best.text
      setFlash(best.text)
      const t = setTimeout(() => setFlash(null), 7000)
      return () => clearTimeout(t)
    }
    return undefined
  }, [data?.events, dataUpdatedAt])

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') {
        setFocus({})
        setModeFilter('all')
        setLatticePick(null)
      }
      if (e.key === 'f' || e.key === 'F') setFollowHunt((v) => !v)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // Auto-follow hunt focus when enabled
  useEffect(() => {
    if (!followHunt || data?.status !== 'running') return
    const tn = data?.search?.testing_now
    if (!tn?.pair) return
    setFocus((prev) => ({
      ...prev,
      pair: tn.pair,
      tf: tn.tf,
      family: tn.family,
      mode: tn.mode,
      source: 'follow',
    }))
    if (tn.family) setModeFilter(tn.family)
  }, [
    followHunt,
    data?.status,
    data?.search?.testing_now?.pair,
    data?.search?.testing_now?.tf,
    data?.search?.testing_now?.mode,
    data?.search?.testing_now?.family,
  ])

  const onFocus = useCallback((hit) => {
    if (!hit) {
      setFocus({})
      return
    }
    setFollowHunt(false)
    if (hit.type === 'pair') {
      setFocus({ pair: hit.pair, source: 'click' })
    } else if (hit.type === 'family') {
      setFocus({ family: hit.family, source: 'click' })
      setModeFilter(hit.family)
    } else if (hit.type === 'mode') {
      setFocus({ family: hit.family, mode: hit.mode, source: 'click' })
      setModeFilter(hit.family)
    } else if (hit.type === 'tf') {
      setFocus((p) => ({ ...p, tf: hit.tf, source: 'click' }))
    }
  }, [])

  const onHover = useCallback((hit, pos) => {
    setHover(hit ? { ...hit, x: pos.x, y: pos.y } : null)
  }, [])

  const nodes = data?.nodes || []
  const search = data?.search || {}
  const status = data?.status || 'idle'
  const testing = search.testing_now || {}
  const adaptive = search.adaptive || {}
  const modes = search.modes || []
  const filteredModes =
    modeFilter === 'all' ? modes : modes.filter((m) => m.family === modeFilter)

  const focusPairNode = useMemo(
    () => nodes.find((n) => n.symbol === focus.pair),
    [nodes, focus.pair]
  )
  const focusTfBlock = useMemo(() => {
    if (!focusPairNode) return null
    const tf = focus.tf || data?.active_tf
    return (focusPairNode.tfs || []).find((t) => t.tf === tf) || null
  }, [focusPairNode, focus.tf, data?.active_tf])

  const focusModeMeta = useMemo(
    () => modes.find((m) => m.id === focus.mode) || search.current_mode,
    [modes, focus.mode, search.current_mode]
  )

  const winnerForFocus = useMemo(() => {
    const wins = adaptive.winners || []
    if (focus.pair && focus.tf) {
      return wins.find((w) => w.symbol === focus.pair && w.tf === focus.tf)
    }
    if (focus.mode) return wins.find((w) => w.mode === focus.mode)
    if (focus.family) return wins.find((w) => w.family === focus.family)
    return null
  }, [adaptive.winners, focus])

  const modePct = data?.modes > 0 ? Math.round((100 * (data.mode_i || 0)) / data.modes) : 0

  return (
    <Layout variant="chart">
      <div className="relative min-h-[calc(100vh-4.25rem)] w-full overflow-hidden bg-[#080807]">
        <div className="absolute inset-0 bottom-[min(42%,380px)] sm:bottom-0 sm:right-[400px]">
          <DeepNebulaCanvas
            nodes={nodes}
            search={search}
            activePair={data?.active_pair}
            activeTf={data?.active_tf}
            pct={data?.pct}
            status={status}
            focus={focus}
            followHunt={followHunt}
            onFocus={onFocus}
            onHover={onHover}
          />
          {hover && (
            <div
              className="pointer-events-none absolute z-20 max-w-[240px] rounded-sm border border-white/10 bg-[#0e0e0c]/92 px-2.5 py-2 text-[10px] text-gray-200 shadow-soft backdrop-blur-md"
              style={{
                left: Math.min(hover.x + 14, (typeof window !== 'undefined' ? window.innerWidth : 800) - 420),
                top: Math.max(8, hover.y - 10),
              }}
            >
              <div className="text-[9px] uppercase tracking-wider text-surface-muted">{hover.type}</div>
              <div className="font-medium text-accent-gold break-all">{hover.label}</div>
              {hover.type === 'family' && hover.meta && (
                <div className="mt-1 text-surface-muted">
                  heat {(hover.meta.heat * 100).toFixed(0)}% · {hover.meta.wins || 0} wins · scanned{' '}
                  {hover.meta.scanned}/{hover.meta.total}
                </div>
              )}
              {hover.type === 'pair' && hover.meta && (
                <div className="mt-1 text-surface-muted">{hover.meta.done_tfs}/4 TFs locked</div>
              )}
              {hover.type === 'mode' && hover.meta && (
                <div className="mt-1 text-surface-muted">
                  {hover.meta.family} · {hover.meta.state}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Top HUD */}
        <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex flex-wrap items-start justify-between gap-4 p-5 sm:p-6 sm:pr-[420px]">
          <div>
            <div className="section-label mb-1">Live · interactive adaptive search</div>
            <h1 className="font-display text-3xl sm:text-4xl text-gray-100 tracking-tight">
              Deep Nebula
            </h1>
            <p className="mt-2 max-w-lg text-sm text-surface-muted">
              {data?.hunt_label
                ? `${data.hunt_label}. `
                : null}
              Click a pair, family, or process star to drill in. Heat grows where winners emerge.
              <span className="text-gray-400"> Esc</span> clears · <span className="text-gray-400">F</span>{' '}
              toggles follow.
              {data?.live_untouched?.length ? (
                <span className="block mt-1 text-accent-buy/80">
                  Live desk stays on: {data.live_untouched.join(' · ')}
                </span>
              ) : null}
            </p>
            {adaptive.insight && (
              <div className="mt-3 max-w-lg rounded-sm border border-accent-gold/20 bg-accent-gold/5 px-3 py-2 text-[11px] text-accent-gold/90">
                {adaptive.insight}
              </div>
            )}
          </div>
          <div className="pointer-events-auto flex flex-col items-end gap-2">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setFollowHunt((v) => !v)}
                className={`rounded-sm border px-2.5 py-1 text-[10px] uppercase tracking-wider ${
                  followHunt
                    ? 'border-accent-buy/40 bg-accent-buy/10 text-accent-buy'
                    : 'border-white/10 text-surface-muted'
                }`}
              >
                {followHunt ? 'Following hunt' : 'Explore'}
              </button>
              <div
                className={`inline-flex items-center gap-2 rounded-sm border px-3 py-1.5 text-xs uppercase tracking-wider ${
                  status === 'running'
                    ? 'border-accent-buy/40 bg-accent-buy/10 text-accent-buy'
                    : status === 'done'
                      ? 'border-accent-gold/40 bg-accent-gold/10 text-accent-gold'
                      : 'border-white/10 text-surface-muted'
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    status === 'running' ? 'bg-accent-buy animate-pulse' : 'bg-surface-muted'
                  }`}
                />
                {status}
              </div>
            </div>
            <div className="text-right text-[11px] text-surface-muted tabular-nums">
              {data?.pair_tf_done ?? 0}/{data?.pair_tf_total ?? 32} pair×TF ·{' '}
              {(data?.cells_done || 0).toLocaleString()}/{(data?.cells_est || 0).toLocaleString()} cells ·{' '}
              {fmtHours(data?.elapsed_sec)}
            </div>
          </div>
        </div>

        {flash && (
          <div className="pointer-events-none absolute left-1/2 top-28 z-20 -translate-x-1/2 animate-fade-in sm:left-[42%]">
            <div className="rounded-sm border border-accent-gold/35 bg-[#0e0e0c]/92 px-4 py-2 text-xs text-accent-gold shadow-soft backdrop-blur-md">
              {flash.replace(/^\s+/, '')}
            </div>
          </div>
        )}

        {/* Right panel */}
        <aside className="absolute inset-x-0 bottom-0 z-10 max-h-[42%] overflow-y-auto border-t border-white/[0.06] bg-[#0a0a09]/92 backdrop-blur-md sm:inset-y-0 sm:left-auto sm:right-0 sm:max-h-none sm:w-[400px] sm:border-l sm:border-t-0">
          <div className="space-y-5 p-4 sm:p-5 sm:pt-6">
            {/* Focus / testing */}
            <div>
              <div className="mb-2 flex items-center justify-between">
                <div className="section-label mb-0">
                  {focus.source === 'click' ? 'Focused' : 'Testing now'}
                </div>
                {(focus.pair || focus.family || focus.mode) && (
                  <button
                    type="button"
                    className="text-[10px] text-accent-gold"
                    onClick={() => {
                      setFocus({})
                      setFollowHunt(true)
                    }}
                  >
                    reset / follow
                  </button>
                )}
              </div>
              <div className="rounded-sm border border-accent-gold/30 bg-accent-gold/5 p-3 space-y-2">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-lg font-semibold text-gray-100">
                    {focus.pair || testing.pair || '—'}
                    <span className="text-accent-gold">
                      {' '}
                      · {focus.tf || testing.tf || '—'}
                    </span>
                  </span>
                  <span className="text-[11px] text-surface-muted tabular-nums">
                    {testing.mode_progress || '—'} ({modePct}%)
                  </span>
                </div>
                <div className="text-xs text-accent-gold break-all">
                  {focus.mode || testing.mode || '—'}
                </div>
                <div className="text-[11px] text-surface-muted">
                  family{' '}
                  <button
                    type="button"
                    className="text-gray-300 underline-offset-2 hover:underline"
                    onClick={() => {
                      const fam = focus.family || testing.family
                      if (!fam) return
                      setFollowHunt(false)
                      setFocus({ family: fam, source: 'click' })
                      setModeFilter(fam)
                    }}
                  >
                    {focus.family || testing.family || '—'}
                  </button>
                  {focusModeMeta?.label ? ` · ${focusModeMeta.label}` : ''}
                </div>
                {focusTfBlock?.n_bars != null && (
                  <div className="text-[10px] text-surface-muted">
                    {Number(focusTfBlock.n_bars).toLocaleString()} bars ·{' '}
                    {Number(focusTfBlock.n_combos || testing.combos_this_tf || 0).toLocaleString()}{' '}
                    combos · XM session spreads on
                  </div>
                )}
              </div>
            </div>

            {/* Lattice */}
            <div>
              <div className="section-label mb-2">Param lattice (this TF)</div>
              <LatticeGrid
                lattice={search.lattice}
                highlight={
                  latticePick ||
                  (winnerForFocus
                    ? {
                        sl_atr: winnerForFocus.sl_atr,
                        rr: winnerForFocus.rr,
                        min_gap: winnerForFocus.min_gap,
                      }
                    : focusTfBlock
                      ? {
                          sl_atr: focusTfBlock.sl_atr,
                          rr: focusTfBlock.rr,
                          min_gap: focusTfBlock.min_gap,
                        }
                      : null)
                }
                onPick={setLatticePick}
              />
              {latticePick && (
                <div className="mt-1.5 text-[10px] text-surface-muted">
                  Selected cell: SL {latticePick.sl_atr} · RR {latticePick.rr} · gap{' '}
                  {latticePick.min_gap}
                </div>
              )}
            </div>

            {/* Horizons for focus */}
            {(winnerForFocus?.horizons || focusTfBlock?.horizons) && (
              <div>
                <div className="section-label mb-2">Multi-horizon (selected)</div>
                <HorizonBars horizons={winnerForFocus?.horizons || focusTfBlock?.horizons} />
                <div className="mt-2 flex flex-wrap gap-2 text-[10px]">
                  {Object.entries(winnerForFocus?.verdicts || focusTfBlock?.verdicts || {}).map(
                    ([k, v]) => (
                      <span key={k} className={`${verdictTone(v)}`}>
                        {k}:{v || '—'}
                      </span>
                    )
                  )}
                </div>
              </div>
            )}

            {/* Top candidates for focused pair×tf */}
            {focusTfBlock?.top?.length > 0 && (
              <div>
                <div className="section-label mb-2">Top candidates · {focus.pair} {focusTfBlock.tf}</div>
                <div className="max-h-40 space-y-1 overflow-y-auto">
                  {focusTfBlock.top.map((c) => (
                    <button
                      key={c.id}
                      type="button"
                      onClick={() => {
                        setFollowHunt(false)
                        setFocus({
                          pair: focus.pair,
                          tf: focusTfBlock.tf,
                          family: c.family,
                          mode: c.mode,
                          source: 'click',
                        })
                        setModeFilter(c.family || 'all')
                        setLatticePick({
                          sl_atr: c.sl_atr,
                          rr: c.rr,
                          min_gap: c.min_gap,
                        })
                      }}
                      className="flex w-full items-center justify-between gap-2 rounded-sm border border-white/[0.05] px-2 py-1.5 text-left hover:bg-white/[0.03]"
                    >
                      <div className="min-w-0">
                        <div className="truncate text-[11px] text-gray-200">{c.mode}</div>
                        <div className="text-[9px] text-surface-muted">
                          {c.family} · SL{c.sl_atr} RR{c.rr} g{c.min_gap}
                        </div>
                      </div>
                      <div className={`shrink-0 text-[11px] tabular-nums ${verdictTone(c.verdict_1y)}`}>
                        {c.verdict_1y} {fmtPnl(c.pnl_1y)}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Adaptive families */}
            <div>
              <div className="section-label mb-2">
                Adaptive heat · {search.active_catalog === 'lean_1m' ? 'lean 1m' : 'full'} (
                {modes.length})
              </div>
              <div className="space-y-1.5">
                {(search.families || []).map((f) => (
                  <button
                    key={f.id}
                    type="button"
                    onClick={() => {
                      setFollowHunt(false)
                      setFocus({ family: f.id, source: 'click' })
                      setModeFilter((cur) => (cur === f.id ? 'all' : f.id))
                    }}
                    className={`flex w-full items-center gap-2 rounded-sm border px-2 py-1.5 text-left transition-colors ${
                      modeFilter === f.id || focus.family === f.id
                        ? 'border-accent-gold/40 bg-accent-gold/10'
                        : 'border-white/[0.05] hover:bg-white/[0.03]'
                    }`}
                  >
                    <span
                      className="h-2 w-2 shrink-0 rounded-full"
                      style={{ background: FAMILY_COLORS[f.id] || FAMILY_COLORS.other }}
                    />
                    <span className="min-w-0 flex-1 truncate text-[11px] text-gray-200">{f.id}</span>
                    <span className="text-[10px] tabular-nums text-surface-muted">
                      {f.wins || 0}W · {f.scanned}/{f.total}
                    </span>
                    <div className="h-1 w-10 overflow-hidden rounded-full bg-white/[0.06]">
                      <div
                        className="h-full rounded-full bg-accent-gold/80"
                        style={{ width: `${Math.max(4, (f.heat || 0) * 100)}%` }}
                      />
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Processes */}
            <div>
              <div className="mb-2 flex items-center justify-between">
                <div className="section-label mb-0">Processes</div>
                {modeFilter !== 'all' && (
                  <button
                    type="button"
                    className="text-[10px] text-accent-gold"
                    onClick={() => setModeFilter('all')}
                  >
                    show all
                  </button>
                )}
              </div>
              <div className="max-h-44 space-y-0.5 overflow-y-auto rounded-sm border border-white/[0.05] bg-black/20 p-1.5 font-mono text-[10px]">
                {isLoading && <div className="px-2 py-3 text-surface-muted">Connecting…</div>}
                {filteredModes.map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => {
                      setFollowHunt(false)
                      setFocus({ family: m.family, mode: m.id, source: 'click' })
                      setModeFilter(m.family)
                    }}
                    className={`flex w-full items-center gap-2 rounded-sm px-1.5 py-1 text-left ${
                      focus.mode === m.id || m.state === 'active'
                        ? 'bg-accent-gold/15 text-accent-gold'
                        : m.state === 'scanned'
                          ? 'text-gray-300 hover:bg-white/[0.04]'
                          : 'text-surface-muted/70 hover:bg-white/[0.03]'
                    }`}
                  >
                    <span
                      className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                        focus.mode === m.id || m.state === 'active'
                          ? 'bg-accent-gold animate-pulse'
                          : m.state === 'scanned'
                            ? 'bg-accent-buy'
                            : 'bg-white/15'
                      }`}
                    />
                    <span className="truncate">{m.id}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Pair grid */}
            <div>
              <div className="section-label mb-2">Pair × TF</div>
              <div className="grid grid-cols-2 gap-1.5">
                {nodes.map((n) => {
                  const active = n.symbol === (focus.pair || data?.active_pair)
                  return (
                    <button
                      key={n.symbol}
                      type="button"
                      onClick={() => {
                        setFollowHunt(false)
                        setFocus({ pair: n.symbol, tf: focus.tf || data?.active_tf, source: 'click' })
                      }}
                      className={`rounded-sm border px-2 py-1.5 text-left ${
                        active ? 'border-accent-gold/40 bg-accent-gold/5' : 'border-white/[0.05]'
                      }`}
                    >
                      <div className="flex justify-between text-[11px]">
                        <span style={{ color: PAIR_COLORS[n.symbol] }}>{n.symbol}</span>
                        <span className="text-surface-muted">{n.done_tfs}/4</span>
                      </div>
                      <div className="mt-1 flex gap-0.5">
                        {(n.tfs || []).map((tf) => (
                          <div
                            key={tf.tf}
                            onClick={(e) => {
                              e.stopPropagation()
                              setFollowHunt(false)
                              setFocus({
                                pair: n.symbol,
                                tf: tf.tf,
                                family: tf.best_family,
                                mode: tf.best_mode,
                                source: 'click',
                              })
                              if (tf.best_family) setModeFilter(tf.best_family)
                              if (tf.sl_atr != null) {
                                setLatticePick({
                                  sl_atr: tf.sl_atr,
                                  rr: tf.rr,
                                  min_gap: tf.min_gap,
                                })
                              }
                            }}
                            title={`${tf.tf} ${tf.best_mode || ''} ${tf.verdict_1y || ''} ${fmtPnl(tf.pnl_1y)}`}
                            className={`h-1.5 flex-1 cursor-pointer rounded-full ${
                              tf.done
                                ? tf.verdict_1y === 'PASS'
                                  ? 'bg-accent-buy'
                                  : tf.verdict_1y === 'SOFT'
                                    ? 'bg-accent-gold'
                                    : 'bg-accent-sell/80'
                                : active && (focus.tf || data?.active_tf) === tf.tf
                                  ? 'bg-accent-gold/50 animate-pulse'
                                  : 'bg-white/[0.08]'
                            }`}
                          />
                        ))}
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>

            {/* Stream */}
            <div>
              <div className="section-label mb-2">Live stream</div>
              <div className="max-h-32 space-y-1 overflow-y-auto font-mono text-[10px] leading-relaxed text-surface-muted">
                {isError && <div className="text-accent-sell">Hunt feed unreachable</div>}
                {(data?.events || [])
                  .slice()
                  .reverse()
                  .slice(0, 16)
                  .map((e, i) => (
                    <div
                      key={`${i}-${e.text.slice(0, 20)}`}
                      className={
                        e.kind === 'best'
                          ? 'text-accent-gold'
                          : e.kind === 'start'
                            ? 'text-sky-300/80'
                            : e.kind === 'error'
                              ? 'text-accent-sell'
                              : ''
                      }
                    >
                      {e.text}
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </aside>
      </div>
    </Layout>
  )
}

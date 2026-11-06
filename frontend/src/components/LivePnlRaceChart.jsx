import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../services/api'

const PAIR_COLORS = {
  GOLD: '#c9a227',
  USDCHF: '#6b8cae',
  USDJPY: '#8f7a5e',
  NAS100: '#3d8b6e',
  UKOIL: '#b85c38',
  AUDUSD: '#5a7d9a',
  BTCUSD: '#d4a017',
  ETHUSD: '#4a6fa5',
  DESK: '#e8e2d6',
}

function colorFor(sym) {
  return PAIR_COLORS[sym] || '#9a8f7d'
}

function fmtMoney(n) {
  if (n == null || !Number.isFinite(Number(n))) return '—'
  const v = Number(n)
  return `${v >= 0 ? '+' : ''}$${v.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  })}`
}

function fmtTime(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso.slice(0, 16)
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function buildPath(xs, ys, xAt, yAt) {
  if (!xs.length) return ''
  let d = `M ${xAt(0)} ${yAt(ys[0])}`
  for (let i = 1; i < xs.length; i += 1) {
    d += ` L ${xAt(i)} ${yAt(ys[i])}`
  }
  return d
}

/**
 * Animated multi-pair cumulative PnL race — 1y Strategy Lab hunt winners.
 */
export default function LivePnlRaceChart() {
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(1)
  const [frame, setFrame] = useState(0)
  const [hidden, setHidden] = useState(() => new Set())
  const [hover, setHover] = useState(null)
  const rafRef = useRef(0)
  const lastTsRef = useRef(0)
  const accumRef = useRef(0)

  const { data, isLoading, isError, error, refetch, isFetching } = useQuery({
    queryKey: ['lab-hunt-pnl-curves'],
    queryFn: async () => (await api.get('/api/strategy-lab/live-pnl')).data,
    refetchInterval: 120000,
    staleTime: 60000,
  })

  const n = data?.times?.length || 0
  const symbols = data?.symbols || []

  useEffect(() => {
    setFrame(0)
    setPlaying(true)
    accumRef.current = 0
  }, [data?.generated_at])

  useEffect(() => {
    if (!playing || n < 2) return undefined

    const stepMs = Math.max(16, 80 / speed)
    lastTsRef.current = 0

    const tick = (ts) => {
      if (!lastTsRef.current) lastTsRef.current = ts
      const dt = ts - lastTsRef.current
      lastTsRef.current = ts
      accumRef.current += dt
      if (accumRef.current >= stepMs) {
        accumRef.current = 0
        setFrame((f) => {
          if (f >= n - 1) {
            setPlaying(false)
            return n - 1
          }
          // Skip frames at high speed so 1y race finishes in ~30–60s
          const jump = speed >= 4 ? 3 : speed >= 2 ? 2 : 1
          return Math.min(n - 1, f + jump)
        })
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [playing, n, speed, data?.generated_at])

  const layout = useMemo(() => {
    const W = 920
    const H = 360
    const pad = { t: 28, r: 160, b: 44, l: 56 }
    const innerW = W - pad.l - pad.r
    const innerH = H - pad.t - pad.b
    const end = Math.max(0, Math.min(frame, n - 1))
    const desk = (data?.desk || []).slice(0, end + 1)
    const series = {}
    for (const s of symbols) {
      series[s] = (data?.by_symbol?.[s] || []).slice(0, end + 1)
    }
    const allVals = [...desk]
    for (const s of symbols) {
      if (!hidden.has(s)) allVals.push(...(series[s] || []))
    }
    let ymin = Math.min(0, ...allVals)
    let ymax = Math.max(0, ...allVals)
    if (ymin === ymax) {
      ymin -= 50
      ymax += 50
    }
    const padY = (ymax - ymin) * 0.08
    ymin -= padY
    ymax += padY

    const xAt = (i) => pad.l + (innerW * i) / Math.max(1, n - 1)
    const yAt = (v) => pad.t + innerH * (1 - (v - ymin) / (ymax - ymin))

    return { W, H, pad, innerW, innerH, end, desk, series, ymin, ymax, xAt, yAt }
  }, [data, frame, n, symbols, hidden])

  const toggleSym = (s) => {
    setHidden((prev) => {
      const next = new Set(prev)
      if (next.has(s)) next.delete(s)
      else next.add(s)
      return next
    })
  }

  if (isLoading && !data) {
    return (
      <div className="zcard p-8 text-sm text-surface-muted">
        Loading live PnL race…
      </div>
    )
  }

  if (isError) {
    return (
      <div className="zcard p-5 text-sm text-accent-sell">
        {error?.response?.data?.detail || error?.message || 'Failed to load PnL curves'}
      </div>
    )
  }

  if (!n || n < 2) {
    return (
      <div className="zcard p-8 text-sm text-surface-muted">
        No 1y hunt equity curves yet. Run the scalp hunt board, then refresh.
      </div>
    )
  }

  const { W, H, pad, end, desk, series, ymin, ymax, xAt, yAt } = layout
  const zeroY = yAt(0)
  const progress = n > 1 ? end / (n - 1) : 0
  const tNow = data.times[end]
  const deskNow = desk[desk.length - 1] ?? 0

  // Rank labels at playhead
  const ranks = symbols
    .filter((s) => !hidden.has(s))
    .map((s) => ({
      symbol: s,
      v: series[s]?.[series[s].length - 1] ?? 0,
    }))
    .sort((a, b) => b.v - a.v)

  const risk = data.risk_usd ?? 200
  const winLabel =
    data.window_start && data.window_end
      ? `${String(data.window_start).slice(0, 10)} → ${String(data.window_end).slice(0, 10)}`
      : 'past 1y'

  return (
    <div className="zcard overflow-hidden">
      <div className="px-5 py-3 border-b border-surface-border flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-gray-100">1y hunt PnL race</div>
          <div className="text-[11px] text-surface-muted mt-0.5">
            {(data.mode === 'adaptive_minute_day_v3'
              ? 'Adaptive v3 absolute winners'
              : data.mode === 'adaptive_regime_v2'
                ? 'Adaptive preferred modes'
                : data.mode === 'desk_recovery_scalp_hunt'
                  ? 'Recovery scalp winners'
                  : 'Dukascopy scalp-hunt winners')}
            {' · cumulative closed P&L · $'}
            {risk}/R · {winLabel}
            {' · '}
            {Number(data.n_trades || 0).toLocaleString()} closes
            {' · desk '}
            <span className={Number(data.desk_final) >= 0 ? 'text-accent-buy' : 'text-accent-sell'}>
              {fmtMoney(data.desk_final)}
            </span>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            className="btn-secondary !px-3 !py-1.5"
            onClick={() => {
              setFrame(0)
              setPlaying(true)
              accumRef.current = 0
            }}
          >
            Replay
          </button>
          <button
            type="button"
            className="btn-secondary !px-3 !py-1.5"
            onClick={() => setPlaying((p) => !p)}
          >
            {playing ? 'Pause' : 'Play'}
          </button>
          <select
            className="bg-surface-raised border border-surface-border rounded-sm text-[11px] text-ink px-2 py-1.5"
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
          >
            <option value={0.5}>0.5×</option>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
            <option value={4}>4×</option>
            <option value={8}>8×</option>
          </select>
          <button
            type="button"
            className="btn-secondary !px-3 !py-1.5"
            onClick={() => refetch()}
            disabled={isFetching}
          >
            {isFetching ? '…' : 'Refresh'}
          </button>
        </div>
      </div>

      <div className="px-5 pt-4 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => toggleSym('__desk')}
          className="text-[10px] uppercase tracking-wide px-2 py-1 rounded border border-white/15 text-ink-soft"
          style={{ opacity: hidden.has('__desk') ? 0.35 : 1 }}
        >
          <span className="inline-block w-2 h-2 rounded-full mr-1.5 align-middle" style={{ background: PAIR_COLORS.DESK }} />
          Desk {fmtMoney(deskNow)}
        </button>
        {symbols.map((s) => {
          const v = data.finals?.[s] ?? 0
          return (
            <button
              key={s}
              type="button"
              onClick={() => toggleSym(s)}
              className="text-[10px] uppercase tracking-wide px-2 py-1 rounded border border-white/10"
              style={{ opacity: hidden.has(s) ? 0.35 : 1, color: colorFor(s) }}
            >
              <span
                className="inline-block w-2 h-2 rounded-full mr-1.5 align-middle"
                style={{ background: colorFor(s) }}
              />
              {s} {fmtMoney(series[s]?.[series[s].length - 1] ?? v)}
            </button>
          )
        })}
      </div>

      <div className="px-3 pb-2 overflow-x-auto">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full min-w-[640px] h-auto mt-2"
          role="img"
          aria-label="Cumulative PnL by pair over time"
          onMouseLeave={() => setHover(null)}
        >
          {/* grid */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const v = ymin + (ymax - ymin) * (1 - t)
            const y = pad.t + (H - pad.t - pad.b) * t
            return (
              <g key={t}>
                <line
                  x1={pad.l}
                  x2={W - pad.r}
                  y1={y}
                  y2={y}
                  stroke="rgba(255,255,255,0.06)"
                  strokeWidth={1}
                />
                <text x={pad.l - 8} y={y + 3} textAnchor="end" fill="#7a7368" fontSize={10}>
                  {fmtMoney(v)}
                </text>
              </g>
            )
          })}
          <line
            x1={pad.l}
            x2={W - pad.r}
            y1={zeroY}
            y2={zeroY}
            stroke="rgba(255,255,255,0.18)"
            strokeWidth={1}
            strokeDasharray="4 4"
          />

          {/* pair lines */}
          {symbols.map((s) => {
            if (hidden.has(s)) return null
            const ys = series[s] || []
            const xs = ys.map((_, i) => i)
            return (
              <path
                key={s}
                d={buildPath(xs, ys, xAt, yAt)}
                fill="none"
                stroke={colorFor(s)}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
                opacity={0.92}
              />
            )
          })}

          {/* desk line on top */}
          {!hidden.has('__desk') && (
            <path
              d={buildPath(
                desk.map((_, i) => i),
                desk,
                xAt,
                yAt
              )}
              fill="none"
              stroke={PAIR_COLORS.DESK}
              strokeWidth={2.5}
              strokeLinejoin="round"
              strokeLinecap="round"
              opacity={0.95}
            />
          )}

          {/* playhead */}
          <line
            x1={xAt(end)}
            x2={xAt(end)}
            y1={pad.t}
            y2={H - pad.b}
            stroke="rgba(232,226,214,0.35)"
            strokeWidth={1}
          />

          {/* end dots + rank labels */}
          {ranks.map((r, idx) => {
            const y = yAt(r.v)
            return (
              <g key={r.symbol}>
                <circle cx={xAt(end)} cy={y} r={3.5} fill={colorFor(r.symbol)} />
                <text
                  x={W - pad.r + 10}
                  y={pad.t + 14 + idx * 16}
                  fill={colorFor(r.symbol)}
                  fontSize={11}
                  fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
                >
                  {r.symbol} {fmtMoney(r.v)}
                </text>
              </g>
            )
          })}

          {/* time axis */}
          <text x={pad.l} y={H - 12} fill="#7a7368" fontSize={10}>
            {fmtTime(data.times[0])}
          </text>
          <text x={W - pad.r} y={H - 12} textAnchor="end" fill="#7a7368" fontSize={10}>
            {fmtTime(tNow)}
          </text>

          {/* hover scrub */}
          <rect
            x={pad.l}
            y={pad.t}
            width={W - pad.l - pad.r}
            height={H - pad.t - pad.b}
            fill="transparent"
            onMouseMove={(e) => {
              const svg = e.currentTarget.ownerSVGElement
              const pt = svg.createSVGPoint()
              pt.x = e.clientX
              pt.y = e.clientY
              const ctm = svg.getScreenCTM()
              if (!ctm) return
              const loc = pt.matrixTransform(ctm.inverse())
              const ratio = (loc.x - pad.l) / (W - pad.l - pad.r)
              const i = Math.max(0, Math.min(n - 1, Math.round(ratio * (n - 1))))
              setHover(i)
              if (!playing) setFrame(i)
            }}
            onClick={() => setPlaying(false)}
          />

          {hover != null && hover <= end && (
            <g>
              <line
                x1={xAt(hover)}
                x2={xAt(hover)}
                y1={pad.t}
                y2={H - pad.b}
                stroke="rgba(201,162,39,0.45)"
                strokeWidth={1}
              />
              <text x={xAt(hover)} y={pad.t - 8} textAnchor="middle" fill="#c9a227" fontSize={10}>
                {fmtTime(data.times[hover])} · desk {fmtMoney(data.desk[hover])}
              </text>
            </g>
          )}
        </svg>
      </div>

      <div className="px-5 pb-4">
        <div className="h-1 rounded-full bg-white/[0.06] overflow-hidden">
          <div
            className="h-full bg-accent-gold/70 transition-[width] duration-75"
            style={{ width: `${progress * 100}%` }}
          />
        </div>
        <div className="mt-2 flex justify-between text-[10px] text-surface-muted tabular-nums">
          <span>
            Frame {end + 1}/{n} · desk {fmtMoney(deskNow)}
          </span>
          <span>Scrub chart to pause · 1y hunt backtest (not live demo)</span>
        </div>
      </div>
    </div>
  )
}

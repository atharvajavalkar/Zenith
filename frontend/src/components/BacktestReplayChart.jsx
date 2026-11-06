/**
 * BacktestReplayChart
 *
 * TradingView-style live market replay.
 * - Chart starts at bar 0 and scrolls right as candles are added
 * - Maintains a fixed ~100-bar visible window (like watching a live chart)
 * - Speed: 1x / 2x / 5x / 10x / 25x
 * - Scrubber to jump to any point
 * - Running P&L counter + trade log
 */
import { useEffect, useRef, useState, useCallback } from 'react'
import { createChart, CrosshairMode, LineStyle } from 'lightweight-charts'

const VISIBLE_BARS = 100   // how many bars to show at once (like TradingView default)
const RIGHT_OFFSET = 8     // empty space on the right (future bars)

const SPEEDS = [
  { label: '1x',  ms: 500 },
  { label: '2x',  ms: 250 },
  { label: '5x',  ms: 100 },
  { label: '10x', ms: 50  },
  { label: '25x', ms: 20  },
]

const pnlCls = (v) =>
  Number(v) > 0.5 ? 'text-green-400' : Number(v) < -0.5 ? 'text-red-400' : 'text-gray-300'

function TradeTag({ trade }) {
  const isBuy = trade.signal === 'BUY'
  const pnl   = trade.pnl
  const isOpen = trade.status === 'open' || !trade.exit_price

  return (
    <div className={`rounded-lg border px-3 py-2 text-xs ${
      isOpen
        ? 'border-accent-gold/30 bg-accent-gold/5'
        : (pnl ?? 0) > 0.5
          ? 'border-green-500/30 bg-green-500/5'
          : (pnl ?? 0) < -0.5
            ? 'border-red-500/30 bg-red-500/5'
            : 'border-gray-700 bg-gray-800/30'
    }`}>
      <div className="flex items-center justify-between gap-2">
        <span className={`font-bold ${isBuy ? 'text-green-400' : 'text-red-400'}`}>{trade.signal}</span>
        <span className="text-gray-500 text-[10px]">{trade.rr_type ?? ''}</span>
        {isOpen
          ? <span className="text-accent-gold text-[10px] font-medium">LIVE</span>
          : <span className={`font-bold ${pnlCls(pnl)}`}>{(pnl ?? 0) >= 0 ? '+' : ''}${(pnl ?? 0).toFixed(2)}</span>
        }
      </div>
      <div className="text-gray-500 tabular-nums mt-0.5 text-[10px]">
        E:{trade.entry}  SL:{trade.sl}  TP:{trade.tp ?? '—'}
      </div>
    </div>
  )
}

export default function BacktestReplayChart({ candles = [], events = [], startAtBar = 0 }) {
  const containerRef = useRef(null)
  const chartRef     = useRef(null)
  const seriesRef    = useRef(null)
  const timerRef     = useRef(null)
  const cursorRef    = useRef(0)   // use ref to avoid stale closure in timer

  const [speedIdx,  setSpeedIdx]  = useState(1)
  const [playing,   setPlaying]   = useState(false)
  const [cursor,    setCursor]    = useState(0)
  const [trades,    setTrades]    = useState([])
  const [pnl,       setPnl]       = useState(0)
  const [wins,      setWins]      = useState(0)
  const [losses,    setLosses]    = useState(0)

  const total = candles.length
  const jumpBar = Math.max(0, Math.min(Number(startAtBar) || 0, Math.max(total - 1, 0)))

  // Event lookup: index → events[]
  const eventMap = useRef({})
  useEffect(() => {
    const map = {}
    ;(events || []).forEach(ev => {
      if (!map[ev.candle_index]) map[ev.candle_index] = []
      map[ev.candle_index].push(ev)
    })
    eventMap.current = map
  }, [events])

  // ── Init chart once ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      layout: {
        background:  { color: '#0d1117' },
        textColor:   '#6b7280',
        fontSize:    11,
      },
      grid: {
        vertLines: { color: '#161b27', style: LineStyle.Dotted },
        horzLines: { color: '#161b27', style: LineStyle.Dotted },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: {
        borderColor:     '#1f2937',
        scaleMarginTop:  0.12,
        scaleMarginBottom: 0.12,
      },
      timeScale: {
        borderColor:      '#1f2937',
        timeVisible:      true,
        secondsVisible:   false,
        rightOffset:      RIGHT_OFFSET,
        barSpacing:       8,
        minBarSpacing:    2,
        fixLeftEdge:      false,
        fixRightEdge:     false,
        lockVisibleTimeRangeOnResize: true,
      },
      handleScroll:  { mouseWheel: true, pressedMouseMove: true },
      handleScale:   { mouseWheel: true, pinch: true },
      width:  containerRef.current.clientWidth,
      height: containerRef.current.clientHeight || 420,
    })

    const series = chart.addCandlestickSeries({
      upColor:         '#26a69a',
      downColor:       '#ef5350',
      borderUpColor:   '#26a69a',
      borderDownColor: '#ef5350',
      wickUpColor:     '#26a69a',
      wickDownColor:   '#ef5350',
    })

    chartRef.current  = chart
    seriesRef.current = series

    const ro = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({
          width:  containerRef.current.clientWidth,
          height: containerRef.current.clientHeight || 420,
        })
      }
    })
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      chart.remove()
      chartRef.current  = null
      seriesRef.current = null
    }
  }, [])

  // ── Advance one bar ────────────────────────────────────────────────────────
  const stepTo = useCallback((idx) => {
    if (idx >= total || !seriesRef.current || !chartRef.current) return

    const c = candles[idx]
    seriesRef.current.update(c)

    // Scroll the visible window to keep this candle in the right portion
    // — just like watching a live TradingView chart
    const from = Math.max(0, idx - VISIBLE_BARS + RIGHT_OFFSET)
    const to   = idx + RIGHT_OFFSET
    chartRef.current.timeScale().setVisibleLogicalRange({ from, to })

    // Handle events at this bar
    const evs = eventMap.current[idx] || []
    evs.forEach(ev => {
      if (ev.type === 'trade_open') {
        setTrades(prev => [{ ...ev, _id: idx }, ...prev])
      }
      if (ev.type === 'trade_close') {
        const pnlVal = ev.pnl ?? 0
        setTrades(prev =>
          prev.map(t =>
            t.candle_index === ev.candle_index && t.type === 'trade_open'
              ? { ...t, ...ev }
              : t
          )
        )
        setPnl(p => +(p + pnlVal).toFixed(2))
        if (pnlVal > 1)       setWins(w => w + 1)
        else if (pnlVal < -1) setLosses(l => l + 1)
      }
    })

    cursorRef.current = idx + 1
    setCursor(idx + 1)
  }, [candles, total])

  // ── Play / pause ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (playing) {
      const ms = SPEEDS[speedIdx].ms
      timerRef.current = setInterval(() => {
        const cur = cursorRef.current
        if (cur >= total) {
          setPlaying(false)
          clearInterval(timerRef.current)
          return
        }
        stepTo(cur)
      }, ms)
    } else {
      clearInterval(timerRef.current)
    }
    return () => clearInterval(timerRef.current)
  }, [playing, speedIdx, stepTo, total])

  // ── Seek (scrubber) ────────────────────────────────────────────────────────
  const seek = useCallback((targetIdx) => {
    if (!seriesRef.current || !chartRef.current) return
    setPlaying(false)
    clearInterval(timerRef.current)

    // Rebuild all state up to target
    setTrades([])
    setPnl(0)
    setWins(0)
    setLosses(0)

    // Feed candles from beginning to target
    const slice = candles.slice(0, targetIdx + 1)
    seriesRef.current.setData(slice)

    // Set visible window
    const from = Math.max(0, targetIdx - VISIBLE_BARS + RIGHT_OFFSET)
    const to   = targetIdx + RIGHT_OFFSET
    chartRef.current.timeScale().setVisibleLogicalRange({ from, to })

    // Replay events
    let runPnl = 0, w = 0, l = 0
    const seen = []
    for (let i = 0; i <= targetIdx; i++) {
      const evs = eventMap.current[i] || []
      evs.forEach(ev => {
        if (ev.type === 'trade_open') {
          seen.unshift({ ...ev, _id: i })
        }
        if (ev.type === 'trade_close') {
          const pnlVal = ev.pnl ?? 0
          const match  = seen.findIndex(t => t.candle_index === ev.candle_index && t.type === 'trade_open')
          if (match >= 0) seen[match] = { ...seen[match], ...ev }
          runPnl = +(runPnl + pnlVal).toFixed(2)
          if (pnlVal > 1)       w++
          else if (pnlVal < -1) l++
        }
      })
    }
    setTrades(seen)
    setPnl(runPnl)
    setWins(w)
    setLosses(l)
    cursorRef.current = targetIdx + 1
    setCursor(targetIdx + 1)
  }, [candles])

  // ── Reset to start ─────────────────────────────────────────────────────────
  const reset = useCallback(() => {
    setPlaying(false)
    clearInterval(timerRef.current)
    if (seriesRef.current) seriesRef.current.setData([])
    setTrades([])
    setPnl(0)
    setWins(0)
    setLosses(0)
    cursorRef.current = 0
    setCursor(0)
  }, [])

  const jumpToFirstTrade = useCallback(() => {
    if (jumpBar > 0) seek(jumpBar)
    else reset()
  }, [jumpBar, seek, reset])

  const pctDone  = total > 0 ? Math.round((cursor / total) * 100) : 0
  const decisive = wins + losses

  return (
    <div className="flex flex-col gap-2">
      {/* ── toolbar ── */}
      <div className="zcard px-4 py-2.5 flex flex-wrap items-center gap-3">

        {/* play / pause */}
        <button
          onClick={() => setPlaying(p => !p)}
          disabled={cursor >= total}
          className="w-9 h-9 rounded-lg bg-accent-gold text-surface font-bold text-base flex items-center justify-center hover:opacity-90 transition disabled:opacity-40"
          title={playing ? 'Pause' : 'Play'}
        >
          {playing ? '⏸' : '▶'}
        </button>

        {/* reset */}
        <button
          onClick={reset}
          className="w-9 h-9 rounded-lg border border-surface-border text-gray-400 hover:text-white hover:border-gray-500 flex items-center justify-center transition"
          title="Reset to start"
        >
          ⏮
        </button>

        {jumpBar > 0 && (
          <button
            type="button"
            onClick={jumpToFirstTrade}
            className="px-3 h-9 rounded-lg border border-accent-gold/40 text-[11px] uppercase tracking-wide text-accent-gold hover:bg-accent-gold/10 transition"
            title={`Jump to first trade (bar ${jumpBar})`}
          >
            First trade · {jumpBar}
          </button>
        )}

        {/* speed selector */}
        <div className="flex rounded-lg overflow-hidden border border-surface-border text-xs font-semibold">
          {SPEEDS.map((sp, i) => (
            <button
              key={sp.label}
              onClick={() => setSpeedIdx(i)}
              className={`px-2.5 py-1.5 transition ${i === speedIdx ? 'bg-accent-gold text-surface' : 'text-surface-muted hover:text-white'}`}
            >
              {sp.label}
            </button>
          ))}
        </div>

        {/* scrubber */}
        <div className="flex-1 flex items-center gap-2 min-w-[140px]">
          <input
            type="range"
            min={0}
            max={Math.max(0, total - 1)}
            value={Math.max(0, cursor - 1)}
            onChange={e => seek(Number(e.target.value))}
            className="flex-1 h-1.5 accent-yellow-400 cursor-pointer"
          />
          <span className="text-xs text-surface-muted tabular-nums whitespace-nowrap">
            {cursor}/{total}
          </span>
        </div>

        {/* live stats */}
        <div className="flex items-center gap-4 text-sm tabular-nums shrink-0">
          <span className={`font-bold ${pnlCls(pnl)}`}>
            {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
          </span>
          <span className="text-green-400 text-xs">W:{wins}</span>
          <span className="text-red-400 text-xs">L:{losses}</span>
          {decisive > 0 && (
            <span className={`text-xs ${wins / decisive >= 0.5 ? 'text-green-300' : 'text-red-300'}`}>
              {(wins / decisive * 100).toFixed(0)}%WR
            </span>
          )}
          <span className="text-xs text-surface-muted">{pctDone}%</span>
        </div>
      </div>

      {/* ── chart + trade log ── */}
      <div className="flex gap-3" style={{ height: '440px' }}>
        {/* chart — takes all available width */}
        <div
          ref={containerRef}
          className="flex-1 rounded-xl overflow-hidden border border-surface-border bg-[#0d1117]"
          style={{ minWidth: 0 }}
        />

        {/* trade log — scrollable sidebar */}
        <div className="w-52 shrink-0 flex flex-col gap-1.5 overflow-y-auto">
          <span className="text-[10px] font-semibold text-surface-muted uppercase tracking-wider px-1">
            Trades · {trades.length}
          </span>
          {trades.length === 0 ? (
            <p className="text-xs text-surface-muted px-1 pt-2">
              {cursor === 0 ? 'Press ▶ to start' : 'No signals yet…'}
            </p>
          ) : (
            trades.map((t, i) => <TradeTag key={i} trade={t} />)
          )}
        </div>
      </div>
    </div>
  )
}

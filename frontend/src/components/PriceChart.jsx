import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'
import { ColorType, createChart } from 'lightweight-charts'
import api from '../services/api'
import { TV_INDICATOR_CATALOG, getIndicatorById } from '../data/tvIndicators'
import { calcIndicator } from '../utils/chartIndicators'

/** Zenith desk palette — matches dashboard void / gold / emerald / oxblood */
const COLORS = {
  bg: '#0e0e0c',
  panel: '#121210',
  text: '#ece8e1',
  muted: '#8a857c',
  border: '#242420',
  crosshair: 'rgba(201, 166, 107, 0.35)',
  upCandle: '#d8c9a8',
  downCandle: '#1c1b18',
  wickUp: '#c9a66b',
  wickDown: '#6a655c',
  buy: '#2f9e6f',
  sell: '#d45b5b',
  live: '#c9a66b',
  bid: '#d45b5b',
  ask: '#2f9e6f',
  tp1: '#c9a66b',
  tp2: '#2f9e6f',
}

function formatPrice(n) {
  if (n == null || Number.isNaN(n)) return '—'
  return Number(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 3 })
}

function parseUtcSeconds(value) {
  if (!value) return null
  const clean = String(value).split('#')[0].trim().replace(' ', 'T')
  const withZ = clean.endsWith('Z') || clean.includes('+') ? clean : `${clean}Z`
  const ms = Date.parse(withZ)
  return Number.isNaN(ms) ? null : Math.floor(ms / 1000)
}

function barsEqual(a, b) {
  if (!a || !b) return false
  return (
    a.time === b.time
    && a.open === b.open
    && a.high === b.high
    && a.low === b.low
    && a.close === b.close
    && (a.volume || 0) === (b.volume || 0)
  )
}

function toVolumeBar(c) {
  return {
    time: c.time,
    value: c.volume || 0,
    color: c.close >= c.open ? 'rgba(201,166,107,0.22)' : 'rgba(138,133,124,0.28)',
  }
}

const CHART_INTERACTION = {
  handleScroll: {
    mouseWheel: false,
    pressedMouseMove: true,
    horzTouchDrag: true,
    vertTouchDrag: false,
  },
  handleScale: {
    mouseWheel: true,
    pinch: true,
    axisPressedMouseMove: { time: true, price: true },
    axisDoubleClickReset: { time: true, price: true },
  },
  kineticScroll: { touch: true, mouse: true },
}

function linkTimeScales(charts, syncingRef) {
  const cleanups = []
  charts.filter(Boolean).forEach((source) => {
    const handler = (range) => {
      if (!range || syncingRef.current) return
      syncingRef.current = true
      charts.forEach((target) => {
        if (target && target !== source) {
          target.timeScale().setVisibleLogicalRange(range)
        }
      })
      syncingRef.current = false
    }
    source.timeScale().subscribeVisibleLogicalRangeChange(handler)
    cleanups.push(() => source.timeScale().unsubscribeVisibleLogicalRangeChange(handler))
  })
  return () => cleanups.forEach((fn) => fn())
}

export default function PriceChart({
  candles,
  timeframe,
  onTimeframeChange,
  signal,
  showSignals = true,
  onToggleSignals,
}) {
  const mainChartRef = useRef(null)
  const rsiChartRef = useRef(null)
  const volChartRef = useRef(null)
  const mainApiRef = useRef(null)
  const rsiApiRef = useRef(null)
  const volApiRef = useRef(null)
  const liveLinesRef = useRef({})
  const signalLinesRef = useRef({})
  const indicatorSeriesRef = useRef({})
  const candlesCacheRef = useRef([])
  const userInteractedRef = useRef(false)
  const syncingRef = useRef(false)
  const crosshairRafRef = useRef(null)
  const ohlcRefs = useRef({ o: null, h: null, l: null, c: null, chg: null })
  const lastBarRef = useRef(null)
  const isHoveringRef = useRef(false)
  const [showIndicators, setShowIndicators] = useState(false)
  const [indicatorSearch, setIndicatorSearch] = useState('')
  const [activeIndicators, setActiveIndicators] = useState(() => new Set())
  const [livePriceDisplay, setLivePriceDisplay] = useState(null)
  const [showLevels, setShowLevels] = useState(true)

  const { data: tick } = useQuery({
    queryKey: ['live-tick'],
    queryFn: async () => (await api.get('/api/signals/tick')).data,
    refetchInterval: 2000,
  })

  const { data: tradesData } = useQuery({
    queryKey: ['chart-trades'],
    queryFn: async () =>
      (await api.get('/api/trades/history', { params: { page: 1, per_page: 100 } })).data,
    refetchInterval: 8000,
  })

  const sortedCandles = useMemo(() => {
    if (!candles?.length) return []
    return [...candles].sort((a, b) => a.time - b.time)
  }, [candles])

  const lastBar = sortedCandles.length ? sortedCandles[sortedCandles.length - 1] : null
  const livePrice = tick?.price ?? signal?.LIVE_PRICE ?? lastBar?.close
  const spread = 0.49
  const bid = livePrice != null ? livePrice - spread / 2 : null
  const ask = livePrice != null ? livePrice + spread / 2 : null

  const hasRsi = [...activeIndicators].some((id) => getIndicatorById(id)?.pane === 'rsi')
  const filteredCatalog = useMemo(() => {
    const q = indicatorSearch.trim().toLowerCase()
    if (!q) return TV_INDICATOR_CATALOG
    return TV_INDICATOR_CATALOG.map((g) => ({
      ...g,
      items: g.items.filter((i) => i.name.toLowerCase().includes(q)),
    })).filter((g) => g.items.length > 0)
  }, [indicatorSearch])

  const ohlcChange = lastBar ? lastBar.close - lastBar.open : null
  const ohlcPct = lastBar?.open ? (ohlcChange / lastBar.open) * 100 : null

  const paintOhlc = (bar) => {
    if (!bar) return
    const { o, h, l, c, chg } = ohlcRefs.current
    if (o) o.textContent = formatPrice(bar.open)
    if (h) h.textContent = formatPrice(bar.high)
    if (l) l.textContent = formatPrice(bar.low)
    if (c) c.textContent = formatPrice(bar.close)
    if (chg) {
      const delta = bar.close - bar.open
      const pct = bar.open ? (delta / bar.open) * 100 : 0
      chg.textContent = `${delta >= 0 ? '+' : ''}${formatPrice(delta)} (${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%)`
      chg.style.color = delta >= 0 ? COLORS.buy : COLORS.sell
    }
  }

  useEffect(() => {
    lastBarRef.current = lastBar
    if (!isHoveringRef.current) paintOhlc(lastBar)
  }, [lastBar])

  useEffect(() => {
    if (livePrice != null) setLivePriceDisplay(livePrice)
  }, [livePrice])

  useEffect(() => {
    userInteractedRef.current = false
    candlesCacheRef.current = []
  }, [timeframe])

  const zoomLogical = (factor) => {
    const chart = mainApiRef.current?.chart
    if (!chart) return
    userInteractedRef.current = true
    const ts = chart.timeScale()
    const range = ts.getVisibleLogicalRange()
    if (!range) return
    const center = (range.from + range.to) / 2
    const span = (range.to - range.from) * factor
    ts.setVisibleLogicalRange({ from: center - span / 2, to: center + span / 2 })
  }

  const fitChart = () => {
    userInteractedRef.current = false
    const chart = mainApiRef.current?.chart
    chart?.timeScale().fitContent()
    chart?.priceScale('right').applyOptions({ autoScale: true })
    volApiRef.current?.chart.timeScale().fitContent()
    rsiApiRef.current?.chart.timeScale().fitContent()
  }

  const toggleIndicator = (id) => {
    setActiveIndicators((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  useEffect(() => {
    if (!mainChartRef.current || !volChartRef.current) return

    const baseOptions = {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: COLORS.bg },
        textColor: COLORS.muted,
        fontFamily: 'Outfit, ui-sans-serif, sans-serif',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.03)', visible: true },
        horzLines: { color: 'rgba(255,255,255,0.03)', visible: true },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: COLORS.crosshair,
          width: 1,
          style: 2,
          labelBackgroundColor: COLORS.panel,
        },
        horzLine: {
          color: COLORS.crosshair,
          width: 1,
          style: 2,
          labelBackgroundColor: COLORS.panel,
        },
      },
      rightPriceScale: {
        borderColor: COLORS.border,
        autoScale: true,
        scaleMargins: { top: 0.08, bottom: 0.08 },
      },
      timeScale: {
        borderColor: COLORS.border,
        visible: false,
        rightOffset: 8,
        barSpacing: 8,
        minBarSpacing: 1,
        fixLeftEdge: false,
        fixRightEdge: false,
      },
      ...CHART_INTERACTION,
    }

    const mainChart = createChart(mainChartRef.current, baseOptions)
    const volChart = createChart(volChartRef.current, {
      ...baseOptions,
      timeScale: {
        ...baseOptions.timeScale,
        visible: true,
        timeVisible: true,
        secondsVisible: false,
      },
    })

    const candleSeries = mainChart.addCandlestickSeries({
      upColor: COLORS.upCandle,
      downColor: COLORS.downCandle,
      borderUpColor: COLORS.wickUp,
      borderDownColor: COLORS.wickDown,
      wickUpColor: COLORS.wickUp,
      wickDownColor: COLORS.wickDown,
      lastValueVisible: true,
      priceLineVisible: false,
    })

    const volumeSeries = volChart.addHistogramSeries({ priceFormat: { type: 'volume' } })
    volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.1, bottom: 0 } })

    mainApiRef.current = { chart: mainChart, candleSeries }
    volApiRef.current = { chart: volChart, volumeSeries }

    const unlink = linkTimeScales([mainChart, volChart], syncingRef)

    mainChart.timeScale().subscribeVisibleLogicalRangeChange(() => {
      if (!syncingRef.current) userInteractedRef.current = true
    })

    mainChart.subscribeCrosshairMove((param) => {
      if (crosshairRafRef.current) cancelAnimationFrame(crosshairRafRef.current)
      crosshairRafRef.current = requestAnimationFrame(() => {
        if (!param.time || !param.seriesData) {
          isHoveringRef.current = false
          paintOhlc(lastBarRef.current)
          return
        }
        const bar = param.seriesData.get(candleSeries)
        if (bar) {
          isHoveringRef.current = true
          paintOhlc({ open: bar.open, high: bar.high, low: bar.low, close: bar.close })
        }
      })
    })

    return () => {
      if (crosshairRafRef.current) cancelAnimationFrame(crosshairRafRef.current)
      unlink()
      mainChart.remove()
      volChart.remove()
      rsiApiRef.current?.chart.remove()
      mainApiRef.current = null
      volApiRef.current = null
      rsiApiRef.current = null
      indicatorSeriesRef.current = {}
      liveLinesRef.current = {}
    }
  }, [])

  useEffect(() => {
    if (!hasRsi || !rsiChartRef.current) {
      if (rsiApiRef.current) {
        rsiApiRef.current.chart.remove()
        rsiApiRef.current = null
        Object.keys(indicatorSeriesRef.current).forEach((k) => {
          if (k.startsWith('rsi_')) delete indicatorSeriesRef.current[k]
        })
      }
      return
    }
    if (rsiApiRef.current) return

    const rsiChart = createChart(rsiChartRef.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: COLORS.bg },
        textColor: COLORS.muted,
        fontFamily: 'Outfit, ui-sans-serif, sans-serif',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.03)', visible: true },
        horzLines: { color: 'rgba(255,255,255,0.03)', visible: true },
      },
      rightPriceScale: { borderColor: COLORS.border },
      timeScale: { visible: false },
      ...CHART_INTERACTION,
    })

    const charts = [mainApiRef.current?.chart, volApiRef.current?.chart, rsiChart].filter(Boolean)
    linkTimeScales(charts, syncingRef)

    rsiApiRef.current = { chart: rsiChart }
  }, [hasRsi])

  useEffect(() => {
    const main = mainApiRef.current
    const vol = volApiRef.current
    if (!main?.candleSeries || !sortedCandles.length) return

    const prev = candlesCacheRef.current
    const shouldFullLoad = !prev.length
      || sortedCandles.length !== prev.length
      || sortedCandles[0]?.time !== prev[0]?.time
      || sortedCandles.length < prev.length

    if (shouldFullLoad) {
      main.candleSeries.setData(sortedCandles)
      vol?.volumeSeries.setData(sortedCandles.map(toVolumeBar))
      candlesCacheRef.current = sortedCandles
      if (!userInteractedRef.current) {
        main.chart.timeScale().fitContent()
        vol?.chart.timeScale().fitContent()
      }
      return
    }

    const lastPrev = prev[prev.length - 1]
    const lastNew = sortedCandles[sortedCandles.length - 1]

    if (lastPrev.time === lastNew.time) {
      if (!barsEqual(lastPrev, lastNew)) {
        main.candleSeries.update(lastNew)
        vol?.volumeSeries.update(toVolumeBar(lastNew))
      }
    } else if (lastNew.time > lastPrev.time) {
      main.candleSeries.update(lastNew)
      vol?.volumeSeries.update(toVolumeBar(lastNew))
    } else {
      main.candleSeries.setData(sortedCandles)
      vol?.volumeSeries.setData(sortedCandles.map(toVolumeBar))
    }

    candlesCacheRef.current = sortedCandles
  }, [sortedCandles])

  useEffect(() => {
    const { candleSeries } = mainApiRef.current || {}
    if (!candleSeries) return

    const upsertLine = (key, price, color, title, lineWidth, lineStyle) => {
      if (price == null || !showLevels) {
        if (liveLinesRef.current[key]) {
          try {
            candleSeries.removePriceLine(liveLinesRef.current[key])
          } catch {
            // removed
          }
          delete liveLinesRef.current[key]
        }
        return
      }
      if (!liveLinesRef.current[key]) {
        liveLinesRef.current[key] = candleSeries.createPriceLine({
          price,
          color,
          lineWidth,
          lineStyle,
          axisLabelVisible: true,
          title,
        })
      } else {
        liveLinesRef.current[key].applyOptions({ price, color, title })
      }
    }

    upsertLine('live', livePrice, COLORS.live, 'Live', 1, 0)
    upsertLine('bid', bid, COLORS.bid, 'Bid', 1, 2)
    upsertLine('ask', ask, COLORS.ask, 'Ask', 1, 2)
  }, [livePrice, bid, ask, showLevels])

  useEffect(() => {
    const { candleSeries } = mainApiRef.current || {}
    if (!candleSeries) return

    const clearSignalLines = () => {
      Object.values(signalLinesRef.current).forEach((line) => {
        try {
          candleSeries.removePriceLine(line)
        } catch {
          // gone
        }
      })
      signalLinesRef.current = {}
    }

    if (!showSignals || !signal || (signal.SIGNAL !== 'BUY' && signal.SIGNAL !== 'SELL')) {
      clearSignalLines()
      return
    }

    const upsertSignalLine = (key, price, color, title, lineStyle) => {
      if (price == null) return false
      const p = Number(price)
      if (!signalLinesRef.current[key]) {
        signalLinesRef.current[key] = candleSeries.createPriceLine({
          price: p,
          color,
          lineWidth: 1,
          lineStyle,
          axisLabelVisible: true,
          title,
        })
      } else {
        signalLinesRef.current[key].applyOptions({ price: p, color, title })
      }
      return true
    }

    const activeKeys = new Set()
    if (upsertSignalLine('entry', signal.ENTRY, COLORS.live, `${signal.SIGNAL}`, 0)) activeKeys.add('entry')
    if (upsertSignalLine('sl', signal.SL, COLORS.sell, 'SL', 2)) activeKeys.add('sl')
    if (upsertSignalLine('tp1', signal.TP1, COLORS.tp1, 'TP1', 2)) activeKeys.add('tp1')
    if (upsertSignalLine('tp2', signal.TP, COLORS.tp2, 'TP2', 2)) activeKeys.add('tp2')

    Object.entries(signalLinesRef.current).forEach(([key, line]) => {
      if (activeKeys.has(key)) return
      try {
        candleSeries.removePriceLine(line)
      } catch {
        // gone
      }
      delete signalLinesRef.current[key]
    })
  }, [signal, showSignals])

  useEffect(() => {
    const { candleSeries } = mainApiRef.current || {}
    if (!candleSeries) return

    if (!showSignals || !sortedCandles.length || !tradesData?.items?.length) {
      candleSeries.setMarkers([])
      return
    }

    const times = sortedCandles.map((c) => c.time)
    const first = times[0]
    const last = times[times.length - 1]

    const snap = (sec) => {
      if (sec <= first) return first
      if (sec >= last) return last
      let lo = 0
      let hi = times.length - 1
      let ans = first
      while (lo <= hi) {
        const mid = (lo + hi) >> 1
        if (times[mid] <= sec) {
          ans = times[mid]
          lo = mid + 1
        } else {
          hi = mid - 1
        }
      }
      return ans
    }

    const seen = new Set()
    const markers = []
    tradesData.items.forEach((t) => {
      if (t.signal !== 'BUY' && t.signal !== 'SELL') return
      const sec = parseUtcSeconds(t.candle_time || t.log_time)
      if (sec == null || sec < first - 300 || sec > last + 300) return
      const time = snap(sec)
      const dedupeKey = `${time}_${t.signal}`
      if (seen.has(dedupeKey)) return
      seen.add(dedupeKey)
      const isBuy = t.signal === 'BUY'
      markers.push({
        time,
        position: isBuy ? 'belowBar' : 'aboveBar',
        color: isBuy ? COLORS.buy : COLORS.sell,
        shape: isBuy ? 'arrowUp' : 'arrowDown',
        text: t.signal,
      })
    })

    markers.sort((a, b) => a.time - b.time)
    candleSeries.setMarkers(markers)
  }, [sortedCandles, tradesData, showSignals])

  useEffect(() => {
    const main = mainApiRef.current
    if (!main?.chart || !sortedCandles.length) return

    const active = new Set(activeIndicators)
    const expectedKeys = new Set()

    active.forEach((id) => {
      const ind = getIndicatorById(id)
      if (!ind) return

      const data = calcIndicator(sortedCandles, ind)

      if (ind.type === 'bb') {
        ;['upper', 'middle', 'lower'].forEach((part) => {
          const key = `${id}_${part}`
          expectedKeys.add(key)
          const points = data[part]
          if (!indicatorSeriesRef.current[key]) {
            indicatorSeriesRef.current[key] = main.chart.addLineSeries({
              color: ind.color,
              lineWidth: part === 'middle' ? 2 : 1,
              lineStyle: part === 'middle' ? 0 : 2,
              priceLineVisible: false,
              lastValueVisible: false,
              crosshairMarkerVisible: false,
            })
          }
          indicatorSeriesRef.current[key].setData(points)
        })
        return
      }

      if (ind.pane === 'rsi' && rsiApiRef.current) {
        const key = `rsi_${id}`
        expectedKeys.add(key)
        if (!indicatorSeriesRef.current[key]) {
          indicatorSeriesRef.current[key] = rsiApiRef.current.chart.addLineSeries({
            color: ind.color,
            lineWidth: 2,
            priceLineVisible: false,
            lastValueVisible: true,
          })
        }
        indicatorSeriesRef.current[key].setData(data.main)
        return
      }

      expectedKeys.add(id)
      if (!indicatorSeriesRef.current[id]) {
        indicatorSeriesRef.current[id] = main.chart.addLineSeries({
          color: ind.color,
          lineWidth: ind.lineWidth || 2,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        })
      }
      indicatorSeriesRef.current[id].setData(data.main)
    })

    Object.entries(indicatorSeriesRef.current).forEach(([key, series]) => {
      if (expectedKeys.has(key)) return
      const chart = key.startsWith('rsi_') ? rsiApiRef.current?.chart : main.chart
      try {
        chart?.removeSeries(series)
      } catch {
        // removed
      }
      delete indicatorSeriesRef.current[key]
    })
  }, [sortedCandles, activeIndicators, hasRsi])

  return (
    <div className="relative h-full flex flex-col" style={{ backgroundColor: COLORS.bg }}>
      {/* Slim chart tools — indicators + view controls */}
      <div
        className="flex items-center justify-between gap-3 px-5 sm:px-8 h-11 border-b shrink-0"
        style={{ backgroundColor: COLORS.panel, borderColor: COLORS.border }}
      >
        <div className="flex items-center gap-1 overflow-x-auto">
          <button
            type="button"
            onClick={() => setShowIndicators((v) => !v)}
            className={`px-2.5 py-1 text-[10px] uppercase tracking-[0.14em] transition ${
              showIndicators || activeIndicators.size > 0
                ? 'text-accent-gold'
                : 'text-surface-muted hover:text-ink-soft'
            }`}
          >
            Indicators
            {activeIndicators.size > 0 && (
              <span className="ml-1.5 text-accent-gold/70 tabular-nums">{activeIndicators.size}</span>
            )}
          </button>
          <button
            type="button"
            onClick={() => (onToggleSignals ? onToggleSignals() : null)}
            className={`px-2.5 py-1 text-[10px] uppercase tracking-[0.14em] transition inline-flex items-center gap-1.5 ${
              showSignals ? 'text-accent-gold' : 'text-surface-muted hover:text-ink-soft'
            }`}
            title="Entry markers & SL / TP"
          >
            Signals
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{ backgroundColor: showSignals ? COLORS.live : COLORS.border }}
            />
          </button>
          <button
            type="button"
            onClick={() => setShowLevels((v) => !v)}
            className={`px-2.5 py-1 text-[10px] uppercase tracking-[0.14em] transition ${
              showLevels ? 'text-ink-soft' : 'text-surface-muted hover:text-ink-soft'
            }`}
            title="Live / bid / ask lines"
          >
            Levels
          </button>
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <button
            type="button"
            onClick={() => zoomLogical(0.75)}
            className="w-8 h-8 text-surface-muted hover:text-ink transition text-sm"
            title="Zoom in"
          >
            +
          </button>
          <button
            type="button"
            onClick={() => zoomLogical(1.35)}
            className="w-8 h-8 text-surface-muted hover:text-ink transition text-sm"
            title="Zoom out"
          >
            −
          </button>
          <button
            type="button"
            onClick={fitChart}
            className="px-2.5 h-8 text-[10px] uppercase tracking-[0.14em] text-surface-muted hover:text-accent-gold transition"
            title="Fit"
          >
            Fit
          </button>
        </div>
      </div>

      {showIndicators && (
        <div
          className="absolute left-5 sm:left-8 top-12 z-20 w-72 max-h-80 overflow-hidden flex flex-col border border-white/[0.08] shadow-soft"
          style={{ backgroundColor: COLORS.panel }}
        >
          <div className="p-3 border-b" style={{ borderColor: COLORS.border }}>
            <input
              type="text"
              placeholder="Search studies…"
              value={indicatorSearch}
              onChange={(e) => setIndicatorSearch(e.target.value)}
              className="field text-xs py-2"
            />
          </div>
          <div className="overflow-y-auto flex-1 p-1.5">
            {filteredCatalog.map((group) => (
              <div key={group.category} className="mb-2">
                <div className="section-label px-2 py-1.5">{group.category}</div>
                {group.items.map((ind) => {
                  const on = activeIndicators.has(ind.id)
                  return (
                    <button
                      key={ind.id}
                      type="button"
                      onClick={() => toggleIndicator(ind.id)}
                      className="w-full flex items-center gap-2.5 px-2 py-1.5 text-xs text-left transition hover:bg-white/[0.03]"
                      style={{ color: COLORS.text }}
                    >
                      <span
                        className="w-3 h-3 border shrink-0 flex items-center justify-center"
                        style={{
                          borderColor: on ? COLORS.live : COLORS.border,
                          backgroundColor: on ? COLORS.live : 'transparent',
                        }}
                      >
                        {on && <span className="text-[#1a1610] text-[8px] font-bold">✓</span>}
                      </span>
                      <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: ind.color }} />
                      <span className="font-light">{ind.name}</span>
                    </button>
                  )
                })}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex-1 min-h-0 flex flex-col">
        <div className="px-5 sm:px-8 pt-4 pb-2 shrink-0" style={{ backgroundColor: COLORS.bg }}>
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <span className="font-display text-lg tracking-[0.04em] text-ink">Gold</span>
                <span className="text-[11px] uppercase tracking-[0.14em] text-surface-muted">
                  {timeframe} · XM
                </span>
                {livePriceDisplay != null && (
                  <span className="text-base font-medium tabular-nums text-accent-gold">
                    {formatPrice(livePriceDisplay)}
                  </span>
                )}
              </div>
              {lastBar && (
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-2 text-[11px] text-surface-muted">
                  <span>
                    O{' '}
                    <span className="text-ink-soft tabular-nums" ref={(el) => { ohlcRefs.current.o = el }}>
                      {formatPrice(lastBar.open)}
                    </span>
                  </span>
                  <span>
                    H{' '}
                    <span className="text-ink-soft tabular-nums" ref={(el) => { ohlcRefs.current.h = el }}>
                      {formatPrice(lastBar.high)}
                    </span>
                  </span>
                  <span>
                    L{' '}
                    <span className="text-ink-soft tabular-nums" ref={(el) => { ohlcRefs.current.l = el }}>
                      {formatPrice(lastBar.low)}
                    </span>
                  </span>
                  <span>
                    C{' '}
                    <span className="text-ink-soft tabular-nums" ref={(el) => { ohlcRefs.current.c = el }}>
                      {formatPrice(lastBar.close)}
                    </span>
                  </span>
                  {ohlcChange != null && (
                    <span
                      ref={(el) => { ohlcRefs.current.chg = el }}
                      className="tabular-nums"
                      style={{ color: ohlcChange >= 0 ? COLORS.buy : COLORS.sell }}
                    >
                      {ohlcChange >= 0 ? '+' : ''}
                      {formatPrice(ohlcChange)}
                      {ohlcPct != null && ` (${ohlcPct >= 0 ? '+' : ''}${ohlcPct.toFixed(2)}%)`}
                    </span>
                  )}
                </div>
              )}
            </div>

            <div className="flex items-stretch gap-2">
              <div className="flex flex-col justify-center px-3.5 py-1.5 min-w-[5.5rem] bg-accent-sell-deep/90">
                <span className="text-[9px] uppercase tracking-[0.16em] text-[#f5f0e8]/70">Sell</span>
                <span className="text-sm font-medium tabular-nums text-[#f5f0e8]">{formatPrice(bid)}</span>
              </div>
              <div className="flex items-center px-2 text-[10px] tabular-nums text-surface-muted border border-white/[0.06]">
                {spread.toFixed(1)}
              </div>
              <div className="flex flex-col justify-center px-3.5 py-1.5 min-w-[5.5rem] bg-accent-buy-deep/90">
                <span className="text-[9px] uppercase tracking-[0.16em] text-[#f5f0e8]/70">Buy</span>
                <span className="text-sm font-medium tabular-nums text-[#f5f0e8]">{formatPrice(ask)}</span>
              </div>
            </div>
          </div>

          {showSignals && signal && (signal.SIGNAL === 'BUY' || signal.SIGNAL === 'SELL') && (
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-3 text-[11px] text-surface-muted">
              <span
                className="font-medium uppercase tracking-[0.12em]"
                style={{ color: signal.SIGNAL === 'BUY' ? COLORS.buy : COLORS.sell }}
              >
                {signal.SIGNAL}
                {signal.DEMAND_TF && signal.SUPPLY_TF ? ` · ${signal.DEMAND_TF}/${signal.SUPPLY_TF}` : ''}
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-3 h-px" style={{ backgroundColor: COLORS.live }} />
                Entry {formatPrice(signal.ENTRY)}
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-3 h-px" style={{ backgroundColor: COLORS.sell }} />
                SL {formatPrice(signal.SL)}
              </span>
              {signal.TP1 != null && (
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-3 h-px" style={{ backgroundColor: COLORS.tp1 }} />
                  TP1 {formatPrice(signal.TP1)}
                </span>
              )}
              <span className="inline-flex items-center gap-1.5">
                <span className="w-3 h-px" style={{ backgroundColor: COLORS.tp2 }} />
                TP2 {formatPrice(signal.TP)}
              </span>
              {signal.RR && <span>RR {signal.RR}</span>}
            </div>
          )}
        </div>

        <div
          ref={mainChartRef}
          className="w-full flex-1 min-h-[280px]"
          style={{ backgroundColor: COLORS.bg }}
        />
        {hasRsi && (
          <div
            ref={rsiChartRef}
            className="w-full border-t shrink-0"
            style={{ backgroundColor: COLORS.bg, borderColor: COLORS.border, height: 72 }}
          />
        )}
        <div
          ref={volChartRef}
          className="w-full border-t shrink-0"
          style={{ backgroundColor: COLORS.bg, borderColor: COLORS.border, height: 80 }}
        />
      </div>
    </div>
  )
}

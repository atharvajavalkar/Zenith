import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'
import { ColorType, createChart } from 'lightweight-charts'
import api from '../services/api'
import { ALL_INDICATORS, TV_INDICATOR_CATALOG, getIndicatorById } from '../data/tvIndicators'
import { calcIndicator } from '../utils/chartIndicators'

const TF_OPTIONS = [
  { value: '1m', label: '1m' },
  { value: '5m', label: '5m' },
  { value: '15m', label: '15m' },
  { value: '1h', label: '1h' },
  { value: '4h', label: '4h' },
]

const TF_LABEL = { '1m': '1', '5m': '5', '15m': '15', '1h': '60', '4h': '240' }

const COLORS = {
  bg: '#D1D4DC',
  panel: '#FFFFFF',
  text: '#131722',
  muted: '#787B86',
  border: '#E0E3EB',
  crosshair: '#9598A1',
  upCandle: '#FFFFFF',
  downCandle: '#000000',
  wick: '#000000',
  buyBtn: '#2962FF',
  sellBtn: '#F23645',
  live: '#2962FF',
  bid: '#F23645',
  ask: '#089981',
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
    color: c.close >= c.open ? 'rgba(0,0,0,0.25)' : 'rgba(0,0,0,0.45)',
  }
}

/** TradingView-like scroll/zoom — wheel zooms time, drag axes for H/V scale */
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

function ToolbarIcon({ children, title, onClick, active }) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className={`w-9 h-9 flex items-center justify-center rounded transition ${
        active ? 'bg-[#E3F2FD] text-[#2962FF]' : 'hover:bg-[#F0F3FA] text-[#787B86]'
      }`}
    >
      {children}
    </button>
  )
}

export default function PriceChart({ candles, timeframe, onTimeframeChange, signal }) {
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
  const [showSignals, setShowSignals] = useState(true)
  const [indicatorSearch, setIndicatorSearch] = useState('')
  const [activeIndicators, setActiveIndicators] = useState(() => new Set())
  const [livePriceDisplay, setLivePriceDisplay] = useState(null)

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
      chg.style.color = delta >= 0 ? '#089981' : '#F23645'
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

  const scrollChart = (bars) => {
    const chart = mainApiRef.current?.chart
    if (!chart) return
    userInteractedRef.current = true
    const ts = chart.timeScale()
    const range = ts.getVisibleLogicalRange()
    if (!range) return
    ts.setVisibleLogicalRange({ from: range.from + bars, to: range.to + bars })
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
        fontFamily: '-apple-system, BlinkMacSystemFont, "Trebuchet MS", Roboto, sans-serif',
        fontSize: 11,
      },
      grid: { vertLines: { visible: false }, horzLines: { visible: false } },
      crosshair: {
        mode: 1,
        vertLine: { color: COLORS.crosshair, width: 1, style: 2, labelBackgroundColor: COLORS.text },
        horzLine: { color: COLORS.crosshair, width: 1, style: 2, labelBackgroundColor: COLORS.text },
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
      borderUpColor: COLORS.wick,
      borderDownColor: COLORS.wick,
      wickUpColor: COLORS.wick,
      wickDownColor: COLORS.wick,
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

  // RSI pane — created when needed
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
        fontSize: 11,
      },
      grid: { vertLines: { visible: false }, horzLines: { visible: false } },
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

  // Live bid / ask / mid price lines — update in place (no remove/recreate)
  useEffect(() => {
    const { candleSeries } = mainApiRef.current || {}
    if (!candleSeries) return

    const upsertLine = (key, price, color, title, lineWidth, lineStyle) => {
      if (price == null) {
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

    upsertLine('live', livePrice, COLORS.live, 'Live', 2, 0)
    upsertLine('bid', bid, COLORS.bid, 'Bid', 1, 2)
    upsertLine('ask', ask, COLORS.ask, 'Ask', 1, 2)
  }, [livePrice, bid, ask])

  // Signal levels — Entry / SL / TP1 / TP2 of the current setup
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
          lineWidth: 2,
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
    if (upsertSignalLine('entry', signal.ENTRY, COLORS.live, `${signal.SIGNAL} ENTRY`, 0)) activeKeys.add('entry')
    if (upsertSignalLine('sl', signal.SL, COLORS.sellBtn, 'SL', 2)) activeKeys.add('sl')
    if (upsertSignalLine('tp1', signal.TP1, '#FB8C00', 'TP1', 2)) activeKeys.add('tp1')
    if (upsertSignalLine('tp2', signal.TP, COLORS.ask, 'TP2', 2)) activeKeys.add('tp2')

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

  // Trade entry markers — BUY (green up) / SELL (red down)
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
        color: isBuy ? COLORS.ask : COLORS.sellBtn,
        shape: isBuy ? 'arrowUp' : 'arrowDown',
        text: t.signal,
      })
    })

    markers.sort((a, b) => a.time - b.time)
    candleSeries.setMarkers(markers)
  }, [sortedCandles, tradesData, showSignals])

  // Manual indicators
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

  const tfDisplay = TF_LABEL[timeframe] || timeframe

  return (
    <div className="rounded-lg overflow-hidden border border-[#E0E3EB] shadow-sm relative" style={{ backgroundColor: COLORS.panel }}>
      <div className="flex items-center justify-between px-2 h-10 border-b border-[#E0E3EB]" style={{ backgroundColor: COLORS.panel }}>
        <div className="flex items-center gap-1">
          <button type="button" className="flex items-center gap-1.5 px-2 py-1 rounded hover:bg-[#F0F3FA]">
            <span className="text-sm font-bold" style={{ color: COLORS.text }}>XAUUSD</span>
            <span className="text-[#787B86] text-xs">▾</span>
          </button>
          <div className="flex items-center gap-0.5 ml-1">
            {TF_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() => onTimeframeChange(opt.value)}
                className="min-w-[32px] h-7 px-2 text-xs font-medium rounded transition"
                style={
                  timeframe === opt.value
                    ? { backgroundColor: '#F0F3FA', color: COLORS.text }
                    : { color: COLORS.muted }
                }
              >
                {opt.label}
              </button>
            ))}
          </div>
          <span className="mx-1 text-[#E0E3EB]">|</span>
          <button
            type="button"
            onClick={() => setShowIndicators((v) => !v)}
            className="px-2 py-1 text-xs rounded hover:bg-[#F0F3FA] flex items-center gap-1"
            style={{ color: showIndicators ? COLORS.buyBtn : COLORS.muted }}
          >
            Indicators
            {activeIndicators.size > 0 && (
              <span className="bg-[#2962FF] text-white text-[10px] px-1.5 rounded-full">
                {activeIndicators.size}
              </span>
            )}
          </button>
          <button
            type="button"
            onClick={() => setShowSignals((v) => !v)}
            className="px-2 py-1 text-xs rounded hover:bg-[#F0F3FA] flex items-center gap-1"
            style={{ color: showSignals ? COLORS.ask : COLORS.muted }}
            title="Toggle BUY/SELL markers and SL/TP levels"
          >
            Signals
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: showSignals ? COLORS.ask : COLORS.border }}
            />
          </button>
        </div>
        <div className="flex items-center gap-1">
          <button type="button" className="px-3 py-1 text-xs font-medium rounded" style={{ color: COLORS.muted }}>
            Save
          </button>
          <button type="button" className="px-3 py-1 text-xs font-semibold rounded text-white" style={{ backgroundColor: COLORS.buyBtn }}>
            Trade
          </button>
        </div>
      </div>

      {showIndicators && (
        <div
          className="absolute left-12 top-10 z-20 w-72 max-h-80 overflow-hidden rounded-lg border border-[#E0E3EB] shadow-xl flex flex-col"
          style={{ backgroundColor: COLORS.panel }}
        >
          <div className="p-2 border-b border-[#E0E3EB]">
            <input
              type="text"
              placeholder="Search indicators..."
              value={indicatorSearch}
              onChange={(e) => setIndicatorSearch(e.target.value)}
              className="w-full px-2 py-1.5 text-xs rounded border border-[#E0E3EB] outline-none focus:border-[#2962FF]"
              style={{ color: COLORS.text }}
            />
            <p className="text-[10px] mt-1.5" style={{ color: COLORS.muted }}>
              TradingView-style indicators — toggle to overlay on chart
            </p>
          </div>
          <div className="overflow-y-auto flex-1 p-1">
            {filteredCatalog.map((group) => (
              <div key={group.category} className="mb-2">
                <div className="text-[10px] font-semibold uppercase px-2 py-1" style={{ color: COLORS.muted }}>
                  {group.category}
                </div>
                {group.items.map((ind) => (
                  <button
                    key={ind.id}
                    type="button"
                    onClick={() => toggleIndicator(ind.id)}
                    className="w-full flex items-center gap-2 px-2 py-1.5 text-xs rounded hover:bg-[#F0F3FA] text-left"
                    style={{ color: COLORS.text }}
                  >
                    <span
                      className="w-3 h-3 rounded border flex items-center justify-center shrink-0"
                      style={{
                        borderColor: activeIndicators.has(ind.id) ? COLORS.buyBtn : COLORS.border,
                        backgroundColor: activeIndicators.has(ind.id) ? COLORS.buyBtn : 'transparent',
                      }}
                    >
                      {activeIndicators.has(ind.id) && <span className="text-white text-[8px]">✓</span>}
                    </span>
                    <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: ind.color }} />
                    {ind.name}
                  </button>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex">
        <div className="flex flex-col items-center py-1 border-r border-[#E0E3EB] w-11 shrink-0" style={{ backgroundColor: COLORS.panel }}>
          <ToolbarIcon title="Crosshair (drag chart to pan)">✛</ToolbarIcon>
          <ToolbarIcon title="Trend line">╱</ToolbarIcon>
          <ToolbarIcon title="Fibonacci">φ</ToolbarIcon>
          <ToolbarIcon title="Brush">✎</ToolbarIcon>
          <ToolbarIcon title="Text">T</ToolbarIcon>
          <ToolbarIcon title="Measure">⌗</ToolbarIcon>
          <ToolbarIcon title="Zoom in (scroll wheel also zooms time)" onClick={() => zoomLogical(0.75)}>+</ToolbarIcon>
          <ToolbarIcon title="Zoom out" onClick={() => zoomLogical(1.35)}>−</ToolbarIcon>
          <ToolbarIcon title="Fit chart (double-click axis to reset scale)" onClick={fitChart}>⊡</ToolbarIcon>
          <ToolbarIcon title="Pan left" onClick={() => scrollChart(-12)}>‹</ToolbarIcon>
          <ToolbarIcon title="Pan right" onClick={() => scrollChart(12)}>›</ToolbarIcon>
        </div>

        <div className="flex-1 min-w-0">
          <div className="px-3 pt-2 pb-1" style={{ backgroundColor: COLORS.bg }}>
            <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
              <span className="text-sm font-semibold" style={{ color: COLORS.text }}>Gold Spot / U.S. Dollar</span>
              <span className="text-xs" style={{ color: COLORS.muted }}>· {tfDisplay} · XM</span>
              {livePriceDisplay != null && (
                <span className="text-sm font-bold tabular-nums ml-1" style={{ color: COLORS.live }}>
                  {formatPrice(livePriceDisplay)}
                </span>
              )}
            </div>
            {lastBar && (
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-1 text-xs" style={{ color: COLORS.text }}>
                <span>O<span className="ml-1 font-medium" ref={(el) => { ohlcRefs.current.o = el }}>{formatPrice(lastBar.open)}</span></span>
                <span>H<span className="ml-1 font-medium" ref={(el) => { ohlcRefs.current.h = el }}>{formatPrice(lastBar.high)}</span></span>
                <span>L<span className="ml-1 font-medium" ref={(el) => { ohlcRefs.current.l = el }}>{formatPrice(lastBar.low)}</span></span>
                <span>C<span className="ml-1 font-medium" ref={(el) => { ohlcRefs.current.c = el }}>{formatPrice(lastBar.close)}</span></span>
                {ohlcChange != null && (
                  <span
                    ref={(el) => { ohlcRefs.current.chg = el }}
                    className={ohlcChange >= 0 ? 'text-[#089981]' : 'text-[#F23645]'}
                  >
                    {ohlcChange >= 0 ? '+' : ''}{formatPrice(ohlcChange)}
                    {ohlcPct != null && ` (${ohlcPct >= 0 ? '+' : ''}${ohlcPct.toFixed(2)}%)`}
                  </span>
                )}
              </div>
            )}
            <div className="flex items-center gap-2 mt-2 mb-1">
              <button type="button" className="flex flex-col items-start px-3 py-1 rounded text-white text-xs min-w-[90px]" style={{ backgroundColor: COLORS.sellBtn }}>
                <span className="opacity-80 text-[10px]">SELL</span>
                <span className="font-semibold tabular-nums">{formatPrice(bid)}</span>
              </button>
              <span className="text-[10px] px-1.5 py-0.5 rounded border border-[#E0E3EB] tabular-nums" style={{ color: COLORS.muted }}>
                {spread.toFixed(1)}
              </span>
              <button type="button" className="flex flex-col items-start px-3 py-1 rounded text-white text-xs min-w-[90px]" style={{ backgroundColor: COLORS.buyBtn }}>
                <span className="opacity-80 text-[10px]">BUY</span>
                <span className="font-semibold tabular-nums">{formatPrice(ask)}</span>
              </button>
            </div>
            {showSignals && signal && (signal.SIGNAL === 'BUY' || signal.SIGNAL === 'SELL') && (
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-1 text-[11px]" style={{ color: COLORS.muted }}>
                <span className="font-semibold" style={{ color: signal.SIGNAL === 'BUY' ? COLORS.ask : COLORS.sellBtn }}>
                  {signal.SIGNAL}
                  {signal.DEMAND_TF && signal.SUPPLY_TF ? ` · ${signal.DEMAND_TF}/${signal.SUPPLY_TF}` : ''}
                  {signal.MOVEMENT_5M ? ` · 5M ${signal.MOVEMENT_5M}` : ''}
                </span>
                <span className="inline-flex items-center gap-1">
                  <span className="w-2.5 h-0.5" style={{ backgroundColor: COLORS.live }} /> Entry {formatPrice(signal.ENTRY)}
                </span>
                <span className="inline-flex items-center gap-1">
                  <span className="w-2.5 h-0.5" style={{ backgroundColor: COLORS.sellBtn }} /> SL {formatPrice(signal.SL)}
                </span>
                {signal.TP1 != null && (
                  <span className="inline-flex items-center gap-1">
                    <span className="w-2.5 h-0.5" style={{ backgroundColor: '#FB8C00' }} /> TP1 {formatPrice(signal.TP1)}
                  </span>
                )}
                <span className="inline-flex items-center gap-1">
                  <span className="w-2.5 h-0.5" style={{ backgroundColor: COLORS.ask }} /> TP2 {formatPrice(signal.TP)}
                </span>
                {signal.RR && <span>RR {signal.RR}</span>}
              </div>
            )}
          </div>

          <div
            ref={mainChartRef}
            className="w-full"
            style={{ backgroundColor: COLORS.bg, height: 'min(52vh, 480px)', minHeight: 320 }}
          />
          {hasRsi && (
            <div
              ref={rsiChartRef}
              className="w-full border-t border-[#E0E3EB]"
              style={{ backgroundColor: COLORS.bg, height: 72 }}
            />
          )}
          <div
            ref={volChartRef}
            className="w-full border-t border-[#E0E3EB]"
            style={{ backgroundColor: COLORS.bg, height: 88 }}
          />

          <div className="flex items-center justify-between px-3 py-1.5 border-t border-[#E0E3EB] text-[11px]" style={{ backgroundColor: COLORS.panel, color: COLORS.muted }}>
            <div className="flex gap-3 items-center">
              <button type="button" onClick={fitChart} className="hover:text-[#131722] transition font-medium" style={{ color: COLORS.buyBtn }}>
                Fit
              </button>
              {['1D', '5D', '1M', '3M', '6M', 'YTD', '1Y', 'All'].map((r) => (
                <button key={r} type="button" className="hover:text-[#131722] transition">{r}</button>
              ))}
            </div>
            <span className="hidden sm:inline">Wheel = zoom · Drag = pan · Drag price/time axis = scale</span>
            <span>{new Date().toLocaleTimeString()}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

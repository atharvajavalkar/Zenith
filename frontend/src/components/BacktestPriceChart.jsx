/**
 * BacktestPriceChart
 *
 * Static candlestick chart for backtest results.
 * Shows the full OHLCV price history with trade entry/exit markers overlaid.
 * Uses lightweight-charts (same library as the live Chart page).
 */
import { useEffect, useRef } from 'react'
import { createChart, CrosshairMode, LineStyle } from 'lightweight-charts'

export default function BacktestPriceChart({ candles = [], height = 420 }) {
  const containerRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current || candles.length === 0) return

    const chart = createChart(containerRef.current, {
      layout: {
        background: { color: '#0d1117' },
        textColor: '#9ca3af',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: '#1a2030', style: LineStyle.Dotted },
        horzLines: { color: '#1a2030', style: LineStyle.Dotted },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#2d3748', scaleMarginTop: 0.1, scaleMarginBottom: 0.1 },
      timeScale: {
        borderColor: '#2d3748',
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 5,
      },
      width:  containerRef.current.clientWidth,
      height: height,
    })

    // ── Candlestick series ────────────────────────────────────────────────────
    const candleSeries = chart.addCandlestickSeries({
      upColor:         '#22c55e',
      downColor:       '#ef4444',
      borderUpColor:   '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor:     '#4ade80',
      wickDownColor:   '#f87171',
    })
    candleSeries.setData(candles)

    // Fit the full range
    chart.timeScale().fitContent()

    const ro = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    })
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      chart.remove()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candles])

  if (candles.length === 0) return null

  return (
    <div className="zcard overflow-hidden">
      <div className="px-4 pt-3 pb-1 flex items-center justify-between">
        <span className="text-sm font-semibold text-gray-300">XAUUSD · Historical Price</span>
        <span className="text-xs text-surface-muted">{candles.length.toLocaleString()} candles</span>
      </div>
      <div ref={containerRef} style={{ height: `${height}px` }} />
    </div>
  )
}

import { memo, useEffect, useRef } from 'react'

const WIDGET_SCRIPT = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js'

const INTERVAL_MAP = {
  '1m': '1',
  '5m': '5',
  '15m': '15',
  '1h': '60',
  '4h': '240',
  D: 'D',
}

/**
 * Official TradingView Advanced Chart embed — full toolbar, drawing tools,
 * and the same Indicators library (80+ studies) as the TradingView app.
 * @see https://www.tradingview.com/widget-docs/widgets/charts/advanced-chart/
 */
function TradingViewWidget({
  symbol = 'XM:XAUUSD',
  interval = '1m',
  theme = 'light',
  timezone = 'Etc/UTC',
}) {
  const containerRef = useRef(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    container.innerHTML = ''

    const widget = document.createElement('div')
    widget.className = 'tradingview-widget-container__widget'
    widget.style.cssText = 'height:100%;width:100%'

    const copyright = document.createElement('div')
    copyright.className = 'tradingview-widget-copyright text-[10px] py-1 text-center'
    copyright.style.color = '#787B86'
    copyright.innerHTML =
      '<a href="https://www.tradingview.com/" rel="noopener noreferrer" target="_blank" style="color:#2962FF">Chart by TradingView</a>'

    const script = document.createElement('script')
    script.type = 'text/javascript'
    script.src = WIDGET_SCRIPT
    script.async = true
    script.innerHTML = JSON.stringify({
      autosize: true,
      symbol,
      interval: INTERVAL_MAP[interval] || interval,
      timezone,
      theme,
      style: '1',
      locale: 'en',
      enable_publishing: false,
      allow_symbol_change: true,
      calendar: false,
      support_host: 'https://www.tradingview.com',
      hide_top_toolbar: false,
      hide_side_toolbar: false,
      hide_legend: false,
      withdateranges: true,
      save_image: true,
      backgroundColor: theme === 'light' ? '#D1D4DC' : '#131722',
      gridColor: theme === 'light' ? 'rgba(0, 0, 0, 0.06)' : 'rgba(255, 255, 255, 0.06)',
      show_popup_button: true,
      popup_width: '1000',
      popup_height: '650',
      details: true,
      hotlist: true,
    })

    container.appendChild(widget)
    container.appendChild(copyright)
    container.appendChild(script)

    return () => {
      container.innerHTML = ''
    }
  }, [symbol, interval, theme, timezone])

  return (
    <div
      ref={containerRef}
      className="tradingview-widget-container h-full w-full"
      style={{ height: '100%', width: '100%' }}
    />
  )
}

export default memo(TradingViewWidget)

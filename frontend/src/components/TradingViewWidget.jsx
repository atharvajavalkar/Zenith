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
 * TradingView Advanced Chart — dark theme to match Zenith desk.
 */
function TradingViewWidget({
  symbol = 'XM:XAUUSD',
  interval = '5m',
  theme = 'dark',
  timezone = 'Etc/UTC',
}) {
  const containerRef = useRef(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    container.innerHTML = ''

    const widget = document.createElement('div')
    widget.className = 'tradingview-widget-container__widget'
    widget.style.cssText = 'height:calc(100% - 22px);width:100%'

    const copyright = document.createElement('div')
    copyright.className = 'text-[10px] py-1 text-center tracking-wide'
    copyright.style.color = '#8a857c'
    copyright.innerHTML =
      '<a href="https://www.tradingview.com/" rel="noopener noreferrer" target="_blank" style="color:#c9a66b">Chart by TradingView</a>'

    const script = document.createElement('script')
    script.type = 'text/javascript'
    script.src = WIDGET_SCRIPT
    script.async = true
    script.innerHTML = JSON.stringify({
      autosize: true,
      symbol,
      interval: INTERVAL_MAP[interval] || interval,
      timezone,
      theme: 'dark',
      style: '1',
      locale: 'en',
      enable_publishing: false,
      allow_symbol_change: true,
      calendar: false,
      support_host: 'https://www.tradingview.com',
      hide_top_toolbar: false,
      hide_side_toolbar: true,
      hide_legend: false,
      withdateranges: true,
      save_image: false,
      backgroundColor: '#0e0e0c',
      gridColor: 'rgba(255, 255, 255, 0.04)',
      show_popup_button: false,
      details: false,
      hotlist: false,
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
      style={{ height: '100%', width: '100%', backgroundColor: '#0e0e0c' }}
    />
  )
}

export default memo(TradingViewWidget)

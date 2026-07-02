import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import Layout from '../components/Layout'
import PriceChart from '../components/PriceChart'
import TradingViewWidget from '../components/TradingViewWidget'
import api from '../services/api'
import { useTimezone } from '../contexts/TimezoneContext'

const TIMEFRAMES = [
  { id: '1m', label: '1m' },
  { id: '5m', label: '5m' },
  { id: '15m', label: '15m' },
  { id: '1h', label: '1H' },
  { id: '4h', label: '4H' },
  { id: 'D', label: 'D' },
]

export default function ChartPage() {
  const [interval, setInterval] = useState('1m')
  const [theme, setTheme] = useState('light')
  const [view, setView] = useState('zenith')
  const [zenithTf, setZenithTf] = useState('5m')
  const { chartTimezone } = useTimezone()

  const { data: candlesData } = useQuery({
    queryKey: ['chart-candles', zenithTf],
    queryFn: async () =>
      (await api.get('/api/chart/candles', { params: { tf: zenithTf, limit: 400 } })).data,
    refetchInterval: 5000,
    enabled: view === 'zenith',
  })

  const { data: signal } = useQuery({
    queryKey: ['signal-latest'],
    queryFn: async () => (await api.get('/api/signals/latest')).data,
    refetchInterval: 3000,
    enabled: view === 'zenith',
  })

  const viewBtn = (id, label) => (
    <button
      type="button"
      onClick={() => setView(id)}
      className="px-3 py-1 text-xs font-medium transition"
      style={view === id ? { backgroundColor: '#F0F3FA', color: '#131722' } : { color: '#787B86' }}
    >
      {label}
    </button>
  )

  return (
    <Layout variant="chart">
      <div className="flex flex-col h-[calc(100vh-56px)]">
        <div
          className="flex flex-wrap items-center justify-between gap-3 px-3 py-2 border-b shrink-0"
          style={{ backgroundColor: '#FFFFFF', borderColor: '#E0E3EB' }}
        >
          <div className="flex items-center gap-3">
            <Link
              to="/dashboard"
              className="text-xs px-3 py-1.5 rounded border hover:bg-[#F0F3FA] transition"
              style={{ color: '#131722', borderColor: '#E0E3EB' }}
            >
              ← Dashboard
            </Link>
            <span className="text-sm font-bold" style={{ color: '#131722' }}>
              XAUUSD · Gold Spot / U.S. Dollar
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex rounded overflow-hidden border" style={{ borderColor: '#E0E3EB' }}>
              {viewBtn('zenith', 'Zenith Signals')}
              {viewBtn('tradingview', 'TradingView')}
            </div>
            {view === 'tradingview' && (
              <>
                <div className="flex rounded overflow-hidden border" style={{ borderColor: '#E0E3EB' }}>
                  {TIMEFRAMES.map((tf) => (
                    <button
                      key={tf.id}
                      type="button"
                      onClick={() => setInterval(tf.id)}
                      className="px-2.5 py-1 text-xs font-medium transition"
                      style={
                        interval === tf.id
                          ? { backgroundColor: '#F0F3FA', color: '#131722' }
                          : { color: '#787B86' }
                      }
                    >
                      {tf.label}
                    </button>
                  ))}
                </div>
                <button
                  type="button"
                  onClick={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
                  className="px-2.5 py-1 text-xs rounded border"
                  style={{ borderColor: '#E0E3EB', color: '#787B86' }}
                >
                  {theme === 'light' ? 'Dark' : 'Light'}
                </button>
              </>
            )}
          </div>
        </div>

        <p className="text-[11px] px-3 py-1.5 shrink-0" style={{ backgroundColor: '#F0F3FA', color: '#787B86' }}>
          {view === 'zenith' ? (
            <>
              <strong style={{ color: '#131722' }}>Zenith Signals</strong> — 5M cadence with live BUY/SELL entry markers
              at 3M/5M supply & demand zones, plus SL, TP1 and TP2 levels. Toggle with the{' '}
              <strong style={{ color: '#131722' }}>Signals</strong> button on the chart.
            </>
          ) : (
            <>
              Use the <strong style={{ color: '#131722' }}>Indicators</strong> button in the chart toolbar to add any
              TradingView study (EMA, RSI, MACD, Volume, etc.).
            </>
          )}
        </p>

        {view === 'zenith' ? (
          <div className="flex-1 min-h-0 overflow-auto p-3" style={{ backgroundColor: '#FFFFFF' }}>
            <PriceChart
              candles={candlesData?.candles || []}
              timeframe={zenithTf}
              onTimeframeChange={setZenithTf}
              signal={signal}
            />
          </div>
        ) : (
          <div className="flex-1 min-h-0">
            <TradingViewWidget
              symbol="XM:XAUUSD"
              interval={interval}
              theme={theme}
              timezone={chartTimezone}
            />
          </div>
        )}
      </div>
    </Layout>
  )
}

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
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

const ZENITH_TFS = [
  { id: '1m', label: '1m' },
  { id: '5m', label: '5m' },
  { id: '15m', label: '15m' },
  { id: '1h', label: '1H' },
  { id: '4h', label: '4H' },
]

function SegBtn({ active, onClick, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`px-3.5 py-1.5 text-[10px] uppercase tracking-[0.16em] font-medium transition ${
        active
          ? 'text-ink border-b border-accent-gold'
          : 'text-surface-muted hover:text-ink-soft border-b border-transparent'
      }`}
    >
      {children}
    </button>
  )
}

function Pill({ active, onClick, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`min-w-[2rem] px-2 py-1 text-[11px] font-medium tracking-wide transition ${
        active ? 'text-accent-gold' : 'text-surface-muted hover:text-ink-soft'
      }`}
    >
      {children}
    </button>
  )
}

export default function ChartPage() {
  const [interval, setInterval] = useState('5m')
  const [view, setView] = useState('zenith')
  const [zenithTf, setZenithTf] = useState('5m')
  const [showSignals, setShowSignals] = useState(true)
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

  const sourceLabel =
    signal?.DATA_SOURCE === 'xm'
      ? `${signal?.SYMBOL || 'GOLD'} · XM`
      : `${signal?.SYMBOL || 'XAUUSD'} · Desk`

  return (
    <Layout variant="chart">
      <div className="flex flex-col h-[calc(100vh-4.25rem)]">
        <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3 px-5 sm:px-8 py-3 border-b border-white/[0.06] bg-surface-raised/60 shrink-0">
          <div className="flex items-center gap-5 min-w-0">
            <div>
              <p className="section-label mb-0.5">Chart</p>
              <h1 className="font-display text-xl tracking-[0.06em] text-ink leading-none truncate">
                {sourceLabel}
              </h1>
            </div>
            <div className="hidden sm:flex items-center gap-0.5 border-l border-white/[0.06] pl-5">
              <SegBtn active={view === 'zenith'} onClick={() => setView('zenith')}>
                Zenith
              </SegBtn>
              <SegBtn active={view === 'tradingview'} onClick={() => setView('tradingview')}>
                TradingView
              </SegBtn>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3 sm:gap-4">
            <div className="sm:hidden flex items-center gap-0.5">
              <SegBtn active={view === 'zenith'} onClick={() => setView('zenith')}>
                Zenith
              </SegBtn>
              <SegBtn active={view === 'tradingview'} onClick={() => setView('tradingview')}>
                TV
              </SegBtn>
            </div>

            <div className="flex items-center gap-0.5">
              {(view === 'zenith' ? ZENITH_TFS : TIMEFRAMES).map((tf) => (
                <Pill
                  key={tf.id}
                  active={(view === 'zenith' ? zenithTf : interval) === tf.id}
                  onClick={() =>
                    view === 'zenith' ? setZenithTf(tf.id) : setInterval(tf.id)
                  }
                >
                  {tf.label}
                </Pill>
              ))}
            </div>

            {view === 'zenith' && (
              <button
                type="button"
                onClick={() => setShowSignals((v) => !v)}
                className={`inline-flex items-center gap-2 px-3 py-1.5 text-[10px] uppercase tracking-[0.16em] border transition ${
                  showSignals
                    ? 'border-accent-gold/40 text-accent-gold'
                    : 'border-white/[0.08] text-surface-muted hover:text-ink-soft'
                }`}
                title="Show entry markers and SL / TP levels"
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    showSignals ? 'bg-accent-gold' : 'bg-surface-border'
                  }`}
                />
                Signals
              </button>
            )}
          </div>
        </div>

        {view === 'zenith' ? (
          <div className="flex-1 min-h-0 bg-surface">
            <PriceChart
              candles={candlesData?.candles || []}
              timeframe={zenithTf}
              onTimeframeChange={setZenithTf}
              signal={signal}
              showSignals={showSignals}
              onToggleSignals={() => setShowSignals((v) => !v)}
            />
          </div>
        ) : (
          <div className="flex-1 min-h-0 bg-surface">
            <TradingViewWidget
              symbol="XM:XAUUSD"
              interval={interval}
              theme="dark"
              timezone={chartTimezone}
            />
          </div>
        )}
      </div>
    </Layout>
  )
}

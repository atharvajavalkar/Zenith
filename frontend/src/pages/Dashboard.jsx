import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import Layout from '../components/Layout'
import { Link } from 'react-router-dom'
import AutopsyBoard from '../components/AutopsyBoard'
import OpportunityBoard from '../components/OpportunityBoard'
import LotDesk from '../components/LotDesk'
import SignalCard from '../components/SignalCard'
import TimezoneSelector from '../components/TimezoneSelector'
import TradePanel, { useExecuteTrade } from '../components/TradePanel'
import TradeStatsBar from '../components/TradeStatsBar'
import { useAuth } from '../contexts/AuthContext'
import api from '../services/api'
import { connectSignalWs } from '../services/ws'
export default function Dashboard() {
  const { accessToken } = useAuth()
  const queryClient = useQueryClient()
  const [signal, setSignal] = useState(null)
  const [wsConnected, setWsConnected] = useState(false)
  const [executeResult, setExecuteResult] = useState('')
  const [executeError, setExecuteError] = useState('')
  const executeMutation = useExecuteTrade()

  const { data: tradeStatus } = useQuery({
    queryKey: ['trade-status'],
    queryFn: async () => (await api.get('/api/trades/status')).data,
    refetchInterval: 5000,
  })

  const isLive = tradeStatus?.trade_mode === 'live' || tradeStatus?.trade_mode === 'demo'
  const symbol = isLive ? (tradeStatus?.symbol || 'GOLD') : 'XAU/USD'
  const marketClosed = tradeStatus?.market_open === false

  const handleExecute = () => {
    setExecuteResult('')
    setExecuteError('')
    executeMutation.mutate(undefined, {
      onSuccess: (data) => setExecuteResult(data?.result?.message || 'Trade executed'),
      onError: (err) =>
        setExecuteError(err.response?.data?.detail || err.message || 'Trade failed'),
    })
  }

  const { data: latestData, isLoading: latestLoading } = useQuery({
    queryKey: ['signal-latest'],
    queryFn: async () => {
      const { data } = await api.get('/api/signals/latest')
      return data
    },
    retry: 3,
    retryDelay: 5000,
  })

  useEffect(() => {
    if (latestData) setSignal(latestData)
  }, [latestData])

  useEffect(() => {
    if (!accessToken) return

    const disconnect = connectSignalWs(accessToken, {
      onMessage: (msg) => {
        if (msg.type === 'signal_update' && msg.signal) {
          setSignal(msg.signal)
          queryClient.invalidateQueries({ queryKey: ['trade-history'] })
          queryClient.invalidateQueries({ queryKey: ['trade-summary'] })
          queryClient.invalidateQueries({ queryKey: ['live-tick'] })
        }
        if (msg.type === 'trade_executed' || msg.type === 'trade_closed') {
          queryClient.invalidateQueries({ queryKey: ['trade-history'] })
          queryClient.invalidateQueries({ queryKey: ['trade-summary'] })
          queryClient.invalidateQueries({ queryKey: ['live-tick'] })
          queryClient.invalidateQueries({ queryKey: ['trade-autopsy'] })
        }
        setWsConnected(true)
      },
      onClose: () => setWsConnected(false),
    })

    return disconnect
  }, [accessToken, queryClient])

  return (
    <Layout>
      <div className="flex flex-wrap items-end justify-between gap-6">
        <div>
          <p className="section-label mb-2">Desk</p>
          <h1 className="page-title">Dashboard</h1>
          <p className="page-subtitle">
            {marketClosed
              ? (tradeStatus?.market_message || 'XM market is closed — the desk pauses until the next session.')
              : isLive
                ? (tradeStatus?.market_status === 'crypto_session'
                    ? (tradeStatus?.market_message
                        || 'Crypto open 24/7 · BTCUSD · ETHUSD · FX/metals closed')
                    : 'GOLD · CHF · JPY · NAS · UKOIL · AUD · BTC · ETH — live pair brains')
                : 'Paper mode — multi-instrument via TradingView'}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {!marketClosed && tradeStatus?.market_status === 'crypto_session' && (
            <div className="status-chip text-accent-gold border-accent-gold/25">
              Crypto session
            </div>
          )}
          {marketClosed && (
            <div className="status-chip text-accent-gold border-accent-gold/25">
              {tradeStatus?.market_next_open
                ? `Closed · opens ${tradeStatus.market_next_open}`
                : 'Market closed'}
            </div>
          )}
          <div className="status-chip">
            <span
              className={`inline-block w-1.5 h-1.5 rounded-full ${
                wsConnected ? 'bg-accent-buy animate-pulse-soft' : 'bg-surface-muted'
              }`}
            />
            {wsConnected ? 'Connected' : 'Connecting'}
          </div>
        </div>
      </div>

      <TradeStatsBar />

      <section className="space-y-3">
        <div className="section-label">Lot desk</div>
        <LotDesk />
      </section>

      <section className="space-y-3">
        <div className="section-label">Universe setups</div>
        <OpportunityBoard />
      </section>

      <section className="space-y-3">
        <div className="section-label">Primary signal (GOLD)</div>
        <SignalCard signal={latestLoading && !signal ? null : signal} />
      </section>

      <TradePanel
        signal={signal}
        onExecute={handleExecute}
        executePending={executeMutation.isPending}
        executeResult={executeResult}
        executeError={executeError}
      />

      <section className="space-y-3">
        <div className="section-label">Trade autopsy</div>
        <AutopsyBoard />
      </section>

      <div className="flex flex-wrap items-center justify-between gap-4 py-1">
        <Link
          to="/world"
          className="text-sm font-light text-surface-muted hover:text-accent-gold transition tracking-wide"
        >
          Open world gold desk →
        </Link>
        <div className="flex-1 min-w-[280px] max-w-xl">
          <TimezoneSelector compact />
        </div>
      </div>
    </Layout>
  )
}

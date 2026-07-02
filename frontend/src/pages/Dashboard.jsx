import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import Layout from '../components/Layout'
import { Link } from 'react-router-dom'
import BiasPanel from '../components/BiasPanel'
import LearningPanel from '../components/LearningPanel'
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
        }
        setWsConnected(true)
      },
      onClose: () => setWsConnected(false),
    })

    return disconnect
  }, [accessToken, queryClient])

  return (
    <Layout>
      <div className="flex flex-wrap items-center justify-between gap-3 animate-fade-in-up">
        <div>
          <h1 className="text-2xl font-bold text-gray-50 tracking-tight">Trading Dashboard</h1>
          <p className="text-sm text-surface-muted mt-0.5">
            Live XAU/USD signals, bias and automated execution
          </p>
        </div>
        <div
          className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium ${
            wsConnected
              ? 'border-accent-buy/30 bg-accent-buy/10 text-accent-buy'
              : 'border-surface-border bg-surface-card text-surface-muted'
          }`}
        >
          <span
            className={`inline-block w-2 h-2 rounded-full ${
              wsConnected ? 'bg-accent-buy animate-pulse-soft' : 'bg-gray-500'
            }`}
          />
          {wsConnected ? 'Live updates connected' : 'Connecting…'}
        </div>
      </div>

      <div className="animate-fade-in-up" style={{ animationDelay: '40ms' }}>
        <TradeStatsBar />
      </div>

      <div className="animate-fade-in-up" style={{ animationDelay: '80ms' }}>
        <BiasPanel />
      </div>

      <div className="animate-fade-in-up" style={{ animationDelay: '120ms' }}>
        <LearningPanel signal={signal} />
      </div>

      <section className="animate-fade-in-up" style={{ animationDelay: '160ms' }}>
        <div className="section-eyebrow mb-2">Live Setup</div>
        <SignalCard signal={latestLoading && !signal ? null : signal} />
      </section>

      <div className="animate-fade-in-up" style={{ animationDelay: '200ms' }}>
        <TradePanel
          signal={signal}
          onExecute={handleExecute}
          executePending={executeMutation.isPending}
          executeResult={executeResult}
          executeError={executeError}
        />
      </div>

      <Link
        to="/chart"
        className="group block rounded-2xl overflow-hidden border border-surface-border bg-surface-card shadow-card transition-all duration-300 hover:-translate-y-0.5 hover:border-accent-gold/30 hover:shadow-card-hover animate-fade-in-up"
        style={{ animationDelay: '240ms' }}
      >
        <div className="flex items-center justify-between gap-4 px-5 py-4">
          <div className="flex items-center gap-4">
            <span className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-gold-gradient text-surface text-xl">
              📈
            </span>
            <div>
              <h3 className="font-semibold text-lg text-gray-100">TradingView Chart</h3>
              <p className="text-sm mt-0.5 text-surface-muted">
                Full chart with official indicators, drawing tools &amp; timeframes
              </p>
            </div>
          </div>
          <span className="px-4 py-2 rounded-lg text-sm font-semibold text-surface bg-gold-gradient shrink-0 transition-transform group-hover:translate-x-0.5">
            Open Chart →
          </span>
        </div>
      </Link>

      <div className="animate-fade-in-up" style={{ animationDelay: '280ms' }}>
        <TimezoneSelector />
      </div>
    </Layout>
  )
}

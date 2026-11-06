import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../services/api'
import { useTimezone } from '../contexts/TimezoneContext'
import { formatInTimezone } from '../utils/formatTime'

const TRADES_PER_PAGE = 15
const TRADE_MODE_OPTIONS = [
  { value: 'paper', label: 'Paper (simulated)' },
  { value: 'demo', label: 'XM Demo' },
  { value: 'live', label: 'XM Live (real money)' },
]

function isBrokerMode(mode) {
  return mode === 'demo' || mode === 'live'
}
function normalizeRR(rr_type) {
  if (!rr_type) return null
  // Handle "1:3.0" → "1:3"
  const parts = rr_type.split(':')
  if (parts.length !== 2) return rr_type
  return `1:${parseInt(parts[1], 10)}`
}

function exitStatusLabel(row) {
  if (row.pnl_status === 'open') return { text: 'Open', cls: 'text-gray-400' }
  if (row.pnl_status === 'partial') return { text: 'TP1 ▶', cls: 'text-yellow-300' }
  if (row.pnl_status === 'partial2') return { text: 'TP2 ▶', cls: 'text-yellow-400' }
  if (row.pnl_status === 'partial3') return { text: 'TP3 ▶', cls: 'text-yellow-500' }
  if (row.pnl_status === 'closed_tp') {
    // Cash wins — never show TP✓ on a red close
    if (row.pnl != null && Number(row.pnl) < 0) {
      return { text: 'Closed', cls: 'text-gray-400' }
    }
    const rr = normalizeRR(row.rr_type) || '1:2'
    const n = parseInt(rr.split(':')[1] || '2', 10)
    return { text: `TP${n} ✓`, cls: 'text-accent-buy font-bold' }
  }
  if (row.pnl_status === 'closed_sl') {
    if (row.outcome === 'breakeven') return { text: 'BE ≈', cls: 'text-accent-gold' }
    if (row.outcome === 'win') return { text: 'Trail ✓', cls: 'text-accent-buy' }
    return { text: 'SL ✗', cls: 'text-accent-sell' }
  }
  if (row.pnl_status === 'closed_broker') {
    if (row.pnl != null && Number(row.pnl) > 0) {
      return { text: 'Closed ✓', cls: 'text-accent-buy' }
    }
    if (row.pnl != null && Number(row.pnl) < 0) {
      return { text: 'Closed ✗', cls: 'text-accent-sell' }
    }
    return { text: 'Closed', cls: 'text-gray-400' }
  }
  if (row.pnl_status === 'closed_manual') return { text: 'Manual', cls: 'text-gray-400' }
  return { text: row.pnl_status ?? '—', cls: 'text-gray-400' }
}

function rrBadge(rr_type) {
  const norm = normalizeRR(rr_type)
  if (!norm) return <span className="text-gray-600">—</span>
  const n = parseInt(norm.split(':')[1] || '2', 10)
  const colorMap = {
    2: 'text-gray-400 border-gray-600 bg-gray-800/40',
    3: 'text-yellow-300 border-yellow-600/70 bg-yellow-900/20',
    4: 'text-emerald-300 border-emerald-600/70 bg-emerald-900/20',
  }
  const cls = colorMap[n] || colorMap[2]
  return (
    <span className={`inline-block text-[10px] font-bold px-1.5 py-px rounded border ${cls} tabular-nums`}>
      {norm}
    </span>
  )
}

function slTrailShort(row) {
  const s = row.sl_trail_status
  if (!s || s === '—') return { text: '—', cls: 'text-gray-600' }
  if (s === 'Closed') return { text: 'Closed', cls: 'text-gray-500' }
  if (s.startsWith('Fixed')) return { text: s.replace('Fixed @ ', ''), cls: 'text-gray-500' }
  if (s.startsWith('Breakeven')) return { text: 'BE ' + (row.sl ?? ''), cls: 'text-orange-300' }
  if (s.startsWith('Trail')) {
    // "Trail 50% @ 4105.50 (+1.33) [TP1]"
    const match = s.match(/@ ([\d.]+)/)
    const stage = s.match(/\[(TP\d)\]/)
    const price = match ? match[1] : ''
    const lbl = stage ? ` [${stage[1]}]` : ''
    return { text: `▲${price}${lbl}`, cls: 'text-accent-gold' }
  }
  return { text: s, cls: 'text-gray-400' }
}

function fmtPx(n, symbol) {
  if (n == null || !Number.isFinite(Number(n))) return '—'
  const s = String(symbol || '').toUpperCase()
  const digits = s.includes('JPY') ? 3 : (s === 'GOLD' || s === 'XAUUSD' || s.startsWith('BTC') ? 2 : 5)
  return Number(n).toFixed(digits)
}

function fmtPnl(n) {
  if (n == null || !Number.isFinite(Number(n))) return '—'
  const v = Number(n)
  const abs = Math.abs(v)
  const text = abs >= 100 ? v.toFixed(0) : abs >= 10 ? v.toFixed(1) : v.toFixed(2)
  return `${v >= 0 ? '+' : ''}${text}`
}

function pnlClass(row) {
  if (row.outcome === 'breakeven') return 'text-accent-gold'
  if (row.outcome === 'win' || (row.pnl != null && row.pnl > 0)) return 'text-accent-buy'
  if (row.outcome === 'loss' || (row.pnl != null && row.pnl < 0)) return 'text-accent-sell'
  return 'text-gray-400'
}

export function useExecuteTrade() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async () => (await api.post('/api/trades/execute')).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trade-history'] })
      queryClient.invalidateQueries({ queryKey: ['trade-status'] })
      queryClient.invalidateQueries({ queryKey: ['trade-summary'] })
    },
  })
}

export default function TradePanel({ signal, onExecute, executePending, executeResult, executeError }) {
  const queryClient = useQueryClient()
  const { displayTimezone } = useTimezone()
  const [page, setPage] = useState(1)

  const { data: status } = useQuery({
    queryKey: ['trade-status'],
    queryFn: async () => (await api.get('/api/trades/status')).data,
    refetchInterval: 5000,
  })

  const { data: settings } = useQuery({
    queryKey: ['trade-settings'],
    queryFn: async () => (await api.get('/api/trades/settings')).data,
    refetchInterval: 5000,
  })

  const updateSettings = useMutation({
    mutationFn: async (payload) =>
      (await api.patch('/api/trades/settings', payload)).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trade-settings'] })
      queryClient.invalidateQueries({ queryKey: ['trade-status'] })
      queryClient.invalidateQueries({ queryKey: ['trade-summary'] })
    },
  })

  const stopAll = useMutation({
    mutationFn: async () => (await api.post('/api/trades/stop-all')).data,
    onSuccess: () => {
      setPage(1)
      queryClient.invalidateQueries({ queryKey: ['trade-history'] })
      queryClient.invalidateQueries({ queryKey: ['trade-status'] })
      queryClient.invalidateQueries({ queryKey: ['trade-settings'] })
      queryClient.invalidateQueries({ queryKey: ['trade-summary'] })
    },
  })

  const syncBroker = useMutation({
    mutationFn: async () => (await api.post('/api/trades/sync-broker')).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trade-history'] })
      queryClient.invalidateQueries({ queryKey: ['trade-status'] })
      queryClient.invalidateQueries({ queryKey: ['trade-summary'] })
    },
  })

  const arch = status?.architecture ?? settings?.architecture
  const execCadence =
    settings?.exec_cadence ??
    settings?.trade_cadence ??
    status?.exec_cadence ??
    status?.trade_cadence ??
    arch?.exec_cadence ??
    arch?.mode ??
    '5m'
  const zoneTf = settings?.zone_timeframes ?? status?.zone_timeframes ?? arch?.zone_timeframes ?? 'MTF'
  const cadenceLabel = `${execCadence}`

  const { data: history } = useQuery({
    queryKey: ['trade-history', page],
    queryFn: async () =>
      (
        await api.get('/api/trades/history', {
          params: { page, per_page: TRADES_PER_PAGE },
        })
      ).data,
    refetchInterval: 5000,
  })

  const total = history?.total ?? 0
  const totalPages = history?.total_pages ?? 1
  const currentPage = history?.page ?? page

  useEffect(() => {
    if (totalPages > 0 && page > totalPages) {
      setPage(totalPages)
    }
  }, [page, totalPages])

  const rangeStart = total === 0 ? 0 : (currentPage - 1) * TRADES_PER_PAGE + 1
  const rangeEnd = Math.min(currentPage * TRADES_PER_PAGE, total)

  const formatTime = (value) => (value ? formatInTimezone(value, displayTimezone) : '—')

  const isBuyOrSell = signal?.SIGNAL === 'BUY' || signal?.SIGNAL === 'SELL'
  const marketClosed = status?.market_open === false
  const mode = settings?.trade_mode || status?.trade_mode || 'paper'
  const broker = (status?.broker || 'xm').toUpperCase()
  const brokerOk = status?.broker_configured
  const xmOpenCount = status?.open_positions?.length ?? 0

  return (
    <div className="zcard p-6 sm:p-8">
      <div className="flex flex-wrap items-start justify-between gap-6 mb-8">
        <div className="max-w-xl">
          <h3 className="panel-title">Auto trade</h3>
          <p className="panel-desc">
            {marketClosed ? (
              <>
                {status?.market_label || 'Market closed'}.{' '}
                {status?.market_next_open
                  ? `Trades resume ${status.market_next_open}.`
                  : 'Trades resume when XM opens (crypto stays 24/7).'}
              </>
            ) : status?.market_status === 'crypto_session' ? (
              <>
                Crypto session — BTC/ETH open now. FX & metals wait for XM.
              </>
            ) : mode === 'live' ? (
              <>
                Live desk — real XM account via MetaAPI. Real money at risk.
              </>
            ) : mode === 'demo' ? (
              <>
                Demo desk — XM demo · multi-instrument (see Lot desk for per-pair size).
              </>
            ) : (
              <>
                Paper desk — TradingView prices, simulated fills.
              </>
            )}
            {!marketClosed && (
              <> Per-pair scan TFs · TP1 books 50%, then trail.</>
            )}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <button
            type="button"
            disabled={stopAll.isPending}
            onClick={() => {
              if (window.confirm('Stop all trades? This closes every open position at market and pauses execution.')) {
                stopAll.mutate()
              }
            }}
            className="btn-danger-ghost"
          >
            {stopAll.isPending ? 'Stopping…' : 'Stop all'}
          </button>
          <button
            type="button"
            disabled={!isBuyOrSell || executePending || status?.trading_halted || marketClosed}
            onClick={onExecute}
            className={
              isBuyOrSell && !status?.trading_halted && !marketClosed
                ? signal.SIGNAL === 'BUY'
                  ? 'btn-buy'
                  : 'btn-sell'
                : 'btn-secondary opacity-40 cursor-not-allowed'
            }
          >
            {marketClosed
              ? 'Market closed'
              : status?.trading_halted
                ? 'Halted'
                : executePending
                  ? 'Executing…'
                  : isBuyOrSell
                    ? `Execute ${signal.SIGNAL}`
                    : 'Awaiting signal'}
          </button>
        </div>
      </div>

      {marketClosed && (
        <div className="mb-8 rounded-sm border border-accent-gold/25 bg-accent-gold/5 px-4 py-3 text-sm font-light text-ink-soft">
          <span className="text-accent-gold font-medium tracking-wide uppercase text-[10px] mr-2">
            Desk paused
          </span>
          {status?.market_message || 'XM gold market is closed for the weekend.'}
        </div>
      )}

      {arch && (
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 mb-8 gate-meta border-y border-white/[0.05] py-3">
          <span>Scan {zoneTf || cadenceLabel}</span>
          <span className="text-white/15">|</span>
          <span className={arch.ml_gate_enabled ? 'gate-meta-on' : 'gate-meta-off'}>
            ML {arch.ml_gate_enabled ? 'on' : 'off'}
          </span>
          <span className={arch.bias_gate_enabled ? 'gate-meta-on' : 'gate-meta-off'}>
            Bias {arch.bias_gate_enabled ? 'on' : 'off'}
          </span>
          {arch?.direction_lock_5m && <span>5M lock</span>}
          {arch.block_opposite_open && <span>No flip</span>}
        </div>
      )}

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-6 gap-y-5 mb-8 text-sm">
        <div className="control-cell">
          <div className="section-label mb-2">Trade mode</div>
          <select
            value={mode}
            disabled={updateSettings.isPending}
            onChange={(e) => {
              const next = e.target.value
              const accounts = status?.broker_accounts || {}
              if (next === 'demo' && accounts.demo === false) {
                const ok = window.confirm(
                  'METAAPI_ACCOUNT_ID_DEMO is not set in .env. Switch to Demo anyway?'
                )
                if (!ok) return
              }
              if (next === 'live' && accounts.live === false) {
                const ok = window.confirm(
                  'METAAPI_ACCOUNT_ID_LIVE is not set in .env. Live orders will fail until you add it. Switch anyway?'
                )
                if (!ok) return
              }
              if (next === 'live') {
                const ok = window.confirm(
                  'Switch to XM LIVE? Real money orders will be sent to your live MetaAPI account.'
                )
                if (!ok) return
              }
              if (next === 'demo') {
                const ok = window.confirm(
                  'Switch to XM DEMO? Orders go to your demo MetaAPI account (practice funds).'
                )
                if (!ok) return
              }
              updateSettings.mutate({ trade_mode: next })
            }}
            className={`w-full bg-transparent border-0 border-b rounded-none px-0 py-1.5 text-sm font-light focus:outline-none focus:border-accent-gold disabled:opacity-50 ${
              mode === 'live'
                ? 'border-red-400/50 text-red-300'
                : mode === 'demo'
                  ? 'border-accent-gold/50 text-accent-gold'
                  : 'border-white/15 text-ink-soft'
            }`}
          >
            {TRADE_MODE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <div className={`text-[10px] mt-2 tracking-wide ${brokerOk ? 'text-accent-buy' : 'text-accent-gold'}`}>
            {brokerOk
              ? isBrokerMode(mode)
                ? `${broker} ${mode.toUpperCase()} · ${xmOpenCount} open`
                : `${broker} accounts ready`
              : isBrokerMode(mode)
                ? `${broker} ${mode} not configured`
                : `${broker} not configured`}
          </div>
          {isBrokerMode(mode) && brokerOk && (
            <button
              type="button"
              disabled={syncBroker.isPending}
              onClick={() => syncBroker.mutate()}
              className="mt-2 text-[10px] uppercase tracking-nav text-surface-muted hover:text-accent-gold transition disabled:opacity-50"
            >
              {syncBroker.isPending ? 'Syncing…' : 'Sync XM'}
            </button>
          )}
        </div>
        <div className="control-cell">
          <div className="section-label mb-2">Auto-trade</div>
          <div className={status?.auto_trade_enabled && !status?.trading_halted ? 'text-accent-buy' : 'text-surface-muted'}>
            {status?.trading_halted ? 'Stopped' : status?.auto_trade_enabled ? 'On' : 'Off'}
          </div>
        </div>
        <div className="control-cell">
          <div className="section-label mb-2">Status</div>
          <div className="text-ink-soft text-xs font-light leading-snug">
            {status?.trading_halted ? 'Stopped — use Stop all / resume via settings' : status?.reason || '—'}
          </div>
        </div>
      </div>

      {syncBroker.isSuccess && syncBroker.data?.synced && (
        <div className="mb-4 text-sm text-accent-buy border-l-2 border-accent-buy/50 pl-3 py-1">
          XM sync: {syncBroker.data.broker_open} open · {syncBroker.data.matched} matched
          {syncBroker.data.imported ? ` · ${syncBroker.data.imported} imported` : ''}
          {syncBroker.data.pnl_synced
            ? ` · realised ${Number(syncBroker.data.realized_pnl ?? 0) >= 0 ? '+' : ''}$${Number(syncBroker.data.realized_pnl ?? 0).toFixed(2)} · floating ${Number(syncBroker.data.floating_pnl ?? 0) >= 0 ? '+' : ''}$${Number(syncBroker.data.floating_pnl ?? 0).toFixed(2)}`
            : ''}
        </div>
      )}
      {stopAll.isError && (
        <div className="mb-4 text-sm text-accent-sell border-l-2 border-accent-sell/50 pl-3 py-1">
          {stopAll.error?.response?.data?.detail || stopAll.error?.message || 'Failed to stop trades'}
        </div>
      )}
      {stopAll.data?.message && (
        <div className="mb-4 text-sm text-accent-gold border-l-2 border-accent-gold/50 pl-3 py-1">
          {stopAll.data.message}
        </div>
      )}
      {executeError && (
        <div className="mb-4 text-sm text-accent-sell border-l-2 border-accent-sell/50 pl-3 py-1">
          {executeError}
        </div>
      )}
      {executeResult && (
        <div className="mb-4 text-sm text-accent-buy border-l-2 border-accent-buy/50 pl-3 py-1">
          {executeResult}
        </div>
      )}

      <div className="overflow-x-auto -mx-1 px-1">
        <table className="w-full text-[12px] leading-snug">
          <thead>
            <tr className="text-left text-surface-muted border-b border-white/[0.06]">
              <th className="py-3 px-2.5 font-medium tracking-wide uppercase text-[9px] whitespace-nowrap">Time</th>
              <th className="py-3 px-2.5 font-medium tracking-wide uppercase text-[9px]">Symbol</th>
              <th className="py-3 px-2.5 font-medium tracking-wide uppercase text-[9px]">Dir</th>
              <th className="py-3 px-2.5 font-medium tracking-wide uppercase text-[9px]">RR</th>
              <th className="py-3 px-2.5 font-medium tracking-wide uppercase text-[9px] tabular-nums">Entry</th>
              <th className="py-3 px-2.5 font-medium tracking-wide uppercase text-[9px] tabular-nums">SL</th>
              <th className="py-3 px-2.5 font-medium tracking-wide uppercase text-[9px] whitespace-nowrap">Trail</th>
              <th className="py-3 px-2.5 font-medium tracking-wide uppercase text-[9px]">TP1</th>
              <th className="py-3 px-2.5 font-medium tracking-wide uppercase text-[9px]">TP2</th>
              <th className="py-3 px-2.5 font-medium tracking-wide uppercase text-[9px]">Final</th>
              <th className="py-3 px-2.5 font-medium tracking-wide uppercase text-[9px]">Now</th>
              <th className="py-3 px-2.5 font-medium tracking-wide uppercase text-[9px] text-right">P&L</th>
              <th className="py-3 px-2.5 font-medium tracking-wide uppercase text-[9px] whitespace-nowrap">Stage</th>
              <th className="py-3 px-2.5 font-medium tracking-wide uppercase text-[9px]">Note</th>
            </tr>
          </thead>
          <tbody>
            {history?.items?.length === 0 && (
              <tr>
                <td colSpan={14} className="py-8 text-center text-gray-500 text-xs font-light">
                  No trade history displayed — frontend reset only.
                </td>
              </tr>
            )}
            {history?.items?.map((row) => {
              const stage = exitStatusLabel(row)
              const trail = slTrailShort(row)
              const isOpen = ['open','partial','partial2','partial3'].includes(row.pnl_status)
              const sym = row.symbol || row.features?.symbol || row.features?.SYMBOL || 'GOLD'
              return (
                <tr
                  key={row.id}
                  className={`border-b border-white/[0.04] transition-colors duration-200 ${
                    isOpen ? 'bg-accent-buy/[0.04]' : 'hover:bg-white/[0.025]'
                  }`}
                >
                  <td className="py-2.5 px-2.5 text-gray-500 whitespace-nowrap tabular-nums text-[11px]">
                    {formatTime(row.log_time || row.candle_time)}
                  </td>
                  <td className="py-2.5 px-2.5 font-semibold text-accent-gold tracking-wide whitespace-nowrap">
                    {sym}
                  </td>
                  <td className={`py-2.5 px-2.5 font-semibold tracking-wide ${row.signal === 'BUY' ? 'text-accent-buy' : row.signal === 'SELL' ? 'text-accent-sell' : 'text-gray-400'}`}>
                    {row.signal === 'BUY' ? '▲' : row.signal === 'SELL' ? '▼' : '—'} {row.signal}
                  </td>
                  <td className="py-2.5 px-2.5">{rrBadge(row.rr_type)}</td>
                  <td className="py-2.5 px-2.5 tabular-nums text-gray-200">{fmtPx(row.entry, sym)}</td>
                  <td className="py-2.5 px-2.5 tabular-nums text-accent-sell/90">{fmtPx(row.sl, sym)}</td>
                  <td className="py-2.5 px-2.5 whitespace-nowrap text-[11px]" title={row.sl_trail_status ?? ''}>
                    <span className={trail.cls}>{trail.text}</span>
                  </td>
                  <td className="py-2.5 px-2.5 tabular-nums text-accent-buy/90">{fmtPx(row.tp1, sym)}</td>
                  <td className={`py-2.5 px-2.5 tabular-nums ${row.tp2 ? 'text-accent-buy/70' : 'text-gray-700'}`}>
                    {fmtPx(row.tp2, sym)}
                  </td>
                  <td className="py-2.5 px-2.5 tabular-nums text-accent-buy font-medium">{fmtPx(row.tp, sym)}</td>
                  <td className="py-2.5 px-2.5 tabular-nums text-gray-300">{fmtPx(row.current_price, sym)}</td>
                  <td className={`py-2.5 px-2.5 font-semibold tabular-nums text-right ${pnlClass(row)}`}>
                    {fmtPnl(row.pnl)}
                  </td>
                  <td className={`py-2.5 px-2.5 font-medium whitespace-nowrap ${stage.cls}`}>
                    {stage.text}
                  </td>
                  <td className="py-2.5 px-2.5 text-gray-500 max-w-[200px] text-[11px] font-light" title={row.message ?? ''}>
                    <span className="block truncate">{row.message ?? '—'}</span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {total > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-3 mt-4 pt-3 border-t border-surface-border text-sm">
          <div className="text-gray-500">
            Showing {rangeStart}–{rangeEnd} of {total} trades
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={currentPage <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="px-3 py-1.5 rounded border border-surface-border text-gray-300 hover:border-accent-gold/50 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            <span className="text-gray-400 tabular-nums px-2">
              Page {currentPage} of {totalPages}
            </span>
            <button
              type="button"
              disabled={currentPage >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              className="px-3 py-1.5 rounded border border-surface-border text-gray-300 hover:border-accent-gold/50 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

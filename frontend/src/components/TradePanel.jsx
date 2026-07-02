import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../services/api'
import { useTimezone } from '../contexts/TimezoneContext'
import { formatInTimezone } from '../utils/formatTime'

const TRADES_PER_PAGE = 15
const TIMEFRAME_OPTIONS = [
  { value: '1m', label: '1 min' },
  { value: '3m', label: '3 min' },
  { value: '5m', label: '5 min' },
]

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
    const rr = normalizeRR(row.rr_type) || '1:2'
    const n = parseInt(rr.split(':')[1] || '2', 10)
    return { text: `TP${n} ✓`, cls: 'text-accent-buy font-bold' }
  }
  if (row.pnl_status === 'closed_sl') {
    if (row.outcome === 'breakeven') return { text: 'BE ≈', cls: 'text-accent-gold' }
    if (row.outcome === 'win') return { text: 'Trail ✓', cls: 'text-accent-buy' }
    return { text: 'SL ✗', cls: 'text-accent-sell' }
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

  const arch = status?.architecture ?? settings?.architecture
  const execCadence = settings?.exec_cadence ?? settings?.trade_cadence ?? status?.exec_cadence ?? status?.trade_cadence ?? arch?.exec_cadence ?? arch?.mode ?? '5m'
  const zoneTf = settings?.zone_timeframes ?? status?.zone_timeframes ?? arch?.zone_timeframes ?? '3M/5M'
  const tradesPerBucket = settings?.max_trades_per_bucket ?? settings?.max_trades_per_minute ?? status?.max_trades_per_bucket ?? status?.max_trades_per_minute ?? 3
  const tradeInterval = status?.trade_interval_seconds ?? settings?.trade_interval_seconds
  const cadenceLabel = TIMEFRAME_OPTIONS.find((o) => o.value === execCadence)?.label ?? '5 min'
  const cadenceShort = execCadence.replace('m', '')

  const { data: history } = useQuery({
    queryKey: ['trade-history', page],
    queryFn: async () =>
      (await api.get('/api/trades/history', { params: { page, per_page: TRADES_PER_PAGE } })).data,
    refetchInterval: 2000,
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
  const mode = status?.trade_mode || 'paper'

  return (
    <div className="zcard p-5">
      <div className="flex flex-wrap items-start justify-between gap-4 mb-4">
        <div>
          <h3 className="font-semibold text-gray-100 text-lg flex items-center gap-2">
            <span className="inline-block h-4 w-1 rounded-full bg-gold-gradient" />
            Auto Trade
          </h3>
          <p className="text-xs text-gray-400 mt-1 max-w-lg">
            TradingView XM data is <strong className="text-gray-300">read-only prices</strong>.
            {' '}{cadenceLabel} cadence · {zoneTf} zones · 0.1 lot · TP1 books 50% then breakeven + 50% profit trail to TP2.
            {' '}Up to {tradesPerBucket} trade{tradesPerBucket === 1 ? '' : 's'} per {cadenceLabel} candle
            {tradeInterval ? ` (~${Math.round(tradeInterval)}s apart)` : ''}.
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
            className="px-5 py-3 rounded-lg font-bold text-base transition bg-red-900/80 border border-red-500/60 text-red-100 hover:bg-red-800 disabled:opacity-50"
          >
            {stopAll.isPending ? 'Stopping...' : 'Stop All Trades'}
          </button>
          <button
            type="button"
            disabled={!isBuyOrSell || executePending || status?.trading_halted}
            onClick={onExecute}
            className={`px-6 py-3 rounded-lg font-bold text-base transition shrink-0 ${
              isBuyOrSell && !status?.trading_halted
                ? signal.SIGNAL === 'BUY'
                  ? 'bg-accent-buy text-white hover:opacity-90'
                  : 'bg-accent-sell text-white hover:opacity-90'
                : 'bg-gray-700 text-gray-500 cursor-not-allowed'
            }`}
          >
            {status?.trading_halted
              ? 'Trading halted'
              : executePending
                ? 'Executing...'
                : isBuyOrSell
                  ? `Execute ${signal.SIGNAL} Now`
                  : 'Waiting for BUY/SELL signal'}
          </button>
        </div>
      </div>

      {arch && (
        <div className="flex flex-wrap gap-2 mb-4">
          <span className="text-xs px-2.5 py-1 rounded-full border border-accent-gold/50 text-accent-gold bg-accent-gold/10">
            {cadenceLabel} · {zoneTf} zones
          </span>
          {arch?.direction_lock_5m && (
            <span className="text-xs px-2.5 py-1 rounded-full border border-surface-border text-gray-400 bg-surface">
              5M direction lock
            </span>
          )}
          <span className={`text-xs px-2.5 py-1 rounded-full border ${
            arch.ml_gate_enabled
              ? 'border-accent-buy/40 text-accent-buy bg-accent-buy/10'
              : 'border-gray-600 text-gray-400 bg-surface'
          }`}>
            ML gate {arch.ml_gate_enabled ? 'ON' : 'OFF'}
          </span>
          <span className={`text-xs px-2.5 py-1 rounded-full border ${
            arch.bias_gate_enabled
              ? 'border-accent-buy/40 text-accent-buy bg-accent-buy/10'
              : 'border-gray-600 text-gray-400 bg-surface'
          }`}>
            Bias gate {arch.bias_gate_enabled ? 'ON' : 'OFF'}
          </span>
          {arch.block_opposite_open && (
            <span className="text-xs px-2.5 py-1 rounded-full border border-surface-border text-gray-400 bg-surface">
              No flip while opposite open
            </span>
          )}
        </div>
      )}

      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 mb-4 text-sm">
        <div className="bg-surface rounded-lg p-3 border border-surface-border">
          <div className="text-xs text-gray-500">Mode</div>
          <div className="text-accent-gold font-medium capitalize">{mode}</div>
        </div>
        <div className="bg-surface rounded-lg p-3 border border-surface-border">
          <div className="text-xs text-gray-500">Lot size</div>
          <div className="text-gray-100 font-medium tabular-nums">
            {status?.lot_size != null ? status.lot_size.toFixed(2) : '0.10'}
          </div>
          <div className="text-[10px] text-gray-500 mt-0.5">
            {status?.trade_units ?? 10} units · 1:2 RR
          </div>
        </div>
        <div className="bg-surface rounded-lg p-3 border border-surface-border">
          <div className="text-xs text-gray-500">Margin (0.1 lot)</div>
          <div className="text-accent-gold font-medium tabular-nums">
            {status?.margin_usd != null ? `$${status.margin_usd.toLocaleString()}` : '—'}
          </div>
          <div className="text-[10px] text-gray-500 mt-0.5">
            {status?.margin_leverage ? `1:${status.margin_leverage} lev` : '1:1000 lev'}
          </div>
        </div>
        <div className="bg-surface rounded-lg p-3 border border-surface-border">
          <div className="text-xs text-gray-500 mb-1.5">Timeframe</div>
          <select
            value={execCadence}
            disabled={updateSettings.isPending || !status?.auto_trade_enabled}
            onChange={(e) => updateSettings.mutate({ exec_cadence: e.target.value })}
            className="w-full bg-surface-card border border-surface-border rounded px-2 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-accent-gold disabled:opacity-50"
          >
            {TIMEFRAME_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <div className="bg-surface rounded-lg p-3 border border-surface-border">
          <div className="text-xs text-gray-500 mb-1.5">Number of trades</div>
          <select
            value={tradesPerBucket}
            disabled={updateSettings.isPending || !status?.auto_trade_enabled}
            onChange={(e) => updateSettings.mutate({ max_trades_per_minute: Number(e.target.value) })}
            className="w-full bg-surface-card border border-surface-border rounded px-2 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-accent-gold disabled:opacity-50"
          >
            {Array.from({ length: 11 }, (_, i) => i).map((n) => (
              <option key={n} value={n}>
                {n === 0 ? '0 — paused' : n}
              </option>
            ))}
          </select>
          <div className="text-[10px] text-gray-500 mt-1">per {cadenceLabel} candle</div>
        </div>
        <div className="bg-surface rounded-lg p-3 border border-surface-border">
          <div className="text-xs text-gray-500">Auto-trade</div>
          <div className={status?.auto_trade_enabled && !status?.trading_halted ? 'text-accent-buy font-medium' : 'text-gray-400'}>
            {status?.trading_halted
              ? 'STOPPED'
              : status?.auto_trade_enabled
                  ? tradesPerBucket === 0
                  ? `ON — paused (0/${cadenceShort}m)`
                  : `ON — ${tradesPerBucket}/${cadenceShort}m`
                : 'OFF'}
          </div>
        </div>
        <div className="bg-surface rounded-lg p-3 border border-surface-border col-span-2 sm:col-span-1">
          <div className="text-xs text-gray-500">Status</div>
          <div className="text-gray-200 text-xs">
            {status?.trading_halted
              ? 'All trades stopped — set number of trades to resume'
              : status?.reason || '—'}
          </div>
        </div>
      </div>

      {stopAll.isError && (
        <div className="mb-3 text-sm text-red-400 bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-2">
          {stopAll.error?.response?.data?.detail || stopAll.error?.message || 'Failed to stop trades'}
        </div>
      )}
      {stopAll.data?.message && (
        <div className="mb-3 text-sm text-orange-300 bg-orange-500/10 border border-orange-500/30 rounded-lg px-4 py-2">
          {stopAll.data.message}
        </div>
      )}
      {executeError && (
        <div className="mb-3 text-sm text-red-400 bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-2">
          {executeError}
        </div>
      )}
      {executeResult && (
        <div className="mb-3 text-sm text-accent-buy bg-accent-buy/10 border border-accent-buy/30 rounded-lg px-4 py-2">
          {executeResult}
        </div>
      )}

      <div className="overflow-x-auto rounded-lg border border-surface-border/40">
        <table className="w-full text-[11px] leading-tight">
          <thead>
            <tr className="text-left text-gray-500 bg-surface/60 border-b border-surface-border">
              <th className="py-2 px-2 font-medium whitespace-nowrap">Time</th>
              <th className="py-2 px-2 font-medium">Dir</th>
              <th className="py-2 px-2 font-medium">RR</th>
              <th className="py-2 px-2 font-medium">Margin</th>
              <th className="py-2 px-2 font-medium tabular-nums">Entry</th>
              <th className="py-2 px-2 font-medium tabular-nums">SL</th>
              <th className="py-2 px-2 font-medium whitespace-nowrap">SL Trail</th>
              <th className="py-2 px-2 font-medium">TP1</th>
              <th className="py-2 px-2 font-medium">TP2</th>
              <th className="py-2 px-2 font-medium">TP3</th>
              <th className="py-2 px-2 font-medium whitespace-nowrap">TP Final</th>
              <th className="py-2 px-2 font-medium">Now</th>
              <th className="py-2 px-2 font-medium">P&L</th>
              <th className="py-2 px-2 font-medium whitespace-nowrap">Stage</th>
              <th className="py-2 px-2 font-medium">Note</th>
            </tr>
          </thead>
          <tbody>
            {history?.items?.length === 0 && (
              <tr>
                <td colSpan={15} className="py-6 text-center text-gray-500 text-xs">
                  Waiting for the first {cadenceShort}M log row...
                </td>
              </tr>
            )}
            {history?.items?.map((row) => {
              const stage = exitStatusLabel(row)
              const trail = slTrailShort(row)
              const isOpen = ['open','partial','partial2','partial3'].includes(row.pnl_status)
              return (
                <tr key={row.id} className={`border-b border-surface-border/30 hover:bg-white/[0.02] transition-colors ${isOpen ? 'bg-accent-buy/[0.03]' : ''}`}>
                  {/* Time */}
                  <td className="py-1.5 px-2 text-gray-500 whitespace-nowrap tabular-nums">
                    {formatTime(row.log_time || row.candle_time)}
                  </td>
                  {/* Direction */}
                  <td className={`py-1.5 px-2 font-bold tracking-wide ${row.signal === 'BUY' ? 'text-accent-buy' : row.signal === 'SELL' ? 'text-accent-sell' : 'text-gray-400'}`}>
                    {row.signal === 'BUY' ? '▲' : row.signal === 'SELL' ? '▼' : '—'} {row.signal}
                  </td>
                  {/* RR badge */}
                  <td className="py-1.5 px-2">{rrBadge(row.rr_type)}</td>
                  {/* Margin */}
                  <td className="py-1.5 px-2 tabular-nums text-accent-gold">
                    {row.margin_usd != null ? `$${row.margin_usd}` : '—'}
                  </td>
                  {/* Entry */}
                  <td className="py-1.5 px-2 tabular-nums text-gray-200">{row.entry ?? '—'}</td>
                  {/* SL */}
                  <td className="py-1.5 px-2 tabular-nums text-accent-sell">{row.sl ?? '—'}</td>
                  {/* SL Trail */}
                  <td className="py-1.5 px-2 whitespace-nowrap" title={row.sl_trail_status ?? ''}>
                    <span className={trail.cls}>{trail.text}</span>
                  </td>
                  {/* TP1 */}
                  <td className="py-1.5 px-2 tabular-nums text-accent-buy">{row.tp1 ?? <span className="text-gray-700">—</span>}</td>
                  {/* TP2 */}
                  <td className={`py-1.5 px-2 tabular-nums ${row.tp2 ? 'text-accent-buy/75' : 'text-gray-700'}`}>
                    {row.tp2 ?? '—'}
                  </td>
                  {/* TP3 */}
                  <td className={`py-1.5 px-2 tabular-nums ${row.tp3 ? 'text-accent-buy/55' : 'text-gray-700'}`}>
                    {row.tp3 ?? '—'}
                  </td>
                  {/* TP Final */}
                  <td className="py-1.5 px-2 tabular-nums text-accent-buy font-medium">{row.tp ?? '—'}</td>
                  {/* Current price */}
                  <td className="py-1.5 px-2 tabular-nums text-gray-300">{row.current_price ?? '—'}</td>
                  {/* P&L */}
                  <td className={`py-1.5 px-2 font-semibold tabular-nums ${pnlClass(row)}`}>
                    {row.pnl != null ? (row.pnl >= 0 ? '+' : '') + row.pnl : '—'}
                  </td>
                  {/* Stage */}
                  <td className={`py-1.5 px-2 font-medium whitespace-nowrap ${stage.cls}`}>
                    {stage.text}
                  </td>
                  {/* Message */}
                  <td className="py-1.5 px-2 text-gray-500 max-w-[180px]" title={row.message ?? ''}>
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

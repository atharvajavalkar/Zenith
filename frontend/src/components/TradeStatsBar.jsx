import { useQuery } from '@tanstack/react-query'
import api from '../services/api'

function PnlDisplay({ value, label }) {
  const positive = value > 0
  const negative = value < 0
  return (
    <div className="zcard zcard-hover p-4 text-center">
      <div className="text-[11px] text-surface-muted uppercase tracking-wide">{label}</div>
      <div
        className={`text-lg font-bold mt-1 tabular-nums leading-tight break-words ${
          positive ? 'text-accent-buy' : negative ? 'text-accent-sell' : 'text-gray-200'
        }`}
      >
        {value > 0 ? '+' : ''}
        {Number(value).toFixed(2)}
      </div>
    </div>
  )
}

export default function TradeStatsBar() {
  const { data: status } = useQuery({
    queryKey: ['trade-status'],
    queryFn: async () => (await api.get('/api/trades/status')).data,
    refetchInterval: 5000,
  })

  const { data: stats } = useQuery({
    queryKey: ['trade-summary'],
    queryFn: async () => (await api.get('/api/trades/summary')).data,
    refetchInterval: 1000,
  })

  const { data: tick } = useQuery({
    queryKey: ['live-tick'],
    queryFn: async () => (await api.get('/api/signals/tick')).data,
    refetchInterval: 1000,
  })

  const livePrice = tick?.price
  const signal = tick?.signal
  const setupRrPct = tick?.rr_pct
  const avgOpenRrPct = stats?.avg_open_rr_pct
  const displayRrPct = setupRrPct ?? avgOpenRrPct
  const rrHealthy = displayRrPct != null && displayRrPct >= 150
  const lotSize = status?.lot_size ?? 0.1

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-10 gap-3">
      <div className="zcard zcard-hover p-4 text-center col-span-2 sm:col-span-1 ring-1 ring-accent-gold/20">
        <div className="text-[11px] text-surface-muted uppercase tracking-wide">XAUUSD Live</div>
        <div className="text-lg font-bold text-accent-gold mt-1 tabular-nums leading-tight break-words">
          {livePrice != null ? `$${Number(livePrice).toLocaleString()}` : '—'}
        </div>
        {signal && (
          <div
            className={`inline-flex items-center justify-center mt-1.5 px-2 py-0.5 rounded-full text-[11px] font-semibold ${
              signal === 'BUY'
                ? 'bg-accent-buy/15 text-accent-buy'
                : signal === 'SELL'
                  ? 'bg-accent-sell/15 text-accent-sell'
                  : 'bg-white/5 text-surface-muted'
            }`}
          >
            {signal}
          </div>
        )}
      </div>
      <div className="zcard zcard-hover p-4 text-center">
        <div className="text-[11px] text-surface-muted uppercase tracking-wide">Lot Size</div>
        <div className="text-lg font-bold text-gray-100 mt-1 tabular-nums leading-tight">{lotSize.toFixed(2)}</div>
        <div className="text-[11px] text-surface-muted mt-1">{status?.trade_units ?? 10} oz</div>
      </div>
      <div className="zcard zcard-hover p-4 text-center">
        <div className="text-[11px] text-surface-muted uppercase tracking-wide">Margin</div>
        <div className="text-lg font-bold text-accent-gold mt-1 tabular-nums leading-tight break-words">
          {status?.margin_usd != null ? `$${status.margin_usd.toLocaleString()}` : '—'}
        </div>
        <div className="text-[11px] text-surface-muted mt-1">
          {lotSize.toFixed(1)} lot · 1:{status?.margin_leverage ?? 1000}
        </div>
      </div>
      <div className="zcard zcard-hover p-4 text-center">
        <div className="text-[11px] text-surface-muted uppercase tracking-wide">Trades Executed</div>
        <div className="text-lg font-bold text-gray-100 mt-1 tabular-nums leading-tight">{stats?.executed ?? 0}</div>
        {stats?.open_trades > 0 && (
          <div className="text-[11px] text-surface-muted mt-1">{stats.open_trades} open</div>
        )}
      </div>
      <PnlDisplay value={stats?.realized_pnl ?? 0} label="Realised Net P&L" />
      <PnlDisplay value={stats?.unrealized_pnl ?? 0} label="Live P&L" />
      <div className="zcard zcard-hover p-4 text-center">
        <div className="text-[11px] text-surface-muted uppercase tracking-wide">Reward / Risk</div>
        <div
          className={`text-lg font-bold mt-1 tabular-nums leading-tight ${
            displayRrPct == null
              ? 'text-gray-400'
              : rrHealthy
                ? 'text-accent-buy'
                : 'text-accent-sell'
          }`}
        >
          {displayRrPct != null ? `${displayRrPct}%` : '—'}
        </div>
        {setupRrPct != null && tick?.risk_usd != null && tick?.reward_usd != null ? (
          <div className="text-[11px] text-surface-muted mt-1 tabular-nums">
            ${tick.risk_usd} risk → ${tick.reward_usd} reward
          </div>
        ) : avgOpenRrPct != null ? (
          <div className="text-[11px] text-surface-muted mt-1">{stats.open_trades} open avg</div>
        ) : null}
        {tick?.live_r != null && (
          <div
            className={`text-[11px] font-semibold mt-0.5 tabular-nums ${
              tick.live_r > 0 ? 'text-accent-buy' : tick.live_r < 0 ? 'text-accent-sell' : 'text-gray-500'
            }`}
          >
            {tick.live_r > 0 ? '+' : ''}
            {tick.live_r}R live
          </div>
        )}
      </div>
      <div className="zcard zcard-hover p-4 text-center">
        <div className="text-[11px] text-surface-muted uppercase tracking-wide">Trades Won</div>
        <div className="text-lg font-bold text-accent-buy mt-1 tabular-nums leading-tight">{stats?.wins ?? 0}</div>
      </div>
      <div className="zcard zcard-hover p-4 text-center ring-1 ring-accent-gold/25">
        <div className="text-[11px] text-surface-muted uppercase tracking-wide">Breakeven</div>
        <div className="text-lg font-bold text-accent-gold mt-1 tabular-nums leading-tight">{stats?.breakevens ?? 0}</div>
        <div className="text-[11px] text-surface-muted mt-1">neither win nor loss</div>
      </div>
      <div className="zcard zcard-hover p-4 text-center">
        <div className="text-[11px] text-surface-muted uppercase tracking-wide">Trades Lost</div>
        <div className="text-lg font-bold text-accent-sell mt-1 tabular-nums leading-tight">{stats?.losses ?? 0}</div>
      </div>
    </div>
  )
}

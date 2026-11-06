import { useQuery } from '@tanstack/react-query'
import api from '../services/api'

function fmtMoney(n, decimals = 2) {
  if (n == null || !Number.isFinite(Number(n))) return '—'
  return `$${Number(n).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })}`
}

function StatTile({ label, value, sub, valueClassName = 'text-ink' }) {
  return (
    <div className="stat-tile">
      <div className="text-[10px] uppercase tracking-label text-surface-muted truncate">{label}</div>
      <div className={`text-[1.35rem] font-display font-medium mt-2 tabular-nums leading-none ${valueClassName}`}>
        {value}
      </div>
      {sub ? <div className="text-[11px] font-light text-surface-muted/80 mt-2 truncate">{sub}</div> : null}
    </div>
  )
}

function PnlTile({ value, label, sub }) {
  const num = Number(value)
  const positive = num > 0
  const negative = num < 0
  const display = Number.isFinite(num)
    ? `${num > 0 ? '+' : ''}${fmtMoney(num)}`
    : '—'
  return (
    <StatTile
      label={label}
      value={display}
      sub={sub}
      valueClassName={positive ? 'text-accent-buy' : negative ? 'text-accent-sell' : 'text-ink-soft'}
    />
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
    refetchInterval: 5000,
  })

  const isLive = ['live', 'demo'].includes(status?.trade_mode || stats?.trade_mode)
  const dataSource = status?.data_source || stats?.data_source || (isLive ? 'xm' : 'tradingview')
  const symbol =
    status?.symbol ||
    stats?.symbol ||
    (dataSource === 'xm' ? status?.broker_symbol || 'GOLD' : 'XAUUSD')
  const brokerAcct = stats?.broker_account || status?.broker_account
  const sourceLabel =
    status?.data_source_label ||
    stats?.data_source_label ||
    (dataSource === 'xm' ? `XM ${symbol} (MetaAPI)` : 'TradingView')

  const capital = isLive ? brokerAcct?.balance ?? stats?.capital : null
  const equity = isLive ? brokerAcct?.equity ?? stats?.equity : null
  const xmOk = isLive && status?.broker_auth_ok !== false && (stats?.xm_synced || brokerAcct?.balance != null)
  const authMessage = status?.broker_auth_message

  return (
    <div className="space-y-3">
      {(isLive || dataSource === 'tradingview') && (
        <div className="text-xs text-surface-muted flex flex-wrap items-center gap-x-2 gap-y-1">
          {dataSource === 'xm' ? (
            <>
              <span>
                Live XM · Multi-market desk
                {brokerAcct?.server ? ` · ${brokerAcct.server}` : ''}
              </span>
              <span className={xmOk ? 'text-accent-buy' : 'text-accent-sell'}>
                {xmOk
                  ? '· XM synced'
                  : status?.broker_auth === 'expired'
                    ? '· MetaAPI token expired'
                    : '· XM sync pending'}
              </span>
              {!xmOk && authMessage ? (
                <span className="text-accent-sell">{authMessage}</span>
              ) : null}
            </>
          ) : (
            <>Paper · {sourceLabel} · Multi-market desk</>
          )}
        </div>
      )}

      <div
        className="grid gap-2.5"
        style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))' }}
      >
        {isLive && (
          <>
            <StatTile label="Capital" value={fmtMoney(capital, 2)} sub="balance" />
            <StatTile
              label="Equity"
              value={fmtMoney(equity, 2)}
              sub="balance + floating"
            />
          </>
        )}

        <StatTile
          label="Open"
          value={String(stats?.open_trades ?? status?.open_positions?.length ?? 0)}
        />

        <PnlTile
          value={
            isLive
              ? (stats?.broker_realized_pnl ?? stats?.realized_pnl ?? 0)
              : (stats?.realized_pnl ?? 0)
          }
          label={isLive ? 'Realised (XM)' : 'Realised P&L'}
          sub={
            isLive && stats?.session_realized_pnl != null
              ? `Zenith book ${Number(stats.session_realized_pnl) >= 0 ? '+' : ''}$${Number(stats.session_realized_pnl).toFixed(0)}`
              : undefined
          }
        />
        <PnlTile
          value={stats?.unrealized_pnl ?? 0}
          label={isLive ? 'Floating' : 'Unrealised'}
        />

        <StatTile
          label="R:R"
          value={
            stats?.avg_rr_pct != null
              ? `${Number(stats.avg_rr_pct).toFixed(0)}%`
              : stats?.reward_risk_pct != null
                ? `${Number(stats.reward_risk_pct).toFixed(0)}%`
                : '—'
          }
          valueClassName="text-ink-soft"
        />

        <StatTile
          label="Won"
          value={String(stats?.wins ?? 0)}
          valueClassName="text-accent-buy"
        />
        <StatTile label="BE" value={String(stats?.breakevens ?? 0)} />
        <StatTile
          label="Lost"
          value={String(stats?.losses ?? 0)}
          valueClassName="text-accent-sell"
        />
      </div>
      {isLive ? (
        <div className="text-[10px] text-surface-muted">
          Capital / Equity / Realised from XM (balance − start
          {stats?.balance_at_reset != null
            ? ` $${Number(stats.balance_at_reset).toLocaleString()}`
            : ''}
          )
          {stats?.desk_reset_at
            ? ` · since ${String(stats.desk_reset_at).slice(0, 19).replace('T', ' ')} UTC`
            : ''}
          {stats?.session_realized_pnl != null &&
          stats?.broker_realized_pnl != null &&
          Math.abs(Number(stats.session_realized_pnl) - Number(stats.broker_realized_pnl)) >= 1
            ? ` · Zenith book $${Number(stats.session_realized_pnl).toFixed(0)} (trade log; XM is truth)`
            : ''}
        </div>
      ) : null}
    </div>
  )
}

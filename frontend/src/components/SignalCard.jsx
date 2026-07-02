function Badge({ children, variant }) {
  const colors = {
    BUY: 'bg-accent-buy/20 text-accent-buy border-accent-buy/40',
    SELL: 'bg-accent-sell/20 text-accent-sell border-accent-sell/40',
    'NO TRADE': 'bg-gray-500/20 text-gray-400 border-gray-500/40',
  }
  return (
    <span
      className={`inline-flex px-3 py-1 rounded-full text-sm font-semibold border ${
        colors[variant] || colors['NO TRADE']
      }`}
    >
      {children}
    </span>
  )
}

export default function SignalCard({ signal }) {
  if (!signal) {
    return (
      <div className="zcard px-5 py-4 animate-pulse">
        <div className="h-4 bg-surface-border rounded w-64" />
      </div>
    )
  }

  const isActive = signal.SIGNAL === 'BUY' || signal.SIGNAL === 'SELL'

  return (
    <div
      className={`zcard px-5 py-4 flex flex-wrap items-center justify-between gap-3 border-l-4 ${
        signal.SIGNAL === 'BUY'
          ? 'border-l-accent-buy'
          : signal.SIGNAL === 'SELL'
            ? 'border-l-accent-sell'
            : 'border-l-surface-border'
      }`}
    >
      <div className="min-w-0 flex-1">
        <p className="text-sm text-gray-400">{signal.REASON || signal.ACTION}</p>
        {isActive && (
          <p className="text-xs text-gray-500 mt-1.5 tabular-nums">
            {signal.DEMAND_TF}/{signal.SUPPLY_TF} zones · Entry {signal.ENTRY ?? '—'} · SL {signal.SL ?? '—'}
            {signal.TP1 != null ? ` · TP1 ${signal.TP1}` : ''}
            {signal.TP != null ? ` · TP2 ${signal.TP}` : ''}
            {signal.RR ? ` · RR ${signal.RR}` : ''}
          </p>
        )}
        {!isActive && signal.DEMAND_TF && (
          <p className="text-xs text-gray-500 mt-1.5 tabular-nums">
            Watching {signal.DEMAND_TF} demand · {signal.SUPPLY_TF} supply
            {signal.MOVEMENT_5M ? ` · 5M ${signal.MOVEMENT_5M}` : ''}
            {signal.BUY_SCORE != null && signal.SELL_SCORE != null
              ? ` · scores ${signal.BUY_SCORE}/${signal.SELL_SCORE}`
              : ''}
          </p>
        )}
      </div>
      <Badge variant={signal.SIGNAL}>{signal.SIGNAL}</Badge>
    </div>
  )
}

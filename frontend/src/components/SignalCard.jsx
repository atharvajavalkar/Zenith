function Badge({ children, variant }) {
  const colors = {
    BUY: 'text-accent-buy border-accent-buy/35 bg-accent-buy/10',
    SELL: 'text-accent-sell border-accent-sell/35 bg-accent-sell/10',
    'NO TRADE': 'text-surface-muted border-surface-border bg-transparent',
  }
  return (
    <span
      className={`inline-flex px-3 py-1 rounded-sm text-[11px] font-medium uppercase tracking-nav border ${
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
      <div className="zcard px-6 py-5 animate-pulse">
        <div className="h-3 bg-surface-border/80 rounded-sm w-56" />
      </div>
    )
  }

  const isActive = signal.SIGNAL === 'BUY' || signal.SIGNAL === 'SELL'
  const marketClosed = signal.MARKET_OPEN === false
  const badge = marketClosed ? 'MARKET CLOSED' : signal.SIGNAL || 'NO TRADE'

  return (
    <div
      className={`zcard px-6 py-5 flex flex-wrap items-center justify-between gap-4 ${
        marketClosed
          ? 'border-accent-gold/20'
          : signal.SIGNAL === 'BUY'
            ? 'border-accent-buy/25'
            : signal.SIGNAL === 'SELL'
              ? 'border-accent-sell/25'
              : ''
      }`}
    >
      <div className="min-w-0 flex-1">
        <p className="text-sm font-light text-ink-soft leading-relaxed">
          {signal.REASON || signal.ACTION}
        </p>
        {marketClosed && signal.MARKET_NEXT_OPEN && (
          <p className="text-[11px] text-accent-gold/80 mt-2 tracking-wide">
            Opens {signal.MARKET_NEXT_OPEN}
            {signal.MARKET_SERVER_TIME ? ` · now ${signal.MARKET_SERVER_TIME}` : ''}
          </p>
        )}
        {isActive && !marketClosed && (
          <p className="text-[11px] text-surface-muted mt-2 tracking-wide">
            {signal.ENTRY != null && `Entry ${signal.ENTRY}`}
            {signal.SL != null && ` · SL ${signal.SL}`}
            {signal.TP != null && ` · TP ${signal.TP}`}
            {signal.SUCCESS_PERCENT != null && ` · MC ${signal.SUCCESS_PERCENT}%`}
          </p>
        )}
      </div>
      <Badge variant={marketClosed ? 'NO TRADE' : signal.SIGNAL || 'NO TRADE'}>{badge}</Badge>
    </div>
  )
}

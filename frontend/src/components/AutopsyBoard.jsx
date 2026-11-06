import { useQuery } from '@tanstack/react-query'
import api from '../services/api'

function meterTone(score) {
  if (score >= 7) return { fill: 'bg-accent-buy', glow: 'shadow-[0_0_12px_rgba(52,211,153,0.25)]' }
  if (score >= 5) return { fill: 'bg-accent-gold', glow: 'shadow-[0_0_10px_rgba(243,185,74,0.2)]' }
  return { fill: 'bg-accent-sell', glow: 'shadow-[0_0_12px_rgba(248,113,113,0.22)]' }
}

function EfficiencyMeter({ score, empty = false }) {
  const s = Math.max(1, Math.min(10, Number(score) || 5))
  // Map 1→10% … 10→100% so Weak still shows a visible bar
  const pct = empty ? 0 : (s / 10) * 100
  const tone = empty
    ? { fill: 'bg-white/10', glow: '' }
    : meterTone(s)

  return (
    <div className="w-full max-w-[220px]" title={empty ? 'No closes yet' : `Efficiency ${s}/10`}>
      <div className="relative h-2.5 rounded-full bg-white/[0.06] border border-surface-border overflow-hidden">
        <div
          className={`absolute inset-y-0 left-0 rounded-full transition-all duration-700 ease-out ${tone.fill} ${tone.glow}`}
          style={{ width: `${Math.max(empty ? 0 : 10, pct)}%` }}
        />
        <div className="absolute inset-0 flex pointer-events-none">
          {Array.from({ length: 9 }).map((_, i) => (
            <div
              key={i}
              className="flex-1 border-r border-black/25 last:border-0"
            />
          ))}
        </div>
      </div>
      <div className="mt-1.5 flex justify-between text-[9px] uppercase tracking-[0.14em] text-surface-muted">
        <span>{empty ? 'No data' : 'Weak'}</span>
        <span>Solid</span>
        <span>Strong</span>
      </div>
    </div>
  )
}

export default function AutopsyBoard() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['trade-autopsy'],
    queryFn: async () =>
      (await api.get('/api/trades/autopsy', { params: { since_desk_reset: true } })).data,
    refetchInterval: 20000,
  })

  const pairs = data?.pairs || []

  return (
    <div className="zcard overflow-hidden">
      <div className="px-5 py-3 border-b border-surface-border flex flex-wrap items-end justify-between gap-2">
        <div>
          <div className="font-semibold text-gray-100 text-sm">Trade autopsy</div>
          <div className="text-xs text-surface-muted mt-0.5">
            Since new brains wired
            {data?.desk_reset_at
              ? ` · ${String(data.desk_reset_at).slice(0, 19).replace('T', ' ')} UTC`
              : ''}
          </div>
        </div>
        <div className="text-[10px] uppercase tracking-wide text-surface-muted tabular-nums">
          {data?.n != null ? `${data.n} closes` : '—'}
          {data?.wins != null
            ? ` · ${data.wins}W / ${data.breakevens ?? 0}BE / ${data.losses ?? 0}L`
            : ''}
          {data?.generated_at ? ` · ${new Date(data.generated_at).toLocaleString()}` : ''}
        </div>
      </div>

      {isLoading && !pairs.length && (
        <div className="px-5 py-6 text-sm text-surface-muted">Reading closed trades…</div>
      )}

      {isError && (
        <div className="px-5 py-6 text-sm text-accent-sell">Could not load autopsy board.</div>
      )}

      {!isLoading && !pairs.length && !isError && (
        <div className="px-5 py-6 text-sm text-surface-muted">
          No closed trades in this window yet — meters appear after exits.
        </div>
      )}

      {pairs.length > 0 && (
        <div className="divide-y divide-surface-border/50">
          {pairs.map((p) => (
            <div
              key={p.symbol}
              className="px-5 py-4 grid grid-cols-1 lg:grid-cols-[7rem_minmax(180px,240px)_1fr] gap-4 items-center"
            >
              <div>
                <div className="font-semibold text-accent-gold tracking-wide text-sm">
                  {p.symbol}
                </div>
                <div className="text-[10px] text-surface-muted mt-1 tabular-nums">
                  {p.wins}W / {p.losses}L · {p.n} closes
                  {p.pnl != null && (
                    <span className={p.pnl >= 0 ? ' text-accent-buy' : ' text-accent-sell'}>
                      {' '}
                      · {p.pnl >= 0 ? '+' : ''}
                      {Number(p.pnl).toFixed(0)}$
                    </span>
                  )}
                </div>
              </div>

              <EfficiencyMeter score={p.efficiency} empty={!!p.no_data || !p.n} />

              <div className="min-w-0">
                <ul className="space-y-1">
                  {(p.issues || []).map((iss) => (
                    <li
                      key={`${p.symbol}-${iss.tag}`}
                      className="text-xs text-gray-300 leading-snug"
                    >
                      <span className="text-surface-muted tabular-nums mr-1.5">
                        ×{iss.count}
                      </span>
                      {iss.label}
                    </li>
                  ))}
                </ul>
                {p.tip && (
                  <div className="mt-2 text-[11px] text-surface-muted italic leading-snug line-clamp-2">
                    {p.tip}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

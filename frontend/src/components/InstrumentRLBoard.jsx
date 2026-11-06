import { useQuery } from '@tanstack/react-query'
import api from '../services/api'

function armTone(arm) {
  if (arm === 'tight') return 'text-sky-300'
  if (arm === 'aggressive') return 'text-orange-300'
  return 'text-accent-gold'
}

function wrTone(wr) {
  if (wr == null) return 'text-surface-muted'
  if (wr >= 0.55) return 'text-accent-buy'
  if (wr < 0.45) return 'text-accent-sell'
  return 'text-gray-300'
}

function expTone(r) {
  if (r == null) return 'text-surface-muted'
  if (r > 0.1) return 'text-accent-buy'
  if (r < -0.05) return 'text-accent-sell'
  return 'text-gray-300'
}

export default function InstrumentRLBoard() {
  const { data, isLoading } = useQuery({
    queryKey: ['trade-learning'],
    queryFn: async () => (await api.get('/api/trades/learning')).data,
    refetchInterval: 8000,
  })
  const { data: gate } = useQuery({
    queryKey: ['gate-check'],
    queryFn: async () => (await api.get('/api/trades/gate-check')).data,
    refetchInterval: 8000,
  })

  const rl = data?.instrument_rl
  const rows = Object.values(rl?.instruments || {})
  const deskBase = gate?.desk_mc_floor ?? data?.adaptive?.mc_floor ?? 42
  const floors = gate?.instrument_floors || {}

  return (
    <div className="zcard overflow-hidden">
      <div className="px-5 py-3 border-b border-surface-border flex flex-wrap items-end justify-between gap-2">
        <div>
          <div className="font-semibold text-gray-100 text-sm">Instrument RL</div>
          <div className="text-xs text-surface-muted mt-0.5">
            Per-pair learning from losses — raises MC/ML, prefers 1:3–1:4, books partials, tightens trail
          </div>
        </div>
        <div className="text-[10px] uppercase tracking-wide text-surface-muted tabular-nums">
          {rl?.total_updates != null ? `${rl.total_updates} updates` : '—'}
          {rl?.updated_at ? ` · ${new Date(rl.updated_at).toLocaleString()}` : ''}
        </div>
      </div>

      {isLoading && !rows.length && (
        <div className="px-5 py-6 text-sm text-surface-muted">Starting reinforcement learners…</div>
      )}

      {rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-surface-border text-surface-muted uppercase tracking-wider">
                <th className="px-4 py-2 text-left">Symbol</th>
                <th className="px-4 py-2 text-left">Profile</th>
                <th className="px-4 py-2 text-right">N</th>
                <th className="px-4 py-2 text-right">WR</th>
                <th className="px-4 py-2 text-right">E[R]</th>
                <th className="px-4 py-2 text-right">PnL</th>
                <th className="px-4 py-2 text-right">MC Δ</th>
                <th className="px-4 py-2 text-right">MC floor</th>
                <th className="px-4 py-2 text-right">Agg</th>
                <th className="px-4 py-2 text-left">Note</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const pol = r.policy || {}
                const wr = r.win_rate
                return (
                  <tr key={r.symbol} className="border-b border-surface-border/40">
                    <td className="px-4 py-2.5 font-semibold text-accent-gold tracking-wide">
                      {r.symbol}
                    </td>
                    <td className={`px-4 py-2.5 uppercase text-[10px] tracking-wide ${armTone(r.active_arm)}`}>
                      {r.active_arm || 'balanced'}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-gray-300">
                      {r.n_trades ?? 0}
                    </td>
                    <td className={`px-4 py-2.5 text-right tabular-nums ${wrTone(wr)}`}>
                      {wr != null ? `${Math.round(wr * 100)}%` : '—'}
                    </td>
                    <td className={`px-4 py-2.5 text-right tabular-nums ${expTone(r.expectancy_r)}`}>
                      {r.expectancy_r != null ? `${r.expectancy_r > 0 ? '+' : ''}${Number(r.expectancy_r).toFixed(2)}R` : '—'}
                    </td>
                    <td className={`px-4 py-2.5 text-right tabular-nums ${expTone(r.sum_pnl)}`}>
                      {r.sum_pnl != null
                        ? `${r.sum_pnl > 0 ? '+' : ''}$${Number(r.sum_pnl).toFixed(0)}`
                        : '—'}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-gray-400">
                      {pol.mc_floor_delta != null
                        ? `${pol.mc_floor_delta > 0 ? '+' : ''}${Number(pol.mc_floor_delta).toFixed(0)}`
                        : '0'}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-accent-gold font-medium">
                      {floors[r.symbol]?.mc_floor != null
                        ? `${Number(floors[r.symbol].mc_floor).toFixed(0)}%`
                        : deskBase != null && pol.mc_floor_delta != null
                          ? `${Math.min(75, Math.max(30, Number(deskBase) + Number(pol.mc_floor_delta))).toFixed(0)}%`
                          : '—'}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-gray-400">
                      {pol.aggression != null ? Number(pol.aggression).toFixed(2) : '—'}
                    </td>
                    <td className="px-4 py-2.5 text-surface-muted font-light max-w-[220px] truncate">
                      {r.message || '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      <div className="px-5 py-2.5 text-[10px] text-surface-muted border-t border-surface-border">
        Effective MC = desk protect base ({deskBase != null ? `${Number(deskBase).toFixed(0)}%` : '—'}) + per-pair RL Δ.
        Arms: <span className="text-sky-300">tight</span> ·{' '}
        <span className="text-accent-gold">balanced</span> ·{' '}
        <span className="text-orange-300">aggressive</span>.
      </div>
    </div>
  )
}

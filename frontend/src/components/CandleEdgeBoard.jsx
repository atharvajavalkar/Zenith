import { useQuery } from '@tanstack/react-query'
import api from '../services/api'

function expTone(r) {
  if (r == null) return 'text-surface-muted'
  if (r > 0.05) return 'text-accent-buy'
  if (r < -0.05) return 'text-accent-sell'
  return 'text-gray-300'
}

export default function CandleEdgeBoard() {
  const { data, isLoading } = useQuery({
    queryKey: ['trade-learning'],
    queryFn: async () => (await api.get('/api/trades/learning')).data,
    refetchInterval: 10000,
  })

  const rows = data?.candle_edge?.symbols || []

  return (
    <div className="zcard overflow-hidden">
      <div className="px-5 py-3 border-b border-surface-border flex flex-wrap items-end justify-between gap-2">
        <div>
          <div className="font-semibold text-gray-100 text-sm">Candle walk-forward</div>
          <div className="text-xs text-surface-muted mt-0.5">
            Every new 1m bar · paper-sim Zenith · retunes knobs live · trade history &amp; RL intact
          </div>
        </div>
        <div className="text-[10px] uppercase tracking-wide text-surface-muted tabular-nums">
          {data?.candle_edge?.updated_at
            ? new Date(data.candle_edge.updated_at).toLocaleString()
            : 'warming'}
        </div>
      </div>

      {isLoading && !rows.length && (
        <div className="px-5 py-6 text-sm text-surface-muted">Waiting for the next 1m candle…</div>
      )}

      {rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-surface-border text-surface-muted uppercase tracking-wider">
                <th className="px-4 py-2 text-left">Symbol</th>
                <th className="px-4 py-2 text-left">Engine</th>
                <th className="px-4 py-2 text-right">N</th>
                <th className="px-4 py-2 text-right">WR</th>
                <th className="px-4 py-2 text-right">E[R]</th>
                <th className="px-4 py-2 text-right">Disp</th>
                <th className="px-4 py-2 text-right">Pull</th>
                <th className="px-4 py-2 text-left">Note</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const e = row.zenith || {}
                return (
                  <tr key={row.symbol} className="border-b border-surface-border/40">
                    <td className="px-4 py-2.5 font-semibold text-accent-gold tracking-wide">
                      {row.symbol}
                    </td>
                    <td className="px-4 py-2.5 uppercase text-[10px] tracking-wide text-sky-300">
                      zenith
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-gray-400">{e.n ?? 0}</td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-gray-300">
                      {e.wr != null ? `${(Number(e.wr) * 100).toFixed(0)}%` : '—'}
                    </td>
                    <td className={`px-4 py-2.5 text-right tabular-nums ${expTone(e.exp_r)}`}>
                      {e.exp_r != null
                        ? `${Number(e.exp_r) > 0 ? '+' : ''}${Number(e.exp_r).toFixed(3)}`
                        : '—'}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-gray-400">
                      {e.disp_atr != null ? Number(e.disp_atr).toFixed(2) : '—'}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-gray-400">
                      {e.pull_max_atr != null ? Number(e.pull_max_atr).toFixed(2) : '—'}
                    </td>
                    <td className="px-4 py-2.5 text-surface-muted font-light truncate max-w-[220px]">
                      {e.message || '—'}
                      {e.block ? ' · BLOCKED' : ''}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

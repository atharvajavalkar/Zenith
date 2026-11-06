import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../services/api'

function classTone(cls) {
  if (cls === 'fx') return 'text-sky-300'
  if (cls === 'gold' || cls === 'silver') return 'text-accent-gold'
  if (cls === 'crypto') return 'text-violet-300'
  return 'text-ink-soft'
}

function money(n) {
  if (n == null || Number.isNaN(Number(n))) return '—'
  const v = Number(n)
  const sign = v < 0 ? '-' : ''
  return `${sign}$${Math.abs(v).toFixed(0)}`
}

export default function LotDesk() {
  const queryClient = useQueryClient()
  const { data, isLoading, error } = useQuery({
    queryKey: ['lot-desk'],
    queryFn: async () => (await api.get('/api/trades/lot-desk')).data,
    refetchInterval: 10000,
    staleTime: 0,
  })

  const repair = useMutation({
    mutationFn: async () => (await api.post('/api/trades/repair-stops')).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trade-history'] })
      queryClient.invalidateQueries({ queryKey: ['trade-status'] })
      queryClient.invalidateQueries({ queryKey: ['lot-desk'] })
    },
  })

  const rows = data?.instruments || []
  const capital = data?.capital
  const target = data?.target_profit_usd ?? 400

  return (
    <div className="zcard overflow-hidden">
      <div className="px-5 py-3 border-b border-surface-border flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="font-semibold text-gray-100 text-sm">Lot desk</div>
          <div className="text-xs text-surface-muted mt-0.5">
            Target ${Number(target).toFixed(0)} at TP · SL ≈ target ÷ RR · live from backend
            {data?.symbol_rr && (
              <span className="ml-2 text-accent-gold/80">
                · map {Object.entries(data.symbol_rr).map(([s, r]) => `${s} 1:${r}`).join(' · ')}
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-[10px] uppercase tracking-wide text-surface-muted tabular-nums">
            {capital != null ? `Capital $${Number(capital).toLocaleString()}` : '—'}
          </div>
          <button
            type="button"
            disabled={repair.isPending}
            onClick={() => repair.mutate()}
            className="text-[10px] uppercase tracking-nav text-surface-muted hover:text-accent-gold transition disabled:opacity-50"
            title="Rewrite missing/truncated SL & TP on open XM positions"
          >
            {repair.isPending ? 'Fixing stops…' : 'Fix XM stops'}
          </button>
        </div>
      </div>

      {isLoading && (
        <div className="px-5 py-5 text-sm text-surface-muted">Loading lot desk…</div>
      )}
      {error && (
        <div className="px-5 py-4 text-sm text-accent-sell">
          {error?.response?.data?.detail || error.message || 'Failed to load lot desk'}
        </div>
      )}
      {repair.isSuccess && (
        <div className="px-5 py-2 text-xs text-accent-buy border-b border-white/[0.04]">
          Stops repaired: {repair.data?.fixed_count ?? 0}
          {repair.data?.failed?.length ? ` · ${repair.data.failed.length} failed` : ''}
        </div>
      )}

      {rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-surface-border text-surface-muted uppercase tracking-wider">
                <th className="px-4 py-2 text-left">Symbol</th>
                <th className="px-4 py-2 text-left">Class</th>
                <th className="px-4 py-2 text-center">Alloc</th>
                <th className="px-4 py-2 text-right">Lot</th>
                <th className="px-4 py-2 text-right">TP $</th>
                <th className="px-4 py-2 text-right">SL $</th>
                <th className="px-4 py-2 text-center">RR</th>
                <th className="px-4 py-2 text-left">Note</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.symbol} className="border-b border-surface-border/40">
                  <td className="px-4 py-2.5 font-semibold text-accent-gold tracking-wide">
                    {r.symbol}
                  </td>
                  <td className={`px-4 py-2.5 uppercase text-[10px] tracking-wide ${classTone(r.asset_class)}`}>
                    {r.asset_class}
                  </td>
                  <td className="px-4 py-2.5 text-center tabular-nums text-gray-300">
                    {r.allocation_mult != null ? `${r.allocation_mult}×` : '—'}
                  </td>
                  <td className="px-4 py-2.5 text-right tabular-nums font-display text-base text-gray-100">
                    {Number(r.lot_size).toFixed(2)}
                  </td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-accent-buy">
                    {money(r.expected_profit_usd)}
                  </td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-accent-sell">
                    {money(-(Math.abs(Number(r.expected_loss_usd) || 0)))}
                  </td>
                  <td className="px-4 py-2.5 text-center tabular-nums text-gray-400">
                    {r.rr != null ? `1:${Number(r.rr)}` : '—'}
                  </td>
                  <td className="px-4 py-2.5 text-surface-muted font-light">{r.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

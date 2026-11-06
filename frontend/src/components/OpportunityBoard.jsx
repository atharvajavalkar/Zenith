import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../services/api'
import { useState } from 'react'

function tone(signal) {
  if (signal === 'BUY') return 'text-accent-buy'
  if (signal === 'SELL') return 'text-accent-sell'
  return 'text-surface-muted'
}

function fmtPx(symbol, v) {
  if (v == null || Number.isNaN(Number(v))) return '—'
  const s = String(symbol || '').toUpperCase()
  const dig =
    s.endsWith('JPY') ? 3
      : s === 'GOLD' || s === 'XAUUSD' || s === 'BTCUSD' || s.startsWith('BTC') ? 2
        : 5
  return Number(v).toFixed(dig)
}

export default function OpportunityBoard() {
  const queryClient = useQueryClient()
  const [flash, setFlash] = useState(null)

  const { data, isLoading, error } = useQuery({
    queryKey: ['opportunities'],
    queryFn: async () => (await api.get('/api/signals/opportunities')).data,
    refetchInterval: 20000,
  })

  const enter = useMutation({
    mutationFn: async (row) =>
      (
        await api.post('/api/signals/opportunities/enter', {
          symbol: row.symbol,
          exec_tf: row.exec_tf,
          signal: row.signal,
          entry: row.entry,
          sl: row.sl,
          tp: row.tp,
          force: true,
        })
      ).data,
    onSuccess: (res) => {
      const msg = res?.result?.message || res?.result?.status || 'Submitted'
      setFlash({ ok: res?.result?.status === 'executed', msg })
      queryClient.invalidateQueries({ queryKey: ['trade-history'] })
      queryClient.invalidateQueries({ queryKey: ['trade-status'] })
      queryClient.invalidateQueries({ queryKey: ['trade-summary'] })
      queryClient.invalidateQueries({ queryKey: ['opportunities'] })
      queryClient.invalidateQueries({ queryKey: ['trade-learning'] })
    },
    onError: (err) => {
      setFlash({
        ok: false,
        msg: err?.response?.data?.detail || err.message || 'Enter failed',
      })
    },
  })

  const rows = data?.opportunities || []
  const universe = data?.universe || []
  const ready = data?.ready ?? rows.filter((r) => r.punchable !== false).length
  const blocked = data?.blocked ?? rows.length - ready

  return (
    <div className="zcard overflow-hidden">
      <div className="px-5 py-3 border-b border-surface-border flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="font-semibold text-gray-100 text-sm">Opportunity board</div>
          <div className="text-xs text-surface-muted mt-0.5">
            Brain setups · Auto column uses the same Markov MC + floor as auto-trade
            (not path MC) · blocked = already open / MC / gates
          </div>
        </div>
        <div className="text-[10px] uppercase tracking-wide text-surface-muted tabular-nums">
          {ready} ready · {blocked} blocked
          {universe.length ? ` · ${universe.join(' · ')}` : ''}
        </div>
      </div>

      {flash && (
        <div
          className={`px-5 py-2 text-xs border-b border-white/[0.04] ${
            flash.ok ? 'text-accent-buy' : 'text-accent-sell'
          }`}
        >
          {flash.msg}
        </div>
      )}

      {isLoading && (
        <div className="px-5 py-6 text-sm text-surface-muted">Scanning universe…</div>
      )}
      {error && (
        <div className="px-5 py-4 text-sm text-accent-sell">
          {error?.response?.data?.detail || error.message || 'Scan failed'}
        </div>
      )}
      {!isLoading && !error && rows.length === 0 && (
        <div className="px-5 py-6 text-sm text-surface-muted">
          No valid setups across the universe right now — waiting.
        </div>
      )}
      {rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-surface-border text-surface-muted uppercase tracking-wider">
                <th className="px-4 py-2 text-left">Symbol</th>
                <th className="px-4 py-2 text-left">TF</th>
                <th className="px-4 py-2 text-left">Signal</th>
                <th className="px-4 py-2 text-right">Entry</th>
                <th className="px-4 py-2 text-right">SL</th>
                <th className="px-4 py-2 text-right">TP</th>
                <th className="px-4 py-2 text-center">RR</th>
                <th className="px-4 py-2 text-center">MC / fl</th>
                <th className="px-4 py-2 text-left">Auto</th>
                <th className="px-4 py-2 text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => {
                const key = `${r.symbol}-${r.exec_tf}-${i}`
                const isReady = r.punchable !== false
                const pending =
                  enter.isPending &&
                  enter.variables?.symbol === r.symbol &&
                  enter.variables?.exec_tf === r.exec_tf
                return (
                  <tr
                    key={key}
                    className={`border-b border-surface-border/40 ${
                      isReady ? '' : 'opacity-60'
                    }`}
                  >
                    <td className="px-4 py-2.5 font-medium text-accent-gold tracking-wide">
                      {r.symbol}
                    </td>
                    <td className="px-4 py-2.5 text-accent-gold/90 uppercase">{r.exec_tf}</td>
                    <td className={`px-4 py-2.5 font-semibold ${tone(r.signal)}`}>{r.signal}</td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-gray-300">
                      {fmtPx(r.symbol, r.entry)}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-red-400/80">
                      {fmtPx(r.symbol, r.sl)}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-green-400/80">
                      {fmtPx(r.symbol, r.tp)}
                    </td>
                    <td className="px-4 py-2.5 text-center text-gray-400">
                      {r.rr ?? (r.target_rr_label || '—')}
                    </td>
                    <td className="px-4 py-2.5 text-center text-accent-gold tabular-nums">
                      {r.mc != null ? `${Number(r.mc).toFixed(0)}%` : '—'}
                      {r.mc_floor != null ? (
                        <span className="text-surface-muted font-normal">
                          {' '}
                          /{Number(r.mc_floor).toFixed(0)}
                        </span>
                      ) : null}
                    </td>
                    <td className="px-4 py-2.5 max-w-[14rem]">
                      {isReady ? (
                        <span className="text-accent-buy uppercase tracking-wide text-[10px]">
                          Ready
                        </span>
                      ) : (
                        <span
                          className="text-accent-sell text-[10px] leading-snug block"
                          title={r.block_reason || 'Blocked'}
                        >
                          Blocked · {r.block_reason || 'gate'}
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      <button
                        type="button"
                        disabled={enter.isPending || r.signal === 'NO TRADE'}
                        onClick={() => enter.mutate(r)}
                        className="text-[10px] uppercase tracking-wide px-2 py-1 rounded border border-accent-gold/40 text-accent-gold hover:bg-accent-gold/10 transition disabled:opacity-40"
                        title={
                          isReady
                            ? 'Manual enter (same as auto)'
                            : `Auto blocked: ${r.block_reason || 'gate'} — Enter still forces`
                        }
                      >
                        {pending ? '…' : isReady ? 'Enter' : 'Force'}
                      </button>
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

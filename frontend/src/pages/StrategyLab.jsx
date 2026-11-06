import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Layout from '../components/Layout'
import LivePnlRaceChart from '../components/LivePnlRaceChart'
import api from '../services/api'

function verdictClass(v) {
  if (v === 'SURVIVES') return 'text-accent-buy'
  if (v === 'FRAGILE') return 'text-accent-gold'
  return 'text-accent-sell'
}

function familyBadge(family) {
  if (family === 'hedge') return 'border-sky-500/30 bg-sky-500/10 text-sky-300'
  return 'border-amber-500/30 bg-amber-500/10 text-amber-200'
}

function formatBarValue(valueKey, v) {
  if (valueKey.includes('rate')) return `${Number(v).toFixed(1)}%`
  if (valueKey === 'exp_r' || valueKey === 'oos_exp_r' || valueKey.includes('exp_r')) {
    return Number(v).toFixed(3)
  }
  if (valueKey === 'n') return Math.round(Number(v)).toLocaleString()
  return `${v >= 0 ? '+' : ''}$${Math.round(v).toLocaleString()}`
}

function BarChart({ rows, valueKey = 'oos_pnl', title, subtitle }) {
  const vals = (rows || []).map((r) => Number(r[valueKey]) || 0)
  const maxAbs = Math.max(1e-9, ...vals.map((v) => Math.abs(v)))
  return (
    <div className="zcard p-5">
      <div className="text-sm font-semibold text-gray-100">{title}</div>
      {subtitle && <div className="text-[11px] text-surface-muted mt-0.5 mb-4">{subtitle}</div>}
      <div className="space-y-2.5">
        {(rows || []).map((r) => {
          const v = Number(r[valueKey]) || 0
          const pct = (Math.abs(v) / maxAbs) * 100
          const pos = v >= 0
          return (
            <div key={r.symbol} className="grid grid-cols-[72px_1fr_88px] gap-2 items-center text-xs">
              <div className="font-medium text-gray-200 truncate">{r.symbol}</div>
              <div className="relative h-5 rounded bg-white/[0.04] overflow-hidden">
                <div
                  className={`absolute inset-y-0 left-0 rounded transition-all ${
                    pos ? 'bg-accent-buy/70' : 'bg-accent-sell/70'
                  }`}
                  style={{ width: `${Math.max(2, pct)}%` }}
                />
                <div className="absolute inset-0 flex items-center px-2 text-[10px] text-white/80 truncate">
                  {r.label || r.strategy || '—'}
                </div>
              </div>
              <div className={`text-right tabular-nums ${pos ? 'text-accent-buy' : 'text-accent-sell'}`}>
                {formatBarValue(valueKey, v)}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function StrategyLabPage() {
  const [lineSym, setLineSym] = useState('ETHUSD')
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ['strategy-lab'],
    queryFn: async () => (await api.get('/api/strategy-lab')).data,
    refetchInterval: 60000,
  })

  const results = data?.results
  const isRecovery = results?.mode === 'desk_recovery_scalp_hunt'
  const isOmniLive = results?.mode === 'omni_horizon_live' || results?.live_wired
  const isAdaptiveXmV4 = results?.mode === 'adaptive_xm_cost_v4'
  const isDeepHunt = results?.mode === 'adaptive_xm_deep_hunt'
  const isAdaptiveV3 = results?.mode === 'adaptive_minute_day_v3'
  const isAdaptive =
    results?.mode === 'adaptive_regime_v2' || isAdaptiveV3 || isAdaptiveXmV4 || isDeepHunt || isOmniLive
  const perPair = results?.per_pair || {}
  const symbols = Object.keys(perPair)
  const bestChart =
    results?.charts?.omni_all_six ||
    results?.charts?.best_oos_by_pair ||
    results?.charts?.oos_exp_r ||
    results?.charts?.pair_1y ||
    []
  const successRows = results?.charts?.success_by_pair || results?.charts?.win_rate || []
  const tradeRows = results?.charts?.trades_by_pair || results?.charts?.trades || []
  const erRows = results?.charts?.exp_r_by_pair || results?.charts?.exp_r_vs_spread || []
  const vsLive = results?.charts?.vs_live || []
  const deskModes = results?.charts?.desk_modes || results?.charts?.desk || []
  const walkForwardDesk =
    results?.charts?.walk_forward_desk ||
    results?.walk_forward?.desk ||
    []
  const fullRows = bestChart.map((r) => ({
    ...r,
    full_pnl: r.full_pnl ?? r.pnl ?? perPair[r.symbol]?.best?.full_pnl ?? 0,
    label: r.label || r.strategy || r.absolute || '—',
    oos_exp_r: r.oos_exp_r ?? r.exp_r,
  }))
  const selected = perPair[lineSym]?.best
  const top10 = perPair[lineSym]?.top10 || []

  const subtitle = isOmniLive
    ? `Omni-horizon LIVE on XM demo — GOLD 3m+5m · BTC 5m+3m · UKOIL 3m (all-six window survivors). Other pairs halted. RR=3.`
    : isDeepHunt
    ? `Deep XM hunt board — prefer omni all-six over 1y-only crowns. ${results?.live_wired ? 'Live map active.' : 'Research board.'}`
    : isAdaptiveXmV4
    ? `Adaptive XM-cost v4 — Dukascopy re-hunt with measured XM session spreads. Desk 1y $${Number(results?.summary?.desk_1y || results?.desk_summary?.absolute_1y || 0).toLocaleString()} · wireable $${Number(results?.summary?.desk_wireable_1y || results?.desk_summary?.wireable_1y || 0).toLocaleString()}. Research only — live auto-trade stays off.`
    : isAdaptiveV3
    ? results?.live_wired
      ? `Adaptive v3 LIVE on XM demo — absolute winners wired. Paper WF desk 7d $${Number(results?.summary?.desk_7d || 0).toLocaleString()} · 90d $${Number(results?.summary?.desk_90d || 0).toLocaleString()} · 1y $${Number(results?.summary?.desk_1y || results?.summary?.desk_full_pnl || 0).toLocaleString()} · 3y $${Number(results?.summary?.desk_3y || 0).toLocaleString()}. Freeze tweaks ${results?.watch_days || 14}d.`
      : 'Adaptive v3 — minute/day equations (session mix, circadian Kalman, entropy TOD). Absolute 1y winner per pair.'
    : isAdaptive
      ? 'Adaptive regime router v2 — static vs gated vs switch per pair (preferred = best 1y). Research only — live brains unchanged.'
      : isRecovery
        ? 'Desk recovery — scalper / intraday hunt (1m · 3m · 5m · 15m). Firm gate + math overlays. Research only — live brains unchanged.'
        : 'Dukascopy scalp / intraday hunt — retail + hedge on 1m · 3m · 5m · 15m (past 1y). Gate: full-span SURVIVES, positive 90d OOS, walk-forward. Research only — live brains unchanged.'


  return (
    <Layout>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold text-gray-100 tracking-tight">Strategy Lab</h1>
          <p className="text-sm text-surface-muted mt-1 max-w-2xl">{subtitle}</p>
        </div>
        <button type="button" className="btn-secondary" onClick={() => refetch()} disabled={isLoading}>
          Refresh
        </button>
      </div>

      <div className="mt-6">
        <LivePnlRaceChart />
      </div>

      {(isError || data?.last_error) && (
        <div className="text-sm text-accent-sell mt-4">
          {data?.last_error || 'Failed to load Strategy Lab'}
        </div>
      )}

      {!results && !isLoading && (
        <div className="zcard p-8 mt-6 text-sm text-surface-muted">
          No deep-hunt board yet. Run the Dukascopy deep hunt to populate results.
        </div>
      )}

      {results && (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-6 gap-3 mt-6">
            {[
              [
                'Mode',
                isOmniLive
                  ? 'Omni LIVE'
                  : isDeepHunt
                    ? 'Deep hunt'
                    : isAdaptiveV3
                      ? results?.live_wired
                        ? 'v3 LIVE'
                        : 'Adaptive v3'
                      : isAdaptive
                        ? 'Adaptive v2'
                        : isRecovery
                          ? 'Recovery scalp'
                          : results.window?.tf || results.mode || 'duka',
              ],
              ['Strategies', results.hunt?.variants_tested || results.summary?.strategies_tested],
              [
                'Deep gate',
                `${results.hunt?.deep_gate_survivors ?? results.summary?.pairs_with_survives ?? '—'}/8`,
              ],
              [
                'Trades',
                results.summary?.trades_total != null
                  ? Number(results.summary.trades_total).toLocaleString()
                  : '—',
              ],
              [
                'Desk full PnL',
                results.summary?.desk_full_pnl != null
                  ? `$${Number(results.summary.desk_full_pnl).toLocaleString()}`
                  : '—',
              ],
              ['Risk', `$${results.risk_usd}/R`],
            ].map(([k, v]) => (
              <div key={k} className="zcard px-4 py-3">
                <div className="text-[10px] uppercase tracking-wider text-surface-muted">{k}</div>
                <div className="text-sm text-gray-100 mt-1 truncate">{v ?? '—'}</div>
              </div>
            ))}
          </div>

          {(results.live_map ||
            results.live_SYMBOL_ENTRY_BRAIN ||
            results.proposed_SYMBOL_ENTRY_BRAIN ||
            results.proposed_SYMBOL_SCAN_TFS) && (
            <div className="zcard p-4 mt-4 space-y-2">
              <div className="text-[10px] uppercase tracking-wider text-surface-muted">
                {results.live_wired ? 'Live map (wired on XM demo)' : 'Proposed map (not wired live)'}
              </div>
              {results.live_map && (
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mb-2">
                  {results.live_map.map((m) => (
                    <div
                      key={m.symbol}
                      className="rounded-sm border border-accent-buy/30 bg-accent-buy/5 px-3 py-2"
                    >
                      <div className="text-sm font-semibold text-accent-buy">{m.symbol}</div>
                      <div className="text-[11px] text-surface-muted mt-0.5">
                        {m.brain} · {(m.tfs || []).join('+')} · RR{m.rr}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {(results.live_SYMBOL_ENTRY_BRAIN || results.proposed_SYMBOL_ENTRY_BRAIN) && (
                <div className="text-xs text-gray-200 font-mono break-all">
                  <span className="text-surface-muted">BRAIN </span>
                  {results.live_SYMBOL_ENTRY_BRAIN || results.proposed_SYMBOL_ENTRY_BRAIN}
                </div>
              )}
              {(results.live_SYMBOL_SCAN_TFS || results.proposed_SYMBOL_SCAN_TFS) && (
                <div className="text-xs text-gray-200 font-mono break-all">
                  <span className="text-surface-muted">TFS </span>
                  {results.live_SYMBOL_SCAN_TFS || results.proposed_SYMBOL_SCAN_TFS}
                </div>
              )}
              {(results.live_SYMBOL_RR || results.proposed_SYMBOL_RR) && (
                <div className="text-xs text-gray-200 font-mono break-all">
                  <span className="text-surface-muted">RR </span>
                  {results.live_SYMBOL_RR || results.proposed_SYMBOL_RR}
                </div>
              )}
              {results.live_TRADING_HALT_SYMBOLS && (
                <div className="text-xs text-accent-sell/90 font-mono break-all">
                  <span className="text-surface-muted">HALTED </span>
                  {results.live_TRADING_HALT_SYMBOLS}
                </div>
              )}
            </div>
          )}

          <div className="grid lg:grid-cols-2 gap-5 mt-6">
            <BarChart
              rows={fullRows}
              valueKey="full_pnl"
              title="Full-span PnL by pair"
              subtitle={
                isAdaptiveV3
                  ? `v3 absolute winner 1y PnL · $${results.risk_usd}/R`
                  : isAdaptive
                    ? `Preferred mode 1y PnL · $${results.risk_usd}/R`
                    : isRecovery
                      ? `Recovery scalp winners · $${results.risk_usd}/R · firm gate`
                      : `Dukascopy scalp 1y · $${results.risk_usd}/R · gated winners`
              }
            />
            <BarChart
              rows={successRows.length ? successRows : bestChart}
              valueKey="success_rate"
              title={isAdaptive ? 'Win rate by pair (preferred)' : 'OOS win rate by pair'}
              subtitle={
                isAdaptiveV3
                  ? 'Win rate for the v3 absolute winner (best 1y)'
                  : isAdaptive
                    ? 'Win rate for the preferred adaptive mode (best 1y)'
                    : 'Out-of-sample win rate for the gated winner'
              }
            />
          </div>

          {isAdaptiveV3 && walkForwardDesk.length > 0 && (
            <div className="zcard overflow-hidden mt-6">
              <div className="px-5 py-3 border-b border-surface-border">
                <div className="text-sm font-semibold text-gray-100">Multi-horizon walk-forward (desk)</div>
                <div className="text-[11px] text-surface-muted mt-0.5">
                  Wired absolute mix vs v2 preferred — paper $200/R
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-[10px] uppercase tracking-wider text-surface-muted border-b border-surface-border">
                      <th className="text-left px-4 py-2 font-medium">Window</th>
                      <th className="text-right px-4 py-2 font-medium">Wired v3</th>
                      <th className="text-right px-4 py-2 font-medium">v2 pref</th>
                    </tr>
                  </thead>
                  <tbody>
                    {walkForwardDesk.map((r) => (
                      <tr key={r.window} className="border-b border-white/[0.04]">
                        <td className="px-4 py-2.5 font-medium text-gray-100">{r.window}</td>
                        <td
                          className={`px-4 py-2.5 text-right tabular-nums ${
                            Number(r.pnl) >= 0 ? 'text-accent-buy' : 'text-accent-sell'
                          }`}
                        >
                          {r.pnl != null ? `$${Number(r.pnl).toLocaleString()}` : '—'}
                        </td>
                        <td
                          className={`px-4 py-2.5 text-right tabular-nums ${
                            Number(r.v2_preferred) >= 0 ? 'text-accent-buy' : 'text-accent-sell'
                          }`}
                        >
                          {r.v2_preferred != null
                            ? `$${Number(r.v2_preferred).toLocaleString()}`
                            : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {isAdaptive && deskModes.length > 0 && (
            <div className="zcard overflow-hidden mt-6">
              <div className="px-5 py-3 border-b border-surface-border">
                <div className="text-sm font-semibold text-gray-100">Desk PnL by mode</div>
                <div className="text-[11px] text-surface-muted mt-0.5">
                  {isAdaptiveV3
                    ? 'Absolute mix vs v2 / top formulas across horizons'
                    : 'Static vs gated vs switch across horizons (all pairs summed)'}
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-[10px] uppercase tracking-wider text-surface-muted border-b border-surface-border">
                      <th className="text-left px-4 py-2 font-medium">Window</th>
                      {isAdaptiveV3 ? (
                        <>
                          <th className="text-right px-4 py-2 font-medium">Absolute</th>
                          <th className="text-right px-4 py-2 font-medium">v2 pref</th>
                          <th className="text-right px-4 py-2 font-medium">Static</th>
                          <th className="text-right px-4 py-2 font-medium">Session</th>
                          <th className="text-right px-4 py-2 font-medium">Circadian</th>
                        </>
                      ) : (
                        <>
                          <th className="text-right px-4 py-2 font-medium">Static</th>
                          <th className="text-right px-4 py-2 font-medium">Gated</th>
                          <th className="text-right px-4 py-2 font-medium">Switch</th>
                        </>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {deskModes.map((r) => (
                      <tr key={r.window} className="border-b border-white/[0.04]">
                        <td className="px-4 py-2.5 font-medium text-gray-100">{r.window}</td>
                        {(isAdaptiveV3
                          ? [r.absolute, r.v2_preferred, r.static, r.session_mix, r.circadian_kalman_z]
                          : [r.static, r.gated, r.switch]
                        ).map((v, i) => (
                          <td
                            key={i}
                            className={`px-4 py-2.5 text-right tabular-nums ${
                              Number(v) >= 0 ? 'text-accent-buy' : 'text-accent-sell'
                            }`}
                          >
                            {v != null ? `$${Number(v).toLocaleString()}` : '—'}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="grid lg:grid-cols-2 gap-5 mt-5">
            <BarChart
              rows={
                erRows.length
                  ? erRows
                  : fullRows.map((r) => ({
                      ...r,
                      exp_r: r.exp_r ?? perPair[r.symbol]?.best?.full_exp_r ?? 0,
                    }))
              }
              valueKey="exp_r"
              title="Expected R by pair"
              subtitle={
                isAdaptive
                  ? '1y E[R] for preferred adaptive mode'
                  : '1y E[R] for scalp punches — higher = stronger edge per trade'
              }
            />
            <BarChart
              rows={
                tradeRows.length
                  ? tradeRows
                  : fullRows.map((r) => ({
                      ...r,
                      n: r.n ?? perPair[r.symbol]?.best?.n ?? 0,
                    }))
              }
              valueKey="n"
              title={isAdaptive ? 'Trade count (preferred mode)' : 'Trade count (scalp frequency)'}
              subtitle={
                isAdaptive
                  ? 'How often the preferred mode trades on 1y'
                  : 'How often the winner punches — recovery prefers high-n survivors'
              }
            />
          </div>

          {vsLive.length > 0 && (
            <div className="zcard overflow-hidden mt-6">
              <div className="px-5 py-3 border-b border-surface-border">
                <div className="text-sm font-semibold text-gray-100">
                  {isAdaptiveV3
                    ? 'v3 absolute vs v2 / static (1y)'
                    : isAdaptive
                      ? 'Static vs gated vs switch (1y)'
                      : 'Live brain vs hunt winner'}
                </div>
                <div className="text-[11px] text-surface-muted mt-0.5">
                  {isAdaptive
                    ? 'Research only — not auto-applied to live'
                    : 'What is wired now vs what the recovery hunt proposes (not auto-applied)'}
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-[10px] uppercase tracking-wider text-surface-muted border-b border-surface-border">
                      <th className="text-left px-4 py-2 font-medium">Pair</th>
                      {isAdaptiveV3 ? (
                        <>
                          <th className="text-left px-4 py-2 font-medium">v3 abs</th>
                          <th className="text-right px-4 py-2 font-medium">Static</th>
                          <th className="text-right px-4 py-2 font-medium">v2</th>
                          <th className="text-right px-4 py-2 font-medium">7d</th>
                          <th className="text-right px-4 py-2 font-medium">n</th>
                          <th className="text-right px-4 py-2 font-medium">E[R]</th>
                          <th className="text-right px-4 py-2 font-medium">1y PnL</th>
                        </>
                      ) : isAdaptive ? (
                        <>
                          <th className="text-left px-4 py-2 font-medium">Preferred</th>
                          <th className="text-right px-4 py-2 font-medium">Static</th>
                          <th className="text-right px-4 py-2 font-medium">Gated</th>
                          <th className="text-right px-4 py-2 font-medium">Switch</th>
                          <th className="text-right px-4 py-2 font-medium">n</th>
                          <th className="text-right px-4 py-2 font-medium">E[R]</th>
                          <th className="text-right px-4 py-2 font-medium">Pref PnL</th>
                        </>
                      ) : (
                        <>
                          <th className="text-left px-4 py-2 font-medium">Live</th>
                          <th className="text-left px-4 py-2 font-medium">Hunt</th>
                          <th className="text-right px-4 py-2 font-medium">TF</th>
                          <th className="text-right px-4 py-2 font-medium">n</th>
                          <th className="text-right px-4 py-2 font-medium">E[R]</th>
                          <th className="text-right px-4 py-2 font-medium">Full PnL</th>
                          <th className="text-right px-4 py-2 font-medium">Δ</th>
                        </>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {vsLive.map((r) => (
                      <tr
                        key={r.symbol}
                        className={`border-b border-white/[0.04] hover:bg-white/[0.02] cursor-pointer ${
                          lineSym === r.symbol ? 'bg-white/[0.03]' : ''
                        }`}
                        onClick={() => setLineSym(r.symbol)}
                      >
                        <td className="px-4 py-2.5 font-medium text-gray-100">{r.symbol}</td>
                        {isAdaptiveV3 ? (
                          <>
                            <td className="px-4 py-2.5 text-xs text-accent-gold font-mono">
                              {r.hunt_strategy}
                            </td>
                            <td className="px-4 py-2.5 text-right tabular-nums text-gray-300">
                              {r.static_pnl != null ? `$${Number(r.static_pnl).toLocaleString()}` : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-right tabular-nums text-gray-300">
                              {r.v2_pnl != null ? `$${Number(r.v2_pnl).toLocaleString()}` : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-right tabular-nums text-gray-300">
                              {r.pnl_7d_abs != null ? `$${Number(r.pnl_7d_abs).toLocaleString()}` : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-right tabular-nums text-gray-300">
                              {r.hunt_n != null ? Number(r.hunt_n).toLocaleString() : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-right tabular-nums text-accent-gold">
                              {r.hunt_exp_r != null ? Number(r.hunt_exp_r).toFixed(3) : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-right tabular-nums text-gray-200">
                              ${Number(r.hunt_pnl || 0).toLocaleString()}
                            </td>
                          </>
                        ) : isAdaptive ? (
                          <>
                            <td className="px-4 py-2.5 text-xs text-accent-gold font-mono uppercase">
                              {r.hunt_strategy}
                            </td>
                            <td className="px-4 py-2.5 text-right tabular-nums text-gray-300">
                              {r.static_pnl != null ? `$${Number(r.static_pnl).toLocaleString()}` : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-right tabular-nums text-gray-300">
                              {r.gated_pnl != null ? `$${Number(r.gated_pnl).toLocaleString()}` : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-right tabular-nums text-gray-300">
                              {r.switch_pnl != null ? `$${Number(r.switch_pnl).toLocaleString()}` : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-right tabular-nums text-gray-300">
                              {r.hunt_n != null ? Number(r.hunt_n).toLocaleString() : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-right tabular-nums text-accent-gold">
                              {r.hunt_exp_r != null ? Number(r.hunt_exp_r).toFixed(3) : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-right tabular-nums text-gray-200">
                              ${Number(r.hunt_pnl || 0).toLocaleString()}
                            </td>
                          </>
                        ) : (
                          <>
                            <td className="px-4 py-2.5 text-xs text-surface-muted font-mono">
                              {r.live_brain || '—'}
                              {r.live_rr != null ? ` · ${r.live_rr}R` : ''}
                            </td>
                            <td className="px-4 py-2.5 text-xs text-gray-200 font-mono truncate max-w-[220px]">
                              {r.hunt_strategy}
                            </td>
                            <td className="px-4 py-2.5 text-right tabular-nums text-gray-300">
                              {r.hunt_tf}
                            </td>
                            <td className="px-4 py-2.5 text-right tabular-nums text-gray-300">
                              {r.hunt_n != null ? Number(r.hunt_n).toLocaleString() : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-right tabular-nums text-accent-gold">
                              {r.hunt_exp_r != null ? Number(r.hunt_exp_r).toFixed(3) : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-right tabular-nums text-gray-200">
                              ${Number(r.hunt_pnl || 0).toLocaleString()}
                            </td>
                            <td
                              className={`px-4 py-2.5 text-right text-xs font-medium ${
                                r.changed ? 'text-accent-gold' : 'text-surface-muted'
                              }`}
                            >
                              {r.changed ? 'CHANGE' : 'same'}
                            </td>
                          </>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="zcard overflow-hidden mt-6">
            <div className="px-5 py-3 border-b border-surface-border">
              <div className="text-sm font-semibold text-gray-100">Deep-gate winner per pair</div>
              <div className="text-[11px] text-surface-muted mt-0.5">
                Click a row for TF / score / runners-up · ranked by scalp-gate score
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-[10px] uppercase tracking-wider text-surface-muted border-b border-surface-border">
                    <th className="text-left px-4 py-2 font-medium">Pair</th>
                    <th className="text-left px-4 py-2 font-medium">Strategy</th>
                    <th className="text-left px-4 py-2 font-medium">Family</th>
                    <th className="text-right px-4 py-2 font-medium">TF</th>
                    <th className="text-right px-4 py-2 font-medium">n</th>
                    <th className="text-right px-4 py-2 font-medium">E[R]</th>
                    <th className="text-right px-4 py-2 font-medium">Full PnL</th>
                    <th className="text-right px-4 py-2 font-medium">MC%</th>
                    <th className="text-right px-4 py-2 font-medium">Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {symbols.map((sym) => {
                    const best = perPair[sym]?.best
                    if (!best) {
                      return (
                        <tr key={sym} className="border-b border-white/[0.04]">
                          <td className="px-4 py-2.5 font-medium">{sym}</td>
                          <td colSpan={8} className="px-4 py-2.5 text-surface-muted text-xs">
                            {perPair[sym]?.error || 'No winner'}
                          </td>
                        </tr>
                      )
                    }
                    return (
                      <tr
                        key={sym}
                        className={`border-b border-white/[0.04] hover:bg-white/[0.02] cursor-pointer ${
                          lineSym === sym ? 'bg-white/[0.03]' : ''
                        }`}
                        onClick={() => setLineSym(sym)}
                      >
                        <td className="px-4 py-2.5 font-medium text-gray-100">{sym}</td>
                        <td className="px-4 py-2.5">
                          <div className="text-gray-100">{best.label}</div>
                          <div className="text-[10px] text-surface-muted">{best.strategy}</div>
                        </td>
                        <td className="px-4 py-2.5">
                          <span
                            className={`text-[10px] uppercase tracking-wide px-2 py-0.5 rounded border ${familyBadge(
                              best.family
                            )}`}
                          >
                            {best.family}
                          </span>
                        </td>
                        <td className="px-4 py-2.5 text-right tabular-nums text-gray-300">
                          {best.tf || '—'}
                        </td>
                        <td className="px-4 py-2.5 text-right tabular-nums text-gray-300">
                          {best.n != null ? Number(best.n).toLocaleString() : '—'}
                        </td>
                        <td className="px-4 py-2.5 text-right tabular-nums text-accent-gold">
                          {best.full_exp_r != null
                            ? Number(best.full_exp_r).toFixed(3)
                            : best.oos_exp_r != null
                              ? Number(best.oos_exp_r).toFixed(3)
                              : '—'}
                        </td>
                        <td className="px-4 py-2.5 text-right tabular-nums text-gray-200">
                          ${Number(best.full_pnl || 0).toLocaleString()}
                        </td>
                        <td className="px-4 py-2.5 text-right tabular-nums text-gray-300">
                          {best.mc_prob != null
                            ? `${Math.round(Number(best.mc_prob) > 1.5 ? Number(best.mc_prob) : Number(best.mc_prob) * 100)}%`
                            : '—'}
                        </td>
                        <td className={`px-4 py-2.5 text-right font-medium ${verdictClass(best.verdict)}`}>
                          {best.verdict}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {selected && (
            <div className="grid lg:grid-cols-2 gap-5 mt-6">
              <div className="zcard p-5">
                <div className="text-sm font-semibold text-gray-100">{lineSym} · winner detail</div>
                <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
                  {[
                    ['Strategy', selected.label],
                    ['ID', selected.strategy],
                    ['Family', selected.family],
                    ['TF / RR', `${selected.tf || '—'} · ${selected.rr ?? '—'}R`],
                    ['Live now', perPair[lineSym]?.live_brain || selected.live_brain || '—'],
                    ['Score', selected.score ?? '—'],
                    ['n / E[R]', `${selected.n ?? '—'} · ${selected.full_exp_r ?? selected.oos_exp_r ?? '—'}`],
                    ['OOS E[R]', selected.oos_exp_r ?? '—'],
                    ['90d OOS E[R]', selected.last_90d_oos_exp_r ?? selected.last_5y_oos_exp_r ?? '—'],
                    ['WF green', `${selected.wf_green ?? '—'} / ${selected.wf_n ?? '—'}`],
                    [
                      'Math',
                      selected.math
                        ? JSON.stringify(selected.math)
                        : '—',
                    ],
                    ['Bars', selected.bars != null ? Number(selected.bars).toLocaleString() : '—'],
                  ].map(([k, v]) => (
                    <div key={k}>
                      <div className="text-[10px] uppercase tracking-wider text-surface-muted">{k}</div>
                      <div className="text-gray-100 mt-0.5 break-all">{v}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="zcard p-5">
                <div className="text-sm font-semibold text-gray-100">{lineSym} · top candidates</div>
                <div className="text-[11px] text-surface-muted mt-0.5 mb-3">
                  Ranked runners from the recovery / deep hunt (gated first)
                </div>
                <div className="space-y-2">
                  {top10.length === 0 && (
                    <div className="text-xs text-surface-muted">No runners-up stored for this pair.</div>
                  )}
                  {top10.map((t, i) => (
                    <div
                      key={`${t.strategy}-${t.tf}-${i}`}
                      className="flex items-center justify-between gap-3 text-xs border-b border-white/[0.04] pb-2"
                    >
                      <div className="min-w-0">
                        <div className="text-gray-100 truncate">
                          {t.label || t.strategy}
                          {t.deep_gate ? (
                            <span className="ml-2 text-[10px] text-accent-buy">GATE</span>
                          ) : null}
                        </div>
                        <div className="text-[10px] text-surface-muted">
                          {t.family} · {t.tf} · {t.rr}R · n={t.n ?? '—'}
                        </div>
                      </div>
                      <div className="text-right shrink-0">
                        <div className={verdictClass(t.verdict)}>{t.verdict || '—'}</div>
                        <div className="text-[10px] text-surface-muted tabular-nums">
                          score {t.score ?? '—'} · E[R] {t.full_exp_r ?? t.oos_exp_r ?? '—'}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          <div className="text-[11px] text-surface-muted mt-4">
            Generated {results.generated_at?.slice(0, 19)?.replace('T', ' ')} UTC · mode{' '}
            {results.mode || 'duka_deep_hunt'}
            {results.note ? ` · ${results.note}` : ''}
          </div>
        </>
      )}
    </Layout>
  )
}

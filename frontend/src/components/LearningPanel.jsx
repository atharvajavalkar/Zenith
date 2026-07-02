import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../services/api'
import { useExecuteTrade } from './TradePanel'

function pct(value) {
  if (value == null) return '—'
  return `${(value * 100).toFixed(1)}%`
}

function Bar({ label, winRate, count, highlight }) {
  const width = Math.min(100, (winRate || 0) * 100)
  const good = (winRate || 0) >= 0.67
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className={highlight ? 'text-accent-gold font-medium' : 'text-gray-400'}>{label}</span>
        <span className={`tabular-nums ${good ? 'text-accent-buy' : 'text-gray-500'}`}>
          {pct(winRate)} · {count} trades
        </span>
      </div>
      <div className="h-1.5 bg-surface rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${good ? 'bg-accent-buy' : 'bg-gray-600'}`}
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  )
}

function McBar({ value, floor }) {
  const pctVal = value != null ? Math.min(100, Math.max(0, value)) : 0
  const floorPct = floor != null ? Math.min(100, Math.max(0, floor)) : null
  const color = pctVal >= 67 ? '#22c55e' : pctVal >= (floor ?? 50) ? '#f59e0b' : '#ef4444'
  return (
    <div className="relative h-3 bg-surface rounded-full overflow-visible mt-1">
      <div
        className="h-full rounded-full transition-all duration-500"
        style={{ width: `${pctVal}%`, backgroundColor: color }}
      />
      {floorPct != null && (
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-accent-gold/70"
          style={{ left: `${floorPct}%` }}
          title={`MC floor ${floorPct}%`}
        />
      )}
    </div>
  )
}

export default function LearningPanel({ signal }) {
  const queryClient = useQueryClient()
  const [lastBucket, setLastBucket] = useState(3)
  const [executeMsg, setExecuteMsg] = useState('')

  const { data, isLoading } = useQuery({
    queryKey: ['trade-learning'],
    queryFn: async () => (await api.get('/api/trades/learning')).data,
    refetchInterval: 8000,
  })

  const { data: status } = useQuery({
    queryKey: ['trade-status'],
    queryFn: async () => (await api.get('/api/trades/status')).data,
    refetchInterval: 5000,
  })

  const { data: gate } = useQuery({
    queryKey: ['gate-check'],
    queryFn: async () => (await api.get('/api/trades/gate-check')).data,
    refetchInterval: 5000,
  })

  const executeTrade = useExecuteTrade()

  const updateSettings = useMutation({
    mutationFn: async (payload) => (await api.patch('/api/trades/settings', payload)).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trade-settings'] })
      queryClient.invalidateQueries({ queryKey: ['trade-status'] })
    },
  })

  const autoOn = status?.max_trades_per_bucket > 0 && !status?.trading_halted

  function toggleAutoTrade() {
    if (autoOn) {
      setLastBucket(status?.max_trades_per_bucket || 3)
      updateSettings.mutate({ max_trades_per_bucket: 0 })
    } else {
      updateSettings.mutate({ max_trades_per_bucket: lastBucket || 3 })
    }
  }

  function handleExecute() {
    setExecuteMsg('')
    executeTrade.mutate(undefined, {
      onSuccess: (d) => setExecuteMsg(d?.result?.message || 'Trade executed'),
      onError: (e) => setExecuteMsg(e.response?.data?.detail || e.message || 'Failed'),
    })
  }

  // Pull from live gate-check endpoint (real calibrator evaluation)
  const mc = gate?.mc_prob ?? null
  const staticMc = gate?.static_mc_prob ?? null
  const mcDelta = gate?.mc_delta ?? null
  const mlProb = gate?.ml_prob ?? null
  const sigDirection = gate?.signal ?? signal?.SIGNAL
  const isBuyOrSell = sigDirection === 'BUY' || sigDirection === 'SELL'
  const gateAllowed = gate?.allowed
  const gateReason = gate?.gate_reason
  const mcFloorLive = gate?.mc_floor ?? null

  const patterns = data?.patterns
  const model = data?.model
  const markov = data?.markov
  const gates = data?.gates
  const adaptive = data?.adaptive
  const gatesOff = gates && !gates.ml_gate_enabled && !gates.adaptive_learning
  const mcFloor = mcFloorLive ?? adaptive?.mc_floor ?? null

  return (
    <div className="zcard p-5">
      <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
        <div>
          <h3 className="font-semibold text-gray-100 text-lg flex items-center gap-2">
            <span className="inline-block h-4 w-1 rounded-full bg-gold-gradient" />
            AI Trade Learning
          </h3>
          <p className="text-xs text-gray-500 mt-1 max-w-3xl leading-relaxed">
            Learns from <strong className="text-gray-300">every closed trade</strong> — Markov, ML, and adaptive
            guards update immediately to push toward a{' '}
            <span className="text-accent-gold">67% win-rate equilibrium</span>.
            {gatesOff ? (
              <>
                {' '}Learning gates are <span className="text-orange-300">OFF</span>.
              </>
            ) : (
              <>
                {' '}Mode:{' '}
                <span className={
                  adaptive?.mode === 'protect' ? 'text-red-400' :
                  adaptive?.mode === 'cautious' ? 'text-orange-300' : 'text-accent-buy'
                }>
                  {adaptive?.mode ?? 'normal'}
                </span>
                {gates?.ml_gate_enabled ? ' · ML gate ON' : ''}
                {gates?.adaptive_learning ? ' · per-trade refresh ON' : ''}
              </>
            )}
          </p>
        </div>
        {model?.trained && (
          <div className="text-right shrink-0">
            <div className="text-xs text-gray-500 uppercase tracking-wide">ML Gate Threshold</div>
            <div className="text-xl font-bold text-accent-gold tabular-nums">
              {pct(model.threshold)}
            </div>
            <div className="text-xs text-gray-500">
              Gated WR {pct(model.filtered_win_rate)} · {model.sample_count} samples
            </div>
          </div>
        )}
      </div>

      {isLoading && !data && (
        <div className="text-sm text-gray-500 animate-pulse py-6 text-center">
          Analysing {patterns?.total_closed ?? '…'} closed trades…
        </div>
      )}

      {adaptive && (
        <div className={`mb-5 rounded-lg border p-4 ${
          adaptive.mode === 'protect'
            ? 'bg-red-500/10 border-red-500/40'
            : adaptive.mode === 'cautious'
              ? 'bg-orange-500/10 border-orange-500/40'
              : 'bg-surface border-surface-border'
        }`}>
          <div className="flex flex-wrap items-start justify-between gap-3 mb-2">
            <div>
              <div className="text-xs text-gray-500 uppercase tracking-wide">Adaptive Guard (live)</div>
              <p className="text-sm text-gray-200 mt-1">{adaptive.message}</p>
            </div>
            <div className="text-right text-xs shrink-0 tabular-nums">
              <div className={adaptive.realized_pnl < 0 ? 'text-red-400 font-semibold' : 'text-accent-buy'}>
                Realized ${adaptive.realized_pnl?.toFixed(2)}
              </div>
              <div className="text-gray-500 mt-1">
                Updates: {adaptive.updates} · Streak: {adaptive.loss_streak}
              </div>
            </div>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
            <div className="bg-surface-card rounded p-2 border border-surface-border">
              <div className="text-gray-500">Rolling WR</div>
              <div className="text-gray-200">{pct(adaptive.rolling_win_rate)}</div>
            </div>
            <div className="bg-surface-card rounded p-2 border border-surface-border">
              <div className="text-gray-500">MC floor</div>
              <div className="text-accent-gold">{adaptive.mc_floor?.toFixed(0)}%</div>
            </div>
            <div className="bg-surface-card rounded p-2 border border-surface-border">
              <div className="text-gray-500">ML min prob</div>
              <div className="text-accent-gold">{pct(adaptive.min_ml_prob)}</div>
            </div>
            <div className="bg-surface-card rounded p-2 border border-surface-border">
              <div className="text-gray-500">Blocked</div>
              <div className="text-gray-300 truncate" title={[...(adaptive.blocked_states || []), ...(adaptive.blocked_hours || []).map((h) => `H${h}`)].join(', ')}>
                {(adaptive.blocked_states?.length || 0) + (adaptive.blocked_hours?.length || 0) || 'none'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Live MC + Gate + Execute / Auto-Trade controls ────────────────────── */}
      <div className="mb-5 rounded-lg border border-accent-gold/40 bg-surface p-4">

        {/* Header row: signal + auto-trade toggle */}
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div className="flex items-center gap-3">
            <div className="text-xs text-gray-500 uppercase tracking-wide">Live Signal</div>
            {isBuyOrSell ? (
              <span className={`text-sm font-bold px-2.5 py-0.5 rounded-full ${
                sigDirection === 'BUY'
                  ? 'bg-accent-buy/20 text-accent-buy border border-accent-buy/40'
                  : 'bg-accent-sell/20 text-accent-sell border border-accent-sell/40'
              }`}>
                {sigDirection}
              </span>
            ) : (
              <span className="text-sm text-gray-500">{sigDirection || '—'}</span>
            )}
            {gate?.state && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-surface-card border border-surface-border text-gray-400">
                {gate.state}
              </span>
            )}
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <span className="text-xs text-gray-400">Auto Trade</span>
            <button
              type="button"
              onClick={toggleAutoTrade}
              disabled={updateSettings.isPending}
              title={autoOn ? 'Pause auto-trading' : 'Resume auto-trading'}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none disabled:opacity-50 ${
                autoOn ? 'bg-accent-buy' : 'bg-gray-600'
              }`}
            >
              <span className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
                autoOn ? 'translate-x-6' : 'translate-x-1'
              }`} />
            </button>
            <span className={`text-xs font-semibold w-5 ${autoOn ? 'text-accent-buy' : 'text-gray-500'}`}>
              {autoOn ? 'ON' : 'OFF'}
            </span>
          </div>
        </div>

        {/* Big MC number + bar */}
        <div className="mb-4">
          <div className="flex items-end gap-3 mb-2">
            <div>
              <div className="text-[10px] text-gray-500 uppercase tracking-wide mb-0.5">Monte Carlo (Markov)</div>
              <span className={`text-3xl font-bold tabular-nums leading-none ${
                mc == null ? 'text-gray-600'
                : mc >= 67 ? 'text-accent-buy'
                : mc >= (mcFloor ?? 50) ? 'text-accent-gold'
                : 'text-accent-sell'
              }`}>
                {mc != null ? `${mc.toFixed(1)}%` : '—'}
              </span>
            </div>
            <div className="pb-1 flex flex-col gap-0.5 text-xs text-gray-500 tabular-nums">
              {staticMc != null && (
                <span>
                  Static {staticMc.toFixed(1)}%
                  {mcDelta != null && (
                    <span className={mcDelta >= 0 ? 'text-accent-buy' : 'text-accent-sell'}>
                      {' '}{mcDelta >= 0 ? '+' : ''}{mcDelta.toFixed(1)}
                    </span>
                  )}
                </span>
              )}
              {mlProb != null && (
                <span>ML score <span className="text-accent-gold">{(mlProb * 100).toFixed(1)}%</span></span>
              )}
              {mcFloor != null && (
                <span>floor <span className="text-accent-gold font-semibold">{mcFloor.toFixed(0)}%</span></span>
              )}
            </div>
          </div>
          <McBar value={mc} floor={mcFloor} />
          <div className="flex justify-between text-[10px] text-gray-600 mt-0.5 px-0.5">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
        </div>

        {/* Gate pass / fail */}
        <div className={`mb-4 rounded-md px-3 py-2 text-xs border ${
          !isBuyOrSell
            ? 'bg-surface-card border-surface-border text-gray-500'
            : gateAllowed === true
              ? 'bg-accent-buy/10 border-accent-buy/40 text-accent-buy'
              : gateAllowed === false
                ? 'bg-red-500/10 border-red-500/40 text-red-300'
                : 'bg-surface-card border-surface-border text-gray-500'
        }`}>
          {!isBuyOrSell
            ? 'No active BUY/SELL setup — MC updates on next signal'
            : gateAllowed === true
              ? '✓ Gate passed — signal meets MC + ML thresholds'
              : gateAllowed === false
                ? `✗ Gate blocked — ${gateReason || 'ML/MC threshold not met'}`
                : 'Evaluating…'}
        </div>

        {/* Execute (override) button + feedback */}
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            disabled={!isBuyOrSell || executeTrade.isPending || status?.trading_halted}
            onClick={handleExecute}
            className={`px-5 py-2.5 rounded-lg font-bold text-sm transition shrink-0 ${
              isBuyOrSell && !status?.trading_halted
                ? sigDirection === 'BUY'
                  ? 'bg-accent-buy text-white hover:opacity-90 active:scale-95'
                  : 'bg-accent-sell text-white hover:opacity-90 active:scale-95'
                : 'bg-gray-700 text-gray-500 cursor-not-allowed'
            }`}
          >
            {status?.trading_halted
              ? 'Trading halted'
              : executeTrade.isPending
                ? 'Executing…'
                : isBuyOrSell
                  ? `⚡ Override Gate & Execute ${sigDirection}`
                  : 'No BUY/SELL signal'}
          </button>
          <span className="text-[10px] text-gray-600 max-w-xs leading-snug">
            Bypasses MC/ML gate — trades immediately at current signal levels
          </span>
        </div>
        {executeMsg && (
          <div className={`mt-2 text-xs px-3 py-1.5 rounded border ${
            executeMsg.toLowerCase().includes('fail') || executeMsg.toLowerCase().includes('error')
              ? 'bg-red-500/10 border-red-500/30 text-red-300'
              : 'bg-accent-buy/10 border-accent-buy/30 text-accent-buy'
          }`}>
            {executeMsg}
          </div>
        )}
      </div>
      {/* ──────────────────────────────────────────────────────────────────────── */}

      {markov && (
        <div className="mb-5 bg-surface rounded-lg border border-accent-gold/30 p-4">
          <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
            <div>
              <div className="text-xs text-gray-500 uppercase tracking-wide">Markov Feedback Loop</div>
              <p className="text-xs text-gray-400 mt-1 max-w-2xl">
                Learns state→state transitions from signal history and per-state drift/vol from closed trades.
                Feeds improved Monte Carlo + ML predictions for future setups.
              </p>
            </div>
            <div className="text-right text-xs text-gray-500 shrink-0">
              <div>Source: <span className="text-accent-gold">{markov.transition_source}</span></div>
              <div>{markov.state_sequence_length ?? 0} state snapshots</div>
            </div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
            <div className="bg-surface-card rounded-lg p-3 border border-surface-border">
              <div className="text-gray-500 uppercase tracking-wide mb-1">Calibration</div>
              <div className="text-gray-200 tabular-nums">
                {markov.calibration?.brier != null
                  ? `Brier ${markov.calibration.brier} · ${markov.calibration.samples} trades`
                  : 'Collecting samples…'}
              </div>
              {markov.calibration?.predicted_wr != null && (
                <div className="text-gray-500 mt-1 tabular-nums">
                  Pred {pct(markov.calibration.predicted_wr)} vs actual {pct(markov.calibration.actual_wr)}
                </div>
              )}
            </div>
            <div className="bg-surface-card rounded-lg p-3 border border-surface-border">
              <div className="text-gray-500 uppercase tracking-wide mb-1">Top Transitions</div>
              <ul className="space-y-1 text-gray-400">
                {markov.top_transitions?.slice(0, 3).map((t) => (
                  <li key={t.from} className="tabular-nums">
                    {t.from} → {t.to} <span className="text-accent-gold">{(t.prob * 100).toFixed(0)}%</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="bg-surface-card rounded-lg p-3 border border-surface-border">
              <div className="text-gray-500 uppercase tracking-wide mb-1">Best States (WR)</div>
              <ul className="space-y-1 text-gray-400">
                {Object.entries(markov.state_win_rates || {})
                  .filter(([, v]) => v.win_rate != null && v.count >= 10)
                  .sort((a, b) => (b[1].win_rate || 0) - (a[1].win_rate || 0))
                  .slice(0, 3)
                  .map(([state, v]) => (
                    <li key={state} className="tabular-nums">
                      {state.replace('_', ' ')} <span className="text-accent-buy">{pct(v.win_rate)}</span>
                    </li>
                  ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {patterns && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div className="bg-surface rounded-lg border border-surface-border p-4 space-y-3">
            <div className="text-xs text-gray-500 uppercase tracking-wide">Win Rate by SL Size</div>
            {patterns.risk_buckets?.map((b) => (
              <Bar
                key={b.label}
                label={b.label}
                winRate={b.win_rate}
                count={b.count}
                highlight={b.win_rate >= 0.67}
              />
            ))}
            <div className="pt-2 border-t border-surface-border text-xs text-gray-500">
              Overall: <span className="text-gray-300">{pct(patterns.overall_win_rate)}</span>
              {' '}→ Target: <span className="text-accent-gold">{pct(patterns.target_win_rate)}</span>
            </div>
          </div>

          <div className="bg-surface rounded-lg border border-surface-border p-4 space-y-3">
            <div className="text-xs text-gray-500 uppercase tracking-wide">Best / Worst Hours (UTC)</div>
            {patterns.best_hours?.slice(0, 3).map((b) => (
              <Bar key={b.label} label={b.label} winRate={b.win_rate} count={b.count} highlight />
            ))}
            {patterns.worst_hours?.map((b) => (
              <Bar key={b.label} label={b.label} winRate={b.win_rate} count={b.count} />
            ))}
          </div>

          <div className="bg-surface rounded-lg border border-surface-border p-4">
            <div className="text-xs text-gray-500 uppercase tracking-wide mb-3">Key Insights</div>
            <ul className="space-y-2 text-xs text-gray-400">
              {patterns.insights?.map((line) => (
                <li key={line} className="flex gap-2">
                  <span className="text-accent-gold shrink-0">•</span>
                  <span>{line}</span>
                </li>
              ))}
            </ul>
            {model?.feature_weights && Object.keys(model.feature_weights).length > 0 && (
              <div className="mt-4 pt-3 border-t border-surface-border">
                <div className="text-xs text-gray-500 uppercase tracking-wide mb-2">Top ML Features</div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(model.feature_weights)
                    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                    .slice(0, 5)
                    .map(([name, weight]) => (
                      <span
                        key={name}
                        className="text-xs px-2 py-1 rounded bg-surface-card border border-surface-border text-gray-400 tabular-nums"
                      >
                        {name}{' '}
                        <span className={weight > 0 ? 'text-accent-buy' : 'text-accent-sell'}>
                          {weight > 0 ? '+' : ''}
                          {weight}
                        </span>
                      </span>
                    ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

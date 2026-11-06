import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import Layout from '../components/Layout'
import api from '../services/api'

const TRADE_MODE_OPTIONS = [
  { value: 'paper', label: 'Paper (simulated)' },
  { value: 'demo', label: 'XM Demo' },
  { value: 'live', label: 'XM Live' },
]

function fmtTs(ts) {
  if (!ts) return '—'
  try {
    return new Date(ts * 1000).toLocaleTimeString()
  } catch {
    return '—'
  }
}

function kindTone(kind) {
  if (kind === 'PUNCH') return 'text-emerald-400'
  if (kind === 'TP1') return 'text-emerald-300'
  if (kind === 'EXIT') return 'text-amber-300'
  if (kind === 'SKIP') return 'text-zinc-500'
  if (kind === 'ARMED') return 'text-cyan-300'
  if (kind === 'STANDBY') return 'text-zinc-400'
  if (kind === 'HALT') return 'text-rose-400'
  return 'text-zinc-300'
}

function fmtMoney(v, signed = false) {
  if (v == null || !Number.isFinite(Number(v))) return '—'
  try {
    const n = Number(v)
    const body = `$${Math.abs(n).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`
    if (!signed) return n < 0 ? `-${body}` : body
    if (n > 0) return `+${body}`
    if (n < 0) return `-${body}`
    return body
  } catch {
    return '—'
  }
}

export default function HyperScalpPage() {
  const qc = useQueryClient()
  const [lotMult, setLotMult] = useState(1)
  const [hold, setHold] = useState(1500)
  const [dispAtr, setDispAtr] = useState(0.65)
  const [pullAtr, setPullAtr] = useState(0.85)
  const [costMult, setCostMult] = useState(3.5)
  const [tp1Frac, setTp1Frac] = useState(0.5)
  const [riskPct, setRiskPct] = useState(10)
  const [emaTf, setEmaTf] = useState('1m')
  const [cooldown, setCooldown] = useState(45)

  const { data, isLoading } = useQuery({
    queryKey: ['hyper-scalp'],
    queryFn: async () => (await api.get('/api/hyper-scalp/status')).data,
    refetchInterval: 1000,
  })

  const { data: tradeStatus } = useQuery({
    queryKey: ['trade-status'],
    queryFn: async () => (await api.get('/api/trades/status')).data,
    refetchInterval: 5000,
  })

  const { data: settings } = useQuery({
    queryKey: ['trade-settings'],
    queryFn: async () => (await api.get('/api/trades/settings')).data,
    refetchInterval: 5000,
  })

  useEffect(() => {
    if (data?.lot_mult != null) setLotMult(Number(data.lot_mult))
    if (data?.max_hold_sec != null) setHold(Number(data.max_hold_sec))
    if (data?.disp_atr != null) setDispAtr(Number(data.disp_atr))
    if (data?.pull_max_atr != null) setPullAtr(Number(data.pull_max_atr))
    if (data?.cost_mult != null) setCostMult(Number(data.cost_mult))
    if (data?.tp1_frac != null) setTp1Frac(Number(data.tp1_frac))
    if (data?.risk_pct != null) setRiskPct(Number(data.risk_pct))
    if (data?.ema_tf != null) setEmaTf(String(data.ema_tf))
    if (data?.cooldown_sec != null) setCooldown(Number(data.cooldown_sec))
  }, [
    data?.lot_mult,
    data?.max_hold_sec,
    data?.disp_atr,
    data?.pull_max_atr,
    data?.cost_mult,
    data?.tp1_frac,
    data?.risk_pct,
    data?.ema_tf,
    data?.cooldown_sec,
  ])

  const toggle = useMutation({
    mutationFn: async (enabled) =>
      (await api.post('/api/hyper-scalp/toggle', { enabled })).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['hyper-scalp'] }),
  })

  const saveSettings = useMutation({
    mutationFn: async (body) =>
      (await api.post('/api/hyper-scalp/settings', body)).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['hyper-scalp'] }),
  })

  const resetPnl = useMutation({
    mutationFn: async () => (await api.post('/api/hyper-scalp/reset-pnl')).data,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['hyper-scalp'] }),
  })

  const updateMode = useMutation({
    mutationFn: async (trade_mode) =>
      (await api.patch('/api/trades/settings', { trade_mode })).data,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['trade-status'] })
      qc.invalidateQueries({ queryKey: ['trade-settings'] })
      qc.invalidateQueries({ queryKey: ['trade-summary'] })
      qc.invalidateQueries({ queryKey: ['hyper-scalp'] })
    },
  })

  const armed = Boolean(data?.enabled)
  const feed = useMemo(() => [...(data?.feed || [])].reverse(), [data?.feed])
  const opens = Object.entries(data?.open || {})
  const mode = settings?.trade_mode || tradeStatus?.trade_mode || 'paper'
  const accounts = tradeStatus?.broker_accounts || {}
  const brokerOk = Boolean(tradeStatus?.broker_configured)

  const switchMode = (next) => {
    if (next === 'demo' && accounts.demo === false) {
      if (
        !window.confirm(
          'METAAPI_ACCOUNT_ID_DEMO is not set in .env. Switch to Demo anyway?'
        )
      )
        return
    }
    if (next === 'live' && accounts.live === false) {
      if (
        !window.confirm(
          'METAAPI_ACCOUNT_ID_LIVE is not set in .env. Live orders will fail until you add it. Switch anyway?'
        )
      )
        return
    }
    if (next === 'live') {
      if (
        !window.confirm(
          'Switch Hyper + desk to XM LIVE? Real money orders will be sent.'
        )
      )
        return
    }
    if (next === 'demo') {
      if (
        !window.confirm(
          'Switch Hyper + desk to XM DEMO? Orders go to your practice account.'
        )
      )
        return
    }
    updateMode.mutate(next)
  }

  return (
    <Layout>
      <div className="relative -mx-5 sm:-mx-10 -mt-10 sm:-mt-12 overflow-hidden">
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              'radial-gradient(ellipse 80% 50% at 20% 0%, rgba(34,211,238,0.12), transparent 55%), radial-gradient(ellipse 60% 40% at 90% 20%, rgba(251,191,36,0.08), transparent 50%), linear-gradient(180deg,#05070a 0%,#080a0e 45%,#050607 100%)',
          }}
        />
        <div
          className="absolute inset-0 opacity-[0.07] pointer-events-none"
          style={{
            backgroundImage:
              'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.35) 3px)',
          }}
        />

        <section className="relative px-5 sm:px-10 pt-12 pb-10 max-w-7xl mx-auto">
          <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-8">
            <div>
              <p className="text-[11px] tracking-[0.35em] uppercase text-cyan-400/80 font-medium mb-3">
                Zenith · Hyper Scalp
              </p>
              <h1 className="font-[Syne,sans-serif] text-4xl sm:text-5xl font-bold tracking-tight text-white leading-[1.05]">
                Machine speed
                <span className="block text-cyan-300">across every market</span>
              </h1>
              <p className="mt-4 max-w-xl text-sm sm:text-[15px] text-zinc-400 leading-relaxed">
                Parked — Zenith-only desk · set HYPER_SCALP_ARMED=true to allow arming
                pullback · TP1 partial at +1R → SL to breakeven · TP must clear{' '}
                {data?.cost_mult ?? 3.5}× XM spread · max {data?.max_open ?? 2} opens ·{' '}
                {data?.cooldown_sec ?? 45}s cooldown.
              </p>
              <span className="mt-3 inline-block text-[10px] tracking-[0.25em] uppercase text-cyan-300/80 border border-cyan-500/30 px-3 py-1">
                Separate engine · own RL · never feeds Dashboard
              </span>
            </div>

            <div className="flex flex-col items-stretch sm:items-end gap-3 min-w-[240px]">
              <div className="w-full border border-white/10 bg-black/50 px-4 py-3">
                <div className="text-[10px] tracking-[0.25em] uppercase text-zinc-500 mb-2">
                  XM account
                </div>
                <select
                  value={mode}
                  disabled={updateMode.isPending}
                  onChange={(e) => switchMode(e.target.value)}
                  className={`w-full bg-transparent border-0 border-b rounded-none px-0 py-1.5 text-sm font-medium focus:outline-none disabled:opacity-50 ${
                    mode === 'live'
                      ? 'border-rose-400/60 text-rose-300'
                      : mode === 'demo'
                        ? 'border-amber-400/50 text-amber-300'
                        : 'border-white/20 text-zinc-300'
                  }`}
                >
                  {TRADE_MODE_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value} className="bg-zinc-900 text-white">
                      {opt.label}
                    </option>
                  ))}
                </select>
                <div
                  className={`mt-2 text-[10px] tracking-wide ${
                    mode === 'live'
                      ? 'text-rose-400'
                      : mode === 'demo'
                        ? 'text-amber-400/90'
                        : 'text-zinc-500'
                  }`}
                >
                  {mode === 'live'
                    ? brokerOk
                      ? 'LIVE — real money · same desk as Dashboard'
                      : 'LIVE selected — MetaAPI live account not configured'
                    : mode === 'demo'
                      ? brokerOk
                        ? 'DEMO — practice funds · shared with Dashboard'
                        : 'DEMO selected — MetaAPI demo not configured'
                      : 'PAPER — simulated only, no XM orders'}
                </div>
              </div>

              <button
                type="button"
                disabled={toggle.isPending}
                onClick={() => toggle.mutate(!armed)}
                className={`relative px-8 py-4 rounded-sm font-[Syne,sans-serif] text-sm tracking-[0.2em] uppercase font-semibold transition-all ${
                  armed
                    ? 'bg-cyan-400 text-black shadow-[0_0_40px_rgba(34,211,238,0.35)]'
                    : 'bg-zinc-900 text-zinc-300 border border-zinc-700 hover:border-cyan-500/50'
                }`}
              >
                {armed ? '● ARMED — LIVE FIRE' : '○ STANDBY — ARM ENGINE'}
              </button>
              <p className="text-[11px] text-zinc-500 tracking-wide text-right">
                Worker {data?.worker_alive ? 'alive' : '…'} · cycle{' '}
                {data?.cycle_sec ?? 1}s · open {data?.open_count ?? 0}/{data?.max_open ?? 6}
              </p>
            </div>
          </div>

          <div className="mt-10 grid grid-cols-2 lg:grid-cols-4 gap-3">
            {[
              {
                label: 'Capital',
                value: fmtMoney(data?.capital ?? data?.last_balance),
                tone: 'text-white',
              },
              {
                label: 'Equity',
                value: fmtMoney(data?.equity ?? data?.last_equity),
                tone: 'text-cyan-300',
              },
              {
                label: 'Realised',
                value: fmtMoney(data?.realized_pnl ?? 0, true),
                tone:
                  (data?.realized_pnl ?? 0) > 0
                    ? 'text-emerald-300'
                    : (data?.realized_pnl ?? 0) < 0
                      ? 'text-rose-300'
                      : 'text-zinc-300',
              },
              {
                label: 'Floating',
                value: fmtMoney(data?.floating_pnl ?? data?.last_floating ?? 0, true),
                tone:
                  (data?.floating_pnl ?? data?.last_floating ?? 0) > 0
                    ? 'text-emerald-300'
                    : (data?.floating_pnl ?? data?.last_floating ?? 0) < 0
                      ? 'text-rose-300'
                      : 'text-zinc-300',
              },
            ].map((c) => (
              <div
                key={c.label}
                className="border border-white/10 bg-black/40 px-4 py-4 backdrop-blur-sm"
              >
                <div className="text-[10px] tracking-[0.25em] uppercase text-zinc-500">
                  {c.label}
                </div>
                <div className={`mt-2 font-[Syne,sans-serif] text-3xl tabular-nums ${c.tone}`}>
                  {isLoading ? '—' : c.value}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-3 grid grid-cols-2 lg:grid-cols-4 gap-3">
            {[
              {
                label: 'Trades / min',
                value: `${data?.tpm ?? 0}/${data?.target_tpm ?? 100}`,
                tone:
                  (data?.tpm ?? 0) >= (data?.target_tpm ?? 100) * 0.7
                    ? 'text-emerald-300'
                    : 'text-cyan-300',
              },
              { label: 'Punches', value: data?.punches ?? 0, tone: 'text-cyan-300' },
              { label: 'Fast exits', value: data?.closes ?? 0, tone: 'text-amber-300' },
              {
                label: 'Session PnL',
                value: fmtMoney(data?.session_pnl ?? 0, true),
                tone:
                  (data?.session_pnl ?? 0) > 0
                    ? 'text-emerald-300'
                    : (data?.session_pnl ?? 0) < 0
                      ? 'text-rose-300'
                      : 'text-zinc-300',
              },
            ].map((c) => (
              <div
                key={c.label}
                className="border border-white/10 bg-black/40 px-4 py-4 backdrop-blur-sm"
              >
                <div className="text-[10px] tracking-[0.25em] uppercase text-zinc-500">
                  {c.label}
                </div>
                <div className={`mt-2 font-[Syne,sans-serif] text-3xl tabular-nums ${c.tone}`}>
                  {isLoading ? '—' : c.value}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
            <p className="text-[11px] text-zinc-500 tracking-wide">
              Session / Realised = Hyper closed trades only · XM deposits never count as profit
            </p>
            <button
              type="button"
              disabled={resetPnl.isPending}
              onClick={() => {
                if (window.confirm('Zero Session + Realised and re-pin risk baseline to live XM equity?')) {
                  resetPnl.mutate()
                }
              }}
              className="px-4 py-2 text-[10px] tracking-[0.2em] uppercase border border-white/20 text-zinc-300 hover:border-cyan-400/50 hover:text-cyan-300 disabled:opacity-50"
            >
              Reset Session PnL
            </button>
          </div>

          <div className="mt-4 border border-white/10 bg-black/40 px-4 py-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] tracking-[0.25em] uppercase text-zinc-500">
                Capital risk budget
              </span>
              <span
                className={`text-[11px] tracking-wide ${
                  data?.halted ? 'text-rose-400' : 'text-zinc-400'
                }`}
              >
                {data?.halted
                  ? 'HALTED — risk stop hit'
                  : `stop at ${fmtMoney(data?.equity_floor)} (−${data?.risk_pct ?? 20}%)`}
              </span>
            </div>
            <div className="h-2 w-full bg-white/5 overflow-hidden rounded-sm">
              <div
                className={`h-full transition-all ${
                  (data?.risk_used_pct ?? 0) >= (data?.risk_pct ?? 20)
                    ? 'bg-rose-500'
                    : (data?.risk_used_pct ?? 0) >= (data?.risk_pct ?? 20) * 0.6
                      ? 'bg-amber-400'
                      : 'bg-emerald-400'
                }`}
                style={{
                  width: `${Math.min(
                    100,
                    ((data?.risk_used_pct ?? 0) / (data?.risk_pct || 20)) * 100
                  )}%`,
                }}
              />
            </div>
            <div className="mt-2 flex justify-between text-[11px] text-zinc-500 tabular-nums">
              <span>
                capital {fmtMoney(data?.capital ?? data?.last_balance)} · equity{' '}
                {fmtMoney(data?.equity ?? data?.last_equity)} / start{' '}
                {fmtMoney(data?.start_equity)}
              </span>
              <span
                className={
                  (data?.drawdown_pct ?? 0) > 0 ? 'text-rose-300' : 'text-emerald-300'
                }
              >
                {data?.drawdown_pct == null
                  ? 'baseline set on arm'
                  : `${data.drawdown_pct > 0 ? '−' : '+'}${Math.abs(
                      data.drawdown_pct
                    ).toFixed(1)}% used of ${data?.risk_pct ?? 20}%`}
              </span>
            </div>
            <div className="mt-1 text-[10px] text-zinc-600">
              Realised / Session = Hyper closed exits only · XM deposits ignored · Floating from XM account
            </div>
          </div>

          {data?.halted && (
            <div className="mt-4 flex items-center justify-between gap-4 border border-rose-500/40 bg-rose-500/10 px-4 py-3">
              <p className="text-sm text-rose-300">
                {data?.halt_reason || 'Hyper halted — capital stop hit.'}
              </p>
              <button
                type="button"
                onClick={() => toggle.mutate(true)}
                disabled={toggle.isPending}
                className="shrink-0 px-4 py-2 text-xs tracking-[0.2em] uppercase border border-rose-400/50 text-rose-200 hover:bg-rose-500/20"
              >
                Re-arm (reset baseline)
              </button>
            </div>
          )}

          <div className="mt-8 grid lg:grid-cols-[1.1fr_0.9fr] gap-6">
            <div className="border border-white/10 bg-black/50 p-5">
              <h2 className="font-[Syne,sans-serif] text-lg text-white mb-4 tracking-tight">
                Live punch tape
              </h2>
              <div className="h-[340px] overflow-y-auto space-y-1.5 font-mono text-[12px]">
                {feed.length === 0 && (
                  <p className="text-zinc-600 py-8 text-center">
                    Waiting for a setup — 5m trend → 1m displacement → pullback entry.
                  </p>
                )}
                {feed.map((row, i) => (
                  <div
                    key={`${row.ts}-${i}`}
                    className="flex gap-3 border-b border-white/[0.04] py-1.5"
                  >
                    <span className="text-zinc-600 w-16 shrink-0">{fmtTs(row.ts)}</span>
                    <span className={`w-14 shrink-0 font-semibold ${kindTone(row.kind)}`}>
                      {row.kind}
                    </span>
                    <span className="text-zinc-300 truncate">{row.detail}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-6">
              <div className="border border-white/10 bg-black/50 p-5">
                <h2 className="font-[Syne,sans-serif] text-lg text-white mb-4">Aggression</h2>
                <label className="block text-[11px] uppercase tracking-wider text-zinc-500 mb-2">
                  Lot multiplier ({lotMult}× desk)
                </label>
                <input
                  type="range"
                  min={1}
                  max={3}
                  step={0.25}
                  value={lotMult}
                  onChange={(e) => setLotMult(Number(e.target.value))}
                  className="w-full accent-cyan-400"
                />
                <label className="block text-[11px] uppercase tracking-wider text-zinc-500 mt-5 mb-2">
                  Entry heat TF
                </label>
                <div className="grid grid-cols-3 gap-1.5">
                  {['1m', '2m', '5m'].map((tf) => (
                    <button
                      key={tf}
                      type="button"
                      onClick={() => setEmaTf(tf)}
                      className={`py-1.5 text-xs tracking-wide border ${
                        emaTf === tf
                          ? 'border-cyan-400 text-cyan-300 bg-cyan-400/10'
                          : 'border-white/10 text-zinc-400 hover:border-white/25'
                      }`}
                    >
                      {tf}
                    </button>
                  ))}
                </div>
                <label className="block text-[11px] uppercase tracking-wider text-zinc-500 mt-5 mb-2">
                  Safety cap ({Math.round(hold / 60)}m) — exits run on TP/SL
                </label>
                <input
                  type="range"
                  min={60}
                  max={1800}
                  step={60}
                  value={hold}
                  onChange={(e) => setHold(Number(e.target.value))}
                  className="w-full accent-amber-400"
                />
                <label className="block text-[11px] uppercase tracking-wider text-zinc-500 mt-5 mb-2">
                  Cooldown / symbol ({cooldown}s)
                </label>
                <input
                  type="range"
                  min={2}
                  max={60}
                  step={1}
                  value={cooldown}
                  onChange={(e) => setCooldown(Number(e.target.value))}
                  className="w-full accent-cyan-400"
                />
                <label className="block text-[11px] uppercase tracking-wider text-zinc-500 mt-5 mb-2">
                  Displacement bar ≥ {dispAtr.toFixed(2)}×ATR
                </label>
                <input
                  type="range"
                  min={0.4}
                  max={1.2}
                  step={0.05}
                  value={dispAtr}
                  onChange={(e) => setDispAtr(Number(e.target.value))}
                  className="w-full accent-emerald-400"
                />
                <label className="block text-[11px] uppercase tracking-wider text-zinc-500 mt-5 mb-2">
                  Pullback within {pullAtr.toFixed(2)}×ATR of EMA mid
                </label>
                <input
                  type="range"
                  min={0.4}
                  max={1.5}
                  step={0.05}
                  value={pullAtr}
                  onChange={(e) => setPullAtr(Number(e.target.value))}
                  className="w-full accent-amber-400"
                />
                <label className="block text-[11px] uppercase tracking-wider text-zinc-500 mt-5 mb-2">
                  XM cost gate — TP ≥ {costMult.toFixed(1)}× round-trip spread
                </label>
                <input
                  type="range"
                  min={2}
                  max={6}
                  step={0.5}
                  value={costMult}
                  onChange={(e) => setCostMult(Number(e.target.value))}
                  className="w-full accent-cyan-400"
                />
                <label className="block text-[11px] uppercase tracking-wider text-zinc-500 mt-5 mb-2">
                  TP1 partial — book {(tp1Frac * 100).toFixed(0)}% at +1R, SL → BE
                </label>
                <input
                  type="range"
                  min={0.2}
                  max={0.8}
                  step={0.05}
                  value={tp1Frac}
                  onChange={(e) => setTp1Frac(Number(e.target.value))}
                  className="w-full accent-emerald-400"
                />
                <label className="block text-[11px] uppercase tracking-wider text-zinc-500 mt-5 mb-2">
                  Capital stop — halt at −{riskPct}% equity
                </label>
                <input
                  type="range"
                  min={1}
                  max={50}
                  step={1}
                  value={riskPct}
                  onChange={(e) => setRiskPct(Number(e.target.value))}
                  className="w-full accent-rose-500"
                />
                <button
                  type="button"
                  className="mt-5 w-full py-2.5 border border-cyan-500/40 text-cyan-300 text-xs tracking-[0.2em] uppercase hover:bg-cyan-400/10"
                  disabled={saveSettings.isPending}
                  onClick={() =>
                    saveSettings.mutate({
                      lot_mult: lotMult,
                      max_hold_sec: hold,
                      disp_atr: dispAtr,
                      pull_max_atr: pullAtr,
                      cost_mult: costMult,
                      tp1_frac: tp1Frac,
                      risk_pct: riskPct,
                      ema_tf: emaTf,
                      cooldown_sec: cooldown,
                    })
                  }
                >
                  Apply settings
                </button>
              </div>

              <div className="border border-white/10 bg-black/50 p-5">
                <h2 className="font-[Syne,sans-serif] text-lg text-white mb-3">Hot opens</h2>
                {opens.length === 0 ? (
                  <p className="text-zinc-600 text-sm">No hyper positions open.</p>
                ) : (
                  <ul className="space-y-2 text-sm">
                    {opens.map(([tag, meta]) => (
                      <li
                        key={tag}
                        className="flex justify-between gap-2 border border-white/5 px-3 py-2"
                      >
                        <span className="text-zinc-200">
                          {meta.symbol}{' '}
                          <span
                            className={
                              meta.signal === 'BUY' ? 'text-emerald-400' : 'text-rose-400'
                            }
                          >
                            {meta.signal}
                          </span>
                          {meta.partial_done && (
                            <span className="ml-2 text-[10px] uppercase tracking-wide text-emerald-300/80">
                              TP1 · BE
                            </span>
                          )}
                        </span>
                        <span className="text-zinc-500 tabular-nums">{meta.lot} lot</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>

          <div className="mt-8 border border-white/10 bg-black/50 p-5">
            <div className="flex items-baseline justify-between mb-3">
              <h2 className="font-[Syne,sans-serif] text-lg text-white tracking-tight">
                5m Value Bias
              </h2>
              <span className="text-[10px] tracking-[0.25em] uppercase text-zinc-500">
                Structure context · punches need 5m trend + 1m pullback
              </span>
            </div>
            {(data?.bias_5m?.symbols || []).length === 0 ? (
              <p className="text-zinc-600 text-sm">
                Waiting for 5m candles — bias appears after the first warm cycle.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-[12px]">
                  <thead className="text-[10px] uppercase tracking-wider text-zinc-500">
                    <tr>
                      <th className="py-2 pr-3 font-medium">Symbol</th>
                      <th className="py-2 pr-3 font-medium">Bias</th>
                      <th className="py-2 pr-3 font-medium">Side</th>
                      <th className="py-2 font-medium">Imbalance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(data?.bias_5m?.symbols || []).map((row) => (
                      <tr key={row.symbol} className="border-t border-white/[0.04]">
                        <td className="py-2 pr-3 text-zinc-200 font-medium">{row.symbol}</td>
                        <td
                          className={`py-2 pr-3 font-semibold ${
                            row.direction === 'BULLISH'
                              ? 'text-emerald-400'
                              : row.direction === 'BEARISH'
                                ? 'text-rose-400'
                                : 'text-zinc-400'
                          }`}
                        >
                          {row.direction || '—'}
                        </td>
                        <td
                          className={`py-2 pr-3 ${
                            row.signal === 'BUY' ? 'text-emerald-300' : 'text-rose-300'
                          }`}
                        >
                          {row.signal || '—'}
                        </td>
                        <td
                          className={`py-2 tabular-nums ${
                            (row.imbalance ?? 0) > 0
                              ? 'text-emerald-300'
                              : (row.imbalance ?? 0) < 0
                                ? 'text-rose-300'
                                : 'text-zinc-400'
                          }`}
                        >
                          {row.imbalance == null
                            ? '—'
                            : `${row.imbalance > 0 ? '+' : ''}${Number(row.imbalance).toFixed(3)}`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="mt-8 border border-white/10 bg-black/50 p-5">
            <div className="flex items-baseline justify-between mb-3">
              <h2 className="font-[Syne,sans-serif] text-lg text-white tracking-tight">
                Momentum Board
              </h2>
              <span className="text-[10px] tracking-[0.25em] uppercase text-zinc-500">
                READY = 5m trend + 1m displacement + pullback · cost gate passed
              </span>
            </div>
            {(data?.flow?.flows || []).length === 0 ? (
              <p className="text-zinc-600 text-sm">
                Probing every ~12s — waiting for a 5m trend with a 1m displacement pullback.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-[12px]">
                  <thead className="text-[10px] uppercase tracking-wider text-zinc-500">
                    <tr>
                      <th className="py-2 pr-3 font-medium">Symbol</th>
                      <th className="py-2 pr-3 font-medium">Status</th>
                      <th className="py-2 pr-3 font-medium">Side</th>
                      <th className="py-2 pr-3 font-medium">Disp ×ATR</th>
                      <th className="py-2 pr-3 font-medium">Pull ×ATR</th>
                      <th className="py-2 pr-3 font-medium">Score</th>
                      <th className="py-2 font-medium">Why</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(data?.flow?.flows || []).map((row) => (
                      <tr key={row.symbol} className="border-t border-white/[0.04]">
                        <td className="py-2 pr-3 text-zinc-200 font-medium">{row.symbol}</td>
                        <td
                          className={`py-2 pr-3 font-semibold ${
                            row.status === 'READY'
                              ? 'text-emerald-400'
                              : row.status === 'COST_KILL' || row.status === 'DEAD_TAPE'
                                ? 'text-rose-300'
                                : row.status === 'NO_BIAS' || row.status === 'NO_DISPLACEMENT'
                                  ? 'text-zinc-500'
                                  : row.status === 'NO_PULLBACK' || row.status === 'TIGHT_STOP'
                                    ? 'text-amber-300'
                                    : 'text-zinc-400'
                          }`}
                        >
                          {row.status || '—'}
                        </td>
                        <td
                          className={`py-2 pr-3 font-semibold ${
                            row.signal === 'BUY' ? 'text-emerald-400' : row.signal === 'SELL' ? 'text-rose-400' : 'text-zinc-500'
                          }`}
                        >
                          {row.signal || '—'}
                        </td>
                        <td className="py-2 pr-3 tabular-nums text-zinc-300">
                          {row.disp_atr == null ? '—' : Number(row.disp_atr).toFixed(2)}
                        </td>
                        <td className="py-2 pr-3 tabular-nums text-zinc-300">
                          {row.pull_atr == null ? '—' : Number(row.pull_atr).toFixed(2)}
                        </td>
                        <td className="py-2 pr-3 tabular-nums text-amber-200/90">
                          {row.score == null ? '—' : Number(row.score).toFixed(2)}
                        </td>
                        <td className="py-2 text-zinc-500 truncate max-w-[220px]">
                          {row.skip || row.bias_5m || '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="mt-8 border border-white/10 bg-black/50 p-5">
            <div className="flex items-baseline justify-between mb-3">
              <h2 className="font-[Syne,sans-serif] text-lg text-white tracking-tight">
                Micro FRVP
              </h2>
              <span className="text-[10px] tracking-[0.25em] uppercase text-zinc-500">
                30 × {emaTf} · pullback filter (bps, not ATR)
              </span>
            </div>
            {(data?.micro_vp?.profiles || []).length === 0 ? (
              <p className="text-zinc-600 text-sm">
                Warming {emaTf} candles — micro profiles appear after the first VP refresh.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-[12px]">
                  <thead className="text-[10px] uppercase tracking-wider text-zinc-500">
                    <tr>
                      <th className="py-2 pr-3 font-medium">Symbol</th>
                      <th className="py-2 pr-3 font-medium">Loc</th>
                      <th className="py-2 pr-3 font-medium">POC</th>
                      <th className="py-2 pr-3 font-medium">VAL</th>
                      <th className="py-2 pr-3 font-medium">VAH</th>
                      <th className="py-2 font-medium">Dist POC</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(data?.micro_vp?.profiles || []).map((row) => (
                      <tr key={row.symbol} className="border-t border-white/[0.04]">
                        <td className="py-2 pr-3 text-zinc-200 font-medium">{row.symbol}</td>
                        <td className="py-2 pr-3 text-zinc-300">{row.location || '—'}</td>
                        <td className="py-2 pr-3 tabular-nums text-cyan-300">
                          {row.poc ?? '—'}
                        </td>
                        <td className="py-2 pr-3 tabular-nums text-zinc-400">
                          {row.val ?? '—'}
                        </td>
                        <td className="py-2 pr-3 tabular-nums text-zinc-400">
                          {row.vah ?? '—'}
                        </td>
                        <td className="py-2 tabular-nums text-zinc-300">
                          {row.dist_to_poc_atr == null
                            ? '—'
                            : `${row.dist_to_poc_atr > 0 ? '+' : ''}${row.dist_to_poc_atr} (legacy)`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="mt-8 border border-white/10 bg-black/50 p-5">
            <div className="flex items-baseline justify-between mb-3">
              <h2 className="font-[Syne,sans-serif] text-lg text-white tracking-tight">
                Candle walk-forward
              </h2>
              <span className="text-[10px] tracking-[0.25em] uppercase text-zinc-500">
                every 1m bar · paper edge · retunes cost/disp/pull · history intact
              </span>
            </div>
            {(data?.candle_edge?.symbols || []).length === 0 ? (
              <p className="text-zinc-600 text-sm">
                Waiting for the next 1m close — both desks learn from forming candles, not only exits.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-[12px]">
                  <thead className="text-[10px] uppercase tracking-wider text-zinc-500">
                    <tr>
                      <th className="py-2 pr-3 font-medium">Symbol</th>
                      <th className="py-2 pr-3 font-medium">E[R]</th>
                      <th className="py-2 pr-3 font-medium">WR</th>
                      <th className="py-2 pr-3 font-medium">Disp</th>
                      <th className="py-2 pr-3 font-medium">Cost×</th>
                      <th className="py-2 pr-3 font-medium">Lot×</th>
                      <th className="py-2 font-medium">Note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(data?.candle_edge?.symbols || []).map((row) => {
                      const h = row.hyper || {}
                      return (
                        <tr key={row.symbol} className="border-t border-white/[0.04]">
                          <td className="py-2 pr-3 text-zinc-200 font-medium">{row.symbol}</td>
                          <td
                            className={`py-2 pr-3 tabular-nums ${
                              (h.exp_r ?? 0) > 0
                                ? 'text-emerald-400'
                                : (h.exp_r ?? 0) < 0
                                  ? 'text-rose-400'
                                  : 'text-zinc-400'
                            }`}
                          >
                            {h.exp_r == null
                              ? '—'
                              : `${h.exp_r > 0 ? '+' : ''}${Number(h.exp_r).toFixed(3)}`}
                          </td>
                          <td className="py-2 pr-3 tabular-nums text-zinc-300">
                            {h.wr == null ? '—' : `${(Number(h.wr) * 100).toFixed(0)}%`}
                          </td>
                          <td className="py-2 pr-3 tabular-nums text-zinc-400">
                            {h.disp_atr == null ? '—' : Number(h.disp_atr).toFixed(2)}
                          </td>
                          <td className="py-2 pr-3 tabular-nums text-cyan-300">
                            {h.cost_mult == null ? '—' : Number(h.cost_mult).toFixed(1)}
                          </td>
                          <td className="py-2 pr-3 tabular-nums text-zinc-300">
                            {h.lot_scale == null ? '—' : `${Number(h.lot_scale).toFixed(2)}×`}
                          </td>
                          <td className="py-2 text-zinc-500 truncate max-w-[240px]">
                            {h.block ? 'BLOCKED · ' : ''}
                            {h.message || '—'}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="mt-8 border border-white/10 bg-black/50 p-5">
            <div className="flex items-baseline justify-between mb-3">
              <h2 className="font-[Syne,sans-serif] text-lg text-white tracking-tight">
                Hyper RL
              </h2>
              <span className="text-[10px] tracking-[0.25em] uppercase text-zinc-500">
                expectancy · streak · block on cold symbols
              </span>
            </div>
            {(data?.rl?.symbols || []).length === 0 ? (
              <p className="text-zinc-600 text-sm">
                No closed hyper trades yet — RL wakes up after the first TP/SL exits.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-[12px]">
                  <thead className="text-[10px] uppercase tracking-wider text-zinc-500">
                    <tr>
                      <th className="py-2 pr-3 font-medium">Symbol</th>
                      <th className="py-2 pr-3 font-medium">N</th>
                      <th className="py-2 pr-3 font-medium">WR</th>
                      <th className="py-2 pr-3 font-medium">E[R]</th>
                      <th className="py-2 pr-3 font-medium">Streak</th>
                      <th className="py-2 pr-3 font-medium">Lot×</th>
                      <th className="py-2 font-medium">State</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(data?.rl?.symbols || []).map((row) => (
                      <tr key={row.symbol} className="border-t border-white/[0.04]">
                        <td className="py-2 pr-3 text-zinc-200 font-medium">{row.symbol}</td>
                        <td className="py-2 pr-3 tabular-nums text-zinc-400">{row.n}</td>
                        <td className="py-2 pr-3 tabular-nums text-zinc-300">{row.win_rate}%</td>
                        <td
                          className={`py-2 pr-3 tabular-nums ${
                            row.expectancy_r > 0
                              ? 'text-emerald-400'
                              : row.expectancy_r < 0
                                ? 'text-rose-400'
                                : 'text-zinc-400'
                          }`}
                        >
                          {row.expectancy_r > 0 ? '+' : ''}
                          {row.expectancy_r}
                        </td>
                        <td
                          className={`py-2 pr-3 tabular-nums ${
                            row.streak > 0
                              ? 'text-emerald-400'
                              : row.streak < 0
                                ? 'text-rose-400'
                                : 'text-zinc-400'
                          }`}
                        >
                          {row.streak > 0 ? `+${row.streak}` : row.streak}
                        </td>
                        <td className="py-2 pr-3 tabular-nums text-zinc-300">{row.lot_scale}×</td>
                        <td className="py-2">
                          {row.blocked ? (
                            <span className="text-rose-400 tracking-wide">COOLING</span>
                          ) : (
                            <span className="text-emerald-400/80 tracking-wide">LIVE</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {data?.last_error && (
            <p className="mt-6 text-sm text-rose-400 border border-rose-500/30 bg-rose-500/5 px-4 py-3">
              {data.last_error}
            </p>
          )}

          <p className="mt-8 text-[11px] text-zinc-600 max-w-2xl leading-relaxed">
            Hyper scalp runs beside the main Zenith desk. Starts in standby — arm only when
            you want machine-speed punches. Spreads and MetaAPI latency still apply; this is
            ultra-fast CFD scalp automation, not colocated HFT.
          </p>
        </section>
      </div>
    </Layout>
  )
}

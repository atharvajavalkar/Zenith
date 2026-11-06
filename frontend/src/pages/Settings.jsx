import { useState, useEffect, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import Layout from '../components/Layout'
import TradeModeSelect from '../components/TradeModeSelect'
import api from '../services/api'

// ── helpers ───────────────────────────────────────────────────────────────────
const fmt = (v, type) => {
  if (type === 'bool') return v ? 'On' : 'Off'
  if (type === 'int')  return String(Math.round(v))
  return typeof v === 'number' ? (v % 1 === 0 ? v.toFixed(0) : v.toFixed(2)) : String(v)
}

const GROUP_ORDER = [
  'MC / Markov',
  'ML Gate',
  'Adaptive Triggers',
  'Risk Management',
  'Execution',
  'Bias Gate',
]

const MODE_COLORS = {
  normal:   { text: 'text-green-400',  bg: 'bg-green-500/10',  border: 'border-green-500/30' },
  cautious: { text: 'text-yellow-400', bg: 'bg-yellow-500/10', border: 'border-yellow-500/30' },
  protect:  { text: 'text-red-400',    bg: 'bg-red-500/10',    border: 'border-red-500/30' },
}

function BrokerCard() {
  const [testResult, setTestResult] = useState(null)
  const [testing, setTesting] = useState(false)

  const { data: status, refetch } = useQuery({
    queryKey: ['trade-status'],
    queryFn: async () => {
      const { data } = await api.get('/api/trades/status')
      return data
    },
    refetchInterval: 30_000,
  })

  const runTest = async () => {
    setTesting(true)
    setTestResult(null)
    try {
      const { data } = await api.get('/api/trades/broker-test')
      setTestResult(data)
      refetch()
    } catch (e) {
      setTestResult({
        ok: false,
        message: e?.response?.data?.detail || e.message || 'Connection test failed',
      })
    } finally {
      setTesting(false)
    }
  }

  const mode = status?.trade_mode || 'paper'
  const configured = status?.broker_configured
  const broker = (status?.broker || 'xm').toUpperCase()
  const accounts = status?.broker_accounts || {}
  const modeBadge =
    mode === 'live'
      ? { label: 'LIVE', cls: 'border-red-400/40 text-red-300 bg-red-500/10' }
      : mode === 'demo'
        ? { label: 'DEMO', cls: 'border-accent-gold/40 text-accent-gold bg-accent-gold/10' }
        : { label: 'PAPER', cls: 'border-blue-500/30 text-blue-400 bg-blue-500/10' }

  return (
    <div className="zcard overflow-hidden">
      <div className="px-5 py-4 border-b border-surface-border flex flex-wrap items-center justify-between gap-3 bg-white/[0.02]">
        <div>
          <div className="font-semibold text-gray-100 text-sm">XM Broker (MetaAPI)</div>
          <div className="text-xs text-surface-muted mt-1">
            Paper / XM demo / XM live — use the Mode dropdown (header or below)
          </div>
        </div>
        <div className="flex items-center gap-2">
          <TradeModeSelect />
          <span className={`text-xs px-3 py-1 rounded-full border font-medium ${modeBadge.cls}`}>
            {modeBadge.label}
          </span>
          <span className={`text-xs px-3 py-1 rounded-full border font-medium ${
            status?.broker_auth === 'expired' || status?.broker_auth === 'invalid'
              ? 'border-red-500/40 text-red-300 bg-red-500/10'
              : configured
                ? 'border-green-500/30 text-green-400 bg-green-500/10'
                : 'border-yellow-500/30 text-yellow-400 bg-yellow-500/10'
          }`}>
            {status?.broker_auth === 'expired'
              ? 'Token expired'
              : status?.broker_auth === 'invalid'
                ? 'Token invalid'
                : configured
                  ? `${broker} configured`
                  : 'Not configured'}
          </span>
        </div>
      </div>

      <div className="p-5 grid gap-4 lg:grid-cols-2">
        <div className="space-y-3 text-sm">
          <p className="text-gray-300 font-medium">Setup (two MetaAPI accounts)</p>
          <ol className="list-decimal list-inside space-y-1.5 text-surface-muted text-xs">
            <li>Create account at <a href="https://app.metaapi.cloud" target="_blank" rel="noreferrer" className="text-accent-gold hover:underline">app.metaapi.cloud</a></li>
            <li>Add <strong className="text-gray-300">XM demo</strong> and <strong className="text-gray-300">XM live</strong> MT5 accounts</li>
            <li>Paste shared <code className="font-mono text-gray-400">METAAPI_TOKEN</code></li>
            <li>Set <code className="font-mono text-gray-400">METAAPI_ACCOUNT_ID_DEMO</code> and <code className="font-mono text-gray-400">METAAPI_ACCOUNT_ID_LIVE</code></li>
            <li>Optional default: <code className="font-mono text-gray-400">TRADE_MODE=demo</code></li>
            <li>Restart backend, then use the Trade mode dropdown (Paper / Demo / Live)</li>
          </ol>
          <p className="text-[11px] text-surface-muted">
            Accounts ready — demo: {accounts.demo ? 'yes' : 'no'} · live: {accounts.live ? 'yes' : 'no'}.
            Symbol must match MT5 Market Watch (<code className="font-mono text-gray-400">GOLD</code>).
          </p>
          {status?.broker_auth_ok === false && status?.broker_auth_message && (
            <p className="text-[11px] text-red-300">{status.broker_auth_message}</p>
          )}
        </div>

        <div className="space-y-3">
          <div className="rounded-xl border border-surface-border p-4 space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-surface-muted">Trade mode</span>
              <span className="text-white font-medium">{mode}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-surface-muted">XM gold market</span>
              <span className={status?.market_open ? 'text-green-400' : 'text-accent-gold'}>
                {status?.market_label || (status?.market_open ? 'Open' : 'Closed')}
              </span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-surface-muted shrink-0">Can trade</span>
              <span className={`text-right ${status?.can_trade ? 'text-green-400' : 'text-yellow-400'}`}>
                {status?.can_trade ? 'Yes' : (status?.reason || '—')}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-surface-muted">Live lot</span>
              <span className="text-accent-gold">{status?.lot_size ?? '—'} lot</span>
            </div>
            <div className="flex justify-between">
              <span className="text-surface-muted">Open positions</span>
              <span className="text-white">{status?.open_positions?.length ?? 0}</span>
            </div>
          </div>

          <button
            onClick={runTest}
            disabled={testing}
            className="btn-primary w-full py-2.5"
          >
            {testing ? 'Testing connection…' : 'Test XM connection'}
          </button>

          {testResult && (
            <div className={`rounded-xl border px-4 py-3 text-xs ${
              testResult.ok
                ? 'border-green-500/40 bg-green-500/10 text-green-400'
                : 'border-red-500/40 bg-red-500/10 text-red-400'
            }`}>
              <p className="font-semibold">{testResult.ok ? '✓ Connected' : '✗ Not connected'}</p>
              <p className="mt-1 opacity-90">{testResult.message}</p>
              {testResult.account && (
                <div className="mt-2 space-y-1 text-gray-300">
                  {testResult.account.balance != null && <p>Balance: ${Number(testResult.account.balance).toFixed(2)}</p>}
                  {testResult.account.equity != null && <p>Equity: ${Number(testResult.account.equity).toFixed(2)}</p>}
                  {testResult.account.server && <p>Server: {testResult.account.server}</p>}
                  {testResult.symbol && <p>Symbol: {testResult.symbol}</p>}
                  {testResult.open_positions != null && <p>Open positions: {testResult.open_positions}</p>}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── sub-components ────────────────────────────────────────────────────────────
function Toggle({ value, onChange }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!value)}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
        value ? 'bg-accent-gold' : 'bg-gray-600'
      }`}
    >
      <span className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
        value ? 'translate-x-6' : 'translate-x-1'
      }`} />
    </button>
  )
}

function Slider({ value, min, max, step, onChange, dirty }) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div className="flex items-center gap-3 flex-1">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => {
          const raw = parseFloat(e.target.value)
          onChange(step < 1 ? raw : Math.round(raw))
        }}
        className="flex-1 h-1.5 appearance-none rounded-full cursor-pointer"
        style={{
          background: `linear-gradient(to right, ${dirty ? '#F0A500' : '#4B5563'} ${pct}%, #374151 ${pct}%)`,
        }}
      />
    </div>
  )
}

function ParamRow({ keyName, meta, value, defaultValue, onChange }) {
  const dirty = value !== defaultValue
  const type  = meta.type

  return (
    <div className={`flex items-center gap-4 px-5 py-3.5 border-b border-surface-border/40 hover:bg-white/[0.015] transition group ${
      dirty ? 'bg-accent-gold/[0.03]' : ''
    }`}>
      {/* Label + description */}
      <div className="w-56 shrink-0">
        <div className={`text-sm font-medium ${dirty ? 'text-accent-gold' : 'text-gray-200'}`}>
          {meta.label}
          {dirty && <span className="ml-1.5 text-[9px] uppercase tracking-wider text-accent-gold/70">modified</span>}
        </div>
        {meta.description && (
          <div className="text-[11px] text-surface-muted mt-0.5 leading-snug">{meta.description}</div>
        )}
      </div>

      {/* Control */}
      <div className="flex-1 flex items-center gap-3">
        {type === 'bool' ? (
          <Toggle value={value} onChange={onChange} />
        ) : (
          <>
            <Slider
              value={value}
              min={meta.min}
              max={meta.max}
              step={meta.step}
              onChange={onChange}
              dirty={dirty}
            />
            <input
              type="number"
              min={meta.min}
              max={meta.max}
              step={meta.step}
              value={value}
              onChange={e => {
                const v = parseFloat(e.target.value)
                if (!isNaN(v)) onChange(type === 'int' ? Math.round(v) : v)
              }}
              className={`w-20 text-right text-sm tabular-nums bg-surface border rounded-lg px-2 py-1 focus:outline-none focus:ring-1 ${
                dirty
                  ? 'border-accent-gold/50 text-accent-gold focus:ring-accent-gold/30'
                  : 'border-surface-border text-gray-300 focus:ring-accent-gold/20'
              }`}
            />
          </>
        )}
      </div>

      {/* Default badge */}
      <div className="w-20 text-right">
        {dirty ? (
          <button
            type="button"
            onClick={() => onChange(defaultValue)}
            className="text-[10px] text-surface-muted hover:text-accent-gold transition"
            title="Reset this parameter"
          >
            default: {fmt(defaultValue, type)}
          </button>
        ) : (
          <span className="text-[11px] text-surface-muted/50">{fmt(value, type)}</span>
        )}
      </div>
    </div>
  )
}

function GroupCard({ title, params, settings, defaults, onChange }) {
  const dirtyCount = params.filter(([k]) => settings[k] !== defaults[k]).length

  return (
    <div className="zcard overflow-hidden">
      <div className="px-5 py-3 border-b border-surface-border flex items-center justify-between">
        <span className="font-medium text-gray-100 text-sm">{title}</span>
        {dirtyCount > 0 && (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-accent-gold/15 text-accent-gold border border-accent-gold/30">
            {dirtyCount} changed
          </span>
        )}
      </div>
      <div>
        {params.map(([key, meta]) => (
          <ParamRow
            key={key}
            keyName={key}
            meta={meta}
            value={settings[key]}
            defaultValue={defaults[key]}
            onChange={v => onChange(key, v)}
          />
        ))}
      </div>
    </div>
  )
}

// ── main page ─────────────────────────────────────────────────────────────────
export default function SettingsPage() {
  const qc = useQueryClient()

  const { data, isLoading, error } = useQuery({
    queryKey: ['algo-settings'],
    queryFn: async () => {
      const { data } = await api.get('/api/algo-settings')
      return data
    },
    refetchInterval: 30_000,
  })

  const [local, setLocal] = useState(null)  // local draft
  useEffect(() => {
    if (data && !local) setLocal(data.settings)
  }, [data])

  const settings  = local  || data?.settings || {}
  const defaults  = data?.defaults || {}
  const meta      = data?.meta     || {}
  const guard     = data?.guard    || {}

  const isDirty = local && data
    ? Object.keys(local).some(k => local[k] !== data.settings[k])
    : false

  const handleChange = useCallback((key, value) => {
    setLocal(prev => ({ ...prev, [key]: value }))
  }, [])

  const saveMut = useMutation({
    mutationFn: async (payload) => {
      const { data: res } = await api.patch('/api/algo-settings', payload)
      return res
    },
    onSuccess: (res) => {
      qc.setQueryData(['algo-settings'], old => ({
        ...old,
        settings: res.settings,
      }))
      setLocal(res.settings)
    },
  })

  const resetMut = useMutation({
    mutationFn: async () => {
      const { data: res } = await api.post('/api/algo-settings/reset')
      return res
    },
    onSuccess: (res) => {
      qc.setQueryData(['algo-settings'], old => ({
        ...old,
        settings: res.settings,
      }))
      setLocal(res.settings)
    },
  })

  const save = () => {
    if (!local) return
    // Only send changed keys
    const changed = {}
    Object.keys(local).forEach(k => {
      if (local[k] !== data?.settings?.[k]) changed[k] = local[k]
    })
    if (Object.keys(changed).length > 0) saveMut.mutate(changed)
  }

  const resetAll = () => resetMut.mutate()

  // Group params in defined order
  const grouped = GROUP_ORDER.map(g => ({
    title: g,
    params: Object.entries(meta).filter(([, m]) => m.group === g),
  })).filter(g => g.params.length > 0)

  // Summary of effective thresholds
  const mcNormal   = settings.mc_floor_base
  const mcCautious = (settings.mc_floor_base || 0) + (settings.mc_cautious_offset || 0)
  const mcProtect  = (settings.mc_floor_base || 0) + (settings.mc_protect_offset  || 0)

  const modeStyle = MODE_COLORS[guard.mode] || MODE_COLORS.normal

  return (
    <Layout>
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <p className="section-label mb-2">Control</p>
          <h1 className="page-title">Settings</h1>
          <p className="page-subtitle">Live parameter control — changes apply instantly</p>
        </div>

        {/* Save / Reset */}
        <div className="flex items-center gap-3">
          <button
            onClick={resetAll}
            disabled={resetMut.isPending}
            className="btn-secondary py-2 disabled:opacity-40"
          >
            {resetMut.isPending ? 'Resetting…' : 'Reset defaults'}
          </button>
          <button
            onClick={save}
            disabled={!isDirty || saveMut.isPending}
            className="btn-primary px-6 py-2 disabled:opacity-40"
          >
            {saveMut.isPending ? 'Saving…' : isDirty ? 'Save changes' : 'No changes'}
          </button>
        </div>
      </div>

      <BrokerCard />

      {isLoading && (
        <div className="zcard p-16 text-center text-surface-muted">Loading settings…</div>
      )}
      {error && (
        <div className="rounded-xl border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-400">
          Failed to load settings: {error.message}
        </div>
      )}

      {data && local && (
        <>
          {/* Status bar */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {/* Adaptive guard mode */}
            <div className={`zcard p-4 border ${modeStyle.border} ${modeStyle.bg}`}>
              <div className="text-[11px] text-surface-muted uppercase tracking-wide">Guard Mode</div>
              <div className={`text-lg font-bold mt-1 capitalize ${modeStyle.text}`}>
                {guard.mode || '—'}
              </div>
              <div className="text-[11px] text-surface-muted mt-1 truncate">{guard.message || 'Awaiting data'}</div>
            </div>

            {/* Effective MC floors */}
            <div className="zcard p-4">
              <div className="text-[11px] text-surface-muted uppercase tracking-wide">MC Floor (effective)</div>
              <div className="text-sm font-bold text-white mt-1 tabular-nums">
                {mcNormal}% · {mcCautious}% · {mcProtect}%
              </div>
              <div className="text-[10px] text-surface-muted mt-1">normal · cautious · protect</div>
            </div>

            {/* ML prob floors */}
            <div className="zcard p-4">
              <div className="text-[11px] text-surface-muted uppercase tracking-wide">ML Min Prob</div>
              <div className="text-sm font-bold text-white mt-1 tabular-nums">
                {(settings.ml_prob_normal * 100).toFixed(0)}% · {(settings.ml_prob_cautious * 100).toFixed(0)}% · {(settings.ml_prob_protect * 100).toFixed(0)}%
              </div>
              <div className="text-[10px] text-surface-muted mt-1">normal · cautious · protect</div>
            </div>

            {/* Lot size */}
            <div className="zcard p-4">
              <div className="text-[11px] text-surface-muted uppercase tracking-wide">Live Lot</div>
              <div className="text-lg font-bold text-accent-gold mt-1 tabular-nums">
                {settings.trade_lot_size} lot
              </div>
              <div className="text-[11px] text-surface-muted mt-1">
                ~${(3300 * settings.trade_lot_size * 100 / 1000).toFixed(0)} margin
              </div>
            </div>
          </div>

          {/* Rolling WR / streak info */}
          {(guard.rolling_win_rate != null || guard.loss_streak != null) && (
            <div className="flex flex-wrap gap-3 text-xs text-surface-muted">
              {guard.rolling_win_rate != null && (
                <span className={`px-3 py-1.5 rounded-full border ${
                  guard.rolling_win_rate >= 0.5
                    ? 'border-green-500/30 text-green-400'
                    : 'border-red-500/30 text-red-400'
                }`}>
                  Rolling WR: {(guard.rolling_win_rate * 100).toFixed(0)}%
                </span>
              )}
              {guard.loss_streak > 0 && (
                <span className="px-3 py-1.5 rounded-full border border-red-500/30 text-red-400">
                  Loss streak: {guard.loss_streak}
                </span>
              )}
              {guard.realized_pnl != null && (
                <span className={`px-3 py-1.5 rounded-full border ${
                  guard.realized_pnl >= 0
                    ? 'border-green-500/30 text-green-400'
                    : 'border-red-500/30 text-red-400'
                }`}>
                  Realized P&L: {guard.realized_pnl >= 0 ? '+' : ''}${guard.realized_pnl?.toFixed(2)}
                </span>
              )}
              {guard.blocked_states?.length > 0 && (
                <span className="px-3 py-1.5 rounded-full border border-yellow-500/30 text-yellow-400">
                  Blocked states: {guard.blocked_states.join(', ')}
                </span>
              )}
              {guard.blocked_hours?.length > 0 && (
                <span className="px-3 py-1.5 rounded-full border border-yellow-500/30 text-yellow-400">
                  Blocked UTC hours: {guard.blocked_hours.join(', ')}
                </span>
              )}
            </div>
          )}

          {data.symbol_rr && Object.keys(data.symbol_rr).length > 0 && (
            <div className="zcard overflow-hidden">
              <div className="px-5 py-3 border-b border-surface-border">
                <div className="font-semibold text-gray-100 text-sm">Per-pair RR (research map)</div>
                <div className="text-xs text-surface-muted mt-0.5">
                  Live TP pinned to SYMBOL_RR from OOS / WF / MC sweep
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-surface-border text-surface-muted uppercase tracking-wider">
                      <th className="px-4 py-2 text-left">Symbol</th>
                      <th className="px-4 py-2 text-left">Brain</th>
                      <th className="px-4 py-2 text-center">Target RR</th>
                      <th className="px-4 py-2 text-center">Entry TF</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(data.symbol_rr).map(([sym, rr]) => (
                      <tr key={sym} className="border-b border-surface-border/40">
                        <td className="px-4 py-2.5 font-medium text-accent-gold tracking-wide">{sym}</td>
                        <td className="px-4 py-2.5 text-surface-muted">
                          {data.symbol_brains?.[sym] || '—'}
                        </td>
                        <td className="px-4 py-2.5 text-center tabular-nums text-gray-200">
                          1:{Number(rr)}
                        </td>
                        <td className="px-4 py-2.5 text-center text-surface-muted tabular-nums">
                          {(data.symbol_scan_tfs?.[sym] || []).join('/') || 'MTF'}
                          {data.htf_entry_5m_exit?.includes(sym) ? ' →5m' : ''}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {isDirty && (
            <div className="rounded-xl border border-accent-gold/40 bg-accent-gold/5 px-4 py-3 text-sm text-accent-gold flex items-center justify-between">
              <span>You have unsaved changes. Click <strong>Save Changes</strong> to apply them to the live algorithm.</span>
              <button onClick={() => setLocal(data.settings)} className="text-xs text-surface-muted hover:text-gray-300 ml-4 shrink-0">
                Discard
              </button>
            </div>
          )}

          {saveMut.isSuccess && (
            <div className="rounded-xl border border-green-500/40 bg-green-500/10 px-4 py-2.5 text-sm text-green-400">
              ✓ Settings saved — live algorithm updated
            </div>
          )}

          {/* Parameter groups */}
          {grouped.map(g => (
            <GroupCard
              key={g.title}
              title={g.title}
              params={g.params}
              settings={settings}
              defaults={defaults}
              onChange={handleChange}
            />
          ))}

          {/* Legend */}
          <div className="text-[11px] text-surface-muted space-y-1 pb-2">
            <p>⚡ Changes apply immediately to the running algorithm — no Docker restart needed.</p>
            <p>↺ "Reset to Defaults" restores all values to what's in your <code className="font-mono text-gray-400">.env</code> file.</p>
            <p>Permanent changes: update <code className="font-mono text-gray-400">.env</code> (so they survive container restarts).</p>
          </div>
        </>
      )}
    </Layout>
  )
}

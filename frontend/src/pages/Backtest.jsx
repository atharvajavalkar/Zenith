/**
 * Backtest — full historical replay with price chart + paginated trade history.
 *
 * Data source: yfinance (GC=F Gold futures)
 *   · 5-minute bars  → up to 60 days back
 *   · 1-hour  bars   → up to 2 years back   (auto-selected for ranges > 60 days)
 *
 * The backend runs the EXACT same algorithm (SMC + MC + ML gates) bar-by-bar,
 * then returns all candles + trade events for display.
 */
import { useMemo, useState, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import Layout from '../components/Layout'
import BacktestPriceChart from '../components/BacktestPriceChart'
import BacktestReplayChart from '../components/BacktestReplayChart'
import api from '../services/api'

// ── helpers ───────────────────────────────────────────────────────────────────
const TODAY = new Date().toISOString().slice(0, 10)
const d = (days) => new Date(Date.now() - days * 86_400_000).toISOString().slice(0, 10)

const PRESETS = [
  { label: 'Last 7d',  start: d(7),  end: TODAY },
  { label: 'Last 14d', start: d(14), end: TODAY },
  { label: 'Last 30d', start: d(30), end: TODAY },
  { label: 'Last 45d', start: d(45), end: TODAY },
  { label: 'Last 60d', start: d(59), end: TODAY },  // max for 5m via yfinance
]

const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)
const pnlCls = (v) =>
  Number(v) > 0.5 ? 'text-green-400' : Number(v) < -0.5 ? 'text-red-400' : 'text-gray-300'
const signalCls = (s) => s === 'BUY' ? 'text-green-400 font-bold' : s === 'SELL' ? 'text-red-400 font-bold' : 'text-gray-500'
const resultCls = (r) => r === 'win' ? 'text-green-400' : r === 'loss' ? 'text-red-400' : 'text-gray-400'

const TRADES_PER_PAGE = 15

function StatCard({ label, value, sub, cls = 'text-gray-100', ring = false }) {
  return (
    <div className={`stat-tile text-center ${ring ? 'ring-1 ring-white/5' : ''}`}>
      <div className="text-[11px] text-surface-muted uppercase tracking-wide">{label}</div>
      <div className={`text-lg font-bold mt-1 tabular-nums leading-tight ${cls}`}>{value}</div>
      {sub && <div className="text-[11px] text-surface-muted mt-1">{sub}</div>}
    </div>
  )
}

function ProgressBar({ progress, message }) {
  return (
    <div className="zcard p-5 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-white flex items-center gap-2">
          <svg className="animate-spin h-4 w-4 text-accent-gold" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
          </svg>
          Running algorithm on historical data…
        </span>
        <span className="text-accent-gold font-bold tabular-nums">{progress}%</span>
      </div>
      <div className="h-2 bg-surface rounded-full overflow-hidden">
        <div className="h-full bg-accent-gold rounded-full transition-all duration-500" style={{ width: `${progress}%` }}/>
      </div>
      <p className="text-xs text-surface-muted">{message || 'Scanning candles…'}</p>
      <p className="text-[11px] text-surface-muted">
        The algorithm scans every candle bar-by-bar using the same SMC + MC + ML logic as live.
        7d takes ~30s · 30d takes ~2 min · 6m–1y takes ~5 min (1h bars).
      </p>
    </div>
  )
}

// ── Paginated trade table ──────────────────────────────────────────────────────
function TradeTable({ trades, startCapital }) {
  const [page, setPage] = useState(0)
  const totalPages = Math.ceil(trades.length / TRADES_PER_PAGE)
  const slice = trades.slice(page * TRADES_PER_PAGE, (page + 1) * TRADES_PER_PAGE)

  if (trades.length === 0) {
    return (
      <div className="zcard p-10 text-center text-surface-muted text-sm">
        No trades found in this date range — algorithm conditions were not met.
      </div>
    )
  }

  return (
    <div className="zcard overflow-hidden">
      <div className="px-5 py-3 border-b border-surface-border flex items-center justify-between">
        <span className="font-semibold text-gray-100">
          Trade History
          <span className="ml-2 text-xs text-surface-muted font-normal">{trades.length} total</span>
        </span>
        {totalPages > 1 && (
          <div className="flex items-center gap-2 text-xs">
            <button
              onClick={() => setPage(p => Math.max(0, p - 1))}
              disabled={page === 0}
              className="px-2 py-1 rounded border border-surface-border text-surface-muted hover:text-white hover:border-gray-500 transition disabled:opacity-30"
            >
              ← Prev
            </button>
            <span className="text-surface-muted tabular-nums">
              {page + 1} / {totalPages}
            </span>
            <button
              onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
              disabled={page === totalPages - 1}
              className="px-2 py-1 rounded border border-surface-border text-surface-muted hover:text-white hover:border-gray-500 transition disabled:opacity-30"
            >
              Next →
            </button>
          </div>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-surface-border text-surface-muted uppercase tracking-wider">
              <th className="px-4 py-2.5 text-left">#</th>
              <th className="px-4 py-2.5 text-left">Time</th>
              <th className="px-4 py-2.5 text-left">TF</th>
              <th className="px-4 py-2.5 text-left">Signal</th>
              <th className="px-4 py-2.5 text-right">Entry</th>
              <th className="px-4 py-2.5 text-right">SL</th>
              <th className="px-4 py-2.5 text-right">TP1</th>
              <th className="px-4 py-2.5 text-right">TP2</th>
              <th className="px-4 py-2.5 text-right">TP3</th>
              <th className="px-4 py-2.5 text-right">Final TP</th>
              <th className="px-4 py-2.5 text-left">SL Trail</th>
              <th className="px-4 py-2.5 text-left">Trail Path</th>
              <th className="px-4 py-2.5 text-right">Exit</th>
              <th className="px-4 py-2.5 text-right">Lot</th>
              <th className="px-4 py-2.5 text-right">P&L</th>
              <th className="px-4 py-2.5 text-right">Capital</th>
              <th className="px-4 py-2.5 text-center">RR</th>
              <th className="px-4 py-2.5 text-center">MC%</th>
              <th className="px-4 py-2.5 text-center">Result</th>
            </tr>
          </thead>
          <tbody>
            {slice.map((t, i) => {
              const globalN = page * TRADES_PER_PAGE + i + 1
              const isOpen = t.status === 'open'
              return (
                <tr key={i} className="border-b border-surface-border/40 hover:bg-white/[0.02] transition">
                  <td className="px-4 py-2.5 text-surface-muted">{globalN}</td>
                  <td className="px-4 py-2.5 text-gray-400 tabular-nums whitespace-nowrap">
                    {t.entry_time
                      ? new Date(t.entry_time * 1000).toLocaleString('en-GB', { day:'2-digit', month:'short', hour:'2-digit', minute:'2-digit', hour12: false })
                      : '—'}
                  </td>
                  <td className="px-4 py-2.5 text-[10px] uppercase tracking-wide text-accent-gold/90 whitespace-nowrap">
                    {t.exec_tf || t.EXEC_TF || '—'}
                  </td>
                  <td className={`px-4 py-2.5 ${signalCls(t.signal)}`}>{t.signal}</td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-gray-300">{t.entry}</td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-red-400/80">{t.sl}</td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-green-400/70">{t.tp1 ?? '—'}</td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-green-400/70">{t.tp2 ?? '—'}</td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-green-400/70">{t.tp3 ?? '—'}</td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-green-400/90">{t.tp ?? '—'}</td>
                  <td className="px-4 py-2.5 text-[10px] text-accent-gold/90 whitespace-nowrap">
                    {t.sl_trail_status ?? '—'}
                  </td>
                  <td className="px-4 py-2.5 text-[10px] text-gray-400 max-w-[260px]">
                    <span className="line-clamp-2">
                      {Array.isArray(t.trail_log) && t.trail_log.length > 0
                        ? t.trail_log.join('  •  ')
                        : (t.message ?? '—')}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-gray-400">
                    {isOpen ? <span className="text-accent-gold text-[10px]">OPEN</span> : t.exit_price ?? '—'}
                  </td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-gray-400">
                    {t.lot_size != null ? Number(t.lot_size).toFixed(2) : '—'}
                  </td>
                  <td className={`px-4 py-2.5 text-right tabular-nums font-semibold ${pnlCls(t.pnl)}`}>
                    {t.pnl != null ? `${t.pnl >= 0 ? '+' : ''}$${Number(t.pnl).toFixed(2)}` : '—'}
                  </td>
                  <td className={`px-4 py-2.5 text-right tabular-nums text-[11px] font-medium ${
                    t.capital_after != null && startCapital != null
                      ? t.capital_after >= startCapital ? 'text-green-400' : 'text-red-400'
                      : 'text-gray-400'
                  }`}>
                    {t.capital_after != null ? `$${t.capital_after.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—'}
                  </td>
                  <td className="px-4 py-2.5 text-center text-gray-400">{t.rr_type ?? '—'}</td>
                  <td className="px-4 py-2.5 text-center text-accent-gold tabular-nums">
                    {t.mc_prob != null ? `${t.mc_prob}%` : '—'}
                  </td>
                  <td className="px-4 py-2.5 text-center">
                    {isOpen
                      ? <span className="text-[10px] text-accent-gold bg-accent-gold/10 rounded-full px-2 py-0.5">open</span>
                      : <span className={`text-[10px] rounded-full px-2 py-0.5 ${
                          t.result === 'win'
                            ? 'text-green-400 bg-green-500/10'
                            : t.result === 'loss'
                              ? 'text-red-400 bg-red-500/10'
                              : 'text-gray-400 bg-gray-700/30'
                        }`}>{t.result ?? '—'}</span>
                    }
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* pagination footer */}
      {totalPages > 1 && (
        <div className="px-5 py-3 border-t border-surface-border flex items-center justify-between text-xs text-surface-muted">
          <span>Showing {page * TRADES_PER_PAGE + 1}–{Math.min((page + 1) * TRADES_PER_PAGE, trades.length)} of {trades.length} trades</span>
          <div className="flex gap-1">
            {Array.from({ length: totalPages }, (_, i) => (
              <button
                key={i}
                onClick={() => setPage(i)}
                className={`w-6 h-6 rounded text-[11px] transition ${
                  i === page
                    ? 'bg-accent-gold text-surface font-bold'
                    : 'border border-surface-border hover:border-gray-500 text-surface-muted hover:text-white'
                }`}
              >
                {i + 1}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Signal Log ────────────────────────────────────────────────────────────────
const DECISION_STYLE = {
  TRADE:      'text-green-400 bg-green-500/10',
  BLOCKED_MC: 'text-yellow-400 bg-yellow-500/10',
  NO_SIGNAL:  'text-gray-500 bg-gray-700/20',
}
const DECISION_LABEL = {
  TRADE:      '✅ TRADED',
  BLOCKED_MC: '🔒 BLOCKED',
  NO_SIGNAL:  '— NO SIGNAL',
}

function SignalLog({ entries }) {
  const [open,    setOpen]    = useState(false)
  const [page,    setPage]    = useState(0)
  const [filter,  setFilter]  = useState('all') // all | traded | blocked | buy | sell
  const LOG_PER_PAGE = 15

  const filtered = entries.filter(e => {
    if (filter === 'traded')  return e.decision === 'TRADE'
    if (filter === 'blocked') return e.decision === 'BLOCKED_MC'
    if (filter === 'buy')     return e.signal === 'BUY'
    if (filter === 'sell')    return e.signal === 'SELL'
    return true
  })
  const totalPages = Math.ceil(filtered.length / LOG_PER_PAGE)
  const slice = filtered.slice(page * LOG_PER_PAGE, (page + 1) * LOG_PER_PAGE)

  const fmtTime = (ts) => {
    if (!ts) return '—'
    const d = new Date(ts * 1000)
    return d.toLocaleString('en-GB', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false })
  }

  return (
    <div className="zcard overflow-hidden">
      {/* Header — click to expand/collapse */}
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full px-5 py-3 border-b border-surface-border flex items-center justify-between hover:bg-white/[0.02] transition"
      >
        <div className="flex items-center gap-3">
          <span className="font-semibold text-gray-100">Signal Log</span>
          <span className="text-xs text-surface-muted">{entries.length.toLocaleString()} bars evaluated</span>
          <span className="text-xs text-green-400">{entries.filter(e => e.decision === 'TRADE').length} traded</span>
          <span className="text-xs text-yellow-400">{entries.filter(e => e.decision === 'BLOCKED_MC').length} blocked</span>
        </div>
        <span className="text-surface-muted text-sm">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <>
          {/* Filter bar */}
          <div className="px-5 py-2.5 border-b border-surface-border flex flex-wrap items-center gap-2">
            <span className="text-xs text-surface-muted">Show:</span>
            {[
              { k: 'all',     l: 'All' },
              { k: 'traded',  l: '✅ Traded' },
              { k: 'blocked', l: '🔒 Blocked' },
              { k: 'buy',     l: 'BUY signals' },
              { k: 'sell',    l: 'SELL signals' },
            ].map(f => (
              <button
                key={f.k}
                onClick={() => { setFilter(f.k); setPage(0) }}
                className={`text-xs px-2.5 py-1 rounded-lg border transition ${
                  filter === f.k
                    ? 'border-accent-gold text-accent-gold bg-accent-gold/10'
                    : 'border-surface-border text-surface-muted hover:border-gray-500 hover:text-white'
                }`}
              >
                {f.l}
              </button>
            ))}
            <span className="ml-auto text-xs text-surface-muted">{filtered.length.toLocaleString()} entries</span>
          </div>

          {/* Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-surface-border text-surface-muted uppercase tracking-wider text-[10px]">
                  <th className="px-3 py-2 text-left">Time</th>
                  <th className="px-3 py-2 text-right">Open</th>
                  <th className="px-3 py-2 text-right">High</th>
                  <th className="px-3 py-2 text-right">Low</th>
                  <th className="px-3 py-2 text-right">Close</th>
                  <th className="px-3 py-2 text-center">Signal</th>
                  <th className="px-3 py-2 text-center">Decision</th>
                  <th className="px-3 py-2 text-center">MC%</th>
                  <th className="px-3 py-2 text-center">Markov%</th>
                  <th className="px-3 py-2 text-center">State</th>
                  <th className="px-3 py-2 text-right">ATR</th>
                  <th className="px-3 py-2 text-right">RSI</th>
                  <th className="px-3 py-2 text-right">EMA9</th>
                  <th className="px-3 py-2 text-right">EMA21</th>
                  <th className="px-3 py-2 text-right">B.Score</th>
                  <th className="px-3 py-2 text-right">S.Score</th>
                  <th className="px-3 py-2 text-right">SMC</th>
                  <th className="px-3 py-2 text-left min-w-[260px]">Why</th>
                </tr>
              </thead>
              <tbody>
                {slice.map((e, i) => {
                  const isBuy  = e.signal === 'BUY'
                  const isSell = e.signal === 'SELL'
                  const decStyle = DECISION_STYLE[e.decision] || 'text-gray-500'
                  const decLabel = DECISION_LABEL[e.decision] || e.decision
                  return (
                    <tr
                      key={i}
                      className={`border-b border-surface-border/40 hover:bg-white/[0.02] transition ${
                        e.decision === 'TRADE' ? 'bg-green-500/[0.03]' : ''
                      }`}
                    >
                      <td className="px-3 py-2 text-gray-400 tabular-nums whitespace-nowrap">{fmtTime(e.time)}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-gray-400">{e.open}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-green-400/70">{e.high}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-red-400/70">{e.low}</td>
                      <td className={`px-3 py-2 text-right tabular-nums font-medium ${
                        e.close >= e.open ? 'text-green-400' : 'text-red-400'
                      }`}>{e.close}</td>
                      <td className="px-3 py-2 text-center">
                        <span className={`font-bold ${isBuy ? 'text-green-400' : isSell ? 'text-red-400' : 'text-gray-500'}`}>
                          {e.signal}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span className={`text-[10px] rounded-full px-2 py-0.5 font-medium ${decStyle}`}>
                          {decLabel}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-center tabular-nums">
                        <span className={`font-medium ${
                          e.mc_prob >= 60 ? 'text-green-400' : e.mc_prob >= 35 ? 'text-accent-gold' : 'text-red-400'
                        }`}>{e.mc_prob ?? '—'}%</span>
                      </td>
                      <td className="px-3 py-2 text-center tabular-nums text-blue-400">
                        {e.markov_prob > 0 ? `${e.markov_prob}%` : '—'}
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span className="text-[9px] text-gray-400 font-mono">{e.state || '—'}</span>
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-gray-500">{e.atr ?? '—'}</td>
                      <td className={`px-3 py-2 text-right tabular-nums ${
                        (e.rsi ?? 50) > 60 ? 'text-red-400' : (e.rsi ?? 50) < 40 ? 'text-green-400' : 'text-gray-400'
                      }`}>{e.rsi ?? '—'}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-gray-500">{e.ema9 ?? '—'}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-gray-500">{e.ema21 ?? '—'}</td>
                      <td className={`px-3 py-2 text-right tabular-nums ${
                        isBuy ? 'text-green-400' : 'text-gray-500'
                      }`}>{e.buy_score ?? '—'}</td>
                      <td className={`px-3 py-2 text-right tabular-nums ${
                        isSell ? 'text-red-400' : 'text-gray-500'
                      }`}>{e.sell_score ?? '—'}</td>
                      <td className={`px-3 py-2 text-right tabular-nums ${
                        (e.smc_score ?? 0) >= 2 ? 'text-accent-gold' : 'text-gray-500'
                      }`}>{e.smc_score ?? '—'}</td>
                      <td className="px-3 py-2 text-gray-400 leading-snug max-w-[300px]">
                        <span className="line-clamp-2 text-[10px]">{e.why || '—'}</span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="px-5 py-3 border-t border-surface-border flex items-center justify-between text-xs text-surface-muted">
              <span>
                {page * LOG_PER_PAGE + 1}–{Math.min((page + 1) * LOG_PER_PAGE, filtered.length)} of {filtered.length.toLocaleString()}
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPage(p => Math.max(0, p - 1))}
                  disabled={page === 0}
                  className="px-2 py-1 rounded border border-surface-border hover:border-gray-500 hover:text-white transition disabled:opacity-30"
                >← Prev</button>
                <span className="tabular-nums">{page + 1} / {totalPages}</span>
                <button
                  onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                  disabled={page === totalPages - 1}
                  className="px-2 py-1 rounded border border-surface-border hover:border-gray-500 hover:text-white transition disabled:opacity-30"
                >Next →</button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

// Desk universe with Dukascopy multi-year hist under data/hist/
const INSTRUMENTS = [
  { value: 'GOLD', label: 'GOLD · Metal' },
  { value: 'USDCHF', label: 'USDCHF · FX' },
  { value: 'USDJPY', label: 'USDJPY · FX' },
  { value: 'NAS100', label: 'NAS100 · Index' },
  { value: 'UKOIL', label: 'UKOIL · Brent' },
  { value: 'AUDUSD', label: 'AUDUSD · FX' },
  { value: 'BTCUSD', label: 'BTCUSD · Crypto' },
  { value: 'ETHUSD', label: 'ETHUSD · Crypto' },
]

const RR_OPTIONS = [
  { value: 'auto', label: 'Per-pair map (SYMBOL_RR)' },
  { value: '1:1.5', label: 'Force 1:1.5' },
  { value: '1:2',  label: 'Force 1:2' },
  { value: '1:2.4', label: 'Force 1:2.4' },
  { value: '1:3',  label: 'Force 1:3' },
  { value: '1:4',  label: 'Force 1:4' },
]

export default function Backtest() {
  const [startDate,    setStartDate]    = useState('')
  const [endDate,      setEndDate]      = useState('')
  const [activePreset, setActivePreset] = useState(null)
  const [viewMode,     setViewMode]     = useState('chart')  // 'chart' | 'replay'
  const [capital,      setCapital]      = useState('10000')   // match live $10k desk
  const [rrMode,       setRrMode]       = useState('auto')
  const [symbol,       setSymbol]       = useState('GOLD')

  const [jobId,   setJobId]   = useState(null)
  const [status,  setStatus]  = useState(null)   // polling state
  const [result,  setResult]  = useState(null)   // final result
  const [error,   setError]   = useState(null)
  const pollRef = useRef(null)

  const parsedCapital = parseFloat(capital) || 10000

  // Live desk lot sizer (SYMBOL_ALLOCATION + TARGET_PROFIT + MAX_RISK)
  const { data: lotDesk } = useQuery({
    queryKey: ['backtest-lot-desk', parsedCapital],
    queryFn: async () =>
      (await api.get('/api/trades/lot-desk', { params: { capital: parsedCapital } })).data,
    staleTime: 15_000,
  })
  const deskRow = useMemo(() => {
    const rows = lotDesk?.instruments || []
    return rows.find((r) => r.symbol === symbol) || null
  }, [lotDesk, symbol])
  const lotSize = Number(deskRow?.lot_size) || 0.01
  const estMargin = deskRow?.expected_loss_usd != null
    ? Number(deskRow.expected_loss_usd)
    : null

  // Human-readable date labels to prevent DD/MM vs MM/DD confusion
  const fmtDate = (iso) => {
    if (!iso) return null
    const [y, m, d] = iso.split('-')
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    return `${d} ${months[parseInt(m,10)-1]} ${y}`
  }
  const readableStart = fmtDate(startDate || d(30))
  const readableEnd   = fmtDate(endDate   || TODAY)

  // Warn if start date is older than 60 days from today
  const startTooOld = (() => {
    const s = startDate || d(30)
    const diffDays = (new Date() - new Date(s)) / 86400000
    return diffDays > 60
  })()

  const applyPreset = (p) => {
    setStartDate(p.start)
    setEndDate(p.end)
    setActivePreset(p.label)
  }

  const run = async () => {
    setError(null)
    setResult(null)
    setStatus({ status: 'running', progress: 0, message: 'Starting…' })

    // Clear previous poll
    if (pollRef.current) clearInterval(pollRef.current)

    try {
      const { data: job } = await api.post('/api/backtest/candles/start', {
        start_date: startDate || d(30),
        end_date:   endDate   || TODAY,
        lot_size:   lotSize, // fallback only — backend sizes each trade via live desk
        mc_floor:   42.0,
        rr_mode:    rrMode,
        capital:    parsedCapital,
        symbol,
        dynamic_lots: true,
      })
      setJobId(job.job_id)

      pollRef.current = setInterval(async () => {
        try {
          const { data: s } = await api.get(`/api/backtest/candles/status/${job.job_id}`)
          setStatus(s)
          if (s.status === 'done' || s.status === 'error') {
            clearInterval(pollRef.current)
            pollRef.current = null
            if (s.status === 'done' && s.result) {
              setResult(s.result)
            } else {
              setError(s.error || 'Backtest failed')
            }
          }
        } catch { clearInterval(pollRef.current); pollRef.current = null }
      }, 2000)
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Failed to start')
      setStatus(null)
    }
  }

  const isRunning = status?.status === 'running'

  // Extract trade_open events for table and chart
  const tradeEvents = (result?.events || []).filter(e =>
    e.type === 'trade_open' || e.type === 'trade_close'
  )
  // trade_close events already contain ALL fields from the open event ({**open_trade, ...})
  // so we can use them directly — no need to merge with trade_open events
  const mergedTrades = (() => {
    let runCap = parsedCapital
    return (result?.events || [])
      .filter(e => e.type === 'trade_close')
      .map(t => {
        const pnl = t.pnl != null ? Number(t.pnl) : null
        if (pnl != null) runCap += pnl
        // Keep backend desk-sized lot — do not overwrite with preview
        return { ...t, capital_after: +runCap.toFixed(2) }
      })
  })()
  const finalCapital = mergedTrades.length > 0
    ? mergedTrades[mergedTrades.length - 1].capital_after
    : null

  const s = result?.stats

  return (
    <Layout>
      {/* ── header ── */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="section-label mb-2">Laboratory</p>
          <h1 className="page-title">Backtest</h1>
          <p className="page-subtitle">
            Desk pairs via Dukascopy hist · live brains · same lot desk ($10k / $200R)
          </p>
        </div>
        {result && (
          <div className="status-chip text-accent-gold border border-accent-gold/20 px-3 py-2">
            <span className="w-1.5 h-1.5 rounded-full bg-accent-gold" />
            {startDate || d(30)} → {endDate || TODAY}
            <span className="text-surface-muted">·</span>
            RR {s?.rr_mode || rrMode}
            <span className="text-surface-muted">·</span>
            {mergedTrades.length} trades
            <span className="text-surface-muted">·</span>
            <span className={s?.realized_pnl >= 0 ? 'text-accent-buy' : 'text-accent-sell'}>
              {s?.realized_pnl >= 0 ? '+' : ''}${s?.realized_pnl?.toFixed(2) ?? '0.00'}
            </span>
          </div>
        )}
      </div>

      {/* ── controls ── */}
      <div className="zcard p-4 flex flex-wrap items-center gap-3">
        {/* Presets */}
        <div className="flex flex-wrap gap-1.5">
          {PRESETS.map(p => (
            <button
              key={p.label}
              onClick={() => applyPreset(p)}
              className={`text-xs px-3 py-1.5 rounded-lg border transition font-medium ${
                activePreset === p.label
                  ? 'border-accent-gold text-accent-gold bg-accent-gold/10'
                  : 'border-surface-border text-surface-muted hover:border-accent-gold/40 hover:text-white'
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>

        <div className="w-px h-6 bg-surface-border hidden sm:block" />

        {/* Instrument — metal / FX / crypto */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-surface-muted whitespace-nowrap">Market</span>
          <select
            value={symbol}
            onChange={e => setSymbol(e.target.value)}
            className="bg-surface border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-accent-gold/50 [color-scheme:dark]"
          >
            {INSTRUMENTS.map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>

        <div className="w-px h-6 bg-surface-border hidden sm:block" />

        {/* Capital input + live desk lot preview */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-surface-muted whitespace-nowrap">Capital $</span>
          <input
            type="number"
            min="100"
            step="100"
            value={capital}
            onChange={e => setCapital(e.target.value)}
            className="w-28 bg-surface border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-accent-gold/50"
            placeholder="10000"
          />
          <div className="flex flex-col items-start leading-tight">
            <span className="text-accent-gold font-bold text-sm tabular-nums">
              {Number(lotSize).toFixed(2)} lot
            </span>
            <span className="text-[10px] text-surface-muted">
              live desk
              {deskRow?.allocation_mult != null ? ` · ×${Number(deskRow.allocation_mult).toFixed(2)}` : ''}
              {estMargin != null ? ` · SL~$${Math.round(estMargin)}` : ''}
            </span>
          </div>
        </div>

        <div className="w-px h-6 bg-surface-border hidden sm:block" />

        {/* RR mode — same algo as live, force profile or auto-pick */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-surface-muted whitespace-nowrap">RR</span>
          <select
            value={rrMode}
            onChange={e => setRrMode(e.target.value)}
            className="bg-surface border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-accent-gold/50 [color-scheme:dark]"
          >
            {RR_OPTIONS.map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>

        <div className="w-px h-6 bg-surface-border hidden sm:block" />

        {/* Date inputs */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-surface-muted">From</span>
          <div className="flex flex-col">
            <input
              type="date"
              value={startDate}
              onChange={e => { setStartDate(e.target.value); setActivePreset(null) }}
              className="bg-surface border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-accent-gold/50 [color-scheme:dark]"
            />
            <span className={`text-[10px] mt-0.5 text-center font-mono ${startTooOld ? 'text-red-400' : 'text-green-400/80'}`}>
              {readableStart}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-surface-muted">To</span>
          <div className="flex flex-col">
            <input
              type="date"
              value={endDate}
              onChange={e => { setEndDate(e.target.value); setActivePreset(null) }}
              className="bg-surface border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-accent-gold/50 [color-scheme:dark]"
            />
            <span className="text-[10px] mt-0.5 text-center font-mono text-green-400/80">
              {readableEnd}
            </span>
          </div>
        </div>

        {/* Run */}
        <button
          onClick={run}
          disabled={isRunning}
          className="btn-primary ml-auto py-2 flex items-center gap-2"
        >
          {isRunning && (
            <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
            </svg>
          )}
          {isRunning ? 'Running…' : 'Run backtest'}
        </button>

        {error && <p className="text-sm text-red-400 w-full mt-1">{error}</p>}
      </div>

      {/* ── date-range warning ── */}
      {startTooOld && !isRunning && !result && (
        <div className="rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm">
          <span className="font-semibold text-red-400">⚠ Start date is older than 60 days</span>
          <span className="text-red-300/80 ml-2">
            ({readableStart}) — 5m data is only available for the last 60 days.
            Use the <strong>Last 7d / 30d / 60d</strong> presets, or pick a start date after{' '}
            <strong>{fmtDate(d(60))}</strong>.
          </span>
        </div>
      )}

      {/* ── info banner (no result yet) ── */}
      {!result && !isRunning && (
        <div className="zcard p-10 text-center space-y-3">
          <div className="text-4xl">📈</div>
          <p className="text-white font-semibold">Enter your capital, pick a date range, and run</p>
          <div className="flex flex-wrap justify-center gap-4 text-xs text-surface-muted mt-2 max-w-2xl mx-auto">
            <span>💰 <strong className="text-gray-300">Capital</strong> → dynamic lots (~$400 TP target, same desk tables)</span>
            <span>📈 <strong className="text-gray-300">Market</strong> — GOLD / FX / BTC historical 5m (TV + Yahoo)</span>
            <span>📊 Always uses <strong className="text-gray-300">5-minute bars</strong> — same 1m/3m/5m/15m/1h/4h as live</span>
            <span>📅 Max range: <strong className="text-gray-300">last 60 days</strong> (yfinance 5m limit)</span>
          </div>
          <div className="flex flex-wrap justify-center gap-3 mt-3 text-[11px] text-surface-muted">
            <span>$500 → 0.05 lot</span>
            <span>$1,000 → 0.10 lot</span>
            <span>$5,000 → 0.50 lot</span>
            <span>$10,000 → 1.00 lot</span>
          </div>
        </div>
      )}

      {/* ── progress ── */}
      {isRunning && <ProgressBar progress={status?.progress || 0} message={status?.message} />}

      {/* ── results ── */}
      {result && !isRunning && (
        <>
          {/* Stats row */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
            <div className="stat-tile text-center ring-1 ring-white/5 col-span-2 sm:col-span-1">
              <div className="text-[11px] text-surface-muted uppercase tracking-wide">XAUUSD</div>
              <div className="text-lg font-bold text-accent-gold mt-1 tabular-nums leading-tight">
                {s?.first_price != null ? `$${Number(s.first_price).toLocaleString()}` : '—'}
              </div>
              {s?.last_price != null && (
                <div className={`text-[11px] mt-1 tabular-nums ${s.last_price >= s.first_price ? 'text-green-400' : 'text-red-400'}`}>
                  {s.last_price >= s.first_price ? '▲' : '▼'} ${Number(s.last_price).toLocaleString()}
                </div>
              )}
            </div>

            <div className="stat-tile text-center">
              <div className="text-[11px] text-surface-muted uppercase tracking-wide">Bars Scanned</div>
              <div className="text-lg font-bold text-gray-100 mt-1 tabular-nums">{(result.bars_scanned || 0).toLocaleString()}</div>
              <div className="text-[11px] text-surface-muted mt-1">{result.data_range?.from} → {result.data_range?.to}</div>
            </div>

            <StatCard label="Trades" value={s?.executed ?? 0} sub={`${s?.closed ?? 0} closed`} />

            {(s?.entry_brain || result.entry_brain) && (
              <div className="stat-tile text-center">
                <div className="text-[11px] text-surface-muted uppercase tracking-wide">Live brain</div>
                <div className="text-sm font-semibold text-accent-gold mt-1 truncate">
                  {s?.entry_brain || result.entry_brain}
                </div>
                <div className="text-[11px] text-surface-muted mt-1 truncate">
                  {(s?.scan_tfs || result.scan_tfs || []).join(' · ') || '—'}
                  {s?.symbol_rr != null ? ` · ${s.symbol_rr}R` : ''}
                </div>
              </div>
            )}

            <div className="stat-tile text-center">
              <div className="text-[11px] text-surface-muted uppercase tracking-wide">First trade</div>
              <div className="text-lg font-bold text-gray-100 mt-1 tabular-nums">
                {s?.first_trade_bar != null ? `Bar ${s.first_trade_bar}` : '—'}
              </div>
              <div className="text-[11px] text-surface-muted mt-1">
                {s?.first_trade_pct != null
                  ? `${s.first_trade_pct}% into range`
                  : s?.first_signal_bar != null
                    ? `Signal @ ${s.first_signal_bar} (MC blocked)`
                    : 'No setup in range'}
              </div>
            </div>

            <div className="stat-tile text-center">
              <div className="text-[11px] text-surface-muted uppercase tracking-wide">Realised P&L</div>
              <div className={`text-lg font-bold mt-1 tabular-nums ${pnlCls(s?.realized_pnl)}`}>
                {s?.realized_pnl != null ? `${s.realized_pnl >= 0 ? '+' : ''}$${Number(s.realized_pnl).toFixed(2)}` : '—'}
              </div>
            </div>

            <div className="stat-tile text-center">
              <div className="text-[11px] text-surface-muted uppercase tracking-wide">Win Rate</div>
              <div className={`text-lg font-bold mt-1 tabular-nums ${(s?.win_rate ?? 0) >= 0.5 ? 'text-green-400' : 'text-red-400'}`}>
                {pct(s?.win_rate)}
              </div>
              <div className="text-[11px] text-surface-muted mt-1">{(s?.wins ?? 0) + (s?.losses ?? 0)} decisive</div>
            </div>

            <StatCard label="Wins"      value={s?.wins ?? 0}    cls="text-green-400" />
            <StatCard label="Losses"    value={s?.losses ?? 0}  cls="text-red-400" />
            <StatCard
              label="Avg lot"
              value={Number(s?.lot_size ?? lotSize).toFixed(2)}
              sub={
                s?.lot_size_min != null && s?.lot_size_max != null
                  ? `${Number(s.lot_size_min).toFixed(2)}–${Number(s.lot_size_max).toFixed(2)} · desk`
                  : `desk · $${parsedCapital.toLocaleString()}`
              }
            />

            {/* Final Capital card */}
            {finalCapital != null && (
              <div className="stat-tile text-center ring-1 ring-white/5 col-span-2 sm:col-span-2">
                <div className="text-[11px] text-surface-muted uppercase tracking-wide">Final Capital</div>
                <div className={`text-lg font-bold mt-1 tabular-nums ${finalCapital >= parsedCapital ? 'text-green-400' : 'text-red-400'}`}>
                  ${finalCapital.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
                <div className={`text-[11px] mt-1 tabular-nums ${finalCapital >= parsedCapital ? 'text-green-400' : 'text-red-400'}`}>
                  {finalCapital >= parsedCapital ? '▲' : '▼'} {finalCapital >= parsedCapital ? '+' : ''}
                  {(((finalCapital - parsedCapital) / parsedCapital) * 100).toFixed(1)}% return
                </div>
              </div>
            )}
          </div>

          {Array.isArray(s?.concept_ranking) && s.concept_ranking.length > 0 && (
            <div className="rounded-xl border border-surface-border bg-surface-raised/40 p-4">
              <div className="text-[11px] text-surface-muted uppercase tracking-wide mb-1">
                Concept expectancy (this timeline)
              </div>
              <p className="text-[11px] text-surface-muted mb-3">
                {s.concept_message || 'Walk-forward ML ranking of which ingredients paid'}
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
                {s.concept_ranking.slice(0, 10).map((c) => (
                  <div key={c.concept} className="rounded-lg border border-surface-border/60 px-2.5 py-2">
                    <div className="text-[10px] text-surface-muted truncate">{c.concept}</div>
                    <div className={`text-sm font-semibold tabular-nums ${Number(c.expectancy) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {Number(c.expectancy) >= 0 ? '+' : ''}{Number(c.expectancy).toFixed(2)}R
                    </div>
                    <div className="text-[10px] text-surface-muted">
                      n={c.n} · WR {Math.round((c.win_rate || 0) * 100)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* View toggle */}
          <div className="flex items-center gap-2">
            <div className="flex rounded-lg overflow-hidden border border-surface-border text-xs font-semibold">
              <button
                onClick={() => setViewMode('chart')}
                className={`px-4 py-2 transition ${viewMode === 'chart' ? 'bg-accent-gold text-surface' : 'text-surface-muted hover:text-white'}`}
              >
                📈 Price Chart
              </button>
              <button
                onClick={() => setViewMode('replay')}
                className={`px-4 py-2 transition ${viewMode === 'replay' ? 'bg-accent-gold text-surface' : 'text-surface-muted hover:text-white'}`}
              >
                📽 Candle Replay
              </button>
            </div>
            <span className="text-xs text-surface-muted">
              {viewMode === 'chart'
                ? 'Static chart — all candles with trade markers'
                : 'Replay from bar 0; jump to first trade when ready'}
            </span>
          </div>

          {s?.first_trade_bar != null && (
            <p className="text-[11px] text-surface-muted">
              Engine takes the first valid setup in your date range (warmup is loaded before the range, not skipped inside it).
              First fill at bar {s.first_trade_bar}
              {s.first_trade_time ? ` · ${s.first_trade_time}` : ''}
              {s.first_trade_pct != null ? ` · ${s.first_trade_pct}% into range` : ''}.
              {s.bars_not_ready > 0 ? ` (${s.bars_not_ready} early bars waited for indicators.)` : ''}
            </p>
          )}

          {/* Chart or Replay */}
          {viewMode === 'chart' ? (
            <BacktestPriceChart
              candles={result.candles || []}
              events={result.events || []}
              height={460}
            />
          ) : (
            <BacktestReplayChart
              candles={result.candles || []}
              events={result.events || []}
              startAtBar={s?.first_trade_bar ?? 0}
            />
          )}

          {/* Trade history (paginated) */}
          <TradeTable trades={mergedTrades} startCapital={parsedCapital} />

          {/* Signal log */}
          <SignalLog entries={result.signal_log || []} />
        </>
      )}
    </Layout>
  )
}

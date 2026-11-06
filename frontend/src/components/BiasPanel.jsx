import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../services/api'
import { useTimezone } from '../contexts/TimezoneContext'
import { formatInTimezone } from '../utils/formatTime'

function BiasDot({ direction, size = 'sm' }) {
  const colors = {
    BULLISH: 'bg-accent-buy',
    BEARISH: 'bg-accent-sell',
    NEUTRAL: 'bg-gray-500',
  }
  const dim = size === 'sm' ? 'w-1.5 h-1.5' : 'w-2 h-2'
  return (
    <span
      className={`inline-block ${dim} rounded-full shrink-0 ${colors[direction] || colors.NEUTRAL}`}
    />
  )
}

function biasColor(direction) {
  if (direction === 'BULLISH') return 'text-accent-buy'
  if (direction === 'BEARISH') return 'text-accent-sell'
  return 'text-gray-400'
}

function shortDir(direction) {
  if (!direction) return 'Flat'
  return direction.charAt(0) + direction.slice(1).toLowerCase()
}

function classTone(cls) {
  if (cls === 'fx') return 'text-sky-300'
  if (cls === 'metals') return 'text-accent-gold'
  if (cls === 'crypto') return 'text-violet-300'
  if (cls === 'energy') return 'text-orange-300'
  if (cls === 'indices') return 'text-emerald-300'
  return 'text-surface-muted'
}

function TfChip({ label, direction, confidence }) {
  return (
    <div className="flex items-center justify-between gap-1 text-[10px] tabular-nums">
      <span className="text-surface-muted w-6">{label}</span>
      <span className={`flex items-center gap-1 ${biasColor(direction)}`}>
        <BiasDot direction={direction} />
        {Math.round(confidence || 0)}%
      </span>
    </div>
  )
}

function InstrumentBiasCard({ row, expanded, onToggle }) {
  const overall = row.overall || {}
  const macro = row.macro || {}
  const news = row.news || {}
  const profile = row.profile || {}
  const tfs = row.timeframes || {}

  return (
    <button
      type="button"
      onClick={onToggle}
      className={`text-left rounded-lg border px-3 py-2.5 transition w-full ${
        expanded
          ? 'border-accent-gold/40 bg-accent-gold/[0.04]'
          : 'border-surface-border bg-surface/30 hover:border-white/15'
      }`}
    >
      <div className="flex items-start justify-between gap-2 mb-1.5">
        <div>
          <div className="text-sm font-semibold text-gray-50 tracking-wide">{row.symbol}</div>
          <div className={`text-[9px] uppercase tracking-wide mt-0.5 ${classTone(row.asset_class)}`}>
            {row.asset_class || '—'}
          </div>
        </div>
        <div className={`text-right ${biasColor(overall.direction)}`}>
          <div className="text-xs font-semibold flex items-center justify-end gap-1.5">
            <BiasDot direction={overall.direction} />
            {overall.label || shortDir(overall.direction)}
          </div>
          <div className="text-[11px] tabular-nums mt-0.5">{overall.confidence ?? '—'}%</div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 mt-2">
        {['5m', '15m', '1h', '4h'].map((tf) => (
          <TfChip
            key={tf}
            label={tf}
            direction={tfs[tf]?.direction}
            confidence={tfs[tf]?.confidence}
          />
        ))}
      </div>

      <div className="mt-2 pt-2 border-t border-white/[0.06] grid grid-cols-3 gap-1 text-[9px] uppercase tracking-wide">
        <div className={biasColor(macro.direction)}>
          <div className="text-surface-muted mb-0.5">Macro</div>
          {shortDir(macro.direction)}
        </div>
        <div className={biasColor(news.direction)}>
          <div className="text-surface-muted mb-0.5">News</div>
          {shortDir(news.direction)}
        </div>
        <div className={biasColor(profile.direction)}>
          <div className="text-surface-muted mb-0.5">Profile</div>
          {profile.available ? shortDir(profile.direction) : '—'}
        </div>
      </div>

      {expanded && (
        <div className="mt-2.5 space-y-2 text-[11px] text-surface-muted border-t border-white/[0.06] pt-2.5">
          {macro.notes?.length > 0 && (
            <div>
              <div className="section-label mb-1">Macro</div>
              <ul className="space-y-0.5">
                {macro.notes.slice(0, 3).map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </div>
          )}
          {news.summary && (
            <div>
              <div className="section-label mb-1">News</div>
              <p className="text-gray-300 leading-snug">{news.summary}</p>
            </div>
          )}
          {profile.available && (
            <div>
              <div className="section-label mb-1">Volume profile</div>
              <p className="text-gray-300 leading-snug">{profile.summary}</p>
              <div className="mt-1 flex flex-wrap gap-2 tabular-nums text-[10px]">
                {profile.poc != null && <span>POC {profile.poc}</span>}
                {profile.vah != null && <span>VAH {profile.vah}</span>}
                {profile.val != null && <span>VAL {profile.val}</span>}
              </div>
            </div>
          )}
          {(overall.physics_weight != null) && (
            <p className="tabular-nums text-[10px]">
              Blend · physics {Math.round((overall.physics_weight ?? 1) * 100)}%
              {overall.news_weight > 0 ? ` · news ${Math.round(overall.news_weight * 100)}%` : ''}
              {overall.profile_weight > 0 ? ` · profile ${Math.round(overall.profile_weight * 100)}%` : ''}
            </p>
          )}
          {row.error && (
            <p className="text-accent-sell text-[10px]">Partial: {row.error}</p>
          )}
        </div>
      )}
    </button>
  )
}

export default function BiasPanel() {
  const { displayTimezone } = useTimezone()
  const [openSym, setOpenSym] = useState(null)

  const { data, isLoading, isError } = useQuery({
    queryKey: ['market-bias'],
    queryFn: async () => (await api.get('/api/signals/bias')).data,
    refetchInterval: 60_000,
    staleTime: 30_000,
  })

  const { data: tradeStatus } = useQuery({
    queryKey: ['trade-status'],
    queryFn: async () => (await api.get('/api/trades/status')).data,
    refetchInterval: 30_000,
  })

  const biasGateOn = tradeStatus?.architecture?.bias_gate_enabled
  const feedLabel =
    data?.data_source === 'xm' || tradeStatus?.data_source === 'xm'
      ? 'XM MetaAPI'
      : 'TradingView'

  const updatedLabel = data?.updated_at
    ? formatInTimezone(data.updated_at, displayTimezone)
    : null

  const instruments = useMemo(() => {
    if (data?.instruments?.length) return data.instruments
    if (data?.symbol) {
      return [
        {
          symbol: data.symbol,
          asset_class: 'metals',
          timeframes: data.timeframes,
          macro: data.macro,
          news: data.news,
          profile: data.profile,
          overall: data.overall,
        },
      ]
    }
    return []
  }, [data])

  return (
    <div className="zcard p-4 sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
        <div>
          <h3 className="panel-title">Desk bias</h3>
          <p className="panel-desc max-w-2xl">
            Algo stack per instrument — multi-TF physics, symbol-aware macro, news lean, and volume
            profile{feedLabel ? ` via ${feedLabel}` : ''}. Tap a card for detail.
            {tradeStatus?.architecture && !biasGateOn && (
              <span className="text-orange-300/90"> Gate off — context only.</span>
            )}
          </p>
        </div>
        {updatedLabel && (
          <div className="text-xs text-surface-muted shrink-0">Updated {updatedLabel}</div>
        )}
      </div>

      {isLoading && !data && (
        <div className="text-sm text-surface-muted py-8 text-center">Loading desk bias…</div>
      )}

      {isError && !data && (
        <div className="text-sm text-accent-sell py-4">Could not load market bias.</div>
      )}

      {instruments.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-5 gap-2.5">
          {instruments.map((row) => (
            <InstrumentBiasCard
              key={row.symbol}
              row={row}
              expanded={openSym === row.symbol}
              onToggle={() => setOpenSym((s) => (s === row.symbol ? null : row.symbol))}
            />
          ))}
        </div>
      )}
    </div>
  )
}

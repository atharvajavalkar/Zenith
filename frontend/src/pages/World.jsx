import { useCallback, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Layout from '../components/Layout'
import BiasPanel from '../components/BiasPanel'
import OrthographicMap from '../components/OrthographicMap'
import api from '../services/api'

function impactTone(impact) {
  const i = String(impact || '').toLowerCase()
  if (i === 'high') return 'text-accent-sell'
  if (i === 'medium') return 'text-accent-gold'
  if (i === 'news') return 'text-sky-300/90'
  return 'text-surface-muted'
}

function biasTone(bias) {
  if (bias === 'BULLISH') return {
    text: 'text-accent-buy',
    border: 'border-accent-buy/40',
    bg: 'bg-accent-buy/10',
    dot: 'bg-accent-buy',
  }
  if (bias === 'BEARISH') return {
    text: 'text-accent-sell',
    border: 'border-accent-sell/40',
    bg: 'bg-accent-sell/10',
    dot: 'bg-accent-sell',
  }
  return {
    text: 'text-gray-300',
    border: 'border-surface-border',
    bg: 'bg-white/[0.03]',
    dot: 'bg-gray-500',
  }
}

function assetClass(sym) {
  const s = String(sym || '').toUpperCase()
  if (s === 'GOLD' || s === 'SILVER' || s === 'XAUUSD' || s === 'XAGUSD') return 'metals'
  if (s === 'BTCUSD' || s === 'ETHUSD' || s.startsWith('BTC') || s.startsWith('ETH')) return 'crypto'
  if (s === 'USOIL' || s === 'UKOIL' || s === 'XTIUSD') return 'energy'
  if (s === 'US30' || s === 'US500' || s === 'NAS100' || s === 'USTEC' || s === 'SPX500') return 'indices'
  return 'fx'
}

function classLabel(cls) {
  return { metals: 'Metals', fx: 'FX', crypto: 'Crypto', energy: 'Energy', indices: 'Indices' }[cls] || cls
}

function BiasBadge({ bias, label, confidence, compact = false }) {
  const tone = biasTone(bias)
  return (
    <div
      className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1 ${tone.bg} ${tone.border}`}
    >
      <span className={`inline-block w-1.5 h-1.5 rounded-full ${tone.dot}`} />
      <span className={`text-xs font-semibold uppercase tracking-wide ${tone.text}`}>
        {label || bias || 'Neutral'}
      </span>
      {!compact && confidence != null && (
        <span className="text-[10px] text-surface-muted tabular-nums">{confidence}%</span>
      )}
    </div>
  )
}

function HoverCard({ marker, anchor }) {
  if (!marker) return null
  const headlines = marker.headlines?.length
    ? marker.headlines
    : [{ headline: 'No fresh headlines right now', detail: marker.role, impact: '—', source: '—' }]
  const tone = biasTone(marker.bias)
  const assets = marker.assets?.length ? marker.assets.join(' · ') : null

  return (
    <div
      className="pointer-events-none absolute z-20 w-[min(340px,calc(100vw-2rem))] rounded-xl border border-accent-gold/40 bg-[#0a0d14]/96 shadow-[0_28px_90px_rgba(0,0,0,0.65)] backdrop-blur-md px-4 py-3.5"
      style={{
        left: Math.min(
          anchor.x + 18,
          typeof window !== 'undefined' ? window.innerWidth - 360 : anchor.x,
        ),
        top: Math.max(72, anchor.y - 10),
      }}
    >
      <div className="flex items-start justify-between gap-3 mb-2">
        <div>
          <div className="text-[11px] uppercase tracking-[0.18em] text-accent-gold/80">Desk pulse</div>
          <div className="text-base font-semibold text-gray-50 tracking-tight mt-0.5">{marker.name}</div>
          <p className="text-xs text-surface-muted mt-1 leading-relaxed">{marker.role}</p>
          {assets && (
            <p className="text-[10px] uppercase tracking-wide text-sky-300/80 mt-1.5">{assets}</p>
          )}
        </div>
        <div className="text-right shrink-0">
          <div className="text-lg font-semibold tabular-nums text-accent-gold">{marker.count ?? 0}</div>
          <div className="text-[10px] text-surface-muted uppercase tracking-wide">items</div>
        </div>
      </div>

      <div className={`mb-3 rounded-lg border px-3 py-2.5 ${tone.bg} ${tone.border}`}>
        <div className="flex items-center justify-between gap-2 mb-1">
          <span className="text-[10px] uppercase tracking-wide text-surface-muted">Hub lean</span>
          <BiasBadge
            bias={marker.bias || 'NEUTRAL'}
            label={marker.bias_label}
            confidence={marker.bias_confidence}
          />
        </div>
        <p className="text-[12px] text-gray-200 leading-snug">
          {marker.bias_summary || 'Waiting on enough headlines to judge this hub.'}
        </p>
      </div>

      <ul className="space-y-2 max-h-44 overflow-hidden">
        {headlines.slice(0, 3).map((h) => (
          <li
            key={h.id || h.headline}
            className="rounded-lg border border-white/[0.06] bg-white/[0.03] px-3 py-2"
          >
            <div className="flex items-center justify-between gap-2 mb-0.5">
              <span className={`text-[10px] uppercase tracking-wide ${impactTone(h.impact)}`}>
                {h.impact || 'Update'}
              </span>
              <span className="text-[10px] text-surface-muted truncate max-w-[140px]">{h.source}</span>
            </div>
            <p className="text-[13px] text-gray-100 leading-snug">{h.headline}</p>
          </li>
        ))}
      </ul>
    </div>
  )
}

const DEFAULT_MARKERS = [
  { id: 'US', name: 'United States', lat: 38.9, lng: -77.0, role: 'Fed & USD', count: 0, headlines: [], bias: 'NEUTRAL', bias_label: 'Neutral', assets: ['GOLD', 'USOIL'] },
  { id: 'CN', name: 'China', lat: 39.9, lng: 116.4, role: 'Demand hub', count: 0, headlines: [], bias: 'NEUTRAL', bias_label: 'Neutral', assets: ['GOLD', 'USOIL'] },
  { id: 'IN', name: 'India', lat: 28.6, lng: 77.2, role: 'Jewellery demand', count: 0, headlines: [], bias: 'NEUTRAL', bias_label: 'Neutral', assets: ['GOLD'] },
  { id: 'GB', name: 'United Kingdom', lat: 51.5, lng: -0.13, role: 'LBMA bullion', count: 0, headlines: [], bias: 'NEUTRAL', bias_label: 'Neutral', assets: ['GOLD'] },
  { id: 'AU', name: 'Australia', lat: -35.3, lng: 149.1, role: 'Mining', count: 0, headlines: [], bias: 'NEUTRAL', bias_label: 'Neutral', assets: ['GOLD'] },
  { id: 'CH', name: 'Switzerland', lat: 46.95, lng: 7.45, role: 'Refining', count: 0, headlines: [], bias: 'NEUTRAL', bias_label: 'Neutral', assets: ['GOLD'] },
  { id: 'AE', name: 'United Arab Emirates', lat: 25.2, lng: 55.27, role: 'Dubai bullion', count: 0, headlines: [], bias: 'NEUTRAL', bias_label: 'Neutral', assets: ['GOLD'] },
  { id: 'SA', name: 'Saudi Arabia', lat: 24.7, lng: 46.7, role: 'OPEC oil', count: 0, headlines: [], bias: 'NEUTRAL', bias_label: 'Neutral', assets: ['USOIL', 'GOLD'] },
]

const FILTERS = [
  { id: 'all', label: 'All' },
  { id: 'metals', label: 'Gold' },
  { id: 'energy', label: 'Oil' },
]

export default function WorldPage() {
  const [hover, setHover] = useState({ marker: null, anchor: { x: 0, y: 0 } })
  const [filter, setFilter] = useState('all')

  const { data, isLoading, isError, isFetching } = useQuery({
    queryKey: ['world-desk-news'],
    queryFn: async () => (await api.get('/api/news/world')).data,
    refetchInterval: 5 * 60_000,
    staleTime: 60_000,
  })

  const markers = data?.markers?.length ? data.markers : DEFAULT_MARKERS
  const feed = data?.feed || []
  const markets = data?.markets || []

  const filteredMarkets = useMemo(() => {
    if (filter === 'all') return markets
    return markets.filter((m) => assetClass(m.symbol) === filter)
  }, [markets, filter])

  const filteredFeed = useMemo(() => {
    if (filter === 'all') return feed
    return feed.filter((item) => {
      const assets = item.assets || []
      if (!assets.length) return filter === 'metals'
      return assets.some((a) => assetClass(a) === filter)
    })
  }, [feed, filter])

  const onHover = useCallback((marker, anchor) => {
    setHover({ marker, anchor })
  }, [])

  const tip = useMemo(
    () => (hover.marker ? null : 'Drag to spin · hover a hub for headlines + lean'),
    [hover.marker],
  )

  return (
    <Layout variant="world">
      <section className="relative -mx-4 sm:-mx-6 -mt-2 mb-2 overflow-hidden rounded-none sm:rounded-2xl border-y sm:border border-accent-gold/20 bg-[#05070c]">
        <div className="absolute inset-x-0 top-0 z-10 px-5 sm:px-8 pt-6 pb-2 flex flex-wrap items-end justify-between gap-4 pointer-events-none">
          <div>
            <p className="text-[11px] uppercase tracking-label text-accent-gold/80">World desk</p>
            <h1 className="mt-2 page-title text-[2rem] sm:text-[2.2rem]">Markets across the map</h1>
            <p className="mt-2 page-subtitle max-w-xl">
              Everyday book — metals, FX, oil & crypto colour from the macro hubs; calendar prints and
              headlines scored per instrument.
            </p>
          </div>
          <div className="text-right text-xs text-surface-muted tabular-nums">
            <div>
              {isLoading ? 'Loading headlines…' : `${data?.total_headlines ?? 0} headlines`}
              {isFetching && !isLoading ? ' · refreshing' : ''}
            </div>
            <div className="text-accent-gold/70 mt-0.5">
              {(data?.sources || []).join(' · ') || 'Forex Factory · Yahoo'}
            </div>
            {tip && <div className="mt-2 text-surface-muted/80">{tip}</div>}
          </div>
        </div>

        <OrthographicMap markers={markers} onHover={onHover} />
        <HoverCard marker={hover.marker} anchor={hover.anchor} />

        {isError && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 text-xs text-red-300 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">
            News feed delayed — map still works. Retry shortly.
          </div>
        )}
      </section>

      <BiasPanel />

      <section className="zcard p-5">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h3 className="panel-title">Instrument board</h3>
            <p className="panel-desc">
              Live lean for the traded OOS book (FX · metals · oil · crypto).
            </p>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {FILTERS.map((f) => (
              <button
                key={f.id}
                type="button"
                onClick={() => setFilter(f.id)}
                className={`text-[10px] uppercase tracking-wide px-2.5 py-1 rounded-md border transition ${
                  filter === f.id
                    ? 'border-accent-gold/50 text-accent-gold bg-accent-gold/10'
                    : 'border-surface-border text-surface-muted hover:text-gray-200'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>
        {isLoading && !markets.length ? (
          <div className="text-sm text-surface-muted animate-pulse py-8 text-center">Scoring markets…</div>
        ) : (
          <ul className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
            {filteredMarkets.map((m) => {
              const tone = biasTone(m.bias)
              const cls = assetClass(m.symbol)
              return (
                <li
                  key={m.symbol}
                  className={`rounded-xl border px-4 py-3.5 min-h-[120px] flex flex-col ${tone.border} ${tone.bg}`}
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div>
                      <div className="text-sm font-semibold text-gray-50 tracking-wide">{m.symbol}</div>
                      <div className="text-[10px] uppercase tracking-wide text-surface-muted mt-0.5">
                        {classLabel(cls)} · {m.count ?? 0} items
                      </div>
                    </div>
                    <BiasBadge bias={m.bias || 'NEUTRAL'} label={m.label} confidence={m.confidence} />
                  </div>
                  <p className="text-[12px] text-gray-200 leading-snug flex-1">
                    {m.summary || 'Collecting updates for this market.'}
                  </p>
                </li>
              )
            })}
          </ul>
        )}
      </section>

      <section className="zcard p-5">
        <div className="mb-4">
          <h3 className="panel-title">Country hub lean</h3>
          <p className="panel-desc">
            One box per hub — macro lean from calendar + headlines, tagged to the markets they move.
          </p>
        </div>
        {isLoading && !data?.markers?.length ? (
          <div className="text-sm text-surface-muted animate-pulse py-8 text-center">Scoring countries…</div>
        ) : (
          <ul className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {markers.map((m) => {
              const tone = biasTone(m.bias)
              const topLine = m.headlines?.[0]?.headline
              return (
                <li
                  key={m.id}
                  className={`rounded-xl border px-4 py-3.5 min-h-[132px] flex flex-col ${tone.border} ${tone.bg}`}
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div>
                      <div className="text-sm font-semibold text-gray-50">{m.name}</div>
                      <div className="text-[11px] text-surface-muted mt-0.5">{m.role}</div>
                      {m.assets?.length ? (
                        <div className="text-[10px] text-sky-300/80 mt-1 uppercase tracking-wide">
                          {m.assets.join(' · ')}
                        </div>
                      ) : null}
                    </div>
                    <BiasBadge
                      bias={m.bias || 'NEUTRAL'}
                      label={m.bias_label}
                      confidence={m.bias_confidence}
                    />
                  </div>
                  <p className="text-[12px] text-gray-200 leading-snug flex-1">
                    {m.bias_summary || 'Collecting updates for this hub.'}
                  </p>
                  {topLine && (
                    <p className="mt-2 text-[11px] text-surface-muted truncate border-t border-white/[0.06] pt-2">
                      Latest: {topLine}
                    </p>
                  )}
                </li>
              )
            })}
          </ul>
        )}
      </section>

      <section className="zcard p-5">
        <div className="mb-4">
          <h3 className="panel-title">Latest market headlines</h3>
          <p className="panel-desc">
            Calendar + Yahoo — filter above to focus gold or oil.
          </p>
        </div>
        {isLoading && !feed.length ? (
          <div className="text-sm text-surface-muted animate-pulse py-8 text-center">Gathering world news…</div>
        ) : (
          <ul className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
            {filteredFeed.slice(0, 20).map((item) => {
              const hub = markers.find((m) => m.id === item.hub)
              return (
                <li
                  key={item.id}
                  className="rounded-lg border border-surface-border bg-surface/40 px-3.5 py-3 min-h-[88px]"
                >
                  <div className="flex items-center justify-between gap-2 mb-1.5">
                    <span className="text-[11px] text-accent-gold/90 font-medium">
                      {item.country || hub?.name || item.hub}
                    </span>
                    <div className="flex items-center gap-2">
                      <span className={`text-[10px] uppercase tracking-wide ${impactTone(item.impact)}`}>
                        {item.impact}
                      </span>
                      <BiasBadge
                        bias={item.bias || hub?.bias || 'NEUTRAL'}
                        label={item.bias_label || hub?.bias_label}
                        confidence={item.bias_confidence ?? hub?.bias_confidence}
                        compact
                      />
                    </div>
                  </div>
                  <p className="text-sm text-gray-100 leading-snug">{item.headline}</p>
                  <div className="mt-1.5 text-[11px] text-surface-muted truncate">
                    {(item.assets || []).join(' · ') || item.source}
                    {item.detail ? ` · ${item.detail}` : ''}
                  </div>
                </li>
              )
            })}
          </ul>
        )}
      </section>
    </Layout>
  )
}

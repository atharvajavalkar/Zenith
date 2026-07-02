import { useQuery } from '@tanstack/react-query'
import api from '../services/api'
import { useTimezone } from '../contexts/TimezoneContext'
import { formatInTimezone } from '../utils/formatTime'

function biasIcon(direction) {
  if (direction === 'BULLISH') return '🟢'
  if (direction === 'BEARISH') return '🔴'
  return '⚪'
}

function biasColor(direction) {
  if (direction === 'BULLISH') return 'text-accent-buy'
  if (direction === 'BEARISH') return 'text-accent-sell'
  return 'text-gray-400'
}

function BiasLine({ label, direction, confidence }) {
  return (
    <div className="flex items-center justify-between gap-4 py-1.5">
      <span className="text-gray-500 w-10 shrink-0 tabular-nums">{label}</span>
      <span className={`font-semibold flex-1 ${biasColor(direction)}`}>
        {biasIcon(direction)} {direction.charAt(0) + direction.slice(1).toLowerCase()} ({confidence}%)
      </span>
    </div>
  )
}

export default function BiasPanel() {
  const { displayTimezone } = useTimezone()

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

  const updatedLabel = data?.updated_at
    ? formatInTimezone(data.updated_at, displayTimezone)
    : null

  return (
    <div className="zcard p-5">
      <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
        <div>
          <h3 className="font-semibold text-gray-100 text-lg flex items-center gap-2">
            <span className="inline-block h-4 w-1 rounded-full bg-gold-gradient" />
            XAU/USD Bias
          </h3>
          <p className="text-xs text-gray-500 mt-1 max-w-3xl leading-relaxed">
            Macro + technical market analyst. Every hour it scans intermarket data, sentiment,
            and multi-timeframe structure to produce a directional bias only — no trade execution
            or entry/exit levels.
            {tradeStatus?.architecture && !biasGateOn && (
              <span className="text-orange-300/90"> Bias gate is OFF — context only during 1M testing.</span>
            )}
          </p>
        </div>
        {updatedLabel && (
          <div className="text-xs text-gray-500 shrink-0">
            Updated {updatedLabel}
          </div>
        )}
      </div>

      {isLoading && !data && (
        <div className="text-sm text-gray-500 animate-pulse py-6 text-center">
          Scanning macro + technical data…
        </div>
      )}

      {isError && !data && (
        <div className="text-sm text-accent-sell py-4">Could not load market bias.</div>
      )}

      {data && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-surface rounded-lg border border-surface-border p-4">
            <div className="text-xs text-gray-500 uppercase tracking-wide mb-3">Timeframe Bias</div>
            <BiasLine label="5m" direction={data.timeframes['5m'].direction} confidence={data.timeframes['5m'].confidence} />
            <BiasLine label="15m" direction={data.timeframes['15m'].direction} confidence={data.timeframes['15m'].confidence} />
            <BiasLine label="1H" direction={data.timeframes['1h'].direction} confidence={data.timeframes['1h'].confidence} />
            <BiasLine label="4H" direction={data.timeframes['4h'].direction} confidence={data.timeframes['4h'].confidence} />

            <div className="mt-4 pt-3 border-t border-surface-border">
              <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Overall</div>
              <div className={`text-xl font-bold ${biasColor(data.overall.direction)}`}>
                {biasIcon(data.overall.direction)} {data.overall.label} ({data.overall.confidence}%)
              </div>
            </div>
          </div>

          <div className="bg-surface rounded-lg border border-surface-border p-4">
            <div className="text-xs text-gray-500 uppercase tracking-wide mb-3">Macro Context</div>
            {data.macro?.available ? (
              <>
                <BiasLine label="Macro" direction={data.macro.direction} confidence={data.macro.confidence} />
                {data.macro.notes?.length > 0 && (
                  <ul className="mt-3 space-y-1.5 text-xs text-gray-400">
                    {data.macro.notes.map((note) => (
                      <li key={note} className="flex gap-2">
                        <span className="text-accent-gold shrink-0">•</span>
                        <span>{note}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </>
            ) : (
              <p className="text-xs text-gray-500">Macro feeds unavailable — using technical structure only.</p>
            )}

            <div className="mt-4 pt-3 border-t border-surface-border">
              <div className="text-xs text-gray-500 uppercase tracking-wide mb-2">Data Sources</div>
              <ul className="text-xs text-gray-500 space-y-1">
                {(data.sources || []).map((src) => (
                  <li key={src}>• {src}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

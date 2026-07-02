import { useEffect, useState } from 'react'
import { useTimezone } from '../contexts/TimezoneContext'
import { formatNowInTimezone } from '../utils/formatTime'

export default function TimezoneSelector() {
  const {
    displayTimezone,
    chartTimezone,
    setDisplayTimezone,
    setChartTimezone,
    timezones,
  } = useTimezone()
  const [now, setNow] = useState(() => formatNowInTimezone(displayTimezone))

  useEffect(() => {
    const tick = () => setNow(formatNowInTimezone(displayTimezone))
    tick()
    const id = window.setInterval(tick, 1000)
    return () => window.clearInterval(id)
  }, [displayTimezone])

  return (
    <div className="zcard p-4">
      <div className="flex flex-wrap items-start justify-between gap-4 mb-4">
        <div>
          <h3 className="font-semibold text-gray-200">Timezone</h3>
          <p className="text-xs text-gray-500 mt-1">
            Trade log timestamps and chart axis use your selected GMT zones.
          </p>
        </div>
        <div className="text-right">
          <div className="text-xs text-gray-500 uppercase tracking-wide">Current time</div>
          <div className="text-lg font-semibold text-accent-gold tabular-nums mt-0.5">{now}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="bg-surface rounded-lg p-3 border border-surface-border">
          <label htmlFor="display-timezone" className="text-xs text-gray-500 block mb-1.5">
            Trade log timezone
          </label>
          <select
            id="display-timezone"
            value={displayTimezone}
            onChange={(e) => setDisplayTimezone(e.target.value)}
            className="w-full bg-surface-card border border-surface-border rounded px-2 py-2 text-sm text-gray-100 focus:outline-none focus:border-accent-gold"
          >
            {timezones.map((tz) => (
              <option key={tz.id} value={tz.id}>
                {tz.label}
              </option>
            ))}
          </select>
        </div>
        <div className="bg-surface rounded-lg p-3 border border-surface-border">
          <label htmlFor="chart-timezone" className="text-xs text-gray-500 block mb-1.5">
            Chart timezone
          </label>
          <select
            id="chart-timezone"
            value={chartTimezone}
            onChange={(e) => setChartTimezone(e.target.value)}
            className="w-full bg-surface-card border border-surface-border rounded px-2 py-2 text-sm text-gray-100 focus:outline-none focus:border-accent-gold"
          >
            {timezones.map((tz) => (
              <option key={tz.id} value={tz.id}>
                {tz.label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  )
}

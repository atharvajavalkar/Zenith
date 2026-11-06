import { useEffect, useState } from 'react'
import { useTimezone } from '../contexts/TimezoneContext'
import { formatNowInTimezone } from '../utils/formatTime'

export default function TimezoneSelector({ compact = false }) {
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

  if (compact) {
    return (
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <span className="text-surface-muted tabular-nums shrink-0">{now}</span>
        <select
          value={displayTimezone}
          onChange={(e) => setDisplayTimezone(e.target.value)}
          className="bg-transparent border border-surface-border rounded-md px-2 py-1 text-gray-300 focus:outline-none focus:border-accent-gold/50"
          aria-label="Trade log timezone"
        >
          {timezones.map((tz) => (
            <option key={tz.id} value={tz.id}>
              Log: {tz.label}
            </option>
          ))}
        </select>
        <select
          value={chartTimezone}
          onChange={(e) => setChartTimezone(e.target.value)}
          className="bg-transparent border border-surface-border rounded-md px-2 py-1 text-gray-300 focus:outline-none focus:border-accent-gold/50"
          aria-label="Chart timezone"
        >
          {timezones.map((tz) => (
            <option key={tz.id} value={tz.id}>
              Chart: {tz.label}
            </option>
          ))}
        </select>
      </div>
    )
  }

  return (
    <div className="zcard p-5">
      <div className="flex flex-wrap items-start justify-between gap-4 mb-4">
        <div>
          <h3 className="panel-title">Timezone</h3>
          <p className="panel-desc">Trade log and chart axis zones.</p>
        </div>
        <div className="text-right">
          <div className="section-label">Now</div>
          <div className="text-base font-medium text-gray-100 tabular-nums mt-0.5">{now}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div>
          <label htmlFor="display-timezone" className="section-label block mb-1.5">
            Trade log
          </label>
          <select
            id="display-timezone"
            value={displayTimezone}
            onChange={(e) => setDisplayTimezone(e.target.value)}
            className="w-full bg-surface border border-surface-border rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:border-accent-gold/50"
          >
            {timezones.map((tz) => (
              <option key={tz.id} value={tz.id}>
                {tz.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="chart-timezone" className="section-label block mb-1.5">
            Chart
          </label>
          <select
            id="chart-timezone"
            value={chartTimezone}
            onChange={(e) => setChartTimezone(e.target.value)}
            className="w-full bg-surface border border-surface-border rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:border-accent-gold/50"
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

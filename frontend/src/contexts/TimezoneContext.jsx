import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { GMT_TIMEZONES } from '../utils/timezones'

const STORAGE_KEY = 'xauusd-display-timezone'
const CHART_STORAGE_KEY = 'xauusd-chart-timezone'

const TimezoneContext = createContext(null)

export function TimezoneProvider({ children }) {
  const [displayTimezone, setDisplayTimezone] = useState(() => {
    return localStorage.getItem(STORAGE_KEY) || 'Etc/UTC'
  })
  const [chartTimezone, setChartTimezone] = useState(() => {
    return localStorage.getItem(CHART_STORAGE_KEY) || 'Etc/UTC'
  })

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, displayTimezone)
  }, [displayTimezone])

  useEffect(() => {
    localStorage.setItem(CHART_STORAGE_KEY, chartTimezone)
  }, [chartTimezone])

  const value = useMemo(
    () => ({
      displayTimezone,
      chartTimezone,
      setDisplayTimezone,
      setChartTimezone,
      timezones: GMT_TIMEZONES,
    }),
    [displayTimezone, chartTimezone],
  )

  return <TimezoneContext.Provider value={value}>{children}</TimezoneContext.Provider>
}

export function useTimezone() {
  const ctx = useContext(TimezoneContext)
  if (!ctx) {
    throw new Error('useTimezone must be used within TimezoneProvider')
  }
  return ctx
}

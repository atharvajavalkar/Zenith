/** Parse backend candle/log timestamps (assumed UTC) and format in a chosen IANA zone. */
export function parseUtcTimestamp(value) {
  if (!value) return null
  const raw = String(value).trim()
  const normalized = raw.includes('T') ? raw : raw.replace(' ', 'T')
  const withZone = /[zZ]|[+-]\d{2}:\d{2}$/.test(normalized) ? normalized : `${normalized}Z`
  const date = new Date(withZone)
  return Number.isNaN(date.getTime()) ? null : date
}

export function formatInTimezone(value, timeZone, options = {}) {
  const date = value instanceof Date ? value : parseUtcTimestamp(value)
  if (!date) return value ?? '—'

  return new Intl.DateTimeFormat('en-GB', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
    ...options,
  }).format(date)
}

export function formatNowInTimezone(timeZone) {
  return formatInTimezone(new Date(), timeZone)
}

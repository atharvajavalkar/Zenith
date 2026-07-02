/** Common GMT-labelled zones for display and chart sync. */
export const GMT_TIMEZONES = [
  { id: 'Etc/UTC', label: 'GMT+0 (UTC)' },
  { id: 'Europe/London', label: 'GMT+0 / +1 (London)' },
  { id: 'Europe/Paris', label: 'GMT+1 (Central Europe)' },
  { id: 'Europe/Athens', label: 'GMT+2 (Eastern Europe)' },
  { id: 'Europe/Moscow', label: 'GMT+3 (Moscow)' },
  { id: 'Asia/Dubai', label: 'GMT+4 (Dubai)' },
  { id: 'Asia/Karachi', label: 'GMT+5 (Pakistan)' },
  { id: 'Asia/Kolkata', label: 'GMT+5:30 (India)' },
  { id: 'Asia/Bangkok', label: 'GMT+7 (Bangkok)' },
  { id: 'Asia/Singapore', label: 'GMT+8 (Singapore)' },
  { id: 'Asia/Tokyo', label: 'GMT+9 (Tokyo)' },
  { id: 'Australia/Sydney', label: 'GMT+10 / +11 (Sydney)' },
  { id: 'America/New_York', label: 'GMT-5 / -4 (New York)' },
  { id: 'America/Chicago', label: 'GMT-6 / -5 (Chicago)' },
  { id: 'America/Denver', label: 'GMT-7 / -6 (Denver)' },
  { id: 'America/Los_Angeles', label: 'GMT-8 / -7 (Los Angeles)' },
]

export function findTimezone(id) {
  return GMT_TIMEZONES.find((tz) => tz.id === id) ?? GMT_TIMEZONES[0]
}

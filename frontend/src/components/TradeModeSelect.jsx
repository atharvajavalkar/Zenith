import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../services/api'

const OPTIONS = [
  { value: 'paper', label: 'Paper' },
  { value: 'demo', label: 'XM Demo' },
  { value: 'live', label: 'XM Live' },
]

/**
 * Always-visible Paper / Demo / Live switcher (testing phase).
 * Patches /api/trades/settings — no Docker recreate needed.
 */
export default function TradeModeSelect({ compact = false, className = '' }) {
  const queryClient = useQueryClient()

  const { data: status } = useQuery({
    queryKey: ['trade-status'],
    queryFn: async () => (await api.get('/api/trades/status')).data,
    refetchInterval: 5000,
  })

  const { data: settings } = useQuery({
    queryKey: ['trade-settings'],
    queryFn: async () => (await api.get('/api/trades/settings')).data,
    refetchInterval: 10_000,
  })

  const update = useMutation({
    mutationFn: async (trade_mode) =>
      (await api.patch('/api/trades/settings', { trade_mode })).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trade-status'] })
      queryClient.invalidateQueries({ queryKey: ['trade-settings'] })
      queryClient.invalidateQueries({ queryKey: ['trade-summary'] })
    },
  })

  const mode = settings?.trade_mode || status?.trade_mode || 'paper'
  const accounts = status?.broker_accounts || {}

  const onChange = (next) => {
    if (next === mode) return
    if (next === 'demo' && accounts.demo === false) {
      if (
        !window.confirm(
          'METAAPI_ACCOUNT_ID_DEMO is not set in .env. Switch to Demo anyway?'
        )
      ) {
        return
      }
    }
    if (next === 'live' && accounts.live === false) {
      if (
        !window.confirm(
          'METAAPI_ACCOUNT_ID_LIVE is not set. Live orders will fail until you add it. Switch anyway?'
        )
      ) {
        return
      }
    }
    if (next === 'live') {
      if (
        !window.confirm(
          'Switch to XM LIVE? Real money orders go to your live MetaAPI account.'
        )
      ) {
        return
      }
    }
    if (next === 'demo') {
      if (
        !window.confirm(
          'Switch to XM DEMO? Orders use practice funds on your demo account.'
        )
      ) {
        return
      }
    }
    update.mutate(next)
  }

  const color =
    mode === 'live'
      ? 'border-red-400/50 text-red-300 bg-red-500/10'
      : mode === 'demo'
        ? 'border-accent-gold/50 text-accent-gold bg-accent-gold/10'
        : 'border-white/20 text-ink-soft bg-white/[0.03]'

  return (
    <label
      className={`inline-flex items-center gap-2 ${className}`}
      title="Paper / Demo / Live — switch anytime while testing"
    >
      {!compact && (
        <span className="text-[10px] uppercase tracking-wider text-surface-muted hidden sm:inline">
          Mode
        </span>
      )}
      <select
        value={mode}
        disabled={update.isPending}
        onChange={(e) => onChange(e.target.value)}
        className={`text-xs sm:text-sm font-medium rounded-sm border px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-accent-gold/40 disabled:opacity-50 cursor-pointer ${color}`}
      >
        {OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value} className="bg-[#121210] text-ink">
            {opt.label}
            {opt.value === 'live' ? ' (real $)' : ''}
          </option>
        ))}
      </select>
    </label>
  )
}

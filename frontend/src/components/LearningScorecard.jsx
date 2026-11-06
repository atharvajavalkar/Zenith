import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../services/api'

function money(n) {
  if (n == null || Number.isNaN(n)) return '—'
  const sign = n > 0 ? '+' : ''
  return `${sign}$${n.toFixed(2)}`
}

function ofTen(rate) {
  if (rate == null || Number.isNaN(rate)) return null
  return Math.max(0, Math.min(10, Math.round(rate * 10)))
}

function avg(nums) {
  const clean = nums.filter((n) => n != null && !Number.isNaN(n))
  if (!clean.length) return null
  return clean.reduce((a, b) => a + b, 0) / clean.length
}

function pickinessLabel(mode) {
  if (mode === 'protect') return { short: 'Very strict', detail: 'Protecting the account after rough results' }
  if (mode === 'cautious') return { short: 'Careful', detail: 'Raising the bar after recent losses' }
  return { short: 'Normal', detail: 'Taking setups that clear the usual filters' }
}

function moodFromStates(stateWinRates) {
  const entries = Object.entries(stateWinRates || {}).filter(
    ([, v]) => v?.win_rate != null && (v.count || 0) >= 3,
  )
  if (!entries.length) return null
  const mean = avg(entries.map(([, v]) => v.win_rate))
  const best = [...entries].sort((a, b) => (b[1].win_rate || 0) - (a[1].win_rate || 0))[0]
  const worst = [...entries].sort((a, b) => (a[1].win_rate || 0) - (b[1].win_rate || 0))[0]
  return {
    mean,
    bestLabel: best?.[0]?.replace(/_/g, ' '),
    bestWr: best?.[1]?.win_rate,
    worstLabel: worst?.[0]?.replace(/_/g, ' '),
    worstWr: worst?.[1]?.win_rate,
    sampleStates: entries.length,
  }
}

function buildTip({ mode, rollingWr, realized, mcFloor, mlFloor, blockedCount, closes }) {
  if (!closes) {
    return 'Not enough closed trades yet — keep running; this box fills in as results come in.'
  }
  if (mode === 'protect' || (realized != null && realized <= -100)) {
    return 'Suggestion: keep filters strict for now. Only loosen the chance / history bars after a few winning days.'
  }
  if (mode === 'cautious' || (rollingWr != null && rollingWr < 0.45)) {
    return 'Suggestion: raise the chance filter a little (MC floor) or skip weak market moods until win rate recovers.'
  }
  if (rollingWr != null && rollingWr >= 0.67 && blockedCount === 0) {
    return 'Suggestion: results look healthy. You can leave gates as-is, or slightly ease the history score if you want more trades.'
  }
  if (blockedCount > 0) {
    return 'Suggestion: review the “Currently skipping” list — those moods/hours have been losing; keep them blocked until they improve.'
  }
  if (mcFloor != null && mcFloor >= 55) {
    return 'Suggestion: chance filter is already high. Focus on quality over quantity until more wins land.'
  }
  if (mlFloor != null && mlFloor >= 0.55) {
    return 'Suggestion: history filter is strict. Good for protection; lower it only if too many decent setups are blocked.'
  }
  return 'Suggestion: watch this scorecard after each session — when win rate and money trend up, gates can stay; when they drop, tighten them.'
}

function ScoreRow({ label, value, note }) {
  return (
    <div className="min-w-0 py-3 border-b border-surface-border last:border-0">
      <div className="text-[11px] uppercase tracking-wider text-surface-muted">{label}</div>
      <div className="mt-1 text-[15px] font-medium text-gray-100 leading-snug">{value}</div>
      {note ? <p className="mt-1 text-xs text-surface-muted leading-relaxed">{note}</p> : null}
    </div>
  )
}

/**
 * Plain-English averages from Markov / ML / closed trades —
 * so gate tweaks can be decided without scientific jargon.
 */
export default function LearningScorecard() {
  const { data, isLoading } = useQuery({
    queryKey: ['trade-learning'],
    queryFn: async () => (await api.get('/api/trades/learning')).data,
    refetchInterval: 8000,
  })

  const { data: summary } = useQuery({
    queryKey: ['trade-summary'],
    queryFn: async () => (await api.get('/api/trades/summary')).data,
    refetchInterval: 8000,
  })

  const score = useMemo(() => {
    const adaptive = data?.adaptive || {}
    const model = data?.model || {}
    const patterns = data?.patterns || {}
    const markov = data?.markov || {}
    const mood = moodFromStates(markov.state_win_rates)

    const closes = patterns.total_closed ?? data?.closes_learned ?? 0
    const rollingWr = adaptive.rolling_win_rate ?? patterns.overall_win_rate ?? null
    const trainWr = model.train_win_rate ?? null
    const gatedWr = model.filtered_win_rate ?? null
    const realized = adaptive.realized_pnl ?? summary?.realized_pnl ?? null
    const mode = adaptive.mode || 'normal'
    const picky = pickinessLabel(mode)
    const mcFloor = adaptive.mc_floor ?? null
    const mlFloor = adaptive.min_ml_prob ?? model.threshold ?? null
    const blockedStates = adaptive.blocked_states || []
    const blockedHours = adaptive.blocked_hours || []
    const blockedCount = blockedStates.length + blockedHours.length

    const avgMcFromCal =
      markov.calibration?.predicted_wr != null
        ? markov.calibration.predicted_wr * 100
        : null
    const avgActualFromCal =
      markov.calibration?.actual_wr != null
        ? markov.calibration.actual_wr * 100
        : null

    const winsOfTen = ofTen(rollingWr)
    const historyOfTen = ofTen(trainWr)
    const gatedOfTen = ofTen(gatedWr)
    const moodOfTen = ofTen(mood?.mean)

    const skipBits = []
    if (blockedStates.length) {
      skipBits.push(blockedStates.slice(0, 3).map((s) => s.replace(/_/g, ' ')).join(', '))
    }
    if (blockedHours.length) {
      const hrs = blockedHours
        .slice(0, 4)
        .map((h) => `${String(h).padStart(2, '0')}:00 UTC`)
        .join(', ')
      skipBits.push(`hours ${hrs}`)
    }

    const tip = buildTip({
      mode,
      rollingWr,
      realized,
      mcFloor,
      mlFloor,
      blockedCount,
      closes,
    })

    return {
      closes,
      picky,
      winsLine:
        winsOfTen != null
          ? `About ${winsOfTen} of every 10 recent trades made money`
          : 'Still collecting recent results',
      winsNote:
        rollingWr != null
          ? `Recent win rate ${(rollingWr * 100).toFixed(0)}% · ${adaptive.recent_count ?? '—'} trades in the window`
          : 'Needs a few closed trades first',
      moneyLine: realized != null ? `Overall ${money(realized)}` : 'No closed P&L yet',
      moneyNote:
        adaptive.loss_streak > 0
          ? `On a ${adaptive.loss_streak}-loss streak — filters get tighter automatically`
          : 'No active losing streak',
      chanceLine:
        mcFloor != null
          ? `Only take setups that look at least ${Math.round(mcFloor)}% likely`
          : 'Chance filter not set yet',
      chanceNote:
        avgMcFromCal != null && avgActualFromCal != null
          ? `Memory guessed ~${avgMcFromCal.toFixed(0)}% wins; real results were ~${avgActualFromCal.toFixed(0)}%`
          : 'Compares “what we thought” vs “what happened” as more trades close',
      historyLine:
        mlFloor != null
          ? `History score must be at least ${Math.round(mlFloor * 100)}%`
          : 'History filter warming up',
      historyNote:
        historyOfTen != null
          ? `From past trades the model sees ~${historyOfTen}/10 wins` +
            (gatedOfTen != null ? ` · after filters ~${gatedOfTen}/10` : '')
          : model.trained
            ? 'Model is training on closed trades'
            : 'Not enough samples to train yet',
      moodLine:
        moodOfTen != null
          ? `Across market moods, wins average about ${moodOfTen} of 10`
          : 'Market-mood memory still filling in',
      moodNote: mood?.bestLabel
        ? `Stronger in “${mood.bestLabel}” · weaker in “${mood.worstLabel}” (${mood.sampleStates} moods tracked)`
        : `${markov.state_sequence_length ?? 0} candle snapshots remembered`,
      skipLine: blockedCount
        ? `Skipping: ${skipBits.join(' · ')}`
        : 'Not skipping any moods or hours right now',
      tip,
      updates: adaptive.updates ?? 0,
    }
  }, [data, summary])

  return (
    <section className="zcard p-5">
      <div className="mb-1">
        <h3 className="panel-title">How the algo is doing</h3>
        <p className="panel-desc max-w-3xl">
          Simple averages from closed trades and candle memory — use this when you want to
          tighten or loosen the chance / history filters later.
        </p>
      </div>

      {isLoading && !data ? (
        <div className="text-sm text-surface-muted animate-pulse py-8 text-center">
          Averaging results…
        </div>
      ) : (
        <>
          <div className="mt-4 flex flex-wrap items-baseline gap-x-4 gap-y-1 text-xs text-surface-muted">
            <span>
              Learned from{' '}
              <span className="text-gray-300 tabular-nums">{score.closes}</span> closed trades
            </span>
            <span className="text-white/10">·</span>
            <span>
              Pickiness:{' '}
              <span className="text-accent-gold">{score.picky.short}</span>
              <span className="text-surface-muted"> — {score.picky.detail}</span>
            </span>
            {score.updates > 0 && (
              <>
                <span className="text-white/10">·</span>
                <span className="tabular-nums">{score.updates} learning updates</span>
              </>
            )}
          </div>

          <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-x-10">
            <ScoreRow label="Win rate lately" value={score.winsLine} note={score.winsNote} />
            <ScoreRow label="Money so far" value={score.moneyLine} note={score.moneyNote} />
            <ScoreRow label="Chance filter" value={score.chanceLine} note={score.chanceNote} />
            <ScoreRow label="History filter" value={score.historyLine} note={score.historyNote} />
            <ScoreRow label="Market mood memory" value={score.moodLine} note={score.moodNote} />
            <ScoreRow
              label="Currently skipping"
              value={score.skipLine}
              note="Weak moods/hours learned from recent losers"
            />
          </div>

          <div className="mt-4 rounded-lg border border-accent-gold/25 bg-accent-gold/[0.04] px-4 py-3">
            <div className="text-[11px] uppercase tracking-wider text-accent-gold/80">
              What this means for gates
            </div>
            <p className="mt-1.5 text-sm text-gray-200 leading-relaxed">{score.tip}</p>
          </div>
        </>
      )}
    </section>
  )
}

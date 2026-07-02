/** Technical indicators — standard formulas matching TradingView defaults. */

export function calcEMA(candles, period) {
  if (!candles?.length || period < 1) return []
  const sorted = [...candles].sort((a, b) => a.time - b.time)
  const k = 2 / (period + 1)
  const out = []
  let ema = null
  for (let i = 0; i < sorted.length; i++) {
    const close = sorted[i].close
    if (ema === null) {
      if (i < period - 1) continue
      ema = sorted.slice(0, period).reduce((s, c) => s + c.close, 0) / period
    } else {
      ema = close * k + ema * (1 - k)
    }
    out.push({ time: sorted[i].time, value: Number(ema.toFixed(4)) })
  }
  return out
}

export function calcSMA(candles, period) {
  if (!candles?.length || period < 1) return []
  const sorted = [...candles].sort((a, b) => a.time - b.time)
  const out = []
  for (let i = period - 1; i < sorted.length; i++) {
    const slice = sorted.slice(i - period + 1, i + 1)
    const avg = slice.reduce((s, c) => s + c.close, 0) / period
    out.push({ time: sorted[i].time, value: Number(avg.toFixed(4)) })
  }
  return out
}

export function calcBollinger(candles, period = 20, mult = 2) {
  if (!candles?.length) return { upper: [], middle: [], lower: [] }
  const sorted = [...candles].sort((a, b) => a.time - b.time)
  const upper = []
  const middle = []
  const lower = []
  for (let i = period - 1; i < sorted.length; i++) {
    const slice = sorted.slice(i - period + 1, i + 1)
    const mean = slice.reduce((s, c) => s + c.close, 0) / period
    const variance = slice.reduce((s, c) => s + (c.close - mean) ** 2, 0) / period
    const std = Math.sqrt(variance)
    const t = sorted[i].time
    middle.push({ time: t, value: Number(mean.toFixed(4)) })
    upper.push({ time: t, value: Number((mean + mult * std).toFixed(4)) })
    lower.push({ time: t, value: Number((mean - mult * std).toFixed(4)) })
  }
  return { upper, middle, lower }
}

export function calcRSI(candles, period = 14) {
  if (!candles?.length || period < 1) return []
  const sorted = [...candles].sort((a, b) => a.time - b.time)
  const out = []
  if (sorted.length <= period) return out

  let gains = 0
  let losses = 0
  for (let i = 1; i <= period; i++) {
    const diff = sorted[i].close - sorted[i - 1].close
    if (diff >= 0) gains += diff
    else losses -= diff
  }
  let avgGain = gains / period
  let avgLoss = losses / period

  const rs = avgLoss === 0 ? 100 : avgGain / avgLoss
  out.push({ time: sorted[period].time, value: Number((100 - 100 / (1 + rs)).toFixed(2)) })

  for (let i = period + 1; i < sorted.length; i++) {
    const diff = sorted[i].close - sorted[i - 1].close
    const gain = diff > 0 ? diff : 0
    const loss = diff < 0 ? -diff : 0
    avgGain = (avgGain * (period - 1) + gain) / period
    avgLoss = (avgLoss * (period - 1) + loss) / period
    const rsi = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss)
    out.push({ time: sorted[i].time, value: Number(rsi.toFixed(2)) })
  }
  return out
}

export function calcIndicator(candles, indicator) {
  switch (indicator.type) {
    case 'ema':
      return { main: calcEMA(candles, indicator.period) }
    case 'sma':
      return { main: calcSMA(candles, indicator.period) }
    case 'bb':
      return calcBollinger(candles, indicator.period, indicator.mult ?? 2)
    case 'rsi':
      return { main: calcRSI(candles, indicator.period) }
    default:
      return { main: [] }
  }
}

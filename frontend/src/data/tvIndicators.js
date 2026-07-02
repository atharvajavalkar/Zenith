/**
 * TradingView-style indicator catalog.
 * Toggle any indicator on the chart for manual analysis (standard TA formulas).
 */
export const TV_INDICATOR_CATALOG = [
  {
    category: 'Moving Averages',
    items: [
      { id: 'ema8', name: 'EMA 8', type: 'ema', period: 8, color: '#2962FF', lineWidth: 2 },
      { id: 'ema9', name: 'EMA 9', type: 'ema', period: 9, color: '#2962FF', lineWidth: 1 },
      { id: 'ema14', name: 'EMA 14', type: 'ema', period: 14, color: '#64B5F6', lineWidth: 2 },
      { id: 'ema21', name: 'EMA 21', type: 'ema', period: 21, color: '#FF6D00', lineWidth: 2 },
      { id: 'ema50', name: 'EMA 50', type: 'ema', period: 50, color: '#AB47BC', lineWidth: 2 },
      { id: 'ema200', name: 'EMA 200', type: 'ema', period: 200, color: '#787B86', lineWidth: 2 },
      { id: 'sma20', name: 'SMA 20', type: 'sma', period: 20, color: '#F23645', lineWidth: 1 },
      { id: 'sma50', name: 'SMA 50', type: 'sma', period: 50, color: '#9C27B0', lineWidth: 1 },
      { id: 'sma200', name: 'SMA 200', type: 'sma', period: 200, color: '#131722', lineWidth: 2 },
    ],
  },
  {
    category: 'Volatility',
    items: [
      { id: 'bb20', name: 'Bollinger Bands (20, 2)', type: 'bb', period: 20, mult: 2, color: '#787B86' },
    ],
  },
  {
    category: 'Oscillators',
    items: [
      { id: 'rsi14', name: 'RSI 14', type: 'rsi', period: 14, color: '#7E57C2', pane: 'rsi' },
    ],
  },
]

export const ALL_INDICATORS = TV_INDICATOR_CATALOG.flatMap((g) => g.items)

export function getIndicatorById(id) {
  return ALL_INDICATORS.find((i) => i.id === id)
}

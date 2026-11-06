# Dual-horizon adaptive policy

Research only — live brains not wired.

Problem: absolute 1y winners can be weak on 7d (or vice versa).
Solution candidates:
1. **Dual-score pick** — maximize short+long composite per pair
2. **Horizon router** — short book for 7–90d recovery, long book for compound
3. **Both-green filter** — only trade formula if short AND long green

## ETHUSD
- 1y absolute: `session_mix` ($17,600)
- dual pick: `session_mix` (7d $-200 · 90d $8,400 · 1y $17,600 · 3y $47,200)
- both-green: `session_mix`
- recommend wire candidate: `session_mix`

## BTCUSD
- 1y absolute: `circadian_kalman_z` ($24,800)
- dual pick: `circadian_kalman_z` (7d $2,800 · 90d $6,000 · 1y $24,800 · 3y $92,600)
- both-green: `circadian_kalman_z`
- recommend wire candidate: `circadian_kalman_z`

## GOLD
- 1y absolute: `entropy_tod` ($30,600)
- dual pick: `static` (7d $0 · 90d $6,200 · 1y $30,000 · 3y $46,600)
- both-green: `static`
- recommend wire candidate: `static`

## NAS100
- 1y absolute: `entropy_tod` ($15,200)
- dual pick: `entropy_tod` (7d $1,400 · 90d $5,000 · 1y $15,200 · 3y $34,800)
- both-green: `entropy_tod`
- recommend wire candidate: `entropy_tod`

## USDJPY
- 1y absolute: `circadian_kalman_z` ($66,400)
- dual pick: `session_mix` (7d $6,000 · 90d $25,000 · 1y $55,000 · 3y $138,800)
- both-green: `session_mix`
- recommend wire candidate: `session_mix`

## AUDUSD
- 1y absolute: `circadian_kalman_z` ($48,600)
- dual pick: `circadian_kalman_z` (7d $-400 · 90d $6,000 · 1y $48,600 · 3y $146,600)
- both-green: `circadian_kalman_z`
- recommend wire candidate: `circadian_kalman_z`

## USDCHF
- 1y absolute: `session_mix` ($61,000)
- dual pick: `session_mix` (7d $2,800 · 90d $28,200 · 1y $61,000 · 3y $187,400)
- both-green: `session_mix`
- recommend wire candidate: `session_mix`

## UKOIL
- 1y absolute: `session_mix` ($51,000)
- dual pick: `session_mix` (7d $7,800 · 90d $13,600 · 1y $51,000 · 3y $129,400)
- both-green: `session_mix`
- recommend wire candidate: `session_mix`

## Desk totals
- absolute 1y mix: $315,200
- dual-pick 1y: $303,200
- dual-pick 7d: $20,200
- dual-pick 3y: $823,400
- pairs with both-green formula: 8/8

## How to be efficient short AND long
1. Stop optimizing only 1y PnL — use dual score (0.5 short + 0.5 long).
2. Prefer both-green formulas; if none, keep long book and throttle size short-term.
3. Per-pair books beat one global adaptive router.
4. v2 gated/switch alone is not enough — calendar equations (session_mix, circadian_kalman_z) carry the edge.
5. Live wire only after confirm — proposed map is research.


# Adaptive v3 — minute/day equations

Research only. Live brains unchanged.

At each bar we recompute sides from calendar (minute-of-day, weekday, session)
and live features. Bake-off vs adaptive v2.

## ETHUSD
- v2 preferred: `static`
- absolute winner: `session_mix` (beat_v2=True)
- 1y: $17,600 · E[R]=0.052 · n=1707

## BTCUSD
- v2 preferred: `switch`
- absolute winner: `circadian_kalman_z` (beat_v2=True)
- 1y: $24,800 · E[R]=0.056 · n=2224

## GOLD
- v2 preferred: `static`
- absolute winner: `entropy_tod` (beat_v2=True)
- 1y: $30,600 · E[R]=0.314 · n=487

## NAS100
- v2 preferred: `static`
- absolute winner: `entropy_tod` (beat_v2=True)
- 1y: $15,200 · E[R]=0.074 · n=1028

## USDJPY
- v2 preferred: `static`
- absolute winner: `circadian_kalman_z` (beat_v2=True)
- 1y: $66,400 · E[R]=0.11 · n=3022

## AUDUSD
- v2 preferred: `static`
- absolute winner: `circadian_kalman_z` (beat_v2=True)
- 1y: $48,600 · E[R]=0.081 · n=2991

## USDCHF
- v2 preferred: `static`
- absolute winner: `session_mix` (beat_v2=True)
- 1y: $61,000 · E[R]=0.064 · n=4795

## UKOIL
- v2 preferred: `static`
- absolute winner: `session_mix` (beat_v2=True)
- 1y: $51,000 · E[R]=0.116 · n=2200

## Desk
- absolute (per-pair best): 1y $315,200
- v2 preferred desk: 1y $191,800
- pairs beating v2: 8/8

## Equation catalog
- `circadian_thr` — SNR thr breathes with sin/cos(TOD)
- `dow_book` — Weekday book switch (trend→fade)
- `session_mix` — Soft α(sess)·trend + β·fade mix
- `harmonic_gate` — Static only on active harmonic minutes
- `vol_clock` — Minute-of-day vol clock → fade/trend
- `weekday_regime_prior` — P(trend|dow) × live regime router
- `tod_fade_boost` — London/NY open fade boost
- `entropy_tod` — Perm-entropy ceiling = f(TOD,week)
- `minute_softmax` — Bar-level softmax book router
- `hybrid_v2_tod` — v2 preferred + TOD/night sit-out
- `circadian_kalman_z` — Kalman |z| thr = f(TOD, DOW)
- `intraday_cusum_clock` — Cusum slope × active hours gate

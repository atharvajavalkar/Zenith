# Adaptive XM-cost v4

Dukascopy hist + measured XM session spreads per trade.
Live SL floors + N×spread min risk. Absolute winner = best 1y under XM costs.

## ETHUSD
- absolute: `vol_clock` (SOFT) wireable=True 1y $3,557 E[R]=0.2223
- SL=2.0×ATR · RR=4.0 · spread_mult=8.0

## BTCUSD
- absolute: `entropy_tod` (FAIL) wireable=False 1y $84 E[R]=0.0836
- SL=2.0×ATR · RR=3.0 · spread_mult=8.0

## GOLD
- absolute: `circadian_thr` (PASS) wireable=True 1y $3,315 E[R]=0.15
- SL=2.0×ATR · RR=4.0 · spread_mult=10.0

## NAS100
- absolute: `entropy_strict` (PASS) wireable=True 1y $4,654 E[R]=0.3422
- SL=2.0×ATR · RR=4.0 · spread_mult=8.0

## USDJPY
- absolute: `weekday_regime_prior` (SOFT) wireable=True 1y $1,102 E[R]=0.0268
- SL=2.5×ATR · RR=2.0 · spread_mult=10.0

## AUDUSD
- absolute: `session_mix_no_off` (SOFT) wireable=True 1y $1,610 E[R]=0.0575
- SL=2.5×ATR · RR=2.0 · spread_mult=10.0

## USDCHF
- absolute: `session_mix` (SOFT) wireable=True 1y $925 E[R]=0.0586
- SL=2.5×ATR · RR=2.0 · spread_mult=10.0

## UKOIL
- absolute: `weekday_regime_prior` (PASS) wireable=True 1y $1,598 E[R]=0.045
- SL=2.0×ATR · RR=4.0 · spread_mult=1.0


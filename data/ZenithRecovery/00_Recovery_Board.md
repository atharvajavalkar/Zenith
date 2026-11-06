# Zenith Desk Recovery Board

Finished: 2026-08-12T12:42:00.969176+00:00
Elapsed: 5.36h · densify rounds: 24
Mode: **scalper / intraday** (1m·3m·5m·15m)
LIVE `.env` **not** auto-wired — review then selective wire only.

| Pair | Live | Winner | TF | RR | n | E[R] | Gate |
|---|---|---|---|---:|---:|---:|---|
| ETHUSD | `breakout_20` | `bb_2+hf0.45_e0.30` | 1m | 4.0 | 448 | 0.272 | True |
| BTCUSD | `bb_2.5` | `bb_2.5+e0.35` | 1m | 4.0 | 88 | 1.045 | True |
| GOLD | `pa_liq_engulf` | `don_30+hf0.55` | 15m | 4.0 | 399 | 0.291 | True |
| NAS100 | `volmom` | `volmom` | 15m | 4.0 | 499 | 0.263 | True |
| USDJPY | `pa_opp_gold` | `bb_2.5+hf0.55` | 1m | 4.0 | 98 | 0.99 | True |
| AUDUSD | `bb_2.5` | `bb_2.5+e0.35` | 1m | 4.0 | 92 | 0.793 | True |
| USDCHF | `bb_2.5` | `bb_2+D1_hf0.58` | 3m | 4.0 | 434 | 0.198 | True |
| UKOIL | `cta_10_40` | `cta_10_40+hf0.55` | 5m | 4.0 | 637 | 0.224 | True |

## Firm scalp candidates (deep_gate)
- **ETHUSD**: `bb_2+hf0.45_e0.30` @1m RR=4.0 n=448 E[R]=0.272 score=15.075000000000001
- **BTCUSD**: `bb_2.5+e0.35` @1m RR=4.0 n=88 E[R]=1.045 score=16.506
- **GOLD**: `don_30+hf0.55` @15m RR=4.0 n=399 E[R]=0.291 score=14.324
- **NAS100**: `volmom` @15m RR=4.0 n=499 E[R]=0.263 score=14.248
- **USDJPY**: `bb_2.5+hf0.55` @1m RR=4.0 n=98 E[R]=0.99 score=16.222
- **AUDUSD**: `bb_2.5+e0.35` @1m RR=4.0 n=92 E[R]=0.793 score=14.991000000000001
- **USDCHF**: `bb_2+D1_hf0.58` @3m RR=4.0 n=434 E[R]=0.198 score=14.558
- **UKOIL**: `cta_10_40+hf0.55` @5m RR=4.0 n=637 E[R]=0.224 score=14.644

## Scalper punch notes
- Prefer 1m BB+math on crypto/FX for frequency (ETH n=448, CHF n=434).
- GOLD hunt winner is Donchian30+HF @15m (not PA liquidity) — more punches than bias-gated PA.
- NAS100 live volmom already wins; UKOIL keeps CTA with HF filter.
- USDJPY: BB+HF beats live opp-GOLD on paper — high E[R], low n (98).

_Paper ≠ XM. Do not wire until reviewed._
_Board rebuilt 2026-08-12T12:51:25.891873+00:00_

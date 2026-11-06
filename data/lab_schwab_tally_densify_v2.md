# Schwab tally densify v2 — wide lookback permutations

- **RESEARCH_MORE**: 8/8 SURVIVES on 1y after wide permute — keep going (WF lock next).
- Lookback sets: **793** (2/3/4/5-way combos)
- Nebula untouched · Dukascopy · $200.0/R

## Best per pair (1y)

| Pair | Verdict | E[R] | PnL$ | Lot | Hold | TP | Mode | Lookbacks | WF | MC |
|---|---|---:|---:|---:|---:|---|---|---|---|---|
| GOLD | SURVIVES | 2.152 | 21090.42 | 2.5 | 5 | None | strict | `[7, 60]` | 4/4 | 100.0 |
| BTCUSD | SURVIVES | 0.953 | 10678.12 | 2.5 | 5 | None | strict | `[5, 10]` | 4/4 | 97.5 |
| UKOIL | SURVIVES | 1.026 | 12110.56 | 2.5 | 5 | None | strict | `[60, 63]` | 4/4 | 95.8 |
| NAS100 | SURVIVES | 0.859 | 12542.41 | 2.5 | 4 | None | strict | `[20, 21, 90]` | 4/4 | 98.8 |
| AUDUSD | SURVIVES | 0.751 | 8256.31 | 2.5 | 5 | None | strict | `[3, 7]` | 4/4 | 94.6 |
| ETHUSD | SURVIVES | 1.004 | 10439.39 | 2.5 | 5 | None | strict | `[28, 56]` | 4/4 | 98.1 |
| USDJPY | SURVIVES | 0.889 | 9956.97 | 2.5 | 5 | None | strict | `[63, 84]` | 2/4 | 91.0 |
| USDCHF | SURVIVES | 0.247 | 4449.43 | 2.5 | 3 | None | strict | `[20, 21]` | 3/4 | 76.9 |

## vs v1 (1y E[R] delta)

- **GOLD**: v1=0.978 → v2=2.152 (Δ 1.174) lb=`[7, 60]`
- **BTCUSD**: v1=0.397 → v2=0.953 (Δ 0.556) lb=`[5, 10]`
- **UKOIL**: v1=0.246 → v2=1.026 (Δ 0.78) lb=`[60, 63]`
- **NAS100**: v1=0.283 → v2=0.859 (Δ 0.576) lb=`[20, 21, 90]`
- **AUDUSD**: v1=0.073 → v2=0.751 (Δ 0.678) lb=`[3, 7]`
- **ETHUSD**: v1=0.176 → v2=1.004 (Δ 0.828) lb=`[28, 56]`
- **USDJPY**: v1=0.077 → v2=0.889 (Δ 0.812) lb=`[63, 84]`
- **USDCHF**: v1=0.032 → v2=0.247 (Δ 0.215) lb=`[20, 21]`

## Top crowns

- **GOLD** E[R]=2.152 pnl=$21090.42 lb=`[7, 60]` lot=2.5 hold=5 tp=None
- **GOLD** E[R]=2.152 pnl=$21090.42 lb=`[7, 60]` lot=2.5 hold=5 tp=None
- **GOLD** E[R]=2.124 pnl=$21238.54 lb=`[14, 42]` lot=2.5 hold=5 tp=None
- **GOLD** E[R]=2.124 pnl=$21238.54 lb=`[14, 42]` lot=2.5 hold=5 tp=None
- **GOLD** E[R]=2.076 pnl=$21178.1 lb=`[7, 42]` lot=2.5 hold=5 tp=None
- **GOLD** E[R]=2.076 pnl=$21178.1 lb=`[7, 42]` lot=2.5 hold=5 tp=None
- **GOLD** E[R]=2.068 pnl=$23576.21 lb=`[21, 28]` lot=2.5 hold=5 tp=None
- **GOLD** E[R]=2.068 pnl=$23576.21 lb=`[21, 28]` lot=2.5 hold=5 tp=None
- **UKOIL** E[R]=1.026 pnl=$12110.56 lb=`[60, 63]` lot=2.5 hold=5 tp=None
- **UKOIL** E[R]=1.026 pnl=$12110.56 lb=`[60, 63]` lot=2.5 hold=5 tp=None
- **ETHUSD** E[R]=1.004 pnl=$10439.39 lb=`[28, 56]` lot=2.5 hold=5 tp=None
- **ETHUSD** E[R]=1.004 pnl=$10439.39 lb=`[28, 56]` lot=2.5 hold=5 tp=None
- **ETHUSD** E[R]=0.966 pnl=$9655.37 lb=`[3, 21]` lot=2.5 hold=5 tp=None
- **ETHUSD** E[R]=0.966 pnl=$9655.37 lb=`[3, 21]` lot=2.5 hold=5 tp=None
- **BTCUSD** E[R]=0.953 pnl=$10678.12 lb=`[5, 10]` lot=2.5 hold=5 tp=None
- **BTCUSD** E[R]=0.953 pnl=$10678.12 lb=`[5, 10]` lot=2.5 hold=5 tp=None
- **UKOIL** E[R]=0.91 pnl=$10009.81 lb=`[30, 56]` lot=2.5 hold=5 tp=None
- **UKOIL** E[R]=0.91 pnl=$10009.81 lb=`[30, 56]` lot=2.5 hold=5 tp=None
- **USDJPY** E[R]=0.889 pnl=$9956.97 lb=`[63, 84]` lot=2.5 hold=5 tp=None
- **USDJPY** E[R]=0.889 pnl=$9956.97 lb=`[63, 84]` lot=2.5 hold=5 tp=None
- **UKOIL** E[R]=0.876 pnl=$12782.95 lb=`[28, 30, 56]` lot=2.5 hold=4 tp=None
- **UKOIL** E[R]=0.871 pnl=$12715.94 lb=`[28, 30, 56]` lot=2.5 hold=4 tp=None
- **NAS100** E[R]=0.859 pnl=$12542.41 lb=`[20, 21, 90]` lot=2.5 hold=4 tp=None
- **UKOIL** E[R]=0.825 pnl=$9240.0 lb=`[5, 56]` lot=2.5 hold=4 tp=None
- **UKOIL** E[R]=0.825 pnl=$9240.0 lb=`[5, 56]` lot=2.5 hold=4 tp=None
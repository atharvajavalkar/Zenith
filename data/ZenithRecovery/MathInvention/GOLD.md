# GOLD — invented math formula

- **Winner:** `perm_order_0.80` · Permutation-entropy order
- **Equation:** $H_{perm}\le 0.8\ \wedge\ side=\mathrm{sign}(\mu)$
- **TF / RR:** 5m / 4.0
- **n / WR / E[R]:** 2427 / 22.5 / 0.1228
- **OOS E[R] / WF / MC:** 0.2835 / 4 / 99.9
- **Deep gate:** True · score=15.839
- **Periods:** `{"1y": {"n": 275, "exp_r": 0.545, "win_rate": null, "pnl": 30000.0, "verdict": "FRAGILE", "wf_green": 0, "mc": 100.0, "oos_exp_r": 0.0}, "3y": {"n": 862, "exp_r": 0.27, "win_rate": null, "pnl": 46600.0, "verdict": "FRAGILE", "wf_green": 0, "mc": 100.0, "oos_exp_r": 0.0}, "5y": {"n": 1467, "exp_r": 0.186, "win_rate": null, "pnl": 54600.0, "verdict": "SURVIVES", "wf_green": 2, "mc": 100.0, "oos_exp_r": 0.1395}, "full": {"n": 2427, "exp_r": 0.123, "win_rate": null, "pnl": 59600.0, "verdict": "SURVIVES", "wf_green": 4, "mc": 99.9, "oos_exp_r": 0.2835}}`

## Top 5
1. `perm_order_0.80` @5m RR=4.0 n=2427 WR=22.5 E[R]=0.1228 gate=True eq=`H_{perm}\le 0.8\ \wedge\ side=\mathrm{sign}(\mu)`
2. `bimod_0.70` @5m RR=4.0 n=995 WR=22.8 E[R]=0.1407 gate=True eq=`\beta=\frac{skew^2+1}{kurt+3}\ge 0.7\ \wedge\ trend`
3. `perm_order_0.80` @5m RR=3.0 n=2630 WR=26.3 E[R]=0.051 gate=True eq=`H_{perm}\le 0.8\ \wedge\ side=\mathrm{sign}(\mu)`
4. `kalman_fade_1.6` @5m RR=2.4 n=21500 WR=30.3 E[R]=0.029 gate=True eq=`z_{kal}=\frac{c-EMA_{20}}{ATR};\ fade\ |z|\ge 1.6`
5. `ou_snap_8_25` @5m RR=3.0 n=12011 WR=26.5 E[R]=0.06 gate=True eq=`t_{1/2}^{OU}\in[8,25];\ fade\ z_{kal}`

_Updated 2026-08-12T16:30:10.858156+00:00_

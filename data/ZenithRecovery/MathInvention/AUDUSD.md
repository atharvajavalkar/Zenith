# AUDUSD — invented math formula

- **Winner:** `kalman_fade_2.4` · Kalman innov fade
- **Equation:** $z_{kal}=\frac{c-EMA_{20}}{ATR};\ fade\ |z|\ge 2.4$
- **TF / RR:** 5m / 1.5
- **n / WR / E[R]:** 9105 / 42.2 / 0.0541
- **OOS E[R] / WF / MC:** 0.0841 / 4 / 100.0
- **Deep gate:** True · score=15.757
- **Periods:** `{"1y": {"n": 1127, "exp_r": 0.105, "win_rate": null, "pnl": 23600.0, "verdict": "FRAGILE", "wf_green": 0, "mc": 99.8, "oos_exp_r": 0.0}, "3y": {"n": 3338, "exp_r": 0.081, "win_rate": null, "pnl": 53900.0, "verdict": "FRAGILE", "wf_green": 0, "mc": 100.0, "oos_exp_r": 0.0}, "5y": {"n": 5545, "exp_r": 0.052, "win_rate": null, "pnl": 57500.0, "verdict": "FRAGILE", "wf_green": 2, "mc": 99.9, "oos_exp_r": -0.0033}, "full": {"n": 9105, "exp_r": 0.054, "win_rate": null, "pnl": 98500.0, "verdict": "SURVIVES", "wf_green": 4, "mc": 100.0, "oos_exp_r": 0.0841}}`

## Top 5
1. `kalman_fade_2.4` @5m RR=1.5 n=9105 WR=42.2 E[R]=0.0541 gate=True eq=`z_{kal}=\frac{c-EMA_{20}}{ATR};\ fade\ |z|\ge 2.4`
2. `kalman_fade_1.6` @5m RR=1.5 n=28464 WR=42.0 E[R]=0.0501 gate=True eq=`z_{kal}=\frac{c-EMA_{20}}{ATR};\ fade\ |z|\ge 1.6`
3. `hf_spike_fade_0.60` @5m RR=1.5 n=16968 WR=42.0 E[R]=0.0504 gate=True eq=`HF\ge 0.6\ \Rightarrow\ fade\ z_{kal}`
4. `kalman_fade_2.4` @5m RR=2.4 n=8681 WR=31.7 E[R]=0.0782 gate=True eq=`z_{kal}=\frac{c-EMA_{20}}{ATR};\ fade\ |z|\ge 2.4`
5. `ou_snap_5_40` @5m RR=1.5 n=27291 WR=41.7 E[R]=0.0432 gate=True eq=`t_{1/2}^{OU}\in[5,40];\ fade\ z_{kal}`

_Updated 2026-08-12T16:44:48.744955+00:00_

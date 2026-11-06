# USDJPY — invented math formula

- **Winner:** `kalman_fade_2.0` · Kalman innov fade
- **Equation:** $z_{kal}=\frac{c-EMA_{20}}{ATR};\ fade\ |z|\ge 2.0$
- **TF / RR:** 5m / 1.5
- **n / WR / E[R]:** 17697 / 41.8 / 0.0448
- **OOS E[R] / WF / MC:** 0.059 / 4 / 100.0
- **Deep gate:** True · score=15.628
- **Periods:** `{"1y": {"n": 2158, "exp_r": 0.095, "win_rate": null, "pnl": 40900.0, "verdict": "FRAGILE", "wf_green": 0, "mc": 100.0, "oos_exp_r": 0.0}, "3y": {"n": 6395, "exp_r": 0.052, "win_rate": null, "pnl": 67000.0, "verdict": "FRAGILE", "wf_green": 0, "mc": 100.0, "oos_exp_r": 0.0}, "5y": {"n": 10862, "exp_r": 0.047, "win_rate": null, "pnl": 103100.0, "verdict": "SURVIVES", "wf_green": 3, "mc": 100.0, "oos_exp_r": 0.0407}, "full": {"n": 17697, "exp_r": 0.045, "win_rate": null, "pnl": 158600.0, "verdict": "SURVIVES", "wf_green": 4, "mc": 100.0, "oos_exp_r": 0.059}}`

## Top 5
1. `kalman_fade_2.0` @5m RR=1.5 n=17697 WR=41.8 E[R]=0.0448 gate=True eq=`z_{kal}=\frac{c-EMA_{20}}{ATR};\ fade\ |z|\ge 2.0`
2. `kalman_fade_1.6` @5m RR=1.5 n=28387 WR=41.0 E[R]=0.0262 gate=True eq=`z_{kal}=\frac{c-EMA_{20}}{ATR};\ fade\ |z|\ge 1.6`
3. `kalman_fade_2.0` @5m RR=2.0 n=16738 WR=35.2 E[R]=0.055 gate=True eq=`z_{kal}=\frac{c-EMA_{20}}{ATR};\ fade\ |z|\ge 2.0`
4. `ou_snap_5_40` @5m RR=1.5 n=27309 WR=41.2 E[R]=0.0302 gate=True eq=`t_{1/2}^{OU}\in[5,40];\ fade\ z_{kal}`
5. `hf_spike_fade_0.60` @5m RR=1.5 n=15925 WR=41.1 E[R]=0.0281 gate=True eq=`HF\ge 0.6\ \Rightarrow\ fade\ z_{kal}`

_Updated 2026-08-12T16:39:48.701136+00:00_

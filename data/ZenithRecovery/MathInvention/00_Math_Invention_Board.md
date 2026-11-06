# Zenith Invented Math Equations

Generated: 2026-08-12T21:12:45.441120+00:00
Elapsed: 5.00h · densify rounds: 35
Research only — **live brains NOT changed**.

## Honesty
No formula wins *all* trades. Below are multi-horizon firm survivors optimized for WR + E[R] + MC + WF across 1y/3y/5y/full.

## Dukascopy hist depth
```json
{
  "ETHUSD": {
    "1m": {
      "start": "2017-12-11 00:00:00",
      "end": "2026-07-23 23:59:00",
      "years": 8.61
    },
    "5m": {
      "start": "2017-12-11 00:00:00",
      "end": "2026-07-23 23:55:00",
      "years": 8.61
    },
    "15m": {
      "start": "2017-12-11 00:00:00",
      "end": "2026-07-23 23:45:00",
      "years": 8.61
    }
  },
  "BTCUSD": {
    "1m": {
      "start": "2017-05-07 00:00:00",
      "end": "2026-07-23 23:59:00",
      "years": 9.21
    },
    "5m": {
      "start": "2017-05-07 00:00:00",
      "end": "2026-07-23 23:55:00",
      "years": 9.21
    },
    "15m": {
      "start": "2017-05-07 00:00:00",
      "end": "2026-07-23 23:45:00",
      "years": 9.21
    }
  },
  "GOLD": {
    "1m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:59:00",
      "years": 10.0
    },
    "5m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:55:00",
      "years": 10.0
    },
    "15m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:45:00",
      "years": 10.0
    }
  },
  "NAS100": {
    "1m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:59:00",
      "years": 10.0
    },
    "5m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:55:00",
      "years": 10.0
    },
    "15m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:45:00",
      "years": 10.0
    }
  },
  "USDJPY": {
    "1m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:59:00",
      "years": 10.0
    },
    "5m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:55:00",
      "years": 10.0
    },
    "15m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:45:00",
      "years": 10.0
    }
  },
  "AUDUSD": {
    "1m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:59:00",
      "years": 10.0
    },
    "5m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:55:00",
      "years": 10.0
    },
    "15m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:45:00",
      "years": 10.0
    }
  },
  "USDCHF": {
    "1m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:59:00",
      "years": 10.0
    },
    "5m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:55:00",
      "years": 10.0
    },
    "15m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:45:00",
      "years": 10.0
    }
  },
  "UKOIL": {
    "1m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:59:00",
      "years": 10.0
    },
    "5m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:55:00",
      "years": 10.0
    },
    "15m": {
      "start": "2016-07-23 00:00:00",
      "end": "2026-07-23 23:45:00",
      "years": 10.0
    }
  }
}
```

## Per-pair winners

### ETHUSD
- **ID:** `bimod_0.55_sl1_D1_sl1.8_D1`
- **Equation:** $$\beta=\frac{skew^2+1}{kurt+3}\ge 0.55\ \wedge\ trend$$
- **TF / RR / SL·ATR:** 3m / 4.0 / 1.8
- **n / WR% / E[R]:** 1613 / 23.7 / 0.1872
- **OOS E[R] / WF / MC%:** 0.1826 / 4 / 100.0
- **Deep gate:** True · score=19.045
- **1y E[R]:** 0.261
- **3y E[R]:** 0.184
- **5y E[R]:** 0.176
- **full E[R]:** 0.187

### BTCUSD
- **ID:** `bimod_0.55_sl1.5_D1`
- **Equation:** $$\beta=\frac{skew^2+1}{kurt+3}\ge 0.55\ \wedge\ trend$$
- **TF / RR / SL·ATR:** 3m / 3.0 / 1.5
- **n / WR% / E[R]:** 1822 / 27.9 / 0.1153
- **OOS E[R] / WF / MC%:** 0.0778 / 3 / 99.9
- **Deep gate:** True · score=18.208
- **1y E[R]:** 0.043
- **3y E[R]:** 0.041
- **5y E[R]:** 0.109
- **full E[R]:** 0.115

### GOLD
- **ID:** `perm_order_0.80`
- **Equation:** $$H_{perm}\le 0.8\ \wedge\ side=\mathrm{sign}(\mu)$$
- **TF / RR / SL·ATR:** 5m / 4.0 / 1.2
- **n / WR% / E[R]:** 2427 / 22.5 / 0.1228
- **OOS E[R] / WF / MC%:** 0.2835 / 4 / 99.9
- **Deep gate:** True · score=15.839
- **1y E[R]:** 0.545
- **3y E[R]:** 0.27
- **5y E[R]:** 0.186
- **full E[R]:** 0.123

### NAS100
- **ID:** `snr_drift_0.40_sl1.2_D1_sl1.5_D1_sl1.8_D1`
- **Equation:** $$SNR_t=\frac{|\mu_{40}(r)|}{\sigma_{40}(r)};\ side=\mathrm{sign}(\mu)\cdot\mathbf{1}[SNR\ge 0.4]$$
- **TF / RR / SL·ATR:** 3m / 2.0 / 1.8
- **n / WR% / E[R]:** 1157 / 37.0 / 0.1098
- **OOS E[R] / WF / MC%:** 0.0091 / 3 / 99.5
- **Deep gate:** True · score=18.331
- **1y E[R]:** 0.022
- **3y E[R]:** 0.1
- **5y E[R]:** 0.103
- **full E[R]:** 0.11

### USDJPY
- **ID:** `kalman_fade_2.0_sl1_D1_sl1.5_D1_sl1_D2`
- **Equation:** $$z_{kal}=\frac{c-EMA_{20}}{ATR};\ fade\ |z|\ge 1.6$$
- **TF / RR / SL·ATR:** 3m / 2.0 / 1.0
- **n / WR% / E[R]:** 31538 / 34.7 / 0.0415
- **OOS E[R] / WF / MC%:** 0.0641 / 4 / 100.0
- **Deep gate:** True · score=18.546
- **1y E[R]:** 0.056
- **3y E[R]:** 0.044
- **5y E[R]:** 0.041
- **full E[R]:** 0.042

### AUDUSD
- **ID:** `kalman_fade_2.4_sl1_D1`
- **Equation:** $$z_{kal}=\frac{c-EMA_{20}}{ATR};\ fade\ |z|\ge 2.4$$
- **TF / RR / SL·ATR:** 3m / 2.0 / 1.0
- **n / WR% / E[R]:** 9666 / 36.0 / 0.0791
- **OOS E[R] / WF / MC%:** 0.0677 / 4 / 100.0
- **Deep gate:** True · score=18.759
- **1y E[R]:** 0.087
- **3y E[R]:** 0.072
- **5y E[R]:** 0.083
- **full E[R]:** 0.079

### USDCHF
- **ID:** `kalman_fade_2.4_sl1_D1`
- **Equation:** $$z_{kal}=\frac{c-EMA_{20}}{ATR};\ fade\ |z|\ge 2.4$$
- **TF / RR / SL·ATR:** 3m / 2.0 / 1.0
- **n / WR% / E[R]:** 10061 / 36.7 / 0.1024
- **OOS E[R] / WF / MC%:** 0.0721 / 4 / 100.0
- **Deep gate:** True · score=18.894
- **1y E[R]:** 0.036
- **3y E[R]:** 0.075
- **5y E[R]:** 0.103
- **full E[R]:** 0.102

### UKOIL
- **ID:** `kalman_fade_2.4_sl1_D1_sl1_D1_sl1_D1`
- **Equation:** $$z_{kal}=\frac{c-EMA_{20}}{ATR};\ fade\ |z|\ge 2.4$$
- **TF / RR / SL·ATR:** 3m / 4.0 / 1.0
- **n / WR% / E[R]:** 7441 / 23.4 / 0.1719
- **OOS E[R] / WF / MC%:** 0.1801 / 4 / 100.0
- **Deep gate:** True · score=18.964
- **1y E[R]:** 0.175
- **3y E[R]:** 0.17
- **5y E[R]:** 0.177
- **full E[R]:** 0.172

## Firm multi-horizon candidates

- **ETHUSD**: `bimod_0.55_sl1_D1_sl1.8_D1`
- **BTCUSD**: `bimod_0.55_sl1.5_D1`
- **GOLD**: `perm_order_0.80`
- **NAS100**: `snr_drift_0.40_sl1.2_D1_sl1.5_D1_sl1.8_D1`
- **USDJPY**: `kalman_fade_2.0_sl1_D1_sl1.5_D1_sl1_D2`
- **AUDUSD**: `kalman_fade_2.4_sl1_D1`
- **USDCHF**: `kalman_fade_2.4_sl1_D1`
- **UKOIL**: `kalman_fade_2.4_sl1_D1_sl1_D1_sl1_D1`

## Catalog (all invented families)

- `snr_drift*`: Drift-diffusion SNR trend
- `hurst_trend*`: Hurst persistence trend
- `hurst_mr*`: Hurst anti-persist mean-revert
- `ou_snap_5*`: OU half-life snapback
- `ou_snap_8*`: OU half-life snapback
- `spec_ent*`: Spectral entropy collapse trend
- `higuchi_trend*`: Low fractal-D smooth trend
- `allan_persist*`: Allan variance persistence
- `perm_order*`: Permutation-entropy order
- `eff_lf*`: Efficiency×LF thrust
- `kalman_fade*`: Kalman innov fade
- `torsion*`: Path-torsion quiet trend
- `zcr*`: Zero-cross compress trend
- `bimod*`: Bimodality breakout
- `cusum*`: CUSUM SNR fire
- `tail_mid*`: Mid-tail + SNR trend
- `hf_spike_fade*`: HF spike fade
- `lf_thrust*`: LF thrust + efficiency
- `vol_compress*`: Vol compress→release
- `skew_align*`: Skew-aligned SNR
- `fractal*`: High fractal-D mean-revert

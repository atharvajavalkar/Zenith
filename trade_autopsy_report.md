# Trade Autopsy Report (read-only)

- Window: last **14** days
- Closed trades scored: **120**
- Generated: `2026-07-20T17:35:44.498159+00:00`
- **No live changes applied** — suggestions are research-only.

## Desk patterns

| Tag | Count |
|---|---:|
| `manual_or_sync_close` | 112 |
| `off_prime_hour` | 38 |
| `soft_mc_entry` | 28 |
| `stretched_rr_loss` | 18 |
| `fast_tf_loss` | 15 |
| `bias_against` | 11 |
| `smc_weak` | 10 |
| `no_strong_flag` | 7 |
| `mid_mc_entry` | 5 |
| `news_against` | 3 |
| `large_htf_loss` | 1 |
| `clean_win_profile` | 1 |

## Per symbol

| Symbol | n | W | L | PnL $ | Top tags |
|---|---:|---:|---:|---:|---|
| ? | 17 | 3 | 14 | -259.58 | manual_or_sync_close(16), stretched_rr_loss(11), soft_mc_entry(8) |
| BTCUSD | 10 | 0 | 10 | -807.63 | manual_or_sync_close(10), fast_tf_loss(4), stretched_rr_loss(2) |
| ETHUSD | 11 | 0 | 11 | -58.48 | manual_or_sync_close(11), off_prime_hour(11), fast_tf_loss(5) |
| EURUSD | 23 | 0 | 22 | -1167.52 | manual_or_sync_close(17), no_strong_flag(6), fast_tf_loss(4) |
| GBPUSD | 25 | 2 | 11 | -467.1 | manual_or_sync_close(24), soft_mc_entry(14), off_prime_hour(14) |
| GOLD | 15 | 2 | 13 | -2260.08 | manual_or_sync_close(15), stretched_rr_loss(1) |
| NAS100 | 2 | 0 | 2 | -147.35 | manual_or_sync_close(2), stretched_rr_loss(2), off_prime_hour(2) |
| SILVER | 8 | 0 | 8 | -3610.0 | manual_or_sync_close(8) |
| USDCHF | 1 | 0 | 1 | -237.85 | manual_or_sync_close(1), soft_mc_entry(1), stretched_rr_loss(1) |
| USDJPY | 8 | 0 | 5 | -128.41 | manual_or_sync_close(8), mid_mc_entry(3), off_prime_hour(3) |

## Research suggestions (do not auto-apply)

- (58×) Process: avoid manual early exits when testing brain edge; let SL/TP decide.
- (7×) Research: session filter / reduce size off-prime for ETHUSD.
- (7×) Research: raise MC soft floor for ? (paper A/B only).
- (5×) Research: prefer 1:2 forced RR on ? vs stretched None.
- (4×) Research: raise MC soft floor for ETHUSD (paper A/B only).
- (2×) Research: raise MC soft floor for GBPUSD (paper A/B only).
- (1×) Research: raise MC soft floor for USDCHF (paper A/B only).
- (1×) Research: prefer 1:2 forced RR on NAS100 vs stretched 1:3.
- (1×) Research: raise MC soft floor for NAS100 (paper A/B only).
- (1×) Research: prefer 1:2 forced RR on BTCUSD vs stretched None.

## Recent trade autopsies

### #2715 USDCHF SELL · loss · $-237.85 · TF=4h · MC=39.15
- Tags: manual_or_sync_close, soft_mc_entry, stretched_rr_loss, off_prime_hour, news_against, smc_weak, large_htf_loss
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Entered with soft MC (39.1%).
- Planned RR≈3.0 and trade lost — stretch may have been optimistic.
- Entry hour_utc=1 was not marked prime.
- World-news bias was not aligned with the trade.
- SMC concept flag was false at entry.
- Large loss on higher TF (4h).
- Suggestion: _Research: raise MC soft floor for USDCHF (paper A/B only)._

### #2714 USDJPY SELL · flat · $-0.23 · TF=1m · MC=47.07
- Tags: manual_or_sync_close, mid_mc_entry, off_prime_hour
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Entry hour_utc=4 was not marked prime.

### #2713 USDJPY SELL · flat · $-0.2 · TF=3m · MC=43.51
- Tags: manual_or_sync_close, mid_mc_entry, off_prime_hour
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Entry hour_utc=4 was not marked prime.

### #2712 USDJPY SELL · flat · $0.24 · TF=5m · MC=42.98
- Tags: manual_or_sync_close, mid_mc_entry, off_prime_hour
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Entry hour_utc=4 was not marked prime.

### #2711 NAS100 BUY · loss · $-145.01 · TF=15m · MC=42.82
- Tags: manual_or_sync_close, mid_mc_entry, stretched_rr_loss, off_prime_hour, bias_against, smc_weak
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Planned RR≈3.0 and trade lost — stretch may have been optimistic.
- Entry hour_utc=2 was not marked prime.
- Desk bias not aligned.
- SMC concept flag was false at entry.
- Suggestion: _Research: prefer 1:2 forced RR on NAS100 vs stretched 1:3._

### #2710 NAS100 BUY · loss · $-2.34 · TF=15m · MC=38.61
- Tags: manual_or_sync_close, soft_mc_entry, stretched_rr_loss, off_prime_hour, bias_against, smc_weak
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Entered with soft MC (38.6%).
- Planned RR≈3.0 and trade lost — stretch may have been optimistic.
- Entry hour_utc=1 was not marked prime.
- Desk bias not aligned.
- SMC concept flag was false at entry.
- Suggestion: _Research: raise MC soft floor for NAS100 (paper A/B only)._

### #2709 ETHUSD BUY · loss · $-4.73 · TF=15m · MC=36.85
- Tags: manual_or_sync_close, soft_mc_entry, stretched_rr_loss, off_prime_hour
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Entered with soft MC (36.9%).
- Planned RR≈4.0 and trade lost — stretch may have been optimistic.
- Entry hour_utc=1 was not marked prime.
- Suggestion: _Research: raise MC soft floor for ETHUSD (paper A/B only)._

### #2707 BTCUSD SELL · loss · $-37.5 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2706 BTCUSD BUY · loss · $-97.54 · TF=None · MC=None
- Tags: manual_or_sync_close, stretched_rr_loss
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Planned RR≈4.0 and trade lost — stretch may have been optimistic.
- Suggestion: _Research: prefer 1:2 forced RR on BTCUSD vs stretched None._

### #2705 ETHUSD BUY · loss · $-0.68 · TF=5m · MC=40.16
- Tags: manual_or_sync_close, soft_mc_entry, off_prime_hour, smc_weak
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Entered with soft MC (40.2%).
- Entry hour_utc=21 was not marked prime.
- SMC concept flag was false at entry.
- Suggestion: _Research: raise MC soft floor for ETHUSD (paper A/B only)._

### #2704 ETHUSD BUY · loss · $-7.92 · TF=3m · MC=40.2
- Tags: manual_or_sync_close, soft_mc_entry, off_prime_hour, smc_weak, fast_tf_loss
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Entered with soft MC (40.2%).
- Entry hour_utc=21 was not marked prime.
- SMC concept flag was false at entry.
- Lost on fast TF (3m).
- Suggestion: _Research: raise MC soft floor for ETHUSD (paper A/B only)._

### #2703 ETHUSD BUY · loss · $-7.92 · TF=5m · MC=41.51
- Tags: manual_or_sync_close, soft_mc_entry, off_prime_hour, smc_weak
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Entered with soft MC (41.5%).
- Entry hour_utc=21 was not marked prime.
- SMC concept flag was false at entry.
- Suggestion: _Research: raise MC soft floor for ETHUSD (paper A/B only)._

### #2702 ETHUSD BUY · loss · $-3.68 · TF=1m · MC=49.14
- Tags: manual_or_sync_close, off_prime_hour, smc_weak, fast_tf_loss
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Entry hour_utc=20 was not marked prime.
- SMC concept flag was false at entry.
- Lost on fast TF (1m).
- Suggestion: _Research: session filter / reduce size off-prime for ETHUSD._

### #2701 ETHUSD BUY · loss · $-3.68 · TF=3m · MC=49.06
- Tags: manual_or_sync_close, off_prime_hour, fast_tf_loss
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Entry hour_utc=20 was not marked prime.
- Lost on fast TF (3m).
- Suggestion: _Research: session filter / reduce size off-prime for ETHUSD._

### #2700 ETHUSD BUY · loss · $-7.76 · TF=5m · MC=47.88
- Tags: manual_or_sync_close, mid_mc_entry, off_prime_hour
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Entry hour_utc=20 was not marked prime.
- Suggestion: _Research: session filter / reduce size off-prime for ETHUSD._

### #2699 ETHUSD BUY · loss · $-7.35 · TF=1m · MC=48.9
- Tags: manual_or_sync_close, off_prime_hour, fast_tf_loss
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Entry hour_utc=20 was not marked prime.
- Lost on fast TF (1m).
- Suggestion: _Research: session filter / reduce size off-prime for ETHUSD._

### #2698 ETHUSD BUY · loss · $-3.63 · TF=3m · MC=53.2
- Tags: manual_or_sync_close, off_prime_hour, fast_tf_loss
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Entry hour_utc=20 was not marked prime.
- Lost on fast TF (3m).
- Suggestion: _Research: session filter / reduce size off-prime for ETHUSD._

### #2697 ETHUSD BUY · loss · $-3.63 · TF=5m · MC=53.82
- Tags: manual_or_sync_close, off_prime_hour
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Entry hour_utc=20 was not marked prime.
- Suggestion: _Research: session filter / reduce size off-prime for ETHUSD._

### #2696 ETHUSD BUY · loss · $-7.5 · TF=5m · MC=49.48
- Tags: manual_or_sync_close, off_prime_hour
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Entry hour_utc=20 was not marked prime.
- Suggestion: _Research: session filter / reduce size off-prime for ETHUSD._

### #2695 SILVER SELL · loss · $-370.0 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2694 SILVER SELL · loss · $-370.0 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2693 GBPUSD SELL · loss · $-90.0 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2691 GBPUSD BUY · loss · $-58.0 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2692 GBPUSD BUY · loss · $-58.0 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2690 GBPUSD BUY · loss · $-60.0 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2688 EURUSD SELL · loss · $-48.0 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2689 GBPUSD SELL · loss · $-62.0 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2687 GOLD SELL · loss · $-69.0 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2686 SILVER SELL · loss · $-410.0 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2685 GOLD SELL · loss · $-34.2 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2684 GOLD SELL · loss · $-91.8 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2683 GOLD SELL · win · $34.2 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Keep — no change suggested._

### #2682 SILVER SELL · loss · $-195.0 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2679 GBPUSD SELL · loss · $-58.0 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2681 GOLD BUY · win · $62.4 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Keep — no change suggested._

### #2680 EURUSD SELL · loss · $-48.0 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2678 USDJPY BUY · loss · $-39.45 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2677 USDJPY BUY · loss · $-24.65 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2676 EURUSD SELL · loss · $-40.0 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

### #2675 GBPUSD SELL · loss · $-48.0 · TF=None · MC=None
- Tags: manual_or_sync_close
- Exit was manual/broker-sync — not a clean SL/TP autopsy.
- Suggestion: _Process: avoid manual early exits when testing brain edge; let SL/TP decide._

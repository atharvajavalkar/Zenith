# skills.md

# XAUUSD Physics-Driven Trading Algorithm

## Project Overview

This project develops a quantitative intraday trading algorithm for XAUUSD (Gold/USD) that combines traditional technical analysis with engineering principles from heat transfer, fluid mechanics, stochastic processes, and probabilistic simulation.

The objective is **not to predict the exact next candle**, but to estimate the probability that a predefined Take Profit (TP) will be reached before the Stop Loss (SL) under the current market state.

---

# Core Philosophy

The algorithm treats the market as a dynamic transport system rather than a sequence of isolated indicators.

Traditional indicators are used only as measurements of the current state.

The decision engine is driven by:

* Market Structure
* Physics State Estimation
* Probabilistic State Transitions
* Monte Carlo Simulation
* Expected Value Optimization

---

# Trading Style

Asset:

* XAUUSD

Execution:

* 5 Minute candles

Higher Timeframe Context:

* 1 Hour
* 4 Hour

Supply / Demand:

* 15 Minute
* 1 Hour

Target Holding Period:

* Approximately 20–60 minutes.

No long-term swing trading.

---

# Market Structure

Uses:

* Supply Zones
* Demand Zones
* Liquidity Sweeps
* Pullbacks
* Rejections
* Breakouts
* Retests

A 20% liquidity buffer is maintained around every Supply and Demand zone.

This buffer is designed to reduce premature stop-outs caused by liquidity grabs.

---

# Physics Layer

The market is modelled using transport phenomena.

## Velocity

Approximation:

EMA9 − EMA21

Represents directional flow.

---

## Acceleration

Discrete Laplacian

Second derivative of price.

Represents acceleration or deceleration of momentum.

---

## Momentum

Derived from

Velocity × Trend Strength

Used to estimate persistence.

---

## Turbulence

Measured using:

ATR Ratio

High ATR Ratio

→ Turbulent Flow

Low ATR Ratio

→ Laminar Flow

---

## Graetz Analogy

Inspired by the heat transfer Graetz Number.

Instead of thermal transport,

the algorithm estimates

Momentum Transport Efficiency.

Conceptually:

High Graetz

↓

Momentum likely survives

↓

Higher TP probability

Low Graetz

↓

Momentum dissipates

↓

Lower TP probability

---

## Boundary Layer

Supply and Demand Zones are treated as boundaries.

The 20% liquidity buffer acts as the boundary layer.

Trades are intentionally positioned outside the expected liquidity sweep region.

---

# Market State Engine

Current market state is classified into:

* LAMINAR_BULL
* LAMINAR_BEAR
* TRANSITION_BULL
* TRANSITION_BEAR
* TURBULENT_BULL
* TURBULENT_BEAR
* CHOP

The state is estimated using:

* EMA Alignment
* Laplacian
* ATR Ratio
* Momentum
* Trend Persistence

---

# Markov Chain

The algorithm models market evolution as a finite-state Markov process.

Instead of predicting prices directly,

it predicts

State → State transitions.

Example:

LAMINAR_BULL

↓

LAMINAR_BULL

↓

TRANSITION

↓

CHOP

↓

TURBULENT_BEAR

Transition probabilities are intended to be learned from historical data.

---

# Monte Carlo Simulation

Monte Carlo is the final validation layer.

The simulation does not generate random candles.

Instead,

it simulates future market paths conditioned on:

* Current Physics State
* Estimated Drift
* Volatility
* Markov Transition Probabilities

Each simulation evaluates:

* TP reached first
* SL reached first
* Neither

The displayed Success Percentage represents

P(TP before SL | Current Market State)

not an arbitrary confidence score.

---

# Risk Management

Dynamic Risk Reward

Possible values:

* 1:2
* 1:3
* 1:4

The algorithm evaluates every candidate RR.

Expected Value (EV) is computed.

The RR with the highest positive EV is selected.

---

# Expected Value

Rather than maximizing raw win rate,

the objective is maximizing

Expected Return.

EV is evaluated using:

Probability of TP

Probability of SL

Reward

Risk

The highest EV configuration is chosen.

---

# Sessions

Primary Trading Window

London

New York

Asian session is generally avoided due to lower liquidity.

---

# News Filters

Future integration:

* Economic Calendar
* USD Index (DXY)
* Oil (WTI / Brent)
* Treasury Yields
* CPI
* PPI
* NFP
* FOMC
* Powell Speeches
* Major Geopolitical Events

These variables act as exogenous inputs to the market state.

---

# Performance Measurement

The objective is not to engineer a fixed 67% win rate.

Instead,

the algorithm measures:

* Win Rate
* Profit Factor
* Expectancy
* Drawdown
* Sharpe Ratio
* Monte Carlo Stability
* Calibration Error

The probability output should converge toward observed historical performance through repeated validation.

---

# Long-Term Roadmap

Future improvements:

* Data-driven Markov transition matrices.
* Bayesian updating of transition probabilities.
* Adaptive Monte Carlo drift and volatility based on realized market conditions.
* Regime detection using historical state persistence.
* Walk-forward optimization.
* Paper trading with automatic journaling.
* Web application dashboard.
* Historical replay and backtesting engine.
* Live analytics and probability calibration.

---

# Guiding Principle

The algorithm is designed to estimate probability, not certainty.

Every trade is treated as a stochastic outcome conditioned on the current market state.

The goal is to maximize long-term expectancy through disciplined probability estimation rather than deterministic prediction.

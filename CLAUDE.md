# CLAUDE.md

Options research. Learning-phase code — clarity beats cleverness, and every
formula in here has to earn trust numerically rather than by assertion.

## Rules

**Stdlib-only where possible.** No numpy, no scipy, no pandas in the math. If a
closed form exists, write it out — `math.erf` gives you the normal CDF, and that
is the only special function this domain needs. Reaching for a dependency to
avoid writing ten lines of arithmetic is how the math stops being legible.

The documented exception is **presentation**: plotting and data-fetching may use
third-party libraries, but they import *inside the function that needs them*, so
the pricing core stays importable with zero dependencies. `plot_greeks()` in
`black_scholes.py` is the pattern to copy.

**Every pricing function needs a finite-difference test verifying it against the
analytic form.** Not a regression test against a hardcoded number someone once
printed — an independent numerical check that the closed form is the derivative
it claims to be:

```python
fd = (f(x + h) - f(x - h)) / (2 * h)   # central difference, h ~ 1e-4 to 1e-6
assert abs(analytic - fd) < tol
```

This is the rule that matters. It costs three lines and it is the difference
between code that looks right and code that is right. It caught nothing the day
it was written, because the code was correct — that is not an argument against
it. Write the test before you trust the formula, not after something breaks.

Test both branches (call *and* put) and any structural identity the model
implies — put-call parity, `theta = -1/2 * gamma * S^2 * sigma^2` at r=q=0. An
identity that holds to 1e-14 is worth more than a hundred assertions.

**Test the edges, not just the happy path.** A verification suite that only ever
prices a near-ATM option with a valid input proves the near-ATM valid-input case.
Solvers get inputs outside their bracket; deep-OTM options have vega near zero
and break tolerance-in-price-space convergence. Go there deliberately.

## Conventions

- `T` is in **years**. 1 calendar day = 1/365, 1 trading day = 1/252.
- `r`, `q`, `sigma` are decimals, continuously compounded (`0.16` == 16 vol).
- Greeks are returned **broker-scaled**: vega and rho per 1 point (÷100), theta
  per calendar day (÷365). Raw partials are per full unit. Scale at the boundary,
  never mid-calculation, and say so in the docstring.
- Charts: one chart per idea, colors from the first three slots of the validated
  palette in `black_scholes.py:_INK`. Don't substitute arbitrary hues.

## What's here

- `black_scholes.py` — European pricer, five greeks, bisection implied vol,
  `verify()` suite, `plot_greeks()`. Run `python black_scholes.py` for the
  verification output, `--plot` for the charts.
  - Both previously known `implied_vol()` bugs are fixed. It now checks the
    bracket endpoints, converges on the width of the **vol** bracket rather
    than on price error, and raises `VolNotIdentifiable` when vega is too
    small for the quoted price to pin the vol down — the exception carries
    `.sigma` and `.vol_error` for callers who want the root anyway. Tests
    `[6]` and `[7]` cover a 120-point strike × expiry × vol grid and the
    refusal cases.

- `vrp_data.py` — SPY (Yahoo) and VIX (FRED) joined daily and cached to
  `spy_vix_daily.csv`, plus `forward_rv()`. The forward alignment is the whole
  point: VIX today forecasts the *next* 30 days, so it must be compared to vol
  that hasn't happened yet. `_verify_alignment()` pins both sides of that
  boundary. Run `python vrp_data.py`; `build(refresh=True)` re-fetches.

- `vrp_study.py` — the VRP cut by VIX regime. Table via `python vrp_study.py`,
  charts via `--plot`. Reuses `black_scholes._INK` for the palette.
  - **The overlap rule.** `forward_rv` looks 21 days ahead, so adjacent rows
    share 20 of 21 days and 5,009 rows carry ~239 independent observations.
    Means are unbiased under overlap; standard errors are not. Every error bar
    here divides by `sqrt(n_indep)` — the count of non-overlapping rows — not
    `sqrt(n)`. Test `[3]` proves the correction recovers the true SE on a
    block-constant series where the answer is known, and that the naive
    version is 4.6x too tight (`sqrt(21)`). Any new statistic on this data
    owes the same treatment.
  - Tail figures cluster: the worst day in four of six buckets is Feb 2020,
    one repricing walking up through every regime. Size tails off events, not
    off row counts. `main()` prints this check automatically.

- `put_write_backtest.py` — the premium turned into a trade: a 25-delta SPY
  put sold every 21 trading days and held to expiry, a 25d/10d put credit
  spread, and SPY buy-and-hold on the same cash-secured capital. Only the entry
  credit is modelled; the payoff is measured on real closes, so the tails owe
  nothing to the model. Tables via `python put_write_backtest.py`, charts
  `--plot`, and `--calibrate` re-measures the vol model from CBOE's free chain.
  - **The vol model is two numbers and one of them was measured.**
    `sigma(K) = atm_ratio * VIX + skew * ln(S/K)`. The 2026-09-03 chain put
    30-day ATM IV at 0.87x VIX and the skew at 0.9 pts per 1% OTM, so reading
    VIX as the put's own IV is ~1.8 points too rich at the money. Three
    settings run side by side; the naked put lands at 4.3-8.3% CAGR across
    them and that width *is* the finding until real option prices exist.
  - **Stale marks smooth the ladder.** The 21-sleeve ladder settles one sleeve
    a day, so a crash leaks across two 21-day periods and the ladder's Sharpe
    reads ~20% higher than any single track. vol and Sharpe come from the
    single-entry-day tracks (median and range). Any equity curve built from
    trades settled at expiry owes the same care.
  - **It has an answer key.** `--validate` runs the 50-delta twin beside
    CBOE's real PUT index over the same twenty years: PUT 7.08%/yr with
    T-bill interest, synthetic 7.32% calibrated and 11.44% flat-VIX without.
    The calibrated model sits where an interest-free model should; the naive
    one is four points a year too generous. Run it after touching the vol
    model.

- `short_vol_sleeve.py` — a **negative result, kept on purpose.** The only
  way a $500 long-only account gets short vol is an inverse-VIX ETF, so this
  tests SVXY (made a consistent -0.5x series) held only when yesterday's
  VIX/VIX3M < 1. 2011-2026: 13.4% CAGR at 31% vol, maxDD -58%, against SPY's
  15.7% at 17% and -34%. The timing beats a block-shuffled signal 72% of the
  time on Sharpe, which is nothing; post-2018 it is worse than always-in; and
  the gap of Feb 2018 is invisible to any curve filter. `--signal` still
  prints the day's position so the rule can be watched, not traded. Tests
  pin no-lookahead (lag 1 reads t-2), one switch cost per change, and the
  -1x -> -0.5x transform against the Feb 2018 event itself.

- `put_manage.py` — managed exits on the same trades: close at a profit
  target and optionally redeploy the capital. Exits pay one half-spread per
  leg (one for the naked put, two for the vertical), and a target that never
  triggers reproduces hold-to-expiry exactly, which is the test that pins the
  engine. Result: taking 50% early makes the 25d/10d put spread worse
  (-0.17% idle, 0.39% redeployed, against 1.42% / 1.84% held), and lifts the
  naked put from 5.34% to 7.37% only through redeployment (2.60% without).

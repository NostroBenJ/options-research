# options-research

Options pricing and variance-risk-premium research in plain Python, standard library only.

I wrote this while learning options from scratch. The rule the whole repo follows: **a formula
isn't trusted because it looks right, it's trusted because a numerical check says so.** Every
pricing function is verified against finite differences, both calls and puts, plus the identities
the model implies (put-call parity, and theta = -1/2 · gamma · S² · sigma² at r = q = 0).

## What's here

| file | what it does | how it's checked |
|---|---|---|
| `black_scholes.py` | European pricer, five greeks, implied vol by bisection | finite differences against every analytic greek; parity; a 120-point strike × expiry × vol grid for implied vol; refuses to return a vol when vega is too small to pin it down |
| `vrp_data.py` | SPY and VIX daily, joined; forward realised vol | tests pin both sides of the forward boundary, so VIX today is only ever compared with vol that hasn't happened yet |
| `vrp_study.py` | the variance risk premium, cut by VIX regime | standard errors corrected for overlapping windows; a test proves the naive version is 4.6x too tight (√21) |
| `put_write_backtest.py` | a 25-delta SPY put sold every 21 trading days, a 25/10-delta put spread, and SPY on the same capital | `--validate` runs the model beside CBOE's real PUT index; `--calibrate` re-measures the vol model from CBOE's free chain |
| `put_manage.py` | managed exits: take profit early, optionally redeploy | a profit target that never triggers must reproduce hold-to-expiry exactly |
| `short_vol_sleeve.py` | an inverse-VIX ETF timing rule. **A negative result, kept on purpose** | no lookahead, one switch cost per change, and the leverage transform checked against the Feb 2018 event itself |

## What it found

- **The put-write model has an answer key, and passes.** Over twenty years CBOE's PUT index
  returned 7.08%/yr including T-bill interest on collateral. The calibrated model returns 7.32%
  without interest, which is where an interest-free model should sit. The naive model, which reads
  VIX as the put's own implied vol, returns 11.44%: **four points a year too generous.**
- **The vol model is two numbers, and one was measured:** 30-day ATM implied vol is 0.87x VIX, with
  0.9 vol points of skew per 1% out of the money.
- **Selling puts works as a Sharpe edge, not a return edge.** The naked 25-delta put lands at
  4.3–8.3% CAGR across three vol settings, against SPY's 11.2%.
- **Credit spreads are worse.** The 25/10-delta put spread returns 1.76% CAGR with the calibrated
  model. Managing it at 50% of max profit makes it worse (−0.17%, or 0.39% with the capital
  redeployed). The naked put improves from 5.34% to 7.37% only through redeployment.
- **The inverse-VIX sleeve doesn't earn its risk:** 13.4% CAGR at 31% vol and a −58% drawdown, against
  SPY's 15.7% at 17% and −34%. Its timing beats a block-shuffled signal only 72% of the time.

## Running it

Python 3.11+. No dependencies for the math; `--plot` needs matplotlib.

```
python black_scholes.py          # verification suite
python vrp_data.py               # downloads and caches SPY + VIX, checks alignment
python vrp_study.py              # the VRP by VIX regime   (--plot for charts)
python put_write_backtest.py     # tables   (--plot, --validate, --calibrate)
python put_manage.py             # managed exits
python short_vol_sleeve.py       # the negative result   (--signal prints today's position)
```

## Data

No market data is in this repo. The scripts download it at runtime from Yahoo Finance, FRED and
CBOE's public endpoints and cache it locally (gitignored).

Conventions (units, greek scaling, the testing rules) are in [`CLAUDE.md`](CLAUDE.md). Research code
for learning, not investment advice.

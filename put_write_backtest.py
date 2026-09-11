"""
put_write_backtest.py -- does selling SPY puts actually collect the VRP?

Standard library only (plotting excepted; it imports inside the function).

vrp_study.py established that VIX has sat ~3.6 vol points above the realized
vol that followed it, in every regime. That is a statement about variance
swaps. Nobody in a retail account sells variance swaps; they sell puts. This
module asks the trader's version of the question: if you had sold a 30-day
out-of-the-money SPY put every month since 2006 and let it expire, what would
you have made, what would it have cost you in 2008 and 2020, and does a long
put underneath (a credit spread) buy protection worth its price.

WHAT IS MEASURED AND WHAT IS MODELLED
--------------------------------------
There is no historical option-price data here, only SPY closes and VIX. The
design leans on that constraint instead of pretending it away:

  * Every trade is held to expiry. The payoff is therefore *measured*: it is
    max(K - S_T, 0) on actual SPY closes and needs no model at all.
  * The only modelled quantity in a trade's P&L is the entry credit, priced
    with Black-Scholes at a volatility built from that day's VIX print, less
    a half-spread haircut. Get that number 10% wrong and the credit is 10%
    wrong -- the tail losses are untouched.
  * A second, model-dependent equity curve marks open positions daily at the
    current VIX so that intra-trade drawdowns (mid-March 2020, before the
    March expiry) are visible. It is labelled as such and feeds only the
    "mtm" drawdown column and the equity chart.

THE VOL MODEL, AND WHERE ITS NUMBERS COME FROM
----------------------------------------------
VIX is a 30-day variance-swap vol, not the implied vol of a 25-delta put.
Two things separate them and they pull opposite ways: the variance swap sits
*above* ATM implied (it is a skew-weighted average), and the put skew sits
above ATM at 25-delta. The implied vol used at strike K is therefore

    sigma(K) = atm_ratio * VIX/100 + skew * ln(S/K)

with two parameters. skew = 1.0 means one vol point per 1% out of the money.
Three settings are run side by side so the reader can see what the answer
costs in assumptions:

    flat VIX      atm_ratio 1.00  skew 0.0   the naive read of VIX as the put's IV
    VIX + skew    atm_ratio 1.00  skew 1.0   skew on top of VIX; over-credits the seller
    calibrated    atm_ratio 0.87  skew 0.9   measured, see below

The calibrated numbers are one real data point, not a study: CBOE's free
delayed SPY chain on 2026-09-03 (SPY 772.80, VIX 14.28, the 29-day expiry)
showed ATM put IV 12.47 -- a ratio of 0.873 to VIX -- and a least-squares
slope of 0.90 vol points per 1% OTM between 1% and 8% out. At that shape the
25-delta put trades *at* VIX and the 10-delta put about 3 points above it.
`python put_write_backtest.py --calibrate` re-measures it from today's chain
so the parameter can be watched rather than believed. Whether the ratio is
stable across regimes is exactly the question the quantdesk chain recorder
exists to answer with time.

PRICES ARE TOTAL-RETURN
-----------------------
`spy` in the cache is Yahoo's dividend-adjusted close. That is the right
series for the buy-and-hold benchmark, and it makes q = 0 the *consistent*
pricing choice: a total-return price has no dividend leakage for the forward
to carry. r is held at 0 as well (FRED's bill series was unreachable when
this was written); on a 30-day put the rate moves the credit by cents. But
note the real CBOE PUT index also earns bill interest on its collateral,
which this does not credit -- the put-write figures here are understated by
roughly the period's average bill yield, on the order of 1-1.5%/yr.

SIZING
------
Returns are on the short strike as collateral -- the cash-secured convention
-- for the naked put *and* the spread, so both curves sit on the same capital
and the spread reads as "the put-write with insurance bought." Return on the
spread's own max risk is reported separately; it looks spectacular, and it is
the number that gets people hurt.

OVERLAP, AND STALE MARKS
------------------------
A trade enters every day in the pooled sample, so trades overlap exactly as
in vrp_study.py: ~5,000 trades, ~239 independent. Pooled means are unbiased;
every standard error divides by sqrt(n_indep). The 21 non-overlapping
"phases" (enter on day p, p+21, ...) are each an honest single track record.

vol and Sharpe are reported from those single tracks (median and range over
the 21), NOT from the 21-sleeve ladder. The ladder settles one sleeve a day,
so a crash's loss leaks across two consecutive 21-day periods and the ladder
looks ~20% smoother than any track you could actually have run. Same
mechanism as appraisal smoothing in private assets; same fix -- measure on
something that is marked honestly. The ladder is kept for CAGR (where it is
just the average of the tracks) and for the drawdown series.

Run it:  python put_write_backtest.py               tables
         python put_write_backtest.py --plot        charts (--dark for the dark palette)
         python put_write_backtest.py --calibrate   re-measure the vol model from CBOE
         python put_write_backtest.py --validate    ATM twin vs CBOE's real PUT index

THE ANSWER KEY
--------------
CBOE publishes the PUT index -- the ATM SPX put sold monthly against T-bill
collateral -- daily since 1991. `--validate` runs the 50-delta twin of this
backtest under each vol model beside it over the same dates. On 2026-09-04:
PUT 7.08%/yr with interest; synthetic ATM 7.32% calibrated, 11.44% flat VIX,
both without interest, scored over the index's own dates. The calibrated model sits where a model that earns no
interest should sit; the flat-VIX one is four points a year too generous.
That is independent, twenty-year confirmation that VIX is not the put's IV.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from datetime import date

import vrp_data
from black_scholes import bs_greeks, bs_price, d1_d2, norm_cdf, norm_pdf
from vrp_study import DEFAULT_EDGES, independent_count, mean, quantile, stdev

HOLD = 21                                 # trading days held; = the VRP forward window
TRADING_DAYS = 252
PERIODS_PER_YEAR = TRADING_DAYS / HOLD    # 12
SIGMA_FLOOR = 0.02                        # the skew line may not drive an ITM vol to zero
STRIKE_STEP = 1.0                         # SPY has listed $1 strikes for the whole sample


# ---------------------------------------------------------------------------
# The vol model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VolModel:
    """sigma(K) = atm_ratio * VIX/100 + skew * ln(S/K), floored at SIGMA_FLOOR."""
    name: str
    atm_ratio: float = 1.0
    skew: float = 0.0          # vol per unit log-moneyness; 1.0 == 1 point per 1% OTM

    def sigma(self, vix: float, S: float, K: float) -> float:
        return max(SIGMA_FLOOR, self.atm_ratio * vix / 100.0 + self.skew * math.log(S / K))


FLAT = VolModel("flat VIX", 1.0, 0.0)
SKEWED = VolModel("VIX + skew 1", 1.0, 1.0)
CALIBRATED = VolModel("calibrated", 0.87, 0.90)     # CBOE SPY chain, 2026-09-03
MODELS = (FLAT, SKEWED, CALIBRATED)


def put_mid(S: float, K: float, T: float, vix: float, vm: VolModel,
            r: float = 0.0) -> float:
    """Model mid of the put at strike K. q = 0: prices are total-return."""
    return bs_price(S, K, T, r, vm.sigma(vix, S, K), 0.0, "put")


def put_delta(S: float, K: float, T: float, vix: float, vm: VolModel,
              r: float = 0.0) -> float:
    return bs_greeks(S, K, T, r, vm.sigma(vix, S, K), 0.0, "put").delta


def dput_dK(S: float, K: float, T: float, vix: float, vm: VolModel,
            r: float = 0.0) -> float:
    """
    Total derivative of the modelled put price with respect to strike.

    Two terms: the Black-Scholes dual delta exp(-rT) N(-d2), which is the
    whole answer at flat vol, plus vega times the slope of the skew line,
    d sigma / dK = -skew / K. The second term is new here and is what the
    finite-difference check in _verify() exists to prove.
    """
    sigma = vm.sigma(vix, S, K)
    d1, d2 = d1_d2(S, K, T, r, sigma)
    dual_delta = math.exp(-r * T) * norm_cdf(-d2)
    vega_raw = S * norm_pdf(d1) * math.sqrt(T)          # per full unit of vol
    dsigma_dK = -vm.skew / K if sigma > SIGMA_FLOOR else 0.0
    return dual_delta + vega_raw * dsigma_dK


def strike_for_delta(S: float, T: float, vix: float, target: float,
                     vm: VolModel, r: float = 0.0, tol: float = 1e-9) -> float:
    """
    The strike whose put delta is -target (target 0.25 -> the 25-delta put),
    with the skew applied self-consistently: a further-OTM strike gets a
    higher vol, which pulls its delta back toward the money.

    Bisection on K. g(K) = delta(K) + target is negative near the money
    (delta ~ -0.5) and positive far below it (delta ~ 0), so the root is
    bracketed on [0.3 S, 1.5 S] for any target in (0, 1) at sane vols.
    """
    lo, hi = 0.3 * S, 1.5 * S

    def g(K: float) -> float:
        return put_delta(S, K, T, vix, vm, r) + target

    if g(lo) <= 0.0 or g(hi) >= 0.0:
        raise ValueError(f"delta {target} not bracketed on [{lo:.2f}, {hi:.2f}]")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0.0:      # delta too small in magnitude: strike too low
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def _round_strike(K: float, step: float = STRIKE_STEP) -> float:
    return round(K / step) * step


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    i: int                 # entry row
    j: int                 # expiry row (i + HOLD)
    entry: str
    expiry: str
    S0: float
    ST: float
    vix: float             # VIX on the entry day
    T: float               # years to expiry, from the calendar dates
    K_short: float
    K_long: float | None   # None for the naked put
    credit: float          # net of half-spreads, per share
    payoff: float          # owed at expiry, per share
    vm: VolModel

    @property
    def pnl(self) -> float:
        return self.credit - self.payoff

    @property
    def ret(self) -> float:
        """Return on the short strike as collateral (cash-secured)."""
        return self.pnl / self.K_short

    @property
    def max_risk(self) -> float:
        width = self.K_short - (self.K_long if self.K_long is not None else 0.0)
        return width - self.credit

    @property
    def ret_on_risk(self) -> float:
        return self.pnl / self.max_risk

    @property
    def spy_ret(self) -> float:
        return self.ST / self.S0 - 1.0


def build_trades(rows: list[dict], short_delta: float = 0.25,
                 long_delta: float | None = None, vm: VolModel = CALIBRATED,
                 half_spread: float = 0.05, hold: int = HOLD,
                 r: float = 0.0) -> list[Trade]:
    """
    One trade entered on every row that has `hold` rows of future.

    The trade at index k enters on row k -- callers rely on that alignment.
    `half_spread` is paid per leg at entry (sell below mid, buy above); the
    expiry settlement costs nothing because the position is simply allowed to
    expire. A long leg that rounds onto or above the short strike is pushed
    one step lower so the spread always has width.
    """
    ords = [date.fromisoformat(r_["date"]).toordinal() for r_ in rows]
    trades: list[Trade] = []
    for i in range(len(rows) - hold):
        j = i + hold
        S0, ST, vix = rows[i]["spy"], rows[j]["spy"], rows[i]["vix"]
        T = (ords[j] - ords[i]) / 365.0
        Ks = _round_strike(strike_for_delta(S0, T, vix, short_delta, vm, r))
        credit = put_mid(S0, Ks, T, vix, vm, r) - half_spread
        payoff = max(Ks - ST, 0.0)
        Kl = None
        if long_delta is not None:
            Kl = _round_strike(strike_for_delta(S0, T, vix, long_delta, vm, r))
            if Kl >= Ks:
                Kl = Ks - STRIKE_STEP
            credit -= put_mid(S0, Kl, T, vix, vm, r) + half_spread
            payoff -= max(Kl - ST, 0.0)
        trades.append(Trade(i, j, rows[i]["date"], rows[j]["date"], S0, ST, vix,
                            T, Ks, Kl, credit, payoff, vm))
    return trades


def unrealized(t: Trade, row: dict, ord_now: int, ord_exp: int) -> float:
    """
    Return on collateral if the open structure were closed at model mid now.
    Model-dependent by construction -- this is the one place the backtest
    prices something other than an entry.
    """
    T = (ord_exp - ord_now) / 365.0
    if T <= 0.0:
        return t.ret
    S, vix = row["spy"], row["vix"]
    value = put_mid(S, t.K_short, T, vix, t.vm)
    if t.K_long is not None:
        value -= put_mid(S, t.K_long, T, vix, t.vm)
    return (t.credit - value) / t.K_short


# ---------------------------------------------------------------------------
# Equity curves
# ---------------------------------------------------------------------------

def settled_equity(trades: list[Trade], phase: int,
                   hold: int = HOLD) -> list[tuple[int, float]]:
    """(row, equity) after each expiry for one phase, starting at 1.0."""
    eq = 1.0
    out = [(trades[phase].i, eq)]
    for k in range(phase, len(trades), hold):
        eq *= 1.0 + trades[k].ret
        out.append((trades[k].j, eq))
    return out


def ladder_equity(trades: list[Trade], rows: list[dict], hold: int = HOLD,
                  mtm: bool = False) -> list[float]:
    """
    Equal capital in each of the `hold` phases, each compounding on its own;
    the ladder is their mean on every row. With mtm=False a sleeve's equity
    moves only when a trade settles, so the curve is a step function and
    understates any drawdown that reversed before an expiry. With mtm=True
    the open trade in each sleeve is marked at model mid every day.
    """
    n = len(rows)
    ords = [date.fromisoformat(r_["date"]).toordinal() for r_ in rows]
    total = [0.0] * n
    for p in range(hold):
        k, eq = p, 1.0
        for t in range(n):
            if k < len(trades) and trades[k].j == t:
                eq *= 1.0 + trades[k].ret
                k += hold
            cur = trades[k] if k < len(trades) else None
            if mtm and cur is not None and cur.i <= t < cur.j:
                total[t] += eq * (1.0 + unrealized(cur, rows[t], ords[t], ords[cur.j]))
            else:
                total[t] += eq
    return [x / hold for x in total]


# ---------------------------------------------------------------------------
# Statistics, written out
# ---------------------------------------------------------------------------

def cagr(e0: float, e1: float, years: float) -> float:
    return (e1 / e0) ** (1.0 / years) - 1.0


def max_drawdown(series: list[float]) -> float:
    """Worst peak-to-trough, as a negative fraction. 0.0 if never below a peak."""
    peak, worst = -math.inf, 0.0
    for x in series:
        peak = max(peak, x)
        worst = min(worst, x / peak - 1.0)
    return worst


def period_returns(series: list[float], step: int) -> list[float]:
    return [series[t] / series[t - step] - 1.0 for t in range(step, len(series), step)]


def ann_vol(rets: list[float]) -> float:
    return stdev(rets) * math.sqrt(PERIODS_PER_YEAR)


def sharpe(rets: list[float]) -> float:
    """Annualized mean over sd of per-period returns. r = 0 throughout, so no excess."""
    return mean(rets) / stdev(rets) * math.sqrt(PERIODS_PER_YEAR)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


@dataclass
class Summary:
    name: str
    cagr: float                              # 21-sleeve ladder = mean of the tracks
    cagr_range: tuple[float, float]          # min, max single track
    vol: float                               # median single track
    sharpe: float                            # median single track
    sharpe_range: tuple[float, float]
    dd_settled: float                        # ladder, settled at expiry
    dd_mtm: float                            # ladder, marked daily at model mid
    phase_dd: float                          # worst settled drawdown of any track
    win: float
    worst_ret: float
    worst_on: str
    n: int


def summarize(name: str, trades: list[Trade], rows: list[dict],
              hold: int = HOLD) -> Summary:
    years = (date.fromisoformat(rows[-1]["date"]).toordinal()
             - date.fromisoformat(rows[0]["date"]).toordinal()) / 365.25
    settled = ladder_equity(trades, rows, hold, mtm=False)
    marked = ladder_equity(trades, rows, hold, mtm=True)

    cagrs, vols, sharpes, dds = [], [], [], []
    for p in range(hold):
        track = trades[p::hold]
        rets = [t.ret for t in track]
        pts = settled_equity(trades, p, hold)
        cagrs.append(cagr(1.0, pts[-1][1], years))
        vols.append(ann_vol(rets))
        sharpes.append(sharpe(rets))
        dds.append(max_drawdown([e for _, e in pts]))

    worst = min(trades, key=lambda t: t.ret)
    return Summary(
        name=name,
        cagr=cagr(settled[0], settled[-1], years),
        cagr_range=(min(cagrs), max(cagrs)),
        vol=_median(vols),
        sharpe=_median(sharpes),
        sharpe_range=(min(sharpes), max(sharpes)),
        dd_settled=max_drawdown(settled),
        dd_mtm=max_drawdown(marked),
        phase_dd=min(dds),
        win=sum(t.pnl > 0 for t in trades) / len(trades),
        worst_ret=worst.ret,
        worst_on=worst.entry,
        n=len(trades),
    )


def spy_summary(rows: list[dict], hold: int = HOLD) -> Summary:
    years = (date.fromisoformat(rows[-1]["date"]).toordinal()
             - date.fromisoformat(rows[0]["date"]).toordinal()) / 365.25
    eq = [r_["spy"] / rows[0]["spy"] for r_ in rows]
    rets = period_returns(eq, hold)
    periods = [(rows[t + hold]["spy"] / rows[t]["spy"] - 1.0, rows[t]["date"])
               for t in range(len(rows) - hold)]
    w = min(periods)
    c, s = cagr(eq[0], eq[-1], years), sharpe(rets)
    return Summary("SPY buy & hold", c, (c, c), ann_vol(rets), s, (s, s),
                   max_drawdown(eq), max_drawdown(eq), max_drawdown(eq),
                   sum(p > 0 for p, _ in periods) / len(periods),
                   w[0], w[1], len(periods))


@dataclass
class Regime:
    label: str
    n: int
    n_indep: int
    mean_credit: float     # entry credit as % of strike
    mean_ret: float        # % on collateral
    se: float
    win: float
    p05: float
    worst: float
    worst_on: str


def regime_table(trades: list[Trade], edges=DEFAULT_EDGES,
                 hold: int = HOLD) -> list[Regime]:
    """Trade return by VIX at entry. Pooled means, sqrt(n_indep) errors."""
    out = []
    for lo, hi in zip(edges, edges[1:]):
        sel = [t for t in trades if lo <= t.vix < hi]
        label = f"{lo:g}-{hi:g}" if math.isfinite(hi) else f"{lo:g}+"
        if not sel:
            out.append(Regime(label, 0, 0, *([math.nan] * 6), ""))
            continue
        rets = [100.0 * t.ret for t in sel]
        n_ind = independent_count([t.i for t in sel], hold)
        se = stdev(rets) / math.sqrt(n_ind) if n_ind >= 2 else math.nan
        w = min(sel, key=lambda t: t.ret)
        out.append(Regime(label, len(sel), n_ind,
                          mean([100.0 * t.credit / t.K_short for t in sel]),
                          mean(rets), se, sum(x > 0 for x in rets) / len(rets),
                          quantile(rets, 0.05), 100.0 * w.ret, w.entry))
    return out


# ---------------------------------------------------------------------------
# Calibration: one real data point for the vol model
# ---------------------------------------------------------------------------

def measure_skew_today(r: float = 0.04, q: float = 0.012) -> dict:
    """
    Pull CBOE's free delayed SPY chain, invert mids to implied vols on the
    expiry nearest 30 days, and report the shape VIX is being asked to stand
    in for: ATM IV / VIX, the skew slope in points per 1% OTM, and where the
    25- and 10-delta puts sit relative to VIX. Stdlib only (urllib + json).

    The session date is taken from the feed's timestamp; a pull before 09:00
    on that clock is the previous session's close, so the date is rolled back
    a day -- otherwise every DTE is one too many, which is 3% of T here and
    33% on a 3-DTE option.
    """
    import json
    import urllib.request
    from datetime import timedelta

    from black_scholes import implied_vol

    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/128 Safari/537.36"}

    def get(sym: str) -> dict:
        url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json"
        with urllib.request.urlopen(urllib.request.Request(url, headers=ua),
                                    timeout=30) as resp:
            return json.loads(resp.read())

    spy, vix = get("SPY"), get("_VIX")
    S, V, asof = spy["data"]["current_price"], vix["data"]["current_price"], spy["timestamp"]
    session = date.fromisoformat(asof[:10])
    if int(asof[11:13]) < 9:
        session -= timedelta(days=1)

    puts = []
    for o in spy["data"]["options"]:
        sym = o["option"]
        if sym[-9] != "P" or o["bid"] <= 0.0 or o["ask"] <= 0.0:
            continue
        exp = date(2000 + int(sym[-15:-13]), int(sym[-13:-11]), int(sym[-11:-9]))
        K = int(sym[-8:]) / 1000.0
        if K < S:
            puts.append((exp, K, 0.5 * (o["bid"] + o["ask"])))
    expiry = min({e for e, _, _ in puts}, key=lambda e: abs((e - session).days - 30))
    dte = (expiry - session).days
    T = dte / 365.0

    pts = []
    for e, K, mid in puts:
        if e != expiry:
            continue
        try:
            iv = implied_vol(mid, S, K, T, r, q, "put")
        except ValueError:
            continue
        pts.append((K, 100.0 * math.log(S / K), 100.0 * iv,
                    bs_greeks(S, K, T, r, iv, q, "put").delta))

    atm = min(pts, key=lambda p: abs(p[1]))
    d25 = min(pts, key=lambda p: abs(p[3] + 0.25))
    d10 = min(pts, key=lambda p: abs(p[3] + 0.10))
    fit = [(m, iv) for _, m, iv, _ in pts if 1.0 <= m <= 8.0]
    mx, my = mean([m for m, _ in fit]), mean([v for _, v in fit])
    slope = (sum((m - mx) * (v - my) for m, v in fit)
             / sum((m - mx) ** 2 for m, _ in fit))
    return {"asof": asof, "session": session.isoformat(), "spot": S, "vix": V,
            "expiry": expiry.isoformat(), "dte": dte, "atm_iv": atm[2],
            "atm_ratio": atm[2] / V, "slope": slope,
            "put25": d25, "put10": d10, "n_fit": len(fit)}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _style(ax, ink, title, xlabel, ylabel):
    ax.set_facecolor(ink["surface"])
    ax.set_title(title, fontsize=13, color=ink["primary"], loc="left", pad=14)
    ax.set_xlabel(xlabel, fontsize=10, color=ink["muted"])
    ax.set_ylabel(ylabel, fontsize=10, color=ink["muted"])
    ax.tick_params(colors=ink["muted"], labelsize=9)
    ax.grid(axis="y", color=ink["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink["axis"])


def plot(rows: list[dict], put: list[Trade], spread: list[Trade],
         out_dir: str = ".", dark: bool = False) -> list[str]:
    """
    Two charts, one idea each.

      equity   the three curves on the same (cash-secured) capital, log scale,
               marked daily so the 2008 and 2020 holes show at their true depth.
      regime   mean trade return by VIX at entry with sqrt(n_indep) intervals,
               naked put next to spread -- where does the insurance cost you.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    from black_scholes import _INK
    ink = _INK[bool(dark)]
    c_put, c_spread, c_spy = ink["series"]
    vm = put[0].vm
    written = []
    dates = [date.fromisoformat(r_["date"]) for r_ in rows]

    # --- chart 1: equity ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 4.9), dpi=160)
    fig.patch.set_facecolor(ink["surface"])
    series = [("put-write 25d", ladder_equity(put, rows, mtm=True), c_put),
              ("put spread 25d/10d", ladder_equity(spread, rows, mtm=True), c_spread),
              ("SPY", [r_["spy"] / rows[0]["spy"] for r_ in rows], c_spy)]
    for label, eq, col in series:
        ax.plot(dates, eq, color=col, linewidth=1.5, zorder=3)
        ax.annotate(f"{label}  {eq[-1]:.2f}x", (dates[-1], eq[-1]),
                    textcoords="offset points", xytext=(6, -3), fontsize=8.5,
                    color=col, va="center")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4, 8])
    ax.set_yticklabels(["0.5x", "1x", "2x", "4x", "8x"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    _style(ax, ink, "Selling the SPY put, marked daily",
           "", "growth of 1, cash-secured (log)")
    fig.text(0.125, 0.005,
             f"Vol model '{vm.name}': {vm.atm_ratio:.2f} x VIX at the money, "
             f"{vm.skew:.1f} pt per 1% OTM. $0.05 half-spread per leg, no interest "
             "on collateral.", fontsize=8.5, color=ink["muted"])
    fig.tight_layout(rect=(0, 0.035, 0.9, 1))
    path = f"{out_dir}/put_write_equity.png"
    fig.savefig(path, facecolor=ink["surface"])
    plt.close(fig)
    written.append(path)

    # --- chart 2: regime --------------------------------------------------------
    rp, rs = regime_table(put), regime_table(spread)
    live = [(a, b) for a, b in zip(rp, rs) if a.n]
    xs = list(range(len(live)))
    fig, ax = plt.subplots(figsize=(7.6, 4.7), dpi=160)
    fig.patch.set_facecolor(ink["surface"])
    w = 0.36
    for off, col, label, key in ((-w / 2, c_put, "put-write", 0),
                                 (w / 2, c_spread, "put spread", 1)):
        vals = [pair[key].mean_ret for pair in live]
        errs = [1.96 * pair[key].se for pair in live]
        ax.bar([x + off for x in xs], vals, width=w, color=col, label=label, zorder=3)
        ax.errorbar([x + off for x in xs], vals, yerr=errs, fmt="none",
                    ecolor=ink["primary"], elinewidth=1.2, capsize=4, zorder=4)
    for x, pair in zip(xs, live):
        top = max(pair[0].mean_ret + 1.96 * pair[0].se, pair[1].mean_ret + 1.96 * pair[1].se)
        ax.annotate(f"n={pair[0].n:,}\nindep {pair[0].n_indep}", (x, top),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=8, color=ink["muted"])
    ax.axhline(0.0, color=ink["axis"], linewidth=1.0, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([pair[0].label for pair in live])
    lo_y = min(min(p.mean_ret - 1.96 * p.se for p in pair) for pair in live)
    hi_y = max(max(p.mean_ret + 1.96 * p.se for p in pair) for pair in live)
    ax.set_ylim(min(lo_y, 0) - 0.1 * (hi_y - lo_y), hi_y + 0.35 * (hi_y - lo_y))
    _style(ax, ink, "Trade return by VIX at entry",
           "VIX on the day of the sale", "return on collateral per trade (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=ink["primary"])
    fig.text(0.125, 0.005,
             f"Vol model '{vm.name}'. 95% intervals use sqrt(n_indep): daily entries "
             "with a 21-day hold overlap.", fontsize=9, color=ink["muted"])
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    path = f"{out_dir}/put_write_by_vix.png"
    fig.savefig(path, facecolor=ink["surface"])
    plt.close(fig)
    written.append(path)
    return written


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _inv_norm(p: float) -> float:
    """N^-1(p) by bisection on norm_cdf. Slow, exact, independent of the solver."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _synthetic_rows(S0: float, ST: float, vix: float, n: int = HOLD + 1) -> list[dict]:
    """Flat at S0 until the last row, which closes at ST. Consecutive calendar days."""
    d0 = date(2020, 1, 1).toordinal()
    rows = [{"date": date.fromordinal(d0 + k).isoformat(), "spy": S0, "vix": vix}
            for k in range(n)]
    rows[-1]["spy"] = ST
    return rows


def _verify() -> None:
    def ok(label: str) -> None:
        print(f"[{label}] ok")

    S, T, vix, r = 640.0, 30 / 365, 17.0, 0.0

    # [1] the flat model is Black-Scholes at VIX; atm_ratio scales the ATM vol
    #     exactly; positive skew lifts OTM puts and lowers ITM ones; the floor
    #     engages instead of going non-positive.
    for K in (560.0, 600.0, 640.0, 700.0):
        assert put_mid(S, K, T, vix, FLAT) == bs_price(S, K, T, r, vix / 100, 0.0, "put")
    assert abs(CALIBRATED.sigma(vix, S, S) - 0.87 * vix / 100) < 1e-15
    assert put_mid(S, 600.0, T, vix, SKEWED) > put_mid(S, 600.0, T, vix, FLAT)
    assert put_mid(S, 700.0, T, vix, SKEWED) < put_mid(S, 700.0, T, vix, FLAT)
    assert SKEWED.sigma(vix, S, 2.0 * S) == SIGMA_FLOOR
    ok("1 vol model reduces to BS, scales ATM, moves the right way")

    # [2] the strike-derivative of the modelled price against a central
    #     difference, for every model in the report plus a steeper one. The
    #     vega * dsigma/dK term is the new formula; at skew 0 it vanishes and
    #     the check reduces to the plain dual delta.
    h = 1e-4
    for vm in MODELS + (VolModel("steep", 0.8, 2.5),):
        for K in (580.0, 620.0, 640.0, 665.0):
            fd = (put_mid(S, K + h, T, vix, vm) - put_mid(S, K - h, T, vix, vm)) / (2 * h)
            an = dput_dK(S, K, T, vix, vm)
            assert abs(an - fd) < 1e-6, (vm.name, K, an, fd)
    ok("2 d(price)/dK matches finite difference for every vol model")

    # [3] delta solver: at flat vol it must reproduce the closed form
    #     K = S exp(-d1 sigma sqrt(T) + sigma^2 T / 2) with d1 = N^-1(1 - delta),
    #     and under any model the delta at the solved strike is the target.
    for target in (0.10, 0.25, 0.40):
        sigma = vix / 100
        d1 = _inv_norm(1.0 - target)
        K_closed = S * math.exp(-d1 * sigma * math.sqrt(T) + 0.5 * sigma * sigma * T)
        K_flat = strike_for_delta(S, T, vix, target, FLAT)
        assert abs(K_flat / K_closed - 1.0) < 1e-7, (target, K_flat, K_closed)
        for vm in (SKEWED, CALIBRATED):
            K_vm = strike_for_delta(S, T, vix, target, vm)
            assert abs(put_delta(S, K_vm, T, vix, vm) + target) < 1e-7
        assert strike_for_delta(S, T, vix, target, SKEWED) < K_flat   # skew pushes OTM
    try:
        strike_for_delta(S, T, vix, 1.5, FLAT)      # no put has delta -1.5
        raise AssertionError("solver accepted an unbracketed delta")
    except ValueError:
        pass
    ok("3 delta-target strike matches closed form; refuses outside bracket")

    # [4] trade arithmetic on a synthetic crash and a synthetic flat month.
    #     The spread's loss is capped at width - credit exactly when S_T is
    #     through the long strike; it equals the naked put plus the long leg's
    #     own P&L otherwise; a flat month pays the whole credit.
    for ST in (75.0, 90.0, 96.0, 100.0):
        rows = _synthetic_rows(100.0, ST, 20.0)
        pw = build_trades(rows, 0.25, None, CALIBRATED, 0.05)[0]
        sp = build_trades(rows, 0.25, 0.10, CALIBRATED, 0.05)[0]
        assert pw.i == 0 and pw.j == HOLD and abs(pw.T - HOLD / 365) < 1e-12
        assert sp.K_long < sp.K_short < 100.0
        assert abs(pw.pnl - (pw.credit - max(pw.K_short - ST, 0.0))) < 1e-12
        long_cost = put_mid(100.0, sp.K_long, sp.T, 20.0, CALIBRATED) + 0.05
        long_pnl = max(sp.K_long - ST, 0.0) - long_cost
        assert abs(sp.pnl - (pw.pnl + long_pnl)) < 1e-12
        assert sp.pnl >= -sp.max_risk - 1e-12
        if ST <= sp.K_long:
            assert abs(sp.pnl + sp.max_risk) < 1e-12 and abs(sp.ret_on_risk + 1.0) < 1e-12
        if ST == 100.0:
            assert abs(pw.pnl - pw.credit) < 1e-12 and abs(sp.pnl - sp.credit) < 1e-12
    ok("4 P&L identities: cap at width - credit, spread = put + long leg")

    # [5] the mark converges to the settlement: on the expiry row the model
    #     value is intrinsic, so unrealized == ret to within the T -> 0 branch.
    rows = _synthetic_rows(100.0, 90.0, 20.0)
    pw = build_trades(rows, 0.25, None, CALIBRATED, 0.05)[0]
    ord_i = date.fromisoformat(rows[0]["date"]).toordinal()
    ord_j = date.fromisoformat(rows[-1]["date"]).toordinal()
    assert abs(unrealized(pw, rows[-1], ord_j, ord_j) - pw.ret) < 1e-12
    at_entry = unrealized(pw, rows[0], ord_i, ord_j)
    assert abs(at_entry + 0.05 / pw.K_short) < 1e-12     # mark at mid: lose the haircut
    ok("5 daily mark equals settlement at expiry, minus haircut at entry")

    # [6] compounding, drawdown, vol and Sharpe on known sequences.
    assert abs(max_drawdown([1.0, 1.2, 0.9, 1.3, 1.0]) + 0.25) < 1e-12
    assert max_drawdown([1.0, 1.1, 1.2]) == 0.0
    assert abs(cagr(1.0, 2.0, 10.0) - (2 ** 0.1 - 1.0)) < 1e-12
    assert period_returns([1.0, 1.1, 1.21, 1.331], 1) == [
        1.1 / 1.0 - 1, 1.21 / 1.1 - 1, 1.331 / 1.21 - 1]
    # returns 0.01 and 0.03: mean 0.02, sd sqrt(2)*0.01; 12 periods a year
    assert abs(ann_vol([0.01, 0.03]) - math.sqrt(2) * 0.01 * math.sqrt(12)) < 1e-15
    assert abs(sharpe([0.01, 0.03]) - 0.02 / (math.sqrt(2) * 0.01) * math.sqrt(12)) < 1e-12
    ok("6 compounding, drawdown, vol, Sharpe on known sequences")

    # [7] the phases partition the trades exactly once each, and the settled
    #     ladder ends at the mean of the sleeves' settled equity -- so the
    #     row-by-row walk agrees with the direct per-phase compounding.
    rows = vrp_data.build()[:300]
    trades = build_trades(rows, 0.25, None, CALIBRATED, 0.05)
    seen = sorted(k for p in range(HOLD) for k in range(p, len(trades), HOLD))
    assert seen == list(range(len(trades)))
    ladder = ladder_equity(trades, rows, mtm=False)
    direct = mean([settled_equity(trades, p)[-1][1] for p in range(HOLD)])
    assert abs(ladder[-1] - direct) < 1e-12, (ladder[-1], direct)
    assert independent_count([t.i for t in trades], HOLD) == math.ceil(len(trades) / HOLD)
    ok("7 phases partition the sample; ladder agrees with per-phase compounding")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_summary(rows_: list[Summary]) -> None:
    print(f"  {'':<32}{'CAGR':>7}{'tracks':>14}{'vol':>7}{'Sharpe':>8}{'tracks':>12}"
          f"{'maxDD':>8}{'mtmDD':>8}{'trkDD':>8}{'win%':>6}{'worst':>8}{'on':>12}")
    print("  " + "-" * 134)
    for s in rows_:
        c0, c1 = s.cagr_range
        s0, s1 = s.sharpe_range
        print(f"  {s.name:<32}{100*s.cagr:>6.2f}%{100*c0:>7.2f}/{100*c1:>5.2f}%"
              f"{100*s.vol:>6.1f}%{s.sharpe:>8.2f}{s0:>7.2f}/{s1:>4.2f}"
              f"{100*s.dd_settled:>7.1f}%{100*s.dd_mtm:>7.1f}%{100*s.phase_dd:>7.1f}%"
              f"{100*s.win:>5.0f}%{100*s.worst_ret:>7.2f}%{s.worst_on:>12}")


def _print_regimes(title: str, regs: list[Regime]) -> None:
    print(f"\n  {title}")
    print(f"  {'VIX':>7}{'n':>7}{'indep':>7}{'credit%':>9}{'ret%':>8}{'+/-95%':>8}"
          f"{'win%':>6}{'p05':>8}{'worst':>8}{'on':>12}")
    print("  " + "-" * 80)
    for g in regs:
        if not g.n:
            continue
        ci = 1.96 * g.se if not math.isnan(g.se) else math.nan
        print(f"  {g.label:>7}{g.n:>7,}{g.n_indep:>7}{g.mean_credit:>9.2f}{g.mean_ret:>8.2f}"
              f"{ci:>8.2f}{100*g.win:>5.0f}%{g.p05:>8.2f}{g.worst:>8.2f}{g.worst_on:>12}")


def print_calibration() -> None:
    c = measure_skew_today()
    k25, m25, iv25, _ = c["put25"]
    k10, m10, iv10, _ = c["put10"]
    print(f"  CBOE delayed chain {c['asof']} (session {c['session']}): "
          f"SPY {c['spot']:.2f}  VIX {c['vix']:.2f}  expiry {c['expiry']} ({c['dte']} dte)")
    print(f"  ATM put IV {c['atm_iv']:.2f}  ->  atm_ratio {c['atm_ratio']:.3f}")
    print(f"  skew slope {c['slope']:.2f} vol pts per 1% OTM  (LS over {c['n_fit']} strikes 1-8% out)")
    print(f"  25d put K {k25:g} ({m25:.2f}% OTM) IV {iv25:.2f}  = VIX {iv25 - c['vix']:+.2f}")
    print(f"  10d put K {k10:g} ({m10:.2f}% OTM) IV {iv10:.2f}  = VIX {iv10 - c['vix']:+.2f}")
    print(f"  in use: VolModel('calibrated', {CALIBRATED.atm_ratio}, {CALIBRATED.skew})")


def validate_against_put_index(rows: list[dict]) -> None:
    """
    The one external answer key this folder has. CBOE's PUT index sells the
    ATM SPX put monthly against T-bill collateral. Its twin here is the
    50-delta trade, which earns no collateral interest, so a faithful model
    should land roughly one bill yield *below* PUT. A model that lands above
    it is over-crediting the seller.
    """
    import urllib.request
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/PUT_History.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; "
                                               "Win64; x64) AppleWebKit/537.36 Chrome/128"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        lines = resp.read().decode().splitlines()
    px = {}
    for row in csv.DictReader(lines):
        m, d, y = row["DATE"].split("/")
        px[f"{y}-{m}-{d}"] = float(row["PUT"])
    days = sorted(d for d in px if rows[0]["date"] <= d <= rows[-1]["date"])
    years = (date.fromisoformat(days[-1]).toordinal()
             - date.fromisoformat(days[0]).toordinal()) / 365.25
    put_cagr = cagr(px[days[0]], px[days[-1]], years)
    dd = max_drawdown([px[d] for d in days])
    print(f"  CBOE PUT index  {days[0]} -> {days[-1]}: CAGR {100*put_cagr:.2f}%  "
          f"maxDD {100*dd:.1f}%   (ATM, monthly, earns T-bill interest)")
    # Score the synthetic over the index's own dates, not the full cache.
    sub = [r_ for r_ in rows if days[0] <= r_["date"] <= days[-1]]
    for vm in MODELS:
        s = summarize("", build_trades(sub, 0.50, None, vm, 0.05), sub)
        print(f"  synthetic 50d   {vm.name:<13}: CAGR {100*s.cagr:.2f}%  mtmDD {100*s.dd_mtm:.1f}%"
              f"   {100*(s.cagr - put_cagr):+.2f} vs PUT, earning no interest")
    print("  a faithful model lands ~1-1.5 points BELOW the index; above it is over-credit.")


def main(plot_charts: bool = False, dark: bool = False, calibrate: bool = False,
         validate: bool = False) -> None:
    print("verify")
    _verify()

    if calibrate:
        print("\ncalibration")
        print_calibration()
    if validate:
        print("\nvalidation against the CBOE PUT index")
        validate_against_put_index(vrp_data.build())
        return

    rows = vrp_data.build()
    print(f"\n{len(rows):,} joined days   {rows[0]['date']} -> {rows[-1]['date']}")

    results = [spy_summary(rows)]
    built: dict[str, tuple[list[Trade], list[Trade]]] = {}
    for vm in MODELS:
        put = build_trades(rows, 0.25, None, vm, 0.05)
        spread = build_trades(rows, 0.25, 0.10, vm, 0.05)
        built[vm.name] = (put, spread)
        results.append(summarize(f"put-write 25d       {vm.name}", put, rows))
        results.append(summarize(f"put spread 25d/10d  {vm.name}", spread, rows))
    n_ind = independent_count([t.i for t in built[FLAT.name][0]], HOLD)
    print(f"{len(built[FLAT.name][0]):,} trades per strategy, {n_ind} non-overlapping\n")
    _print_summary(results)
    print("\n  CAGR and maxDD from the 21-sleeve ladder (settled at expiry); mtmDD marks open")
    print("  trades daily at model mid. 'tracks' = min/max over the 21 single-entry-day tracks;")
    print("  vol and Sharpe are the median track, not the ladder (stale marks smooth it).")
    print("  Returns are on the short strike as collateral for the put AND the spread; no")
    print("  interest is credited on that collateral. $0.05 half-spread per leg at entry.")

    put, spread = built[CALIBRATED.name]
    _print_regimes(f"put-write 25d, {CALIBRATED.name} -- return on collateral by VIX at entry",
                   regime_table(put))
    _print_regimes(f"put spread 25d/10d, {CALIBRATED.name}", regime_table(spread))

    ror = [100.0 * t.ret_on_risk for t in spread]
    cred = [100.0 * t.credit / (t.K_short - t.K_long) for t in spread]
    print(f"\n  spread on its own max risk: mean {mean(ror):.1f}% per trade, "
          f"credit {mean(cred):.0f}% of width, "
          f"{sum(x <= -99.9 for x in ror)} of {len(ror)} trades a total loss")

    print(f"\n  worst put-write trades ({CALIBRATED.name}), return on collateral")
    print(f"  {'entry':>12}{'VIX':>7}{'S0':>8}{'ST':>8}{'K':>7}{'credit':>8}{'ret':>9}")
    for t in sorted(put, key=lambda t: t.ret)[:8]:
        print(f"  {t.entry:>12}{t.vix:>7.1f}{t.S0:>8.2f}{t.ST:>8.2f}{t.K_short:>7.0f}"
              f"{t.credit:>8.2f}{100*t.ret:>8.2f}%")
    months: dict[str, int] = {}
    for t in sorted(put, key=lambda t: t.ret)[:50]:
        months[t.entry[:7]] = months.get(t.entry[:7], 0) + 1
    top = sorted(months.items(), key=lambda kv: -kv[1])[:4]
    print("  the 50 worst trades by entry month: " +
          ", ".join(f"{m} x{c}" for m, c in top))

    if plot_charts:
        for p in plot(rows, put, spread, dark=dark):
            print(f"  wrote {p}")


if __name__ == "__main__":
    main(plot_charts="--plot" in sys.argv, dark="--dark" in sys.argv,
         calibrate="--calibrate" in sys.argv, validate="--validate" in sys.argv)

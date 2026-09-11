"""
short_vol_sleeve.py -- the variance risk premium for a $500, long-only account.

Standard library only (plotting excepted; it imports inside the function).

put_write_backtest.py showed the premium is collectable by selling puts: ~5%
a year on cash-secured capital at 40% of SPY's vol. A Level 2 Robinhood cash
account with $500 cannot sell anything -- no spreads, and one SPY put needs
~$75k of collateral. The only way such an account gets *short* volatility is
to buy something that is already short it: an inverse VIX-futures ETF.

SVXY is -0.5x the S&P 500 VIX Short-Term Futures index, daily rebalanced. It
earns the roll-down of the front two VIX futures when the curve is upward
sloping (contango), which is most of the time, and it is destroyed when VIX
gaps: its predecessor at -1x lost 83% over 5-6 Feb 2018. The structure here
is deliberately the dumbest rule that addresses the known failure mode:

    hold SVXY when yesterday's VIX / VIX3M < 1.0   (curve in contango)
    hold cash  otherwise                          (curve in backwardation)

Threshold 1.0 is the contango/backwardation boundary, not a fitted number.
The lag is one full day: the ratio is read at yesterday's close and the
trade happens at today's close, which is what a person with a phone can
actually do. A sensitivity table is printed for other thresholds and for a
zero lag; it is there to show the result is not a knife edge, not to pick
from.

WHAT THE FILTER CAN AND CANNOT DO
---------------------------------
Backwardation persists for weeks in a crisis (2011, 2020), so the filter
sidesteps most of the grinding losses and, more importantly, the recovery
days when SVXY is worth the least. It does nothing about a gap: 2 Feb 2018
closed in contango and the ETF lost half its value over the next two
sessions. That event is in the sample, on purpose, and every drawdown figure
below includes it. **Position size is the only defence against the gap** and
the sizing table is the most important table in the output.

DATA
----
  VIX, VIX3M   CBOE's free daily-history CSVs (VIX3M starts 2009-09).
  SVXY         Yahoo adjusted close from launch, 2011-10-04. The fund was -1x
               until 2018-02-27 and -0.5x from 2018-02-28. A daily-rebalanced
               -0.5x product returns exactly half the daily return of the
               -1x product before fees, so pre-change daily returns are
               halved to make one consistent -0.5x series. That transform is
               verified against the Feb 2018 event itself in _verify().
  SPY          Yahoo adjusted close, the benchmark.
Costs: Robinhood charges no commission; 5 bp is taken on every switch for
the spread. Cash earns nothing. SVXY's 0.95% expense ratio is already inside
its price history.

Overlap is not a problem here -- daily returns do not overlap -- but vol
clusters, so the null is a block shuffle of the signal (21-day blocks), not
an i.i.d. one. The question it answers: does *timing* add anything beyond
just holding the ETF at the same average exposure?

Run it:  python short_vol_sleeve.py            tables (fetches + caches on first run)
         python short_vol_sleeve.py --plot     charts (--dark for the dark palette)
         python short_vol_sleeve.py --signal   today's position for the sleeve
         python short_vol_sleeve.py --refresh  re-fetch the cache
This script never places an order. It prints what the rule says; you trade.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime

from vrp_study import mean, stdev

CACHE = "short_vol_daily.csv"
HALF_LEVERAGE_FROM = "2018-02-28"     # first session SVXY ran at -0.5x
THRESHOLD = 1.0                       # VIX / VIX3M below this = contango = hold
LAG = 1                               # read yesterday's ratio, trade today's close
SWITCH_COST = 0.0005                  # 5 bp per switch, for the spread
TRADING_DAYS = 252
BLOCK = 21                            # block length for the shuffled-signal null

CBOE_HIST = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=20y&interval=1d"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_cboe(name: str) -> dict[str, float]:
    """{YYYY-MM-DD: close} from CBOE's daily-history CSV (dates are MM/DD/YYYY)."""
    out = {}
    for row in csv.DictReader(_get(CBOE_HIST.format(name=name)).decode().splitlines()):
        m, d, y = row["DATE"].split("/")
        out[f"{y}-{m}-{d}"] = float(row["CLOSE"])
    return out


def fetch_yahoo(sym: str) -> dict[str, float]:
    """
    {YYYY-MM-DD: adjusted close}. Today's row is dropped: during the session
    Yahoo reports the live print there, and a live print is not a close.
    """
    raw = json.loads(_get(YAHOO.format(sym=sym)))
    result = raw["chart"]["result"][0]
    stamps = result["timestamp"]
    closes = result["indicators"]["adjclose"][0]["adjclose"]
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    out = {}
    for ts, px in zip(stamps, closes):
        if px is None:
            continue
        day = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
        if day != today:
            out[day] = float(px)
    return out


FIELDS = ("date", "svxy", "spy", "vix", "vix3m")


def build(path: str = CACHE, refresh: bool = False) -> list[dict]:
    """Inner-join the four series on date, cache to CSV, return the rows."""
    if os.path.exists(path) and not refresh:
        with open(path, newline="") as f:
            return [{"date": r["date"], **{k: float(r[k]) for k in FIELDS[1:]}}
                    for r in csv.DictReader(f)]
    svxy, spy = fetch_yahoo("SVXY"), fetch_yahoo("SPY")
    vix, vix3m = fetch_cboe("VIX"), fetch_cboe("VIX3M")
    days = sorted(svxy.keys() & spy.keys() & vix.keys() & vix3m.keys())
    rows = [{"date": d, "svxy": svxy[d], "spy": spy[d], "vix": vix[d], "vix3m": vix3m[d]}
            for d in days]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------

def etf_returns(rows: list[dict]) -> list[float | None]:
    """
    Daily returns of a consistent -0.5x product, aligned to rows (first None).
    Before HALF_LEVERAGE_FROM the fund was -1x, so those days are halved.
    """
    out: list[float | None] = [None]
    for prev, cur in zip(rows, rows[1:]):
        r = cur["svxy"] / prev["svxy"] - 1.0
        out.append(0.5 * r if cur["date"] < HALF_LEVERAGE_FROM else r)
    return out


def positions(rows: list[dict], threshold: float = THRESHOLD,
              lag: int = LAG) -> list[int | None]:
    """
    pos[t] is the exposure carried over the return of day t (close t-1 to
    close t). It is decided from the ratio at close t-1-lag: with lag 1 that
    is *yesterday's* close relative to the trade at close t-1. None until
    enough history exists. No element of pos[t] may depend on any row >= t
    -- test [1] pins this.
    """
    out: list[int | None] = []
    for t in range(len(rows)):
        k = t - 1 - lag
        if k < 0:
            out.append(None)
            continue
        out.append(1 if rows[k]["vix"] / rows[k]["vix3m"] < threshold else 0)
    return out


def strategy_returns(rets: list[float | None], pos: list[int | None],
                     fraction: float = 1.0,
                     cost: float = SWITCH_COST) -> list[float | None]:
    """
    fraction * pos * ETF return, less fraction * cost on any day the position
    differs from the day before. The sleeve starts flat.
    """
    out: list[float | None] = []
    prev = 0
    for r, p in zip(rets, pos):
        if r is None or p is None:
            out.append(None)
            continue
        day = fraction * p * r
        if p != prev:
            day -= fraction * cost
        prev = p
        out.append(day)
    return out


def constant_exposure(rets: list[float | None], pos: list[int | None],
                      fraction: float = 1.0) -> list[float | None]:
    """Always-in at the same average exposure the rule had: timing removed."""
    live = [p for p in pos if p is not None]
    tim = sum(live) / len(live)
    return [None if (r is None or p is None) else fraction * tim * r
            for r, p in zip(rets, pos)]


def equity(rets: list[float | None]) -> list[float]:
    """Compounded growth of 1 over the non-None returns."""
    out, eq = [], 1.0
    for r in rets:
        if r is None:
            continue
        eq *= 1.0 + r
        out.append(eq)
    return out


# ---------------------------------------------------------------------------
# Statistics, written out
# ---------------------------------------------------------------------------

def max_drawdown(series: list[float]) -> float:
    peak, worst = -math.inf, 0.0
    for x in series:
        peak = max(peak, x)
        worst = min(worst, x / peak - 1.0)
    return worst


def cagr(eq: list[float], years: float) -> float:
    return eq[-1] ** (1.0 / years) - 1.0


def sharpe(rets: list[float]) -> float:
    return mean(rets) / stdev(rets) * math.sqrt(TRADING_DAYS)


def worst_window(rets: list[float], n: int) -> float:
    """Worst compounded return over any n consecutive days."""
    worst = 0.0
    for i in range(len(rets) - n + 1):
        g = 1.0
        for r in rets[i:i + n]:
            g *= 1.0 + r
        worst = min(worst, g - 1.0)
    return worst


@dataclass
class Summary:
    name: str
    cagr: float
    vol: float
    sharpe: float
    max_dd: float
    worst_day: float
    worst_5d: float
    time_in: float
    switches_per_year: float


def summarize(name: str, rets: list[float | None], pos: list[int | None],
              years: float) -> Summary:
    live = [r for r in rets if r is not None]
    p_live = [p for r, p in zip(rets, pos) if r is not None and p is not None]
    switches = sum(1 for a, b in zip(p_live, p_live[1:]) if a != b)
    eq = equity(live)
    return Summary(name, cagr(eq, years), stdev(live) * math.sqrt(TRADING_DAYS),
                   sharpe(live), max_drawdown(eq), min(live), worst_window(live, 5),
                   (sum(p_live) / len(p_live)) if p_live else 1.0, switches / years)


def block_shuffle(pos: list[int], block: int, rng: random.Random) -> list[int]:
    """Shuffle whole blocks of the signal; keeps its autocorrelation and mean."""
    blocks = [pos[i:i + block] for i in range(0, len(pos), block)]
    rng.shuffle(blocks)
    return [p for b in blocks for p in b]


def timing_null(rets: list[float | None], pos: list[int | None], years: float,
                n: int = 300, seed: int = 0) -> tuple[list[float], list[float]]:
    """Sharpe and CAGR of n block-shuffled signals with the same time in market."""
    rng = random.Random(seed)
    live_r = [r for r, p in zip(rets, pos) if r is not None and p is not None]
    live_p = [p for r, p in zip(rets, pos) if r is not None and p is not None]
    sh, cg = [], []
    for _ in range(n):
        shuffled = block_shuffle(live_p, BLOCK, rng)
        s = strategy_returns(live_r, shuffled)
        s_live = [x for x in s if x is not None]
        sh.append(sharpe(s_live))
        cg.append(cagr(equity(s_live), years))
    return sh, cg


def percentile_of(x: float, sample: list[float]) -> float:
    return sum(1 for s in sample if s < x) / len(sample)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot(rows: list[dict], curves: list[tuple[str, list[float | None]]],
         out_dir: str = ".", dark: bool = False) -> list[str]:
    """
    Two charts, one idea each: growth of 1 (log) and the drawdown that came
    with it. Same three series on both so the eye can pair them.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    from black_scholes import _INK
    ink = _INK[bool(dark)]
    colors = ink["series"]
    written = []

    def style(ax, title, ylabel):
        ax.set_facecolor(ink["surface"])
        ax.set_title(title, fontsize=13, color=ink["primary"], loc="left", pad=14)
        ax.set_ylabel(ylabel, fontsize=10, color=ink["muted"])
        ax.tick_params(colors=ink["muted"], labelsize=9)
        ax.grid(axis="y", color=ink["grid"], linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(ink["axis"])

    # Align every curve to the first row where all of them are live.
    start = max(next(i for i, r in enumerate(c) if r is not None) for _, c in curves)
    dates = [date.fromisoformat(r["date"]) for r in rows[start:]]
    eqs = [(name, equity(c[start:])) for name, c in curves]

    fig, ax = plt.subplots(figsize=(8.4, 4.9), dpi=160)
    fig.patch.set_facecolor(ink["surface"])
    for (name, eq), col in zip(eqs, colors):
        ax.plot(dates, eq, color=col, linewidth=1.4, zorder=3)
        ax.annotate(f"{name}  {eq[-1]:.2f}x", (dates[-1], eq[-1]),
                    textcoords="offset points", xytext=(6, 0), fontsize=8.5,
                    color=col, va="center")
    ax.set_yscale("log")
    ticks = [0.25, 0.5, 1, 2, 4, 8, 16]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}x" for t in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())
    style(ax, "Short vol for a long-only account", "growth of 1 (log)")
    fig.text(0.125, 0.005,
             f"SVXY as a consistent -0.5x series. Rule: hold when yesterday's "
             f"VIX/VIX3M < {THRESHOLD:g}, else cash. 5 bp per switch.",
             fontsize=8.5, color=ink["muted"])
    fig.tight_layout(rect=(0, 0.035, 0.86, 1))
    path = f"{out_dir}/short_vol_equity.png"
    fig.savefig(path, facecolor=ink["surface"])
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=160)
    fig.patch.set_facecolor(ink["surface"])
    for (name, eq), col in zip(eqs, colors):
        peak, dd = -math.inf, []
        for x in eq:
            peak = max(peak, x)
            dd.append(100.0 * (x / peak - 1.0))
        ax.plot(dates, dd, color=col, linewidth=1.2, zorder=3, label=name)
    style(ax, "The drawdowns that came with it", "drawdown from peak (%)")
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=ink["primary"])
    fig.text(0.125, 0.005,
             "Feb 2018 is the gap the filter cannot see; 2020 is the regime it can.",
             fontsize=8.5, color=ink["muted"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = f"{out_dir}/short_vol_drawdown.png"
    fig.savefig(path, facecolor=ink["surface"])
    plt.close(fig)
    written.append(path)
    return written


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _rows(ratios: list[float], svxy: list[float] | None = None,
          start: str = "2020-01-01") -> list[dict]:
    d0 = date.fromisoformat(start).toordinal()
    svxy = svxy or [100.0] * len(ratios)
    return [{"date": date.fromordinal(d0 + i).isoformat(), "svxy": s, "spy": 100.0,
             "vix": 20.0 * q, "vix3m": 20.0} for i, (q, s) in enumerate(zip(ratios, svxy))]


def _verify(rows: list[dict]) -> None:
    def ok(label: str) -> None:
        print(f"[{label}] ok")

    # [1] alignment. Ratios by day: contango, contango, BACKWARDATION, then
    #     contango. With lag 1 the position over day t comes from day t-2;
    #     with lag 0 from day t-1. Nothing at t may look at t or later.
    syn = _rows([0.9, 0.9, 1.1, 0.9, 0.9, 0.9])
    assert positions(syn, 1.0, 1) == [None, None, 1, 1, 0, 1]
    assert positions(syn, 1.0, 0) == [None, 1, 1, 0, 1, 1]
    # Perturbing the future must not change the past: change the last ratio.
    syn2 = _rows([0.9, 0.9, 1.1, 0.9, 0.9, 5.0])
    assert positions(syn2, 1.0, 1)[:-1] == positions(syn, 1.0, 1)[:-1]
    assert positions(syn2, 1.0, 1)[-1] == positions(syn, 1.0, 1)[-1]   # t-2 unchanged
    ok("1 signal alignment: lag 1 reads t-2, lag 0 reads t-1, no lookahead")

    # [2] the -1x -> -0.5x transform, synthetic and on the real event.
    syn = _rows([0.9] * 4, [100.0, 110.0, 99.0, 108.9], start="2018-02-26")
    r = etf_returns(syn)                      # dates 02-26, 02-27, 02-28, 03-01
    assert r[0] is None
    assert abs(r[1] - 0.05) < 1e-12           # 02-27: still -1x, halved
    assert abs(r[2] - (99.0 / 110.0 - 1.0)) < 1e-12   # 02-28: -0.5x, as is
    assert abs(r[3] - 0.10) < 1e-12
    idx = {row["date"]: i for i, row in enumerate(rows)}
    if "2018-02-02" in idx and "2018-02-06" in idx:
        i0, i2 = idx["2018-02-02"], idx["2018-02-06"]
        raw = rows[i2]["svxy"] / rows[i0]["svxy"] - 1.0
        rr = etf_returns(rows)
        halved = (1.0 + rr[i0 + 1]) * (1.0 + rr[i2]) - 1.0
        assert raw < -0.80, raw                   # the event as it happened at -1x
        assert -0.60 < halved < -0.40, halved     # what -0.5x would have done
    ok("2 leverage transform; Feb 2018 reproduces at -1x and halves as -0.5x")

    # [3] strategy arithmetic: exposure, and the switch cost exactly once per change.
    rets = [None, 0.01, 0.02, -0.05, 0.01, 0.03]
    pos = [None, 1, 1, 0, 0, 1]
    s = strategy_returns(rets, pos, 0.5, 0.001)
    exp = [None, 0.0045, 0.01, -0.0005, 0.0, 0.0145]
    assert s[0] is None and all(abs(a - b) < 1e-15 for a, b in zip(s[1:], exp[1:])), s
    assert strategy_returns(rets, [None, 1, 1, 1, 1, 1], 1.0, 0.0)[1:] == rets[1:] or \
        all(abs(a - b) < 1e-15 for a, b in
            zip(strategy_returns(rets, [None, 1, 1, 1, 1, 1], 1.0, 0.0)[1:], rets[1:]))
    ce = constant_exposure(rets, pos, 1.0)    # time in market 3/5
    assert all(abs(a - 0.6 * b) < 1e-15 for a, b in zip(ce[1:], rets[1:]))
    ok("3 exposure, one switch cost per change, constant-exposure benchmark")

    # [4] statistics on known sequences.
    assert abs(max_drawdown([1.0, 1.2, 0.9, 1.3, 1.0]) + 0.25) < 1e-12
    assert abs(cagr([1.0, 1.5, 2.0], 10.0) - (2 ** 0.1 - 1.0)) < 1e-12
    assert abs(sharpe([0.01, 0.03]) - 0.02 / (math.sqrt(2) * 0.01) * math.sqrt(252)) < 1e-12
    assert abs(worst_window([0.1, -0.1, -0.1, 0.2], 2) - (0.81 - 1.0)) < 1e-12
    assert equity([None, 0.1, None, -0.5]) == [1.1, 0.55]
    ok("4 drawdown, CAGR, Sharpe, worst window, compounding")

    # [5] the block shuffle keeps the mean and the multiset; the percentile helper.
    p = [1] * 50 + [0] * 13
    sh = block_shuffle(p, BLOCK, random.Random(1))
    assert sorted(sh) == sorted(p) and len(sh) == len(p)
    assert sh[:21] in ([1] * 21, [1] * 8 + [0] * 13) or sh[:21] == [1] * 21
    assert percentile_of(2.0, [1.0, 3.0, 1.5, 2.5]) == 0.5
    ok("5 block shuffle preserves the signal's content; percentile")

    # [6] cache round trip.
    tmp = os.path.join(os.environ.get("TEMP", "."), "short_vol_roundtrip.csv")
    syn = _rows([0.9, 1.1], [100.0, 101.5])
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(syn)
    assert build(tmp) == syn
    os.remove(tmp)
    ok("6 cache round trip")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_summaries(rows_: list[Summary]) -> None:
    print(f"  {'':<34}{'CAGR':>7}{'vol':>7}{'Sharpe':>8}{'maxDD':>8}{'worst d':>9}"
          f"{'worst 5d':>10}{'in mkt':>8}{'sw/yr':>7}")
    print("  " + "-" * 98)
    for s in rows_:
        print(f"  {s.name:<34}{100*s.cagr:>6.1f}%{100*s.vol:>6.1f}%{s.sharpe:>8.2f}"
              f"{100*s.max_dd:>7.1f}%{100*s.worst_day:>8.1f}%{100*s.worst_5d:>9.1f}%"
              f"{100*s.time_in:>7.0f}%{s.switches_per_year:>7.1f}")


def _years(rows: list[dict], start: int = 0) -> float:
    return (date.fromisoformat(rows[-1]["date"]).toordinal()
            - date.fromisoformat(rows[start]["date"]).toordinal()) / 365.25


def report(rows: list[dict], plot_charts: bool = False, dark: bool = False) -> None:
    rets = etf_returns(rows)
    pos = positions(rows)
    first = next(i for i, p in enumerate(pos) if p is not None)
    years = _years(rows, first)
    print(f"\n{len(rows):,} joined days   {rows[0]['date']} -> {rows[-1]['date']}   "
          f"({years:.1f} years scored)")

    spy = [None] + [b["spy"] / a["spy"] - 1.0 for a, b in zip(rows, rows[1:])]
    ones = [None if p is None else 1 for p in pos]
    live_rets = [None if p is None else r for r, p in zip(rets, pos)]

    rule = strategy_returns(rets, pos)
    always = strategy_returns(live_rets, ones, 1.0, 0.0)
    const = constant_exposure(rets, pos)
    spy_live = [None if p is None else r for r, p in zip(spy, pos)]

    print()
    _print_summaries([
        summarize("sleeve rule  (contango -> SVXY)", rule, pos, years),
        summarize("SVXY -0.5x   always in", always, ones, years),
        summarize("SVXY -0.5x   constant exposure", const, ones, years),
        summarize("SPY buy & hold", spy_live, ones, years),
    ])
    print("  'constant exposure' holds the ETF every day at the rule's average weight:")
    print("  it has the rule's beta and none of its timing.")

    # --- is the timing real? ------------------------------------------------
    sh_null, cg_null = timing_null(rets, pos, years)
    rule_s = summarize("", rule, pos, years)
    print(f"\n  timing null: 300 block-shuffled signals (21-day blocks, same time in market)")
    print(f"    rule Sharpe {rule_s.sharpe:.2f}  vs null median {sorted(sh_null)[150]:.2f}  "
          f"-> rule beats {100*percentile_of(rule_s.sharpe, sh_null):.0f}% of shuffles")
    print(f"    rule CAGR  {100*rule_s.cagr:.1f}%  vs null median {100*sorted(cg_null)[150]:.1f}%  "
          f"-> rule beats {100*percentile_of(rule_s.cagr, cg_null):.0f}% of shuffles")

    # --- halves -------------------------------------------------------------
    cut = next(i for i, r in enumerate(rows) if r["date"] >= HALF_LEVERAGE_FROM)
    print(f"\n  by period (split at the leverage change, {HALF_LEVERAGE_FROM})")
    for label, lo, hi in (("before", first, cut), ("after", cut, len(rows))):
        yrs = _years(rows[lo:hi])
        sub_rule = strategy_returns(rets[lo:hi], pos[lo:hi])
        sub_alw = strategy_returns(rets[lo:hi], ones[lo:hi], 1.0, 0.0)
        a, b = summarize("", sub_rule, pos[lo:hi], yrs), summarize("", sub_alw, ones[lo:hi], yrs)
        print(f"    {label:<7}{rows[lo]['date']} -> {rows[hi-1]['date']}   "
              f"rule {100*a.cagr:>6.1f}% CAGR, Sharpe {a.sharpe:.2f}, maxDD {100*a.max_dd:.0f}%   "
              f"| always in {100*b.cagr:>6.1f}%, {b.sharpe:.2f}, {100*b.max_dd:.0f}%")

    # --- sensitivity: shown, not selected ----------------------------------
    print("\n  sensitivity (the rule is threshold 1.00, lag 1; the rest is context)")
    print(f"  {'thr':>6}{'lag':>5}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'in mkt':>8}{'sw/yr':>7}")
    for thr in (0.90, 0.95, 1.00, 1.05, 1.10):
        for lag in (0, 1):
            p = positions(rows, thr, lag)
            s = summarize("", strategy_returns(rets, p), p, years)
            mark = "  <- the rule" if (thr == THRESHOLD and lag == LAG) else ""
            print(f"  {thr:>6.2f}{lag:>5}{100*s.cagr:>7.1f}%{s.sharpe:>8.2f}"
                  f"{100*s.max_dd:>7.0f}%{100*s.time_in:>7.0f}%{s.switches_per_year:>7.1f}{mark}")

    # --- sizing: the table that matters ------------------------------------
    account = 500.0
    print(f"\n  sizing on a ${account:.0f} account (the rest sits in cash)")
    print(f"  {'in SVXY':>9}{'CAGR':>7}{'maxDD':>8}{'worst day':>11}"
          f"{'  $ worst day':>14}{'  $ maxDD':>10}{'  $ 2018 repeat':>16}{'  $/yr':>8}")
    for frac in (0.25, 0.50, 1.00):
        s = summarize("", strategy_returns(rets, pos, frac), pos, years)
        repeat = frac * account * 0.51           # the two Feb 2018 sessions at -0.5x
        print(f"  {frac*account:>8.0f}${100*s.cagr:>6.1f}%{100*s.max_dd:>7.0f}%"
              f"{100*s.worst_day:>10.1f}%{account*s.worst_day:>13.0f}"
              f"{account*s.max_dd:>10.0f}{-repeat:>16.0f}{account*s.cagr:>8.0f}")
    print("  '$ 2018 repeat' = the sleeve's share of a -51% two-session gap, the event the")
    print("  filter cannot see coming. If that number is not survivable, the size is wrong.")

    # --- the worst days, and where the rule was -----------------------------
    print("\n  worst ten days for the ETF, and whether the rule was holding it")
    print(f"  {'date':>12}{'SVXY -0.5x':>12}{'ratio t-2':>11}{'held':>6}{'VIX':>7}")
    order = sorted((r, i) for i, r in enumerate(rets) if r is not None and pos[i] is not None)
    for r, i in order[:10]:
        k = i - 1 - LAG
        ratio = rows[k]["vix"] / rows[k]["vix3m"]
        print(f"  {rows[i]['date']:>12}{100*r:>11.1f}%{ratio:>11.3f}"
              f"{'yes' if pos[i] else 'no':>6}{rows[i]['vix']:>7.1f}")

    if plot_charts:
        curves = [("sleeve rule", rule), ("SVXY -0.5x always in", always), ("SPY", spy_live)]
        for p in plot(rows, curves, dark=dark):
            print(f"  wrote {p}")


def signal(sleeve: float) -> None:
    """Today's instruction, from the latest closes CBOE has published."""
    vix, vix3m = fetch_cboe("VIX"), fetch_cboe("VIX3M")
    day = max(vix.keys() & vix3m.keys())
    ratio = vix[day] / vix3m[day]
    hold = ratio < THRESHOLD
    try:
        svxy = fetch_yahoo("SVXY")
        px_day = max(svxy)
        px = svxy[px_day]
        shares = f"about {sleeve / px:.2f} shares at the {px_day} close of {px:.2f}"
    except Exception as exc:                        # the instruction stands without it
        shares = f"(SVXY price unavailable: {exc})"
    print(f"\n  as of the {day} close:  VIX {vix[day]:.2f}   VIX3M {vix3m[day]:.2f}   "
          f"ratio {ratio:.3f}   threshold {THRESHOLD:.2f}")
    print(f"  curve is in {'CONTANGO' if hold else 'BACKWARDATION'}")
    if hold:
        print(f"  -> target: ${sleeve:.0f} in SVXY, {shares}. Hold it if already held; "
              f"buy at today's close if not.")
    else:
        print("  -> target: $0 in SVXY. Sell at today's close if held; stay in cash if not.")
    print("  This script places no orders. Re-run after each close; act once a day at most.")


def main(argv: list[str]) -> None:
    rows = build(refresh="--refresh" in argv)
    print("verify")
    _verify(rows)
    if "--signal" in argv:
        sleeve = 250.0
        if "--sleeve" in argv:
            sleeve = float(argv[argv.index("--sleeve") + 1])
        signal(sleeve)
        return
    report(rows, plot_charts="--plot" in argv, dark="--dark" in argv)


if __name__ == "__main__":
    main(sys.argv[1:])

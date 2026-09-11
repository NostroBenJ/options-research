"""
vrp_data.py -- daily SPY and VIX, joined and cached, for the variance risk
premium study.

Standard library only.

The variance risk premium is the gap between what the option market charges for
variance and what the underlying actually delivers:

    VRP = implied vol (VIX today)  -  realized vol (SPY over the NEXT 30 days)

The alignment is the whole thing. VIX today is a forecast of the *following*
30 days, so it must be compared to vol that has not happened yet. Line VIX up
against trailing realized vol and you will "discover" a fat premium that is
really just you reading the answer key backwards. `forward_rv()` below shifts
in the correct direction; the test at the bottom pins that down.

Sources (both free, no key):
  SPY   Yahoo Finance chart API -- adjusted closes, so dividends don't show up
        as fake one-day gaps in the return series.
  VIX   FRED series VIXCLS -- daily close, back to 1990.

Run it:  python vrp_data.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import urllib.request

SPY_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/SPY"
           "?range={range_}&interval=1d")
VIX_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"

CACHE = "spy_vix_daily.csv"
TRADING_DAYS = 252


def _get(url: str, ua: str | None = None) -> bytes:
    """Fetch a URL. Yahoo requires a browser UA; FRED rejects one. Hence the flag."""
    headers = {"User-Agent": ua} if ua else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                timeout=60) as r:
        return r.read()


def fetch_spy(range_: str = "20y") -> dict[str, float]:
    """{date: adjusted close} for SPY."""
    raw = json.loads(_get(SPY_URL.format(range_=range_), "Mozilla/5.0"))
    result = raw["chart"]["result"][0]
    stamps = result["timestamp"]
    # adjclose, not close -- see module docstring.
    closes = result["indicators"]["adjclose"][0]["adjclose"]

    import datetime as dt
    out = {}
    for ts, px in zip(stamps, closes):
        if px is None:
            continue  # Yahoo emits nulls on halted/holiday rows
        day = dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d")
        out[day] = float(px)
    return out


def fetch_vix() -> dict[str, float]:
    """{date: VIX close}."""
    text = _get(VIX_URL).decode("utf-8")
    out = {}
    for row in csv.DictReader(text.splitlines()):
        val = row["VIXCLS"]
        if val in (".", "", None):
            continue  # FRED marks holidays with a bare dot
        out[row["observation_date"]] = float(val)
    return out


def build(path: str = CACHE, range_: str = "20y", refresh: bool = False) -> list[dict]:
    """
    Join SPY and VIX on date, cache to CSV, return the rows.

    Inner join on purpose: a date with only one of the two series is useless
    here, and silently forward-filling the missing side would invent data.
    """
    if os.path.exists(path) and not refresh:
        with open(path, newline="") as f:
            return [{"date": r["date"], "spy": float(r["spy"]),
                     "vix": float(r["vix"])} for r in csv.DictReader(f)]

    spy, vix = fetch_spy(range_), fetch_vix()
    rows = [{"date": d, "spy": spy[d], "vix": vix[d]}
            for d in sorted(spy.keys() & vix.keys())]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "spy", "vix"])
        w.writeheader()
        w.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# Derived series
# ---------------------------------------------------------------------------

def log_returns(rows: list[dict]) -> list[float | None]:
    """Daily log returns, aligned to `rows`. First element is None."""
    out: list[float | None] = [None]
    for prev, cur in zip(rows, rows[1:]):
        out.append(math.log(cur["spy"] / prev["spy"]))
    return out


def forward_rv(rows: list[dict], window: int = 21) -> list[float | None]:
    """
    Annualized realized vol over the `window` trading days FOLLOWING each row.

    Forward, not trailing. Row i uses returns from i+1 through i+window, so it
    is strictly out-of-sample relative to the VIX print on row i. The last
    `window` rows have no future left and come back None -- they are not zero,
    and dropping them is the caller's job.

    Zero-mean convention: variance is measured about 0, not about the sample
    mean. Over 21 days the drift estimate is pure noise, and this is what the
    VIX's own formula assumes, so the two sides stay comparable.
    """
    rets = log_returns(rows)
    out: list[float | None] = []
    for i in range(len(rows)):
        future = rets[i + 1: i + 1 + window]
        if len(future) < window or any(r is None for r in future):
            out.append(None)
            continue
        var = sum(r * r for r in future) / window
        out.append(math.sqrt(var * TRADING_DAYS) * 100.0)  # in VIX points
    return out


def _verify_alignment() -> None:
    """
    Prove forward_rv looks forward. Synthetic series: flat until day 10, then a
    burst of volatility. A forward window must see the burst BEFORE it starts.
    """
    rows = [{"date": f"d{i}", "spy": 100.0, "vix": 0.0} for i in range(40)]
    for i in range(10, 40):  # alternating +-2% from day 10 on
        rows[i]["spy"] = 100.0 * (1.02 if i % 2 else 0.98)
    rv = forward_rv(rows, window=5)
    # The burst's first RETURN is at index 10, so with a 5-day forward window
    # row 5 (returns 6..10) is the first row that can see it and row 4
    # (returns 5..9) is the last that cannot. Pinning both sides of that exact
    # boundary is what distinguishes a forward window from a trailing one.
    assert rv[4] == 0.0, "row 4 looks at flat days only and must read zero"
    assert rv[5] > 0.0, "row 5 must already see the burst that begins at day 10"
    assert rv[0] == 0.0, "rows well before the burst should see zero vol ahead"
    assert rv[-1] is None, "last rows have no future and must be None"
    print("[0] forward-window alignment ....... ok")


if __name__ == "__main__":
    _verify_alignment()

    rows = build()
    rv = forward_rv(rows)
    paired = [(r["vix"], v) for r, v in zip(rows, rv) if v is not None]

    print(f"\n{len(rows):,} joined trading days   "
          f"{rows[0]['date']} -> {rows[-1]['date']}")
    print(f"{len(paired):,} rows with 21d of future realized vol\n")

    mean_iv = sum(iv for iv, _ in paired) / len(paired)
    mean_rv = sum(v for _, v in paired) / len(paired)
    diffs = sorted(iv - v for iv, v in paired)
    median = diffs[len(diffs) // 2]
    positive = sum(1 for d in diffs if d > 0) / len(diffs)

    print(f"  mean VIX (implied)        {mean_iv:6.2f}")
    print(f"  mean forward realized     {mean_rv:6.2f}")
    print(f"  mean premium              {mean_iv - mean_rv:6.2f} vol points")
    print(f"  median premium            {median:6.2f}")
    print(f"  premium positive          {positive:6.1%} of days")
    print(f"  worst day for a seller    {diffs[0]:6.2f}")
    print(f"\n  Cached to {CACHE}")

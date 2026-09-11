"""
vrp_study.py -- the variance risk premium, cut by VIX regime.

Standard library only (plotting excepted, and it imports inside the function).

`vrp_data.py` builds the series and reports one unconditional number: VIX has
averaged about 3.6 vol points above the realized vol that followed it, and the
premium was positive on 84% of days. That average is real, and on its own it is
close to useless -- it is the average of a distribution with a -69 vol point
day in it. The question that matters to anyone actually short variance is not
"what is the premium" but "what is the premium *when*", and specifically
whether the compensation shows up in the regimes that also carry the tail.

So: bucket by the VIX level at the time of the sale, and look at each regime's
premium, its dispersion, and its worst outcome separately.

THE SAMPLE-SIZE TRAP
--------------------
`forward_rv` looks 21 trading days ahead. Row i and row i+1 therefore share 20
of their 21 days -- the observations overlap almost completely, and the series
of 5,009 "days" contains nothing like 5,009 independent facts. Treat them as
independent and every standard error comes out roughly sqrt(21) ~ 4.6x too
small, which is how a study talks itself into a precision it does not have.

Point estimates are fine under overlap: the mean is still unbiased. It is the
*uncertainty* that breaks. So this module reports every mean from the full
sample, and every error bar from `n_indep` -- the number of observations in
that bucket whose forward windows do not overlap at all. In the top VIX bucket
that difference is 253 rows versus roughly a dozen independent ones, and it is
the difference between "the tail regime pays more" as a finding and as a
coincidence.

`n_indep` is an upper bound on the information available, not a guarantee of
independence: volatility clusters, so even non-overlapping windows drawn from
the same regime carry related news. Read the intervals as optimistic.

Run it:  python vrp_study.py        table
         python vrp_study.py --plot charts
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import vrp_data

# Must match the forward window used to build the realized-vol series: two rows
# are non-overlapping exactly when their indices differ by at least this much.
WINDOW = 21

# Trader-legible VIX regimes rather than equal-count quantiles. The buckets come
# out uneven on purpose -- that the 35+ regime is 5% of history is part of what
# there is to know about it, and quantile edges would hide it.
DEFAULT_EDGES = (0.0, 13.0, 16.0, 20.0, 26.0, 35.0, math.inf)


# ---------------------------------------------------------------------------
# Small statistics, written out
# ---------------------------------------------------------------------------

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def stdev(xs: list[float]) -> float:
    """Sample standard deviation (n-1). Undefined for n < 2."""
    if len(xs) < 2:
        return math.nan
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def quantile(xs: list[float], p: float) -> float:
    """Lower-interpolation quantile. p in [0, 1]."""
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def independent_count(indices: list[int], min_gap: int = WINDOW) -> int:
    """
    How many of these rows have mutually non-overlapping forward windows.

    Greedy first-fit over sorted indices: take the earliest, skip everything
    within `min_gap` of it, repeat. For a contiguous run of length n this
    returns ceil(n / min_gap), which is the intuition -- one usable observation
    per window length. For the scattered indices of a real VIX bucket it
    returns however many survive, which is the number that should be under the
    square-root sign.
    """
    count, last = 0, None
    for i in sorted(indices):
        if last is None or i - last >= min_gap:
            count += 1
            last = i
    return count


# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------

@dataclass
class Bucket:
    lo: float
    hi: float
    n: int              # rows in the bucket
    n_indep: int        # of those, how many are non-overlapping
    mean_iv: float
    mean_rv: float
    mean_premium: float
    median_premium: float
    sd_premium: float
    pct_positive: float
    p05: float          # 5th percentile of the premium -- the bad tail
    worst: float
    worst_date: str     # carried because WHEN the worst day fell is the point

    @property
    def label(self) -> str:
        hi = "+" if math.isinf(self.hi) else f"-{self.hi:g}"
        return f"{self.lo:g}{hi}" if math.isinf(self.hi) else f"{self.lo:g}-{self.hi:g}"

    @property
    def se(self) -> float:
        """
        Standard error of the mean premium, discounted for overlap.

        sd / sqrt(n_indep), NOT sd / sqrt(n). Same spread, honest denominator.
        """
        if self.n_indep < 2 or math.isnan(self.sd_premium):
            return math.nan
        return self.sd_premium / math.sqrt(self.n_indep)

    @property
    def se_naive(self) -> float:
        """What the error bar would be if you ignored the overlap. For contrast."""
        if self.n < 2 or math.isnan(self.sd_premium):
            return math.nan
        return self.sd_premium / math.sqrt(self.n)

    @property
    def ci95(self) -> tuple[float, float]:
        """Normal-approximation 95% interval on the mean premium."""
        if math.isnan(self.se):
            return (math.nan, math.nan)
        return (self.mean_premium - 1.96 * self.se,
                self.mean_premium + 1.96 * self.se)


def bucket_premium(rows: list[dict], rv: list[float | None],
                   edges: tuple[float, ...] = DEFAULT_EDGES) -> list[Bucket]:
    """
    Split the VRP series by the VIX level on the day of the (hypothetical) sale.

    Buckets are half-open [lo, hi), so every row lands in exactly one and the
    counts reconcile against the paired total -- verified in `_verify`.

    Rows whose forward window runs off the end of the data have rv None and are
    dropped. Their indices are kept for the overlap count, so `n_indep` refers
    to positions in the original series, not to positions in the filtered list.
    """
    out = []
    for lo, hi in zip(edges, edges[1:]):
        idx, ivs, rvs, prem, dates = [], [], [], [], []
        for i, (row, v) in enumerate(zip(rows, rv)):
            if v is None or not (lo <= row["vix"] < hi):
                continue
            idx.append(i)
            ivs.append(row["vix"])
            rvs.append(v)
            prem.append(row["vix"] - v)
            dates.append(row["date"])

        if not prem:
            out.append(Bucket(
                lo=lo, hi=hi, n=0, n_indep=0, mean_iv=math.nan,
                mean_rv=math.nan, mean_premium=math.nan,
                median_premium=math.nan, sd_premium=math.nan,
                pct_positive=math.nan, p05=math.nan, worst=math.nan,
                worst_date=""))
            continue

        w = min(range(len(prem)), key=prem.__getitem__)
        out.append(Bucket(
            lo=lo, hi=hi, n=len(prem),
            n_indep=independent_count(idx),
            mean_iv=mean(ivs), mean_rv=mean(rvs),
            mean_premium=mean(prem), median_premium=median(prem),
            sd_premium=stdev(prem),
            pct_positive=sum(1 for p in prem if p > 0) / len(prem),
            p05=quantile(prem, 0.05), worst=prem[w], worst_date=dates[w],
        ))
    return out


# ---------------------------------------------------------------------------
# Plots -- matplotlib imported inside, per the house rule.
# ---------------------------------------------------------------------------

def plot_buckets(buckets: list[Bucket], out_dir: str = ".",
                 dark: bool = False) -> list[str]:
    """
    Two charts, one idea each.

      premium   mean premium per regime with the overlap-corrected 95% interval.
                The claim is "high VIX pays more"; the bars say how much of that
                is signal.
      tail      mean premium against the 5th percentile and the worst day in the
                same regime. Same axis on purpose -- the point is the ratio.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from black_scholes import _INK
    ink = _INK[bool(dark)]
    c_prem, c_tail, c_worst = ink["series"]

    live = [b for b in buckets if b.n]
    labels = [b.label for b in live]
    xs = list(range(len(live)))
    written = []

    # --- chart 1: premium and its honest error bar -------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.7), dpi=160)
    fig.patch.set_facecolor(ink["surface"])
    ax.set_facecolor(ink["surface"])

    errs = [1.96 * b.se if not math.isnan(b.se) else 0.0 for b in live]
    ax.bar(xs, [b.mean_premium for b in live], width=0.6, color=c_prem,
           zorder=3)
    ax.errorbar(xs, [b.mean_premium for b in live], yerr=errs,
                fmt="none", ecolor=ink["primary"], elinewidth=1.4,
                capsize=5, zorder=4)
    # Counts ride above the interval, not inside the bar -- muted ink on the
    # series colour is unreadable, and the whole point of the pair is that the
    # second number is much smaller than the first.
    for x, b, e in zip(xs, live, errs):
        ax.annotate(f"n={b.n:,}\nindep {b.n_indep}", (x, b.mean_premium + e),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=8, color=ink["muted"], zorder=5)

    top = max(b.mean_premium + e for b, e in zip(live, errs))
    bot = min([0.0] + [b.mean_premium - e for b, e in zip(live, errs)])
    ax.set_ylim(bot - 0.06 * (top - bot), top + 0.26 * (top - bot))
    ax.axhline(0.0, color=ink["axis"], linewidth=1.0, zorder=2)
    ax.set_title("Variance risk premium by VIX regime",
                 fontsize=13, color=ink["primary"], loc="left", pad=14)
    ax.set_xlabel("VIX level on the day of the sale", fontsize=10,
                  color=ink["muted"])
    ax.set_ylabel("VIX - forward realized vol (points)", fontsize=10,
                  color=ink["muted"])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.tick_params(colors=ink["muted"], labelsize=9)
    ax.grid(axis="y", color=ink["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink["axis"])
    fig.text(0.125, 0.005,
             "95% intervals use sqrt(n_indep): overlapping windows are not "
             "independent.", fontsize=9, color=ink["muted"])
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    path = f"{out_dir}/vrp_premium_by_vix.png"
    fig.savefig(path, facecolor=ink["surface"])
    plt.close(fig)
    written.append(path)

    # --- chart 2: what you are paid vs what you are exposed to -------------
    fig, ax = plt.subplots(figsize=(7.6, 4.7), dpi=160)
    fig.patch.set_facecolor(ink["surface"])
    ax.set_facecolor(ink["surface"])

    w = 0.27
    ax.bar([x - w for x in xs], [b.mean_premium for b in live], width=w,
           color=c_prem, label="mean premium", zorder=3)
    ax.bar(xs, [b.p05 for b in live], width=w, color=c_tail,
           label="5th percentile", zorder=3)
    ax.bar([x + w for x in xs], [b.worst for b in live], width=w,
           color=c_worst, label="worst day", zorder=3)

    # Date every worst-day bar. Without these the chart reads as six separate
    # disasters; with them the shared month is the first thing you see.
    for x, b in zip(xs, live):
        ax.annotate(b.worst_date[:7], (x + w, b.worst),
                    textcoords="offset points", xytext=(0, -12), ha="center",
                    fontsize=7.5, color=ink["muted"], zorder=5)
    floor = min(b.worst for b in live)
    ax.set_ylim(floor * 1.14, max(b.mean_premium for b in live) * 1.6)

    ax.axhline(0.0, color=ink["axis"], linewidth=1.0, zorder=2)
    ax.set_title("The premium against the tail it is paid for",
                 fontsize=13, color=ink["primary"], loc="left", pad=14)
    ax.set_xlabel("VIX level on the day of the sale", fontsize=10,
                  color=ink["muted"])
    ax.set_ylabel("VIX - forward realized vol (points)", fontsize=10,
                  color=ink["muted"])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.tick_params(colors=ink["muted"], labelsize=9)
    ax.grid(axis="y", color=ink["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink["axis"])
    leg = ax.legend(frameon=False, fontsize=9, loc="lower left",
                    labelcolor=ink["primary"])
    fig.text(0.125, 0.005,
             "Same axis throughout. Dates label each worst day -- note the "
             "shared month.", fontsize=9, color=ink["muted"])
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    path = f"{out_dir}/vrp_tail_by_vix.png"
    fig.savefig(path, facecolor=ink["surface"])
    plt.close(fig)
    written.append(path)

    return written


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify() -> None:
    import random

    # --- 1. The statistics helpers against hand-checkable values -----------
    xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert mean(xs) == 5.0
    # population sd of this classic set is 2, so sample sd is 2*sqrt(8/7)
    assert abs(stdev(xs) - 2.0 * math.sqrt(8 / 7)) < 1e-12
    assert median(xs) == 4.5
    assert median([3.0, 1.0, 2.0]) == 2.0
    print("[1] mean / stdev / median ................ ok")

    # --- 2. independent_count on a contiguous run --------------------------
    # One usable observation per window length, no more.
    for n in (1, 20, 21, 22, 100, 5000):
        for gap in (1, 7, 21):
            got = independent_count(list(range(n)), gap)
            want = -(-n // gap)          # ceil division
            assert got == want, (n, gap, got, want)
    # Gaps in the index list must be respected, not collapsed.
    assert independent_count([0, 5, 10, 500, 505], 21) == 2
    assert independent_count([0, 21, 42], 21) == 3      # exactly at the bound
    assert independent_count([0, 20, 40], 21) == 2      # one short of it
    print("[2] independent_count .................... ok")

    # --- 3. The overlap correction actually recovers the true error bar ----
    # Block-constant series: m identical values per block, blocks drawn iid.
    # The mean of the whole series IS the mean of nb independent block draws,
    # so the true standard error is known exactly. This is the cleanest
    # available model of an overlapping window, and it is the whole argument
    # for reporting sqrt(n_indep) instead of sqrt(n).
    rng = random.Random(0)
    m, nb = 21, 240
    blocks = [rng.gauss(0.0, 1.0) for _ in range(nb)]
    series = [b for b in blocks for _ in range(m)]
    n = len(series)

    true_se = stdev(blocks) / math.sqrt(nb)
    naive_se = stdev(series) / math.sqrt(n)
    n_indep = independent_count(list(range(n)), m)
    adj_se = stdev(series) / math.sqrt(n_indep)

    assert n_indep == nb, (n_indep, nb)
    assert abs(adj_se / true_se - 1.0) < 0.01, adj_se / true_se
    assert abs(naive_se * math.sqrt(m) / true_se - 1.0) < 0.01
    print(f"[3] overlap correction ................... ok")
    print(f"      true se {true_se:.5f}   corrected {adj_se:.5f}   "
          f"naive {naive_se:.5f}  ({true_se / naive_se:.1f}x too tight)")

    # --- 4. Bucketing is a partition -- nothing lost, nothing double-counted
    rows = [{"date": f"d{i}", "spy": 100.0, "vix": float(i % 40)}
            for i in range(400)]
    rv = [float(i % 7) for i in range(400)]
    bs = bucket_premium(rows, rv, DEFAULT_EDGES)
    assert sum(b.n for b in bs) == 400, sum(b.n for b in bs)
    # Half-open [lo, hi): a value exactly on an edge goes to the upper bucket.
    edge = bucket_premium([{"date": "d", "spy": 1.0, "vix": 16.0}], [1.0],
                          (0.0, 13.0, 16.0, math.inf))
    assert [b.n for b in edge] == [0, 0, 1], [b.n for b in edge]
    print("[4] buckets partition the sample ......... ok")

    # --- 5. Known-answer bucket -------------------------------------------
    # Two rows in one bucket with premiums 5 and 15: mean 10, sd 5*sqrt(2).
    rows = [{"date": "a", "spy": 1.0, "vix": 21.0},
            {"date": "b", "spy": 1.0, "vix": 25.0}]
    b = bucket_premium(rows, [16.0, 10.0], (20.0, 26.0))[0]
    assert b.n == 2 and abs(b.mean_premium - 10.0) < 1e-12
    assert abs(b.sd_premium - 5.0 * math.sqrt(2)) < 1e-12
    assert b.pct_positive == 1.0 and b.worst == 5.0
    # Adjacent rows overlap, so two rows are one independent observation --
    # and one observation cannot support an interval.
    assert b.n_indep == 1 and math.isnan(b.se)
    assert not math.isnan(b.se_naive)
    print("[5] known-answer bucket .................. ok")

    # --- 6. Edges: empty buckets and dropped tail rows ---------------------
    empty = bucket_premium([{"date": "d", "spy": 1.0, "vix": 12.0}], [1.0],
                           (30.0, 40.0))[0]
    assert empty.n == 0 and math.isnan(empty.mean_premium)
    assert math.isnan(empty.se) and empty.label == "30-40"
    # A None rv must drop the row entirely, not count as a zero premium.
    dropped = bucket_premium(
        [{"date": "a", "spy": 1.0, "vix": 21.0},
         {"date": "b", "spy": 1.0, "vix": 21.0}], [16.0, None], (20.0, 26.0))[0]
    assert dropped.n == 1 and dropped.mean_premium == 5.0
    assert math.isinf(DEFAULT_EDGES[-1]) and bucket_premium(
        [{"date": "d", "spy": 1.0, "vix": 1e6}], [1.0], (35.0, math.inf))[0].n == 1
    print("[6] empty buckets, None rows, open top ... ok")


# ---------------------------------------------------------------------------

def main(plot: bool = False, dark: bool = False) -> None:
    _verify()

    rows = vrp_data.build()
    rv = vrp_data.forward_rv(rows, WINDOW)
    buckets = bucket_premium(rows, rv)

    paired = [(r["vix"], v) for r, v in zip(rows, rv) if v is not None]
    total_indep = independent_count(
        [i for i, v in enumerate(rv) if v is not None])

    print(f"\n{len(rows):,} joined days   {rows[0]['date']} -> "
          f"{rows[-1]['date']}")
    print(f"{len(paired):,} with {WINDOW}d of forward realized vol, of which "
          f"{total_indep} are non-overlapping\n")

    hdr = (f"  {'VIX':>7} {'n':>6} {'indep':>6} {'IV':>6} {'RV':>6} "
           f"{'prem':>7} {'med':>7} {'+/-95%':>8} {'win%':>6} {'p05':>7} "
           f"{'worst':>8} {'on':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for b in buckets:
        if not b.n:
            continue
        ci = "     n/a" if math.isnan(b.se) else f"{1.96 * b.se:8.2f}"
        print(f"  {b.label:>7} {b.n:>6,} {b.n_indep:>6} {b.mean_iv:>6.2f} "
              f"{b.mean_rv:>6.2f} {b.mean_premium:>7.2f} "
              f"{b.median_premium:>7.2f} {ci} {b.pct_positive:>5.0%} "
              f"{b.p05:>7.2f} {b.worst:>8.2f} {b.worst_date:>12}")

    print("\n  prem   mean VIX minus mean forward realized vol, in vol points")
    print("  +/-95% 1.96 standard errors on sqrt(indep), not sqrt(n)")
    print("  p05    5th percentile of the premium in that regime")

    # The `worst` column is the one most likely to be misread, so say it out
    # loud rather than leaving it to be noticed: if the dates cluster, that
    # column is one episode seen from six angles, not six separate warnings.
    worst_dates = [b.worst_date for b in buckets if b.n]
    months = {d[:7] for d in worst_dates}
    if len(months) < len(worst_dates):
        common = max(months, key=lambda m: sum(d.startswith(m)
                                               for d in worst_dates))
        share = sum(d.startswith(common) for d in worst_dates)
        print(f"\n  NOTE  {share} of the {len(worst_dates)} bucket-worst days "
              f"fall in {common}. The tail column is not six independent")
        print("        disasters -- it is one repricing walking up through "
              "every VIX regime it")
        print("        passed on the way. Size the tail off events, not off "
              "bucket rows.")

    if plot:
        print()
        for path in plot_buckets(buckets, dark=dark):
            print("wrote", path)


if __name__ == "__main__":
    import sys

    main(plot="--plot" in sys.argv, dark="--dark" in sys.argv)

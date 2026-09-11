"""
black_scholes.py -- European option pricing and greeks, built from scratch.

Standard library only. No numpy, no scipy.

The point of this module is NOT to give you a pricer -- every broker has one.
The point is that every greek below is a partial derivative of one function,
and the verification suite at the bottom proves that numerically instead of
asking you to take it on faith.

Run it:  python black_scholes.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SQRT_2PI = math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# The two normal-distribution primitives everything is built on
# ---------------------------------------------------------------------------

def norm_pdf(x: float) -> float:
    """Standard normal density, phi(x)."""
    return math.exp(-0.5 * x * x) / SQRT_2PI


def norm_cdf(x: float) -> float:
    """Standard normal CDF, N(x). Built on erf, so no scipy needed."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def d1_d2(S: float, K: float, T: float, r: float, sigma: float,
          q: float = 0.0) -> tuple[float, float]:
    """
    The two arguments every Black-Scholes quantity is built from.

        d1 = [ln(S/K) + (r - q + sigma^2 / 2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

    Interpretation worth internalizing:
      N(d2) is the risk-neutral probability the option expires in the money.
      N(d1) is the delta (before dividend discounting).

    S     spot price of the underlying
    K     strike
    T     time to expiry IN YEARS (1 trading day ~= 1/252, 1 calendar day 1/365)
    r     risk-free rate, continuously compounded, as a decimal
    sigma implied volatility, annualized, as a decimal (0.18 == 18 vol)
    q     continuous dividend yield (SPY is roughly 0.012)
    """
    if T <= 0.0:
        raise ValueError("T must be positive; expired options have no greeks")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    vol_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_t
    return d1, d1 - vol_t


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------

def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0, kind: str = "call") -> float:
    """Black-Scholes-Merton price of a European call or put."""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if kind == "call":
        return S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2)
    if kind == "put":
        return K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1)
    raise ValueError("kind must be 'call' or 'put'")


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

@dataclass
class Greeks:
    price: float
    delta: float      # d(price) / d(spot)
    gamma: float      # d(delta) / d(spot)
    vega: float       # d(price) / d(vol), per 1 vol POINT (i.e. /100)
    theta: float      # d(price) / d(time), per CALENDAR DAY (i.e. /365)
    rho: float        # d(price) / d(rate), per 1 percentage POINT (i.e. /100)

    def __str__(self) -> str:
        return (f"price {self.price:9.4f}   delta {self.delta:8.4f}   "
                f"gamma {self.gamma:8.4f}\n"
                f"vega  {self.vega:9.4f}   theta {self.theta:8.4f}   "
                f"rho   {self.rho:8.4f}")


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              q: float = 0.0, kind: str = "call") -> Greeks:
    """
    All five greeks, analytically.

    Scaling note -- this trips everyone up. The raw derivatives are per one
    full unit (1.00 of vol = 100 vol points, 1 year of time, 1.00 of rate).
    Brokers quote vega and rho per POINT and theta per DAY, so we divide.
    """
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    sqrt_t = math.sqrt(T)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    pdf_d1 = norm_pdf(d1)

    price = bs_price(S, K, T, r, sigma, q, kind)

    # Gamma and vega are identical for calls and puts.
    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * disc_q * pdf_d1 * sqrt_t

    # This term is the decay from volatility alone -- the "rent on gamma".
    decay = -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_t)

    if kind == "call":
        delta = disc_q * norm_cdf(d1)
        theta = decay - r * K * disc_r * norm_cdf(d2) + q * S * disc_q * norm_cdf(d1)
        rho = K * T * disc_r * norm_cdf(d2)
    else:
        delta = -disc_q * norm_cdf(-d1)
        theta = decay + r * K * disc_r * norm_cdf(-d2) - q * S * disc_q * norm_cdf(-d1)
        rho = -K * T * disc_r * norm_cdf(-d2)

    return Greeks(price=price, delta=delta, gamma=gamma,
                  vega=vega / 100.0, theta=theta / 365.0, rho=rho / 100.0)


# ---------------------------------------------------------------------------
# Implied volatility -- bisection. Slower than Newton, never diverges.
# ---------------------------------------------------------------------------

class VolNotIdentifiable(ValueError):
    """
    The solve succeeded but the answer is meaningless.

    Carries the root it found (`sigma`) and the vol uncertainty that the
    price's own precision implies (`vol_error`), so a caller who genuinely
    wants the number can catch this and take it -- deliberately, in the open,
    rather than by being handed it as though it were a fact.
    """

    def __init__(self, message: str, sigma: float, vol_error: float) -> None:
        super().__init__(message)
        self.sigma = sigma
        self.vol_error = vol_error


def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                q: float = 0.0, kind: str = "call",
                lo: float = 1e-6, hi: float = 5.0, tol: float = 1e-10,
                max_iter: int = 200, price_tick: float = 0.005,
                max_vol_error: float = 0.01) -> float:
    """
    Back out the volatility the market is pricing in. Bisection.

    This is the function you will actually use most. Option prices are quoted
    in dollars, but dollars are not comparable across strikes or expiries.
    Implied vol is. Every serious options conversation happens in vol space.

    Which is exactly why the answer has to be converged in vol space too.
    Three things this function refuses to do, each of which it used to do:

      1. Return a bracket endpoint. Price is strictly increasing in sigma
         (vega > 0 everywhere), so the two endpoints say up front whether a
         root is inside the bracket at all. If it is not, that is an error,
         not a quiet `hi`.

      2. Stop on price error. `abs(price_error) < tol` is satisfied on the
         first iteration for any option whose price is near zero -- the solver
         halts wherever bisection happens to be standing and reports that as
         the answer. Fed four different vols at K=1000/30d it returned
         0.156251 four times: not an estimate, a constant. `tol` is now the
         width of the VOL bracket, which is the space the root lives in.

      3. Answer when the price cannot pin the vol down. Even an exact solve is
         only as sharp as vega lets it be. Vega IS the exchange rate between
         the two spaces, so `price_tick` of price ambiguity is exactly
         `price_tick / vega` of vol ambiguity. Deep OTM vega goes to zero and
         that ratio explodes; the honest output there is a refusal, not a
         number with six decimal places on it.

    price_tick     how precisely you actually know `market_price`. Default is
                   half a cent -- the resolution of a penny-quoted market.
    max_vol_error  how much vol uncertainty you will accept before the
                   function refuses. Raises VolNotIdentifiable past this;
                   pass math.inf to disable the check and take what you get.
    """
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")

    # The bracket endpoints. p_lo is the zero-vol floor -- the discounted
    # intrinsic -- so this subsumes the old below-intrinsic check rather than
    # restating the same formula a second way.
    p_lo = bs_price(S, K, T, r, lo, q, kind)
    p_hi = bs_price(S, K, T, r, hi, q, kind)

    if market_price < p_lo - price_tick:
        raise ValueError(
            f"price {market_price:.6g} is below the zero-vol floor {p_lo:.6g} "
            f"-- no implied vol exists")
    if market_price > p_hi + price_tick:
        raise ValueError(
            f"price {market_price:.6g} exceeds {p_hi:.6g}, the most this "
            f"contract can be worth at the bracket ceiling sigma={hi:g} "
            f"-- widen `hi` or re-check the inputs")

    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, r, mid, q, kind) > market_price:
            hi = mid
        else:
            lo = mid
    else:
        raise RuntimeError(f"bisection failed to converge in {max_iter} iters")

    sigma = 0.5 * (lo + hi)

    # bs_greeks scales vega to a vol POINT; undo that to get the raw partial
    # d(price)/d(sigma), which is the unit the conversion above is written in.
    vega_raw = bs_greeks(S, K, T, r, sigma, q, kind).vega * 100.0
    vol_error = price_tick / vega_raw if vega_raw > 0.0 else math.inf
    if vol_error > max_vol_error:
        raise VolNotIdentifiable(
            f"vol is not identifiable here: vega is {vega_raw:.3g}, so "
            f"+-{price_tick:g} of price is +-{vol_error:.3g} of vol, past the "
            f"{max_vol_error:g} limit. Nearest root is sigma={sigma:.6g}; it "
            f"is a real root and still tells you nothing",
            sigma=sigma, vol_error=vol_error)
    return sigma


# ---------------------------------------------------------------------------
# Plots -- the only part of this file that reaches outside the stdlib.
# matplotlib is imported INSIDE the function, so importing this module for
# pricing still costs you nothing and still works with no dependencies.
# ---------------------------------------------------------------------------

# Chart ink. Index by `dark`. Series colors are slots 1-3 of a palette that has
# been checked for colorblind separation -- don't swap them for arbitrary hues.
_INK = {
    False: {"surface": "#fcfcfb", "primary": "#0b0b0b", "muted": "#898781",
            "grid": "#e1e0d9", "axis": "#c3c2b7",
            "series": ("#2a78d6", "#eb6834", "#1baf7a")},
    True: {"surface": "#1a1a19", "primary": "#ffffff", "muted": "#898781",
           "grid": "#2c2c2a", "axis": "#383835",
           "series": ("#3987e5", "#d95926", "#199e70")},
}


# Where a series' direct label hangs, and which way it leans off the curve.
# Peaks lean up-and-left so they clear the spot marker's text; troughs hang
# below so they don't sit on top of the curve they belong to.
_LABEL_OFFSET = {"peak": ((-8, 8), "right"), "trough": ((0, -16), "center")}


def _label_point(xs, ys, mode: str) -> tuple[float, float]:
    """Where to hang a series' direct label: its peak, its trough, or a fraction in."""
    if mode == "peak":
        i = max(range(len(ys)), key=ys.__getitem__)
    elif mode == "trough":
        i = min(range(len(ys)), key=ys.__getitem__)
    else:
        i = int((len(xs) - 1) * float(mode))
    return xs[i], ys[i]


def plot_greeks(S: float = 640.0, r: float = 0.042, sigma: float = 0.16,
                q: float = 0.012, expiries: tuple[int, ...] = (45, 14, 3),
                width: float = 0.12, n: int = 401,
                out_dir: str = ".", dark: bool = False) -> list[str]:
    """
    Delta, gamma and theta across a range of strikes. One chart per greek.

    Each chart holds one line per expiry, because the shape is only half the
    lesson -- the other half is what happens to that shape as expiry closes in.
    Read them as a set:

      delta  an S-curve through the money. It steepens as expiry approaches;
             in the limit it becomes a step function at the strike.
      gamma  a hump centred near the money -- gamma is a LOCAL phenomenon. It
             is the slope of the delta curve, so it peaks exactly where delta
             is steepest, and rises as expiry closes in.
      theta  the mirror image: a well, deepest where gamma is tallest.

    That last correspondence is the whole point. Put the gamma chart and the
    theta chart side by side and they are the same picture flipped, because
    theta ~= -1/2 * gamma * S^2 * sigma^2. Convexity and its rent are the same
    curve with opposite signs. You cannot buy the hump without digging the well.

    width   half-range of strikes to plot, as a fraction of spot (0.12 = +-12%)
    Returns the list of file paths written.
    """
    import matplotlib
    matplotlib.use("Agg")  # write files, never try to open a window
    import matplotlib.pyplot as plt

    ink = _INK[bool(dark)]
    lo, hi = S * (1.0 - width), S * (1.0 + width)
    strikes = [lo + (hi - lo) * i / (n - 1) for i in range(n)]

    curves = {}
    for days in expiries:
        T = days / 365.0
        gs = [bs_greeks(S, K, T, r, sigma, q, "call") for K in strikes]
        curves[days] = gs

    charts = (
        # Delta's three curves all pin to 1.0 in the left wing and 0.0 in the
        # right one, so label them just past the money where they fan out.
        ("delta", "Delta -- the hedge ratio",
         "delta (shares of stock per option)", "0.63",
         "Steeper into expiry. In the limit it is a step at the strike."),
        ("gamma", "Gamma -- convexity, and where it lives",
         "gamma (delta gained per $1 of spot)", "peak",
         "A hump, not a level. Gamma is local to the money, and it grows."),
        ("theta", "Theta -- the rent you pay for it",
         "theta ($ per calendar day)", "trough",
         "The gamma hump upside down. Same curve, opposite sign."),
    )

    written = []
    for attr, title, ylabel, anchor, caption in charts:
        fig, ax = plt.subplots(figsize=(7.6, 4.7), dpi=160)
        fig.patch.set_facecolor(ink["surface"])
        ax.set_facecolor(ink["surface"])

        for color, days in zip(ink["series"], expiries):
            ys = [getattr(g, attr) for g in curves[days]]
            ax.plot(strikes, ys, color=color, linewidth=2.0,
                    label=f"{days}d", zorder=3)
            lx, ly = _label_point(strikes, ys, anchor)
            offset, align = _LABEL_OFFSET.get(anchor, ((0, 9), "center"))
            ax.annotate(f"{days}d", (lx, ly), textcoords="offset points",
                        xytext=offset, ha=align, fontsize=9,
                        color=ink["primary"], zorder=4)

        # Spot marker. Everything interesting on these charts happens here.
        ax.axvline(S, color=ink["muted"], linewidth=1.0, linestyle=(0, (4, 4)),
                   zorder=1)
        ax.annotate(f"spot {S:g}", (S, 1.0), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(5, -12), fontsize=9,
                    color=ink["muted"], zorder=4)
        if attr == "theta":
            ax.axhline(0.0, color=ink["axis"], linewidth=1.0, zorder=1)

        ax.set_title(title, fontsize=13, color=ink["primary"], loc="left",
                     pad=14)
        ax.set_xlabel("strike", fontsize=10, color=ink["muted"])
        ax.set_ylabel(ylabel, fontsize=10, color=ink["muted"])
        ax.tick_params(colors=ink["muted"], labelsize=9)
        ax.grid(axis="y", color=ink["grid"], linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(ink["axis"])

        leg = ax.legend(title="expiry", frameon=False, fontsize=9,
                        loc="best", labelcolor=ink["primary"])
        leg.get_title().set_fontsize(9)
        leg.get_title().set_color(ink["muted"])

        fig.text(0.125, 0.005, caption, fontsize=9, color=ink["muted"])
        fig.tight_layout(rect=(0, 0.035, 1, 1))

        path = f"{out_dir}/greeks_{attr}.png"
        fig.savefig(path, facecolor=ink["surface"])
        plt.close(fig)
        written.append(path)

    return written


# ---------------------------------------------------------------------------
# Verification -- the part that actually teaches you something
# ---------------------------------------------------------------------------

def _fd(f, x: float, h: float) -> float:
    """Central finite difference: (f(x+h) - f(x-h)) / 2h."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def verify() -> None:
    S, K, T, r, sigma, q = 640.0, 645.0, 30 / 365, 0.042, 0.16, 0.012

    print("=" * 66)
    print("SPY-ish call: S=640 K=645 T=30d r=4.2% vol=16% q=1.2%")
    print("=" * 66)
    print(bs_greeks(S, K, T, r, sigma, q, "call"))

    # --- 1. Put-call parity -------------------------------------------------
    # C - P must equal S*e^(-qT) - K*e^(-rT). This is pure arbitrage: it holds
    # regardless of your volatility assumption or any model at all.
    c = bs_price(S, K, T, r, sigma, q, "call")
    p = bs_price(S, K, T, r, sigma, q, "put")
    lhs = c - p
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    print(f"\n[1] put-call parity     C-P = {lhs:.10f}")
    print(f"                     target = {rhs:.10f}   err {abs(lhs-rhs):.2e}")

    # --- 2. Greeks are real derivatives ------------------------------------
    # Bump each input, re-price, compare the slope to the analytic formula.
    print("\n[2] analytic greek  vs  finite-difference slope")
    for kind in ("call", "put"):
        g = bs_greeks(S, K, T, r, sigma, q, kind)
        fd_delta = _fd(lambda x: bs_price(x, K, T, r, sigma, q, kind), S, 1e-4)
        fd_gamma = _fd(lambda x: bs_greeks(x, K, T, r, sigma, q, kind).delta, S, 1e-4)
        fd_vega = _fd(lambda x: bs_price(S, K, T, r, x, q, kind), sigma, 1e-6) / 100
        fd_theta = -_fd(lambda x: bs_price(S, K, x, r, sigma, q, kind), T, 1e-6) / 365
        fd_rho = _fd(lambda x: bs_price(S, K, T, x, sigma, q, kind), r, 1e-6) / 100
        print(f"  {kind:>4}  delta {g.delta:9.5f} / {fd_delta:9.5f}   "
              f"gamma {g.gamma:9.5f} / {fd_gamma:9.5f}")
        print(f"        vega  {g.vega:9.5f} / {fd_vega:9.5f}   "
              f"theta {g.theta:9.5f} / {fd_theta:9.5f}   "
              f"rho {g.rho:8.5f} / {fd_rho:8.5f}")

    # --- 3. Theta IS the rent on gamma -------------------------------------
    # With r = q = 0 the identity is exact:   theta = -1/2 * gamma * S^2 * vol^2
    # You cannot own convexity without paying for it. This is the whole game.
    print("\n[3] theta = -(1/2) * gamma * S^2 * sigma^2   (exact when r=q=0)")
    g0 = bs_greeks(S, K, T, 0.0, sigma, 0.0, "call")
    theta_annual = g0.theta * 365
    identity = -0.5 * g0.gamma * S * S * sigma * sigma
    print(f"    theta/yr = {theta_annual:.8f}")
    print(f"    -1/2 G S^2 v^2 = {identity:.8f}   err {abs(theta_annual-identity):.2e}")

    # --- 4. Round-trip the implied vol solver -------------------------------
    iv = implied_vol(c, S, K, T, r, q, "call")
    print(f"\n[4] implied vol round-trip: fed in {sigma:.6f}, recovered {iv:.6f}")

    # --- 5. Why 0DTE is a different animal ----------------------------------
    # Same strike, same vol. Only time changes. Watch gamma and theta.
    print("\n[5] identical ATM option at different expiries (S=K=640, vol=16%)")
    print(f"    {'days':>6} {'price':>9} {'gamma':>10} {'theta/day':>11} "
          f"{'theta/price':>12}")
    for days in (45, 30, 14, 7, 3, 1):
        g = bs_greeks(640.0, 640.0, days / 365, r, sigma, q, "call")
        print(f"    {days:>6} {g.price:>9.3f} {g.gamma:>10.5f} "
              f"{g.theta:>11.4f} {abs(g.theta)/g.price:>11.2%}")
    print("\n    Gamma rises and theta/price explodes as expiry approaches.")
    print("    On the last day you are renting convexity at the highest rate")
    print("    of the contract's entire life. That is the 0DTE trade-off.")

    # --- 6. The solver round-trips in VOL space, across a grid ---------------
    # Price a known vol, solve it back, measure the error in vol -- the space
    # the answer lives in. A near-ATM spot check proves the near-ATM case and
    # nothing else, so this sweeps the wings and the short expiries too, where
    # the price-space exit used to quietly fail.
    print("\n[6] implied vol round-trip over a strike x expiry x vol grid")
    worst, worst_at, solved, refused = 0.0, None, 0, 0
    for Kx in (520.0, 580.0, 640.0, 700.0, 780.0):
        for days in (3, 30, 180):
            for true_vol in (0.08, 0.16, 0.35, 0.90):
                Tx = days / 365
                for kind in ("call", "put"):
                    px = bs_price(S, Kx, Tx, r, true_vol, q, kind)
                    try:
                        got = implied_vol(px, S, Kx, Tx, r, q, kind)
                    except VolNotIdentifiable:
                        refused += 1   # correctly declined; see [7]
                        continue
                    solved += 1
                    if abs(got - true_vol) > worst:
                        worst = abs(got - true_vol)
                        worst_at = (Kx, days, true_vol, kind)
    print(f"    {solved} solved, {refused} refused as unidentifiable")
    print(f"    worst vol error {worst:.2e}  at K={worst_at[0]:g} "
          f"{worst_at[1]}d vol={worst_at[2]} {worst_at[3]}")
    assert worst < 1e-8, "round-trip must converge in vol space, not price space"

    # --- 7. The edges -- what it refuses, and why ---------------------------
    # Every one of these used to return a confident float.
    print("\n[7] the edges: inputs with no trustworthy answer")

    def _refusal(label, **kw) -> None:
        try:
            got = implied_vol(**kw)
        except (ValueError, RuntimeError) as e:
            head = str(e).split(" -- ")[0].split(", past")[0]
            print(f"    {label:<26} raises {type(e).__name__}: {head}")
        else:
            raise AssertionError(f"{label} should have raised, returned {got}")

    _refusal("above bracket ceiling", market_price=600.0, S=S, K=K, T=T, r=r,
             q=q, kind="call")
    _refusal("below zero-vol floor", market_price=0.0, S=S, K=520.0, T=T, r=r,
             q=q, kind="call")
    _refusal("deep OTM, vega ~ 0", market_price=bs_price(S, 900.0, T, r, 0.16, q),
             S=S, K=900.0, T=T, r=r, q=q, kind="call")

    # Why the refusal is the right call, with the check switched off so the
    # raw roots are visible. Converging in vol space is a genuine improvement
    # here -- 0.25 comes back exact, where the old price-space exit returned
    # 0.156251 for it -- but that is luck, not robustness: it survives only
    # because its price is still a distinct float. Below 0.20 the price
    # underflows to exactly 0.0 and every vol maps to the same root. The dvol
    # column is the honest error bar, and it condemns all four equally.
    print("\n    K=1000 30d with max_vol_error=inf (check off):")
    print(f"      {'fed':>5} {'price':>12} {'root':>10} {'+-vol':>11}")
    for true_vol in (0.10, 0.16, 0.20, 0.25):
        px = bs_price(S, 1000.0, T, r, true_vol, q, "call")
        got = implied_vol(px, S, 1000.0, T, r, q, "call",
                          max_vol_error=math.inf)
        vega_raw = bs_greeks(S, 1000.0, T, r, got, q, "call").vega * 100.0
        dvol = 0.005 / vega_raw if vega_raw > 0.0 else math.inf
        print(f"      {true_vol:>5.2f} {px:>12.3e} {got:>10.6f} {dvol:>11.2e}")
    print("    Two of these are the same root from different inputs, and not")
    print("    one of them is pinned down to better than thousands of vol")
    print("    points. A returned float implies a precision none of them has.")


if __name__ == "__main__":
    import sys

    if "--plot" in sys.argv:
        for path in plot_greeks(dark="--dark" in sys.argv):
            print("wrote", path)
    else:
        verify()

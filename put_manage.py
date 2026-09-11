"""Managing short premium early, instead of holding to expiry.

    python put_manage.py                 # the comparison table
    python put_manage.py --take 0.5      # a different profit target

THE QUESTION. `put_write_backtest.py` holds every position to expiry and
measured the 25d/10d put credit spread at 1.76% CAGR, with `142 of 5009
trades a total loss`. Those total losses happen in the final week, where a
short put's gamma turns convex and a move that would have been survivable at
20 DTE is fatal at 3. Closing at a fraction of maximum profit is the standard
response among premium sellers, and it was untested here.

This changes EXACTLY ONE THING. Same entry dates, same strikes, same vol
model, same cadence. Only the exit differs. Capital freed by an early close
sits in cash until the next scheduled entry -- it is NOT redeployed, because
assuming instant redeployment would credit the variant with returns that
require a second, untested decision.

TWO HONEST COSTS, both of which the hold-to-expiry version avoids for free:

1. **Closing pays the spread again.** `build_trades` charges a half-spread per
   leg at entry and nothing at expiry, "because the position is simply allowed
   to expire". A managed exit is a real trade: one half-spread for the naked
   put, two for the vertical. Charging entry-only would be the single easiest
   way to manufacture an improvement here, so it is charged explicitly and the
   sensitivity is printed.

2. **The exit is MODEL-PRICED.** The hold-to-expiry design's great virtue is
   that the payoff is measured on real closes, so the tails owe nothing to the
   vol model. That virtue is LOST here: an early close is priced with
   `unrealized()`, whose own docstring calls it "model-dependent by
   construction". So a managed result is a weaker class of evidence than the
   number it is being compared against, and the comparison says so out loud
   rather than putting them in the same column and hoping.

Standard library only, as everything in this directory is.
"""

from __future__ import annotations

import argparse
import statistics as st
from dataclasses import dataclass
from datetime import date

import vrp_data
from put_write_backtest import (
    CALIBRATED, HOLD, Trade, ann_vol, build_trades, cagr, max_drawdown,
    put_mid, sharpe,
)


def load_rows() -> list[dict]:
    """The same cached SPY+VIX series the parent backtest uses."""
    return vrp_data.build()

TAKE_PROFIT = 0.50
HALF_SPREAD = 0.05


@dataclass
class Managed:
    """One trade, exited early or held to expiry, with the reason recorded."""

    trade: Trade
    pnl: float                 # per share, net of the exit cost
    held_days: int
    exit_reason: str           # "target" | "expiry"
    exit_cost: float

    @property
    def ret(self) -> float:
        return self.pnl / self.trade.K_short

    @property
    def ret_on_risk(self) -> float:
        return self.pnl / self.trade.max_risk


def legs(t: Trade) -> int:
    return 1 if t.K_long is None else 2


def structure_value(t: Trade, spy: float, vix: float, T: float) -> float:
    """What it would cost to buy the structure back, per share, at model mid."""
    if T <= 0.0:
        payoff = max(t.K_short - spy, 0.0)
        if t.K_long is not None:
            payoff -= max(t.K_long - spy, 0.0)
        return payoff
    value = put_mid(spy, t.K_short, T, vix, t.vm)
    if t.K_long is not None:
        value -= put_mid(spy, t.K_long, T, vix, t.vm)
    return value


def manage(t: Trade, rows: list[dict], ords: list[int],
           take_profit: float = TAKE_PROFIT,
           half_spread: float = HALF_SPREAD) -> Managed:
    """Walk the position forward; close the first day the target is reached.

    The target is on the CREDIT, which is the convention: "take 50%" means buy
    the structure back for half what you sold it for. Checked once per day at
    the close, never intraday -- an intraday touch would need data this study
    does not have, and assuming it would be lookahead.
    """
    target_value = (1.0 - take_profit) * t.credit
    exit_cost = legs(t) * half_spread

    for k in range(t.i + 1, t.j):
        T = (ords[t.j] - ords[k]) / 365.0
        value = structure_value(t, rows[k]["spy"], rows[k]["vix"], T)
        if value <= target_value:
            return Managed(trade=t, pnl=t.credit - value - exit_cost,
                           held_days=k - t.i, exit_reason="target",
                           exit_cost=exit_cost)

    # Never hit the target: expires, and expiry costs nothing.
    return Managed(trade=t, pnl=t.credit - t.payoff, held_days=t.j - t.i,
                   exit_reason="expiry", exit_cost=0.0)


def equity_curve(managed: list[Managed], hold: int = HOLD) -> list[float]:
    """One non-overlapping track: enter every `hold` rows, compound the result.

    The same single-track construction the parent file uses for vol and Sharpe,
    for the reason recorded there -- a 21-sleeve ladder settles one sleeve a
    day and its stale marks smooth the curve, reading ~20% high on Sharpe.
    """
    by_i = {m.trade.i: m for m in managed}
    curve, equity, i = [1.0], 1.0, 0
    while i in by_i:
        equity *= 1.0 + by_i[i].ret
        curve.append(equity)
        i += hold
    return curve


def summarize(label: str, managed: list[Managed], years: float,
              hold: int = HOLD) -> dict:
    curve = equity_curve(managed, hold)
    rets = [curve[k + 1] / curve[k] - 1.0 for k in range(len(curve) - 1)]
    per_year = 252.0 / hold
    wins = sum(1 for m in managed if m.pnl > 0)
    total_loss = sum(1 for m in managed
                     if m.trade.max_risk > 0
                     and m.pnl <= -0.999 * m.trade.max_risk)
    early = [m for m in managed if m.exit_reason == "target"]
    return {
        "label": label,
        "n": len(managed),
        "cagr": cagr(curve[0], curve[-1], (len(curve) - 1) / per_year),
        "vol": ann_vol(rets) if len(rets) > 2 else float("nan"),
        "sharpe": sharpe(rets) * (per_year ** 0.5) if len(rets) > 2 else float("nan"),
        "maxdd": max_drawdown(curve),
        "win": wins / len(managed) if managed else float("nan"),
        "total_losses": total_loss,
        "early_pct": len(early) / len(managed) if managed else float("nan"),
        "mean_days": st.fmean([m.held_days for m in early]) if early else float("nan"),
        "mean_ret": st.fmean([m.ret for m in managed]) if managed else float("nan"),
    }


def row(s: dict) -> str:
    return (f"  {s['label']:<34}{s['cagr']:>7.2%}{s['vol']:>8.1%}"
            f"{s['sharpe']:>8.2f}{s['maxdd']:>8.1%}{s['win']:>7.0%}"
            f"{s['total_losses']:>8}{s['early_pct']:>8.0%}{s['mean_days']:>7.1f}")


def verify() -> int:
    """Numerical checks. The load-bearing one is [1]."""
    ok = fail = 0

    def check(label, cond, detail=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  ok   {label}")
        else:
            fail += 1
            print(f"  FAIL {label}  {detail}")

    rows = load_rows()
    ords = [date.fromisoformat(r["date"]).toordinal() for r in rows]
    spread = build_trades(rows, 0.25, 0.10, CALIBRATED, HALF_SPREAD)

    print("[1] a target that can never trigger REPRODUCES hold-to-expiry")
    # take_profit > 1 means the buy-back price would have to be negative.
    never = [manage(t, rows, ords, take_profit=1.5) for t in spread[:400]]
    worst = max(abs(m.pnl - m.trade.pnl) for m in never)
    check(f"identical P&L on 400 trades (worst {worst:.2e})", worst < 1e-12)
    check("all of them report an expiry exit",
          all(m.exit_reason == "expiry" for m in never))
    check("and pay no exit cost", all(m.exit_cost == 0.0 for m in never))

    print("\n[2] the exit cost is charged, and scales with the legs")
    naked = build_trades(rows, 0.25, None, CALIBRATED, HALF_SPREAD)
    mn = [m for m in (manage(t, rows, ords, 0.5) for t in naked[:400])
          if m.exit_reason == "target"]
    ms = [m for m in (manage(t, rows, ords, 0.5) for t in spread[:400])
          if m.exit_reason == "target"]
    check("naked put pays one half-spread",
          all(abs(m.exit_cost - HALF_SPREAD) < 1e-12 for m in mn), str(len(mn)))
    check("the vertical pays two",
          all(abs(m.exit_cost - 2 * HALF_SPREAD) < 1e-12 for m in ms), str(len(ms)))

    print("\n[3] an early exit cannot beat the credit it collected")
    every = [manage(t, rows, ords, 0.5) for t in spread]
    check("no trade profits more than its credit",
          all(m.pnl <= m.trade.credit + 1e-12 for m in every))
    check("no trade loses more than its max risk",
          all(m.pnl >= -m.trade.max_risk - 1e-9 for m in every))
    check("every exit is inside the holding window",
          all(1 <= m.held_days <= HOLD for m in every))

    print("\n[4] a tighter target exits sooner and more often")
    tight = [manage(t, rows, ords, 0.25) for t in spread]
    loose = [manage(t, rows, ords, 0.75) for t in spread]
    n_t = sum(1 for m in tight if m.exit_reason == "target")
    n_l = sum(1 for m in loose if m.exit_reason == "target")
    check(f"25% target triggers more often than 75% ({n_t} vs {n_l})", n_t > n_l)
    d_t = st.fmean([m.held_days for m in tight if m.exit_reason == "target"])
    d_l = st.fmean([m.held_days for m in loose if m.exit_reason == "target"])
    check(f"and sooner ({d_t:.1f}d vs {d_l:.1f}d)", d_t < d_l)

    print("\n[5] structure_value agrees with the payoff at expiry")
    t0 = spread[0]
    for spy in (t0.K_short - 20, t0.K_short - 1, t0.K_short + 20):
        v = structure_value(t0, spy, 0.20, 0.0)
        expected = max(t0.K_short - spy, 0.0) - max(t0.K_long - spy, 0.0)
        check(f"S={spy:.0f}: value {v:.4f} == intrinsic {expected:.4f}",
              abs(v - expected) < 1e-12)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", type=float, default=TAKE_PROFIT)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        return verify()

    rows = load_rows()
    ords = [date.fromisoformat(r["date"]).toordinal() for r in rows]
    years = (ords[-1] - ords[0]) / 365.25
    print(f"{len(rows)} sessions, {rows[0]['date']} .. {rows[-1]['date']}"
          f"  ({years:.1f} years)\n")

    print(f"  {'':<34}{'CAGR':>7}{'vol':>8}{'Sharpe':>8}{'maxDD':>8}"
          f"{'win%':>7}{'total':>8}{'early':>8}{'days':>7}")
    out = []
    for name, ld in (("put-write 25d", None), ("put spread 25d/10d", 0.10)):
        trades = build_trades(rows, 0.25, ld, CALIBRATED, HALF_SPREAD)
        held = [Managed(t, t.pnl, HOLD, "expiry", 0.0) for t in trades]
        out.append(summarize(f"{name}  hold to expiry", held, years))
        print(row(out[-1]))
        for take in (0.25, 0.50, 0.75):
            man = [manage(t, rows, ords, take) for t in trades]
            out.append(summarize(f"{name}  take {take:.0%}", man, years))
            print(row(out[-1]))
        print()

    print("  'total' counts trades losing ~100% of max risk. 'early' is the share")
    print("  closed at the target; 'days' their mean holding period.")
    print("\n  THE EXIT IS MODEL-PRICED. Hold-to-expiry settles on real closes and")
    print("  owes the vol model nothing; every managed row prices its exit with")
    print("  the same model that set the entry credit. A managed number is")
    print("  therefore WEAKER EVIDENCE than the row above it, not merely a")
    print("  different number, and the two are not interchangeable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

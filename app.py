"""
Options Desk — a small tool for trading stock options.

Three questions, in order:
  1. Is anything worth trading?      -> Today
  2. Which strike, and at what price? -> Trade
  3. Am I actually any good at this?  -> Journal

Deliberately nothing else. No stock screener, no pattern scanner, no sentiment
gauge. Those are research tools; this is for placing trades.

Run locally:  streamlit run app.py
"""

import math
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Options Desk", page_icon="◆", layout="wide",
                   initial_sidebar_state="collapsed")

VERSION = "v7"
DATA = Path(__file__).parent / "data"
FNO_DIR = DATA / "fno"
HIST_DIR = DATA / "history"
MEETINGS = DATA / "board_meetings.csv"
IV_HIST = DATA / "iv_history.csv"
LOTS = DATA / "lot_sizes.csv"
JOURNAL = DATA / "journal.csv"

RISK_FREE = 0.065

# Round-trip costs. Editable in the UI — rates change and differ by broker.
COSTS = {
    "brokerage_per_order": 20.0,
    "stt_sell_pct": 0.10,
    "exchange_pct": 0.035,
    "sebi_per_cr": 10.0,
    "stamp_buy_pct": 0.003,
    "gst_pct": 18.0,
    "slippage_pct": 1.0,
}


# ============================================================== PRICING ======

def _cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs(spot, strike, t, vol, is_call=True, r=RISK_FREE):
    """Black-Scholes price. Falls back to intrinsic value at or past expiry."""
    if t <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return max((spot - strike) if is_call else (strike - spot), 0.0)
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    disc = math.exp(-r * t)
    if is_call:
        return spot * _cdf(d1) - strike * disc * _cdf(d2)
    return strike * disc * _cdf(-d2) - spot * _cdf(-d1)


def implied_vol(price, spot, strike, t, is_call=True):
    """
    Volatility that reproduces the observed price, by bisection.

    Returns NaN where no solution exists — a price below intrinsic value, a
    contract that did not trade, an expiry too close. Substituting a default
    here would make every downstream number confidently wrong, which is worse
    than showing nothing.
    """
    for v in (price, spot, strike, t):
        if v is None or v != v:
            return float("nan")
    if price <= 0 or spot <= 0 or strike <= 0 or t <= 1 / 365:
        return float("nan")
    intrinsic = max((spot - strike) if is_call else (strike - spot), 0.0)
    if price < intrinsic - 1e-6:
        return float("nan")
    if bs(spot, strike, t, 5.0, is_call) < price:
        return float("nan")

    lo, hi = 1e-4, 5.0
    for _ in range(70):
        mid = (lo + hi) / 2
        if bs(spot, strike, t, mid, is_call) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    v = (lo + hi) / 2
    return v * 100 if 0.001 < v < 4.99 else float("nan")


def greeks(spot, strike, t, vol, is_call=True, r=RISK_FREE):
    """Delta, gamma, theta per calendar day, vega per volatility point."""
    if t <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return {k: float("nan") for k in ("delta", "gamma", "theta", "vega")}
    sq = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t) / (vol * sq)
    d2 = d1 - vol * sq
    disc = math.exp(-r * t)
    pdf = _pdf(d1)
    theta_y = -(spot * pdf * vol) / (2 * sq)
    theta_y += (-r * strike * disc * _cdf(d2)) if is_call else (r * strike * disc * _cdf(-d2))
    return {"delta": _cdf(d1) if is_call else _cdf(d1) - 1,
            "gamma": pdf / (spot * vol * sq),
            "theta": theta_y / 365.0,
            "vega": spot * pdf * sq / 100.0}


def spot_for_premium(target, spot, strike, days_left, vol, is_call):
    """Underlying price at which the option is worth `target`."""
    lo, hi = spot * 0.4, spot * 2.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (bs(mid, strike, days_left / 365.0, vol, is_call) < target) == is_call:
            lo = mid
        else:
            hi = mid
        if hi - lo < spot * 1e-5:
            break
    return (lo + hi) / 2


def round_trip_cost(entry, exit_px, qty, cfg=None):
    """Full cost of one options round trip, including slippage both ways."""
    c = {**COSTS, **(cfg or {})}
    buy, sell = entry * qty, exit_px * qty
    brokerage = c["brokerage_per_order"] * 2
    exch = (buy + sell) * c["exchange_pct"] / 100
    total = (brokerage
             + sell * c["stt_sell_pct"] / 100
             + exch
             + (buy + sell) * c["sebi_per_cr"] / 1e7
             + buy * c["stamp_buy_pct"] / 100
             + (brokerage + exch) * c["gst_pct"] / 100
             + (buy + sell) * c["slippage_pct"] / 100)
    return total


# ================================================================ DATA =======

@st.cache_data(ttl=1800)
def load_fno() -> pd.DataFrame:
    """Latest stored F&O snapshot — every contract on every underlying."""
    if not FNO_DIR.exists():
        return pd.DataFrame()
    files = sorted(FNO_DIR.glob("*.csv.gz"))
    if not files:
        return pd.DataFrame()
    try:
        df = pd.read_csv(files[-1], skipinitialspace=True)
    except Exception:
        return pd.DataFrame()
    df = df[df["DATE"] == df["DATE"].max()].copy()
    for c in ("STRIKE", "CLOSE", "PREV_CLOSE", "UNDERLYING", "OI", "CHG_OI", "VOLUME"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "EXPIRY" in df.columns:
        df["EXPIRY_DT"] = pd.to_datetime(df["EXPIRY"], errors="coerce")
    return df


@st.cache_data(ttl=1800)
def load_history() -> pd.DataFrame:
    """Daily closes for the underlyings, used for the typical-move comparison."""
    if not HIST_DIR.exists():
        return pd.DataFrame()
    frames = []
    for f in sorted(HIST_DIR.glob("*.csv.gz")):
        try:
            frames.append(pd.read_csv(f, skipinitialspace=True,
                                      usecols=lambda c: c in
                                      ("DATE", "SYMBOL", "CLOSE", "HIGH", "LOW")))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    for c in ("CLOSE", "HIGH", "LOW"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df.dropna(subset=["DATE", "CLOSE"]).sort_values(["SYMBOL", "DATE"])


@st.cache_data(ttl=1800)
def load_meetings() -> pd.DataFrame:
    if not MEETINGS.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(MEETINGS, skipinitialspace=True)
        df["MEETING_DATE"] = pd.to_datetime(df["MEETING_DATE"], errors="coerce",
                                            dayfirst=True)
        return df.dropna(subset=["MEETING_DATE"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def load_lots() -> dict:
    if not LOTS.exists():
        return {}
    try:
        df = pd.read_csv(LOTS, skipinitialspace=True)
        return {str(r["SYMBOL"]): int(r["LOT_SIZE"]) for _, r in df.iterrows()
                if pd.notna(r["LOT_SIZE"])}
    except Exception:
        return {}


JOURNAL_TEXT = ["date", "symbol", "type", "expiry", "why", "status", "notes"]
JOURNAL_NUM = ["strike", "lots", "entry", "target", "stop", "exit"]
JOURNAL_COLS = ["date", "symbol", "strike", "type", "expiry", "lots", "entry",
                "target", "stop", "why", "status", "exit", "notes"]


def load_journal() -> pd.DataFrame:
    """
    The journal with column types pinned.

    An empty text column round-trips through CSV as float NaN, which then
    clashes with a text column config in the editor and raises. Types are set
    explicitly here rather than inferred.
    """
    if JOURNAL.exists():
        try:
            df = pd.read_csv(JOURNAL, skipinitialspace=True)
        except Exception:
            df = pd.DataFrame(columns=JOURNAL_COLS)
    else:
        df = pd.DataFrame(columns=JOURNAL_COLS)

    for c in JOURNAL_COLS:
        if c not in df.columns:
            df[c] = None
    for c in JOURNAL_TEXT:
        df[c] = df[c].fillna("").astype(str).replace("nan", "")
    for c in JOURNAL_NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[JOURNAL_COLS]


# ============================================================ ANALYSIS =======

def typical_move(hist: pd.DataFrame, symbol: str, days: int) -> float:
    """
    Median absolute move over `days`, from stored history.

    Absolute, because an option buyer is paid for magnitude in either
    direction. Returns NaN rather than a guess when history is short.
    """
    g = hist[hist["SYMBOL"] == symbol]
    closes = g["CLOSE"].dropna().reset_index(drop=True)
    if len(closes) < days + 40:
        return float("nan")
    moves = ((closes.shift(-days) / closes - 1) * 100).dropna().abs()
    return float(moves.median()) if len(moves) else float("nan")


def daily_range(hist: pd.DataFrame, symbol: str) -> float:
    """Typical daily move as a percentage — used to sanity-check stops."""
    g = hist[hist["SYMBOL"] == symbol].tail(60)
    closes = g["CLOSE"].dropna()
    if len(closes) < 20:
        return float("nan")
    return float((closes.pct_change().abs().median()) * 100)


@st.cache_data(ttl=3600)
def find_candidates(min_oi: int = 200000, max_dte: int = 45) -> pd.DataFrame:
    """
    Underlyings where options look cheap for the move the stock actually makes.

    The comparison is the whole idea. An option buyer needs the move to beat
    the premium, so a stock priced for 4% that routinely does 7% is a better
    starting point than one that merely looks bullish. There is no view on
    direction here — that part the data cannot supply.
    """
    fno, hist = load_fno(), load_history()
    if fno.empty:
        return pd.DataFrame()

    opts = fno[fno["OPT"].isin(["CE", "PE"])] if "OPT" in fno.columns else fno
    today = pd.Timestamp(datetime.now().date())
    snap = pd.Timestamp(fno["DATE"].iloc[0])

    events = {}
    mtg = load_meetings()
    if not mtg.empty and "SYMBOL" in mtg.columns:
        soon = mtg[(mtg["MEETING_DATE"] >= today)
                   & (mtg["MEETING_DATE"] <= today + pd.Timedelta(days=max_dte))]
        for _, r in soon.iterrows():
            s = str(r["SYMBOL"]).upper()
            if s not in events or r["MEETING_DATE"] < events[s]:
                events[s] = r["MEETING_DATE"]

    rows = []
    for sym, g in opts.groupby("SYMBOL"):
        oi = g["OI"].sum()
        if oi < min_oi:
            continue
        exps = sorted(x for x in g["EXPIRY_DT"].dropna().unique()
                      if pd.Timestamp(x) > snap)
        if not exps:
            continue
        near = pd.Timestamp(exps[0])
        dte = (near - snap).days
        if not 3 <= dte <= max_dte:
            continue

        chain = g[g["EXPIRY_DT"] == near]
        spot_s = chain["UNDERLYING"].dropna()
        if spot_s.empty:
            continue
        spot = float(spot_s.iloc[0])

        atm = chain.iloc[(chain["STRIKE"] - spot).abs().argsort()[:1]]["STRIKE"]
        if atm.empty:
            continue
        k = float(atm.iloc[0])
        ce = chain[(chain["OPT"] == "CE") & (chain["STRIKE"] == k)]["CLOSE"]
        pe = chain[(chain["OPT"] == "PE") & (chain["STRIKE"] == k)]["CLOSE"]
        if ce.empty or pe.empty or pd.isna(ce.iloc[0]) or pd.isna(pe.iloc[0]):
            continue

        straddle = float(ce.iloc[0]) + float(pe.iloc[0])
        implied = (straddle / spot) * 100
        actual = typical_move(hist, sym, dte) if not hist.empty else float("nan")

        ev = events.get(sym.upper())
        rows.append({
            "Symbol": sym, "Spot": spot, "Expiry": near.date(),
            "Days": dte, "ATM strike": k,
            "Options price a move of": implied,
            "It usually moves": actual,
            "Cheapness": implied / actual if actual == actual and actual else float("nan"),
            "Results in": (ev - today).days if ev is not None else None,
            "Open interest": oi,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    score = pd.Series(0.0, index=out.index)
    score += (out["Cheapness"] < 1.0).fillna(False).astype(float) * 3
    score += (out["Cheapness"] < 0.8).fillna(False).astype(float) * 2
    ev = out["Results in"]
    score += (ev.notna() & (ev <= out["Days"])).astype(float) * 3
    score += ev.notna().astype(float) * 1
    score += (out["Open interest"] > out["Open interest"].median()).astype(float)
    out["Score"] = score
    return out.sort_values(["Score", "Cheapness"], ascending=[False, True])


def pick_strikes(fno: pd.DataFrame, symbol: str, expiry, opt_type: str,
                 spot: float, snap, n: int = 5) -> pd.DataFrame:
    """
    Strikes near the money with the numbers that decide between them.

    Delta doubles as a rough chance of finishing in the money, and "move
    needed" is what actually separates a cheap far strike from a costly near
    one. Contracts whose price yields no implied volatility are dropped rather
    than shown with an invented one.
    """
    g = fno[(fno["SYMBOL"] == symbol) & (fno["OPT"] == opt_type)
            & (fno["EXPIRY_DT"] == pd.Timestamp(expiry))].copy()
    if g.empty:
        return pd.DataFrame()

    dte = max((pd.Timestamp(expiry) - pd.Timestamp(snap)).days, 1)
    t = dte / 365.0
    is_call = opt_type == "CE"
    g = g.iloc[(g["STRIKE"] - spot).abs().argsort()[:n * 2]].sort_values("STRIKE")

    rows = []
    for _, r in g.iterrows():
        k, prem = float(r["STRIKE"]), float(r["CLOSE"])
        iv = implied_vol(prem, spot, k, t, is_call)
        if iv != iv:
            continue
        gk = greeks(spot, k, t, iv / 100, is_call)
        be = k + prem if is_call else k - prem
        rows.append({
            "Strike": k, "Premium": prem, "IV %": iv,
            "Delta": gk["delta"],
            "Chance of finishing ITM": abs(gk["delta"]) * 100,
            "Breakeven": be,
            "Move needed %": ((be - spot) / spot) * 100,
            "Theta/day": gk["theta"],
            "OI": r.get("OI", float("nan")),
        })
    out = pd.DataFrame(rows)
    return out.iloc[(out["Strike"] - spot).abs().argsort()[:n]].sort_values("Strike") \
        if not out.empty else out


def build_plan(spot, strike, entry, dte, iv, is_call, qty, hold,
               target_pct, stop_pct, support=None, resistance=None, cfg=None):
    """Entry, target and stop as premium levels, with the move each requires."""
    vol = max(iv, 0.1) / 100
    days_left = max(dte - hold, 0)

    def at_level(level):
        return bs(level, strike, days_left / 365.0, vol, is_call)

    def move_for(prem):
        lvl = spot_for_premium(prem, spot, strike, days_left, vol, is_call)
        return lvl, ((lvl - spot) / spot) * 100

    rows = [{"Plan": "Entry", "Premium": entry, "Stock at": spot,
             "Stock move %": 0.0, "P&L after costs": 0.0, "Return %": 0.0}]

    items = [("Target (premium +%d%%)" % target_pct, entry * (1 + target_pct / 100)),
             ("Stop (premium -%d%%)" % stop_pct, entry * (1 - stop_pct / 100))]
    lvl_t = resistance if is_call else support
    lvl_s = support if is_call else resistance
    if lvl_t is not None and lvl_t == lvl_t:
        items.append(("Target (chart level)", at_level(lvl_t)))
    if lvl_s is not None and lvl_s == lvl_s:
        items.append(("Stop (chart level)", at_level(lvl_s)))

    for label, prem in items:
        if prem is None or prem != prem:
            continue
        prem = max(prem, 0.05)
        lvl, move = move_for(prem)
        net = (prem - entry) * qty - round_trip_cost(entry, prem, qty, cfg)
        rows.append({"Plan": label, "Premium": prem, "Stock at": lvl,
                     "Stock move %": move, "P&L after costs": net,
                     "Return %": (net / (entry * qty)) * 100 if entry else float("nan")})
    return pd.DataFrame(rows)


# ========================================================== OPEN POSITIONS ===
# What you are holding right now, and what it is doing.
#
# One real limitation: the stored data is a previous close, so during market
# hours every premium here is yesterday's. Rather than present stale numbers
# as live, the current premium is something you type in from your broker. An
# app that looked live and was not would be worse than one that asks.

def position_status(row, current_premium, lots_map, today=None):
    """
    One open position, marked to a premium you supply.

    Everything is computed from your own entry, target and stop rather than
    re-derived, because those are what you actually committed to.
    """
    today = today or pd.Timestamp(datetime.now().date())
    sym = str(row.get("symbol", ""))
    qty = int(row.get("lots") or 0) * lots_map.get(sym, 1)

    try:
        entry = float(row.get("entry") or 0)
        target = float(row.get("target") or 0)
        stop = float(row.get("stop") or 0)
    except (TypeError, ValueError):
        return None
    if entry <= 0 or qty <= 0:
        return None

    cost = entry * qty
    value = current_premium * qty
    gross = value - cost
    net = gross - round_trip_cost(entry, current_premium, qty)

    try:
        expiry = pd.Timestamp(str(row.get("expiry")))
        days_left = max((expiry - today).days, 0)
    except Exception:
        expiry, days_left = pd.NaT, None

    to_target = ((target / current_premium) - 1) * 100 if current_premium else float("nan")
    to_stop = ((stop / current_premium) - 1) * 100 if current_premium else float("nan")

    # How much of the distance from entry to target or stop has been covered.
    if current_premium >= entry and target > entry:
        progress = min((current_premium - entry) / (target - entry), 1.0) * 100
        toward = "target"
    elif current_premium < entry and entry > stop:
        progress = min((entry - current_premium) / (entry - stop), 1.0) * 100
        toward = "stop"
    else:
        progress, toward = 0.0, "—"

    flags = []
    if current_premium <= stop:
        flags.append("STOP HIT")
    if current_premium >= target and target > 0:
        flags.append("TARGET HIT")
    if days_left is not None and days_left <= 5:
        flags.append(f"{days_left}d to expiry")

    return {
        "Symbol": sym, "Contract": f"{row.get('strike','')} {row.get('type','')}",
        "Expiry": expiry.date() if pd.notna(expiry) else "—",
        "Days left": days_left, "Lots": row.get("lots"), "Qty": qty,
        "Entry": entry, "Now": current_premium,
        "Target": target, "Stop": stop,
        "Cost": cost, "Value": value, "P&L net": net,
        "Return %": (net / cost) * 100 if cost else float("nan"),
        "To target %": to_target, "To stop %": to_stop,
        "Progress %": progress, "Toward": toward,
        "Flags": ", ".join(flags) if flags else "",
        "Why": str(row.get("why", ""))[:90],
    }


# ======================================================= PRE-TRADE CHECKS ====
# Five questions before placing. None of them predicts anything; each one
# catches a specific way trades go wrong that is obvious afterwards and
# invisible beforehand.

def pre_trade_checks(entry, qty, capital, stop_move_pct, daily_range_pct,
                     breakeven_move_pct, typical_move_pct, days_to_expiry,
                     event_days, why_written, lots=1) -> list:
    """Returns (passed, title, detail) for each check."""
    out = []
    cost = entry * qty
    risk_pct = (cost / capital * 100) if capital else float("nan")

    # Indian lot sizes are large enough that one lot of a cheap option can
    # exceed a sensible position limit on a small account. That is worth
    # saying outright rather than reporting as a generic failure — the answer
    # is a different contract or more capital, not a smaller size.
    one_lot_cost = cost / max(lots, 1) if lots else cost
    one_lot_pct = (one_lot_cost / capital * 100) if capital else float("nan")
    detail = (f"This costs Rs {cost:,.0f}, which is {risk_pct:.1f}% of your "
              f"capital. A bought option can go to zero, so the whole premium "
              f"is the risk — not the notional.")
    if one_lot_pct == one_lot_pct and one_lot_pct > 2.0:
        detail += (f" Note that a single lot already costs Rs {one_lot_cost:,.0f} "
                   f"({one_lot_pct:.1f}%), so no size fixes this — you need a "
                   f"cheaper contract, a further strike, or more capital before "
                   f"this instrument fits the account.")
    out.append((risk_pct <= 2.0 if risk_pct == risk_pct else False,
                "Position is 2% of capital or less", detail))

    ok_stop = (stop_move_pct == stop_move_pct and daily_range_pct == daily_range_pct
               and abs(stop_move_pct) > daily_range_pct)
    out.append((
        ok_stop,
        "Stop is wider than the stock's daily noise",
        f"Your stop triggers on a {abs(stop_move_pct):.2f}% move; this stock "
        f"typically moves {daily_range_pct:.1f}% a day. A stop inside that "
        f"range gets hit on ordinary days."
        if stop_move_pct == stop_move_pct else "Could not be computed."))

    ok_be = (breakeven_move_pct == breakeven_move_pct
             and typical_move_pct == typical_move_pct
             and abs(breakeven_move_pct) <= typical_move_pct)
    out.append((
        ok_be,
        "Breakeven is within a typical move",
        f"You need {abs(breakeven_move_pct):.1f}% to break even; this stock "
        f"usually covers {typical_move_pct:.1f}% over {days_to_expiry} days. "
        f"Needing more than usual is not impossible, but the odds are against it."
        if breakeven_move_pct == breakeven_move_pct else "Could not be computed."))

    has_event = event_days is not None and event_days == event_days
    out.append((
        has_event,
        "There is a reason for it to move",
        f"Results in {int(event_days)} days."
        if has_event else
        "No results date found before expiry. Buying premium with no catalyst "
        "means paying for time and hoping — decay is certain, the move is not."))

    out.append((
        bool(str(why_written).strip()),
        "You have written down why",
        "One sentence, before you know the outcome. Six months from now this "
        "is the only honest record of your reasoning."))

    return out


# ======================================================= UNEXPLAINED FALLS ===
# The idea: a stock dragged down by the market, with nothing wrong at the
# company, tends to come back — while one that fell on its own news may not.
#
# Making that precise needs two things. First a market return, taken as the
# median move across every stock that traded, which needs no index feed and is
# harder to distort than a cap-weighted index. Second a beta, because a stock
# that habitually moves 1.5x the market falling 1.5x is not unusual at all —
# it is behaving normally. What is left after removing beta times the market
# is the part the market does NOT explain, and that is what this looks for.

def market_returns(hist: pd.DataFrame) -> pd.Series:
    """Daily median return across all stocks — a robust market proxy."""
    if hist.empty:
        return pd.Series(dtype=float)
    wide = hist.pivot_table(index="DATE", columns="SYMBOL", values="CLOSE",
                            aggfunc="last").sort_index()
    return wide.pct_change().median(axis=1).dropna()


def compute_betas(hist: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    """
    Beta and residual volatility per stock, against the market proxy.

    Residual volatility matters as much as beta: a 6% unexplained fall is
    remarkable in a placid stock and unremarkable in a jumpy one, so the
    ranking below uses how many of its OWN residual standard deviations the
    move represents rather than the raw percentage.
    """
    if hist.empty:
        return pd.DataFrame()

    wide = hist.pivot_table(index="DATE", columns="SYMBOL", values="CLOSE",
                            aggfunc="last").sort_index().tail(window + 1)
    rets = wide.pct_change().dropna(how="all")
    mkt = rets.median(axis=1)
    if len(rets) < 40:
        return pd.DataFrame()

    rows = []
    mkt_var = mkt.var()
    for sym in rets.columns:
        r = rets[sym].dropna()
        if len(r) < 40 or mkt_var == 0:
            continue
        aligned = mkt.reindex(r.index)
        beta = float(r.cov(aligned) / mkt_var)
        resid = r - beta * aligned
        rows.append({"Symbol": sym, "Beta": beta,
                     "Residual vol %": float(resid.std() * 100),
                     "Days": len(r)})
    return pd.DataFrame(rows)


def unexplained_falls(hist: pd.DataFrame, lookback: int = 5,
                      min_sigma: float = 1.5, window: int = 120) -> pd.DataFrame:
    """
    Stocks that fell further than the market and their own beta explain.

    Ranked by standard deviations of their own residual, not by raw percentage
    — otherwise the list is simply the most volatile stocks every time.
    """
    betas = compute_betas(hist, window)
    if betas.empty:
        return pd.DataFrame()

    wide = hist.pivot_table(index="DATE", columns="SYMBOL", values="CLOSE",
                            aggfunc="last").sort_index()
    if len(wide) <= lookback:
        return pd.DataFrame()

    period = (wide.iloc[-1] / wide.iloc[-lookback - 1] - 1) * 100
    mkt_period = float(period.median())

    rows = []
    for _, b in betas.iterrows():
        sym = b["Symbol"]
        if sym not in period.index or period[sym] != period[sym]:
            continue
        actual = float(period[sym])
        expected = b["Beta"] * mkt_period
        resid = actual - expected
        # Scale the residual to the holding period, then express in sigmas.
        sigma = b["Residual vol %"] * (lookback ** 0.5)
        if sigma <= 0:
            continue
        rows.append({
            "Symbol": sym,
            "Now": float(wide[sym].iloc[-1]),
            f"{lookback}d move %": actual,
            "Market did %": mkt_period,
            "Beta": b["Beta"],
            "Expected %": expected,
            "Unexplained %": resid,
            "Sigmas": resid / sigma,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[out["Sigmas"] <= -abs(min_sigma)]
    return out.sort_values("Sigmas")


def test_reversion(hist: pd.DataFrame, lookback: int = 5, horizon: int = 5,
                   min_sigma: float = 1.5, window: int = 120) -> dict:
    """
    Does the idea actually work on your own stored history?

    Finds every past instance of an unexplained fall and measures what the
    next `horizon` days did, against the base rate for the same stocks over
    the same windows. Overlapping windows are accounted for — otherwise a
    hundred observations look like a hundred independent ones when they are
    closer to twenty.
    """
    if hist.empty:
        return {}

    wide = hist.pivot_table(index="DATE", columns="SYMBOL", values="CLOSE",
                            aggfunc="last").sort_index()
    if len(wide) < window + lookback + horizon + 10:
        return {"error": f"needs about {window + lookback + horizon + 10} "
                         f"trading days of history, has {len(wide)}"}

    rets = wide.pct_change()
    mkt = rets.median(axis=1)

    hits, base = [], []
    step = max(horizon // 2, 1)
    for i in range(window, len(wide) - horizon, step):
        hist_slice = rets.iloc[i - window:i]
        mkt_slice = mkt.iloc[i - window:i]
        mvar = mkt_slice.var()
        if mvar == 0 or hist_slice.empty:
            continue

        period = (wide.iloc[i] / wide.iloc[i - lookback] - 1) * 100
        mkt_period = float(period.median())
        fwd = (wide.iloc[i + horizon] / wide.iloc[i] - 1) * 100
        base.extend(fwd.dropna().tolist())

        for sym in wide.columns:
            r = hist_slice[sym].dropna()
            if len(r) < 40:
                continue
            beta = float(r.cov(mkt_slice.reindex(r.index)) / mvar)
            resid_vol = float((r - beta * mkt_slice.reindex(r.index)).std() * 100)
            if resid_vol <= 0:
                continue
            if period.get(sym) != period.get(sym):
                continue
            resid = float(period[sym]) - beta * mkt_period
            if resid / (resid_vol * (lookback ** 0.5)) <= -abs(min_sigma):
                f = fwd.get(sym)
                if f == f:
                    hits.append(float(f))

    if not hits:
        return {"error": "no historical instances found at that threshold"}

    h = pd.Series(hits)
    b = pd.Series(base)
    eff_n = max(len(h) / max(horizon, 1), 1)
    se = h.std() / (eff_n ** 0.5) if len(h) > 1 else float("inf")
    edge = float(h.median() - b.median())

    return {
        "n": len(h), "effective_n": eff_n,
        "median": float(h.median()), "base_median": float(b.median()),
        "edge": edge,
        "hit_rate": float((h > 0).mean() * 100),
        "base_hit": float((b > 0).mean() * 100),
        "significant": bool(se != float("inf") and abs(edge) > 2 * se and eff_n >= 5),
        "threshold": 2 * se if se != float("inf") else float("nan"),
    }


# ============================================================= PATTERNS ======
# A reference built from YOUR stored data rather than textbook drawings.
#
# Every pattern guide shows the examples that worked. This shows real recent
# instances from the market you trade, and — more usefully — what happened
# afterwards across every instance in the history, including the failures.
# A pattern with a 52% hit rate looks very different once you can see that.

PATTERNS = {
    "Breakout above 20-day high": {
        "what": "Price closes above the highest point of the previous 20 days.",
        "why": "The idea is that everyone who bought in that range is now in "
               "profit, so there is no trapped supply overhead.",
        "catch": "Breakouts fail often, and a breakout without a volume "
                 "increase fails more. It also fires after the move has begun.",
    },
    "Breakdown below 20-day low": {
        "what": "Price closes below the lowest point of the previous 20 days.",
        "why": "The mirror image — everyone who bought recently is now losing.",
        "catch": "In a range-bound market this is simply the bottom of the "
                 "range, which is where it stops falling.",
    },
    "Higher highs and higher lows": {
        "what": "Both recent peaks and recent troughs are above the previous "
                "ones.",
        "why": "The textbook definition of an uptrend, and the most durable of "
               "these patterns because it describes structure rather than a "
               "single bar.",
        "catch": "It describes what has already happened. Trends end without "
                 "warning, and the structure only breaks after the fact.",
    },
    "Consolidation": {
        "what": "The last 20 days span less than half the range of the last 60.",
        "why": "Volatility contracts before it expands, so a tight range often "
               "precedes a larger move.",
        "catch": "It says nothing about direction. A tight range breaks both "
                 "ways, and traders routinely assume it will break the way they "
                 "are positioned.",
    },
    "Above 200-day average": {
        "what": "Price is above its 200-day moving average.",
        "why": "The line most institutional mandates watch. Roughly separates "
               "stocks in long uptrends from the rest.",
        "catch": "Slow. A stock can lose a third of its value while still "
                 "above the line.",
    },
    "Oversold (RSI below 30)": {
        "what": "Relative Strength Index under 30 — recent losses heavily "
                "outweigh recent gains.",
        "why": "Conventionally read as a bounce candidate.",
        "catch": "RSI stays below 30 for months in a falling stock. Oversold "
                 "means fast, not finished.",
    },
    "Near 52-week high": {
        "what": "Within 2% of the highest close in the past year.",
        "why": "Momentum research repeatedly finds recent winners continue "
               "over medium horizons.",
        "catch": "Also where a stock tops out. The pattern cannot tell you "
                 "which of the two you are looking at.",
    },
    "Fell more than the market explains": {
        "what": "A drop well beyond what this stock's beta and the market "
                "move account for.",
        "why": "If the market did not cause it and no news did either, there "
               "may be nothing wrong.",
        "catch": "Company news moves price before it reaches a headline. "
                 "Unexplained is not the same as unfair.",
    },
}


def detect_pattern_series(closes: pd.Series, highs=None, lows=None) -> dict:
    """
    Boolean series for each pattern across a stock's whole history.

    Computed for every day, not just today, so instances can be found in the
    past and their outcomes measured.
    """
    c = closes.dropna()
    if len(c) < 210:
        return {}
    h = highs.reindex(c.index).fillna(c) if highs is not None else c
    lo = lows.reindex(c.index).fillna(c) if lows is not None else c

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))

    hi20 = h.rolling(20).max().shift(1)
    lo20 = lo.rolling(20).min().shift(1)
    sma200 = c.rolling(200).mean()
    hi52 = c.rolling(250).max()

    r20 = h.rolling(20).max() - lo.rolling(20).min()
    r60 = h.rolling(60).max() - lo.rolling(60).min()

    sh = h.rolling(11, center=True).max()
    sl = lo.rolling(11, center=True).min()
    swing_hi = (h == sh)
    swing_lo = (lo == sl)

    return {
        "Breakout above 20-day high": c > hi20,
        "Breakdown below 20-day low": c < lo20,
        "Higher highs and higher lows": (
            (h.rolling(20).max() > h.rolling(20).max().shift(20))
            & (lo.rolling(20).min() > lo.rolling(20).min().shift(20))),
        "Consolidation": r20 < r60 * 0.45,
        "Above 200-day average": c > sma200,
        "Oversold (RSI below 30)": rsi < 30,
        "Near 52-week high": c >= hi52 * 0.98,
        "_swing_hi": swing_hi, "_swing_lo": swing_lo,
    }


def pattern_outcomes(hist: pd.DataFrame, pattern: str, horizon: int = 10,
                     max_symbols: int = 120) -> dict:
    """
    Every historical instance of a pattern, and what followed.

    The base rate — what the same stocks did over the same windows regardless
    of the pattern — is reported alongside. Without it a 55% hit rate sounds
    impressive when the stock rose 55% of the time anyway.
    """
    if hist.empty:
        return {}
    wide = hist.pivot_table(index="DATE", columns="SYMBOL", values="CLOSE",
                            aggfunc="last").sort_index()
    highs = hist.pivot_table(index="DATE", columns="SYMBOL", values="HIGH",
                             aggfunc="last").sort_index() \
        if "HIGH" in hist.columns else None
    lows = hist.pivot_table(index="DATE", columns="SYMBOL", values="LOW",
                            aggfunc="last").sort_index() \
        if "LOW" in hist.columns else None

    symbols = list(wide.columns)[:max_symbols]
    after, base, examples = [], [], []

    for sym in symbols:
        c = wide[sym].dropna()
        if len(c) < 210 + horizon:
            continue
        flags = detect_pattern_series(
            c,
            highs[sym] if highs is not None and sym in highs else None,
            lows[sym] if lows is not None and sym in lows else None)
        if pattern not in flags:
            continue

        fwd = (c.shift(-horizon) / c - 1) * 100
        base.extend(fwd.dropna().tolist())

        sig = flags[pattern].fillna(False)
        # Only the day a pattern first appears, so a condition that persists
        # for weeks is not counted as dozens of separate signals.
        onset = sig & ~sig.shift(1).fillna(False)
        hit_dates = c.index[onset.reindex(c.index).fillna(False)]
        for d in hit_dates:
            v = fwd.get(d)
            if v == v:
                after.append(float(v))
                examples.append((sym, d, float(v)))

    if not after:
        return {}
    a, b = pd.Series(after), pd.Series(base)
    eff_n = max(len(a) / max(horizon, 1), 1)
    se = a.std() / (eff_n ** 0.5) if len(a) > 1 else float("inf")
    edge = float(a.median() - b.median())
    return {
        "n": len(a), "effective_n": eff_n,
        "median": float(a.median()), "base_median": float(b.median()),
        "edge": edge, "hit": float((a > 0).mean() * 100),
        "base_hit": float((b > 0).mean() * 100),
        "significant": bool(se != float("inf") and abs(edge) > 2 * se and eff_n >= 5),
        "examples": sorted(examples, key=lambda x: x[1], reverse=True)[:8],
    }


def strikes_for_candidate(fno: pd.DataFrame, symbol: str, expiry, spot: float,
                          snap, typical_pct: float, lot: int, n: int = 4):
    """
    Calls and puts around the money, with the one thing that decides between
    them: whether the breakeven is inside what this stock actually covers.

    A strike needing a 9% move on a stock that typically manages 5% is cheap
    for a reason. Flagging that is a statement about arithmetic, not direction
    — both the call and the put are shown, because which one to buy depends on
    a view the data cannot supply.
    """
    g = fno[(fno["SYMBOL"] == symbol)
            & (fno["EXPIRY_DT"] == pd.Timestamp(expiry))
            & (fno["OPT"].isin(["CE", "PE"]))].copy()
    if g.empty:
        return pd.DataFrame()

    dte = max((pd.Timestamp(expiry) - pd.Timestamp(snap)).days, 1)
    t = dte / 365.0
    rows = []
    for opt in ("CE", "PE"):
        side = g[g["OPT"] == opt]
        if side.empty:
            continue
        near = side.iloc[(side["STRIKE"] - spot).abs().argsort()[:n]]
        for _, r in near.sort_values("STRIKE").iterrows():
            k, prem = float(r["STRIKE"]), float(r["CLOSE"])
            iv = implied_vol(prem, spot, k, t, opt == "CE")
            if iv != iv:
                continue
            gk = greeks(spot, k, t, iv / 100, opt == "CE")
            be = k + prem if opt == "CE" else k - prem
            need = ((be - spot) / spot) * 100
            rows.append({
                "Type": opt, "Strike": k, "Premium": prem,
                "Cost per lot": prem * lot,
                "Move needed %": need,
                "Reachable": (abs(need) <= typical_pct
                              if typical_pct == typical_pct else None),
                "Chance ITM %": abs(gk["delta"]) * 100,
                "IV %": iv,
                "OI": r.get("OI", float("nan")),
            })
    return pd.DataFrame(rows)


# ============================================================== STYLING ======

st.markdown("""
<style>
  .stApp { background: #0e1116; }
  h1,h2,h3 { letter-spacing:-0.02em; font-weight:640; }
  .note { font-size:0.8rem; color:#c9a227; background:#1e1a10;
          border-left:2px solid #b8860b; padding:0.65rem 0.95rem; margin:0.6rem 0; }
  .bad  { font-size:0.8rem; color:#f0c8ca; background:#2a1618;
          border-left:2px solid #e5484d; padding:0.65rem 0.95rem; margin:0.6rem 0; }
  .card { border-left:3px solid #2fbf71; padding:0.7rem 1rem; background:#151a21;
          margin:0.4rem 0; color:#e6eaef; }
</style>
""", unsafe_allow_html=True)


@contextmanager
def safe(name):
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        st.error(f"{name}: {type(exc).__name__}: {exc}")
        with st.expander("Details"):
            st.code(traceback.format_exc())


def tint(df, cols):
    present = [c for c in cols if c in df.columns]

    def shade(v):
        if v is None or v != v:
            return "color:#6b7684"
        return "color:#2fbf71" if v > 0 else "color:#e5484d" if v < 0 else ""
    return df.style.map(shade, subset=present).format(precision=2, na_rep="—")


# ================================================================= APP =======

fno = load_fno()
hist = load_history()


def data_banner():
    """
    A one-line statement of what the app can actually see.

    Without this an empty tab is ambiguous: no data, or something broken?
    Stating it on every screen removes the guesswork.
    """
    bits = []
    bits.append(f"F&O: {fno['SYMBOL'].nunique()} stocks" if not fno.empty
                else "F&O: **missing**")
    if hist.empty:
        bits.append("History: **missing**")
    else:
        bits.append(f"History: {hist['DATE'].nunique()} days, "
                    f"{hist['SYMBOL'].nunique()} stocks")
    st.caption(f"Options Desk {VERSION} · " + " · ".join(bits))
    if fno.empty or hist.empty:
        st.markdown(
            '<div class="note">Some data has not loaded. Run the <i>Collect</i> '
            'workflow in your repo, then reboot the app from share.streamlit.io '
            '— Streamlit does not pick up newly committed files on its own. '
            'The Data tab shows exactly what is present.</div>',
            unsafe_allow_html=True)

(tab_today, tab_falls, tab_trade, tab_open, tab_journal, tab_patterns,
 tab_data) = st.tabs(["Today", "Unfair falls", "Trade", "Open positions",
                      "Journal", "Patterns", "Data"])


with tab_today, safe("Today"):
    st.markdown("### Is anything worth trading?")
    data_banner()
    st.caption(
        "Stock options where the market is pricing a smaller move than the "
        "stock usually makes, with a results date coming. No view on direction "
        "— that part is yours."
    )

    if fno.empty:
        st.info("No data yet. Open the Data tab.")
    else:
        c1, c2 = st.columns(2)
        max_dte = c1.slider("Expiry within (days)", 7, 60, 45, key="t_dte")
        min_oi = c2.number_input("Minimum open interest", value=200000,
                                 step=50000, key="t_oi")

        cands = find_candidates(int(min_oi), int(max_dte))
        if cands.empty:
            st.warning("Nothing passed the filters.")
        else:
            top = cands.head(5)
            st.caption(f"Top 5 of {len(cands)} liquid underlyings.")

            lots_map = load_lots()
            for _, r in top.iterrows():
                cheap = r["Cheapness"]
                ev = r["Results in"]
                bits = []
                if cheap == cheap:
                    bits.append(
                        f"Options price a <b>{r['Options price a move of']:.1f}%</b> "
                        f"move; it usually does <b>{r['It usually moves']:.1f}%</b>")
                if ev is not None and pd.notna(ev):
                    bits.append(f"results in <b>{int(ev)} days</b>"
                                + (" — before expiry" if ev <= r["Days"] else ""))
                st.markdown(
                    f'<div class="card"><b style="font-size:1.05rem;">{r["Symbol"]}</b>'
                    f'<span style="color:#8b95a1;"> · Rs {r["Spot"]:,.2f} · expiry '
                    f'{r["Expiry"]} ({int(r["Days"])}d)</span><br>'
                    f'<span style="color:#8b95a1;font-size:0.87rem;">'
                    f'{". ".join(bits) if bits else "Liquid, no catalyst found"}.'
                    f'</span></div>', unsafe_allow_html=True)

                lot = lots_map.get(r["Symbol"], 1)
                ladder = strikes_for_candidate(
                    fno, r["Symbol"], r["Expiry"], r["Spot"],
                    fno["DATE"].iloc[0], r["It usually moves"], lot)

                with st.expander(f"Strikes for {r['Symbol']}"):
                    if ladder.empty:
                        st.caption("No strikes here could be priced — they may "
                                   "not have traded.")
                    else:
                        reachable = ladder[ladder["Reachable"] == True]  # noqa: E712
                        if len(reachable):
                            best_c = reachable[reachable["Type"] == "CE"]
                            best_p = reachable[reachable["Type"] == "PE"]
                            picks = []
                            if len(best_c):
                                b = best_c.loc[best_c["Move needed %"].abs().idxmin()]
                                picks.append(
                                    f"<b>If you think it rises:</b> the "
                                    f"{b['Strike']:,.0f} CE at Rs {b['Premium']:.2f} "
                                    f"needs only {abs(b['Move needed %']):.1f}% "
                                    f"— Rs {b['Cost per lot']:,.0f} a lot")
                            if len(best_p):
                                b = best_p.loc[best_p["Move needed %"].abs().idxmin()]
                                picks.append(
                                    f"<b>If you think it falls:</b> the "
                                    f"{b['Strike']:,.0f} PE at Rs {b['Premium']:.2f} "
                                    f"needs only {abs(b['Move needed %']):.1f}% "
                                    f"— Rs {b['Cost per lot']:,.0f} a lot")
                            st.markdown(
                                '<div class="card">'
                                + "<br><br>".join(picks)
                                + f'<br><br><span style="color:#8b95a1;'
                                  f'font-size:0.84rem;">Both are listed because '
                                  f'nothing here tells you which way it goes. '
                                  f'These are simply the strikes whose breakeven '
                                  f'sits inside the {r["It usually moves"]:.1f}% '
                                  f'this stock typically covers in {int(r["Days"])} '
                                  f'days — the ones that can work, not the ones '
                                  f'that will.</span></div>',
                                unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f'<div class="note">No strike here has a '
                                f'breakeven within the {r["It usually moves"]:.1f}% '
                                f'this stock typically covers. Every one needs a '
                                f'bigger-than-usual move to pay off, which is a '
                                f'reason to look elsewhere rather than to pick '
                                f'the cheapest.</div>', unsafe_allow_html=True)

                        show = ladder.copy()
                        show["Reachable"] = show["Reachable"].map(
                            {True: "yes", False: "no"}).fillna("—")
                        st.dataframe(
                            tint(show, ["Move needed %"]),
                            column_config={
                                "Strike": st.column_config.NumberColumn(format="%.0f"),
                                "Premium": st.column_config.NumberColumn(format="%.2f"),
                                "Cost per lot": st.column_config.NumberColumn(format="%.0f"),
                                "Move needed %": st.column_config.NumberColumn(
                                    format="%.1f%%",
                                    help="How far the stock must travel for this "
                                         "to break even at expiry."),
                                "Reachable": st.column_config.TextColumn(
                                    help="Whether that move is within what this "
                                         "stock typically covers by expiry."),
                                "Chance ITM %": st.column_config.NumberColumn(format="%.0f%%"),
                                "IV %": st.column_config.NumberColumn(format="%.1f"),
                            },
                            use_container_width=True, hide_index=True)
                        st.caption(
                            f"Lot size {lot}. Take a strike to the **Trade** tab "
                            "for entry, target and stop in premium terms.")

            with st.expander("Full list"):
                st.dataframe(
                    tint(cands, ["Options price a move of", "It usually moves"]),
                    use_container_width=True, hide_index=True)

            st.markdown(
                '<div class="note"><b>Cheapness below 1 means the market is '
                'pricing less movement than the stock normally delivers.</b> That '
                'favours buying options — but only if something causes the move. '
                'It also says nothing about direction, so a cheap option is cheap '
                'both ways. One caveat: "usually moves" is measured over ordinary '
                'periods, and stocks genuinely move more around results, so '
                'options that look expensive before earnings may simply be priced '
                'correctly.</div>', unsafe_allow_html=True)


with tab_falls, safe("Unfair falls"):
    st.markdown("### Which stocks fell more than the market explains?")
    data_banner()
    st.markdown(
        "Your idea, made precise. A stock that falls because the whole market "
        "fell is a different thing from one that falls on its own news — the "
        "first has nothing wrong with it. Separating them needs **beta**: a "
        "stock that habitually moves 1.5x the market dropping 1.5x is behaving "
        "normally, not being punished."
    )

    if hist.empty:
        st.info("Needs price history. Open the Data tab.")
    else:
        f1, f2, f3 = st.columns(3)
        look = f1.select_slider("Fall measured over (days)", options=[3, 5, 10, 20],
                                value=5, key="uf_look")
        sig = f2.slider("How unusual (sigmas)", 1.0, 3.0, 1.5, 0.25, key="uf_sig",
                        help="How many of the stock's own residual standard "
                             "deviations the unexplained part represents. Higher "
                             "means rarer.")
        fno_only = f3.checkbox("Only stocks with options", value=True, key="uf_fno")

        with st.spinner("Estimating betas and residuals…"):
            falls = unexplained_falls(hist, look, sig)

        if fno_only and not falls.empty and not fno.empty:
            tradable = set(fno["SYMBOL"].dropna().unique())
            falls = falls[falls["Symbol"].isin(tradable)]

        if falls.empty:
            st.info("Nothing fell unusually today at that threshold. Lower the "
                    "sigma slider, or check back after a down day.")
        else:
            st.caption(f"{len(falls)} stocks. The market moved "
                       f"{falls['Market did %'].iloc[0]:+.2f}% over {look} days.")

            for _, r in falls.head(6).iterrows():
                st.markdown(
                    f'<div class="card"><b style="font-size:1.05rem;">{r["Symbol"]}</b>'
                    f'<span style="color:#8b95a1;"> · Rs {r["Now"]:,.2f}</span><br>'
                    f'<span style="color:#8b95a1;font-size:0.87rem;">'
                    f'Fell <b>{r[f"{look}d move %"]:.1f}%</b> while the market did '
                    f'{r["Market did %"]:+.1f}%. With a beta of {r["Beta"]:.2f} it '
                    f'"should" have fallen {r["Expected %"]:.1f}%, so '
                    f'<b>{abs(r["Unexplained %"]):.1f}%</b> of the drop is '
                    f'unexplained — {abs(r["Sigmas"]):.1f} standard deviations of '
                    f'its own normal wobble.</span></div>',
                    unsafe_allow_html=True)

            with st.expander("Full list"):
                st.dataframe(tint(falls, [f"{look}d move %", "Market did %",
                                          "Unexplained %", "Sigmas"]),
                             use_container_width=True, hide_index=True)

            st.markdown(
                '<div class="note"><b>This finds the fall, not the reason.</b> '
                'An unexplained drop means the market did not cause it — it does '
                'not mean nothing caused it. Company news often moves the price '
                'before it reaches a headline, and this app has no news feed, so '
                'check the stock on your broker or the exchange filings page '
                'before assuming the fall was unfair. The ones worth trading are '
                'those where you look and genuinely find nothing.</div>',
                unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Does this actually work?")
        st.caption(
            "The strategy tested against your own stored history: every past "
            "instance of an unexplained fall, and what the following days did — "
            "measured against the base rate for the same stocks over the same "
            "windows."
        )

        if st.button("Test it", key="uf_test"):
            st.session_state["uf_ran"] = True

        if st.session_state.get("uf_ran"):
            h1, h2 = st.columns(2)
            horizon = h1.select_slider("Hold for (days)", options=[3, 5, 10, 20],
                                       value=5, key="uf_hor")
            test_sig = h2.slider("At what sigma", 1.0, 3.0, float(sig), 0.25,
                                 key="uf_tsig")

            with st.spinner("Walking through the history…"):
                res = test_reversion(hist, look, horizon, test_sig)

            if "error" in res:
                st.warning(res["error"])
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Instances found", res["n"])
                m2.metric(f"Median {horizon}d after", f"{res['median']:+.2f}%",
                          delta=f"{res['edge']:+.2f}% vs base")
                m3.metric("Went up", f"{res['hit_rate']:.0f}%",
                          delta=f"{res['hit_rate'] - res['base_hit']:+.0f}pp")
                m4.metric("Independent windows", f"{res['effective_n']:.0f}",
                          help="Overlapping windows share most of their days, so "
                               "this governs reliability rather than the raw count.")

                colour = "#2fbf71" if res["significant"] else "#c9a227"
                verdict = ("This beats the base rate by more than noise explains"
                           if res["significant"] else
                           "Not distinguishable from the base rate")
                st.markdown(
                    f'<div style="border-left:3px solid {colour};padding:0.7rem 1rem;'
                    f'background:#151a21;margin:0.5rem 0;color:#e6eaef;">'
                    f'<b>{verdict}.</b><br>'
                    f'<span style="color:#8b95a1;font-size:0.86rem;">'
                    f'After an unexplained fall these stocks did '
                    f'{res["median"]:+.2f}% over the next {horizon} days, against '
                    f'{res["base_median"]:+.2f}% for the same stocks generally — '
                    f'an edge of {res["edge"]:+.2f}% against a noise threshold of '
                    f'±{res["threshold"]:.2f}%.</span></div>',
                    unsafe_allow_html=True)

                st.markdown(
                    '<div class="note">Two things to hold onto whatever this '
                    'says. It is one market regime — the period you happen to '
                    'have collected — and mean reversion works until it does '
                    'not, which is usually when something genuinely was wrong. '
                    'And it takes no account of costs; at 3% round trip on '
                    'options, an edge under about 3% is not an edge you can '
                    'trade.</div>', unsafe_allow_html=True)


with tab_trade, safe("Trade"):
    st.markdown("### Which strike, and where do I get out?")
    data_banner()

    if fno.empty:
        st.info("No data yet. Open the Data tab.")
    else:
        opts = fno[fno["OPT"].isin(["CE", "PE"])]
        s1, s2, s3 = st.columns(3)
        sym = s1.selectbox("Stock", sorted(opts["SYMBOL"].dropna().unique()),
                           key="tr_sym")
        sub = opts[opts["SYMBOL"] == sym]
        exps = sorted(sub["EXPIRY_DT"].dropna().unique())
        labels = [pd.Timestamp(e).strftime("%d-%b-%Y") for e in exps]
        exp = exps[labels.index(s2.selectbox("Expiry", labels, key="tr_exp"))]
        view = s3.radio("Your view", ["Up (call)", "Down (put)"],
                        horizontal=True, key="tr_view")
        opt_type = "CE" if view.startswith("Up") else "PE"

        spot_s = sub["UNDERLYING"].dropna()
        spot = float(spot_s.iloc[0]) if len(spot_s) else float("nan")
        snap = pd.Timestamp(fno["DATE"].iloc[0])
        dte = max((pd.Timestamp(exp) - snap).days, 1)
        lot = load_lots().get(sym, 1)
        dr = daily_range(hist, sym) if not hist.empty else float("nan")

        st.caption(f"Spot {spot:,.2f} · {dte} days to expiry · lot size {lot}"
                   + (f" · this stock typically moves {dr:.1f}% a day"
                      if dr == dr else ""))

        ladder = pick_strikes(fno, sym, exp, opt_type, spot, snap)
        if ladder.empty:
            st.warning("No strikes here could be priced — they may not have "
                       "traded. Try another expiry.")
        else:
            st.markdown("#### Choose a strike")
            st.dataframe(
                tint(ladder, ["Move needed %"]),
                column_config={
                    "Strike": st.column_config.NumberColumn(format="%.0f"),
                    "Premium": st.column_config.NumberColumn(format="%.2f"),
                    "Chance of finishing ITM": st.column_config.NumberColumn(
                        format="%.0f%%",
                        help="Delta used as a rough probability. Not exact, but "
                             "the right way to compare a cheap far strike "
                             "against a costly near one."),
                    "Move needed %": st.column_config.NumberColumn(
                        format="%.1f%%",
                        help="How far the stock must travel just to break even "
                             "at expiry."),
                    "Theta/day": st.column_config.NumberColumn(format="%.2f"),
                },
                use_container_width=True, hide_index=True)

            if dr == dr:
                reach = dr * math.sqrt(max(dte, 1))
                far = ladder[ladder["Move needed %"].abs() > reach]
                if len(far):
                    st.markdown(
                        f'<div class="note">Strikes needing more than '
                        f'<b>{reach:.1f}%</b> are asking for more than this stock '
                        f'typically covers in {dte} days: '
                        f'{", ".join(f"{int(k)}" for k in far["Strike"])}. They '
                        f'are cheap because they usually expire worthless.</div>',
                        unsafe_allow_html=True)

            st.divider()
            st.markdown("#### Your plan")

            p1, p2, p3, p4 = st.columns(4)
            strike = p1.selectbox("Strike", ladder["Strike"].tolist(), key="tr_k")
            row = ladder[ladder["Strike"] == strike].iloc[0]
            entry = p2.number_input("Price you'll pay",
                                    value=float(round(row["Premium"], 2)),
                                    min_value=0.05, step=0.05, key="tr_entry")
            lots = p3.number_input("Lots", value=1, min_value=1, step=1, key="tr_lots")
            hold = p4.slider("Days you'll hold", 1, max(dte, 2),
                             min(7, dte), key="tr_hold")

            qty = int(lots) * lot
            iv = float(row["IV %"])

            g1, g2 = st.columns(2)
            tgt = g1.number_input("Target: premium +%", value=50.0, step=10.0,
                                  key="tr_tgt")
            stp = g2.number_input("Stop: premium -%", value=30.0, step=5.0,
                                  key="tr_stp")

            support = resistance = None
            if not hist.empty:
                gh = hist[hist["SYMBOL"] == sym].tail(120)
                if len(gh) > 20 and "LOW" in gh.columns and gh["LOW"].notna().any():
                    lows = gh["LOW"].dropna()
                    highs = gh["HIGH"].dropna()
                    below = lows[lows < spot]
                    above = highs[highs > spot]
                    support = float(below.quantile(0.75)) if len(below) else None
                    resistance = float(above.quantile(0.25)) if len(above) else None

            plan = build_plan(spot, strike, entry, dte, iv, opt_type == "CE",
                              qty, hold, tgt, stp, support, resistance)

            k1, k2, k3 = st.columns(3)
            k1.metric("You pay", f"Rs {entry * qty:,.0f}")
            tr = plan[plan["Plan"].str.startswith("Target (premium")]
            sr = plan[plan["Plan"].str.startswith("Stop (premium")]
            if len(tr) and len(sr):
                gain = float(tr["P&L after costs"].iloc[0])
                loss = abs(float(sr["P&L after costs"].iloc[0]))
                k2.metric("If target hits", f"Rs {gain:+,.0f}")
                k3.metric("If stop hits", f"Rs {-loss:+,.0f}",
                          help=f"Reward to risk {gain / loss:.2f}" if loss else None)

            st.dataframe(
                tint(plan, ["Stock move %", "P&L after costs", "Return %"]),
                column_config={
                    "Premium": st.column_config.NumberColumn(
                        format="%.2f", help="The price to place your order at."),
                    "Stock at": st.column_config.NumberColumn(format="%.2f"),
                    "Stock move %": st.column_config.NumberColumn(format="%.2f%%"),
                    "P&L after costs": st.column_config.NumberColumn(format="%.0f"),
                    "Return %": st.column_config.NumberColumn(format="%.0f%%"),
                },
                use_container_width=True, hide_index=True)

            if len(sr) and dr == dr:
                stop_move = abs(float(sr["Stock move %"].iloc[0]))
                if stop_move < dr:
                    st.markdown(
                        f'<div class="bad"><b>Your {stp:.0f}% stop triggers on a '
                        f'{stop_move:.2f}% move in the stock — less than its '
                        f'normal daily range of {dr:.1f}%.</b> You would be '
                        f'stopped out on an ordinary day when nothing had gone '
                        f'wrong. Options lose value quickly on small adverse '
                        f'moves, so a percentage stop on premium is much tighter '
                        f'than it sounds. Widen it, or accept the stop is really '
                        f'a time stop.</div>', unsafe_allow_html=True)

            cost = round_trip_cost(entry, entry, qty)
            st.caption(
                f"Round-trip costs Rs {cost:,.0f} — {cost / (entry * qty) * 100:.1f}% "
                f"of what you pay, mostly slippage. Time decay is about "
                f"Rs {abs(row['Theta/day']) * qty:,.0f} a day if the stock does "
                f"nothing. Every premium here is a model value from the stored "
                f"close, not a live quote — check the bid and ask before acting."
            )

            st.divider()
            st.markdown("#### Before you place it")

            cap = st.number_input("Your capital (Rs)", value=500000.0,
                                  step=50000.0, key="tr_cap")
            why = st.text_input(
                "Why are you taking it?", key="tr_why",
                placeholder="e.g. results in 6 days, options cheap vs usual move")

            stop_move = float(sr["Stock move %"].iloc[0]) if len(sr) else float("nan")
            be_move = float(row["Move needed %"])
            typ = typical_move(hist, sym, dte) if not hist.empty else float("nan")
            ev_days = None
            mtg = load_meetings()
            if not mtg.empty and "SYMBOL" in mtg.columns:
                mine = mtg[(mtg["SYMBOL"].astype(str).str.upper() == sym.upper())
                           & (mtg["MEETING_DATE"] >= pd.Timestamp(datetime.now().date()))
                           & (mtg["MEETING_DATE"] <= pd.Timestamp(exp))]
                if not mine.empty:
                    ev_days = (mine["MEETING_DATE"].min()
                               - pd.Timestamp(datetime.now().date())).days

            checks = pre_trade_checks(entry, qty, cap, stop_move, dr, be_move,
                                      typ, dte, ev_days, why, int(lots))
            passed = sum(1 for ok, _, _ in checks if ok)

            for ok, title, detail in checks:
                mark = "✓" if ok else "✕"
                colour = "#2fbf71" if ok else "#e5484d"
                st.markdown(
                    f'<div style="padding:0.35rem 0;border-bottom:1px solid #1c222b;">'
                    f'<span style="color:{colour};font-weight:700;">{mark}</span> '
                    f'<b style="color:#e6eaef;">{title}</b><br>'
                    f'<span style="color:#8b95a1;font-size:0.84rem;margin-left:1.1rem;">'
                    f'{detail}</span></div>', unsafe_allow_html=True)

            if passed < 3:
                st.markdown(
                    f'<div class="bad"><b>{passed} of 5 checks pass.</b> None of '
                    'these predicts anything — each one catches a specific way '
                    'trades go wrong that is obvious afterwards and invisible '
                    'beforehand. Failing three or more usually means a different '
                    'strike, a later expiry, or no trade at all.</div>',
                    unsafe_allow_html=True)
            else:
                st.caption(f"{passed} of 5 checks pass.")

            with st.form("log_trade", clear_on_submit=True):
                if st.form_submit_button("Add to journal") and why.strip():
                    j = load_journal()
                    new = pd.DataFrame([{
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "symbol": sym, "strike": strike, "type": opt_type,
                        "expiry": pd.Timestamp(exp).strftime("%Y-%m-%d"),
                        "lots": lots, "entry": entry,
                        "target": round(entry * (1 + tgt / 100), 2),
                        "stop": round(entry * (1 - stp / 100), 2),
                        "why": why.strip(), "status": "open", "exit": "",
                        "notes": ""}])
                    out = pd.concat([j, new], ignore_index=True)
                    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
                    out.to_csv(JOURNAL, index=False)
                    st.success("Logged. Download it from the Journal tab to keep it.")


with tab_open, safe("Open positions"):
    st.markdown("### What am I holding?")
    data_banner()

    j = load_journal()
    open_rows = j[j["status"].astype(str).str.lower() != "closed"] if not j.empty \
        else pd.DataFrame()

    if open_rows.empty:
        st.info("Nothing open. Log a trade on the Trade tab.")
    else:
        st.caption(
            f"{len(open_rows)} open. Enter the current premium from your broker "
            "for each — the stored data is a previous close, so anything shown "
            "during market hours would be yesterday's price."
        )
        lots_map = load_lots()
        capital = st.number_input("Your capital (Rs)", value=500000.0,
                                  step=50000.0, key="op_cap",
                                  help="Used to show what share of it each "
                                       "position represents.")

        cards = []
        for i, (_, r) in enumerate(open_rows.iterrows()):
            label = f"{r.get('symbol','')} {r.get('strike','')} {r.get('type','')}"
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                now = c1.number_input(
                    f"{label} — premium now",
                    value=float(pd.to_numeric(r.get("entry"), errors="coerce") or 1.0),
                    min_value=0.05, step=0.05, key=f"op_now_{i}")
                s = position_status(r, now, lots_map)
                if s is None:
                    c2.warning("This entry is missing an entry price or lot count.")
                    continue
                cards.append(s)

                pnl_col = "#2fbf71" if s["P&L net"] >= 0 else "#e5484d"
                flag_html = (f'<span style="color:#e5484d;font-weight:600;">'
                             f'{s["Flags"]}</span><br>' if s["Flags"] else "")
                c2.markdown(
                    f'{flag_html}'
                    f'<span style="font-size:1.35rem;font-weight:650;'
                    f'color:{pnl_col};">Rs {s["P&L net"]:+,.0f}</span>'
                    f'<span style="color:#8b95a1;"> ({s["Return %"]:+.0f}% '
                    f'after costs)</span><br>'
                    f'<span style="color:#8b95a1;font-size:0.87rem;">'
                    f'Cost Rs {s["Cost"]:,.0f} · now worth Rs {s["Value"]:,.0f} · '
                    f'{s["Days left"] if s["Days left"] is not None else "?"} days '
                    f'to expiry<br>'
                    f'Target {s["Target"]:.2f} ({s["To target %"]:+.0f}% away) · '
                    f'Stop {s["Stop"]:.2f} ({s["To stop %"]:+.0f}% away)</span>',
                    unsafe_allow_html=True)
                if s["Toward"] != "—":
                    c2.progress(min(s["Progress %"] / 100, 1.0),
                                text=f"{s['Progress %']:.0f}% of the way to your "
                                     f"{s['Toward']}")
                if s["Why"]:
                    c2.caption(f"You wrote: {s['Why']}")

        if cards:
            summary = pd.DataFrame(cards)
            st.divider()
            t1, t2, t3, t4 = st.columns(4)
            total_cost = summary["Cost"].sum()
            total_pnl = summary["P&L net"].sum()
            t1.metric("Deployed", f"Rs {total_cost:,.0f}",
                      help=f"{total_cost / capital * 100:.1f}% of capital"
                           if capital else None)
            t2.metric("Open P&L", f"Rs {total_pnl:+,.0f}")
            t3.metric("Positions", len(summary))
            near = summary[summary["Days left"].fillna(99) <= 5]
            t4.metric("Expiring within 5 days", len(near))

            if total_cost > capital * 0.1 and capital:
                st.markdown(
                    f'<div class="note">You have {total_cost / capital * 100:.0f}% '
                    'of capital in open option positions. Bought options can go '
                    'to zero together — they are not independent bets when the '
                    'whole market moves against you.</div>',
                    unsafe_allow_html=True)
            if len(near):
                st.markdown(
                    f'<div class="note"><b>{len(near)} position(s) expire within '
                    'five days.</b> Time decay accelerates sharply in the final '
                    'week — an option that has not worked by now usually will '
                    'not, and the last days lose value fastest.</div>',
                    unsafe_allow_html=True)

            st.dataframe(
                tint(summary[["Symbol", "Contract", "Days left", "Entry", "Now",
                              "Target", "Stop", "Cost", "P&L net", "Return %",
                              "Flags"]], ["P&L net", "Return %"]),
                use_container_width=True, hide_index=True)

        st.caption("Mark positions closed on the Journal tab once you exit, so "
                   "they count toward your record.")


with tab_journal, safe("Journal"):
    st.markdown("### Am I actually any good at this?")
    data_banner()
    st.caption(
        "The only question that matters before sizing up. Log every trade with "
        "the reason, then let the numbers answer it."
    )

    j = load_journal()
    if j.empty:
        st.markdown(
            '<div class="card"><b>No trades logged yet — this is the empty '
            'state, not an error.</b><br><span style="color:#8b95a1;'
            'font-size:0.88rem;">Go to the <b>Trade</b> tab, pick a stock and '
            'strike, write one line about why, and press <i>Add to journal</i>. '
            'Paper trades count the same — you do not need to place money to '
            'find out whether your reasoning holds up.<br><br>After about thirty '
            'closed trades the win rate and average R at the top of this page '
            'become the only numbers that matter.</span></div>',
            unsafe_allow_html=True)
        st.caption("Journal file location: data/journal.csv in your repo.")
    else:
        d = j.copy()
        for c in ("entry", "exit", "lots", "stop", "target"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        lots_map = load_lots()
        d["qty"] = d.apply(
            lambda r: (r["lots"] or 0) * lots_map.get(str(r["symbol"]), 1), axis=1)
        closed = d[(d["status"].astype(str).str.lower() == "closed")
                   & d["exit"].notna()]

        if len(closed):
            closed = closed.copy()
            closed["P&L"] = ((closed["exit"] - closed["entry"]) * closed["qty"]
                             - closed.apply(lambda r: round_trip_cost(
                                 r["entry"], r["exit"], r["qty"]), axis=1))
            closed["R"] = (closed["exit"] - closed["entry"]) / \
                          (closed["entry"] - closed["stop"]).replace(0, float("nan"))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Closed trades", len(closed))
            m2.metric("Win rate", f"{(closed['P&L'] > 0).mean() * 100:.0f}%")
            m3.metric("Net P&L", f"Rs {closed['P&L'].sum():,.0f}")
            m4.metric("Average R", f"{closed['R'].mean():+.2f}",
                      help="Profit as a multiple of what you risked. Above +0.3 "
                           "across many trades is a real result.")

            if len(closed) < 20:
                st.markdown(
                    f'<div class="note"><b>{len(closed)} closed trades is not yet '
                    'a result.</b> A win rate over a handful of trades is noise, '
                    'and reading anything into it is how journals mislead. Thirty '
                    'or more before you draw conclusions, and before you '
                    'increase size.</div>', unsafe_allow_html=True)
        else:
            st.caption("No closed trades yet — set status to 'closed' and fill "
                       "in the exit price to score them.")

        st.markdown("**All trades**")
        edited = st.data_editor(
            j, num_rows="dynamic", use_container_width=True, key="j_edit",
            column_config={
                "date": st.column_config.TextColumn("Date", width="small"),
                "symbol": st.column_config.TextColumn("Stock", width="small"),
                "strike": st.column_config.NumberColumn("Strike", format="%.0f"),
                "type": st.column_config.SelectboxColumn("Type", options=["CE", "PE"]),
                "expiry": st.column_config.TextColumn("Expiry", width="small"),
                "lots": st.column_config.NumberColumn("Lots", format="%d"),
                "entry": st.column_config.NumberColumn("Entry", format="%.2f"),
                "target": st.column_config.NumberColumn("Target", format="%.2f"),
                "stop": st.column_config.NumberColumn("Stop", format="%.2f"),
                "why": st.column_config.TextColumn("Why I took it", width="large"),
                "status": st.column_config.SelectboxColumn(
                    "Status", options=["open", "closed"]),
                "exit": st.column_config.NumberColumn("Exit", format="%.2f"),
                "notes": st.column_config.TextColumn("Afterwards", width="large"),
            })
        c1, c2 = st.columns(2)
        if c1.button("Save changes"):
            JOURNAL.parent.mkdir(parents=True, exist_ok=True)
            edited.to_csv(JOURNAL, index=False)
            st.success("Saved for this session.")
        c2.download_button("Download journal.csv",
                           data=edited.to_csv(index=False).encode("utf-8"),
                           file_name="journal.csv", mime="text/csv",
                           help="Commit this to data/ in your repo to keep it — "
                                "hosted apps reset their disk on rebuild.")


with tab_patterns, safe("Patterns"):
    st.markdown("### Chart patterns, and whether they work")
    data_banner()
    st.markdown(
        "Every pattern guide shows the examples that worked. This finds real "
        "instances in **your own stored data** and reports what happened after "
        "all of them — including the failures — against the base rate for the "
        "same stocks over the same windows."
    )

    if hist.empty:
        st.markdown(
            '<div class="card"><b>No price history loaded — this is the empty '
            'state, not an error.</b><br><span style="color:#8b95a1;'
            'font-size:0.88rem;">These patterns are found in your own stored '
            'data, so there has to be some. Run the <i>Collect</i> workflow with '
            'a backfill of <b>400</b>, then reboot the app.<br><br>Each stock '
            'needs 210 trading days before patterns can be computed at all.'
            '</span></div>', unsafe_allow_html=True)
    else:
        days = hist["DATE"].nunique()
        if days < 260:
            st.markdown(
                f'<div class="note">Only {days} trading days stored. These '
                'patterns need 210 days per stock before they can be computed '
                'at all, and a year or more before the outcomes mean much. '
                'Run the collector with a larger backfill.</div>',
                unsafe_allow_html=True)

        pick = st.selectbox("Pattern", list(PATTERNS.keys()), key="pt_pick")
        info = PATTERNS[pick]

        st.markdown(
            f'<div class="card"><b>{pick}</b><br>'
            f'<span style="color:#8b95a1;font-size:0.9rem;">'
            f'<b>What it is:</b> {info["what"]}<br>'
            f'<b>Why people use it:</b> {info["why"]}<br>'
            f'<b>The catch:</b> {info["catch"]}</span></div>',
            unsafe_allow_html=True)

        if pick == "Fell more than the market explains":
            st.info("This one has its own tab — see **Unfair falls**, which "
                    "includes a test of it against your history.")
        else:
            horizon = st.select_slider("Measure what happened over (days)",
                                       options=[5, 10, 20], value=10, key="pt_hor")
            if st.button("Find real instances and outcomes", key="pt_go"):
                st.session_state["pt_ran"] = pick

            if st.session_state.get("pt_ran") == pick:
                with st.spinner("Scanning your history…"):
                    res = pattern_outcomes(hist, pick, horizon)

                if not res:
                    st.warning("No instances found — usually not enough history "
                               "per stock yet.")
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Instances", res["n"])
                    m2.metric(f"Median {horizon}d after", f"{res['median']:+.2f}%",
                              delta=f"{res['edge']:+.2f}% vs base")
                    m3.metric("Went up", f"{res['hit']:.0f}%",
                              delta=f"{res['hit'] - res['base_hit']:+.0f}pp")
                    m4.metric("Independent windows", f"{res['effective_n']:.0f}")

                    colour = "#2fbf71" if res["significant"] else "#c9a227"
                    verdict = ("Beats the base rate by more than noise explains"
                               if res["significant"] else
                               "Not distinguishable from the base rate")
                    st.markdown(
                        f'<div style="border-left:3px solid {colour};'
                        f'padding:0.7rem 1rem;background:#151a21;margin:0.5rem 0;'
                        f'color:#e6eaef;"><b>{verdict}.</b><br>'
                        f'<span style="color:#8b95a1;font-size:0.86rem;">'
                        f'After this pattern: {res["median"]:+.2f}% over '
                        f'{horizon} days, up {res["hit"]:.0f}% of the time. '
                        f'The same stocks generally: {res["base_median"]:+.2f}%, '
                        f'up {res["base_hit"]:.0f}% of the time.</span></div>',
                        unsafe_allow_html=True)

                    st.markdown("**Real recent instances**")
                    ex = pd.DataFrame(res["examples"],
                                      columns=["Symbol", "Date", f"{horizon}d after %"])
                    ex["Date"] = pd.to_datetime(ex["Date"]).dt.date
                    st.dataframe(tint(ex, [f"{horizon}d after %"]),
                                 use_container_width=True, hide_index=True)

                    if len(ex):
                        chart_sym = st.selectbox("Chart one of these",
                                                 ex["Symbol"].unique(), key="pt_chart")
                        row = ex[ex["Symbol"] == chart_sym].iloc[0]
                        g = hist[hist["SYMBOL"] == chart_sym].set_index("DATE")["CLOSE"]
                        when = pd.Timestamp(row["Date"])
                        window = g[(g.index >= when - pd.Timedelta(days=90))
                                   & (g.index <= when + pd.Timedelta(days=60))]
                        if len(window) > 5:
                            st.line_chart(window, height=280)
                            st.caption(
                                f"{chart_sym} around {row['Date']}, when the "
                                f"pattern appeared. It went "
                                f"{row[f'{horizon}d after %']:+.1f}% over the "
                                f"following {horizon} days. One instance is an "
                                f"anecdote — the numbers above are the evidence."
                            )

                    st.markdown(
                        '<div class="note">Read the <b>vs base</b> figures rather '
                        'than the headline ones. A pattern followed by +1.2% looks '
                        'good until you see the same stocks averaged +1.1% anyway. '
                        'And most of these will come back "not distinguishable" — '
                        'that is the honest result, and it is why patterns are '
                        'better used to organise what you are looking at than to '
                        'decide what to buy.</div>', unsafe_allow_html=True)


with tab_data, safe("Data"):
    st.markdown("### Data")
    st.caption("Everything here is collected nightly by a GitHub Action. "
               "A gap is a gap in the analysis.")

    def status(path, label, why):
        if path.is_dir():
            files = list(path.glob("*.csv.gz"))
            return {"Data": label, "Status": "ready" if files else "missing",
                    "Detail": f"{len(files)} file(s)", "Needed for": why}
        ok = path.exists()
        n = ""
        if ok:
            try:
                n = f"{len(pd.read_csv(path, skipinitialspace=True)):,} rows"
            except Exception:
                ok, n = False, "unreadable"
        return {"Data": label, "Status": "ready" if ok else "missing",
                "Detail": n or "not collected", "Needed for": why}

    rows = [status(FNO_DIR, "F&O chains", "Everything"),
            status(HIST_DIR, "Price history", "Typical move, stop sanity check"),
            status(MEETINGS, "Results calendar", "Catalysts"),
            status(LOTS, "Lot sizes", "Position sizing")]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if any(r["Status"] == "missing" for r in rows):
        st.markdown(
            '<div class="note">Run the <i>Collect</i> workflow in your repo\'s '
            'Actions tab. It gathers all of this in one pass and then repeats '
            'every weekday evening.</div>', unsafe_allow_html=True)
    else:
        st.success(f"All collected. Snapshot: {fno['DATE'].iloc[0] if not fno.empty else '—'}")

    st.caption(f"Options Desk {VERSION}. Delayed exchange data, model prices, "
               "not advice. Roughly nine in ten individual F&O traders in India "
               "lose money — SEBI has found this in three separate studies. "
               "Paper trade until your journal says otherwise.")

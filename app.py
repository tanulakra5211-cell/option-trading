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

VERSION = "v1"
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


def load_journal() -> pd.DataFrame:
    cols = ["date", "symbol", "strike", "type", "expiry", "lots", "entry",
            "target", "stop", "why", "status", "exit", "notes"]
    if JOURNAL.exists():
        try:
            df = pd.read_csv(JOURNAL, skipinitialspace=True)
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            return df[cols]
        except Exception:
            pass
    return pd.DataFrame(columns=cols)


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

tab_today, tab_trade, tab_journal, tab_data = st.tabs(
    ["Today", "Trade", "Journal", "Data"])


with tab_today, safe("Today"):
    st.markdown("### Is anything worth trading?")
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

            for _, r in top.iterrows():
                cheap = r["Cheapness"]
                ev = r["Results in"]
                bits = []
                if cheap == cheap:
                    bits.append(
                        f"Options price a **{r['Options price a move of']:.1f}%** "
                        f"move; it usually does **{r['It usually moves']:.1f}%**")
                if ev is not None and pd.notna(ev):
                    bits.append(f"results in **{int(ev)} days**"
                                + (" — before expiry" if ev <= r["Days"] else ""))
                st.markdown(
                    f'<div class="card"><b style="font-size:1.05rem;">{r["Symbol"]}</b>'
                    f'<span style="color:#8b95a1;"> · Rs {r["Spot"]:,.2f} · expiry '
                    f'{r["Expiry"]} ({int(r["Days"])}d)</span><br>'
                    f'<span style="color:#8b95a1;font-size:0.87rem;">'
                    f'{". ".join(bits) if bits else "Liquid, no catalyst found"}.'
                    f'</span></div>', unsafe_allow_html=True)

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


with tab_trade, safe("Trade"):
    st.markdown("### Which strike, and where do I get out?")

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

            with st.form("log_trade", clear_on_submit=True):
                st.markdown("**Log this trade**")
                why = st.text_input("Why are you taking it?",
                                    placeholder="e.g. results in 6 days, options "
                                                "cheap vs usual move")
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


with tab_journal, safe("Journal"):
    st.markdown("### Am I actually any good at this?")
    st.caption(
        "The only question that matters before sizing up. Log every trade with "
        "the reason, then let the numbers answer it."
    )

    j = load_journal()
    if j.empty:
        st.info("Nothing logged yet. Take a trade on the Trade tab, or paper "
                "trade — the numbers work the same either way.")
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
                "status": st.column_config.SelectboxColumn(
                    options=["open", "closed"]),
                "type": st.column_config.SelectboxColumn(options=["CE", "PE"]),
                "why": st.column_config.TextColumn("Why I took it", width="large"),
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

"""
Collector for Options Desk.

Gathers only what the app needs: F&O chains, underlying price history, lot
sizes and results dates. Runs on a GitHub Action each weekday evening and
commits the files back to the repo.

Runs from GitHub rather than the app because NSE rate-limits cloud hosts that
make many requests, and the app reading from disk is both faster and immune to
that.

Usage:
    python collect.py                 # daily
    python collect.py --backfill 400  # seed history once
"""

import argparse
import io
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).parent / "data"
FNO_DIR = ROOT / "fno"
HIST_DIR = ROOT / "history"
MEETINGS = ROOT / "board_meetings.csv"
LOTS = ROOT / "lot_sizes.csv"

BASE = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{BASE}/",
}
ARCHIVE_HEADERS = {"Referer": f"{BASE}/all-reports",
                   "Accept": "text/csv,application/zip,*/*"}

EQ_BHAV = ("https://nsearchives.nseindia.com/products/content/"
           "sec_bhavdata_full_{d:%d%m%Y}.csv")
FO_BHAV = ("https://nsearchives.nseindia.com/content/fo/"
           "BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")

HIST_COLS = ["DATE", "SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE", "PREV_CLOSE",
             "VOLUME", "DELIV_PER"]


def client(attempts: int = 3):
    """
    Cookie-primed client. NSE rejects requests without a prior page visit.

    Every call is guarded. The previous version let a connection error during
    priming escape, which killed the whole script with a raw traceback before
    a single line of useful output. Network calls to NSE fail often enough
    that this has to be handled rather than assumed.

    HTTP/2 is tried first because NSE fingerprints plain HTTP/1.1 clients, but
    HTTP/2 itself sometimes breaks from CI runners — so both are attempted.
    """
    last = ""
    for attempt in range(attempts):
        for use_h2 in (True, False):
            try:
                c = httpx.Client(http2=use_h2, headers=HEADERS, timeout=30.0,
                                 follow_redirects=True)
                r = c.get(BASE)
                if r.status_code >= 400:
                    last = f"homepage returned HTTP {r.status_code}"
                    c.close()
                    continue
                time.sleep(0.6)
                c.get(f"{BASE}/market-data/live-equity-market")
                time.sleep(0.6)
                print(f"  session established (HTTP/{'2' if use_h2 else '1.1'})")
                return c
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {str(exc)[:70]}"
                try:
                    c.close()
                except Exception:
                    pass
        if attempt < attempts - 1:
            wait = 3 * (attempt + 1)
            print(f"  handshake failed ({last}) — retrying in {wait}s")
            time.sleep(wait)

    print(f"  could not reach NSE after {attempts} attempts: {last}")
    return None


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from headers and values.

    NSE writes "SYMBOL, SERIES, ..." with a space after each comma. Under
    pandas 3 string columns are dtype 'str' rather than 'object', so a dtype
    check misses them and every value keeps a leading space — which silently
    breaks every filter downstream.
    """
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    return df


def fetch_equity(cl: httpx.Client, day: datetime):
    """One trading day of cash-market data, or None."""
    try:
        r = cl.get(EQ_BHAV.format(d=day), headers=ARCHIVE_HEADERS)
        if r.status_code != 200 or r.content[:1] == b"<":
            return None, f"HTTP {r.status_code}"
        df = clean(pd.read_csv(io.StringIO(r.text), skipinitialspace=True))
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}"

    if "SERIES" in df.columns:
        df = df[df["SERIES"].str.upper() == "EQ"]
    if "SYMBOL" not in df.columns or "CLOSE_PRICE" not in df.columns:
        return None, "unexpected columns"

    out = pd.DataFrame({
        "DATE": day.strftime("%Y-%m-%d"), "SYMBOL": df["SYMBOL"],
        "OPEN": pd.to_numeric(df.get("OPEN_PRICE"), errors="coerce"),
        "HIGH": pd.to_numeric(df.get("HIGH_PRICE"), errors="coerce"),
        "LOW": pd.to_numeric(df.get("LOW_PRICE"), errors="coerce"),
        "CLOSE": pd.to_numeric(df["CLOSE_PRICE"], errors="coerce"),
        "PREV_CLOSE": pd.to_numeric(df.get("PREV_CLOSE"), errors="coerce"),
        "VOLUME": pd.to_numeric(df.get("TTL_TRD_QNTY"), errors="coerce"),
        "DELIV_PER": pd.to_numeric(df.get("DELIV_PER"), errors="coerce"),
    }).dropna(subset=["CLOSE"])
    return out, "ok"


def fetch_fno(cl: httpx.Client, day: datetime):
    """Every F&O contract for one day, in a single request."""
    try:
        r = cl.get(FO_BHAV.format(d=day), headers=ARCHIVE_HEADERS)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            raw = clean(pd.read_csv(z.open(z.namelist()[0]), skipinitialspace=True))
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}"

    keep = {"TckrSymb": "SYMBOL", "FinInstrmTp": "TYPE", "XpryDt": "EXPIRY",
            "StrkPric": "STRIKE", "OptnTp": "OPT", "ClsPric": "CLOSE",
            "PrvsClsgPric": "PREV_CLOSE", "UndrlygPric": "UNDERLYING",
            "OpnIntrst": "OI", "ChngInOpnIntrst": "CHG_OI",
            "TtlTradgVol": "VOLUME", "NewBrdLotQty": "LOT_SIZE"}
    have = {k: v for k, v in keep.items() if k in raw.columns}
    if "TckrSymb" not in have:
        return None, f"unexpected columns: {list(raw.columns)[:8]}"

    df = raw[list(have)].rename(columns=have)
    for c in ("STRIKE", "CLOSE", "PREV_CLOSE", "UNDERLYING", "OI", "CHG_OI",
              "VOLUME", "LOT_SIZE"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["DATE"] = day.strftime("%Y-%m-%d")
    return df, "ok"


def store_month(directory: Path, df: pd.DataFrame, cols=None) -> None:
    """Append into a monthly file, replacing that date if already present."""
    if df is None or df.empty:
        return
    directory.mkdir(parents=True, exist_ok=True)
    date_str = df["DATE"].iloc[0]
    path = directory / f"{date_str[:7]}.csv.gz"
    if path.exists():
        prev = pd.read_csv(path, skipinitialspace=True)
        prev = prev[prev["DATE"] != date_str]
        df = pd.concat([prev, df], ignore_index=True)
    if cols:
        for c in cols:
            if c not in df.columns:
                df[c] = float("nan")
        df = df[cols]
    df.to_csv(path, index=False, compression="gzip")


def stored_dates(directory: Path) -> set:
    out = set()
    if not directory.exists():
        return out
    for f in directory.glob("*.csv.gz"):
        try:
            out.update(pd.read_csv(f, usecols=["DATE"],
                                   skipinitialspace=True)["DATE"].unique())
        except Exception:
            continue
    return out


def collect_meetings(cl: httpx.Client) -> int:
    """Board meeting dates — when results are approved, which is the catalyst."""
    start = datetime.now() - timedelta(days=3)
    end = datetime.now() + timedelta(days=45)
    url = (f"{BASE}/api/corporate-board-meetings?index=equities"
           f"&from_date={start:%d-%m-%Y}&to_date={end:%d-%m-%Y}")
    try:
        r = cl.get(url, headers={"Referer": f"{BASE}/companies-listing/"
                                            f"corporate-filings-board-meetings"})
        if r.status_code != 200:
            print(f"  meetings: HTTP {r.status_code}")
            return 0
        data = r.json()
        data = data.get("data", data) if isinstance(data, dict) else data
    except Exception as exc:  # noqa: BLE001
        print(f"  meetings: {type(exc).__name__}")
        return 0

    rows = [{"SYMBOL": i.get("bm_symbol") or i.get("symbol", ""),
             "COMPANY": i.get("sm_name") or i.get("company", ""),
             "MEETING_DATE": i.get("bm_date", ""),
             "PURPOSE": str(i.get("bm_purpose", ""))[:120]}
            for i in (data or []) if isinstance(i, dict)]
    if not rows:
        return 0

    df = pd.DataFrame(rows)
    MEETINGS.parent.mkdir(parents=True, exist_ok=True)
    if MEETINGS.exists():
        prev = pd.read_csv(MEETINGS, skipinitialspace=True, dtype=str)
        df = pd.concat([prev, df.astype(str)], ignore_index=True)
    df = df.drop_duplicates(subset=["SYMBOL", "MEETING_DATE", "PURPOSE"])
    df.to_csv(MEETINGS, index=False)
    print(f"  meetings: {len(rows)} fetched, {len(df)} on file")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, default=0,
                    help="Calendar days of price history to seed")
    args = ap.parse_args()

    print("Connecting to NSE…")
    cl = client()
    if cl is None:
        print("\nNothing collected — NSE could not be reached at all. This is "
              "usually temporary; try again in a few minutes. If it persists, "
              "NSE is refusing this runner and the collector needs to run "
              "somewhere else.")
        return 1

    done = {"fno": False, "history": 0, "meetings": 0}

    # --- F&O: only the newest trading day is needed
    print("\nF&O chains…")
    try:
        for off in range(6):
            day = datetime.now() - timedelta(days=off)
            if day.weekday() >= 5:
                continue
            df, note = fetch_fno(cl, day)
            if df is None:
                print(f"  {day:%Y-%m-%d}: {note}")
                time.sleep(0.4)
                continue
            opts = df[df["TYPE"].isin(["STO", "IDO"])] if "TYPE" in df else df
            print(f"  {day:%Y-%m-%d}: {len(df):,} contracts, "
                  f"{df['SYMBOL'].nunique()} underlyings, {len(opts):,} options")
            store_month(FNO_DIR, df)
            if "LOT_SIZE" in df.columns:
                lots = df.dropna(subset=["LOT_SIZE"]).groupby("SYMBOL")["LOT_SIZE"].max()
                LOTS.parent.mkdir(parents=True, exist_ok=True)
                lots.reset_index().to_csv(LOTS, index=False)
                print(f"  lot sizes: {len(lots)} underlyings")
            done["fno"] = True
            break
        else:
            print("  no F&O file found in the last six days")
    except Exception as exc:  # noqa: BLE001
        print(f"  F&O collection failed: {type(exc).__name__}: {str(exc)[:90]}")

    # --- price history, newest first
    print("\nPrice history…")
    try:
        days_back = args.backfill or 6
        have = stored_dates(HIST_DIR)
        print(f"  {len(have)} day(s) already stored")
        got = miss = 0
        for off in range(days_back):
            day = datetime.now() - timedelta(days=off)
            if day.weekday() >= 5 or day.strftime("%Y-%m-%d") in have:
                continue
            try:
                df, note = fetch_equity(cl, day)
            except Exception as exc:  # noqa: BLE001
                df, note = None, f"{type(exc).__name__}"
            if df is None:
                miss += 1
                if got == 0 and miss >= 10:
                    print(f"  the ten most recent days all failed ({note}) — stopping")
                    break
                if got > 0 and miss > 30:
                    print(f"  past the end of NSE's archive after {got} new day(s)")
                    break
            else:
                store_month(HIST_DIR, df, HIST_COLS)
                got += 1
                miss = 0
                if got % 10 == 0 or got == 1:
                    print(f"  {day:%Y-%m-%d}: {len(df):,} stocks ({got} so far)")
            time.sleep(0.35)
        done["history"] = got
        print(f"  {got} new day(s) stored")
    except Exception as exc:  # noqa: BLE001
        print(f"  history collection failed: {type(exc).__name__}: {str(exc)[:90]}")

    print("\nResults calendar…")
    try:
        done["meetings"] = collect_meetings(cl)
    except Exception as exc:  # noqa: BLE001
        print(f"  meetings failed: {type(exc).__name__}: {str(exc)[:90]}")

    try:
        cl.close()
    except Exception:
        pass

    print(f"\nSummary: F&O {'ok' if done['fno'] else 'MISSING'} · "
          f"{done['history']} new price day(s) · {done['meetings']} meeting rows")

    # The app is usable with F&O alone, so only a total failure is an error.
    if not done["fno"] and done["history"] == 0:
        print("Nothing was collected. Re-run — NSE refuses requests "
              "intermittently and a retry often succeeds.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Options Desk

A small tool for trading Indian stock options. Four screens, nothing else.

**Today** — is anything worth trading?
**Trade** — which strike, and where do I get out?
**Journal** — am I actually any good at this?
**Data** — what's been collected.

Deliberately no stock screener, pattern scanner, sentiment gauge or backtest.
Those are research tools. This is for placing trades.

---

## Setup

**1. Put these files in a new GitHub repo.**

Drag `app.py`, `collect.py` and `requirements.txt` in and commit. Then use
**Add file → Create new file**, type the filename as
`.github/workflows/collect.yml` exactly — the slashes create the folders — and
paste that file's contents in.

Make the repo **private**. Your journal is in it.

**2. Deploy.** Go to share.streamlit.io, sign in with GitHub, deploy from your
repo with main file path `app.py`. Under Advanced settings set Python to 3.12.

**3. Collect the data.** In your repo's **Actions** tab, run the *Collect*
workflow with a backfill of `400`. That seeds about eighteen months of price
history, which is what the "it usually moves" comparison needs. Takes ten
minutes or so.

**4. Reboot the app** from share.streamlit.io so it picks up the committed
data.

After that it runs itself: the workflow fires every weekday at 19:30 IST,
collects the day's chains and prices, and commits them.

---

## How to use it

**Today** lists up to five stocks where the market is pricing a *smaller* move
than the stock normally makes, with a results date coming.

That comparison is the whole idea. An option buyer doesn't just need the stock
to move the right way — the move has to beat the premium. A stock priced for a
4% move that routinely does 7% is a better starting point than one that merely
looks bullish on a chart.

It has **no view on direction**. That part is yours, and no data here supplies
it.

**Trade** shows the strikes nearest the money with the numbers that actually
separate them: premium, implied volatility, chance of finishing in the money,
and how far the stock must travel just to break even. Strikes needing more than
the stock typically covers get flagged — they're cheap because they usually
expire worthless.

Pick one and you get a plan in **premium terms**: entry, target and stop as
prices you can place orders at, each with the stock move it requires and the
P&L after costs.

**Watch the stop row.** A 30% stop on premium often triggers on a move of well
under 1% in the stock — inside a normal day's range. The app warns when that
happens. Options lose value quickly on small adverse moves, so a percentage
stop is far tighter than it sounds.

**Journal** is the point of the whole thing. Log every trade with the reason
before you know the outcome. After thirty closed trades your average R tells
you whether you have an edge. Nothing else can answer that.

---

## What this cannot do

It cannot tell you which way a stock will go. It finds where options look
underpriced relative to a stock's own movement; the direction call is yours.

It cannot produce a target monthly income. Markets don't pay a salary, and
anything that appeared to promise one would be lying.

Worth knowing plainly: SEBI has run three studies on this and found that
roughly **nine in ten individual F&O traders in India lose money** — 89% in
FY22, 93% across FY22–FY24, 91% in FY25. The average loss in FY25 was about
₹1.1 lakh. Individuals lost over ₹61,000 crore gross in a year when
proprietary desks made ₹33,000 crore and foreign investors ₹28,000 crore.

The sensible use of this tool is to paper trade for a few months, fill the
journal honestly, and only put money behind it if your own numbers say you
should.

---

## Notes on the data

Every premium is a **model value from the stored closing price**, not a live
quote. Thinly traded strikes can be stale or reflect a single odd print. Check
the live bid and ask before acting.

Where a strike's price yields no implied volatility — usually because it barely
traded — it is dropped rather than shown with an invented number.

"It usually moves" is measured over ordinary periods. Stocks genuinely move
more around results, so options that look expensive before earnings may simply
be priced correctly.

Costs default to typical discount-broker rates and are editable in the Trade
tab. Slippage is included and is usually the largest single line — it's not a
fee, it's the gap between the price you see and the price you get, paid on the
way in and again on the way out.

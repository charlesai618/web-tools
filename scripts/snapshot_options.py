#!/usr/bin/env python3
"""Fetch CBOE delayed option chains and write compact per-contract vol/OI snapshots.

Run by .github/workflows/options-snapshot.yml after the US market close. The
committed snapshot is read by options-chain.html as the *shared* day-over-day
baseline, so every browser sees the same Vol/OI change (unlike localStorage,
which is per-browser).

Output: options-data/<TICKER>.json  ->  {"symbol", "date", "map": {contract: [vol, oi]}}
"""
import json
import os
import sys
import urllib.request

TICKERS = ["SPY"]                       # extend this list to snapshot more symbols
CBOE = "https://cdn.cboe.com/api/global/delayed_quotes/options/"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "options-data")


def fetch(sym):
    """Equities/ETFs live at SYM.json; indices (SPX, VIX, ...) at _SYM.json."""
    for path in (sym + ".json", "_" + sym + ".json"):
        try:
            req = urllib.request.Request(
                CBOE + path, headers={"User-Agent": "Mozilla/5.0 options-snapshot"}
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                j = json.load(r)
            if j.get("data", {}).get("options"):
                return j
        except Exception as e:  # noqa: BLE001 - log and try the next variant
            print(f"  {path}: {e}", file=sys.stderr)
    return None


def snapshot(sym):
    j = fetch(sym)
    if not j:
        print(f"{sym}: no data", file=sys.stderr)
        return False
    d = j["data"]
    date = (d.get("last_trade_time") or j.get("timestamp") or "")[:10]
    m = {}
    for o in d["options"]:
        vol = int(o.get("volume") or 0)
        oi = int(o.get("open_interest") or 0)
        if vol == 0 and oi == 0:
            continue                    # prune dead contracts to keep the file small
        m[o["option"]] = [vol, oi]
    out = {"symbol": d.get("symbol") or sym, "date": date, "map": m}
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, sym + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{sym}: {len(m)} contracts, session {date} -> {os.path.relpath(path)}")
    return True


def main():
    ok = False
    for t in TICKERS:
        ok = snapshot(t) or ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

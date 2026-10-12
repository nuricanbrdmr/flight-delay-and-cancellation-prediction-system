"""Quick structural profile of every CSV in ../../csv (chunked, low memory)."""
import sys, pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parents[2] / "csv"
for f in sorted(D.glob("*.csv")):
    with open(f, encoding="utf-8", errors="replace") as fh:
        head = fh.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    n = 0; cols = None; canc = None
    for ch in pd.read_csv(f, sep=sep, chunksize=500_000, low_memory=False):
        cols = list(ch.columns); n += len(ch)
        for c in ch.columns:
            if c.strip().lower() in ("cancelled",):
                canc = (canc or 0) + (ch[c] == 1).sum()
    print(f"\n== {f.name} sep={sep!r} rows={n:,} cancelled={canc}")
    print(cols)

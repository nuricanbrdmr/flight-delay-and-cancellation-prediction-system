"""'Day-of-departure' (Mode B) features: operational state of the SAME DAY observable LEAD minutes before a flight's scheduled departure.
For a flight with scheduled departure s on date D at origin O (airline A) only events that happened at or before s - LEAD are used:
    * flights that have ACTUALLY departed by then  -> their delays (delayed>15 share, mean delay)
    * flights SCHEDULED to depart by then          -> cancelled share (a cancellation is known at the latest at its scheduled departure) and
                                                      BACKLOG = share of the scheduled, non-cancelled flights that have not yet departed
  levels: origin airport, airline (nation-wide), nation-wide. Rates are shrunk to TRAIN-period priors (no counts -> robust to the 10x denser Jan-2024 file).

TIME ZONES. BTS reports scheduled times in the LOCAL time of the origin airport. The cut-off is therefore computed on an ABSOLUTE (UTC) time axis:
  scheduled UTC = local scheduled time - UTC offset of the origin airport on that date (IANA time zone, daylight saving included);
  actual departure UTC = scheduled UTC + departure delay.
For airline- and nation-level features the events of all airports are compared with the cut-off of the target flight in absolute time. To keep
"same day" meaningful, events are assigned to the calendar day of the TARGET's own time zone (its UTC offset applied to every event) and only
events of the target's own day are used (the same-day window starts at local midnight of the target's date; nothing from the previous day is used).
A flight scheduled less than LEAD minutes after local midnight therefore has no same-day information (neutral prior).
Midnight: an event belongs to the day of its SCHEDULED departure; a delayed flight that departs after midnight is not counted in the next day's state.
Self-test (absolute time): changing every event that happens after the cut-off (or the flight itself) must leave the target's features unchanged.
Output: csv/features_v3.pkl (= features_v2 + L_* columns)"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as C

LEAD, K, W = 180, 30, 4000
TZ_FALLBACK = {"ISN": "America/Chicago"}          # Williston (ND) has no time zone in airportsdata
EPOCH = pd.Timestamp("2018-01-01")


def airport_tz(codes):
    import airportsdata
    ap = airportsdata.load("IATA")
    out = {}
    for a in pd.unique(codes):
        tz = (ap.get(a) or {}).get("tz") or TZ_FALLBACK.get(a)
        if not tz: raise KeyError(f"no time zone for airport {a}")
        out[a] = tz
    return out


def utc_times(df):
    """scheduled departure in absolute minutes (UTC, minutes since EPOCH) and the UTC offset (minutes) of the origin at that moment."""
    n = len(df)
    loc_min = (df.FL_DATE.values - np.datetime64(EPOCH)) / np.timedelta64(1, "m") + C._hm_to_min(df.CRS_DEP_TIME).values
    sched_utc = np.empty(n, dtype=np.int64); off = np.empty(n, dtype=np.int64)
    tzmap = airport_tz(df.ORIGIN.values)
    tzs = df.ORIGIN.map(tzmap).values
    for tz in pd.unique(tzs):
        idx = np.where(tzs == tz)[0]
        naive = EPOCH + pd.to_timedelta(loc_min[idx].astype(np.int64), unit="m")
        aware = pd.DatetimeIndex(naive).tz_localize(tz, ambiguous=np.ones(len(idx), bool), nonexistent="shift_forward")
        utc = aware.tz_convert("UTC").tz_localize(None)
        u = ((utc - EPOCH) / pd.Timedelta(minutes=1)).astype(np.int64)
        sched_utc[idx] = u
        off[idx] = loc_min[idx].astype(np.int64) - u
    return sched_utc, off


def live_features(df, pri, lead=LEAD):
    """df: FL_DATE, ORIGIN, AIRLINE_CODE, CRS_DEP_TIME, CANCELLED, DEP_DELAY. pri: dict of FIXED train-period priors."""
    n = len(df)
    sched_utc, off = utc_times(df)
    loc = C._hm_to_min(df.CRS_DEP_TIME).values.astype(np.int64)
    canc = df.CANCELLED.values == 1
    dly = df.DEP_DELAY.values.astype(float)
    dly0 = np.nan_to_num(dly)
    dep = ~canc & ~np.isnan(dly)                    # departed flights (event time = actual departure)
    ones = np.ones(n)
    keyvals = {"ORG": [df.ORIGIN.values], "AIR": [df.AIRLINE_CODE.values], "NAT": []}
    res = {f"L_{t}_{m}": np.full(n, np.nan) for t in keyvals for m in ["DELAY15", "MEANDELAY", "CANCEL", "BACKLOG"]}
    for o in np.unique(off):
        q = np.where(off == o)[0]                   # target flights whose origin has UTC offset o
        shifted = sched_utc + o                     # every event on the clock of the targets' time zone
        day = np.floor_divide(shifted, 1440)
        minute = shifted - day * 1440               # scheduled minute of the event on that clock, 0..1439
        act = minute + dly0                         # actual departure minute (only used where dep)
        cutoff_q = loc[q] - lead                    # == minute[q] - lead
        for tag, kv in keyvals.items():
            gid = pd.MultiIndex.from_arrays([day] + kv).factorize()[0].astype(np.int64)
            base = gid * W + 400
            gq = gid[q]; qkey = gq * W + 400 + np.clip(cutoff_q, -300, 3000)

            def cum_at(event_mask, event_minute, weights):
                ek = (base + np.clip(event_minute, -300, 3000))[event_mask]
                order = np.argsort(ek, kind="stable"); ek = ek[order]
                cs = np.concatenate([[0.0], np.cumsum(weights[event_mask][order])])
                hi = np.searchsorted(ek, qkey, side="right"); lo = np.searchsorted(ek, gq * W, side="left")
                return cs[hi] - cs[lo]

            act_i = act.astype(np.int64)
            n_dep = cum_at(dep, act_i, ones)
            n_d15 = cum_at(dep, act_i, (dly0 > 15).astype(float))
            s_dly = cum_at(dep, act_i, np.clip(dly0, -30, 180))
            allm = np.ones(n, bool)
            n_sch = cum_at(allm, minute, ones)
            n_can = cum_at(allm, minute, canc.astype(float))
            n_sch_nc = n_sch - n_can
            backlog = np.clip(n_sch_nc - n_dep, 0, None)
            res[f"L_{tag}_DELAY15"][q] = (n_d15 + K * pri["d15"]) / (n_dep + K)
            res[f"L_{tag}_MEANDELAY"][q] = (s_dly + K * pri["mean"]) / (n_dep + K)
            res[f"L_{tag}_CANCEL"][q] = (n_can + K * pri["canc"]) / (n_sch + K)
            res[f"L_{tag}_BACKLOG"][q] = (backlog + K * pri["backlog"]) / (n_sch_nc + K)
    return pd.DataFrame(res, index=df.index)


def priors(train):
    ok = (train.CANCELLED == 0)
    return dict(d15=float((train.DEP_DELAY[ok] > 15).mean()), mean=float(train.DEP_DELAY[ok].clip(-30, 180).mean()),
                canc=float(train.CANCELLED.mean()), backlog=0.05)


def selftest(df, pri, n_targets=300, fn=None, days=3, seed=1, quiet=False):
    """Alter every event that occurs AFTER the absolute cut-off of the target (not yet scheduled, or scheduled but not yet departed) and the target
    itself; the target's features must not change. Returns the number of targets whose features changed (0 expected). `fn` = feature function."""
    fn = fn or live_features
    rng = np.random.RandomState(seed)
    d0 = df.FL_DATE.drop_duplicates().sort_values().iloc[500]
    sub = df[df.FL_DATE.between(d0 - pd.Timedelta(days=1), d0 + pd.Timedelta(days=days - 2))].copy().reset_index(drop=True)
    a = fn(sub, pri)
    sched_utc, _ = utc_times(sub)
    dly = sub.DEP_DELAY.fillna(0).values
    canc = sub.CANCELLED.values == 1
    act_utc = np.where(canc, np.inf, sched_utc + dly)
    cand = np.where(sub.FL_DATE == d0)[0]
    bad = 0
    for i in rng.choice(cand, n_targets, replace=False):
        cut = sched_utc[i] - LEAD
        pending = (sched_utc > cut) & (act_utc > cut)       # not yet scheduled and not yet departed at the cut-off (includes the flight itself)
        later_dep = (sched_utc <= cut) & (act_utc > cut) & ~canc   # scheduled before the cut-off, still on the ground (cancellation already known)
        s2 = sub.copy()
        s2.loc[pending | later_dep, "DEP_DELAY"] = s2.loc[pending | later_dep, "DEP_DELAY"].fillna(0) + 300
        s2.loc[pending, "CANCELLED"] = 1 - s2.loc[pending, "CANCELLED"]
        b = fn(s2, pri)
        if not np.allclose(a.iloc[i].values, b.iloc[i].values, equal_nan=True): bad += 1
    if not quiet:
        print(f"SELF-TEST: {n_targets} target flights on {d0.date()} (all time zones): features changed for {bad} targets", flush=True)
    return bad


if __name__ == "__main__":
    df = C.load()
    pri = priors(df[df.SPLIT == "train"])
    print("train-period priors:", {k: round(v, 4) for k, v in pri.items()}, flush=True)
    bad = selftest(df, pri)
    assert bad == 0, f"live-feature leakage test FAILED for {bad} targets"
    print("SELF-TEST OK: features unchanged when every event after the absolute cut-off / the flight itself is altered", flush=True)
    L = live_features(df, pri)
    print(L.describe().T[["mean", "std", "min", "max"]].round(3).to_string(), flush=True)
    tr = df.SPLIT == "train"
    print("train corr with CANCELLED (|r|):", {c: round(abs(np.corrcoef(L[c][tr], df.CANCELLED[tr])[0, 1]), 3) for c in L.columns if c.startswith("L_ORG")}, flush=True)
    del df
    base = pd.read_pickle(C.D / "features_v2.pkl")
    assert len(base) == len(L)
    base = base.drop(columns=[c for c in base.columns if c.startswith("L_")], errors="ignore")
    base = base.join(L)
    base.to_pickle(C.D / "features_v3.pkl")
    print("saved", base.shape, flush=True)

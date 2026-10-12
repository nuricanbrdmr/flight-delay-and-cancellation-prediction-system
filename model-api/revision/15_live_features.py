"""'Day-of-departure' (Mode B) features: operational state of the SAME DAY observable LEAD minutes before a flight's scheduled departure.
For a flight with scheduled departure s on date D at origin O (airline A):
  only events with time <= s - LEAD are used:
    * flights that have ACTUALLY departed by then  -> their delays (delayed>15 share, mean delay)
    * flights SCHEDULED to depart by then          -> cancelled share (a cancellation is known at its scheduled time), and BACKLOG = share of
                                                      scheduled-but-not-yet-departed flights (a live indicator of delays building up)
  levels: origin airport, airline (nation-wide), nation-wide. Rates are shrunk to TRAIN-period priors (no counts -> robust to the 10x denser Jan-2024 file).
Self-test: modifying every flight that happens after the cutoff (or the flight itself) must leave the features unchanged.
Output: csv/features_v3.pkl (= features_v2 + L_* columns)"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as C

LEAD, K, W = 180, 30, 4000


def live_features(df, pri):
    """df: FL_DATE, ORIGIN, AIRLINE_CODE, CRS_DEP_TIME, CANCELLED, DIVERTED, DEP_DELAY. pri: dict of FIXED train-period priors."""
    crs = C._hm_to_min(df.CRS_DEP_TIME).values.astype(np.int64)
    canc = df.CANCELLED.values == 1
    dly = df.DEP_DELAY.values
    act = np.where(canc | np.isnan(dly), np.nan, crs + np.nan_to_num(dly))
    cutoff = crs - LEAD
    date = df.FL_DATE.values
    out = pd.DataFrame(index=df.index)
    for tag, keycols in [("ORG", ["ORIGIN"]), ("AIR", ["AIRLINE_CODE"]), ("NAT", [])]:
        gid = pd.MultiIndex.from_frame(df[["FL_DATE"] + keycols]).factorize()[0].astype(np.int64)
        base = gid * W + 400
        qkey = base + np.clip(cutoff, -300, 3000)

        def cum_at(event_mask, event_minute, weights):
            """sum of `weights` over events (same group) with event_minute <= cutoff of the target row."""
            ek = (base + np.clip(event_minute, -300, 3000))[event_mask]
            o = np.argsort(ek, kind="stable")
            ek = ek[o]
            cs = np.concatenate([[0.0], np.cumsum(weights[event_mask][o])])
            hi = np.searchsorted(ek, qkey, side="right")
            lo = np.searchsorted(ek, gid * W, side="left")
            return cs[hi] - cs[lo]

        ones = np.ones(len(df))
        dep = ~canc & ~np.isnan(act)                    # departed flights, event time = actual departure minute
        n_dep = cum_at(dep, np.nan_to_num(act).astype(np.int64), ones)
        n_d15 = cum_at(dep, np.nan_to_num(act).astype(np.int64), (np.nan_to_num(dly) > 15).astype(float))
        s_dly = cum_at(dep, np.nan_to_num(act).astype(np.int64), np.clip(np.nan_to_num(dly), -30, 180))
        sch_all = np.ones(len(df), bool)                # scheduled events, event time = scheduled minute
        n_sch = cum_at(sch_all, crs, ones)
        n_can = cum_at(sch_all, crs, canc.astype(float))
        n_sch_nc = n_sch - n_can
        backlog = np.clip(n_sch_nc - n_dep, 0, None)    # scheduled by cutoff, not departed by cutoff
        out[f"L_{tag}_DELAY15"] = (n_d15 + K * pri["d15"]) / (n_dep + K)
        out[f"L_{tag}_MEANDELAY"] = (s_dly + K * pri["mean"]) / (n_dep + K)
        out[f"L_{tag}_CANCEL"] = (n_can + K * pri["canc"]) / (n_sch + K)
        out[f"L_{tag}_BACKLOG"] = (backlog + K * pri["backlog"]) / (n_sch_nc + K)
    return out


def priors(train):
    ok = (train.CANCELLED == 0)
    return dict(d15=float((train.DEP_DELAY[ok] > 15).mean()), mean=float(train.DEP_DELAY[ok].clip(-30, 180).mean()),
                canc=float(train.CANCELLED.mean()), backlog=0.05)


def selftest(df, pri, n_targets=300):
    rng = np.random.RandomState(1)
    day = df.FL_DATE.drop_duplicates().sort_values().iloc[500]
    sub = df[df.FL_DATE == day].copy().reset_index(drop=True)
    a = live_features(sub, pri)
    crs = C._hm_to_min(sub.CRS_DEP_TIME).values
    bad = 0
    for i in rng.choice(len(sub), n_targets, replace=False):
        cut = crs[i] - LEAD
        s2 = sub.copy()
        act = np.where(s2.CANCELLED == 1, np.nan, crs + s2.DEP_DELAY.fillna(0).values)
        after = (crs > cut) & ((np.isnan(act)) | (act > cut))     # events strictly after the cutoff (not yet scheduled, or not yet departed)
        after[i] = True                                           # the flight itself
        s2.loc[after, "DEP_DELAY"] = s2.loc[after, "DEP_DELAY"].fillna(0) + 300      # later still
        flip = after.copy()          # only events that happen after the cutoff (a flight that already departed early is a known fact)
        s2.loc[flip, "CANCELLED"] = 1 - s2.loc[flip, "CANCELLED"]
        b = live_features(s2, pri)
        if not np.allclose(a.iloc[i].values, b.iloc[i].values, equal_nan=True): bad += 1
    assert bad == 0, f"live-feature leakage test FAILED for {bad}/{n_targets} targets"
    print(f"SELF-TEST OK: {n_targets} target flights on {day.date()}: features unchanged when every later event / the flight itself is altered", flush=True)


if __name__ == "__main__":
    df = pd.read_pickle(C.D / "features_v2.pkl")
    pri = priors(df[df.SPLIT == "train"])
    print("train-period priors:", {k: round(v, 4) for k, v in pri.items()}, flush=True)
    selftest(df, pri)
    L = live_features(df, pri)
    df = df.join(L)
    print(df[list(L.columns)].describe().T[["mean", "std", "min", "max"]].round(3).to_string(), flush=True)
    print("train corr with CANCELLED (|r|):", {c: round(abs(np.corrcoef(L[c][df.SPLIT == 'train'], df.CANCELLED[df.SPLIT == 'train'])[0, 1]), 3) for c in L.columns if c.startswith('L_ORG')}, flush=True)
    df.to_pickle(C.D / "features_v3.pkl")
    print("saved", df.shape, flush=True)

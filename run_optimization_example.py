
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_optimization_example.py (V2)
Minimal example on one house/day.
"""
import argparse, pandas as pd
from run_optimization_batch import try_import_user_loader, load_local_csvs, load_from_github, dt_hours
from optimization_multiobj import optimize_day_multiobj

def first_complete_day_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dt = (idx.to_series().diff().dropna().median() or pd.Timedelta('1H'))
    steps = int(round(24.0 / (dt.total_seconds()/3600.0)))
    start = idx[0].replace(hour=0, minute=0, second=0, microsecond=0)
    return pd.date_range(start, periods=steps, freq=dt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--inc", type=str, default="10-15")
    ap.add_argument("--dec", type=str, default="18-23,6-8")
    ap.add_argument("--data_dir", type=str, default=None)
    args = ap.parse_known_args()[0]

    user_loader = try_import_user_loader()
    if user_loader is not None:
        df = user_loader(args.house)
    elif args.data_dir:
        df = load_local_csvs(args.data_dir, args.house)
    else:
        df = load_from_github(args.house)

    df = df.sort_index().ffill().bfill()
    df['load_kW'] = pd.to_numeric(df['Consumption (kW)'], errors='coerce').clip(lower=0.0)
    df['pv_kW']   = pd.to_numeric(df['PV Power Generation (kW)'], errors='coerce').clip(lower=0.0)

    day_idx = first_complete_day_index(df.index)
    day_df = df.loc[day_idx, ['load_kW','pv_kW']].copy()

    res = optimize_day_multiobj(day_df, alpha=args.alpha, inc_win=args.inc, dec_win=args.dec,
                                lambda_import=1.0, lambda_curt=0.0, lambda_peak=0.0, lambda_ramp=0.0)
    print("Status:", res['status'])
    print("Before:", res['before'])
    print("After:", res['after'])

if __name__ == "__main__":
    main()

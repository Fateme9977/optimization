
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_optimization_batch.py (V2)
Batch runner across houses and representative seasonal weeks.
"""
import os, json, argparse
import pandas as pd, numpy as np
from datetime import timedelta

def try_import_user_loader():
    try:
        import create_master_dataset as cmd
        if hasattr(cmd,"load_and_merge_house_data"): return cmd.load_and_merge_house_data
    except Exception: pass
    return None

def load_local_csvs(data_dir, house_id):
    import pandas as pd, os
    L=os.path.join(data_dir,f"Load House {house_id}.csv")
    P=os.path.join(data_dir,f"PV Generation House {house_id}.csv")
    dfL=pd.read_csv(L, parse_dates=[0]); tL=[c for c in dfL.columns if "time" in c.lower() or "date" in c.lower()][0]
    dfL=dfL.rename(columns={tL:"Timestamp"}).set_index("Timestamp")
    dfP=pd.read_csv(P, parse_dates=[0]); tP=[c for c in dfP.columns if "time" in c.lower() or "date" in c.lower() or "stamp" in c.lower()][0]
    dfP=dfP.rename(columns={tP:"Timestamp"}).set_index("Timestamp")
    if "PV Power Generation (kW)" not in dfP.columns and "PV Power Generation (W)" in dfP.columns:
        dfP["PV Power Generation (kW)"]=dfP["PV Power Generation (W)"]/1000.0
    dfP=dfP[["PV Power Generation (kW)"]]
    df=dfL[["Consumption (kW)"]].join(dfP, how="outer").sort_index().ffill().bfill()
    df["house_id"]=house_id
    return df

def load_from_github(house_id:int):
    base="https://raw.githubusercontent.com/Fateme9977/dataseat/Fateme9977-data/"
    L=f"{base}Load%20House%20{house_id}.csv"
    P=f"{base}PV%20Generation%20House%20{house_id}.csv"
    dfL=pd.read_csv(L, parse_dates=['DateTime']).rename(columns={'DateTime':'Timestamp'}).set_index('Timestamp')
    dfP=pd.read_csv(P, parse_dates=['Timestamp']).set_index('Timestamp')
    if "PV Power Generation (kW)" not in dfP.columns and "PV Power Generation (W)" in dfP.columns:
        dfP['PV Power Generation (kW)']=dfP['PV Power Generation (W)']/1000.0
    dfP=dfP[['PV Power Generation (kW)']]
    df=dfL[['Consumption (kW)']].join(dfP, how='outer').sort_index().ffill().bfill()
    df['house_id']=house_id
    return df

def dt_hours(idx): 
    dt=(pd.Series(idx).diff().dropna().median() or pd.Timedelta("1H"))
    return dt.total_seconds()/3600.0
def steps_per_day(idx): return int(round(24.0/dt_hours(idx)))

def pick_weeks(df):
    m2s={1:'DJF',2:'DJF',12:'DJF',3:'MAM',4:'MAM',5:'MAM',6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'}
    tmp=df.copy(); tmp['date']=tmp.index.date; tmp['season']=[m2s[m] for m in tmp.index.month]
    daily=tmp.groupby(['season','date'])['PV Power Generation (kW)'].sum().reset_index()
    daily['Epv']=daily['PV Power Generation (kW)']*dt_hours(df.index); picks={}
    for s,g in daily.groupby('season'):
        if len(g)<7: continue
        med=g['Epv'].median(); c=g.iloc[(g['Epv']-med).abs().argsort().values[0]]['date']
        start=pd.Timestamp(c)
        S=set(g['date'])
        def ok(u): return all((u+pd.Timedelta(days=i)).date() in S for i in range(7))
        u=start
        for k in range(8):
            if ok(u): break
            u=start-pd.Timedelta(days=k+1)
        if not ok(u):
            u=start
            for k in range(8):
                if ok(u): break
                u=start+pd.Timedelta(days=k+1)
        if ok(u): picks[s]=[(pd.Timestamp(u.date()), pd.Timestamp((u+pd.Timedelta(days=6)).date()))]
    return picks

def slice_day(df, day):
    h=steps_per_day(df.index); dt=(pd.Series(df.index).diff().dropna().median() or pd.Timedelta("1H"))
    start=pd.Timestamp(day).replace(hour=0,minute=0,second=0,microsecond=0)
    rng=pd.date_range(start, periods=h, freq=dt)
    return df.loc[rng].copy() if set(rng).issubset(set(df.index)) else pd.DataFrame()

def _load_windows_json(path):
    if not path or not os.path.isfile(path): return {}
    with open(path,"r",encoding="utf-8") as f: return json.load(f)

def _wins_for(house, season, inc_def, dec_def, W):
    k=str(house)
    if k in W and season in W[k]:
        inc=W[k][season].get("inc",inc_def); dec=W[k][season].get("dec",dec_def)
        return inc,dec
    return inc_def, dec_def

def run_day(df_day, params):
    from optimization_multiobj import optimize_day_multiobj
    return optimize_day_multiobj(
        df_day[['load_kW','pv_kW']],
        alpha=params['alpha'], inc_win=params['inc'], dec_win=params['dec'],
        lambda_import=params['l_import'], lambda_curt=params['l_curt'],
        lambda_peak=params['l_peak'], lambda_ramp=params['l_ramp'],
        peak_cap=params['peak_cap'], ramp_cap=params['ramp_cap'])

def run_batch(houses="1-13", alpha=0.2, inc="10-15", dec="18-23,6-8",
              l_import=1.0, l_curt=0.0, l_peak=0.0, l_ramp=0.0,
              peak_cap=None, ramp_cap=None, data_dir=None, outdir="outputs_batch",
              save_solutions=False, solutions_dir=None, windows_json=None):
    os.makedirs(outdir, exist_ok=True)
    params=dict(alpha=alpha, inc=inc, dec=dec, l_import=l_import, l_curt=l_curt,
                l_peak=l_peak, l_ramp=l_ramp, peak_cap=peak_cap, ramp_cap=ramp_cap)
    with open(os.path.join(outdir,"run_params.json"),"w",encoding="utf-8") as f: json.dump(params,f,indent=2)
    W=_load_windows_json(windows_json)

    houses_list = list(range(int(houses.split("-")[0]), int(houses.split("-")[1])+1)) if "-" in str(houses) else [int(x) for x in str(houses).split(",")]

    user_loader=try_import_user_loader()
    daily_rows=[]; picked_meta={}

    for h in houses_list:
        if user_loader: df=user_loader(h)
        elif data_dir:  df=load_local_csvs(data_dir,h)
        else:           df=load_from_github(h)
        if df is None or df.empty: 
            print(f"[WARN] House {h}: no data"); continue
        df=df.sort_index().ffill().bfill()
        df['load_kW']=pd.to_numeric(df['Consumption (kW)'],errors='coerce').clip(lower=0)
        df['pv_kW']=pd.to_numeric(df['PV Power Generation (kW)'],errors='coerce').clip(lower=0)

        picks=pick_weeks(df)
        picked_meta[str(h)]={s:[(str(a),str(b)) for (a,b) in spans] for s,spans in picks.items()}
        if not picks: 
            print(f"[INFO] House {h}: no eligible weeks"); continue

        for season,spans in picks.items():
            inc_season, dec_season = _wins_for(h, season, inc, dec, W)
            params['inc']=inc_season; params['dec']=dec_season
            for (start,end) in spans:
                day=start
                while day<=end:
                    dd=slice_day(df,day)
                    if not dd.empty and not dd[['load_kW','pv_kW']].isna().any().any():
                        res=run_day(dd,params)
                        b,a=res['before'],res['after']
                        row=dict(house=h,season=season,date=str(pd.Timestamp(day).date()),
                                 **{f"before_{k}":v for k,v in b.items()},
                                 **{f"after_{k}":v for k,v in a.items()})
                        for k in b.keys(): row[f"delta_{k}"]=a[k]-b[k]
                        daily_rows.append(row)
                        if save_solutions:
                            sdir = solutions_dir or os.path.join(outdir,"solutions")
                            os.makedirs(os.path.join(sdir,f"house{h}"), exist_ok=True)
                            res['solution'].to_csv(os.path.join(sdir,f"house{h}",f"{str(pd.Timestamp(day).date())}.csv"))
                    day += timedelta(days=1)

    T1=pd.DataFrame(daily_rows); T1.to_csv(os.path.join(outdir,"T1_daily_house.csv"), index=False)
    if not T1.empty:
        metrics=[c.replace("before_","") for c in T1.columns if c.startswith("before_")]
        agg={}
        for k in metrics:
            agg[f"before_{k}"]="mean"; agg[f"after_{k}"]="mean"; agg[f"delta_{k}"]=["mean","median","std"]
        T2=T1.groupby(['house','season']).agg(agg)
        T2.columns=["_".join([c for c in col if c]) for col in T2.columns.to_flat_index()]
        T2=T2.reset_index(); T2.to_csv(os.path.join(outdir,"T2_season_house.csv"), index=False)
        def iqr(x): return x.quantile(0.75)-x.quantile(0.25)
        rows=[]
        for s,g in T1.groupby("season"):
            d={ "season":s }
            for k in metrics:
                v=g.groupby("house")[f"delta_{k}"].mean()
                d[f"delta_{k}_median"]=float(v.median()); d[f"delta_{k}_IQR"]=float(iqr(v))
            rows.append(d)
        pd.DataFrame(rows).sort_values("season").to_csv(os.path.join(outdir,"T3_season_cohort.csv"), index=False)
    else:
        pd.DataFrame().to_csv(os.path.join(outdir,"T2_season_house.csv"), index=False)
        pd.DataFrame().to_csv(os.path.join(outdir,"T3_season_cohort.csv"), index=False)

    with open(os.path.join(outdir,"picked_weeks.json"),"w",encoding="utf-8") as f: json.dump(picked_meta,f,indent=2)
    print("[OK] Finished. See outputs in", outdir)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--houses", type=str, default="1-13")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--inc", type=str, default="10-15")
    ap.add_argument("--dec", type=str, default="18-23,6-8")
    ap.add_argument("--l_import", type=float, default=1.0)
    ap.add_argument("--l_curt", type=float, default=0.0)
    ap.add_argument("--l_peak", type=float, default=0.0)
    ap.add_argument("--l_ramp", type=float, default=0.0)
    ap.add_argument("--peak_cap", type=float, default=None)
    ap.add_argument("--ramp_cap", type=float, default=None)
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="outputs_batch")
    ap.add_argument("--save_solutions", action="store_true")
    ap.add_argument("--solutions_dir", type=str, default=None)
    ap.add_argument("--windows_json", type=str, default=None)
    args=ap.parse_known_args()[0]
    run_batch(houses=args.houses, alpha=args.alpha, inc=args.inc, dec=args.dec,
              l_import=args.l_import, l_curt=args.l_curt, l_peak=args.l_peak, l_ramp=args.l_ramp,
              peak_cap=args.peak_cap, ramp_cap=args.ramp_cap, data_dir=args.data_dir, outdir=args.outdir,
              save_solutions=args.save_solutions, solutions_dir=args.solutions_dir, windows_json=args.windows_json)
if __name__=="__main__": main()

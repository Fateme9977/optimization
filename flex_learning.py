
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flex_learning.py
Learn house/season-specific increase/decrease windows from saved LP solutions.
"""
import os, json, argparse, pandas as pd, numpy as np
SEASON_MAP={1:'DJF',2:'DJF',12:'DJF',3:'MAM',4:'MAM',5:'MAM',6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'}

def contiguous_blocks(flags):
    blocks=[]; inb=False; s=0
    for h in range(24):
        if flags[h] and not inb: inb=True; s=h
        if inb and (h==23 or not flags[h+1]): blocks.append((s,h)); inb=False
    return blocks
def to_str(blocks): return ",".join(f"{a}-{b+1}" for a,b in blocks)  # [a,b+1)

def learn_for_house(hdir, prob_thresh=0.4, eps=1e-4):
    rows=[]
    for fn in os.listdir(hdir):
        if not fn.endswith(".csv"): continue
        df=pd.read_csv(os.path.join(hdir,fn), parse_dates=[0], index_col=0)
        if not {'s_plus_kW','s_minus_kW'}.issubset(df.columns): continue
        date=pd.to_datetime(fn.replace(".csv","")); season=SEASON_MAP[date.month]
        hrs=df.index.hour; inc=(df['s_plus_kW'].values>eps).astype(int); dec=(df['s_minus_kW'].values>eps).astype(int)
        for h,i,d in zip(hrs,inc,dec): rows.append(dict(season=season,hour=int(h),inc=int(i),dec=int(d)))
    if not rows: return {}
    T=pd.DataFrame(rows)
    out={}
    for s,g in T.groupby("season"):
        f_inc=g.groupby("hour")["inc"].mean().reindex(range(24),fill_value=0).values
        f_dec=g.groupby("hour")["dec"].mean().reindex(range(24),fill_value=0).values
        inc_b=contiguous_blocks(f_inc>=prob_thresh); dec_b=contiguous_blocks(f_dec>=prob_thresh)
        out[s]=dict(inc=to_str(inc_b) if inc_b else "", dec=to_str(dec_b) if dec_b else "")
    return out

def learn_windows(solutions_dir, out_json, prob_thresh=0.4, eps=1e-4):
    W={}
    for name in os.listdir(solutions_dir):
        if not name.startswith("house"): continue
        hdir=os.path.join(solutions_dir,name)
        if not os.path.isdir(hdir): continue
        hid=int(name.replace("house",""))
        w=learn_for_house(hdir, prob_thresh=prob_thresh, eps=eps)
        if w: W[str(hid)]=w
    with open(out_json,"w",encoding="utf-8") as f: json.dump(W,f,indent=2)
    print("[OK] learned windows ->", out_json)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--solutions_dir", required=True)
    ap.add_argument("--out_json", default="learned_windows.json")
    ap.add_argument("--prob_thresh", type=float, default=0.4)
    ap.add_argument("--eps", type=float, default=1e-4)
    args=ap.parse_known_args()[0]
    learn_windows(args.solutions_dir, args.out_json, prob_thresh=args.prob_thresh, eps=args.eps)
if __name__=="__main__": main()

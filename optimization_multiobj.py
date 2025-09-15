
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimization_multiobj.py (V2)
Multi-objective LP for intra-day demand shifting (no battery, no tariffs).
"""
import pandas as pd, numpy as np, pulp
from typing import Dict, Any

def _parse_windows(win_str: str):
    if not win_str: return []
    out=[]
    for seg in str(win_str).split(","):
        seg=seg.strip()
        if not seg: continue
        a,b=seg.split("-"); out.append((int(a),int(b)))
    return sorted(out)

def _mask_from_windows(index: pd.DatetimeIndex, windows):
    if not windows:
        return pd.Series(False, index=index)
    m = pd.Series(False, index=index)
    for (a,b) in windows:
        m |= ((index.hour >= a) & (index.hour < b))  # [a,b)
    return m

def _metrics(load_kW: pd.Series, pv_kW: pd.Series, dt_h: float) -> Dict[str,float]:
    pv_to_load = np.minimum(load_kW.values, pv_kW.values)
    E_pv   = float(pv_kW.sum() * dt_h)
    E_load = float(load_kW.sum() * dt_h)
    E_use  = float(pv_to_load.sum() * dt_h)
    net    = load_kW.values - pv_kW.values
    return dict(
        E_load_kWh=E_load,
        E_pv_kWh=E_pv,
        E_pv_used_kWh=E_use,
        E_import_kWh=float(np.maximum(net,0).sum()*dt_h),
        Curtailment_kWh=float(np.maximum(-net,0).sum()*dt_h),
        SCR=(E_use/E_pv if E_pv>0 else float("nan")),
        SSR=(E_use/E_load if E_load>0 else float("nan")),
        NetLoad_peak_kW=float(np.nanmax(net)),
        NetLoad_mean_kW=float(np.nanmean(net)),
        NetLoad_std_kW=float(np.nanstd(net)),
        NetLoad_ramps_L1=float(np.abs(np.diff(net)).sum())
    )

def optimize_day_multiobj(df_day: pd.DataFrame,
                          alpha: float=0.2,
                          inc_win: str="10-15",
                          dec_win: str="18-23,6-8",
                          lambda_import: float=1.0,
                          lambda_curt: float=0.0,
                          lambda_peak: float=0.0,
                          lambda_ramp: float=0.0,
                          peak_cap: float=None,
                          ramp_cap: float=None) -> Dict[str,Any]:
    df = df_day.copy().astype(float).sort_index()
    assert {'load_kW','pv_kW'}.issubset(df.columns)

    dt   = (df.index.to_series().diff().dropna().median() or pd.Timedelta("1H"))
    dt_h = dt.total_seconds()/3600.0
    T    = range(len(df))

    incm = _mask_from_windows(df.index, _parse_windows(inc_win))
    decm = _mask_from_windows(df.index, _parse_windows(dec_win))

    load = df['load_kW'].values
    pv   = df['pv_kW'].values
    E_day = float(load.sum()*dt_h)

    m = pulp.LpProblem("DemandShiftLP", pulp.LpMinimize)
    s_plus  = pulp.LpVariable.dicts('s_plus',  T, lowBound=0.0)
    s_minus = pulp.LpVariable.dicts('s_minus', T, lowBound=0.0)
    y = pulp.LpVariable.dicts('y', T, lowBound=0.0)     # grid import
    z = pulp.LpVariable.dicts('z', T, lowBound=0.0)     # unused PV
    r = pulp.LpVariable.dicts('r', T, lowBound=0.0)     # L1 ramp(|Δy|)
    Ppeak = pulp.LpVariable('Ppeak', lowBound=0.0)

    for t in T:
        if not bool(incm.iloc[t]):  m += s_plus[t]  == 0.0
        if not bool(decm.iloc[t]):  m += s_minus[t] == 0.0
        net = load[t] + s_plus[t] - s_minus[t] - pv[t]
        m += y[t] >= net
        m += z[t] >= -net
        m += Ppeak >= y[t]
        if peak_cap is not None:
            m += y[t] <= peak_cap

    m += pulp.lpSum([s_plus[t] for t in T]) == pulp.lpSum([s_minus[t] for t in T])
    m += pulp.lpSum([s_plus[t]*dt_h for t in T]) <= alpha * E_day

    for t in T:
        prev = y[t-1] if t>0 else 0.0
        m += r[t] >= y[t] - prev
        m += r[t] >= prev - y[t]
        if ramp_cap is not None:
            m += r[t] <= ramp_cap

    obj = (lambda_import*dt_h*pulp.lpSum([y[t] for t in T]) +
           lambda_curt  *dt_h*pulp.lpSum([z[t] for t in T]) +
           lambda_peak  *Ppeak +
           lambda_ramp  *pulp.lpSum([r[t] for t in T]))
    m += obj
    m.solve(pulp.PULP_CBC_CMD(msg=False))

    s_plus_sol  = np.array([pulp.value(s_plus[t])  for t in T])
    s_minus_sol = np.array([pulp.value(s_minus[t]) for t in T])
    y_sol       = np.array([pulp.value(y[t]) for t in T])
    z_sol       = np.array([pulp.value(z[t]) for t in T])
    load_after  = load + s_plus_sol - s_minus_sol

    before = _metrics(df['load_kW'], df['pv_kW'], dt_h)
    after  = _metrics(pd.Series(load_after, index=df.index), df['pv_kW'], dt_h)

    sol = pd.DataFrame({
        'load_kW': df['load_kW'].values,
        'pv_kW': df['pv_kW'].values,
        's_plus_kW': s_plus_sol,
        's_minus_kW': s_minus_sol,
        'import_kW': y_sol,
        'curtail_kW': z_sol,
        'load_after_kW': load_after
    }, index=df.index)

    return dict(status=pulp.LpStatus[m.status], before=before, after=after, solution=sol)

#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
OUT.mkdir(exist_ok=True)
scenarios=['S0_BASE','S2_CONN','S3_BUFF','S6_UPRISK']
forecast_cols=['fcst_max_receptor_dd','fcst_max_compliance_dd','fcst_stage4_mean_inflow','fcst_recovery_years']

def parameter_group(name:str)->str:
    if name.startswith('upp_'): return 'upper pilot points'
    if name.startswith('main_'): return 'main pilot points'
    if name.startswith('low_'): return 'lower pilot points'
    return {
        'recharge_mult':'recharge','ghb_mult':'regional support','riverbed_mult':'receptor conductance',
        'channel_mult':'palaeochannel multiplier','backfill_mult':'backfill conductivity'
    }.get(name,name)

pg=[]; og=[]
for s in scenarios:
    d=ROOT/'data/scenario_outputs'/s
    par=pd.read_csv(d/f'{s}_final_parameters.csv',index_col=0)
    obs=pd.read_csv(d/f'{s}_final_observations.csv',index_col=0)
    common=par.index.intersection(obs.index)
    par=par.loc[common]; obs=obs.loc[common]
    for p in par.columns:
        for f in forecast_cols:
            rho,pval=spearmanr(par[p],obs[f],nan_policy='omit')
            pg.append([s,parameter_group(p),p,f,len(common),rho,abs(rho),pval])
    cond=[c for c in obs.columns if c not in forecast_cols]
    for o in cond:
        for f in forecast_cols:
            rho,pval=spearmanr(obs[o],obs[f],nan_policy='omit')
            og.append([s,o,f,len(common),rho,abs(rho),pval])
P=pd.DataFrame(pg,columns=['scenario','parameter_group','parameter','forecast','n','spearman_rho','abs_rho','p_value'])
G=(P.groupby(['scenario','parameter_group','forecast'])
   .agg(max_abs_rho=('abs_rho','max'),median_abs_rho=('abs_rho','median'),n_parameters=('parameter','nunique')).reset_index())
G.to_csv(OUT/'ensemble_parameter_group_forecast_associations.csv',index=False)
O=pd.DataFrame(og,columns=['scenario','observation','forecast','n','spearman_rho','abs_rho','p_value'])
top=(O.sort_values(['scenario','forecast','abs_rho'],ascending=[True,True,False])
     .groupby(['scenario','forecast'],group_keys=False).head(3))
top.to_csv(OUT/'top_observation_forecast_associations.csv',index=False)
print('Wrote',OUT)

from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def test_retained_counts():
    expected={'S0_BASE':58,'S2_CONN':38,'S3_BUFF':89,'S6_UPRISK':42}
    for s,n in expected.items():
        df=pd.read_csv(ROOT/'data/scenario_outputs'/s/f'{s}_final_observations.csv',index_col=0)
        assert len(df)==n

def test_backfill_group_is_weak_relative_to_spatial_groups():
    df=pd.read_csv(ROOT/'data/processed/ensemble_parameter_group_forecast_associations.csv')
    back=df[df.parameter_group=='backfill conductivity'].max_abs_rho.median()
    spatial=df[df.parameter_group.isin(['upper pilot points','main pilot points','lower pilot points'])].max_abs_rho.median()
    assert back < spatial

def test_threshold_table_not_empty():
    df=pd.read_csv(ROOT/'data/processed/new_threshold_sensitivity.csv')
    assert len(df)>0

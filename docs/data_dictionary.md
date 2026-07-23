# Data dictionary

## Scenario output files

- `*_final_parameters.csv`: retained final parameter ensemble; rows are realizations and columns are adjustable parameters.
- `*_final_observations.csv`: retained final simulated-equivalent and zero-weight forecast ensemble.
- `prior_parensemble.csv`: common 300-draw prior parameter ensemble plus the base row.
- `stage_c_forecast_summary.csv`: deterministic, mean, percentile, and exceedance summaries.
- `stage_c_phi_history.csv`: objective-function history by IES iteration.
- `parameters_metadata.csv`: parameter names, groups, prior values, bounds, and spatial metadata.
- `obs_targets.csv`: scenario-conditioned pseudo-observation targets and weights.

## Processed files

- `deterministic_screening_values.csv`: actual deterministic values shown in main Figure 4.
- `new_scenario_factor_response_screen.csv`: deterministic changes relative to `S0_BASE`, normalized by the full seven-scenario range; shown in SI Figure S3.
- `successful_prior_forecast_realizations.csv`: successful iteration-0 simulated-equivalent rows used in SI Figure S5. These are not all 300 requested draws; scenario-specific counts differ because some prior runs did not yield complete outputs.
- `new_threshold_sensitivity.csv`: exceedance frequency over a range of illustrative benchmark thresholds.
- `new_benchmark_exceedance_wilson_intervals.csv`: finite-ensemble Wilson intervals.
- `new_bootstrap_uncertainty.csv`: bootstrap uncertainty for means, quantiles, and exceedance frequencies.
- `ensemble_parameter_group_forecast_associations.csv`: maximum and median absolute Spearman association by parameter group, forecast, and scenario.
- `dominant_parameter_group_associations.csv`: strongest parameter-group association for each scenario/forecast combination.
- `top_observation_forecast_associations.csv`: leading simulated-observation/forecast associations.
- `dominant_observation_forecast_linkages.csv`: one strongest weighted simulated-observation linkage per scenario and forecast; shown in SI Figure S8.

## Association definitions

For scenario `s`, parameter group `g`, and forecast `F`, the manuscript reports

`Amax(s,g,F) = max over p in g of |rhoS(p,F)|`

and the SI also reports

`Amed(s,g,F) = median over p in g of |rhoS(p,F)|`,

where `rhoS` is the Spearman rank-correlation coefficient calculated across paired retained final realizations. Every main Figure 5 cell is populated because every reported parameter group contained at least one retained parameter with sufficient paired finite values. These statistics are exploratory nonlinear association diagnostics; they are not derivatives, causal effects, PREDVAR/PREDUNC outputs, or model probabilities.

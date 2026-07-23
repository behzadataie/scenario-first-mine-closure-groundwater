# Scenario-first mine-closure groundwater uncertainty benchmark

This repository contains the source code, compact model outputs, processed analysis tables, and editable figures supporting the manuscript **A scenario-first workflow for sparse-data mine-closure groundwater forecasting**.

## Scientific scope

The repository demonstrates a scenario-first groundwater uncertainty workflow:

1. define the decision quantities and benchmark thresholds;
2. register plausible conceptual hypotheses;
3. screen hypotheses against common evidence in real applications;
4. use deterministic forecasts to identify decision-distinct cases; and
5. apply within-scenario stochastic refinement only to retained cases.

The completed stochastic runs are scenario-conditioned synthetic experiments. Each retained scenario was conditioned to pseudo-observations generated from that same scenario. The results quantify variability within internally consistent conceptual states. They do not assign probabilities to conceptual scenarios and are not a common-data multi-model analysis.

## Included

- deterministic MODFLOW 6 model-building and quality-control scripts;
- continuous-K / pilot-point PESTPP-IES setup, forward-model, execution, and post-processing scripts;
- compact final parameter and simulated-observation ensembles for four retained scenarios;
- processed deterministic-screening, threshold-sensitivity, Wilson-interval, bootstrap, parameter-forecast, and observation-forecast tables;
- editable SVG masters used in the manuscript and Supporting Information;
- validation, reproduction, and test scripts;
- documentation for processed-results reproduction and optional full numerical reruns.

## Deliberately excluded

The public repository does not contain manuscript drafts, reviewer reports, response letters, editorial correspondence, personal workstation paths, credentials, licensed executables, transient worker folders, or large MODFLOW binary outputs. MODFLOW 6 and PESTPP-IES must be obtained separately from their official sources.

## Repository map

- `src/stage_a/` - deterministic MODFLOW 6 construction and post-processing
- `src/stage_c/` - PESTPP-IES setup, forward model, execution, and post-processing
- `src/analysis/` - saved-ensemble analysis utilities
- `scripts/` - repository validation, processed-result reproduction, and figure generation
- `data/scenario_outputs/` - compact final scenario inputs and saved ensembles
- `data/processed/` - analysis tables used by the manuscript
- `figures/` - editable SVG figure masters
- `docs/` - method scope, data dictionary, provenance, rerun, and release guidance
- `tests/` - consistency tests

## Quick start: reproduce processed analyses without rerunning MODFLOW or PESTPP-IES

Copy the HTTPS clone address from the GitHub repository page, then run:

```bash
git clone <HTTPS-CLONE-ADDRESS>
cd scenario-first-mine-closure-groundwater
conda env create -f environment.yml
conda activate scenario-first-mine-closure
python scripts/validate_repository.py
python scripts/reproduce_processed_results.py
python scripts/generate_refined_figures.py
pytest -q
```

Generated files are written to `outputs/`, which is excluded from version control.

## Reproducibility levels

1. **Processed-results reproduction:** no external groundwater executables are needed.
2. **Forward-model reproduction:** requires MODFLOW 6 on `PATH`.
3. **Full PESTPP-IES reproduction:** requires MODFLOW 6, PESTPP-IES, and substantial computation time.

See `docs/full_model_rerun.md` for the numerical workflow and `docs/data_dictionary.md` for file definitions.

## Release and DOI

The intended archival workflow is:

1. review this repository in a private GitHub staging repository;
2. make it public only after both authors approve the exact file tree;
3. connect GitHub to Zenodo and enable the repository;
4. create the GitHub release `v1.0.0`;
5. verify the Zenodo software record and version DOI;
6. add the final repository URL and Zenodo DOI to `CITATION.cff` and the manuscript.

Detailed instructions are in `docs/github_zenodo_release_guide.md` and `PUBLIC_RELEASE_CHECKLIST.md`.

## Citation

Use the metadata in `CITATION.cff`. After Zenodo creates the release DOI, add that DOI to `CITATION.cff` and cite the version-specific Zenodo record for reproducibility.

## License

The repository is currently provided under the BSD 3-Clause License. Both authors and their institutions should confirm that this license is appropriate before the repository is made public.

## Security

Do not commit passwords, access tokens, private keys, confidential site data, personal absolute paths, licensed executables, or unpublished peer-review files. See `SECURITY.md` and run:

```bash
python scripts/validate_repository.py
```

before every public release.

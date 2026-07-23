# Optional full model rerun

The paper can be reproduced at the processed-results level without new PESTPP-IES runs. A full numerical rerun is optional.

## External executables

- MODFLOW 6 (`mf6`)
- PESTPP-IES (`pestpp-ies`)

Record executable versions before running.

## Deterministic benchmark

```bash
python src/stage_a/stage_a_basecase_flopy.py --help
python src/stage_a/stage_a_postprocess.py --help
```

Use a clean workspace. Verify mass-balance error, solver convergence, stage hand-off, pit drainage components, and final recovery heads before proceeding.

## PESTPP-IES branch

The compact scenario directories under `data/scenario_outputs/` include the forward-model script, templates, instructions, PEST control file, and starting parameter files. Their `base_config.json` files use `mf6` as the executable name; ensure MODFLOW 6 is on `PATH`.

For each scenario directory:

```bash
python src/stage_c/stage_c_build_pst.py --scenario-dir PATH_TO_SCENARIO
python src/stage_c/stage_c_run_ies.py --scenario-dir PATH_TO_SCENARIO --pestpp PATH_TO_PESTPP_IES
python src/stage_c/stage_c_postprocess_ies.py --scenario-dir PATH_TO_SCENARIO
```

Start with `S0_BASE` and one short test. Confirm that templates, instructions, model commands, and output files are portable before launching parallel workers.

## Run records

For every run, save:

- software versions and operating system;
- random seed;
- PEST++ options;
- prior and observation ensembles;
- worker count and run manager;
- failed realization count and reason;
- final parameter and observation matrices;
- model-list and PEST record files.


## Portability check before a full rerun

```bash
python -m compileall src data/scenario_outputs
python scripts/validate_repository.py
```

The compact directories are sufficient to inspect or restart the PESTPP-IES branch, but the very large original transient workspaces are archived separately. A completely independent rerun should begin with a serial `NOPTMAX=0` forward-model check and verification of mass balance, observation extraction, and forecast names.

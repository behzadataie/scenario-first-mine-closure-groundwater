# Publish and archive the repository

This document is a concise repository-level checklist. The separate publication guide supplied with the release candidate contains the full step-by-step procedure and current official links.

## 1. Validate locally

```bash
conda env create -f environment.yml
conda activate scenario-first-mine-closure
python scripts/validate_repository.py
python scripts/reproduce_processed_results.py
python scripts/generate_refined_figures.py
pytest -q
python -m compileall -q src scripts tests
```

## 2. Create a private GitHub staging repository

Create an empty private repository. Do not initialize it with another README, license, or `.gitignore`.

```bash
git init
git branch -M main
git add .
git status --short
git diff --cached --check
git commit -m "Prepare public reproducibility release v1.0.0"
git remote add origin <HTTPS-REPOSITORY-URL>
git push -u origin main
```

Review the private repository with the co-author before changing visibility.

## 3. Connect Zenodo

1. Link the GitHub account in Zenodo.
2. Enable this repository in the Zenodo GitHub integration.
3. Confirm that `CITATION.cff` is present and `.zenodo.json` is absent.
4. Make the GitHub repository public after the final private review.
5. Create GitHub release `v1.0.0`.
6. Verify the Zenodo software record and DOI.

## 4. Update metadata after DOI creation

Add these fields to `CITATION.cff` after the public repository and Zenodo record exist:

```yaml
date-released: "YYYY-MM-DD"
repository-code: "https://github.com/OWNER/scenario-first-mine-closure-groundwater"
identifiers:
  - type: doi
    value: "10.5281/zenodo.REPLACE"
    description: "Zenodo DOI for version 1.0.0"
```

Commit the metadata update to the default branch. Do not rewrite the already archived `v1.0.0` tag.

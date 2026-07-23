#!/usr/bin/env python3
"""Validate the compact public repository before release."""
from __future__ import annotations

from pathlib import Path
import json
import re
import zipfile

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ["S0_BASE", "S2_CONN", "S3_BUFF", "S6_UPRISK"]
EXPECTED_ROWS = {"S0_BASE": 58, "S2_CONN": 38, "S3_BUFF": 89, "S6_UPRISK": 42}
FORECASTS = [
    "fcst_max_receptor_dd",
    "fcst_max_compliance_dd",
    "fcst_stage4_mean_inflow",
    "fcst_recovery_years",
]

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".yml", ".yaml", ".csv", ".pst",
    ".cff", ".dat", ".tpl", ".ins", ".info",
}
FORBIDDEN_SUFFIXES = {
    ".docx", ".pdf", ".zip", ".7z", ".rar", ".exe", ".dll", ".msi",
    ".pem", ".key", ".p12", ".pfx", ".hds", ".cbc", ".grb",
}
FORBIDDEN_NAMES = {".zenodo.json", ".env", "id_rsa", "id_ed25519"}
SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|client[_-]?secret|access[_-]?token|password)\s*[:=]\s*['\"][^'\"]+"),
]
ABS_WINDOWS = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
ABS_HOME = re.compile(r"/(?:home|Users)/[^\s\"']+")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PLACEHOLDERS = ("REPLACE_WITH_ACCOUNT", "YOUR_ACCOUNT", "YOUR_GITHUB", "AtaieAB", "/mnt/data")


def validate_ensembles() -> None:
    for scenario in SCENARIOS:
        folder = ROOT / "data" / "scenario_outputs" / scenario
        par_path = folder / f"{scenario}_final_parameters.csv"
        obs_path = folder / f"{scenario}_final_observations.csv"
        assert par_path.exists() and obs_path.exists(), f"Missing ensemble CSV for {scenario}"
        par = pd.read_csv(par_path, index_col=0)
        obs = pd.read_csv(obs_path, index_col=0)
        assert len(par) == EXPECTED_ROWS[scenario], (scenario, len(par), EXPECTED_ROWS[scenario])
        assert len(obs) == EXPECTED_ROWS[scenario], (scenario, len(obs), EXPECTED_ROWS[scenario])
        assert set(par.index) == set(obs.index), f"Index mismatch: {scenario}"
        for forecast in FORECASTS:
            assert forecast in obs.columns, (scenario, forecast)
        cfg = json.loads((folder / "base_config.json").read_text(encoding="utf-8"))
        assert cfg.get("mf6_exe") == "mf6", f"Nonportable mf6_exe in {scenario}/base_config.json"
        pst = (folder / "continuous_k.pst").read_text(encoding="utf-8", errors="ignore")
        assert "python forward_model.py" in pst, f"Portable model command not found in {scenario}/continuous_k.pst"


def validate_required_files() -> None:
    required = [
        "README.md", "LICENSE", "CITATION.cff", "VERSION", "SECURITY.md",
        "PUBLIC_RELEASE_CHECKLIST.md", "PUBLIC_RELEASE_AUDIT.md", "environment.yml", "requirements.txt",
        "RELEASE_NOTES.md", "docs/github_zenodo_release_guide.md",
        "figures/main/Fig01_generic_scenario_first_workflow.svg",
        "figures/main/Fig04_deterministic_screening.svg",
        "figures/main/Fig05_ensemble_parameter_forecast_associations.svg",
        "figures/supporting_information/FigS04_prior_parameter_distributions.svg",
        "figures/supporting_information/FigS05_successful_prior_forecast_distributions.svg",
        "figures/supporting_information/FigS08_observation_forecast_linkage.svg",
        "data/processed/deterministic_screening_values.csv",
        "data/processed/successful_prior_forecast_realizations.csv",
        "scripts/generate_refined_figures.py", "docs/figure_provenance.md",
    ]
    for rel in required:
        assert (ROOT / rel).exists(), f"Missing required file: {rel}"
    assert not (ROOT / "manuscript").exists(), "Private manuscript/review folder must not be public"
    assert not (ROOT / ".zenodo.json").exists(), "Use one metadata source; .zenodo.json must be absent"


def validate_citation() -> None:
    path = ROOT / "CITATION.cff"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data.get("cff-version") == "1.2.0"
    assert data.get("title")
    assert data.get("type") == "software"
    assert data.get("authors") and len(data["authors"]) >= 1
    assert data.get("version") == (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    assert "release-date" not in data, "CFF field is date-released, not release-date"


def validate_release_hygiene() -> None:
    bad: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        rel_posix = rel.as_posix()
        if ".git" in rel.parts or "__pycache__" in rel.parts or ".pytest_cache" in rel.parts:
            continue
        if path.stat().st_size > 50 * 1024 * 1024:
            bad.append(f"file over 50 MiB: {rel_posix}")
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            bad.append(f"forbidden public-release file: {rel_posix}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"README.md", ".gitignore", ".gitattributes", "VERSION"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if rel_posix != "scripts/validate_repository.py":
                if ABS_WINDOWS.search(text) or ABS_HOME.search(text):
                    bad.append(f"absolute local path: {rel_posix}")
                if EMAIL.search(text):
                    bad.append(f"email address in public text file: {rel_posix}")
                for token in PLACEHOLDERS:
                    if token in text:
                        bad.append(f"unresolved/private token {token!r}: {rel_posix}")
                for pattern in SECRET_PATTERNS:
                    if pattern.search(text):
                        bad.append(f"possible secret: {rel_posix}")
    assert not bad, "Release-hygiene problems:\n- " + "\n- ".join(sorted(set(bad)))


def validate_zip_safety_if_present() -> None:
    # The repository itself should not contain archives, but this helper is useful
    # when the validator is run against a staged tree containing a release ZIP.
    for path in ROOT.rglob("*.zip"):
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                parts = Path(info.filename).parts
                assert not Path(info.filename).is_absolute() and ".." not in parts, f"Unsafe ZIP member: {path}: {info.filename}"


def main() -> None:
    validate_required_files()
    validate_ensembles()
    validate_citation()
    validate_release_hygiene()
    validate_zip_safety_if_present()
    print("Repository validation passed.")


if __name__ == "__main__":
    main()

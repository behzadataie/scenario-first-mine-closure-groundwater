#!/usr/bin/env python3
"""Finalize public repository metadata before or after the Zenodo release.

Examples
--------
Before creating the GitHub release:
    python scripts/finalize_release_metadata.py --owner GITHUB_OWNER --date 2026-08-01

After Zenodo mints the version DOI:
    python scripts/finalize_release_metadata.py --owner GITHUB_OWNER --date 2026-08-01 --doi 10.5281/zenodo.1234567
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFF = ROOT / "CITATION.cff"
README = ROOT / "README.md"
REPO_NAME = "scenario-first-mine-closure-groundwater"


def valid_iso_date(value: str) -> str:
    date.fromisoformat(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True, help="GitHub user or organization name")
    parser.add_argument("--date", type=valid_iso_date, required=True, help="Release date, YYYY-MM-DD")
    parser.add_argument("--doi", default=None, help="Version-specific Zenodo DOI, without https://doi.org/")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", args.owner):
        raise ValueError("The GitHub owner name is not in an expected form")
    if args.doi and not re.fullmatch(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", args.doi):
        raise ValueError("The DOI is not in an expected form")

    data = yaml.safe_load(CFF.read_text(encoding="utf-8"))
    data["date-released"] = args.date
    data["repository-code"] = f"https://github.com/{args.owner}/{REPO_NAME}"
    if args.doi:
        data["identifiers"] = [
            {
                "type": "doi",
                "value": args.doi,
                "description": "Zenodo DOI for version 1.0.0",
            }
        ]

    CFF.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=1000),
        encoding="utf-8",
        newline="\n",
    )

    text = README.read_text(encoding="utf-8")
    repo_url = f"https://github.com/{args.owner}/{REPO_NAME}"
    marker = "## Public identifiers"
    block = f"{marker}\n\n- GitHub: {repo_url}\n"
    if args.doi:
        block += f"- Zenodo version DOI: https://doi.org/{args.doi}\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n\n" + block
    else:
        text = text.rstrip() + "\n\n" + block
    README.write_text(text, encoding="utf-8", newline="\n")

    print(f"Updated {CFF.relative_to(ROOT)} and {README.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

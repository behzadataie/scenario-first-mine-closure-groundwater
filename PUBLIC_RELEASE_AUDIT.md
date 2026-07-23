# Public release security audit

Audit date: 2026-07-23

## Source archive

- Source file: `scenario-first-mine-closure-groundwater_refined_GitHub_ready(1).zip`
- SHA-256: `950a37182eacdbf846bd0b9e1c80873984bf3283d34fa669d3253ddf26920484`
- ZIP entries: 248
- Regular files: 226
- Uncompressed size: 8,689,692 bytes
- Path traversal entries: none detected
- Symbolic links: none detected
- Executable binaries: none detected
- Obvious passwords, API tokens, private keys, or credential files: none detected

## Issues corrected before public release

1. Removed the entire `manuscript/` directory. It contained manuscript drafts, Supporting Information, a reviewer response, a technical note, a restoration rationale, and an author-check guide. These are unnecessary for software/data reproducibility and may expose peer-review material and author contact details.
2. Removed `.zenodo.json`. Zenodo gives `.zenodo.json` priority when both `.zenodo.json` and `CITATION.cff` are present, so keeping both creates an avoidable metadata conflict.
3. Replaced the invalid CFF field `release-date` with a valid release-ready structure. `date-released`, `repository-code`, and DOI identifiers are intentionally added only after the public repository and Zenodo record exist.
4. Removed unresolved repository-owner placeholders from release metadata and public-facing instructions.
5. Updated README and release notes so they no longer claim that private manuscript-review files are part of the public repository.
6. Added `SECURITY.md`, `PUBLIC_RELEASE_CHECKLIST.md`, `VERSION`, and Dependabot configuration.
7. Hardened GitHub Actions with read-only repository permissions, disabled persisted checkout credentials, and added a timeout.
8. Expanded `.gitignore` to reduce accidental publication of credentials, manuscript-review files, licensed executables, transient workspaces, and large MODFLOW binaries.
9. Strengthened `scripts/validate_repository.py` to reject email addresses in public text files, private paths, unresolved placeholders, common secret patterns, manuscript-review files, archives, executables, and large model binaries.
10. Removed generated caches and temporary output folders, regenerated data-driven figures, and normalized trailing whitespace so a clean Git staging check succeeds.

## Sanitized repository status

- Public-release files: 222
- Uncompressed size: 4,789,527 bytes
- Largest file: 421,366 bytes
- Files over 50 MiB: none
- Git LFS required: no
- Email addresses in public text files: none detected
- Personal absolute paths: none detected
- Unresolved GitHub-owner placeholders in release metadata: none
- `.zenodo.json`: absent
- `CITATION.cff`: present and parsed successfully
- Manuscript/reviewer DOCX files: absent
- Licensed executables: absent
- Large MODFLOW `*.hds`, `*.cbc`, and `*.grb` outputs: absent

## Validation completed

- repository validation: passed
- processed-result reproduction: passed
- data-driven figure generation: passed
- automated tests: 3 passed
- Python source compilation: passed
- clean Git initialization, staging, `git diff --cached --check`, and commit simulation: passed

## Validation boundary

The audit did not rerun MODFLOW 6 or PESTPP-IES and did not install external executables. The compact repository is intended to reproduce processed analyses and figures directly; a full groundwater-model rerun remains optional and requires separately obtained MODFLOW 6 and PESTPP-IES executables.

No automated audit can guarantee that all institutional, intellectual-property, or confidentiality obligations have been satisfied. Both authors should complete `PUBLIC_RELEASE_CHECKLIST.md` before the repository is made public or archived in Zenodo.

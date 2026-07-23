# Public release checklist

Complete this checklist before changing the GitHub repository from private to public or creating a Zenodo release.

## Authorship and rights

- [ ] Both authors approve the exact repository file tree.
- [ ] University of Newcastle and Jacobs publication, intellectual-property, and licensing requirements have been checked.
- [ ] The BSD 3-Clause license is approved for the code, data, figures, and documentation included here.
- [ ] Author names and ORCID identifiers in `CITATION.cff` are correct.

## Privacy and security

- [ ] GitHub two-factor authentication or a passkey is enabled.
- [ ] The Git commit email uses the intended public or GitHub no-reply address.
- [ ] No password, token, private key, credential file, or `.env` file is present.
- [ ] No personal absolute path, confidential site data, manuscript draft, reviewer file, or response letter is present.
- [ ] No licensed executable is present.
- [ ] No file exceeds the intended ordinary Git size limit.

## Technical validation

- [ ] `python scripts/validate_repository.py` passes.
- [ ] `python scripts/reproduce_processed_results.py` passes.
- [ ] `python scripts/generate_refined_figures.py` completes.
- [ ] `pytest -q` passes.
- [ ] `python -m compileall -q src scripts tests` passes.
- [ ] `sha256sum -c SHA256SUMS.txt` passes where `sha256sum` is available.
- [ ] GitHub Actions passes in the private staging repository.

## GitHub and Zenodo

- [ ] The private GitHub staging repository has been reviewed by both authors.
- [ ] Repository description and topics are correct.
- [ ] Zenodo is connected to the correct GitHub account.
- [ ] The repository is enabled in Zenodo before the GitHub release is created.
- [ ] `CITATION.cff` is valid and `.zenodo.json` is absent.
- [ ] GitHub release `v1.0.0` is created only after the public repository is final.
- [ ] The Zenodo title, creators, ORCIDs, license, version, keywords, and files are verified.
- [ ] The version-specific Zenodo DOI is inserted in the manuscript and citation metadata.

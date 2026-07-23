# GitHub and archival release

The repository is designed for a private GitHub staging review followed by a versioned Zenodo archive.

## Required sequence

1. Confirm authorship, institutional rights, and the repository license.
2. Run repository validation, processed-result reproduction, figure generation, tests, and Python compilation.
3. Initialize Git locally and inspect every staged file before committing.
4. Push first to a private GitHub repository and invite the co-author to review it.
5. Change the repository to public only after both authors approve the final tree.
6. Connect GitHub to Zenodo and enable the repository.
7. Create GitHub release `v1.0.0` after Zenodo has been enabled.
8. Verify the Zenodo software record and version-specific DOI.
9. Add the DOI and public repository URL to the manuscript and `CITATION.cff`.

Do not publish personal workstation paths, credentials, confidential site data, licensed executables, manuscript drafts, reviewer reports, response letters, or large transient model workspaces.

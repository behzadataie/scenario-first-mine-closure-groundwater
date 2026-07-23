# Security policy

## Reporting a security or privacy issue

Do not open a public issue if you discover a credential, private key, confidential file, personal path, or other sensitive information. Contact the repository owner privately through the contact method shown on the GitHub profile or institutional page.

## Supported release

Security and privacy corrections are applied to the current public release and the default branch.

## Repository rules

The repository must not contain:

- passwords, API tokens, private keys, or authentication files;
- private or client data;
- personal workstation paths or user names;
- unpublished reviewer reports, response letters, or editorial correspondence;
- licensed MODFLOW 6 or PEST++ executables;
- large transient model workspaces or worker directories;
- files whose redistribution rights have not been confirmed.

Run `python scripts/validate_repository.py` before each release and review the staged Git file list manually before pushing.

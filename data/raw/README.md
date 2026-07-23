# Raw and large numerical outputs

The GitHub repository intentionally omits large transient workspaces and binary MODFLOW outputs (`*.hds`, `*.cbc`, `*.grb`, full `model_run/` folders). They are not required to reproduce the processed manuscript tables and figures.

For complete numerical preservation, place the following in a Zenodo release or institutional archive:

- predevelopment, four operational stages, and recovery workspaces;
- MODFLOW 6 list, head, and budget files;
- full PESTPP-IES worker/master directories;
- any original ZIP archives used to assemble the compact scenario outputs.

Add a checksum manifest and the archive DOI to this file before public release.

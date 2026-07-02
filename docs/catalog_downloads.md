# Star Catalog Download Paths

Use `scripts/download_catalogs.py` to download the required first-pass catalogs:

```bash
conda run -n hgastro python scripts/download_catalogs.py
```

The script creates `catalog/` if it does not exist and skips complete existing
files by default.

## Bright Star Catalog / Yale BSC

Use this first for bright-star and constellation-level interactive matching.

- Catalog page: https://cdsarc.cds.unistra.fr/viz-bin/cat/V/50
- FTP directory: https://cdsarc.cds.unistra.fr/ftp/V/50/
- Data file: https://cdsarc.cds.unistra.fr/ftp/V/50/catalog.gz
- ReadMe: https://cdsarc.cds.unistra.fr/ftp/V/50/ReadMe
- Optional notes: https://cdsarc.cds.unistra.fr/ftp/V/50/notes.gz
- Local data file: `catalog/yale_bsc/catalog.gz`
- Local ReadMe: `catalog/yale_bsc/ReadMe`

## Hipparcos Original Catalog

Use this for wide-angle and medium-wide images when the BSC is too sparse.

- Catalog page: https://cdsarc.cds.unistra.fr/viz-bin/cat/I/239
- FTP directory: https://cdsarc.cds.unistra.fr/ftp/I/239/
- Data file: https://cdsarc.cds.unistra.fr/ftp/I/239/hip_main.dat
- ReadMe: https://cdsarc.cds.unistra.fr/ftp/I/239/ReadMe
- Local data file: `catalog/hipparcos_i239/hip_main.dat`
- Local ReadMe: `catalog/hipparcos_i239/ReadMe`

## IAU Catalog of Star Names (CSN)

Use this to add IAU-approved English display names for bright stars in the
render window.

- Catalog page: https://exopla.net/star-names/modern-iau-star-names/
- Local HTML snapshot: `catalog/iau_csn/modern_iau_star_names.html`

## d3-celestial Milky Way GeoJSON

Use this to render the Milky Way as a semi-transparent band in the simulated
camera view.

- Project: https://github.com/ofrohn/d3-celestial
- Versioned data file: https://cdn.jsdelivr.net/npm/d3-celestial@0.7.35/data/mw.json
- Source data note: d3-celestial documents `mw.json` as Milky Way outlines in
  five brightness steps, converted to GeoJSON J2000 coordinates.
- Local data file: `catalog/d3_celestial/mw.json`

## Hipparcos New Reduction

This is a useful alternative to the original Hipparcos main catalog.

- Catalog page: https://cdsarc.cds.unistra.fr/viz-bin/cat/I/311
- FTP directory: https://cdsarc.cds.unistra.fr/ftp/I/311/
- Data file: https://cdsarc.cds.unistra.fr/ftp/I/311/hip2.dat.gz
- ReadMe: https://cdsarc.cds.unistra.fr/ftp/I/311/ReadMe
- Suggested local path if downloaded later: `catalog/hipparcos_i311/hip2.dat.gz`

For the first implementation, filter to approximately `mag <= 6.5` for manual
bright-star matching and `mag <= 8.5` for assisted wide-field matching. No dark
star catalog is needed for the current plan.

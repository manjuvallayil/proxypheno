# Data

The Dryad deposit is not committed. Download it manually (Dryad blocks bots).

## Required

From https://datadryad.org/dataset/doi:10.5061/dryad.kwh70rz1z:

- `README.dryad.md`
- `phens.tar.gz` (10.6 MB)

Extract so the layout is:

```
data/raw/
  README.dryad.md
  phens/
    phen_1.1.txt
    ...
```

Or place the same `phens/` folder at a sibling `../dryad/` path. The resolver in
`src/proxypheno/config.py` checks both.

## Optional (genetic validation)

- `gens.tar.gz` (8.2 GB): PLINK genotypes, including `gen_2.*` / `gen_3.*`
  prioritised mutations and `gen_11/12` simulated genotypes.
- `other.tar.gz` (5.6 GB): pedigree, GWAS summary statistics, inbreeding and
  breed information.

Extract as `data/raw/gens/` and `data/raw/other/`.

## Licence

Follow the Dryad deposit terms and cite the dataset and paper in any derivative
work. Do not commit raw genotypes or phenotypes to the public repository.

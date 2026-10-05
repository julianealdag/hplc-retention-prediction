# Data

The MCMRT workbooks are not stored in this repository. Download them with:

```bash
python scripts/download_data.py
```

This puts the 30 `.xlsx` files here and checks their MD5 checksums:

```
data/raw/Dataset 01.xlsx ... Dataset 30.xlsx
```

Source: Science Data Bank, https://doi.org/10.57760/sciencedb.15823 (CC0).
If the script fails, download the files by hand from that page.

> Zhang, Y. *et al.* Retention time dataset for heterogeneous molecules in
> reversed-phase liquid chromatography. *Scientific Data* **11**, 946 (2024).
> https://doi.org/10.1038/s41597-024-03780-5

To exercise the pipeline without the real files:

```bash
python scripts/make_synthetic_data.py --out data/synthetic --n-experiments 3
hplc-rt evaluate --data-dir data/synthetic --pooled-only --models Ridge
```

`data/raw/` and `data/synthetic/` are gitignored.

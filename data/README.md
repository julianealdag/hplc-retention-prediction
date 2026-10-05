# Data

The MCMRT workbooks are not redistributed in this repository (they are not ours
to republish). Download the 30 `.xlsx` files from the source publication and put
them here:

```
data/raw/*.xlsx
```

> Zhang, Y. *et al.* Retention time dataset for heterogeneous molecules in
> reversed-phase liquid chromatography. *Scientific Data* **11**, 946 (2024).
> https://doi.org/10.1038/s41597-024-03780-5
> Dataset: https://doi.org/10.57760/sciencedb.15823

To exercise the pipeline without the real files:

```bash
python scripts/make_synthetic_data.py --out data/synthetic --n-experiments 3
hplc-rt evaluate --data-dir data/synthetic --pooled-only --models Ridge
```

`data/raw/` and `data/synthetic/` are gitignored.

# Trying DonorShift-Bench on public breast-cancer data

The first release of the split comparison uses synthetic counts. A suitable real-data follow-up is the breast-cancer atlas from Wu et al. (2021), GEO **GSE176078**.

- [GSE176078 on NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176078)
- [Wu et al., Nature Genetics (2021)](https://www.nature.com/articles/s41588-021-00911-1)
- [Processed scRNA-seq data in the Broad Single Cell Portal, study SCP1039](https://singlecell.broadinstitute.org/single_cell/study/SCP1039)

The paper includes single-cell measurements from 26 primary breast tumors (11 ER+, 5 HER2+, 10 TNBC). GEO lists a processed scRNA-seq tarball of approximately 533 MB. Raw sequencing data are controlled-access at EGA; **do not confuse them with the public processed counts**. These public sources describe where to obtain data, but we have not downloaded and benchmarked them for v0.2.

## Prepare your input

1. Obtain a processed **raw-count** matrix and matching cell annotations from the sources above. Inspect the data files rather than assuming a particular filename, column name or `.h5ad` layer exists.
2. Confirm that each expression row refers to exactly one cell and that its barcode maps to the correct sample and patient. Keep the mapping as a reproducible preprocessing script outside this package.
3. Check whether a patient contributes more than one sample. For donor-held-out evaluation, **all samples from one patient must have the same donor ID**.
4. Check that your cell-type labels are comparable across samples. Do not claim an unannotated matrix has reference cell-type labels.
5. Preserve the original count matrix for model input. Normalized/log-transformed data should not be passed into the raw-count loader.
6. Record sample inclusion/exclusion, filtering thresholds, annotation source and accession/version in your study notes. Any global learned preprocessing (including HVG selection and batch correction) needs a leakage assessment.

The expected `.h5ad` object has an observation (`.obs`) column for donor IDs and one for cell types. Raw integer counts can be in `.X` or a named layer. Don't guess which: inspect the object first.

```python
import anndata as ad
import numpy as np
from scipy import sparse

adata = ad.read_h5ad('cohort.h5ad')
print(adata.shape)
print(adata.obs.columns.tolist())
print(list(adata.layers.keys()))
print(adata.obs[['patient_id', 'cell_type']].isna().sum())
X = adata.layers['counts']  # only if this layer has the raw UMI counts
values = X.data if sparse.issparse(X) else np.asarray(X)
print('finite:', np.isfinite(values).all(), 'integer-like:', np.allclose(values, np.rint(values)))
```

For raw data in `.X`, omit `--counts-layer`. The sample column names in the snippet above are placeholders until you verify your file.

## Validate, then compare

```bash
python -m pip install -e '.[h5ad]'

donorshift validate --input cohort.h5ad \
  --donor-col patient_id --label-col cell_type --counts-layer counts

donorshift compare --input cohort.h5ad \
  --donor-col patient_id --label-col cell_type --counts-layer counts \
  --models logreg random_forest --top-genes 3000 \
  --test-fraction 0.2 --random-repeats 3 \
  --dataset-kind public --accession GSE176078 \
  --output results/GSE176078
```

The comparison saves a `cohort_audit.csv` with cells per donor and type. Review that file before interpreting results. Check particularly for donors with very few cells or classes missing from other donors' training data.

## What to report

Use the same fixed model settings across protocols, but state clearly that the training cohorts are not the same size. Report equal-donor mean and per-donor metrics, random-split replicate variation, all model failures and the exact version of the annotation dataset. Any claimed patient-level generalization should be based on the **donor-held-out** results, not on the within-donor cell split.

Do not upload controlled-access counts, patient identifiers, unlicensed matrices or private clinical metadata to a public GitHub repository. Only the synthetic example is committed to this project.

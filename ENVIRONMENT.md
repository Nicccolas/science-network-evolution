# Research environment

This environment supports the study of whether the spread of deep-learning methods is associated with increasing cross-disciplinary connections in scientific citation networks over time.

## Start

```bash
conda activate macs40123
jupyter lab
```

Run these commands from the `science-network-evolution` project directory. Select the `Python (macs40123)` kernel in notebooks.

## Recreate

```bash
conda env create --file environment.yml
```

`environment.yml` pins Python to the 3.12 series and direct dependencies to their installed versions, using only conda-forge. Conda resolves platform-specific dependencies. For an exact package/build recreation on Apple Silicon macOS, use the additional explicit package list:

```bash
conda create --name macs40123 --file conda-osx-arm64.lock.txt
```

These creation commands assume the target environment does not already exist; use another name to create a separate copy. The explicit lock is specific to `osx-arm64` and includes transitive dependencies.

## Package choices

| Task | Packages |
| --- | --- |
| OpenAlex requests, progress, local configuration | requests, tqdm, python-dotenv |
| Data cleaning and Parquet datasets | pandas, pyarrow |
| Numerical work and sparse matrices | numpy, scipy |
| Title/abstract TF-IDF features, classification, evaluation | scikit-learn |
| Statistical models of changes over time | statsmodels |
| Directed citation networks and disciplinary mixing | networkx |
| Larger graph analysis and Leiden communities | python-igraph, leidenalg |
| Notebooks | jupyter, jupyterlab, ipykernel |
| Figures | matplotlib, seaborn |

The initial text-classification tools support keyword and TF-IDF baselines without downloading pretrained models. Package installation does not determine the study's community definition, graph projection, treatment of edge direction, temporal windows, or validation strategy; those remain research design choices.

Community-detection installation references: [igraph](https://python.igraph.org/en/main/install.html) and [leidenalg](https://leidenalg.readthedocs.io/en/latest/install.html).

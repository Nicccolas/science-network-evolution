# Research environment

This environment supports the study of how the semantic breadth of a scientific field changes as its publication volume grows (see [`docs/research_design.md`](docs/research_design.md)). The network packages remain from the project's earlier citation-network phase. It also includes PyTorch, Hugging Face `transformers`, and `adapters` for the SPECTER2 document-embedding model.

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
conda run -n macs40123 pip install --no-deps adapters==1.3.0
```

The explicit list records conda packages only, so `adapters` (PyPI-only) is installed separately. `--no-deps` stops pip from replacing the conda-installed `transformers`.

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
| Document embeddings (SPECTER2, MiniLM) on the Apple GPU (`mps`) | pytorch, transformers, adapters, sentence-transformers, huggingface_hub |
| Notebooks | jupyter, jupyterlab, ipykernel |
| Figures | matplotlib, seaborn |

The initial text-classification tools support keyword and TF-IDF baselines without downloading pretrained models. Package installation does not determine the study's community definition, graph projection, treatment of edge direction, temporal windows, or validation strategy; those remain research design choices.

## Embedding-stack constraints

- **Install PyTorch from conda-forge, never with `pip install torch`.** The pip wheel bundles its own OpenMP runtime, which collides with conda-forge's `llvm-openmp`, so `import torch` aborts with `OMP: Error #15`. Do not work around this with `KMP_DUPLICATE_LIB_OK=TRUE`; it can silently corrupt results.
- **`transformers` is held at 4.57.x** because `adapters` 1.3.0 requires `transformers~=4.57.6`. Upgrading `transformers` or `sentence-transformers` to 5.x/6.x breaks `adapters`. Check `pip check` after any change.
- **SPECTER2 = base model + proximity adapter.** Load `allenai/specter2_base` with `adapters.AutoAdapterModel`, then `load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)`. Input is `title + tokenizer.sep_token + abstract`, truncated at 512 tokens; the embedding is the first-token (`[CLS]`) vector, 768 dimensions. The message "There are adapters available but none are activated" appears while the adapter loads and is harmless; confirm with `model.active_adapters` (should show `Stack[proximity]`). Record the model revision hashes when generating research embeddings.
- Model weights download to `~/.cache/huggingface` (SPECTER2 is about 0.5 GB), which counts against the laptop's limited free disk.

Community-detection installation references: [igraph](https://python.igraph.org/en/main/install.html) and [leidenalg](https://leidenalg.readthedocs.io/en/latest/install.html).

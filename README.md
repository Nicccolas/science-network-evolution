# Science Network Evolution

This repository supports a research project using OpenAlex publication metadata to study the spread of deep-learning methods across scientific disciplines.

The main research question is:

> As deep-learning methods spread across scientific disciplines, do scientific research communities become increasingly cross-disciplinary, or do they remain separated along disciplinary boundaries?

The planned analysis combines paper titles and abstracts, OpenAlex disciplinary labels, publication dates, and citation relationships. Papers form network nodes, references form directed edges, and the OpenAlex primary topic hierarchy supplies predefined topic, subfield, field, and domain labels.

## Current status

The repository currently contains:

- A reusable OpenAlex API extractor for balanced yearly samples.
- Saved 2024 and 2013–2016 API samples.
- An exploratory notebook for abstract vectorization and similarity analysis.
- A reproducible Conda environment for text, statistical, and network analysis.
- A bounded AWS Athena/S3 technical pilot for the OpenAlex Parquet snapshot.

The saved API data are research-development samples, not complete yearly cohorts. Their raw JSON contains the primary taxonomy hierarchy, but the saved CSVs predate the explicit field, subfield, and domain columns now produced by `extract.py`; the notebook temporarily reconstructs those columns in memory. The Athena work is also a small technical pilot rather than a complete Psychology dataset.

## Quick start

Create and activate the Conda environment:

```bash
conda env create --file environment.yml
conda activate macs40123
```

Start Jupyter from the repository root:

```bash
jupyter lab
```

Open `exploratory_analysis.ipynb` and select the `Python (macs40123)` kernel. See [`ENVIRONMENT.md`](ENVIRONMENT.md) for package details and exact Apple Silicon recreation instructions.

To collect an API sample before analysis, follow [`api_test/README.md`](api_test/README.md). For example:

```bash
export OPENALEX_API_KEY="your-key"
python3 api_test/extract.py \
  --min-year 2013 --max-year 2016 \
  --size 200 \
  --output api_test/data_2013_2016
```

## Repository guide

### Research and environment files

| Path | Purpose |
| --- | --- |
| `README.md` | Project overview, workflow, and repository guide |
| `ENVIRONMENT.md` | Environment setup and explanation of package choices |
| `environment.yml` | Reproducible cross-platform Conda environment using conda-forge |
| `conda-osx-arm64.lock.txt` | Exact package/build list for Apple Silicon macOS |
| `exploratory_analysis.ipynb` | Text cleaning, count vectors, TF-IDF, sentence embeddings, similarity, and early network-analysis exploration |

### API extraction

| Path | Purpose |
| --- | --- |
| `api_test/README.md` | Complete usage guide for the API extractor |
| `api_test/extract.py` | Samples OpenAlex works, reconstructs abstracts, flattens primary taxonomy, and builds citation tables |
| `api_test/test_extract.py` | Offline automated tests using simulated API responses |
| `api_test/data/` | Saved 50-paper English-article sample from 2024; its CSV uses the earlier 19-column format |
| `api_test/data_2013_2016/` | Saved 800-paper sample with 200 English-language articles from each year, 2013–2016; its CSV uses the earlier format |

Each saved API data directory contains:

| File | Purpose |
| --- | --- |
| `works.csv` | Analysis-ready paper metadata; new extractions include explicit primary taxonomy columns |
| `works_raw.json` | Selected nested OpenAlex records retained for provenance and later transformations |
| `citation_edges.csv` | Directed `citing work → cited work` edge list |
| `citation_nodes.csv` | Sampled works and external referenced-work endpoints |
| `metadata.json` | Extraction parameters, timestamps, validation counts, missingness, and API cost |

### Athena/S3 pilot

| Path | Purpose |
| --- | --- |
| `athena_pilot/README.md` | Provenance, AWS resources, costs, query IDs, results, and reproduction notes for the bounded pilot |
| `athena_pilot/source.sql` | Athena table definition for nested OpenAlex Parquet works |
| `athena_pilot/partitions.sql` | Registration of the five selected update-date partitions |
| `athena_pilot/cohort_count.sql` | Count query for 2015 primary-field Psychology works |
| `athena_pilot/source_counts.sql` | Counts validating the bounded source data |
| `athena_pilot/nested_fields_check.sql` | Readability checks for nested references, topics, authorships, and abstract data |

## Data workflow

```text
OpenAlex API
    │
    ├── works_raw.json       nested source records and provenance
    ├── works.csv            analysis-ready paper metadata
    ├── citation_edges.csv   directed citation relationships
    └── citation_nodes.csv   sampled and external work IDs
             │
             ▼
exploratory_analysis.ipynb
    ├── abstract cleaning
    ├── count and TF-IDF representations
    ├── sentence embeddings
    ├── similarity comparisons
    └── later community and disciplinary-mixing analysis
```

The extractor uses canonical OpenAlex IDs as relational keys. `works.csv.openalex_id` joins sampled paper metadata to citation-edge `Source` and `Target` values. Referenced works outside the sample remain ID-only nodes unless their metadata is fetched separately.

## Methodological notes

- API extractions are balanced samples with up to the requested number of papers per year; they do not preserve the actual yearly publication distribution.
- OpenAlex's `language` value describes title and abstract metadata, not necessarily full-text language.
- The current CSV taxonomy represents the single `primary_topic` hierarchy. It does not represent all secondary OpenAlex topic assignments.
- An edge points from the citing work to the cited work. The extractor collects outgoing references only; it does not fetch incoming citations or recursively expand external works.
- Field-to-field citation analysis requires taxonomy metadata for both endpoints. Initially this is available only for edges whose source and target are both sampled works.
- OpenAlex metadata and citation counts change over time. Preserve `works_raw.json` and `metadata.json` with research outputs for reproducibility.

## Validation

Run the offline extractor tests from the repository root:

```bash
python3 -m unittest api_test.test_extract -v
```

The tests do not call OpenAlex or overwrite the saved datasets. They check paging, multi-year sampling, filters, taxonomy flattening, citation relationships, checkpoints, duplicate IDs, and missing values.

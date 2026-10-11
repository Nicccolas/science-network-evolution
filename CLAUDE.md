# CLAUDE.md

Guidance for AI coding agents working in this repository.

## Project purpose

A **publication-ready social-science research paper**, developed as a class project for
MACS 40123 (UChicago). The working title is *Publication Growth and the Semantic Breadth
of Science: A Longitudinal Case Study Using Document Embeddings and Clustering*.

> How does the semantic breadth of research change as publication volume grows within a
> scientific field, and how do shifts in thematic composition accompany this relationship?

The core pipeline: OpenAlex titles and abstracts for one field → SPECTER2 embeddings
(provisional) → annual **knowledge extent** (KE: maximum distance to centroid, in fixed
2,000-paper samples resampled 1,000 times, following Hao et al. 2026) → shared k-means
**thematic clusters** → **time-series** regression of Δlog KE on Δlog publication count.

## Read these first

| File | What it holds |
| --- | --- |
| [`docs/research_design.md`](docs/research_design.md) | **Canonical proposal.** Do not change the methods silently. |
| [`docs/ideas_backlog.md`](docs/ideas_backlog.md) | Extensions not to forget: nearest-neighbour composition (Motohashi et al. 2024), sliding 3-year windows, journal-based labels, metric/centroid pairing, citation-pair validation, etc. |
| [`docs/decision_log.md`](docs/decision_log.md) | Open and settled analytic choices, with dates and reasons. |
| [`docs/data_strategy.md`](docs/data_strategy.md) | Storage/compute budget and API-vs-AWS plan. |
| [`docs/literature.md`](docs/literature.md) | How each cited work is used; citations still need verification. |

When the user mentions a new idea or option they "don't want to forget", add it to
`docs/ideas_backlog.md`. When a choice is made, add it to `docs/decision_log.md`.

## Repository layout

| Path | Role |
| --- | --- |
| `WeekN_notebook.ipynb` (e.g. `Week1_notebook.ipynb`) | **Course assignment notebooks.** Each is graded on its own: a literature review, a method demonstration, and a reflection. Keep them self-contained and readable, and do not restructure or rename them without asking. They may prototype methods, but the research pipeline should not depend on them. |
| `exploratory_analysis.ipynb` | Early exploration: count vectors, TF-IDF, MiniLM embeddings, cosine similarity, neighbour overlap. |
| `api_test/extract.py` (+ `test_extract.py`) | OpenAlex API extractor: seeded per-year samples, abstract reconstruction, primary-topic taxonomy, citation edges. Tests are offline. |
| `api_test/data*/` | Small development samples (50 and 800 papers). Not analysis cohorts. Their CSVs predate the taxonomy columns, so the notebooks backfill them from `works_raw.json`. |
| `athena_pilot/` | Tiny AWS Athena/S3 technical pilot (5 records). See its README. |
| `environment.yml`, `conda-osx-arm64.lock.txt`, `ENVIRONMENT.md` | Conda env `macs40123` (Python 3.12, conda-forge). |

The planned, not yet created, layout for the research pipeline is `src/` (reusable
modules), `data/` (git-ignored; see `docs/data_strategy.md`), `outputs/figures`,
`outputs/tables`, and `paper/`. Keep reusable logic in `src/` modules that notebooks
import, not duplicated across notebooks.

## Commands

```bash
conda activate macs40123
jupyter lab                                     # select kernel "Python (macs40123)"
python3 -m unittest api_test.test_extract -v    # offline extractor tests
export OPENALEX_API_KEY=...                     # never write the key to files
python3 api_test/extract.py --min-year 2013 --max-year 2016 --size 200 --output api_test/data_2013_2016
```

When adding a package, add it to `environment.yml`, then update `ENVIRONMENT.md` and
regenerate the lock (`conda list -n macs40123 --explicit > conda-osx-arm64.lock.txt`).
The embedding stack has fragile pins (see `ENVIRONMENT.md`): install PyTorch only from
conda-forge (the pip wheel crashes with `OMP: Error #15`), and keep `transformers`
4.57.x for `adapters` 1.3.0. Run `pip check` after any change.

## Hard constraints

- **Disk:** about 13 GB free on 2026-10-10. Never download the OpenAlex snapshot or any
  multi-GB file locally without checking `df -h ~` and asking first. Keep the project's
  data under about 8 GB. Store embeddings in chunks; float16 storage is acceptable.
- **Data goes in `data/` (git-ignored).** Do not commit large data, embeddings, or API
  keys. The small `api_test/data*` samples are already tracked; leave them.
- **AWS:** profile `macs40123`, selected Region **`us-east-2`**. Follow `AGENTS.md` (the
  new-AWS-experience rules): no resources or cross-Region actions outside `us-east-2`;
  the public OpenAlex bucket is in `us-east-1`. The user has **limited AWS experience**
  and chose **help level HIGH** for AWS work (saved in `AGENTS.md`): explain what each
  step does and why before running it, suggest better alternatives, flag best practices
  and trade-offs, estimate cost first, confirm before creating resources, and offer
  cleanup afterward. AWS is a fallback; the default route is the OpenAlex API.
- **OpenAlex API:** a free key gives $1/day (resets at midnight UTC). A list/filter or
  `group_by` request costs $0.0001 and returns up to 100 works, so about 1M works per
  day. Keyword `search` costs 10× as much; define fields with filters. `sample=` is
  capped at 10,000; use cursor paging beyond that. Keep checkpoint/resume behaviour, and save
  `metadata.json` provenance with every pull.

## Research-integrity conventions

These matter for publication readiness. Flag any code that violates them.

- **Pre-specify, then look.** Choose the case field, inclusion rules, sample sizes, and
  cluster count by the documented criteria, **not** by which gives the strongest
  growth–KE pattern. Log each choice in `docs/decision_log.md` before running the
  outcome analysis.
- **Fixed-size samples.** KE must be computed on equal-size samples across years (2,000
  by default). Never compute max-distance on whole annual corpora; larger years would
  win mechanically. The same logic applies to kNN reference sets.
- **Resamples are not observations.** The 1,000 KE draws describe sampling variation
  within a year. Do not use them as the n of the time-series model or as confidence
  intervals for the growth relationship. Overlapping 3-year windows are serially
  correlated by construction.
- **Two count series.** Report all eligible papers and papers with usable abstracts
  separately, so coverage changes are not mistaken for growth.
- **One fixed embedding setup.** Use one model, revision hash, input format, and
  truncation rule for all years. Embed each paper once; everything downstream reuses the
  saved vectors.
- **Metric consistency.** Euclidean distance with mean centroids and k-means; Manhattan
  with medians and k-medians; cosine with L2-normalized vectors. Raw vs normalized
  vectors for KE is an open decision.
- **Reproducibility.** Set and record random seeds for sampling, KE resampling, k-means,
  and OpenAlex `seed`. Keep raw pulls and their metadata, because OpenAlex changes over
  time. Figures should be regenerable from code.
- **Interpretation limits.** KE is semantic dispersion in a fixed-size sample, not
  "amount of knowledge", novelty, or quality. Clusters are retrospective categories.
  Associations are not causal.

## Working with the user

- The user is a graduate student. Explain methodological trade-offs briefly and
  concretely, and recommend an option rather than listing every one.
- Writing style for the paper and docs: plain, precise academic prose; define each
  measure before using it; no overclaiming.
- Do not commit or push unless asked. Untracked files at setup time: `AGENTS.md`,
  `.cursor/`, `Week1_notebook.ipynb`, `conda-osx-arm64.lock.txt`.

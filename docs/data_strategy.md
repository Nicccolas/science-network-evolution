# Data, storage, and compute strategy

**Constraint:** On 2026-10-10 the laptop had about **13 GB free** (`df -h ~`). Plan
for a working dataset of **≤ 8 GB** to leave headroom for the Hugging Face model
cache (`~/.cache/huggingface`, about 0.5–1 GB for SPECTER2), notebooks, and
temporary files. Re-check `df -h ~` before any large download.

## Key insight: the design needs counts for everything, but text only for samples

| Quantity | What it needs | Storage |
| --- | --- | --- |
| Annual publication counts (and with-abstract counts) | OpenAlex API `group_by=publication_year` with the field, type, and language filters | Kilobytes; no paper downloads |
| Annual KE (2,000 × 1,000 resamples) | A **uniform random sample** of each year's eligible papers, large enough that 2,000-paper draws rarely repeat the same sets | Embeddings for the sample only |
| Clustering | A year-balanced pooled sample (a subset of the above) | Same vectors |
| kNN composition (idea 1) | Focal-field sample **plus** a multi-field reference sample | Additional vectors |

Drawing 2,000 papers from a uniform random subsample of a year is statistically
equivalent to drawing 2,000 from the full year. You therefore **do not need every
paper in the field**, only a sufficiently large random sample per year plus exact
population counts from the API.

## Size arithmetic

| Item | Per paper | Per 1M papers |
| --- | --- | --- |
| SPECTER2 vector, 768-d float32 | 3.1 KB | ~3.1 GB |
| Same, stored as float16 (cast back to float32 for analysis) | 1.5 KB | ~1.5 GB |
| Title + abstract + metadata as zstd Parquet | ~0.5–1 KB | ~0.5–1 GB |

Rough working-set totals:

| Tier | Corpus | Papers | Disk (float32) | Fits locally? |
| --- | --- | --- | --- | --- |
| Pilot | 3–5 candidate fields × 3–4 years × 2,000 | ≤ 40k | < 0.2 GB | Yes |
| **Recommended core** | 1 field × ~15–20 years × up to **30k random papers/year** (all papers if fewer) | ≤ 600k | ~2.5–3 GB | Yes |
| + kNN reference | Stratified multi-field sample, ~20k/year | +300–400k | +1.5–2 GB | Yes |
| Full field census | Every paper in a large field (several million) | 2–5M | 10–20 GB | **No** (needs float16 + S3, or AWS) |

**Recommendation:** Start with the recommended core plus a kNN reference sample, about
5 GB in total. This fits on the laptop and needs no AWS. Use 30k per year as a planning
number and confirm it in the pilot: compare KE medians from 10k vs 30k pools. If they
agree, the pool is large enough.

## Collection route

**Primary: OpenAlex API (no AWS).** This extends the existing `api_test/extract.py`.

- **Pricing (checked 2026-10-10, [help.openalex.org/access/pricing](https://help.openalex.org/access/pricing/)):**
  a free API key gives a **$1/day** usage budget (no key: $0.10/day), which resets at
  midnight UTC. A list/filter request costs $0.0001 and returns up to 100 works, so the
  free budget covers about **1,000,000 works per day**. `group_by` costs the same per
  call. Keyword `search` costs 10× as much, so define the field with filters, not
  `--search`. Single-work lookups are free. The observed costs in
  `api_test/data_2013_2016/metadata.json` match: $0.0003 per year = 1 count request + 2
  pages.
- The recommended core (≤ 600k works ≈ 6,000 requests ≈ $0.60) plus a kNN reference
  sample (~400k) fits in **one to two days of the free budget**. Paid top-ups exist but
  should not be needed. The extractor already checkpoints and resumes when the daily
  budget runs out.
- `extract.py` currently uses `sample=` with a seed, which OpenAlex caps at **10,000 per
  request**. For 30k per year, either (a) add **cursor paging** to pull every eligible
  paper in field-years that are not too large, or (b) combine several seeds and
  de-duplicate (record the resulting non-uniformity). Option (a) is cleaner when a
  field-year has fewer than about 100k papers.
- For the embedding corpus, request only the needed fields (drop `referenced_works` and
  `authorships` unless used) to cut transfer and storage. Keep `referenced_works` for the
  citation-pair validation subset (backlog idea 6).
- Population counts come from `group_by` queries, which cost very little.

**Secondary: AWS (only if the API route proves inadequate).** The AWS profile is
`macs40123` and the selected Region is `us-east-2`.

- The public OpenAlex snapshot is in `us-east-1`. The project may **not** perform
  cross-Region actions or create resources outside `us-east-2`. The 2026-10-02 pilot
  (`athena_pilot/README.md`) worked within this by streaming objects through the laptop
  into the project bucket `openalex-macs40123-bucket` without saving them locally.
  Copying the full works snapshot this way is slow and large, so it is not a good first
  route for someone new to AWS.
- Reasonable AWS uses, if needed later: (1) S3 as cold storage for embeddings or raw
  pulls that do not fit on the laptop; (2) a cloud GPU for embedding if laptop throughput
  is too low. Before relying on a service, check availability with
  `aws freetier get-account-plan-state --profile macs40123` and check spend limits in
  AWS Settings > Billing.
- Existing AWS resources from the pilot: the bucket `openalex-macs40123-bucket` (tiny
  Parquet test files under `raw/`) and the Athena database `macs40123_pilot`. Decide
  whether to keep or clean these up.

## Embedding compute

- SPECTER2 is BERT-base sized with a 512-token limit. Typical abstracts fit; record how
  many get truncated.
- Packages: SPECTER2 is `allenai/specter2_base` plus the `allenai/specter2` proximity
  adapter, loaded with the `adapters` library. Installed and tested on 2026-10-10
  (768-d output on `mps`). See `ENVIRONMENT.md` for version pins and loading code. Record
  the model revision hashes when generating research embeddings.
- Use Apple Silicon `mps` in PyTorch. Measure throughput in the pilot before committing to
  a corpus size. As a planning assumption, expect tens of papers per second, so 600k
  papers takes several hours to overnight. Write embeddings in chunks (for example one
  `.npy` per year or per 50k rows) with a matching ID index, so runs can resume.
- Embed each paper once. All resampling, clustering, and kNN work operates on saved
  vectors.

## Planned local layout (git-ignored)

```text
data/
  raw/<pull-name>/          API pages / works.parquet + metadata.json (provenance)
  interim/                  cleaned corpus: one row per eligible paper
  embeddings/<model>@<rev>/ vectors_<year>.npy + ids_<year>.parquet
  derived/                  KE draws, cluster assignments, kNN compositions
outputs/figures/, outputs/tables/
```

Keep every `metadata.json` (filters, seeds, retrieval timestamps, OpenAlex snapshot
date, counts). OpenAlex changes over time, so these files are the reproducibility record.

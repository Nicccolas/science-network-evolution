# OpenAlex API extractor

`extract.py` creates balanced, seeded samples of OpenAlex works for local analysis. It collects English-language records classified by OpenAlex as `article`, reconstructs available abstracts, preserves the primary topic hierarchy, and creates an outgoing citation network.

Run commands from the `science-network-evolution` directory.

## Quick start

The default command samples up to 50 papers from 2024 with seed 42 and writes them to `api_test/data`:

```bash
python3 api_test/extract.py
```

For a larger run, set an OpenAlex API key first. The key is sent in an authorization header and is never written to the output files.

```bash
export OPENALEX_API_KEY="your-key"
```

This example requests up to 200 papers from each year from 2013 through 2016, with no field restriction:

```bash
python3 api_test/extract.py \
  --min-year 2013 --max-year 2016 \
  --size 200 \
  --output api_test/data_2013_2016
```

This example requests up to 500 papers per year whose primary field is one of fields 30, 31, or 32:

```bash
python3 api_test/extract.py \
  --min-year 2010 --max-year 2025 \
  --size 500 --fields "30,31,32" \
  --output api_test/data_2010_2025
```

## Options

| Option | Meaning | Default |
| --- | --- | --- |
| `--size N` | Maximum sampled works per year; accepts 1–10,000 | `50` |
| `--year YYYY` | Select one publication year | — |
| `--min-year YYYY` | First publication year, inclusive | `2024` |
| `--max-year YYYY` | Last publication year, inclusive | `2024` |
| `--fields "30,31,32"` | Match any listed primary-field ID | all fields |
| `--search "terms"` | Add an OpenAlex keyword search | no search |
| `--seed N` | Seed used by OpenAlex sampling | `42` |
| `--output PATH` | Directory for completed files and temporary checkpoints | `api_test/data` |
| `--refresh-existing` | Refresh the exact saved IDs in an output containing at most 100 works | off |

`--year` cannot be combined with `--min-year` or `--max-year`. Omitting `--fields` removes the disciplinary restriction. `--search` is keyword search, not an exact topic classifier, and costs more API credits than a normal list request.

The extractor always applies these filters:

```text
type:article
language:en
```

OpenAlex's language value describes the title and abstract metadata. It does not verify the language of the full paper, and the `article` type does not guarantee peer review.

## Sampling and paging

`--size` is interpreted per year. For example, `--size 200 --min-year 2013 --max-year 2016` requests up to 800 works: as many as 200 from each of four years. A year with fewer matching works contributes fewer rows.

OpenAlex returns at most 100 supported results per page. The extractor requests additional pages automatically and combines them into one set of output files. OpenAlex limits a seeded sample to 10,000 works, which is also the maximum accepted `--size`.

The seed makes the sampling procedure repeatable against a fixed OpenAlex corpus. OpenAlex changes over time, so running the same command later may not produce an identical sample.

## Checkpoints and daily API budget

Every completed API page is saved under:

```text
<output>/.extract-checkpoint/
```

If the OpenAlex daily budget is exhausted, the extractor prints the reset time and exits while retaining the checkpoint. Rerun the exact same command to continue. Changing the years, size, fields, seed, search, or selected response fields requires another output directory or removal of the unfinished checkpoint after it has been inspected.

Existing completed output files are replaced only after all pages have been collected and validated. On success, the checkpoint directory is removed. During assembly, allow enough disk space for both the checkpoints and the completed output.

## Output files

Each successful run publishes five files:

| File | Contents |
| --- | --- |
| `works.csv` | One row per sampled paper, ready for pandas or other tabular tools |
| `works_raw.json` | The selected original OpenAlex work objects from every API page |
| `citation_edges.csv` | Unique directed edges where `Source` cites `Target` |
| `citation_nodes.csv` | Sampled work nodes plus external referenced-work IDs |
| `metadata.json` | Parameters, per-year counts and times, API cost, missingness, and network counts |

### `works.csv` columns

| Column | Meaning |
| --- | --- |
| `openalex_id` | Canonical OpenAlex work ID; primary key |
| `doi` | DOI URL, when available |
| `title` | Work title |
| `abstract` | Plaintext reconstructed from `abstract_inverted_index` |
| `publication_year` | Publication year |
| `publication_date` | OpenAlex publication date |
| `type` | OpenAlex work type; always `article` for a normal extraction |
| `language` | OpenAlex metadata-language code; always `en` for a normal extraction |
| `cited_by_count` | Incoming citation count at retrieval time |
| `referenced_works_count` | Number of returned outgoing references |
| `referenced_works` | Semicolon-separated IDs cited by this work |
| `is_oa` | Open-access flag |
| `oa_status` | Open-access category |
| `oa_url` | Open-access URL, when available |
| `source_name` | Source at the primary location |
| `primary_topic` | Primary topic name |
| `primary_topic_id` | Primary topic ID |
| `primary_topic_score` | OpenAlex score for the primary topic |
| `primary_subfield` | Subfield containing the primary topic |
| `primary_subfield_id` | Primary subfield ID |
| `primary_field` | Field containing the primary topic |
| `primary_field_id` | Primary field ID |
| `primary_domain` | Domain containing the primary topic |
| `primary_domain_id` | Primary domain ID |
| `authors` | Author names in API order, separated by semicolons |
| `author_ids` | Corresponding OpenAlex author IDs, separated by semicolons |

Blank cells mean missing or unavailable values, not zero. The taxonomy columns describe only the hierarchy of the paper's single `primary_topic`; secondary topic assignments are not extracted.

## Citation-network interpretation

An edge `A → B` means work A cites work B. References outside the sample are retained as external nodes, but only their OpenAlex IDs are known. Their titles, taxonomy, incoming links, and outgoing references are not fetched.

The output is therefore a one-step outgoing citation network. A referenced external node with zero observed outgoing degree should not be interpreted as a work with no bibliography. Incoming citations from works outside the sample are also absent.

For field-to-field network analysis, join the sampled nodes to `works.csv` using the OpenAlex work ID. Both endpoints have taxonomy data only when both works are part of the sample.

## Refresh an existing small sample

To update the metadata for the exact IDs already saved in an output directory:

```bash
python3 api_test/extract.py \
  --refresh-existing \
  --output api_test/data
```

Refresh is limited to 100 saved IDs because OpenAlex supports at most 100 OR values in one filtered request. It cannot be combined with year, field, or search filters. Larger outputs should be regenerated or handled with a separate batched-enrichment workflow.

## Tests

The tests simulate OpenAlex responses, so they do not spend API credits or replace saved datasets:

```bash
python3 -m unittest api_test.test_extract -v
```

They cover multi-year and multi-page sampling, field parsing, citation endpoints, duplicate IDs, missing taxonomy, zero-result years, safe publication, and resumption after a simulated daily-budget limit.

References: [OpenAlex API](https://help.openalex.org/api/), [paging and limits](https://help.openalex.org/api/paging/), and [authentication and daily budgets](https://help.openalex.org/api/authentication/).

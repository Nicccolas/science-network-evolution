# OpenAlex dataset MVP

The saved MVP contains 50 randomly sampled works classified by OpenAlex as English-language `article`s, published in 2024, with seed 42. The extractor can also make balanced samples across a range of publication years, optionally restricted to listed primary fields. OpenAlex's language label describes the title/abstract metadata, not necessarily the full text. This demonstrates extraction, not a representative research design. The `article` classification alone does not guarantee peer review or a journal source.

## Files

- `data/works.csv`: one row per work, including reconstructed abstracts, reference IDs, and the primary OpenAlex topic hierarchy.
- `data/citation_edges.csv`: directed edge list (`Source`, `Target`, `Type`); source cites target.
- `data/citation_nodes.csv`: all sampled and referenced work IDs, including isolated sampled works. Titles and years are populated only for sampled works; external nodes use their ID as a label.
- `data/works_raw.json`: original work objects in a `results` array. New runs combine objects from all pages; the saved MVP retains its original single-response envelope.
- `data/metadata.json`: query and validation details. New runs add yearly filters/counts/retrieval times and API cost; the saved MVP has its original request URL and counts.
- `extract.py`: reusable extractor using only the Python standard library.

## Run

From the `science-network-evolution` directory:

```bash
python3 api_test/extract.py
```

Set an OpenAlex API key and request up to 500 works **per year** from 2010 through 2025, whose primary field is any of 30, 31, or 32:

```bash
export OPENALEX_API_KEY="your-key"
python3 api_test/extract.py \
  --min-year 2010 --max-year 2025 \
  --size 500 --fields "30,31,32" \
  --output api_test/data_2010_2025
```

For one year, use equal bounds or the `--year` shortcut. For example:

```bash
python3 api_test/extract.py --size 100 --year 2025 --search "social inequality" --output api_test/data_inequality
```

`--min-year` and `--max-year` are inclusive and both default to 2024. `--year` selects one year and cannot be combined with either bound. `--size` defaults to 50 and accepts 1–10,000 **per year**. OpenAlex returns at most 100 works per page, so larger samples use multiple requests; a year with fewer matching works yields fewer rows. `--fields` accepts up to 100 comma-separated field numbers and matches **any listed primary field**; omit it to include all fields. `--search` is optional keyword search and uses more daily API credits. `--seed` defaults to 42. These are samples, not complete yearly cohorts.

An API key is optional; set `OPENALEX_API_KEY` for the larger keyed daily budget. It is sent in an authorization header and is not saved in metadata. The extractor checks OpenAlex's remaining daily credits. If the budget runs out, it keeps completed request pages in `<output>/.extract-checkpoint/` and prints the reset time (midnight UTC). Rerun **the same command** to resume; different options need a different output directory. Each checkpoint stores its request URL, retrieval time, rate-limit status, and response. Once every page passes validation, the five completed files replace any existing files in the output directory and the checkpoint is removed. Allow temporary disk space for checkpoints plus the new outputs during assembly.

To enrich or refresh the exact existing sample instead of resampling:

```bash
python3 api_test/extract.py --refresh-existing
```

This refresh uses stored work IDs and is limited to **100 IDs**; it does not resample a multi-year extraction. Year, size, seed, and search options do not control it. The saved metadata records the actual ID lookup request.

Keyword search is not a topic-ID filter. For an exact topic, author, institution, or journal, first resolve its name to an OpenAlex ID, then filter by that ID. See the [OpenAlex API reference](https://help.openalex.org/api/).

## Columns

| Column | Meaning |
| --- | --- |
| openalex_id | Unique work URL; primary key |
| doi | DOI URL, if available |
| title | Work title |
| abstract | Plaintext reconstructed from OpenAlex's `abstract_inverted_index`, if available |
| publication_year | Publication year |
| publication_date | Publication date reported by OpenAlex |
| type | OpenAlex work type; filtered to article |
| language | Language code, if available |
| cited_by_count | Citation count at retrieval time |
| referenced_works_count | Number of reference IDs returned by OpenAlex, computed from the list |
| referenced_works | Semicolon-separated OpenAlex IDs of works this paper cites |
| is_oa | OpenAlex open-access flag |
| oa_status | Open-access category |
| oa_url | Open-access link, if available |
| source_name | Source at the primary location |
| primary_topic | Assigned primary topic name |
| primary_topic_id | Assigned primary topic URL |
| primary_topic_score | OpenAlex confidence score for the primary topic |
| primary_subfield | Subfield containing the primary topic |
| primary_subfield_id | OpenAlex ID for the primary subfield |
| primary_field | Field containing the primary topic |
| primary_field_id | OpenAlex ID for the primary field |
| primary_domain | Domain containing the primary topic |
| primary_domain_id | OpenAlex ID for the primary domain |
| authors | Author names in API order, joined by semicolons |
| author_ids | Author IDs in corresponding order, joined by semicolons |

Blank CSV cells mean missing/not supplied, not zero. JSON preserves nulls, booleans, and nested authorship/institution records. Use raw JSON for reliable author relationships; names themselves can contain delimiters. Extremely large author lists may be truncated by OpenAlex.

## Validation and limits

### Citation network

Import `citation_nodes.csv` as a nodes table and `citation_edges.csv` as a directed edges table in a network tool such as Gephi. An edge A → B means A cites B. The extractor retains references outside the sampled works, deduplicates source-target pairs, and includes every edge endpoint in the nodes table.

This is a one-step outgoing citation network. `references_fetched=False` means an external node's own references have not been fetched; its zero observed outgoing degree is not evidence that it cites nothing. An empty reference list for a sampled work means no linked references were returned, not necessarily an empty bibliography. Reference coverage depends on OpenAlex's matching and source coverage. Incoming citations from other papers and links among external nodes are not collected.

The saved 50 papers were randomly sampled across disciplines, so they may have few or no links to each other. The multi-year mode takes up to the requested number from each year and therefore does not preserve the real distribution of papers by year. For a connected research network, use a focused topic or expand references from selected seed papers. Definitions: [OpenAlex work attributes](https://help.openalex.org/data/works/attributes/).

The extractor checks unique IDs and requested year/type/language/primary-field filters before publishing data. Missing values are counted in metadata. It retries transient rate limits, server failures, and network failures up to five attempts. A daily-budget limit pauses the run with its checkpoints intact.

The seed makes the sampling procedure repeatable, but OpenAlex changes over time, including during a multi-day extraction. Per-year retrieval times show when pages were fetched; retain the saved raw response for the exact historical sample. Citation counts also change. The sample comes from the default core corpus and should not be treated as representative of all scholarship. This extracts bibliographic metadata, not paper PDFs or full text. See [OpenAlex paging and limits](https://help.openalex.org/api/paging/) and [authentication and daily budgets](https://help.openalex.org/api/authentication/).

Source: [OpenAlex](https://openalex.org/), accessed through [its API](https://help.openalex.org/api/). Query parameters and retrieval timestamps are in the output's `metadata.json`.

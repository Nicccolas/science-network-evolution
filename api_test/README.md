# OpenAlex dataset MVP

A small, real API extraction: 50 randomly sampled works classified by OpenAlex as English-language `article`s, published in 2024, with seed 42. There is no subject restriction. OpenAlex's language label describes the title/abstract metadata, not necessarily the full text. This demonstrates extraction, not a representative research design. The `article` classification alone does not guarantee peer review or a journal source.

## Files

- `data/works.csv`: one row per work, 19 columns for analysis, including reconstructed abstracts, reference IDs, and their count.
- `data/citation_edges.csv`: directed edge list (`Source`, `Target`, `Type`); source cites target.
- `data/citation_nodes.csv`: all sampled and referenced work IDs, including isolated sampled works. Titles and years are populated only for sampled works; external nodes use their ID as a label.
- `data/works_raw.json`: original API response, retaining the selected nested fields.
- `data/metadata.json`: exact request URL, parameters, UTC retrieval time, counts, and missingness.
- `extract.py`: reusable extractor using only the Python standard library.

## Run

From the `science-network-evolution` directory:

```bash
python3 api_test/extract.py
```

Change the sample or add a keyword search:

```bash
python3 api_test/extract.py --size 100 --year 2025 --search "social inequality" --output api_test/data_inequality
```

The MVP accepts 1–100 records in one request. An API key is optional; if needed, set `OPENALEX_API_KEY` in your environment. It is sent in an authorization header and is not saved in metadata. Rerunning with the same output directory replaces its five data files.

To enrich or refresh the exact existing sample instead of resampling:

```bash
python3 api_test/extract.py --refresh-existing
```

This refresh uses stored work IDs, so year, size, seed, and search options do not control it. The saved metadata records the actual ID lookup request.

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
| authors | Author names in API order, joined by semicolons |
| author_ids | Author IDs in corresponding order, joined by semicolons |

Blank CSV cells mean missing/not supplied, not zero. JSON preserves nulls, booleans, and nested authorship/institution records. Use raw JSON for reliable author relationships; names themselves can contain delimiters. Extremely large author lists may be truncated by OpenAlex.

## Validation and limits

### Citation network

Import `citation_nodes.csv` as a nodes table and `citation_edges.csv` as a directed edges table in a network tool such as Gephi. An edge A → B means A cites B. The extractor retains references outside the 50 sampled works, deduplicates source-target pairs, and includes every edge endpoint in the nodes table.

This is a one-step outgoing citation network. `references_fetched=False` means an external node's own references have not been fetched; its zero observed outgoing degree is not evidence that it cites nothing. An empty reference list for a sampled work means no linked references were returned, not necessarily an empty bibliography. Reference coverage depends on OpenAlex's matching and source coverage. Incoming citations from other papers and links among external nodes are not collected.

The 50 papers were randomly sampled across disciplines, so they may have few or no links to each other. For a connected research network, use a focused topic or expand references from selected seed papers. Definitions: [OpenAlex work attributes](https://help.openalex.org/data/works/attributes/).

The extractor checks unique IDs and requested year/type/language filters before writing data. Missing values are counted in metadata. It retries rate limits, transient server failures, and network failures up to five attempts.

The seed makes the sampling procedure repeatable, but OpenAlex changes over time; retain the saved raw response for the exact historical sample. Citation counts also change. The sample comes from the default core corpus and should not be treated as representative of all scholarship. This extracts bibliographic metadata, not paper PDFs or full text.

Source: [OpenAlex](https://openalex.org/), accessed through [its API](https://help.openalex.org/api/). The actual retrieval timestamp and request are in `data/metadata.json`.

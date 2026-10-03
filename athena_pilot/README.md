# OpenAlex streaming/Athena pilot

**Bounded technical pilot—not the complete 2015 Psychology dataset.**

On 2026-10-02, five complete update-date partitions from the public OpenAlex
Parquet snapshot dated 2026-09-23 were streamed through the local computer to
the private S3 bucket `openalex-macs40123-bucket` in `us-east-2`. The transfer
used `curl` on the public HTTPS source and `aws s3 cp -` on standard input; no
Parquet file was saved to local disk. The public snapshot manifest is
`s3://openalex/data/parquet/manifest.json` (`format: parquet`).

| `updated_date` | Manifest bytes | Manifest records | Destination ETag |
|---|---:|---:|---|
| 2025-10-30 | 57,376 | 1 | `38e1481bee4a916cb5a1c64267d04055` |
| 2025-10-15 | 62,600 | 1 | `4e17f6f0ec42d55b70487ee3f344f5e7` |
| 2018-10-12 | 63,359 | 1 | `6379b212d7d10a1e2f9782d79dd255a1` |
| 2017-09-15 | 63,927 | 1 | `d3f57d021767036f24fc52deb84a89dc` |
| 2019-08-22 | 64,250 | 1 | `65417afb63947ba870616c5e35487baf` |

Each source object is
`s3://openalex/data/parquet/works/updated_date=<date>/part_0000.parquet`;
each destination object is
`s3://openalex-macs40123-bucket/raw/openalex/snapshot=2026-09-23/updated_date=<date>/part_0000.parquet`.
The five objects total **311,512 bytes** and five records. The destination
sizes matched the manifest, and each uploaded object reported SSE-S3 encryption.
An earlier 1,014,006-byte transfer test also remains at
`raw/openalex/snapshot=2026-09-23/updated_date=2016-06-24/part_0000.parquet`;
it was **not registered** as an Athena partition. Its destination ETag matched
the source ETag `95f1bb84e281c6f3ce73b357903682be`.

The existing S3 bucket has all four Block Public Access settings enabled and
default AES256 encryption. The `athena/results/` prefix expires after seven
days. Research source files under `raw/` have no lifecycle expiration.

## Athena test

Database: `macs40123_pilot`. Table: `openalex_works_source`. Region: `us-east-2`.
The table projects the Parquet fields needed for the planned relational data;
`referenced_works` is an array, `topics` and `authorships` are nested arrays,
and `abstract_inverted_index` is a JSON string. Only the five partitions in
[`partitions.sql`](partitions.sql) were registered. SQL is in
[`source.sql`](source.sql), [`cohort_count.sql`](cohort_count.sql), and
[`source_counts.sql`](source_counts.sql). The nested-field check is in
[`nested_fields_check.sql`](nested_fields_check.sql).

| Query | Athena query ID | Result | Scanned bytes |
|---|---|---:|---:|
| Create source table | `4bcdda40-96c4-49ff-9d95-644df73c1415` | succeeded | 0 |
| Add five partitions | `93a88607-f9a3-432a-b674-c760de617678` | succeeded | 0 |
| 2015 primary Psychology, non-XPAC count | `07bec8e4-8bec-416c-a419-7f992ff46c71` | 0 | 0 |
| Source counts | `bf5dbda0-21bb-4e1d-a89b-c18c9da2ef93` | 5 total; 0 from 2015; 0 primary Psychology | 468 |
| Nested-field check | `716a2041-0674-4f70-8983-74bf5f6a076f` | succeeded; all five records readable | 5,713 |

The cohort query applied `publication_year = 2015`,
`primary_topic.field.id = 'https://openalex.org/fields/32'`, and
`is_xpac = false`. Athena confirmed that the source table has five records,
so the empty cohort is caused by this intentionally tiny selection. No
`works`, `citations`, `work_topics`, `topics`, `authors`, `authorships`, or
`external_works` research tables were materialized because they would all be
empty. No abstract reconstruction or network validation was possible.
The nested-field check successfully read canonical work IDs, reference-array
lengths, topic-array lengths, authorship-array lengths, and abstract-index
presence. These five records have zero references and no abstract indexes, so
they cannot exercise citation-edge or abstract reconstruction logic.

The `primary` workgroup had a temporary 100 MiB per-query scan cutoff during
the test. Its original configuration was restored and verified afterward.
There were three scanning SELECT queries, well below the ten-query limit.
Athena query result objects total 1,034 bytes; they are temporary CSV/metadata
files under `athena/results/`, not research outputs. The research inputs are
Parquet only. Actual billing may include Athena's per-query minimum, S3
requests/storage, and Glue usage; the tiny pilot is far below the $5 ceiling.

To reproduce the transfer for one date, with the configured `macs40123`
profile and region:

```bash
set -o pipefail
d=2025-10-30
curl -fsSL "https://openalex.s3.amazonaws.com/data/parquet/works/updated_date=${d}/part_0000.parquet" \
  | aws s3 cp - "s3://openalex-macs40123-bucket/raw/openalex/snapshot=2026-09-23/updated_date=${d}/part_0000.parquet" \
      --profile macs40123 --region us-east-2 --sse AES256 --no-progress
```

The next useful pilot requires a different bounded source selection that
contains 2015 primary-field Psychology works, while still respecting the
scan and spending limits. Update dates describe when OpenAlex refreshed
records, not their publication years.

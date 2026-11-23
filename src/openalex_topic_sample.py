#!/usr/bin/env python3
"""Download the topic-stratified sample for the topic panel, with the topic counts it is drawn from.

For every year, two OpenAlex group_by requests count one primary field's journal articles by
primary topic: all of them (N_kt, the topic's size in the panel) and those with abstracts (which
decide panel membership and the sample size). Every topic-year with at least --min-papers
articles with abstracts is in the panel, and for each one a seeded OpenAlex random sample of
min(articles with abstracts, --cap) papers is downloaded (research design, "Topic panel";
decision log 2026-11-02 and 2026-11-19). All counts are pulled before any sample, so the panel
rests on one database state, and the run stops before sampling if the disk is too full.

Outputs in --output:
  topic_counts.csv    one row per topic-year in either count, with in_panel and sample_size
  sample.parquet      one row per sampled paper: the pool's columns, publisher fields, in_pool
  to_embed.parquet    the sampled papers not in the pool (all of them without --pool), with the
                      columns src/embed_pool.py reads, for: python3 -m src.embed_pool --pool
                      <output>/to_embed.parquet --output data/embeddings/<model>_topic_sample
  sample_report.csv   per-year totals, requests, and reported cost
  metadata.json       filters, thresholds, seeds, request totals, and notes
  raw/                every response; sample pages are gzip-compressed JSON

Run with --counts-only first: it makes only the 2 count requests per year and prints the plan
(topic-years, papers, requests, estimated cost). The full 2001-2024 pull is about 6,600 requests,
about $0.66 of the $1 daily budget. If the budget runs out, checkpoints are kept and the same
command resumes.

    python3 -m src.openalex_topic_sample --field 32 --min-year 2001 --max-year 2024 \\
        --pool data/pool/psychology_field32/pool.parquet \\
        --output data/topic_sample/psychology_field32 --counts-only
"""
import argparse
import csv
import gzip
import json
import math
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_test import extract  # noqa: E402  (fetch, request_url, checkpointing, atomic writes)
from src import openalex_counts, openalex_pool  # noqa: E402  (the journal frame, page checks, flatten)


MIN_PAPERS = 100        # articles with abstracts for a topic-year to enter the panel
CAP = 300               # papers sampled per topic-year (all of them if fewer)
SAMPLE_SEED = 1         # OpenAlex seed, as for the pool
MAX_GROUPS = 200        # one group_by page; a year with more topics than this would need paging
COST_PER_REQUEST = 0.0001
BYTES_PER_PAPER = 1_500  # gzip raw page (~0.96 KB) + parquet row (~0.56 KB), measured on the pool
DISK_MARGIN = 2 * 1024 ** 3  # free space to leave for everything else on the laptop

PUBLISHER_COLUMNS = ("source_host_organization", "source_host_organization_name",
                     "source_host_organization_lineage_names")
COLUMNS = (*openalex_pool.COLUMNS, *PUBLISHER_COLUMNS, "in_pool")
EMBED_COLUMNS = ("openalex_id", "publication_year", "title", "abstract")  # embed_pool.POOL_COLUMNS
COUNT_COLUMNS = ("topic_id", "topic_name", "year", "journal_articles", "journal_articles_abstract",
                 "abstract_share", "in_panel", "sample_size")

SAMPLING_NOTE = ("Each panel topic-year is one OpenAlex random sample (sample=sample_size, "
                 "seed=sample_seed) of its journal articles with abstracts, paged with basic paging "
                 "(per_page=100). sample_size = min(journal_articles_abstract, cap), so topic-years "
                 "with fewer than cap such articles contribute all of them.")
COUNT_NOTE = ("journal_articles is N_kt, the topic's size in the panel (decision log, 2026-11-19); "
              "journal_articles_abstract sets panel membership (>= min_papers) and sample_size. "
              "Both are group_by primary_topic.id counts pulled before any sample.")
POOL_NOTE = ("in_pool marks papers that are also in the 240,000-paper KE pool, so their pool "
             "vectors can be reused instead of embedding them again.")


def frame_filter(field, year):
    """All journal articles in the field's N_t frame for one year; N_kt counts these by topic."""
    return (f"primary_topic.field.id:{field},type:article,language:en,is_paratext:false,"
            f"{openalex_counts.JOURNAL},publication_year:{year}")


def abstract_filter(field, year):
    """The same frame restricted to works with abstracts (the KE pool's filter)."""
    return openalex_pool.year_filter(field, year)


def topic_filter(field, year, topic):
    return f"{abstract_filter(field, year)},primary_topic.id:{topic}"


def short_id(openalex_id):
    return str(openalex_id).rsplit("/", 1)[-1]


def topic_number(topic):
    return int(topic[1:])


def topic_counts(payload):
    """{topic_id: (name, count)} from one group_by response, checked against meta.count."""
    groups = payload.get("group_by")
    meta = payload.get("meta") or {}
    total = meta.get("count")
    if not isinstance(groups, list) or not isinstance(total, int):
        raise RuntimeError("OpenAlex response lacks group_by counts")
    if isinstance(meta.get("groups_count"), int) and meta["groups_count"] > len(groups):
        raise RuntimeError(f"OpenAlex returned {len(groups)} of {meta['groups_count']} topic groups; "
                           "this year needs group_by paging")
    counts = {}
    for group in groups:
        topic, count = short_id(group.get("key")), group.get("count")
        if not re.fullmatch(r"T\d+", topic) or not isinstance(count, int) or count < 0 or topic in counts:
            raise RuntimeError(f"Unexpected topic group in OpenAlex response: {group!r}")
        counts[topic] = (group.get("key_display_name"), count)
    if sum(count for _, count in counts.values()) != total:
        raise RuntimeError(f"Topic groups sum to {sum(c for _, c in counts.values())}, "
                           f"but meta.count is {total}")
    return counts


def count_request(path, filter_value):
    url = extract.request_url({"filter": filter_value, "group_by": "primary_topic.id",
                               "per_page": MAX_GROUPS, "corpus": "core"})
    return extract.checkpoint_request(path, url, topic_counts)


def collect_counts(raw, field, year):
    """Return the year's two count records: all journal articles, and those with abstracts."""
    directory = raw / f"year={year}"
    return (count_request(directory / "counts_all.json", frame_filter(field, year)),
            count_request(directory / "counts_abstract.json", abstract_filter(field, year)))


def count_rows(year, every, with_abstract, min_papers, cap):
    """One row per topic in either count, in topic-number order."""
    all_counts = topic_counts(every["payload"])
    abstract_counts = topic_counts(with_abstract["payload"])
    rows = []
    for topic in sorted(set(all_counts) | set(abstract_counts), key=topic_number):
        name_all, n_all = all_counts.get(topic, (None, 0))
        name_abstract, n_abstract = abstract_counts.get(topic, (None, 0))
        if n_abstract > n_all:
            raise RuntimeError(f"{year} {topic}: {n_abstract} articles with abstracts but only "
                               f"{n_all} articles; re-pull this year's counts")
        in_panel = n_abstract >= min_papers
        rows.append({"topic_id": topic, "topic_name": name_all or name_abstract, "year": year,
                     "journal_articles": n_all, "journal_articles_abstract": n_abstract,
                     "abstract_share": round(n_abstract / n_all, 4) if n_all else None,
                     "in_panel": in_panel, "sample_size": min(n_abstract, cap) if in_panel else 0})
    return rows


def pages_needed(size):
    return math.ceil(size / 100)


def read_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def atomic_gzip_json(path, value):
    """Like extract.atomic_json, but gzip-compressed (sample pages are about 370 KB raw)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".checkpoint-", delete=False) as file:
        temporary = Path(file.name)
        try:
            with gzip.GzipFile(fileobj=file, mode="wb", mtime=0) as compressed:
                compressed.write(json.dumps(value, ensure_ascii=False).encode("utf-8") + b"\n")
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def checkpoint_request_gz(path, url, validator):
    """extract.checkpoint_request with gzip-compressed checkpoint files."""
    if path.exists():
        record = read_gzip_json(path)
        if record.get("request_url") != url:
            raise RuntimeError(f"Checkpoint query differs from this run: {path}")
    else:
        payload, rate = extract.fetch(url)
        validator(payload)
        record = {"request_url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                  "rate": rate, "payload": payload}
        atomic_gzip_json(path, record)
        if rate.get("remaining") is not None and rate["remaining"] < 10:
            print(f"OpenAlex daily credits remaining: {rate['remaining']}", file=sys.stderr)
    validator(record["payload"])
    return record


def validate_sample_page(payload, field, year, topic, size, expected):
    count = (payload.get("meta") or {}).get("count")
    if count != size:
        raise RuntimeError(f"{year} {topic}: OpenAlex reports {count} works for a sample of {size}. "
                           f"The topic-year changed since the counts were pulled; delete "
                           f"raw/year={year}/ and rerun to redraw this year.")
    openalex_pool.validate_page(payload, field, year, expected)
    topic_url = f"https://openalex.org/{topic}"
    for work in payload["results"]:
        if (work.get("primary_topic") or {}).get("id") != topic_url:
            raise RuntimeError(f"{year} {topic}: OpenAlex returned a work from another topic: {work['id']}")


def sample_topic_year(raw, field, year, topic, size):
    """Fetch one topic-year's seeded sample, page by page. Returns (works, records)."""
    params = {"filter": topic_filter(field, year, topic), "corpus": "core", "sample": size,
              "seed": SAMPLE_SEED, "per_page": 100, "select": openalex_pool.SELECT}
    directory = raw / f"year={year}" / f"topic={topic}" / f"seed={SAMPLE_SEED}"
    works, records, seen = [], [], set()
    for page in range(1, pages_needed(size) + 1):
        expected = min(100, size - (page - 1) * 100)
        record = checkpoint_request_gz(
            directory / f"page_{page:04d}.json.gz", extract.request_url({**params, "page": page}),
            lambda payload, n=expected: validate_sample_page(payload, field, year, topic, size, n))
        for work in record["payload"]["results"]:
            if work["id"] in seen:
                raise RuntimeError(f"{year} {topic}: the sample repeats {work['id']}")
            seen.add(work["id"])
            works.append(work)
        records.append(record)
    return works, records


def flatten(work, position, pool_ids):
    row = openalex_pool.flatten(work, SAMPLE_SEED, position)
    source = (work.get("primary_location") or {}).get("source") or {}
    row["source_host_organization"] = source.get("host_organization")
    row["source_host_organization_name"] = source.get("host_organization_name")
    row["source_host_organization_lineage_names"] = (
        "; ".join(source.get("host_organization_lineage_names") or []) or None)
    row["in_pool"] = None if pool_ids is None else work["id"] in pool_ids
    return row


def year_report(year, rows, papers, records, pool_ids):
    panel = [row for row in rows if row["in_panel"]]
    with_abstract = sum(row["journal_articles_abstract"] for row in rows)
    times = [record["retrieved_at_utc"] for record in records]
    return {
        "year": year,
        "journal_articles": sum(row["journal_articles"] for row in rows),
        "journal_articles_abstract": with_abstract,
        "topics": len(rows), "panel_topics": len(panel),
        "panel_share_abstract": (round(sum(row["journal_articles_abstract"] for row in panel)
                                       / with_abstract, 4) if with_abstract else None),
        "planned_sample": sum(row["sample_size"] for row in panel),
        "planned_requests": 2 + sum(pages_needed(row["sample_size"]) for row in panel),
        "sampled": len(papers),
        "in_pool": None if pool_ids is None else sum(bool(paper["in_pool"]) for paper in papers),
        "missing_title": sum(not paper["title"] for paper in papers),
        "requests": len(records),
        "api_cost_usd": round(sum(record["payload"]["meta"].get("cost_usd") or 0
                                  for record in records), 6),
        "first_retrieved_at_utc": min(times), "last_retrieved_at_utc": max(times),
    }


def check_disk(output, papers):
    """Stop before sampling if the rest of the sample would leave less than DISK_MARGIN free."""
    already = sum(path.stat().st_size for path in (output / "raw").rglob("page_*.json.gz"))
    need = max(0, papers * BYTES_PER_PAPER - already)
    free = shutil.disk_usage(output).free
    if free < need + DISK_MARGIN:
        raise RuntimeError(f"The sample needs about {need / 1e9:.1f} GB and the disk has "
                           f"{free / 1e9:.1f} GB free; keep at least {DISK_MARGIN / 1024 ** 3:.0f} GB "
                           "spare. Free some space first.")


def write_csv(path, rows, columns):
    temporary = path.parent / f".{path.name}.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_sample(output, frames):
    sample = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    if sample["openalex_id"].duplicated().any():
        raise RuntimeError("The assembled sample repeats a work")
    temporary = output / ".sample.parquet.tmp"
    sample.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output / "sample.parquet")
    # The papers src/embed_pool.py still has to embed: pool papers already have vectors.
    to_embed = sample.loc[sample["in_pool"].ne(True), list(EMBED_COLUMNS)]
    temporary = output / ".to_embed.parquet.tmp"
    to_embed.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output / "to_embed.parquet")
    return len(sample)


def write_metadata(output, args, reports, sampled_rows):
    extract.atomic_json(output / "metadata.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "src/openalex_topic_sample.py",
        "git_commit": openalex_counts.git_commit(),
        "field": args.field, "years": [args.min_year, args.max_year],
        "counts_only": args.counts_only,
        "filters": {"journal_articles": frame_filter(args.field, "<year>"),
                    "journal_articles_abstract": abstract_filter(args.field, "<year>"),
                    "sample": topic_filter(args.field, "<year>", "<topic>")},
        "group_by": "primary_topic.id", "corpus": "core", "select": openalex_pool.SELECT,
        "min_papers": args.min_papers, "cap": args.cap, "sample_seed": SAMPLE_SEED,
        "pool": str(args.pool) if args.pool else None,
        "topic_years_in_panel": sum(r["panel_topics"] for r in reports),
        "planned_sample": sum(r["planned_sample"] for r in reports),
        "rows": sampled_rows,
        "requests": sum(r["requests"] for r in reports),
        "reported_cost_usd": round(sum(r["api_cost_usd"] for r in reports), 6),
        "per_year": reports,
        "notes": [COUNT_NOTE, SAMPLING_NOTE, POOL_NOTE, openalex_counts.CLASSIFIER_NOTE],
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--field", type=int, required=True, help="OpenAlex field ID, e.g. 32 (Psychology)")
    parser.add_argument("--min-year", type=int, required=True)
    parser.add_argument("--max-year", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool", type=Path, help="pool.parquet, to mark papers already in the KE pool")
    parser.add_argument("--min-papers", type=int, default=MIN_PAPERS,
                        help="articles with abstracts for a topic-year to enter the panel")
    parser.add_argument("--cap", type=int, default=CAP, help="papers sampled per topic-year, 1-10,000")
    parser.add_argument("--counts-only", action="store_true",
                        help="pull only the topic counts and print the sampling plan")
    args = parser.parse_args(argv)
    if args.field < 1 or args.min_year > args.max_year:
        parser.error("need a positive field ID and min-year <= max-year")
    if not 1 <= args.cap <= 10_000:
        parser.error("--cap must be between 1 and 10,000 (the OpenAlex sample cap)")
    if args.min_papers < 1:
        parser.error("--min-papers must be positive")

    years = range(args.min_year, args.max_year + 1)
    raw = args.output / "raw"
    counts = {year: collect_counts(raw, args.field, year) for year in years}
    rows = {year: count_rows(year, *counts[year], args.min_papers, args.cap) for year in years}
    all_rows = [row for year in years for row in rows[year]]
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "topic_counts.csv", all_rows, COUNT_COLUMNS)

    panel = [row for row in all_rows if row["in_panel"]]
    planned = sum(row["sample_size"] for row in panel)
    requests = 2 * len(years) + sum(pages_needed(row["sample_size"]) for row in panel)
    print(f"Plan: {len(panel):,} topic-years from {len({row['topic_id'] for row in panel})} topics, "
          f"{planned:,} papers, {requests:,} requests (about ${requests * COST_PER_REQUEST:.2f})")

    pool_ids = None
    if args.pool:
        pool_ids = set(pd.read_parquet(args.pool, columns=["openalex_id"])["openalex_id"])
    reports, frames = [], []
    if not args.counts_only:
        check_disk(args.output, planned)
    for year in years:
        papers, records = [], list(counts[year])
        if not args.counts_only:
            for row in rows[year]:
                if row["in_panel"]:
                    works, sample_records = sample_topic_year(raw, args.field, year, row["topic_id"],
                                                              row["sample_size"])
                    papers += [flatten(work, position, pool_ids) for position, work in enumerate(works)]
                    records += sample_records
            frames.append(pd.DataFrame(papers, columns=COLUMNS))
        report = year_report(year, rows[year], papers, records, pool_ids)
        reports.append(report)
        share = report["panel_share_abstract"] or 0
        print(f"{year}: {report['panel_topics']} of {report['topics']} topics in the panel "
              f"({share:.1%} of articles with abstracts); "
              + (f"{report['planned_sample']:,} papers to sample" if args.counts_only else
                 f"{report['sampled']:,} papers sampled"
                 + (f", {report['in_pool']:,} already in the pool" if pool_ids is not None else "")))

    sampled_rows = None if args.counts_only else write_sample(args.output, frames)
    write_csv(args.output / "sample_report.csv", reports, list(reports[0]))
    write_metadata(args.output, args, reports, sampled_rows)
    if args.counts_only:
        print(f"Counts saved to {args.output / 'topic_counts.csv'}; no papers were sampled")
    else:
        print(f"Saved {sampled_rows:,} papers to {args.output / 'sample.parquet'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    try:
        main()
    except extract.DailyBudgetExceeded as error:
        sys.exit(f"{error}. Checkpoints were kept; rerun the same command to resume.")
    except RuntimeError as error:
        sys.exit(str(error))

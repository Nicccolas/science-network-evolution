#!/usr/bin/env python3
"""Download a fixed-size random pool of journal articles per year for one primary field."""
import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_test import extract  # noqa: E402  (reuses fetch, checkpointing, and atomic writes)
from src import openalex_counts  # noqa: E402  (git_commit and the classifier note)


POOL_SIZE = 10_000      # papers per year; KE draws are 1,000 (decision log, 2026-10-23)
SAMPLE_SEED = 1         # OpenAlex seed for the main sample
TOPUP_SIZE = 2_000      # size of each top-up sample, used only with --min-abstract-chars
MAX_TOPUP_SAMPLES = 5   # top-up seeds SAMPLE_SEED + 1 ... SAMPLE_SEED + 5
TOPUP_RNG_SEED = 0      # numpy seed for choosing top-up papers
SELECT = ("id,doi,title,abstract_inverted_index,publication_year,publication_date,type,language,"
          "is_paratext,primary_topic,primary_location,cited_by_count")
COLUMNS = (
    "openalex_id", "publication_year", "doi", "title", "abstract", "abstract_chars",
    "publication_date", "type", "language", "primary_topic", "primary_topic_id",
    "primary_topic_score", "primary_subfield", "primary_subfield_id", "primary_field",
    "primary_field_id", "source_id", "source_name", "source_issn_l", "source_type",
    "cited_by_count", "source_seed", "sample_position",
)
SAMPLING_NOTE = ("Each year is one OpenAlex random sample (sample=pool_size, seed=sample_seed) of "
                 "the filtered year, paged with basic paging (per_page=100). OpenAlex documents "
                 "sample= as random results and seed= as reproducible sampling; a later re-run can "
                 "differ as records are added or reclassified, so the raw pages are the record.")
CLEANING_NOTE = ("No abstract cleaning beyond has_abstract:true, following Hao et al. (2026). "
                 "abstract_chars is kept so short abstracts can be inspected later.")


def year_filter(field, year):
    return (f"primary_topic.field.id:{field},type:article,language:en,is_paratext:false,"
            f"{openalex_counts.JOURNAL},has_abstract:true,publication_year:{year}")


def validate_page(payload, field, year, expected):
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != expected:
        count = len(results) if isinstance(results, list) else None
        raise RuntimeError(f"{year}: OpenAlex returned {count} works on a page; expected {expected}")
    field_id = f"https://openalex.org/fields/{field}"
    for work in results:
        source = ((work.get("primary_location") or {}).get("source") or {}) if isinstance(work, dict) else {}
        if (not isinstance(work, dict) or not isinstance(work.get("id"), str)
                or work.get("publication_year") != year or work.get("type") != "article"
                or work.get("language") != "en" or work.get("is_paratext") is not False
                or ((work.get("primary_topic") or {}).get("field") or {}).get("id") != field_id
                or source.get("type") != "journal"):
            raise RuntimeError(f"{year}: OpenAlex returned a work outside the filtered frame: "
                               f"{work.get('id') if isinstance(work, dict) else work!r}")
        if not isinstance(work.get("abstract_inverted_index"), dict) or not work["abstract_inverted_index"]:
            raise RuntimeError(f"{year}: {work['id']} has an empty abstract index")


def sample_works(directory, field, year, seed, size):
    """Fetch one seeded OpenAlex sample, page by page. Returns (works, records)."""
    params = {"filter": year_filter(field, year), "corpus": "core", "sample": size,
              "seed": seed, "per_page": 100, "select": SELECT}
    works, records, seen = [], [], set()
    for page in range(1, (size + 99) // 100 + 1):
        expected = min(100, size - (page - 1) * 100)
        record = extract.checkpoint_request(
            directory / f"seed={seed}" / f"page_{page:04d}.json",
            extract.request_url({**params, "page": page}),
            lambda payload, n=expected: validate_page(payload, field, year, n))
        if record["payload"].get("meta", {}).get("count") != size:
            raise RuntimeError(f"{year}: seed {seed} sample count is not {size}")
        for work in record["payload"]["results"]:
            if work["id"] in seen:
                raise RuntimeError(f"{year}: seed {seed} sample repeats {work['id']}")
            seen.add(work["id"])
            works.append(work)
        records.append(record)
    return works, records


def flatten(work, seed, position):
    source = (work.get("primary_location") or {}).get("source") or {}
    topic = work.get("primary_topic") or {}
    subfield = topic.get("subfield") or {}
    field = topic.get("field") or {}
    abstract = extract.reconstruct_abstract(work["abstract_inverted_index"])
    return {
        "openalex_id": work["id"], "publication_year": work["publication_year"],
        "doi": work.get("doi"), "title": work.get("title"), "abstract": abstract,
        "abstract_chars": len(abstract), "publication_date": work.get("publication_date"),
        "type": work.get("type"), "language": work.get("language"),
        "primary_topic": topic.get("display_name"), "primary_topic_id": topic.get("id"),
        "primary_topic_score": topic.get("score"),
        "primary_subfield": subfield.get("display_name"), "primary_subfield_id": subfield.get("id"),
        "primary_field": field.get("display_name"), "primary_field_id": field.get("id"),
        "source_id": source.get("id"), "source_name": source.get("display_name"),
        "source_issn_l": source.get("issn_l"), "source_type": source.get("type"),
        "cited_by_count": work.get("cited_by_count"),
        "source_seed": seed, "sample_position": position,
    }


def collect_year(raw, field, year, pool_size, min_chars):
    """Return (rows, report) for one year's pool of exactly pool_size papers."""
    directory = raw / f"year={year}"
    count_record = extract.checkpoint_request(
        directory / "count.json",
        extract.request_url({"filter": year_filter(field, year), "corpus": "core",
                             "per_page": 1, "select": "id"}),
        extract.validate_count)
    frame = count_record["payload"]["meta"]["count"]
    if frame < pool_size:
        raise RuntimeError(f"{year}: only {frame:,} eligible works; the pool needs {pool_size:,}")

    works, records = sample_works(directory, field, year, SAMPLE_SEED, pool_size)
    rows = [flatten(work, SAMPLE_SEED, position) for position, work in enumerate(works)]
    kept = [row for row in rows if row["abstract_chars"] >= min_chars]
    dropped = len(rows) - len(kept)
    topped_up, seed = 0, SAMPLE_SEED
    rng = np.random.default_rng([TOPUP_RNG_SEED, year])  # per year, so runs over any year range agree
    while len(kept) < pool_size:
        seed += 1
        if seed > SAMPLE_SEED + MAX_TOPUP_SAMPLES:
            raise RuntimeError(f"{year}: still {pool_size - len(kept)} short after "
                               f"{MAX_TOPUP_SAMPLES} top-up samples")
        extra, extra_records = sample_works(directory, field, year, seed, min(TOPUP_SIZE, frame))
        records += extra_records
        have = {row["openalex_id"] for row in kept} | {row["openalex_id"] for row in rows}
        candidates = [row for row in (flatten(work, seed, position) for position, work in enumerate(extra))
                      if row["openalex_id"] not in have and row["abstract_chars"] >= min_chars]
        need = pool_size - len(kept)
        if len(candidates) > need:
            chosen = sorted(rng.choice(len(candidates), size=need, replace=False))
            candidates = [candidates[i] for i in chosen]
        kept += candidates
        topped_up += len(candidates)

    times = [count_record["retrieved_at_utc"]] + [r["retrieved_at_utc"] for r in records]
    report = {
        "year": year, "frame_count": frame, "sampled": len(rows),
        "dropped_short": dropped, "topped_up": topped_up, "final": len(kept),
        "topup_samples": seed - SAMPLE_SEED,
        "missing_title": sum(not row["title"] for row in kept),
        "median_abstract_chars": int(np.median([row["abstract_chars"] for row in kept])),
        "requests": len(records) + 1,
        "api_cost_usd": round(sum(r["payload"]["meta"].get("cost_usd") or 0
                                  for r in [count_record, *records]), 6),
        "first_retrieved_at_utc": min(times), "last_retrieved_at_utc": max(times),
    }
    return kept, report


def write_outputs(output, rows, reports, args):
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if frame["openalex_id"].duplicated().any():
        raise RuntimeError("The assembled pool repeats a work across years")
    temporary = output / ".pool.parquet.tmp"
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output / "pool.parquet")

    temporary = output / ".pool_report.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    temporary.replace(output / "pool_report.csv")

    extract.atomic_json(output / "metadata.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "src/openalex_pool.py",
        "git_commit": openalex_counts.git_commit(),
        "field": args.field, "years": [args.min_year, args.max_year],
        "filter_template": year_filter(args.field, "<year>"), "corpus": "core", "select": SELECT,
        "pool_size": args.pool_size, "sample_seed": SAMPLE_SEED,
        "min_abstract_chars": args.min_abstract_chars,
        "topup": {"sample_size": TOPUP_SIZE, "max_samples": MAX_TOPUP_SAMPLES,
                  "seeds_start_at": SAMPLE_SEED + 1,
                  "numpy_rng": f"default_rng([{TOPUP_RNG_SEED}, year])",
                  "used": any(r["topped_up"] for r in reports)},
        "rows": len(frame),
        "requests": sum(r["requests"] for r in reports),
        "reported_cost_usd": round(sum(r["api_cost_usd"] for r in reports), 6),
        "per_year": reports,
        "notes": [SAMPLING_NOTE, CLEANING_NOTE, openalex_counts.CLASSIFIER_NOTE],
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=int, required=True, help="OpenAlex field ID, e.g. 32 (Psychology)")
    parser.add_argument("--min-year", type=int, required=True)
    parser.add_argument("--max-year", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE, help="Papers per year, 1–10,000")
    parser.add_argument("--min-abstract-chars", type=int, default=0,
                        help="Optional: drop abstracts shorter than this and top up (off by default)")
    args = parser.parse_args(argv)
    if args.field < 1 or args.min_year > args.max_year:
        parser.error("need a positive field ID and min-year <= max-year")
    if not 1 <= args.pool_size <= 10_000:
        parser.error("--pool-size must be between 1 and 10,000 (the OpenAlex sample cap)")
    if args.min_abstract_chars < 0:
        parser.error("--min-abstract-chars must not be negative")

    rows, reports = [], []
    for year in range(args.min_year, args.max_year + 1):
        kept, report = collect_year(args.output / "raw", args.field, year, args.pool_size,
                                    args.min_abstract_chars)
        rows += kept
        reports.append(report)
        print(f"{year}: {report['final']:,} papers from {report['frame_count']:,} eligible "
              f"(dropped {report['dropped_short']}, topped up {report['topped_up']})")
    write_outputs(args.output, rows, reports, args)
    print(f"Saved {len(rows):,} papers to {args.output / 'pool.parquet'}")


if __name__ == "__main__":
    try:
        main()
    except extract.DailyBudgetExceeded as error:
        sys.exit(f"{error}. Checkpoints were kept; rerun the same command to resume.")

#!/usr/bin/env python3
"""Count OpenAlex works per publication year for one primary field under several inclusion rules."""
import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_test import extract  # noqa: E402  (reuses fetch, checkpointing, and atomic writes)


POOL_SIZE = 10_000  # KE pool per year; draws are 1,000 papers (decision log, 2026-10-23)
CLASSIFIER_NOTE = ("Primary-topic labels come from the OpenAlex topic classifier introduced on "
                   "2026-10-06; earlier pulls (e.g. api_test/data*) carry the old labels.")
CORPUS_NOTE = ("All main series use corpus=core (the OpenAlex default, passed explicitly). "
               "articles_expansion counts the same articles in the opt-in expansion corpus "
               "(formerly XPAC) and is reported for context only.")
ABSTRACT_NOTE = ("has_abstract:true is an upper bound on usable abstracts: about 6% of abstracts "
                 "in the 2013-2016 development sample were junk (<40 tokens), so pool_ratio "
                 "should stay comfortably above 1.")
JOURNAL_NOTE = ("journal_articles is N_t (decision log, 2026-10-25): articles whose primary "
                "location is a journal (primary_location.source.type:journal). The KE pool is "
                "drawn from journal_articles_abstract; meets_10k_pool and pool_ratio refer to it.")
JOURNAL = "primary_location.source.type:journal"


def series_filters(field, min_year, max_year):
    """Return {series: (filter, corpus)} in CSV column order."""
    dates = f"from_publication_date:{min_year}-01-01,to_publication_date:{max_year}-12-31"
    base = f"primary_topic.field.id:{field},language:en,is_paratext:false,{dates}"
    return {
        "journal_articles": (f"{base},type:article,{JOURNAL}", "core"),
        "journal_articles_abstract": (f"{base},type:article,{JOURNAL},has_abstract:true", "core"),
        "articles": (f"{base},type:article", "core"),
        "articles_abstract": (f"{base},type:article,has_abstract:true", "core"),
        "articles_reviews": (f"{base},type:article|review", "core"),
        "articles_reviews_abstract": (f"{base},type:article|review,has_abstract:true", "core"),
        "field_all": (f"primary_topic.field.id:{field},{dates}", "core"),
        "any_topic_articles": (f"topics.field.id:{field},language:en,is_paratext:false,type:article,{dates}", "core"),
        "articles_expansion": (f"{base},type:article", "expansion"),
    }


def year_counts(payload, min_year, max_year):
    groups = payload.get("group_by")
    total = payload.get("meta", {}).get("count")
    if not isinstance(groups, list) or not isinstance(total, int):
        raise RuntimeError("OpenAlex response lacks group_by counts")
    counts = {}
    for group in groups:
        key, count = group.get("key"), group.get("count")
        if not isinstance(count, int) or count < 0 or not str(key).isdigit():
            raise RuntimeError(f"Unexpected year group in OpenAlex response: {group!r}")
        year = int(key)
        if not min_year <= year <= max_year or year in counts:
            raise RuntimeError(f"OpenAlex returned an unexpected or repeated year: {year}")
        counts[year] = count
    if sum(counts.values()) != total:
        raise RuntimeError(f"Year groups sum to {sum(counts.values())}, but meta.count is {total}")
    return counts


def validate_field(payload):
    if not isinstance(payload.get("display_name"), str) or not payload.get("id"):
        raise RuntimeError("OpenAlex field lookup returned no field")


def git_commit():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "api_test"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def collect(field, min_year, max_year, output):
    raw = output / "raw"
    field_record = extract.checkpoint_request(
        raw / "field.json", f"https://api.openalex.org/fields/{field}", validate_field)
    records, counts = {}, {}
    for name, (filter_value, corpus) in series_filters(field, min_year, max_year).items():
        url = extract.request_url({"filter": filter_value, "group_by": "publication_year",
                                   "corpus": corpus})
        records[name] = extract.checkpoint_request(
            raw / f"{name}.json", url, lambda payload: year_counts(payload, min_year, max_year))
        counts[name] = year_counts(records[name]["payload"], min_year, max_year)
    return field_record, records, counts


def build_rows(counts, min_year, max_year):
    rows = []
    for year in range(min_year, max_year + 1):
        row = {"year": year, **{name: series.get(year, 0) for name, series in counts.items()}}
        row["abstract_share"] = round(row["articles_abstract"] / row["articles"], 4) if row["articles"] else None
        row["journal_share"] = round(row["journal_articles"] / row["articles"], 4) if row["articles"] else None
        row["journal_abstract_share"] = (round(row["journal_articles_abstract"] / row["journal_articles"], 4)
                                         if row["journal_articles"] else None)
        row["pool_ratio"] = round(row["journal_articles_abstract"] / POOL_SIZE, 2)
        row["meets_10k_pool"] = row["journal_articles_abstract"] >= POOL_SIZE
        rows.append(row)
    return rows


def write_outputs(output, rows, field_record, records, field, min_year, max_year):
    columns = list(rows[0])
    temporary = output / ".yearly_counts.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "yearly_counts.csv")

    field_payload = field_record["payload"]
    filters = series_filters(field, min_year, max_year)
    extract.atomic_json(output / "metadata.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "src/openalex_counts.py",
        "git_commit": git_commit(),
        "field": {"id": field_payload.get("id"), "display_name": field_payload.get("display_name"),
                  "domain": (field_payload.get("domain") or {}).get("display_name"),
                  "works_count_at_lookup": field_payload.get("works_count"),
                  "subfields": [s.get("display_name") for s in field_payload.get("subfields") or []]},
        "years": [min_year, max_year],
        "group_by": "publication_year",
        "pool_size": POOL_SIZE,
        "series": {name: {"filter": filters[name][0], "corpus": filters[name][1],
                          "request_url": record["request_url"],
                          "retrieved_at_utc": record["retrieved_at_utc"],
                          "total": record["payload"]["meta"]["count"]}
                   for name, record in records.items()},
        "requests": len(records) + 1,
        "reported_cost_usd": round(sum(r["payload"]["meta"].get("cost_usd") or 0
                                       for r in records.values()), 6),
        "notes": [JOURNAL_NOTE, CLASSIFIER_NOTE, CORPUS_NOTE, ABSTRACT_NOTE,
                  "Years with no matching works are written as 0."],
    })


def print_table(rows):
    columns = ("year", "journal_articles", "journal_articles_abstract", "journal_share",
               "articles", "articles_abstract", "any_topic_articles", "pool_ratio")
    print("  ".join(f"{c:>18s}" if c != "year" else "year" for c in columns))
    for row in rows:
        cells = [str(row["year"])]
        for column in columns[1:]:
            value = row[column]
            cells.append(f"{value:>18,}" if isinstance(value, int) else f"{value!s:>18}")
        print("  ".join(cells))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=int, required=True, help="OpenAlex field ID, e.g. 32 (Psychology)")
    parser.add_argument("--min-year", type=int, required=True)
    parser.add_argument("--max-year", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.field < 1 or args.min_year > args.max_year or args.max_year - args.min_year >= 200:
        parser.error("need a positive field ID and a year range of at most 200 years")

    field_record, records, counts = collect(args.field, args.min_year, args.max_year, args.output)
    rows = build_rows(counts, args.min_year, args.max_year)
    write_outputs(args.output, rows, field_record, records, args.field, args.min_year, args.max_year)
    print(f"{field_record['payload']['display_name']} (field {args.field}); "
          f"outputs in {args.output}")
    print_table(rows)


if __name__ == "__main__":
    try:
        main()
    except extract.DailyBudgetExceeded as error:
        sys.exit(str(error))

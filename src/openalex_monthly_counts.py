#!/usr/bin/env python3
"""Monthly OpenAlex counts of one field's journal articles by primary topic, for the time-series notebook.

For every year, OpenAlex group_by primary_topic.id counts of the year's journal articles (the N_kt
frame of src/openalex_topic_sample.py): one for the whole year, one per calendar month
(from_publication_date and to_publication_date), and one for papers dated 1 January. OpenAlex dates
a paper 1 January when only its year is known, so the 1 January count is what the notebook spreads
over the year. All of a year's counts come from one run, so the twelve months should add up to
the year; the run stops if any field-year differs by more than --max-gap.

Outputs in --output:
  monthly_counts.csv  topic_id, topic_name, year, month, journal_articles; every topic counted in the
                      year has all twelve months, zeros included
  annual_counts.csv   topic_id, topic_name, year, journal_articles, months_sum, jan1_articles
  metadata.json       filters, per-year totals and gaps, request totals, and notes
  raw/                every response

For 2001-2024 this is 336 requests (about $0.03). Requests are checkpointed, so the same command
resumes.

    python3 -m src.openalex_monthly_counts --field 32 --min-year 2001 --max-year 2024 \\
        --output data/topic_sample/psychology_field32/monthly
"""
import argparse
import calendar
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_test import extract  # noqa: E402  (checkpointing, atomic writes)
from src import openalex_counts  # noqa: E402  (git_commit, classifier note)
from src.openalex_topic_sample import (  # noqa: E402  (the N_kt frame and its checked group_by counts)
    count_request, frame_filter, topic_counts, topic_number, write_csv)


MAX_GAP = 0.005  # largest allowed |months - year| / year for a field-year
MONTHLY_COLUMNS = ("topic_id", "topic_name", "year", "month", "journal_articles")
ANNUAL_COLUMNS = ("topic_id", "topic_name", "year", "journal_articles", "months_sum", "jan1_articles")

DATE_NOTE = ("OpenAlex sets publication_date to 1 January when only the year is known, so January "
             "holds both January papers and papers with no month. jan1_articles counts papers dated "
             "exactly 1 January; a real 1 January issue cannot be told apart from a missing month.")
COUNT_NOTE = ("journal_articles counts every journal article in the frame (the N_kt series), with or "
              "without an abstract. Counts move by a few papers from day to day, so months_sum and "
              "journal_articles can differ slightly within a topic-year.")


def month_filter(field, year, month):
    last = calendar.monthrange(year, month)[1]
    return (f"{frame_filter(field, year)},from_publication_date:{year}-{month:02d}-01,"
            f"to_publication_date:{year}-{month:02d}-{last:02d}")


def jan1_filter(field, year):
    return f"{frame_filter(field, year)},publication_date:{year}-01-01"


def collect_year(raw, field, year):
    """The year's records: the annual count, twelve monthly counts, and the 1 January count."""
    directory = raw / f"year={year}"
    annual = count_request(directory / "annual.json", frame_filter(field, year))
    months = [count_request(directory / f"month={month:02d}.json", month_filter(field, year, month))
              for month in range(1, 13)]
    jan1 = count_request(directory / "jan1.json", jan1_filter(field, year))
    return annual, months, jan1


def year_rows(year, annual, months, jan1):
    """Monthly rows (zeros filled) and annual rows for every topic in any of the year's counts."""
    annual_counts = topic_counts(annual["payload"])
    month_counts = [topic_counts(record["payload"]) for record in months]
    jan1_counts = topic_counts(jan1["payload"])
    names = {}
    for counts in (jan1_counts, *month_counts, annual_counts):  # the annual count's name wins
        names.update({topic: name for topic, (name, _) in counts.items()})
    monthly, yearly = [], []
    for topic in sorted(names, key=topic_number):
        by_month = [counts.get(topic, (None, 0))[1] for counts in month_counts]
        monthly += [{"topic_id": topic, "topic_name": names[topic], "year": year, "month": month,
                     "journal_articles": n} for month, n in enumerate(by_month, start=1)]
        yearly.append({"topic_id": topic, "topic_name": names[topic], "year": year,
                       "journal_articles": annual_counts.get(topic, (None, 0))[1],
                       "months_sum": sum(by_month), "jan1_articles": jan1_counts.get(topic, (None, 0))[1]})
    return monthly, yearly


def year_report(year, monthly, yearly, records):
    total = sum(row["journal_articles"] for row in yearly)
    months_sum = sum(row["months_sum"] for row in yearly)
    january = {row["topic_id"]: row["journal_articles"] for row in monthly if row["month"] == 1}
    jan1 = sum(row["jan1_articles"] for row in yearly)
    times = [record["retrieved_at_utc"] for record in records]
    return {
        "year": year, "journal_articles": total, "months_sum": months_sum,
        "gap": round((months_sum - total) / total, 6) if total else None,
        "january": sum(january.values()), "jan1_articles": jan1,
        "jan1_share": round(jan1 / total, 4) if total else None,
        "topics": len(yearly),
        # Possible only through count drift between the January and 1 January requests.
        "topics_jan1_above_january": sum(row["jan1_articles"] > january[row["topic_id"]] for row in yearly),
        "requests": len(records),
        "api_cost_usd": round(sum(record["payload"]["meta"].get("cost_usd") or 0 for record in records), 6),
        "first_retrieved_at_utc": min(times), "last_retrieved_at_utc": max(times),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--field", type=int, required=True, help="OpenAlex field ID, e.g. 32 (Psychology)")
    parser.add_argument("--min-year", type=int, required=True)
    parser.add_argument("--max-year", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gap", type=float, default=MAX_GAP,
                        help="largest allowed |months - year| / year for a field-year")
    args = parser.parse_args(argv)
    if args.field < 1 or args.min_year > args.max_year:
        parser.error("need a positive field ID and min-year <= max-year")

    raw = args.output / "raw"
    monthly, yearly, reports = [], [], []
    for year in range(args.min_year, args.max_year + 1):
        annual, months, jan1 = collect_year(raw, args.field, year)
        year_monthly, year_yearly = year_rows(year, annual, months, jan1)
        report = year_report(year, year_monthly, year_yearly, [annual, *months, jan1])
        print(f"{year}: {report['journal_articles']:,} articles, months sum to {report['months_sum']:,} "
              f"(gap {report['gap']:+.3%}), {report['jan1_share']:.1%} dated 1 January")
        monthly += year_monthly
        yearly += year_yearly
        reports.append(report)

    off = [report for report in reports if report["gap"] is None or abs(report["gap"]) > args.max_gap]
    if off:
        raise RuntimeError("Monthly counts do not add up to the annual count in "
                           + ", ".join(str(report["year"]) for report in off)
                           + f" (more than {args.max_gap:.1%}); delete raw/year=<year>/ and rerun")

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "monthly_counts.csv", monthly, MONTHLY_COLUMNS)
    write_csv(args.output / "annual_counts.csv", yearly, ANNUAL_COLUMNS)
    extract.atomic_json(args.output / "metadata.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "src/openalex_monthly_counts.py",
        "git_commit": openalex_counts.git_commit(),
        "field": args.field, "years": [args.min_year, args.max_year],
        "filters": {"annual": frame_filter(args.field, "<year>"),
                    "month": (f"{frame_filter(args.field, '<year>')},from_publication_date:"
                              "<year>-<month>-01,to_publication_date:<year>-<month>-<last day>"),
                    "jan1": jan1_filter(args.field, "<year>")},
        "group_by": "primary_topic.id", "corpus": "core", "max_gap": args.max_gap,
        "requests": sum(report["requests"] for report in reports),
        "reported_cost_usd": round(sum(report["api_cost_usd"] for report in reports), 6),
        "per_year": reports,
        "notes": [COUNT_NOTE, DATE_NOTE, openalex_counts.CLASSIFIER_NOTE],
    })
    print(f"Saved {len(monthly):,} topic-months to {args.output / 'monthly_counts.csv'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    try:
        main()
    except extract.DailyBudgetExceeded as error:
        sys.exit(f"{error}. Checkpoints were kept; rerun the same command to resume.")
    except RuntimeError as error:
        sys.exit(str(error))

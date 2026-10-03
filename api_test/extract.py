#!/usr/bin/env python3
"""Collect balanced, seeded OpenAlex samples and their outgoing citations."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SELECT = "id,doi,title,abstract_inverted_index,publication_year,publication_date,type,language,cited_by_count,open_access,primary_location,primary_topic,authorships,referenced_works"
OUTPUT_FILES = ("works_raw.json", "works.csv", "citation_edges.csv", "citation_nodes.csv", "metadata.json")
WORK_COLUMNS = (
    "openalex_id", "doi", "title", "abstract", "publication_year", "publication_date",
    "type", "language", "cited_by_count", "referenced_works_count",
    "referenced_works", "is_oa", "oa_status", "oa_url", "source_name",
    "primary_topic", "primary_topic_id", "primary_topic_score",
    "primary_subfield", "primary_subfield_id", "primary_field", "primary_field_id",
    "primary_domain", "primary_domain_id", "authors", "author_ids",
)


class DailyBudgetExceeded(RuntimeError):
    def __init__(self, reset_seconds):
        if reset_seconds is None:
            message = "OpenAlex daily budget exhausted; resume after midnight UTC"
        else:
            reset_at = datetime.now(timezone.utc) + timedelta(seconds=reset_seconds)
            message = f"OpenAlex daily budget exhausted; resume after {reset_at.isoformat()}"
        super().__init__(message)


def header_int(headers, name):
    try:
        return int(headers.get(name))
    except (TypeError, ValueError):
        return None


def fetch(url):
    headers = {"User-Agent": "OpenAlex-MVP/1.0", "Accept": "application/json"}
    if os.environ.get("OPENALEX_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENALEX_API_KEY"]
    for attempt in range(5):
        try:
            with urlopen(Request(url, headers=headers), timeout=30) as response:
                rate = {"remaining": header_int(response.headers, "X-RateLimit-Remaining"),
                        "reset_seconds": header_int(response.headers, "X-RateLimit-Reset")}
                return json.load(response), rate
        except HTTPError as error:
            if error.code == 429:
                remaining = header_int(error.headers, "X-RateLimit-Remaining")
                if remaining == 0:
                    raise DailyBudgetExceeded(header_int(error.headers, "X-RateLimit-Reset")) from None
            elif error.code not in (500, 502, 503, 504):
                raise RuntimeError(f"OpenAlex returned HTTP {error.code}") from None
        except (URLError, TimeoutError):
            pass
        if attempt < 4:
            time.sleep(2 ** attempt)
    raise RuntimeError("OpenAlex request failed after five attempts; checkpoints were kept")

def reconstruct_abstract(index):
    if not index:
        return None
    words = [(position, word)
             for word, positions in index.items()
             for position in positions]
    return " ".join(word for _, word in sorted(words))

def flatten(work):
    source = (work.get("primary_location") or {}).get("source") or {}
    topic = work.get("primary_topic") or {}
    subfield = topic.get("subfield") or {}
    field = topic.get("field") or {}
    domain = topic.get("domain") or {}
    oa = work.get("open_access") or {}
    authors = work.get("authorships") or []
    return {
        "openalex_id": work["id"],
        "doi": work.get("doi"),
        "title": work.get("title"),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "type": work.get("type"),
        "language": work.get("language"),
        "cited_by_count": work.get("cited_by_count"),
        "referenced_works_count": len(work["referenced_works"]),
        "referenced_works": "; ".join(work["referenced_works"]),
        "is_oa": oa.get("is_oa"),
        "oa_status": oa.get("oa_status"),
        "oa_url": oa.get("oa_url"),
        "source_name": source.get("display_name"),
        "primary_topic": topic.get("display_name"),
        "primary_topic_id": topic.get("id"),
        "primary_topic_score": topic.get("score"),
        "primary_subfield": subfield.get("display_name"),
        "primary_subfield_id": subfield.get("id"),
        "primary_field": field.get("display_name"),
        "primary_field_id": field.get("id"),
        "primary_domain": domain.get("display_name"),
        "primary_domain_id": domain.get("id"),
        "authors": "; ".join(a["author"].get("display_name") or "" for a in authors),
        "author_ids": "; ".join(a["author"].get("id") or "" for a in authors),
    }


def parse_fields(value):
    if value is None:
        return ()
    value = value.strip()
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1]
    parts = [part.strip() for part in value.split(",")]
    if any(not re.fullmatch(r"[0-9]+", part) or int(part) < 1 for part in parts):
        raise ValueError("--fields must be positive, comma-separated IDs such as 30,31,32")
    fields = tuple(dict.fromkeys(str(int(part)) for part in parts))
    if len(fields) > 100:
        raise ValueError("--fields accepts at most 100 distinct IDs")
    return fields


def works_filter(year, fields):
    value = f"publication_year:{year},type:article,language:en"
    if fields:
        value += ",primary_topic.field.id:" + "|".join(fields)
    return value


def request_url(params):
    return "https://api.openalex.org/works?" + urlencode(params)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".checkpoint-", delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(value, file, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def validate_page(payload, year, fields, expected=None):
    results = payload.get("results")
    if not isinstance(results, list) or len(results) > 100:
        raise RuntimeError("OpenAlex returned an invalid results page")
    if expected is not None and len(results) != expected:
        raise RuntimeError(f"OpenAlex returned {len(results)} works; expected {expected}")
    allowed = {f"https://openalex.org/fields/{field}" for field in fields}
    for work in results:
        if (not isinstance(work, dict) or not isinstance(work.get("id"), str)
                or work.get("publication_year") != year or work.get("type") != "article"
                or work.get("language") != "en"):
            raise RuntimeError(f"OpenAlex returned a work outside the {year} English-article cohort")
        if fields and ((work.get("primary_topic") or {}).get("field") or {}).get("id") not in allowed:
            raise RuntimeError("OpenAlex returned a work outside the requested primary fields")
        refs = work.get("referenced_works")
        if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
            raise RuntimeError("OpenAlex did not return a valid reference list")


def validate_count(payload):
    count = payload.get("meta", {}).get("count")
    if not isinstance(count, int) or count < 0:
        raise RuntimeError("OpenAlex response lacks a population count")


def checkpoint_request(path, url, validator):
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("request_url") != url:
            raise RuntimeError(f"Checkpoint query differs from this run: {path}")
    else:
        payload, rate = fetch(url)
        validator(payload)
        record = {"request_url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                  "rate": rate, "payload": payload}
        atomic_json(path, record)
        if rate.get("remaining") is not None and rate["remaining"] < 10:
            print(f"OpenAlex daily credits remaining: {rate['remaining']}", file=sys.stderr)
    validator(record["payload"])
    return record


def collect_pages(checkpoint, config):
    page_paths = []
    per_year = []
    fields = tuple(config["fields"])
    for year in range(config["min_year"], config["max_year"] + 1):
        directory = checkpoint / f"year={year}"
        common = {"filter": works_filter(year, fields)}
        if config["search"]:
            common["search"] = config["search"]
        count_url = request_url({**common, "per_page": 1, "select": "id"})
        count_record = checkpoint_request(directory / "count.json", count_url, validate_count)
        population = count_record["payload"]["meta"]["count"]
        cost = count_record["payload"]["meta"].get("cost_usd") or 0
        available = min(config["size"], population)
        times = [count_record["retrieved_at_utc"]]
        sampled_count = 0
        pages = 0
        if available:
            sample_params = {**common, "sample": available, "seed": config["seed"],
                             "per_page": 100, "select": SELECT}
            first_url = request_url({**sample_params, "page": 1})
            first = checkpoint_request(
                directory / "page_0001.json", first_url,
                lambda payload: validate_page(payload, year, fields),
            )
            sampled_count = first["payload"].get("meta", {}).get("count")
            if not isinstance(sampled_count, int) or not 1 <= sampled_count <= available:
                raise RuntimeError(f"Invalid sampled count for {year}")
            pages = (sampled_count + 99) // 100
            for page in range(1, pages + 1):
                path = directory / f"page_{page:04d}.json"
                expected = min(100, sampled_count - (page - 1) * 100)
                url = request_url({**sample_params, "page": page})
                record = checkpoint_request(
                    path, url,
                    lambda payload, n=expected: validate_page(payload, year, fields, n),
                )
                if record["payload"].get("meta", {}).get("count") != sampled_count:
                    raise RuntimeError(f"Sample count changed across pages for {year}")
                times.append(record["retrieved_at_utc"])
                cost += record["payload"]["meta"].get("cost_usd") or 0
                page_paths.append(path)
        per_year.append({"year": year, "population_count": population,
                         "requested_sample": config["size"], "sample_count": sampled_count,
                         "pages": pages, "filter": common["filter"],
                         "first_retrieved_at_utc": min(times),
                         "last_retrieved_at_utc": max(times), "api_cost_usd": cost})
        print(f"{year}: {sampled_count} sampled works from {population:,} matching works")
    return page_paths, per_year


def assemble(output, page_paths, config, per_year, refresh=False):
    """Stream checkpoint pages into staged files, using SQLite for network keys."""
    with tempfile.TemporaryDirectory(prefix=".extract-build-", dir=output) as temporary:
        stage = Path(temporary)
        database = sqlite3.connect(stage / "network.sqlite3")
        try:
            database.execute("CREATE TABLE works (id TEXT PRIMARY KEY, title TEXT, year INTEGER)")
            database.execute("CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, PRIMARY KEY (source, target))")
            work_count = 0
            with (stage / "works.csv").open("w", encoding="utf-8", newline="") as csv_file, \
                 (stage / "works_raw.json").open("w", encoding="utf-8") as raw_file:
                writer = csv.DictWriter(csv_file, fieldnames=WORK_COLUMNS)
                writer.writeheader()
                missing = dict.fromkeys(WORK_COLUMNS, 0)
                raw_file.write('{"meta":')
                json.dump({"combined_pages": True, "sample_strategy": "equal_per_year",
                           "per_year": per_year}, raw_file, ensure_ascii=False)
                raw_file.write(',"results":[\n')
                for path in page_paths:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    for work in record["payload"]["results"]:
                        try:
                            database.execute("INSERT INTO works VALUES (?, ?, ?)",
                                             (work["id"], work.get("title"), work["publication_year"]))
                        except sqlite3.IntegrityError:
                            raise RuntimeError(f"Duplicate work ID across pages: {work['id']}") from None
                        database.executemany("INSERT OR IGNORE INTO edges VALUES (?, ?)",
                                             ((work["id"], target) for target in work["referenced_works"]))
                        row = flatten(work)
                        writer.writerow(row)
                        for key, value in row.items():
                            missing[key] += value is None or value == ""
                        if work_count:
                            raw_file.write(',\n')
                        json.dump(work, raw_file, ensure_ascii=False)
                        work_count += 1
                raw_file.write(']\n}\n')
            database.commit()
            expected = sum(item["sample_count"] for item in per_year)
            if work_count != expected:
                raise RuntimeError(f"Combined {work_count} works; expected {expected}")
            edge_count = database.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            internal_edges = database.execute(
                "SELECT COUNT(*) FROM edges JOIN works ON edges.target = works.id"
            ).fetchone()[0]
            external_nodes = database.execute(
                "SELECT COUNT(DISTINCT target) FROM edges WHERE target NOT IN (SELECT id FROM works)"
            ).fetchone()[0]
            sampled_with_refs = database.execute("SELECT COUNT(DISTINCT source) FROM edges").fetchone()[0]
            with (stage / "citation_edges.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Source", "Target", "Type"])
                writer.writerows((source, target, "Directed") for source, target in
                                 database.execute("SELECT source, target FROM edges ORDER BY source, target"))
            with (stage / "citation_nodes.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Id", "Label", "is_sampled", "references_fetched", "publication_year"])
                for work_id, title, year in database.execute("SELECT id, title, year FROM works ORDER BY id"):
                    writer.writerow([work_id, title or work_id, True, True, year])
                for (work_id,) in database.execute(
                    "SELECT DISTINCT target FROM edges WHERE target NOT IN (SELECT id FROM works) ORDER BY target"
                ):
                    writer.writerow([work_id, work_id, False, False, None])
            metadata = {
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "parameters": config, "per_year": per_year,
                "corpus": "OpenAlex default (core)",
                "requested_rows": sum(item["requested_sample"] for item in per_year),
                "returned_rows": work_count, "refresh_existing": refresh,
                "request_count": len(page_paths) + (0 if refresh else len(per_year)),
                "estimated_api_cost_usd": round(sum(item.get("api_cost_usd", 0) for item in per_year), 6),
                "citation_network": {
                    "nodes": work_count + external_nodes, "edges": edge_count,
                    "direction": "citing work -> cited work",
                    "sampled_works_with_references": sampled_with_refs,
                    "edges_within_sample": internal_edges,
                    "scope": "Outgoing references of sampled works only; external nodes have no fetched metadata or outgoing references.",
                },
                "unique_work_ids": work_count, "missing_values_by_column": missing,
                "raw_json_note": "works_raw.json combines original work objects from all API pages",
            }
            (stage / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            for name in OUTPUT_FILES:
                os.replace(stage / name, output / name)
        finally:
            database.close()
    print(f"Saved {work_count} unique works to {output.resolve()}")
    print(f"Citation network: {work_count + external_nodes} nodes, {edge_count} directed edges")


def refresh_existing(output):
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        count = json.loads(metadata_path.read_text(encoding="utf-8")).get("returned_rows")
        if isinstance(count, int) and count > 100:
            raise RuntimeError("--refresh-existing supports at most 100 saved works")
    previous = json.loads((output / "works_raw.json").read_text(encoding="utf-8"))
    existing = [work["id"] for work in previous["results"]]
    if not 1 <= len(existing) <= 100:
        raise RuntimeError("--refresh-existing supports 1–100 saved works")
    url = request_url({"filter": "openalex_id:" + "|".join(existing),
                       "per_page": 100, "select": SELECT})
    payload, rate = fetch(url)
    works = payload.get("results", [])
    if len(works) != len(existing) or {work["id"] for work in works} != set(existing):
        raise RuntimeError("Refresh did not return every saved work; outputs were not changed")
    by_id = {work["id"]: work for work in works}
    works = [by_id[work_id] for work_id in existing]
    for work in works:
        validate_page({"results": [work]}, work["publication_year"], ())
    with tempfile.TemporaryDirectory(prefix=".refresh-", dir=output) as temporary:
        page = Path(temporary) / "page.json"
        now = datetime.now(timezone.utc).isoformat()
        atomic_json(page, {"request_url": url, "retrieved_at_utc": now,
                           "rate": rate, "payload": {"meta": payload.get("meta", {}),
                                                      "results": works}})
        counts = {}
        for work in works:
            counts[work["publication_year"]] = counts.get(work["publication_year"], 0) + 1
        per_year = [{"year": year, "population_count": None, "requested_sample": count,
                     "sample_count": count, "pages": 1, "filter": "openalex_id lookup",
                     "first_retrieved_at_utc": now, "last_retrieved_at_utc": now,
                     "api_cost_usd": payload.get("meta", {}).get("cost_usd") or 0}
                    for year, count in sorted(counts.items())]
        config = {"min_year": min(counts), "max_year": max(counts),
                  "size": len(works), "fields": [], "seed": None, "search": "",
                  "refresh_lookup_url": url}
        assemble(output, [page], config, per_year, refresh=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=50, help="Works per year, 1–10,000")
    parser.add_argument("--year", type=int, help="One-year shortcut; cannot combine with year bounds")
    parser.add_argument("--min-year", type=int, help="First publication year, inclusive; default 2024")
    parser.add_argument("--max-year", type=int, help="Last publication year, inclusive; default 2024")
    parser.add_argument("--fields", help='Primary field IDs, e.g. "30,31,32"; omit for all fields')
    parser.add_argument("--search", default="", help="Optional keyword search; costs more API credits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-existing", action="store_true", help="Refresh up to 100 exact work IDs in the output directory")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args(argv)
    if args.year is not None and (args.min_year is not None or args.max_year is not None):
        parser.error("--year cannot be combined with --min-year or --max-year")
    minimum = args.year if args.year is not None else (args.min_year if args.min_year is not None else 2024)
    maximum = args.year if args.year is not None else (args.max_year if args.max_year is not None else 2024)
    if minimum < 1 or maximum < minimum:
        parser.error("year bounds must be positive and min-year must not exceed max-year")
    if not 1 <= args.size <= 10000:
        parser.error("--size must be between 1 and 10,000 works per year")
    try:
        fields = parse_fields(args.fields)
    except ValueError as error:
        parser.error(str(error))
    output = args.output.expanduser()
    if args.refresh_existing:
        if any(value is not None for value in (args.year, args.min_year, args.max_year, args.fields)) or args.search:
            parser.error("--refresh-existing cannot be combined with year, field, or search filters")
        refresh_existing(output)
        return 0
    config = {"min_year": minimum, "max_year": maximum, "size": args.size,
              "fields": list(fields), "seed": args.seed, "search": args.search, "select": SELECT}
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / ".extract-checkpoint"
    config_path = checkpoint / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError("Unfinished checkpoint has different options; rerun the same command or choose another --output")
    else:
        if checkpoint.exists() and any(checkpoint.iterdir()):
            raise RuntimeError("Checkpoint has no config.json; inspect it before restarting")
        atomic_json(config_path, config)
    try:
        page_paths, per_year = collect_pages(checkpoint, config)
    except DailyBudgetExceeded as error:
        print(f"{error}. Checkpoints kept in {checkpoint}; rerun the same command to resume.",
              file=sys.stderr)
        return 2
    assemble(output, page_paths, config, per_year)
    shutil.rmtree(checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())

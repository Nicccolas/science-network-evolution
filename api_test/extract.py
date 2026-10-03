#!/usr/bin/env python3
"""Download a small seeded OpenAlex sample; Python standard library only."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def fetch(url):
    headers = {"User-Agent": "OpenAlex-MVP/1.0", "Accept": "application/json"}
    if os.environ.get("OPENALEX_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENALEX_API_KEY"]
    for attempt in range(5):
        try:
            with urlopen(Request(url, headers=headers), timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"OpenAlex returned HTTP {error.code}") from None
        except (URLError, TimeoutError):
            pass
        if attempt < 4:
            time.sleep(2 ** attempt)
    raise RuntimeError("OpenAlex request failed after five attempts; check network or API budget.")

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
        "authors": "; ".join(a["author"].get("display_name") or "" for a in authors),
        "author_ids": "; ".join(a["author"].get("id") or "" for a in authors),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=50, help="Sample size, 1–100")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--search", default="", help="Optional keyword search, not an exact topic classification")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-existing", action="store_true", help="Refresh the exact work IDs in the output directory instead of resampling")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    if not 1 <= args.size <= 100:
        parser.error("--size must be between 1 and 100 for this single-request MVP")
    params = {
        "filter": f"publication_year:{args.year},type:article,language:en",
        "sample": args.size, "seed": args.seed, "per_page": args.size,
        "select": "id,doi,title,abstract_inverted_index,publication_year,publication_date,type,language,cited_by_count,open_access,primary_location,primary_topic,authorships,referenced_works",
    }
    if args.search:
        params["search"] = args.search
    existing_ids = None
    if args.refresh_existing:
        previous = json.loads((args.output / "works_raw.json").read_text(encoding="utf-8"))
        existing_ids = [w["id"] for w in previous["results"]]
        if not 1 <= len(existing_ids) <= 100:
            parser.error("Existing sample must contain 1–100 works")
        params = {"filter": "openalex_id:" + "|".join(existing_ids), "per_page": 100, "select": params["select"]}
    url = "https://api.openalex.org/works?" + urlencode(params)
    payload = fetch(url)
    works = payload["results"]
    if not works:
        raise RuntimeError("No matching works; change the year or search terms.")
    ids = [w["id"] for w in works]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Unexpected duplicate work IDs")
    if existing_ids is not None:
        if set(ids) != set(existing_ids):
            raise RuntimeError("Refresh did not return every existing work; saved files were not changed")
        by_id = {w["id"]: w for w in works}
        works = [by_id[work_id] for work_id in existing_ids]
        payload["results"] = works
    elif any(w.get("publication_year") != args.year or w.get("type") != "article" for w in works):
        raise RuntimeError("API returned works outside the requested filters")
    if any(w.get("language") != "en" for w in works):
        raise RuntimeError("API returned non-English works; saved files were not changed")
    if any(not isinstance(w.get("referenced_works"), list) for w in works):
        raise RuntimeError("API did not return reference lists; saved files were not changed")
    rows = [flatten(w) for w in works]
    edges = sorted({(w["id"], target) for w in works for target in w["referenced_works"]})
    sampled = {w["id"]: w for w in works}
    node_ids = sorted(set(sampled) | {target for _, target in edges})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "works_raw.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output / "works.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "citation_edges.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Source", "Target", "Type"])
        writer.writerows((source, target, "Directed") for source, target in edges)
    with (args.output / "citation_nodes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Label", "is_sampled", "references_fetched", "publication_year"])
        for work_id in node_ids:
            work = sampled.get(work_id, {})
            writer.writerow([work_id, work.get("title") or work_id, work_id in sampled, work_id in sampled, work.get("publication_year")])
    metadata = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_url": url, "parameters": params,
        "corpus": "OpenAlex default (core)",
        "requested_rows": len(existing_ids) if existing_ids is not None else args.size, "returned_rows": len(rows),
        "refresh_existing": args.refresh_existing,
        "citation_network": {"nodes": len(node_ids), "edges": len(edges), "direction": "citing work -> cited work", "sampled_works_with_references": sum(bool(w["referenced_works"]) for w in works), "edges_within_sample": sum(target in sampled for _, target in edges), "scope": "Outgoing references of sampled works only; external nodes have no fetched metadata or outgoing references."},
        "unique_work_ids": len(set(ids)),
        "api_meta": payload.get("meta"),
        "missing_values_by_column": {key: sum(r[key] is None or r[key] == "" for r in rows) for key in rows[0]},
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} unique works to {args.output.resolve()}")
    print(f"Citation network: {len(node_ids)} nodes, {len(edges)} directed edges")
    if len(rows) < metadata["requested_rows"]:
        print("Fewer works returned than requested; see metadata.json.")


if __name__ == "__main__":
    main()

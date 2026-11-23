#!/usr/bin/env python3
"""Which pool papers enter the KE draws: the exclusion rules fixed on 2026-11-21 (decision log).

The downloaded pool stays at 10,000 papers per year; the rules are applied at the KE step, before
the 1,000-paper draws (decision log, 2026-10-26). A paper is excluded if any of these hold, with
text compared after lowercasing and trimming:

  missing_title         no title (Hao et al. 2026 also drop these)
  duplicate_record      same title and abstract as another paper of the same year: the same work
                        indexed twice; the copy with the lowest OpenAlex ID is kept
  boilerplate_abstract  an abstract shared by papers with different titles anywhere in the pool:
                        publisher or platform placeholder text; every copy is dropped
  short_abstract        an abstract under 200 characters ("-", "N/A", "No abstract", citation
                        strings): found by content, so it is caught in every year whether or not
                        another copy happens to be in the pool

The first two depend on which papers are in the pool, which is right for duplicate records (a
work counts twice in a draw only if both copies are in the pool) and catches the large
boilerplate groups in every year. The length rule needs no second copy.

Writes eligibility.parquet (pool order), eligibility_report.csv, and metadata.json:

    python3 -m src.ke_eligibility --pool data/pool/psychology_field32/pool.parquet \\
        --output data/ke/psychology_field32
"""
import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_test import extract  # noqa: E402  (atomic_json)
from src import embed_pool, openalex_counts  # noqa: E402  (file_sha256; git_commit)


MIN_ABSTRACT_CHARS = 200
DRAW_SIZE = 1_000  # KE draw (decision log, 2026-10-23), for the pool-to-draw ratio in the report
RULES = {
    "missing_title": "no title",
    "duplicate_record": "same title and abstract as another paper of the same year; the copy with "
                        "the lowest OpenAlex ID is kept",
    "boilerplate_abstract": "abstract shared by papers with two or more different non-missing "
                            "titles anywhere in the pool; every copy is dropped",
    "short_abstract": f"abstract under {MIN_ABSTRACT_CHARS} characters",
}
FLAGS = tuple(RULES)
COLUMNS = ["openalex_id", "publication_year", "title", "abstract", "abstract_chars"]


def normalize(text):
    return text.fillna("").str.lower().str.strip()


def id_number(openalex_id):
    return openalex_id.str.rsplit("W", n=1).str[-1].astype("int64")


def flags(papers):
    """One row per paper (same index): the four exclusion flags and ke_eligible."""
    title, abstract = normalize(papers["title"]), normalize(papers["abstract"])
    missing_title = title == ""

    # Same work twice in a year: keep the copy with the lowest OpenAlex ID.
    order = id_number(papers["openalex_id"]).sort_values(kind="stable").index
    same = pd.DataFrame({"year": papers["publication_year"], "title": title, "abstract": abstract})
    duplicate_record = same.loc[order].duplicated(["year", "title", "abstract"]).reindex(papers.index)

    # Placeholder text: one abstract under two or more different (non-missing) titles.
    titled = pd.DataFrame({"title": title, "abstract": abstract})[~missing_title]
    titles_per_abstract = titled.groupby("abstract")["title"].nunique()
    boilerplate_abstract = abstract.map(titles_per_abstract).fillna(0) >= 2

    short_abstract = papers["abstract"].fillna("").str.len() < MIN_ABSTRACT_CHARS
    out = pd.DataFrame({"missing_title": missing_title, "duplicate_record": duplicate_record,
                        "boilerplate_abstract": boilerplate_abstract,
                        "short_abstract": short_abstract}, index=papers.index).astype(bool)
    out["ke_eligible"] = ~out[list(FLAGS)].any(axis=1)
    return out


def year_report(papers, table):
    rows = []
    for year, group in table.groupby(papers["publication_year"]):
        eligible = int(group["ke_eligible"].sum())
        rows.append({"year": int(year), "pool": len(group),
                     **{flag: int(group[flag].sum()) for flag in FLAGS},
                     "excluded": len(group) - eligible, "eligible": eligible,
                     "excluded_share": round(1 - eligible / len(group), 4),
                     "pool_to_draw": round(eligible / DRAW_SIZE, 2)})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", type=Path, required=True, help="pool.parquet from src/openalex_pool.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    papers = pd.read_parquet(args.pool, columns=COLUMNS)
    if papers["openalex_id"].duplicated().any():
        raise RuntimeError(f"{args.pool} repeats an openalex_id")
    if not (papers["abstract"].fillna("").str.len() == papers["abstract_chars"]).all():
        raise RuntimeError("abstract_chars does not match the abstract length")
    table = flags(papers)
    rows = year_report(papers, table)

    args.output.mkdir(parents=True, exist_ok=True)
    result = pd.concat([papers[["openalex_id", "publication_year"]], table], axis=1)
    temporary = args.output / ".eligibility.parquet.tmp"
    result.to_parquet(temporary, index=False)
    temporary.replace(args.output / "eligibility.parquet")
    temporary = args.output / ".eligibility_report.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output / "eligibility_report.csv")
    extract.atomic_json(args.output / "metadata.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "src/ke_eligibility.py", "git_commit": openalex_counts.git_commit(),
        "pool": str(args.pool), "pool_sha256": embed_pool.file_sha256(args.pool),
        "rules": RULES, "min_abstract_chars": MIN_ABSTRACT_CHARS,
        "papers": len(result), "eligible": int(table["ke_eligible"].sum()),
        "excluded_by_flag": {flag: int(table[flag].sum()) for flag in FLAGS},
        "notes": ["Text is compared after lowercasing and trimming.",
                  "Flags overlap; a paper is excluded if any flag is set.",
                  "Decision log 2026-11-21. Supersedes theme_mix.ke_eligible (missing title or any "
                  "shared abstract), which the Week 2 pilot and eda.ipynb used."],
    })
    print(f"{int(table['ke_eligible'].sum()):,} of {len(result):,} pool papers are KE-eligible")
    for row in rows:
        print(f"{row['year']}: {row['eligible']:,} eligible ({row['excluded_share']:.1%} excluded; "
              f"{row['short_abstract']} short, {row['boilerplate_abstract']} boilerplate, "
              f"{row['duplicate_record']} duplicate records, {row['missing_title']} missing titles)")


if __name__ == "__main__":
    main()

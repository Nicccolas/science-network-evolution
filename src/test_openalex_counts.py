"""Offline checks for the yearly OpenAlex count script."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from api_test import extract
from src import openalex_counts


FIELD = {"id": "https://openalex.org/fields/32", "display_name": "Psychology",
         "domain": {"display_name": "Social Sciences"}, "works_count": 10,
         "subfields": [{"display_name": "Social Psychology"}]}


def counts_for(filter_value, corpus):
    if corpus == "expansion":
        return {2000: 5}
    counts = {2000: 12_000, 2001: 9_000}
    if "has_abstract:true" in filter_value:
        counts = {2000: 10_500, 2001: 7_200}
    if "type:article|review" in filter_value:
        counts = {year: count + 100 for year, count in counts.items()}
    if "primary_location.source.type:journal" in filter_value:
        counts = {year: count - 500 for year, count in counts.items()}
    return counts


def response(url, mismatch=False):
    if "/fields/" in url:
        return FIELD, {"remaining": None, "reset_seconds": None}
    params = parse_qs(urlsplit(url).query)
    counts = counts_for(params["filter"][0], params["corpus"][0])
    total = sum(counts.values()) + (1 if mismatch else 0)
    payload = {"meta": {"count": total, "cost_usd": 0.0001},
               "group_by": [{"key": str(y), "key_display_name": str(y), "count": c}
                            for y, c in counts.items()]}
    return payload, {"remaining": None, "reset_seconds": None}


def run(directory):
    openalex_counts.main(["--field", "32", "--min-year", "2000", "--max-year", "2002",
                          "--output", str(directory)])


class CountTests(unittest.TestCase):
    def test_table_and_metadata(self):
        with TemporaryDirectory() as directory, patch.object(extract, "fetch", side_effect=response):
            run(Path(directory))
            with open(Path(directory) / "yearly_counts.csv", newline="") as file:
                rows = {int(r["year"]): r for r in csv.DictReader(file)}
            metadata = json.loads((Path(directory) / "metadata.json").read_text())

        self.assertEqual(sorted(rows), [2000, 2001, 2002])
        self.assertEqual(rows[2000]["articles"], "12000")
        self.assertEqual(rows[2000]["articles_reviews_abstract"], "10600")
        self.assertEqual(rows[2000]["abstract_share"], "0.875")
        self.assertEqual(rows[2000]["journal_articles"], "11500")
        self.assertEqual(rows[2000]["journal_articles_abstract"], "10000")
        self.assertEqual(rows[2000]["journal_share"], "0.9583")
        self.assertEqual(rows[2000]["pool_ratio"], "1.0")
        self.assertEqual(rows[2000]["meets_10k_pool"], "True")
        self.assertEqual(rows[2001]["meets_10k_pool"], "False")
        self.assertEqual(list(rows[2000])[1:3], ["journal_articles", "journal_articles_abstract"])
        self.assertEqual(rows[2001]["articles_expansion"], "0")
        self.assertEqual(rows[2002]["articles"], "0")
        self.assertEqual(rows[2002]["abstract_share"], "")
        self.assertEqual(metadata["field"]["display_name"], "Psychology")
        self.assertEqual(metadata["requests"], 10)
        self.assertEqual(metadata["series"]["articles"]["corpus"], "core")
        self.assertEqual(metadata["series"]["articles_expansion"]["corpus"], "expansion")

    def test_mismatched_total_is_rejected_before_outputs(self):
        with TemporaryDirectory() as directory, patch.object(
                extract, "fetch", side_effect=lambda url: response(url, mismatch=True)):
            with self.assertRaisesRegex(RuntimeError, "meta.count"):
                run(Path(directory))
            self.assertFalse((Path(directory) / "yearly_counts.csv").exists())
            self.assertFalse((Path(directory) / "raw" / "articles.json").exists())

    def test_rerun_reuses_checkpoints(self):
        with TemporaryDirectory() as directory:
            with patch.object(extract, "fetch", side_effect=response):
                run(Path(directory))
            with patch.object(extract, "fetch", side_effect=AssertionError("network call")) as fetch:
                run(Path(directory))
            fetch.assert_not_called()

    def test_series_filters_use_primary_field_and_fixed_dates(self):
        filters = openalex_counts.series_filters(32, 1980, 2025)
        self.assertIn("primary_topic.field.id:32", filters["articles"][0])
        self.assertIn("topics.field.id:32", filters["any_topic_articles"][0])
        self.assertIn("primary_location.source.type:journal", filters["journal_articles"][0])
        self.assertNotIn("primary_location", filters["articles"][0])
        for filter_value, _ in filters.values():
            self.assertIn("from_publication_date:1980-01-01", filter_value)
            self.assertIn("to_publication_date:2025-12-31", filter_value)


if __name__ == "__main__":
    unittest.main()

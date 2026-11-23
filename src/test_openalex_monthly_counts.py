"""Offline checks for the monthly topic counts."""

import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pandas as pd

from api_test import extract
from src import openalex_monthly_counts as monthly


RATE = {"remaining": None, "reset_seconds": None}
# topic: (twelve monthly counts, papers dated 1 January). T2 has no papers in March.
TOPICS = {"T1": ([30, 5, 5, 6, 5, 5, 4, 5, 6, 5, 5, 9], 20),
          "T2": ([3, 1, 0, 2, 1, 1, 1, 0, 1, 2, 1, 1], 2)}


def response(url, extra=0):
    """A fake OpenAlex group_by. extra adds papers to every annual topic count."""
    filter_value = parse_qs(urlsplit(url).query)["filter"][0]
    if "publication_date:" in filter_value and "from_publication_date" not in filter_value:
        counts = {topic: jan1 for topic, (_, jan1) in TOPICS.items()}
    elif (match := re.search(r"from_publication_date:\d{4}-(\d{2})-01", filter_value)):
        counts = {topic: months[int(match.group(1)) - 1] for topic, (months, _) in TOPICS.items()}
    else:
        counts = {topic: sum(months) + extra for topic, (months, _) in TOPICS.items()}
    groups = [{"key": f"https://openalex.org/{topic}", "key_display_name": f"Topic {topic}", "count": n}
              for topic, n in counts.items() if n]  # OpenAlex leaves out empty groups
    return {"meta": {"count": sum(g["count"] for g in groups), "groups_count": len(groups),
                     "cost_usd": 0.0001}, "group_by": groups}, RATE


def run(directory, *extra):
    monthly.main(["--field", "32", "--min-year", "2004", "--max-year", "2005",
                  "--output", str(directory), *extra])


class MonthlyCountTests(unittest.TestCase):
    def test_months_add_up_and_zeros_are_filled(self):
        with TemporaryDirectory() as directory, patch.object(extract, "fetch", side_effect=response) as fetch:
            run(directory)
            months = pd.read_csv(Path(directory) / "monthly_counts.csv")
            annual = pd.read_csv(Path(directory) / "annual_counts.csv")
            metadata = json.loads((Path(directory) / "metadata.json").read_text())
            urls = [call.args[0] for call in fetch.call_args_list]

        self.assertEqual(len(urls), 2 * 14)  # annual + 12 months + 1 January, per year
        self.assertEqual(len(months), 2 * 2 * 12)
        t2_march = months.query("topic_id == 'T2' and year == 2004 and month == 3")["journal_articles"]
        self.assertEqual(t2_march.tolist(), [0])
        self.assertTrue((annual["months_sum"] == annual["journal_articles"]).all())
        self.assertEqual(annual.query("year == 2004")["jan1_articles"].tolist(), [20, 2])
        self.assertEqual(metadata["requests"], 28)
        self.assertEqual([r["gap"] for r in metadata["per_year"]], [0, 0])
        self.assertEqual(metadata["per_year"][0]["jan1_share"], round(22 / 104, 4))

        filters = [parse_qs(urlsplit(url).query)["filter"][0] for url in urls]
        self.assertIn("from_publication_date:2004-02-01,to_publication_date:2004-02-29", filters[2])  # leap year
        self.assertIn("to_publication_date:2005-02-28", filters[14 + 2])
        self.assertTrue(filters[13].endswith("publication_date:2004-01-01"))
        for filter_value in filters:  # the N_kt frame: every journal article, abstract or not
            self.assertIn("primary_topic.field.id:32,type:article", filter_value)
            self.assertIn("primary_location.source.type:journal", filter_value)
            self.assertNotIn("has_abstract", filter_value)

    def test_rerun_uses_checkpoints(self):
        with TemporaryDirectory() as directory:
            with patch.object(extract, "fetch", side_effect=response):
                run(directory)
            with patch.object(extract, "fetch", side_effect=AssertionError("network call")) as fetch:
                run(directory)
            fetch.assert_not_called()

    def test_months_that_miss_the_annual_count_stop_the_run(self):
        with TemporaryDirectory() as directory, patch.object(
                extract, "fetch", side_effect=lambda url: response(url, extra=1)):
            with self.assertRaisesRegex(RuntimeError, "do not add up to the annual count in 2004, 2005"):
                run(directory)  # 2 of 104 papers missing from the months: a 1.9% gap
            self.assertFalse((Path(directory) / "monthly_counts.csv").exists())
            self.assertTrue((Path(directory) / "raw" / "year=2004" / "month=12.json").exists())
        with TemporaryDirectory() as directory, patch.object(
                extract, "fetch", side_effect=lambda url: response(url, extra=1)):
            run(directory, "--max-gap", "0.02")
            annual = pd.read_csv(Path(directory) / "annual_counts.csv")
        self.assertEqual((annual["journal_articles"] - annual["months_sum"]).tolist(), [1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()

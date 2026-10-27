"""Offline checks for the per-year pool download script."""

import json
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pandas as pd

from api_test import extract
from src import openalex_pool


FRAME = 600  # eligible works per year in the fake API
RATE = {"remaining": None, "reset_seconds": None}


def fake_work(year, i):
    words = ["Short."] if i % 25 == 0 else ["This", "abstract", "is", "long", "enough", f"paper{i}"]
    return {"id": f"https://openalex.org/W{year}{i:04d}", "doi": None, "title": f"Paper {i}",
            "abstract_inverted_index": {word: [k] for k, word in enumerate(words)},
            "publication_year": year, "publication_date": f"{year}-01-01", "type": "article",
            "language": "en", "is_paratext": False, "cited_by_count": i,
            "primary_topic": {"id": "T1", "display_name": "Topic", "score": 0.9,
                              "subfield": {"id": "S1", "display_name": "Sub"},
                              "field": {"id": "https://openalex.org/fields/32", "display_name": "Psychology"}},
            "primary_location": {"source": {"id": "S", "display_name": "J", "issn_l": "0000-0000",
                                            "type": "journal"}}}


def response(url, frame=FRAME, corrupt=False):
    params = parse_qs(urlsplit(url).query)
    year = int(re.search(r"publication_year:(\d+)", params["filter"][0]).group(1))
    if "sample" not in params:
        return {"meta": {"count": frame, "cost_usd": 0.0001}, "results": []}, RATE
    size, seed, page = (int(params[key][0]) for key in ("sample", "seed", "page"))
    chosen = random.Random(f"{year}-{seed}").sample(range(frame), size)
    results = [fake_work(year, i) for i in chosen[(page - 1) * 100: page * 100]]
    if corrupt:
        results[0]["publication_year"] = year - 1
    return {"meta": {"count": size, "cost_usd": 0.0001}, "results": results}, RATE


def run(directory, *extra):
    openalex_pool.main(["--field", "32", "--min-year", "2001", "--max-year", "2002",
                        "--pool-size", "250", "--output", str(directory), *extra])


def read_pool(directory):
    return pd.read_parquet(Path(directory) / "pool.parquet")


@patch.object(openalex_pool, "TOPUP_SIZE", 200)
class PoolTests(unittest.TestCase):
    def test_default_keeps_exactly_the_seed_one_sample(self):
        with TemporaryDirectory() as directory, patch.object(extract, "fetch", side_effect=response) as fetch:
            run(directory)
            pool = read_pool(directory)
            metadata = json.loads((Path(directory) / "metadata.json").read_text())
            urls = [call.args[0] for call in fetch.call_args_list]

        self.assertEqual(pool.groupby("publication_year").size().to_dict(), {2001: 250, 2002: 250})
        self.assertTrue(pool["openalex_id"].is_unique)
        self.assertEqual(set(pool["source_seed"]), {1})
        self.assertIn(6, set(pool["abstract_chars"]))  # short abstracts are kept by default
        expected = [f"https://openalex.org/W2001{i:04d}"
                    for i in random.Random("2001-1").sample(range(FRAME), 250)]
        self.assertEqual(pool.loc[pool["publication_year"] == 2001, "openalex_id"].tolist(), expected)
        self.assertFalse(any("seed=2" in url for url in urls))
        self.assertEqual(len(urls), 2 * (1 + 3))  # count + 3 pages per year
        self.assertEqual(metadata["rows"], 500)
        self.assertEqual(metadata["min_abstract_chars"], 0)
        self.assertFalse(metadata["topup"]["used"])
        for params in (parse_qs(urlsplit(url).query) for url in urls):
            self.assertIn("primary_location.source.type:journal", params["filter"][0])
            self.assertIn("has_abstract:true", params["filter"][0])
            self.assertEqual(params["corpus"], ["core"])

    def test_min_abstract_chars_tops_up_reproducibly(self):
        pools = []
        for _ in range(2):
            with TemporaryDirectory() as directory, patch.object(extract, "fetch", side_effect=response):
                run(directory, "--min-abstract-chars", "20")
                pools.append(read_pool(directory))
                report = pd.read_csv(Path(directory) / "pool_report.csv")
        pool = pools[0]
        self.assertEqual(pool.groupby("publication_year").size().to_dict(), {2001: 250, 2002: 250})
        self.assertTrue(pool["openalex_id"].is_unique)
        self.assertTrue((pool["abstract_chars"] >= 20).all())
        self.assertIn(2, set(pool["source_seed"]))
        self.assertTrue((report["sampled"] - report["dropped_short"] + report["topped_up"]
                         == report["final"]).all())
        self.assertTrue((report["dropped_short"] > 0).all())
        self.assertEqual(pools[0]["openalex_id"].tolist(), pools[1]["openalex_id"].tolist())

    def test_one_year_run_matches_the_same_year_in_a_longer_run(self):
        with TemporaryDirectory() as one, TemporaryDirectory() as two, \
                patch.object(extract, "fetch", side_effect=response):
            openalex_pool.main(["--field", "32", "--min-year", "2002", "--max-year", "2002",
                                "--pool-size", "250", "--min-abstract-chars", "20", "--output", one])
            run(two, "--min-abstract-chars", "20")
            alone = read_pool(one)["openalex_id"].tolist()
            within = read_pool(two).query("publication_year == 2002")["openalex_id"].tolist()
        self.assertEqual(alone, within)

    def test_frame_smaller_than_pool_is_an_error(self):
        with TemporaryDirectory() as directory, patch.object(
                extract, "fetch", side_effect=lambda url: response(url, frame=200)):
            with self.assertRaisesRegex(RuntimeError, "only 200 eligible"):
                run(directory)
            self.assertFalse((Path(directory) / "pool.parquet").exists())

    def test_work_outside_frame_is_rejected_before_checkpoint(self):
        with TemporaryDirectory() as directory, patch.object(
                extract, "fetch", side_effect=lambda url: response(url, corrupt=True)):
            with self.assertRaisesRegex(RuntimeError, "outside the filtered frame"):
                run(directory)
            self.assertFalse((Path(directory) / "raw" / "year=2001" / "seed=1" / "page_0001.json").exists())

    def test_resume_after_daily_budget_and_rerun_without_network(self):
        calls = {"n": 0}

        def budget_runs_out(url):
            calls["n"] += 1
            if calls["n"] > 5:
                raise extract.DailyBudgetExceeded(None)
            return response(url)

        with TemporaryDirectory() as directory:
            with patch.object(extract, "fetch", side_effect=budget_runs_out):
                with self.assertRaises(extract.DailyBudgetExceeded):
                    run(directory)
            with patch.object(extract, "fetch", side_effect=response) as fetch:
                run(directory)
            self.assertEqual(fetch.call_count, 8 - 5)
            with patch.object(extract, "fetch", side_effect=AssertionError("network call")) as fetch:
                run(directory)
            fetch.assert_not_called()
            self.assertEqual(len(read_pool(directory)), 500)


if __name__ == "__main__":
    unittest.main()

"""Offline checks for the topic-stratified sample download script."""

import gzip
import json
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pandas as pd

from api_test import extract
from src import embed_pool, openalex_topic_sample as topic_sample


RATE = {"remaining": None, "reset_seconds": None}
# topic: (all journal articles, those with abstracts). With min_papers = 100 and cap = 300:
# T1 is sampled at the cap, T2 in full, T3 is below the threshold, and T4 has no abstracts.
TOPICS = {"T1": (500, 450), "T2": (260, 150), "T3": (120, 80), "T4": (30, 0)}
PLENTY = SimpleNamespace(free=10 ** 12)


def fake_work(year, topic, i):
    return {"id": f"https://openalex.org/W{year}{topic_sample.topic_number(topic):02d}{i:04d}",
            "doi": None, "title": f"Paper {i}",
            "abstract_inverted_index": {"An": [0], "abstract": [1], f"w{i}": [2]},
            "publication_year": year, "publication_date": f"{year}-01-01", "type": "article",
            "language": "en", "is_paratext": False, "cited_by_count": i,
            "primary_topic": {"id": f"https://openalex.org/{topic}", "display_name": f"Topic {topic}",
                              "score": 0.9, "subfield": {"id": "S1", "display_name": "Sub"},
                              "field": {"id": "https://openalex.org/fields/32", "display_name": "Psychology"}},
            "primary_location": {"source": {
                "id": "S", "display_name": "J", "issn_l": "0000-0000", "type": "journal",
                "host_organization": "https://openalex.org/P1", "host_organization_name": "Publisher",
                "host_organization_lineage_names": ["Publisher", "Parent"]}}}


def chosen(year, topic, seed, size):
    return random.Random(f"{topic}-{year}-{seed}").sample(range(TOPICS[topic][1]), size)


def response(url, wrong_topic=False, drift=None, bad_sum=False):
    params = parse_qs(urlsplit(url).query)
    filter_value = params["filter"][0]
    year = int(re.search(r"publication_year:(\d+)", filter_value).group(1))
    abstracts = "has_abstract:true" in filter_value
    if "group_by" in params:
        groups = [{"key": f"https://openalex.org/{topic}", "key_display_name": f"Topic {topic}",
                   "count": counts[1] if abstracts else counts[0]}
                  for topic, counts in TOPICS.items() if (counts[1] if abstracts else counts[0])]
        total = sum(group["count"] for group in groups) + (1 if bad_sum else 0)
        return {"meta": {"count": total, "groups_count": len(groups), "cost_usd": 0.0001},
                "group_by": groups}, RATE
    topic = re.search(r"primary_topic\.id:(T\d+)", filter_value).group(1)
    size, seed, page = (int(params[key][0]) for key in ("sample", "seed", "page"))
    results = [fake_work(year, topic, i) for i in chosen(year, topic, seed, size)[(page - 1) * 100: page * 100]]
    if wrong_topic:
        results[0]["primary_topic"]["id"] = "https://openalex.org/T9"
    count = size - 1 if topic == drift else size
    return {"meta": {"count": count, "cost_usd": 0.0001}, "results": results}, RATE


def run(directory, *extra):
    topic_sample.main(["--field", "32", "--min-year", "2001", "--max-year", "2002",
                       "--output", str(directory), *extra])


def read(directory, name):
    path = Path(directory) / name
    return pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path)


@patch.object(topic_sample.shutil, "disk_usage", return_value=PLENTY)
class TopicSampleTests(unittest.TestCase):
    def test_full_run_samples_each_panel_topic_year(self, _disk):
        with TemporaryDirectory() as directory, patch.object(extract, "fetch", side_effect=response) as fetch:
            run(directory)
            sample = read(directory, "sample.parquet")
            counts = read(directory, "topic_counts.csv")
            to_embed = read(directory, "to_embed.parquet")
            metadata = json.loads((Path(directory) / "metadata.json").read_text())
            urls = [call.args[0] for call in fetch.call_args_list]
            page = Path(directory) / "raw" / "year=2001" / "topic=T1" / "seed=1" / "page_0001.json.gz"
            with gzip.open(page, "rt", encoding="utf-8") as file:
                self.assertEqual(len(json.load(file)["payload"]["results"]), 100)

        sizes = sample.groupby(["publication_year", "primary_topic_id"]).size().to_dict()
        self.assertEqual(sizes, {(year, f"https://openalex.org/{topic}"): size
                                 for year in (2001, 2002) for topic, size in (("T1", 300), ("T2", 150))})
        self.assertTrue(sample["openalex_id"].is_unique)
        expected = [f"https://openalex.org/W200101{i:04d}" for i in chosen(2001, "T1", 1, 300)]
        self.assertEqual(sample.query("publication_year == 2001 and primary_topic_id.str.endswith('T1')",
                                      engine="python")["openalex_id"].tolist(), expected)
        self.assertEqual(set(sample["source_host_organization_name"]), {"Publisher"})
        self.assertEqual(set(sample["source_host_organization_lineage_names"]), {"Publisher; Parent"})
        self.assertTrue(sample["in_pool"].isna().all())
        self.assertEqual(to_embed["openalex_id"].tolist(), sample["openalex_id"].tolist())  # no pool given

        row = counts.query("year == 2001 and topic_id == 'T1'").iloc[0]
        self.assertEqual((row["journal_articles"], row["journal_articles_abstract"], row["sample_size"]),
                         (500, 450, 300))
        self.assertAlmostEqual(row["abstract_share"], 0.9)
        self.assertEqual(counts.query("in_panel")["topic_id"].tolist(), ["T1", "T2", "T1", "T2"])
        t4 = counts.query("year == 2001 and topic_id == 'T4'").iloc[0]
        self.assertEqual((t4["journal_articles_abstract"], t4["sample_size"], bool(t4["in_panel"])), (0, 0, False))

        self.assertEqual(len(urls), 2 * (2 + 3 + 2))  # 2 count requests + 5 sample pages per year
        self.assertEqual(metadata["rows"], 900)
        self.assertEqual(metadata["topic_years_in_panel"], 4)
        for params in (parse_qs(urlsplit(url).query) for url in urls):
            filter_value = params["filter"][0]
            self.assertIn("primary_location.source.type:journal", filter_value)
            self.assertIn("type:article", filter_value)
            self.assertEqual(params["corpus"], ["core"])
            if "sample" in params:
                self.assertIn("has_abstract:true", filter_value)
                self.assertIn("primary_topic.id:T", filter_value)
        all_count_urls = [url for url in urls if "group_by" in url and "has_abstract" not in url]
        self.assertEqual(len(all_count_urls), 2)  # N_kt counts every journal article

    def test_in_pool_marks_papers_already_in_the_pool(self, _disk):
        in_both = [f"https://openalex.org/W200101{i:04d}" for i in chosen(2001, "T1", 1, 300)[:10]]
        with TemporaryDirectory() as directory, patch.object(extract, "fetch", side_effect=response):
            pool = Path(directory) / "pool.parquet"
            pd.DataFrame({"openalex_id": in_both + ["https://openalex.org/W1"]}).to_parquet(pool)
            run(Path(directory) / "out", "--pool", str(pool))
            sample = read(Path(directory) / "out", "sample.parquet")
            report = read(Path(directory) / "out", "sample_report.csv")
            to_embed = read(Path(directory) / "out", "to_embed.parquet")
        self.assertEqual(sorted(sample.loc[sample["in_pool"], "openalex_id"]), sorted(in_both))
        self.assertEqual(report["in_pool"].tolist(), [10, 0])
        # Only papers without pool vectors are left to embed, in sample order, as embed_pool reads them.
        self.assertEqual(list(to_embed), embed_pool.POOL_COLUMNS)
        self.assertEqual(to_embed["openalex_id"].tolist(),
                         sample.loc[~sample["in_pool"], "openalex_id"].tolist())
        self.assertEqual(len(to_embed), 900 - 10)

    def test_counts_only_pulls_counts_and_a_later_run_reuses_them(self, _disk):
        with TemporaryDirectory() as directory:
            with patch.object(extract, "fetch", side_effect=response) as fetch:
                run(directory, "--counts-only")
                self.assertTrue(all("group_by" in call.args[0] for call in fetch.call_args_list))
                self.assertEqual(fetch.call_count, 4)
            self.assertFalse((Path(directory) / "sample.parquet").exists())
            self.assertEqual(read(directory, "sample_report.csv")["planned_sample"].tolist(), [450, 450])
            self.assertTrue(json.loads((Path(directory) / "metadata.json").read_text())["counts_only"])
            with patch.object(extract, "fetch", side_effect=response) as fetch:
                run(directory)
                self.assertFalse(any("group_by" in call.args[0] for call in fetch.call_args_list))
                self.assertEqual(fetch.call_count, 10)
            self.assertEqual(len(read(directory, "sample.parquet")), 900)

    def test_resume_after_daily_budget_and_rerun_without_network(self, _disk):
        calls = {"n": 0}

        def budget_runs_out(url):
            calls["n"] += 1
            if calls["n"] > 6:
                raise extract.DailyBudgetExceeded(None)
            return response(url)

        with TemporaryDirectory() as directory:
            with patch.object(extract, "fetch", side_effect=budget_runs_out):
                with self.assertRaises(extract.DailyBudgetExceeded):
                    run(directory)
            with patch.object(extract, "fetch", side_effect=response) as fetch:
                run(directory)
            self.assertEqual(fetch.call_count, 14 - 6)
            with patch.object(extract, "fetch", side_effect=AssertionError("network call")) as fetch:
                run(directory)
            fetch.assert_not_called()
            self.assertEqual(len(read(directory, "sample.parquet")), 900)

    def test_work_from_another_topic_is_rejected_before_checkpoint(self, _disk):
        with TemporaryDirectory() as directory, patch.object(
                extract, "fetch", side_effect=lambda url: response(url, wrong_topic=True)):
            with self.assertRaisesRegex(RuntimeError, "another topic"):
                run(directory)
            self.assertFalse(any((Path(directory) / "raw").rglob("page_*.json.gz")))

    def test_sample_smaller_than_the_count_is_an_error(self, _disk):
        with TemporaryDirectory() as directory, patch.object(
                extract, "fetch", side_effect=lambda url: response(url, drift="T2")):
            with self.assertRaisesRegex(RuntimeError, "changed since the counts were pulled"):
                run(directory)

    def test_topic_groups_must_sum_to_the_total(self, _disk):
        with TemporaryDirectory() as directory, patch.object(
                extract, "fetch", side_effect=lambda url: response(url, bad_sum=True)):
            with self.assertRaisesRegex(RuntimeError, "Topic groups sum to"):
                run(directory)

    def test_full_disk_stops_before_any_sample_request(self, disk):
        disk.return_value = SimpleNamespace(free=10 ** 9)
        with TemporaryDirectory() as directory, patch.object(extract, "fetch", side_effect=response) as fetch:
            with self.assertRaisesRegex(RuntimeError, "Free some space first"):
                run(directory)
            self.assertTrue(all("group_by" in call.args[0] for call in fetch.call_args_list))
            self.assertTrue((Path(directory) / "topic_counts.csv").exists())


if __name__ == "__main__":
    unittest.main()

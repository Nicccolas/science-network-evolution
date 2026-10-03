"""Behavior checks for multi-page sampling and safe output publication."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from api_test import extract


def work(year, number, field=32):
    references = []
    if year == 2023 and number == 0:
        references = ["https://openalex.org/W2024000", "https://openalex.org/W9999999"]
    return {
        "id": f"https://openalex.org/W{year}{number:03d}",
        "title": f"Work {year}-{number}", "publication_year": year,
        "type": "article", "language": "en", "referenced_works": references,
        "primary_topic": None if field is None else {
            "id": "https://openalex.org/T10001", "display_name": "Example topic",
            "score": 0.95,
            "subfield": {"id": "https://openalex.org/subfields/3201",
                         "display_name": "Example subfield"},
            "field": {"id": f"https://openalex.org/fields/{field}",
                      "display_name": "Psychology"},
            "domain": {"id": "https://openalex.org/domains/2",
                       "display_name": "Social Sciences"},
        },
        "abstract_inverted_index": {"Hello": [0], "world": [1]},
        "authorships": [],
    }


def response(url, duplicate=False):
    params = parse_qs(urlsplit(url).query)
    year = int(params["filter"][0].split("publication_year:")[1].split(",")[0])
    population = 101 if year == 2023 else 2
    if "sample" not in params:
        return {"meta": {"count": population, "cost_usd": 0.0001},
                "results": [{"id": "unused"}]}, {"remaining": 9999, "reset_seconds": 100}
    page = int(params["page"][0])
    sampled = min(int(params["sample"][0]), population)
    start = (page - 1) * 100
    values = [work(year, number) for number in range(start, min(start + 100, sampled))]
    if duplicate and page == 2:
        values[0] = work(year, 0)
    return {"meta": {"count": sampled, "cost_usd": 0.0001},
            "results": values}, {"remaining": 9999, "reset_seconds": 100}


class ExtractTests(unittest.TestCase):
    def test_balanced_pages_and_network(self):
        with TemporaryDirectory() as directory, patch.object(extract, "fetch", side_effect=response):
            output = Path(directory)
            result = extract.main(["--min-year", "2023", "--max-year", "2024",
                                   "--size", "101", "--fields", "(30, 31, 32)",
                                   "--output", str(output)])
            self.assertEqual(result, 0)
            with (output / "works.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            with (output / "citation_edges.csv").open(newline="") as file:
                edges = list(csv.DictReader(file))
            with (output / "citation_nodes.csv").open(newline="") as file:
                nodes = list(csv.DictReader(file))
            metadata = json.loads((output / "metadata.json").read_text())
            raw = json.loads((output / "works_raw.json").read_text())
            self.assertEqual(len(rows), len(raw["results"]))
            self.assertEqual(len(rows), 103)
            self.assertEqual({r["publication_year"] for r in rows}, {"2023", "2024"})
            self.assertTrue(all(r["abstract"] == "Hello world" for r in rows))
            self.assertEqual(rows[0]["primary_topic_score"], "0.95")
            self.assertEqual(rows[0]["primary_subfield"], "Example subfield")
            self.assertEqual(rows[0]["primary_subfield_id"],
                             "https://openalex.org/subfields/3201")
            self.assertEqual(rows[0]["primary_field"], "Psychology")
            self.assertEqual(rows[0]["primary_field_id"],
                             "https://openalex.org/fields/32")
            self.assertEqual(rows[0]["primary_domain"], "Social Sciences")
            self.assertEqual(rows[0]["primary_domain_id"],
                             "https://openalex.org/domains/2")
            self.assertEqual([(x["year"], x["sample_count"]) for x in metadata["per_year"]],
                             [(2023, 101), (2024, 2)])
            self.assertEqual(len(edges), 2)
            self.assertEqual(len(nodes), 104)
            self.assertEqual({(edge["Source"], edge["Target"]) for edge in edges}, {
                ("https://openalex.org/W2023000", "https://openalex.org/W2024000"),
                ("https://openalex.org/W2023000", "https://openalex.org/W9999999"),
            })
            self.assertEqual(metadata["citation_network"]["edges_within_sample"], 1)
            self.assertEqual({node["Id"] for node in nodes},
                             {row["openalex_id"] for row in rows} | {"https://openalex.org/W9999999"})
            self.assertFalse((output / ".extract-checkpoint").exists())

    def test_budget_stop_resumes_without_repeating_pages(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "works.csv").write_text("previous data\n")
            calls = []

            def interrupted(url):
                calls.append(url)
                if "page=2" in url:
                    raise extract.DailyBudgetExceeded(120)
                return response(url)

            args = ["--year", "2023", "--size", "101", "--output", str(output)]
            with patch.object(extract, "fetch", side_effect=interrupted):
                self.assertEqual(extract.main(args), 2)
            self.assertEqual((output / "works.csv").read_text(), "previous data\n")
            self.assertTrue((output / ".extract-checkpoint/year=2023/count.json").exists())
            self.assertTrue((output / ".extract-checkpoint/year=2023/page_0001.json").exists())
            with patch.object(extract, "fetch", side_effect=response) as resumed:
                self.assertEqual(extract.main(args), 0)
            self.assertEqual(resumed.call_count, 1)
            self.assertIn("page=2", resumed.call_args.args[0])
            self.assertIn("openalex_id", (output / "works.csv").read_text())

    def test_duplicate_page_does_not_replace_outputs(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "works.csv").write_text("previous data\n")
            with patch.object(extract, "fetch", side_effect=lambda url: response(url, duplicate=True)):
                with self.assertRaisesRegex(RuntimeError, "Duplicate work ID"):
                    extract.main(["--year", "2023", "--size", "101",
                                  "--output", str(output)])
            self.assertEqual((output / "works.csv").read_text(), "previous data\n")
            self.assertTrue((output / ".extract-checkpoint").exists())

    def test_zero_matching_year_writes_empty_outputs(self):
        def empty_response(url):
            self.assertNotIn("sample=", url)
            return {"meta": {"count": 0, "cost_usd": 0.0001}, "results": []}, {
                "remaining": 9999, "reset_seconds": 100,
            }

        with TemporaryDirectory() as directory, patch.object(extract, "fetch", side_effect=empty_response):
            output = Path(directory)
            self.assertEqual(extract.main(["--year", "1900", "--output", str(output)]), 0)
            with (output / "works.csv").open(newline="") as file:
                self.assertEqual(list(csv.DictReader(file)), [])
            with (output / "citation_edges.csv").open(newline="") as file:
                self.assertEqual(list(csv.DictReader(file)), [])
            self.assertEqual(json.loads((output / "metadata.json").read_text())["returned_rows"], 0)
            self.assertFalse((output / ".extract-checkpoint").exists())

    def test_missing_primary_topic_is_retained_with_blank_taxonomy(self):
        def unclassified_response(url):
            params = parse_qs(urlsplit(url).query)
            if "sample" not in params:
                return {"meta": {"count": 1, "cost_usd": 0.0001}, "results": []}, {
                    "remaining": 9999, "reset_seconds": 100,
                }
            return {"meta": {"count": 1, "cost_usd": 0.0001},
                    "results": [work(2024, 0, field=None)]}, {
                        "remaining": 9999, "reset_seconds": 100,
                    }

        with TemporaryDirectory() as directory, patch.object(
                extract, "fetch", side_effect=unclassified_response):
            output = Path(directory)
            self.assertEqual(extract.main(["--year", "2024", "--size", "1",
                                           "--output", str(output)]), 0)
            with (output / "works.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            taxonomy_columns = [
                "primary_topic_score", "primary_subfield", "primary_subfield_id",
                "primary_field", "primary_field_id", "primary_domain", "primary_domain_id",
            ]
            self.assertTrue(all(rows[0][column] == "" for column in taxonomy_columns))
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertTrue(all(metadata["missing_values_by_column"][column] == 1
                                for column in taxonomy_columns))

    def test_field_list_and_range_validation(self):
        self.assertEqual(extract.parse_fields("(30, 31, 32, 31)"), ("30", "31", "32"))
        with self.assertRaises(ValueError):
            extract.parse_fields("30,,32")
        with self.assertRaises(SystemExit):
            extract.main(["--min-year", "2025", "--max-year", "2024"])


if __name__ == "__main__":
    unittest.main()

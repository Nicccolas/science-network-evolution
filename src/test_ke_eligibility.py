"""Offline checks for the KE exclusion rules (decision log, 2026-11-21)."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src import ke_eligibility


LONG = "A real abstract that is long enough to pass the length rule. " * 5  # 305 characters


def papers():
    rows = [
        # id, year, title, abstract
        ("W10", 2001, "Kept", LONG + "one"),
        ("W30", 2001, "Same work", LONG + "two"),           # duplicate record, higher ID: dropped
        ("W20", 2001, "same work ", " " + LONG + "two"),     # its copy with the lowest ID: kept
        ("W40", 2002, "Same work", LONG + "two"),            # same text in another year: kept
        ("W50", 2001, "Paper A", LONG + "placeholder"),      # boilerplate: two different titles
        ("W60", 2002, "Paper B", LONG + "placeholder"),
        ("W70", 2001, None, LONG + "three"),                 # missing title
        ("W80", 2001, "Also kept", LONG + "four"),
        ("W90", 2002, "", LONG + "four"),                    # missing title; not boilerplate for W80
        ("W95", 2002, "Short", "N/A"),                       # short abstract, a single copy
    ]
    frame = pd.DataFrame(rows, columns=["openalex_id", "publication_year", "title", "abstract"])
    frame["openalex_id"] = "https://openalex.org/" + frame["openalex_id"]
    frame["abstract_chars"] = frame["abstract"].str.len()
    return frame


class EligibilityTests(unittest.TestCase):
    def test_each_rule(self):
        table = ke_eligibility.flags(papers()).set_axis(papers()["openalex_id"].str[21:])
        expected = {
            "missing_title": {"W70", "W90"},
            "duplicate_record": {"W30"},
            "boilerplate_abstract": {"W50", "W60"},
            "short_abstract": {"W95"},
        }
        for flag, ids in expected.items():
            self.assertEqual(set(table.index[table[flag]]), ids, flag)
        self.assertEqual(set(table.index[table["ke_eligible"]]), {"W10", "W20", "W40", "W80"})

    def test_lowest_id_is_kept_whatever_the_row_order(self):
        shuffled = papers().sample(frac=1, random_state=3)
        table = ke_eligibility.flags(shuffled)
        kept = shuffled.loc[table["ke_eligible"], "openalex_id"].str[21:]
        self.assertIn("W20", set(kept))
        self.assertNotIn("W30", set(kept))

    def test_main_writes_flags_report_and_metadata(self):
        with TemporaryDirectory() as directory:
            pool = Path(directory) / "pool.parquet"
            papers().to_parquet(pool, index=False)
            ke_eligibility.main(["--pool", str(pool), "--output", str(Path(directory) / "out")])
            result = pd.read_parquet(Path(directory) / "out" / "eligibility.parquet")
            report = pd.read_csv(Path(directory) / "out" / "eligibility_report.csv")
            metadata = json.loads((Path(directory) / "out" / "metadata.json").read_text())
        self.assertEqual(result["openalex_id"].tolist(), papers()["openalex_id"].tolist())
        self.assertEqual(report.set_index("year")["eligible"].to_dict(), {2001: 3, 2002: 1})
        self.assertEqual(report["pool"].sum(), 10)
        self.assertEqual(metadata["eligible"], 4)
        self.assertEqual(set(metadata["rules"]), set(ke_eligibility.FLAGS))


if __name__ == "__main__":
    unittest.main()

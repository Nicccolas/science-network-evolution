"""Offline checks for the pool embedding run (fake model and tokenizer, no download)."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import embed_pool, embeddings
from src.test_embeddings import FakeModel, FakeTokenizer


class SepTokenizer(FakeTokenizer):
    sep_token = " [SEP] "


INFO = {"name": "fake", "repo": "fake/base", "revision": "a" * 40, "adapter": "fake/adapter",
        "adapter_revision": "b" * 40, "device": "cpu", "max_tokens": embeddings.MAX_TOKENS,
        "input": "title + sep_token + abstract", "output": "CLS token, raw (not normalized)"}


def write_pool(path, years=(2001, 2002), per_year=5):
    rows = []
    for year in years:
        for i in range(per_year):
            title = None if i == 0 else f"Paper {i}"
            abstract = " ".join(["word"] * (embeddings.MAX_TOKENS + 10 if i == 1 else i + 1))
            rows.append({"openalex_id": f"https://openalex.org/W{year}{i}", "publication_year": year,
                         "title": title, "abstract": abstract, "sample_position": i})
    pd.DataFrame(rows).to_parquet(path, index=False)
    return pd.DataFrame(rows)


class EmbedPoolTests(unittest.TestCase):
    def run_main(self, pool, output, info=INFO, extra=()):
        with patch.object(embeddings, "load", return_value=(FakeModel(), SepTokenizer(), dict(info))):
            embed_pool.main(["--pool", str(pool), "--output", str(output), *extra])

    def test_writes_aligned_vectors_ids_and_report(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pool = write_pool(tmp / "pool.parquet")
            self.run_main(tmp / "pool.parquet", tmp / "out")
            for year in (2001, 2002):
                papers = pool[pool["publication_year"] == year]
                vectors = np.load(tmp / "out" / f"vectors_{year}.npy")
                ids = pd.read_parquet(tmp / "out" / f"ids_{year}.parquet")
                self.assertEqual(vectors.dtype, np.float32)
                self.assertEqual(vectors.shape, (5, FakeModel.config.hidden_size))
                self.assertEqual(ids["openalex_id"].tolist(), papers["openalex_id"].tolist())
                # FakeModel's vector is the truncated token count, so rows line up with n_tokens.
                expected = np.minimum(ids["n_tokens"], embeddings.MAX_TOKENS)
                self.assertEqual(vectors[:, 0].tolist(), expected.tolist())
            report = pd.read_csv(tmp / "out" / "embedding_report.csv")
            self.assertEqual(report["year"].tolist(), [2001, 2002])
            self.assertEqual(report["missing_titles"].tolist(), [1, 1])
            self.assertEqual(report["over_max_tokens"].tolist(), [1, 1])
            metadata = json.loads((tmp / "out" / "metadata.json").read_text())
            self.assertEqual(metadata["config"]["model"]["revision"], "a" * 40)
            self.assertEqual(len(metadata["config"]["pool_sha256"]), 64)
            self.assertEqual(list(tmp.joinpath("out").glob(".*")), [])  # no temporary files left

    def test_rerun_skips_finished_years_and_resumes_the_rest(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_pool(tmp / "pool.parquet")
            self.run_main(tmp / "pool.parquet", tmp / "out", extra=["--max-year", "2001"])
            self.assertFalse((tmp / "out" / "vectors_2002.npy").exists())
            with patch.object(embed_pool, "embed_year", wraps=embed_pool.embed_year) as embed:
                self.run_main(tmp / "pool.parquet", tmp / "out")
            self.assertEqual(embed.call_count, 1)  # only 2002
            report = pd.read_csv(tmp / "out" / "embedding_report.csv")
            self.assertEqual(report["year"].tolist(), [2001, 2002])

    def test_year_without_its_ids_file_is_embedded_again(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_pool(tmp / "pool.parquet")
            self.run_main(tmp / "pool.parquet", tmp / "out")
            (tmp / "out" / "ids_2002.parquet").unlink()
            with patch.object(embed_pool, "embed_year", wraps=embed_pool.embed_year) as embed:
                self.run_main(tmp / "pool.parquet", tmp / "out")
            self.assertEqual(embed.call_count, 1)
            self.assertTrue((tmp / "out" / "ids_2002.parquet").exists())

    def test_rerun_with_a_different_setup_stops(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_pool(tmp / "pool.parquet")
            self.run_main(tmp / "pool.parquet", tmp / "out", extra=["--max-year", "2001"])
            with self.assertRaisesRegex(RuntimeError, "model"):
                self.run_main(tmp / "pool.parquet", tmp / "out", info={**INFO, "revision": "c" * 40})
            with self.assertRaisesRegex(RuntimeError, "batch_size"):
                self.run_main(tmp / "pool.parquet", tmp / "out", extra=["--batch-size", "8"])
            write_pool(tmp / "pool.parquet", per_year=6)
            with self.assertRaisesRegex(RuntimeError, "pool_sha256"):
                self.run_main(tmp / "pool.parquet", tmp / "out")

    def test_missing_revision_stops_before_writing(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_pool(tmp / "pool.parquet")
            with self.assertRaisesRegex(RuntimeError, "revision"):
                self.run_main(tmp / "pool.parquet", tmp / "out", info={**INFO, "revision": None})
            self.assertFalse((tmp / "out").exists())

    def test_year_range_outside_the_pool_is_rejected(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_pool(tmp / "pool.parquet")
            with self.assertRaises(SystemExit):
                self.run_main(tmp / "pool.parquet", tmp / "out", extra=["--max-year", "2003"])


if __name__ == "__main__":
    unittest.main()

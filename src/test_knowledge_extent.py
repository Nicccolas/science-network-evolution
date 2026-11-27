"""Offline checks for the annual KE computation."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src import knowledge_extent as ke


def fake_embeddings(directory, years=(2001, 2002), n=60, dim=4):
    rng = np.random.default_rng(0)
    rows = []
    for year in years:
        ids = [f"https://openalex.org/W{year}{i:03d}" for i in range(n)]
        np.save(Path(directory) / f"vectors_{year}.npy", rng.normal(size=(n, dim)).astype(np.float32))
        pd.DataFrame({"openalex_id": ids, "n_tokens": 10}).to_parquet(Path(directory) / f"ids_{year}.parquet")
        rows += [{"openalex_id": i, "publication_year": year, "ke_eligible": k % 10 != 0}
                 for k, i in enumerate(ids)]
    (Path(directory) / "metadata.json").write_text(json.dumps({"updated_at_utc": "t"}))
    return pd.DataFrame(rows)


class DistanceTests(unittest.TestCase):
    def test_max_and_p95_on_a_known_sample(self):
        X = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]])
        ke_max, ke_p95, farthest = ke.distance_stats(X, np.array([[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]]))
        np.testing.assert_allclose(ke_max, [2.0, 2.0])  # centroid is the origin
        np.testing.assert_allclose(ke_p95, [2.0, 2.0])
        self.assertIn(farthest[0], {1, 2, 3, 4})

    def test_centroid_is_the_draws_own_mean(self):
        X = np.array([[10.0], [12.0], [100.0], [0.0]])
        ke_max, _, farthest = ke.distance_stats(X, np.array([[0, 1]]))
        self.assertAlmostEqual(ke_max[0], 1.0)  # around 11, ignoring rows not drawn
        self.assertIn(farthest[0], {0, 1})

    def test_l2_normalize(self):
        X = np.array([[3.0, 4.0], [0.0, 2.0]])
        np.testing.assert_allclose(np.linalg.norm(ke.l2_normalize(X), axis=1), [1.0, 1.0])


class DrawTests(unittest.TestCase):
    def test_draws_are_distinct_and_reproducible_per_year(self):
        a = ke.draw_indices(50, 2001, n_draws=20, size=10)
        b = ke.draw_indices(50, 2001, n_draws=20, size=10)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(all(len(set(row)) == 10 for row in a))
        self.assertFalse(np.array_equal(a, ke.draw_indices(50, 2002, n_draws=20, size=10)))

    def test_too_few_papers_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "only 5 eligible"):
            ke.draw_indices(5, 2001, n_draws=2, size=10)

    def test_field_draws_use_only_eligible_papers_and_cache(self):
        with TemporaryDirectory() as directory:
            eligibility = fake_embeddings(directory)
            out = Path(directory) / "draws.parquet"
            draws = ke.field_draws(directory, eligibility, [2001, 2002], output=out, n_draws=30, size=20)
            self.assertEqual(len(draws), 60)
            excluded = set(eligibility.loc[~eligibility["ke_eligible"], "openalex_id"])
            self.assertFalse(set(draws["farthest_id"]) & excluded)
            self.assertEqual(set(draws["eligible"]), {54})
            again = ke.field_draws(directory, eligibility, [2001, 2002], output=out, n_draws=30, size=20)
            pd.testing.assert_frame_equal(draws, again)
            alone = ke.field_draws(directory, eligibility, [2002], n_draws=30, size=20)
            np.testing.assert_allclose(alone["ke_max"], draws.query("year == 2002")["ke_max"])
            summary = ke.summarize(draws)
            self.assertEqual(summary["year"].tolist(), [2001, 2002])
            self.assertTrue((summary["ke_max_q1"] <= summary["ke_max_median"]).all())


class SanityCheckTests(unittest.TestCase):
    def test_matched_random_subset(self):
        with TemporaryDirectory() as directory:
            eligibility = fake_embeddings(directory)
        sizes = {2001: 30, 2002: 40}
        a = ke.matched_random_subset(eligibility, sizes, replicate=0)
        self.assertEqual(a.groupby("publication_year")["ke_eligible"].sum().to_dict(), sizes)
        self.assertFalse((a["ke_eligible"] & ~eligibility["ke_eligible"]).any())  # only eligible papers kept
        pd.testing.assert_frame_equal(a, ke.matched_random_subset(eligibility, sizes, replicate=0))
        self.assertFalse(a["ke_eligible"].equals(ke.matched_random_subset(eligibility, sizes, replicate=1)["ke_eligible"]))
        with self.assertRaisesRegex(RuntimeError, "cannot keep"):
            ke.matched_random_subset(eligibility, {2001: 100}, replicate=0)

    def test_saved_run_is_not_reused_for_other_papers_of_the_same_count(self):
        with TemporaryDirectory() as directory:
            eligibility = fake_embeddings(directory)
            out = Path(directory) / "draws.parquet"
            first = ke.field_draws(directory, ke.matched_random_subset(eligibility, {2001: 30}, 0), [2001],
                                   output=out, n_draws=5, size=20, l2=False)
            second = ke.field_draws(directory, ke.matched_random_subset(eligibility, {2001: 30}, 1), [2001],
                                    output=out, n_draws=5, size=20, l2=False)
            self.assertNotIn("ke_max_l2", first)
            self.assertFalse(np.allclose(first["ke_max"], second["ke_max"]))
            self.assertEqual(list(ke.summarize(second).filter(like="_l2")), [])

    def test_numpy_years_can_be_saved(self):
        with TemporaryDirectory() as directory:
            eligibility = fake_embeddings(directory)
            out = Path(directory) / "draws.parquet"
            ke.field_draws(directory, eligibility, np.arange(2001, 2003), output=out, n_draws=5, size=20, l2=False)
            self.assertEqual(json.loads(out.with_suffix(".json").read_text())["config"]["years"], [2001, 2002])


class ReplacementTests(unittest.TestCase):
    def test_replaced_vectors_keep_the_draws_and_change_only_their_papers(self):
        with TemporaryDirectory() as directory:
            eligibility = fake_embeddings(directory)
            target = "https://openalex.org/W2001001"   # eligible, 2001
            original = ke.load_year(directory, 2001, set(eligibility["openalex_id"]))
            same = pd.DataFrame(original[0][original[1] == target], index=[target])
            far = pd.DataFrame(np.full((1, 4), 50.0), index=[target])
            settings = dict(n_draws=30, size=20, l2=False)
            base = ke.field_draws(directory, eligibility, [2001, 2002], **settings)
            pd.testing.assert_frame_equal(base, ke.field_draws(directory, eligibility, [2001, 2002], replace=same, **settings))
            moved = ke.field_draws(directory, eligibility, [2001, 2002], replace=far, **settings)
            pd.testing.assert_frame_equal(base.query("year == 2002"), moved.query("year == 2002"))
            in_draw = moved.query("year == 2001")["farthest_id"].eq(target)
            self.assertTrue(in_draw.any())
            unchanged = base.query("year == 2001")["ke_max"].to_numpy()[~in_draw.to_numpy()]
            np.testing.assert_array_equal(unchanged, moved.query("year == 2001")["ke_max"].to_numpy()[~in_draw.to_numpy()])

    def test_saved_run_is_not_reused_with_other_vectors(self):
        with TemporaryDirectory() as directory:
            eligibility = fake_embeddings(directory)
            out = Path(directory) / "draws.parquet"
            target = "https://openalex.org/W2001001"
            settings = dict(output=out, n_draws=30, size=20, l2=False)
            base = ke.field_draws(directory, eligibility, [2001], **settings)
            moved = ke.field_draws(directory, eligibility, [2001], replace=pd.DataFrame(np.full((1, 4), 50.0), index=[target]),
                                   **settings)
            self.assertFalse(np.allclose(base["ke_max"], moved["ke_max"]))
            config = json.loads(out.with_suffix(".json").read_text())["config"]
            self.assertEqual(config["replaced_vectors"]["papers"], 1)
            pd.testing.assert_frame_equal(base, ke.field_draws(directory, eligibility, [2001], **settings))

    def test_replacing_an_ineligible_paper_is_an_error(self):
        with TemporaryDirectory() as directory:
            eligibility = fake_embeddings(directory)
            excluded = eligibility.loc[~eligibility["ke_eligible"], "openalex_id"].iloc[0]
            with self.assertRaisesRegex(RuntimeError, "not eligible"):
                ke.field_draws(directory, eligibility, [2001], replace=pd.DataFrame(np.zeros((1, 4)), index=[excluded]),
                               n_draws=5, size=20, l2=False)


class CumulativeTests(unittest.TestCase):
    def test_draws_cover_the_years_so_far_in_proportion_to_their_weights(self):
        with TemporaryDirectory() as directory:
            eligibility = fake_embeddings(directory)
            draws = ke.cumulative_draws(directory, eligibility, [2001, 2002], {2001: 1.0, 2002: 3.0}, n_draws=200, size=20)
            first, second = draws.query("year == 2001"), draws.query("year == 2002")
            self.assertTrue((first["newest_share"] == 1).all())           # the first year has only itself
            self.assertTrue((first["farthest_year"] == 2001).all())
            self.assertAlmostEqual(second["newest_share"].mean(), 0.75, delta=0.03)
            self.assertEqual(set(second["farthest_year"]), {2001, 2002})
            self.assertEqual(second["eligible"].iloc[0], 108)
            excluded = set(eligibility.loc[~eligibility["ke_eligible"], "openalex_id"])
            self.assertFalse(set(draws["farthest_id"]) & excluded)

    def test_first_year_is_a_sample_of_that_year_only(self):
        with TemporaryDirectory() as directory:
            eligibility = fake_embeddings(directory)
            X, ids = ke.load_year(directory, 2001, set(eligibility.loc[eligibility["ke_eligible"], "openalex_id"]))
            cumulative = ke.cumulative_draws(directory, eligibility, [2001], {2001: 1.0}, n_draws=5, size=20)
            for _, row in cumulative.iterrows():
                self.assertIn(row["farthest_id"], set(ids))
            self.assertTrue((cumulative["ke_max"] >= cumulative["ke_p95"]).all())

    def test_saved_run_is_reused_only_with_the_same_weights(self):
        with TemporaryDirectory() as directory:
            eligibility = fake_embeddings(directory)
            out = Path(directory) / "cumulative.parquet"
            settings = dict(output=out, n_draws=30, size=20)
            a = ke.cumulative_draws(directory, eligibility, [2001, 2002], {2001: 1.0, 2002: 1.0}, **settings)
            pd.testing.assert_frame_equal(a, ke.cumulative_draws(directory, eligibility, [2001, 2002],
                                                                 {2001: 1.0, 2002: 1.0}, **settings))
            b = ke.cumulative_draws(directory, eligibility, [2001, 2002], {2001: 1.0, 2002: 9.0}, **settings)
            self.assertGreater(b.query("year == 2002")["newest_share"].mean(), a.query("year == 2002")["newest_share"].mean())

    def test_bad_weights_and_oversized_draws_are_errors(self):
        with TemporaryDirectory() as directory:
            eligibility = fake_embeddings(directory)
            with self.assertRaisesRegex(RuntimeError, "weights"):
                ke.cumulative_draws(directory, eligibility, [2001, 2002], {2001: 1.0}, n_draws=5, size=20)
            with self.assertRaisesRegex(RuntimeError, "more papers"):
                ke.cumulative_draws(directory, eligibility, [2001], {2001: 1.0}, n_draws=5, size=60)


class GrowthTests(unittest.TestCase):
    def test_elasticity_recovers_a_power_law(self):
        years = np.arange(2001, 2025)
        n = pd.Series(1000 * 1.07 ** (years - 2001), index=years)
        out = ke.elasticity(n ** 0.3, n)
        self.assertAlmostEqual(out["elasticity"], 0.3, places=2)

    def test_first_difference_slope(self):
        years = np.arange(2001, 2025)
        growth = np.where(years % 2 == 0, 1.05, 1.09)
        n = pd.Series(1000 * np.cumprod(growth), index=years)
        frame, fit = ke.first_difference(n ** 0.5, n)
        self.assertEqual(fit["pairs"], 23)
        self.assertAlmostEqual(fit["beta"], 0.5, places=6)
        self.assertEqual(len(frame), 23)


if __name__ == "__main__":
    unittest.main()

"""Annual knowledge extent (KE) of the field from the saved pool vectors (decision log, 2026-11-22).

For each year: 1,000 draws of 1,000 distinct KE-eligible papers. In each draw, KE is the largest
Euclidean distance from a paper's vector to the draw's mean vector (Hao et al. 2026). The same
draws also give the 95th percentile of the distances, and both statistics again on L2-normalized
vectors (robustness). A year's draws come from numpy.random.default_rng([SEED, year]), so they do
not depend on which other years are computed.

The draws describe sampling variation within a year. They are not observations for the
growth-KE relationship: each year contributes one median.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261122
REMOVAL_SEED = 20261123  # size-matched random removal in the subfield sanity check (decision log, 2026-11-22)
N_DRAWS = 1_000
DRAW_SIZE = 1_000
EARLY, LATE = (2001, 2003), (2022, 2024)  # windows for the whole-period elasticity
STATISTICS = ("ke_max", "ke_p95", "ke_max_l2", "ke_p95_l2")


def load_year(embedding_dir, year, eligible_ids):
    """Vectors (float64) and ids of one year's KE-eligible papers, in pool order."""
    embedding_dir = Path(embedding_dir)
    ids = pd.read_parquet(embedding_dir / f"ids_{year}.parquet")["openalex_id"].to_numpy(dtype=str)
    vectors = np.load(embedding_dir / f"vectors_{year}.npy")
    if len(ids) != len(vectors):
        raise RuntimeError(f"{year}: {len(ids)} ids but {len(vectors)} vectors")
    keep = np.isin(ids, list(eligible_ids))
    return vectors[keep].astype(np.float64), ids[keep]


def draw_indices(n, year, n_draws=N_DRAWS, size=DRAW_SIZE, seed=SEED):
    """n_draws rows of `size` distinct indices into the year's n eligible papers."""
    if n < size:
        raise RuntimeError(f"{year}: only {n} eligible papers for draws of {size}")
    rng = np.random.default_rng([seed, year])
    return np.stack([rng.choice(n, size=size, replace=False) for _ in range(n_draws)])


def l2_normalize(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def distance_stats(X, draws):
    """Per draw: the maximum and 95th-percentile distance to the draw's centroid, and the row
    index (into X) of the farthest paper."""
    ke_max, ke_p95, farthest = (np.empty(len(draws)), np.empty(len(draws)),
                                np.empty(len(draws), dtype=np.int64))
    for d, rows in enumerate(draws):
        sample = X[rows]
        distances = np.linalg.norm(sample - sample.mean(axis=0), axis=1)
        k = int(np.argmax(distances))
        ke_max[d], ke_p95[d], farthest[d] = distances[k], np.percentile(distances, 95), rows[k]
    return ke_max, ke_p95, farthest


def year_draws(embedding_dir, year, eligible_ids, n_draws=N_DRAWS, size=DRAW_SIZE, seed=SEED, l2=True):
    """One row per draw: the statistics and the farthest paper (raw, and L2 unless l2=False)."""
    X, ids = load_year(embedding_dir, year, eligible_ids)
    draws = draw_indices(len(X), year, n_draws, size, seed)
    ke_max, ke_p95, farthest = distance_stats(X, draws)
    frame = pd.DataFrame({"year": year, "draw": np.arange(n_draws), "eligible": len(X),
                          "ke_max": ke_max, "ke_p95": ke_p95, "farthest_id": ids[farthest]})
    if l2:
        ke_max_l2, ke_p95_l2, farthest_l2 = distance_stats(l2_normalize(X), draws)
        frame = frame.assign(ke_max_l2=ke_max_l2, ke_p95_l2=ke_p95_l2, farthest_id_l2=ids[farthest_l2])
    return frame


def ids_fingerprint(ids):
    """SHA-256 of the sorted ids, so a saved run is reused only for exactly the same papers."""
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def matched_random_subset(eligibility, sizes, replicate, seed=REMOVAL_SEED):
    """A copy of `eligibility` in which each year keeps a random `sizes[year]` of its eligible papers
    (numpy.random.default_rng([seed, year, replicate])); all other papers become ineligible."""
    keep = np.zeros(len(eligibility), dtype=bool)
    years = eligibility["publication_year"].to_numpy()
    eligible = eligibility["ke_eligible"].to_numpy()
    for year, size in sizes.items():
        rows = np.flatnonzero(eligible & (years == year))
        if size > len(rows):
            raise RuntimeError(f"{year}: cannot keep {size} of {len(rows)} eligible papers")
        rng = np.random.default_rng([seed, int(year), replicate])
        keep[rng.choice(rows, size=int(size), replace=False)] = True
    return eligibility.assign(ke_eligible=keep)


def field_draws(embedding_dir, eligibility, years, output=None, **settings):
    """All years' draws. With `output`, saves them and a metadata file next to them, and reuses a
    saved file made with the same inputs and settings."""
    eligible = eligibility.loc[eligibility["ke_eligible"]]
    config = {"embedding_dir": str(embedding_dir), "years": [int(min(years)), int(max(years))],
              "n_draws": settings.get("n_draws", N_DRAWS), "draw_size": settings.get("size", DRAW_SIZE),
              "seed": settings.get("seed", SEED), "l2": settings.get("l2", True),
              "eligible_papers": int(len(eligible)), "eligible_ids_sha256": ids_fingerprint(eligible["openalex_id"]),
              "embedding_metadata_updated": json.loads(
                  (Path(embedding_dir) / "metadata.json").read_text())["updated_at_utc"]}
    if output is not None:
        output = Path(output)
        meta_path = output.with_suffix(".json")
        if output.exists() and meta_path.exists() and json.loads(meta_path.read_text())["config"] == config:
            return pd.read_parquet(output)
    frames = []
    for year in years:
        ids = set(eligible.loc[eligible["publication_year"] == year, "openalex_id"])
        frames.append(year_draws(embedding_dir, year, ids, **settings))
    draws = pd.concat(frames, ignore_index=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        draws.to_parquet(output, index=False)
        meta_path.write_text(json.dumps({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": "src/knowledge_extent.py", "config": config,
            "statistics": {"ke_max": "max distance to the draw's mean centroid (main KE)",
                           "ke_p95": "95th percentile of the draw's distances (numpy linear)",
                           "ke_max_l2": "ke_max on L2-normalized vectors",
                           "ke_p95_l2": "ke_p95 on L2-normalized vectors"},
            "rng": f"numpy.random.default_rng([{config['seed']}, year]); choice without replacement per draw",
        }, indent=2) + "\n")
    return draws


def summarize(draws, statistics=STATISTICS):
    """Per year and statistic: the median over draws and its spread across draws."""
    statistics = [name for name in statistics if name in draws]
    rows = []
    for year, group in draws.groupby("year"):
        row = {"year": int(year), "eligible": int(group["eligible"].iloc[0])}
        for name in statistics:
            values = group[name].to_numpy()
            row[f"{name}_median"] = float(np.median(values))
            for label, q in (("q1", 25), ("q3", 75), ("p2.5", 2.5), ("p97.5", 97.5)):
                row[f"{name}_{label}"] = float(np.percentile(values, q))
        rows.append(row)
    return pd.DataFrame(rows)


def window_mean(series, window):
    return series.loc[window[0]:window[1]].mean()


def elasticity(ke, n, early=EARLY, late=LATE):
    """(log KE_late − log KE_early) / (log N_late − log N_early), means over each 3-year window.
    `ke` and `n` are Series indexed by year."""
    d_ke = np.log(window_mean(ke, late)) - np.log(window_mean(ke, early))
    d_n = np.log(window_mean(n, late)) - np.log(window_mean(n, early))
    return {"early": f"{early[0]}–{early[1]}", "late": f"{late[0]}–{late[1]}",
            "ke_early": window_mean(ke, early), "ke_late": window_mean(ke, late),
            "n_early": window_mean(n, early), "n_late": window_mean(n, late),
            "ke_change": float(np.exp(d_ke) - 1), "n_growth": float(np.exp(d_n)),
            "elasticity": float(d_ke / d_n)}


def first_difference(ke, n):
    """OLS of Δlog KE_t on Δlog N_t over adjacent years (descriptive check), HC1 standard error."""
    import statsmodels.api as sm

    frame = pd.DataFrame({"d_log_ke": np.log(ke).diff(), "d_log_n": np.log(n).diff()}).dropna()
    fit = sm.OLS(frame["d_log_ke"], sm.add_constant(frame["d_log_n"])).fit(cov_type="HC1")
    low, high = fit.conf_int().loc["d_log_n"]
    return frame, {"pairs": int(fit.nobs), "alpha": float(fit.params["const"]),
                   "beta": float(fit.params["d_log_n"]), "beta_se_hc1": float(fit.bse["d_log_n"]),
                   "beta_ci_low": float(low), "beta_ci_high": float(high), "r_squared": float(fit.rsquared)}

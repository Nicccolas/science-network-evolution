"""Annual knowledge extent (KE) of the field from the saved pool vectors (decision log, 2026-11-22).

For each year: 1,000 draws of 1,000 distinct KE-eligible papers. In each draw, KE is the largest
Euclidean distance from a paper's vector to the draw's mean vector (Hao et al. 2026). The same
draws also give the 95th percentile of the distances, and both statistics again on L2-normalized
vectors (robustness). A year's draws come from numpy.random.default_rng([SEED, year]), so they do
not depend on which other years are computed.

The draws describe sampling variation within a year. They are not observations for the
growth-KE relationship: each year contributes one median.

`cumulative_draws` is an exploratory comparison, not the project's KE: the same statistics on
samples of all papers published from the first year up to year t (see its docstring).
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
CUMULATIVE_SEED = 20261127  # cumulative KE draws (exploratory comparison in field_ke.ipynb)
STATISTICS = ("ke_max", "ke_p95", "ke_max_l2", "ke_p95_l2")


def load_year(embedding_dir, year, eligible_ids, replace=None):
    """Vectors (float64) and ids of one year's KE-eligible papers, in pool order. `replace`, a
    DataFrame of vectors indexed by openalex_id, swaps in other vectors for the papers it lists
    (e.g. a paper re-embedded from edited text); its other rows are ignored."""
    embedding_dir = Path(embedding_dir)
    ids = pd.read_parquet(embedding_dir / f"ids_{year}.parquet")["openalex_id"].to_numpy(dtype=str)
    vectors = np.load(embedding_dir / f"vectors_{year}.npy")
    if len(ids) != len(vectors):
        raise RuntimeError(f"{year}: {len(ids)} ids but {len(vectors)} vectors")
    keep = np.isin(ids, list(eligible_ids))
    X, kept = vectors[keep].astype(np.float64), ids[keep]
    if replace is not None:
        if replace.shape[1] != X.shape[1]:
            raise RuntimeError(f"replacement vectors have {replace.shape[1]} dimensions, not {X.shape[1]}")
        rows = np.flatnonzero(np.isin(kept, replace.index.to_numpy(dtype=str)))
        X[rows] = replace.loc[kept[rows]].to_numpy(dtype=np.float64)
    return X, kept


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
        sample = X[rows].astype(np.float64, copy=False)
        distances = np.linalg.norm(sample - sample.mean(axis=0), axis=1)
        k = int(np.argmax(distances))
        ke_max[d], ke_p95[d], farthest[d] = distances[k], np.percentile(distances, 95), rows[k]
    return ke_max, ke_p95, farthest


def year_draws(embedding_dir, year, eligible_ids, n_draws=N_DRAWS, size=DRAW_SIZE, seed=SEED, l2=True,
               replace=None):
    """One row per draw: the statistics and the farthest paper (raw, and L2 unless l2=False).
    The draws depend only on the number of eligible papers, so `replace` (see load_year) keeps
    the same draws and changes only the vectors of the papers it lists."""
    X, ids = load_year(embedding_dir, year, eligible_ids, replace)
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


def vectors_fingerprint(vectors):
    """SHA-256 of a DataFrame of vectors indexed by openalex_id (sorted ids, then float32 values),
    so a saved run is reused only with exactly the same replacement vectors."""
    vectors = vectors.sort_index()
    digest = hashlib.sha256(ids_fingerprint(vectors.index).encode())
    digest.update(np.ascontiguousarray(vectors.to_numpy(dtype=np.float32)).tobytes())
    return digest.hexdigest()


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


def field_draws(embedding_dir, eligibility, years, output=None, replace=None, **settings):
    """All years' draws. With `output`, saves them and a metadata file next to them, and reuses a
    saved file made with the same inputs and settings. `replace` (see load_year) must list only
    eligible papers; it is fingerprinted, so a run with replaced vectors is never confused with
    one without."""
    eligible = eligibility.loc[eligibility["ke_eligible"]]
    config = {"embedding_dir": str(embedding_dir), "years": [int(min(years)), int(max(years))],
              "n_draws": settings.get("n_draws", N_DRAWS), "draw_size": settings.get("size", DRAW_SIZE),
              "seed": settings.get("seed", SEED), "l2": settings.get("l2", True),
              "eligible_papers": int(len(eligible)), "eligible_ids_sha256": ids_fingerprint(eligible["openalex_id"]),
              "embedding_metadata_updated": json.loads(
                  (Path(embedding_dir) / "metadata.json").read_text())["updated_at_utc"]}
    if replace is not None:
        unknown = set(replace.index.astype(str)) - set(eligible["openalex_id"].astype(str))
        if unknown:
            raise RuntimeError(f"{len(unknown)} replacement vectors are for papers that are not eligible")
        config["replaced_vectors"] = {"papers": int(len(replace)), "sha256": vectors_fingerprint(replace)}
    saved = _saved_draws(output, config)
    if saved is not None:
        return saved
    frames = []
    for year in years:
        ids = set(eligible.loc[eligible["publication_year"] == year, "openalex_id"])
        frames.append(year_draws(embedding_dir, year, ids, replace=replace, **settings))
    draws = pd.concat(frames, ignore_index=True)
    _save_draws(output, draws, config,
                statistics={"ke_max": "max distance to the draw's mean centroid (main KE)",
                            "ke_p95": "95th percentile of the draw's distances (numpy linear)",
                            "ke_max_l2": "ke_max on L2-normalized vectors",
                            "ke_p95_l2": "ke_p95 on L2-normalized vectors"},
                rng=f"numpy.random.default_rng([{config['seed']}, year]); choice without replacement per draw")
    return draws


def _saved_draws(output, config):
    """The draws saved at `output` if they were made with exactly `config`, else None."""
    if output is None:
        return None
    output, meta_path = Path(output), Path(output).with_suffix(".json")
    if output.exists() and meta_path.exists() and json.loads(meta_path.read_text())["config"] == config:
        return pd.read_parquet(output)
    return None


def _save_draws(output, draws, config, **notes):
    if output is None:
        return
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    draws.to_parquet(output, index=False)
    output.with_suffix(".json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "src/knowledge_extent.py", "config": config, **notes}, indent=2) + "\n")


def cumulative_draws(embedding_dir, eligibility, years, weights, output=None, n_draws=N_DRAWS, size=DRAW_SIZE,
                     seed=CUMULATIVE_SEED, replace=None):
    """Exploratory comparison, not the project's KE (which is annual): the same statistics on
    samples of the literature published from the first of `years` up to each year t.

    Per t, n_draws samples of `size` distinct papers from the eligible papers of all years up to t.
    The pool holds about the same number of papers every year while real output grows, so each
    draw first splits its `size` papers among those years (multinomial, with probabilities
    proportional to `weights[year]`, the number of papers each year's pool stands for) and then
    draws each year's share uniformly from that year's eligible papers. A draw is thus a random
    sample of the cumulative literature, not one that gives every year the same weight.
    numpy.random.default_rng([seed, t]) per t.

    Successive years' pools are nested, so the yearly values are serially correlated by
    construction, and the newest year is only a small share of each draw. Raw vectors only.
    Columns: year, draw, eligible (papers in the cumulative pool), newest_share (share of the
    draw from year t), ke_max, ke_p95, farthest_id, farthest_year."""
    years = [int(year) for year in years]
    weights = pd.Series(weights, dtype=float).reindex(years)
    if weights.isna().any() or (weights <= 0).any():
        raise RuntimeError("weights need a positive value for every year")
    eligible = eligibility.loc[eligibility["ke_eligible"]]
    config = {"kind": "cumulative", "embedding_dir": str(embedding_dir), "years": [min(years), max(years)],
              "n_draws": n_draws, "draw_size": size, "seed": seed,
              "weights": {str(year): float(weight) for year, weight in weights.items()},
              "eligible_papers": int(len(eligible[eligible["publication_year"].isin(years)])),
              "eligible_ids_sha256": ids_fingerprint(eligible.loc[eligible["publication_year"].isin(years), "openalex_id"]),
              "embedding_metadata_updated": json.loads(
                  (Path(embedding_dir) / "metadata.json").read_text())["updated_at_utc"]}
    if replace is not None:
        if set(replace.index.astype(str)) - set(eligible.loc[eligible["publication_year"].isin(years), "openalex_id"].astype(str)):
            raise RuntimeError("some replacement vectors are for papers that are not eligible")
        config["replaced_vectors"] = {"papers": int(len(replace)), "sha256": vectors_fingerprint(replace)}
    saved = _saved_draws(output, config)
    if saved is not None:
        return saved

    blocks, all_ids, offsets, counts = [], [], {}, {}
    for year in years:
        X, ids = load_year(embedding_dir, year, set(eligible.loc[eligible["publication_year"] == year, "openalex_id"]),
                           replace)
        offsets[year], counts[year] = sum(len(block) for block in blocks), len(X)
        blocks.append(X.astype(np.float32))   # the saved vectors are float32, so this loses nothing
        all_ids.append(ids)
    X, ids = np.concatenate(blocks), np.concatenate(all_ids)
    year_of_row = np.repeat(years, [counts[year] for year in years])

    frames = []
    for i, year in enumerate(years):
        upto = years[:i + 1]
        p = weights.loc[upto].to_numpy() / weights.loc[upto].sum()
        rng = np.random.default_rng([seed, year])
        draws, newest = np.empty((n_draws, size), dtype=np.int64), np.empty(n_draws)
        for d in range(n_draws):
            split = rng.multinomial(size, p)
            if (split > np.array([counts[y] for y in upto])).any():
                raise RuntimeError(f"{year}: a draw needs more papers from one year than it has eligible")
            draws[d] = np.concatenate([offsets[y] + rng.choice(counts[y], size=k, replace=False)
                                       for y, k in zip(upto, split) if k])
            newest[d] = split[-1] / size
        ke_max, ke_p95, farthest = distance_stats(X, draws)
        frames.append(pd.DataFrame({"year": year, "draw": np.arange(n_draws), "eligible": sum(counts[y] for y in upto),
                                    "newest_share": newest, "ke_max": ke_max, "ke_p95": ke_p95,
                                    "farthest_id": ids[farthest], "farthest_year": year_of_row[farthest]}))
    draws = pd.concat(frames, ignore_index=True)
    _save_draws(output, draws, config,
                statistics={"ke_max": "max distance to the draw's mean centroid",
                            "ke_p95": "95th percentile of the draw's distances (numpy linear)",
                            "newest_share": "share of the draw's papers published in year t"},
                rng=f"numpy.random.default_rng([{seed}, year]); per draw, multinomial split of the draw over the years "
                    "up to t with probabilities proportional to `weights`, then choice without replacement within each year")
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

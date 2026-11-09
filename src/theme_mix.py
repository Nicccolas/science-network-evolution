"""Theme-mix comparisons: does the mix of k-means themes change between two periods?

Used by pilot_theme_mix.ipynb on the 2,400-paper Week 2 pilot (100 KE-eligible pool papers per
year), and written so the same functions run on the full pool once it is embedded.

The rule was fixed before any theme share was computed (decision log, 2026-11-02):
  - measure: total variation distance (TVD) between two theme mixes, i.e. the share of papers
    that would have to change theme to turn one mix into the other;
  - verdict: one permutation test of TVD between 2001-03 and 2022-24, changed if p < 0.05;
    size = observed TVD minus its noise level (the mean TVD when the window labels are shuffled);
  - per-theme changes get bootstrap intervals and are described, not tested one by one.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data/pool/psychology_field32/pool.parquet"
PILOT_VECTORS = ROOT / "data/validation/citation_pairs/vectors_specter2_proximity.npz"
COUNTS = ROOT / "data/counts/psychology_field32/yearly_counts.csv"

SEED = 20261026  # Week 2 pilot seed: pilot sample, k-means, bootstrap
WEEK2_SIZES = (289, 417, 497, 253, 363, 323, 258)  # k = 7 cluster sizes in Week2_notebook.ipynb
# Short names from each cluster's top OpenAlex topics and its most distinctive TF-IDF words.
THEME_NAMES = {
    0: "Teaching and teams",
    1: "Well-being and personality",
    2: "Mental health and suicide",
    3: "Reading and language",
    4: "Psychoanalysis and theory",
    5: "Child and family development",
    6: "Perception and human factors",
}
PAPER_COLUMNS = ["openalex_id", "publication_year", "title", "abstract", "primary_topic",
                 "primary_subfield"]


def ke_eligible(pool):
    """Pool papers kept at the KE step: a title, and an abstract not shared with another pool paper."""
    missing_title = pool["title"].fillna("").str.strip() == ""
    shared_abstract = pool["abstract"].str.lower().str.strip().duplicated(keep=False)
    return ~(missing_title | shared_abstract)


def load_pilot(pool_path=POOL, vectors_path=PILOT_VECTORS):
    """The pilot's SPECTER2 proximity vectors and its papers, row-aligned (Week 2 order)."""
    saved = np.load(vectors_path)
    ids = saved["pilot_ids"].astype(str)
    pool = pd.read_parquet(pool_path, columns=PAPER_COLUMNS).set_index("openalex_id")
    papers = pool.loc[ids].reset_index()
    return saved["pilot"], papers


def fit_themes(X, k=7, seed=SEED, n_init=10):
    """k-means on the raw vectors (Euclidean, mean centres), as in Week 2."""
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
    return model.labels_, model


def window_of(years, start=2001, width=3):
    """Window label per year, e.g. 2001 -> '2001–03' for 3-year windows starting in 2001."""
    years = np.asarray(years, dtype=int)
    first = start + (years - start) // width * width
    return np.array([f"{a}–{(a + width - 1) % 100:02d}" for a in first])


def mix(codes, k):
    """Share of papers in each of k categories (codes are 0..k-1)."""
    codes = np.asarray(codes)
    return np.bincount(codes, minlength=k) / len(codes)


def tvd(p, q):
    """Total variation distance: half the summed absolute share differences (0 = same mix, 1 = disjoint)."""
    return 0.5 * np.abs(np.asarray(p, float) - np.asarray(q, float)).sum()


def evenness(p):
    """Entropy of a mix divided by log k: 0 = all papers in one category, 1 = equal shares."""
    p = np.asarray(p, float)
    nonzero = p[p > 0]
    return float(-(nonzero * np.log(nonzero)).sum() / np.log(len(p)))


def wilson(x, n, z=1.96):
    """Wilson score interval for a share x / n."""
    x, n = np.asarray(x, float), np.asarray(n, float)
    p = x / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def counts(labels, groups):
    """Papers per category (rows) and group (columns)."""
    return pd.crosstab(pd.Series(np.asarray(labels), name="theme"),
                       pd.Series(np.asarray(groups), name="window"))


def shares(labels, groups):
    """Long table: window, theme, papers, n, share and its Wilson 95% interval."""
    table = counts(labels, groups)
    long = table.stack().rename("papers").reset_index()
    long["n"] = long["window"].map(table.sum(axis=0))
    long["share"] = long["papers"] / long["n"]
    long["lo"], long["hi"] = wilson(long["papers"], long["n"])
    return long[["window", "theme", "papers", "n", "share", "lo", "hi"]]


def chi_square(labels, groups):
    """Chi-square test of homogeneity of the mix across groups: statistic, df, p."""
    result = stats.chi2_contingency(counts(labels, groups))
    return {"chi2": float(result.statistic), "df": int(result.dof), "p": float(result.pvalue)}


def _two_windows(labels, groups, a, b):
    codes, categories = pd.factorize(np.asarray(labels), sort=True)
    groups = np.asarray(groups)
    keep = np.isin(groups, [a, b])
    return codes[keep], groups[keep] == a, len(categories), categories


def permutation_test(labels, groups, a, b, n_perm=10_000, seed=SEED):
    """TVD between the mixes of windows a and b, against the TVD from shuffled window labels.

    Returns the observed TVD, its noise level (mean TVD under shuffling), the size above noise,
    the 95th percentile of the shuffled TVDs, and the permutation p-value.
    """
    codes, in_a, k, _ = _two_windows(labels, groups, a, b)
    observed = tvd(mix(codes[in_a], k), mix(codes[~in_a], k))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(in_a)
        null[i] = tvd(mix(codes[shuffled], k), mix(codes[~shuffled], k))
    return {"n_a": int(in_a.sum()), "n_b": int((~in_a).sum()), "categories": k,
            "tvd": observed, "noise": float(null.mean()), "above_noise": observed - float(null.mean()),
            "null_95": float(np.percentile(null, 95)),
            "p": float((1 + (null >= observed).sum()) / (n_perm + 1))}


def detection_rate(p_a, p_b, n_a, n_b, n_sim=1_000, n_perm=999, alpha=0.05, seed=SEED):
    """How often the permutation test finds p < alpha when n_a papers come from mix p_a and n_b from p_b.

    With p_a == p_b this is the false-positive rate (about alpha); otherwise it is the power to
    detect that shift with windows of this size.
    """
    rng = np.random.default_rng(seed)
    k = len(p_a)
    groups = np.repeat(["a", "b"], [n_a, n_b])
    hits = 0
    for _ in range(n_sim):
        labels = np.concatenate([rng.choice(k, n_a, p=p_a), rng.choice(k, n_b, p=p_b)])
        hits += permutation_test(labels, groups, "a", "b", n_perm=n_perm, seed=rng.integers(2**32))["p"] < alpha
    return hits / n_sim


def bootstrap_diff(labels, groups, a, b, n_boot=10_000, seed=SEED):
    """Per-category shares in windows a and b and the change b - a, with percentile 95% intervals.

    Papers are resampled with replacement within each window.
    """
    codes, in_a, k, categories = _two_windows(labels, groups, a, b)
    code_a, code_b = codes[in_a], codes[~in_a]
    rng = np.random.default_rng(seed)
    diffs = np.empty((n_boot, k))
    for i in range(n_boot):
        diffs[i] = (mix(rng.choice(code_b, code_b.size), k) - mix(rng.choice(code_a, code_a.size), k))
    share_a, share_b = mix(code_a, k), mix(code_b, k)
    return pd.DataFrame({"theme": categories, f"share {a}": share_a, f"share {b}": share_b,
                         "change": share_b - share_a,
                         "lo": np.percentile(diffs, 2.5, axis=0), "hi": np.percentile(diffs, 97.5, axis=0)})


def evenness_by_group(labels, groups, n_boot=2_000, seed=SEED):
    """Evenness of the mix in each group, with a bootstrap 95% interval (papers resampled within group)."""
    codes, categories = pd.factorize(np.asarray(labels), sort=True)
    groups = np.asarray(groups)
    k = len(categories)
    rng = np.random.default_rng(seed)
    rows = []
    for group in sorted(set(groups)):
        own = codes[groups == group]
        boot = [evenness(mix(rng.choice(own, own.size), k)) for _ in range(n_boot)]
        rows.append({"window": group, "n": own.size, "evenness": evenness(mix(own, k)),
                     "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5)})
    return pd.DataFrame(rows)


def top_topics(topics, labels, n=3):
    """The n most common OpenAlex primary topics in each cluster."""
    return (pd.Series(np.asarray(topics)).groupby(np.asarray(labels))
            .agg(lambda s: s.value_counts().index[:n].tolist()))


def distinctive_words(texts, labels, n=10, min_df=10):
    """Words that set each cluster apart (Week 2 method): mean paper TF-IDF in the cluster minus overall."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    overall = np.asarray(tfidf.mean(axis=0)).ravel()
    labels = np.asarray(labels)
    return {k: ", ".join(terms[np.argsort(np.asarray(tfidf[labels == k].mean(axis=0)).ravel() - overall)[::-1][:n]])
            for k in np.unique(labels)}

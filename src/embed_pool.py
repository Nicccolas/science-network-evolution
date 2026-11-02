#!/usr/bin/env python3
"""Embed every paper in the KE pool once, one year at a time, with the fixed setup in src/embeddings.py.

For each year it writes, in pool order:
  vectors_<year>.npy     float32, one 768-d row per pool paper (raw CLS vectors)
  ids_<year>.parquet     openalex_id and the untruncated token count, row-aligned with the vectors
and, for the whole run, embedding_report.csv (per-year diagnostics) and metadata.json (setup and
provenance). Finished years are skipped on a re-run, so an interrupted run resumes where it
stopped. A re-run with a different pool, model, revision, device, batch size, or package versions
stops instead of mixing setups (CLAUDE.md, "One fixed embedding setup").

All 10,000 pool papers per year are embedded. Papers with missing titles get an empty title and
are dropped at the KE step, not here (decision log, 2026-10-26).

The full run takes about 4 hours on mps (about 17 papers/s, Week 2 pilot). Keep the laptop
plugged in and awake:

    caffeinate -is python -m src.embed_pool --pool data/pool/psychology_field32/pool.parquet \\
        --output data/embeddings/specter2_proximity
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_test import extract  # noqa: E402  (atomic_json)
from src import embeddings, openalex_counts  # noqa: E402  (the embedding setup; git_commit)


BATCH_SIZE = 32  # the batch size of the Week 2 speed check
PACKAGES = ["torch", "transformers", "adapters", "tokenizers", "huggingface_hub", "numpy"]
POOL_COLUMNS = ["openalex_id", "publication_year", "title", "abstract"]


def now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_pool(path):
    pool = pd.read_parquet(path, columns=POOL_COLUMNS)
    if pool["openalex_id"].duplicated().any():
        raise RuntimeError(f"{path} repeats an openalex_id")
    if pool["abstract"].isna().any():
        raise RuntimeError(f"{path} has papers without an abstract")
    return pool


def vectors_path(output, year):
    return output / f"vectors_{year}.npy"


def ids_path(output, year):
    return output / f"ids_{year}.parquet"


def run_config(pool_path, info, batch_size):
    """Everything that must be identical for every year in one output directory."""
    return {"pool_sha256": file_sha256(pool_path), "model": info, "batch_size": batch_size,
            "dtype": "float32", "packages": package_versions()}


def start_metadata(output, config, pool_path):
    """Return the run's metadata, reusing an earlier run's only if its setup is identical."""
    path = output / "metadata.json"
    if path.exists():
        earlier = json.loads(path.read_text(encoding="utf-8"))
        changed = [key for key in config if earlier["config"].get(key) != config[key]]
        if changed:
            raise RuntimeError(f"{output} holds vectors from a different setup (changed: "
                               f"{', '.join(changed)}). Use a new --output directory.")
        return earlier
    if any(output.glob("vectors_*.npy")) or any(output.glob("ids_*.parquet")):
        raise RuntimeError(f"{output} has vector files but no metadata.json. Use a new --output directory.")
    return {"created_at_utc": now(), "script": "src/embed_pool.py", "pool": str(pool_path),
            "config": config, "per_year": [],
            "notes": ["Rows follow pool.parquet order within each year; ids_<year>.parquet gives the "
                      "openalex_id of each vector row.",
                      "n_tokens counts tokens before truncation, including special tokens; inputs "
                      f"over {embeddings.MAX_TOKENS} were truncated.",
                      "Missing titles are embedded as an empty title and dropped at the KE step."]}


def finished(output, year, ids, hidden_size, metadata):
    """True if this year's files are complete, match the pool, and are recorded in metadata."""
    if year not in {row["year"] for row in metadata["per_year"]}:
        return False
    try:
        saved = pd.read_parquet(ids_path(output, year))["openalex_id"].to_numpy(dtype=str)
        shape = np.load(vectors_path(output, year), mmap_mode="r").shape
    except (OSError, ValueError):
        return False
    return shape == (len(ids), hidden_size) and np.array_equal(saved, ids)


def embed_year(model, tokenizer, papers, batch_size):
    texts = [embeddings.input_text(title, abstract, tokenizer.sep_token)
             for title, abstract in zip(papers["title"], papers["abstract"])]
    n_tokens = embeddings.token_counts(tokenizer, texts)
    vectors = embeddings.encode(model, tokenizer, texts, batch_size=batch_size)
    return vectors, n_tokens


def save_year(output, year, ids, vectors, n_tokens):
    """Write both files under temporary names, then rename (the ids file last)."""
    temporary_vectors = output / f".vectors_{year}.npy.tmp"
    with open(temporary_vectors, "wb") as file:
        np.save(file, vectors)
        file.flush()
        os.fsync(file.fileno())
    temporary_ids = output / f".ids_{year}.parquet.tmp"
    pd.DataFrame({"openalex_id": ids, "n_tokens": n_tokens}).to_parquet(temporary_ids, index=False)
    temporary_vectors.replace(vectors_path(output, year))
    temporary_ids.replace(ids_path(output, year))


def year_report(year, papers, vectors, n_tokens, seconds):
    over = n_tokens > embeddings.MAX_TOKENS
    norms = np.linalg.norm(vectors, axis=1)
    has_title = papers["title"].map(lambda title: isinstance(title, str) and title.strip() != "")
    return {
        "year": int(year), "papers": len(papers), "missing_titles": int((~has_title).sum()),
        "tokens_median": float(np.median(n_tokens)), "tokens_max": int(n_tokens.max()),
        "over_max_tokens": int(over.sum()), "share_over_max_tokens": round(float(over.mean()), 4),
        "norm_mean": round(float(norms.mean()), 4), "norm_sd": round(float(norms.std()), 4),
        "seconds": round(seconds, 1), "papers_per_second": round(len(papers) / max(seconds, 1e-9), 1),
        "finished_at_utc": now(), "git_commit": openalex_counts.git_commit(),
    }


def write_report(output, metadata):
    rows = sorted(metadata["per_year"], key=lambda row: row["year"])
    temporary = output / ".embedding_report.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "embedding_report.csv")


def release_device_memory(device):
    if device == "mps":
        import torch
        torch.mps.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", type=Path, required=True, help="pool.parquet from src/openalex_pool.py")
    parser.add_argument("--output", type=Path, required=True, help="one directory per model")
    parser.add_argument("--model", default="specter2_proximity", choices=sorted(embeddings.MODELS))
    parser.add_argument("--min-year", type=int, help="default: first year in the pool")
    parser.add_argument("--max-year", type=int, help="default: last year in the pool")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    pool = load_pool(args.pool)
    available = set(pool["publication_year"].tolist())
    first = args.min_year if args.min_year is not None else min(available)
    last = args.max_year if args.max_year is not None else max(available)
    years = list(range(first, last + 1))
    missing = [year for year in years if year not in available]
    if not years or missing:
        parser.error(f"the pool has no papers for {missing or 'that range'}")

    model, tokenizer, info = embeddings.load(args.model)
    if info["revision"] is None or (info["adapter"] and info["adapter_revision"] is None):
        raise RuntimeError("Could not read the model revision from the Hugging Face cache")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = start_metadata(args.output, run_config(args.pool, info, args.batch_size), args.pool)
    extract.atomic_json(args.output / "metadata.json", metadata)
    print(f"{info['name']} (revision {info['revision'][:8]}"
          + (f", adapter {info['adapter_revision'][:8]}" if info["adapter"] else "")
          + f") on {info['device']}, years {first}-{last}")

    timings = []
    for position, year in enumerate(years):
        papers = pool[pool["publication_year"] == year]
        ids = papers["openalex_id"].to_numpy(dtype=str)
        if finished(args.output, year, ids, model.config.hidden_size, metadata):
            print(f"{year}: already done, skipped")
            continue
        start = time.perf_counter()
        vectors, n_tokens = embed_year(model, tokenizer, papers, args.batch_size)
        seconds = time.perf_counter() - start
        save_year(args.output, year, ids, vectors, n_tokens)
        report = year_report(year, papers, vectors, n_tokens, seconds)
        metadata["per_year"] = [row for row in metadata["per_year"] if row["year"] != year] + [report]
        metadata["updated_at_utc"] = now()
        extract.atomic_json(args.output / "metadata.json", metadata)
        write_report(args.output, metadata)
        release_device_memory(info["device"])
        timings.append(seconds)
        recorded = {row["year"] for row in metadata["per_year"]}
        left = sum(1 for later in years[position + 1:] if later not in recorded)
        print(f"{year}: {report['papers']:,} papers in {seconds / 60:.1f} min "
              f"({report['papers_per_second']}/s), {report['share_over_max_tokens']:.1%} over "
              f"{embeddings.MAX_TOKENS} tokens; about {left * np.mean(timings) / 60:.0f} min left")

    done = sorted(row["year"] for row in metadata["per_year"])
    print(f"Done: {len(done)} years in {args.output} ({done[0]}-{done[-1]})")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # show progress even when output goes to a log file
    try:
        main()
    except RuntimeError as error:
        sys.exit(str(error))
    except KeyboardInterrupt:
        sys.exit("Stopped. Finished years are saved; rerun the same command to resume.")

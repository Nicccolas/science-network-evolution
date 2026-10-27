"""Embed paper titles and abstracts with one fixed setup: input format, 512-token truncation, CLS output.

Used by the Week 2 model check and by the full embedding run, so both see exactly the same
inputs and outputs (decision log, 2026-10-26).
"""
from pathlib import Path

import numpy as np


MAX_TOKENS = 512  # model maximum; Hao et al. (2026) used 288
MODELS = {
    # name: (base model repository, adapter repository or None)
    "specter2_proximity": ("allenai/specter2_base", "allenai/specter2"),
    "specter2_base": ("allenai/specter2_base", None),
    "bge_base": ("BAAI/bge-base-en-v1.5", None),
}


def input_text(title, abstract, sep):
    """Title + separator token + abstract (the SPECTER convention); a missing title (None/NaN) becomes ''."""
    return (title if isinstance(title, str) else "") + sep + abstract


def token_counts(tokenizer, texts, batch_size=10_000):
    """Tokens each input would have without truncation, including the special tokens."""
    counts = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start:start + batch_size], truncation=False, verbose=False)
        counts += [len(ids) for ids in encoded["input_ids"]]
    return np.array(counts)


def cached_revision(repo, filename="config.json"):
    """Commit hash of the locally cached snapshot of a Hugging Face repository, or None."""
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(repo, filename)
    return Path(path).parent.name if isinstance(path, str) else None


def load(name, device=None):
    """Return (model, tokenizer, info) for one of MODELS, in eval mode on `device`."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    repo, adapter = MODELS[name]
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(repo)
    if adapter:
        from adapters import AutoAdapterModel
        model = AutoAdapterModel.from_pretrained(repo)
        model.load_adapter(adapter, source="hf", load_as="proximity", set_active=True)
    else:
        model = AutoModel.from_pretrained(repo)
    model.to(device).eval()
    info = {"name": name, "repo": repo, "revision": cached_revision(repo),
            "adapter": adapter,
            "adapter_revision": cached_revision(adapter, "adapter_config.json") if adapter else None,
            "device": device, "max_tokens": MAX_TOKENS,
            "input": "title + sep_token + abstract", "output": "CLS token, raw (not normalized)"}
    return model, tokenizer, info


def encode(model, tokenizer, texts, batch_size=32):
    """CLS vectors (float32) in input order. Batches group inputs of similar length for speed."""
    import torch

    device = next(model.parameters()).device
    lengths = token_counts(tokenizer, texts)
    order = np.argsort(-np.minimum(lengths, MAX_TOKENS), kind="stable")
    vectors = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            rows = order[start:start + batch_size]
            batch = tokenizer([texts[i] for i in rows], padding=True, truncation=True,
                              max_length=MAX_TOKENS, return_tensors="pt").to(device)
            vectors[rows] = model(**batch).last_hidden_state[:, 0, :].float().cpu().numpy()
    return vectors

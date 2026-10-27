"""Offline checks for the shared embedding code (no model download)."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from src import embeddings


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    """One token per word, plus [CLS] and [SEP]."""

    def __call__(self, texts, padding=False, truncation=False, max_length=None,
                 return_tensors=None, verbose=True):
        ids = [[101] + [7] * len(text.split()) + [102] for text in texts]
        if truncation:
            ids = [row[:max_length] for row in ids]
        if return_tensors != "pt":
            return {"input_ids": ids}
        width = max(len(row) for row in ids)
        mask = [[1] * len(row) + [0] * (width - len(row)) for row in ids]
        return FakeBatch(input_ids=torch.tensor([row + [0] * (width - len(row)) for row in ids]),
                         attention_mask=torch.tensor(mask))


class FakeModel:
    """The CLS vector repeats the number of unpadded tokens in the input."""

    config = SimpleNamespace(hidden_size=3)

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, input_ids, attention_mask):
        tokens = attention_mask.sum(dim=1, keepdim=True).float()
        hidden = tokens[:, :, None].expand(-1, input_ids.shape[1], 3)
        return SimpleNamespace(last_hidden_state=hidden)


class EmbeddingTests(unittest.TestCase):
    def test_input_text_joins_title_and_abstract(self):
        self.assertEqual(embeddings.input_text("A title", "An abstract.", "[SEP]"), "A title[SEP]An abstract.")
        self.assertEqual(embeddings.input_text(None, "An abstract.", "[SEP]"), "[SEP]An abstract.")
        self.assertEqual(embeddings.input_text(float("nan"), "An abstract.", "[SEP]"), "[SEP]An abstract.")

    def test_token_counts_include_special_tokens(self):
        counts = embeddings.token_counts(FakeTokenizer(), ["one two", "one", "one two three four"], batch_size=2)
        self.assertEqual(counts.tolist(), [4, 3, 6])

    def test_encode_returns_vectors_in_input_order(self):
        texts = ["a b c", "a", "a b c d e f", "a b", "a b c d"]
        vectors = embeddings.encode(FakeModel(), FakeTokenizer(), texts, batch_size=2)
        self.assertEqual(vectors.dtype, np.float32)
        self.assertEqual(vectors[:, 0].tolist(), [len(t.split()) + 2 for t in texts])

    def test_encode_truncates_at_max_tokens(self):
        long_text = " ".join(["word"] * (embeddings.MAX_TOKENS + 50))
        vectors = embeddings.encode(FakeModel(), FakeTokenizer(), [long_text, "short"])
        self.assertEqual(vectors[:, 0].tolist(), [embeddings.MAX_TOKENS, 3])


if __name__ == "__main__":
    unittest.main()

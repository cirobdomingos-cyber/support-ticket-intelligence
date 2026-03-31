"""Mock heavy optional dependencies so tests run without torch/faiss/sentence-transformers."""
import sys
from unittest.mock import MagicMock

for _mod in [
    "faiss",
    "sentence_transformers",
    "sentence_transformers.SentenceTransformer",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

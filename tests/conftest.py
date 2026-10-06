from pathlib import Path

import pytest

from emphtts.tts.model.utils import get_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def vocab():
    vocab_char_map, vocab_size = get_tokenizer(REPO_ROOT / "data" / "vocab.txt")
    return vocab_char_map, vocab_size

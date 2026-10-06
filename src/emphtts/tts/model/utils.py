from __future__ import annotations

import re
import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence


# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


# Text tokenization. Multi-character "[...]" vocab entries are matched greedily, everything else per character.

_multi_char_cache: dict[int, list[str]] = {}
_regex_cache: dict[int, re.Pattern] = {}


def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],
    padding_value=-1,
) -> int["b nt"]:  # noqa: F722
    _key = id(vocab_char_map)

    if _key not in _multi_char_cache:
        _multi_char_cache[_key] = sorted(
            [k for k in vocab_char_map if (len(k) > 1 and k.startswith('[') and k.endswith(']'))], key=len, reverse=True
        )
    multi_char = _multi_char_cache[_key]

    if _key not in _regex_cache:
        if multi_char:
            # Alternation tries left-to-right, so longest-first order gives greedy match.
            pattern = "|".join(re.escape(k) for k in multi_char) + "|."
        else:
            pattern = "."
        _regex_cache[_key] = re.compile(pattern, re.DOTALL)
    tok_re = _regex_cache[_key]

    def _tokenize(t):
        if isinstance(t, list):
            return [vocab_char_map.get(c, 0) for c in t]
        return [vocab_char_map.get(m, 0) for m in tok_re.findall(t)]

    list_idx_tensors = [torch.tensor(_tokenize(t), dtype=torch.long) for t in text]
    return pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)


# Get tokenizer


def get_tokenizer(vocab_file, tokenizer: str = "custom"):
    """Load the character vocabulary used by an EmphTTS checkpoint."""
    if tokenizer != "custom":
        raise ValueError("This release uses a custom vocabulary file; set tokenizer=custom.")
    with open(vocab_file, "r", encoding="utf-8") as stream:
        vocab_char_map = {line.rstrip("\n"): index for index, line in enumerate(stream)}
    if vocab_char_map.get(" ") != 0:
        raise ValueError("The first vocabulary entry must be a space.")
    return vocab_char_map, len(vocab_char_map)

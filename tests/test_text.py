import pytest

from emphtts.duration.grpo_utils import TTSRewardFn
from emphtts.tts.eval.utils_eval import tinystress_emphasized_text
from emphtts.tts.model.utils import list_str_to_idx


def test_vocab_has_space_first_and_emphasis_marker(vocab):
    vocab_char_map, vocab_size = vocab
    assert vocab_char_map[" "] == 0
    assert "*" in vocab_char_map
    assert vocab_size == len(vocab_char_map)


def test_list_str_to_idx_pads_with_minus_one(vocab):
    vocab_char_map, _ = vocab
    ids = list_str_to_idx(["I *really* did.", "ok"], vocab_char_map)
    assert ids.shape == (2, len("I *really* did."))
    assert ids[0, 2].item() == vocab_char_map["*"]
    assert (ids[1, 2:] == -1).all()


def test_tinystress_emphasis_wraps_whole_token():
    words = "She replied, very slowly.".split()
    assert tinystress_emphasized_text(words, [1, 3]) == "She *replied,* very *slowly.*"
    assert tinystress_emphasized_text(words, []) == "She replied, very slowly."


@pytest.mark.parametrize(
    "text, clean, stressed",
    [
        ("You *chose* to do this?", "You chose to do this?", [1]),
        ("on a *Saturday*!", "on a Saturday!", [2]),
        ("She *replied,* slowly", "She replied, slowly", [1]),
        ('he said "*now*" twice', 'he said "now" twice', [2]),
        ("*I* *cannot* answer.", "I cannot answer.", [0, 1]),
        ("no emphasis here", "no emphasis here", []),
        ("a * b", "a * b", []),
    ],
)
def test_parse_emphasized_text(text, clean, stressed):
    assert TTSRewardFn._parse_emphasized_text(text) == (clean, stressed)

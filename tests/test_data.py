import soundfile as sf

from emphtts.tts.eval.utils_eval import TINYSTRESS_DEFAULT_PROMPT_MAP, load_tinystress_prompt_map


def test_bundled_prompts_resolve_relative_to_prompt_map(repo_root):
    prompt_map = load_tinystress_prompt_map(repo_root / TINYSTRESS_DEFAULT_PROMPT_MAP)
    assert len(prompt_map) == 10
    for voice, entry in prompt_map.items():
        info = sf.info(entry["audio"])
        assert info.samplerate == 48000 and info.channels == 1, voice
        assert 4.0 <= info.duration <= 5.0, voice
        assert entry["text"].strip(), voice


def test_prompt_map_reports_missing_voice(repo_root, tmp_path):
    prompt_map_path = repo_root / TINYSTRESS_DEFAULT_PROMPT_MAP
    try:
        load_tinystress_prompt_map(prompt_map_path, voices=["en-US-standard-A", "not-a-voice"])
    except ValueError as error:
        assert "not-a-voice" in str(error)
    else:
        raise AssertionError("an unknown voice should be reported")

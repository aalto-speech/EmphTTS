"""Read-only sanity check of the real TinyStress-15K test shard.

Verifies row count and uniqueness, selected prompt coverage/duration/distinctness,
emphasis-index bounds, and that the embedded test WAV payloads decode as 48 kHz
mono through both librosa and torchaudio file-like interfaces.

    python scripts/validate_tinystress_shard.py [--parquet PATH] [--audio-samples N]
"""

import argparse
import io
import re

import librosa
import torchaudio

from emphtts.tts.eval.utils_eval import (
    TINYSTRESS_DEFAULT_PARQUET,
    TINYSTRESS_DEFAULT_PROMPT_MAP,
    _read_tinystress_audio,
    _read_tinystress_rows,
    load_tinystress_prompt_map,
    tinystress_emphasized_text,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=TINYSTRESS_DEFAULT_PARQUET)
    parser.add_argument("--prompt-map", default=TINYSTRESS_DEFAULT_PROMPT_MAP)
    parser.add_argument("--expected-rows", type=int, default=1000)
    parser.add_argument("--audio-samples", type=int, default=25, help="How many WAVs to decode")
    args = parser.parse_args()

    rows = _read_tinystress_rows(args.parquet)
    print(f"Parquet          : {args.parquet}")
    print(f"Rows             : {len(rows)} (unique ids: {len({r['id'] for r in rows})})")
    assert len(rows) == args.expected_rows, f"expected {args.expected_rows} rows, got {len(rows)}"
    assert len({r["id"] for r in rows}) == len(rows), "duplicate ids"

    ids = sorted(r["id"] for r in rows)
    gaps = sorted(set(range(ids[0], ids[-1] + 1)) - set(ids))
    print(f"Id range         : {ids[0]}..{ids[-1]} with {len(gaps)} gaps {gaps}")

    voices = sorted({r["voice"] for r in rows})
    print(f"Voices ({len(voices)})       : {voices}")

    prompt_map = load_tinystress_prompt_map(args.prompt_map, voices=voices)
    normalized_prompts = {
        re.sub(r"[^a-z0-9]+", " ", prompt_map[voice]["text"].casefold()).strip()
        for voice in voices
    }
    assert len(normalized_prompts) == len(voices), "prompt transcriptions are not distinct"

    prompt_durations = []
    for voice in voices:
        wav, sample_rate = torchaudio.load(prompt_map[voice]["audio"], backend="soundfile")
        duration = wav.shape[-1] / sample_rate
        assert sample_rate == 48000 and wav.shape[0] == 1, (
            f"{voice} prompt is {tuple(wav.shape)} @ {sample_rate} Hz"
        )
        assert 4.0 <= duration <= 5.0, f"{voice} prompt is {duration:.3f}s"
        prompt_durations.append(duration)
    print(
        f"Prompt map       : {args.prompt_map} — {len(voices)} distinct 4–5s prompts, "
        f"mean {sum(prompt_durations) / len(prompt_durations):.2f}s"
    )

    # _read_tinystress_rows already rejects out-of-range indices; re-state it explicitly.
    for row in rows:
        assert all(0 <= i < len(row["words"]) for i in row["emphasis_indices"]), row["id"]
    num_stressed = sum(len(r["emphasis_indices"]) for r in rows)
    num_words = sum(len(r["words"]) for r in rows)
    print(f"Emphasis indices : all in range; {num_stressed}/{num_words} tokens stressed "
          f"({100 * num_stressed / num_words:.1f}%)")

    sample_ids = [r["id"] for r in rows[: args.audio_samples]]
    audio_by_id = _read_tinystress_audio(args.parquet, sample_ids)
    total_seconds = 0.0
    for sample_id in sample_ids:
        handle = audio_by_id[sample_id]
        assert isinstance(handle, io.BytesIO)

        handle.seek(0)
        audio, sr = librosa.load(handle, sr=None)
        assert sr == 48000, f"id={sample_id} sample rate {sr}"
        assert audio.ndim == 1, f"id={sample_id} is not mono"

        handle.seek(0)
        wav, sr2 = torchaudio.load(handle, backend="soundfile")
        assert sr2 == 48000 and wav.shape[0] == 1, f"id={sample_id} torchaudio {wav.shape} @ {sr2}"
        assert wav.shape[-1] == audio.shape[-1], f"id={sample_id} length mismatch"
        total_seconds += wav.shape[-1] / sr2
    print(f"Audio            : {len(sample_ids)} payloads decoded, 48 kHz mono, "
          f"mean {total_seconds / len(sample_ids):.2f}s via librosa and torchaudio")

    example = rows[0]
    print("\nExample target text:")
    print("  " + repr(" " + tinystress_emphasized_text(example["words"], example["emphasis_indices"])))
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

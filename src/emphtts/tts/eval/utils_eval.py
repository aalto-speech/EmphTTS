"""TinyStress-15K metadata and prompt handling."""
import io
import json
import os

TINYSTRESS_DEFAULT_PROMPT_MAP = "data/tinystress-15k/prompts.json"

TINYSTRESS_REQUIRED_COLUMNS = ("id", "transcription", "emphasis_indices", "metadata")
TINYSTRESS_DEFAULT_PARQUET = "data/tinystress-15k/test-00000-of-00001.parquet"


def load_tinystress_prompt_map(prompt_map_path, voices=None):
    """Load ``prompts.json`` and check that every requested voice is usable.

    Args:
        prompt_map_path: JSON file mapping voice name -> {"text": ..., "audio": ...}. Relative
            ``audio`` paths are resolved against the directory of the JSON file.
        voices: voices that must be covered by an entry. ``None`` only validates the structure.

    Every problem found is reported together.
    """
    with open(prompt_map_path, "r", encoding="utf-8") as f:
        prompt_map = json.load(f)

    if not isinstance(prompt_map, dict):
        raise ValueError(f"TinyStress prompt map {prompt_map_path} must be a JSON object.")

    base_dir = os.path.dirname(os.path.abspath(prompt_map_path))
    problems = []
    for voice, entry in prompt_map.items():
        if not isinstance(entry, dict) or not entry.get("text") or not entry.get("audio"):
            problems.append(f"  {voice}: entry must be an object with non-empty 'text' and 'audio'")
            continue
        entry["audio"] = os.path.join(base_dir, entry["audio"])  # no-op for absolute paths

    if voices is not None:
        for voice in sorted(set(voices)):
            entry = prompt_map.get(voice)
            if entry is None:
                problems.append(f"  {voice}: missing from {prompt_map_path}")
            elif isinstance(entry, dict) and entry.get("audio") and not os.path.exists(entry["audio"]):
                problems.append(f"  {voice}: 'audio' path does not exist ({entry['audio']})")

    if problems:
        raise ValueError(
            f"TinyStress prompt map {prompt_map_path} is incomplete:\n" + "\n".join(problems)
        )

    return prompt_map


def _read_tinystress_rows(parquet_path):
    """Read the light-weight columns of the TinyStress Parquet shard and validate them."""
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(parquet_path)
    available = set(parquet_file.schema_arrow.names)
    missing = [c for c in TINYSTRESS_REQUIRED_COLUMNS if c not in available]
    if missing:
        raise ValueError(
            f"TinyStress Parquet {parquet_path} is missing required column(s): {missing}. "
            f"Found: {sorted(available)}"
        )

    rows = []
    seen_ids = set()
    duplicate_ids = []
    index_problems = []

    for batch in parquet_file.iter_batches(
        batch_size=256, columns=list(TINYSTRESS_REQUIRED_COLUMNS)
    ):
        for row in batch.to_pylist():
            sample_id = row["id"]
            if sample_id in seen_ids:
                duplicate_ids.append(sample_id)
                continue
            seen_ids.add(sample_id)

            transcription = row["transcription"]
            words = transcription.split()
            emphasis_indices = sorted(set(row["emphasis_indices"] or []))
            out_of_range = [i for i in emphasis_indices if i < 0 or i >= len(words)]
            if out_of_range:
                index_problems.append(
                    f"  id={sample_id}: emphasis indices {out_of_range} outside [0, {len(words)})"
                )
                continue

            metadata = row["metadata"] or {}
            rows.append(
                {
                    # Original IDs are kept as-is: the split has nine gaps and
                    # renumbering would desync us from the published dataset.
                    "utt": f"tinystress_{sample_id:05d}",
                    "id": sample_id,
                    "transcription": transcription,
                    "words": words,
                    "emphasis_indices": emphasis_indices,
                    "voice": metadata.get("voice_name"),
                }
            )

    if duplicate_ids:
        raise ValueError(
            f"TinyStress Parquet {parquet_path} contains duplicate ids: {sorted(set(duplicate_ids))}"
        )
    if index_problems:
        raise ValueError(
            f"TinyStress Parquet {parquet_path} has invalid emphasis indices:\n"
            + "\n".join(index_problems)
        )
    if not rows:
        raise ValueError(f"TinyStress Parquet {parquet_path} yielded no rows.")

    missing_voice = [r["id"] for r in rows if not r["voice"]]
    if missing_voice:
        raise ValueError(
            f"TinyStress Parquet {parquet_path} has rows without metadata.voice_name: {missing_voice[:10]}"
        )

    return rows


def _read_tinystress_audio(parquet_path, wanted_ids):
    """Return {id: BytesIO} for the embedded WAV payloads (roughly 338 MB for the test split)."""
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(parquet_path)
    if "audio" not in set(parquet_file.schema_arrow.names):
        raise ValueError(f"TinyStress Parquet {parquet_path} has no 'audio' column.")

    wanted = set(wanted_ids)
    audio_by_id = {}
    for batch in parquet_file.iter_batches(batch_size=64, columns=["id", "audio"]):
        for row in batch.to_pylist():
            if row["id"] not in wanted:
                continue
            payload = (row["audio"] or {}).get("bytes")
            if not payload:
                raise ValueError(f"TinyStress row id={row['id']} has an empty embedded audio payload.")
            audio_by_id[row["id"]] = io.BytesIO(payload)

    absent = sorted(wanted - set(audio_by_id))
    if absent:
        raise ValueError(f"TinyStress Parquet {parquet_path} is missing audio for ids: {absent[:10]}")
    return audio_by_id


def tinystress_emphasized_text(words, emphasis_indices):
    """Mark every stressed whitespace token with asterisks, punctuation included.

    ``emphasis_indices`` index whitespace-split tokens, so ``replied,`` becomes
    ``*replied,*`` and adjacent stressed tokens are wrapped independently.
    """
    emphasis_set = set(emphasis_indices)
    return " ".join(f"*{word}*" if i in emphasis_set else word for i, word in enumerate(words))


def get_tinystress_rows(parquet_path, prompt_map_path):
    """Rows of the TinyStress test split joined with their per-voice reference prompt."""
    rows = _read_tinystress_rows(parquet_path)
    prompt_map = load_tinystress_prompt_map(prompt_map_path, voices=[r["voice"] for r in rows])
    for row in rows:
        entry = prompt_map[row["voice"]]
        row["prompt_text"] = entry["text"]
        row["prompt_wav"] = entry["audio"]
    return rows


# tinystress testset metainfo: utt, prompt_text, prompt_wav, gt_text, gt_wav
def get_tinystress_testset_metainfo(parquet_path, prompt_map_path, include_gt_audio=False):
    """Build F5-TTS metainfo directly from the TinyStress test Parquet shard.

    ``include_gt_audio`` exposes the embedded WAV bytes as file-like objects so
    ground-truth durations can be measured; leave it off to skip the ~338 MB read.
    """
    rows = get_tinystress_rows(parquet_path, prompt_map_path)

    audio_by_id = {}
    if include_gt_audio:
        audio_by_id = _read_tinystress_audio(parquet_path, [r["id"] for r in rows])

    metainfo = []
    for row in rows:
        gt_text = " " + tinystress_emphasized_text(row["words"], row["emphasis_indices"])
        metainfo.append(
            (
                row["utt"],
                row["prompt_text"],
                row["prompt_wav"],
                gt_text,
                audio_by_id.get(row["id"]),
            )
        )
    return metainfo


def get_tinystress_testset(parquet_path, prompt_map_path, generated_dir, eval_ground_truth=False):
    """Return every test item; fail if a generation is missing."""
    rows = get_tinystress_rows(parquet_path, prompt_map_path)
    embedded = _read_tinystress_audio(parquet_path, [row["id"] for row in rows]) if eval_ground_truth else {}
    result = []
    missing = []
    for row in rows:
        if eval_ground_truth:
            wav = embedded[row["id"]]
            wav.name = row["utt"] + ".wav"
        else:
            wav = os.path.join(generated_dir, row["utt"] + ".wav")
            if not os.path.isfile(wav):
                missing.append(wav)
                continue
        result.append((wav, row["prompt_wav"], row["transcription"]))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} of {len(rows)} TinyStress generations. First: {missing[:5]}")
    return result

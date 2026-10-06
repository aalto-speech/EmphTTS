"""Training data for the duration predictor.

Every source goes through ``datasets.load_dataset``, so it can be a Hugging Face Hub repository (e.g. the Emilia
WebDataset shards in ``amphion/Emilia-Dataset``) or local files through a ``datasets`` builder
(``path: webdataset`` / ``parquet`` with local ``data_files``). Pretraining streams its sources; fine-tuning and
validation load them as map-style datasets.
"""

import io
import random

import librosa
import torch
from datasets import Audio, Features, Value, interleave_datasets, load_dataset
from datasets.distributed import split_dataset_by_node
from omegaconf import OmegaConf

from emphtts.tts.model.dataset import HFDataset
from emphtts.tts.model.modules import MelSpec

MIN_SECONDS, MAX_SECONDS = 0.3, 30.0  # same duration filter as the map-style F5-TTS datasets

AUDIO_TEXT = Features({"audio": Audio(decode=False), "text": Value("string")})


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _load_kwargs(source):
    kwargs = dict(split=source["split"])
    if source.get("name"):
        kwargs["name"] = source["name"]
    if source.get("data_files"):
        kwargs["data_files"] = _plain(source["data_files"])
    return kwargs


def load_streaming_source(source):
    """Stream one source as undecoded ``audio`` bytes and a ``text`` string.

    ``source`` has ``path``, ``split``, ``audio_column`` and ``text_column`` and optional ``name`` and ``data_files``.
    A dotted ``text_column`` reads a nested field, e.g. ``json.text`` for Emilia's per-utterance JSON.
    """
    audio_column = source["audio_column"]
    text_keys = source["text_column"].split(".")

    dataset = load_dataset(source["path"], streaming=True, **_load_kwargs(source))
    dataset = dataset.cast_column(audio_column, Audio(decode=False))

    def to_audio_text(example):
        text = example
        for key in text_keys:
            text = text[key]
        return {"audio": example[audio_column], "text": text}

    return dataset.map(to_audio_text, remove_columns=list(dataset.features), features=AUDIO_TEXT)


def build_streaming_dataset(
    sources,
    mel_spec_kwargs,
    *,
    probabilities=None,
    seed=666,
    shuffle_buffer=1000,
    rank=0,
    world_size=1,
):
    """Interleave ``sources``, keep this process's share, shuffle, and decode to mel spectrograms.

    Several sources are interleaved with ``probabilities`` (``None`` alternates between them) and the
    stream ends when the first source is exhausted. Items are ``{"mel_spec": (n_mel, T), "text": str}``.
    """
    streams = [load_streaming_source(source) for source in sources]
    if len(streams) == 1:
        dataset = streams[0]
    else:
        dataset = interleave_datasets(streams, probabilities=_plain(probabilities), seed=seed)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)

    mel_spec = MelSpec(**_plain(mel_spec_kwargs))
    target_sample_rate = mel_spec.target_sample_rate
    frame_seconds = mel_spec.hop_length / target_sample_rate

    def decode(example):
        audio, sample_rate = librosa.load(io.BytesIO(example["audio"]["bytes"]), sr=None)
        if sample_rate != target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
        mel = mel_spec(torch.from_numpy(audio).float().unsqueeze(0)).squeeze(0)  # (n_mel, T)
        return {"mel_spec": mel, "text": example["text"]}

    dataset = dataset.map(decode, remove_columns=["audio"])
    return dataset.filter(lambda example: MIN_SECONDS <= example["mel_spec"].shape[-1] * frame_seconds <= MAX_SECONDS)


def load_map_source(source, num_proc=None):
    """Load one source as a map-style dataset with undecoded ``audio`` and ``text`` columns.

    With ``fraction`` set, a fixed random subset is kept: ``random.seed(seed)`` then
    ``random.sample(range(num_rows), int(fraction * num_rows))``, with ``seed`` defaulting to 666.
    """
    dataset = load_dataset(source["path"], num_proc=num_proc, **_load_kwargs(source))
    dataset = dataset.cast_column(source["audio_column"], Audio(decode=False))
    if source["audio_column"] != "audio":
        dataset = dataset.rename_column(source["audio_column"], "audio")
    if source["text_column"] != "text":
        dataset = dataset.rename_column(source["text_column"], "text")
    dataset = dataset.select_columns(["audio", "text"])

    if source.get("fraction"):
        random.seed(source.get("seed", 666))
        indices = random.sample(range(dataset.num_rows), int(source["fraction"] * dataset.num_rows))
        dataset = dataset.select(indices)
    return dataset


def build_map_dataset(sources, mel_spec_kwargs, *, seed=666, num_proc=None):
    """Interleave map-style sources one item at a time until the first is exhausted (``interleave_datasets``)."""
    datasets = [load_map_source(source, num_proc=num_proc) for source in sources]
    for source, dataset in zip(sources, datasets):
        print(f"{source['path']} ({source['split']}): {dataset.num_rows} samples")
    dataset = datasets[0] if len(datasets) == 1 else interleave_datasets(datasets, seed=seed)
    return HFDataset(dataset, **_plain(mel_spec_kwargs))


def load_validation_dataset(cfg):
    """Map-style validation split (``datasets.validation``), e.g. LibriTTS-R dev-clean from the Hub."""
    validation = dict(_plain(cfg.datasets.validation))
    validation.setdefault("audio_column", "audio")
    validation.setdefault("text_column", "text")
    dataset = load_map_source(validation)
    return HFDataset(dataset, **_plain(cfg.model.mel_spec))

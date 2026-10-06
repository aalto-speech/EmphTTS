import io
import random

import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import datasets as hf_datasets
from emphtts.tts.model.modules import MelSpec


class GRPOParquetDataset(Dataset):
    """
    Map-style dataset that wraps any HuggingFace dataset whose audio column
    is stored as raw bytes (not pre-decoded).

    Loading examples:

        # Local parquet files
        GRPOParquetDataset(path="parquet", data_files="data/*.parquet")

        # HuggingFace Hub repo (default config)
        GRPOParquetDataset(path="username/my-tts-dataset")

        # HuggingFace Hub repo with a named config/subset
        GRPOParquetDataset(path="username/my-tts-dataset", name="default")

    Each item returns:
        mel_spec    : (n_mel, T)  channels-first mel spectrogram
        text        : str         transcript of the prompt utterance
        target_text : str         transcript randomly drawn from a different item

    Args:
        path              : first argument to `datasets.load_dataset`
                            (HF repo id, or "parquet" for local files)
        name              : dataset configuration/subset name (e.g. "default");
                            required for datasets that have named configs
        data_files        : passed to `load_dataset` when loading local files
        split             : dataset split to use (default "train")
        target_sample_rate: output sample rate for mel computation
        n_mel_channels    : number of mel bins
        hop_length        : STFT hop length
        n_fft             : FFT size
        win_length        : STFT window length
        mel_spec_type     : "vocos"
        audio_column      : name of the audio column
        text_column       : name of the text column
        num_sample        : optional subset size (int) or fraction (float in (0, 1))
    """

    def __init__(
        self,
        path: str,
        name: str | None = None,
        data_files=None,
        split: str = "train",
        target_sample_rate: int = 24_000,
        n_mel_channels: int = 100,
        hop_length: int = 256,
        n_fft: int = 1024,
        win_length: int = 1024,
        mel_spec_type: str = "vocos",
        audio_column: str = "audio",
        text_column: str = "text",
        num_sample: float | int | None = None,
    ):
        super().__init__()

        self.target_sample_rate = target_sample_rate
        self.audio_column = audio_column
        self.text_column = text_column

        # Load dataset; keep audio as raw bytes (no automatic decoding)
        load_kwargs = dict(split=split, keep_in_memory=True)
        if name is not None:
            load_kwargs["name"] = name
        if data_files is not None:
            load_kwargs["data_files"] = data_files

        self.dataset = hf_datasets.load_dataset(path, **load_kwargs) \
            .cast_column(audio_column, hf_datasets.Audio(decode=False))

        if num_sample is not None:
            n = int(num_sample * len(self.dataset)) if 0 < num_sample < 1 else int(num_sample)
            self.dataset = self.dataset.shuffle(seed=666).take(n)

        # Pre-cache all texts for O(1) random target sampling (no audio I/O)
        self._texts = self.dataset[text_column]

        # Mel spectrogram extractor (CPU)
        self.mel_spec = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _decode_audio(self, audio_bytes: bytes) -> torch.Tensor:
        """Decode raw audio bytes to a (1, T) float32 tensor at target_sample_rate."""
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        if sr != self.target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sample_rate)
        return torch.from_numpy(audio).float().unsqueeze(0)  # (1, T)

    def _wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """(1, T) → (n_mel, T)"""
        with torch.no_grad():
            mel = self.mel_spec(wav)   # (1, n_mel, T)
        return mel.squeeze(0)          # (n_mel, T)

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        audio_bytes = item[self.audio_column]["bytes"]
        wav = self._decode_audio(audio_bytes)               # (1, T)

        duration_s = wav.shape[-1] / self.target_sample_rate
        if duration_s < 0.3 or duration_s > 30.0:
            return self[random.randint(0, len(self) - 1)]

        mel = self._wav_to_mel(wav)                         # (n_mel, T)
        text = item[self.text_column]                       # raw string

        # Sample target text from a different item (text only, no audio I/O)
        target_idx = idx
        while target_idx == idx:
            target_idx = random.randint(0, len(self) - 1)
        target_text = self._texts[target_idx]               # raw string

        return dict(
            mel_spec=mel,
            text=text,
            target_text=target_text,
        )


def grpo_collate_fn(batch: list[dict]) -> dict:
    """
    Collate a list of items from GRPOParquetDataset into a batch dict.

    Returns:
        mel          : (B, n_mel, T_max)  right-padded with zeros
        mel_lengths  : (B,)               original frame counts
        text         : list[str]          B prompt transcripts
        text_lengths : (B,)               prompt text lengths (in chars)
        target_text  : list[str]          B target transcripts
    """
    mel_specs   = [item["mel_spec"] for item in batch]          # each (n_mel, T_i)
    mel_lengths = torch.tensor([m.shape[-1] for m in mel_specs], dtype=torch.long)
    max_T       = mel_lengths.amax().item()

    padded_mels = []
    for m in mel_specs:
        pad_len = max_T - m.shape[-1]
        padded_mels.append(F.pad(m, (0, pad_len), value=0.0))
    mel = torch.stack(padded_mels)  # (B, n_mel, T_max)

    text         = [item["text"]        for item in batch]
    target_text  = [item["target_text"] for item in batch]
    text_lengths = torch.tensor([len(t) for t in text], dtype=torch.long)

    return dict(
        mel=mel,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
        target_text=target_text,
    )

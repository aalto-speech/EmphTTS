import io

import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from emphtts.tts.model.modules import MelSpec
from emphtts.tts.model.utils import default


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        if 'duration' in row.keys():
            return row['duration'] * self.target_sample_rate / self.hop_length
        audio_bytes = row["audio"]['bytes']
        return sf.info(io.BytesIO(audio_bytes)).duration * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio_bytes = row["audio"]['bytes']

        audio, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)

        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        if sample_rate != self.target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.target_sample_rate)

        audio_tensor = torch.from_numpy(audio).float()

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        item = dict(
            mel_spec=mel_spec,
            text=text,
        )
        return item


class CustomWidsDataset(Dataset):
    """Random-access WebDataset shards described by a WIDS index JSON.

    Each sample needs a ``.flac`` payload and a ``.json`` metadata record with a transcript
    (``transcript``, ``whisper_transcript`` or ``text``) and either ``duration`` or ``start``/``end``.
    """

    def __init__(
        self,
        wids_index_json_path: str,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        mel_spec_module: nn.Module | None = None,
    ):
        import wids  # shipped with webdataset; only needed for this dataset type

        # Load using ShardListDataset for random access
        self.dataset = wids.ShardListDataset(wids_index_json_path)

        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = default(
            mel_spec_module,
            MelSpec(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mel_channels=n_mel_channels,
                target_sample_rate=target_sample_rate,
                mel_spec_type=mel_spec_type,
            ),
        )

    @staticmethod
    def _metadata(sample):
        return sample[".json"] if ".json" in sample else sample["..json"]

    def get_frame_len(self, index):
        metadata = self._metadata(self.dataset[index])
        duration = metadata['duration'] if 'duration' in metadata else (metadata["end"] - metadata["start"])
        return duration * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Get sample using random access
        sample = self.dataset[index]
        metadata = self._metadata(sample)

        if 'transcript' in metadata:
            text = metadata['transcript']
        elif 'whisper_transcript' in metadata:
            text = metadata['whisper_transcript']
        elif 'text' in metadata:
            text = metadata['text']
        else:
            raise ValueError(f"Cannot find transcript key in the metadata. Metadata keys: {metadata.keys()}")

        duration = metadata['duration'] if 'duration' in metadata else (metadata["end"] - metadata["start"])

        # Filter by duration
        if not (0.3 <= duration <= 30):
            return self.__getitem__((index + 1) % len(self))

        try:
            audio, source_sample_rate = librosa.load(sample['.flac'], sr=None)  # mono by default
        except Exception:
            print(f"{sample['__key__']} from {sample['__shard__']} is broken.")
            return self.__getitem__((index + 1) % len(self))

        if source_sample_rate != self.target_sample_rate:
            audio = librosa.resample(audio, orig_sr=source_sample_rate, target_sr=self.target_sample_rate)
        audio = torch.from_numpy(audio).unsqueeze(0)  # 't -> 1 t'
        mel_spec = self.mel_spectrogram(audio)
        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        item = {
            "mel_spec": mel_spec,
            "text": text,
        }
        return item


# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = int(frames_threshold)
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(dataset_name, dataset_type="HFDataset", mel_spec_kwargs=None, index_path=None):
    """Load a Hugging Face audio dataset or a local WIDS index."""
    mel_spec_kwargs = mel_spec_kwargs or {}
    if dataset_type == "HFDataset":
        from datasets import Audio, load_dataset as load_hf_dataset

        dataset = load_hf_dataset(dataset_name, split="train").cast_column("audio", Audio(decode=False))
        if "sentence" in dataset.column_names and "text" not in dataset.column_names:
            dataset = dataset.rename_column("sentence", "text")
        return HFDataset(dataset, **mel_spec_kwargs)
    if dataset_type == "CustomWidsDataset":
        if not index_path:
            raise ValueError("CustomWidsDataset requires index_path")
        return CustomWidsDataset(index_path, **mel_spec_kwargs)
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    collated = dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
    )

    return collated

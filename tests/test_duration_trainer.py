import io
import os
import random

import numpy as np
import pytest
import soundfile as sf
import torch
from datasets import Dataset
from torch.utils.data import IterableDataset

from emphtts.duration.data import load_map_source
from emphtts.duration.duration_predictor import SpeechLengthPredictor
from emphtts.duration.trainer import Trainer, masked_cross_entropy_loss, masked_l1_loss
from emphtts.tts.model.dataset import DynamicBatchSampler


class TinyStream(IterableDataset):
    """A finite stream of random mel/text items, like one pass over a streamed dataset."""

    def __init__(self, n_items=6):
        self.n_items = n_items

    def __iter__(self):
        generator = torch.Generator().manual_seed(0)
        for i in range(self.n_items):
            frames = 20 + 5 * i
            yield {"mel_spec": torch.randn(100, frames, generator=generator), "text": "some *text* " * (i + 1)}


@pytest.mark.parametrize("loss_fn", ["L1", "CE", "L1_and_CE"])
def test_streaming_training_saves_and_rotates_checkpoints(tmp_path, vocab, loss_fn):
    torch.manual_seed(0)
    vocab_char_map, vocab_size = vocab
    n_class = 301 if loss_fn != "L1" else 1
    model = SpeechLengthPredictor(vocab_size=vocab_size, hidden_dim=32, n_head=2, n_text_layer=1, n_cross_layer=1,
                                  output_dim=n_class)  # n_class 1: the forward squeezes to (B, T) lengths

    trainer = Trainer(
        model, vocab_size, vocab_char_map, loss_fn=loss_fn, n_class=n_class, total_updates=5,
        num_warmup_updates=1, batch_size=2, save_per_updates=2, last_per_updates=3,
        keep_last_n_checkpoints=1, checkpoint_path=str(tmp_path), logger=None,
    )
    validation = [{"mel_spec": torch.randn(100, 30), "text": "valid"}] * 2
    trainer.train_streaming(TinyStream(), validation, num_workers=0)  # 3 batches per pass, so the stream is re-read

    assert sorted(os.listdir(tmp_path)) == ["model_4.pt", "model_last.pt"]
    assert torch.load(tmp_path / "model_last.pt", weights_only=True)["step"] == 5


def test_epoch_training_runs_all_epochs(tmp_path, vocab):
    vocab_char_map, vocab_size = vocab
    model = SpeechLengthPredictor(vocab_size=vocab_size, hidden_dim=32, n_head=2, n_text_layer=1, n_cross_layer=1,
                                  output_dim=301)
    trainer = Trainer(
        model, vocab_size, vocab_char_map, loss_fn="CE", epochs=3, num_warmup_updates=1, batch_size=2,
        save_per_updates=100, last_per_updates=100, checkpoint_path=str(tmp_path), logger=None, use_ema=True,
    )
    items = list(TinyStream(n_items=4))  # 2 batches per epoch
    trainer.train(items, items[:2], num_workers=0)

    checkpoint = torch.load(tmp_path / "model_last.pt", weights_only=True)
    assert checkpoint["step"] == 6
    assert "ema_model_state_dict" in checkpoint


def test_map_source_keeps_fixed_random_fraction(tmp_path):
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(2400, dtype=np.float32), 24000, format="WAV")
    rows = [{"mp3": {"bytes": buffer.getvalue(), "path": f"{i}.wav"}, "text": f"utt {i}", "extra": i} for i in range(200)]
    Dataset.from_list(rows).to_parquet(str(tmp_path / "part-0.parquet"))

    source = dict(path="parquet", data_files={"en": str(tmp_path / "*.parquet")}, split="en",
                  audio_column="mp3", text_column="text", fraction=0.05, seed=666)
    subset = load_map_source(source)

    random.seed(666)
    expected = random.sample(range(200), 10)
    assert subset.column_names == ["audio", "text"]
    assert subset["text"] == [f"utt {i}" for i in expected]


def test_masked_losses_ignore_padding():
    tar = torch.tensor([[2, 1, 0, 0, 0]])
    assert masked_l1_loss(torch.tensor([[2.0, 1.0, 0.0, 9.0, 9.0]]), tar.float()).item() == 0.0
    logits = torch.full((1, 5, 3), -10.0)
    logits[0, torch.arange(5), torch.tensor([2, 1, 0, 2, 2])] = 10.0
    assert masked_cross_entropy_loss(logits, tar).item() < 1e-6


class Lengths:
    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def get_frame_len(self, index):
        return self.lengths[index]


def test_dynamic_batch_sampler_respects_frame_budget():
    data = Lengths([30, 10, 50, 20, 40, 60])
    sampler = DynamicBatchSampler(torch.utils.data.SequentialSampler(data), frames_threshold=70, random_seed=0)
    batches = list(sampler)
    assert sorted(i for batch in batches for i in batch) == list(range(6))
    assert all(sum(data.lengths[i] for i in batch) <= 70 for batch in batches)

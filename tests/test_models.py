import torch
from omegaconf import OmegaConf

from emphtts.duration.duration_predictor import SpeechLengthPredictor, calculate_remaining_lengths
from emphtts.tts.infer.utils_infer import build_duration_model, build_f5_model, predict_total_mel_len

TINY_ARCH = dict(dim=32, depth=2, heads=2, dim_head=16, ff_mult=2, text_dim=16, conv_layers=1)


def tiny_f5(repo_root, vocab_char_map):
    cfg = OmegaConf.load(repo_root / "src/emphtts/tts/configs/EmphTTS.yaml")
    cfg.model.arch = OmegaConf.merge(cfg.model.arch, TINY_ARCH)
    return build_f5_model(cfg, vocab_char_map)


def test_cfm_forward_and_sample(repo_root, vocab):
    torch.manual_seed(0)
    vocab_char_map, _ = vocab
    model = tiny_f5(repo_root, vocab_char_map)

    mel = torch.randn(2, 40, 100)
    loss, _, pred = model(mel, text=["a *short* one", "and another"], lens=torch.tensor([40, 30]))
    assert torch.isfinite(loss)
    assert pred.shape == mel.shape

    out, trajectory = model.sample(
        cond=mel[:1, :20], text=["ref text *now*"], duration=50, steps=2, cfg_strength=2.0, sway_sampling_coef=-1.0
    )
    assert out.shape == (1, 50, 100)
    assert torch.equal(out[:, :20], mel[:1, :20])  # the prompt region is kept
    assert trajectory.shape[0] == 3


def test_duration_predictor_shapes(repo_root):
    cfg = OmegaConf.load(repo_root / "src/emphtts/duration/config/duration_predictor.yaml")
    cfg.model.hidden_dim, cfg.model.n_text_layer, cfg.model.n_cross_layer = 32, 1, 1
    model = build_duration_model(cfg, vocab_size=79)
    assert isinstance(model, SpeechLengthPredictor)
    logits = model(torch.randint(0, 79, (2, 12)), torch.randn(2, 30, 100))
    assert logits.shape == (2, 30, cfg.loss.n_class)


def test_remaining_lengths():
    remaining = calculate_remaining_lengths(torch.tensor([3, 1]))
    assert remaining.tolist() == [[2, 1, 0], [0, 0, 0]]


class FixedLogits(torch.nn.Module):
    """Duration predictor stub whose last-frame logits peak at ``best_bin``."""

    def __init__(self, best_bin, n_class=301):
        super().__init__()
        self.best_bin, self.n_class = best_bin, n_class

    def forward(self, text_ids, mel):
        logits = torch.zeros(1, mel.shape[1], self.n_class)
        logits[:, -1, self.best_bin] = 1.0
        return logits


def test_predict_total_mel_len_keeps_paper_conversion(vocab):
    # Locks the evaluation behaviour behind the reported results: classes read as 0.1 s bins,
    # search limited to [0.7, 1.7] x the heuristic length.
    vocab_char_map, _ = vocab
    ref_mel, ref_len = torch.zeros(1, 100, 100), 100
    heuristic_len = ref_len + 375  # 375 frames = 4.0 s -> heuristic bin 40, window [28, 69)

    total = predict_total_mel_len(FixedLogits(50), vocab_char_map, "a b", ref_mel, ref_len, heuristic_len)
    assert total == ref_len + int(50 / 10 * 24000 / 256)

    clamped_low = predict_total_mel_len(FixedLogits(5), vocab_char_map, "a b", ref_mel, ref_len, heuristic_len)
    assert clamped_low == ref_len + int(28 / 10 * 24000 / 256)

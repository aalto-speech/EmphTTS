"""Shared F5-TTS inference helpers for EmphTTS."""
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from safetensors.torch import load_file
from vocos import Vocos

from emphtts.duration.duration_predictor import SpeechLengthPredictor
from emphtts.tts.model import CFM, DiT
from emphtts.tts.model.utils import get_tokenizer, list_str_to_idx

SAMPLE_RATE = 24000
HOP_LENGTH = 256
cfg_strength = 2.0
nfe_step = 32
sway_sampling_coef = -1.0


def load_vocoder(local_path=None, device=None):
    """Load Vocos from its Hub release or an explicitly supplied local directory."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if local_path:
        config_path = Path(local_path) / "config.yaml"
        model_path = Path(local_path) / "pytorch_model.bin"
    else:
        repo = "charactr/vocos-mel-24khz"
        config_path = hf_hub_download(repo, "config.yaml")
        model_path = hf_hub_download(repo, "pytorch_model.bin")
    vocoder = Vocos.from_hparams(str(config_path))
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    from vocos.feature_extractors import EncodecFeatures
    if isinstance(vocoder.feature_extractor, EncodecFeatures):
        state.update({"feature_extractor.encodec." + key: value for key, value in vocoder.feature_extractor.encodec.state_dict().items()})
    vocoder.load_state_dict(state)
    return vocoder.eval().to(device)


def load_checkpoint(model, checkpoint_path, device, use_ema=True, dtype=None):
    path = Path(checkpoint_path)
    if path.suffix == ".safetensors":
        state = load_file(str(path), device="cpu")
        checkpoint = {"ema_model_state_dict" if use_ema else "model_state_dict": state}
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if use_ema:
        state = {key.removeprefix("ema_model."): value for key, value in checkpoint["ema_model_state_dict"].items() if key not in ("initted", "step", "update")}
    else:
        state = checkpoint["model_state_dict"]
    for key in ("mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"):
        state.pop(key, None)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model.load_state_dict(state)
    return model.to(device).eval()


def build_f5_model(cfg, vocab_char_map, ode_method="euler"):
    """Build an untrained CFM/DiT model from an F5-TTS config (``model.arch`` and ``model.mel_spec``)."""
    if cfg.model.backbone != "DiT" or cfg.model.mel_spec.mel_spec_type != "vocos":
        raise ValueError("This release supports the DiT backbone with Vocos mel features.")
    mel = OmegaConf.to_container(cfg.model.mel_spec, resolve=True)
    arch = OmegaConf.to_container(cfg.model.arch, resolve=True)
    return CFM(
        transformer=DiT(**arch, text_num_embeds=len(vocab_char_map), mel_dim=mel["n_mel_channels"]),
        mel_spec_kwargs=mel,
        odeint_kwargs=dict(method=ode_method),
        vocab_char_map=vocab_char_map,
    )


def build_duration_model(cfg, vocab_size):
    """Build an untrained duration predictor from a duration config (``model`` and ``loss.n_class``)."""
    return SpeechLengthPredictor(
        vocab_size=vocab_size,
        n_mel=cfg.model.mel_spec.n_mel_channels,
        hidden_dim=cfg.model.hidden_dim,
        n_head=cfg.model.n_head,
        n_text_layer=cfg.model.n_text_layer,
        n_cross_layer=cfg.model.n_cross_layer,
        output_dim=cfg.loss.n_class,
    )


def load_f5_model(config_path, checkpoint_path, vocab_path, device, use_ema=True):
    cfg = OmegaConf.load(config_path)
    vocab_map, _ = get_tokenizer(vocab_path, "custom")
    model = build_f5_model(cfg, vocab_map)
    mel = OmegaConf.to_container(cfg.model.mel_spec, resolve=True)
    return load_checkpoint(model, checkpoint_path, device, use_ema), vocab_map, mel


def load_duration_model(config_path, checkpoint_path, vocab_size, device, use_ema=True):
    model = build_duration_model(OmegaConf.load(config_path), vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if use_ema and ("ema_model_state_dict" in checkpoint or "ema_state_dict" in checkpoint):
        ema_state = checkpoint.get("ema_model_state_dict", checkpoint.get("ema_state_dict"))
        state = {key.removeprefix("ema_model."): value for key, value in ema_state.items() if key not in ("initted", "step", "update")}
    else:
        state = checkpoint["model_state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval()


def reference_mel(audio_path, mel_spec, device, target_rms=0.1):
    audio, sr = torchaudio.load(audio_path)
    audio = audio.mean(dim=0, keepdim=True)
    rms = audio.square().mean().sqrt()
    if rms.item() == 0:
        raise ValueError(f"Reference audio is silent: {audio_path}")
    if rms < target_rms:
        audio = audio * target_rms / rms
    sample_rate = mel_spec.target_sample_rate
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    audio = audio.to(device)
    ref_mel_len = audio.shape[-1] // mel_spec.hop_length
    mel = mel_spec(audio).permute(0, 2, 1)
    return mel, ref_mel_len, rms


def heuristic_total_mel_len(ref_len, ref_text, gen_text, speed=1.0):
    if not ref_text.strip() or speed <= 0:
        raise ValueError("Reference text must be nonempty and speed must be positive.")
    return ref_len + max(1, int(ref_len * len(gen_text.encode("utf-8")) / len(ref_text.encode("utf-8")) / speed))


def predict_total_mel_len(predictor, vocab_map, text, ref_mel, ref_len, heuristic_len, sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH):
    """Total mel length (prompt + generation) from the duration predictor's remaining-length class.

    The search is limited to [0.7, 1.7] x the text-length heuristic. Classes are read as 0.1 s bins here, as in the
    evaluation behind the reported results; training labels use ``loss.n_frame_per_class`` = 10 mel frames
    (about 0.107 s), so this conversion is kept deliberately rather than derived from the duration config.
    """
    text_ids = list_str_to_idx([text], vocab_map).to(ref_mel.device)
    logits = predictor(text_ids, ref_mel)[:, -1, :]
    linear_bin = (heuristic_len - ref_len) * hop_length / sample_rate * 10
    lower = max(0, int(linear_bin * 0.7))
    upper = min(logits.shape[-1], max(lower + 1, int(linear_bin * 1.7) + 1))
    logits[:, :lower] = float("-inf")
    logits[:, upper:] = float("-inf")
    remaining_bin = logits.argmax(dim=-1).item()
    return ref_len + max(1, int(remaining_bin / 10 * sample_rate / hop_length))


def synthesize(model, vocoder, ref_mel, ref_len, rms, text, total_len, *, steps=nfe_step, cfg=cfg_strength, sway=sway_sampling_coef, seed=None, target_rms=0.1):
    with torch.inference_mode():
        generated, _ = model.sample(cond=ref_mel, text=[text], duration=total_len, lens=torch.tensor([ref_len], device=ref_mel.device), steps=steps, cfg_strength=cfg, sway_sampling_coef=sway, seed=seed)
        mel = generated[:, ref_len:total_len, :].permute(0, 2, 1).float()
        wav = vocoder.decode(mel).cpu()
        if rms < target_rms:
            wav *= rms / target_rms
    return wav

"""Single-utterance emphasis-controlled synthesis."""
import argparse
from pathlib import Path

import torch
import torchaudio

from emphtts.tts.infer.utils_infer import (
    heuristic_total_mel_len, load_duration_model, load_f5_model, load_vocoder,
    predict_total_mel_len, reference_mel, synthesize,
)


def main():
    parser = argparse.ArgumentParser(description="Synthesize speech with *emphasized* words")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--text", required=True, help="Use *word* to request emphasis")
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-config")
    parser.add_argument("--duration-checkpoint")
    parser.add_argument("--vocos-dir", help="Optional local Vocos directory")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if bool(args.duration_config) != bool(args.duration_checkpoint):
        parser.error("--duration-config and --duration-checkpoint must be supplied together")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab_map, mel = load_f5_model(args.config, args.checkpoint, args.vocab, device)
    vocoder = load_vocoder(args.vocos_dir, device)
    ref_mel, ref_len, rms = reference_mel(args.ref_audio, model.mel_spec, device)
    ref_text = args.ref_text.rstrip() + " "
    text = ref_text + args.text
    total_len = heuristic_total_mel_len(ref_len, ref_text, args.text, args.speed)
    if args.duration_config:
        predictor = load_duration_model(args.duration_config, args.duration_checkpoint, len(vocab_map), device)
        total_len = predict_total_mel_len(predictor, vocab_map, text, ref_mel, ref_len, total_len, mel["target_sample_rate"], mel["hop_length"])
    wav = synthesize(model, vocoder, ref_mel, ref_len, rms, text, total_len, steps=args.steps, cfg=args.cfg, sway=args.sway, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output), wav, mel["target_sample_rate"])
    print(output)


if __name__ == "__main__":
    main()

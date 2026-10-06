"""Generate the full TinyStress-15K test split, optionally using a duration predictor."""
import argparse
from pathlib import Path

import torch
import torchaudio
from accelerate import Accelerator
from tqdm import tqdm

from emphtts.tts.eval.utils_eval import TINYSTRESS_DEFAULT_PROMPT_MAP, get_tinystress_testset_metainfo
from emphtts.tts.infer.utils_infer import (
    heuristic_total_mel_len, load_duration_model, load_f5_model, load_vocoder,
    predict_total_mel_len, reference_mel, synthesize,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--tinystress-parquet", required=True)
    parser.add_argument("--tinystress-prompt-map", default=TINYSTRESS_DEFAULT_PROMPT_MAP)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration-config")
    parser.add_argument("--duration-checkpoint")
    parser.add_argument("--vocos-dir")
    parser.add_argument("--true-duration", action="store_true", help="Report duration-predictor MAE against the embedded test WAV durations")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if bool(args.duration_config) != bool(args.duration_checkpoint):
        parser.error("--duration-config and --duration-checkpoint must be supplied together")
    if args.true_duration and not args.duration_config:
        parser.error("--true-duration measures the duration predictor; also pass --duration-config and --duration-checkpoint")
    accelerator = Accelerator()
    device = accelerator.device
    metainfo = get_tinystress_testset_metainfo(args.tinystress_parquet, args.tinystress_prompt_map, args.true_duration)
    model, vocab_map, mel = load_f5_model(args.config, args.checkpoint, args.vocab, device)
    vocoder = load_vocoder(args.vocos_dir, device)
    predictor = load_duration_model(args.duration_config, args.duration_checkpoint, len(vocab_map), device) if args.duration_config else None
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    with accelerator.split_between_processes(metainfo) as rows:
        for utt, ref_text, ref_audio, gen_text, gt_audio in tqdm(rows, disable=not accelerator.is_local_main_process):
            ref_mel, ref_len, rms = reference_mel(ref_audio, model.mel_spec, device)
            ref_text = ref_text.rstrip() + " "
            text = ref_text + gen_text
            heuristic = heuristic_total_mel_len(ref_len, ref_text, gen_text, args.speed)
            total_len = heuristic
            if predictor:
                total_len = predict_total_mel_len(predictor, vocab_map, text, ref_mel, ref_len, heuristic, mel["target_sample_rate"], mel["hop_length"])
            if args.true_duration:
                gt_audio.seek(0)
                audio, sr = torchaudio.load(gt_audio)
                if sr != mel["target_sample_rate"]:
                    audio = torchaudio.functional.resample(audio, sr, mel["target_sample_rate"])
                gold_len = ref_len + int(audio.shape[-1] / mel["hop_length"] / args.speed)
                errors.append(abs(gold_len - total_len) * mel["hop_length"] / mel["target_sample_rate"])
            wav = synthesize(model, vocoder, ref_mel, ref_len, rms, text, total_len, steps=args.steps, cfg=args.cfg, sway=args.sway, seed=args.seed)
            torchaudio.save(str(output / f"{utt}.wav"), wav, mel["target_sample_rate"])
    accelerator.wait_for_everyone()
    if args.true_duration:
        local_stats = torch.tensor([[sum(errors), len(errors)]], dtype=torch.float64, device=device)
        stats = accelerator.gather(local_stats)
        if accelerator.is_main_process:
            print(f"Duration MAE: {stats[:, 0].sum().item() / stats[:, 1].sum().item():.4f} s")
    if accelerator.is_main_process:
        print(f"Generated {len(metainfo)} TinyStress items in {output}")


if __name__ == "__main__":
    main()

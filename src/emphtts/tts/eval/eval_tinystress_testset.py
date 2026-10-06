"""Score TinyStress-15K intelligibility (WER/CER) or speaker similarity."""
import argparse
import json
import string
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from emphtts.tts.eval.utils_eval import TINYSTRESS_DEFAULT_PROMPT_MAP, get_tinystress_testset


def normalized(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()


def score_wer(rows, model_name, device):
    from faster_whisper import WhisperModel
    from jiwer import wer, cer
    model = WhisperModel(model_name, device=device, compute_type="float16" if device == "cuda" else "int8")
    for wav, _, truth in tqdm(rows, desc="WER/CER"):
        if hasattr(wav, "seek"):
            wav.seek(0)
        segments, _ = model.transcribe(wav, beam_size=5, language="en", without_timestamps=True)
        hypothesis = " ".join(segment.text for segment in segments)
        yield {"utt": Path(getattr(wav, "name", wav)).stem, "truth": truth, "hypothesis": hypothesis, "wer": wer(normalized(truth), normalized(hypothesis)), "cer": cer(normalized(truth), normalized(hypothesis))}


def load_speaker_model(checkpoint):
    """WavLM-ECAPA speaker verifier with every ECAPA head weight taken from ``checkpoint``.

    The WavLM feature extractor comes from S3PRL, so only ``feature_extract.*`` keys (and the training-only
    ``loss_calculator.*`` head) may differ from the checkpoint; anything else means the wrong checkpoint.
    """
    from emphtts.tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" not in state:
        raise ValueError(f"{checkpoint} is not a WavLM-ECAPA speaker-verification checkpoint (no 'model' entry).")
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    missing = [key for key in missing if not key.startswith("feature_extract.")]
    unexpected = [key for key in unexpected if not key.startswith(("feature_extract.", "loss_calculator."))]
    if missing or unexpected:
        raise ValueError(
            f"{checkpoint} does not match the WavLM-ECAPA speaker model: "
            f"missing {missing[:5]}, unexpected {unexpected[:5]}."
        )
    return model


def score_sim(rows, checkpoint, device):
    model = load_speaker_model(checkpoint).to(device).eval()
    for wav, prompt, _ in tqdm(rows, desc="Speaker similarity"):
        if hasattr(wav, "seek"):
            wav.seek(0)
        generated, sr_gen = torchaudio.load(wav)
        reference, sr_ref = torchaudio.load(prompt)
        generated = generated.mean(0, keepdim=True)
        reference = reference.mean(0, keepdim=True)
        if sr_gen != 16000:
            generated = torchaudio.functional.resample(generated, sr_gen, 16000)
        if sr_ref != 16000:
            reference = torchaudio.functional.resample(reference, sr_ref, 16000)
        with torch.no_grad():
            embedding_gen = model(generated.to(device))
            embedding_ref = model(reference.to(device))
        yield {"utt": Path(getattr(wav, "name", wav)).stem, "sim": F.cosine_similarity(embedding_gen, embedding_ref)[0].item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("wer", "sim"), required=True)
    parser.add_argument("--tinystress-parquet", required=True)
    parser.add_argument("--tinystress-prompt-map", default=TINYSTRESS_DEFAULT_PROMPT_MAP)
    parser.add_argument("--gen-wav-dir", required=True, help="Generated WAVs; results are written here too")
    parser.add_argument("--whisper-model", default="large-v3", help="faster-whisper model name or local model directory")
    parser.add_argument("--wavlm-checkpoint", default="checkpoints/wavlm_large_finetune.pth",
                        help="WavLM-ECAPA speaker-verification weights for --task sim (see README, Checkpoints)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--eval-ground-truth", action="store_true", help="Score embedded reference WAVs for WER/CER")
    args = parser.parse_args()
    if args.eval_ground_truth and args.task != "wer":
        parser.error("--eval-ground-truth only applies to WER/CER")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.task == "sim" and not Path(args.wavlm_checkpoint).is_file():
        parser.error(
            f"WavLM-ECAPA checkpoint not found at {args.wavlm_checkpoint}; download wavlm_large_finetune.pth "
            "as described in the README (Checkpoints) or pass --wavlm-checkpoint"
        )
    rows = get_tinystress_testset(args.tinystress_parquet, args.tinystress_prompt_map, args.gen_wav_dir, args.eval_ground_truth)
    results = list(score_wer(rows, args.whisper_model, device) if args.task == "wer" else score_sim(rows, args.wavlm_checkpoint, device))
    output = Path(args.gen_wav_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Ground-truth scores get their own prefix so they never overwrite a system's results in the same directory.
    prefix = f"_gt_{args.task}" if args.eval_ground_truth else f"_{args.task}"
    with (output / f"{prefix}_results.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    metrics = {args.task: round(float(np.mean([row[args.task] for row in results])), 5)}
    if args.task == "wer":
        metrics["cer"] = round(float(np.mean([row["cer"] for row in results])), 5)
    summary = {"evaluated": len(results), "metrics": metrics}
    (output / f"{prefix}_results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()

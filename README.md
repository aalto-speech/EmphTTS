# EmphTTS: emphasis-controlled TTS with reinforcement learning

> **Paper:** [EmphTTS: an emphasis-control TTS with reinforcement learning](https://arxiv.org/abs/2609.27599) (arXiv:2609.27599).

## Installation

Use Python 3.10 or newer. First install a PyTorch/torchaudio pair that matches your CPU or CUDA setup, then:

```bash
python -m pip install -e ".[all]"          # everything: training, evaluation, GRPO and tests
# or only what you need, e.g.
python -m pip install -e ".[eval]"         # inference plus TinyStress scoring
```

Notes:

- **FFmpeg.** `datasets` 4.x decodes audio through TorchCodec, which needs FFmpeg 4–7 installed on the system.
- **Multiple GPUs.** The training scripts start a single process. For several GPUs, run the module with `accelerate launch -m ...` instead, e.g. `accelerate launch -m emphtts.duration.train`.

## Data

All data and model weights come from the Hugging Face Hub or from local files.

| Asset | Source | Used for |
|---|---|---|
| LibriTTS-R | [LibriTTS-R](https://www.openslr.org/141/) training splits, packed into local WebDataset shards (see below) | F5-TTS pretraining |
| Emilia EN | [`amphion/Emilia-Dataset`](https://huggingface.co/datasets/amphion/Emilia-Dataset)  | Duration pretraining (streamed) |
| Emilia EN subset | [`ylacombe/emilia-subset`](https://huggingface.co/datasets/ylacombe/emilia-subset) (3.4M utterances, ~245 GB) | Duration fine-tuning (a fixed 5%) |
| Expresso | [`ylacombe/expresso`](https://huggingface.co/datasets/ylacombe/expresso) | Duration fine-tuning, GRPO |
| VCTK | [`kth-tmh/vctk`](https://huggingface.co/datasets/kth-tmh/vctk) | GRPO |
| LibriTTS-R dev-clean | [`blabble-io/libritts_r`](https://huggingface.co/datasets/blabble-io/libritts_r) | Duration-predictor validation |
| TinyStress-15K | [`slprl/TinyStress-15K`](https://huggingface.co/datasets/slprl/TinyStress-15K) | Evaluation; voice prompts in `data/tinystress-15k/` |

F5-TTS was pretrained on LibriTTS-R packed into WebDataset `.tar` shards and read through a [WIDS](https://github.com/webdataset/webdataset) index (`datasets.dataset_type=CustomWidsDataset`). Each sample in a shard has two files:

- `<key>.flac` holds the 24 kHz audio.
- `<key>.json` holds the transcript in `text` and the length in seconds in `duration`.

F5-TTS can also train on any Hugging Face dataset with `audio` and `text` columns (the default `HFDataset`).

## Checkpoints

Two pretrained checkpoints have to be downloaded by hand. Put them in `checkpoints/` and the code looks for them there by default.

| File | Used for | Source |
|---|---|---|
| `checkpoints/wavlm_large_finetune.pth` | Speaker similarity (`scripts/eval_tinystress.sh --task sim`) | WavLM-Large speaker verifier from [UniSpeech](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification) |
| `checkpoints/campplus_cn_en_common.pt` | GRPO speaker-similarity reward | CAM++ from [3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker), hosted on [ModelScope](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced) |

Put the TTS model and duration predictor you train in `checkpoints/` too. The GRPO config expects them at `checkpoints/f5_model.pt` and `checkpoints/duration_model.pt`.

## Training

### F5-TTS

```bash
scripts/train_f5.sh datasets.name=your_hf_dataset
scripts/finetune_f5.sh checkpoints/f5_base.pt datasets.name=your_hf_dataset ckpts.save_dir=ckpts/finetuned
```

- **Config.** `src/emphtts/tts/configs/EmphTTS.yaml` holds the model architecture. Treat its dataset and schedule values as a template to adapt.
- **Data.** Mark emphasized words in the training transcripts with asterisks, as Expresso does (`I *cannot* answer that.`).
- **WIDS input.** For a local WIDS index, pass `datasets.dataset_type=CustomWidsDataset datasets.index_path=/path/to/index.json`.
- **Fine-tuning.** The checkpoint is copied into `ckpts.save_dir`, and checkpoints already in that directory take precedence. Use a fresh `ckpts.save_dir` for each run.

### Duration predictor

The duration predictor is trained in two stages, as in the paper:

```bash
# 1. Pretrain on Emilia EN, streamed from the Hub (85k updates, 48 utterances per batch)
scripts/train_duration.sh
# 2. Fine-tune on Expresso alternating with a fixed 5% of ylacombe/emilia-subset (21 epochs, lr 1e-5, EMA)
scripts/finetune_duration.sh DurPred_ckpts/emilia_en_48_CE/model_70000.pt
```

The configs are `src/emphtts/duration/config/duration_predictor.yaml` (pretraining) and `duration_finetune.yaml` (fine-tuning). Hydra overrides on the command line replace config values.

**Pretraining** streams `datasets.train` with `datasets.load_dataset(..., streaming=True)`. It runs for `optim.total_updates` and re-reads the stream with a new shuffle whenever it ends. Each entry under `datasets.train` is one source:

- `path`, `name`, `data_files` and `split` are passed to `datasets.load_dataset`.
- `audio_column` names the audio column.
- `text_column` names the transcript column. It may be a dotted path, such as Emilia's `json.text`.

When there are several sources, they are interleaved: they alternate by default, or follow `datasets.probabilities` if set. Streaming Emilia from the Hub depends on your bandwidth. To train from a local download of the same shards instead, override the source:

```bash
scripts/train_duration.sh datasets.train.emilia_en.path=webdataset \
  'datasets.train.emilia_en.data_files=/data/emilia/EN/*.tar'
```

**Fine-tuning** loads the `datasets.finetune` sources as regular (map-style) datasets and trains for `optim.epochs` epochs:

- **Emilia subset.** A source with `fraction` keeps a fixed random subset: `random.seed(seed)`, then `random.sample(range(num_rows), int(fraction * num_rows))`. For `ylacombe/emilia-subset` this is about 170k utterances, but the whole ~245 GB dataset is downloaded to select them.
- **Mixing.** The sources alternate one item at a time until the first runs out (`datasets.interleave_datasets`), so each epoch pairs all of Expresso with as many Emilia utterances.
- **Warmup.** Warmup stays at 20,000 updates as in the paper's run. With about 10k updates in total on one GPU, the learning rate is still rising when training ends.

**Checkpoints used in the paper.** The fine-tuning started from the 70k-update pretraining checkpoint, and GRPO started from the fine-tuned checkpoint after 4,000 updates (`model_4000.pt`). Both stages were launched with `accelerate launch --mixed_precision fp16`. Pretraining used 4 GPUs with 48 utterances each, and fine-tuning used one GPU. Since warmup and decay scale with the number of processes, use the same setup to reproduce them.

### Duration GRPO

With the checkpoints in `checkpoints/` (see [Checkpoints](#checkpoints)), run:

```bash
scripts/train_grpo.sh
# other locations: scripts/train_grpo.sh durpred.ckpt=... tts.ckpt=... reward.campplus_ckpt=...
```

By default the reward combines Wav2Vec2 intelligibility and CAMPPlus speaker similarity. The rollout texts come from the configured VCTK and Expresso sources.

For the emphasis-aware reward:

1. Clone [WhiStress](https://github.com/slp-rl/WhiStress).
2. Install its requirements and run its `download_weights.py`.
3. Run `scripts/train_grpo_with_whistress.sh WhiStress` with the same overrides as above.

The launcher puts the checkout on `PYTHONPATH` and switches the reward to Whisper WER plus WhiStress stress balanced accuracy.

## TinyStress-15K evaluation

Generate all 1,000 test utterances. Output files keep the dataset IDs, as in `tinystress_00042.wav`. Each item uses the prompt for its voice from `data/tinystress-15k/prompts.json`.

```bash
scripts/infer_tinystress.sh \
  --config src/emphtts/tts/configs/EmphTTS.yaml \
  --checkpoint checkpoints/f5_model.pt \
  --vocab data/vocab.txt \
  --tinystress-parquet data/tinystress-15k/test-00000-of-00001.parquet \
  --output-dir results/tinystress
```

Two optional flags apply:

- `--duration-config` and `--duration-checkpoint` enable the duration predictor.
- `--true-duration` reports the predictor's duration MAE against the reference recordings. Those durations are used only for this measurement.

Score WER/CER and speaker similarity. A missing generation is an error, not a skipped item.

```bash
scripts/eval_tinystress.sh --task wer \
  --tinystress-parquet data/tinystress-15k/test-00000-of-00001.parquet --gen-wav-dir results/tinystress
scripts/eval_tinystress.sh --task sim \
  --tinystress-parquet data/tinystress-15k/test-00000-of-00001.parquet --gen-wav-dir results/tinystress
```

Each run writes `_wer_results.jsonl` / `_sim_results.jsonl` (per sample) and a matching `.json` summary into the generation directory. `--eval-ground-truth` scores the reference recordings instead and writes to `_gt_*` files.

### Sentence-stress detection with StressTest

The patch adds three things to StressTest:

- a TinyStress Parquet loader;
- support for scoring generated WAVs;
- explicit result-file names.

It also fixes Qwen2Audio input conversion and switches StressLM generation to `max_new_tokens`. Apply it to a separate StressTest checkout at upstream commit `e72a8c0`:

```bash
git clone https://github.com/slp-rl/StressTest.git StressTest
git -C StressTest checkout e72a8c0
scripts/apply_stresstest_patch.sh StressTest
python -m pip install -r StressTest/requirements.txt   # preferably in a separate environment
```

StressTest pins its own Transformers version and uses an OpenAI model as the answer judge. Set up its credentials as described in its README. Then, from this repository root:

```bash
scripts/run_stresstest.sh StressTest \
  --model_to_evaluate stresslm \
  --tinystress_parquet data/tinystress-15k/test-00000-of-00001.parquet \
  --audio_dir results/tinystress \
  --inference_results_file tinystress_ssd.json \
  --metrics_file tinystress_metrics_ssd.json
```

- **Output.** Results go to `StressTest/results/`.
- **Reference recordings.** Omit `--audio_dir` to score the reference recordings instead.
- **Subsets.** To re-aggregate WER, SIM and StressLM F1 over a subset defined by the number of emphasized words, run:

```bash
scripts/filter_tinystress.sh --max-emphasis 2 \
  --wav-dir EmphTTS=results/tinystress \
  --ssd-json EmphTTS=StressTest/results/tinystress_ssd.json
```

## Acknowledgements

We borrowed code from the following projects:

1. [F5-TTS](https://github.com/SWivid/F5-TTS): the TTS model, its trainer, and the inference and evaluation utilities.
2. [DMOSpeech 2](https://github.com/yl4579/DMOSpeech2): the duration predictor, its trainer, and the duration GRPO trainer and rewards.
3. [3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker): the CAM++ speaker-verification model.
4. [UniSpeech](https://github.com/microsoft/UniSpeech) and [ECAPA-TDNN](https://github.com/lawlict/ECAPA-TDNN): the WavLM-ECAPA speaker-similarity model.
5. [StressTest](https://github.com/slp-rl/StressTest): our TinyStress patch extends its evaluator.

We thank the authors for releasing their code.

## Citation

If you use this code, please cite our paper:

```bibtex
@misc{li2026emphtts,
  title         = {EmphTTS: an emphasis-control TTS with reinforcement learning},
  author        = {Zirui Li and Rech Silas and Lauri Juvela and Tom Backstrom and Mikko Kurimo},
  year          = {2026},
  eprint        = {2609.27599},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url           = {https://arxiv.org/abs/2609.27599}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE), except for code borrowed from the projects above, which keeps its original license.

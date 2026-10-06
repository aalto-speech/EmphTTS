import re
import string

import jiwer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from emphtts.duration.campplus import CAMP_plus
from emphtts.tts.infer.utils_infer import build_f5_model, load_checkpoint, load_vocoder

# A whitespace token whose word is wrapped in asterisks, with optional punctuation outside them:
# *word*, *word*!, *word,* and "*word*" (the punctuation is kept in the clean text).
_EMPHASIZED_TOKEN = re.compile(r"^(\W*)\*(.+?)\*(\W*)$")


class TTSRewardFn(nn.Module):
    """
    Reward function for GRPODurationTrainer.

    For each generated mel spectrogram it computes:
      - loss_ctc : CTC loss from a frozen Wav2Vec2 ASR model (intelligibility)
      - wer      : Word Error Rate via jiwer (intelligibility)
      - loss_sim : cosine similarity in [0, 1] from a frozen CAMPPlus speaker-
                   verification model (1 = identical speaker)
    With ``use_stress_metric`` the ASR model is replaced by WhiStress, which yields a
    Whisper-based ``wer`` and a word-level ``stress_balanced_acc`` against the *emphasized*
    words in the target text.

    Pipeline:
        est_mel    (B, T, 100) --[vocos]--> wav --[wav2vec2 + jiwer]--> loss_ctc, wer
        est_mel    (B, T, 100) --[vocos]--> wav --[campplus]--+
        prompt_mel (B, T, 100) --[vocos]--> wav --[campplus]--+--> loss_sim

    Args:
        campplus_ckpt_path  : path to campplus_cn_en_common.pt checkpoint
        vocos_local_path    : local directory containing vocos config.yaml and
                              pytorch_model.bin; if None, downloads from HF
                              (charactr/vocos-mel-24khz)
        tts_sample_rate     : sample rate of TTS output mel (default 24 kHz)
        asr_model_id        : HuggingFace model ID for Wav2Vec2 ASR
    """

    def __init__(
        self,
        campplus_ckpt_path: str,
        vocos_local_path: str | None = None,
        tts_sample_rate: int = 24_000,
        asr_model_id: str = "facebook/wav2vec2-large-960h-lv60-self",
        use_stress_metric: bool = False,
    ):
        super().__init__()

        self.tts_sample_rate = tts_sample_rate

        # ── Vocoder ──────────────────────────────────────────────────────────
        self.vocoder = load_vocoder(vocos_local_path, device="cpu")
        for p in self.vocoder.parameters():
            p.requires_grad = False

        # ── CAMPPlus speaker-verification model (frozen) ─────────────────────
        # CAMP_plus handles its own feature extraction (LogMelBank at 16 kHz)
        # and weight loading from ckpt_path.
        self.sv_model = CAMP_plus(ckpt_path=campplus_ckpt_path)
        for p in self.sv_model.parameters():
            p.requires_grad = False
        self.sv_model.eval()

        self.use_stress_metric = use_stress_metric
        # ── WhiStress (frozen) — stress F1 + Whisper-based WER ──────────────
        if use_stress_metric:
            try:
                from whistress import WhiStressInferenceClient
            except ImportError as exc:
                raise RuntimeError("GRPO stress reward requires the external WhiStress repository on PYTHONPATH and its weights installed.") from exc
            self.whistress_client = WhiStressInferenceClient(device="cpu")
            for p in self.whistress_client.whistress.parameters():
                p.requires_grad = False
            self.whistress_client.whistress.eval()
        else:
            # ── ASR model (frozen) — used for transcription only ────────────────
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            self.asr_processor = Wav2Vec2Processor.from_pretrained(asr_model_id)
            self.asr_model = Wav2Vec2ForCTC.from_pretrained(asr_model_id)
            for p in self.asr_model.parameters():
                p.requires_grad = False
            self.asr_model.eval()

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_emphasized_text(text: str) -> tuple[str, list[int]]:
        """
        Parse *emphasized* word markers; punctuation may sit inside or outside the asterisks.
        "You *chose* to do *this*?" → ("You chose to do this?", [1, 4])
        Returns clean text and 0-based indices of stressed words in the clean split.
        """
        clean_words, stressed_indices = [], []
        for index, word in enumerate(text.split()):
            match = _EMPHASIZED_TOKEN.match(word)
            if match:
                word = "".join(match.groups())
                stressed_indices.append(index)
            clean_words.append(word)
        return " ".join(clean_words), stressed_indices

    @staticmethod
    def _balanced_accuracy(preds: list[int], refs: list[int]) -> float:
        """
        Balanced accuracy: average of per-class recall.

          pos_acc = TP / (TP+FN)  — stressed words correctly identified
          neg_acc = TN / (TN+FP)  — unstressed words correctly left alone

        When there are no gold-stressed words pos_acc is vacuously 1.0,
        so the score reduces to neg_acc (penalises false emphases).
        Always in [0, 1]; 1.0 = perfect, 0.5 = random, 0.0 = inverse-perfect.
        """
        tp = sum(p == 1 and g == 1 for p, g in zip(preds, refs))
        fp = sum(p == 1 and g == 0 for p, g in zip(preds, refs))
        tn = sum(p == 0 and g == 0 for p, g in zip(preds, refs))
        fn = sum(p == 0 and g == 1 for p, g in zip(preds, refs))
        pos_acc = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # vacuously 1 if no stressed words
        neg_acc = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        return (pos_acc + neg_acc) / 2

    @torch.no_grad()
    def _compute_whistress_metrics(
        self, est_wav: torch.Tensor, target_texts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Whisper-based WER and WhiStress stress balanced accuracy for a single utterance.

        predict() is called without a GT transcription so Whisper auto-generates
        the transcript. WER compares those predicted words against the clean GT.
        Balanced accuracy uses jiwer word-level alignment to match predicted words to GT words,
        then compares stress labels only on equal/substituted word positions.

        Args:
            est_wav      : (1, T_audio) waveform at tts_sample_rate
            target_texts : list of 1 string with *emphasized* words marked

        Returns:
            (whistress_wer, balanced_acc) — scalar tensors
        """
        assert est_wav.shape[0] == 1, "batch size must be 1"
        device = est_wav.device
        self.whistress_client.device = str(device)

        norm = lambda s: s.lower().translate(str.maketrans("", "", string.punctuation))

        wav_np = est_wav[0].cpu().float().numpy()
        clean_text, stressed_indices = self._parse_emphasized_text(target_texts[0])
        clean_words = clean_text.split()
        gold_set = set(stressed_indices)
        gt_stress = [1 if j in gold_set else 0 for j in range(len(clean_words))]

        # predict() handles resampling from tts_sample_rate → 16 kHz internally
        audio_dict = {"array": wav_np, "sampling_rate": self.tts_sample_rate}
        pairs = self.whistress_client.predict(
            audio=audio_dict, transcription=None, return_pairs=True
        )
        pred_stress = [s for _, s in pairs]
        pred_text = " ".join(w for w, _ in pairs)

        # ── WER ──────────────────────────────────────────────────────────────
        wer_val = jiwer.wer(norm(clean_text), norm(pred_text))

        # ── Stress balanced accuracy via word-level alignment ────────────────
        # Normalise before alignment so case/punctuation don't cause false
        # substitutions; word count is preserved by norm() so indices stay
        # consistent with clean_words / pred_stress.
        alignment = jiwer.process_words(
            reference=norm(clean_text), hypothesis=norm(pred_text)
        )
        matched_pred, matched_gt = [], []
        for chunk in alignment.alignments[0]:
            if chunk.type in ("equal", "substitute"):
                for pi, gi in zip(
                    range(chunk.hyp_start_idx, chunk.hyp_end_idx),
                    range(chunk.ref_start_idx, chunk.ref_end_idx),
                ):
                    matched_pred.append(pred_stress[pi])
                    matched_gt.append(gt_stress[gi])

        balanced_acc = self._balanced_accuracy(matched_pred, matched_gt) if matched_pred else 0.5

        return (
            torch.tensor(wer_val, device=device, dtype=torch.float32),
            torch.tensor(balanced_acc, device=device, dtype=torch.float32),
        )

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def _mel_to_wav(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, T, 100) → (B, T_audio)"""
        return self.vocoder.decode(mel.permute(0, 2, 1))

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def _extract_sv_embedding(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Extract CAMPPlus speaker embeddings.

        Args:
            wav: (B, T_audio) waveform at tts_sample_rate

        Returns:
            embs: (B, emb_dim)
        """
        if self.tts_sample_rate != 16_000:
            wav = torchaudio.functional.resample(wav, self.tts_sample_rate, 16_000)
        # CAMP_plus.forward(wav) handles its own LogMelBank feature extraction
        return self.sv_model(wav)  # (B, emb_dim)

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def _compute_asr_metrics(
        self, est_wav: torch.Tensor, target_texts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single ASR forward pass returning both CTC loss and WER.

        Args:
            est_wav      : (B, T_audio) waveform at tts_sample_rate
            target_texts : list of B ground-truth text strings

        Returns:
            (loss_ctc, wer) — both scalar tensors on the same device as est_wav
              loss_ctc: CTC loss (unbounded, lower = more intelligible)
              wer:      WER clamped to [0, 1]
        """
        device = est_wav.device
        wav_16k = torchaudio.functional.resample(est_wav, self.tts_sample_rate, 16_000)

        input_values = self.asr_processor(
            [w.cpu().float().numpy() for w in wav_16k],
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        ).input_values.to(device)

        # Prepare labels for CTC loss (wav2vec2 vocab is uppercase, no punctuation)
        norm = lambda s: s.upper().translate(str.maketrans("", "", string.punctuation))
        pad_id = self.asr_processor.tokenizer.pad_token_id
        labels = self.asr_processor.tokenizer(
            [norm(t) for t in target_texts],
            return_tensors="pt",
            padding=True,
        ).input_ids.to(device)
        labels[labels == pad_id] = -100

        output = self.asr_model(input_values, labels=labels)
        loss_ctc = output.loss

        # Decode for WER
        transcriptions = self.asr_processor.batch_decode(output.logits.argmax(dim=-1))
        refs = [norm(t) for t in target_texts]
        hyps = [norm(t) for t in transcriptions]

        wer_value = jiwer.wer(refs, hyps)
        wer = torch.tensor(wer_value, device=device, dtype=torch.float32)
        return loss_ctc, wer

    # ── main forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        prompt_mel: torch.Tensor,
        est_mel: torch.Tensor,
        target_text: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            prompt_mel  : (B, T,  100) reference mel, time-first
            est_mel     : (B, T', 100) generated mel, time-first
            target_text : list of B raw text strings (original, not token IDs)

        Returns:
            dict of scalar tensors:
              'loss_ctc' — CTC loss (lower = more intelligible; ASR reward only)
              'wer'      — WER (lower = more intelligible)
              'loss_sim' — cosine similarity in [0, 1]
                           (higher = more similar to the reference speaker)
              'stress_balanced_acc' — WhiStress reward only
        """
        prompt_wav = self._mel_to_wav(prompt_mel)
        try:
            est_wav = self._mel_to_wav(est_mel)
        except Exception:
            # The vocoder rejects degenerate rollouts (e.g. a sampled duration that is too short);
            # score them as the worst outcome so GRPO pushes away from them.
            zero = prompt_wav.new_zeros(())
            return {
                "loss_ctc": zero + 500.0,
                "wer": zero + 1.0,
                "loss_sim": zero,
                "stress_balanced_acc": zero + 0.5,
            }

        # Speaker similarity: cosine similarity scaled to [0, 1]
        prompt_emb = self._extract_sv_embedding(prompt_wav)
        est_emb    = self._extract_sv_embedding(est_wav)
        loss_sim   = ((F.cosine_similarity(prompt_emb, est_emb) + 1.0) / 2.0).mean()

        # Whisper WER + stress balanced accuracy via WhiStress
        if self.use_stress_metric:
            whistress_wer, balanced_acc = self._compute_whistress_metrics(est_wav, target_text)

            return {
                "wer": whistress_wer,
                "loss_sim": loss_sim,
                "stress_balanced_acc": balanced_acc,
            }
        else:
            # Intelligibility: CTC loss + WER in a single ASR forward pass
            loss_ctc, wer = self._compute_asr_metrics(est_wav, target_text)
            return {
                    "loss_ctc": loss_ctc,
                    "wer": wer,
                    "loss_sim": loss_sim,
                }


class TTSInferenceFn(nn.Module):
    """
    Frozen CFM (F5-TTS) model used as `inference_fn` in GRPODurationTrainer.

    Call signature (see GRPODurationTrainer.generate_duration_samples):
        __call__(
            full_text_ids  : (B, L)      text token IDs for prompt + target
            prompt_mel     : (B, T, 100) reference mel spectrogram
            target_duration: (B,)        target duration in mel frames (float)
        ) -> est_mel : (B, 100, T')      generated mel (target portion only)
    """

    def __init__(
        self,
        tts_config,             # path to model config YAML or OmegaConf object
        tts_ckpt_path: str,     # path to TTS model checkpoint (.pt / .safetensors)
        vocab_char_map: dict,   # char → id mapping from the tokenizer
        ode_method: str = "euler",
        use_ema: bool = True,
        cfg_strength: float = 1.0,
        steps: int = 32,
    ):
        super().__init__()
        from omegaconf import OmegaConf

        self.cfg_strength = cfg_strength
        self.steps = steps

        if isinstance(tts_config, str):
            tts_config = OmegaConf.load(tts_config)
        tts_model = build_f5_model(tts_config, vocab_char_map, ode_method=ode_method)
        # Load to CPU; the trainer moves the module to its device
        self.tts_model = load_checkpoint(
            tts_model, tts_ckpt_path, device="cpu", dtype=torch.float32, use_ema=use_ema
        )
        self.tts_model.requires_grad_(False)

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def forward(
        self,
        full_text_ids: torch.Tensor,
        prompt_mel: torch.Tensor,
        target_duration: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            full_text_ids:   (B, L)      — token IDs (prompt text + target text)
            prompt_mel:      (B, T, 100) — reference mel spectrogram, time-first
            target_duration: (B,)        — number of mel frames to generate (float)

        Returns:
            est_mel: (B, 100, T')        — generated mel, channels-first
        """
        prompt_len = prompt_mel.shape[1]

        # target_duration comes from Gumbel-softmax sampling — cast to long
        target_dur = target_duration.long()                          # (B,)
        total_duration = prompt_len + target_dur                     # (B,)

        lens = torch.full(
            (prompt_mel.shape[0],),
            prompt_len,
            device=prompt_mel.device,
            dtype=torch.long,
        )

        # CFM.sample() expects cond in (B, T, n_mel) format — prompt_mel already is
        out, _ = self.tts_model.sample(
            cond=prompt_mel,
            text=full_text_ids,
            duration=total_duration,
            lens=lens,
            steps=self.steps,
            cfg_strength=self.cfg_strength,
        )
        # out: (B, max_total_duration, 100)
        # The prompt region is restored to cond inside CFM.sample(), so the
        # generated target starts at index prompt_len.
        est_mel = out[:, prompt_len:, :]   # (B, T', 100)

        return est_mel.permute(0, 2, 1)                              # (B, 100, T')

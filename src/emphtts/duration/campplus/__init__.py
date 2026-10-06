import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from emphtts.duration.campplus.DTDNN import CAMPPlus


class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super(PreEmphasis, self).__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class LogMelBank(torch.nn.Module):
    def __init__(self, out_dim=64, mean_nor=False, **unused):
        super(LogMelBank, self).__init__()
        SAMPING_RATE = 16000
        self.mean_nor = mean_nor
        win_length = 400
        hop_length = 160

        self.pre = PreEmphasis()
        self.mel_bank = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPING_RATE, n_fft=512,
                                                             win_length=win_length, hop_length=hop_length,
                                                             window_fn=lambda x: torch.hann_window(x, periodic=False).pow(0.85), n_mels=out_dim)
        self.out_dim = out_dim

    def forward_base(self, wav_input):
        wav_input = wav_input.unsqueeze(1)
        wav_input = self.pre(wav_input)
        wav_input = wav_input.squeeze(1)
        wav_out = self.mel_bank(wav_input)
        wav_out = torch.log(wav_out + 1e-6)
        return wav_out

    def forward(self, wav_input):
        wav_out = self.forward_base(wav_input)
        if self.mean_nor:
            wav_out = wav_out - torch.mean(wav_out, dim=-1, keepdim=True)
        wav_out = wav_out.transpose(1, 2)
        return wav_out


class CAMP_plus(nn.Module):
    """CAMPPlus speaker embedder for 16 kHz waveforms (e.g. 3D-Speaker's campplus_cn_en_common.pt)."""

    def __init__(self, ckpt_path, norm_emb=False, **unused):
        super(CAMP_plus, self).__init__()
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"CAMPPlus checkpoint not found at {ckpt_path}; download campplus_cn_en_common.pt as described "
                "in the README (Checkpoints) or set reward.campplus_ckpt"
            )
        pretrained_state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        self.norm_emb = norm_emb

        embedding_model = CAMPPlus(feat_dim=80, embedding_size=192)
        embedding_model.load_state_dict(pretrained_state)
        self.embedding_model = embedding_model
        self.embedding_model.eval()

        self.feature_extractor = LogMelBank(80, mean_nor=True)

    def forward(self, wav, mel=None, with_grad=False):
        if mel is None:
            feat = self.feature_extractor(wav)
        else:
            feat = mel
        if with_grad:
            embedding = self.embedding_model(feat)
        else:
            with torch.no_grad():
                embedding = self.embedding_model(feat).detach()
        if self.norm_emb:
            embedding = F.normalize(embedding, dim=-1)
        return embedding

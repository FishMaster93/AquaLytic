import torch.nn as nn
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

from models.modules import init_bn


class Audio_Frontend(nn.Module):
    """Waveform -> log-mel spectrogram frontend, shared by both backbones."""

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        super(Audio_Frontend, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.mel_bins = mel_bins

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size, win_length=window_size,
            window=window, center=center, pad_mode=pad_mode, freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax,
            ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(self.mel_bins)
        init_bn(self.bn0)

    def forward(self, input):
        """Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = nn.ZeroPad2d((0, 0, 2, 0))(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)
        return x


class AudioModel(nn.Module):
    """Thin wrapper gluing frontend + backbone, exposing both the CORAL
    logits (clipwise_output) and the pre-classifier embedding."""

    def __init__(self, frontend, backbone, **kwargs):
        super().__init__(**kwargs)
        self.frontend = frontend
        self.backbone = backbone

    def forward(self, input):
        clipwise_output, embedding = self.backbone(self.frontend(input))
        return {'clipwise_output': clipwise_output, 'embedding': embedding}

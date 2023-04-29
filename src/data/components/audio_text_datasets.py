from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, Resample


@dataclass
class AudioTextDataset(Dataset):

    """
    Dataset for audio-text dataset
    manifest file format:
        [
            {
                "text": "hello",
                "wav_path": "path/to/wav"
            },
            ...
        ]
    Returns:
        spec (torch.Tensor)[Channel, Dim, Time]: mel spectrogram
        text_ids (torch.Tensor)[Len]: text ids
    """

    manifest: str
    sampling_rate: int = 16000
    n_mel: int = 80

    def __post_init__(self):
        tokens = []
        for m in self.manifest:
            # validation
            assert "text" in m
            assert "wav_path" in m
            assert Path(m["wav_path"]).exists()
            # make tokens
            tokens += list(m["text"])

        self.tokens = list(set(tokens))
        self.token_ids = {token: idx for idx, token in enumerate(self.tokens)}
        self.transform_melspec = MelSpectrogram(self.sampling_rate, n_mels=self.n_mel)

    def __len__(self):
        return len(self.manifest)

    def get_vocab_size(self):
        """
        return vocab size
        * added 1 token for blank
        """
        return len(self.tokens) + 1

    def get_vocab_list(self):
        return self.tokens

    def text_to_ids(self, text) -> torch.Tensor:
        id_list = [self.token_ids[token] for token in text]
        return torch.tensor(id_list, dtype=torch.long)

    def transform_wav_to_melspec(self, wav_path):
        waveform, sample_rate = torchaudio.load(wav_path)
        if sample_rate != self.sampling_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.sampling_rate)
            waveform = resampler(waveform)
        melspec = self.transform_melspec(waveform)
        return melspec

    def __getitem__(self, idx):
        manifest = self.manifest[idx]
        wav_path = manifest["wav_path"]
        spec = self.transform_wav_to_melspec(wav_path)
        text = manifest["text"]
        text_ids = self.text_to_ids(text)
        return spec, text_ids


def audio_to_text_collate_fn(batch):
    """
    collate function for audio-text dataset
    batch: [(spec, text_ids), ...)]
        spec: (C, D, T)
        text_ids: (L)

    return:
        spec_padded: (B, T, D)
        spec_lengths: (B)
    """
    max_spec_len = max(x[0].size(2) for x in batch)
    max_text_len = max(x[1].size(0) for x in batch)
    spec_dim = batch[0][0].size(1)
    batch_size = len(batch)

    spec_lengths = torch.LongTensor(batch_size)
    text_lengths = torch.LongTensor(batch_size)
    spec_padded = torch.zeros(batch_size, max_spec_len, spec_dim, dtype=torch.float32)
    text_padded = torch.zeros(batch_size, max_text_len, dtype=torch.long)

    for i, (spec, text) in enumerate(batch, 0):
        spec_padded[i, : spec.size(2)] = spec[0, :].transpose(0, 1)
        spec_lengths[i] = spec.size(2)

        text_padded[i, : text.size(0)] = text
        text_lengths[i] = text.size(0)

    return (spec_padded, spec_lengths, text_padded, text_lengths)

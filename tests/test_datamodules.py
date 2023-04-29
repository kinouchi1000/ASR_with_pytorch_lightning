from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.components.audio_text_datasets import (
    AudioTextDataset,
    audio_to_text_collate_fn,
)
from src.data.formatter.jsut import JSUTFormatter
from src.data.jsut_datamodule import JSUTDataModule


def test_format_json():
    jsut = JSUTFormatter("./tests/tmp")
    manifest_path = jsut.get_manifest_path()
    assert Path(manifest_path).exists()
    manifest = jsut.get_manifest()
    assert len(manifest) > 0
    for item in manifest:
        assert "text" in item
        assert "wav_path" in item
        assert "id" in item
        assert "speaker_id" in item


def test_datamodules():
    datamodule = JSUTDataModule("./tests/tmp")
    datamodule.prepare_data()
    datamodule.setup()


def test_datasets():
    jsut = JSUTFormatter("./tests/tmp")
    dataset = AudioTextDataset(jsut.get_manifest(), sampling_rate=16000, n_mel=80)
    wav, text_ids = dataset[0]

    assert wav.shape[0] == 1
    assert wav.dim() == 3
    assert wav.shape[1] == 80
    assert len(text_ids) > 0

    assert type(wav) == torch.Tensor
    assert type(text_ids) == torch.Tensor

    dataloader = DataLoader(dataset, collate_fn=audio_to_text_collate_fn, batch_size=2, shuffle=False)

    for batch in dataloader:
        spec, spec_len, text, text_len = batch
        break

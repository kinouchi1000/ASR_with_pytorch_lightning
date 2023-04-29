import random
from dataclasses import dataclass
from typing import Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .components.audio_text_datasets import AudioTextDataset
from .formatter.jsut import JSUTFormatter


@dataclass
class JSUTDataModule(LightningDataModule):
    """
    Example of LightningDataModule for JSUT courpus
    日本語女性音声のコーパスのJSUTのDatamodule
    reference : https://sites.google.com/site/shinnosuketakamichi/publication/jsut?pli=1

    """

    data_dir: str = "data/"
    train_val_test_split_per: Tuple[float, float, float] = (0.9, 0.05, 0.05)
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False

    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """
        Download JSUT
        """
        JSUTFormatter(self.data_dir)

    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            jsut = JSUTFormatter(self.data_dir)
            all_manifest = jsut.get_manifest()
            random.shuffle(all_manifest)
            train_idx = int(len(all_manifest) * self.train_val_test_split_per[0])
            valid_idx = int(len(all_manifest) * self.train_val_test_split_per[1])
            train = all_manifest[:train_idx]
            val = all_manifest[train_idx : train_idx + valid_idx]
            test = all_manifest[train_idx + valid_idx :]

            self.data_train = AudioTextDataset(train)
            self.data_val = AudioTextDataset(val)
            self.data_test = AudioTextDataset(test)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evan import inception
from evan.models import resnet


class DummyDataset(Dataset):
    length = 16
    size = 112
    channels = 3

    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return np.random.rand(self.channels, self.length, self.size, self.size).astype(
            np.float32
        )


class TestInception(unittest.TestCase):
    def test_preapre_inception_model(self):
        temp_dir = tempfile.TemporaryDirectory()
        model = inception.prepare_inception_model(
            torch.device("cpu"), Path(temp_dir.name)
        )

        filename = "resnet-101-kinetics-ucf101_split1.pth"
        model_file = Path(temp_dir.name) / filename
        self.assertTrue(model_file.exists())

        self.assertEqual(2048, model.fc.in_features)
        self.assertEqual(101, model.fc.out_features)

        temp_dir.cleanup()

    def test_forward_videos(self):
        model = resnet.resnet101(
            num_classes=101, shortcut_type="B", sample_size=112, sample_duration=16
        )

        batchsize = 5
        n_workers = 1
        n_samples = 10
        dataset = DummyDataset(n_samples)
        dataloader = DataLoader(
            dataset, batch_size=batchsize, num_workers=n_workers, pin_memory=True
        )

        device = torch.device("cpu")
        features, probs = inception.forward_videos(model, dataloader, device, False)

        self.assertEqual((n_samples, 2048), features.shape)
        self.assertEqual((n_samples, 101), probs.shape)

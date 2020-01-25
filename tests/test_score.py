import tempfile
import unittest
from pathlib import Path

import numpy as np
import skvideo.io
import torch
from torch.utils.data import DataLoader, Dataset

from evan import score

TEST_VIDEO = "tests/data/small.mp4"


def new_video_directory(n_samples: int) -> tempfile.TemporaryDirectory:
    temp = tempfile.TemporaryDirectory()
    temp_dir = Path(temp.name)

    videogen = skvideo.io.vreader(TEST_VIDEO)
    video = np.stack([frame for frame in videogen])
    t, h, w, _ = video.shape

    length = 48
    size = 256
    cropped_video = video[
        (t - length) // 2 : (t - length) // 2 + length,
        (h - size) // 2 : (h - size) // 2 + size,
        (w - size) // 2 : (w - size) // 2 + size,
    ]

    fps = 30
    noise_scale = 100
    for i in range(n_samples):
        v = cropped_video + noise_scale * np.random.rand(*cropped_video.shape)
        v = np.clip(v, 0, 255).astype(np.uint8)

        writer = skvideo.io.FFmpegWriter(
            str(temp_dir / f"{i}.mp4"), inputdict={"-r": str(fps)}
        )

        for frame in v:
            writer.writeFrame(frame)
        writer.close()

    return temp


class TestScore(unittest.TestCase):
    def test_inception_score(self):
        N = 10
        d = new_video_directory(N)
        s = score.compute_inception_score(d.name)
        d.cleanup()

        t = 10
        self.assertTrue(s < t)

    def test_frechet_distance(self):
        N = 10
        g = new_video_directory(N)
        r = new_video_directory(N)
        s = score.compute_frechet_distance(g.name, r.name)
        g.cleanup()
        r.cleanup()

        t = 10
        self.assertTrue(s < t)

    def test_precision_recall(self):
        N = 10
        g = new_video_directory(N)
        r = new_video_directory(N)
        s = score.compute_frechet_distance(g.name, r.name)
        g.cleanup()
        r.cleanup()

        t = 10
        self.assertTrue(s < t)

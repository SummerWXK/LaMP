import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from starVLA.dataloader.libero import LiberoRawDataset


class FakeBackend:
    fps = 20

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class FakeBundledReader:
    def __init__(self):
        self.all_steps = [(3, 4)]

    def __len__(self):
        return 1

    def get_step_data(self, episode_index, frame_index):
        assert (episode_index, frame_index) == self.all_steps[0]
        record = {
            "video.primary_image": np.zeros((1, 16, 20, 3), dtype=np.uint8),
            "video.wrist_image": np.zeros((1, 16, 20, 3), dtype=np.uint8),
            "annotation.human.action.task_description": ["pick up the block"],
        }
        for offset, name in enumerate(("x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper")):
            record[f"state.{name}"] = np.array([[offset]], dtype=np.float32)
        for offset, name in enumerate(("x", "y", "z", "roll", "pitch", "yaw", "gripper")):
            record[f"action.{name}"] = np.full((10, 1), offset, dtype=np.float32)
        return record


def make_record(index=0):
    image = torch.zeros(3, 32, 48)
    return {
        "observation.images.image": image,
        "observation.images.wrist_image": image,
        "observation.state": torch.arange(8, dtype=torch.float32),
        "action": torch.zeros(10, 7),
        "task": "pick up the block",
        "episode_index": torch.tensor(index),
        "frame_index": torch.tensor(index + 1),
    }


class LiberoRawDatasetTest(unittest.TestCase):
    def test_returns_raw_shapes_and_metadata(self):
        dataset = LiberoRawDataset("/unused", "libero_spatial", backend=FakeBackend([make_record()]))
        sample = dataset[0]
        self.assertEqual([image.size for image in sample["image"]], [(48, 32), (48, 32)])
        self.assertEqual(sample["state"].shape, (8,))
        self.assertEqual(sample["state"].dtype, np.float32)
        self.assertEqual(sample["action"].shape, (10, 7))
        self.assertEqual(sample["dataset_key"], "libero_spatial_no_noops_lerobot")

    def test_uses_bundled_gr00t_lerobot_reader(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata_directory = root / "libero_spatial_no_noops_lerobot" / "meta"
            metadata_directory.mkdir(parents=True)
            metadata = {
                "codebase_version": "v2.1",
                "fps": 20,
                "features": {
                    "observation.images.image": {},
                    "observation.images.wrist_image": {},
                    "observation.state": {},
                    "action": {},
                },
            }
            (metadata_directory / "info.json").write_text(json.dumps(metadata), encoding="utf-8")

            with mock.patch(
                "starVLA.dataloader.libero._build_gr00t_lerobot_reader",
                return_value=FakeBundledReader(),
            ) as build_reader:
                dataset = LiberoRawDataset(root, "libero_spatial", video_backend="torchvision")
                sample = dataset[0]

            build_reader.assert_called_once_with(
                root / "libero_spatial_no_noops_lerobot",
                video_backend="torchvision",
            )
            self.assertEqual(sample["lang"], "pick up the block")
            np.testing.assert_array_equal(sample["state"], np.arange(8, dtype=np.float32))
            self.assertEqual(sample["action"].shape, (10, 7))
            self.assertEqual(sample["episode_index"], 3)
            self.assertEqual(sample["frame_index"], 4)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import pickle

import torch

from opensr_model.training.data import TacoPairedDataset, _aligned_crop, _augment


def test_aligned_crop_preserves_hr_lr_and_mask_coordinates() -> None:
    lr = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    hr = lr.repeat_interleave(4, -2).repeat_interleave(4, -1)
    mask = torch.ones(1, 16, 16)

    cropped_hr, cropped_lr, cropped_mask = _aligned_crop(
        hr, lr, mask, patch=(8, 8), factor=4, random=False
    )

    assert cropped_lr.shape == (1, 2, 2)
    assert cropped_hr.shape == (1, 8, 8)
    assert cropped_mask.shape == (1, 8, 8)
    assert torch.equal(
        cropped_hr, cropped_lr.repeat_interleave(4, -2).repeat_interleave(4, -1)
    )


def test_augmentation_applies_identical_geometry_to_pair_and_mask() -> None:
    lr = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
    hr = lr.repeat_interleave(4, -2).repeat_interleave(4, -1)
    mask = hr.clone()

    augmented_hr, augmented_lr, augmented_mask = _augment(
        hr, lr, mask, horizontal=1.0, vertical=1.0, rotate_90=False
    )

    assert torch.equal(augmented_hr, augmented_mask)
    assert torch.equal(
        augmented_hr,
        augmented_lr.repeat_interleave(4, -2).repeat_interleave(4, -1),
    )


def test_dataset_pickle_drops_process_local_reader_cache(fake_taco_factory) -> None:
    bundle = fake_taco_factory(include_nodata=False)
    dataset = TacoPairedDataset(bundle.root, training=False)
    dataset[0]

    restored = pickle.loads(pickle.dumps(dataset))

    assert dataset._readers
    assert restored._reader_pid is None
    assert restored._readers == {}

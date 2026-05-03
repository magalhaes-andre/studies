from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .ct import crop_image, expand_bbox, load_dicom_as_hu, resize_to_tensor, sample_negative_bbox, window_ct


@dataclass(frozen=True)
class SampleRow:
    image_path: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    label: int
    seed: int


class LungTumorCropDataset(Dataset):
    def __init__(
        self,
        annotations: pd.DataFrame,
        image_size: int = 224,
        negatives_per_positive: int = 1,
        window_center: float = 40.0,
        window_width: float = 350.0,
        padding: float = 0.25,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.image_size = image_size
        self.window_center = window_center
        self.window_width = window_width
        self.padding = padding
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.samples = self._build_samples(annotations, negatives_per_positive)

    def _build_samples(self, annotations: pd.DataFrame, negatives_per_positive: int) -> list[SampleRow]:
        rows: list[SampleRow] = []
        for row in annotations.itertuples(index=False):
            seed = int(self.rng.integers(0, 2**31 - 1))
            rows.append(
                SampleRow(
                    image_path=row.image_path,
                    xmin=int(row.xmin),
                    ymin=int(row.ymin),
                    xmax=int(row.xmax),
                    ymax=int(row.ymax),
                    label=1,
                    seed=seed,
                )
            )
            for _ in range(negatives_per_positive):
                seed = int(self.rng.integers(0, 2**31 - 1))
                rows.append(
                    SampleRow(
                        image_path=row.image_path,
                        xmin=int(row.xmin),
                        ymin=int(row.ymin),
                        xmax=int(row.xmax),
                        ymax=int(row.ymax),
                        label=0,
                        seed=seed,
                    )
                )
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, tensor: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        if rng.random() < 0.5:
            tensor = torch.flip(tensor, dims=[2])
        if rng.random() < 0.5:
            tensor = torch.flip(tensor, dims=[1])
        if rng.random() < 0.2:
            noise = torch.randn_like(tensor) * 0.01
            tensor = torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = load_dicom_as_hu(sample.image_path)
        image = window_ct(image, center=self.window_center, width=self.window_width)
        image_shape = image.shape
        positive_bbox = (sample.xmin, sample.ymin, sample.xmax, sample.ymax)

        if sample.label == 1:
            bbox = expand_bbox(positive_bbox, image_shape, self.padding)
        else:
            rng = np.random.default_rng(sample.seed)
            bbox = sample_negative_bbox(image_shape, positive_bbox, self.padding, rng)

        crop = crop_image(image, bbox)
        tensor = resize_to_tensor(crop, self.image_size)

        if self.augment:
            rng = np.random.default_rng(sample.seed)
            tensor = self._augment(tensor, rng)

        label = torch.tensor([float(sample.label)], dtype=torch.float32)
        return tensor, label

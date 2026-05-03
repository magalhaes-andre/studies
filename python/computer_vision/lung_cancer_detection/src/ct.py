from __future__ import annotations

from typing import Tuple

import numpy as np
import pydicom
import torch
import torch.nn.functional as F


BBox = Tuple[int, int, int, int]


def load_dicom_as_hu(dicom_path: str) -> np.ndarray:
    ds = pydicom.dcmread(dicom_path)
    pixels = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = pixels * slope + intercept
    return hu


def window_ct(image_hu: np.ndarray, center: float = 40.0, width: float = 350.0) -> np.ndarray:
    lower = center - width / 2.0
    upper = center + width / 2.0
    clipped = np.clip(image_hu, lower, upper)
    normalized = (clipped - lower) / max(width, 1e-6)
    return normalized.astype(np.float32)


def bbox_size(bbox: BBox) -> tuple[int, int]:
    xmin, ymin, xmax, ymax = bbox
    return xmax - xmin, ymax - ymin


def clip_bbox(bbox: BBox, image_shape: tuple[int, int]) -> BBox:
    height, width = image_shape
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, min(xmin, width - 1))
    ymin = max(0, min(ymin, height - 1))
    xmax = max(xmin + 1, min(xmax, width))
    ymax = max(ymin + 1, min(ymax, height))
    return xmin, ymin, xmax, ymax


def expand_bbox(bbox: BBox, image_shape: tuple[int, int], padding: float) -> BBox:
    height, width = image_shape
    clipped_bbox = clip_bbox(bbox, image_shape)
    xmin, ymin, xmax, ymax = clipped_bbox
    box_w, box_h = bbox_size(clipped_bbox)
    pad_w = int(round(box_w * padding))
    pad_h = int(round(box_h * padding))

    xmin = max(0, xmin - pad_w)
    ymin = max(0, ymin - pad_h)
    xmax = min(width, xmax + pad_w)
    ymax = min(height, ymax + pad_h)
    return xmin, ymin, xmax, ymax


def crop_image(image: np.ndarray, bbox: BBox) -> np.ndarray:
    xmin, ymin, xmax, ymax = clip_bbox(bbox, image.shape)
    crop = image[ymin:ymax, xmin:xmax]
    if crop.size == 0:
        return np.zeros((16, 16), dtype=image.dtype)
    return crop


def compute_iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def sample_negative_bbox(image_shape: tuple[int, int], positive_bbox: BBox, padding: float, rng: np.random.Generator) -> BBox:
    height, width = image_shape
    positive_bbox = clip_bbox(positive_bbox, image_shape)
    crop_w, crop_h = bbox_size(expand_bbox(positive_bbox, image_shape, padding))
    crop_w = min(max(crop_w, 16), width)
    crop_h = min(max(crop_h, 16), height)

    max_x = max(0, width - crop_w)
    max_y = max(0, height - crop_h)

    for _ in range(64):
        xmin = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        ymin = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        candidate = (xmin, ymin, xmin + crop_w, ymin + crop_h)
        if compute_iou(candidate, positive_bbox) < 0.05:
            return candidate

    return (0, 0, crop_w, crop_h)


def resize_to_tensor(image: np.ndarray, size: int) -> torch.Tensor:
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0)

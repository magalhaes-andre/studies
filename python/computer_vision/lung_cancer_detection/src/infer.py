from __future__ import annotations

import argparse

import torch

from .ct import expand_bbox, crop_image, load_dicom_as_hu, resize_to_tensor, window_ct
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for a single CT crop.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dicom-path", required=True)
    parser.add_argument("--xmin", type=int, required=True)
    parser.add_argument("--ymin", type=int, required=True)
    parser.add_argument("--xmax", type=int, required=True)
    parser.add_argument("--ymax", type=int, required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=350.0)
    parser.add_argument("--padding", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = load_dicom_as_hu(args.dicom_path)
    image = window_ct(image, center=args.window_center, width=args.window_width)
    bbox = expand_bbox((args.xmin, args.ymin, args.xmax, args.ymax), image.shape, args.padding)
    crop = crop_image(image, bbox)
    tensor = resize_to_tensor(crop, args.img_size).unsqueeze(0).to(device)

    with torch.no_grad():
        probability = torch.sigmoid(model(tensor)).item()

    print(f"tumor_probability={probability:.4f}")


if __name__ == "__main__":
    main()

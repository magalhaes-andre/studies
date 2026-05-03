from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class AnnotationRecord:
    image_path: str
    patient_id: str
    xml_path: str
    object_name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class AnnotationResolver:
    def __init__(self, image_root: Path) -> None:
        self.image_root = image_root
        self._filename_cache: dict[str, Path | None] = {}

    def resolve(self, xml_path: Path, filename: str | None, raw_path: str | None) -> Path | None:
        candidates: list[Path] = []

        if raw_path:
            candidates.append(Path(raw_path))
            if filename:
                candidates.append(self.image_root / Path(raw_path).name)

        if filename:
            candidates.append(self.image_root / filename)

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        if not filename:
            return None

        if filename not in self._filename_cache:
            match = next(self.image_root.rglob(filename), None)
            self._filename_cache[filename] = match

        return self._filename_cache[filename]


def _safe_text(element: ET.Element | None, tag: str) -> str | None:
    if element is None:
        return None
    node = element.find(tag)
    if node is None or node.text is None:
        return None
    value = node.text.strip()
    return value or None


def _patient_id(image_root: Path, image_path: Path) -> str:
    try:
        relative = image_path.relative_to(image_root)
        if relative.parts:
            return relative.parts[0]
    except ValueError:
        pass
    stem = image_path.stem
    return stem.split("_")[0]


def load_annotations(annotations_dir: str | Path, image_root: str | Path) -> pd.DataFrame:
    annotations_dir = Path(annotations_dir)
    image_root = Path(image_root)
    resolver = AnnotationResolver(image_root)

    xml_files = sorted(annotations_dir.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found under {annotations_dir}")

    rows: list[dict[str, object]] = []
    unresolved = 0

    for xml_path in tqdm(xml_files, desc="Parsing XML annotations"):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = _safe_text(root, "filename")
        raw_path = _safe_text(root, "path")
        image_path = resolver.resolve(xml_path, filename, raw_path)
        if image_path is None:
            unresolved += 1
            continue

        patient_id = _patient_id(image_root, image_path)
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            try:
                xmin = int(float(_safe_text(bbox, "xmin") or 0))
                ymin = int(float(_safe_text(bbox, "ymin") or 0))
                xmax = int(float(_safe_text(bbox, "xmax") or 0))
                ymax = int(float(_safe_text(bbox, "ymax") or 0))
            except ValueError:
                continue

            if xmax <= xmin or ymax <= ymin:
                continue

            rows.append(
                {
                    "image_path": str(image_path),
                    "patient_id": patient_id,
                    "xml_path": str(xml_path),
                    "object_name": (_safe_text(obj, "name") or "tumor").lower(),
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                }
            )

    if not rows:
        raise RuntimeError("No usable annotation rows were parsed from the XML files.")

    frame = pd.DataFrame(rows).drop_duplicates()
    if unresolved:
        print(f"Warning: {unresolved} XML files could not be resolved to a local image path.")
    return frame.reset_index(drop=True)

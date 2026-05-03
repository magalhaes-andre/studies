# Minimal Lung Cancer CT MVP

This project implements a narrow research MVP for the `Lung-PET-CT-Dx` archive:

- input: single CT DICOM slices referenced by XML annotations
- supervision: tumor bounding boxes from Pascal VOC XML files
- task: binary classification of `tumor crop` vs `background crop`
- output: a lesion probability for a crop, not a diagnosis

This is intentionally small so you can get an end-to-end pipeline working before attempting full detection, 3D models, or clinical claims.

## What This Starter Does

- parses Pascal VOC XML annotations
- resolves the referenced DICOM slice under an image root
- converts CT pixels to HU and applies windowing
- samples positive crops from tumor boxes
- samples random negative crops away from the tumor box
- trains a `ResNet18` binary classifier on 2D crops
- saves patient-level train/val/test splits and the best checkpoint

## Recommended Dataset Layout

The code expects two paths:

- `--image-root`: a directory that contains the CT DICOM files
- `--annotations-dir`: a directory that contains the XML annotation files

A layout like this works well:

```text
lung_pet_ct_dx/
  images/
    PATIENT_A001/
      ... dicom files ...
  annotations/
    PATIENT_A001_0001.xml
    PATIENT_A001_0002.xml
```

The XML parser first tries the `<path>` field from each XML file. If that path is unusable on your machine, it falls back to searching for the XML `filename` under `--image-root`.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m src.train   --image-root "C:/data/lung_pet_ct_dx/images"   --annotations-dir "C:/data/lung_pet_ct_dx/annotations"   --output-dir "C:/data/lung_pet_ct_dx_runs/mvp"   --epochs 10   --batch-size 16
```

Useful options:

- `--img-size 224`
- `--negatives-per-positive 1`
- `--window-center 40`
- `--window-width 350`
- `--padding 0.25`
- `--no-pretrained`

## Inference

Run inference on a known box from one DICOM slice:

```bash
python -m src.infer   --checkpoint "C:/data/lung_pet_ct_dx_runs/mvp/best.pt"   --dicom-path "C:/data/lung_pet_ct_dx/images/example.dcm"   --xmin 120 --ymin 140 --xmax 220 --ymax 240
```

## Limitations

- This is not a full lung cancer detector.
- The negative class is sampled from cancer scans, so it is only a weak background class.
- Performance on this task does not imply patient-level diagnostic usefulness.
- The archive is small enough that overfitting is a real risk.

## Good Next Steps

- replace crop classification with object detection
- add benign or non-cancer CT cases from another dataset
- move from 2D slices to 3D crops
- validate on a completely separate cohort

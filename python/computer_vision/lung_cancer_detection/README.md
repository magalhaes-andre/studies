# Lung Cancer Detection Notebook

This project implements a small teaching project for the `Lung-PET-CT-Dx` archive:

- input: single CT DICOM slices referenced by XML annotations
- supervision: tumor bounding boxes from Pascal VOC XML files
- task: binary classification of `tumor crop` vs `background crop`
- output: a lesion probability for a crop, not a diagnosis

This version is intentionally simple so it can work as an introductory CNN assignment.

## What This Notebook Does

- parses Pascal VOC XML annotations
- resolves the referenced DICOM slice under an image root
- converts CT pixels to HU and applies windowing
- samples positive crops from tumor boxes
- samples random negative crops away from the tumor box
- trains a small `TensorFlow/Keras` CNN on 2D crops
- saves patient-level train/val/test splits and the best model

## Main File

- Open `lung_cancer_detection_workflow.ipynb`
- Run the cells from top to bottom
- Edit the dataset paths in the path configuration cell before training

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

## How To Use

- Set `IMAGE_ROOT`, `ANNOTATIONS_DIR`, and `OUTPUT_DIR` in the notebook
- Run the training cell to create `best.keras`, `best_info.json`, and `metrics.json`
- Run the inference example cell to test one DICOM crop

Useful notebook settings:

- `IMAGE_SIZE = 128`
- `NEGATIVES_PER_POSITIVE = 1`
- `WINDOW_CENTER = 40.0`
- `WINDOW_WIDTH = 350.0`
- `PADDING = 0.25`
- `EPOCHS = 10`
- `BATCH_SIZE = 16`

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

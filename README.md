# Real-time ROI Detection using Depth and Optical Flow

## Overview

This project implements a lightweight ROI (Region of Interest) detection pipeline using monocular depth estimation and optical flow.

Instead of relying on deep-learning-based object detectors, the system fuses multiple visual cues including:

- monocular depth (MiDaS)
- optical flow
- edge saliency

to detect candidate object regions in real time.

The pipeline is designed to run on CPU without requiring GPU-based object detection models.

---

## System Pipeline

Smartphone Camera + Depth (MiDaS)
        ↓
Frame Transmission (TCP)
        ↓
Optical Flow (Lucas–Kanade)
        ↓
Depth Confidence Estimation
        ↓
Cue Fusion
        ↓
ROI Detection

---

## Implementation

Main components:

- Depth normalization and confidence estimation
- Lucas–Kanade optical flow motion estimation
- Motion reliability estimation
- Multi-cue fusion (depth + motion + edge)
- ROI detection using connected components

The system processes frames in real time using a multi-threaded pipeline.

---

## Example Results

Example ROI detection output:

![ROI-ezgif com-video-to-gif-converter2](https://github.com/user-attachments/assets/8d37c15d-3c84-4b5a-bdf3-006a354e443d)


---

## Notes

The Android client used in this project has been modified in later experiments to include additional sensors such as accelerometer and gyroscope.  
Therefore, the Android code included in this repository may not exactly match the version used in the initial experiments.

The main focus of this repository is the perception pipeline implemented on the PC side.

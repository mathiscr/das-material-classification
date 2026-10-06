# DAS Material Classification

Classification of ground materials from Distributed Acoustic Sensing (DAS) signals using time-series processing and k-Nearest Neighbors (k-NN).

The objective of this project is to process raw DAS measurements, automatically extract acoustic events, and classify the material surrounding the optical fiber into three classes:

- **Asphalt**
- **Sand**
- **Technical chamber**

Three time-series distance metrics are evaluated for k-NN classification:

- Euclidean distance
- Dynamic Time Warping (DTW)
- Cross-correlation

The best performance obtained on the curated dataset is **90% accuracy using cross-correlation with k-NN**.

---

## Project Overview

Distributed Acoustic Sensing (DAS) uses optical fibers as continuous acoustic sensors. Vibrations occurring along the fiber modify the optical signal, making it possible to detect acoustic events and locate them in space and time.

In this project, we investigate whether the acoustic response measured by DAS can be used to identify the surrounding ground material.

The complete pipeline is:

```text
Raw DAS data (.mat)
        │
        ▼
Differential phase extraction
        │
        ▼
Spatial channel selection
        │
        ▼
Butterworth bandpass filtering
        │
        ▼
Event detection
        │
        ▼
0.5 s time-series extraction
        │
        ▼
Dataset curation (.npz)
        │
        ▼
Pairwise distance matrices
        │
        ▼
k-NN + Leave-One-Out cross-validation
        │
        ▼
Material classification

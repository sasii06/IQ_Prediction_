# IQ Prediction from Resting-State fMRI using Temporal, Similarity, and Hybrid Representations

This repository contains the implementation of the proposed temporal, similarity-based, and hybrid frameworks for IQ prediction from resting-state fMRI. The repository includes the complete source code for the experiments conducted on the ABIDE dataset, along with an executed notebook for the HCP dataset.

---

# Datasets

This study utilizes two publicly available neuroimaging datasets:

- Autism Brain Imaging Data Exchange (ABIDE)
- Human Connectome Project (HCP)

## Dataset Download

### ABIDE

The preprocessed ABIDE dataset can be downloaded using:

https://github.com/ShawonBarman/How-to-download-ABIDE-Preprocessed-dataset-for-autism-detection

For Subjects IDs used in this study-- select SITEID as NYU and exclude the files containing IQ values as nan or -9999

### HCP

The HCP preprocessing pipeline is available at:

https://github.com/Washington-University/HCPpipelines

Please download the datasets from their official sources before running the experiments.

---
Subject IDs are provided in a Jupyter notebook.
# Data Availability

The complete HCP dataset is not included because of its large storage requirements and data-sharing restrictions.

For the ABIDE experiments, the required processed data are provided in the following compressed files:

- Temporal data.rar
- Spatial data.rar

---

# Repository Contents

## temp_iq.py

Implements the **Temporal-based IQ Prediction** model.

Required data:

- Temporal data.rar

---

## similarity_and_hybrid_based_iq_prediction.py

Implements:

- Similarity-based IQ Prediction
- Hybrid Temporal + Similarity IQ Prediction

Required data:

- Temporal data.rar
- Spatial data.rar

---

## ablation_studies.py

Reproduces all ablation studies reported in the manuscript, including:

- HYDRA parameter analysis
- CNN architecture analysis
- Machine learning model comparison
- Hybrid feature ablations
- Similarity feature analysis

Required data:

- Temporal data.rar
- Spatial data.rar

---

# HCP Experiments

The HCP experiments require substantial computational resources and large storage space. Therefore, the processed HCP data are not included in this repository.

Instead, an **executed Jupyter Notebook** is provided that contains:

- Complete implementation
- Experimental workflow
- Training procedure
- Final results
- Figures generated in the manuscript

Researchers can reproduce these experiments after downloading the HCP dataset and preprocessing it using the official HCP pipelines.


# Running the Experiments

## Temporal Model
python temp_iq.py

## Similarity and Hybrid Models
python similarity_and_hybrid_based_iq_prediction.py

## Ablation Studies
python ablation_studies.py

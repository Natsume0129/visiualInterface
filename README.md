# visualInterface

This repository contains my experimental implementation for **structure-oriented edge / boundary detection** based on **multi-scale wavelet responses**.

Instead of detecting all local gradients as edges, I focus on extracting boundaries that are closer to **human perceptual contours**.  
To evaluate this, I manually draw sketch-based contours and use them as pixel-level ground truth for quantitative evaluation.

This project was developed as part of a visual interface / vision assignment, and the full methodology and results are documented in the accompanying report.

---

## Repository Structure

The repository is organized as follows:

    visualInterface/

    ├── codes/

    ├── cat_experiment/

    ├── lena_experiment/

    ├── REPORT.md

    └── VI25-Report.pdf


### `codes/`
This directory contains all core Python implementations of the method.

The code covers the full pipeline:
- Ground-truth generation from hand-drawn sketches
- Multi-scale wavelet-based edge response computation
- Baseline multi-scale fusion
- Cross-scale consistency filtering
- Pixel-level evaluation (Precision / Recall / F1 with tolerance)
- Parameter grid search for optimization

All experiments in this repository are driven by the scripts in this folder.

---

### `cat_experiment/`
This folder contains the **first-stage experiment** using a cat image.

The purpose of this experiment is exploratory:
- The cat image contains strong texture (fur), which produces many non-structural edges
- This experiment demonstrates the limitation of the method when texture dominates over shape
- The results motivated switching to a cleaner test image (Lena) and introducing systematic parameter optimization

---

### `lena_experiment/`
This folder contains the **main experiments** conducted on the Lena image.

Key characteristics:
- Two different hand-drawn sketches are used as ground truth (different annotation styles)
- Both baseline and consistency-based edge maps are evaluated independently
- Extensive parameter search is performed to find the best configuration for each sketch

Subdirectories include:
- Visualization results (overlay of detected edges and sketches)
- Optimization results (best-performing parameter sets)
- Comparison between baseline and consistency-based outputs

This folder represents the core contribution of the project.

---

### `REPORT.md`
Markdown version of the full experimental report.

It includes:
- Problem formulation
- Method description
- Experimental design
- Quantitative evaluation
- Parameter optimization results
- Discussion of failure cases and limitations

---

### `VI25-Report.pdf`
PDF version of the final report, formatted for submission and presentation.

---

## Method Overview

My approach consists of two **independently evaluated pipelines**:

1. **Baseline multi-scale fusion**
   - Multi-scale wavelet responses are computed separately along x and y directions
   - Responses are fused across scales to produce an edge probability map (`P_base`)
   - This approach tends to achieve higher recall but includes more texture edges

2. **Consistency-based edge detection**
   - Cross-scale consistency constraints are applied
   - Only edges that are stable across multiple scales are retained (`P_cons`)
   - This approach improves precision but may suppress weak or sketch-dependent contours

Both outputs are evaluated separately against the same ground truth.

---

## Ground Truth Construction

Ground truth is generated from hand-drawn sketches using the following steps:
1. Resize to match the original image resolution
2. Gaussian smoothing for stability
3. Otsu thresholding
4. Automatic inversion if foreground/background is flipped
5. Optional morphological opening to remove small noise

This process converts subjective sketches into a consistent binary contour representation suitable for pixel-level evaluation.

---

## Evaluation

- Evaluation is performed at pixel level
- A spatial tolerance is allowed to account for sketch imprecision
- Precision, Recall, and F1-score are reported
- Baseline and consistency-based results are evaluated independently

Parameter optimization is performed using grid search, with F1-score as the selection criterion.

---

## Notes and Observations

- Optimization does not monotonically improve performance; stronger consistency constraints often increase precision while significantly reducing recall
- Results are highly sensitive to the annotation style of the ground truth sketch
- Texture-heavy images remain challenging for this approach

---

## How to Read This Repository

1. Start with `REPORT.md` or `VI25-Report.pdf` to understand the motivation and conclusions
2. Inspect visual results in `lena_experiment/` for qualitative comparison
3. Read the implementation details in `codes/` alongside the report sections

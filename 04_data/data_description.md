# Dataset Description

## Overview

This research evaluates the proposed Multiplier-Free FPGA Acceleration of Convolutional Kolmogorov-Arnold Networks for Edge Inference on standard image classification benchmarks commonly used in the ultra-low-latency, resource-constrained neural network hardware literature. The paper reports results across three datasets — **MNIST**, **Fashion MNIST**, and **CIFAR-10** — with MNIST and Fashion MNIST serving as the primary design-under-test (DUT) and the latter two used for comparative benchmarking. We chose these three datasets primarily because their small size makes them ideal for training a lightweight model. Additionally, their widespread use provides ample existing models for comparison.

---

## Datasets

### 1. MNIST (Primary Benchmark #1)

| Property | Detail |
|---|---|
| Task | Handwritten digit recognition (10-class classification) |
| Input dimensions | 28 × 28 × 1 (grayscale) |
| Classes | 10 (digits 0–9) |
| Standard train split | 60,000 images |
| Standard test split | 10,000 images |

MNIST is the sole dataset for which a full end-to-end hardware inference pipeline is implemented and characterized. The paper describes a two-layer CKAN network specifically configured for MNIST's 28×28 grayscale inputs, making it the definitive benchmark for resource utilization, latency, and accuracy reporting.

### 2. Fashion MNIST (Primary Benchmark #2)

| Property | Detail |
|---|---|
| Task | Clothing item classification (10-class) |
| Input dimensions | 28 × 28 × 1 (grayscale) |
| Classes | 10 (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot) |
| Standard train split | 60,000 images |
| Standard test split | 10,000 images |

Fashion MNIST shares the same spatial dimensions as MNIST and is included in comparison Table III and Table IV alongside competing architectures (KANELÉ, TreeLUT, PolyLUT, hls4ml).

### 3. CIFAR-10 (Comparative Benchmark)

| Property | Detail |
|---|---|
| Task | Natural image classification (10-class) |
| Input dimensions | 32 × 32 × 3 (RGB) |
| Classes | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Standard train split | 50,000 images |
| Standard test split | 10,000 images |

---

## Train / Validation / Test Split

- **MNIST**: 60,000 train / 10,000 test 
- **Fashion MNIST**: 60,000 train / 10,000 test
- **CIFAR-10**: 50,000 train / 10,000 test

---

## Preprocessing
- **Quantization**: The CKAN model undergoes **quantization-aware training** with B-spline pruning. The trained floating-point B-spline functions are discretized into compact truth tables (lookup tables) for hardware deployment.
- **Input bit-width**: Pixel values are represented at a reduced bit-width consistent with the hardware DATA_WIDTH parameter. For the MNIST DUT, Layer 1 uses `DATA_WIDTH = 4` bits for input data.
- **B-spline discretization**: Each trained basis function φ_k(·) is converted to a ROM with a **6-bit input address** (64 entries) and a **4-bit signed output**, implemented via a Python conversion script that discretizes the continuous B-spline. This discretization constitutes the primary quantization step applied to the model weights.
- **Pixel streaming**: Inputs are streamed in **raster-scan order** at one pixel per clock cycle; no spatial reordering or normalization beyond quantization is described.
- **Hexdecimal File Conversion** : Converted the dataset into a hexadecimal format to facilitate inference on an FPGA.

---

## Subset / Reduced Setup

- The MNIST network uses only **2 output channels** per convolutional layer and a small feature map pipeline (28×28 → 26×26 → 13×13 → 11×11 → 5×5 → flatten to 400 values), making it a lightweight architecture rather than a full-scale model.
- The design targets a **Xilinx Zynq-7020 (PYNQ-Z2)** FPGA, a resource-constrained edge device, which motivates the compact model configuration.

---


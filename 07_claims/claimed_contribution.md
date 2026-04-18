# Claimed Contribution

---

## What We Reproduced

We reproduced the core idea from KANELÉ — converting trained B-spline functions into lookup tables and storing them in FPGA distributed RAM. We also reproduced the quantization-aware training setup from the KAN literature, including B-spline pruning to reduce the number of active functions before export. The MNIST training pipeline and the basic LUT-ROM structure in Verilog follow established patterns from prior work.

---

## What We Modified

We took the LUT-ROM evaluation approach from KANELÉ, which only works for fully connected layers, and redesigned it to support convolution. This meant building a sliding window buffer that feeds a K×K pixel window into the LUT bank at each step, and adding a pipelined adder tree to accumulate the K² lookup results. We also added multi-channel support so the same pipeline can handle multiple input and output feature maps in parallel. The full inference pipeline — convolution, max-pooling, flattening, and a KAN-based MLP classifier — was integrated into a single design running on a Zynq-7020 FPGA.

---

## What Did Not Work

Getting high accuracy while keeping the LUT tables small enough for the TinyImageNet was not feasible.We also could not get CIFAR-10 working within the resource limits of the Zynq-7020 — the network needed too many channels to fit. 
---

## What We Believe Is Our Contribution

No prior work has built a hardware accelerator specifically for convolutional KAN layers. Everything before this — KANELÉ, ArKANe, Nilsson, Alam et al. — only targets fully connected KAN layers. Our design is the first to handle the spatial structure of CKANs in hardware: parallel LUT evaluation over a sliding window, cross-channel accumulation, and a complete multi-layer inference pipeline. The result runs on a low-cost FPGA using zero DSP blocks and zero BRAM, which we think is a meaningful result for edge deployment.

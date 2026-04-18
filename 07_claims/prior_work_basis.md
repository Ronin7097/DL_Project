# Prior Work Basis

This document lists the core literature that formed the foundation for the CKAN (Convolutional KAN on FPGA) architecture, detailing how each specific work influenced our design decisions.

## Foundations of LUT-Based Neural Networks

**1. LogicNets: Co-Designed Neural Networks and Circuits for Extreme-Throughput Applications**  
*Umuroglu, Y., Akhauri, Y., Fraser, N. J., & Blott, M. (FPL 2020).*  
**Influence:** Provided the foundational paradigm of mapping quantized neural network layers directly into FPGA Lookup Tables (LUTs), entirely bypassing DSP blocks and multiply-accumulate (MAC) operations to achieve extreme throughput. This principle guided our core pipeline goal of achieving O(1) hardware inference.

**2. NeuraLUT: Hiding Neural Network Density in Boolean Synthesizable Functions**  
*Andronic, M., & Constantinides, G. A. (FPL 2024).*  
**Influence:** Heavily influenced our quantization framework. Our project directly leverages algorithms and concepts from NeuraLUT (as seen in our `quant.py` and `KANQuant.py` adaptations) to systematically extract truth tables from quantization-aware networks and map dense activations into hardware-efficient boolean structures.

**3. PolyLUT: Learning Piecewise Polynomials for Ultra-Low Latency FPGA LUT-based Inference**  
*Andronic, M., & Constantinides, G. A. (FPGA 2024).*  
**Influence:** Demonstrated the viability of utilizing LUTs to directly evaluate piecewise polynomial functions in hardware. Since Kolmogorov-Arnold Networks naturally rely on learning B-splines (piecewise polynomials), PolyLUT served as a theoretical and architectural catalyst for directly baking B-spline curves into single ROM primitives (`KAN_LUT_ROM_opt`).

**4. PolyLUT-Add: FPGA-based LUT Inference with Wide Inputs**  
*Andronic, M., & Constantinides, G. A. (FPL 2024).*  
**Influence:** Motivated the design of the pipelined adder-tree reduction logic utilized in our CKAN feature mapping. It influenced how we combine the independent univariate functions (spline connections) across multiple input channels seamlessly in Verilog.

## Kolmogorov-Arnold Networks in Hardware

**5. KANELÉ: Kolmogorov-Arnold Networks for Efficient LUT-Based Evaluation**  
*Kjeang, W. et al. (FPGA 2026).*  
**Influence:** The most direct inspiration and dependency for this project. KANELÉ established the baseline methodology for mapping KAN-based MLP primitives into VHDL using specialized LUTs. We directly utilize the KANELÉ infrastructure to deploy the final fully-connected KAN layers, allowing us to focus on building the novel Convolutional KAN mechanisms on top of this foundation.

**6. ArKANe: Accelerating Kolmogorov-Arnold Networks on Reconfigurable Spatial Architectures**  
*Nitu, P. et al. (IEEE Embedded Systems Letters 2026).*  
**Influence:** Provided critical architectural context on the spatial dataflow mapping of B-splines. Insights from ArKANe influenced our core "Split-ROM Optimization" strategy, driving our decision to partition complex multi-variable convolutions into purely 1D, 6-bit standalone memory blocks to maximize parallel spatial throughput and minimize LUT usage.

**7. Design of a Kolmogorov-Arnold Network Hardware Accelerator**  
*Nilsson, E. (Master's Thesis, Lund University, 2025).*  
**Influence:** Served as an exploratory baseline illustrating early challenges in mapping KAN configurations to FPGAs. This work guided our implementation's resource profiling, validating the need to dramatically shrink our internal LUT footprints stringently via algorithmic connection pruning during training.

**8. Hardware Acceleration of Kolmogorov-Arnold Networks for Lightweight Edge Inference**  
*Alam, R. et al. (ASP-DAC 2025).*  
**Influence:** Confirmed the practical utility of deploying KAN architectures specifically for resource-constrained edge environments. This shaped our design requirements, compelling us to use dynamically parameterized modules (configurable bitwidths, stride, kernel sizes) explicitly targeting edge FPGA deployment (such as the target board xc7z020).

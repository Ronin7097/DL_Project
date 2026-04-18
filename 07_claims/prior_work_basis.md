# Prior Work Basis

These are the main papers we read before and during the project, and what we took from each.

---

**1. Kolmogorov-Arnold Networks (Liu et al., 2025)**

This is the original KAN paper. It introduced the idea of putting learnable functions on network edges instead of fixed weights. We used this as the starting point to understand what KANs are and why they might be easier to compress for hardware.

---

**2. Convolutional Kolmogorov-Arnold Networks (Bodner et al., 2025)**

This paper extended KANs to work like CNNs by applying the same learnable function idea to convolution. Our whole project is built on this — we took the CKAN concept and asked how to run it efficiently on an FPGA.

---

**3. KAN 2.0 (Liu et al., 2025)**

A follow-up to the original KAN paper with improvements to B-spline training and pruning. We used the pruning ideas here to reduce how many functions the network keeps, which directly affects how much ROM the FPGA needs.

---

**4. KANELÉ (Kjeang et al., 2026)**

This was the most directly related hardware paper. It showed that KAN B-spline functions can be stored as lookup tables in FPGA distributed RAM and evaluated without any multipliers. We used the same core idea for our convolution engine. The difference is that KANELÉ only handles fully connected layers — we extended it to handle convolution.

---

**5. LogicNets (Umuroglu et al., 2020)**

This paper showed that heavily quantized neurons can be expressed as truth tables and mapped directly to FPGA LUTs. It gave us confidence that the lookup table approach was practical for neural network inference on FPGAs.

---

**6. PolyLUT and PolyLUT-Add (Andronic et al., 2024)**

These two papers improved on LogicNets by putting more complex functions inside each LUT. We used them as benchmarks in our comparison tables and also read them to understand the tradeoffs between LUT size and accuracy.

---

**7. Efficient Processing of Deep Neural Networks (Sze et al., 2017)**

A broad survey of CNN accelerator design. We read this to understand dataflow strategies like row-stationary and weight-stationary, which helped us think about how to handle data reuse in the sliding window buffer.

---

**8. Eyeriss (Chen et al., 2017)**

A well-known CNN accelerator that uses a spatial array of MAC units. We used it as a reference for what a traditional CNN accelerator looks like, and as a contrast to show why our design is different — we have no MAC units at all.


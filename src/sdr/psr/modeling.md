# BTA-dist modeling

## Desirable models

### FLOPS counts model

We want to get the theoretical FLOPS of our algorythm and compare them with both
CPU and GPU achieved FLOPS.

- CPU profiling using *PAPI*
- GPU profiling using NVIDIA and AMD tools

### Operations ratios model

We want to get the theoretical operations ratio of our algorythm and compare them
with both CPU and GPU achieved operations ratio. Similar to what Lisa's did in  
her analysis.

- GEMM
- POTRF
- TRSM..

### Alpha-Beta-(Gamma) communication model

Use alpha-beta-gamma model to show the communication cost of our algorythm and compare
them with both CPU and GPU achieved communication cost.

- SLI analysis: *https://github.com/LLNL/Aluminum* library

## Desirable analysis

### 1. "Classical" weak-scaling

We use the **FLOPS count model** to show weak scaling analysis.Because of the size
of the reduced system evolving with the number of processes the operations ratio
will evolve. We use the **operations ratios model** to show how this evolves.

### 2. "Ratio" weak-scaling

We distinguishe a classical weak-scalling analysis to a "ratio" weak-scaling, where  
the size of the problem grow but the ratio between the partitions sizes and the  
reduced system size is constant. We use the **FLOPS count model** and the  
**operations ratios model** to show this being constant.  

### 3. Strong-scaling

We run strong scaling in some relevant test-cases.

- **FLOPS count model**
- **operations ratios model** 

### 4. Communication analysis

- **Alpha-Beta-(Gamma) communication model**
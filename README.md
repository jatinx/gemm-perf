# Gemm comparision

A trivially naive gemm comparision among different libs.

## How to compile

Use the compiler of the platform you are on, i.e. `hipcc` if `ROCm` or `nvcc` if `CUDA`.
Make sure you have `openblas` installed, thats is our gold standard against which all results are compared.

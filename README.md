# CUDA_cutlass_benchmark

**This cutlass code will find the best MNK to stress the best performance in different GPU.**

**usage(This script will boost you CPU and GPU clocks):**
>
make<br />
<br />
#Find Best MNK Usage         :  ./all_gemm --gemmType=N  --findBest   <br />
#Find Best MNK + Stress Usage:  ./all_gemm --gemmType=N  --autoStress  <br />
#Stress Usage                :  ./all_gemm --gemmType=N  --stress --mn=xxx --k=yyy  <br />


**Result as following.** <br /> <br />
>./all_gemm --gemmType=4  --findBest
[TensorCore FP16(FP16 accumulation) Time and TFLOPS Result]<br />
    m      n      k          Time (msec)         TFLOPS  <br />
   1024,   1024,   1024,        0.06304,          34.07, <br />
   1024,   1024,   2048,        0.05139,          83.57, <br />
   1024,   1024,   3072,         0.0729,          88.38, <br />
   1024,   1024,   4096,         0.1659,          51.78, <br />
   1024,   1024,   5120,         0.2053,          52.31, <br />
   2048,   2048,   1024,        0.09267,          92.69, <br />
   2048,   2048,   2048,         0.1679,          102.3, <br />
   2048,   2048,   3072,         0.2458,          104.9, <br />
   2048,   2048,   4096,         0.3221,          106.7, <br />
[Peak TFLOPS]=106.7, m=n=2048, k=4096

**Tensor( 32 accumulation):**<br />
INT8_Tensor(INT8->INT32 accumulation)<br />
FP16_Tensor(FP16->FP16 accumulation) <br />
FP16_32_Tensor(FP16->FP32 accumulation) <br />

**Gemm without Tensor** <br />
HGEMM->FP16 GEMM <br />
SGEMM->FP32 GEMM <br />
DEGMM->FP64 GEMM <br />


**Theoretical FMA for Flops**<br />
**( number of core * peak freq in graphic clock * instruction per clock) <br />**

| Core/ GPU | Core(streaming processor) | Peak freq(GPU) | Instruction per clock | Flops | GFlops|
| ------------- | ------------- | ------------- | ------------- |------------- | ------------- |
|Tensor CORE|	640(64)=40960|	1.53(SXM2-V100)|	2|	125337.6	|125.34|
|FP16|	5120	|1.53(SXM2-V100)|	4|	31334.4|	31.33|
|FP32|	5120	|1.53(SXM2-V100)|	2|	15667.2|	15.67|
|FP64|	2560	|1.53(SXM2-V100)|	2	|7833.6	|7.83|




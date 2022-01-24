#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cufft.h"
#include "cuda.h"
#include "time.h"

#define SPECTRA     512
#define CHANNELS    16384
#define SAMPLES     CHANNELS * SPECTRA

void main()
{
    // Normal mode
    // data buffer on the host
    cufftReal *data_host = (cufftReal*) malloc(SAMPLES * sizeof(cufftReal));   
    cufftComplex *data_host_out = (cufftComplex*) malloc(SAMPLES * sizeof(cufftComplex));
    // data buffer on the GPU
    cufftComplex *data_gpu_out;
    cufftReal *data_gpu;
    cudaMalloc((void**)&data_gpu, SAMPLES * sizeof(cufftReal));
    cudaMalloc((void**)&data_gpu_out, SAMPLES * sizeof(cufftComplex));
    // copy data from host to GPU
     cudaMemcpy(data_gpu, data_host, SAMPLES * sizeof(cufftReal), cudaMemcpyHostToDevice);
    // cufft
    cufftPlanMany(...);
    cufftExecR2C(...);
    // copy result from GPU to host
    cudaMemcpy(data_host_out, data_gpu_out, SAMPLES * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // Zero copy mode
    // data buffer on the host
    cufftComplex *data_host_out;
    cudaHostAlloc((void **)&data_host_out, SAMPLES * sizeof(cufftComplex), cudaHostAllocMapped);
    cufftReal *data_host;
    cudaHostAlloc((void **)&data_host, SAMPLES * sizeof(cufftReal), cudaHostAllocMapped);
    // share the memory between host and GPU
    cufftComplex *data_gpu_out;
    cufftReal *data_gpu;
    cudaHostGetDevicePointer((void**)&data_gpu, data_host, 0);
    cudaHostGetDevicePointer((void**)&data_gpu_out, data_host_out, 0);
    // cufft
    cufftPlanMany(...);
    cufftExecR2C(...);
    
}


/*******************************************************************************
All of the GPU related code are here.
We will compile the code as a .so, and then link the code in the hashpipe code.
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "cufft.h"
#include "cuda.h"

//extern "C" {
#include "fast_gpu.h"
// This is the PFB FIR code from James
#include "pfb_fir.cuh"

// The following buffers are the GPU buffers.
char            *data_in_gpu;       // input data
float           *weights_gpu;       // PFB FIR weights
cufftReal       *pfbfir_out_gpu;    // the output of PFB FIR
cufftComplex    *data_out_gpu;      // output data on GPU
cufftComplex    *data_out_host;     // output data on host

// cufft plan
cufftHandle plan;

// PFB FIR parameters
int step        = CHANNELS;
int out_n       = step * SPECTRA;
int stepy       = out_n/(256*1024)*step;
int groupsx     = step/WGS;
int groupsy     = (out_n + stepy - 1)/stepy;
dim3 dimgrid(groupsx*WGS, groupsy);
dim3 dimblock(WGS,1);

int GPU_GetDevInfo()
{
    cudaDeviceProp prop;
    int deviceID;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&prop, deviceID);
    printf("GPU Device Info:\r\n");
    printf("%-25s: %d\r\n", "MaxThreadsPerBlock", prop.maxThreadsPerBlock);
    printf("%-25s: %d %d %d\r\n","maxThreadsDim", prop.maxThreadsDim[0], \
                                                  prop.maxThreadsDim[1], \
                                                  prop.maxThreadsDim[2]);
    printf("%-25s: %d %d %d\r\n","maxGridSize",   prop.maxGridSize[0], \
                                                  prop.maxGridSize[1], \
                                                  prop.maxGridSize[2]);

    if(!prop.deviceOverlap)
        return -1;
    else
        return 0;
}

// This func is used for allocating pinned memory on the host computer 
//int Host_MallocBuffer(DIN_TYPE *buf_in, DOUT_TYPE *buf_out)
int Host_MallocBuffer(DIN_TYPE **buf_in, DOUT_TYPE **buf_out)
{
    cudaError_t status;
    status = cudaMallocHost((void **)buf_in, SAMPLES * sizeof(DIN_TYPE));
    if(status != cudaSuccess)
        return -1;
    status = cudaMallocHost((void **)buf_out, OUTPUT_LEN * sizeof(DOUT_TYPE));
    if(status != cudaSuccess)
        return -2;
    return 0;
}

// This func is used for allocating memory on the GPU
void GPU_MallocBuffer()
{
    cudaMalloc((void**)&data_in_gpu, SAMPLES * sizeof(char));
    cudaMalloc((void**)&weights_gpu, TAPS*CHANNELS*sizeof(float));
    cudaMalloc((void**)&pfbfir_out_gpu, CHANNELS*SPECTRA*sizeof(cufftReal));
    cudaMalloc((void**)&data_out_gpu, CHANNELS*SPECTRA * sizeof(cufftComplex));
    cudaMallocHost((void**)&data_out_host, OUTPUT_LEN * sizeof(cufftComplex));
}

// This func is used for creating cufft plan
int GPU_CreateFFTPlan()
{
    int rank = 1;
    int n[1];
    n[0] = CHANNELS;
    int istride = 1;
    int idist = CHANNELS;
    int ostride = 1;
    int odist = CHANNELS;
    
    int inembed[1], onembed[1];
    inembed[0] = CHANNELS*SPECTRA;
    onembed[0] = CHANNELS*SPECTRA;
    cufftResult fft_ret = cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, SPECTRA);
    if( fft_ret != CUFFT_SUCCESS )
        return -1;
    else
        return 0;
}

// move weights from host to GPU
void GPU_MoveWeightsFromHost(float *weights)
{
    cudaMemcpy(weights_gpu, weights, TAPS * CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
}

// move data from host to GPU
void GPU_MoveDataFromHost(DIN_TYPE *din)
{
    cudaMemcpy(data_in_gpu, din, SAMPLES * sizeof(DIN_TYPE), cudaMemcpyHostToDevice);
}

// move data from GPU to host
void GPU_MoveDataToHost(DOUT_TYPE *dout)
{
//#pragma unroll 
    for(int i = 0; i < SPECTRA; i++)
    //cudaMemcpy(dout + i * CH_PER_SPEC, data_out_gpu + i * CHANNELS + START_BIN, OUTPUT_LEN * sizeof(DOUT_TYPE), cudaMemcpyDeviceToHost);
        cudaMemcpy(data_out_host + i * CH_PER_SPEC, \
                   data_out_gpu + i * CHANNELS + START_BIN, 
                   OUTPUT_LEN * sizeof(cufftComplex), 
                   cudaMemcpyDeviceToHost);
    for(int i = 0; i < OUTPUT_LEN; i++)
        dout[i] = data_out_host[i].x * data_out_host[i].x + \
                  data_out_host[i].y * data_out_host[i].y;

}

// do PFB
int GPU_DoPFB()
{

    pfb_fir<<<dimgrid,dimblock>>>(
        (float *)pfbfir_out_gpu,  
        (char*)data_in_gpu,   
        weights_gpu,    
        out_n,
        step,
        stepy,
        0,
        0
        ); 
    cudaDeviceSynchronize();
    cufftResult fft_ret;
    fft_ret = cufftExecR2C(plan, (cufftReal*)pfbfir_out_gpu, (cufftComplex*) data_out_gpu);
    if (fft_ret != CUFFT_SUCCESS)
        return -1; 
    else
        return 0;
}

void GPU_DestroyPlan()
{
    cufftDestroy(plan);
}

void Host_FreeBuffer(DIN_TYPE *buf_in, DOUT_TYPE *buf_out)
{
    cudaFreeHost(buf_in);
    cudaFreeHost(buf_out);
}

void GPU_FreeBuffer()
{
    cudaFree(data_in_gpu);
    cudaFree(weights_gpu);
    cudaFree(pfbfir_out_gpu);
    cudaFree(data_out_gpu);
    cudaFreeHost(data_out_host);
}
//}
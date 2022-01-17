#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cufft.h"
#include "cuda.h"
#include "time.h"

#ifndef SPECTRA
#define SPECTRA     4096
#endif

#define CHANNELS    16384
#define SAMPLES     CHANNELS * SPECTRA

#define WR_TO_FILE
//#define NORMAL

#define ELAPSED_NS(start,stop) \
  (((int64_t)stop.tv_sec-start.tv_sec)*1000*1000*1000+(stop.tv_nsec-start.tv_nsec))

void gen_fake_data(float *data) {
   float fs = 1024;
   float fin  = 128;
   for( size_t t=0; t<SAMPLES; t++ ) { 
       double f = 2*M_PI * t *fin/fs;
       float res = 127 * sin(f);
       *(data+t) = res;
   }
}

int main()
{
    struct timespec start, stop;
    int64_t elapsed_gpu_ns  = 0;


 #ifdef NORMAL
    printf("Normal Mode\r\n");
    // data buffer on the host computer
    cufftReal *data_host = (cufftReal*) malloc(SAMPLES * sizeof(cufftReal));   
    cufftComplex *data_host_out;
    cudaHostAlloc((void **)&data_host_out, SAMPLES * sizeof(cufftComplex), cudaHostAllocMapped);
 #else
    printf("Zero Copy Mode\r\n");
    cufftComplex *data_host_out;
    cudaHostAlloc((void **)&data_host_out, SAMPLES * sizeof(cufftComplex), cudaHostAllocMapped);
    cufftReal *data_host;
    cudaHostAlloc((void **)&data_host, SAMPLES * sizeof(cufftReal), cudaHostAllocMapped);
#endif

    int64_t elapsed_gpu_ns3  = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // generate fake data
    float *fake_data = (float*) malloc(SAMPLES * sizeof(float));
    gen_fake_data(fake_data);
    clock_gettime(CLOCK_MONOTONIC, &stop);
    elapsed_gpu_ns3 = ELAPSED_NS(start, stop);
    printf("%-25s: %f ms\r\n","Generating fake data time", elapsed_gpu_ns3/1000000.0);

    // init data buffer
    for(int i = 0; i < SAMPLES; i++)
    {
        data_host[i] = fake_data[i];
    }
    
    // data buffer on GPU
    cufftComplex *data_gpu_out;
    cufftReal *data_gpu;
#ifdef NORMAL
    cudaMalloc((void**)&data_gpu, SAMPLES * sizeof(cufftReal));
#else
    // do nothing here
#endif
    cudaMalloc((void**)&data_gpu_out, SAMPLES * sizeof(cufftComplex));

    // record the start time
    int64_t elapsed_gpu_ns0  = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // copy data from host to GPU
#ifdef NORMAL
    cudaMemcpy(data_gpu, data_host, SAMPLES * sizeof(cufftReal), cudaMemcpyHostToDevice);
#else
    cudaHostGetDevicePointer((void**)&data_gpu, data_host, 0);
#endif
    clock_gettime(CLOCK_MONOTONIC, &stop);
    elapsed_gpu_ns0 = ELAPSED_NS(start, stop);
    printf("%-25s: %f ms\r\n","copy time(host to dev)", elapsed_gpu_ns0/1000000.0);

    int64_t elapsed_gpu_ns1  = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // exec fft
    cufftHandle plan;
    /*
    *   1d fft
    */
    //cufftPlan1d(&plan, SAMPLES, CUFFT_C2C,1);
    
    /*
    * Many fft
    */
    int rank = 1;
    int n[1];
    n[0] = CHANNELS;
    int istride = 1;
    int idist = CHANNELS;
    int ostride = 1;
    int odist = CHANNELS;
    
    int inembed[1], onembed[1];
    inembed[0] = CHANNELS * SPECTRA;
    onembed[0] = CHANNELS * SPECTRA;
    //inembed[1] = SPECTRA;
    //onembed[1] = SPECTRA;
    
    cufftResult fft_ret = cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, SPECTRA);
    //cufftResult fft_ret = cufftPlanMany(&plan, rank, n, NULL, istride, idist, NULL, ostride, odist, CUFFT_C2C, SPECTRA);

    if( fft_ret != CUFFT_SUCCESS ) {
        printf("cufftPlanMany failed\r\n");
    }

    //cufftExecC2C(plan, (cufftComplex*) data_gpu, (cufftComplex*) data_gpu, CUFFT_FORWARD);
    fft_ret = cufftExecR2C(plan, data_gpu, (cufftComplex*) data_gpu_out);
    if (fft_ret != CUFFT_SUCCESS) {
        printf("forward transform fail\r\n"); 
    }
    cudaDeviceSynchronize();

    // record the end time
    clock_gettime(CLOCK_MONOTONIC, &stop);
    elapsed_gpu_ns1 = ELAPSED_NS(start, stop);
    printf("%-25s: %f ms\r\n","Processing time", elapsed_gpu_ns1/1000000.0);

    // copy data from GPU to host
    int64_t elapsed_gpu_ns2  = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    cudaMemcpy(data_host_out, data_gpu_out, SAMPLES * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC, &stop);
    elapsed_gpu_ns2 = ELAPSED_NS(start, stop);
    printf("%-25s: %f ms\r\n","copy time(dev to host)", elapsed_gpu_ns2/1000000.0);

    elapsed_gpu_ns = elapsed_gpu_ns0 + elapsed_gpu_ns1 + elapsed_gpu_ns2;
    printf("%-25s: %f ms\r\n","total time", elapsed_gpu_ns/1000000.0);

    // cal power
    float *res = (float*) malloc(SAMPLES * sizeof(float));
    for(int i = 0; i < SAMPLES; i++)
    {
        res[i] = data_host_out[i].x * data_host_out[i].x + data_host_out[i].y * data_host_out[i].y;
    }

    // write data to file
#ifdef WR_TO_FILE
    FILE *fp;
    fp = fopen("fft.dat","w");
    if(fp==NULL)
    {
        fprintf(stderr, "the file can not be create.\r\n");
        return -1;
    }
    else
    {
        fprintf(stderr, "file created.\r\n");
    }
    fwrite(res,SAMPLES*sizeof(float),1,fp);
    fclose(fp);
#else
    // do nothing
#endif
    
    // end
    cufftDestroy(plan);
    cudaFree(data_gpu_out);
    free(res);
#ifdef NORMAL
    cudaFree(data_gpu);
    free(data_host);
#else
    cudaFreeHost(data_host);
#endif

    return 0;
}
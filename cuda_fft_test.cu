#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cufft.h"
#include "time.h"

#define SAMPLES 256

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
    
    // data buffer on the host computer
    cufftComplex *data_host = (cufftComplex*) malloc(SAMPLES * sizeof(cufftComplex));
    // generate fake data
    float *fake_data = (float*) malloc(SAMPLES * sizeof(float));
    gen_fake_data(fake_data);
    // init data buffer
    for(int i = 0; i < SAMPLES; i++)
    {
        data_host[i].x = fake_data[i];
        data_host[i].y = 0;
    }
    
    // data buffer on GPU
    cufftComplex *data_gpu;
    cudaMalloc((void**)&data_gpu, SAMPLES * sizeof(cufftComplex));
    
    // record the start time
    int64_t elapsed_gpu_ns0  = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // copy data from host to GPU
    cudaMemcpy(data_gpu, data_host, SAMPLES * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC, &stop);
    elapsed_gpu_ns0 = ELAPSED_NS(start, stop);
    printf("%-25s: %f ms\r\n","copy time(host to dev)", elapsed_gpu_ns0/1000000.0);

    int64_t elapsed_gpu_ns1  = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // exec fft
    cufftHandle plan;
    cufftPlan1d(&plan, SAMPLES, CUFFT_C2C,1);
    cufftExecC2C(plan, (cufftComplex*) data_gpu, (cufftComplex*) data_gpu, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    // record the end time
    clock_gettime(CLOCK_MONOTONIC, &stop);
    elapsed_gpu_ns1 = ELAPSED_NS(start, stop);
    printf("%-25s: %f ms\r\n","Processing time", elapsed_gpu_ns1/1000000.0);

    // copy data from GPU to host
    cudaMemcpy(data_host, data_gpu, SAMPLES * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // cal power
    float *res = (float*) malloc(SAMPLES * sizeof(float));
    for(int i = 0; i < SAMPLES; i++)
    {
        res[i] = data_host[i].x * data_host[i].x + data_host[i].y * data_host[i].y;
    }

    // write data to file
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

    // end
    cufftDestroy(plan);
    cudaFree(data_gpu);
    free(data_host);
    free(res);
    return 0;
}
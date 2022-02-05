#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "time.h"
#include "fast_gpu.h"

#define REPEAT      2
#define ELAPSED_NS(start,stop) \
  (((int64_t)stop.tv_sec-start.tv_sec)*1000*1000*1000+(stop.tv_nsec-start.tv_nsec))

// This func is used for generating fake data
void gen_fake_data(float *data) {
   float fs = 1024;
   float fin  = 2;
   for( size_t t=0; t< SAMPLES; t++ ) { 
       double f = 2*M_PI * t *fin/fs;
       float res = 127 * sin(f);
       *(data+t) = res;
       //*(data+t) = 1;
   }
}

int main()
{
    struct timespec start, stop;
    int64_t elapsed_gpu_ns  = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    clock_gettime(CLOCK_MONOTONIC, &stop);
    elapsed_gpu_ns = ELAPSED_NS(start, stop);

    int status = 0;
    
    // Check gpu status
    status = GPU_GetDevInfo();
    if(status < 0)
        printf("No device will handle overlaps.\r\n");
    else   
        printf("overlaps are supported on the device.\r\n");
    
    // Malloc buffer on GPU
    GPU_MallocBuffer();
    
    // Preparing weights for PFB FIR
    float *weights;
    weights = (float*) malloc(TAPS*CHANNELS*sizeof(float));
    printf("preparing for weights...\r\n");
    for(int i = 0; i<(TAPS*CHANNELS); i++)weights[i] = 1.0;
    printf("weights ready.\r\n");
    GPU_MoveWeightsFromHost(weights);


    int64_t elapsed_gpu_ns3  = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // Generate fake data
    float *fake_data = (float*) malloc(SAMPLES * sizeof(float));
    gen_fake_data(fake_data);
    clock_gettime(CLOCK_MONOTONIC, &stop);
    elapsed_gpu_ns3 = ELAPSED_NS(start, stop);
    printf("%-25s: %f ms\r\n","Generating fake data time", elapsed_gpu_ns3/1000000.0);
    
    
    // init data buffer
    DIN_TYPE *din;
    DOUT_TYPE *dout;
    status = Host_MallocBuffer(&din, &dout);
    if(status == -1)
        printf("Malloc din on pinned memory failed!\r\n");
    else if(status == -2)
        printf("Malloc dout on pinned memory failed!\r\n");
    else
        printf("Malloc din and dout on pinned memory successfully!\r\n");
    for(unsigned int i = 0; i < SAMPLES; i++)
    {
        din[i] = fake_data[i];
    }

    // create cufft plan
    status = GPU_CreateFFTPlan();
    if(status == -1)
    {
        printf("The cuFFT plan can't be created!\r\n");
        return 0;
    }
    else
        printf("The cuFFT plan is created successfully!\r\n");

    // move data from host to GPU and do FFT
    for(int i = 0; i < REPEAT; i++)
    {
        GPU_MoveDataFromHost(din);
        status = GPU_DoPFB();
        if(status == -1)
        {
            printf("PFB failed!\r\n");
            return 0;
        }
        GPU_MoveDataToHost(dout);
    }
    
    
    printf("Done!\r\n");
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
    fwrite(dout,OUTPUT_LEN * sizeof(float),1,fp);
    fclose(fp);
    
    // close everything
    GPU_DestroyPlan();
    Host_FreeBuffer(din, dout);
    GPU_FreeBuffer();
    return 0;
}
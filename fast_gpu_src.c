#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "time.h"
#include "fast_gpu.h"

#define REPEAT      2
#define ELAPSED_NS(start,stop) \
  (((int64_t)stop.tv_sec-start.tv_sec)*1000*1000*1000+(stop.tv_nsec-start.tv_nsec))

void gen_fake_data(float *data) {
   float fs = 1024;
   float fin  = 128;
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
    status = Host_MallocBuffer(din, dout);
    printf("status=%d\r\n",status);
    printf("1\r\n");
    if(status == -1)
        printf("Malloc din on pinned memory failed!\r\n");
    else if(status == -2)
        printf("Malloc dout on pinned memory failed!\r\n");
    else
        printf("Malloc din and dout on pinned memory successfully!\r\n");
    printf("SAMPLES=%d\r\n",SAMPLES);
    for(unsigned long i = 0; i < SAMPLES; i++)
    {
        //din[i] = fake_data[i];
        din[i] = 1;
    }
    printf("2\r\n");
    Host_FreeBuffer(din, dout);
    printf("3\r\n");
    GPU_FreeBuffer();
    
    return 0;
}
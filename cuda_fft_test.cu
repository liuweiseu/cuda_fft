#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cufft.h"
#include "cuda.h"
#include "time.h"

#include "pfb_fir.cuh"

#define WGS         128
#define CHANNELS    16384
#define TAPS        4
#define SPECTRA     512
#define SAMPLES     CHANNELS * (SPECTRA + TAPS - 1)

#define WR_TO_FILE
#define NORMAL

#define REPEAT      6
#define ELAPSED_NS(start,stop) \
  (((int64_t)stop.tv_sec-start.tv_sec)*1000*1000*1000+(stop.tv_nsec-start.tv_nsec))

void gen_fake_data(float *data) {
   float fs = 1024;
   float fin  = 128;
   for( size_t t=0; t<SAMPLES; t++ ) { 
       double f = 2*M_PI * t *fin/fs;
       float res = 127 * sin(f)+127;
       *(data+t) = res;
       //*(data+t) = 1;
   }
}

int GetDevInfo()
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
    printf("%-25s: %d %d %d\r\n","maxGridSize", prop.maxGridSize[0], \
                                                prop.maxGridSize[1], \
                                                prop.maxGridSize[2]);

    if(!prop.deviceOverlap)
        return -1;
    else
        return 0;
}

int main()
{
    struct timespec start, stop;
    int64_t elapsed_gpu_ns  = 0;

    cufftReal test0 = 4;
    float test1 = 4;
    printf("%f %f\r\n",(float)test0,(cufftReal)test1);
    printf("sizeof cufftReal: %ld\r\n",sizeof(cufftReal));

    int gpu_status = 0;
    gpu_status = GetDevInfo();
    if(gpu_status < 0)
        printf("No device will handle overlaps.\r\n");
    else   
        printf("overlaps are supported on the device.\r\n");

    //////////////////////////////////////////////////////////////////////////////////////////
    /*
    * preparing for pfb_fir
    */
    float *weights;
    weights = (float*) malloc(TAPS*CHANNELS*sizeof(float));
    printf("preparing for weights...\r\n");
    for(int i = 0; i<(TAPS*CHANNELS); i++)weights[i] = 1.0;
    printf("weights ready.\r\n");
    float *weights_gpu;
    cudaMalloc((void**)&weights_gpu, TAPS*CHANNELS*sizeof(float));
    cudaMemcpy(weights_gpu, weights, TAPS*CHANNELS*sizeof(float), cudaMemcpyHostToDevice);

    cufftReal *pfbfir_out_gpu;
    cudaMalloc((void**)&pfbfir_out_gpu, CHANNELS*SPECTRA*sizeof(cufftReal));

    float *pfbfir = (float*) malloc(sizeof(float)*CHANNELS*SPECTRA);
    for(int i = 0;i<CHANNELS*SPECTRA;i++)pfbfir[i] = 4.0;
    //cudaMemcpy(pfbfir_out_gpu, pfbfir, CHANNELS*SPECTRA*sizeof(float), cudaMemcpyHostToDevice);
    free(pfbfir);

    long long int step = CHANNELS;
    printf("%-10s : %lld\r\n","step",step);
    long long int out_n = step * SPECTRA;
    printf("%-10s : %lld\r\n","out_n",out_n);
    long long int stepy = (step * out_n + 256 * 1024 - 1)/(256*1024);
    printf("%-10s : %lld\r\n","stepy",stepy);
    int groupsx = step/WGS;
    printf("%-10s : %d\r\n","groupsx",groupsx);
    int groupsy = (out_n + stepy - 1)/stepy;
    printf("%-10s : %d\r\n","groupsy",groupsy);
    dim3 dimgrid(groupsx*WGS, groupsy);
    dim3 dimblock(WGS,1);
    ///////////////////////////////////////////////////////////////////////////////////////////

 #ifdef NORMAL
    printf("Normal Mode\r\n");
    // data buffer on the host computer
    //cufftReal *data_host = (cufftReal*) malloc(SAMPLES * sizeof(cufftReal));
    unsigned char *data_host = (unsigned char*) malloc(SAMPLES * sizeof(unsigned char));   
    cufftComplex *data_host_out = (cufftComplex*) malloc(SAMPLES * sizeof(cufftComplex));
 #else
    printf("Zero Copy Mode\r\n");
    cufftComplex *data_host_out;
    cudaHostAlloc((void **)&data_host_out, SAMPLES * sizeof(cufftComplex), cudaHostAllocMapped);
    cufftReal *data_host;
    cudaHostAlloc((void **)&data_host, SAMPLES * sizeof(cufftReal), cudaHostAllocMapped);
    //unsigned char *data_host;
    //cudaHostAlloc((void **)&data_host, SAMPLES * sizeof(unsigned char), cudaHostAllocMapped);
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
    //cufftReal *data_gpu;
    unsigned char *data_gpu;
#ifdef NORMAL
    //cudaMalloc((void**)&data_gpu, SAMPLES * sizeof(cufftReal));
    cudaMalloc((void**)&data_gpu, SAMPLES * sizeof(unsigned char));
    cudaMalloc((void**)&data_gpu_out, SAMPLES * sizeof(cufftComplex));
#else
    // do nothing here
#endif

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
    inembed[0] = SAMPLES;
    onembed[0] = SAMPLES;
    //inembed[1] = SPECTRA;
    //onembed[1] = SPECTRA;
    
    cufftResult fft_ret = cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, SPECTRA);
    //cufftResult fft_ret = cufftPlanMany(&plan, rank, n, NULL, istride, idist, NULL, ostride, odist, CUFFT_C2C, SPECTRA);

    if( fft_ret != CUFFT_SUCCESS ) {
        printf("cufftPlanMany failed\r\n");
    }

    // record the start time
    int64_t elapsed_gpu_ns0  = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i = 0; i < REPEAT; i++)
    {
    
    // copy data from host to GPU
#ifdef NORMAL
        //cudaMemcpy(data_gpu, data_host, SAMPLES * sizeof(cufftReal), cudaMemcpyHostToDevice);
        cudaMemcpy(data_gpu, data_host, SAMPLES * sizeof(unsigned char), cudaMemcpyHostToDevice);
        //cudaMemset(data_gpu,1, SAMPLES*sizeof(cufftReal));
#else
        cudaHostGetDevicePointer((void**)&data_gpu, data_host, 0);
        cudaHostGetDevicePointer((void**)&data_gpu_out, data_host_out, 0);
#endif
        
        pfb_fir<<<dimgrid,dimblock>>>(
        (float *)pfbfir_out_gpu,  
        (unsigned char*)data_gpu,   
        weights_gpu,    
        out_n,
        step,
        stepy,
        0,
        0
        );
     

        //cufftExecC2C(plan, (cufftComplex*) data_gpu, (cufftComplex*) data_gpu, CUFFT_FORWARD);


        fft_ret = cufftExecR2C(plan, (cufftReal*)pfbfir_out_gpu, (cufftComplex*) data_gpu_out);
        //fft_ret = cufftExecR2C(plan, (cufftReal*)data_gpu, (cufftComplex*) data_gpu_out);
        if (fft_ret != CUFFT_SUCCESS) {
            printf("forward transform fail\r\n"); 
        }
    }
    /*
    cufftReal *pfbfir_out = (cufftReal*)malloc(CHANNELS*SPECTRA*sizeof(cufftReal));
    cudaMemcpy(pfbfir_out, pfbfir_out_gpu, CHANNELS*SPECTRA*sizeof(cufftReal), cudaMemcpyDeviceToHost);
    for(long long unsigned int i = 0;i<CHANNELS*SPECTRA ;i++)
    {
        if(pfbfir_out[i]!=4.0)
        {
            printf("%lld %f\r\n",i,pfbfir_out[i]);
            break;
        }       
    }
    for(int i = 0;i<200;i++)printf("%d %f\r\n",i,pfbfir_out[i]);
    free(pfbfir_out);
    */
    cudaDeviceSynchronize();
    // record the end time
    clock_gettime(CLOCK_MONOTONIC, &stop);
    elapsed_gpu_ns0 = ELAPSED_NS(start, stop);
    printf("%-25s: %f ms\r\n","Processing and copy time", elapsed_gpu_ns0/1000000.0);

    // copy data from GPU to host
    int64_t elapsed_gpu_ns2  = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
#ifdef NORMAL
    cudaMemcpy(data_host_out, data_gpu_out, SAMPLES * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
#else
    // do nothing here
#endif
    clock_gettime(CLOCK_MONOTONIC, &stop);
    elapsed_gpu_ns2 = ELAPSED_NS(start, stop);
    printf("%-25s: %f ms\r\n","copy time(dev to host)", elapsed_gpu_ns2/1000000.0);

    elapsed_gpu_ns = elapsed_gpu_ns0  + elapsed_gpu_ns2;
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
    cudaFree(weights_gpu);
    cudaFree(pfbfir_out_gpu);
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
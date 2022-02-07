/* fast_gpu.h */
#ifndef _FASTGPU_H
#define _FASTGPU_H

#define TAPS 4
 
#define DIN_TYPE    char
#define DOUT_TYPE   float //struct FFT_RES

#define CHANNELS    65536
#define SPECTRA     4096
#define SAMPLES     CHANNELS * (SPECTRA + TAPS - 1)

#define START_BIN   0
#define STOP_BIN    255
#define CH_PER_SPEC (STOP_BIN - START_BIN + 1)
#define OUTPUT_LEN  SPECTRA * CH_PER_SPEC

int GPU_GetDevInfo();
int Host_MallocBuffer(DIN_TYPE **buf_in, DOUT_TYPE **buf_out);
void GPU_MallocBuffer();
int GPU_CreateFFTPlan();
void GPU_MoveWeightsFromHost(float *weights);
void GPU_MoveDataFromHost(DIN_TYPE *din);
void GPU_MoveDataToHost(DOUT_TYPE *dout);
int GPU_DoPFB();
void GPU_DestroyPlan();
void Host_FreeBuffer(DIN_TYPE *buf_in, DOUT_TYPE *buf_out);
void GPU_FreeBuffer();
#endif
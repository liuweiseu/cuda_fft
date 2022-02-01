/* fast_gpu.h */
#ifndef _FASTGPU_H
#define _FASTGPU_H

#define DIN_TYPE    char
#define DOUT_TYPE   float

#define CHANNELS    65536
#define SPECTRA     512
#define SAMPLES     CHANNELS * (SPECTRA + TAPS - 1)

#define START_BIN   0
#define STOP_BIN    255
#define OUTPUT_LEN  SPECTRA * (STOP_BIN - START_BIN)

#endif
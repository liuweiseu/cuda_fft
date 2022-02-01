NVCC	= nvcc

DEF		= 
TARGET  = cuda_fft_test

INC 	= pfb_fir.cuh
SRC		= cuda_fft_test.cu

CCFLAG 	= -I/usr/local/cuda/include \
		  -L/usr/local/cuda/lib64   \
		  -lcufft
#-O3 -std=c++11			
${TARGET}: ${SRC} ${INC}
	${NVCC} ${CCFLAG} ${SRC} ${DEF} -o $@ 

.PHONY: clean
clean:
	rm ${TARGET}
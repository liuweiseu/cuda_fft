NVCC	= nvcc

DEF		= 
TARGET  = cuda_fft_test

SRC		= cuda_fft_test.cu

CCFLAG 	= -I/usr/local/cuda/include \
		  -L/usr/local/cuda/lib64   \
		  -lcufft
#-O3 -std=c++11			
${TARGET}: ${SRC}
	${NVCC} ${CCFLAG} ${SRC} ${DEF} -o $@ 

.PHONY: clean
clean:
	rm ${TARGET}
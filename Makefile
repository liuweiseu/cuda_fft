NVCC	= nvcc

DEF		= 
TARGET  = cuda_fft_test

SRC		= cuda_fft_test.cu

CCFLAG 	= -O3 -std=c++11			\
		  -I/usr/local/cuda/include \
		  -L/usr/local/cuda/lib64   \
		  -lcufft

${TARGET}: ${SRC}
	${NVCC} ${CCFLAG} ${SRC} ${DEF} -o $@ 

.PHONY: clean
clean:
	rm ${TARGET}
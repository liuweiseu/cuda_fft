NVCC	= nvcc

TARGET  = cuda_fft_test

SRC		= cuda_fft_test.cu

CCFLAG 	= -I/usr/local/cuda/include \
		  -L/usr/local/cuda/lib64   \
		  -lcufft

${TARGET}: ${SRC}
	${NVCC} ${CCFLAG} ${SRC} -o $@ 

.PHONY: clean
clean:
	rm ${TARGET}
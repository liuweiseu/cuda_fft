NVCC	= nvcc

TARGET  = libfastgpu.so

INC 	= pfb_fir.cuh \
		  fast_gpu.h
SRC		= fast_gpu.cu

NVCCFLAG  = -Xcompiler -fPIC --shared \
			-I/usr/local/cuda/include \
		  	-L/usr/local/cuda/lib64   \
		  	-lcufft					

${TARGET}: ${SRC} ${INC} 
	${NVCC} -o $@ ${SRC} ${NVCCFLAG}


.PHONY: clean install
install:
	install -p ${TARGET} "/usr/local/lib" 
clean:
	rm ${TARGET} 
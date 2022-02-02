CC	= gcc

DEF		= 
TARGET  = fast_gpu_test

INC		= fast_gpu.h
SRC		= fast_gpu_src.c

FLAG  	= -I./fast_gpu \
          -L./fast_gpu \
		  -lfastgpu -lm


${TARGET}: ${SRC}
	${CC} ${SRC} ${DEF} -o $@  ${FLAG} 

.PHONY: clean
clean:
	rm ${TARGET}
CC	= gcc

DEF		= 
TARGET  = fast_gpu_test

INC		= fast_gpu.h
SRC		= fast_gpu_src.c

FLAG  	= -I./fast_gpu \
          -L./fast_gpu \
		  -lfastgpu -lm

SUB_DIR = fast_gpu

${TARGET}: ${SRC} ${SUB_DIR}
	${CC} ${SRC} ${DEF} -o $@  ${FLAG} 

${SUB_DIR}: ECHO
	make -C $@

ECHO:
	echo ${SUB_DIR}
	
.PHONY: clean
clean:
	rm ${TARGET}